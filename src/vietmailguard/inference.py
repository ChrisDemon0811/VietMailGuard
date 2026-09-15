"""Stable Version 2 inference API with separate ML and security decisions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from vietmailguard.content_analyzer import analyze_content, load_security_config
from vietmailguard.dataset_standardizer import normalize_text
from vietmailguard.eml_parser import parse_eml_bytes, parse_eml_file
from vietmailguard.explainability import LinearFeatureExplainer, combine_explanations
from vietmailguard.file_utils import sha256_file
from vietmailguard.language_detection import detect_language_details
from vietmailguard.risk_engine import calculate_risk, load_risk_config
from vietmailguard.sender_analyzer import analyze_sender
from vietmailguard.url_analyzer import analyze_urls

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "v2_bilingual" / "production_pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "v2_bilingual" / "model_metadata.json"
DEFAULT_RISK_CONFIG_PATH = PROJECT_ROOT / "config" / "risk_rules_v2.json"
DEFAULT_SECURITY_CONFIG_PATH = PROJECT_ROOT / "config" / "security_rules_v2.json"

RECOMMENDED_ACTIONS = {
    "normal": "ALLOW",
    "spam": "MOVE_TO_SPAM",
    "phishing": "QUARANTINE",
}

INFERENCE_RESULT_FIELDS = {
    "model_name",
    "model_version",
    "detected_language",
    "language_support_status",
    "prediction",
    "class_probabilities",
    "confidence",
    "risk_score",
    "risk_level",
    "model_explanation",
    "security_findings",
    "url_findings",
    "reasons",
    "recommended_action",
    "limitations",
}


def model_display_name(metadata: dict[str, Any]) -> str:
    """Build a presentation name from saved metadata without changing the artifact."""
    version = str(metadata.get("model_version", "unknown"))
    major = version.split(".", maxsplit=1)[0]
    suffix = f"V{major}" if major.isdigit() else version
    languages = {str(value) for value in metadata.get("training_languages", [])}
    scope = " Bilingual" if {"en", "vi"}.issubset(languages) else ""
    return f"VietMailGuard {suffix}{scope}".strip()


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _model_classes(model: Any) -> list[str]:
    classifier = getattr(model, "named_steps", {}).get("classifier")
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        classes = getattr(model, "classes_", None)
    return [] if classes is None else [str(value) for value in classes]


def _test_metadata(model: Any) -> dict[str, Any]:
    """Provide explicit test-only metadata when a model is dependency-injected."""
    return {
        "model_version": "test-double",
        "class_names": _model_classes(model),
        "supported_content_languages": {
            "en": {"status": "supported", "evaluation": "test double"},
            "vi": {"status": "experimental", "evaluation": "test double"},
            "mixed": {"status": "experimental_not_independently_benchmarked"},
        },
        "native_benchmark_status": "Native Vietnamese benchmark is not available.",
        "limitations": ["Dependency-injected model; production metadata was not loaded."],
    }


def _merge_configs(
    risk_config: dict[str, Any], security_config: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(security_config)
    merged["risk_engine"] = risk_config["risk_engine"]
    return merged


def _file_fingerprint(path: Path) -> tuple[str, int, int, str]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns), sha256_file(resolved)


def _flatten_security_findings(
    content_findings: list[dict[str, Any]],
    url_analyses: list[dict[str, Any]],
    sender_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = [{"source": "content", **finding} for finding in content_findings]
    output.extend(
        {"source": "url", "url": analysis.get("url", ""), **finding}
        for analysis in url_analyses
        for finding in analysis.get("findings", [])
    )
    output.extend({"source": "sender", **finding} for finding in sender_findings)
    return output


def _recommendation(
    *,
    prediction: str,
    risk_level: str,
    class_probabilities: dict[str, float],
    content_findings: list[dict[str, Any]],
    url_analyses: list[dict[str, Any]],
    sender_findings: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[str, list[str]]:
    """Recommend review on disagreement while preserving the ML class unchanged."""
    base = RECOMMENDED_ACTIONS[prediction]
    content_codes = {str(value.get("code", "")) for value in content_findings}
    url_codes = {
        str(finding.get("code", ""))
        for analysis in url_analyses
        for finding in analysis.get("findings", [])
    }
    sender_codes = {str(value.get("code", "")) for value in sender_findings}
    review_reasons: list[str] = []

    if prediction == "normal":
        strong_content = content_codes & set(policy["normal_review_content_codes"])
        if strong_content:
            review_reasons.append(
                "normal prediction conflicts with content-security indicators: "
                + ", ".join(sorted(strong_content))
            )
        strong_urls = url_codes & set(policy["normal_review_url_codes"])
        if strong_urls:
            review_reasons.append(
                "normal prediction conflicts with URL indicators: "
                + ", ".join(sorted(strong_urls))
            )
        strong_sender = sender_codes & set(policy["normal_review_sender_codes"])
        if strong_sender:
            review_reasons.append(
                "normal prediction conflicts with sender indicators: "
                + ", ".join(sorted(strong_sender))
            )
        commercial_code = str(policy["commercial_code"])
        commercial_matches = sum(
            int(finding.get("match_count", 0))
            for finding in content_findings
            if finding.get("code") == commercial_code
        )
        if commercial_matches >= int(policy["minimum_commercial_matches_for_review"]):
            review_reasons.append(
                "normal prediction conflicts with multiple promotional indicators"
            )
    elif prediction == "spam":
        strong_content = content_codes & set(policy["spam_review_content_codes"])
        if strong_content:
            review_reasons.append(
                "spam prediction conflicts with phishing/scam indicators: "
                + ", ".join(sorted(strong_content))
            )

    level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    threshold = str(policy["review_at_or_above_risk_level"])
    if prediction != "phishing" and level_order.get(risk_level, -1) >= level_order.get(threshold, 2):
        review_reasons.append(f"risk level {risk_level} conflicts with automatic delivery action")

    spam_review_threshold = float(policy["normal_spam_probability_review_threshold"])
    if prediction == "normal" and class_probabilities.get("spam", 0.0) >= spam_review_threshold:
        review_reasons.append("calibrated spam probability is close to the normal decision boundary")
    return ("REVIEW", list(dict.fromkeys(review_reasons))) if review_reasons else (base, [])


class EmailSecurityInference:
    """Load one frozen model and expose a stable, UI-independent analysis API."""

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        risk_config_path: Path = DEFAULT_RISK_CONFIG_PATH,
        security_config_path: Path = DEFAULT_SECURITY_CONFIG_PATH,
        config_path: Path | None = None,
        model: Any | None = None,
        metadata: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.risk_config_path = Path(risk_config_path)
        self.security_config_path = Path(security_config_path)
        self.model = model if model is not None else joblib.load(self.model_path)

        if metadata is not None:
            self.metadata = metadata
        elif model is not None:
            self.metadata = _test_metadata(self.model)
        else:
            self.metadata = _read_json(self.metadata_path)
            expected_hash = str(self.metadata.get("artifact_sha256", ""))
            if not expected_hash or sha256_file(self.model_path) != expected_hash:
                raise RuntimeError("Production model hash does not match model_metadata.json")

        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_risk_config(Path(config_path))
        else:
            risk_config = load_risk_config(self.risk_config_path)
            security_config = load_security_config(self.security_config_path)
            self.config = _merge_configs(risk_config, security_config)

        required_config = {
            "url_rules", "sender_rules", "content_rules", "risk_engine",
            "explainability", "recommendation_policy",
        }
        missing = required_config - set(self.config)
        legacy_optional = {"sender_rules", "recommendation_policy"}
        if missing and (model is None or not missing <= legacy_optional):
            raise ValueError(f"Inference configuration is missing sections: {sorted(missing)}")
        if missing:
            self.config = {
                **self.config,
                "sender_rules": {"maximum_at_signs": 1},
                "recommendation_policy": {
                    "normal_review_content_codes": [],
                    "normal_review_url_codes": [],
                    "normal_review_sender_codes": [],
                    "spam_review_content_codes": [],
                    "commercial_code": "commercial_promotion",
                    "minimum_commercial_matches_for_review": 2,
                    "normal_spam_probability_review_threshold": 0.45,
                    "review_at_or_above_risk_level": "CRITICAL",
                },
            }
        self.explainer = LinearFeatureExplainer(self.model)
        self._validate_model_contract()

    def _validate_model_contract(self) -> None:
        classes = _model_classes(self.model)
        metadata_classes = [str(value) for value in self.metadata.get("class_names", [])]
        if set(classes) != set(RECOMMENDED_ACTIONS):
            raise RuntimeError(f"Model classes do not match the inference contract: {classes}")
        if metadata_classes and set(metadata_classes) != set(classes):
            raise RuntimeError("Model classes do not match model metadata")
        if self.metadata.get("model_version") != "test-double" and not hasattr(
            self.model, "predict_proba"
        ):
            raise RuntimeError("Production model must expose calibrated predict_proba")

    def _predict(
        self, model_input: pd.DataFrame
    ) -> tuple[str, float | None, dict[str, float]]:
        prediction = str(self.model.predict(model_input)[0])
        if prediction not in RECOMMENDED_ACTIONS:
            raise RuntimeError(f"Model returned an unsupported internal class: {prediction}")
        if not hasattr(self.model, "predict_proba"):
            return prediction, None, {}
        probability_array = np.asarray(self.model.predict_proba(model_input), dtype=float)
        classes = _model_classes(self.model)
        if probability_array.shape != (1, len(classes)):
            raise RuntimeError("Model probability output does not align with classifier classes")
        if not np.isfinite(probability_array).all() or (
            (probability_array < 0.0).any() or (probability_array > 1.0).any()
        ):
            raise RuntimeError("Model returned invalid class probabilities")
        if not np.allclose(probability_array.sum(axis=1), 1.0, atol=1e-7):
            raise RuntimeError("Calibrated class probabilities do not sum to one")
        probabilities = {
            label: float(probability)
            for label, probability in zip(classes, probability_array[0], strict=True)
        }
        return prediction, probabilities[prediction], probabilities

    def _language_support(self, detected_language: str) -> dict[str, Any]:
        support = self.metadata.get("supported_content_languages", {})
        metadata_value = support.get(detected_language, {}) if isinstance(support, dict) else {}
        if isinstance(metadata_value, str):
            metadata_value = {"status": metadata_value}
        status = str(metadata_value.get("status", "unknown"))
        limitations: list[str] = []
        if detected_language == "vi":
            limitations.extend(
                [
                    "Vietnamese training and evaluation are predominantly translated data.",
                    "The held-out translated-Vietnamese subset contained no phishing examples.",
                    str(self.metadata.get(
                        "native_benchmark_status",
                        "Native Vietnamese benchmark is not available.",
                    )),
                ]
            )
        elif detected_language == "mixed":
            limitations.append("Mixed-language email has not been independently benchmarked.")
        elif detected_language == "unknown":
            limitations.append("Content language could not be determined reliably.")
        return {
            "status": status,
            "metadata": metadata_value,
            "limitations": list(dict.fromkeys(limitations)),
        }

    def analyze_email(
        self,
        *,
        sender: object = "",
        subject: object = "",
        body: object = "",
        parsed_eml: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze an email; security rules never overwrite the ML prediction."""
        if parsed_eml is not None:
            sender = sender or parsed_eml.get("sender", "")
            subject = subject or parsed_eml.get("subject", "")
            body = body or parsed_eml.get("body", "")
        raw_sender = "" if sender is None else str(sender)
        raw_subject = "" if subject is None else str(subject)
        raw_body = "" if body is None else str(body)
        if not normalize_text(raw_subject) and not normalize_text(raw_body):
            raise ValueError("At least one of subject or body must contain text")

        email_text = f"{raw_subject}\n{raw_body}"
        language_detection = detect_language_details(email_text)
        detected_language = language_detection.language
        url_analyses = analyze_urls(email_text, self.config["url_rules"])
        content_analysis = analyze_content(
            email_text, self.config["content_rules"], detected_language=detected_language
        )
        sender_analysis = analyze_sender(raw_sender, self.config["sender_rules"])

        # The frozen pipeline owns subject/body composition and TF-IDF transformation.
        model_input = pd.DataFrame([{"subject": raw_subject, "body": raw_body}])
        prediction, confidence, probabilities = self._predict(model_input)

        explanation_config = self.config["explainability"]
        model_explanation = self.explainer.explain_structured(
            model_input,
            prediction,
            max_features=int(explanation_config["max_model_features"]),
            include_character_features=bool(explanation_config["include_character_features"]),
            minimum_positive_contribution=float(
                explanation_config["minimum_positive_contribution"]
            ),
        )
        content_findings = content_analysis["findings"]
        sender_findings = sender_analysis["findings"]
        reasons = combine_explanations(
            model_explanation["features"],
            url_analyses,
            content_findings,
            maximum_security_reasons=int(explanation_config["max_security_reasons"]),
            sender_findings=sender_findings,
        )
        risk = calculate_risk(
            prediction=prediction,
            class_probabilities=probabilities,
            url_analyses=url_analyses,
            content_findings=content_findings,
            sender_findings=sender_findings,
            rules=self.config,
        )
        recommended_action, recommendation_reasons = _recommendation(
            prediction=prediction,
            risk_level=risk.level,
            class_probabilities=probabilities,
            content_findings=content_findings,
            url_analyses=url_analyses,
            sender_findings=sender_findings,
            policy=self.config["recommendation_policy"],
        )
        language_support = self._language_support(detected_language)
        limitations = [str(value) for value in self.metadata.get("limitations", [])]
        limitations.extend(language_support["limitations"])
        if not self.explainer.supported:
            limitations.append(self.explainer.limitation)

        return {
            "model_name": model_display_name(self.metadata),
            "model_version": str(self.metadata.get("model_version", "unknown")),
            "detected_language": detected_language,
            "language_detection": language_detection.to_dict(),
            "language_support_status": language_support,
            "prediction": prediction,
            "confidence": confidence,
            "class_probabilities": probabilities,
            "probability_source": "calibrated_predict_proba",
            "risk_score": risk.score,
            "risk_level": risk.level,
            "risk_score_type": "decision_support_heuristic",
            "risk_components": risk.components,
            "model_explanation": model_explanation,
            "security_findings": _flatten_security_findings(
                content_findings, url_analyses, sender_findings
            ),
            "reasons": reasons,
            "urls": [str(analysis["url"]) for analysis in url_analyses],
            "url_findings": url_analyses,
            "sender_analysis": sender_analysis,
            "sender_findings": sender_findings,
            "content_findings": content_findings,
            "content_statistics": content_analysis["statistics"],
            "base_recommended_action": RECOMMENDED_ACTIONS[prediction],
            "recommended_action": recommended_action,
            "recommendation_reasons": recommendation_reasons,
            "limitations": list(dict.fromkeys(limitations)),
            "model_explanation_supported": self.explainer.supported,
            "model_explanation_method": self.explainer.method,
        }

    def analyze_eml_bytes(self, data: bytes) -> dict[str, Any]:
        """Parse and analyze raw .eml bytes."""
        return self.analyze_email(parsed_eml=parse_eml_bytes(data))

    def analyze_eml_file(self, path: Path) -> dict[str, Any]:
        """Parse and analyze one local .eml file."""
        return self.analyze_email(parsed_eml=parse_eml_file(path))


@lru_cache(maxsize=8)
def _load_inference_engine_cached(
    model_path: str,
    metadata_path: str,
    risk_config_path: str,
    security_config_path: str,
    model_fingerprint: tuple[str, int, int, str],
    metadata_fingerprint: tuple[str, int, int, str],
    risk_fingerprint: tuple[str, int, int, str],
    security_fingerprint: tuple[str, int, int, str],
) -> EmailSecurityInference:
    del model_fingerprint, metadata_fingerprint, risk_fingerprint, security_fingerprint
    return EmailSecurityInference(
        model_path=Path(model_path),
        metadata_path=Path(metadata_path),
        risk_config_path=Path(risk_config_path),
        security_config_path=Path(security_config_path),
    )


def load_inference_engine(
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    risk_config_path: Path = DEFAULT_RISK_CONFIG_PATH,
    security_config_path: Path = DEFAULT_SECURITY_CONFIG_PATH,
) -> EmailSecurityInference:
    """Load a cached engine; cache keys change whenever an artifact file changes."""
    paths = [
        Path(model_path).resolve(),
        Path(metadata_path).resolve(),
        Path(risk_config_path).resolve(),
        Path(security_config_path).resolve(),
    ]
    fingerprints = [_file_fingerprint(path) for path in paths]
    return _load_inference_engine_cached(
        *(str(path) for path in paths),
        *fingerprints,
    )


def analyze_email(
    *,
    sender: object = "",
    subject: object = "",
    body: object = "",
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    risk_config_path: Path = DEFAULT_RISK_CONFIG_PATH,
    security_config_path: Path = DEFAULT_SECURITY_CONFIG_PATH,
) -> dict[str, Any]:
    """Analyze one email through the single cached production inference layer."""
    engine = load_inference_engine(
        model_path, metadata_path, risk_config_path, security_config_path
    )
    return engine.analyze_email(sender=sender, subject=subject, body=body)
