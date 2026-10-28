"""Opt-in V2.1 development-candidate inference; V2 remains the default.

The configured candidate is verified before deserializing. It is neither a
production release nor a calibrated probability model. No test data is read.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from vietmailguard.explainability import LinearFeatureExplainer
from vietmailguard.file_utils import sha256_file
from vietmailguard.inference import EmailSecurityInference
from vietmailguard.modeling import EmailTextComposer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG = PROJECT_ROOT / "config" / "v21" / "runtime_model.json"
_CLASSES = {"normal", "spam", "phishing"}
_HAN = re.compile(r"[\u3400-\u9fff]")
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
_LINK_OR_ADDRESS = re.compile(r"(?:https?://|www\.)\S+|\b\S+@\S+\b", re.IGNORECASE)


class V21ModelUnavailable(RuntimeError):
    """An explicitly configured development candidate failed verification."""


def _verified_paths(config_path: Path, root: Path) -> tuple[dict[str, Any], Path, Path]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config["schema_version"] != 1
                or config["scope"] != "development_candidate_only"
                or config["probability_status"] != "unavailable_uncalibrated"
                or set(config["supported_classes"]) != _CLASSES):
            raise ValueError("Unexpected runtime model policy")
        artifact = (root / config["artifact"]).resolve()
        metrics_path = (root / config["validation_metrics"]).resolve()
        training_path = (root / config["training_config"]).resolve()
        candidate_root = (root / "models" / "v21_multilingual" / "candidates").resolve()
        results_root = (root / "results" / "v21_multilingual").resolve()
        configuration_root = (root / "config" / "v21").resolve()
        if (not artifact.is_relative_to(candidate_root)
                or not metrics_path.is_relative_to(results_root)
                or not training_path.is_relative_to(configuration_root)):
            raise ValueError("V2.1 artifact path is outside its isolated namespace")
        if sha256_file(artifact) != config["artifact_sha256"]:
            raise ValueError("V2.1 artifact SHA-256 mismatch")
        if sha256_file(metrics_path) != config["validation_metrics_sha256"]:
            raise ValueError("V2.1 validation manifest SHA-256 mismatch")
        if sha256_file(training_path) != config["training_config_sha256"]:
            raise ValueError("V2.1 training preprocessing SHA-256 mismatch")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (metrics.get("scope") != "development_validation_only"
                or metrics.get("config_sha256") != config["training_config_sha256"]
                or metrics.get("development_leakage_preflight", {}).get("passed") is not True
                or not any(row.get("candidate") == config["candidate_id"]
                           and row.get("artifact") == config["artifact"]
                           for row in metrics.get("candidates", []))):
            raise ValueError("Candidate is not in the verified development results")
        training = json.loads(training_path.read_text(encoding="utf-8"))
        limit = training.get("max_input_characters")
        if (training.get("scope") != "development_bootstrap_validation_only"
                or set(training.get("classes", [])) != _CLASSES
                or type(limit) is not int or limit <= 0):
            raise ValueError("V2.1 training preprocessing policy is incompatible")
        config = {**config, "verified_training_config": training}
        return config, artifact, metrics_path
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        raise V21ModelUnavailable("V2.1 development artifact is absent or unverified") from error


def model_availability(config_path: Path = DEFAULT_RUNTIME_CONFIG,
                       root: Path = PROJECT_ROOT) -> tuple[bool, str]:
    """Verify hash, validation provenance and deserialized pipeline contract."""
    try:
        load_v21_engine(config_path, root)
    except Exception:  # fail closed at the untrusted artifact boundary
        return False, "V2.1 artifact/configuration verification failed"
    return True, "V2.1 development candidate verified; not a production model"


def _validate_pipeline(model: Any, training_config: dict[str, Any] | None = None) -> None:
    if not isinstance(model, Pipeline) or list(model.named_steps) != [
            "compose_text", "tfidf", "classifier"]:
        raise V21ModelUnavailable("V2.1 pipeline structure is incompatible")
    if not isinstance(model.named_steps["compose_text"], EmailTextComposer):
        raise V21ModelUnavailable("V2.1 text composer is incompatible")
    composer = model.named_steps["compose_text"]
    if (composer.subject_column, composer.body_column) != ("subject", "body"):
        raise V21ModelUnavailable("V2.1 composer input columns are incompatible")
    union = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    if (not isinstance(union, FeatureUnion)
            or set(dict(union.transformer_list)) != {"word", "character"}
            or not all(isinstance(item, TfidfVectorizer)
                       for _, item in union.transformer_list)
            or not isinstance(classifier, LinearSVC)
            or len(classifier.classes_) != 3
            or {str(value) for value in classifier.classes_} != _CLASSES
            or hasattr(model, "predict_proba")):
        raise V21ModelUnavailable("V2.1 candidate class or calibration contract differs")
    feature_count = 0
    for name, vectorizer in union.transformer_list:
        check_is_fitted(vectorizer)
        feature_count += len(vectorizer.get_feature_names_out())
        expected_analyzer = "word" if name == "word" else "char_wb"
        if (vectorizer.analyzer != expected_analyzer
                or vectorizer.preprocessor is not None
                or vectorizer.tokenizer is not None):
            raise V21ModelUnavailable("V2.1 feature preprocessing is incompatible")
        if training_config is not None:
            expected = training_config["tfidf"][name]
            for parameter in ("ngram_range", "min_df", "max_features", "sublinear_tf", "dtype"):
                actual = getattr(vectorizer, parameter)
                target = expected[parameter]
                if parameter == "ngram_range":
                    target = tuple(target)
                elif parameter == "dtype":
                    actual, target = np.dtype(actual), np.dtype(target)
                if actual != target:
                    raise V21ModelUnavailable("V2.1 vectorizer differs from its training configuration")
    check_is_fitted(classifier)
    if classifier.coef_.shape != (3, feature_count):
        raise V21ModelUnavailable("V2.1 feature/classifier dimensions do not align")
    if training_config is not None:
        for parameter, expected in training_config["tfidf"]["svm"].items():
            if getattr(classifier, parameter) != expected:
                raise V21ModelUnavailable("V2.1 classifier differs from its training configuration")
        if classifier.random_state != training_config["seed"]:
            raise V21ModelUnavailable("V2.1 classifier seed differs from its training configuration")


def _bounded_model_input(frame: pd.DataFrame, maximum_characters: int) -> pd.DataFrame:
    """Mirror the verified development loader; never truncate stored email text."""
    result = frame.copy()
    for column in ("subject", "body"):
        result[column] = result[column].fillna("").astype(str).str.slice(0, maximum_characters)
    return result


class _DevelopmentFeatureExplainer(LinearFeatureExplainer):
    def __init__(self, pipeline: Pipeline, maximum_characters: int) -> None:
        super().__init__(pipeline)
        self.maximum_characters = maximum_characters

    def explain(self, rows: pd.DataFrame, predicted_class: str, **kwargs: Any) -> list[dict[str, Any]]:
        return super().explain(
            _bounded_model_input(rows, self.maximum_characters), predicted_class, **kwargs)


class V21DevelopmentInference(EmailSecurityInference):
    """Reuse the V2 security/risk pipeline while replacing only the ML artifact."""

    def __init__(self, model: Pipeline, version: str, *, maximum_characters: int = 4096,
                 artifact_provenance: dict[str, Any] | None = None) -> None:
        self.maximum_characters = maximum_characters
        self.artifact_provenance = dict(artifact_provenance or {})
        metadata = {
            "model_version": version,
            "class_names": sorted(_CLASSES),
            "training_languages": ["en", "vi"],
            "limitations": [
                "Development/Validation candidate only; no independent V2.1 final test.",
                "LinearSVC is uncalibrated; model confidence and class probabilities are unavailable.",
                "Vietnamese development data is translated; Chinese has no supervised development examples.",
                "V2 English/Vietnamese security rules are reused; Chinese-specific security rules are unavailable.",
            ],
        }
        super().__init__(model=model, metadata=metadata)
        self.explainer = _DevelopmentFeatureExplainer(model, maximum_characters)

    def _validate_model_contract(self) -> None:
        _validate_pipeline(self.model)

    def _predict(self, model_input: pd.DataFrame) -> tuple[str, None, dict[str, float]]:
        prediction = str(self.model.predict(
            _bounded_model_input(model_input, self.maximum_characters))[0])
        if prediction not in _CLASSES:
            raise V21ModelUnavailable("V2.1 candidate returned an unsupported class")
        return prediction, None, {}

    def _language_support(self, detected_language: str) -> dict[str, Any]:
        status = "experimental" if detected_language in {"en", "vi"} else "unvalidated"
        return {"status": status, "metadata": {"scope": "development_validation_only"},
                "limitations": ["No independent native Vietnamese or Chinese final benchmark exists."]}

    def analyze_email(self, **kwargs: Any) -> dict[str, Any]:
        result = super().analyze_email(**kwargs)
        parsed = kwargs.get("parsed_eml") or {}
        subject = str(kwargs.get("subject") or parsed.get("subject") or "")
        body = str(kwargs.get("body") or parsed.get("body") or "")
        text = f"{subject}\n{body}"
        if _HAN.search(text):
            language = "mixed" if _LATIN_WORD.search(_LINK_OR_ADDRESS.sub(" ", text)) else "zh"
            result["detected_language"] = language
            result["language_support_status"] = self._language_support(language)
            result["language_detection"] = {
                "language": language, "reason": "Han characters observed; independent benchmark unavailable"}
            result["limitations"] = list(dict.fromkeys(
                [*result["limitations"], *result["language_support_status"]["limitations"]]))
        result["model_name"] = "VietMailGuard V2.1 Development"
        result["probability_source"] = "unavailable_uncalibrated"
        result["confidence"] = None
        result["class_probabilities"] = {}
        result["model_scope"] = "development_candidate_only"
        result["model_artifact"] = self.artifact_provenance.copy()
        result["preprocessing"] = {
            "composer": "EmailTextComposer", "fields": ["subject", "body"],
            "maximum_characters_per_field": self.maximum_characters,
            "truncated_fields": [field for field, value in (("subject", subject), ("body", body))
                                 if len(value) > self.maximum_characters],
            "security_analysis_scope": "full_email_text",
        }
        if result["preprocessing"]["truncated_fields"]:
            result["limitations"].append(
                f"ML uses the first {self.maximum_characters} characters per field, matching "
                "development training; security analysis still examines the full email.")
        result["model_explanation"]["limitation"] = (
            "Contributions describe the underlying LinearSVC score, not probability.")
        return result


@lru_cache(maxsize=2)
def _load_verified(artifact: str, digest: str, version: str,
                   training_json: str, training_digest: str,
                   risk_digest: str, security_digest: str) -> V21DevelopmentInference:
    del risk_digest, security_digest  # included to invalidate cached security configuration
    training = json.loads(training_json)
    # Recheck immediately before deserializing, including on a changed cache key.
    if sha256_file(Path(artifact)) != digest:
        raise V21ModelUnavailable("V2.1 artifact changed during verification")
    model = joblib.load(artifact)
    _validate_pipeline(model, training)
    return V21DevelopmentInference(
        model, version, maximum_characters=training["max_input_characters"],
        artifact_provenance={"candidate_id": Path(artifact).stem,
                             "artifact_sha256": digest,
                             "training_config_sha256": training_digest})


def load_v21_engine(config_path: Path = DEFAULT_RUNTIME_CONFIG,
                    root: Path = PROJECT_ROOT) -> V21DevelopmentInference:
    config, artifact, _ = _verified_paths(Path(config_path), Path(root))
    try:
        return _load_verified(
            str(artifact), config["artifact_sha256"], config["model_version"],
            json.dumps(config["verified_training_config"], sort_keys=True),
            config["training_config_sha256"],
            sha256_file(PROJECT_ROOT / "config" / "risk_rules_v2.json"),
            sha256_file(PROJECT_ROOT / "config" / "security_rules_v2.json"))
    except Exception as error:  # never activate a partially loaded candidate
        raise V21ModelUnavailable("V2.1 candidate pipeline cannot be loaded") from error


def analyze_v21_email(*, sender: object = "", subject: object = "",
                      body: object = "") -> dict[str, Any]:
    """MailService-compatible entry point for an explicitly verified V2.1 mode."""
    return load_v21_engine().analyze_email(sender=sender, subject=subject, body=body)
