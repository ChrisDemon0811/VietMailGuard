"""Feature-based model explanations combined with observed security indicators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


def _linear_parameters(classifier: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if hasattr(classifier, "coef_") and hasattr(classifier, "classes_"):
        return np.asarray(classifier.coef_), np.asarray(classifier.classes_, dtype=object)
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    if not calibrated:
        return None
    estimators = [item.estimator for item in calibrated]
    if not all(hasattr(estimator, "coef_") and hasattr(estimator, "classes_") for estimator in estimators):
        return None
    classes = np.asarray(estimators[0].classes_, dtype=object)
    if any(not np.array_equal(classes, np.asarray(estimator.classes_)) for estimator in estimators[1:]):
        return None
    coefficients = np.mean([np.asarray(estimator.coef_) for estimator in estimators], axis=0)
    return coefficients, classes


def _coefficient_for_class(
    coefficients: np.ndarray, classes: np.ndarray, predicted_class: str
) -> np.ndarray | None:
    matching = np.flatnonzero(classes == predicted_class)
    if not len(matching):
        return None
    class_index = int(matching[0])
    if coefficients.shape[0] == len(classes):
        return coefficients[class_index]
    if coefficients.shape[0] == 1 and len(classes) == 2:
        return coefficients[0] if class_index == 1 else -coefficients[0]
    return None


def _readable_feature(name: str, include_character_features: bool) -> tuple[str, str] | None:
    if name.startswith("word__"):
        return name.removeprefix("word__"), "word"
    if name.startswith("character__"):
        if not include_character_features:
            return None
        return name.removeprefix("character__"), "character"
    return name, "word"


class LinearFeatureExplainer:
    """Explain positive sparse-feature contributions for supported linear pipelines."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        classifier = getattr(pipeline, "named_steps", {}).get("classifier")
        self.method = "unsupported"
        self.limitation = "The fitted pipeline does not expose compatible linear coefficients."
        parameters = _linear_parameters(classifier)
        transformer = getattr(pipeline, "named_steps", {}).get("tfidf")
        if parameters is None or transformer is None or not hasattr(transformer, "get_feature_names_out"):
            self.supported = False
            self.coefficients = np.empty((0, 0))
            self.classes = np.empty(0, dtype=object)
            self.feature_names = np.empty(0, dtype=object)
            return
        self.coefficients, self.classes = parameters
        self.feature_names = np.asarray(transformer.get_feature_names_out(), dtype=object)
        self.supported = self.coefficients.shape[1] == len(self.feature_names)
        if self.supported:
            self.method = "observed_tfidf_value_times_underlying_linear_coefficient"
            self.limitation = (
                "Contributions explain the underlying linear classifier score, not a direct "
                "decomposition of the sigmoid-calibrated probability."
            )
        else:
            self.limitation = "Linear coefficient and TF-IDF feature dimensions do not align."

    def explain(
        self,
        rows: pd.DataFrame,
        predicted_class: str,
        *,
        max_features: int = 5,
        include_character_features: bool = False,
        minimum_positive_contribution: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return only observed features with positive contribution to the predicted class."""
        if not self.supported or len(rows) != 1:
            return []
        class_coefficients = _coefficient_for_class(
            self.coefficients, self.classes, predicted_class
        )
        if class_coefficients is None:
            return []
        transformed = self.pipeline[:-1].transform(rows)
        if sparse.issparse(transformed):
            contributions = transformed.getrow(0).multiply(class_coefficients).toarray().ravel()
        else:
            contributions = np.asarray(transformed)[0] * class_coefficients
        candidate_indices = np.flatnonzero(contributions > minimum_positive_contribution)
        ranked_indices = candidate_indices[np.argsort(contributions[candidate_indices])[::-1]]

        reasons: list[dict[str, Any]] = []
        for feature_index in ranked_indices:
            readable = _readable_feature(
                str(self.feature_names[feature_index]), include_character_features
            )
            if readable is None:
                continue
            feature, feature_kind = readable
            if not feature.strip() or feature.strip().isdigit():
                continue
            contribution = float(contributions[feature_index])
            reasons.append(
                {
                    "source": "model",
                    "category": "model_feature",
                    "feature": feature,
                    "feature_kind": feature_kind,
                    "contribution": contribution,
                    "description": (
                        f"Observed feature '{feature}' contributes positively to the "
                        f"underlying linear score for '{predicted_class}'."
                    ),
                }
            )
            if len(reasons) >= max_features:
                break
        return reasons

    def explain_structured(
        self,
        rows: pd.DataFrame,
        predicted_class: str,
        *,
        max_features: int = 5,
        include_character_features: bool = False,
        minimum_positive_contribution: float = 0.0,
    ) -> dict[str, Any]:
        """Return explanation metadata together with real observed contributions."""
        items = self.explain(
            rows,
            predicted_class,
            max_features=max_features,
            include_character_features=include_character_features,
            minimum_positive_contribution=minimum_positive_contribution,
        )
        return {
            "supported": self.supported,
            "method": self.method,
            "target_class": predicted_class,
            "features": items,
            "limitation": self.limitation,
        }


def security_rule_reasons(
    url_analyses: list[dict[str, Any]],
    content_findings: list[dict[str, Any]],
    *,
    maximum: int,
    sender_findings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert actual rule matches to explanations without maliciousness claims."""
    reasons: list[dict[str, Any]] = []
    for finding in content_findings:
        reasons.append(
            {
                "source": "content_rule",
                "category": str(finding["code"]),
                "description": str(finding["description"]),
                "evidence": list(finding.get("evidence", [])),
                "rule_ids": list(finding.get("rule_ids", [])),
            }
        )
    for analysis in url_analyses:
        for finding in analysis.get("findings", []):
            reasons.append(
                {
                    "source": "url_rule",
                    "category": str(finding["code"]),
                    "description": str(finding["description"]),
                    "evidence": str(finding.get("evidence", "")),
                    "url": str(analysis.get("url", "")),
                }
            )
    for finding in sender_findings or []:
        reasons.append(
            {
                "source": "sender_rule",
                "category": str(finding["code"]),
                "description": str(finding["description"]),
                "evidence": list(finding.get("evidence", [])),
                "rule_ids": list(finding.get("rule_ids", [])),
            }
        )
    return reasons[:maximum]


def combine_explanations(
    model_reasons: list[dict[str, Any]],
    url_analyses: list[dict[str, Any]],
    content_findings: list[dict[str, Any]],
    *,
    maximum_security_reasons: int,
    sender_findings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Combine real model contributions with actually triggered security rules."""
    return [
        *model_reasons,
        *security_rule_reasons(
            url_analyses,
            content_findings,
            maximum=maximum_security_reasons,
            sender_findings=sender_findings,
        ),
    ]
