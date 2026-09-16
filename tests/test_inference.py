from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from vietmailguard.explainability import LinearFeatureExplainer
from vietmailguard.inference import (
    INFERENCE_RESULT_FIELDS,
    EmailSecurityInference,
    RECOMMENDED_ACTIONS,
)
from vietmailguard.modeling import EmailTextComposer
from vietmailguard.risk_engine import load_risk_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_risk_config(ROOT / "config" / "risk_rules.json")


class FixedNormalModel:
    classes_ = np.asarray(["normal", "phishing", "spam"], dtype=object)

    def __init__(self) -> None:
        self.named_steps = {"classifier": self}

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(["normal"] * len(rows), dtype=object)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return np.tile(np.asarray([[0.98, 0.01, 0.01]]), (len(rows), 1))


def test_inference_result_schema_and_recommended_action() -> None:
    engine = EmailSecurityInference(model=FixedNormalModel(), config=CONFIG)
    result = engine.analyze_email(
        sender="manager@example.com",
        subject="Meeting notes",
        body="Please review the notes at https://www.example.com/about.",
    )
    assert INFERENCE_RESULT_FIELDS <= set(result)
    assert result["prediction"] == "normal"
    assert result["recommended_action"] == "ALLOW"
    assert result["confidence"] == pytest.approx(0.98)
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0)
    assert 0 <= result["risk_score"] <= 100


def test_recommended_action_mapping_uses_internal_classes() -> None:
    assert RECOMMENDED_ACTIONS == {
        "normal": "ALLOW",
        "spam": "MOVE_TO_SPAM",
        "phishing": "QUARANTINE",
    }


def test_normal_url_does_not_override_ml_prediction() -> None:
    engine = EmailSecurityInference(model=FixedNormalModel(), config=CONFIG)
    result = engine.analyze_email(
        subject="Documentation",
        body="Read https://www.example.com/about for the project documentation.",
    )
    assert result["prediction"] == "normal"
    assert result["recommended_action"] == "ALLOW"
    assert result["url_findings"][0]["findings"] == []


def test_empty_email_is_rejected_before_prediction() -> None:
    engine = EmailSecurityInference(model=FixedNormalModel(), config=CONFIG)
    with pytest.raises(ValueError, match="subject or body"):
        engine.analyze_email(sender="sender@example.com", subject=" \n", body="\t")


def test_linear_explanation_uses_observed_model_features() -> None:
    rows = pd.DataFrame(
        [
            {"subject": "team meeting", "body": "project agenda notes"},
            {"subject": "project update", "body": "team schedule meeting"},
            {"subject": "meeting notes", "body": "project team agenda"},
            {"subject": "discount sale", "body": "product offer promotion"},
            {"subject": "product offer", "body": "discount sale promotion"},
            {"subject": "sale promotion", "body": "discount product offer"},
            {"subject": "verify account", "body": "password login security"},
            {"subject": "account login", "body": "verify password security"},
            {"subject": "password verification", "body": "account login security"},
        ]
    )
    labels = ["normal"] * 3 + ["spam"] * 3 + ["phishing"] * 3
    pipeline = Pipeline(
        [
            ("compose_text", EmailTextComposer()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(rows, labels)
    sample = pd.DataFrame([{"subject": "verify account", "body": "password login"}])
    predicted = str(pipeline.predict(sample)[0])
    reasons = LinearFeatureExplainer(pipeline).explain(sample, predicted, max_features=3)
    vocabulary = set(pipeline.named_steps["tfidf"].get_feature_names_out())
    assert reasons
    assert all(reason["feature"] in vocabulary for reason in reasons)
    assert all(reason["contribution"] > 0 for reason in reasons)


@pytest.mark.production_artifact
def test_production_model_runs_through_stable_inference_schema() -> None:
    engine = EmailSecurityInference()
    result = engine.analyze_email(
        subject="Project meeting notes",
        body="Hello team, here are the notes and action items from today's meeting.",
    )
    assert INFERENCE_RESULT_FIELDS <= set(result)
    assert result["prediction"] in {"normal", "spam", "phishing"}
    assert set(result["class_probabilities"]) == {"normal", "spam", "phishing"}
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0)
    assert result["model_explanation_supported"]
    assert any(reason["source"] == "model" for reason in result["reasons"])
    assert result["model_version"] == "2.0.0"
    assert result["probability_source"] == "calibrated_predict_proba"
