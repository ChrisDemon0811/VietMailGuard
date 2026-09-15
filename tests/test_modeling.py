from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from vietmailguard.modeling import (
    EmailTextComposer,
    assert_disjoint_splits,
    build_experiment_pipeline,
    compute_metrics,
    iter_experiment_specs,
    load_json,
    rank_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "model_experiments.json"


def test_email_text_composer_preserves_subject_and_body() -> None:
    frame = pd.DataFrame(
        [
            {"subject": "Quarterly review", "body": "Agenda follows."},
            {"subject": "", "body": "Body only"},
        ]
    )
    values = EmailTextComposer().fit_transform(frame)
    assert values.tolist() == ["Quarterly review\nAgenda follows.", "\nBody only"]


def test_tfidf_is_fitted_only_on_rows_passed_to_pipeline_fit() -> None:
    train = pd.DataFrame(
        [
            {"subject": "alpha normal", "body": "meeting agenda"},
            {"subject": "beta spam", "body": "commercial offer"},
            {"subject": "gamma phishing", "body": "verify account"},
        ]
    )
    pipeline = Pipeline(
        [
            ("compose_text", EmailTextComposer()),
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("classifier", MultinomialNB()),
        ]
    )
    pipeline.fit(train, ["normal", "spam", "phishing"])
    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    assert "alpha" in vocabulary
    assert "validationonlytoken" not in vocabulary
    pipeline.predict(
        pd.DataFrame([{"subject": "validationonlytoken", "body": "meeting"}])
    )
    assert "validationonlytoken" not in pipeline.named_steps["tfidf"].vocabulary_


def test_config_expands_to_required_nine_experiments() -> None:
    config = load_json(CONFIG_PATH)
    specs = iter_experiment_specs(config)
    assert len(specs) == 9
    assert len({spec.feature_name for spec in specs}) == 3
    assert len({spec.classifier_name for spec in specs}) == 3
    combined = next(spec for spec in specs if spec.feature_name == "word_character_tfidf")
    pipeline = build_experiment_pipeline(combined, config)
    assert isinstance(pipeline.named_steps["tfidf"], FeatureUnion)


def test_model_selection_config_does_not_reference_test_data() -> None:
    config_text = CONFIG_PATH.read_text(encoding="utf-8").casefold()
    assert "test.csv" not in config_text
    assert "held_out_test" not in config_text


def test_calibrated_linear_svm_exposes_probabilities() -> None:
    config = load_json(CONFIG_PATH)
    svm_spec = next(
        spec
        for spec in iter_experiment_specs(config)
        if spec.feature_name == "word_tfidf" and spec.classifier_name == "linear_svm"
    )
    local_config = json.loads(json.dumps(config))
    local_config["features"]["word_tfidf"]["min_df"] = 1
    local_config["features"]["word_tfidf"]["max_features"] = 100
    local_config["classifiers"]["linear_svm"]["calibration_cv"] = 2
    rows = []
    labels = []
    for label, phrase in (
        ("normal", "team meeting schedule"),
        ("spam", "discount product sale"),
        ("phishing", "verify password account"),
    ):
        for index in range(4):
            rows.append({"subject": phrase, "body": f"message {index} {phrase}"})
            labels.append(label)
    pipeline = build_experiment_pipeline(svm_spec, local_config)
    pipeline.fit(pd.DataFrame(rows), labels)
    probabilities = pipeline.predict_proba(pd.DataFrame(rows[:2]))
    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_metrics_include_all_required_values() -> None:
    labels = ["normal", "spam", "phishing"]
    metrics = compute_metrics(
        ["normal", "spam", "phishing", "phishing"],
        ["normal", "spam", "normal", "phishing"],
        labels,
    )
    assert {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "per_class",
        "confusion_matrix",
    } <= set(metrics)
    assert list(metrics["per_class"]) == labels
    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)


def test_ranking_uses_macro_f1_then_phishing_recall() -> None:
    comparison = pd.DataFrame(
        [
            {
                "experiment_id": "a",
                "macro_f1": 0.8,
                "phishing_recall": 0.7,
                "weighted_f1": 0.9,
                "accuracy": 0.9,
            },
            {
                "experiment_id": "b",
                "macro_f1": 0.8,
                "phishing_recall": 0.8,
                "weighted_f1": 0.8,
                "accuracy": 0.8,
            },
        ]
    )
    config = load_json(CONFIG_PATH)
    ranked = rank_comparison(comparison, config["selection"])
    assert ranked.iloc[0]["experiment_id"] == "b"
    assert bool(ranked.iloc[0]["selected"])


def test_split_overlap_is_rejected() -> None:
    left = pd.DataFrame({"group_id": ["g1"], "content_hash": ["h1"]})
    right = pd.DataFrame({"group_id": ["g1"], "content_hash": ["h2"]})
    with pytest.raises(ValueError, match="Leakage detected"):
        assert_disjoint_splits(left, right, left_name="train", right_name="validation")
