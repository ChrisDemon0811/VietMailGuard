from __future__ import annotations

from pathlib import Path

import pandas as pd

from vietmailguard.modeling import build_experiment_pipeline, iter_experiment_specs, load_json
from vietmailguard.v2_modeling import (
    assert_no_cross_split_values,
    assert_only_model_selection_inputs,
    classifier_confidence_capability,
    compute_subset_metrics,
    rank_bilingual_comparison,
    validation_subset_masks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "v2_tfidf_experiments.json"


def test_v2_config_accepts_only_train_and_validation() -> None:
    config = load_json(CONFIG_PATH)
    assert_only_model_selection_inputs(config)
    assert set(config["data"]) == {"train", "validation"}
    assert len(iter_experiment_specs(config)) == 9


def test_v2_linear_svm_is_not_presented_as_probability_model() -> None:
    config = load_json(CONFIG_PATH)
    spec = next(
        item
        for item in iter_experiment_specs(config)
        if item.feature_name == "word_tfidf" and item.classifier_name == "linear_svm"
    )
    pipeline = build_experiment_pipeline(spec, config)
    assert classifier_confidence_capability(pipeline) == "unavailable_uncalibrated"
    assert not hasattr(pipeline.named_steps["classifier"], "predict_proba")


def test_absent_subset_class_metrics_are_unavailable() -> None:
    metrics = compute_subset_metrics(
        ["normal", "normal", "spam"],
        ["normal", "spam", "spam"],
        ["normal", "spam", "phishing"],
    )
    assert metrics["supported_classes"] == ["normal", "spam"]
    assert metrics["three_class_macro_f1"] is None
    assert metrics["per_class"]["phishing"]["recall"] is None
    assert metrics["per_class"]["phishing"]["support"] == 0


def test_validation_origin_masks_are_disjoint_where_expected() -> None:
    frame = pd.DataFrame(
        [
            {"language": "en", "data_origin": "", "source": "Enron", "augmentation_type": ""},
            {"language": "vi", "data_origin": "translated", "source": "data_vi", "augmentation_type": ""},
            {
                "language": "vi",
                "data_origin": "translated",
                "source": "controlled_translation",
                "augmentation_type": "controlled_translation",
            },
        ]
    )
    masks = validation_subset_masks(frame)
    assert masks["english_original_artifact"].tolist() == [True, False, False]
    assert masks["existing_translated_vietnamese"].tolist() == [False, True, False]
    assert masks["controlled_translated_augmentation"].tolist() == [False, False, True]


def test_cross_split_leakage_key_is_rejected() -> None:
    train = pd.DataFrame({"final_group_id": ["g1", "g2"]})
    validation = pd.DataFrame({"final_group_id": ["g2", "g3"]})
    try:
        assert_no_cross_split_values(train, validation, ["final_group_id"])
    except ValueError as exc:
        assert "Leakage detected" in str(exc)
    else:
        raise AssertionError("Expected leakage to be rejected")


def test_bilingual_ranking_uses_declared_priority() -> None:
    frame = pd.DataFrame(
        [
            {
                "experiment_id": "a",
                "macro_f1": 0.9,
                "phishing_recall": 0.8,
                "language_balance_gap": 0.01,
                "normal_false_positive_rate": 0.02,
                "deployability_rank": 0,
            },
            {
                "experiment_id": "b",
                "macro_f1": 0.9,
                "phishing_recall": 0.9,
                "language_balance_gap": 0.05,
                "normal_false_positive_rate": 0.01,
                "deployability_rank": 1,
            },
        ]
    )
    ranked = rank_bilingual_comparison(frame)
    assert ranked.iloc[0]["experiment_id"] == "b"
    assert bool(ranked.iloc[0]["recommended_tfidf_candidate"])
