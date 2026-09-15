from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vietmailguard.v2_finalization import (
    OneShotGuard,
    assert_one_shot_not_started,
    build_frozen_pipeline,
    build_group_calibration_splits,
    classification_report_frame,
    evaluate_final_subsets,
    validate_controlled_test_descendants,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = PROJECT_ROOT / "config" / "v2_production_frozen.json"


def _synthetic_final_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "en-normal",
                "parent_id": "",
                "subject": "meeting",
                "body": "project agenda",
                "label": "normal",
                "language": "en",
                "data_origin": "",
                "source": "Enron",
                "augmentation_type": "",
            },
            {
                "id": "vi-spam",
                "parent_id": "",
                "subject": "khuyến mãi",
                "body": "ưu đãi sản phẩm",
                "label": "spam",
                "language": "vi",
                "data_origin": "translated",
                "source": "data_vi",
                "augmentation_type": "",
            },
            {
                "id": "en-phishing",
                "parent_id": "",
                "subject": "verify account",
                "body": "enter password",
                "label": "phishing",
                "language": "en",
                "data_origin": "",
                "source": "Nazario",
                "augmentation_type": "",
            },
        ]
    )


def test_group_calibration_never_splits_one_group_across_fold_sides() -> None:
    labels = ["normal", "spam", "phishing"] * 6
    groups = [f"group-{index}" for index in range(len(labels))]
    folds = build_group_calibration_splits(labels, groups, n_splits=3, seed=42)
    group_array = np.asarray(groups)
    seen: list[int] = []
    for fit, calibrate in folds:
        assert not (set(group_array[fit]) & set(group_array[calibrate]))
        seen.extend(calibrate.tolist())
    assert sorted(seen) == list(range(len(labels)))


def test_frozen_pipeline_returns_calibrated_probabilities() -> None:
    config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    labels: list[str] = []
    groups: list[str] = []
    phrases = {
        "normal": "team meeting project agenda",
        "spam": "commercial discount product promotion",
        "phishing": "verify password suspended account",
    }
    for label, phrase in phrases.items():
        for index in range(10):
            rows.append({"subject": phrase, "body": f"{phrase} example {index}"})
            labels.append(label)
            groups.append(f"{label}-{index}")
    folds = build_group_calibration_splits(labels, groups, n_splits=3, seed=42)
    local = json.loads(json.dumps(config))
    local["representation"]["min_df"] = 1
    local["representation"]["max_features"] = 200
    pipeline = build_frozen_pipeline(local, folds)
    pipeline.fit(pd.DataFrame(rows), labels)
    probabilities = pipeline.predict_proba(pd.DataFrame(rows[:3]))
    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_absent_vietnamese_phishing_is_reported_as_na() -> None:
    frame = _synthetic_final_frame().iloc[:2].copy()
    metrics = evaluate_final_subsets(frame, ["normal", "spam"], ["normal", "spam", "phishing"])
    vietnamese = metrics["translated_vietnamese_test"]
    assert vietnamese["per_class"]["phishing"]["recall"] is None
    assert vietnamese["three_class_macro_f1"] is None
    report = classification_report_frame(metrics, ["normal", "spam", "phishing"])
    row = report[
        report["evaluation_scope"].eq("translated_vietnamese_test")
        & report["class"].eq("phishing")
    ].iloc[0]
    assert row["availability_note"] == "N/A: class absent from subset"


def test_controlled_test_translation_requires_parent_in_same_test() -> None:
    development = pd.DataFrame({"id": ["development-parent"]})
    test = pd.DataFrame(
        [
            {
                "id": "test-parent",
                "parent_id": "",
                "source": "Enron",
                "augmentation_type": "",
            },
            {
                "id": "test-translation",
                "parent_id": "test-parent",
                "source": "controlled_translation",
                "augmentation_type": "controlled_translation",
            },
        ]
    )
    result = validate_controlled_test_descendants(development, test)
    assert result == {"controlled_test_rows": 1, "verified_test_parent_rows": 1}

    test.loc[1, "parent_id"] = "development-parent"
    with pytest.raises(ValueError, match="appears in development"):
        validate_controlled_test_descendants(development, test)


def test_one_shot_guard_refuses_started_or_completed_evaluation(tmp_path: Path) -> None:
    guard = OneShotGuard(
        started_path=tmp_path / "started.json",
        final_metrics_path=tmp_path / "metrics.json",
    )
    assert_one_shot_not_started(guard)
    guard.started_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="journal already exists"):
        assert_one_shot_not_started(guard)
    guard.started_path.unlink()
    guard.final_metrics_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exist"):
        assert_one_shot_not_started(guard)


def test_frozen_config_matches_saved_validation_winner() -> None:
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    validation = json.loads(
        (PROJECT_ROOT / "results" / "v2_bilingual" / "tfidf_validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert frozen["selected_experiment_id"] == validation["recommended_tfidf_candidate"][
        "experiment_id"
    ]
    assert frozen["held_out_test_accessed_at_selection"] is False
    assert frozen["post_test_tuning_allowed"] is False
