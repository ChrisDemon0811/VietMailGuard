from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vietmailguard.dashboard_data import (
    aggregate_split_distribution,
    duplicate_summary,
    load_dataset_dashboard_data,
    load_model_dashboard_data,
    split_summary_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DASHBOARD_ARTIFACTS = [
    PROJECT_ROOT / "models" / "v2_bilingual" / "model_metadata.json",
    PROJECT_ROOT / "results" / "v2_bilingual" / "final_test_metrics.json",
    PROJECT_ROOT / "results" / "v2_bilingual" / "tfidf_model_comparison.csv",
    PROJECT_ROOT / "results" / "v2_bilingual" / "final_confusion_matrix.csv",
    PROJECT_ROOT / "results" / "v2_bilingual" / "final_test_classification_report.csv",
    PROJECT_ROOT / "results" / "v2_bilingual" / "robustness_metrics.csv",
    PROJECT_ROOT / "results" / "v2_bilingual" / "short_form_challenge_metrics.csv",
]
DATASET_DASHBOARD_ARTIFACTS = [
    PROJECT_ROOT / "data" / "splits" / "v2" / f"{name}.csv"
    for name in ("train", "validation", "test")
] + [
    PROJECT_ROOT / "data" / "curated" / "v2" / "Vietnamese_Curated_Dataset.csv",
    PROJECT_ROOT / "data" / "processed" / "Vietnamese_Translated_Augmentation.csv",
]


@pytest.mark.skipif(
    not all(path.exists() for path in MODEL_DASHBOARD_ARTIFACTS),
    reason="optional saved dashboard artifacts are not all distributed with a clean clone",
)
def test_model_dashboard_uses_saved_test_metrics() -> None:
    saved = json.loads(
        (PROJECT_ROOT / "results" / "v2_bilingual" / "final_test_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    dashboard = load_model_dashboard_data(PROJECT_ROOT)
    assert dashboard.test_metrics is not None
    assert dashboard.test_metrics["macro_f1"] == saved["overall_test_metrics"]["macro_f1"]
    assert dashboard.translated_vietnamese_metrics is not None
    assert dashboard.translated_vietnamese_metrics["rows"] == 179
    assert len(dashboard.comparison) == 9
    assert not dashboard.robustness_metrics.empty
    assert not dashboard.short_form_metrics.empty
    assert dashboard.errors == []


@pytest.mark.skipif(
    not all(path.exists() for path in DATASET_DASHBOARD_ARTIFACTS),
    reason="local generated dataset artifacts are not distributed with a clean clone",
)
def test_dataset_dashboard_uses_only_eligible_v2_splits() -> None:
    dashboard = load_dataset_dashboard_data(PROJECT_ROOT)
    assert dashboard.errors == []
    assert dashboard.split_report is not None
    assert dashboard.split_report["total_rows"] == 44_717
    assert dashboard.split_report["leakage_check"]["passed"] is True
    assert dashboard.summary["english_samples"] == 42_112
    assert dashboard.summary["translated_vietnamese_samples"] == 2_605
    assert dashboard.summary["review_rows"] == 1_330
    assert dashboard.summary["excluded_rows"] == 1
    assert dashboard.summary["augmentation_eligible"] == 1_460


def test_missing_dashboard_artifacts_return_empty_states(tmp_path: Path) -> None:
    model_data = load_model_dashboard_data(tmp_path)
    dataset_data = load_dataset_dashboard_data(tmp_path)
    assert model_data.test_metrics is None
    assert model_data.comparison.empty
    assert model_data.errors
    assert dataset_data.split_report is None
    assert dataset_data.standardization.empty
    assert dataset_data.errors


def test_split_distribution_aggregation_and_summary() -> None:
    report = {
        "splits": {
            "train": {
                "rows": 7,
                "groups": 6,
                "actual_ratio": 0.7,
                "target_ratio": 0.7,
                "class_distribution": {"normal": {"count": 4}, "spam": {"count": 3}},
            },
            "validation": {
                "rows": 2,
                "groups": 2,
                "actual_ratio": 0.2,
                "target_ratio": 0.15,
                "class_distribution": {"normal": {"count": 1}, "spam": {"count": 1}},
            },
            "test": {
                "rows": 1,
                "groups": 1,
                "actual_ratio": 0.1,
                "target_ratio": 0.15,
                "class_distribution": {"normal": {"count": 1}},
            },
        }
    }
    distribution = aggregate_split_distribution(report, "class_distribution")
    assert dict(zip(distribution["category"], distribution["count"], strict=True)) == {
        "normal": 6,
        "spam": 4,
    }
    summary = split_summary_frame(report)
    assert summary["split"].tolist() == ["train", "validation", "test"]
    assert summary["rows"].sum() == 10


def test_duplicate_summary_reads_reported_actions() -> None:
    duplicates = pd.DataFrame(
        [
            {"duplicate_type": "exact", "group_id": "g1"},
            {"duplicate_type": "exact", "group_id": "g1"},
            {"duplicate_type": "template", "group_id": "g2"},
        ]
    )
    removed = pd.DataFrame(
        [{"reason": "exact_duplicate"}, {"reason": "empty_or_placeholder_body"}]
    )
    assert duplicate_summary(duplicates, removed) == {
        "reported_rows": 3,
        "exact_groups": 1,
        "template_groups": 1,
        "exact_rows_removed": 1,
    }
