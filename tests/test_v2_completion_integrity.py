"""Integration checks for frozen Version 2 scientific artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from vietmailguard.bilingual_dataset import validate_bilingual_splits
from vietmailguard.dashboard_data import load_model_dashboard_data
from vietmailguard.file_utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]
V2_RESULTS = ROOT / "results" / "v2_bilingual"
V2_MODEL = ROOT / "models" / "v2_bilingual"
RAW_AUDIT_ARTIFACTS = [
    ROOT / "reports" / "dataset_audit.csv",
    ROOT / "reports" / "v2" / "vietnamese_dataset_audit.md",
] + [
    ROOT / "data" / "raw" / filename
    for filename in (
        "CEAS_08.csv",
        "Enron.csv",
        "Ling.csv",
        "Nazario.csv",
        "Nigerian_Fraud.csv",
        "SpamAssasin.csv",
        "data_vi.csv",
    )
]
V2_SPLIT_ARTIFACTS = [
    ROOT / "data" / "splits" / "v2" / f"{name}.csv"
    for name in ("train", "validation", "test")
]
SPLIT_COLUMNS = [
    "id",
    "parent_id",
    "source",
    "language",
    "data_origin",
    "augmentation_type",
    "label",
    "raw_label",
    "label_status",
    "label_provenance",
    "training_eligible",
    "content_hash",
    "group_id",
    "normalized_content_hash",
    "template_hash",
    "translation_group_id",
    "final_group_id",
    "split_constraint",
]


@pytest.fixture(scope="module")
def actual_splits() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(
            ROOT / "data" / "splits" / "v2" / f"{name}.csv",
            usecols=SPLIT_COLUMNS,
            keep_default_na=False,
        )
        for name in ("train", "validation", "test")
    }


@pytest.mark.skipif(
    not all(path.exists() for path in RAW_AUDIT_ARTIFACTS),
    reason="raw datasets and generated audit files are local-only artifacts",
)
def test_raw_files_match_saved_audit_hashes() -> None:
    english_audit = pd.read_csv(ROOT / "reports" / "dataset_audit.csv", dtype=str)
    for row in english_audit.itertuples(index=False):
        assert sha256_file(ROOT / "data" / "raw" / row.file) == row.sha256

    vietnamese_report = (
        ROOT / "reports" / "v2" / "vietnamese_dataset_audit.md"
    ).read_text(encoding="utf-8")
    match = re.search(r"SHA-256: `([0-9a-f]{64})`", vietnamese_report)
    assert match is not None
    assert sha256_file(ROOT / "data" / "raw" / "data_vi.csv") == match.group(1)


@pytest.mark.skipif(
    not all(path.exists() for path in V2_SPLIT_ARTIFACTS),
    reason="generated V2 split datasets are local-only artifacts",
)
def test_actual_v2_splits_preserve_eligibility_provenance_and_origin(
    actual_splits: dict[str, pd.DataFrame],
) -> None:
    leakage = validate_bilingual_splits(actual_splits)
    assert leakage["passed"] is True
    assert set(leakage["cross_split_overlap_counts"].values()) == {0}
    assert leakage["parent_translation_pairs_checked"] == 2_605
    assert leakage["train_only_rows_checked"] == 1_460

    combined = pd.concat(actual_splits.values(), ignore_index=True)
    assert set(combined["label"]) == {"normal", "spam", "phishing"}
    assert not combined["label_status"].isin({"review", "exclude"}).any()
    assert combined["label_provenance"].astype(str).str.strip().ne("").all()
    assert combined["raw_label"].astype(str).str.strip().ne("").all()
    vietnamese = combined.loc[combined["language"].eq("vi")]
    assert len(vietnamese) == 2_605
    assert set(vietnamese["data_origin"]) == {"translated"}
    assert "native" not in set(combined["data_origin"])
    controlled = combined.loc[
        combined["augmentation_type"].eq("controlled_translation")
    ]
    assert len(controlled) == 1_460
    assert set(controlled["source"]) == {"controlled_translation"}
    assert set(controlled["split_constraint"]) == {"train_only"}


def _expanded_confusion(matrix: np.ndarray, labels: list[str]) -> tuple[list[str], list[str]]:
    actual: list[str] = []
    predicted: list[str] = []
    for row_index, actual_label in enumerate(labels):
        for column_index, predicted_label in enumerate(labels):
            count = int(matrix[row_index, column_index])
            actual.extend([actual_label] * count)
            predicted.extend([predicted_label] * count)
    return actual, predicted


def test_frozen_metrics_recompute_and_agree_across_artifacts() -> None:
    final = json.loads((V2_RESULTS / "final_test_metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((V2_MODEL / "model_metadata.json").read_text(encoding="utf-8"))
    saved = final["overall_test_metrics"]
    labels = saved["confusion_matrix_label_order"]
    matrix = np.asarray(saved["confusion_matrix"], dtype=int)
    actual, predicted = _expanded_confusion(matrix, labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    recomputed = {
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
    }
    for key, value in recomputed.items():
        assert saved[key] == pytest.approx(value, abs=1e-15)
        assert metadata["test_metrics"][key] == pytest.approx(value, abs=1e-15)

    for index, label in enumerate(labels):
        row = saved["per_class"][label]
        assert row["precision"] == pytest.approx(precision[index], abs=1e-15)
        assert row["recall"] == pytest.approx(recall[index], abs=1e-15)
        assert row["f1"] == pytest.approx(f1[index], abs=1e-15)
        assert row["support"] == int(support[index])
        assert metadata["test_metrics"]["per_class"][label] == row

    confusion_csv = pd.read_csv(V2_RESULTS / "final_confusion_matrix.csv", index_col=0)
    assert np.array_equal(confusion_csv.to_numpy(dtype=int), matrix)
    report = pd.read_csv(V2_RESULTS / "final_test_classification_report.csv")
    combined = report.loc[
        report["evaluation_scope"].eq("combined")
        & report["row_type"].eq("class_metric")
    ].set_index("class")
    for label in labels:
        assert combined.loc[label, "precision"] == pytest.approx(
            saved["per_class"][label]["precision"]
        )
        assert combined.loc[label, "recall"] == pytest.approx(
            saved["per_class"][label]["recall"]
        )
        assert combined.loc[label, "f1"] == pytest.approx(saved["per_class"][label]["f1"])
        assert int(combined.loc[label, "support"]) == saved["per_class"][label]["support"]

    dashboard = load_model_dashboard_data(ROOT)
    assert dashboard.errors == []
    assert dashboard.test_metrics == saved
    assert dashboard.translated_vietnamese_metrics == final["test_subset_metrics"][
        "translated_vietnamese_test"
    ]


@pytest.mark.production_artifact
def test_one_shot_selection_and_production_hash_contract() -> None:
    frozen = json.loads((ROOT / "config" / "v2_production_frozen.json").read_text("utf-8"))
    final = json.loads((V2_RESULTS / "final_test_metrics.json").read_text("utf-8"))
    metadata = json.loads((V2_MODEL / "model_metadata.json").read_text("utf-8"))
    assert frozen["selection_basis"] == "validation_only"
    assert frozen["held_out_test_accessed_at_selection"] is False
    assert frozen["post_test_tuning_allowed"] is False
    assert "test" not in frozen["data"]["train"].lower()
    assert "test" not in frozen["data"]["validation"].lower()
    assert final["held_out_test_evaluation_count"] == 1
    assert final["selection_completed_before_test_access"] is True
    assert final["post_test_tuning_allowed"] is False
    assert set(final["final_refit"]["input_splits"]) == {"train", "validation"}
    assert final["leakage_checks"]["passed"] is True
    assert sha256_file(ROOT / final["frozen_config"]) == final["frozen_config_sha256"]
    assert sha256_file(ROOT / final["selection_rationale"]) == final[
        "selection_rationale_sha256"
    ]
    assert sha256_file(V2_MODEL / "production_pipeline.joblib") == metadata[
        "artifact_sha256"
    ]

    model = joblib.load(V2_MODEL / "production_pipeline.joblib")
    classifier = model.named_steps["classifier"]
    assert classifier.__class__.__name__ == "CalibratedClassifierCV"
    assert hasattr(model, "predict_proba")
    assert set(classifier.classes_) == {"normal", "spam", "phishing"}


def test_readme_contains_complete_v2_reproducibility_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_commands = [
        "python -m venv .venv",
        "python -m pip install -r requirements.txt",
        "python scripts\\audit_datasets.py",
        "python scripts\\audit_vietnamese_dataset.py",
        "python scripts\\standardize_datasets.py",
        "python scripts\\standardize_vietnamese_dataset.py",
        "python scripts\\audit_cross_language_overlap.py",
        "python scripts\\curate_vietnamese_labels.py",
        "python scripts\\build_translated_augmentation.py --offline",
        "python scripts\\build_bilingual_dataset.py",
        "python scripts\\train_v2_tfidf_models.py",
        "python scripts\\train_v2_multilingual_embeddings.py --device auto",
        "python scripts\\finalize_v2_model.py",
        "python scripts\\evaluate_v2_robustness.py",
        "python -m streamlit run app\\app.py",
        "python -m pytest -q",
    ]
    assert all(command in readme for command in required_commands)
    assert "do not rerun in the completed repository" in readme
    assert "No independent native Vietnamese benchmark exists" in readme
    assert "171 normal, 8 spam, 0 phishing" in readme
    assert "short promotional emails" in readme.casefold()
