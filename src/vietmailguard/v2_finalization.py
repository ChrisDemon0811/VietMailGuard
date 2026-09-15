"""Frozen Version 2 refit and one-shot evaluation helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from vietmailguard.file_utils import sha256_file
from vietmailguard.modeling import EmailTextComposer
from vietmailguard.v2_modeling import compute_subset_metrics, validation_subset_masks


@dataclass(frozen=True)
class OneShotGuard:
    """Paths used to prevent a second held-out evaluation."""

    started_path: Path
    final_metrics_path: Path


def verify_frozen_artifacts(config: dict[str, Any], project_root: Path) -> None:
    """Reject any selection artifact changed after the pre-test freeze."""
    if config.get("selection_basis") != "validation_only":
        raise ValueError("Frozen selection is not marked validation_only")
    if config.get("held_out_test_accessed_at_selection") is not False:
        raise ValueError("Frozen selection does not attest a clean pre-test decision")
    if config.get("post_test_tuning_allowed") is not False:
        raise ValueError("Frozen config must prohibit post-test tuning")
    rationale = project_root / config["selection_artifacts"]["rationale"]
    if not rationale.is_file():
        raise FileNotFoundError("Pre-test model-selection rationale is missing")

    artifact_config = config["selection_artifacts"]
    pairs = (
        ("tfidf_comparison", "tfidf_comparison_sha256"),
        ("tfidf_metrics", "tfidf_metrics_sha256"),
        ("embedding_comparison", "embedding_comparison_sha256"),
        ("embedding_metrics", "embedding_metrics_sha256"),
        ("selected_candidate", "selected_candidate_sha256"),
    )
    for path_key, hash_key in pairs:
        path = project_root / artifact_config[path_key]
        if not path.is_file():
            raise FileNotFoundError(f"Frozen selection artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != artifact_config[hash_key]:
            raise ValueError(f"Frozen selection artifact changed: {path_key}")

    for split_name in ("train", "validation"):
        path = project_root / config["data"][split_name]
        expected = config["data"][f"{split_name}_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen {split_name} split changed after selection")


def build_group_calibration_splits(
    labels: Iterable[str],
    groups: Iterable[str],
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic stratified folds without splitting a leakage group."""
    label_array = np.asarray(list(labels), dtype=object)
    group_array = np.asarray(list(groups), dtype=object)
    if len(label_array) != len(group_array):
        raise ValueError("Calibration labels and groups have different lengths")
    if len(label_array) == 0:
        raise ValueError("Calibration data is empty")
    if any(not str(group).strip() for group in group_array):
        raise ValueError("Calibration group values must be non-empty")
    splitter = StratifiedGroupKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed)
    )
    dummy = np.zeros(len(label_array), dtype=np.uint8)
    splits = [
        (np.asarray(fit, dtype=int), np.asarray(calibrate, dtype=int))
        for fit, calibrate in splitter.split(dummy, label_array, group_array)
    ]
    assert_group_calibration_integrity(splits, group_array)
    expected_classes = set(label_array)
    for fold_index, (fit, calibrate) in enumerate(splits):
        if set(label_array[fit]) != expected_classes or set(label_array[calibrate]) != expected_classes:
            raise ValueError(f"Calibration fold {fold_index} does not contain every class")
    return splits


def assert_group_calibration_integrity(
    splits: Iterable[tuple[np.ndarray, np.ndarray]], groups: np.ndarray
) -> None:
    """Ensure no calibration fold places one group on both sides."""
    seen_calibration_rows: set[int] = set()
    for fold_index, (fit, calibrate) in enumerate(splits):
        overlap = set(groups[fit]) & set(groups[calibrate])
        if overlap:
            raise ValueError(
                f"Calibration group leakage in fold {fold_index}: {sorted(overlap)[:5]}"
            )
        seen_calibration_rows.update(int(value) for value in calibrate)
    if seen_calibration_rows != set(range(len(groups))):
        raise ValueError("Calibration folds do not cover every development row exactly once")


def build_frozen_pipeline(
    config: dict[str, Any],
    calibration_splits: list[tuple[np.ndarray, np.ndarray]],
) -> Pipeline:
    """Construct the exact pipeline recorded in the immutable snapshot."""
    representation = config["representation"]
    classifier_config = config["classifier"]
    calibration = config["calibration"]
    if representation["kind"] != "word" or classifier_config["name"] != "LinearSVC":
        raise ValueError("Unsupported frozen production approach")
    dtype_name = str(representation["dtype"])
    if dtype_name != "float32":
        raise ValueError(f"Unsupported frozen TF-IDF dtype: {dtype_name}")
    svm = LinearSVC(
        C=float(classifier_config["C"]),
        class_weight=classifier_config["class_weight"],
        max_iter=int(classifier_config["max_iter"]),
        random_state=int(classifier_config["random_state"]),
    )
    calibrated = CalibratedClassifierCV(
        estimator=svm,
        method=str(calibration["method"]),
        cv=calibration_splits,
        n_jobs=int(calibration["n_jobs"]),
        ensemble=bool(calibration["ensemble"]),
    )
    return Pipeline(
        [
            ("compose_text", EmailTextComposer()),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=str(representation["analyzer"]),
                    ngram_range=tuple(representation["ngram_range"]),
                    lowercase=bool(representation["lowercase"]),
                    min_df=int(representation["min_df"]),
                    max_features=int(representation["max_features"]),
                    sublinear_tf=bool(representation["sublinear_tf"]),
                    dtype=np.float32,
                ),
            ),
            ("classifier", calibrated),
        ]
    )


def final_subset_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return required combined, language, and origin test subsets."""
    masks = validation_subset_masks(frame)
    return {
        "combined": masks["combined"],
        "english_test": frame["language"].fillna("").astype(str).eq("en"),
        "translated_vietnamese_test": masks["translated_vietnamese"],
        "original_english": masks["english_original_artifact"],
        "existing_translated_vietnamese": masks["existing_translated_vietnamese"],
        "controlled_translation_augmentation_descendants": masks[
            "controlled_translated_augmentation"
        ],
    }


def evaluate_final_subsets(
    frame: pd.DataFrame,
    predictions: Iterable[str],
    label_order: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute honest subset metrics, leaving unsupported classes unavailable."""
    predicted = np.asarray(list(predictions), dtype=object)
    if len(predicted) != len(frame):
        raise ValueError("Test frame and prediction lengths differ")
    output: dict[str, dict[str, Any]] = {}
    for name, mask in final_subset_masks(frame).items():
        positions = np.flatnonzero(mask.to_numpy())
        truth = frame.iloc[positions]["label"].astype(str).to_numpy()
        subset_predictions = predicted[positions]
        metrics = compute_subset_metrics(truth, subset_predictions, label_order)
        if len(truth):
            supported = metrics["supported_classes"]
            _, _, weighted_f1, _ = precision_recall_fscore_support(
                truth,
                subset_predictions,
                labels=supported,
                average="weighted",
                zero_division=0,
            )
            metrics["weighted_f1_supported_classes"] = float(weighted_f1)
        else:
            metrics["weighted_f1_supported_classes"] = None
        output[name] = metrics
    return output


def classification_report_frame(
    subset_metrics: dict[str, dict[str, Any]],
    label_order: list[str],
) -> pd.DataFrame:
    """Flatten aggregate and per-class test metrics into a machine-readable table."""
    rows: list[dict[str, Any]] = []
    for scope, metrics in subset_metrics.items():
        rows.append(
            {
                "evaluation_scope": scope,
                "row_type": "scope_summary",
                "class": "all_supported_classes",
                "rows": metrics["rows"],
                "supported_classes": ",".join(metrics["supported_classes"]),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision_supported_classes"],
                "macro_recall": metrics["macro_recall_supported_classes"],
                "macro_f1": metrics["macro_f1_supported_classes"],
                "weighted_f1": metrics["weighted_f1_supported_classes"],
                "precision": None,
                "recall": None,
                "f1": None,
                "support": metrics["rows"],
                "availability_note": (
                    "all three classes supported"
                    if metrics["three_class_macro_f1"] is not None
                    else "macro metrics cover supported classes only; absent classes are N/A"
                ),
            }
        )
        for label in label_order:
            values = metrics["per_class"][label]
            rows.append(
                {
                    "evaluation_scope": scope,
                    "row_type": "class_metric",
                    "class": label,
                    "rows": metrics["rows"],
                    "supported_classes": ",".join(metrics["supported_classes"]),
                    "accuracy": None,
                    "macro_precision": None,
                    "macro_recall": None,
                    "macro_f1": None,
                    "weighted_f1": None,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                    "support": values["support"],
                    "availability_note": "" if values["support"] else "N/A: class absent from subset",
                }
            )
    return pd.DataFrame(rows)


def validate_controlled_test_descendants(
    development: pd.DataFrame, test: pd.DataFrame
) -> dict[str, int]:
    """Require controlled translations in test to remain with their test parent."""
    controlled = test[
        test["augmentation_type"].fillna("").astype(str).eq("controlled_translation")
        | test["source"].fillna("").astype(str).eq("controlled_translation")
    ]
    if controlled.empty:
        return {"controlled_test_rows": 0, "verified_test_parent_rows": 0}
    parent_ids = controlled["parent_id"].fillna("").astype(str)
    if parent_ids.str.strip().eq("").any():
        raise ValueError("Controlled test translation is missing parent_id")
    development_ids = set(development["id"].astype(str))
    leaked = set(parent_ids) & development_ids
    if leaked:
        raise ValueError(f"Controlled test parent appears in development: {sorted(leaked)[:5]}")
    test_ids = set(test["id"].astype(str))
    missing = set(parent_ids) - test_ids
    if missing:
        raise ValueError(f"Controlled test parent is absent from test: {sorted(missing)[:5]}")
    return {
        "controlled_test_rows": int(len(controlled)),
        "verified_test_parent_rows": int(parent_ids.nunique()),
    }


def assert_one_shot_not_started(guard: OneShotGuard) -> None:
    """Refuse any repeat or recovery run against the same held-out test."""
    if guard.final_metrics_path.exists():
        raise FileExistsError(
            f"Final test metrics already exist: {guard.final_metrics_path}. "
            "The Version 2 held-out test cannot be evaluated again."
        )
    if guard.started_path.exists():
        raise FileExistsError(
            f"One-shot evaluation journal already exists: {guard.started_path}. "
            "Do not reopen the held-out test; use a new Version 2.1 test protocol."
        )


def create_one_shot_journal(path: Path, payload: dict[str, Any]) -> None:
    """Atomically record authorization immediately before the first test read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except BaseException:
        raise
