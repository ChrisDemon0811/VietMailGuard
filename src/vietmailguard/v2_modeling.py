"""Validation-only helpers for Version 2 bilingual model experiments."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


def assert_only_model_selection_inputs(config: dict[str, Any]) -> None:
    """Reject configs that include anything beyond train and validation inputs."""
    observed = set(config.get("data", {}))
    if observed != {"train", "validation"}:
        raise ValueError(
            "Version 2 model selection accepts exactly train and validation inputs; "
            f"observed={sorted(observed)}"
        )


def assert_no_cross_split_values(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """Reject any non-empty leakage key shared by train and validation."""
    for column in columns:
        if column not in train.columns or column not in validation.columns:
            raise ValueError(f"Leakage column is missing: {column}")
        train_values = {value for value in train[column].astype(str) if value.strip()}
        validation_values = {
            value for value in validation[column].astype(str) if value.strip()
        }
        overlap = train_values & validation_values
        if overlap:
            raise ValueError(
                f"Leakage detected for {column} between train and validation: "
                f"count={len(overlap)}, examples={sorted(overlap)[:5]}"
            )


def validation_subset_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Create explicit language and data-origin evaluation subsets."""
    required = {"language", "data_origin", "source", "augmentation_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Validation data is missing subset columns: {sorted(missing)}")

    language = frame["language"].fillna("").astype(str)
    origin = frame["data_origin"].fillna("").astype(str)
    source = frame["source"].fillna("").astype(str)
    augmentation = frame["augmentation_type"].fillna("").astype(str)
    translated = language.eq("vi") & origin.eq("translated")
    controlled = source.eq("controlled_translation") | augmentation.eq(
        "controlled_translation"
    )
    return {
        "combined": pd.Series(True, index=frame.index, dtype=bool),
        "english_original_artifact": language.eq("en") & origin.eq(""),
        "translated_vietnamese": translated,
        "existing_translated_vietnamese": translated & ~controlled,
        "controlled_translated_augmentation": controlled,
    }


def compute_subset_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    label_order: list[str],
) -> dict[str, Any]:
    """Compute subset metrics while leaving unsupported classes explicitly unavailable."""
    truth = np.asarray(list(y_true), dtype=object)
    predictions = np.asarray(list(y_pred), dtype=object)
    if len(truth) != len(predictions):
        raise ValueError("Truth and prediction lengths differ")
    if len(truth) == 0:
        return {
            "rows": 0,
            "supported_classes": [],
            "accuracy": None,
            "macro_precision_supported_classes": None,
            "macro_recall_supported_classes": None,
            "macro_f1_supported_classes": None,
            "three_class_macro_f1": None,
            "per_class": {
                label: {"precision": None, "recall": None, "f1": None, "support": 0}
                for label in label_order
            },
            "confusion_matrix": [[0 for _ in label_order] for _ in label_order],
            "confusion_matrix_label_order": list(label_order),
        }

    supported = [label for label in label_order if int(np.sum(truth == label)) > 0]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=label_order,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth,
        predictions,
        labels=supported,
        average="macro",
        zero_division=0,
    )
    per_class: dict[str, dict[str, float | int | None]] = {}
    for index, label in enumerate(label_order):
        class_support = int(support[index])
        per_class[label] = {
            "precision": float(precision[index]) if class_support else None,
            "recall": float(recall[index]) if class_support else None,
            "f1": float(f1[index]) if class_support else None,
            "support": class_support,
        }

    return {
        "rows": int(len(truth)),
        "supported_classes": supported,
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision_supported_classes": float(macro_precision),
        "macro_recall_supported_classes": float(macro_recall),
        "macro_f1_supported_classes": float(macro_f1),
        "three_class_macro_f1": float(macro_f1)
        if len(supported) == len(label_order)
        else None,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=label_order
        ).astype(int).tolist(),
        "confusion_matrix_label_order": list(label_order),
    }


def evaluate_validation_subsets(
    frame: pd.DataFrame,
    predictions: Iterable[str],
    label_order: list[str],
) -> dict[str, dict[str, Any]]:
    """Evaluate one prediction vector over all declared validation subsets."""
    prediction_array = np.asarray(list(predictions), dtype=object)
    if len(prediction_array) != len(frame):
        raise ValueError("Validation frame and prediction lengths differ")
    results: dict[str, dict[str, Any]] = {}
    for name, mask in validation_subset_masks(frame).items():
        positions = np.flatnonzero(mask.to_numpy())
        results[name] = compute_subset_metrics(
            frame.iloc[positions]["label"].astype(str),
            prediction_array[positions],
            label_order,
        )
    return results


def classifier_confidence_capability(pipeline: Pipeline) -> str:
    """Describe probability support without treating decision scores as probabilities."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return "native_predict_proba"
    return "unavailable_uncalibrated"


def rank_bilingual_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates using the Version 2 validation-selection contract."""
    required = {
        "experiment_id",
        "macro_f1",
        "phishing_recall",
        "language_balance_gap",
        "normal_false_positive_rate",
        "deployability_rank",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Comparison table is missing ranking columns: {sorted(missing)}")
    ranked = frame.sort_values(
        by=[
            "macro_f1",
            "phishing_recall",
            "language_balance_gap",
            "normal_false_positive_rate",
            "deployability_rank",
            "experiment_id",
        ],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "validation_rank", np.arange(1, len(ranked) + 1))
    ranked["recommended_tfidf_candidate"] = ranked["validation_rank"].eq(1)
    return ranked
