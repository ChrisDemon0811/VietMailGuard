"""Reproducible text-classification experiments for VietMailGuard."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from vietmailguard.file_utils import sha256_file

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REQUIRED_DATA_COLUMNS = {
    "id",
    "subject",
    "body",
    "label",
    "source",
    "language",
    "content_hash",
    "group_id",
}


class EmailTextComposer(BaseEstimator, TransformerMixin):
    """Combine subject and body while keeping the raw split rows unchanged."""

    def __init__(self, subject_column: str = "subject", body_column: str = "body") -> None:
        self.subject_column = subject_column
        self.body_column = body_column

    def fit(self, X: pd.DataFrame, y: object = None) -> "EmailTextComposer":
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._validate_input(X)
        subject = X[self.subject_column].fillna("").astype(str)
        body = X[self.body_column].fillna("").astype(str)
        return (subject + "\n" + body).to_numpy(dtype=object)

    def _validate_input(self, X: object) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("EmailTextComposer expects a pandas DataFrame")
        missing = {self.subject_column, self.body_column} - set(X.columns)
        if missing:
            raise ValueError(f"Email input is missing columns: {sorted(missing)}")


@dataclass(frozen=True)
class ExperimentSpec:
    """One configured feature/classifier combination."""

    experiment_id: str
    feature_name: str
    feature_display_name: str
    classifier_name: str
    classifier_display_name: str


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def runtime_versions() -> dict[str, str]:
    """Return runtime versions recorded with experiment artifacts."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def read_email_split(path: Path) -> pd.DataFrame:
    """Read and validate one standardized split without altering it."""
    if not path.exists():
        raise FileNotFoundError(f"Split does not exist: {path}")
    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    missing = REQUIRED_DATA_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    if frame["body"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{path} contains an empty email body")
    return frame


def validate_labels(frame: pd.DataFrame, label_order: list[str], split_name: str) -> None:
    """Require exactly the configured internal class identifiers."""
    observed = set(frame["label"].astype(str))
    expected = set(label_order)
    if observed != expected:
        raise ValueError(
            f"{split_name} labels differ from configured labels: "
            f"observed={sorted(observed)}, expected={sorted(expected)}"
        )


def assert_disjoint_splits(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Reject group or exact-content leakage between two splits."""
    for column in ("group_id", "content_hash"):
        overlap = set(left[column].astype(str)) & set(right[column].astype(str))
        if overlap:
            examples = sorted(overlap)[:5]
            raise ValueError(
                f"Leakage detected for {column} between {left_name} and {right_name}: "
                f"count={len(overlap)}, examples={examples}"
            )


def iter_experiment_specs(config: dict[str, Any]) -> list[ExperimentSpec]:
    """Expand configured features and classifiers into deterministic experiments."""
    specs: list[ExperimentSpec] = []
    for feature_name, feature in config["features"].items():
        for classifier_name, classifier in config["classifiers"].items():
            specs.append(
                ExperimentSpec(
                    experiment_id=f"{feature_name}__{classifier_name}",
                    feature_name=feature_name,
                    feature_display_name=str(feature["display_name"]),
                    classifier_name=classifier_name,
                    classifier_display_name=str(classifier["display_name"]),
                )
            )
    return specs


def _tfidf_dtype(name: str) -> type[np.floating[Any]]:
    if name == "float32":
        return np.float32
    if name == "float64":
        return np.float64
    raise ValueError(f"Unsupported TF-IDF dtype: {name}")


def _word_vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(config["ngram_range"]),
        lowercase=True,
        min_df=int(config["min_df"]),
        max_features=int(config["max_features"]),
        sublinear_tf=bool(config["sublinear_tf"]),
        dtype=_tfidf_dtype(str(config["dtype"])),
    )


def _character_vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=str(config.get("analyzer", "char_wb")),
        ngram_range=tuple(config["ngram_range"]),
        lowercase=True,
        min_df=int(config["min_df"]),
        max_features=int(config["max_features"]),
        sublinear_tf=bool(config["sublinear_tf"]),
        dtype=_tfidf_dtype(str(config["dtype"])),
    )


def build_feature_transformer(config: dict[str, Any]) -> BaseEstimator:
    """Build one configured TF-IDF representation."""
    kind = config["kind"]
    if kind == "word":
        return _word_vectorizer(config)
    if kind == "character":
        return _character_vectorizer(config)
    if kind == "word_character":
        return FeatureUnion(
            [
                ("word", _word_vectorizer(config["word"])),
                ("character", _character_vectorizer(config["character"])),
            ],
            n_jobs=1,
        )
    raise ValueError(f"Unsupported feature kind: {kind}")


def build_classifier(config: dict[str, Any], seed: int) -> BaseEstimator:
    """Build one configured classifier without inventing probability support."""
    kind = config["kind"]
    if kind == "multinomial_nb":
        return MultinomialNB(
            alpha=float(config["alpha"]),
            fit_prior=bool(config["fit_prior"]),
        )
    if kind == "logistic_regression":
        return LogisticRegression(
            C=float(config["C"]),
            class_weight=config["class_weight"],
            solver=str(config["solver"]),
            max_iter=int(config["max_iter"]),
            random_state=seed,
        )
    if kind == "linear_svm":
        return LinearSVC(
            C=float(config["C"]),
            class_weight=config["class_weight"],
            max_iter=int(config["max_iter"]),
            random_state=seed,
        )
    if kind == "calibrated_linear_svm":
        svm = LinearSVC(
            C=float(config["C"]),
            class_weight=config["class_weight"],
            max_iter=int(config["max_iter"]),
            random_state=seed,
        )
        return CalibratedClassifierCV(
            estimator=svm,
            method=str(config["calibration_method"]),
            cv=int(config["calibration_cv"]),
            n_jobs=1,
            ensemble=bool(config["calibration_ensemble"]),
        )
    raise ValueError(f"Unsupported classifier kind: {kind}")


def build_experiment_pipeline(
    spec: ExperimentSpec,
    config: dict[str, Any],
    *,
    cache_directory: Path | None = None,
) -> Pipeline:
    """Build an end-to-end subject/body to class pipeline."""
    memory = str(cache_directory) if cache_directory is not None else None
    return Pipeline(
        [
            ("compose_text", EmailTextComposer()),
            ("tfidf", build_feature_transformer(config["features"][spec.feature_name])),
            (
                "classifier",
                build_classifier(config["classifiers"][spec.classifier_name], int(config["seed"])),
            ),
        ],
        memory=memory,
    )


def compute_metrics(
    y_true: Iterable[str], y_pred: Iterable[str], label_order: list[str]
) -> dict[str, Any]:
    """Compute all required multiclass metrics using an explicit class order."""
    truth = np.asarray(list(y_true), dtype=object)
    predictions = np.asarray(list(y_pred), dtype=object)
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=label_order,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth,
        predictions,
        labels=label_order,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        truth,
        predictions,
        labels=label_order,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(truth, predictions, labels=label_order)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            label: {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(per_support[index]),
            }
            for index, label in enumerate(label_order)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_label_order": list(label_order),
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    label_order: list[str],
) -> tuple[dict[str, Any], np.ndarray, float]:
    """Predict once and return metrics, predictions, and elapsed seconds."""
    started = time.perf_counter()
    predictions = np.asarray(pipeline.predict(X), dtype=object)
    elapsed = time.perf_counter() - started
    return compute_metrics(y, predictions, label_order), predictions, elapsed


def metrics_row(
    spec: ExperimentSpec,
    metrics: dict[str, Any],
    *,
    fit_seconds: float,
    predict_seconds: float,
) -> dict[str, Any]:
    """Flatten validation metrics for the comparison CSV."""
    row: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "feature_set": spec.feature_display_name,
        "classifier": spec.classifier_display_name,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }
    for label, values in metrics["per_class"].items():
        row[f"{label}_precision"] = values["precision"]
        row[f"{label}_recall"] = values["recall"]
        row[f"{label}_f1"] = values["f1"]
        row[f"{label}_support"] = values["support"]
    return row


def rank_comparison(frame: pd.DataFrame, selection: dict[str, Any]) -> pd.DataFrame:
    """Rank experiments without consulting held-out test data."""
    metrics = [
        str(selection["primary_metric"]),
        str(selection["secondary_metric"]),
        *(str(value) for value in selection.get("additional_tie_breakers", [])),
    ]
    missing = set(metrics) - set(frame.columns)
    if missing:
        raise ValueError(f"Comparison table is missing ranking metrics: {sorted(missing)}")
    ranked = frame.sort_values(
        by=[*metrics, "experiment_id"],
        ascending=[*[False] * len(metrics), True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "validation_rank", np.arange(1, len(ranked) + 1))
    ranked["selected"] = ranked["validation_rank"].eq(1)
    return ranked


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write formatted UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_metric_artifacts(
    output_directory: Path,
    artifact_stem: str,
    metrics: dict[str, Any],
) -> None:
    """Write JSON, per-class CSV, confusion CSV, and confusion PNG."""
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / f"{artifact_stem}_metrics.json", metrics)

    per_class = pd.DataFrame.from_dict(metrics["per_class"], orient="index")
    per_class.index.name = "class"
    per_class.reset_index().to_csv(
        output_directory / f"{artifact_stem}_per_class_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    labels = metrics["confusion_matrix_label_order"]
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    matrix.index.name = "actual\\predicted"
    matrix.to_csv(output_directory / f"{artifact_stem}_confusion_matrix.csv", encoding="utf-8")
    plot_confusion_matrix(
        matrix.to_numpy(),
        labels,
        output_directory / f"{artifact_stem}_confusion_matrix.png",
        title=artifact_stem.replace("_", " ").title(),
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save a compact confusion-matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="Actual class",
        title=title,
    )
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot validation Macro F1 for every configured experiment."""
    ordered = comparison.sort_values("macro_f1", ascending=True)
    labels = ordered["feature_set"] + " / " + ordered["classifier"]
    colors = ["#0f766e" if selected else "#64748b" for selected in ordered["selected"]]
    height = max(5.5, 0.55 * len(ordered))
    figure, axis = plt.subplots(figsize=(10, height))
    bars = axis.barh(labels, ordered["macro_f1"], color=colors)
    axis.set_xlabel("Validation Macro F1")
    axis.set_xlim(0.0, 1.0)
    axis.set_title("VietMailGuard model comparison")
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, ordered["macro_f1"]):
        axis.text(
            min(value + 0.01, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
        )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize rows, groups, classes, sources, and languages."""
    return {
        "rows": int(len(frame)),
        "groups": int(frame["group_id"].nunique()),
        "class_distribution": {
            str(key): int(value) for key, value in frame["label"].value_counts().sort_index().items()
        },
        "source_distribution": {
            str(key): int(value) for key, value in frame["source"].value_counts().sort_index().items()
        },
        "language_distribution": {
            str(key): int(value) for key, value in frame["language"].value_counts().sort_index().items()
        },
    }


def confidence_for_rows(pipeline: Pipeline, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted classes and calibrated/native probabilities."""
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "predict_proba"):
        raise TypeError("Production classifier does not expose calibrated probabilities")
    predictions = np.asarray(pipeline.predict(rows), dtype=object)
    probabilities = np.asarray(pipeline.predict_proba(rows), dtype=float)
    return predictions, probabilities
