"""Read saved experiment and dataset artifacts for presentation layers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ModelDashboardData:
    """Actual saved artifacts used by the model dashboard."""

    selected_model: dict[str, Any] | None = None
    final_evaluation: dict[str, Any] | None = None
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    confusion_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_class_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    short_form_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list[str] = field(default_factory=list)

    @property
    def test_metrics(self) -> dict[str, Any] | None:
        """Return the held-out test metric block when it exists."""
        if self.final_evaluation is None:
            return None
        metrics = self.final_evaluation.get("overall_test_metrics")
        if not isinstance(metrics, dict):
            metrics = self.final_evaluation.get("test_metrics")
        return metrics if isinstance(metrics, dict) else None

    @property
    def translated_vietnamese_metrics(self) -> dict[str, Any] | None:
        """Return the translated-Vietnamese held-out subset, when saved."""
        if self.final_evaluation is None:
            return None
        subsets = self.final_evaluation.get("test_subset_metrics")
        if not isinstance(subsets, dict):
            return None
        metrics = subsets.get("translated_vietnamese_test")
        return metrics if isinstance(metrics, dict) else None


@dataclass
class DatasetDashboardData:
    """Actual saved artifacts used by the dataset dashboard."""

    split_report: dict[str, Any] | None = None
    standardization: pd.DataFrame = field(default_factory=pd.DataFrame)
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    removed_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicates: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)
    duplicate_stats: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_optional_json(
    path: Path, errors: list[str], *, artifact_name: str
) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"Missing {artifact_name}: {path.as_posix()}")
        return None
    try:
        return _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"Cannot read {artifact_name}: {exc}")
        return None


def _load_optional_csv(
    path: Path,
    errors: list[str],
    *,
    artifact_name: str,
    report_missing: bool = True,
) -> pd.DataFrame:
    if not path.exists():
        if report_missing:
            errors.append(f"Missing {artifact_name}: {path.as_posix()}")
        return pd.DataFrame()
    try:
        return _read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        errors.append(f"Cannot read {artifact_name}: {exc}")
        return pd.DataFrame()


def load_model_dashboard_data(project_root: Path) -> ModelDashboardData:
    """Load model-selection and held-out-test artifacts without inventing values."""
    root = Path(project_root)
    errors: list[str] = []
    results = root / "results" / "v2_bilingual"
    models = root / "models" / "v2_bilingual"
    return ModelDashboardData(
        selected_model=_load_optional_json(
            models / "model_metadata.json",
            errors,
            artifact_name="selected model metadata",
        ),
        final_evaluation=_load_optional_json(
            results / "final_test_metrics.json",
            errors,
            artifact_name="held-out test metrics",
        ),
        comparison=_load_optional_csv(
            results / "tfidf_model_comparison.csv",
            errors,
            artifact_name="model comparison",
        ),
        confusion_matrix=_load_optional_csv(
            results / "final_confusion_matrix.csv",
            errors,
            artifact_name="held-out confusion matrix",
        ),
        per_class_metrics=_load_optional_csv(
            results / "final_test_classification_report.csv",
            errors,
            artifact_name="held-out per-class metrics",
        ),
        robustness_metrics=_load_optional_csv(
            results / "robustness_metrics.csv",
            errors,
            artifact_name="Vietnamese robustness metrics",
            report_missing=False,
        ),
        short_form_metrics=_load_optional_csv(
            results / "short_form_challenge_metrics.csv",
            errors,
            artifact_name="short-form challenge metrics",
            report_missing=False,
        ),
        errors=errors,
    )


def load_dataset_dashboard_data(project_root: Path) -> DatasetDashboardData:
    """Build the V2 dashboard view from processed and split artifacts only."""
    root = Path(project_root)
    errors: list[str] = []
    split_root = root / "data" / "splits" / "v2"
    desired_columns = {
        "source",
        "label",
        "language",
        "data_origin",
        "augmentation_type",
        "training_eligible",
        "label_status",
        "content_hash",
        "group_id",
        "template_hash",
        "translation_group_id",
        "final_group_id",
    }
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        path = split_root / f"{split_name}.csv"
        if not path.exists():
            errors.append(f"Missing V2 {split_name} split: {path.as_posix()}")
            continue
        try:
            split_frames[split_name] = pd.read_csv(
                path,
                usecols=lambda column: column in desired_columns,
                keep_default_na=False,
            )
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            errors.append(f"Cannot read V2 {split_name} split: {exc}")

    if len(split_frames) != 3:
        return DatasetDashboardData(errors=errors)

    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    total_rows = sum(len(frame) for frame in split_frames.values())
    split_details: dict[str, Any] = {}
    for split_name, frame in split_frames.items():
        split_details[split_name] = {
            "rows": int(len(frame)),
            "groups": int(frame.get("final_group_id", pd.Series(dtype=str)).nunique()),
            "actual_ratio": len(frame) / total_rows if total_rows else 0.0,
            "target_ratio": targets[split_name],
            "class_distribution": _count_distribution(frame, "label"),
            "source_distribution": _count_distribution(frame, "source"),
            "language_distribution": _count_distribution(frame, "language"),
            "data_origin_distribution": _count_distribution(frame, "data_origin"),
        }

    leakage_columns = (
        "final_group_id",
        "content_hash",
        "group_id",
        "template_hash",
        "translation_group_id",
    )
    leakage_counts: dict[str, int] = {}
    for column in leakage_columns:
        sets = {
            name: set(frame[column].astype(str)) - {""}
            for name, frame in split_frames.items()
            if column in frame.columns
        }
        count = 0
        if len(sets) == 3:
            count = sum(
                len(sets[left] & sets[right])
                for left, right in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                )
            )
        leakage_counts[column] = count

    combined = pd.concat(split_frames.values(), ignore_index=True)
    valid_labels = combined["label"].isin({"normal", "spam", "phishing"})
    eligible = combined["training_eligible"].map(_as_bool)
    review_or_exclude = combined.get("label_status", pd.Series("", index=combined.index)).isin(
        {"review", "exclude"}
    )
    leakage_passed = (
        all(value == 0 for value in leakage_counts.values())
        and bool(valid_labels.all())
        and bool(eligible.all())
        and not bool(review_or_exclude.any())
    )
    split_report = {
        "total_rows": int(total_rows),
        "total_groups": int(combined["final_group_id"].nunique()),
        "splits": split_details,
        "leakage_check": {"passed": leakage_passed, **leakage_counts},
    }

    curated = _read_selected_columns(
        root / "data" / "curated" / "v2" / "Vietnamese_Curated_Dataset.csv",
        {"label", "training_eligible"},
        errors,
        "Vietnamese curated dataset",
    )
    augmentation = _read_selected_columns(
        root / "data" / "processed" / "Vietnamese_Translated_Augmentation.csv",
        {"training_eligible", "quality_status"},
        errors,
        "controlled translation augmentation",
    )
    summary = {
        "english_samples": int(combined["language"].eq("en").sum()),
        "translated_vietnamese_samples": int(combined["language"].eq("vi").sum()),
        "review_rows": int(curated.get("label", pd.Series(dtype=str)).eq("review").sum()),
        "excluded_rows": int(curated.get("label", pd.Series(dtype=str)).eq("exclude").sum()),
        "augmentation_rows": int(len(augmentation)),
        "augmentation_eligible": int(
            augmentation.get("training_eligible", pd.Series(dtype=object)).map(_as_bool).sum()
        ),
    }
    summary["augmentation_failed"] = summary["augmentation_rows"] - summary[
        "augmentation_eligible"
    ]
    duplicate_stats = {
        "final_groups": int(combined["final_group_id"].nunique()),
        "translation_groups": _repeated_group_count(combined, "translation_group_id"),
        "exact_duplicate_groups": _repeated_group_count(combined, "content_hash"),
        "template_duplicate_groups": _repeated_group_count(combined, "template_hash"),
    }
    return DatasetDashboardData(
        split_report=split_report,
        summary=summary,
        duplicate_stats=duplicate_stats,
        errors=errors,
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "eligible"}


def _count_distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    values = frame[column].astype(str).replace("", "not_declared")
    return {str(key): int(value) for key, value in values.value_counts().items()}


def _repeated_group_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = frame[column].astype(str)
    counts = values.loc[values.ne("")].value_counts()
    return int(counts.gt(1).sum())


def _read_selected_columns(
    path: Path,
    columns: set[str],
    errors: list[str],
    artifact_name: str,
) -> pd.DataFrame:
    if not path.exists():
        errors.append(f"Missing {artifact_name}: {path.as_posix()}")
        return pd.DataFrame()
    try:
        return pd.read_csv(
            path,
            usecols=lambda column: column in columns,
            keep_default_na=False,
        )
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        errors.append(f"Cannot read {artifact_name}: {exc}")
        return pd.DataFrame()


def aggregate_split_distribution(
    split_report: dict[str, Any], distribution_key: str
) -> pd.DataFrame:
    """Aggregate a count distribution across the saved train/validation/test splits."""
    totals: dict[str, int] = {}
    splits = split_report.get("splits", {})
    if not isinstance(splits, dict):
        return pd.DataFrame(columns=["category", "count"])
    for split in splits.values():
        if not isinstance(split, dict):
            continue
        distribution = split.get(distribution_key, {})
        if not isinstance(distribution, dict):
            continue
        for category, details in distribution.items():
            if isinstance(details, dict):
                count = details.get("count", 0)
            else:
                count = details
            try:
                totals[str(category)] = totals.get(str(category), 0) + int(count)
            except (TypeError, ValueError):
                continue
    return pd.DataFrame(
        [{"category": category, "count": count} for category, count in totals.items()]
    ).sort_values("category", ignore_index=True)


def split_summary_frame(split_report: dict[str, Any]) -> pd.DataFrame:
    """Return one summary row per split from the saved report."""
    rows: list[dict[str, Any]] = []
    splits = split_report.get("splits", {})
    if not isinstance(splits, dict):
        return pd.DataFrame()
    for split_name in ("train", "validation", "test"):
        details = splits.get(split_name)
        if not isinstance(details, dict):
            continue
        rows.append(
            {
                "split": split_name,
                "rows": details.get("rows"),
                "groups": details.get("groups"),
                "actual_ratio": details.get("actual_ratio"),
                "target_ratio": details.get("target_ratio"),
            }
        )
    return pd.DataFrame(rows)


def duplicate_summary(duplicates: pd.DataFrame, removed_rows: pd.DataFrame) -> dict[str, int]:
    """Summarize duplicate rows and groups using saved standardization reports."""
    result = {
        "reported_rows": int(len(duplicates)),
        "exact_groups": 0,
        "template_groups": 0,
        "exact_rows_removed": 0,
    }
    required_duplicate_columns = {"duplicate_type", "group_id"}
    if required_duplicate_columns.issubset(duplicates.columns):
        exact = duplicates.loc[duplicates["duplicate_type"].eq("exact")]
        template = duplicates.loc[duplicates["duplicate_type"].eq("template")]
        result["exact_groups"] = int(exact["group_id"].nunique())
        result["template_groups"] = int(template["group_id"].nunique())
    if "reason" in removed_rows.columns:
        result["exact_rows_removed"] = int(
            removed_rows["reason"].eq("exact_duplicate").sum()
        )
    return result
