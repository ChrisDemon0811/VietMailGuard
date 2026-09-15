"""Leakage-safe, group-aware dataset splitting utilities."""

from __future__ import annotations

import hashlib
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from vietmailguard.dataset_standardizer import normalized_duplicate_text

SPLIT_ORDER = ("train", "validation", "test")


class LeakageError(RuntimeError):
    """Raised when any duplicate identifier crosses split boundaries."""


def integer_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Convert fractional split targets to integers using largest remainders."""
    if tuple(ratios) != SPLIT_ORDER:
        raise ValueError(f"Ratios must be ordered as {SPLIT_ORDER}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    raw = {name: total * ratios[name] for name in SPLIT_ORDER}
    targets = {name: int(raw[name]) for name in SPLIT_ORDER}
    remainder = total - sum(targets.values())
    ranked = sorted(SPLIT_ORDER, key=lambda name: (-(raw[name] - targets[name]), SPLIT_ORDER.index(name)))
    for name in ranked[:remainder]:
        targets[name] += 1
    return targets


def _allocate_stratum(
    groups: list[tuple[str, int]], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    targets = integer_targets(sum(size for _, size in groups), ratios)
    counts = {name: 0 for name in SPLIT_ORDER}
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=lambda item: -item[1])
    assignments: dict[str, str] = {}

    for group_id, size in shuffled:
        fitting = [name for name in SPLIT_ORDER if counts[name] + size <= targets[name]]
        if fitting:
            chosen = max(
                fitting,
                key=lambda name: (
                    (targets[name] - counts[name]) / max(targets[name], 1),
                    -SPLIT_ORDER.index(name),
                ),
            )
        else:
            chosen = min(
                SPLIT_ORDER,
                key=lambda name: (
                    (counts[name] + size - targets[name]) / max(targets[name], 1),
                    SPLIT_ORDER.index(name),
                ),
            )
        assignments[group_id] = chosen
        counts[chosen] += size
    return assignments


def assign_group_splits(
    frame: pd.DataFrame,
    ratios: dict[str, float],
    *,
    seed: int = 42,
    group_column: str = "group_id",
    label_column: str = "label",
) -> pd.Series:
    """Assign whole groups while approximately stratifying by class and source."""
    required = {group_column, label_column, "source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing split columns: {sorted(missing)}")
    if frame[group_column].isna().any() or frame[group_column].astype(str).str.strip().eq("").any():
        raise ValueError("Every row must have a non-empty group_id")

    grouped = frame.groupby(group_column, sort=False).agg(
        size=(group_column, "size"),
        label_count=(label_column, "nunique"),
        label=(label_column, "first"),
        source_count=("source", "nunique"),
        source=("source", "first"),
    )
    conflicts = grouped.index[grouped["label_count"] > 1].tolist()
    if conflicts:
        raise ValueError(f"group_id spans multiple labels: {conflicts[:5]}")
    grouped.loc[grouped["source_count"] > 1, "source"] = "<mixed-source-group>"

    assignments: dict[str, str] = {}
    for stratum_index, ((label, source), group_rows) in enumerate(
        grouped.groupby(["label", "source"], sort=True)
    ):
        units = [(str(group_id), int(row["size"])) for group_id, row in group_rows.iterrows()]
        assignments.update(_allocate_stratum(units, ratios, seed + stratum_index))

    split_names = frame[group_column].astype(str).map(assignments)
    if split_names.isna().any():
        raise RuntimeError("At least one group did not receive a split assignment")
    return split_names


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def add_leakage_fingerprints(frame: pd.DataFrame) -> pd.DataFrame:
    """Add independently recomputed normalized and template fingerprints."""
    result = frame.copy()
    result["_normalized_content_hash"] = result.apply(
        lambda row: _sha256_text(normalized_duplicate_text(row["subject"], row["body"])), axis=1
    )
    result["_template_hash"] = result.apply(
        lambda row: _sha256_text(
            normalized_duplicate_text(row["subject"], row["body"], template=True)
        ),
        axis=1,
    )
    return result


def check_cross_split_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Check group, stored content, normalized content, and template intersections."""
    if set(splits) != set(SPLIT_ORDER):
        raise ValueError(f"Expected splits {SPLIT_ORDER}, received {sorted(splits)}")
    fingerprinted = {name: add_leakage_fingerprints(frame) for name, frame in splits.items()}
    columns = {
        "group_id": "group_id",
        "content_hash": "content_hash",
        "normalized_content_hash": "_normalized_content_hash",
        "template_hash": "_template_hash",
    }
    comparisons: dict[str, dict[str, Any]] = {}
    passed = True
    for left, right in combinations(SPLIT_ORDER, 2):
        pair = f"{left}_vs_{right}"
        comparisons[pair] = {}
        for check_name, column in columns.items():
            overlap = set(fingerprinted[left][column]) & set(fingerprinted[right][column])
            comparisons[pair][check_name] = {
                "overlap_count": len(overlap),
                "examples": sorted(str(item) for item in overlap)[:5],
            }
            passed = passed and not overlap
    report = {
        "passed": passed,
        "checks": list(columns),
        "comparisons": comparisons,
    }
    if not passed:
        raise LeakageError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_splits(
    frame: pd.DataFrame, ratios: dict[str, float], *, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Return deterministic group-safe train, validation, and test frames."""
    assignments = assign_group_splits(frame, ratios, seed=seed)
    splits: dict[str, pd.DataFrame] = {}
    for index, name in enumerate(SPLIT_ORDER):
        selected = frame.loc[assignments.eq(name)].copy()
        selected = selected.sample(frac=1.0, random_state=seed + index).reset_index(drop=True)
        splits[name] = selected
    check_cross_split_leakage(splits)
    return splits


def _distribution(series: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = series.value_counts().sort_index()
    total = len(series)
    return {
        str(value): {
            "count": int(count),
            "percent": round(100 * int(count) / total, 6) if total else 0.0,
        }
        for value, count in counts.items()
    }


def build_split_report(
    splits: dict[str, pd.DataFrame], ratios: dict[str, float], seed: int
) -> dict[str, Any]:
    """Create a JSON-serializable split distribution and leakage report."""
    total_rows = sum(len(frame) for frame in splits.values())
    report: dict[str, Any] = {
        "seed": seed,
        "target_ratios": ratios,
        "total_rows": total_rows,
        "total_groups": len(set().union(*(set(frame["group_id"]) for frame in splits.values()))),
        "group_integrity_priority": (
            "Group integrity takes precedence over exact ratios; deviations are retained and reported."
        ),
        "splits": {},
    }
    for name in SPLIT_ORDER:
        frame = splits[name]
        actual_ratio = len(frame) / total_rows if total_rows else 0.0
        report["splits"][name] = {
            "rows": len(frame),
            "groups": int(frame["group_id"].nunique()),
            "actual_ratio": round(actual_ratio, 8),
            "target_ratio": ratios[name],
            "deviation_percentage_points": round(100 * (actual_ratio - ratios[name]), 6),
            "class_distribution": _distribution(frame["label"]),
            "source_distribution": _distribution(frame["source"]),
            "language_distribution": _distribution(frame["language"]),
        }
    report["leakage_check"] = check_cross_split_leakage(splits)
    return report


def write_splits_atomically(
    splits: dict[str, pd.DataFrame], outputs: dict[str, Path]
) -> None:
    """Verify serialized temporary files before replacing split outputs."""
    temporary: dict[str, Path] = {}
    try:
        for name in SPLIT_ORDER:
            path = outputs[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(f"{path.suffix}.tmp")
            splits[name].to_csv(temp_path, index=False, encoding="utf-8")
            temporary[name] = temp_path
        reloaded = {
            name: pd.read_csv(path, keep_default_na=False, low_memory=False)
            for name, path in temporary.items()
        }
        check_cross_split_leakage(reloaded)
        for name in SPLIT_ORDER:
            temporary[name].replace(outputs[name])
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()
