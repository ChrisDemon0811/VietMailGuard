"""Read-only profiling utilities for raw email datasets."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from vietmailguard.file_utils import sha256_file

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}
LABEL_COLUMN_NAMES = {"label", "labels", "class", "category", "target", "type"}
SUBJECT_COLUMN_NAMES = {"subject", "title"}
BODY_COLUMN_NAMES = {"body", "message", "text", "content", "email", "email_body"}
OUTPUT_NAME_PATTERN = re.compile(
    r"(^|[_\-.])(result|results|output|outputs|prediction|predictions|metric|metrics|evaluation)([_\-.]|$)",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def normalize_column_name(value: object) -> str:
    """Return a comparison-safe column name without changing source data."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_for_comparison(value: object, *, template: bool = False) -> str:
    """Normalize text for exact or URL/email-insensitive duplicate comparison."""
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    text = text.lower()
    if template:
        text = URL_PATTERN.sub(" <url> ", text)
        text = EMAIL_PATTERN.sub(" <email> ", text)
    return " ".join(text.split())


def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)
            return frame, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("; ".join(errors))


def read_dataset(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a supported tabular dataset and report the selected encoding."""
    if path.suffix.lower() in {".csv", ".tsv"}:
        return _read_csv(path)
    return pd.read_excel(path), "binary/xlsx"


def _safe_value(value: object, max_length: int = 240) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    return text if len(text) <= max_length else f"{text[:max_length]}…"


def _find_columns(columns: Iterable[object], candidates: set[str]) -> list[str]:
    return [str(column) for column in columns if normalize_column_name(column) in candidates]


def _content_series(frame: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    subjects = _find_columns(frame.columns, SUBJECT_COLUMN_NAMES)
    bodies = _find_columns(frame.columns, BODY_COLUMN_NAMES)
    selected = subjects[:1] + bodies[:1]
    if not selected:
        return None, []
    parts = [frame[column].fillna("").astype(str).str.strip() for column in selected]
    content = parts[0]
    for part in parts[1:]:
        content = content.str.cat(part, sep="\n")
    return content, selected


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_repository(raw_dir: Path, reports_dir: Path, sample_size: int = 3) -> dict[str, Any]:
    """Audit every supported raw dataset and write reproducible reports."""
    raw_dir = raw_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No supported datasets found under {raw_dir}")

    dataset_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    samples: dict[str, list[dict[str, object]]] = {}
    inventory: dict[str, dict[str, Any]] = {}
    exact_owners: dict[str, Counter[str]] = defaultdict(Counter)
    template_owners: dict[str, Counter[str]] = defaultdict(Counter)

    for path in paths:
        relative_name = path.relative_to(raw_dir).as_posix()
        frame, encoding = read_dataset(path)
        label_columns = _find_columns(frame.columns, LABEL_COLUMN_NAMES)
        content, content_columns = _content_series(frame)
        likely_output = bool(OUTPUT_NAME_PATTERN.search(path.stem))
        missing_cells = int(frame.isna().sum().sum())
        total_cells = int(frame.shape[0] * frame.shape[1])

        exact_duplicate_rows = 0
        template_duplicate_rows = 0
        unique_exact_hashes = 0
        unique_template_hashes = 0
        if content is not None:
            exact_hashes = content.map(lambda value: _hash_text(normalize_for_comparison(value)))
            template_hashes = content.map(
                lambda value: _hash_text(normalize_for_comparison(value, template=True))
            )
            exact_duplicate_rows = int(exact_hashes.duplicated(keep=False).sum())
            template_duplicate_rows = int(template_hashes.duplicated(keep=False).sum())
            unique_exact_hashes = int(exact_hashes.nunique())
            unique_template_hashes = int(template_hashes.nunique())
            for digest, count in exact_hashes.value_counts().items():
                exact_owners[str(digest)][relative_name] += int(count)
            for digest, count in template_hashes.value_counts().items():
                template_owners[str(digest)][relative_name] += int(count)

        dataset_rows.append(
            {
                "file": relative_name,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "encoding": encoding,
                "rows": len(frame),
                "columns": len(frame.columns),
                "column_names": json.dumps([str(column) for column in frame.columns], ensure_ascii=False),
                "missing_cells": missing_cells,
                "missing_percent": round(100 * missing_cells / total_cells, 4) if total_cells else 0.0,
                "exact_duplicate_rows_all_columns": int(frame.duplicated(keep=False).sum()),
                "content_columns": json.dumps(content_columns, ensure_ascii=False),
                "exact_duplicate_rows_content": exact_duplicate_rows,
                "template_duplicate_rows_content": template_duplicate_rows,
                "unique_exact_content": unique_exact_hashes,
                "unique_template_content": unique_template_hashes,
                "label_columns": json.dumps(label_columns, ensure_ascii=False),
                "likely_generated_output": likely_output,
                "initial_training_status": "exclude" if likely_output else "review",
                "status_reason": (
                    "Filename indicates a generated result/output artifact."
                    if likely_output
                    else "Raw labels and corpus provenance require explicit review before mapping."
                ),
            }
        )

        for column in frame.columns:
            series = frame[column]
            non_null = series.dropna()
            example_values = [_safe_value(value, 120) for value in non_null.drop_duplicates().head(5)]
            column_rows.append(
                {
                    "file": relative_name,
                    "column": str(column),
                    "normalized_column": normalize_column_name(column),
                    "dtype": str(series.dtype),
                    "non_null": int(series.notna().sum()),
                    "missing": int(series.isna().sum()),
                    "missing_percent": round(100 * series.isna().mean(), 4),
                    "unique_non_null": int(series.nunique(dropna=True)),
                    "sample_unique_values": json.dumps(example_values, ensure_ascii=False),
                }
            )

        for label_column in label_columns:
            counts = frame[label_column].value_counts(dropna=False)
            for raw_value, count in counts.items():
                label_rows.append(
                    {
                        "file": relative_name,
                        "label_column": label_column,
                        "raw_label": "<MISSING>" if pd.isna(raw_value) else str(raw_value),
                        "count": int(count),
                        "mapping_status": "unmapped_pending_provenance_review",
                        "mapped_label": "",
                    }
                )

        samples[relative_name] = [
            {str(column): _safe_value(value) for column, value in row.items()}
            for row in frame.head(sample_size).to_dict(orient="records")
        ]
        normalized_to_source = {
            normalize_column_name(column): str(column) for column in frame.columns
        }
        inventory[relative_name] = {
            "path": f"data/raw/{relative_name}",
            "format": path.suffix.lower().lstrip("."),
            "encoding": encoding,
            "source_name": path.stem,
            "language": "en",
            "observed_columns": [str(column) for column in frame.columns],
            "field_mapping": {
                "id": normalized_to_source.get("id"),
                "sender": normalized_to_source.get("sender"),
                "receiver": normalized_to_source.get("receiver"),
                "date": normalized_to_source.get("date"),
                "subject": normalized_to_source.get("subject"),
                "body": normalized_to_source.get("body"),
                "raw_label": label_columns[0] if len(label_columns) == 1 else None,
                "label": None,
                "source": None,
                "language": None,
                "url_count": normalized_to_source.get("url_count"),
                "content_hash": normalized_to_source.get("content_hash"),
                "group_id": normalized_to_source.get("group_id"),
            },
            "label_columns_detected": label_columns,
            "label_mapping": {},
            "mapping_status": "pending_provenance_review",
            "training_status": "exclude" if likely_output else "review",
            "notes": [
                "Column names are recorded from the file; semantic mappings are not yet approved.",
                "No raw label has been mapped to normal, spam, or phishing.",
            ],
        }

    overlap_rows = _build_overlap_rows(paths, raw_dir, exact_owners, template_owners)
    pd.DataFrame(dataset_rows).to_csv(reports_dir / "dataset_audit.csv", index=False, encoding="utf-8")
    pd.DataFrame(column_rows).to_csv(reports_dir / "dataset_columns.csv", index=False, encoding="utf-8")
    pd.DataFrame(label_rows).to_csv(reports_dir / "dataset_labels.csv", index=False, encoding="utf-8")
    pd.DataFrame(overlap_rows).to_csv(reports_dir / "dataset_overlap.csv", index=False, encoding="utf-8")
    (reports_dir / "dataset_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown_summary(reports_dir / "initial_audit_summary.md", dataset_rows, overlap_rows)
    return {
        "datasets": inventory,
        "summary": dataset_rows,
        "overlaps": overlap_rows,
        "report_files": [
            "dataset_audit.csv",
            "dataset_columns.csv",
            "dataset_labels.csv",
            "dataset_overlap.csv",
            "dataset_samples.json",
            "initial_audit_summary.md",
        ],
    }


def _write_markdown_summary(
    path: Path, dataset_rows: list[dict[str, Any]], overlap_rows: list[dict[str, Any]]
) -> None:
    likely_outputs = [row["file"] for row in dataset_rows if row["likely_generated_output"]]
    representation_candidates = [
        row for row in overlap_rows
        if row["assessment"] in {
            "likely_alternate_representation_or_subset",
            "substantial_overlap_review_before_combining",
        }
    ]
    partial_overlaps = [
        row for row in overlap_rows if row["assessment"] == "partial_overlap_group_before_splitting"
    ]
    lines = [
        "# Initial raw dataset audit",
        "",
        "This report is generated from file contents. It does not approve any label mapping or dataset provenance.",
        "",
        "## Inventory",
        "",
        "| File | Rows | Columns | Missing cells | Exact content duplicate rows | Template duplicate rows | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['file']} | {row['rows']} | {row['columns']} | {row['missing_cells']} | "
            f"{row['exact_duplicate_rows_content']} | {row['template_duplicate_rows_content']} | "
            f"{row['initial_training_status']} |"
        )
    lines.extend(["", "## Label decision", ""])
    lines.append(
        "This structural audit intentionally records observed labels as unmapped in `dataset_labels.csv`; it does "
        "not overwrite or supersede reviewed decisions. Current semantics and approved mappings are maintained "
        "separately in `reports/label_semantics_report.md` and `config/datasets.json`."
    )
    lines.extend(["", "## Generated output/result files", ""])
    lines.append(
        ", ".join(likely_outputs)
        if likely_outputs
        else "No filename matched the conservative generated result/output naming rule."
    )
    lines.extend(["", "## Possible alternate representations", ""])
    if representation_candidates:
        for row in representation_candidates:
            lines.append(
                f"- {row['file_a']} vs {row['file_b']}: {row['assessment']} "
                f"(exact overlap {row['exact_overlap_of_smaller_unique_set_percent']}%)."
            )
    else:
        lines.append("No pair met the substantial-overlap threshold (20% of the smaller unique-content set).")
    lines.extend(["", "## Partial cross-corpus overlap", ""])
    if partial_overlaps:
        for row in partial_overlaps:
            lines.append(
                f"- {row['file_a']} vs {row['file_b']}: "
                f"{row['shared_unique_exact_content']} shared exact content hashes and "
                f"{row['shared_unique_template_content']} shared template hashes."
            )
    else:
        lines.append("No cross-corpus content overlap was detected.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Filename, numeric label value, and single-class composition are not treated as proof of label meaning. "
            "The `urls` source column is also not mapped to `url_count`; its observed values require provenance review.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_overlap_rows(
    paths: list[Path],
    raw_dir: Path,
    exact_owners: dict[str, Counter[str]],
    template_owners: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    names = [path.relative_to(raw_dir).as_posix() for path in paths]
    unique_counts: dict[str, dict[str, int]] = {"exact": {}, "template": {}}
    for name in names:
        unique_counts["exact"][name] = sum(name in owners for owners in exact_owners.values())
        unique_counts["template"][name] = sum(name in owners for owners in template_owners.values())

    rows: list[dict[str, Any]] = []
    for left, right in combinations(names, 2):
        exact_shared = sum(left in owners and right in owners for owners in exact_owners.values())
        template_shared = sum(left in owners and right in owners for owners in template_owners.values())
        exact_denominator = min(unique_counts["exact"][left], unique_counts["exact"][right])
        template_denominator = min(unique_counts["template"][left], unique_counts["template"][right])
        exact_ratio = exact_shared / exact_denominator if exact_denominator else 0.0
        template_ratio = template_shared / template_denominator if template_denominator else 0.0
        if exact_ratio >= 0.8 or template_ratio >= 0.8:
            assessment = "likely_alternate_representation_or_subset"
        elif exact_ratio >= 0.2 or template_ratio >= 0.2:
            assessment = "substantial_overlap_review_before_combining"
        elif exact_shared or template_shared:
            assessment = "partial_overlap_group_before_splitting"
        else:
            assessment = "no_content_overlap_detected"
        rows.append(
            {
                "file_a": left,
                "file_b": right,
                "shared_unique_exact_content": exact_shared,
                "exact_overlap_of_smaller_unique_set_percent": round(exact_ratio * 100, 4),
                "shared_unique_template_content": template_shared,
                "template_overlap_of_smaller_unique_set_percent": round(template_ratio * 100, 4),
                "assessment": assessment,
            }
        )
    return rows
