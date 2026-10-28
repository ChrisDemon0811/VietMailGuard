"""Source-specific, review-only CSV adapters for four V2.1 input corpora.

No raw binary label is promoted to a VietMailGuard training label here.
The validation file is isolated as challenge material. Every input row is
retained in a chunk file, including incomplete/empty rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path


PARSER_VERSION = 1
EMPTY_VALUES = {"", "empty", "nan", "none", "null", "n/a", "<empty>"}
BIN_LABELS = {"0", "0.0", "1", "1.0"}
MEAJOR_COLUMNS = (
    "sender", "sender_domain", "receiver", "receiver_domain", "date",
    "subject", "content_types", "body", "urls", "url_count",
    "url_length_max", "url_length_avg", "url_subdom_max", "url_subdom_avg",
    "attachment_count", "has_attachments", "attachment_types", "language",
    "source", "label",
)
OUTPUT_COLUMNS = (
    "id", "dataset", "source", "source_row", "source_file", "raw_label",
    "candidate_label", "label", "review_status", "training_eligible",
    "dataset_role", "language", "sender", "sender_domain", "receiver",
    "receiver_domain", "date", "subject", "body", "content_types", "urls",
    "url_count", "url_length_max", "url_length_avg", "url_subdom_max",
    "url_subdom_avg", "attachment_count", "has_attachments",
    "attachment_types", "content_hash", "empty", "missing_label",
    "decode_warning", "extra_fields_json",
)


@dataclass(frozen=True)
class CsvSource:
    filename: str
    dataset: str
    required: tuple[str, ...]
    role: str
    text_column: str
    label_column: str


SOURCES = {
    "Merged_Dataset.csv": CsvSource("Merged_Dataset.csv", "merged_dataset", ("body", "label"), "review_only", "body", "label"),
    "Balanced_Dataset.csv": CsvSource("Balanced_Dataset.csv", "balanced_dataset", ("body", "label"), "review_only_related_to_merged", "body", "label"),
    "meajor_cleaned_preprocessed.csv": CsvSource("meajor_cleaned_preprocessed.csv", "meajor", MEAJOR_COLUMNS, "review_only", "body", "label"),
    "Phishing_validation_emails.csv": CsvSource("Phishing_validation_emails.csv", "phishing_validation", ("Email Text", "Email Type"), "challenge_only", "Email Text", "Email Type"),
}


def _missing(value: str) -> bool:
    return value.strip().casefold() in EMPTY_VALUES


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _content_hash(subject: str, body: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", subject + "\n" + body).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def inspect_schema(path: Path, source: CsvSource) -> list[str]:
    """Check the real header before consuming any row; preserve extra columns."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate header fields in {path.name}")
    missing = set(source.required) - set(header)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")
    return header


def candidate_label(source: CsvSource, raw_label: str) -> str:
    """Candidate taxonomy is descriptive only, never a verified training label."""
    label = raw_label.strip().casefold()
    if source.dataset == "phishing_validation":
        return {"safe email": "safe_candidate", "phishing email": "phishing_or_scam_candidate"}.get(label, "unknown")
    if label in {"0", "0.0"}:
        return "normal_candidate"
    if label in {"1", "1.0"}:
        return "spam_or_scam_or_phishing_candidate"
    return "unknown"


def adapt_row(source: CsvSource, raw: dict[str, str], row_number: int) -> dict[str, str]:
    """Adapt a single source row without assigning unverified class labels."""
    body = _nfc(raw.get(source.text_column, "") or "")
    subject = _nfc(raw.get("subject", "") or "")
    raw_label = raw.get(source.label_column, "") or ""
    source_name = (raw.get("source", "") or "") if source.dataset == "meajor" else source.dataset
    output = {key: "" for key in OUTPUT_COLUMNS}
    output.update({
        "id": f"{source.dataset}:{row_number}", "dataset": source.dataset,
        "source": source_name, "source_row": str(row_number),
        "source_file": source.filename, "raw_label": raw_label,
        "candidate_label": candidate_label(source, raw_label),
        "label": "review", "review_status": "needs_review",
        "training_eligible": "false", "dataset_role": source.role,
        "language": raw.get("language", "") if source.dataset == "meajor" else "",
        "subject": subject, "body": body,
        "content_hash": _content_hash(subject, body),
        "empty": str(_missing(subject) and _missing(body)).lower(),
        "missing_label": str(_missing(raw_label)).lower(),
        "decode_warning": str("\ufffd" in subject or "\ufffd" in body).lower(),
        "extra_fields_json": raw.get("_extra_fields", ""),
    })
    if source.dataset == "meajor":
        for key in MEAJOR_COLUMNS:
            if key not in {"source", "label", "language", "subject", "body"}:
                output[key] = _nfc(raw.get(key, "") or "")
    return output


def _new_stats(path: Path, source: CsvSource, header: list[str], chunk_size: int) -> dict:
    stat = path.stat()
    return {
        "parser_version": PARSER_VERSION, "filename": source.filename,
        "dataset": source.dataset, "role": source.role,
        "input_size_bytes": stat.st_size, "input_mtime_ns": stat.st_mtime_ns,
        "header": header, "chunk_size": chunk_size, "rows_read": 0, "chunks_written": 0,
        "raw_labels": {}, "candidate_labels": {}, "review_status": {},
        "missing_by_column": {key: 0 for key in header},
        "empty_rows": 0, "missing_label_rows": 0, "decode_warning_rows": 0,
        "malformed_rows": 0, "complete": False,
    }


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def _record_stats(stats: dict, raw: dict[str, str], adapted: dict[str, str]) -> None:
    stats["rows_read"] += 1
    for key in stats["missing_by_column"]:
        if _missing(raw.get(key, "") or ""):
            stats["missing_by_column"][key] += 1
    for field, value in (("raw_labels", adapted["raw_label"]),
                         ("candidate_labels", adapted["candidate_label"]),
                         ("review_status", adapted["review_status"])):
        stats[field][value] = stats[field].get(value, 0) + 1
    for field, count_field in (("empty", "empty_rows"),
                               ("missing_label", "missing_label_rows"),
                               ("decode_warning", "decode_warning_rows")):
        stats[count_field] += adapted[field] == "true"
    stats["malformed_rows"] += None in raw


def run_source(path: Path, output_dir: Path, report_path: Path, *, chunk_size: int = 5000) -> dict:
    """Stream one immutable CSV to deterministic parts with resumable checkpoints.

    The checkpoint is committed after each atomic part replacement. A crash
    between those steps safely rewrites the same part on resume.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    path = Path(path)
    source = SOURCES[path.name]
    header = inspect_schema(path, source)
    report_path = Path(report_path)
    output_dir = Path(output_dir)
    if report_path.exists():
        stats = json.loads(report_path.read_text(encoding="utf-8"))
        fresh = _new_stats(path, source, header, chunk_size)
        for field in ("parser_version", "filename", "input_size_bytes", "input_mtime_ns", "header", "chunk_size"):
            if stats[field] != fresh[field]:
                raise ValueError(f"Input changed since checkpoint: {path.name} ({field})")
        if stats["complete"]:
            parts = list(output_dir.glob("part-*.csv"))
            if len(parts) != stats["chunks_written"]:
                raise ValueError(f"Checkpoint/part count mismatch for {source.filename}")
            return stats
    else:
        if output_dir.exists() and any(output_dir.glob("part-*.csv")):
            raise ValueError(f"Output parts exist without checkpoint for {source.filename}")
        stats = _new_stats(path, source, header, chunk_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(100_000_000)
    completed = stats["rows_read"]
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        batch: list[tuple[dict[str, str], dict[str, str]]] = []
        for row_number, raw in enumerate(reader, start=1):
            if row_number <= completed:
                continue
            if None in raw:
                # Retain malformed extra fields in a sidecar field rather than
                # dropping the row; this condition is counted for review.
                raw = {**raw, "_extra_fields": json.dumps(raw[None], ensure_ascii=False)}
            adapted = adapt_row(source, raw, row_number)
            batch.append((raw, adapted))
            if len(batch) == chunk_size:
                _write_batch(batch, output_dir, stats, report_path)
                batch.clear()
        if batch:
            _write_batch(batch, output_dir, stats, report_path)
    stats["complete"] = True
    _save_json(report_path, stats)
    return stats


def _write_batch(batch: list[tuple[dict[str, str], dict[str, str]]], output_dir: Path,
                 stats: dict, report_path: Path) -> None:
    number = stats["chunks_written"] + 1
    part = output_dir / f"part-{number:06d}.csv"
    temp = part.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for _, adapted in batch:
            writer.writerow(adapted)
    os.replace(temp, part)
    for raw, adapted in batch:
        _record_stats(stats, raw, adapted)
    stats["chunks_written"] = number
    _save_json(report_path, stats)
