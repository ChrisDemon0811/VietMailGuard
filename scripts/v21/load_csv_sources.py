"""Audit four V2.1 CSV sources into isolated, review-only chunk files."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.v21.csv_adapters import SOURCES, run_source  # noqa: E402
from vietmailguard.v21.paths import resolve_v21_paths  # noqa: E402


def _rows(parts_dir: Path):
    csv.field_size_limit(100_000_000)
    for part in sorted(parts_dir.glob("part-*.csv")):
        with part.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)


def compare_merged_balanced(merged_dir: Path, balanced_dir: Path, sqlite_path: Path) -> dict:
    """Bounded-memory exact normalized overlap check; no data are merged."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS merged_hashes (hash TEXT PRIMARY KEY, raw_label TEXT)")
        connection.execute("DELETE FROM merged_hashes")
        merged_rows = 0
        merged_unique = 0
        for row in _rows(merged_dir):
            merged_rows += 1
            if row["content_hash"]:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO merged_hashes (hash, raw_label) VALUES (?, ?)",
                    (row["content_hash"], row["raw_label"]),
                )
                merged_unique += cursor.rowcount
        connection.commit()
        balanced_rows = 0
        matched_rows = 0
        conflicting_labels = 0
        for row in _rows(balanced_dir):
            balanced_rows += 1
            parent = connection.execute(
                "SELECT raw_label FROM merged_hashes WHERE hash = ?", (row["content_hash"],)
            ).fetchone() if row["content_hash"] else None
            if parent:
                matched_rows += 1
                conflicting_labels += parent[0] != row["raw_label"]
    return {
        "merged_rows": merged_rows, "merged_unique_normalized_hashes": merged_unique,
        "balanced_rows": balanced_rows,
        "balanced_rows_matching_merged_normalized_text": matched_rows,
        "matched_rows_with_conflicting_raw_labels": conflicting_labels,
        "relationship_policy": "related_corpora_not_independent",
        "identity_limit": "A matching normalized body is evidence of overlap, not original-message provenance.",
    }


def summarize_parts(parts_dir: Path, sqlite_path: Path) -> dict:
    """Count normalized duplicates and provenance fields without storing text."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    sources: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("DROP TABLE IF EXISTS seen_hashes")
        connection.execute("CREATE TABLE seen_hashes (hash TEXT PRIMARY KEY, first_label TEXT, count INTEGER)")
        rows = duplicate_rows = conflicting_label_rows = 0
        for row in _rows(parts_dir):
            rows += 1
            sources[row["source"]] += 1
            languages[row["language"]] += 1
            content_hash = row["content_hash"]
            if not content_hash:
                continue
            previous = connection.execute(
                "SELECT first_label FROM seen_hashes WHERE hash = ?", (content_hash,)
            ).fetchone()
            if previous:
                duplicate_rows += 1
                conflicting_label_rows += previous[0] != row["raw_label"]
                connection.execute("UPDATE seen_hashes SET count = count + 1 WHERE hash = ?", (content_hash,))
            else:
                connection.execute(
                    "INSERT INTO seen_hashes VALUES (?, ?, 1)", (content_hash, row["raw_label"])
                )
        groups = connection.execute("SELECT COUNT(*) FROM seen_hashes WHERE count > 1").fetchone()[0]
    return {
        "rows": rows, "normalized_duplicate_rows": duplicate_rows,
        "normalized_duplicate_groups": groups,
        "conflicting_raw_label_rows_against_first_occurrence": conflicting_label_rows,
        "source_counts": dict(sources), "language_counts": dict(languages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--dataset-dir", type=Path, help="Overrides configured input directory for this run")
    parser.add_argument("--only", choices=tuple(SOURCES), help="One source for an incremental run")
    args = parser.parse_args()
    paths = resolve_v21_paths()
    input_dir = args.dataset_dir or paths.dataset_dir
    names = [args.only] if args.only else list(SOURCES)
    summaries = {}
    for name in names:
        input_path = input_dir / name
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        source = SOURCES[name]
        base = paths.evaluation if source.role == "challenge_only" else paths.processed
        stats = run_source(
            input_path, base / "csv_sources" / source.dataset,
            paths.reports / f"csv_{source.dataset}_audit.json", chunk_size=args.chunk_size,
        )
        summaries[name] = {
            "rows": stats["rows_read"], "raw_labels": stats["raw_labels"],
            "empty": stats["empty_rows"], "missing_label": stats["missing_label_rows"],
            "complete": stats["complete"], "role": source.role,
        }
        print(json.dumps({name: summaries[name]}, ensure_ascii=False))
    if not args.only:
        overlap = compare_merged_balanced(
            paths.processed / "csv_sources" / "merged_dataset",
            paths.processed / "csv_sources" / "balanced_dataset",
            paths.cache / "csv_merged_balanced_overlap.sqlite3",
        )
        overlap_path = paths.reports / "csv_merged_balanced_overlap.json"
        overlap_path.parent.mkdir(parents=True, exist_ok=True)
        overlap_path.write_text(json.dumps(overlap, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"overlap": overlap}, ensure_ascii=False))
    for name in names:
        source = SOURCES[name]
        base = paths.evaluation if source.role == "challenge_only" else paths.processed
        details = summarize_parts(
            base / "csv_sources" / source.dataset,
            paths.cache / f"csv_{source.dataset}_duplicates.sqlite3",
        )
        details_path = paths.reports / f"csv_{source.dataset}_details.json"
        details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({f"{source.dataset}_details": details}, ensure_ascii=False))


if __name__ == "__main__":
    main()
