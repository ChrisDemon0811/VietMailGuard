"""Prepare a metadata-only intake for a future independent Chinese benchmark.

This module never opens a held-out split or a classifier. Fingerprint checks use
the pre-split V2.1 audit index read-only; they do not inspect labels or errors.
No candidate is automatically promoted to a final benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from .dedup_audit import dedup_hashes
from .paths import V21Paths, resolve_v21_paths


INPUT_FIELDS = (
    "candidate_id", "source", "source_record_id", "source_version", "source_url",
    "license_status", "permission_evidence", "data_origin", "language_claim",
    "raw_label", "proposed_label", "label_evidence", "reviewer_1_id",
    "reviewer_1_label", "reviewer_2_id", "reviewer_2_label", "parent_id",
    "subject", "body",
)
MANIFEST_FIELDS = (
    "candidate_id", "source", "source_record_id", "source_version", "source_url",
    "license_status", "permission_evidence", "data_origin", "language_claim", "raw_label",
    "proposed_label", "label_evidence", "reviewer_1_id", "reviewer_1_label",
    "reviewer_2_id", "reviewer_2_label", "parent_id", "exact_hash", "normalized_hash",
    "template_hash", "internal_duplicate_kinds", "reference_overlap",
    "screening_flags", "benchmark_status",
)
LABELS = {"normal", "spam", "phishing"}
BLOCKED_SOURCES = {"trec06c", "llmgen_phishing_email_dataset", "v2_reference"}


def _output_dir(paths: V21Paths) -> Path:
    return paths.evaluation / "chinese_real_world_benchmark"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_hashes(paths: V21Paths) -> set[str]:
    """Read research synthetic fingerprints only, never their email text."""
    values: set[str] = set()
    directory = paths.processed / "research_synthetic_zh"
    for name in ("train.csv", "synthetic_challenge.csv"):
        path = directory / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("content_hash", "normalized_content_hash", "template_hash"):
                    if row.get(key):
                        values.add(row[key])
    return values


def _reference_connection(paths: V21Paths) -> sqlite3.Connection:
    index = paths.cache / "dedup_audit.sqlite3"
    if not index.is_file():
        raise FileNotFoundError("Pre-split V2.1 dedup index is required before screening candidates")
    connection = sqlite3.connect(f"file:{index.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    coverage = connection.execute(
        "SELECT COUNT(*) FROM rows WHERE dataset='v2_reference'"
    ).fetchone()[0]
    # The pre-split audit indexed 38,013 V2 Train/Validation plus 6,704
    # frozen Test rows. A fingerprint-only lookup must cover both without
    # opening any held-out email or label file.
    if coverage < 44_717:
        connection.close()
        raise ValueError("Pre-split V2 reference fingerprint coverage is incomplete")
    return connection


def _reference_hits(
    connection: sqlite3.Connection, hashes: tuple[str, str, str], synthetic: set[str],
) -> list[str]:
    found: set[str] = set()
    for column, value in zip(("exact_hash", "normalized_hash", "template_hash"), hashes):
        if not value:
            continue
        if value in synthetic:
            found.add("research_synthetic_zh")
        # Column names are constants above; only digest values are parameters.
        for (dataset,) in connection.execute(
            f"SELECT DISTINCT dataset FROM rows WHERE {column}=?", (value,)
        ):
            found.add(dataset)
    return sorted(found)


def _flags(row: dict[str, str], hashes: tuple[str, str, str]) -> list[str]:
    flags: list[str] = []
    if row["source"].casefold() in BLOCKED_SOURCES:
        flags.append("already_used_or_synthetic_source")
    if row["data_origin"].casefold() != "original":
        flags.append("not_original_email")
    if row["license_status"] != "verified_research_use" or not row["permission_evidence"]:
        flags.append("permission_not_documented")
    if row["language_claim"].casefold() != "zh":
        flags.append("language_not_claimed_zh")
    han = sum("\u3400" <= char <= "\u9fff" for char in row["subject"] + row["body"])
    if han < 20:
        flags.append("insufficient_han_text")
    if not hashes[1]:
        flags.append("empty_email")
    if row["proposed_label"] not in LABELS:
        flags.append("missing_three_class_proposal")
    if not row["label_evidence"]:
        flags.append("missing_label_evidence")
    if (not row["reviewer_1_id"] or not row["reviewer_2_id"]
            or row["reviewer_1_id"] == row["reviewer_2_id"]
            or row["reviewer_1_label"] != row["proposed_label"]
            or row["reviewer_2_label"] != row["proposed_label"]):
        flags.append("independent_chinese_review_unverified")
    if row["parent_id"]:
        flags.append("parent_derivative_relation_requires_review")
    return flags


def _prepare_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE candidates (
            candidate_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL,
            exact_hash TEXT, normalized_hash TEXT, template_hash TEXT
        );
        CREATE INDEX idx_candidates_exact ON candidates(exact_hash);
        CREATE INDEX idx_candidates_normalized ON candidates(normalized_hash);
        CREATE INDEX idx_candidates_template ON candidates(template_hash);
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE duplicate_kinds (
            candidate_id TEXT NOT NULL, kind TEXT NOT NULL,
            PRIMARY KEY (candidate_id, kind)
        );
    """)
    return connection


def prepare_candidate_manifest(
    input_path: Path | None = None, *, paths: V21Paths | None = None,
) -> dict[str, Any]:
    """Screen an explicitly supplied local CSV, or publish an empty pending manifest.

    Only metadata and fingerprints are exported. The original email remains
    solely in the caller's local source file. No row becomes benchmark-ready.
    """
    paths = paths or resolve_v21_paths()
    output = _output_dir(paths)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "candidate_manifest.csv"
    store_path = output / "screening.sqlite3"
    summary_path = output / "manifest.json"
    if input_path is None and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("candidate_rows", 0):
            raise ValueError("Existing candidate intake must not be replaced by an empty run")
    input_sha = None
    source_rows = 0
    references_checked = False
    synthetic = set()
    reference = None
    if input_path is not None:
        input_path = Path(input_path)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        input_sha = _sha256_file(input_path)
        synthetic = _synthetic_hashes(paths)
        reference = _reference_connection(paths)
        references_checked = True

    committed_rows = 0
    if input_path is not None and store_path.is_file():
        existing = sqlite3.connect(store_path)
        try:
            has_metadata = existing.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
            ).fetchone()
            saved_sha = (existing.execute(
                "SELECT value FROM metadata WHERE key='input_sha256'"
            ).fetchone() if has_metadata else None)
            if saved_sha:
                if saved_sha[0] != input_sha:
                    raise ValueError("Candidate input changed since screening checkpoint")
                committed = existing.execute(
                    "SELECT value FROM metadata WHERE key='committed_rows'"
                ).fetchone()
                committed_rows = int(committed[0]) if committed else 0
            elif summary_path.is_file() and json.loads(
                summary_path.read_text(encoding="utf-8")
            ).get("candidate_rows", 0):
                raise ValueError("Existing candidate screening has no verifiable checkpoint")
        finally:
            existing.close()
        if not saved_sha:
            store_path.unlink()  # Only the fixed empty derived store, never source email data.
    elif input_path is None and store_path.exists():
        store_path.unlink()  # Empty-manifest rerun only; nonempty prior run is blocked above.

    store = sqlite3.connect(store_path) if store_path.exists() else _prepare_store(store_path)
    try:
        if input_path is not None:
            def write_progress(rows: int) -> None:
                summary_path.write_text(json.dumps({
                    "scope": "chinese_real_world_benchmark_preparation_only",
                    "benchmark_status": "incomplete",
                    "final_evaluation_ready": False,
                    "input_sha256": input_sha,
                    "committed_rows": rows,
                    "held_out_test_opened": False,
                    "model_evaluated": False,
                }, indent=2) + "\n", encoding="utf-8")

            store.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('input_sha256', ?)", (input_sha,)
            )
            store.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('committed_rows', ?)",
                (str(committed_rows),),
            )
            store.commit()
            write_progress(committed_rows)
            csv.field_size_limit(100_000_000)
            with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not set(INPUT_FIELDS).issubset(reader.fieldnames or ()):
                    raise ValueError("Candidate input is missing required intake columns")
                for raw in reader:
                    source_rows += 1
                    if source_rows <= committed_rows:
                        continue
                    row = {key: (raw.get(key) or "").strip() for key in INPUT_FIELDS}
                    if not row["candidate_id"] or not row["source_record_id"] or not row["source"]:
                        raise ValueError(f"Missing stable candidate/source ID at intake row {source_rows}")
                    hashes = dedup_hashes(row["subject"], row["body"])
                    hits = _reference_hits(reference, hashes, synthetic)
                    flags = _flags(row, hashes)
                    if hits:
                        flags.append("existing_corpus_fingerprint_overlap")
                    metadata = {key: row[key] for key in MANIFEST_FIELDS if key in row and key not in {"subject", "body"}}
                    metadata.update({
                        "exact_hash": hashes[0], "normalized_hash": hashes[1],
                        "template_hash": hashes[2], "reference_overlap": ";".join(hits),
                        "screening_flags": ";".join(flags), "benchmark_status": "pending",
                    })
                    store.execute(
                        "INSERT INTO candidates VALUES (?,?,?,?,?)",
                        (row["candidate_id"], json.dumps(metadata, ensure_ascii=False), *hashes),
                    )
                    if source_rows % 1000 == 0:
                        store.execute(
                            "UPDATE metadata SET value=? WHERE key='committed_rows'",
                            (str(source_rows),),
                        )
                        store.commit()
                        write_progress(source_rows)
            store.execute(
                "UPDATE metadata SET value=? WHERE key='committed_rows'",
                (str(source_rows),),
            )
            store.commit()
            write_progress(source_rows)

        store.execute("DELETE FROM duplicate_kinds")
        duplicate_hash_groups: dict[str, int] = {}
        for kind, column in (("exact", "exact_hash"), ("normalized", "normalized_hash"),
                             ("template", "template_hash")):
            duplicate_hash_groups[kind] = store.execute(
                f"SELECT COUNT(*) FROM (SELECT {column} FROM candidates "
                f"WHERE {column}<>'' GROUP BY {column} HAVING COUNT(*)>1)"
            ).fetchone()[0]
            store.execute(
                f"INSERT INTO duplicate_kinds "
                f"SELECT candidate_id, ? FROM candidates WHERE {column}<>'' "
                f"AND {column} IN (SELECT {column} FROM candidates WHERE {column}<>'' "
                f"GROUP BY {column} HAVING COUNT(*)>1)", (kind,),
            )
        store.commit()
        duplicate_rows = store.execute(
            "SELECT COUNT(DISTINCT candidate_id) FROM duplicate_kinds"
        ).fetchone()[0]
        flags_count: Counter[str] = Counter()
        overlap_count = 0
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for candidate_id, serialized in store.execute(
                "SELECT candidate_id, manifest_json FROM candidates ORDER BY candidate_id"
            ):
                metadata = json.loads(serialized)
                kinds = [kind for (kind,) in store.execute(
                    "SELECT kind FROM duplicate_kinds WHERE candidate_id=? ORDER BY kind",
                    (candidate_id,),
                )]
                metadata["internal_duplicate_kinds"] = ";".join(kinds)
                if kinds:
                    metadata["screening_flags"] = ";".join(filter(None, (
                        metadata["screening_flags"], "internal_duplicate_group")))
                if metadata["reference_overlap"]:
                    overlap_count += 1
                flags_count.update(filter(None, metadata["screening_flags"].split(";")))
                writer.writerow({key: metadata.get(key, "") for key in MANIFEST_FIELDS})
    finally:
        store.close()
        if reference is not None:
            reference.close()

    summary = {
        "scope": "chinese_real_world_benchmark_preparation_only",
        "benchmark_status": "pending",
        "final_evaluation_ready": False,
        "input_sha256": input_sha,
        "candidate_rows": source_rows,
        "accepted_benchmark_rows": 0,
        "reference_fingerprints_checked": references_checked,
        "reference_overlap_rows": overlap_count,
        "internal_duplicate_rows": duplicate_rows,
        "internal_duplicate_hash_groups": duplicate_hash_groups,
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "screening_flag_counts": dict(sorted(flags_count.items())),
        "held_out_test_opened": False,
        "model_evaluated": False,
        "limitations": [
            "No independently licensed, Chinese-language, human-verified three-class corpus was admitted.",
            "Screening cannot verify a reviewer's Chinese-language competence or legal permission by itself.",
            "Exact/normalized/template fingerprints cannot prove absence of semantic or cross-language overlap.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
