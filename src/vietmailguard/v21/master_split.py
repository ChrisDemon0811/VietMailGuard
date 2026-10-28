"""V2.1-only, review-gated master preparation and disk-backed group splitting.

The current audit has no V2.1 training-eligible rows. The real run therefore
emits a blocked manifest, not empty or weakly labeled train/test files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .dedup_audit import dedup_hashes
from .paths import V21Paths, resolve_v21_paths


SCHEMA = (
    "id", "source", "language", "subject", "body", "raw_label", "label",
    "label_status", "label_provenance", "data_origin", "parent_id",
    "translation_source_id", "final_group_id", "content_hash", "split",
)
REQUIRED_INPUT = set(SCHEMA) - {"final_group_id", "content_hash", "split", "translation_source_id"}
SPLITS = ("train", "validation", "test")
LABELS = ("normal", "spam", "phishing")
LANGUAGES = ("en", "vi", "zh")


class PreparationBlocked(RuntimeError):
    """The scientific eligibility/leakage preflight did not pass."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _policy(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("seed") != 42:
        raise ValueError("V2.1 split policy requires schema version 1 and seed 42")
    if tuple(data.get("ratios", {}).keys()) != SPLITS:
        raise ValueError("Split ratios must list train, validation, test in order")
    if abs(sum(data["ratios"].values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one")
    if set(data.get("allowed_labels", [])) != set(LABELS):
        raise ValueError("All three canonical labels must be configured")
    if set(data.get("allowed_languages", [])) != set(LANGUAGES):
        raise ValueError("English, Vietnamese, and Chinese must be configured")
    return data


def _base_manifest(paths: V21Paths, policy_path: Path, audit_summary: Path) -> dict:
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    index = paths.cache / "dedup_audit.sqlite3"
    if not index.is_file():
        raise FileNotFoundError(f"V2.1 audit index missing: {index}")
    return {
        "schema_version": 1, "created_at_utc": _utc(), "seed": 42,
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "schema": list(SCHEMA), "audit_summary_sha256": sha256_file(audit_summary),
        "audit_index_sha256": sha256_file(index),
        "policy_sha256": sha256_file(policy_path),
        "audit_v21_source_rows": audit.get("v21_source_rows"),
        "audit_v21_training_eligible_rows": audit.get("v21_training_eligible_rows"),
        "audit_status_counts": audit.get("label_status", {}),
        "frozen_v2_reference_rows": audit.get("read_only_v2_reference_rows"),
        "v2_reference_rule": "No V2 reference row or exact/normalized/template/parent overlap is admitted",
        "language_class_counts": {language: {label: 0 for label in LABELS} for language in LANGUAGES},
        "missing_language_classes": [f"{language}/{label}" for language in LANGUAGES for label in LABELS],
        "input_files": [], "output_files": {}, "leakage_checks": "not_run",
    }


def _write_report(paths: V21Paths, manifest: dict) -> None:
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "multilingual_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# V2.1 multilingual master and split preparation", "",
             f"Status: **{manifest['status'].upper()}**", "",
             "Command: `python -m scripts.v21.build_multilingual_master`.", "",
             f"Seed: {manifest['seed']}; target train/validation/test: 70/15/15.", "",
             f"Audited V2.1 source rows: {manifest['audit_v21_source_rows']:,}; "
             f"V2.1 training-eligible after audit: {manifest['audit_v21_training_eligible_rows']:,}.", ""]
    if manifest["status"] == "blocked":
        lines += ["No Multilingual Master Dataset or split CSV was created. Reasons:", ""]
        lines += [f"- {reason}" for reason in manifest["blocked_reasons"]]
    else:
        lines += [f"Eligible rows in master: {manifest['eligible_rows']:,}; final groups: {manifest['final_groups']:,}.",
                  "", "| Split | Rows |", "|---|---:|"]
        lines += [f"| {name} | {manifest['split_counts'][name]:,} |" for name in SPLITS]
        lines += ["", "Observed ratios: " + ", ".join(
            f"{name} {manifest['observed_ratios'][name]:.1%}" for name in SPLITS) + ".",
            "Group integrity has priority over the target ratios; small or large linked groups can cause deviations.",
            "", f"Leakage checks: **{manifest['leakage_checks']}**."]
    lines += ["", "## Confirmed labels by email language", "",
              "| Language | normal | spam | phishing |", "|---|---:|---:|---:|"]
    for language in LANGUAGES:
        counts = manifest["language_class_counts"][language]
        lines.append(f"| {language} | {counts['normal']:,} | {counts['spam']:,} | {counts['phishing']:,} |")
    lines += ["", "Zero means no confirmed eligible examples; it is not an evaluation metric.",
              "No native-language or phishing performance claim is made.",
              "V2 frozen held-out data is not used as V2.1 training/test material.",
              "Source rows with `review`, `exclude`, or unapproved provenance never enter supervised splits.",
              "The exact input/output hashes and any leakage checks are recorded in `multilingual_split_manifest.json`."]
    (paths.reports / "multilingual_master_split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _blocked(paths: V21Paths, manifest: dict, reasons: list[str]) -> None:
    manifest.update(status="blocked", blocked_reasons=reasons)
    _write_report(paths, manifest)
    raise PreparationBlocked("; ".join(reasons))


def _stage(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, language TEXT NOT NULL,
            label TEXT NOT NULL, raw_label TEXT, label_provenance TEXT NOT NULL,
            data_origin TEXT NOT NULL, parent_id TEXT, translation_source_id TEXT,
            declared_group TEXT, exact_hash TEXT NOT NULL, normalized_hash TEXT NOT NULL,
            template_hash TEXT, input_path TEXT NOT NULL, input_row INTEGER NOT NULL,
            root_id TEXT, split TEXT
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            path TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
            sha256 TEXT NOT NULL, processed_rows INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rejected (reason TEXT PRIMARY KEY, rows INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS parents (id TEXT PRIMARY KEY, parent TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_items_exact ON items(exact_hash);
        CREATE INDEX IF NOT EXISTS idx_items_normalized ON items(normalized_hash);
        CREATE INDEX IF NOT EXISTS idx_items_template ON items(template_hash);
        CREATE INDEX IF NOT EXISTS idx_items_parent ON items(parent_id);
        CREATE INDEX IF NOT EXISTS idx_items_declared ON items(declared_group);
        CREATE INDEX IF NOT EXISTS idx_items_root ON items(root_id);
    """)
    return conn


def _audit_match(conn: sqlite3.Connection, row_id: str, source: str, hashes: tuple[str, str, str]) -> tuple[str, str]:
    rows = conn.execute("""SELECT source, exact_hash, audit_reason FROM rows
        WHERE dataset!='v2_reference' AND original_id=?""", (row_id,)).fetchall()
    if len(rows) != 1 or rows[0][0] != source or rows[0][1] != hashes[0]:
        raise ValueError(f"Curated row {row_id!r} does not match one audited source/text")
    return rows[0][1], rows[0][2]


def _v2_overlap(conn: sqlite3.Connection, row_id: str, parent_id: str, declared_group: str,
                hashes: tuple[str, str, str]) -> bool:
    exact, normalized, template = hashes
    return conn.execute("""SELECT 1 FROM rows WHERE dataset='v2_reference' AND (
        (exact_hash=? AND exact_hash!='') OR (normalized_hash=? AND normalized_hash!='')
        OR (template_hash=? AND template_hash!='') OR original_id=? OR original_id=?
        OR (v2_group_id=? AND v2_group_id!='')) LIMIT 1""",
        (exact, normalized, template, row_id, parent_id, declared_group)).fetchone() is not None


def _candidate(raw: dict[str, str], policy: dict, audit: sqlite3.Connection) -> tuple[tuple | None, str | None]:
    row_id = (raw.get("id") or "").strip()
    if not row_id:
        raise ValueError("Curated input row is missing id")
    if raw.get("label_status") != "eligible" or str(raw.get("training_eligible", "")).casefold() != "true":
        return None, "not_eligible"
    if raw.get("label") not in policy["allowed_labels"]:
        return None, "unverified_label"
    if raw.get("label_provenance") not in policy["approved_label_provenance"]:
        return None, "unapproved_label_provenance"
    if raw.get("language") not in policy["allowed_languages"]:
        return None, "unsupported_or_unverified_language"
    if not raw.get("data_origin") or raw["data_origin"] == "unknown":
        return None, "unknown_data_origin"
    if raw["data_origin"] == "translated" and not raw.get("parent_id"):
        return None, "translated_row_missing_parent"
    hashes = dedup_hashes(raw.get("subject"), raw.get("body"))
    if not hashes[1]:
        return None, "empty_content"
    _, audit_reason = _audit_match(audit, row_id, raw.get("source", ""), hashes)
    if audit_reason in {"empty_content", "duplicate_label_conflict", "candidate_mapping_inconsistent"}:
        return None, "audit_review_or_exclude"
    parent_id = raw.get("parent_id", "") or ""
    declared_group = raw.get("final_group_id", "") or ""
    if _v2_overlap(audit, row_id, parent_id, declared_group, hashes):
        return None, "frozen_v2_reference_overlap"
    return (
        row_id, raw["source"], raw["language"], raw["label"], raw.get("raw_label", ""),
        raw["label_provenance"], raw["data_origin"], parent_id,
        raw.get("translation_source_id", "") or "", declared_group,
        hashes[0], hashes[1], hashes[2], "", 0, None, None,
    ), None


def _ingest_part(stage: sqlite3.Connection, audit: sqlite3.Connection, path: Path,
                 input_sha256: str, policy: dict) -> None:
    if sha256_file(path) != input_sha256:
        raise RuntimeError(f"Curated input changed during preparation: {path}")
    key = str(path.resolve())
    stat = path.stat()
    old = stage.execute("SELECT size_bytes,mtime_ns,sha256,processed_rows FROM checkpoints WHERE path=?", (key,)).fetchone()
    if old and old[:3] != (stat.st_size, stat.st_mtime_ns, input_sha256):
        raise RuntimeError(f"Curated input changed since checkpoint: {path}")
    processed = old[3] if old else 0
    csv.field_size_limit(100_000_000)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_INPUT.union({"training_eligible"}).issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid curated schema: {path}")
        batch: list[tuple] = []
        skipped: Counter[str] = Counter()
        for line, raw in enumerate(reader, start=1):
            if line <= processed:
                continue
            candidate, reason = _candidate(raw, policy, audit)
            if candidate:
                mutable = list(candidate)
                mutable[-4], mutable[-3] = key, line
                batch.append(tuple(mutable))
            else:
                skipped[reason or "unknown"] += 1
            if line % 1000 == 0:
                _commit_batch(stage, key, stat, input_sha256, line, batch, skipped)
                batch.clear()
                skipped.clear()
        final_line = line if 'line' in locals() else 0
        if final_line > processed:
            _commit_batch(stage, key, stat, input_sha256, final_line, batch, skipped)


def _commit_batch(stage: sqlite3.Connection, key: str, stat: os.stat_result, input_sha256: str, line: int,
                  batch: list[tuple], skipped: Counter[str]) -> None:
    with stage:
        stage.executemany("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        stage.executemany("INSERT INTO parents(id,parent) VALUES (?,?)", ((row[0], row[0]) for row in batch))
        stage.executemany("""INSERT INTO rejected(reason,rows) VALUES (?,?)
            ON CONFLICT(reason) DO UPDATE SET rows=rows+excluded.rows""", skipped.items())
        stage.execute("""INSERT INTO checkpoints(path,size_bytes,mtime_ns,sha256,processed_rows)
            VALUES (?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET processed_rows=excluded.processed_rows""",
            (key, stat.st_size, stat.st_mtime_ns, input_sha256, line))


def _find(conn: sqlite3.Connection, key: str) -> str:
    path: list[str] = []
    while True:
        result = conn.execute("SELECT parent FROM parents WHERE id=?", (key,)).fetchone()
        if result is None:
            raise ValueError(f"Unknown group member: {key}")
        parent = result[0]
        if parent == key:
            break
        path.append(key)
        key = parent
    if path:
        conn.executemany("UPDATE parents SET parent=? WHERE id=?", ((key, child) for child in path))
    return key


def _union(conn: sqlite3.Connection, left: str, right: str) -> None:
    a, b = _find(conn, left), _find(conn, right)
    if a != b:
        smaller, larger = sorted((a, b))
        conn.execute("UPDATE parents SET parent=? WHERE id=?", (smaller, larger))


def _link_by_column(conn: sqlite3.Connection, column: str) -> None:
    previous_value = ""
    first_id = ""
    with conn:
        for row_id, value in conn.execute(f"SELECT id,{column} FROM items WHERE {column}!='' ORDER BY {column},id"):
            if value != previous_value:
                first_id, previous_value = row_id, value
            else:
                _union(conn, first_id, row_id)


def _group(conn: sqlite3.Connection) -> None:
    for column in ("exact_hash", "normalized_hash", "template_hash", "declared_group", "parent_id", "translation_source_id"):
        _link_by_column(conn, column)
    with conn:
        for row_id, parent_id in conn.execute("SELECT id,parent_id FROM items WHERE parent_id!=''"):
            if conn.execute("SELECT 1 FROM parents WHERE id=?", (parent_id,)).fetchone():
                _union(conn, row_id, parent_id)
        updates: list[tuple[str, str]] = []
        for (row_id,) in conn.execute("SELECT id FROM items"):
            updates.append((_find(conn, row_id), row_id))
            if len(updates) >= 1000:
                conn.executemany("UPDATE items SET root_id=? WHERE id=?", updates)
                updates.clear()
        if updates:
            conn.executemany("UPDATE items SET root_id=? WHERE id=?", updates)


def _assign(conn: sqlite3.Connection, policy: dict) -> dict[str, int]:
    conflicts = conn.execute("""SELECT count(*) FROM (SELECT root_id FROM items
        GROUP BY root_id HAVING count(DISTINCT label)>1)""").fetchone()[0]
    if conflicts:
        raise PreparationBlocked(f"{conflicts} related groups have conflicting confirmed labels; review required")
    totals = dict(conn.execute("SELECT label,count(*) FROM items GROUP BY label"))
    groups = dict(conn.execute("SELECT label,count(*) FROM (SELECT root_id,label FROM items GROUP BY root_id,label) GROUP BY label"))
    minimum = policy["minimum_independent_groups_per_class"]
    if any(groups.get(label, 0) < minimum for label in LABELS):
        raise PreparationBlocked(f"Each class needs at least {minimum} independent groups; observed {groups}")
    assigned = {label: Counter() for label in LABELS}
    group_ordinals = Counter()
    with conn:
        for root, label, size in conn.execute("""SELECT root_id,label,count(*) AS n FROM items
            GROUP BY root_id,label ORDER BY label,n DESC,root_id"""):
            ordinal = group_ordinals[label]
            group_ordinals[label] += 1
            if ordinal < 3:
                split = SPLITS[ordinal]
            else:
                # Largest remaining class-specific deficit; stable seed-based tie order.
                order = sorted(SPLITS, key=lambda value: hashlib.sha256(f"{policy['seed']}:{root}:{value}".encode()).hexdigest())
                split = max(order, key=lambda value: (totals[label] * policy["ratios"][value] - assigned[label][value], -order.index(value)))
            conn.execute("UPDATE items SET split=? WHERE root_id=?", (split, root))
            assigned[label][split] += size
    return dict(conn.execute("SELECT split,count(*) FROM items GROUP BY split"))


def _validate(conn: sqlite3.Connection) -> dict[str, int]:
    findings: dict[str, int] = {}
    for column in ("root_id", "exact_hash", "normalized_hash", "template_hash", "declared_group", "parent_id", "translation_source_id"):
        findings[column] = conn.execute(f"""SELECT count(*) FROM (SELECT {column} FROM items
            WHERE {column}!='' GROUP BY {column} HAVING count(DISTINCT split)>1)""").fetchone()[0]
    invalid = conn.execute("""SELECT count(*) FROM items WHERE split NOT IN ('train','validation','test')
        OR label NOT IN ('normal','spam','phishing') OR split IS NULL""").fetchone()[0]
    findings["invalid_rows"] = invalid
    if any(findings.values()):
        raise PreparationBlocked(f"Cross-split leakage or invalid rows: {findings}")
    return findings


def _output_row(raw: dict[str, str], staged: tuple[str, str, str]) -> dict[str, str]:
    root, split, exact_hash = staged
    return {
        "id": raw["id"], "source": raw["source"], "language": raw["language"],
        "subject": raw.get("subject", ""), "body": raw.get("body", ""),
        "raw_label": raw.get("raw_label", ""), "label": raw["label"],
        "label_status": "eligible", "label_provenance": raw["label_provenance"],
        "data_origin": raw["data_origin"], "parent_id": raw.get("parent_id", ""),
        "translation_source_id": raw.get("translation_source_id", ""),
        "final_group_id": "v21group_" + hashlib.sha256(root.encode()).hexdigest()[:24],
        "content_hash": exact_hash, "split": split,
    }


def _export(paths: V21Paths, parts: list[Path], conn: sqlite3.Connection) -> dict[str, dict]:
    master = paths.processed / "Multilingual_Master_Dataset.csv"
    outputs = {"master": master, **{name: paths.splits / f"{name}.csv" for name in SPLITS}}
    if any(path.exists() for path in outputs.values()):
        raise PreparationBlocked("Refusing to overwrite an existing V2.1 master or split")
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temp = {name: path.with_suffix(".csv.incomplete") for name, path in outputs.items()}
    handles = {}
    try:
        writers = {}
        for name, path in temp.items():
            handles[name] = path.open("w", encoding="utf-8", newline="")
            writers[name] = csv.DictWriter(handles[name], fieldnames=SCHEMA)
            writers[name].writeheader()
        for part in parts:
            with part.open("r", encoding="utf-8-sig", newline="") as handle:
                for line, raw in enumerate(csv.DictReader(handle), start=1):
                    staged = conn.execute("""SELECT root_id,split,exact_hash FROM items
                        WHERE id=? AND input_path=? AND input_row=?""",
                        (raw.get("id", ""), str(part.resolve()), line)).fetchone()
                    if staged is None:
                        continue
                    row = _output_row(raw, staged)
                    writers["master"].writerow(row)
                    writers[staged[1]].writerow(row)
        for handle in handles.values():
            handle.close()
        handles.clear()
        for name, path in outputs.items():
            os.replace(temp[name], path)
    finally:
        for handle in handles.values():
            handle.close()
    return {name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for name, path in outputs.items()}


def build_master_split(paths: V21Paths | None = None, *, policy_path: Path | None = None) -> dict:
    """Build only from independently curated V2.1 rows; block otherwise."""
    paths = paths or resolve_v21_paths()
    policy_path = policy_path or paths.config / "master_split_policy.json"
    policy = _policy(policy_path)
    summary_path = paths.reports / "dedup_audit_summary.json"
    manifest = _base_manifest(paths, policy_path, summary_path)
    parts = sorted(paths.curated.glob(policy["curated_glob"]))
    curated_root = paths.curated.resolve()
    if any(not part.resolve().is_relative_to(curated_root) for part in parts):
        raise ValueError("Curated input glob resolved outside the V2.1 curated directory")
    if not parts:
        _blocked(paths, manifest, ["No approved V2.1 curated CSV parts exist under data/v21/curated/eligible/",
                                   "Audit reports zero V2.1 training-eligible rows; all V2 reference rows are read-only"])
    if not policy["approved_label_provenance"]:
        _blocked(paths, manifest, ["No approved V2.1 label provenance has been configured after curation"])
    if (paths.processed / "Multilingual_Master_Dataset.csv").exists() or any(
        (paths.splits / f"{name}.csv").exists() for name in SPLITS
    ):
        raise PreparationBlocked("Refusing to overwrite an existing V2.1 master or split")
    manifest["input_files"] = [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                               for path in parts]
    stage = _stage(paths.cache / "master_split_stage.sqlite3")
    audit = sqlite3.connect(paths.cache / "dedup_audit.sqlite3")
    try:
        for key in ("policy_sha256", "audit_index_sha256"):
            saved = stage.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
            if saved and saved[0] != manifest[key]:
                _blocked(paths, manifest, [f"Stage checkpoint was created with a different {key}; rebuild the local stage after review"])
            stage.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)", (key, manifest[key]))
        stage.commit()
        for part, fingerprint in zip(parts, manifest["input_files"]):
            _ingest_part(stage, audit, part, fingerprint["sha256"], policy)
        count = stage.execute("SELECT count(*) FROM items").fetchone()[0]
        manifest["curated_rows_seen"] = sum(row[0] for row in stage.execute("SELECT processed_rows FROM checkpoints"))
        manifest["curated_rejection_reasons"] = dict(stage.execute("SELECT reason,rows FROM rejected"))
        if not count:
            _blocked(paths, manifest, ["No curated rows pass eligibility, provenance, audit, and V2-reference guards"])
        missing_parents = stage.execute("""SELECT count(*) FROM items child
            WHERE child.data_origin='translated' AND child.parent_id!=''
            AND NOT EXISTS (SELECT 1 FROM items parent WHERE parent.id=child.parent_id)""").fetchone()[0]
        if missing_parents:
            _blocked(paths, manifest, [f"{missing_parents} translated eligible rows lack an eligible parent in this build"])
        _group(stage)
        try:
            split_counts = _assign(stage, policy)
            checks = _validate(stage)
        except PreparationBlocked as error:
            _blocked(paths, manifest, [str(error)])
        language_counts = {language: {label: 0 for label in LABELS} for language in LANGUAGES}
        for language, label, n in stage.execute("SELECT language,label,count(*) FROM items GROUP BY language,label"):
            language_counts[language][label] = n
        manifest["language_class_counts"] = language_counts
        manifest["missing_language_classes"] = [f"{language}/{label}" for language in LANGUAGES
                                                for label in LABELS if language_counts[language][label] == 0]
        manifest["eligible_rows"] = count
        manifest["final_groups"] = stage.execute("SELECT count(DISTINCT root_id) FROM items").fetchone()[0]
        manifest["split_counts"] = split_counts
        manifest["observed_ratios"] = {name: split_counts.get(name, 0) / count for name in SPLITS}
        by_split = {split: {language: {label: 0 for label in LABELS} for language in LANGUAGES}
                    for split in SPLITS}
        for split, language, label, n in stage.execute("""SELECT split,language,label,count(*)
            FROM items GROUP BY split,language,label"""):
            by_split[split][language][label] = n
        manifest["split_language_class_counts"] = by_split
        manifest["leakage_checks"] = checks
        if sha256_file(paths.cache / "dedup_audit.sqlite3") != manifest["audit_index_sha256"]:
            _blocked(paths, manifest, ["Audit index changed during V2.1 preparation"])
        manifest["output_files"] = _export(paths, parts, stage)
        manifest["status"] = "complete"
        _write_report(paths, manifest)
        return manifest
    finally:
        audit.close()
        stage.close()
