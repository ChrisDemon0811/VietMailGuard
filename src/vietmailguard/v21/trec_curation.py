"""TREC06C-only, evidence-gated human curation. Never changes source or V2 data."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .dedup_audit import dedup_hashes
from .paths import V21Paths, resolve_v21_paths


SEED = 42
QUEUE_FIELDS = (
    "id", "source", "raw_label", "original_path", "part_path", "subject",
    "body_excerpt", "length_bucket", "quality_stratum", "duplicate_stratum",
    "quality_flags", "audit_reason", "exact_group_id", "normalized_group_id",
    "template_group_id", "human_decision", "human_note",
)
DECISION_FIELDS = (
    "id", "source", "raw_label", "original_path", "decision", "label_status",
    "label_provenance", "curation_rule_id", "reviewer", "review_method",
    "reviewed_at_utc", "evidence", "content_hash", "normalized_hash",
    "exact_group_id", "normalized_group_id", "template_group_id", "final_group_id",
    "eligibility_blocker", "review_cohort", "review_stratum",
)
LABELS = {"normal", "spam", "phishing"}
DECISIONS = LABELS | {"review", "exclude"}
PART_DIR = "trec06c_mime_v2"
RULE_ID = "trec06c_individual_human_content_review_v1"
MANUAL_METHODS = {"full_message_manual_review", "dual_human_review"}


def _validate_policy(paths: V21Paths) -> None:
    policy = json.loads((paths.config / "trec06c_curation_policy.json").read_text(encoding="utf-8"))
    if (policy.get("schema_version") != 1 or policy.get("source") != "TREC06C"
            or policy.get("bulk_approval") is not False
            or policy.get("automatic_phishing_from_spam") is not False
            or policy.get("individual_review_rule_id") != RULE_ID
            or set(policy.get("allowed_manual_decisions", [])) != DECISIONS):
        raise ValueError("Unsafe or inconsistent TREC curation policy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: str) -> bool:
    return value.casefold() == "true"


def quality_flags(row: dict[str, str]) -> list[str]:
    flags = []
    for field in ("empty", "missing_content", "decode_error", "mime_recovered", "mime_defects", "html_only"):
        if row.get(field) and (_bool(row[field]) if field not in {"mime_defects"} else True):
            flags.append(field)
    if not _bool(row.get("contains_chinese", "")):
        flags.append("no_chinese_characters")
    if row.get("status") != "ok":
        flags.append("source_error")
    return flags


def _quality_stratum(flags: list[str]) -> str:
    for flag in ("source_error", "empty", "missing_content", "decode_error", "no_chinese_characters", "mime_recovered", "mime_defects", "html_only"):
        if flag in flags:
            return flag
    return "clean"


def _length_bucket(subject: str, body: str) -> str:
    length = len(subject) + len(body)
    return "short" if length < 80 else "medium" if length < 600 else "long"


def _duplicate_stratum(audit: sqlite3.Row) -> str:
    for field, name in (("exact_group_id", "exact"), ("normalized_group_id", "normalized"), ("template_group_id", "template")):
        if audit[field]:
            return name
    return "singleton"


def _parts(paths: V21Paths) -> list[Path]:
    parts = sorted((paths.processed / PART_DIR).glob("part-*.csv"))
    if not parts:
        raise FileNotFoundError("Authoritative TREC06C MIME-v2 CSV parts are missing")
    return parts


def _audit_connection(paths: V21Paths) -> sqlite3.Connection:
    path = paths.cache / "dedup_audit.sqlite3"
    if not path.is_file():
        raise FileNotFoundError("V2.1 dedup audit index is missing")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _validate_part_checkpoint(conn: sqlite3.Connection, part: Path) -> None:
    stat = part.stat()
    checkpoint = conn.execute(
        "SELECT size_bytes,mtime_ns FROM checkpoints WHERE path=?", (str(part.resolve()),)
    ).fetchone()
    if checkpoint is None or tuple(checkpoint) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(f"TREC source changed since dedup audit: {part}")


def _conflict_groups(conn: sqlite3.Connection) -> dict[str, set[str]]:
    result = {}
    for column in ("exact_group_id", "normalized_group_id", "template_group_id"):
        result[column] = {
            row[0] for row in conn.execute(
                f"SELECT {column} FROM rows WHERE dataset='trec06c' AND {column}!='' "
                f"GROUP BY {column} HAVING COUNT(DISTINCT raw_label)>1"
            )
        }
    return result


def _read_decisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        optional = {"eligibility_blocker", "review_cohort", "review_stratum"}
        if not (set(DECISION_FIELDS) - optional).issubset(reader.fieldnames or ()):
            raise ValueError("TREC decision manifest schema is invalid")
        decisions = {}
        for row in reader:
            for field in optional:
                row.setdefault(field, "")
            if row["id"] in decisions or row["decision"] not in DECISIONS:
                raise ValueError("Duplicate or invalid TREC decision")
            if not all(row.get(key, "").strip() for key in ("reviewer", "review_method", "reviewed_at_utc", "evidence")):
                raise ValueError("TREC decision lacks reviewer, method, time or evidence")
            if (row["source"] != "TREC06C" or not row["id"].startswith("trec06c:")
                    or row["raw_label"] not in {"ham", "spam"}
                    or row["review_method"] not in MANUAL_METHODS
                    or not row["original_path"] or not row["content_hash"]
                    or not row["normalized_hash"] or not row["final_group_id"]):
                raise ValueError("TREC decision manifest has invalid provenance")
            if row["decision"] in LABELS:
                if row["label_status"] == "eligible":
                    if (row["curation_rule_id"] != RULE_ID
                            or row["label_provenance"] != "trec06c_individual_human_review"
                            or row.get("eligibility_blocker", "")):
                        raise ValueError("TREC eligible decision lacks approved rule/provenance")
                elif (row["label_status"] != "review" or not row.get("eligibility_blocker")
                      or row["curation_rule_id"] or row["label_provenance"]):
                    raise ValueError("Blocked TREC class decision was marked eligible")
            elif row["label_status"] != row["decision"] or row["curation_rule_id"] or row["label_provenance"]:
                raise ValueError("Noneligible TREC decision was marked eligible")
            parsed = datetime.fromisoformat(row["reviewed_at_utc"].replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
                raise ValueError("TREC decision time must be UTC")
            decisions[row["id"]] = row
        return decisions


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _eligible_gate(conn: sqlite3.Connection, row: sqlite3.Row, flags: list[str], conflicts: dict[str, set[str]]) -> str | None:
    if row["dataset"] != "trec06c" or row["label_status"] == "exclude":
        return "excluded_or_wrong_source"
    if row["raw_label"] not in {"ham", "spam"} or row["audit_reason"] != "trec_ham_spam_candidate_not_vmg_three_class_label":
        return "provenance_or_label_inconsistent"
    if any(flag in flags for flag in ("empty", "missing_content", "decode_error", "no_chinese_characters", "source_error")):
        return "content_quality_review"
    if row["language"] != "zh" or row["data_origin"] != "original":
        return "language_or_origin_unverified"
    for column in conflicts:
        if row[column] in conflicts[column] and row[column]:
            return "duplicate_label_conflict"
        if row[column]:
            return "duplicate_group_needs_joint_review"
    for column in ("exact_hash", "normalized_hash", "template_hash"):
        value = row[column]
        if value and conn.execute(
            f"SELECT 1 FROM rows WHERE dataset!='trec06c' AND {column}=? LIMIT 1", (value,)
        ).fetchone():
            return "cross_source_overlap"
    return None


def build_review_queue(paths: V21Paths | None = None, *, per_stratum: int = 3) -> dict:
    """Stream authoritative parts and make a bounded, deterministic local queue.

    The queue has excerpts and must remain local. No label is promoted here.
    Existing reviewed decisions are preserved and omitted from the queue.
    """
    paths = paths or resolve_v21_paths()
    _validate_policy(paths)
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    local = paths.curated / "trec06c"
    decisions = _read_decisions(local / "decisions.csv")
    if not (local / "decisions.csv").is_file():
        _write_csv(local / "decisions.csv", DECISION_FIELDS, [])
    conn = _audit_connection(paths)
    csv.field_size_limit(100_000_000)
    samples: dict[tuple[str, ...], list[tuple[str, dict[str, str]]]] = defaultdict(list)
    counts = Counter()
    flag_counts = Counter()
    try:
        conflicts = _conflict_groups(conn)
        for part in _parts(paths):
            _validate_part_checkpoint(conn, part)
            with part.open("r", encoding="utf-8-sig", newline="") as handle:
                for raw in csv.DictReader(handle):
                    audit = conn.execute(
                        "SELECT * FROM rows WHERE audit_id=?", (f"trec06c|{raw['id']}",)
                    ).fetchone()
                    if audit is None or audit["raw_label"] != raw["raw_label"] or audit["original_id"] != raw["id"]:
                        raise RuntimeError("TREC source/audit identity mismatch")
                    counts[("raw", raw["raw_label"])] += 1
                    flags = quality_flags(raw)
                    for flag in flags:
                        flag_counts[(raw["raw_label"], flag)] += 1
                    if (raw["status"] == "ok" and not _bool(raw["empty"])
                            and not _bool(raw["missing_content"])
                            and not _bool(raw["decode_error"])
                            and _bool(raw["contains_chinese"])):
                        counts[("strict_chinese", raw["raw_label"])] += 1
                    for column in conflicts:
                        if audit[column] and audit[column] in conflicts[column]:
                            counts[("conflict_rows", column)] += 1
                    bucket = _length_bucket(raw["subject"], raw["body"])
                    duplicate = _duplicate_stratum(audit)
                    counts[("length", raw["raw_label"], bucket)] += 1
                    counts[("duplicate", raw["raw_label"], duplicate)] += 1
                    counts[("quality", raw["raw_label"], _quality_stratum(flags))] += 1
                    if raw["id"] in decisions:
                        continue
                    key = (raw["raw_label"], bucket, _quality_stratum(flags), duplicate)
                    priority = hashlib.sha256(f"{SEED}:{raw['id']}".encode()).hexdigest()
                    entry = {
                        "id": raw["id"], "source": raw["source"], "raw_label": raw["raw_label"],
                        "original_path": raw["original_path"],
                        "part_path": str(part.relative_to(paths.project_root)),
                        "subject": raw["subject"], "body_excerpt": raw["body"][:300],
                        "length_bucket": bucket, "quality_stratum": key[2],
                        "duplicate_stratum": duplicate, "quality_flags": ";".join(flags),
                        "audit_reason": audit["audit_reason"],
                        "exact_group_id": audit["exact_group_id"],
                        "normalized_group_id": audit["normalized_group_id"],
                        "template_group_id": audit["template_group_id"],
                        "human_decision": "", "human_note": "",
                    }
                    choices = samples[key]
                    choices.append((priority, entry))
                    choices.sort(key=lambda item: item[0])
                    if len(choices) > per_stratum:
                        choices.pop()
        queue = [entry for choices in samples.values() for _, entry in choices]
        queue.sort(key=lambda item: (item["raw_label"], item["quality_stratum"], item["length_bucket"], item["duplicate_stratum"], item["id"]))
        # A normalized duplicate/template family should not crowd out other examples.
        seen_groups: set[str] = set()
        diverse = []
        for item in queue:
            group = item["normalized_group_id"] or item["template_group_id"] or item["id"]
            if group in seen_groups:
                continue
            seen_groups.add(group)
            diverse.append(item)
        _write_csv(local / "review_queue.csv", QUEUE_FIELDS, diverse)
        summary = {
            "source": "TREC06C", "seed": SEED, "per_stratum": per_stratum,
            "raw_labels": {label: counts[("raw", label)] for label in ("ham", "spam")},
            "strict_chinese_quality_candidates": {
                label: counts[("strict_chinese", label)] for label in ("ham", "spam")
            },
            "length_distribution": {"|".join(map(str, key[1:])): value for key, value in counts.items() if key[0] == "length"},
            "quality_distribution": {"|".join(map(str, key[1:])): value for key, value in counts.items() if key[0] == "quality"},
            "duplicate_distribution": {"|".join(map(str, key[1:])): value for key, value in counts.items() if key[0] == "duplicate"},
            "quality_flags": {"|".join(key): value for key, value in flag_counts.items()},
            "conflicting_groups": {column: len(values) for column, values in conflicts.items()},
            "conflict_rows": {column: counts[("conflict_rows", column)] for column in conflicts},
            "queue_rows": len(diverse), "existing_decisions": len(decisions),
            "eligible": sum(row["label_status"] == "eligible" for row in decisions.values()),
            "decision_manifest_sha256": _sha256(local / "decisions.csv") if (local / "decisions.csv").is_file() else None,
        }
        paths.reports.mkdir(parents=True, exist_ok=True)
        (paths.reports / "trec06c_curation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        conn.close()


def load_full_message(paths: V21Paths, queue_row: dict[str, str]) -> dict[str, str]:
    part = (paths.project_root / queue_row["part_path"]).resolve()
    allowed = (paths.processed / PART_DIR).resolve()
    if not part.is_relative_to(allowed) or not part.name.startswith("part-"):
        raise ValueError("Review queue contains an unsafe part path")
    conn = _audit_connection(paths)
    try:
        _validate_part_checkpoint(conn, part)
        with part.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["id"] == queue_row["id"]:
                    if row["raw_label"] != queue_row["raw_label"] or row["original_path"] != queue_row["original_path"]:
                        raise ValueError("Review queue provenance mismatch")
                    return row
    finally:
        conn.close()
    raise LookupError("Queued TREC message not found")


def record_human_decision(
    paths: V21Paths, row: dict[str, str], *, decision: str, reviewer: str,
    method: str, evidence: str, reviewed_at_utc: str | None = None,
    review_cohort: str = "individual", review_stratum: str = "",
) -> dict[str, str]:
    """Record one explicit human decision; never infer approval from raw labels."""
    if decision not in DECISIONS or method not in MANUAL_METHODS or not all(value.strip() for value in (reviewer, method, evidence)):
        raise ValueError("Decision, reviewer, method and evidence are required")
    _validate_policy(paths)
    timestamp = reviewed_at_utc or datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Review timestamp must be UTC")
    local = paths.curated / "trec06c"
    manifest = local / "decisions.csv"
    previous = _read_decisions(manifest)
    if row["id"] in previous:
        raise ValueError("This TREC email already has a decision; resume skips it")
    conn = _audit_connection(paths)
    try:
        audit = conn.execute("SELECT * FROM rows WHERE audit_id=?", (f"trec06c|{row['id']}",)).fetchone()
        if audit is None or audit["raw_label"] != row["raw_label"] or row["source"] != "TREC06C":
            raise ValueError("Decision provenance does not match TREC audit")
        if row["original_path"] != audit["source_path"] or dedup_hashes(row["subject"], row["body"])[0] != audit["exact_hash"]:
            raise ValueError("Decision text/path differs from the frozen dedup audit")
        flags = quality_flags(row)
        conflicts = _conflict_groups(conn)
        blocker = _eligible_gate(conn, audit, flags, conflicts)
        status = ("eligible" if blocker is None else "review") if decision in LABELS else decision
        record = {
            "id": row["id"], "source": "TREC06C", "raw_label": row["raw_label"],
            "original_path": row["original_path"], "decision": decision,
            "label_status": status,
            "label_provenance": "trec06c_individual_human_review" if status == "eligible" else "",
            "curation_rule_id": RULE_ID if status == "eligible" else "",
            "reviewer": reviewer.strip(), "review_method": method.strip(),
            "reviewed_at_utc": timestamp, "evidence": evidence.strip(),
            "content_hash": row["content_hash"], "normalized_hash": audit["normalized_hash"],
            "exact_group_id": audit["exact_group_id"],
            "normalized_group_id": audit["normalized_group_id"],
            "template_group_id": audit["template_group_id"],
            "final_group_id": (
                audit["normalized_group_id"] or audit["exact_group_id"]
                or audit["template_group_id"] or f"trec06c:singleton:{audit['normalized_hash']}"
            ),
            "eligibility_blocker": blocker if decision in LABELS and blocker else "",
            "review_cohort": review_cohort.strip(), "review_stratum": review_stratum.strip(),
        }
        _write_csv(manifest, DECISION_FIELDS, [*previous.values(), record])
        return record
    finally:
        conn.close()
