"""Read-only source-label approval dossier for the frozen V2.1 audit.

This module cannot grant eligibility. It summarizes documentary proposals,
quality flags and the already-curated status without opening any held-out test.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .paths import V21Paths, resolve_v21_paths


LABELS = {"normal", "spam", "phishing"}
RULE_STATUSES = {
    "pending_user", "blocked_mixed_taxonomy", "blocked_unlabeled", "blocked_derived_copy",
}
CANDIDATE_FIELDS = (
    "rule_id", "dataset", "raw_label", "rows", "proposed_label", "approval_status",
    "eligible_now", "review_or_weak_now", "excluded_now", "conflict_rows",
    "exact_duplicate_rows", "normalized_duplicate_rows", "template_duplicate_rows",
    "unknown_language_rows", "unknown_origin_rows", "evidence", "conditions",
)


def load_source_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != 1 or policy.get("decision_state") != "pending_user_confirmation":
        raise ValueError("Source approval dossier must remain pending user confirmation")
    rules = policy.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("Source approval rules missing")
    seen: set[tuple[str, str]] = set()
    ids: set[str] = set()
    for rule in rules:
        key = (rule["dataset"], rule["raw_label"])
        if key in seen or rule["rule_id"] in ids:
            raise ValueError("Duplicate source/raw-label rule or rule ID")
        seen.add(key)
        ids.add(rule["rule_id"])
        if rule["status"] not in RULE_STATUSES:
            raise ValueError("An unapproved rule cannot be activated in this audit")
        if rule["proposed_label"] not in LABELS | {None}:
            raise ValueError("Invalid proposed label")
        if rule["status"] != "pending_user" and rule["proposed_label"] is not None:
            raise ValueError("Blocked source cannot propose a training label")
        if not rule.get("evidence") or not rule.get("conditions"):
            raise ValueError("Every source rule needs evidence and conditions")
    if policy.get("v2_train_validation_reuse", {}).get("status") != "pending_separate_protocol":
        raise ValueError("V2 reuse requires a separate approved protocol")
    return policy


def source_group_counts(conn: sqlite3.Connection) -> dict[tuple[str, str], dict[str, int]]:
    """Aggregate flags using the indexed audit; no source email bodies are loaded."""
    query = """SELECT dataset, raw_label, COUNT(*),
        SUM(label_status='exclude'), SUM(audit_reason='duplicate_label_conflict'),
        SUM(exact_group_id!=''), SUM(normalized_group_id!=''), SUM(template_group_id!=''),
        SUM(language NOT IN ('en','vi','zh')),
        SUM(data_origin IN ('','unknown'))
        FROM rows WHERE dataset!='v2_reference' GROUP BY dataset,raw_label"""
    fields = (
        "rows", "excluded_now", "conflict_rows", "exact_duplicate_rows",
        "normalized_duplicate_rows", "template_duplicate_rows", "unknown_language_rows",
        "unknown_origin_rows",
    )
    return {
        (dataset, raw_label): dict(zip(fields, values))
        for dataset, raw_label, *values in conn.execute(query)
    }


def inspect_v2_development_splits(root: Path) -> dict:
    """Read only V2 train/validation metadata; never access the test CSV."""
    csv.field_size_limit(100_000_000)
    by_split = Counter()
    by_language_label = Counter()
    by_provenance = Counter()
    seen_groups: dict[str, str] = {}
    crossing_groups: set[str] = set()
    invalid_rows = 0
    for split in ("train", "validation"):
        path = root / "data" / "splits" / "v2" / f"{split}.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "label", "language", "label_provenance", "label_status",
                        "training_eligible", "final_group_id"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"V2 development split lacks provenance fields: {path}")
            for row in reader:
                by_split[split] += 1
                by_language_label[(row["language"], row["label"])] += 1
                by_provenance[row["label_provenance"]] += 1
                group = row["final_group_id"]
                if (row["label"] not in LABELS or row["training_eligible"].lower() != "true"
                        or row["label_status"] not in {"eligible", "evidence_confirmed"}
                        or not row["label_provenance"] or not group):
                    invalid_rows += 1
                old = seen_groups.setdefault(group, split) if group else split
                if old != split:
                    crossing_groups.add(group)
    return {
        "rows_by_split": dict(sorted(by_split.items())),
        "rows_by_language_label": [
            {"language": lang, "label": label, "rows": count}
            for (lang, label), count in sorted(by_language_label.items())
        ],
        "label_provenance_counts": dict(sorted(by_provenance.items())),
        "invalid_rows": invalid_rows,
        "train_validation_group_crossings": len(crossing_groups),
        "test_csv_accessed": False,
        "reuse_approved": False,
    }


def run_source_approval_audit(paths: V21Paths | None = None) -> dict:
    """Write a pending-approval dossier, never modifying V2 or curation data."""
    paths = paths or resolve_v21_paths()
    policy = load_source_policy(paths.config / "source_label_approval_policy.json")
    curation = json.loads((paths.reports / "label_curation_summary.json").read_text(encoding="utf-8"))
    audit = sqlite3.connect(f"file:{(paths.cache / 'dedup_audit.sqlite3').as_posix()}?mode=ro", uri=True)
    try:
        audit.execute("PRAGMA query_only=ON")
        grouped = source_group_counts(audit)
        audited_total = audit.execute(
            "SELECT COUNT(*) FROM rows WHERE dataset!='v2_reference'").fetchone()[0]
        audit_conflicts = audit.execute(
            "SELECT COUNT(*) FROM rows WHERE dataset!='v2_reference' "
            "AND audit_reason='duplicate_label_conflict'").fetchone()[0]
        audit_promotions = audit.execute(
            "SELECT COUNT(*) FROM rows WHERE dataset!='v2_reference' "
            "AND training_eligible=1").fetchone()[0]
    finally:
        audit.close()
    if sum(item["rows"] for item in grouped.values()) != audited_total:
        raise RuntimeError("Audit source-group row counts do not reconcile")
    if curation["v21_rows"] != audited_total or sum(curation["status_counts"].values()) != audited_total:
        raise RuntimeError("Curation status counts do not reconcile with frozen audit")
    if audit_promotions:
        raise RuntimeError("V2.1 audit contains unexpected eligible source rows")
    if curation["status_counts"].get("eligible", 0):
        raise RuntimeError("Pending source dossier cannot summarize already-eligible curated rows")
    if curation.get("automatic_promotions", 0):
        raise RuntimeError("An earlier automatic promotion must be independently reviewed")
    rule_keys = {(r["dataset"], r["raw_label"]) for r in policy["rules"]}
    uncovered = sorted(set(grouped) - rule_keys)
    if uncovered:
        raise RuntimeError(f"Source policy does not cover audited raw labels: {uncovered}")
    v2 = inspect_v2_development_splits(paths.project_root)
    rows = []
    for rule in policy["rules"]:
        stats = grouped.get((rule["dataset"], rule["raw_label"]), {})
        count = stats.get("rows", 0)
        rows.append({
            "rule_id": rule["rule_id"], "dataset": rule["dataset"],
            "raw_label": rule["raw_label"], "rows": count,
            "proposed_label": rule["proposed_label"] or "",
            "approval_status": rule["status"], "eligible_now": 0,
            "review_or_weak_now": count - stats.get("excluded_now", 0),
            "excluded_now": stats.get("excluded_now", 0),
            "conflict_rows": stats.get("conflict_rows", 0),
            "exact_duplicate_rows": stats.get("exact_duplicate_rows", 0),
            "normalized_duplicate_rows": stats.get("normalized_duplicate_rows", 0),
            "template_duplicate_rows": stats.get("template_duplicate_rows", 0),
            "unknown_language_rows": stats.get("unknown_language_rows", 0),
            "unknown_origin_rows": stats.get("unknown_origin_rows", 0),
            "evidence": "; ".join(rule["evidence"]),
            "conditions": "; ".join(rule["conditions"]),
        })
    paths.reports.mkdir(parents=True, exist_ok=True)
    with (paths.reports / "source_label_approval_candidates.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "v21_source_rows": audited_total,
        "status_counts": curation["status_counts"],
        "eligible_by_language_label": curation["eligible_by_language_label"],
        "source_rules_pending": sum(r["status"] == "pending_user" for r in policy["rules"]),
        "source_rules_blocked": sum(r["status"].startswith("blocked") for r in policy["rules"]),
        "source_rules_active": 0,
        "new_eligible_from_source_approval": 0,
        "audit_conflict_rows": audit_conflicts,
        "v2_development_splits": v2,
        "v2_test_csv_accessed": False,
        "training_or_splitting_performed": False,
    }
    (paths.reports / "source_label_approval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
