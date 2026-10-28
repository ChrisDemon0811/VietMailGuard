"""Evidence-gated V2.1 label curation over the read-only dedup audit index.

This stage never infers ground truth from raw/candidate labels. It streams
standardized parts only to fill a small review sample or approved human rows.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .dedup_audit import DATASETS, dedup_hashes
from .paths import V21Paths, resolve_v21_paths


LABELS = {"normal", "spam", "phishing"}
DECISION_FIELDS = {"audit_id", "decision", "reviewer", "evidence", "reviewed_at_utc"}
REVIEW_FIELDS = (
    "audit_id", "dataset", "source", "language", "raw_label", "candidate_label",
    "audit_reason", "source_group_rows", "subject", "body_excerpt", "curation_status", "human_decision",
    "human_evidence",
)
ELIGIBLE_FIELDS = (
    "id", "source", "language", "subject", "body", "raw_label", "label",
    "label_status", "label_provenance", "curation_rule_id", "curation_evidence",
    "data_origin", "parent_id", "translation_source_id", "final_group_id",
    "training_eligible",
)


def curation_status(row: sqlite3.Row, source_policy: dict) -> str:
    """Classify label evidence without promoting any candidate to ground truth."""
    if row["label_status"] == "exclude":
        return "exclude"
    if row["audit_reason"] in {"duplicate_label_conflict", "candidate_mapping_inconsistent", "parse_or_source_warning"}:
        return "review"
    raw = row["raw_label"] or ""
    configured = source_policy.get("raw_labels", {}).get(raw, "review")
    return configured if configured in {"review", "weak_label"} else "review"


def _load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != 1 or set(policy.get("taxonomy", [])) != LABELS:
        raise ValueError("Invalid V2.1 label curation policy")
    if any(source.get("automatic_mapping_enabled") for source in policy["sources"].values()):
        raise ValueError("Automatic source mapping needs a separate documented approval and implementation")
    return policy


def _read_decisions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    decisions: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not DECISION_FIELDS.issubset(reader.fieldnames or []):
            raise ValueError(f"Decision ledger lacks required fields: {path}")
        for row in reader:
            audit_id = row["audit_id"].strip()
            if not audit_id:
                continue
            if audit_id in decisions:
                raise ValueError(f"Duplicate human decision for {audit_id}")
            if row["decision"] not in LABELS | {"review", "exclude"}:
                raise ValueError(f"Unsupported human decision for {audit_id}")
            if any(not row[field].strip() for field in ("reviewer", "evidence", "reviewed_at_utc")):
                raise ValueError(f"Human decision lacks reviewer/evidence/time: {audit_id}")
            decisions[audit_id] = row
    return decisions


def _safe_manual_decision(conn: sqlite3.Connection, row: sqlite3.Row) -> str | None:
    """Return a blocking reason; conservative duplicate gates are intentional."""
    if row["dataset"] == "v2_reference" or row["label_status"] == "exclude":
        return "excluded_or_frozen_reference"
    if row["audit_reason"] in {"duplicate_label_conflict", "candidate_mapping_inconsistent", "parse_or_source_warning"}:
        return row["audit_reason"]
    if not row["normalized_hash"]:
        return "empty_content"
    if row["language"] not in {"en", "vi", "zh"}:
        return "language_unverified"
    if row["data_origin"] in {"", "unknown"}:
        return "data_origin_unverified"
    if row["data_origin"] == "translated":
        return "translated_group_review_required" if row["parent_id"] else "translated_parent_unverified"
    for column in ("exact_hash", "normalized_hash", "template_hash", "provenance_group_id"):
        value = row[column]
        if value and conn.execute(f"SELECT count(*) FROM rows WHERE {column}=?", (value,)).fetchone()[0] > 1:
            return f"linked_{column}_needs_group_review"
    if conn.execute("""SELECT 1 FROM rows WHERE dataset='v2_reference' AND
        (original_id=? OR (exact_hash=? AND exact_hash!='') OR
         (normalized_hash=? AND normalized_hash!='') OR
         (template_hash=? AND template_hash!='') OR
         (v2_group_id=? AND v2_group_id!='')) LIMIT 1""",
        (row["original_id"], row["exact_hash"], row["normalized_hash"],
         row["template_hash"], row["v2_group_id"])).fetchone():
        return "frozen_v2_reference_overlap"
    return None


def _parts(paths: V21Paths):
    for dataset, relative in DATASETS:
        for part in sorted((paths.processed / relative).glob("part-*.csv")):
            yield dataset, part


def _collect_source_rows(paths: V21Paths, target_ids: set[str], audit: sqlite3.Connection) -> dict[str, dict[str, str]]:
    """Stream parts with per-part SQLite checkpoints; retain selected IDs only."""
    found: dict[str, dict[str, str]] = {}
    if not target_ids:
        return found
    csv.field_size_limit(100_000_000)
    selection_key = hashlib.sha256("\n".join(sorted(target_ids)).encode("utf-8")).hexdigest()
    cache_path = paths.cache / "label_curation_samples.sqlite3"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_path) as cache:
        cache.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        cache.execute("CREATE TABLE IF NOT EXISTS selected (audit_id TEXT PRIMARY KEY, row_json TEXT NOT NULL)")
        cache.execute("""CREATE TABLE IF NOT EXISTS checkpoints
            (path TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL)""")
        old_key = cache.execute("SELECT value FROM metadata WHERE key='selection_key'").fetchone()
        if old_key is None or old_key[0] != selection_key:
            cache.execute("DELETE FROM selected")
            cache.execute("DELETE FROM checkpoints")
            cache.execute("INSERT OR REPLACE INTO metadata VALUES ('selection_key',?)", (selection_key,))
        for dataset, part in _parts(paths):
            resolved = str(part.resolve())
            stat = part.stat()
            audited = audit.execute("SELECT size_bytes, mtime_ns FROM checkpoints WHERE path=?", (resolved,)).fetchone()
            if audited is None or tuple(audited) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Standardized part changed since dedup audit: {part}")
            checkpoint = cache.execute("SELECT size_bytes, mtime_ns FROM checkpoints WHERE path=?", (resolved,)).fetchone()
            if checkpoint:
                if checkpoint != (stat.st_size, stat.st_mtime_ns):
                    raise RuntimeError(f"Standardized part changed since curation checkpoint: {part}")
                continue
            with part.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    audit_id = f"{dataset}|{row['id']}"
                    if audit_id in target_ids:
                        cache.execute("INSERT OR REPLACE INTO selected VALUES (?,?)",
                                      (audit_id, json.dumps(row, ensure_ascii=False)))
            cache.execute("INSERT INTO checkpoints VALUES (?,?,?)", (resolved, stat.st_size, stat.st_mtime_ns))
            cache.commit()
            if cache.execute("SELECT count(*) FROM selected").fetchone()[0] == len(target_ids):
                break
        found = {audit_id: json.loads(payload) for audit_id, payload in cache.execute(
            "SELECT audit_id,row_json FROM selected") if audit_id in target_ids}
    missing = target_ids - found.keys()
    if missing:
        raise RuntimeError(f"Audited IDs absent from standardized parts: {len(missing)}")
    return found


def run_curation(
    paths: V21Paths | None = None, *, policy_path: Path | None = None,
    decisions_path: Path | None = None,
) -> dict:
    """Produce a deterministic review queue/report and approved eligible parts."""
    paths = paths or resolve_v21_paths()
    policy_path = policy_path or paths.config / "label_curation_policy.json"
    policy = _load_policy(policy_path)
    audit_path = paths.cache / "dedup_audit.sqlite3"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Run the V2.1 dedup audit first: {audit_path}")
    decisions_path = decisions_path or paths.curated / "label_decisions.csv"
    decisions = _read_decisions(decisions_path)
    conn = sqlite3.connect(audit_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    try:
        if conn.execute("SELECT count(*) FROM rows").fetchone()[0] == 0:
            raise RuntimeError("Audit index is empty")
        grouped = conn.execute("""SELECT dataset, source, language, raw_label, candidate_label,
            label_status, audit_reason, count(*) AS n FROM rows WHERE dataset!='v2_reference'
            GROUP BY dataset, source, language, raw_label, candidate_label, label_status, audit_reason
            ORDER BY dataset, source, raw_label, language""").fetchall()
        status_counts: Counter[str] = Counter()
        by_source: Counter[tuple[str, str]] = Counter()
        by_language_label: Counter[tuple[str, str]] = Counter()
        review_groups: list[dict] = []
        for group in grouped:
            status = curation_status(group, policy["sources"].get(group["dataset"], {}))
            status_counts[status] += group["n"]
            by_source[(group["dataset"], status)] += group["n"]
            if status in {"review", "weak_label"}:
                review_groups.append(dict(group) | {"curation_status": status})

        accepted: dict[str, tuple[sqlite3.Row, dict[str, str]]] = {}
        noneligible_decisions: dict[str, tuple[sqlite3.Row, dict[str, str]]] = {}
        rejected: Counter[str] = Counter()
        for audit_id, decision in decisions.items():
            row = conn.execute("SELECT * FROM rows WHERE audit_id=?", (audit_id,)).fetchone()
            if row is None:
                rejected["unknown_audit_id"] += 1
                continue
            if not policy["sources"].get(row["dataset"], {}).get("manual_curation_enabled", True):
                rejected["manual_curation_disabled"] += 1
                continue
            if row["label_status"] == "exclude" and decision["decision"] != "exclude":
                rejected["audit_exclude_cannot_be_restored"] += 1
                continue
            if decision["decision"] in {"review", "exclude"}:
                noneligible_decisions[audit_id] = (row, decision)
                continue
            reason = _safe_manual_decision(conn, row)
            if reason:
                rejected[reason] += 1
                continue
            accepted[audit_id] = (row, decision)
        # Deterministic bounded-memory sample across the whole source/label
        # group, rather than the first few adjacent rows of an input part.
        sample_heaps: dict[tuple[str, ...], list[tuple[int, str]]] = {}
        for row in conn.execute("""SELECT audit_id,dataset,source,language,raw_label,
            candidate_label,label_status,audit_reason FROM rows
            WHERE dataset!='v2_reference' ORDER BY audit_id"""):
            if curation_status(row, policy["sources"].get(row["dataset"], {})) not in {"review", "weak_label"}:
                continue
            key = (row["dataset"], row["source"], row["language"], row["raw_label"],
                   row["candidate_label"], row["label_status"], row["audit_reason"])
            rank = int.from_bytes(hashlib.blake2b(row["audit_id"].encode("utf-8"), digest_size=8).digest(), "big")
            heap = sample_heaps.setdefault(key, [])
            item = (-rank, row["audit_id"])
            if len(heap) < policy["review_sample_per_source_label"]:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        sample_ids = {audit_id for heap in sample_heaps.values() for _, audit_id in heap}
        sample_ids -= set(accepted)
        sample_ids -= {audit_id for audit_id, (_, decision) in noneligible_decisions.items()
                       if decision["decision"] == "exclude"}
        selected = _collect_source_rows(paths, sample_ids | accepted.keys(), conn)
        # Validate immutable source text against the audit fingerprint before export.
        for audit_id, (row, _) in list(accepted.items()):
            raw = selected[audit_id]
            if dedup_hashes(raw.get("subject"), raw.get("body"))[0] != row["exact_hash"]:
                rejected["source_text_changed_since_audit"] += 1
                del accepted[audit_id]
        reports = paths.reports
        reports.mkdir(parents=True, exist_ok=True)
        review_path = reports / "label_curation_review_queue.csv"
        sample_group_sizes = {}
        for group in review_groups:
            key = (group["dataset"], group["source"], group["language"],
                   group["raw_label"], group["candidate_label"],
                   group["label_status"], group["audit_reason"])
            sample_group_sizes[key] = group["n"]
        with review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for audit_id in sorted(sample_ids):
                row = conn.execute("SELECT * FROM rows WHERE audit_id=?", (audit_id,)).fetchone()
                raw = selected[audit_id]
                writer.writerow({
                    "audit_id": audit_id, "dataset": row["dataset"], "source": row["source"],
                    "language": row["language"], "raw_label": row["raw_label"],
                    "candidate_label": row["candidate_label"], "audit_reason": row["audit_reason"],
                    "source_group_rows": sample_group_sizes[(row["dataset"], row["source"],
                        row["language"], row["raw_label"], row["candidate_label"],
                        row["label_status"], row["audit_reason"])],
                    "subject": raw.get("subject", ""), "body_excerpt": (raw.get("body") or "")[:500],
                    "curation_status": (noneligible_decisions[audit_id][1]["decision"]
                        if audit_id in noneligible_decisions else
                        curation_status(row, policy["sources"].get(row["dataset"], {}))),
                    "human_decision": decisions.get(audit_id, {}).get("decision", ""),
                    "human_evidence": decisions.get(audit_id, {}).get("evidence", ""),
                })
        eligible_dir = paths.curated / "eligible"
        if accepted:
            eligible_dir.mkdir(parents=True, exist_ok=True)
            output = eligible_dir / "part-00001.csv"
            if output.exists():
                raise RuntimeError(f"Refusing to overwrite curated eligible part: {output}")
            partial_eligible = output.with_suffix(".csv.partial")
            with partial_eligible.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ELIGIBLE_FIELDS)
                writer.writeheader()
                for audit_id, (row, decision) in sorted(accepted.items()):
                    raw = selected[audit_id]
                    label = decision["decision"]
                    writer.writerow({
                        "id": row["original_id"], "source": row["source"],
                        "language": row["language"], "subject": raw.get("subject", ""),
                        "body": raw.get("body", ""), "raw_label": row["raw_label"],
                        "label": label, "label_status": "eligible",
                        "label_provenance": "human_review_v21",
                        "curation_rule_id": policy["manual_rule_id"],
                        "curation_evidence": decision["evidence"],
                        "data_origin": row["data_origin"], "parent_id": row["parent_id"],
                        "translation_source_id": raw.get("translation_source_id", ""),
                        "final_group_id": "v21_" + row["normalized_hash"][:20],
                        "training_eligible": "true",
                    })
                    old = curation_status(row, policy["sources"].get(row["dataset"], {}))
                    status_counts[old] -= 1
                    status_counts["eligible"] += 1
                    by_source[(row["dataset"], old)] -= 1
                    by_source[(row["dataset"], "eligible")] += 1
                    by_language_label[(row["language"], label)] += 1
            partial_eligible.replace(output)
        for row, decision in noneligible_decisions.values():
            old = curation_status(row, policy["sources"].get(row["dataset"], {}))
            new = decision["decision"]
            if old != new:
                status_counts[old] -= 1
                status_counts[new] += 1
                by_source[(row["dataset"], old)] -= 1
                by_source[(row["dataset"], new)] += 1
        counts_path = reports / "label_curation_counts.csv"
        with counts_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("dataset", "label_status", "rows"))
            for dataset in sorted(name for name in policy["sources"] if name != "v2_reference"):
                writer.writerows((dataset, status, by_source[(dataset, status)])
                                 for status in ("eligible", "review", "weak_label", "exclude"))
        distribution_path = reports / "label_curation_distribution.csv"
        distribution_counts: Counter[tuple[str, ...]] = Counter()
        for group in grouped:
            key = (group["dataset"], group["source"], group["language"], group["raw_label"],
                   group["candidate_label"], group["label_status"], group["audit_reason"],
                   curation_status(group, policy["sources"].get(group["dataset"], {})))
            distribution_counts[key] += group["n"]
        for row, decision in list(accepted.values()) + list(noneligible_decisions.values()):
            prefix = (row["dataset"], row["source"], row["language"], row["raw_label"],
                      row["candidate_label"], row["label_status"], row["audit_reason"])
            old = curation_status(row, policy["sources"].get(row["dataset"], {}))
            new = "eligible" if decision["decision"] in LABELS else decision["decision"]
            if old != new:
                distribution_counts[prefix + (old,)] -= 1
                distribution_counts[prefix + (new,)] += 1
        with distribution_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("dataset", "source", "language", "raw_label", "candidate_label",
                             "audit_status", "audit_reason", "curation_status", "rows"))
            writer.writerows((*key, count) for key, count in sorted(distribution_counts.items()) if count)
        summary = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "v21_rows": sum(status_counts.values()),
            "status_counts": dict(sorted((status, count) for status, count in status_counts.items() if count)),
            "eligible_by_language_label": {lang: {label: by_language_label[(lang, label)] for label in sorted(LABELS)}
                                           for lang in ("en", "vi", "zh")},
            "human_decisions_read": len(decisions), "accepted_human_labels": len(accepted),
            "human_review_decisions": sum(decision["decision"] == "review" for _, decision in noneligible_decisions.values()),
            "human_exclude_decisions": sum(decision["decision"] == "exclude" for _, decision in noneligible_decisions.values()),
            "rejected_decisions": dict(sorted(rejected.items())),
            "review_groups": len(review_groups), "review_queue_sample_rows": len(sample_ids),
            "review_sampling": "deterministic_hash_reservoir_v1",
            "automatic_promotions": 0, "v2_reference_used_for_training": 0,
            "sources_without_automatic_mapping": sorted(policy["sources"]),
            "eligible_part": str((eligible_dir / "part-00001.csv").relative_to(paths.project_root)) if accepted else None,
        }
        manifest_path = paths.curated / "label_status_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = manifest_path.with_suffix(".csv.partial")
        with partial_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("audit_id", "dataset", "source", "language", "raw_label", "candidate_label",
                             "label", "label_status", "label_provenance", "curation_rule_id",
                             "final_group_id", "training_eligible", "audit_reason"))
            for row in conn.execute("SELECT * FROM rows WHERE dataset!='v2_reference' ORDER BY audit_id"):
                decision = accepted.get(row["audit_id"])
                noneligible = noneligible_decisions.get(row["audit_id"])
                status = ("eligible" if decision else noneligible[1]["decision"] if noneligible
                          else curation_status(row, policy["sources"].get(row["dataset"], {})))
                writer.writerow((row["audit_id"], row["dataset"], row["source"], row["language"],
                                 row["raw_label"], row["candidate_label"],
                                 decision[1]["decision"] if decision else row["label"], status,
                                 "human_review_v21" if decision or noneligible else row["label_provenance"],
                                 policy["manual_rule_id"] if decision or noneligible else "",
                                 "v21_" + row["normalized_hash"][:20] if decision else "",
                                 "true" if decision else "false", row["audit_reason"]))
        partial_path.replace(manifest_path)
        (reports / "label_curation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        conn.close()
