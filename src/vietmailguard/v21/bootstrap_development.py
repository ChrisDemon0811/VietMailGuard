"""Build a V2.1 development-only bootstrap from frozen V2 train/validation.

The V2 test CSV is read for identifiers/fingerprints only. Its text and labels
are never inspected, copied or exported. No new-source curation state changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .dedup_audit import dedup_hashes
from .paths import V21Paths, resolve_v21_paths


class BootstrapBlocked(RuntimeError):
    """Eligibility, provenance or leakage evidence is insufficient."""


GUARD_FIELDS = (
    "id", "final_group_id", "group_id", "translation_group_id", "parent_id",
    "translation_source_id", "content_hash", "normalized_content_hash", "template_hash",
)
AUDIT_GUARD_FIELDS = ("exact_hash", "normalized_hash", "template_hash", "v2_group_id", "provenance_group_id")
DEVELOPMENT_FIELDS = (
    "id", "source", "language", "subject", "body", "raw_label", "label",
    "label_status", "label_provenance", "data_origin", "parent_id",
    "translation_source_id", "final_group_id", "content_hash",
    "normalized_content_hash", "template_hash", "group_id", "translation_group_id",
    "augmentation_type", "split_constraint", "training_eligible",
    "split", "curation_rule_id", "bootstrap_source_split", "bootstrap_role",
)
CURATION_RULE = "v21_reuse_frozen_v2_development_split"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy(paths: V21Paths) -> dict:
    policy = json.loads((paths.config / "bootstrap_development_policy.json").read_text(encoding="utf-8"))
    if (policy.get("schema_version") != 1 or policy.get("mode") != "development_bootstrap_only"
            or policy.get("final_test_available") is not False):
        raise BootstrapBlocked("Invalid or final-test-enabled bootstrap policy")
    if set(policy.get("source_splits", {})) != {"train", "validation", "test_guard"}:
        raise BootstrapBlocked("Missing explicit frozen test guard")
    if any(not value or value not in {"normal", "spam", "phishing"}
           for value in policy.get("allowed_labels_by_provenance", {}).values()):
        raise BootstrapBlocked("Invalid provenance-to-label policy")
    return policy


def _input_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise BootstrapBlocked(f"Missing or out-of-project bootstrap input: {relative}")
    return path


def _split_metadata(path: Path, split: str, policy: dict) -> tuple[dict[str, dict], list[str]]:
    """Keep metadata only. For test, never access label/subject/body fields."""
    csv.field_size_limit(100_000_000)
    records: dict[str, dict] = {}
    issues: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(GUARD_FIELDS) | {"split"}
        if split != "test":
            required |= {"label", "label_status", "label_provenance", "training_eligible",
                         "language", "source", "data_origin", "augmentation_type", "split_constraint"}
        if not required.issubset(reader.fieldnames or ()):
            raise BootstrapBlocked(f"Missing V2 metadata fields in {path.name}")
        for row in reader:
            identifier = row["id"]
            if not identifier or identifier in records:
                issues.append(f"missing_or_duplicate_id:{split}")
                continue
            if row["split"] != split:
                issues.append(f"split_field_mismatch:{split}")
            item = {field: (row.get(field) or "").strip() for field in GUARD_FIELDS}
            item["split"] = split
            if split != "test":
                item.update({field: (row.get(field) or "").strip() for field in (
                    "label", "label_status", "label_provenance", "training_eligible",
                    "language", "source", "data_origin", "augmentation_type", "split_constraint",
                )})
                if (item["label_provenance"] not in policy["allowed_labels_by_provenance"]
                        or policy["allowed_labels_by_provenance"].get(item["label_provenance"]) != item["label"]
                        or item["label_status"] not in policy["allowed_label_status"]
                        or item["training_eligible"].casefold() != "true"
                        or item["language"] not in policy["allowed_languages"]
                        or not item["source"] or not item["content_hash"]
                        or not item["final_group_id"]):
                    issues.append(f"unverified_development_row:{identifier}")
                if (item["language"] == "vi" and item["data_origin"] != "translated"):
                    issues.append(f"vietnamese_origin_not_translated:{identifier}")
                if (item["augmentation_type"] == "controlled_translation"
                        and (split != "train" or item["split_constraint"] != "train_only")):
                    issues.append(f"augmentation_not_train_only:{identifier}")
            records[identifier] = item
    return records, issues


def _audit_metadata(path: Path, split_by_id: dict[str, str]) -> tuple[dict[str, dict], list[str]]:
    """Match every V2 split ID to the read-only V2 reference audit index."""
    if not path.is_file():
        raise BootstrapBlocked("V2.1 dedup audit index missing")
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    records: dict[str, dict] = {}
    issues: list[str] = []
    try:
        conn.execute("PRAGMA query_only=ON")
        for row in conn.execute("""SELECT original_id, exact_hash, normalized_hash,
            template_hash, v2_group_id, provenance_group_id FROM rows
            WHERE dataset='v2_reference'"""):
            identifier = row[0]
            if identifier not in split_by_id or identifier in records:
                issues.append("audit_reference_id_mismatch")
                continue
            records[identifier] = dict(zip(AUDIT_GUARD_FIELDS, row[1:]))
    finally:
        conn.close()
    if len(records) != len(split_by_id):
        issues.append("audit_reference_coverage_incomplete")
    return records, issues


def _leakage_checks(splits: dict[str, dict[str, dict]], audited: dict[str, dict]) -> dict:
    """Require disjoint content, templates and related groups across all V2 splits."""
    issues: list[str] = []
    all_rows: dict[str, dict] = {}
    for split, rows in splits.items():
        for identifier, metadata in rows.items():
            if identifier in all_rows:
                issues.append(f"cross_split_id:{identifier}")
            all_rows[identifier] = metadata
            if audited.get(identifier, {}).get("v2_group_id") != metadata["final_group_id"]:
                issues.append(f"audit_group_mismatch:{identifier}")
    for field in GUARD_FIELDS[1:]:
        owners: dict[str, str] = {}
        for identifier, metadata in all_rows.items():
            value = metadata.get(field)
            if not value:
                continue
            prior = owners.setdefault(value, metadata["split"])
            if prior != metadata["split"]:
                issues.append(f"cross_split_{field}")
    for field in AUDIT_GUARD_FIELDS:
        owners: dict[str, str] = {}
        for identifier, metadata in all_rows.items():
            value = audited[identifier].get(field)
            if not value:
                continue
            prior = owners.setdefault(value, metadata["split"])
            if prior != metadata["split"]:
                issues.append(f"cross_split_audit_{field}")
    for identifier, metadata in all_rows.items():
        parent = metadata["parent_id"]
        if parent and (parent not in all_rows or all_rows[parent]["split"] != metadata["split"]):
            issues.append(f"parent_split_or_identity_mismatch:{identifier}")
        if (metadata["split"] != "test" and metadata.get("augmentation_type") == "controlled_translation"
                and not parent):
            issues.append(f"augmentation_parent_missing:{identifier}")
        if (parent and metadata["split"] != "test" and parent in all_rows
                and all_rows[parent].get("label") != metadata.get("label")):
            issues.append(f"parent_label_conflict:{identifier}")
    dev_labels: dict[str, str] = {}
    for split in ("train", "validation"):
        for metadata in splits[split].values():
            group = metadata["final_group_id"]
            prior = dev_labels.setdefault(group, metadata["label"])
            if prior != metadata["label"]:
                issues.append(f"development_group_label_conflict:{group}")
    return {"passed": not issues, "issues": sorted(set(issues)),
            "checked_v2_test_metadata_rows": len(splits["test"]),
            "test_label_or_text_used": False}


def _write_split(source: Path, target: Path, metadata: dict[str, dict], audit: dict[str, dict],
                 split: str) -> tuple[int, str]:
    """Stream verified V2 development content to a V2.1-only atomic CSV."""
    partial = target.with_suffix(".csv.partial")
    count = 0
    csv.field_size_limit(100_000_000)
    with source.open("r", encoding="utf-8-sig", newline="") as input_handle, partial.open(
            "w", encoding="utf-8", newline="") as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=DEVELOPMENT_FIELDS)
        writer.writeheader()
        for row in reader:
            identifier = row["id"]
            if identifier not in metadata:
                raise BootstrapBlocked(f"Source split changed during bootstrap: {identifier}")
            hashes = dedup_hashes(row.get("subject"), row.get("body"))
            evidence = audit[identifier]
            if hashes != tuple(evidence[field] for field in AUDIT_GUARD_FIELDS[:3]):
                raise BootstrapBlocked(f"V2 source text differs from read-only audit: {identifier}")
            if any((row.get(field) or "").strip() != metadata[identifier][field]
                   for field in GUARD_FIELDS):
                raise BootstrapBlocked(f"V2 source metadata changed: {identifier}")
            output = {field: row.get(field, "") for field in DEVELOPMENT_FIELDS}
            output.update(curation_rule_id=CURATION_RULE, bootstrap_source_split=split,
                          bootstrap_role="development_train" if split == "train"
                          else "development_validation_reused_from_v2")
            writer.writerow(output)
            count += 1
    if count != len(metadata):
        raise BootstrapBlocked(f"V2 source row count changed during bootstrap: {split}")
    partial.replace(target)
    return count, file_hash(target)


def _completed_split(target: Path, checkpoint: Path, source_hash: str,
                     expected_rows: int) -> tuple[int, str] | None:
    """Resume only a fully checksummed split from the same frozen input."""
    if not checkpoint.exists():
        if target.exists():
            raise BootstrapBlocked(f"Bootstrap CSV lacks checkpoint: {target.name}")
        return None
    if not target.is_file():
        raise BootstrapBlocked(f"Bootstrap checkpoint lacks CSV: {target.name}")
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (saved.get("source_sha256") != source_hash or saved.get("rows") != expected_rows
            or saved.get("output_sha256") != file_hash(target)):
        raise BootstrapBlocked(f"Bootstrap checkpoint mismatch: {target.name}")
    return expected_rows, saved["output_sha256"]


def build_development_bootstrap(paths: V21Paths | None = None) -> dict:
    paths = paths or resolve_v21_paths()
    policy = _policy(paths)
    sources = {name: _input_path(paths.project_root, relative)
               for name, relative in policy["source_splits"].items()}
    audit_path = _input_path(paths.project_root, policy["source_audit_index"])
    curation_path = _input_path(paths.project_root, policy["source_curation_summary"])
    source_hashes = {name: file_hash(path) for name, path in sources.items()}
    audit_hash = file_hash(audit_path)
    curation_hash = file_hash(curation_path)
    curation = json.loads(curation_path.read_text(encoding="utf-8"))
    if curation["status_counts"].get("eligible", 0) or curation["v21_rows"] != sum(
            curation["status_counts"].values()):
        raise BootstrapBlocked("V2.1 source curation status changed or does not reconcile")
    splits: dict[str, dict[str, dict]] = {}
    issues: list[str] = []
    for name, path in sources.items():
        split = "test" if name == "test_guard" else name
        splits[split], found = _split_metadata(path, split, policy)
        issues.extend(found)
    split_by_id = {identifier: split for split, rows in splits.items() for identifier in rows}
    if len(split_by_id) != sum(map(len, splits.values())):
        issues.append("cross_split_id")
    audited, found = _audit_metadata(audit_path, split_by_id)
    issues.extend(found)
    if issues:
        raise BootstrapBlocked("Bootstrap preflight failed: " + ", ".join(sorted(set(issues))[:10]))
    leakage = _leakage_checks(splits, audited)
    if not leakage["passed"]:
        raise BootstrapBlocked("Bootstrap leakage check failed: " + ", ".join(leakage["issues"][:10]))
    counts = Counter()
    sources_count = Counter()
    origins = Counter()
    for split in ("train", "validation"):
        for row in splits[split].values():
            counts[(split, row["language"], row["label"])] += 1
            sources_count[(split, row["source"])] += 1
            origins[(split, row["language"], row["data_origin"] or "not_declared")] += 1
    if any(not all(counts[("train", lang, label)] or counts[("validation", lang, label)]
                   for label in ("normal", "spam", "phishing")) for lang in ("en",)):
        raise BootstrapBlocked("English development pool lacks a required class")
    output_dir = _input_output_dir(paths, policy["bootstrap_output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (old.get("source_sha256") != source_hashes
                or old.get("audit_index_sha256") != audit_hash
                or old.get("curation_summary_sha256") != curation_hash):
            raise BootstrapBlocked("Existing bootstrap inputs differ; refusing to overwrite")
        for split in ("train", "validation"):
            target = output_dir / f"{split}.csv"
            if not target.is_file() or file_hash(target) != old["output_sha256"][split]:
                raise BootstrapBlocked("Existing bootstrap output checksum mismatch")
        return old
    output_hashes = {}
    output_counts = {}
    for split in ("train", "validation"):
        target = output_dir / f"{split}.csv"
        checkpoint = output_dir / f"{split}.checkpoint.json"
        completed = _completed_split(target, checkpoint, source_hashes[split], len(splits[split]))
        if completed is None:
            completed = _write_split(sources[split], target, splits[split], audited, split)
            temporary_checkpoint = checkpoint.with_suffix(".json.partial")
            temporary_checkpoint.write_text(json.dumps({
                "source_sha256": source_hashes[split], "rows": completed[0],
                "output_sha256": completed[1],
            }, indent=2) + "\n", encoding="utf-8")
            temporary_checkpoint.replace(checkpoint)
        output_counts[split], output_hashes[split] = completed
    if any(file_hash(path) != source_hashes[name] for name, path in sources.items()):
        raise BootstrapBlocked("Frozen V2 split changed during bootstrap")
    if file_hash(curation_path) != curation_hash:
        raise BootstrapBlocked("V2.1 curation summary changed during bootstrap")
    if file_hash(audit_path) != audit_hash:
        raise BootstrapBlocked("V2.1 dedup audit index changed during bootstrap")
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_ready", "mode": "development_bootstrap_only",
        "curation_rule_id": CURATION_RULE,
        "source_sha256": source_hashes, "curation_summary_sha256": curation_hash,
        "audit_index_sha256": audit_hash, "output_sha256": output_hashes,
        "output_rows": output_counts, "v2_test_guard_rows": len(splits["test"]),
        "v2_test_rows_exported": 0, "leakage": leakage,
        "counts_by_split_language_label": [
            {"split": s, "language": l, "label": y, "rows": n}
            for (s, l, y), n in sorted(counts.items())],
        "counts_by_split_source": [
            {"split": s, "source": src, "rows": n}
            for (s, src), n in sorted(sources_count.items())],
        "counts_by_split_language_origin": [
            {"split": s, "language": l, "data_origin": o, "rows": n}
            for (s, l, o), n in sorted(origins.items())],
        "v21_new_source_status_counts_unchanged": curation["status_counts"],
        "final_test_available": False,
        "validation_reuse_warning": "V2 validation previously participated in V2 model selection; development only",
    }
    temporary_manifest = manifest_path.with_suffix(".json.partial")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "bootstrap_development_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _input_output_dir(paths: V21Paths, relative: str) -> Path:
    target = (paths.project_root / relative).resolve()
    if not target.is_relative_to(paths.curated.resolve()):
        raise BootstrapBlocked("Bootstrap output must stay inside V2.1 curated area")
    return target
