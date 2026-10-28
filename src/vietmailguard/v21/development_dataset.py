"""Materialize a development-only V2.1 master from the verified V2 bootstrap.

No resplitting, model fitting or final-test construction occurs here. Email
content is streamed; only bounded identifiers, fingerprints and counts remain
in memory for duplicate and lineage checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .bootstrap_development import (
    CURATION_RULE,
    DEVELOPMENT_FIELDS,
    BootstrapBlocked,
    build_development_bootstrap,
    file_hash,
)
from .dedup_audit import dedup_hashes
from .master_split import SCHEMA
from .paths import V21Paths, resolve_v21_paths


class DevelopmentDatasetBlocked(RuntimeError):
    """The bootstrap or its development-only integrity checks did not pass."""


MASTER_FIELDS = SCHEMA + tuple(field for field in DEVELOPMENT_FIELDS if field not in SCHEMA)
SPLITS = ("train", "validation")
LABELS = ("normal", "spam", "phishing")
FINGERPRINT_FIELDS = (
    "final_group_id", "group_id", "translation_group_id", "content_hash",
    "normalized_content_hash", "template_hash", "audit_exact_hash",
    "audit_normalized_hash", "audit_template_hash",
)


def _safe_output(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise DevelopmentDatasetBlocked("Development output path escapes its V2.1 area")
    return resolved


def _load_verified_bootstrap(paths: V21Paths) -> tuple[dict, dict[str, Path]]:
    directory = paths.curated / "bootstrap_v2_development"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise DevelopmentDatasetBlocked("Prompt 06.7 bootstrap manifest is missing")
    try:
        manifest = build_development_bootstrap(paths)
    except (BootstrapBlocked, KeyError, ValueError) as error:
        raise DevelopmentDatasetBlocked(f"Bootstrap verification failed: {error}") from error
    if (manifest.get("status") != "development_ready"
            or manifest.get("mode") != "development_bootstrap_only"
            or manifest.get("final_test_available") is not False
            or manifest.get("v2_test_rows_exported") != 0
            or manifest.get("leakage", {}).get("passed") is not True
            or set(manifest.get("output_rows", {})) != set(SPLITS)):
        raise DevelopmentDatasetBlocked("Bootstrap is not a verified development-only input")
    inputs = {split: directory / f"{split}.csv" for split in SPLITS}
    for split, path in inputs.items():
        if not path.is_file() or file_hash(path) != manifest["output_sha256"][split]:
            raise DevelopmentDatasetBlocked(f"Bootstrap {split} checksum mismatch")
    return manifest, inputs


def _audit_rows(inputs: dict[str, Path], manifest: dict, provenance_policy: dict) -> dict:
    """Recompute content fingerprints and preserve all inherited assignments."""
    csv.field_size_limit(100_000_000)
    allowed_provenance = provenance_policy["allowed_labels_by_provenance"]
    allowed_status = set(provenance_policy["allowed_label_status"])
    count = Counter()
    by_language_label = Counter()
    by_language_origin = Counter()
    by_source = Counter()
    by_provenance = Counter()
    by_status = Counter()
    group_sizes = {field: Counter() for field in FINGERPRINT_FIELDS}
    fingerprint_owner: dict[str, dict[str, tuple[str, str]]] = {
        field: {} for field in FINGERPRINT_FIELDS
    }
    identities: dict[str, tuple[str, str, str]] = {}
    parent_links = 0
    translated_parent_links = 0

    for split in SPLITS:
        with inputs[split].open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(MASTER_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise DevelopmentDatasetBlocked(
                    f"Bootstrap {split} schema is incomplete: {sorted(missing)}")
            for row in reader:
                identifier = row["id"].strip()
                if not identifier or identifier in identities:
                    raise DevelopmentDatasetBlocked("Missing or duplicate development ID")
                if (row["split"] != split or row["bootstrap_source_split"] != split
                        or row["curation_rule_id"] != CURATION_RULE
                        or row["bootstrap_role"] != (
                            "development_train" if split == "train"
                            else "development_validation_reused_from_v2")):
                    raise DevelopmentDatasetBlocked(f"Inherited split/curation mismatch: {identifier}")
                label = row["label"]
                if (label not in LABELS or row["label_status"] not in allowed_status
                        or row["training_eligible"].casefold() != "true"
                        or allowed_provenance.get(row["label_provenance"]) != label
                        or not row["source"] or not row["raw_label"]
                        or not row["final_group_id"] or not row["group_id"]
                        or not row["content_hash"] or not row["normalized_content_hash"]):
                    raise DevelopmentDatasetBlocked(f"Unverified development label/provenance: {identifier}")
                language, origin = row["language"], row["data_origin"]
                if language not in {"en", "vi"}:
                    raise DevelopmentDatasetBlocked(f"Unexpected bootstrap language: {identifier}")
                if language == "vi" and origin != "translated":
                    raise DevelopmentDatasetBlocked(f"Vietnamese origin is not translated: {identifier}")
                if origin not in {"", "native", "translated", "synthetic"}:
                    raise DevelopmentDatasetBlocked(f"Invalid data origin: {identifier}")
                if row["augmentation_type"] == "controlled_translation" and (
                        split != "train" or row["split_constraint"] != "train_only"):
                    raise DevelopmentDatasetBlocked(f"Augmentation escaped train: {identifier}")
                exact, normalized, template = dedup_hashes(row["subject"], row["body"])
                if not exact or not normalized:
                    raise DevelopmentDatasetBlocked(f"Empty development email: {identifier}")
                values = {field: row[field] for field in FINGERPRINT_FIELDS[:6]}
                values.update(audit_exact_hash=exact, audit_normalized_hash=normalized,
                              audit_template_hash=template)
                for field, value in values.items():
                    if not value:
                        continue
                    previous = fingerprint_owner[field].setdefault(value, (split, label))
                    if previous[0] != split:
                        raise DevelopmentDatasetBlocked(f"Cross-split {field}: {identifier}")
                    if previous[1] != label:
                        raise DevelopmentDatasetBlocked(f"Conflicting-label {field}: {identifier}")
                    group_sizes[field][value] += 1
                identities[identifier] = (split, label, row["final_group_id"])
                if row["parent_id"]:
                    parent_links += 1
                    if origin == "translated":
                        translated_parent_links += 1
                count[split] += 1
                by_language_label[(split, language, label)] += 1
                by_language_origin[(split, language, origin or "not_declared")] += 1
                by_source[(split, row["source"])] += 1
                by_provenance[(split, row["label_provenance"])] += 1
                by_status[(split, row["label_status"])] += 1

    if dict(count) != manifest["output_rows"]:
        raise DevelopmentDatasetBlocked("Bootstrap row counts differ from its manifest")
    if any(by_language_label[(split, language, label)] != entry["rows"]
           for entry in manifest["counts_by_split_language_label"]
           for split, language, label in [(entry["split"], entry["language"], entry["label"])]):
        raise DevelopmentDatasetBlocked("Bootstrap class/language counts differ from its manifest")
    if any(not count[(split)] or not sum(by_language_label[(split, lang, label)]
                                        for lang in ("en", "vi"))
           for split in SPLITS for label in LABELS):
        raise DevelopmentDatasetBlocked("All three classes are required in Train and Validation")

    for split in SPLITS:
        with inputs[split].open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                parent_id = row["parent_id"]
                if not parent_id:
                    if row["data_origin"] == "translated":
                        raise DevelopmentDatasetBlocked(f"Translated parent missing: {row['id']}")
                    continue
                parent = identities.get(parent_id)
                if parent is None or parent != (
                        split, row["label"], row["final_group_id"]):
                    raise DevelopmentDatasetBlocked(f"Parent/translation crossing: {row['id']}")

    duplicate_groups = {
        field: {"total_groups": len(sizes),
                "groups_with_multiple_rows": sum(size > 1 for size in sizes.values()),
                "rows_in_multiple_groups": sum(size for size in sizes.values() if size > 1),
                "largest_group": max(sizes.values(), default=0)}
        for field, sizes in group_sizes.items()
    }
    return {
        "rows_by_split": dict(count),
        "by_split_language_label": [
            {"split": split, "language": lang, "label": label, "rows": rows}
            for (split, lang, label), rows in sorted(by_language_label.items())],
        "by_split_language_origin": [
            {"split": split, "language": lang, "data_origin": origin, "rows": rows}
            for (split, lang, origin), rows in sorted(by_language_origin.items())],
        "by_split_source": [
            {"split": split, "source": source, "rows": rows}
            for (split, source), rows in sorted(by_source.items())],
        "by_split_provenance": [
            {"split": split, "label_provenance": provenance, "rows": rows}
            for (split, provenance), rows in sorted(by_provenance.items())],
        "by_split_label_status": [
            {"split": split, "label_status": status, "rows": rows}
            for (split, status), rows in sorted(by_status.items())],
        "parent_linked_rows": parent_links,
        "translated_parent_linked_rows": translated_parent_links,
        "duplicate_groups": duplicate_groups,
        "leakage": {"passed": True, "cross_split_groups": 0,
                    "conflicting_label_groups": 0, "parent_crossings": 0},
    }


def _write_csv(sources: tuple[Path, ...], target: Path) -> tuple[int, str]:
    """Project the verified schema without editing email text or metadata."""
    partial = target.with_suffix(target.suffix + ".partial")
    count = 0
    csv.field_size_limit(100_000_000)
    with partial.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=MASTER_FIELDS)
        writer.writeheader()
        for source in sources:
            with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
                for row in csv.DictReader(input_handle):
                    writer.writerow({field: row[field] for field in MASTER_FIELDS})
                    count += 1
    partial.replace(target)
    return count, file_hash(target)


def _materialize(sources: tuple[Path, ...], target: Path, expected_rows: int,
                 source_hashes: tuple[str, ...], checkpoint: Path) -> tuple[int, str]:
    input_key = hashlib.sha256("|".join(source_hashes).encode("ascii")).hexdigest()
    if checkpoint.exists():
        if not target.is_file():
            raise DevelopmentDatasetBlocked(f"Checkpoint has no output: {target.name}")
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (saved.get("input_key") != input_key or saved.get("rows") != expected_rows
                or saved.get("output_sha256") != file_hash(target)):
            raise DevelopmentDatasetBlocked(f"Output checkpoint mismatch: {target.name}")
        return expected_rows, saved["output_sha256"]
    if target.exists():
        raise DevelopmentDatasetBlocked(f"Output exists without checkpoint: {target.name}")
    rows, digest = _write_csv(sources, target)
    if rows != expected_rows:
        raise DevelopmentDatasetBlocked(f"Output row count mismatch: {target.name}")
    partial = checkpoint.with_suffix(checkpoint.suffix + ".partial")
    partial.write_text(json.dumps({"input_key": input_key, "rows": rows,
                                   "output_sha256": digest}, indent=2) + "\n", encoding="utf-8")
    partial.replace(checkpoint)
    return rows, digest


def _report(manifest: dict) -> str:
    labels = {(r["split"], r["language"], r["label"]): r["rows"]
              for r in manifest["by_split_language_label"]}
    origins = {(r["split"], r["language"], r["data_origin"]): r["rows"]
               for r in manifest["by_split_language_origin"]}
    duplicates = manifest["duplicate_groups"]
    lines = ["# V2.1 Development/Bootstrap dataset", "",
             "Status: **DEVELOPMENT TRAINING READY**; final evaluation: **NOT READY**.", "",
             "The V2 Train/Validation assignment is inherited exactly. No random split,",
             "augmentation, model training, Test creation or final evaluation occurred.", "",
             "| Split | Normal | Spam | Phishing | Total |",
             "|---|---:|---:|---:|---:|" ]
    for split in SPLITS:
        counts = [sum(labels.get((split, lang, label), 0) for lang in ("en", "vi", "zh"))
                  for label in LABELS]
        lines.append(f"| {split} | {counts[0]:,} | {counts[1]:,} | {counts[2]:,} | {sum(counts):,} |")
    lines += ["", "| Language | Split | Normal | Spam | Phishing | Total |",
              "|---|---|---:|---:|---:|---:|"]
    for language in ("en", "vi", "zh"):
        for split in SPLITS:
            counts = [labels.get((split, language, label), 0) for label in LABELS]
            lines.append(f"| {language} | {split} | {counts[0]:,} | {counts[1]:,} | "
                         f"{counts[2]:,} | {sum(counts):,} |")
    lines += ["", "| Language / recorded origin | Train | Validation | Total |",
              "|---|---:|---:|---:|"]
    for lang, origin in (("en", "not_declared"), ("vi", "translated"), ("zh", "not_declared")):
        train = origins.get(("train", lang, origin), 0)
        val = origins.get(("validation", lang, origin), 0)
        lines.append(f"| {lang} / {origin} | {train:,} | {val:,} | {train + val:,} |")
    lines += ["", "Native rows with verified `data_origin=native`: **0**. English V2",
              "origin is blank and is reported as `not_declared`, not invented as native.",
              "Vietnamese data is translated; Vietnamese Validation has no phishing rows.",
              "Chinese has no eligible development rows.", "",
              "## Duplicate and leakage audit", "",
              "| Fingerprint/group | Total groups | Groups with multiple rows | Rows in those groups |",
              "|---|---:|---:|---:|"]
    for field in ("final_group_id", "content_hash", "normalized_content_hash",
                  "template_hash", "audit_exact_hash", "audit_normalized_hash",
                  "audit_template_hash"):
        d = duplicates[field]
        lines.append(f"| {field} | {d['total_groups']:,} | {d['groups_with_multiple_rows']:,} | "
                     f"{d['rows_in_multiple_groups']:,} |")
    lines += ["", f"Parent-linked rows: **{manifest['parent_linked_rows']:,}**; all retain",
              "the parent's split, class and final group. Cross-split fingerprint/group",
              "collisions, conflicting-label groups and parent crossings: **0** in the",
              "development output. The Prompt 06.7 guard also checked 6,704 frozen V2",
              "Test metadata rows and exported none. Its Test labels/text were not used.", "",
              "## Scientific boundary", "",
              "The 1,102,345 new V2.1 source records remain unapproved and are absent.",
              "V2 Validation previously participated in V2 selection, so this is",
              "development validation only. No independent V2.1 Final Test exists;",
              "`final_evaluation_ready=false`. No native Vietnamese or Chinese",
              "performance claim can follow from this dataset. Prompt 08 may fit only",
              "the development Train and compare on development Validation, not on",
              "the frozen V2 Test or an invented final Test.", "",
              "Machine-readable paths, SHA-256 checksums, schema, split assignment",
              "and readiness flags are in the V2.1 development manifest.", ""]
    return "\n".join(lines)


def build_development_dataset(paths: V21Paths | None = None) -> dict:
    """Verify, project and checkpoint V2.1 development Train/Validation."""
    paths = paths or resolve_v21_paths()
    bootstrap, inputs = _load_verified_bootstrap(paths)
    policy_path = paths.config / "bootstrap_development_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    input_hashes = {split: file_hash(path) for split, path in inputs.items()}
    curation_status = paths.curated / "label_status_manifest.csv"
    if not curation_status.is_file():
        raise DevelopmentDatasetBlocked("V2.1 source curation-status manifest is missing")
    curation_status_hash = file_hash(curation_status)
    statistics = _audit_rows(inputs, bootstrap, policy)

    split_dir = _safe_output(paths.splits / "development_bootstrap", paths.splits)
    master_path = _safe_output(paths.processed / "Development_Master_Dataset.csv", paths.processed)
    manifest_path = split_dir / "manifest.json"
    if (split_dir / "test.csv").exists():
        raise DevelopmentDatasetBlocked("Unexpected Test CSV in development-only output")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (old.get("bootstrap_manifest_sha256") != file_hash(
                paths.curated / "bootstrap_v2_development/manifest.json")
                or old.get("bootstrap_input_sha256") != input_hashes
                or old.get("source_curation_status_sha256") != curation_status_hash
                or old.get("final_evaluation_ready") is not False):
            raise DevelopmentDatasetBlocked("Existing development manifest inputs changed")
        outputs = {"train": split_dir / "train.csv", "validation": split_dir / "validation.csv",
                   "master": master_path}
        if any(not path.is_file() or file_hash(path) != old["output_sha256"][key]
               for key, path in outputs.items()):
            raise DevelopmentDatasetBlocked("Existing development output checksum mismatch")
        if (old.get("development_training_ready") is not True
                or old.get("schema") != list(MASTER_FIELDS)
                or any(old.get(key) != value for key, value in statistics.items())):
            raise DevelopmentDatasetBlocked("Existing development manifest statistics mismatch")
        paths.reports.mkdir(parents=True, exist_ok=True)
        summary_path = paths.reports / "development_dataset_summary.json"
        report_path = paths.reports / "development_dataset_report.md"
        summary_text = json.dumps(old, ensure_ascii=False, indent=2) + "\n"
        report_text = _report(old)
        if not summary_path.exists() or summary_path.read_text(encoding="utf-8") != summary_text:
            summary_path.write_text(summary_text, encoding="utf-8")
        if not report_path.exists() or report_path.read_text(encoding="utf-8") != report_text:
            report_path.write_text(report_text, encoding="utf-8")
        return old

    split_dir.mkdir(parents=True, exist_ok=True)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    output_rows = {}
    for split in SPLITS:
        target = split_dir / f"{split}.csv"
        output_rows[split], output_hashes[split] = _materialize(
            (inputs[split],), target, statistics["rows_by_split"][split],
            (input_hashes[split],), split_dir / f"{split}.checkpoint.json")
    output_rows["master"], output_hashes["master"] = _materialize(
        (inputs["train"], inputs["validation"]), master_path,
        sum(statistics["rows_by_split"].values()),
        (input_hashes["train"], input_hashes["validation"]),
        split_dir / "master.checkpoint.json")
    if ({split: file_hash(path) for split, path in inputs.items()} != input_hashes
            or file_hash(curation_status) != curation_status_hash):
        raise DevelopmentDatasetBlocked("Bootstrap or curation manifest changed during materialization")

    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "development_bootstrap", "status": "development_training_ready",
        "development_training_ready": True, "final_evaluation_ready": False,
        "final_test_path": None, "random_seed_used": None,
        "split_strategy": "inherited_v2_train_validation_no_resplit",
        "schema": list(MASTER_FIELDS),
        "bootstrap_manifest_sha256": file_hash(paths.curated / "bootstrap_v2_development/manifest.json"),
        "bootstrap_input_sha256": input_hashes,
        "source_curation_status_sha256": curation_status_hash,
        "v2_test_guard_rows": bootstrap["v2_test_guard_rows"],
        "v2_test_rows_exported": 0,
        "new_v21_source_status_counts_unchanged": bootstrap["v21_new_source_status_counts_unchanged"],
        "output_paths": {
            "train": "data/v21/splits/development_bootstrap/train.csv",
            "validation": "data/v21/splits/development_bootstrap/validation.csv",
            "master": "data/v21/processed/Development_Master_Dataset.csv",
        },
        "output_rows": output_rows, "output_sha256": output_hashes,
        "validation_reuse_warning": bootstrap["validation_reuse_warning"],
        **statistics,
    }
    report = _report(manifest)
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "development_dataset_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (paths.reports / "development_dataset_report.md").write_text(report, encoding="utf-8")
    return manifest
