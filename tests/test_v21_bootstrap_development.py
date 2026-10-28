"""Development bootstrap cannot promote weak labels or consume frozen V2 test."""

import csv
import json
from contextlib import closing

import pytest

from scripts.v21 import build_multilingual_master as master_cli
from vietmailguard.v21.bootstrap_development import (
    BootstrapBlocked,
    build_development_bootstrap,
    file_hash,
)
from vietmailguard.v21.dedup_audit import FIELDS, _connect, dedup_hashes
from vietmailguard.v21.paths import resolve_v21_paths


SPLIT_FIELDS = (
    "id", "source", "subject", "body", "label", "raw_label", "language",
    "content_hash", "group_id", "label_provenance", "parent_id",
    "translation_source_id", "data_origin", "augmentation_type", "label_status",
    "training_eligible", "split_constraint", "normalized_content_hash",
    "template_hash", "translation_group_id", "final_group_id", "split",
)


def _row(identifier, label, provenance, split, *, language="en", parent="", group=None):
    subject = f"Subject {identifier}"
    body = f"Distinct message content for {identifier}."
    exact, normalized, template = dedup_hashes(subject, body)
    return {
        "id": identifier, "source": "Nazario" if label == "phishing" else "Enron",
        "subject": subject, "body": body, "label": label, "raw_label": "1" if label != "normal" else "0",
        "language": language, "content_hash": exact, "group_id": group or identifier,
        "label_provenance": provenance, "parent_id": parent,
        "translation_source_id": parent, "data_origin": "translated" if language == "vi" else "",
        "augmentation_type": "controlled_translation" if language == "vi" else "",
        "label_status": "eligible", "training_eligible": "True",
        "split_constraint": "train_only" if language == "vi" else "",
        "normalized_content_hash": normalized, "template_hash": template,
        "translation_group_id": group or identifier, "final_group_id": group or identifier,
        "split": split,
    }


def _fixture(tmp_path):
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    paths.config.mkdir(parents=True)
    paths.cache.mkdir(parents=True)
    paths.reports.mkdir(parents=True)
    split_dir = paths.project_root / "data/splits/v2"
    split_dir.mkdir(parents=True)
    provenance = {
        "verified_legitimate_source": "normal", "high_confidence_curated_spam": "spam",
        "nazario_phishing": "phishing", "controlled_translation_from_nazario_phishing": "phishing",
    }
    policy = {
        "schema_version": 1, "mode": "development_bootstrap_only",
        "source_splits": {name: f"data/splits/v2/{name if name != 'test_guard' else 'test'}.csv"
                          for name in ("train", "validation", "test_guard")},
        "source_audit_index": "data/v21/cache/dedup_audit.sqlite3",
        "source_curation_summary": "reports/v21/label_curation_summary.json",
        "allowed_labels_by_provenance": provenance,
        "allowed_label_status": ["eligible"], "allowed_languages": ["en", "vi"],
        "bootstrap_output": "data/v21/curated/bootstrap_v2_development",
        "final_test_available": False,
    }
    (paths.config / "bootstrap_development_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    summary = {"v21_rows": 3, "status_counts": {"review": 2, "weak_label": 1}}
    (paths.reports / "label_curation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = {
        "train": [
            _row("n1", "normal", "verified_legitimate_source", "train"),
            _row("s1", "spam", "high_confidence_curated_spam", "train"),
            _row("p1", "phishing", "nazario_phishing", "train"),
            _row("p1_vi", "phishing", "controlled_translation_from_nazario_phishing", "train",
                 language="vi", parent="p1", group="p1"),
        ],
        "validation": [_row("n2", "normal", "verified_legitimate_source", "validation")],
        "test": [_row("test_only", "normal", "verified_legitimate_source", "test")],
    }
    for split, group in rows.items():
        _write_csv(split_dir / f"{split}.csv", group)
    # sqlite3.Connection.__exit__ commits but does not close the WAL writer.
    # Seal it before bootstrap fingerprints the fixture's database bytes.
    with closing(_connect(paths.cache / "dedup_audit.sqlite3")) as conn, conn:
        for group in rows.values():
            for source in group:
                audit = dict.fromkeys(FIELDS, "")
                exact, normalized, template = dedup_hashes(source["subject"], source["body"])
                audit.update(audit_id=f"v2_reference|{source['id']}", dataset="v2_reference",
                             original_id=source["id"], source=source["source"],
                             label_status="eligible", training_eligible=1,
                             exact_hash=exact, normalized_hash=normalized,
                             template_hash=template, v2_group_id=source["final_group_id"],
                             provenance_group_id=f"v2_{source['final_group_id']}")
                conn.execute(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})",
                             tuple(audit[field] for field in FIELDS))
    return paths, rows


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_bootstrap_preserves_v2_and_never_exports_test(tmp_path):
    paths, rows = _fixture(tmp_path)
    split_dir = paths.project_root / "data/splits/v2"
    # The guard must not consult held-out labels, even to validate taxonomy.
    rows["test"][0]["label"] = "not_for_development"
    rows["test"][0]["label_provenance"] = ""
    _write_csv(split_dir / "test.csv", rows["test"])
    frozen = {split: file_hash(split_dir / f"{split}.csv")
              for split in ("train", "validation", "test")}
    curation = file_hash(paths.reports / "label_curation_summary.json")
    manifest = build_development_bootstrap(paths)
    assert manifest["output_rows"] == {"train": 4, "validation": 1}
    assert manifest["v2_test_rows_exported"] == 0
    assert manifest["final_test_available"] is False
    assert manifest["leakage"]["passed"] is True
    assert {split: file_hash(split_dir / f"{split}.csv") for split in frozen} == frozen
    assert file_hash(paths.reports / "label_curation_summary.json") == curation
    output_dir = paths.curated / "bootstrap_v2_development"
    assert not (output_dir / "test.csv").exists()
    with (output_dir / "train.csv").open(encoding="utf-8", newline="") as handle:
        output = list(csv.DictReader(handle))
    assert {row["label"] for row in output} == {"normal", "spam", "phishing"}
    assert all(row["label_provenance"] and row["curation_rule_id"]
               and row["final_group_id"] and row["training_eligible"] == "True"
               for row in output)
    assert build_development_bootstrap(paths)["output_sha256"] == manifest["output_sha256"]


def test_test_group_collision_blocks_bootstrap(tmp_path):
    paths, rows = _fixture(tmp_path)
    rows["test"][0]["final_group_id"] = "n1"
    _write_csv(paths.project_root / "data/splits/v2/test.csv", rows["test"])
    with pytest.raises(BootstrapBlocked, match="leakage"):
        build_development_bootstrap(paths)
    assert not (paths.curated / "bootstrap_v2_development/train.csv").exists()


def test_test_fingerprint_collision_blocks_bootstrap(tmp_path):
    paths, rows = _fixture(tmp_path)
    rows["test"][0]["normalized_content_hash"] = rows["train"][0]["normalized_content_hash"]
    _write_csv(paths.project_root / "data/splits/v2/test.csv", rows["test"])
    with pytest.raises(BootstrapBlocked, match="cross_split_normalized_content_hash"):
        build_development_bootstrap(paths)


def test_independent_audit_fingerprint_collision_blocks_bootstrap(tmp_path):
    paths, rows = _fixture(tmp_path)
    with closing(_connect(paths.cache / "dedup_audit.sqlite3")) as conn, conn:
        conn.execute("UPDATE rows SET normalized_hash=? WHERE original_id='test_only'",
                     (rows["train"][0]["normalized_content_hash"],))
    with pytest.raises(BootstrapBlocked, match="cross_split_audit_normalized_hash"):
        build_development_bootstrap(paths)


def test_parent_cross_split_blocks_bootstrap(tmp_path):
    paths, rows = _fixture(tmp_path)
    rows["validation"][0]["parent_id"] = "p1"
    _write_csv(paths.project_root / "data/splits/v2/validation.csv", rows["validation"])
    with pytest.raises(BootstrapBlocked, match="parent_split_or_identity_mismatch"):
        build_development_bootstrap(paths)


def test_unverified_provenance_never_promoted(tmp_path):
    paths, rows = _fixture(tmp_path)
    rows["train"][1]["label_provenance"] = "rule_only_guess"
    _write_csv(paths.project_root / "data/splits/v2/train.csv", rows["train"])
    with pytest.raises(BootstrapBlocked, match="unverified_development_row"):
        build_development_bootstrap(paths)


def test_existing_output_without_checkpoint_is_not_overwritten(tmp_path):
    paths, _ = _fixture(tmp_path)
    target = paths.curated / "bootstrap_v2_development/train.csv"
    target.parent.mkdir(parents=True)
    target.write_text("unsafely existing output", encoding="utf-8")
    with pytest.raises(BootstrapBlocked, match="lacks checkpoint"):
        build_development_bootstrap(paths)
    assert target.read_text(encoding="utf-8") == "unsafely existing output"


def test_prompt07_development_entrypoint_never_calls_full_split(monkeypatch, capsys):
    monkeypatch.setattr(master_cli, "build_master_split",
                        lambda: pytest.fail("Full split must not run in bootstrap mode"))
    monkeypatch.setattr(master_cli, "build_development_dataset",
                        lambda: {"status": "development_training_ready", "mode": "development_bootstrap",
                                 "output_rows": {"train": 4, "validation": 1, "master": 5},
                                 "development_training_ready": True,
                                 "final_evaluation_ready": False, "v2_test_rows_exported": 0})
    assert master_cli.main(["--mode", "development-bootstrap"]) == 0
    assert json.loads(capsys.readouterr().out)["final_evaluation_ready"] is False


def test_prompt07_development_entrypoint_rejects_test_export(monkeypatch):
    monkeypatch.setattr(master_cli, "build_development_dataset",
                        lambda: {"final_evaluation_ready": True, "v2_test_rows_exported": 1})
    assert master_cli.main(["--mode", "development-bootstrap"]) == 2
