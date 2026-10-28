"""Prompt 07 uses inherited development splits and never creates a final test."""

import csv
import json
from contextlib import closing

import pytest

from test_v21_bootstrap_development import _fixture, _row, _write_csv
from vietmailguard.v21.bootstrap_development import build_development_bootstrap, file_hash
from vietmailguard.v21.dedup_audit import FIELDS, _connect, dedup_hashes
from vietmailguard.v21.development_dataset import (
    DevelopmentDatasetBlocked,
    _audit_rows,
    build_development_dataset,
)


def _ready_fixture(tmp_path):
    paths, rows = _fixture(tmp_path)
    for identifier, label, provenance in (
        ("s2", "spam", "high_confidence_curated_spam"),
        ("p2", "phishing", "nazario_phishing"),
    ):
        source = _row(identifier, label, provenance, "validation")
        rows["validation"].append(source)
        with closing(_connect(paths.cache / "dedup_audit.sqlite3")) as conn, conn:
            audit = dict.fromkeys(FIELDS, "")
            exact, normalized, template = dedup_hashes(source["subject"], source["body"])
            audit.update(audit_id=f"v2_reference|{identifier}", dataset="v2_reference",
                         original_id=identifier, source=source["source"],
                         label_status="eligible", training_eligible=1,
                         exact_hash=exact, normalized_hash=normalized,
                         template_hash=template, v2_group_id=identifier,
                         provenance_group_id=f"v2_{identifier}")
            conn.execute(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})",
                         tuple(audit[field] for field in FIELDS))
    _write_csv(paths.project_root / "data/splits/v2/validation.csv", rows["validation"])
    status_manifest = paths.curated / "label_status_manifest.csv"
    status_manifest.parent.mkdir(parents=True, exist_ok=True)
    status_manifest.write_text("id,label_status\nnew_source_1,review\n", encoding="utf-8")
    build_development_bootstrap(paths)
    return paths, rows, status_manifest


def _rows(path):
    csv.field_size_limit(100_000_000)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rewrite(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_materialized_master_preserves_split_and_all_provenance(tmp_path):
    paths, _, status_manifest = _ready_fixture(tmp_path)
    source_dir = paths.curated / "bootstrap_v2_development"
    original_hashes = {name: file_hash(source_dir / f"{name}.csv")
                       for name in ("train", "validation")}
    status_hash = file_hash(status_manifest)
    manifest = build_development_dataset(paths)

    assert manifest["development_training_ready"] is True
    assert manifest["final_evaluation_ready"] is False
    assert manifest["final_test_path"] is None
    assert manifest["random_seed_used"] is None
    assert manifest["output_rows"] == {"train": 4, "validation": 3, "master": 7}
    assert manifest["v2_test_rows_exported"] == 0
    assert manifest["leakage"]["passed"] is True
    assert manifest["new_v21_source_status_counts_unchanged"] == {"review": 2, "weak_label": 1}

    split_dir = paths.splits / "development_bootstrap"
    assert not (split_dir / "test.csv").exists()
    source_rows = {name: _rows(source_dir / f"{name}.csv")
                   for name in ("train", "validation")}
    target_rows = {name: _rows(split_dir / f"{name}.csv")
                   for name in ("train", "validation")}
    for name in ("train", "validation"):
        assert [row["id"] for row in target_rows[name]] == [row["id"] for row in source_rows[name]]
        assert all(row["split"] == name and row["bootstrap_source_split"] == name
                   and row["training_eligible"] == "True" and row["label_provenance"]
                   and row["final_group_id"] and row["content_hash"]
                   for row in target_rows[name])
    master = _rows(paths.processed / "Development_Master_Dataset.csv")
    assert [row["id"] for row in master] == [row["id"] for row in target_rows["train"] + target_rows["validation"]]
    assert "test_only" not in {row["id"] for row in master}
    assert all(row["label"] in {"normal", "spam", "phishing"} for row in master)
    assert {name: file_hash(source_dir / f"{name}.csv") for name in original_hashes} == original_hashes
    assert file_hash(status_manifest) == status_hash
    assert build_development_dataset(paths)["output_sha256"] == manifest["output_sha256"]


def test_development_stage_requires_completed_bootstrap(tmp_path):
    paths, _ = _fixture(tmp_path)
    with pytest.raises(DevelopmentDatasetBlocked, match="bootstrap manifest is missing"):
        build_development_dataset(paths)


def test_development_stage_refuses_tampered_bootstrap(tmp_path):
    paths, _, _ = _ready_fixture(tmp_path)
    target = paths.curated / "bootstrap_v2_development/train.csv"
    with target.open("a", encoding="utf-8") as handle:
        handle.write("tampering\n")
    with pytest.raises(DevelopmentDatasetBlocked, match="Bootstrap verification failed"):
        build_development_dataset(paths)
    assert not (paths.processed / "Development_Master_Dataset.csv").exists()


def test_development_stage_refuses_uncheckpointed_output(tmp_path):
    paths, _, _ = _ready_fixture(tmp_path)
    split_dir = paths.splits / "development_bootstrap"
    split_dir.mkdir(parents=True)
    (split_dir / "train.csv").write_text("unexpected", encoding="utf-8")
    with pytest.raises(DevelopmentDatasetBlocked, match="without checkpoint"):
        build_development_dataset(paths)


def test_independent_fingerprint_guard_rejects_cross_split_group(tmp_path):
    paths, _, _ = _ready_fixture(tmp_path)
    split_dir = paths.curated / "bootstrap_v2_development"
    rows = {name: _rows(split_dir / f"{name}.csv") for name in ("train", "validation")}
    rows["validation"][0]["template_hash"] = rows["train"][0]["content_hash"]
    rows["train"][0]["template_hash"] = rows["train"][0]["content_hash"]
    for name in ("train", "validation"):
        _rewrite(split_dir / f"{name}.csv", rows[name])
    policy = json.loads((paths.config / "bootstrap_development_policy.json").read_text(encoding="utf-8"))
    input_paths = {name: split_dir / f"{name}.csv" for name in ("train", "validation")}
    fake_manifest = {"output_rows": {"train": 4, "validation": 3},
                     "counts_by_split_language_label": []}
    with pytest.raises(DevelopmentDatasetBlocked, match="Cross-split template_hash"):
        _audit_rows(input_paths, fake_manifest, policy)


def test_translated_parent_cannot_change_group(tmp_path):
    paths, _, _ = _ready_fixture(tmp_path)
    split_dir = paths.curated / "bootstrap_v2_development"
    rows = {name: _rows(split_dir / f"{name}.csv") for name in ("train", "validation")}
    rows["train"][3]["final_group_id"] = "wrong_group"
    _rewrite(split_dir / "train.csv", rows["train"])
    policy = json.loads((paths.config / "bootstrap_development_policy.json").read_text(encoding="utf-8"))
    input_paths = {name: split_dir / f"{name}.csv" for name in ("train", "validation")}
    fake_manifest = {"output_rows": {"train": 4, "validation": 3},
                     "counts_by_split_language_label": []}
    with pytest.raises(DevelopmentDatasetBlocked, match="Parent/translation crossing"):
        _audit_rows(input_paths, fake_manifest, policy)
