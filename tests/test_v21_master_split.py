"""V2.1 split tests use synthetic labels; no frozen V2 test is read."""

import csv
import json
import sqlite3
from collections import defaultdict

import pytest

from vietmailguard.v21.dedup_audit import FIELDS, _connect, dedup_hashes
from vietmailguard.v21.master_split import (
    PreparationBlocked, _ingest_part, _stage, build_master_split, sha256_file,
)
from vietmailguard.v21.paths import resolve_v21_paths


def _fake_input(tmp_path, *, conflict=False, orphan=False):
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    paths.reports.mkdir(parents=True)
    (paths.reports / "dedup_audit_summary.json").write_text(json.dumps({
        "v21_source_rows": 13, "v21_training_eligible_rows": 0,
        "label_status": {"review": 13}, "read_only_v2_reference_rows": 1,
    }), encoding="utf-8")
    policy = {
        "schema_version": 1, "seed": 42,
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "curated_glob": "eligible/part-*.csv",
        "approved_label_provenance": ["human_review_v21"],
        "allowed_labels": ["normal", "spam", "phishing"],
        "allowed_languages": ["en", "vi", "zh"],
        "minimum_independent_groups_per_class": 3,
    }
    paths.config.mkdir(parents=True)
    policy_path = paths.config / "master_split_policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    rows = []
    def add(row_id, label, body, *, language="en", parent="", eligible=True):
        rows.append({
            "id": row_id, "source": "synthetic_curated_source", "language": language,
            "subject": f"Subject {row_id}", "body": body, "raw_label": "1",
            "label": label, "label_status": "eligible" if eligible else "review",
            "label_provenance": "human_review_v21", "data_origin": "translated" if parent else "original",
            "parent_id": parent, "translation_source_id": "", "final_group_id": "",
            "training_eligible": "true" if eligible else "false",
        })
    add("n1", "normal", "Please find the ordinary meeting minutes attached for review tomorrow.")
    add("n1vi", "normal", "Đây là biên bản cuộc họp ngày mai để bạn xem lại.",
        language="vi", parent="missing_parent" if orphan else "n1")
    add("n2", "normal", "The team will meet after lunch to discuss the project timetable.")
    add("n3", "normal", "The requested document is ready and can be collected at the office.")
    offer = "This is a commercial product promotion with a detailed sale offer for our subscribers; read the full terms at {}"
    add("s1", "spam", offer.format("https://offers.example/one"))
    # Same subject is required for a URL-template group across distinct IDs.
    add("s1b", "spam", offer.format("https://offers.example/two"))
    rows[-1]["subject"] = rows[-2]["subject"]
    rows[-1]["source"] = "synthetic_other_source"
    add("s2", "spam", "Our catalogue announces a new product discount and an optional commercial newsletter.")
    add("s3", "spam", "Giảm giá sản phẩm hôm nay và nhận bản tin khuyến mãi.", language="zh")
    add("p1", "phishing", "Sign in at the deceptive account page and submit your password now.")
    add("p2", "phishing", "We impersonate your bank and request private account credentials.")
    add("p3", "phishing", "This fake security notice asks for your login verification code.")
    if conflict:
        rows[-1]["subject"], rows[-1]["body"] = rows[2]["subject"], rows[2]["body"]
    add("locked", "normal", "This content belongs to a frozen V2 reference row and must not enter V2.1.")
    add("review", "spam", "This row has no completed review decision.", eligible=False)
    with _connect(paths.cache / "dedup_audit.sqlite3") as conn:
        for row in rows:
            exact, normalized, template = dedup_hashes(row["subject"], row["body"])
            values = dict.fromkeys(FIELDS, "")
            values.update(audit_id=f"synthetic|{row['id']}", dataset="synthetic", source=row["source"],
                          original_id=row["id"], exact_hash=exact, normalized_hash=normalized,
                          template_hash=template, label_status="review", training_eligible=0,
                          audit_reason="unverified_v21_label")
            conn.execute(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})",
                         tuple(values[field] for field in FIELDS))
        reference = dict.fromkeys(FIELDS, "")
        reference.update(audit_id="v2_reference|v2_locked", dataset="v2_reference", source="Enron",
                         original_id="v2_locked", exact_hash=dedup_hashes(rows[-2]["subject"], rows[-2]["body"])[0],
                         normalized_hash=dedup_hashes(rows[-2]["subject"], rows[-2]["body"])[1],
                         label_status="eligible", training_eligible=0)
        conn.execute(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})",
                     tuple(reference[field] for field in FIELDS))
    rows.append({**rows[0], "body": "A duplicate id without a human decision.",
                 "label_status": "review", "training_eligible": "false"})
    part = paths.curated / "eligible/part-00001.csv"
    part.parent.mkdir(parents=True)
    with part.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return paths, policy_path, part


def test_real_preflight_blocks_when_no_curated_rows(tmp_path):
    paths, policy_path, part = _fake_input(tmp_path)
    part.unlink()
    with pytest.raises(PreparationBlocked, match="No approved V2.1 curated CSV"):
        build_master_split(paths, policy_path=policy_path)
    manifest = json.loads((paths.reports / "multilingual_split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["language_class_counts"]["zh"]["phishing"] == 0
    assert not (paths.processed / "Multilingual_Master_Dataset.csv").exists()


def test_group_split_and_v2_lock(tmp_path):
    paths, policy_path, _ = _fake_input(tmp_path)
    result = build_master_split(paths, policy_path=policy_path)
    assert result["status"] == "complete"
    assert result["eligible_rows"] == 11
    assert result["curated_rejection_reasons"] == {
        "frozen_v2_reference_overlap": 1, "not_eligible": 2,
    }
    assert result["language_class_counts"]["vi"]["normal"] == 1
    assert result["language_class_counts"]["zh"]["phishing"] == 0
    assert sum(result["split_counts"].values()) == 11
    assert result["observed_ratios"]["train"] != pytest.approx(0.70)
    assert set(result["split_counts"]) == {"train", "validation", "test"}
    with (paths.processed / "Multilingual_Master_Dataset.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in rows}
    assert len(rows) == 11 and "locked" not in by_id and "review" not in by_id
    assert by_id["n1"]["final_group_id"] == by_id["n1vi"]["final_group_id"]
    assert by_id["n1"]["split"] == by_id["n1vi"]["split"]
    assert by_id["s1"]["final_group_id"] == by_id["s1b"]["final_group_id"]
    assert by_id["s1"]["source"] != by_id["s1b"]["source"]
    groups = defaultdict(set)
    for row in rows:
        groups[row["final_group_id"]].add(row["split"])
        assert row["label_status"] == "eligible"
    assert all(len(splits) == 1 for splits in groups.values())
    assert all(value == 0 for value in result["leakage_checks"].values())
    for split in ("train", "validation", "test"):
        with (paths.splits / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            assert all(row["split"] == split for row in csv.DictReader(handle))
    with pytest.raises(PreparationBlocked, match="Refusing to overwrite"):
        build_master_split(paths, policy_path=policy_path)


def test_conflicting_confirmed_labels_block_export(tmp_path):
    paths, policy_path, _ = _fake_input(tmp_path, conflict=True)
    with pytest.raises(PreparationBlocked, match="conflicting confirmed labels"):
        build_master_split(paths, policy_path=policy_path)
    manifest = json.loads((paths.reports / "multilingual_split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["output_files"] == {}
    assert not (paths.splits / "test.csv").exists()


def test_checkpoint_resume_and_changed_input_guard(tmp_path):
    paths, policy_path, part = _fake_input(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    with _stage(paths.cache / "stage_test.sqlite3") as stage, sqlite3.connect(paths.cache / "dedup_audit.sqlite3") as audit:
        fingerprint = sha256_file(part)
        _ingest_part(stage, audit, part, fingerprint, policy)
        first = stage.execute("SELECT count(*) FROM items").fetchone()[0]
        _ingest_part(stage, audit, part, fingerprint, policy)
        assert stage.execute("SELECT count(*) FROM items").fetchone()[0] == first == 11
        with part.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with pytest.raises(RuntimeError, match="changed since checkpoint"):
            _ingest_part(stage, audit, part, sha256_file(part), policy)


def test_translated_orphan_blocks_split(tmp_path):
    paths, policy_path, _ = _fake_input(tmp_path, orphan=True)
    with pytest.raises(PreparationBlocked, match="translated eligible rows lack an eligible parent"):
        build_master_split(paths, policy_path=policy_path)
    assert not (paths.splits / "test.csv").exists()
