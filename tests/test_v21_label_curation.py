"""Curation tests use synthetic rows and never open frozen V2 test files."""

import csv
import json

import pytest

from vietmailguard.v21.dedup_audit import FIELDS, _connect, dedup_hashes
from vietmailguard.v21.label_curation import run_curation
from vietmailguard.v21.paths import resolve_v21_paths


def _fixture(tmp_path, *, duplicate=False, conflict=False, v2_overlap=False):
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    policy = {
        "schema_version": 1, "taxonomy": ["normal", "spam", "phishing"],
        "manual_rule_id": "human_content_review_v21", "review_sample_per_source_label": 3,
        "sources": {"trec06c": {"raw_labels": {"ham": "weak_label", "spam": "weak_label"},
                               "automatic_mapping_enabled": False},
                    "v2_reference": {"automatic_mapping_enabled": False, "manual_curation_enabled": False}},
    }
    paths.config.mkdir(parents=True)
    policy_path = paths.config / "label_curation_policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    part = paths.processed / "trec06c_mime_v2/part-00001.csv"
    part.parent.mkdir(parents=True)
    raw_rows = [{"id": "zh1", "source": "trec06c", "language_candidate": "zh",
                 "raw_label": "spam", "candidate_label": "spam", "subject": "特价广告",
                 "body": "本周商品促销，请查看商品目录和价格。"}]
    if duplicate:
        raw_rows.append({**raw_rows[0], "id": "zh2"})
    with part.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_rows[0])
        writer.writeheader()
        writer.writerows(raw_rows)
    with _connect(paths.cache / "dedup_audit.sqlite3") as conn:
        stat = part.stat()
        conn.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)",
                     (str(part.resolve()), stat.st_size, stat.st_mtime_ns, len(raw_rows), 1))
        for raw in raw_rows:
            exact, normalized, template = dedup_hashes(raw["subject"], raw["body"])
            values = dict.fromkeys(FIELDS, "")
            values.update(audit_id=f"trec06c|{raw['id']}", dataset="trec06c", source="trec06c",
                          original_id=raw["id"], language="zh", data_origin="original",
                          raw_label="spam", candidate_label="spam", label="review",
                          label_status="review", training_eligible=0,
                          audit_reason="duplicate_label_conflict" if conflict else "unverified_v21_label",
                          exact_hash=exact, normalized_hash=normalized, template_hash=template)
            conn.execute(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})",
                         tuple(values[field] for field in FIELDS))
        if v2_overlap:
            values = dict.fromkeys(FIELDS, "")
            values.update(audit_id="v2_reference|old", dataset="v2_reference", source="Nazario",
                          original_id="zh1", label_status="eligible", training_eligible=0,
                          exact_hash="other_exact", normalized_hash="other_normalized")
            conn.execute(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})",
                         tuple(values[field] for field in FIELDS))
    return paths, policy_path


def _decision(paths, *, decision="spam", evidence="Human read complete email: unsolicited product promotion; no credential or fraud request"):
    path = paths.curated / "label_decisions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", "decision", "reviewer", "evidence", "reviewed_at_utc"])
        writer.writeheader()
        writer.writerow({"audit_id": "trec06c|zh1", "decision": decision, "reviewer": "human_auditor",
                         "evidence": evidence, "reviewed_at_utc": "2026-10-20T00:00:00Z"})
    return path


def test_no_automatic_promotion_and_review_queue(tmp_path):
    paths, policy = _fixture(tmp_path)
    summary = run_curation(paths, policy_path=policy)
    assert summary["status_counts"] == {"weak_label": 1}
    assert summary["automatic_promotions"] == 0
    assert summary["review_queue_sample_rows"] == 1
    assert not (paths.curated / "eligible/part-00001.csv").exists()
    with (paths.curated / "label_status_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert manifest[0]["label_status"] == "weak_label"
    assert manifest[0]["training_eligible"] == "false"
    assert run_curation(paths, policy_path=policy)["status_counts"] == summary["status_counts"]


def test_manual_evidence_creates_eligible_row_with_provenance(tmp_path):
    paths, policy = _fixture(tmp_path)
    summary = run_curation(paths, policy_path=policy, decisions_path=_decision(paths))
    assert summary["accepted_human_labels"] == 1
    assert summary["eligible_by_language_label"]["zh"]["spam"] == 1
    with (paths.curated / "eligible/part-00001.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["raw_label"] == "spam"
    assert rows[0]["label_provenance"] == "human_review_v21"
    assert rows[0]["curation_rule_id"] == "human_content_review_v21"
    assert rows[0]["final_group_id"] and rows[0]["training_eligible"] == "true"


@pytest.mark.parametrize("flags,reason", [
    ({"duplicate": True}, "linked_exact_hash_needs_group_review"),
    ({"conflict": True}, "duplicate_label_conflict"),
    ({"v2_overlap": True}, "frozen_v2_reference_overlap"),
])
def test_duplicate_conflict_and_v2_overlap_cannot_become_eligible(tmp_path, flags, reason):
    paths, policy = _fixture(tmp_path, **flags)
    summary = run_curation(paths, policy_path=policy, decisions_path=_decision(paths))
    assert summary["accepted_human_labels"] == 0
    assert summary["rejected_decisions"][reason] == 1
    assert not (paths.curated / "eligible/part-00001.csv").exists()


def test_missing_human_evidence_is_error(tmp_path):
    paths, policy = _fixture(tmp_path)
    with pytest.raises(ValueError, match="lacks reviewer/evidence/time"):
        run_curation(paths, policy_path=policy, decisions_path=_decision(paths, evidence=""))


@pytest.mark.parametrize("decision", ["review", "exclude"])
def test_noneligible_human_decision_is_persisted_without_training(tmp_path, decision):
    paths, policy = _fixture(tmp_path)
    summary = run_curation(paths, policy_path=policy,
                           decisions_path=_decision(paths, decision=decision))
    assert summary["status_counts"] == {decision: 1}
    assert summary["accepted_human_labels"] == 0
    assert not (paths.curated / "eligible/part-00001.csv").exists()
    with (paths.curated / "label_status_manifest.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["label_status"] == decision
    assert row["training_eligible"] == "false"


def test_changed_standardized_part_is_rejected(tmp_path):
    paths, policy = _fixture(tmp_path)
    run_curation(paths, policy_path=policy)
    part = paths.processed / "trec06c_mime_v2/part-00001.csv"
    with part.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="changed since dedup audit"):
        run_curation(paths, policy_path=policy)
