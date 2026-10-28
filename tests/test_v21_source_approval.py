"""Source approval remains evidence-only until a separate user decision."""

import csv
import json

import pytest

from vietmailguard.v21.dedup_audit import FIELDS, _connect
from vietmailguard.v21.paths import resolve_v21_paths
from vietmailguard.v21.source_approval import (
    inspect_v2_development_splits,
    load_source_policy,
    run_source_approval_audit,
)


def _fixture(tmp_path):
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    paths.config.mkdir(parents=True)
    paths.cache.mkdir(parents=True)
    paths.reports.mkdir(parents=True)
    policy = {
        "schema_version": 1,
        "decision_state": "pending_user_confirmation",
        "rules": [
            {"rule_id": "h", "dataset": "trec06c", "raw_label": "ham",
             "proposed_label": "normal", "status": "pending_user",
             "evidence": ["source document"], "conditions": ["human verification"]},
            {"rule_id": "s", "dataset": "trec06c", "raw_label": "spam",
             "proposed_label": None, "status": "blocked_mixed_taxonomy",
             "evidence": ["source document"], "conditions": ["class review"]},
        ],
        "v2_train_validation_reuse": {"status": "pending_separate_protocol"},
    }
    policy_path = paths.config / "source_label_approval_policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with _connect(paths.cache / "dedup_audit.sqlite3") as conn:
        for raw_label, reason in (("ham", "unverified_v21_label"),
                                  ("spam", "duplicate_label_conflict")):
            row = dict.fromkeys(FIELDS, "")
            row.update(audit_id=f"trec06c|{raw_label}", dataset="trec06c",
                       source="TREC06C", raw_label=raw_label, language="zh",
                       data_origin="original", label_status="review",
                       training_eligible=0, audit_reason=reason,
                       exact_group_id="exact_same" if raw_label == "spam" else "")
            conn.execute(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})",
                         tuple(row[field] for field in FIELDS))
    summary = {
        "v21_rows": 2, "status_counts": {"weak_label": 1, "review": 1},
        "eligible_by_language_label": {
            lang: {label: 0 for label in ("normal", "spam", "phishing")}
            for lang in ("en", "vi", "zh")},
        "automatic_promotions": 0,
    }
    (paths.reports / "label_curation_summary.json").write_text(
        json.dumps(summary), encoding="utf-8")
    split_dir = paths.project_root / "data/splits/v2"
    split_dir.mkdir(parents=True)
    fields = ("id", "label", "language", "label_provenance", "label_status",
              "training_eligible", "final_group_id")
    for split in ("train", "validation"):
        with (split_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"id": split, "label": "normal", "language": "en",
                             "label_provenance": "verified_legitimate_source",
                             "label_status": "eligible", "training_eligible": "True",
                             "final_group_id": split})
    return paths, policy, policy_path


def test_source_approval_integration_stays_pending_with_conflict(tmp_path):
    paths, _, _ = _fixture(tmp_path)
    result = run_source_approval_audit(paths)
    assert result["v21_source_rows"] == 2
    assert result["new_eligible_from_source_approval"] == 0
    assert result["audit_conflict_rows"] == 1
    assert result["source_rules_pending"] == 1
    assert result["source_rules_blocked"] == 1
    assert result["v2_test_csv_accessed"] is False
    assert result["v2_development_splits"]["rows_by_split"] == {"train": 1, "validation": 1}
    with (paths.reports / "source_label_approval_candidates.csv").open(
            encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["proposed_label"] == "normal"
    assert all(row["eligible_now"] == "0" for row in rows)
    assert rows[1]["conflict_rows"] == "1"


@pytest.mark.parametrize("change", [
    {"decision_state": "approved"},
    {"rule_status": "approved"},
    {"blocked_proposal": "spam"},
])
def test_policy_rejects_implicit_or_blocked_approval(tmp_path, change):
    _, policy, path = _fixture(tmp_path)
    if "decision_state" in change:
        policy["decision_state"] = change["decision_state"]
    if "rule_status" in change:
        policy["rules"][0]["status"] = change["rule_status"]
    if "blocked_proposal" in change:
        policy["rules"][1]["proposed_label"] = change["blocked_proposal"]
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError):
        load_source_policy(path)


def test_v2_development_group_crossing_is_flagged(tmp_path):
    paths, _, _ = _fixture(tmp_path)
    validation = paths.project_root / "data/splits/v2/validation.csv"
    with validation.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0]["final_group_id"] = "train"
    with validation.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = inspect_v2_development_splits(paths.project_root)
    assert result["train_validation_group_crossings"] == 1
    assert result["reuse_approved"] is False


def test_uncovered_raw_label_is_not_silently_ignored(tmp_path):
    paths, policy, path = _fixture(tmp_path)
    policy["rules"] = policy["rules"][:1]
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not cover"):
        run_source_approval_audit(paths)
