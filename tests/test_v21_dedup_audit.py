"""Small synthetic source audits; never load a corpus or a frozen test set."""

import csv
import sqlite3
from pathlib import Path

import pytest

from vietmailguard.v21.dedup_audit import DATASETS, dedup_hashes, run_audit
from vietmailguard.v21.paths import resolve_v21_paths


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "source", "raw_label", "candidate_label", "label", "training_eligible",
            "subject", "body", "language", "original_row_id", "parent_id", "final_group_id",
            "v2_enron_csv_row", "data_origin", "empty", "status",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_hash_levels_unicode_and_placeholders():
    assert dedup_hashes("Café", "Một thư")[1] == dedup_hashes("Cafe\u0301", "Một   thư")[1]
    body = "Please review the complete commercial service offer in this message and compare all details before submitting the form at {}"
    first = dedup_hashes("Offer", body.format("https://offer.example/a"))
    second = dedup_hashes("Offer", body.format("https://offer.example/b"))
    assert first[0] != second[0] and first[1] != second[1] and first[2] == second[2]
    assert dedup_hashes("Hi", "Visit https://a.example")[2] == ""
    assert dedup_hashes("", "empty") == ("", "", "")
    assert dedup_hashes("Your message", "Completely unrelated text")[1] != dedup_hashes("Your message", "Different unrelated text")[1]


def test_audit_status_conflict_provenance_and_resume(tmp_path):
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    common = "A shared fully written email with enough context to identify this exact message and its provenance."
    for dataset, relative in DATASETS:
        row = {"id": f"{dataset}:1", "source": dataset, "label": "review", "raw_label": "1",
               "candidate_label": "spam_or_scam_or_phishing_candidate", "subject": dataset,
               "body": common + dataset, "status": "ok", "language": "en"}
        if dataset == "enron_original":
            row["v2_enron_csv_row"] = "100"
        _write(paths.processed / relative / "part-00001.csv", [row])
    merged = paths.processed / "csv_sources/merged_dataset/part-00001.csv"
    _write(merged, [
        {"id": "merged:1", "raw_label": "0", "candidate_label": "normal_candidate", "label": "review", "subject": "Duplicate", "body": common},
        {"id": "merged:2", "raw_label": "1", "candidate_label": "spam_or_scam_or_phishing_candidate", "label": "review", "subject": "Duplicate", "body": common},
        {"id": "merged:3", "raw_label": "", "label": "review", "subject": "", "body": "empty", "empty": "true"},
    ])
    balanced = paths.processed / "csv_sources/balanced_dataset/part-00001.csv"
    _write(balanced, [{"id": "balanced:1", "raw_label": "0", "candidate_label": "normal_candidate",
                       "label": "review", "subject": "Duplicate", "body": common}])
    v2 = tmp_path / "data/processed/Bilingual_Master_Dataset.csv"
    _write(v2, [
        {"id": "Enron:100", "original_row_id": "100", "source": "Enron", "final_group_id": "group100", "label": "normal",
         "training_eligible": "true", "subject": "V2", "body": "A verified normal V2 reference row."},
        {"id": "data_vi:1", "source": "data_vi", "parent_id": "Enron:100", "final_group_id": "group100",
         "label": "normal", "training_eligible": "true", "subject": "Chào", "body": "Thư dịch tiếng Việt.", "language": "vi", "data_origin": "translated"},
    ])
    first = run_audit(paths)
    assert first["input_rows"] == first["output_rows"] == 9
    assert first["physically_removed_rows"] == 0
    assert first["v21_training_eligible_rows"] == 0
    assert first["conflict_rows"] == 3
    assert first["cross_source_pairs"] >= 1
    assert first["cross_corpus_source_pairs"] >= 1
    assert first["provenance_groups"] >= 2
    assert first["label_status"] == {"eligible": 2, "exclude": 1, "review": 6}
    with sqlite3.connect(paths.cache / "dedup_audit.sqlite3") as conn:
        group = conn.execute("SELECT provenance_group_id FROM rows WHERE audit_id='enron_original|enron_original:1'").fetchone()[0]
        assert group == "v2_enron_row_100"
        assert conn.execute("SELECT provenance_group_id FROM rows WHERE audit_id='v2_reference|Enron:100'").fetchone()[0] == group
        assert conn.execute("SELECT training_eligible FROM rows WHERE audit_id='v2_reference|Enron:100'").fetchone()[0] == 0
        assert conn.execute("SELECT label_status FROM rows WHERE audit_id='merged_dataset|merged:1'").fetchone()[0] == "review"
    second = run_audit(paths)
    assert second["input_rows"] == first["input_rows"]
    assert second["resumed_parts"] == len(DATASETS) + 1
    with merged.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="Input changed since checkpoint"):
        run_audit(paths)
