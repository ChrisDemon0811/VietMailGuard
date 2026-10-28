"""Small synthetic fixtures; no private CSV corpus required in source CI."""

import csv
from pathlib import Path

import pytest

import vietmailguard.v21.csv_adapters as adapters
from vietmailguard.v21.csv_adapters import MEAJOR_COLUMNS, run_source


def _write(path: Path, fields: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def _load(path: Path, output: Path, report: Path, size: int = 2) -> tuple[dict, list[dict]]:
    stats = run_source(path, output, report, chunk_size=size)
    result = []
    for part in sorted(output.glob("part-*.csv")):
        with part.open(encoding="utf-8", newline="") as handle:
            result.extend(csv.DictReader(handle))
    return stats, result


def test_merged_adapter_positive_is_not_phishing(tmp_path: Path) -> None:
    path = tmp_path / "Merged_Dataset.csv"
    _write(path, ["body", "label"], [["Discount offer", "1.0"], ["Meeting", "0.0"], ["empty", ""]])
    stats, rows = _load(path, tmp_path / "out", tmp_path / "audit.json")
    assert stats["rows_read"] == 3
    assert stats["raw_labels"] == {"1.0": 1, "0.0": 1, "": 1}
    assert [row["candidate_label"] for row in rows] == [
        "spam_or_scam_or_phishing_candidate", "normal_candidate", "unknown"
    ]
    assert {row["label"] for row in rows} == {"review"}
    assert {row["training_eligible"] for row in rows} == {"false"}
    assert stats["missing_by_column"]["label"] == 1
    assert stats["empty_rows"] == 1


def test_balanced_adapter_is_related_not_independent(tmp_path: Path) -> None:
    path = tmp_path / "Balanced_Dataset.csv"
    _write(path, ["body", "label"], [["Same text", "1.0"]])
    _, rows = _load(path, tmp_path / "out", tmp_path / "audit.json")
    assert rows[0]["dataset_role"] == "review_only_related_to_merged"
    assert rows[0]["candidate_label"] != "phishing"
    assert rows[0]["content_hash"]


def test_meajor_adapter_preserves_all_metadata(tmp_path: Path) -> None:
    path = tmp_path / "meajor_cleaned_preprocessed.csv"
    values = {key: f"value-{key}" for key in MEAJOR_COLUMNS}
    values.update({"subject": "Tài khoản", "body": "Đăng nhập tại https://example.test",
                   "source": "trec5", "language": "vi", "urls": "https://example.test", "label": "1.0"})
    _write(path, list(MEAJOR_COLUMNS), [[values[key] for key in MEAJOR_COLUMNS]])
    stats, rows = _load(path, tmp_path / "out", tmp_path / "audit.json")
    row = rows[0]
    for key in MEAJOR_COLUMNS:
        if key == "label":
            assert row["raw_label"] == values[key]
        else:
            assert row[key] == values[key]
    assert row["label"] == "review"
    assert stats["missing_by_column"]["subject"] == 0


def test_phishing_validation_is_challenge_only(tmp_path: Path) -> None:
    path = tmp_path / "Phishing_validation_emails.csv"
    _write(path, ["Email Text", "Email Type"], [["Prize claim", "Phishing Email"], ["Hello", "Safe Email"]])
    _, rows = _load(path, tmp_path / "challenge", tmp_path / "audit.json")
    assert {row["dataset_role"] for row in rows} == {"challenge_only"}
    assert {row["training_eligible"] for row in rows} == {"false"}
    assert [row["candidate_label"] for row in rows] == ["phishing_or_scam_candidate", "safe_candidate"]
    assert {row["label"] for row in rows} == {"review"}


def test_resume_does_not_duplicate_parts_or_counts(tmp_path: Path) -> None:
    path = tmp_path / "Merged_Dataset.csv"
    _write(path, ["body", "label"], [["One", "0"], ["Two", "1"], ["Three", "1"]])
    first, rows = _load(path, tmp_path / "out", tmp_path / "audit.json")
    second = run_source(path, tmp_path / "out", tmp_path / "audit.json", chunk_size=2)
    assert first == second
    assert len(rows) == 3
    assert len(list((tmp_path / "out").glob("part-*.csv"))) == 2


def test_interrupted_run_resumes_from_committed_chunk(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "Merged_Dataset.csv"
    _write(path, ["body", "label"], [["One", "0"], ["Two", "1"], ["Three", "1"]])
    original = adapters._write_batch
    calls = 0

    def interrupt_after_first(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        original(*args)

    monkeypatch.setattr(adapters, "_write_batch", interrupt_after_first)
    with pytest.raises(RuntimeError, match="interruption"):
        run_source(path, tmp_path / "out", tmp_path / "audit.json", chunk_size=2)
    monkeypatch.setattr(adapters, "_write_batch", original)
    stats, rows = _load(path, tmp_path / "out", tmp_path / "audit.json")
    assert stats["rows_read"] == 3
    assert stats["complete"] is True
    assert [row["source_row"] for row in rows] == ["1", "2", "3"]


def test_schema_failure_does_not_create_outputs(tmp_path: Path) -> None:
    path = tmp_path / "Merged_Dataset.csv"
    _write(path, ["body", "wrong"], [["test", "1"]])
    with pytest.raises(ValueError, match="missing required columns"):
        run_source(path, tmp_path / "out", tmp_path / "audit.json")
    assert not (tmp_path / "out").exists()
