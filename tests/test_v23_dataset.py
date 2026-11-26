from __future__ import annotations
import csv
import io
import sqlite3
import zlib
from pathlib import Path

import pytest
from vietmailguard.v23 import dataset


POLICY = {"label_status": "weak_label", "candidate_label": "spam_or_phishing",
          "rights_evidence": "https://www.untroubled.org/spam/"}


def test_duplicate_does_not_approve_source_label_and_keeps_lineage(tmp_path):
    with dataset.connect(tmp_path / "sample.sqlite3") as connection:
        first = {"id": "1", "subject": "Offer", "body": "Here is a commercial message."}
        assert dataset.add_email(connection, "source", first, POLICY) == "unique_quality_candidate"
        second = {**first, "id": "2", "subject": "OFFER", "body": "HERE  IS A COMMERCIAL MESSAGE."}
        assert dataset.add_email(connection, "source", second, POLICY) == "duplicate_normalized_full_text"
        assert connection.execute("SELECT COUNT(*) FROM thu").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM ban_ghi").fetchone()[0] == 2
        assert connection.execute("SELECT trang_thai_nhan FROM thu").fetchone()[0] == "weak_label"


def test_shared_template_does_not_delete_distinct_messages(tmp_path):
    with dataset.connect(tmp_path / "sample.sqlite3") as connection:
        text = "This is a sufficiently long project notification with the same template and details at "
        for index, url in enumerate(["https://example.com/one", "https://example.org/two"]):
            dataset.add_email(connection, "source", {"id": str(index), "subject": "Notice", "body": text + url}, POLICY)
        assert connection.execute("SELECT COUNT(*) FROM thu").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(DISTINCT ma_bam_mau) FROM thu").fetchone()[0] == 1


def test_unicode_csv_resume_and_input_change(tmp_path):
    directory = tmp_path / "中文 data"
    directory.mkdir()
    path = directory / "part-00001.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "subject", "body"])
        writer.writeheader()
        writer.writerows([{"id": "1", "subject": "会议", "body": "明天下午三点开会，请带上会议资料。"},
                         {"id": "2", "subject": "Empty", "body": ""}])
    policy = {**POLICY, "parts": directory.name}
    config = {"batch_rows": 1, "min_body_alphanumeric": 8}
    with dataset.connect(tmp_path / "sample.sqlite3") as connection:
        dataset.ingest_csv(connection, tmp_path, "source", policy, config)
        dataset.ingest_csv(connection, tmp_path, "source", policy, config)
        assert dataset.summary(connection)["input_records"] == 2
        value = connection.execute("SELECT van_ban_nen FROM thu").fetchone()[0]
        assert "明天" in zlib.decompress(value).decode("utf-8")
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Input part changed"):
            dataset.ingest_csv(connection, tmp_path, "source", policy, config)


def test_archive_crc_stops_wrong_member_attribution(tmp_path, monkeypatch):
    raw = b"Subject: Synthetic\n\nIndependent synthetic message for this test."
    archive = tmp_path / "example.7z"
    archive.write_bytes(b"synthetic archive handle")
    member = {"Path": "mail/1", "Size": str(len(raw)), "CRC": "00000000"}
    monkeypatch.setattr(dataset, "archive_members", lambda *args: iter([member]))

    class Process:
        stdout = io.BytesIO(raw)
        def poll(self): return 0
        def wait(self): return 0

    monkeypatch.setattr(dataset.subprocess, "Popen", lambda *args, **kwargs: Process())
    with dataset.connect(tmp_path / "sample.sqlite3") as connection:
        with pytest.raises(ValueError, match="CRC mismatch"):
            dataset.ingest_archive(connection, tmp_path, "source", {**POLICY, "archive": archive.name},
                                   {"max_email_bytes": 1024, "batch_rows": 10, "min_body_alphanumeric": 8}, Path("7z"))
        assert connection.execute("SELECT COUNT(*) FROM thu").fetchone()[0] == 0


def test_guard_uses_only_saved_fingerprints_and_propagates_known_groups(tmp_path):
    directory = tmp_path / "data/v21/cache"
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / "dedup_audit.sqlite3") as old:
        # There are deliberately no Test text or label columns to read.
        old.execute("CREATE TABLE rows(dataset,normalized_hash,template_hash,normalized_group_id,template_group_id,provenance_group_id)")
        old.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?)", [
            ("v2_reference", "first", "", "known", "", ""),
            ("enron_original", "second", "", "known", "", "")])
    with dataset.connect(tmp_path / "intake.sqlite3") as connection:
        result = dataset.protected_metadata_guard(connection, tmp_path)
        assert result["legacy_test_file_opened"] is False
        assert set(connection.execute("SELECT ma_bam FROM bam_bao_ve")) == {("first",), ("second",)}


def test_sparse_jaccard_matches_exact_prefix_join_without_uint8_overflow():
    from vietmailguard.v23.train_guard import ReferenceJoin
    base = " ".join(f"word{i}" for i in range(800))
    texts = [base, base + " additional content", "entirely unrelated " * 20, ""]
    queries = [base, "changed " + base, "unrelated query " * 20, ""]
    reference = ReferenceJoin(texts)
    expected = {i for i, text in enumerate(queries) if reference.matches(text)}
    assert reference.matches_many(queries) == expected
    assert expected == {0, 1}


def test_incremental_idf_equals_train_only_fit_and_validation_does_not_refit():
    import numpy as np
    from vietmailguard.v23.modeling import StreamingHashTfidf
    texts = ["project meeting tomorrow", "commercial marketing offer", "verify password account"]
    whole = StreamingHashTfidf(128).fit(texts)
    incremental = StreamingHashTfidf(128)
    for text in texts:
        incremental.partial_fit([text])
    np.testing.assert_array_equal(whole.document_frequency_, incremental.document_frequency_)
    np.testing.assert_allclose(whole.idf_, incremental.idf_)
    before = incremental.idf_.copy()
    matrix = incremental.transform(["validation-only vocabulary unrelated to training"])
    assert matrix.shape == (1, 256) and (matrix.data >= 0).all()
    assert incremental.documents_seen_ == 3
    np.testing.assert_array_equal(before, incremental.idf_)


def test_known_group_guard_readonly_attach_with_unicode_spaces(tmp_path):
    from vietmailguard.v23.train_guard import propagate_known_constraints
    root = tmp_path / "dữ liệu 中文"
    old_path = root / "data/v22/phase3/expansion.sqlite3"
    old_path.parent.mkdir(parents=True)
    with dataset.connect(root / "data/v23/intake.sqlite3") as connection:
        for i in range(2):
            dataset.add_email(connection, "source", {"id": str(i), "subject": str(i),
                "body": "Independent local fixture content " + str(i)}, POLICY)
        hashes = list(connection.execute("SELECT id_thu,ma_bam_chuan FROM thu ORDER BY id_thu"))
        with sqlite3.connect(old_path) as old:
            old.execute("CREATE TABLE van_ban(id_van_ban INTEGER,ma_bam TEXT)")
            old.execute("CREATE TABLE nhom(id_van_ban INTEGER,id_nhom TEXT)")
            old.executemany("INSERT INTO van_ban VALUES (?,?)", hashes)
            old.executemany("INSERT INTO nhom VALUES (?,'shared_source_component')", [(i,) for i, _ in hashes])
        before = dataset.digest(old_path)
        connection.execute("INSERT INTO cach_ly VALUES (?,'validation_link')", (hashes[0][0],))
        connection.commit()
        propagate_known_constraints(connection, root)
        assert connection.execute("SELECT COUNT(DISTINCT id_thu) FROM cach_ly").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM thu").fetchone()[0] == 2
        assert dataset.digest(old_path) == before
