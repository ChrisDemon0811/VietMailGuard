"""TREC06C loader checks with a tiny archive and optional real-archive smoke test."""

from __future__ import annotations

import csv
import json
import os
from email.message import EmailMessage
from pathlib import Path
from zipfile import ZipFile

import pytest

from vietmailguard.v21.paths import resolve_v21_paths
from vietmailguard.v21.trec06c import load_trec06c, parse_message


def _message(subject: str, body: str, *, charset: str = "gb18030") -> bytes:
    message = EmailMessage()
    message["From"] = "sender@example.cn"
    message["To"] = "receiver@example.cn"
    message["Subject"] = subject
    message.set_content(body, charset=charset)
    return message.as_bytes()


def _archive(path: Path) -> None:
    base = "trec06c-utf8/"
    first = _message("项目会议", "请于明天开会。")
    html = EmailMessage()
    html["Subject"] = "优惠"
    html.set_content('<p>特价商品</p><a href="https://example.cn/offer">详情</a>', subtype="html", charset="gb18030")
    with ZipFile(path, "w") as archive:
        archive.writestr(base + "label/index", "ham ../data/000/000\nspam ../data/000/001\nspam ../data/000/002\nham ../data/000/003\nspam ../data/000/004\n")
        archive.writestr(base + "data/000/000", first)
        archive.writestr(base + "data_cut/000/000", b"DO NOT LOAD THIS COPY")
        archive.writestr(base + "data/000/001", html.as_bytes())
        archive.writestr(base + "data/000/002", first)
        archive.writestr(base + "data/000/003", b"")
        # 000/004 intentionally missing to verify index traceability.


def _rows(output: Path) -> list[dict[str, str]]:
    records = []
    for part in sorted(output.glob("part-*.csv")):
        with part.open(encoding="utf-8", newline="") as handle:
            records.extend(csv.DictReader(handle))
    return records


def test_gb18030_mime_and_html() -> None:
    parsed = parse_message(_message("账户通知", "请阅读中文正文。"))
    assert parsed["subject"] == "账户通知"
    assert parsed["body"] == "请阅读中文正文。"
    assert parsed["contains_chinese"] is True
    assert parsed["decode_error"] is False


def test_transcoded_utf8_body_recovers_stale_base64_header() -> None:
    raw = (
        b"Subject: =?utf-8?b?5rWL6K+V?=\n"
        b"Content-Type: text/plain; charset=gb2312\n"
        b"Content-Transfer-Encoding: base64\n\n"
        + "这是已转成 UTF-8 的正文。".encode("utf-8")
    )
    parsed = parse_message(raw)
    assert parsed["body"] == "这是已转成 UTF-8 的正文。"
    assert parsed["mime_recovered"] is True
    assert parsed["body_source"] == "utf8_raw_recovery"


def test_streaming_original_only_resume_and_candidate_policy(tmp_path: Path) -> None:
    archive = tmp_path / "2025-123.zip"
    output = tmp_path / "processed"
    report = tmp_path / "report.json"
    _archive(archive)

    partial = load_trec06c(archive, output, report, batch_size=2, max_batches=1)
    assert partial["complete"] is False
    assert partial["processed"] == 2
    assert len(_rows(output)) == 2

    final = load_trec06c(archive, output, report, batch_size=2)
    assert final["complete"] is True
    assert final["index_rows"] == 5
    assert final["processed"] == 5
    assert final["counts"]["exact_duplicates"] == 1
    assert final["counts"]["normalized_duplicates"] == 1
    assert final["counts"]["empty"] == 1
    assert final["counts"]["missing_member"] == 1
    assert final["counts"]["has_html"] == 1
    assert final["counts"]["contains_chinese"] == 3
    assert final["archive_members"]["data_cut_members"] == 1
    rows = _rows(output)
    assert len(rows) == 5
    assert all(row["label"] == "review" and row["training_eligible"] == "False" for row in rows)
    assert [row["raw_label"] for row in rows] == ["ham", "spam", "spam", "ham", "spam"]
    assert rows[0]["body"] == "请于明天开会。"
    assert "DO NOT LOAD" not in json.dumps(rows, ensure_ascii=False)
    assert "https://example.cn/offer" in rows[1]["body"]
    assert rows[2]["duplicate_raw_of"] == rows[0]["id"]
    assert rows[3]["status"] == "empty"
    assert rows[4]["status"] == "missing_member"
    assert json.loads(report.read_text(encoding="utf-8"))["complete"] is True

    repeated = load_trec06c(archive, output, report, batch_size=2)
    assert repeated["processed"] == 5
    assert len(_rows(output)) == 5


def test_unsafe_index_path_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("trec06c-utf8/label/index", "spam ../data_cut/000/000\n")
    with pytest.raises(ValueError, match="Unsafe/non-original"):
        load_trec06c(archive, tmp_path / "output", tmp_path / "report.json")


def test_resume_rejects_changed_limit_or_committed_part(tmp_path: Path) -> None:
    archive = tmp_path / "2025-123.zip"
    output = tmp_path / "processed"
    report = tmp_path / "report.json"
    _archive(archive)
    load_trec06c(archive, output, report, batch_size=2, max_batches=1)
    with pytest.raises(ValueError, match="size limit changed"):
        load_trec06c(archive, output, report, batch_size=2, max_member_bytes=100)
    with (output / "part-00001.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="missing/changed"):
        load_trec06c(archive, output, report, batch_size=2)


@pytest.mark.trec06c_real
def test_real_zip_first_batch_when_opted_in(tmp_path: Path) -> None:
    if os.environ.get("VIETMAILGUARD_RUN_TREC06C_INTEGRATION") != "1":
        pytest.skip("Opt in to the real-archive integration test")
    archive = resolve_v21_paths().dataset_dir / "2025-123.zip"
    if not archive.is_file():
        pytest.skip("Configured TREC06C archive is unavailable")
    result = load_trec06c(
        archive, tmp_path / "processed", tmp_path / "report.json",
        batch_size=10, max_batches=1,
    )
    assert result["processed"] == 10
    assert result["complete"] is False
    assert result["index_rows"] > 10
    assert len(_rows(tmp_path / "processed")) == 10
