"""Small-source tests before streaming the original Enron archive."""

from __future__ import annotations

import csv
import io
import tarfile
from email.message import EmailMessage
from pathlib import Path
from zipfile import ZipFile

import pytest

from vietmailguard.v21.enron_loader import (
    detect_enron_format,
    discover_enron_source,
    load_enron,
    parse_enron_message,
)


def _email(subject: str, body: str, *, html: bool = False) -> bytes:
    message = EmailMessage()
    message["Message-ID"] = "<sample@example.com>"
    message["From"] = "Alice <alice@enron.com>"
    message["To"] = "Bob <bob@enron.com>"
    message["Date"] = "Mon, 1 Jan 2001 10:00:00 -0600"
    message["Subject"] = subject
    message.set_content(body, subtype="html" if html else "plain", charset="utf-8")
    return message.as_bytes()


def _tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _rows(directory: Path) -> list[dict[str, str]]:
    output = []
    for path in sorted(directory.glob("part-*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            output.extend(csv.DictReader(handle))
    return output


def test_detect_actual_format_not_name(tmp_path: Path) -> None:
    directory = tmp_path / "enron.maildir"
    directory.mkdir()
    assert detect_enron_format(directory) == "maildir"
    archive = tmp_path / "enron.unknown"
    _tar(archive, [("maildir/a/inbox/1.", _email("Hello", "Meeting"))])
    assert detect_enron_format(archive) == "tar"
    zipped = tmp_path / "enron.blob"
    with ZipFile(zipped, "w") as handle:
        handle.writestr("maildir/a/inbox/1.", _email("Hello", "Meeting"))
    assert detect_enron_format(zipped) == "zip"
    csv_path = tmp_path / "enron.csv"
    csv_path.write_text("subject,body,label\nHello,World,0\n", encoding="utf-8")
    assert detect_enron_format(csv_path) == "csv"


def test_discover_standard_maildir_name(tmp_path: Path) -> None:
    source = tmp_path / "maildir"
    source.mkdir()
    assert discover_enron_source(tmp_path) == source


def test_parse_headers_unicode_html_and_forwarded() -> None:
    raw = _email(
        "Fwd: Réunion",
        '<p>See the plan.</p><script>do not run</script><a href="https://example.org/x">Open</a>',
        html=True,
    )
    parsed = parse_enron_message(raw)
    assert parsed["sender"] == "Alice <alice@enron.com>"
    assert parsed["receiver"] == "Bob <bob@enron.com>"
    assert parsed["subject"] == "Fwd: Réunion"
    assert "See the plan" in parsed["body"]
    assert "https://example.org/x" in parsed["body"]
    assert "do not run" not in parsed["body"]
    assert parsed["is_forwarded"] is True
    assert parsed["has_html"] is True


def test_tar_checkpoint_duplicate_and_v2_trace_remain_review(tmp_path: Path) -> None:
    body = "Project planning discussion with the operations team next Monday. " * 3
    first = _email("Project update", body)
    source = tmp_path / "original.tar.gz"
    _tar(source, [
        ("maildir/a/inbox/1.", first),
        ("maildir/a/sent/2.", first),
        ("maildir/a/inbox/3.", _email("HTML", "<p>Quarterly report</p>", html=True)),
        ("maildir/a/inbox/4.", b""),
    ])
    reference = tmp_path / "Enron.csv"
    with reference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "body", "label"])
        writer.writeheader()
        writer.writerow({"subject": "Project update", "body": body, "label": "0"})
    output, report = tmp_path / "out", tmp_path / "report.json"
    partial = load_enron(source, output, report, v2_reference=reference, batch_size=2, max_batches=1)
    assert partial["complete"] is False
    assert partial["processed"] == 2
    complete = load_enron(source, output, report, v2_reference=reference, batch_size=2)
    assert complete["complete"] is True
    assert complete["processed"] == 4
    assert complete["counts"]["duplicate_raw_of"] == 1
    assert complete["counts"]["duplicate_text_of"] == 1
    assert complete["counts"]["empty"] == 1
    rows = _rows(output)
    assert len(rows) == 4
    assert rows[0]["v2_enron_csv_row"] == "1"
    assert rows[0]["v2_overlap_type"] == "strict_full"
    assert rows[1]["duplicate_raw_of"] == rows[0]["id"]
    assert all(row["label"] == "review" and row["training_eligible"] == "False" for row in rows)
    assert all(row["raw_label"] == "" for row in rows)
    assert rows[0]["source_path"] == "maildir/a/inbox/1."
    again = load_enron(source, output, report, v2_reference=reference, batch_size=2)
    assert again["processed"] == 4
    assert len(_rows(output)) == 4


def test_maildir_and_zip_small_sources(tmp_path: Path) -> None:
    message = _email("Hello", "Plain text")
    directory = tmp_path / "maildir"
    (directory / "person" / "inbox").mkdir(parents=True)
    (directory / "person" / "inbox" / "1.").write_bytes(message)
    maildir_result = load_enron(directory, tmp_path / "maildir-out", tmp_path / "maildir.json")
    assert maildir_result["processed"] == 1
    assert maildir_result["input_format"] == "maildir"
    archive = tmp_path / "mail.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("maildir/person/inbox/1.", message)
    zip_result = load_enron(archive, tmp_path / "zip-out", tmp_path / "zip.json")
    assert zip_result["processed"] == 1
    assert zip_result["input_format"] == "zip"


def test_csv_not_silently_treated_as_original(tmp_path: Path) -> None:
    source = tmp_path / "Enron.csv"
    source.write_text("subject,body,label\nHello,World,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="derived representation"):
        load_enron(source, tmp_path / "out", tmp_path / "report.json")


def test_resume_rejects_changed_source_or_committed_part(tmp_path: Path) -> None:
    source = tmp_path / "original.tar.gz"
    messages = [("maildir/a/inbox/1.", _email("Hello", "Body"))]
    _tar(source, messages)
    output, report = tmp_path / "out", tmp_path / "report.json"
    load_enron(source, output, report, batch_size=1, max_batches=1)
    with (output / "part-00001.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="missing/changed"):
        load_enron(source, output, report)
    other_output = tmp_path / "other-out"
    load_enron(source, other_output, tmp_path / "other.json", batch_size=1, max_batches=1)
    _tar(source, messages + [("maildir/a/inbox/2.", _email("Later", "More body"))])
    with pytest.raises(ValueError, match="source_fingerprint"):
        load_enron(source, other_output, tmp_path / "other.json")


def test_attachment_is_not_included_in_body() -> None:
    message = EmailMessage()
    message["Subject"] = "Report"
    message.set_content("Please read the note.")
    message.add_attachment(b"SECRET_ATTACHMENT_BYTES", maintype="application", subtype="octet-stream", filename="secret.bin")
    parsed = parse_enron_message(message.as_bytes())
    assert parsed["has_attachment"] is True
    assert parsed["body"] == "Please read the note."
    assert "SECRET_ATTACHMENT_BYTES" not in parsed["body"]
