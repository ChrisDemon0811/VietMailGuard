from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pytest

from scripts.seed_demo_mailbox import DEMO_EMAILS, seed_demo_mailbox
from vietmailguard.mail_service import MailService


def _analysis(prediction: str = "normal") -> dict[str, Any]:
    action = {
        "normal": "ALLOW",
        "spam": "MOVE_TO_SPAM",
        "phishing": "QUARANTINE",
    }[prediction]
    return {
        "model_version": "2.0.0-test-double",
        "detected_language": "vi" if prediction == "spam" else "en",
        "prediction": prediction,
        "confidence": 0.9,
        "risk_score": 90 if prediction == "phishing" else 12,
        "risk_level": "CRITICAL" if prediction == "phishing" else "LOW",
        "base_recommended_action": action,
        "recommended_action": action,
        "security_findings": [],
    }


def _analyzer(prediction: str = "normal"):
    def analyze(**_: object) -> dict[str, Any]:
        return _analysis(prediction)

    return analyze


def _plain_eml(*, subject: str, body: str, sender: str = "alice@example.com") -> bytes:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = "bob@example.org"
    message["Date"] = "Mon, 15 Sep 2026 08:00:00 +0000"
    message["Subject"] = subject
    message.set_content(body)
    return message.as_bytes()


def test_seed_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "seed.db"
    first = seed_demo_mailbox(database_path, analyzer=_analyzer())
    second = seed_demo_mailbox(database_path, analyzer=_analyzer())

    assert len(DEMO_EMAILS) == 24
    assert first["created"] == 24
    assert first["skipped_duplicates"] == 0
    assert first["total"] == 24
    assert second["created"] == 0
    assert second["skipped_duplicates"] == 24
    assert second["total"] == 24


def test_seed_contains_english_vietnamese_mixed_and_short_content() -> None:
    texts = [f"{item['subject']} {item['body']}" for item in DEMO_EMAILS]
    assert any("Weekly project" in text for text in texts)
    assert any("Tài khoản" in text for text in texts)
    assert any("Your account cần" in text for text in texts)
    assert any(len(item["body"].split()) < 8 for item in DEMO_EMAILS)


def test_duplicate_eml_is_analyzed_and_stored_only_once(tmp_path: Path) -> None:
    calls = 0

    def counting_analyzer(**_: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _analysis("normal")

    raw = _plain_eml(subject="Project update", body="The project remains on schedule.")
    with MailService.from_database_path(
        tmp_path / "import.db", analyzer=counting_analyzer
    ) as service:
        first = service.import_eml_bytes(raw)
        second = service.import_eml_bytes(raw)
        stored = service.repository.list_emails()

    assert first["created"] is True
    assert second["duplicate"] is True
    assert first["email"]["id"] == second["email"]["id"]
    assert first["analysis_id"] == second["analysis_id"]
    assert calls == 1
    assert len(stored) == 1


def test_english_and_vietnamese_eml_import(tmp_path: Path) -> None:
    with MailService.from_database_path(
        tmp_path / "languages.db", analyzer=_analyzer("normal")
    ) as service:
        english = service.import_eml_bytes(
            _plain_eml(subject="Meeting", body="Meeting moved to 3 PM.")
        )
        vietnamese = service.import_eml_bytes(
            _plain_eml(
                subject="Lịch họp",
                body="Cuộc họp được chuyển sang 15 giờ chiều nay.",
                sender="lan@example.com",
            )
        )

    assert english["email"]["subject"] == "Meeting"
    assert vietnamese["email"]["subject"] == "Lịch họp"
    assert "chuyển sang" in vietnamese["email"]["body"]


def test_phishing_prediction_controls_eml_route(tmp_path: Path) -> None:
    with MailService.from_database_path(
        tmp_path / "phishing.db", analyzer=_analyzer("phishing")
    ) as service:
        imported = service.import_eml_bytes(
            _plain_eml(
                subject="Account alert",
                body="Verify your password at http://192.0.2.1/login.",
            )
        )

    assert imported["email"]["folder"] == "cach_ly"
    assert imported["analysis"]["prediction"] == "phishing"


@pytest.mark.parametrize("raw", [b"", b"From: sender@example.com\r\n\r\n"])
def test_malformed_or_empty_eml_is_rejected(tmp_path: Path, raw: bytes) -> None:
    with MailService.from_database_path(
        tmp_path / "malformed.db", analyzer=_analyzer()
    ) as service:
        with pytest.raises(ValueError):
            service.import_eml_bytes(raw)
        assert service.repository.list_emails() == []


def test_html_eml_keeps_plain_body_and_url_without_html_execution(tmp_path: Path) -> None:
    raw = (
        b"From: notice@example.com\r\n"
        b"To: user@example.org\r\n"
        b"Subject: HTML notice\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n\r\n"
        b"<html><body><p>Read this notice.</p>"
        b"<a href='https://example.com/notice'>Open notice</a>"
        b"<img src='https://tracker.example/pixel.png'>"
        b"<script>window.alert('unsafe')</script></body></html>"
    )
    with MailService.from_database_path(
        tmp_path / "html.db", analyzer=_analyzer()
    ) as service:
        imported = service.import_eml_bytes(raw)

    assert "https://example.com/notice" in imported["email"]["body"]
    assert imported["email"]["body_html"] == ""
    assert "<script>" not in imported["email"]["body"]
    assert "window.alert" not in imported["email"]["body"]
    assert "tracker.example" not in imported["email"]["body"]


def test_multipart_eml_prefers_plain_text_and_ignores_attachment(
    tmp_path: Path,
) -> None:
    message = EmailMessage()
    message["From"] = "sender@example.com"
    message["To"] = "recipient@example.org"
    message["Subject"] = "Multipart notice"
    message.set_content("Plain text body with https://example.com/safe")
    message.add_alternative(
        "<p>HTML alternative</p><img src='https://tracker.example/pixel.png'>",
        subtype="html",
    )
    message.add_attachment(
        b"attachment payload must not enter the body",
        maintype="application",
        subtype="octet-stream",
        filename="sample.bin",
    )

    with MailService.from_database_path(
        tmp_path / "multipart.db", analyzer=_analyzer()
    ) as service:
        imported = service.import_eml_bytes(message.as_bytes())

    assert "Plain text body" in imported["email"]["body"]
    assert "HTML alternative" not in imported["email"]["body"]
    assert "attachment payload" not in imported["email"]["body"]
    assert imported["email"]["body_html"] == ""


def test_malformed_but_recoverable_eml_is_imported(tmp_path: Path) -> None:
    raw = (
        b"Subject: Recoverable message\r\n"
        b"From sender@example.com\r\n"
        b"\r\n"
        b"The recoverable body remains available.\r\n"
    )
    with MailService.from_database_path(
        tmp_path / "recoverable.db", analyzer=_analyzer()
    ) as service:
        imported = service.import_eml_bytes(raw)

    assert imported["email"]["subject"] == "Recoverable message"
    assert "recoverable body" in imported["email"]["body"]


def test_import_eml_file_uses_same_fingerprint_path(tmp_path: Path) -> None:
    eml_path = tmp_path / "message.eml"
    eml_path.write_bytes(_plain_eml(subject="File import", body="Stored from a file."))
    with MailService.from_database_path(
        tmp_path / "file.db", analyzer=_analyzer()
    ) as service:
        first = service.import_eml_file(eml_path)
        second = service.import_eml_file(eml_path)

    assert first["created"] is True
    assert second["duplicate"] is True
