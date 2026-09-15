from __future__ import annotations

from vietmailguard.eml_parser import parse_eml_bytes


def test_parse_plain_text_eml() -> None:
    raw = (
        b"From: Alice <alice@example.com>\r\n"
        b"To: Bob <bob@example.org>\r\n"
        b"Date: Tue, 10 Sep 2024 09:30:00 +0000\r\n"
        b"Subject: Project update\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"Hello Bob,\r\nThe project is on schedule.\r\n"
    )
    parsed = parse_eml_bytes(raw)
    assert parsed["sender"] == "Alice <alice@example.com>"
    assert parsed["receiver"] == "Bob <bob@example.org>"
    assert parsed["subject"] == "Project update"
    assert "project is on schedule" in parsed["body"].casefold()


def test_parse_html_fallback_preserves_link_target() -> None:
    raw = (
        b"From: notice@example.com\r\n"
        b"To: user@example.org\r\n"
        b"Subject: HTML message\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"\r\n"
        b"<html><body><p>Read the notice.</p>"
        b"<a href='https://example.com/notice'>Open</a></body></html>"
    )
    parsed = parse_eml_bytes(raw)
    assert "Read the notice" in parsed["body"]
    assert "https://example.com/notice" in parsed["body"]
