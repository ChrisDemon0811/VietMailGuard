from __future__ import annotations

from app.mailbox_ui import (
    display_timestamp,
    is_security_message,
    message_preview,
    normalize_mailbox_view,
)


def test_mailbox_view_uses_safe_internal_values() -> None:
    assert normalize_mailbox_view("thu_rac") == "thu_rac"
    assert normalize_mailbox_view("Hộp thư đến") == "hop_thu_den"
    assert normalize_mailbox_view(None) == "hop_thu_den"


def test_message_preview_is_compact_and_non_destructive() -> None:
    original = "First line\n\nSecond line with   spaces and more text"
    assert message_preview(original, maximum=28) == "First line Second line with…"
    assert original == "First line\n\nSecond line with   spaces and more text"


def test_display_timestamp_handles_iso_and_unknown_values() -> None:
    assert display_timestamp("2026-09-15T08:30:00Z") == "2026-09-15 08:30"
    assert display_timestamp("not-a-date") == "not-a-date"
    assert display_timestamp("") == "—"


def test_security_view_uses_only_stored_analysis() -> None:
    assert is_security_message({"analysis": {"has_warning": True, "risk_level": "LOW"}})
    assert is_security_message({"analysis": {"has_warning": False, "risk_level": "HIGH"}})
    assert not is_security_message(
        {"analysis": {"has_warning": False, "risk_level": "MEDIUM"}}
    )
    assert not is_security_message({"analysis": None})
