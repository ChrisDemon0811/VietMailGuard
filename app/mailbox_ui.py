"""Pure presentation helpers shared by the Streamlit mailbox page."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


MAILBOX_VIEWS = (
    "hop_thu_den",
    "gan_sao",
    "thu_rac",
    "cach_ly",
    "da_xoa",
    "security",
)

FOLDER_VIEW_KEYS = {
    "hop_thu_den": "folder_inbox",
    "gan_sao": "folder_starred",
    "thu_rac": "folder_spam",
    "cach_ly": "folder_quarantine",
    "da_xoa": "folder_deleted",
    "security": "security_view",
}


def normalize_mailbox_view(value: object) -> str:
    """Return a safe internal mailbox view identifier."""

    candidate = str(value or "")
    return candidate if candidate in MAILBOX_VIEWS else "hop_thu_den"


def message_preview(value: object, *, maximum: int = 105) -> str:
    """Create a compact one-line preview without modifying stored content."""

    text = " ".join(str(value or "").split())
    if len(text) <= maximum:
        return text
    return text[: max(1, maximum - 1)].rstrip() + "…"


def display_timestamp(value: object) -> str:
    """Format an ISO/RFC timestamp compactly and preserve unknown formats."""

    text = str(value or "").strip()
    if not text:
        return "—"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return parsed.strftime("%Y-%m-%d %H:%M")


def is_security_message(message: Mapping[str, Any]) -> bool:
    """Identify rows eligible for Security View from stored analysis only."""

    analysis = message.get("analysis")
    if not isinstance(analysis, Mapping):
        return False
    return bool(analysis.get("has_warning")) or str(analysis.get("risk_level")) in {
        "HIGH",
        "CRITICAL",
    }
