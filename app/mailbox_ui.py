"""Pure presentation helpers shared by the Streamlit mailbox page."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, MutableMapping


MAILBOX_VIEWS = (
    "hop_thu_den",
    "gan_sao",
    "thu_rac",
    "cach_ly",
    "da_xoa",
    "security",
)

MAILBOX_MODES = ("list", "detail")

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


def resolve_mailbox_view(
    requested_view: object,
    state: Mapping[str, Any],
) -> str:
    """Resolve a view while tolerating query parameters lost by page reruns."""

    candidate = str(requested_view or "")
    if candidate in MAILBOX_VIEWS:
        return candidate
    return normalize_mailbox_view(state.get("mailbox_current_view"))


def resolve_mailbox_mode(
    requested_mode: object,
    state: Mapping[str, Any],
) -> str:
    """Resolve list/detail mode with session state as the durable fallback."""

    candidate = str(requested_mode or "")
    if candidate in MAILBOX_MODES:
        return candidate
    stored = str(state.get("mailbox_mode") or "")
    return stored if stored in MAILBOX_MODES else "list"


def sync_mailbox_navigation(
    state: MutableMapping[str, Any],
    current_view: object,
    requested_mode: object = None,
) -> str:
    """Synchronize explicit list/detail state with safe query parameters."""

    view = normalize_mailbox_view(current_view)
    requested = str(requested_mode or "")
    previous_view = state.get("mailbox_current_view")
    state["mailbox_current_view"] = view

    if previous_view != view and requested != "detail":
        state["mailbox_return_view"] = view
        state["mailbox_mode"] = "list"

    if requested == "list":
        state["mailbox_mode"] = "list"
        state["mailbox_return_view"] = view
    elif requested == "detail" and state.get("mailbox_selected_id") is not None:
        state["mailbox_mode"] = "detail"
    elif state.get("mailbox_mode") not in MAILBOX_MODES:
        state["mailbox_mode"] = "list"

    if state["mailbox_mode"] == "detail" and state.get("mailbox_selected_id") is None:
        state["mailbox_mode"] = "list"
    state.setdefault("mailbox_return_view", view)
    return str(state["mailbox_mode"])


def select_mailbox_message(
    state: MutableMapping[str, Any],
    email_id: int,
    current_view: object,
) -> None:
    """Transition from a mailbox list to one directly addressable detail view."""

    view = normalize_mailbox_view(current_view)
    state["mailbox_selected_id"] = int(email_id)
    state["mailbox_current_view"] = view
    state["mailbox_return_view"] = view
    state["mailbox_mode"] = "detail"


def return_to_mailbox_list(state: MutableMapping[str, Any]) -> str:
    """Return to the originating view without discarding the selected id."""

    view = normalize_mailbox_view(
        state.get("mailbox_return_view", state.get("mailbox_current_view"))
    )
    state["mailbox_current_view"] = view
    state["mailbox_mode"] = "list"
    return view


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
