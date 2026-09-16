from __future__ import annotations

import pytest

from app.mailbox_ui import (
    display_timestamp,
    is_security_message,
    message_preview,
    normalize_mailbox_view,
    resolve_mailbox_mode,
    resolve_mailbox_view,
    return_to_mailbox_list,
    select_mailbox_message,
    sync_mailbox_navigation,
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


def test_list_detail_navigation_preserves_origin_and_selected_email() -> None:
    state: dict[str, object] = {}
    assert sync_mailbox_navigation(state, "thu_rac", "list") == "list"

    select_mailbox_message(state, 42, "thu_rac")
    assert state == {
        "mailbox_current_view": "thu_rac",
        "mailbox_return_view": "thu_rac",
        "mailbox_mode": "detail",
        "mailbox_selected_id": 42,
    }
    assert sync_mailbox_navigation(state, "thu_rac", "detail") == "detail"
    assert return_to_mailbox_list(state) == "thu_rac"
    assert state["mailbox_mode"] == "list"
    assert state["mailbox_selected_id"] == 42


def test_detail_mode_does_not_depend_on_filtered_list_membership() -> None:
    state: dict[str, object] = {}
    sync_mailbox_navigation(state, "cach_ly", "list")
    select_mailbox_message(state, 99, "cach_ly")

    available_ids: set[int] = set()
    assert 99 not in available_ids
    assert sync_mailbox_navigation(state, "cach_ly", "detail") == "detail"
    assert state["mailbox_selected_id"] == 99


def test_sidebar_view_change_returns_to_list_mode() -> None:
    state: dict[str, object] = {}
    select_mailbox_message(state, 7, "hop_thu_den")

    assert sync_mailbox_navigation(state, "cach_ly", "list") == "list"
    assert state["mailbox_current_view"] == "cach_ly"
    assert state["mailbox_return_view"] == "cach_ly"


@pytest.mark.parametrize(
    "view",
    ["hop_thu_den", "thu_rac", "cach_ly", "da_xoa", "security"],
)
def test_every_mailbox_list_can_transition_to_shared_detail(view: str) -> None:
    state: dict[str, object] = {}
    sync_mailbox_navigation(state, view, "list")
    select_mailbox_message(state, 123, view)

    assert sync_mailbox_navigation(state, view, "detail") == "detail"
    assert state["mailbox_selected_id"] == 123
    assert state["mailbox_return_view"] == view


def test_star_or_language_change_does_not_discard_open_message_state() -> None:
    state: dict[str, object] = {"interface_language": "en"}
    select_mailbox_message(state, 88, "thu_rac")

    state["interface_language"] = "vi"
    state["mailbox_star_changed"] = True

    assert sync_mailbox_navigation(state, "thu_rac", "detail") == "detail"
    assert state["mailbox_selected_id"] == 88
    assert state["mailbox_return_view"] == "thu_rac"


@pytest.mark.parametrize("view", ["thu_rac", "cach_ly"])
def test_lost_query_params_do_not_bounce_protected_folders_to_inbox(
    view: str,
) -> None:
    state: dict[str, object] = {
        "mailbox_current_view": view,
        "mailbox_return_view": view,
        "mailbox_mode": "list",
    }

    assert resolve_mailbox_view(None, state) == view
    assert resolve_mailbox_mode(None, state) == "list"

    select_mailbox_message(state, 55, view)

    assert resolve_mailbox_view(None, state) == view
    assert resolve_mailbox_mode(None, state) == "detail"
    assert sync_mailbox_navigation(
        state,
        resolve_mailbox_view(None, state),
        resolve_mailbox_mode(None, state),
    ) == "detail"
    assert state["mailbox_selected_id"] == 55
    assert state["mailbox_return_view"] == view
