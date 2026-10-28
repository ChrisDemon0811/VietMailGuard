from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
from vietmailguard.mail_database import open_initialized_database
from vietmailguard.mail_repository import MailRepository
from vietmailguard.mail_service import MailService


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


def test_sidebar_folder_switch_opens_message_from_selected_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query_param_values,
) -> None:
    """A folder change must replace stale detail query parameters before a click."""

    database_path = tmp_path / "mailbox-navigation.db"
    connection = open_initialized_database(database_path)
    repository = MailRepository(connection)
    ids = {
        view: repository.create_email(
            sender=f"{view}@example.test",
            subject=f"Subject {view}",
            body=f"Body {view}",
            folder=view,
            is_read=True,
        )
        for view in ("hop_thu_den", "thu_rac", "cach_ly", "da_xoa")
    }
    connection.commit()
    connection.close()

    original_factory = MailService.from_database_path.__func__

    def from_test_database(cls: type[MailService], *_args: object, **kwargs: object) -> MailService:
        return original_factory(cls, database_path, **kwargs)

    monkeypatch.setattr(MailService, "from_database_path", classmethod(from_test_database))
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception

    for view in ("hop_thu_den", "thu_rac", "cach_ly", "da_xoa"):
        next(button for button in app.button if button.key == f"mail_nav_{view}").click()
        app.run()
        assert not app.exception
        assert query_param_values(app.query_params["view"]) == [view]
        assert query_param_values(app.query_params["mode"]) == ["list"]

        next(button for button in app.button if button.key == f"mail_select_{ids[view]}").click()
        app.run()
        assert not app.exception
        assert query_param_values(app.query_params["view"]) == [view]
        assert query_param_values(app.query_params["mode"]) == ["detail"]
        assert app.session_state["mailbox_selected_id"] == ids[view]
        assert any(f"Subject {view}" in item.value for item in app.markdown)
        assert any(f"Body {view}" in item.value for item in app.text)


def test_stored_security_details_are_specific_to_each_email(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mailbox details use persisted evidence, not static text or fresh inference."""

    database_path = tmp_path / "security-details.db"
    connection = open_initialized_database(database_path)
    repository = MailRepository(connection)
    normal_id = repository.create_email(
        sender="colleague@example.test",
        subject="Project meeting",
        body="We will meet tomorrow.",
        folder="hop_thu_den",
        is_read=True,
    )
    phishing_id = repository.create_email(
        sender="notice@example.test",
        subject="Account alert",
        body="Verify your password at http://192.0.2.1/login",
        folder="cach_ly",
        is_read=True,
    )
    repository.save_analysis(
        normal_id,
        model_version="V2",
        detected_language="en",
        prediction="normal",
        confidence=0.91,
        risk_score=4.0,
        risk_level="LOW",
        base_recommended_action="ALLOW",
        recommended_action="ALLOW",
        has_warning=False,
        security_finding_count=0,
        result={
            "prediction": "normal",
            "risk_level": "LOW",
            "recommended_action": "ALLOW",
            "class_probabilities": {"normal": 0.91, "spam": 0.06, "phishing": 0.03},
            "risk_components": {
                "ml_probability_and_prediction": 4.0,
                "url_indicators": 0.0,
                "content_indicators": 0.0,
                "sender_indicators": 0.0,
            },
            "model_explanation": {
                "target_class": "normal",
                "features": [{"feature": "meeting", "contribution": 0.25}],
            },
            "content_findings": [],
            "url_findings": [],
            "sender_findings": [],
            "limitations": [],
        },
    )
    repository.save_analysis(
        phishing_id,
        model_version="V2",
        detected_language="en",
        prediction="phishing",
        confidence=0.82,
        risk_score=79.0,
        risk_level="HIGH",
        base_recommended_action="QUARANTINE",
        recommended_action="QUARANTINE",
        has_warning=True,
        security_finding_count=2,
        result={
            "prediction": "phishing",
            "risk_level": "HIGH",
            "recommended_action": "QUARANTINE",
            "class_probabilities": {"normal": 0.11, "spam": 0.07, "phishing": 0.82},
            "risk_components": {
                "ml_probability_and_prediction": 59.0,
                "url_indicators": 8.0,
                "content_indicators": 12.0,
                "sender_indicators": 0.0,
            },
            "model_explanation": {
                "target_class": "phishing",
                "features": [{"feature": "verify password", "contribution": 0.34}],
            },
            "content_findings": [
                {"code": "credential_request", "evidence": ["password"], "description": "Credential request"}
            ],
            "url_findings": [
                {"url": "http://192.0.2.1/login", "findings": [
                    {"code": "ip_host", "evidence": ["192.0.2.1"], "description": "IP host"}
                ]}
            ],
            "sender_findings": [],
            "limitations": [],
        },
    )
    connection.commit()
    connection.close()

    original_factory = MailService.from_database_path.__func__

    def from_test_database(cls: type[MailService], *_args: object, **kwargs: object) -> MailService:
        return original_factory(cls, database_path, **kwargs)

    monkeypatch.setattr(MailService, "from_database_path", classmethod(from_test_database))
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception

    next(button for button in app.button if button.key == f"mail_select_{normal_id}").click()
    app.run()
    assert not app.exception
    normal_display = "\n".join(item.value for item in app.markdown)
    assert "91.00%" in normal_display
    assert "4.00" in normal_display
    assert "meeting" in normal_display
    assert "verify password" not in normal_display

    next(button for button in app.button if button.key == "mail_nav_cach_ly").click()
    app.run()
    next(button for button in app.button if button.key == f"mail_select_{phishing_id}").click()
    app.run()
    assert not app.exception
    phishing_display = "\n".join(item.value for item in app.markdown)
    assert "82.00%" in phishing_display
    assert "59.00" in phishing_display
    assert "verify password" in phishing_display
    assert "192.0.2.1" in phishing_display
    assert "91.00%" not in phishing_display


def test_demo_compose_analyzes_and_routes_via_mail_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query_param_values,
) -> None:
    """A composed demo mail uses the normal local analysis/routing workflow."""

    database_path = tmp_path / "demo-compose.db"
    analysis_calls: list[dict[str, object]] = []

    def fake_analyzer(**fields: object) -> dict[str, object]:
        analysis_calls.append(fields)
        return {
            "model_version": "test-double",
            "detected_language": "en",
            "prediction": "phishing",
            "confidence": 0.8,
            "risk_score": 72,
            "risk_level": "HIGH",
            "base_recommended_action": "QUARANTINE",
            "recommended_action": "QUARANTINE",
            "security_findings": [],
        }

    original_factory = MailService.from_database_path.__func__

    def from_test_database(cls: type[MailService], *_args: object, **kwargs: object) -> MailService:
        return original_factory(cls, database_path, analyzer=fake_analyzer)

    monkeypatch.setattr(MailService, "from_database_path", classmethod(from_test_database))
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception

    next(button for button in app.button if button.key == "mailbox_demo_compose_toggle").click()
    app.run()
    assert not app.exception
    next(item for item in app.text_input if item.label == "From (demo sender)").set_value(
        "sender@example.test"
    )
    next(item for item in app.text_input if item.label == "To (optional)").set_value(
        "recipient@example.test"
    )
    next(item for item in app.text_input if item.label == "Subject").set_value(
        "Account notice"
    )
    next(item for item in app.text_area if item.label == "Message body").set_value(
        "Please verify your account."
    )
    next(button for button in app.button if button.label == "Analyze and deliver locally").click()
    app.run()
    assert not app.exception
    assert len(analysis_calls) == 1
    assert query_param_values(app.query_params["view"]) == ["cach_ly"]
    assert query_param_values(app.query_params["mode"]) == ["detail"]
    assert any("Account notice" in item.value for item in app.markdown)
    assert any("Please verify your account." in item.value for item in app.text)

    connection = open_initialized_database(database_path)
    repository = MailRepository(connection)
    stored = repository.list_emails(folder="cach_ly")
    assert len(stored) == 1
    assert stored[0]["source"] == "demo_compose"
    assert repository.get_latest_analysis(stored[0]["id"])["prediction"] == "phishing"
    connection.close()
