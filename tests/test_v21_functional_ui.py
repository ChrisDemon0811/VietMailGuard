"""Actual local candidate + disposable UI/mailbox; no held-out or external API."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vietmailguard.mail_service import MailService
from vietmailguard.v21.runtime_inference import V21DevelopmentInference, model_availability


APP = Path(__file__).resolve().parents[1] / "app/app.py"


def widget(items, key):
    return next(item for item in items if item.key == key)


def local_app(tmp_path, monkeypatch):
    if not model_availability()[0]:
        pytest.skip("Configured V2.1 candidate not available locally")
    monkeypatch.setenv("VIETMAILGUARD_MAIL_DB", str(tmp_path / "mail.db"))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    return app


def test_real_candidate_analyze_save_open_mock_and_draft_without_implicit_inference(
    tmp_path, monkeypatch,
):
    calls = []
    original = V21DevelopmentInference.analyze_email

    def counted(self, **fields):
        calls.append(dict(fields))
        return original(self, **fields)

    monkeypatch.setattr(V21DevelopmentInference, "analyze_email", counted)
    import vietmailguard.v21.gemini_reviewer as gemini

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline UI must never construct Gemini")

    monkeypatch.setattr(gemini, "GeminiReviewerProvider", forbidden)
    app = local_app(tmp_path, monkeypatch)
    assert app.session_state["analysis_mode"] == "v2"
    widget(app.selectbox, "analysis_mode_selector").set_value("v21_hybrid").run()
    assert app.session_state["reviewer_provider"] == "mock"
    app.switch_page("views/email_analyzer.py").run()
    widget(app.text_input, "input_quick_sender").set_value("team@example.test").run()
    widget(app.text_input, "input_quick_subject").set_value("Lịch họp dự án").run()
    widget(app.text_area, "input_quick_body").set_value("Chào cả nhóm, cuộc họp bắt đầu sáng mai.").run()
    assert calls == []
    widget(app.selectbox, "language_selector").set_value("vi").run()
    assert widget(app.text_input, "input_quick_subject").value == "Lịch họp dự án"
    assert widget(app.text_area, "input_quick_body").value.startswith("Chào cả nhóm")
    assert calls == []
    widget(app.button, "quick_analyze_button").click().run()
    assert not app.exception
    assert len(calls) == 1
    before = copy.deepcopy(app.session_state["vietmailguard_analysis_result"]["result"])
    assert before["model_version"] == "2.1-development"
    assert before["confidence"] is None and before["class_probabilities"] == {}
    widget(app.button, "quick_llm_review_button").click().run()
    assert app.session_state["vietmailguard_llm_review"]["reviewer"]["status"] == "mock_only"
    assert app.session_state["vietmailguard_analysis_result"]["result"] == before
    widget(app.button, "quick_save_to_mailbox").click().run()
    assert not app.exception and not app.error
    assert len(calls) == 1
    saved_id = app.session_state["vietmailguard_analysis_result"]["saved_email_id"]
    # Repeated saves are idempotent; changing a draft does not change saved input.
    widget(app.text_input, "input_quick_subject").set_value("Unsaved replacement draft").run()
    widget(app.button, "quick_save_to_mailbox").click().run()
    assert app.session_state["vietmailguard_analysis_result"]["saved_email_id"] == saved_id
    widget(app.button, "quick_open_saved").click().run()
    assert not app.exception
    assert app.session_state["mailbox_mode"] == "detail"
    assert len(calls) == 1
    # AppTest needs its active script synchronized after programmatic switch_page
    # before a second-page widget interaction (the real browser follows the URL).
    app.switch_page("views/mail_client.py").run()
    widget(app.button, f"detail_read_{saved_id}").click().run()
    assert not app.exception
    with MailService.from_database_path(tmp_path / "mail.db") as service:
        assert service.open_email(saved_id, mark_as_read=False)["email"]["is_read"] is False
    assert app.session_state["mailbox_mode"] == "detail"
    assert len(calls) == 1
    with MailService.from_database_path(tmp_path / "mail.db") as service:
        stored = service.open_email(saved_id)
        assert stored["email"]["subject"] == "Lịch họp dự án"
        assert stored["analysis"]["result"] == before
        assert stored["llm_review"]["result"]["reviewer"]["status"] == "mock_only"
        assert service.mailbox_counts()["folders"]["hop_thu_den"] == 1
    app.switch_page("views/email_analyzer.py").run()
    assert widget(app.text_input, "input_quick_subject").value == "Unsaved replacement draft"
    assert len(calls) == 1


def test_missing_gemini_key_blocks_external_review_but_not_real_ml(tmp_path, monkeypatch):
    app = local_app(tmp_path, monkeypatch)
    widget(app.selectbox, "analysis_mode_selector").set_value("v21_hybrid").run()
    widget(app.selectbox, "reviewer_provider_selector").set_value("gemini").run()
    app.switch_page("views/email_analyzer.py").run()
    widget(app.text_input, "input_quick_subject").set_value("Project meeting").run()
    widget(app.text_area, "input_quick_body").set_value("Please review the project notes tomorrow.").run()
    widget(app.button, "quick_analyze_button").click().run()
    assert not app.exception
    assert app.session_state["vietmailguard_analysis_result"]["result"]["model_version"] == "2.1-development"
    assert widget(app.button, "quick_llm_review_button").disabled
    widget(app.checkbox, "quick_llm_consent").check().run()
    assert widget(app.button, "quick_llm_review_button").disabled
    assert "vietmailguard_llm_review" not in app.session_state


def test_empty_submission_clears_old_prediction_without_erasing_draft(tmp_path, monkeypatch):
    app = local_app(tmp_path, monkeypatch)
    widget(app.selectbox, "analysis_mode_selector").set_value("v21_ml").run()
    app.switch_page("views/email_analyzer.py").run()
    widget(app.text_input, "input_quick_subject").set_value("Local meeting").run()
    widget(app.button, "quick_analyze_button").click().run()
    assert app.session_state["vietmailguard_analysis_result"]["result"]["prediction"] in {"normal", "spam", "phishing"}
    widget(app.text_input, "input_quick_subject").set_value("").run()
    widget(app.button, "quick_analyze_button").click().run()
    assert not app.exception and app.error
    assert "vietmailguard_analysis_result" not in app.session_state
