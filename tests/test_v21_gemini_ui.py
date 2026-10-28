"""Gemini UI wiring with synthetic emails and a fake SDK transport only."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from vietmailguard.mail_service import MailService


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "app.py"
SUBJECT = "Synthetic meeting notice"
BODY = "Please review the meeting agenda tomorrow."
PRIVATE_ERROR = "DO_NOT_DISPLAY_PROVIDER_REQUEST_OR_CREDENTIAL"


def _widget(items, key):
    return next(item for item in items if item.key == key)


def _analysis(prediction="normal"):
    action = {"normal": "ALLOW", "spam": "MOVE_TO_SPAM", "phishing": "QUARANTINE"}[prediction]
    return {
        "model_version": "2.1-development", "model_name": "Synthetic ML test double",
        "prediction": prediction, "detected_language": "en",
        "language_support_status": {"status": "experimental"},
        "confidence": None, "class_probabilities": {},
        "risk_score": 25, "risk_level": "LOW",
        "base_recommended_action": action, "recommended_action": action,
        "security_findings": [], "limitations": ["Synthetic test fixture only."],
    }


@pytest.fixture
def fake_gemini(tmp_path, monkeypatch, streamlit_loopback):
    """Keep the real provider/service validation; fake only SDK I/O and ML."""
    from vietmailguard.v21 import gemini_reviewer as gemini
    from vietmailguard.v21 import llm_reviewer, runtime_inference

    monkeypatch.setenv("VIETMAILGUARD_MAIL_DB", str(tmp_path / "mail.db"))
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-test-placeholder-not-a-real-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("GEMINI_REVIEW_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(gemini, "_DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(gemini, "_sdk_version", lambda _package: "test-version")
    monkeypatch.setattr(runtime_inference, "model_availability", lambda: (True, "test fixture"))
    state = SimpleNamespace(
        constructions=[], calls=[], options=[], closed=0, ml_calls=[],
        response={"assessment": "phishing", "summary": "Independent synthetic interpretation.",
                  "evidence": [{"field": "subject", "quote": SUBJECT,
                                "explanation": "Literal subject used for test only."}],
                  "suspicious_indicators": ["Synthetic disagreement; not a benchmark."]},
        error=None, database=tmp_path / "mail.db",
    )

    class FakeSDKClient:
        def __init__(self):
            self.models = self

        def generate_content(self, **kwargs):
            state.calls.append(kwargs)
            if state.error is not None:
                raise state.error
            text = state.response if isinstance(state.response, str) else json.dumps(state.response)
            return SimpleNamespace(text=text)

        def close(self):
            state.closed += 1

    def fake_client(_key, timeout_seconds):
        state.constructions.append(timeout_seconds)
        return FakeSDKClient()

    class FakeMLEngine:
        def analyze_email(self, **fields):
            state.ml_calls.append(fields)
            return _analysis()

    actual_review = llm_reviewer.review_email

    def capture_review(**kwargs):
        state.options.append({key: kwargs[key] for key in (
            "timeout_seconds", "max_retries", "allow_external", "explicit_email_sharing_consent")})
        return actual_review(**kwargs)

    monkeypatch.setattr(gemini, "_client", fake_client)
    monkeypatch.setattr(runtime_inference, "load_v21_engine", lambda: FakeMLEngine())
    monkeypatch.setattr(llm_reviewer, "review_email", capture_review)
    return state


def _hybrid_app():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    _widget(app.selectbox, "analysis_mode_selector").set_value("v21_hybrid").run()
    assert app.session_state["reviewer_provider"] == "mock"
    return app


def _quick_result(app):
    app.switch_page("views/email_analyzer.py").run()
    _widget(app.text_input, "input_quick_sender").set_value("demo@example.com").run()
    _widget(app.text_input, "input_quick_subject").set_value(SUBJECT).run()
    _widget(app.text_area, "input_quick_body").set_value(BODY).run()
    _widget(app.button, "quick_analyze_button").click().run()
    assert not app.exception
    return copy.deepcopy(app.session_state["vietmailguard_analysis_result"]["result"])


def _visible_text(app):
    kinds = ("caption", "text", "markdown", "warning", "error", "info", "code")
    return "\n".join(str(item.value) for kind in kinds for item in app.get(kind))


def test_gemini_quick_explicit_click_and_consent_keep_ml_and_saved_routing(fake_gemini):
    state = fake_gemini
    app = _hybrid_app()
    original = _quick_result(app)
    assert state.calls == state.constructions == []
    _widget(app.selectbox, "reviewer_provider_selector").set_value("gemini").run()
    assert state.calls == state.constructions == []
    assert _widget(app.button, "quick_llm_review_button").disabled
    assert "have not been checked with google" in _visible_text(app).lower()
    assert "Google" in _visible_text(app)
    _widget(app.checkbox, "quick_llm_consent").check().run()
    assert state.calls == state.constructions == []
    assert not _widget(app.button, "quick_llm_review_button").disabled
    _widget(app.button, "quick_llm_review_button").click().run()
    assert not app.exception
    assert len(state.calls) == state.closed == 1
    assert state.constructions == [30.0]
    assert state.options == [{"timeout_seconds": 30.0, "max_retries": 0,
                              "allow_external": True, "explicit_email_sharing_consent": True}]
    request = json.loads(state.calls[0]["contents"])["untrusted_email_data"]
    assert request == {"sender": "demo@example.com", "subject": SUBJECT, "body": BODY}
    assert state.calls[0]["config"]["tools"] == []
    assert state.calls[0]["config"]["response_mime_type"] == "application/json"
    assert "UNTRUSTED DATA" in state.calls[0]["config"]["system_instruction"]
    reviewed = app.session_state["vietmailguard_llm_review"]
    assert reviewed["reviewer"]["provider"] == "gemini"
    assert reviewed["reviewer"]["assessment"] == "phishing"
    assert reviewed["reviewer"]["recommendation"] == "REVIEW"
    assert reviewed["reviewer"]["calibrated_probability"] is None
    assert app.session_state["vietmailguard_analysis_result"]["result"] == original
    _widget(app.button, "quick_save_to_mailbox").click().run()
    email_id = app.session_state["vietmailguard_analysis_result"]["saved_email_id"]
    with MailService.from_database_path(state.database) as service:
        stored = service.open_email(email_id)
        assert stored["email"]["folder"] == "hop_thu_den"
        assert stored["analysis"]["result"] == original
        assert stored["llm_review"]["result"]["reviewer"]["assessment"] == "phishing"
    assert len(state.calls) == 1


def test_mock_stays_default_and_provider_changes_reset_consent(fake_gemini):
    state = fake_gemini
    app = _hybrid_app()
    _quick_result(app)
    _widget(app.button, "quick_llm_review_button").click().run()
    assert app.session_state["vietmailguard_llm_review"]["reviewer"]["status"] == "mock_only"
    assert state.calls == state.constructions == []
    _widget(app.selectbox, "reviewer_provider_selector").set_value("gemini").run()
    _widget(app.checkbox, "quick_llm_consent").check().run()
    _widget(app.selectbox, "reviewer_provider_selector").set_value("mock").run()
    _widget(app.selectbox, "reviewer_provider_selector").set_value("gemini").run()
    assert not _widget(app.checkbox, "quick_llm_consent").value
    assert _widget(app.button, "quick_llm_review_button").disabled
    assert state.calls == state.constructions == []


@pytest.mark.parametrize(("failure", "expected_status", "expected_category", "expected_message"), [
    ("quota", "provider_error", "rate_or_quota", "quota"),
    ("timeout", "provider_error", "timeout", "time"),
    ("invalid_json", "invalid_response", "invalid_response", "JSON"),
])
def test_gemini_api_failures_are_safe_no_retry_and_keep_ml(
    fake_gemini, failure, expected_status, expected_category, expected_message,
):
    state = fake_gemini
    if failure == "quota":
        class FakeAPIError(RuntimeError):
            code = 429
            status = "RESOURCE_EXHAUSTED"

        state.error = FakeAPIError(PRIVATE_ERROR)
    elif failure == "timeout":
        state.error = TimeoutError(PRIVATE_ERROR)
    else:
        state.response = '{"invalid": "' + PRIVATE_ERROR + '"'
    app = _hybrid_app()
    original = _quick_result(app)
    _widget(app.selectbox, "reviewer_provider_selector").set_value("gemini").run()
    _widget(app.checkbox, "quick_llm_consent").check().run()
    _widget(app.button, "quick_llm_review_button").click().run()
    assert not app.exception
    assert len(state.calls) == state.closed == 1
    assert state.options[0]["max_retries"] == 0
    reviewed = app.session_state["vietmailguard_llm_review"]
    assert reviewed["reviewer"]["status"] == expected_status
    assert reviewed["reviewer"]["diagnostic"]["category"] == expected_category
    assert reviewed["reviewer"]["recommendation"] == "REVIEW"
    assert reviewed["reviewer"]["assessment"] is None
    assert app.session_state["vietmailguard_analysis_result"]["result"] == original
    visible = _visible_text(app)
    assert expected_message.casefold() in visible.casefold()
    assert PRIVATE_ERROR not in visible + repr(reviewed)
    assert "synthetic-test-placeholder-not-a-real-key" not in visible + repr(reviewed)
    assert app.error
    if failure == "quota":
        assert "HTTP 429" in visible


def test_missing_key_prevents_gemini_construction_even_with_checkbox(fake_gemini, monkeypatch):
    state = fake_gemini
    monkeypatch.setenv("GEMINI_API_KEY", "")
    app = _hybrid_app()
    _quick_result(app)
    _widget(app.selectbox, "reviewer_provider_selector").set_value("gemini").run()
    _widget(app.checkbox, "quick_llm_consent").check().run()
    assert _widget(app.button, "quick_llm_review_button").disabled
    assert state.calls == state.constructions == []


def test_gemini_mailbox_review_preserves_quarantine_and_exact_analysis(fake_gemini, query_param_values):
    state = fake_gemini
    original = _analysis("phishing")
    state.response["assessment"] = "normal"
    with MailService.from_database_path(state.database) as service:
        email_id = service.save_analyzed_email(
            sender="demo@example.com", subject=SUBJECT, body=BODY,
            analysis_result=original,
        )["email"]["id"]
    app = _hybrid_app()
    _widget(app.selectbox, "reviewer_provider_selector").set_value("gemini").run()
    _widget(app.button, "mail_nav_cach_ly").click().run()
    _widget(app.button, f"mail_select_{email_id}").click().run()
    assert not app.exception
    assert state.calls == state.constructions == []
    assert _widget(app.button, f"llm_review_{email_id}").disabled
    _widget(app.checkbox, f"llm_consent_{email_id}").check().run()
    assert state.calls == state.constructions == []
    _widget(app.button, f"llm_review_{email_id}").click().run()
    assert not app.exception
    assert len(state.calls) == state.closed == 1
    assert state.options == [{"timeout_seconds": 30.0, "max_retries": 0,
                              "allow_external": True, "explicit_email_sharing_consent": True}]
    assert state.ml_calls == []
    with MailService.from_database_path(state.database) as service:
        stored = service.open_email(email_id)
        assert stored["email"]["folder"] == "cach_ly"
        assert stored["analysis"]["result"] == original
        assert stored["llm_review"]["result"]["reviewer"]["assessment"] == "normal"
        assert stored["llm_review"]["result"]["reviewer"]["recommendation"] == "REVIEW"
    assert query_param_values(app.query_params["view"]) == ["cach_ly"]
    assert query_param_values(app.query_params["mode"]) == ["detail"]
