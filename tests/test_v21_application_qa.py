"""Application QA with disposable mailboxes and no external Gemini request."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vietmailguard.mail_database import open_initialized_database
from vietmailguard.mail_repository import MailRepository
from vietmailguard.mail_service import MailService
from vietmailguard.v21.runtime_inference import model_availability
from vietmailguard.v21 import runtime_modes  # import before any temporary availability patch


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


def _temporary_mailbox(monkeypatch: pytest.MonkeyPatch, path: Path) -> list[str]:
    original = MailService.from_database_path.__func__
    errors: list[str] = []

    def factory(cls: type[MailService], *_args: object, **kwargs: object) -> MailService:
        try:
            return original(cls, path, **kwargs)
        except Exception as error:
            errors.append(repr(error))
            raise

    monkeypatch.setattr(MailService, "from_database_path", classmethod(factory))
    return errors


def _selector(app: AppTest, key: str):
    return next(item for item in app.selectbox if item.key == key)


def test_default_v2_and_invalid_v21_artifact_is_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _temporary_mailbox(monkeypatch, tmp_path / "mail.db")
    import vietmailguard.v21.runtime_inference as inference

    monkeypatch.setattr(inference, "model_availability", lambda: (False, "bad artifact"))
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert app.session_state["analysis_mode"] == "v2"
    assert _selector(app, "analysis_mode_selector").options == ["V2 frozen production"]
    assert any("artifact" in item.value.lower() for item in app.sidebar.caption)


def test_v21_and_hybrid_ui_are_opt_in_and_do_not_call_provider_on_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not model_availability()[0]:
        pytest.skip("V2.1 candidate artifact not present in clean clone")
    _temporary_mailbox(monkeypatch, tmp_path / "mail.db")
    monkeypatch.setenv("GEMINI_API_KEY", "qa-placeholder-never-sent")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    import vietmailguard.v21.gemini_reviewer as gemini

    provider_calls: list[str] = []

    class LocalFakeProvider:
        name = "qa-fake"
        is_external = True

        def review(self, request, *, timeout_seconds):
            del timeout_seconds
            provider_calls.append(request.subject)
            return {"assessment": "review", "summary": "Human review suggested.",
                    "evidence": [{"field": "subject", "quote": request.subject,
                                  "explanation": "Observed subject only."}],
                    "suspicious_indicators": []}

    monkeypatch.setattr(gemini, "GeminiReviewerProvider", LocalFakeProvider)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    assert app.session_state["analysis_mode"] == "v2"

    _selector(app, "analysis_mode_selector").set_value("v21_ml").run()
    assert not app.exception
    assert app.session_state["analysis_mode"] == "v21_ml"
    app.switch_page("views/email_analyzer.py").run()
    assert not app.exception
    app.text_input[1].set_value("Local meeting")
    app.text_area[0].set_value("Please meet at three tomorrow.")
    app.button[0].click().run(timeout=60)
    assert not app.exception
    result = dict(app.session_state["vietmailguard_analysis_result"]["result"])
    assert result["model_version"] == "2.1-development"
    assert result["confidence"] is None
    assert result["class_probabilities"] == {}
    assert result["risk_score"] != result["confidence"]
    assert provider_calls == []

    _selector(app, "analysis_mode_selector").set_value("v21_hybrid").run()
    assert not app.exception
    assert app.session_state["analysis_mode"] == "v21_hybrid"
    assert provider_calls == []
    _selector(app, "reviewer_provider_selector").set_value("gemini").run()
    # Changing mode invalidates the old displayed result; analyze again.
    app.button[0].click().run(timeout=60)
    assert not app.exception
    ml_before = dict(app.session_state["vietmailguard_analysis_result"]["result"])
    assert provider_calls == []
    review_button = next(button for button in app.button if button.key == "quick_llm_review_button")
    assert review_button.disabled
    next(box for box in app.checkbox if box.key == "quick_llm_consent").check().run()
    assert provider_calls == []
    next(button for button in app.button if button.key == "quick_llm_review_button").click().run()
    assert not app.exception
    assert provider_calls == ["Local meeting"]
    assert app.session_state["vietmailguard_analysis_result"]["result"] == ml_before
    assert app.session_state["vietmailguard_llm_review"]["reviewer"]["recommendation"] == "REVIEW"


def test_disposable_mailbox_eml_feedback_persistence_and_integrity(tmp_path: Path) -> None:
    path = tmp_path / "mail.db"
    calls = []

    def analyzer(**fields):
        calls.append(fields)
        return {"model_version": "qa-double", "detected_language": "en",
                "prediction": "spam", "confidence": None, "risk_score": 40,
                "risk_level": "MEDIUM", "base_recommended_action": "MOVE_TO_SPAM",
                "recommended_action": "MOVE_TO_SPAM", "security_findings": []}

    eml = (b"From: qa@example.test\r\nTo: me@example.test\r\n"
           b"Subject: Local offer\r\nMessage-ID: <qa-11a@example.test>\r\n"
           b"Content-Type: text/plain; charset=utf-8\r\n\r\nSample offer only.\r\n")
    with MailService.from_database_path(path, analyzer=analyzer) as service:
        imported = service.import_eml_bytes(eml)
        email_id = imported["email"]["id"]
        assert imported["email"]["folder"] == "thu_rac"
        assert service.import_eml_bytes(eml)["duplicate"] is True
        assert len(calls) == 1
        service.record_feedback(email_id, action="khong_phai_thu_rac")
        assert service.open_email(email_id)["analysis"]["prediction"] == "spam"
    with MailService.from_database_path(path, analyzer=analyzer) as restarted:
        saved = restarted.open_email(email_id)
        assert saved["email"]["folder"] == "hop_thu_den"
        assert saved["analysis"]["confidence"] is None
        assert saved["analysis"]["prediction"] == "spam"
        assert len(saved["folder_history"]) >= 2
        assert len(calls) == 1
    connection = open_initialized_database(path)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM phan_hoi_nguoi_dung").fetchone()[0] >= 1
    finally:
        connection.close()


def test_hybrid_mailbox_review_does_not_reanalyze_or_reroute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    query_param_values,
) -> None:
    if not model_availability()[0]:
        pytest.skip("V2.1 candidate artifact not present in clean clone")
    path = tmp_path / "hybrid-mail.db"
    analysis_calls = []

    def analyzer(**fields):
        analysis_calls.append(fields)
        return {"model_version": "qa-double", "detected_language": "en",
                "prediction": "phishing", "confidence": .8, "risk_score": 80,
                "risk_level": "HIGH", "base_recommended_action": "QUARANTINE",
                "recommended_action": "QUARANTINE", "security_findings": []}

    with MailService.from_database_path(path, analyzer=analyzer) as service:
        email_id = service.receive_email(sender="qa@example.test", subject="QA alert",
                                         body="Local test email only.")["email"]["id"]
    _temporary_mailbox(monkeypatch, path)
    monkeypatch.setenv("GEMINI_API_KEY", "qa-placeholder-never-sent")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    import vietmailguard.v21.gemini_reviewer as gemini

    provider_calls = []

    class LocalFakeProvider:
        name = "qa-fake"
        is_external = True

        def review(self, request, *, timeout_seconds):
            del timeout_seconds
            provider_calls.append(request.subject)
            return {"assessment": "normal", "summary": "Review the conflicting interpretation.",
                    "evidence": [{"field": "subject", "quote": "QA alert",
                                  "explanation": "Observed subject."}],
                    "suspicious_indicators": []}

    monkeypatch.setattr(gemini, "GeminiReviewerProvider", LocalFakeProvider)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    _selector(app, "analysis_mode_selector").set_value("v21_hybrid").run()
    _selector(app, "reviewer_provider_selector").set_value("gemini").run()
    next(button for button in app.button if button.key == "mail_nav_cach_ly").click().run()
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]
    assert any(button.key == f"mail_select_{email_id}" for button in app.button)
    next(button for button in app.button if button.key == f"mail_select_{email_id}").click().run()
    assert not app.exception
    assert query_param_values(app.query_params["view"]) == ["cach_ly"]
    assert query_param_values(app.query_params["mode"]) == ["detail"]
    assert provider_calls == []
    assert len(analysis_calls) == 1
    assert app.session_state["analysis_mode"] == "v21_hybrid", [
        (button.key, button.label) for button in app.button
    ]
    assert next(button for button in app.button if button.key == f"llm_review_{email_id}").disabled
    next(box for box in app.checkbox if box.key == f"llm_consent_{email_id}").check().run()
    assert provider_calls == []
    next(button for button in app.button if button.key == f"llm_review_{email_id}").click().run()
    assert not app.exception
    assert provider_calls == ["QA alert"]
    assert len(analysis_calls) == 1
    with MailService.from_database_path(path, analyzer=analyzer) as service:
        stored = service.open_email(email_id)
        assert stored["email"]["folder"] == "cach_ly"
        assert stored["analysis"]["prediction"] == "phishing"
        assert stored["analysis"]["confidence"] == .8
        assert stored["llm_review"]["result"]["reviewer"]["assessment"] == "normal"
        assert stored["llm_review"]["result"]["reviewer"]["recommendation"] == "REVIEW"
