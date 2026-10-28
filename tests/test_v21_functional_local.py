"""Real local-candidate smoke integration, not a scientific evaluation.

All email examples below are invented fixtures. No corpus, validation subset,
held-out Test, external provider, model fitting or network is used. A clean
source-only clone skips only the tests requiring the optional candidate binary.
"""

from __future__ import annotations

from copy import deepcopy
from email.message import EmailMessage
import json
from pathlib import Path

import pytest

from vietmailguard.mail_database import open_initialized_database
from vietmailguard.mail_service import MailService
from vietmailguard.v21.llm_reviewer import (
    MockReviewerProvider, ReviewerPrivacyError, review_email,
)
from vietmailguard.v21.runtime_inference import (
    DEFAULT_RUNTIME_CONFIG, PROJECT_ROOT, V21ModelUnavailable,
    load_v21_engine,
)


SYNTHETIC_EMAILS = [
    ("en", "Project meeting", "Hello team, please join tomorrow's project meeting at three."),
    ("vi", "Lịch họp dự án", "Chào nhóm, cuộc họp dự án bắt đầu lúc ba giờ chiều mai."),
    ("zh", "会议通知", "明天下午三点在办公室开会，请提前准备项目资料。"),
]
FOLDERS = ("hop_thu_den", "thu_rac", "cach_ly", "da_xoa")


@pytest.fixture(scope="module")
def real_engine():
    config = json.loads(DEFAULT_RUNTIME_CONFIG.read_text(encoding="utf-8"))
    if not (PROJECT_ROOT / config["artifact"]).is_file():
        pytest.skip("Optional real V2.1 candidate binary is not supplied in source-only checkout")
    # If an artifact exists but is invalid, fail rather than turn it into a skip.
    return load_v21_engine()


def _eml(subject: str, body: str, *, html: bool = False) -> bytes:
    message = EmailMessage()
    message["From"] = "Synthetic Sender <sender@example.com>"
    message["To"] = "reader@example.org"
    message["Subject"] = subject
    message["Message-ID"] = "<functional-v21@example.com>"
    message.set_content(body)
    if html:
        message.add_alternative(
            "<html><body><p>Invented meeting notice.</p>"
            '<img src="https://example.net/tracker.png">'
            '<script>alert("synthetic")</script></body></html>', subtype="html")
    return message.as_bytes()


@pytest.mark.parametrize("language,subject,body", SYNTHETIC_EMAILS)
def test_real_candidate_uses_three_classes_without_fake_probability(
    real_engine, language: str, subject: str, body: str,
) -> None:
    result = real_engine.analyze_email(sender="sender@example.com", subject=subject, body=body)
    assert result["model_version"] == "2.1-development"
    assert result["model_scope"] == "development_candidate_only"
    assert result["prediction"] in {"normal", "spam", "phishing"}
    assert result["confidence"] is None
    assert result["class_probabilities"] == {}
    assert result["probability_source"] == "unavailable_uncalibrated"
    assert result["detected_language"] == language
    assert 0 <= result["risk_score"] <= 100
    assert result["model_artifact"]["artifact_sha256"]
    assert result["model_explanation"]["limitation"]
    expected_status = "unvalidated" if language == "zh" else "experimental"
    assert result["language_support_status"]["status"] == expected_status
    if language == "zh":
        assert any("Chinese" in value for value in result["limitations"])


@pytest.mark.parametrize("language,subject,body", SYNTHETIC_EMAILS)
def test_real_eml_import_duplicate_open_every_folder_and_restart(
    real_engine, tmp_path: Path, language: str, subject: str, body: str,
) -> None:
    database = tmp_path / "local-mail.db"
    calls = []

    def analyze(**fields):
        calls.append(fields)
        return real_engine.analyze_email(**fields)

    with MailService.from_database_path(database, analyzer=analyze) as service:
        first = service.import_eml_bytes(_eml(subject, body, html=True))
        identifier = first["email"]["id"]
        original = deepcopy(first["analysis"]["result"])
        expected_folder = {"normal": "hop_thu_den", "spam": "thu_rac", "phishing": "cach_ly"}[
            original["prediction"]]
        assert first["email"]["folder"] == expected_folder
        assert first["email"]["subject"] == subject
        assert first["email"]["body"].strip() == body
        assert first["analysis"]["detected_language"] == language
        duplicate = service.import_eml_bytes(_eml(subject, body, html=True))
        assert duplicate["duplicate"] is True
        assert duplicate["email"]["id"] == identifier
        assert len(calls) == 1
        for folder in FOLDERS:
            service.move_email(identifier, folder)
            opened = service.open_email(identifier)
            assert opened["email"]["folder"] == folder
            assert opened["analysis"]["result"] == original
            assert opened["email"]["body"].strip() == body
        service.toggle_star(identifier)
        service.mark_read(identifier, False)
    with MailService.from_database_path(database, analyzer=analyze) as service:
        restored = service.open_email(identifier, mark_as_read=False)
        assert restored["email"]["folder"] == "da_xoa"
        assert restored["email"]["is_starred"] is True
        assert restored["email"]["is_read"] is False
        assert restored["analysis"]["result"] == original
        assert restored["folder_history"]
        assert len(calls) == 1
    connection = open_initialized_database(database)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM phan_hoi_nguoi_dung").fetchone()[0] >= 3
    finally:
        connection.close()


def test_real_analyze_save_mock_review_and_reopen_preserves_exact_ml(
    real_engine, tmp_path: Path,
) -> None:
    fields = {"sender": "sender@example.com", "subject": "Synthetic account notice",
              "body": "Please verify your password at http://192.0.2.10/login immediately."}
    result = real_engine.analyze_email(**fields)
    before = deepcopy(result)
    other_label = "normal" if result["prediction"] != "normal" else "phishing"
    mock_review = review_email(**fields, ml_result=result, provider=MockReviewerProvider({
        "assessment": other_label, "summary": "Invented offline reviewer disagreement.",
        "evidence": [{"field": "subject", "quote": fields["subject"],
                      "explanation": "Literal fixture subject; not real semantic evidence."}],
        "suspicious_indicators": [],
    }))

    def never_reanalyze(**_fields):
        pytest.fail("Saving or opening an already analyzed email must not rerun ML")

    database = tmp_path / "saved-analysis.db"
    with MailService.from_database_path(database, analyzer=never_reanalyze) as service:
        saved = service.save_analyzed_email(**fields, analysis_result=result, llm_review=mock_review)
        identifier = saved["email"]["id"]
        assert service.save_analyzed_email(**fields, analysis_result=result)["duplicate"] is True
        opened = service.open_email(identifier)
        assert opened["analysis"]["result"] == before == result
        assert opened["llm_review"]["result"]["ml_result"] == before
        assert opened["llm_review"]["result"]["reviewer"]["recommendation"] == "REVIEW"
        assert opened["llm_review"]["result"]["reviewer"]["calibrated_probability"] is None
        service.record_feedback(identifier, action="chuyen_vao_hop_thu_den")
        assert service.open_email(identifier)["analysis"]["result"] == before
    with MailService.from_database_path(database, analyzer=never_reanalyze) as service:
        opened = service.open_email(identifier)
        assert opened["analysis"]["result"] == before
        assert opened["llm_review"]["result"]["reviewer"]["assessment"] == other_label


def test_real_ml_works_without_key_and_external_reviewer_requires_consent(
    real_engine, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    fields = {"sender": "sender@example.com", "subject": "Synthetic meeting",
              "body": "Please attend tomorrow's project meeting."}
    result = real_engine.analyze_email(**fields)

    class NeverExternal(MockReviewerProvider):
        is_external = True

        def review(self, *_args, **_kwargs):
            pytest.fail("External provider cannot execute without both explicit consent flags")

    for activation, consent in ((False, False), (True, False), (False, True)):
        with pytest.raises(ReviewerPrivacyError):
            review_email(**fields, ml_result=result, provider=NeverExternal(),
                         allow_external=activation, explicit_email_sharing_consent=consent)
    failed = review_email(**fields, ml_result=result, provider=MockReviewerProvider("not JSON"))
    assert failed["reviewer"]["status"] == "invalid_response"
    assert failed["ml_result"] == result
    assert failed["reviewer"]["assessment"] is None


def test_missing_local_model_fails_closed_without_falling_back_to_v2(tmp_path: Path) -> None:
    with pytest.raises(V21ModelUnavailable):
        load_v21_engine(tmp_path / "missing.json", tmp_path)


def test_real_candidate_rejects_empty_input_before_prediction(real_engine) -> None:
    with pytest.raises(ValueError):
        real_engine.analyze_email(subject=" \n", body="\t")
