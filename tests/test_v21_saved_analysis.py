"""Save trusted analysis snapshots without running inference or external APIs."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from vietmailguard.mail_service import MailService
from vietmailguard.v21.llm_reviewer import (
    MockReviewerProvider, ProviderTimeoutError, ReviewerPrivacyError, review_email,
)


def analysis(prediction: str = "normal") -> dict:
    action = {"normal": "ALLOW", "spam": "MOVE_TO_SPAM", "phishing": "QUARANTINE"}[prediction]
    return {
        "model_version": "2.1-development", "model_name": "Synthetic test double",
        "candidate_id": "fixture-v21", "artifact_sha256": "0" * 64,
        "detected_language": "vi", "prediction": prediction,
        "confidence": None, "class_probabilities": {},
        "probability_source": "unavailable_uncalibrated",
        "risk_score": 44, "risk_level": "MEDIUM",
        "recommended_action": action, "security_findings": [],
        "limitations": ["Synthetic smoke fixture; not a benchmark."],
    }


def forbidden_analyzer(**_):
    raise AssertionError("Saving or reopening an analyzed email must not run inference")


@pytest.mark.parametrize(("prediction", "folder"), [
    ("normal", "hop_thu_den"), ("spam", "thu_rac"), ("phishing", "cach_ly"),
])
def test_save_exact_result_routes_once_and_restarts_without_inference(
    tmp_path: Path, prediction: str, folder: str,
) -> None:
    path = tmp_path / "saved.db"
    original = analysis(prediction)
    snapshot = copy.deepcopy(original)
    with MailService.from_database_path(path, analyzer=forbidden_analyzer) as service:
        saved = service.save_analyzed_email(
            sender="demo@example.com", receiver="local@example.org",
            subject="Thông báo thử nghiệm", body="Nội dung giả để kiểm thử.",
            analysis_result=original,
        )
        assert saved["created"] is True
        assert saved["email"]["folder"] == folder
        assert saved["email"]["source"] == "quick_analyzer"
        assert saved["analysis"]["result"] == snapshot
        assert saved["analysis"]["confidence"] is None
        email_id = saved["email"]["id"]
        original["prediction"] = "phishing"
    with MailService.from_database_path(path, analyzer=forbidden_analyzer) as service:
        opened = service.open_email(email_id)
        assert opened["email"]["folder"] == folder
        assert opened["analysis"]["result"] == snapshot
        assert opened["analysis"]["confidence"] is None
        assert opened["analysis"]["result"]["artifact_sha256"] == "0" * 64


def test_duplicate_save_is_idempotent_and_different_model_history_is_not_replaced(tmp_path):
    with MailService.from_database_path(tmp_path / "mail.db", analyzer=forbidden_analyzer) as service:
        fields = {"subject": "Same input", "body": "Synthetic content."}
        first = service.save_analyzed_email(**fields, analysis_result=analysis())
        repeated = service.save_analyzed_email(**fields, analysis_result=analysis())
        assert repeated["duplicate"] is True
        assert repeated["email"]["id"] == first["email"]["id"]
        assert service.mailbox_counts()["total"] == 1
        assert service.repository.connection.execute("SELECT COUNT(*) FROM phan_tich_thu").fetchone()[0] == 1
        other = analysis("spam")
        other["model_version"] = "another-explicit-development-candidate"
        second = service.save_analyzed_email(**fields, analysis_result=other)
        assert second["email"]["id"] != first["email"]["id"]
        assert service.open_email(first["email"]["id"])["analysis"]["result"] == analysis()


@pytest.mark.parametrize("change", [
    {"prediction": "ham"}, {"confidence": .8}, {"confidence": float("nan")},
    {"risk_score": float("inf")}, {"risk_score": -1}, {"risk_level": "unknown"},
    {"class_probabilities": {"normal": .8, "spam": .7, "phishing": .1}},
    {"class_probabilities": {"normal": 1}}, {"model_version": ""},
])
def test_invalid_or_fake_probability_result_is_not_saved(tmp_path, change):
    with MailService.from_database_path(tmp_path / "mail.db", analyzer=forbidden_analyzer) as service:
        result = {**analysis(), **change}
        with pytest.raises(ValueError):
            service.save_analyzed_email(subject="Fixture", body="Synthetic text", analysis_result=result)
        assert service.mailbox_counts()["total"] == 0


def test_empty_or_incomplete_result_cannot_create_mail(tmp_path):
    with MailService.from_database_path(tmp_path / "mail.db", analyzer=forbidden_analyzer) as service:
        for fields, result in (({"body": " "}, analysis()), ({"body": "body"}, {})):
            with pytest.raises(ValueError):
                service.save_analyzed_email(**fields, analysis_result=result)
        assert service.mailbox_counts()["total"] == 0


def test_matching_mock_review_is_saved_separately_and_repeated_save_does_not_duplicate(tmp_path):
    result = analysis("spam")
    fields = {"sender": "demo@example.com", "subject": "Fixture", "body": "Synthetic test text"}
    review = review_email(**fields, ml_result=result, provider=MockReviewerProvider({
        "assessment": "phishing", "summary": "Synthetic conflicting review fixture.",
        "evidence": [{"field": "subject", "quote": "Fixture", "explanation": "Observed subject."}],
        "suspicious_indicators": [],
    }))
    with MailService.from_database_path(tmp_path / "mail.db", analyzer=forbidden_analyzer) as service:
        first = service.save_analyzed_email(**fields, analysis_result=result, llm_review=review)
        service.save_analyzed_email(**fields, analysis_result=result, llm_review=review)
        opened = service.open_email(first["email"]["id"])
        assert opened["email"]["folder"] == "thu_rac"
        assert opened["analysis"]["result"] == result
        assert opened["llm_review"]["result"] == review
        assert opened["llm_review"]["result"]["reviewer"]["recommendation"] == "REVIEW"
        assert service.repository.connection.execute("SELECT COUNT(*) FROM danh_gia_llm").fetchone()[0] == 1


def test_mismatched_or_fabricated_review_is_rejected_before_persistence(tmp_path):
    fields = {"sender": "demo@example.com", "subject": "Fixture", "body": "Synthetic text"}
    review = review_email(**fields, ml_result=analysis(), provider=MockReviewerProvider())
    with MailService.from_database_path(tmp_path / "mail.db", analyzer=forbidden_analyzer) as service:
        with pytest.raises(ValueError, match="does not match"):
            service.save_analyzed_email(**fields, analysis_result=analysis("spam"), llm_review=review)
        review["reviewer"]["calibrated_probability"] = .9
        with pytest.raises(ValueError, match="Invalid reviewer"):
            service.save_analyzed_email(**fields, analysis_result=analysis(), llm_review=review)
        assert service.mailbox_counts()["total"] == 0


@pytest.mark.parametrize("response", ["not valid JSON", ProviderTimeoutError("do not expose this")])
def test_stored_review_failure_is_safe_and_does_not_touch_routing_or_analysis(tmp_path, response):
    class FailingLocalProvider:
        name = "offline-failure-fixture"
        is_external = False

        def review(self, request, *, timeout_seconds):
            if isinstance(response, Exception):
                raise response
            return response

    with MailService.from_database_path(tmp_path / "mail.db", analyzer=forbidden_analyzer) as service:
        saved = service.save_analyzed_email(subject="Fixture", body="Synthetic text", analysis_result=analysis("spam"))
        email_id = saved["email"]["id"]
        service.review_stored_email(email_id, provider=FailingLocalProvider())
        opened = service.open_email(email_id)
        assert opened["analysis"]["result"] == analysis("spam")
        assert opened["email"]["folder"] == "thu_rac"
        assert opened["llm_review"]["result"]["reviewer"]["status"] in {"invalid_response", "provider_error"}
        assert "do not expose this" not in str(opened["llm_review"])


def test_external_stored_review_without_consent_never_invokes_provider(tmp_path):
    class NeverCalled:
        name = "external-fixture"
        is_external = True

        def review(self, *_args, **_kwargs):
            raise AssertionError("External request forbidden without consent")

    with MailService.from_database_path(tmp_path / "mail.db", analyzer=forbidden_analyzer) as service:
        saved = service.save_analyzed_email(subject="Fixture", body="Synthetic text", analysis_result=analysis())
        with pytest.raises(ReviewerPrivacyError):
            service.review_stored_email(saved["email"]["id"], provider=NeverCalled(), allow_external=True)
        assert service.open_email(saved["email"]["id"])["llm_review"] is None
