"""Development-mode integration without touching frozen test data or paid APIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vietmailguard.mail_database import (
    _CREATE_VERSION_TABLE, _VERSION_ONE_STATEMENTS, open_connection,
    open_initialized_database,
)
from vietmailguard.mail_repository import MailRepository
from vietmailguard.mail_service import MailService
from vietmailguard.v21.dashboard_artifacts import (
    load_development_dataset, load_development_validation,
)
from vietmailguard.v21.llm_reviewer import MockReviewerProvider, ReviewerPrivacyError
from vietmailguard.v21.runtime_inference import (
    PROJECT_ROOT, V21ModelUnavailable, _verified_paths, analyze_v21_email,
    model_availability,
)
from vietmailguard.v21.runtime_modes import V2_MODE, V21_ML_MODE, analyzer_for_mode


def _analysis(prediction: str = "spam", *, confidence: float | None = None) -> dict:
    action = {"normal": "ALLOW", "spam": "MOVE_TO_SPAM", "phishing": "QUARANTINE"}[prediction]
    return {
        "model_version": "2.1-development", "detected_language": "zh",
        "prediction": prediction, "confidence": confidence,
        "class_probabilities": {}, "risk_score": 47, "risk_level": "MEDIUM",
        "base_recommended_action": action, "recommended_action": action,
        "security_findings": [],
    }


def test_v21_artifact_is_not_enabled_when_hash_or_manifest_invalid(tmp_path: Path) -> None:
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({
        "schema_version": 1, "scope": "development_candidate_only",
        "probability_status": "unavailable_uncalibrated",
        "supported_classes": ["normal", "spam", "phishing"],
        "artifact": "models/v21_multilingual/candidates/missing.joblib",
        "artifact_sha256": "0" * 64,
        "validation_metrics": "results/v21_multilingual/missing.json",
        "validation_metrics_sha256": "0" * 64,
    }), encoding="utf-8")
    with pytest.raises(V21ModelUnavailable):
        _verified_paths(config, tmp_path)


def test_v21_saved_development_artifacts_are_distinct_from_final_test() -> None:
    dataset = load_development_dataset(PROJECT_ROOT)
    validation = load_development_validation(PROJECT_ROOT)
    if dataset is None or validation is None:
        pytest.skip("V2.1 development artifacts not present in source-only clone")
    assert dataset["final_evaluation_ready"] is False
    assert dataset["final_test_path"] is None
    assert validation["scope"] == "development_validation_only"
    assert dataset["output_rows"]["validation"] == validation["validation_rows"]


def test_v21_candidate_is_uncalibrated_if_local_artifact_present() -> None:
    available, _ = model_availability()
    if not available:
        pytest.skip("V2.1 candidate binary not present in source-only clone")
    result = analyze_v21_email(subject="账户通知", body="请查看账户通知。")
    assert result["model_version"] == "2.1-development"
    assert result["confidence"] is None
    assert result["class_probabilities"] == {}
    assert result["probability_source"] == "unavailable_uncalibrated"
    assert result["detected_language"] == "zh"
    assert result["language_support_status"]["status"] == "unvalidated"
    assert result["prediction"] in {"normal", "spam", "phishing"}


def test_v2_mode_still_points_to_frozen_inference_entry() -> None:
    from vietmailguard.inference import analyze_email
    assert analyzer_for_mode(V2_MODE) is analyze_email


def test_verified_v21_mailbox_receive_and_open_uses_stored_analysis(tmp_path: Path) -> None:
    if not model_availability()[0]:
        pytest.skip("V2.1 candidate binary not present in source-only clone")
    analyzer = analyzer_for_mode(V21_ML_MODE)
    calls = []
    def counted(**fields):
        calls.append(fields)
        return analyzer(**fields)
    service = MailService.from_database_path(tmp_path / "v21-mail.db", analyzer=counted)
    try:
        received = service.receive_email(sender="local@example.com", subject="会议信息",
                                         body="明天三点见。")
        email_id = received["email"]["id"]
        opened = service.open_email(email_id)
        assert len(calls) == 1
        assert opened["analysis"]["model_version"] == "2.1-development"
        assert opened["analysis"]["confidence"] is None
        assert opened["analysis"]["detected_language"] == "zh"
        assert opened["email"]["folder"] in {"hop_thu_den", "thu_rac", "cach_ly"}
    finally:
        service.close()


def test_v1_database_migrates_without_losing_mail_or_analysis(tmp_path: Path) -> None:
    db_path = tmp_path / "v1.db"
    old = open_connection(db_path)
    old.execute(_CREATE_VERSION_TABLE)
    for statement in _VERSION_ONE_STATEMENTS:
        old.execute(statement)
    old.execute("INSERT INTO phien_ban_co_so_du_lieu VALUES (1, '2026-01-01T00:00:00Z')")
    old.execute("INSERT INTO thu(id_thu, tieu_de, noi_dung) VALUES (7, 'old', 'body')")
    old.execute("""INSERT INTO phan_tich_thu(
        id_thu, phien_ban_mo_hinh, ngon_ngu_phat_hien, nhan_du_doan,
        do_tin_cay, diem_rui_ro, muc_rui_ro, hanh_dong_goc,
        hanh_dong_de_xuat, ket_qua_json)
        VALUES (7, '2.0.0', 'en', 'normal', .8, 2, 'LOW', 'ALLOW', 'ALLOW', '{}')""")
    old.close()
    migrated = open_initialized_database(db_path)
    try:
        assert migrated.execute("SELECT tieu_de FROM thu WHERE id_thu=7").fetchone()[0] == "old"
        assert migrated.execute("SELECT do_tin_cay FROM phan_tich_thu WHERE id_thu=7").fetchone()[0] == .8
        assert migrated.execute("SELECT MAX(phien_ban) FROM phien_ban_co_so_du_lieu").fetchone()[0] == 2
        assert migrated.execute("SELECT COUNT(*) FROM danh_gia_llm").fetchone()[0] == 0
    finally:
        migrated.close()


def test_review_is_opt_in_persisted_separately_and_never_reroutes(tmp_path: Path) -> None:
    calls = []
    def analyzer(**fields):
        calls.append(fields)
        return _analysis()
    service = MailService.from_database_path(tmp_path / "runtime.db", analyzer=analyzer)
    try:
        received = service.receive_email(sender="sender@example.com", subject="Offer", body="Buy this")
        email_id = received["email"]["id"]
        before = service.open_email(email_id)
        assert before["llm_review"] is None
        assert len(calls) == 1
        response = {"assessment": "phishing", "summary": "Needs investigation.",
                    "evidence": [{"field": "subject", "quote": "Offer", "explanation": "Observed subject."}],
                    "suspicious_indicators": []}
        provider = MockReviewerProvider(response)
        service.review_stored_email(email_id, provider=provider)
        after = service.open_email(email_id)
        assert len(calls) == 1
        assert after["email"]["folder"] == "thu_rac"
        assert after["analysis"]["prediction"] == "spam"
        assert after["analysis"]["confidence"] is None
        assert after["llm_review"]["result"]["reviewer"]["assessment"] == "phishing"
        assert after["llm_review"]["result"]["reviewer"]["recommendation"] == "REVIEW"
        assert len(after["folder_history"]) == len(before["folder_history"])
    finally:
        service.close()


def test_external_review_requires_both_activation_and_consent(tmp_path: Path) -> None:
    class NeverCalledProvider(MockReviewerProvider):
        is_external = True
        def review(self, *args, **kwargs):
            raise AssertionError("Network provider must not be called")
    service = MailService.from_database_path(tmp_path / "privacy.db", analyzer=lambda **_: _analysis("normal"))
    try:
        email_id = service.receive_email(subject="hello", body="body")["email"]["id"]
        with pytest.raises(ReviewerPrivacyError):
            service.review_stored_email(email_id, provider=NeverCalledProvider(),
                                        allow_external=True,
                                        explicit_email_sharing_consent=False)
        assert service.open_email(email_id)["llm_review"] is None
    finally:
        service.close()
