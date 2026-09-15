from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from vietmailguard.mail_database import open_initialized_database
from vietmailguard.mail_repository import MailNotFoundError, MailRepository


@pytest.fixture
def repository(tmp_path: Path) -> MailRepository:
    connection = open_initialized_database(tmp_path / "repository.db")
    repo = MailRepository(connection)
    try:
        yield repo
    finally:
        connection.close()


def _create_email(repository: MailRepository, *, subject: str = "Status update") -> int:
    return repository.create_email(
        sender="alice@example.com",
        receiver="bob@example.org",
        subject=subject,
        body="The project meeting starts at 3 PM.",
        sent_at="2026-09-15T08:00:00Z",
        source="test_fixture",
        source_id="mail-001",
    )


def _save_analysis(
    repository: MailRepository,
    email_id: int,
    *,
    prediction: str = "normal",
    risk_score: float = 8,
    risk_level: str = "LOW",
    language: str = "en",
) -> int:
    action = {
        "normal": "ALLOW",
        "spam": "MOVE_TO_SPAM",
        "phishing": "QUARANTINE",
    }[prediction]
    return repository.save_analysis(
        email_id,
        model_version="2.0.0",
        detected_language=language,
        prediction=prediction,
        confidence=0.91,
        risk_score=risk_score,
        risk_level=risk_level,
        base_recommended_action=action,
        recommended_action=action,
        has_warning=prediction == "phishing",
        security_finding_count=0,
        result={"prediction": prediction, "confidence": 0.91},
    )


def test_repository_crud_flags_move_and_history(repository: MailRepository) -> None:
    email_id = _create_email(repository)

    assert repository.get_email(email_id)["folder"] == "hop_thu_den"
    assert repository.get_email_by_external_id("missing") is None
    assert [row["id"] for row in repository.list_emails(folder="hop_thu_den")] == [
        email_id
    ]
    assert repository.update_read_status(email_id, True)["is_read"] is True
    assert repository.update_starred_status(email_id, True)["is_starred"] is True

    move = repository.move_email(email_id, "thu_rac", reason="nguoi_dung_chuyen")
    assert move["moved"] is True
    assert move["previous_folder"] == "hop_thu_den"
    assert move["new_folder"] == "thu_rac"
    assert repository.list_folder_history(email_id)[0]["reason"] == "nguoi_dung_chuyen"


def test_latest_analysis_is_returned_without_recomputation(
    repository: MailRepository,
) -> None:
    email_id = _create_email(repository)
    first_id = _save_analysis(repository, email_id)
    second_id = _save_analysis(
        repository,
        email_id,
        prediction="phishing",
        risk_score=92,
        risk_level="CRITICAL",
    )

    latest = repository.get_latest_analysis(email_id)
    assert first_id < second_id
    assert latest["id"] == second_id
    assert latest["prediction"] == "phishing"
    assert latest["result"]["prediction"] == "phishing"


def test_feedback_insert_does_not_change_analysis(repository: MailRepository) -> None:
    email_id = _create_email(repository)
    _save_analysis(repository, email_id, prediction="spam")
    feedback_id = repository.save_user_feedback(
        email_id,
        initial_prediction="spam",
        previous_folder="thu_rac",
        new_folder="hop_thu_den",
        action="khong_phai_thu_rac",
        note="Legitimate newsletter",
    )

    assert feedback_id > 0
    assert repository.get_latest_analysis(email_id)["prediction"] == "spam"
    stored = repository.connection.execute(
        "SELECT hanh_dong_nguoi_dung FROM phan_hoi_nguoi_dung WHERE id_phan_hoi = ?",
        (feedback_id,),
    ).fetchone()
    assert stored[0] == "khong_phai_thu_rac"


def test_search_matches_sender_subject_body_and_escapes_wildcards(
    repository: MailRepository,
) -> None:
    first_id = _create_email(repository, subject="Quarterly 50% summary")
    second_id = repository.create_email(
        sender="security@example.org",
        subject="Account notice",
        body="Review the security policy.",
    )
    _save_analysis(repository, first_id, prediction="normal")
    _save_analysis(
        repository,
        second_id,
        prediction="phishing",
        risk_score=90,
        risk_level="CRITICAL",
    )

    assert [row["id"] for row in repository.search("security@example.org")] == [second_id]
    assert [row["id"] for row in repository.search("50%")] == [first_id]
    assert [
        row["id"]
        for row in repository.search("Account", prediction="phishing", risk_level="CRITICAL")
    ] == [second_id]
    assert repository.search("' OR 1=1 --") == []


def test_search_supports_empty_query_and_all_mailbox_filters(
    repository: MailRepository,
) -> None:
    english_id = repository.create_email(
        sender="alerts@example.org",
        subject="Security [notice]",
        body="Literal underscore_value and 100% complete.",
        folder="cach_ly",
        is_read=False,
        is_starred=True,
    )
    vietnamese_id = repository.create_email(
        sender="banbe@example.vn",
        subject="Lịch họp",
        body="Nội dung bình thường",
        folder="hop_thu_den",
        is_read=True,
        is_starred=False,
    )
    _save_analysis(
        repository,
        english_id,
        prediction="phishing",
        risk_score=93,
        risk_level="CRITICAL",
    )
    _save_analysis(
        repository,
        vietnamese_id,
        prediction="normal",
        risk_score=7,
        risk_level="LOW",
        language="vi",
    )

    assert {row["id"] for row in repository.search("")} == {
        english_id,
        vietnamese_id,
    }
    assert [
        row["id"]
        for row in repository.search(
            "",
            folder="cach_ly",
            unread=True,
            starred=True,
            prediction="phishing",
            risk_level="CRITICAL",
            language="en",
        )
    ] == [english_id]
    assert [row["id"] for row in repository.search("underscore_value")] == [
        english_id
    ]
    assert [row["id"] for row in repository.search("100% complete")] == [english_id]
    assert repository.search("%") == [repository.get_email(english_id)]
    assert repository.search("'; DROP TABLE thu; --") == []
    assert repository.connection.execute("SELECT COUNT(*) FROM thu").fetchone()[0] == 2


def test_security_query_sorts_risk_descending_and_filters_prediction(
    repository: MailRepository,
) -> None:
    low_id = _create_email(repository, subject="Low risk")
    high_id = _create_email(repository, subject="High risk")
    medium_id = _create_email(repository, subject="Medium risk")
    _save_analysis(repository, low_id, prediction="normal", risk_score=5)
    _save_analysis(
        repository,
        high_id,
        prediction="phishing",
        risk_score=91,
        risk_level="CRITICAL",
    )
    _save_analysis(
        repository,
        medium_id,
        prediction="spam",
        risk_score=50,
        risk_level="MEDIUM",
    )

    ordered = repository.search("", require_analysis=True, order_by_risk=True)
    assert [row["id"] for row in ordered] == [high_id, medium_id, low_id]
    assert [
        row["id"]
        for row in repository.search(
            "",
            prediction="spam",
            require_analysis=True,
            order_by_risk=True,
        )
    ] == [medium_id]


def test_unknown_email_updates_raise(repository: MailRepository) -> None:
    with pytest.raises(MailNotFoundError):
        repository.update_read_status(999, True)
    with pytest.raises(MailNotFoundError):
        repository.move_email(999, "thu_rac", reason="nguoi_dung_chuyen")


def test_repository_uses_foreign_keys(repository: MailRepository) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        repository.connection.execute(
            """
            INSERT INTO lich_su_thu_muc(id_thu, thu_muc_cu, thu_muc_moi, ly_do)
            VALUES (?, ?, ?, ?)
            """,
            (999, "hop_thu_den", "thu_rac", "nguoi_dung_chuyen"),
        )
