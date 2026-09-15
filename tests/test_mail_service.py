from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vietmailguard.mail_database import open_initialized_database
from vietmailguard.mail_repository import MailRepository
from vietmailguard.mail_router import route_analysis
from vietmailguard.mail_service import MailService


def _analysis(
    prediction: str,
    *,
    recommended_action: str | None = None,
    findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base_action = {
        "normal": "ALLOW",
        "spam": "MOVE_TO_SPAM",
        "phishing": "QUARANTINE",
    }[prediction]
    return {
        "model_version": "2.0.0-test-double",
        "detected_language": "en",
        "prediction": prediction,
        "confidence": 0.88,
        "risk_score": 72 if prediction == "phishing" else 22,
        "risk_level": "HIGH" if prediction == "phishing" else "LOW",
        "base_recommended_action": base_action,
        "recommended_action": recommended_action or base_action,
        "security_findings": findings or [],
    }


def _analyzer(result: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    def analyze(**_: object) -> dict[str, Any]:
        return dict(result)

    return analyze


@pytest.fixture
def service_factory(tmp_path: Path):
    connections = []

    def create(result: dict[str, Any]) -> MailService:
        connection = open_initialized_database(tmp_path / f"mail-{len(connections)}.db")
        connections.append(connection)
        return MailService(MailRepository(connection), analyzer=_analyzer(result))

    yield create
    for connection in connections:
        connection.close()


@pytest.mark.parametrize(
    ("prediction", "expected_folder", "expected_reason"),
    [
        ("normal", "hop_thu_den", "tu_dong_normal"),
        ("spam", "thu_rac", "tu_dong_spam"),
        ("phishing", "cach_ly", "tu_dong_phishing"),
    ],
)
def test_automatic_routing_by_ml_prediction(
    service_factory,
    prediction: str,
    expected_folder: str,
    expected_reason: str,
) -> None:
    service = service_factory(_analysis(prediction))
    received = service.receive_email(subject="Test", body="Message body")

    assert received["email"]["folder"] == expected_folder
    assert received["analysis"]["prediction"] == prediction
    assert received["routing"]["reason"] == expected_reason


def test_normal_review_stays_in_inbox_with_warning(service_factory) -> None:
    service = service_factory(
        _analysis(
            "normal",
            recommended_action="REVIEW",
            findings=[{"code": "commercial_promotion"}],
        )
    )
    received = service.receive_email(subject="Sale", body="Buy now")

    assert received["email"]["folder"] == "hop_thu_den"
    assert received["analysis"]["prediction"] == "normal"
    assert received["analysis"]["recommended_action"] == "REVIEW"
    assert received["analysis"]["has_warning"] is True


def test_spam_warning_does_not_change_spam_folder(service_factory) -> None:
    service = service_factory(
        _analysis(
            "spam",
            recommended_action="REVIEW",
            findings=[{"code": "credential_request"}],
        )
    )
    received = service.receive_email(subject="Review", body="Message")

    assert received["email"]["folder"] == "thu_rac"
    assert received["analysis"]["prediction"] == "spam"
    assert received["analysis"]["has_warning"] is True


def test_user_restores_spam_and_prediction_is_unchanged(service_factory) -> None:
    service = service_factory(_analysis("spam"))
    received = service.receive_email(subject="Offer", body="Commercial message")
    email_id = received["email"]["id"]

    result = service.record_feedback(email_id, action="khong_phai_thu_rac")

    assert result["email"]["folder"] == "hop_thu_den"
    assert result["analysis"]["prediction"] == "spam"
    feedback = service.repository.connection.execute(
        """
        SELECT nhan_du_doan_ban_dau, thu_muc_truoc, thu_muc_sau,
               hanh_dong_nguoi_dung
        FROM phan_hoi_nguoi_dung
        WHERE id_phan_hoi = ?
        """,
        (result["feedback_id"],),
    ).fetchone()
    assert tuple(feedback) == (
        "spam",
        "thu_rac",
        "hop_thu_den",
        "khong_phai_thu_rac",
    )

    history = service.repository.list_folder_history(email_id)
    assert [(item["new_folder"], item["reason"]) for item in history] == [
        ("thu_rac", "tu_dong_spam"),
        ("hop_thu_den", "khong_phai_thu_rac"),
    ]


def test_router_ignores_security_action_for_destination() -> None:
    decision = route_analysis(
        {
            "prediction": "normal",
            "recommended_action": "QUARANTINE",
            "security_findings": [{"code": "suspicious_indicator"}],
        }
    )
    assert decision.folder == "hop_thu_den"
    assert decision.prediction == "normal"
    assert decision.has_warning is True


def test_import_email_reuses_existing_parser(service_factory) -> None:
    service = service_factory(_analysis("normal"))
    raw = (
        b"From: Alice <alice@example.com>\r\n"
        b"To: Bob <bob@example.org>\r\n"
        b"Subject: Imported message\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"The imported body.\r\n"
    )
    result = service.import_email(raw, source_id="fixture-eml")

    assert result["email"]["sender"] == "Alice <alice@example.com>"
    assert result["email"]["subject"] == "Imported message"
    assert result["email"]["source"] == "eml_import"
    assert result["email"]["source_id"] == "fixture-eml"


def test_open_read_star_list_and_search_service_methods(service_factory) -> None:
    service = service_factory(_analysis("normal"))
    received = service.receive_email(
        sender="team@example.com",
        subject="Project Delta",
        body="Meeting agenda",
    )
    email_id = received["email"]["id"]

    assert service.open_email(email_id)["email"]["is_read"] is True
    assert service.toggle_star(email_id)["is_starred"] is True
    assert [item["id"] for item in service.list_folder("hop_thu_den", starred=True)] == [
        email_id
    ]
    assert [item["id"] for item in service.search_email("Delta")] == [email_id]


def test_open_email_uses_exact_stored_analysis_without_reinference(tmp_path: Path) -> None:
    calls = 0
    exact_result = _analysis(
        "normal",
        recommended_action="REVIEW",
        findings=[{"source": "content", "code": "commercial_promotion"}],
    )
    exact_result["limitations"] = ["stored fixture limitation"]
    exact_result["nested_evidence"] = {"values": [1, 2, 3]}

    def counting_analyzer(**_: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return exact_result

    connection = open_initialized_database(tmp_path / "stored-analysis.db")
    service = MailService(MailRepository(connection), analyzer=counting_analyzer)
    try:
        received = service.receive_email(subject="Offer", body="Promotional message")
        listed = service.list_folder("hop_thu_den")
        counts = service.mailbox_counts()
        security_rows = service.list_security_view()
        first_open = service.open_email(received["email"]["id"])
        second_open = service.open_email(received["email"]["id"])
    finally:
        connection.close()

    assert calls == 1
    assert listed[0]["analysis"]["prediction"] == "normal"
    assert counts["folders"]["hop_thu_den"] == 1
    assert security_rows[0]["id"] == received["email"]["id"]
    assert received["analysis"]["result"] == exact_result
    assert first_open["analysis"]["result"] == exact_result
    assert second_open["analysis"]["result"] == exact_result
    assert first_open["routing"]["warning_severity"] == "MEDIUM"


def test_phishing_base_action_is_canonical_while_json_stays_exact(tmp_path: Path) -> None:
    exact_result = _analysis("phishing")
    exact_result["base_recommended_action"] = "ALLOW"
    connection = open_initialized_database(tmp_path / "phishing-contract.db")
    service = MailService(MailRepository(connection), analyzer=_analyzer(exact_result))
    try:
        received = service.receive_email(subject="Alert", body="Verify credentials")
    finally:
        connection.close()

    assert received["email"]["folder"] == "cach_ly"
    assert received["analysis"]["prediction"] == "phishing"
    assert received["analysis"]["base_recommended_action"] == "QUARANTINE"
    assert received["analysis"]["result"] == exact_result


def test_detail_state_actions_are_audited_without_mutating_analysis(
    service_factory,
) -> None:
    service = service_factory(_analysis("spam"))
    received = service.receive_email(subject="Offer", body="Commercial message")
    email_id = received["email"]["id"]
    original_analysis = service.repository.get_latest_analysis(email_id)
    immutable_values = (
        original_analysis["prediction"],
        original_analysis["confidence"],
        original_analysis["risk_score"],
    )

    service.open_email(email_id)
    service.mark_read(email_id, False)
    service.toggle_star(email_id)
    service.toggle_star(email_id)
    service.move_email(email_id, "hop_thu_den")
    service.move_email(email_id, "cach_ly")
    service.move_email(email_id, "da_xoa")

    action_rows = service.repository.connection.execute(
        """
        SELECT hanh_dong_nguoi_dung
        FROM phan_hoi_nguoi_dung
        WHERE id_thu = ?
        ORDER BY id_phan_hoi
        """,
        (email_id,),
    ).fetchall()
    assert [row[0] for row in action_rows] == [
        "danh_dau_da_doc",
        "danh_dau_chua_doc",
        "gan_sao",
        "bo_gan_sao",
        "chuyen_vao_hop_thu_den",
        "chuyen_vao_cach_ly",
        "xoa",
    ]
    history = service.repository.list_folder_history(email_id)
    assert [event["new_folder"] for event in history] == [
        "thu_rac",
        "hop_thu_den",
        "cach_ly",
        "da_xoa",
    ]
    unchanged = service.repository.get_latest_analysis(email_id)
    assert (
        unchanged["prediction"],
        unchanged["confidence"],
        unchanged["risk_score"],
    ) == immutable_values


def test_feedback_corrections_preserve_prediction_confidence_and_risk(
    service_factory,
) -> None:
    service = service_factory(_analysis("normal", recommended_action="REVIEW"))
    received = service.receive_email(subject="Review", body="Promotional wording")
    email_id = received["email"]["id"]
    before = service.repository.get_latest_analysis(email_id)

    service.record_feedback(email_id, action="danh_dau_thu_rac")
    service.record_feedback(email_id, action="khong_phai_thu_rac")
    service.record_feedback(email_id, action="bao_cao_lua_dao")

    after = service.repository.get_latest_analysis(email_id)
    assert after["prediction"] == before["prediction"] == "normal"
    assert after["confidence"] == before["confidence"]
    assert after["risk_score"] == before["risk_score"]
    assert service.repository.get_email(email_id)["folder"] == "cach_ly"


def test_security_view_uses_stored_analysis_filters_and_risk_order(
    service_factory,
) -> None:
    normal_service = service_factory(_analysis("normal"))
    low = normal_service.receive_email(
        sender="team@example.org",
        subject="Low risk update",
        body="Project status",
    )
    normal_service._analyzer = _analyzer(_analysis("phishing"))
    high = normal_service.receive_email(
        sender="security@example.org",
        subject="Critical account alert",
        body="Verify credentials",
    )

    rows = normal_service.list_security_view()
    assert [row["id"] for row in rows] == [high["email"]["id"], low["email"]["id"]]
    filtered = normal_service.list_security_view(
        query="account",
        folder="cach_ly",
        prediction="phishing",
        risk_level="HIGH",
        language="en",
    )
    assert [row["id"] for row in filtered] == [high["email"]["id"]]


def test_restart_preserves_complete_mailbox_state_without_reanalysis(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "restart-complete.db"
    with MailService.from_database_path(
        database_path,
        analyzer=_analyzer(_analysis("spam")),
    ) as first:
        received = first.receive_email(
            sender="newsletter@example.org",
            subject="Stored message",
            body="This state must survive restart.",
        )
        email_id = int(received["email"]["id"])
        first.open_email(email_id)
        first.toggle_star(email_id)
        first.record_feedback(email_id, action="khong_phai_thu_rac")

    def forbidden_analyzer(**_: object) -> dict[str, Any]:
        raise AssertionError("Opening persisted mail must not run inference")

    with MailService.from_database_path(
        database_path,
        analyzer=forbidden_analyzer,
    ) as second:
        reopened = second.open_email(email_id)
        feedback = second.repository.connection.execute(
            """
            SELECT hanh_dong_nguoi_dung
            FROM phan_hoi_nguoi_dung
            WHERE id_thu = ?
            ORDER BY id_phan_hoi
            """,
            (email_id,),
        ).fetchall()

    assert reopened["email"]["folder"] == "hop_thu_den"
    assert reopened["email"]["is_read"] is True
    assert reopened["email"]["is_starred"] is True
    assert reopened["analysis"]["prediction"] == "spam"
    assert reopened["analysis"]["confidence"] == pytest.approx(0.88)
    assert reopened["analysis"]["risk_score"] == pytest.approx(22)
    assert [row[0] for row in feedback] == [
        "danh_dau_da_doc",
        "gan_sao",
        "khong_phai_thu_rac",
    ]
    assert [event["new_folder"] for event in reopened["folder_history"]] == [
        "thu_rac",
        "hop_thu_den",
    ]
