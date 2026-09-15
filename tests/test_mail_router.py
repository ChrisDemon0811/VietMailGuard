from __future__ import annotations

from typing import Any

import pytest

from vietmailguard.mail_router import route_analysis


ROUTING_DECISION_MATRIX: tuple[
    tuple[str, dict[str, Any], str, str, bool, str], ...
] = (
    (
        "normal_low_clean",
        {"prediction": "normal", "risk_level": "LOW"},
        "hop_thu_den",
        "ALLOW",
        False,
        "NONE",
    ),
    (
        "normal_promotional_review",
        {
            "prediction": "normal",
            "risk_level": "LOW",
            "recommended_action": "REVIEW",
            "security_findings": [{"source": "content", "code": "commercial_promotion"}],
        },
        "hop_thu_den",
        "ALLOW",
        True,
        "MEDIUM",
    ),
    (
        "normal_promotional_signal_only",
        {
            "prediction": "normal",
            "risk_level": "LOW",
            "security_findings": [{"source": "content", "code": "commercial_promotion"}],
        },
        "hop_thu_den",
        "ALLOW",
        True,
        "LOW",
    ),
    (
        "normal_medium_without_indicator",
        {"prediction": "normal", "risk_level": "MEDIUM"},
        "hop_thu_den",
        "ALLOW",
        False,
        "NONE",
    ),
    (
        "normal_high_risk",
        {"prediction": "normal", "risk_level": "HIGH"},
        "hop_thu_den",
        "ALLOW",
        True,
        "HIGH",
    ),
    (
        "normal_critical_risk",
        {"prediction": "normal", "risk_level": "CRITICAL"},
        "hop_thu_den",
        "ALLOW",
        True,
        "CRITICAL",
    ),
    (
        "normal_suspicious_url",
        {
            "prediction": "normal",
            "risk_level": "LOW",
            "url_findings": [{"url": "http://192.0.2.1", "findings": [{"code": "ip_host"}]}],
        },
        "hop_thu_den",
        "ALLOW",
        True,
        "MEDIUM",
    ),
    (
        "normal_sender_indicator",
        {
            "prediction": "normal",
            "risk_level": "LOW",
            "sender_findings": [{"code": "display_domain_mismatch"}],
        },
        "hop_thu_den",
        "ALLOW",
        True,
        "MEDIUM",
    ),
    (
        "spam_low_clean",
        {"prediction": "spam", "risk_level": "LOW"},
        "thu_rac",
        "MOVE_TO_SPAM",
        False,
        "NONE",
    ),
    (
        "spam_phishing_like",
        {
            "prediction": "spam",
            "risk_level": "MEDIUM",
            "recommended_action": "REVIEW",
            "security_findings": [{"source": "content", "code": "credential_request"}],
        },
        "thu_rac",
        "MOVE_TO_SPAM",
        True,
        "HIGH",
    ),
    (
        "spam_critical",
        {"prediction": "spam", "risk_level": "CRITICAL"},
        "thu_rac",
        "MOVE_TO_SPAM",
        True,
        "CRITICAL",
    ),
    (
        "phishing_low",
        {"prediction": "phishing", "risk_level": "LOW"},
        "cach_ly",
        "QUARANTINE",
        True,
        "HIGH",
    ),
    (
        "phishing_critical",
        {
            "prediction": "phishing",
            "risk_level": "CRITICAL",
            "recommended_action": "QUARANTINE",
        },
        "cach_ly",
        "QUARANTINE",
        True,
        "CRITICAL",
    ),
)


@pytest.mark.parametrize(
    ("case_name", "analysis", "folder", "base_action", "has_warning", "severity"),
    ROUTING_DECISION_MATRIX,
    ids=[case[0] for case in ROUTING_DECISION_MATRIX],
)
def test_complete_routing_decision_matrix(
    case_name: str,
    analysis: dict[str, Any],
    folder: str,
    base_action: str,
    has_warning: bool,
    severity: str,
) -> None:
    del case_name
    decision = route_analysis(analysis)
    assert decision.prediction == analysis["prediction"]
    assert decision.folder == folder
    assert decision.base_action == base_action
    assert decision.has_warning is has_warning
    assert decision.warning_severity == severity


def test_recommendation_never_changes_folder_or_prediction() -> None:
    decision = route_analysis(
        {
            "prediction": "normal",
            "risk_level": "LOW",
            "recommended_action": "QUARANTINE",
        }
    )
    assert decision.prediction == "normal"
    assert decision.folder == "hop_thu_den"
    assert decision.base_action == "ALLOW"
    assert decision.recommended_action == "QUARANTINE"
    assert "ml_security_disagreement" in decision.warning_reasons


def test_invalid_prediction_and_risk_are_rejected() -> None:
    with pytest.raises(ValueError, match="prediction"):
        route_analysis({"prediction": "review", "risk_level": "LOW"})
    with pytest.raises(ValueError, match="risk level"):
        route_analysis({"prediction": "normal", "risk_level": "EXTREME"})
