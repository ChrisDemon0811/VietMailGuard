"""Deterministic mailbox routing that never rewrites the ML prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PREDICTION_FOLDERS = {
    "normal": "hop_thu_den",
    "spam": "thu_rac",
    "phishing": "cach_ly",
}

AUTOMATIC_ROUTING_REASONS = {
    "normal": "tu_dong_normal",
    "spam": "tu_dong_spam",
    "phishing": "tu_dong_phishing",
}

BASE_ACTIONS = {
    "normal": "ALLOW",
    "spam": "MOVE_TO_SPAM",
    "phishing": "QUARANTINE",
}

RISK_LEVEL_ORDER = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

PHISHING_LIKE_CONTENT_CODES = {
    "credential_request",
    "account_suspension",
    "financial_bait",
    "prize_winner",
    "threat",
}


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """A mailbox destination derived solely from an internal ML class."""

    prediction: str
    folder: str
    reason: str
    base_action: str
    recommended_action: str
    has_warning: bool
    warning_severity: str
    warning_reasons: tuple[str, ...]


def _mapping_items(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _maximum_severity(current: str, minimum: str) -> str:
    return minimum if RISK_LEVEL_ORDER[minimum] > RISK_LEVEL_ORDER[current] else current


def route_analysis(analysis: Mapping[str, Any]) -> RoutingDecision:
    """Return routing metadata while keeping security disagreement separate.

    The destination is based only on ``prediction``. Security findings and a
    ``REVIEW`` recommendation can set ``has_warning`` but cannot change it.
    """

    prediction = str(analysis.get("prediction", ""))
    if prediction not in PREDICTION_FOLDERS:
        raise ValueError(f"Unsupported ML prediction: {prediction!r}")

    raw_risk_level = str(analysis.get("risk_level", "LOW") or "LOW").upper()
    if raw_risk_level not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        raise ValueError(f"Unsupported risk level: {raw_risk_level!r}")

    base_action = BASE_ACTIONS[prediction]
    recommended_action = str(analysis.get("recommended_action", base_action) or base_action)
    security_findings = _mapping_items(analysis.get("security_findings", []))
    security_codes = {str(finding.get("code", "")) for finding in security_findings}
    security_sources = {str(finding.get("source", "")) for finding in security_findings}
    url_analyses = _mapping_items(analysis.get("url_findings", []))
    sender_findings = _mapping_items(analysis.get("sender_findings", []))
    has_suspicious_url = "url" in security_sources or any(
        bool(_mapping_items(url_analysis.get("findings", [])))
        for url_analysis in url_analyses
    )
    has_sender_indicator = "sender" in security_sources or bool(sender_findings)
    has_phishing_like_content = bool(security_codes & PHISHING_LIKE_CONTENT_CODES)

    warning_reasons: list[str] = []
    if prediction == "phishing":
        warning_reasons.append("ml_phishing_prediction")
    if recommended_action == "REVIEW":
        warning_reasons.append("recommendation_review")
    elif recommended_action != base_action:
        warning_reasons.append("ml_security_disagreement")
    if raw_risk_level in {"HIGH", "CRITICAL"}:
        warning_reasons.append("elevated_risk")
    if security_findings:
        warning_reasons.append("security_findings")
    if has_suspicious_url:
        warning_reasons.append("suspicious_url_findings")
    if has_sender_indicator:
        warning_reasons.append("sender_indicators")
    if prediction == "spam" and has_phishing_like_content:
        warning_reasons.append("spam_with_phishing_like_indicators")

    warning_severity = "NONE"
    if warning_reasons:
        warning_severity = raw_risk_level
        if recommended_action == "REVIEW" or recommended_action != base_action:
            warning_severity = _maximum_severity(warning_severity, "MEDIUM")
        if has_suspicious_url or has_sender_indicator:
            warning_severity = _maximum_severity(warning_severity, "MEDIUM")
        if prediction == "phishing" or (
            prediction == "spam" and has_phishing_like_content
        ):
            warning_severity = _maximum_severity(warning_severity, "HIGH")

    return RoutingDecision(
        prediction=prediction,
        folder=PREDICTION_FOLDERS[prediction],
        reason=AUTOMATIC_ROUTING_REASONS[prediction],
        base_action=base_action,
        recommended_action=recommended_action,
        has_warning=bool(warning_reasons),
        warning_severity=warning_severity,
        warning_reasons=tuple(dict.fromkeys(warning_reasons)),
    )
