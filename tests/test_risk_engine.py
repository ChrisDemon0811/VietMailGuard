from __future__ import annotations

from pathlib import Path

import pytest

from vietmailguard.risk_engine import (
    calculate_risk,
    load_risk_config,
    risk_level_for_score,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_risk_config(ROOT / "config" / "risk_rules.json")
LEVELS = CONFIG["risk_engine"]["levels"]


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "LOW"),
        (30, "LOW"),
        (31, "MEDIUM"),
        (60, "MEDIUM"),
        (61, "HIGH"),
        (80, "HIGH"),
        (81, "CRITICAL"),
        (100, "CRITICAL"),
    ],
)
def test_risk_threshold_boundaries(score: int, expected: str) -> None:
    assert risk_level_for_score(score, LEVELS) == expected


def test_risk_score_is_always_bounded() -> None:
    all_url_findings = [
        {"url": "http://192.0.2.10/login", "findings": [{"code": code}]}
        for code in CONFIG["risk_engine"]["url_finding_weights"]
    ]
    all_content_findings = [
        {"code": code} for code in CONFIG["risk_engine"]["content_finding_weights"]
    ]
    result = calculate_risk(
        prediction="phishing",
        class_probabilities={"normal": -5.0, "spam": 2.0, "phishing": 8.0},
        url_analyses=all_url_findings,
        content_findings=all_content_findings,
        rules=CONFIG,
    )
    assert result.score == 100
    assert result.level == "CRITICAL"


def test_low_risk_is_not_ml_confidence() -> None:
    result = calculate_risk(
        prediction="normal",
        class_probabilities={"normal": 0.99, "spam": 0.005, "phishing": 0.005},
        url_analyses=[],
        content_findings=[],
        rules=CONFIG,
    )
    assert 0 <= result.score <= 100
    assert result.score != 99
    assert result.level == "LOW"


def test_sender_signal_is_a_separate_bounded_component() -> None:
    config = load_risk_config(ROOT / "config" / "risk_rules_v2.json")
    result = calculate_risk(
        prediction="normal",
        class_probabilities={"normal": 0.99, "spam": 0.005, "phishing": 0.005},
        url_analyses=[],
        content_findings=[],
        sender_findings=[{"code": "sender_ip_domain"}],
        rules=config,
    )
    assert result.components["sender_indicators"] == 8.0
    assert 0 <= result.score <= 100
