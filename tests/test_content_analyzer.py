from __future__ import annotations

from pathlib import Path

import pytest

from vietmailguard.content_analyzer import analyze_content, load_security_config
from vietmailguard.risk_engine import load_risk_config


ROOT = Path(__file__).resolve().parents[1]
RULES = load_risk_config(ROOT / "config" / "risk_rules.json")["content_rules"]
V2_RULES = load_security_config(ROOT / "config" / "security_rules_v2.json")[
    "content_rules"
]


def test_content_rules_return_only_observed_evidence() -> None:
    text = (
        "URGENT: Your account will be suspended within 24 hours. "
        "Verify your password immediately and click the link!!!!!"
    )
    analysis = analyze_content(text, RULES)
    codes = {finding["code"] for finding in analysis["findings"]}
    assert {
        "urgency",
        "credential_request",
        "account_suspension",
        "call_to_action",
        "excessive_punctuation",
    } <= codes
    assert all(finding["evidence"] for finding in analysis["findings"])


def test_neutral_content_has_no_security_rule_findings() -> None:
    analysis = analyze_content(
        "Hello team, the project meeting starts at ten tomorrow morning.", RULES
    )
    assert analysis["findings"] == []
    assert 0.0 <= analysis["statistics"]["uppercase_ratio"] <= 1.0


@pytest.mark.parametrize(
    ("text", "expected_code"),
    [
        ("Please send the bank details for the wire transfer.", "financial_bait"),
        ("You have won the lottery. Claim your prize today.", "prize_winner"),
        ("Failure to respond will result in legal action.", "threat"),
        ("THIS MESSAGE USES CAPITAL LETTERS THROUGHOUT THE ENTIRE NOTICE", "excessive_uppercase"),
    ],
)
def test_each_security_category_requires_matching_evidence(
    text: str, expected_code: str
) -> None:
    analysis = analyze_content(text, RULES)
    match = next(
        finding for finding in analysis["findings"] if finding["code"] == expected_code
    )
    assert match["evidence"]


def test_vietnamese_security_rules_report_observed_categories() -> None:
    text = (
        "Khẩn cấp: tài khoản của bạn đã bị khóa. Hãy đăng nhập ngay và nhập mật khẩu "
        "để xác minh tài khoản trong vòng 24 giờ."
    )
    analysis = analyze_content(text, V2_RULES, detected_language="vi")
    codes = {finding["code"] for finding in analysis["findings"]}
    assert {"urgency", "credential_request", "account_suspension", "call_to_action"} <= codes
    assert all(finding["language"] in {"vi", "shared"} for finding in analysis["findings"])


def test_vietnamese_commercial_rules_are_signals_not_labels() -> None:
    analysis = analyze_content(
        "Khuyến mãi lớn, giảm giá hôm nay. Mua ngay để nhận ưu đãi.",
        V2_RULES,
        detected_language="vi",
    )
    finding = next(
        value for value in analysis["findings"] if value["code"] == "commercial_promotion"
    )
    assert finding["match_count"] >= 3
    assert "label" not in finding
