from __future__ import annotations

from vietmailguard.sender_analyzer import analyze_sender


RULES = {"maximum_at_signs": 1}


def test_sender_domain_is_parsed_without_brand_assumptions() -> None:
    result = analyze_sender("Project Manager <manager@example.com>", RULES)
    assert result["domain"] == "example.com"
    assert result["findings"] == []


def test_sender_ip_domain_is_only_a_suspicious_indicator() -> None:
    result = analyze_sender("Notice <user@[192.0.2.10]>", RULES)
    assert result["is_ip_domain"]
    assert any(value["code"] == "sender_ip_domain" for value in result["findings"])


def test_literal_display_domain_mismatch_is_reported() -> None:
    result = analyze_sender("accounts.example.com <notice@other.example>", RULES)
    codes = {value["code"] for value in result["findings"]}
    assert "display_domain_mismatch" in codes
