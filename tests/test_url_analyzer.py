from __future__ import annotations

from pathlib import Path

from vietmailguard.risk_engine import load_risk_config
from vietmailguard.url_analyzer import analyze_url, analyze_urls, extract_urls


ROOT = Path(__file__).resolve().parents[1]
RULES = load_risk_config(ROOT / "config" / "risk_rules.json")["url_rules"]


def test_url_extraction_strips_trailing_punctuation() -> None:
    text = "Open https://example.com/path, or www.example.org/help)."
    assert extract_urls(text) == ["https://example.com/path", "www.example.org/help"]


def test_suspicious_ip_url_is_reported_as_indicator() -> None:
    result = analyze_url("http://192.0.2.10/login", RULES)
    codes = {finding.code for finding in result.findings}
    assert result.scheme == "http"
    assert result.hostname == "192.0.2.10"
    assert result.is_ip_host
    assert not result.is_https
    assert result.subdomain_depth == 0
    assert "login" in result.suspicious_tokens
    assert {"ip_host", "plain_http", "suspicious_token"} <= codes


def test_normal_https_url_has_no_automatic_suspicious_finding() -> None:
    result = analyze_url("https://www.example.com/about", RULES)
    assert result.scheme == "https"
    assert result.hostname == "www.example.com"
    assert result.registered_domain == "example.com"
    assert result.subdomain_depth == 0
    assert result.is_https
    assert not result.findings


def test_shortener_and_punycode_are_configurable_indicators() -> None:
    shortener = analyze_url("https://bit.ly/abc", RULES)
    punycode = analyze_url("https://xn--example-9za.test/path", RULES)
    assert shortener.is_shortener
    assert any(finding.code == "shortener" for finding in shortener.findings)
    assert punycode.has_punycode
    assert any(finding.code == "punycode" for finding in punycode.findings)


def test_length_and_subdomain_depth_use_configured_thresholds() -> None:
    url = "https://a.b.c.d.example.com/" + "x" * 140
    result = analyze_url(url, RULES)
    codes = {finding.code for finding in result.findings}
    assert result.length == len(url)
    assert result.subdomain_depth == 4
    assert {"long_url", "excessive_subdomains"} <= codes


def test_obfuscated_url_is_preserved_and_analyzed() -> None:
    analyses = analyze_urls("Mở hxxp : / / 192 . 0 . 2 . 10 / login ngay", RULES)
    assert len(analyses) == 1
    result = analyses[0]
    assert result["url"] == "hxxp : / / 192 . 0 . 2 . 10 / login"
    assert result["analysis_url"] == "http://192.0.2.10/login"
    assert result["is_obfuscated"]
    codes = {finding["code"] for finding in result["findings"]}
    assert {"obfuscated_url", "ip_host", "plain_http", "suspicious_token"} <= codes
