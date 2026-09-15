"""Offline URL extraction and explainable suspicious-indicator analysis."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

import tldextract

from vietmailguard.dataset_standardizer import (
    extract_malformed_urls as _extract_malformed_urls,
    extract_urls as _extract_urls,
)

_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


@dataclass(frozen=True)
class UrlFinding:
    """One deterministic URL indicator; it is not a maliciousness verdict."""

    code: str
    description: str
    evidence: str


@dataclass(frozen=True)
class UrlAnalysis:
    """Structured offline analysis for one extracted URL."""

    url: str
    analysis_url: str
    scheme: str
    hostname: str
    registered_domain: str
    is_ip_host: bool
    is_https: bool
    length: int
    subdomain_depth: int
    suspicious_tokens: tuple[str, ...]
    has_punycode: bool
    is_shortener: bool
    is_obfuscated: bool
    matched_domain_patterns: tuple[str, ...]
    findings: tuple[UrlFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["suspicious_tokens"] = list(self.suspicious_tokens)
        result["matched_domain_patterns"] = list(self.matched_domain_patterns)
        result["findings"] = [asdict(finding) for finding in self.findings]
        return result


def extract_urls(text: object) -> list[str]:
    """Extract HTTP(S) and www-style URLs using the standardization logic."""
    return _extract_urls(text)


def extract_obfuscated_urls(text: object) -> list[str]:
    """Extract whitespace-broken or hxxp-style URL candidates without rewriting text."""
    return _extract_malformed_urls(text)


def extract_url_candidates(text: object) -> list[str]:
    """Return standard and obfuscated URL occurrences in stable input order."""
    candidates = [*extract_urls(text), *extract_obfuscated_urls(text)]
    return list(dict.fromkeys(candidates))


def _normalize_obfuscated_url(value: str) -> str:
    """Create a parse-only URL representation while preserving the reported original."""
    normalized = value.strip()
    normalized = re.sub(r"(?i)^hxxps(?=\s*:)", "https", normalized)
    normalized = re.sub(r"(?i)^hxxp(?=\s*:)", "http", normalized)
    normalized = re.sub(r"\[\s*\.\s*\]", ".", normalized)
    normalized = re.sub(r"\s*([:/.])\s*", r"\1", normalized)
    return normalized


def _matches_domain(hostname: str, domain: str) -> bool:
    candidate = domain.casefold().strip(".")
    return bool(candidate) and (hostname == candidate or hostname.endswith(f".{candidate}"))


def _hostname_details(hostname: str, ignored_labels: set[str]) -> tuple[str, int]:
    if not hostname:
        return "", 0
    try:
        ipaddress.ip_address(hostname)
        return hostname, 0
    except ValueError:
        pass
    extracted = _TLD_EXTRACTOR(hostname)
    registered_domain = (
        f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
    )
    labels = [
        label
        for label in extracted.subdomain.split(".")
        if label and label.casefold() not in ignored_labels
    ]
    return registered_domain, len(labels)


def analyze_url(url: str, rules: dict[str, Any]) -> UrlAnalysis:
    """Analyze one URL without network access or a malicious-domain assertion."""
    raw_url = str(url).strip()
    analysis_url = _normalize_obfuscated_url(raw_url)
    is_obfuscated = analysis_url != raw_url
    has_explicit_scheme = bool(re.match(r"(?i)^https?://", analysis_url))
    parse_target = analysis_url if has_explicit_scheme else f"http://{analysis_url}"
    try:
        parsed = urlsplit(parse_target)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
    except ValueError:
        parsed = urlsplit("http://invalid.local")
        hostname = ""
    scheme = parsed.scheme.casefold() if has_explicit_scheme else ""
    try:
        ipaddress.ip_address(hostname)
        is_ip_host = True
    except ValueError:
        is_ip_host = False

    ignored_labels = {str(value).casefold() for value in rules.get("ignore_subdomain_labels", [])}
    registered_domain, subdomain_depth = _hostname_details(hostname, ignored_labels)
    token_source = unquote(f"{hostname} {parsed.path} {parsed.query} {parsed.fragment}").casefold()
    observed_tokens = set(re.findall(r"[a-z0-9]+", token_source))
    suspicious_tokens = tuple(
        sorted(observed_tokens & {str(value).casefold() for value in rules["suspicious_tokens"]})
    )
    has_punycode = any(label.startswith("xn--") for label in hostname.split("."))
    is_shortener = any(
        _matches_domain(hostname, str(domain)) for domain in rules.get("shortener_domains", [])
    )

    matched_domain_patterns: list[str] = []
    for pattern in rules.get("domain_patterns", []):
        if re.search(str(pattern["regex"]), hostname, flags=re.IGNORECASE):
            matched_domain_patterns.append(str(pattern["id"]))

    findings: list[UrlFinding] = []
    if is_obfuscated:
        findings.append(
            UrlFinding(
                "obfuscated_url",
                "The URL uses spacing or hxxp-style obfuscation; this is a suspicious indicator, not a maliciousness verdict.",
                raw_url,
            )
        )
    if not hostname:
        findings.append(
            UrlFinding("missing_hostname", "The URL has no parseable hostname.", raw_url)
        )
    if is_ip_host:
        findings.append(
            UrlFinding("ip_host", "The URL hostname is an IP literal.", hostname)
        )
    if scheme == "http":
        findings.append(
            UrlFinding("plain_http", "The URL explicitly uses HTTP instead of HTTPS.", raw_url)
        )
    if len(analysis_url) > int(rules["long_url_length"]):
        findings.append(
            UrlFinding(
                "long_url",
                "The URL exceeds the configured length threshold.",
                str(len(analysis_url)),
            )
        )
    if subdomain_depth > int(rules["excessive_subdomain_depth"]):
        findings.append(
            UrlFinding(
                "excessive_subdomains",
                "The hostname exceeds the configured subdomain-depth threshold.",
                str(subdomain_depth),
            )
        )
    if suspicious_tokens:
        findings.append(
            UrlFinding(
                "suspicious_token",
                "The URL contains security-sensitive terms.",
                ", ".join(suspicious_tokens),
            )
        )
    if has_punycode:
        findings.append(
            UrlFinding("punycode", "The hostname contains a punycode label.", hostname)
        )
    if is_shortener:
        findings.append(
            UrlFinding(
                "shortener",
                "The hostname matches a configured URL-shortener domain.",
                hostname,
            )
        )
    descriptions = {
        str(pattern["id"]): str(pattern["description"])
        for pattern in rules.get("domain_patterns", [])
    }
    for pattern_id in matched_domain_patterns:
        findings.append(
            UrlFinding("domain_pattern", descriptions[pattern_id], pattern_id)
        )

    return UrlAnalysis(
        url=raw_url,
        analysis_url=analysis_url,
        scheme=scheme,
        hostname=hostname,
        registered_domain=registered_domain,
        is_ip_host=is_ip_host,
        is_https=scheme == "https",
        length=len(analysis_url),
        subdomain_depth=subdomain_depth,
        suspicious_tokens=suspicious_tokens,
        has_punycode=has_punycode,
        is_shortener=is_shortener,
        is_obfuscated=is_obfuscated,
        matched_domain_patterns=tuple(matched_domain_patterns),
        findings=tuple(findings),
    )


def analyze_urls(text: object, rules: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and analyze every URL occurrence in input order."""
    return [analyze_url(url, rules).to_dict() for url in extract_url_candidates(text)]
