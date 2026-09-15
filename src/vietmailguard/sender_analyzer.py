"""Offline sender-address and display-name consistency indicators."""

from __future__ import annotations

import ipaddress
import re
from email.utils import parseaddr
from typing import Any

from vietmailguard.dataset_standardizer import normalize_text


_DOMAIN_PATTERN = re.compile(r"(?i)\b(?:[a-z0-9-]+\.)+[a-z]{2,63}\b")


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip("[]"))
        return True
    except ValueError:
        return False


def analyze_sender(sender: object, rules: dict[str, Any]) -> dict[str, Any]:
    """Analyze available sender metadata without asserting maliciousness."""
    raw = normalize_text(sender, "NFKC")
    display_name, address = parseaddr(raw)
    if not address:
        ip_literal = re.search(
            r"(?i)([a-z0-9._%+-]+)@\[((?:\d{1,3}\.){3}\d{1,3}|[0-9a-f:]+)\]",
            raw,
        )
        if ip_literal:
            address = ip_literal.group(0)
            display_name = raw.split("<", 1)[0].strip().strip('"')
    address = address.casefold()
    local_part, separator, domain = address.rpartition("@")
    domain = domain.rstrip(".") if separator else ""
    findings: list[dict[str, Any]] = []

    def add(code: str, description: str, evidence: str) -> None:
        findings.append(
            {
                "code": code,
                "description": description,
                "evidence": [evidence] if evidence else [],
                "rule_ids": [code],
                "match_count": 1,
            }
        )

    if not raw:
        return {
            "raw_sender": "",
            "display_name": "",
            "address": "",
            "domain": "",
            "is_ip_domain": False,
            "findings": [],
        }
    if not separator or not local_part or not domain:
        add("invalid_sender_format", "Sender address could not be parsed reliably.", raw)
    if raw.count("@") > int(rules.get("maximum_at_signs", 1)):
        add("suspicious_sender_format", "Sender field contains multiple at-signs.", raw)
    if any(ord(character) < 32 for character in raw):
        add("suspicious_sender_format", "Sender field contains control characters.", raw)

    is_ip_domain = _is_ip(domain) if domain else False
    if is_ip_domain:
        add("sender_ip_domain", "Sender domain is an IP literal.", domain)
    if domain and any(label.startswith("xn--") for label in domain.split(".")):
        add("sender_punycode", "Sender domain contains a punycode label.", domain)

    # Only compare literal domains exposed by the display name; no brand allow/deny list is used.
    display_domains = {
        match.group(0).casefold().rstrip(".") for match in _DOMAIN_PATTERN.finditer(display_name)
    }
    mismatches = sorted(value for value in display_domains if value != domain)
    if domain and mismatches:
        add(
            "display_domain_mismatch",
            "Display name contains a domain different from the sender address domain.",
            f"display={','.join(mismatches)}; sender={domain}",
        )

    return {
        "raw_sender": raw,
        "display_name": display_name,
        "address": address,
        "domain": domain,
        "is_ip_domain": is_ip_domain,
        "findings": findings,
    }
