"""Explainable bilingual social-engineering and formatting indicators."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from vietmailguard.dataset_standardizer import normalize_text


def load_security_config(path: Path) -> dict[str, Any]:
    """Load the configurable URL, sender, content, and explanation rules."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "url_rules",
        "sender_rules",
        "content_rules",
        "explainability",
        "recommendation_policy",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Security config is missing sections: {sorted(missing)}")
    return config


def _unique_evidence(values: list[str], limit: int = 5) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = normalize_text(value, "NFKC")[:160]
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _pattern_findings(
    text: str, rules: dict[str, Any], *, rule_language: str
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for category, patterns in rules["patterns"].items():
        evidence: list[str] = []
        rule_ids: list[str] = []
        descriptions: list[str] = []
        match_count = 0
        for pattern in patterns:
            matches = list(re.finditer(str(pattern["regex"]), text, flags=re.IGNORECASE))
            if not matches:
                continue
            match_count += len(matches)
            rule_ids.append(str(pattern["id"]))
            descriptions.append(str(pattern["description"]))
            evidence.extend(match.group(0) for match in matches)
        if rule_ids:
            findings.append(
                {
                    "code": str(category),
                    "language": rule_language,
                    "rule_ids": rule_ids,
                    "description": " ".join(dict.fromkeys(descriptions)),
                    "evidence": _unique_evidence(evidence),
                    "match_count": match_count,
                }
            )
    return findings


def _language_rule_sets(
    rules: dict[str, Any], detected_language: str
) -> list[tuple[str, dict[str, Any]]]:
    """Select configured rule sets without using language as an ML feature."""
    languages = rules.get("languages")
    if not isinstance(languages, dict):
        return [("en", rules)]
    if detected_language in languages and detected_language in {"en", "vi"}:
        return [(detected_language, languages[detected_language])]
    # Mixed and unknown content are scanned with both dictionaries so signals are not hidden.
    return [
        (language, language_rules)
        for language, language_rules in languages.items()
        if isinstance(language_rules, dict)
    ]


def analyze_content(
    text: object, rules: dict[str, Any], *, detected_language: str = "unknown"
) -> dict[str, Any]:
    """Return matched content indicators and transparent formatting statistics."""
    normalized = normalize_text(text, "NFKC")
    findings = [
        finding
        for language, language_rules in _language_rule_sets(rules, detected_language)
        for finding in _pattern_findings(
            normalized, language_rules, rule_language=str(language)
        )
    ]

    alpha_characters = [character for character in normalized if character.isalpha()]
    uppercase_characters = sum(character.isupper() for character in alpha_characters)
    uppercase_ratio = (
        uppercase_characters / len(alpha_characters) if alpha_characters else 0.0
    )
    exclamation_count = normalized.count("!")
    question_count = normalized.count("?")
    formatting = rules["formatting"]
    minimum_alpha = int(formatting["minimum_alpha_characters"])
    uppercase_threshold = float(formatting["uppercase_ratio"])
    punctuation_threshold = int(formatting["repeated_punctuation_run"])
    punctuation_runs = re.findall(rf"[!?]{{{punctuation_threshold},}}", normalized)

    if len(alpha_characters) >= minimum_alpha and uppercase_ratio >= uppercase_threshold:
        findings.append(
            {
                "code": "excessive_uppercase",
                "language": "shared",
                "rule_ids": ["uppercase_ratio"],
                "description": "Uppercase usage exceeds the configured ratio threshold.",
                "evidence": [f"uppercase_ratio={uppercase_ratio:.3f}"],
                "match_count": 1,
            }
        )
    if (
        exclamation_count >= int(formatting["exclamation_count"])
        or question_count >= int(formatting["question_count"])
        or punctuation_runs
    ):
        evidence = [
            f"exclamation_count={exclamation_count}",
            f"question_count={question_count}",
        ]
        evidence.extend(punctuation_runs[:3])
        findings.append(
            {
                "code": "excessive_punctuation",
                "language": "shared",
                "rule_ids": ["punctuation_threshold"],
                "description": "Punctuation usage exceeds a configured threshold.",
                "evidence": evidence,
                "match_count": max(1, len(punctuation_runs)),
            }
        )

    return {
        "language_scope": detected_language,
        "findings": findings,
        "statistics": {
            "alpha_character_count": len(alpha_characters),
            "uppercase_character_count": uppercase_characters,
            "uppercase_ratio": uppercase_ratio,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
        },
    }
