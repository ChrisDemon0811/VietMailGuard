"""Configurable decision-support risk scoring separate from ML confidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RiskResult:
    """A bounded heuristic score with an auditable component breakdown."""

    score: int
    level: str
    components: dict[str, float]
    raw_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_risk_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the risk/security configuration."""
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"risk_engine"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Risk config is missing sections: {sorted(missing)}")
    _validate_levels(config["risk_engine"]["levels"])
    return config


def _validate_levels(levels: list[dict[str, Any]]) -> None:
    ordered = sorted(levels, key=lambda item: int(item["minimum"]))
    if not ordered or int(ordered[0]["minimum"]) != 0 or int(ordered[-1]["maximum"]) != 100:
        raise ValueError("Risk levels must cover scores from 0 through 100")
    expected_minimum = 0
    for level in ordered:
        minimum = int(level["minimum"])
        maximum = int(level["maximum"])
        if minimum != expected_minimum or maximum < minimum:
            raise ValueError("Risk levels must be contiguous and non-overlapping")
        expected_minimum = maximum + 1


def risk_level_for_score(score: int, levels: list[dict[str, Any]]) -> str:
    """Map a bounded integer score to its configured internal level."""
    if score < 0 or score > 100:
        raise ValueError("Risk score must be between 0 and 100")
    _validate_levels(levels)
    for level in levels:
        if int(level["minimum"]) <= score <= int(level["maximum"]):
            return str(level["name"])
    raise RuntimeError(f"No risk level configured for score {score}")


def _bounded_probability(value: object) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, probability))


def calculate_risk(
    *,
    prediction: str,
    class_probabilities: dict[str, float],
    url_analyses: list[dict[str, Any]],
    content_findings: list[dict[str, Any]],
    rules: dict[str, Any],
    sender_findings: list[dict[str, Any]] | None = None,
) -> RiskResult:
    """Calculate a bounded heuristic score without changing the ML prediction."""
    engine = rules["risk_engine"]
    caps = engine["component_caps"]

    ml_score = 0.0
    for label, weight in engine["ml_probability_weights"].items():
        ml_score += _bounded_probability(class_probabilities.get(label, 0.0)) * float(weight)
    ml_score += float(engine["prediction_bonus"].get(prediction, 0.0))
    ml_score = min(float(caps["ml"]), ml_score)

    url_score = 0.0
    seen_urls: set[str] = set()
    url_weights = engine["url_finding_weights"]
    for analysis in url_analyses:
        url = str(analysis.get("url", ""))
        if url in seen_urls:
            continue
        seen_urls.add(url)
        for finding in analysis.get("findings", []):
            url_score += float(url_weights.get(str(finding.get("code", "")), 0.0))
    url_score = min(float(caps["url"]), url_score)

    content_score = 0.0
    seen_content_codes: set[str] = set()
    content_weights = engine["content_finding_weights"]
    for finding in content_findings:
        code = str(finding.get("code", ""))
        if code in seen_content_codes:
            continue
        seen_content_codes.add(code)
        content_score += float(content_weights.get(code, 0.0))
    content_score = min(float(caps["content"]), content_score)

    sender_score = 0.0
    seen_sender_codes: set[str] = set()
    sender_weights = engine.get("sender_finding_weights", {})
    for finding in sender_findings or []:
        code = str(finding.get("code", ""))
        if code in seen_sender_codes:
            continue
        seen_sender_codes.add(code)
        sender_score += float(sender_weights.get(code, 0.0))
    sender_score = min(float(caps.get("sender", 0.0)), sender_score)

    raw_score = ml_score + url_score + content_score + sender_score
    score = min(100, max(0, int(round(raw_score))))
    level = risk_level_for_score(score, engine["levels"])
    return RiskResult(
        score=score,
        level=level,
        components={
            "ml_probability_and_prediction": round(ml_score, 6),
            "url_indicators": round(url_score, 6),
            "content_indicators": round(content_score, 6),
            "sender_indicators": round(sender_score, 6),
        },
        raw_score=round(raw_score, 6),
    )
