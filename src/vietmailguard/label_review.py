"""Create review-only candidate groups without assigning ground-truth labels."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REVIEW_COLUMNS = [
    "source",
    "original_row_id",
    "subject",
    "body_excerpt",
    "raw_label",
    "candidate_group",
    "triggered_rules",
    "proposed_label",
    "review_status",
]


@dataclass(frozen=True)
class ReviewRule:
    """A named set of regex patterns belonging to one review family."""

    name: str
    family: str
    patterns: tuple[re.Pattern[str], ...]

    def matches(self, text: str) -> bool:
        return any(pattern.search(text) is not None for pattern in self.patterns)


def load_review_config(path: Path) -> tuple[dict[str, Any], list[ReviewRule]]:
    """Load and compile the rule-assisted review configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    rules = [
        ReviewRule(
            name=item["name"],
            family=item["family"],
            patterns=tuple(re.compile(pattern, re.IGNORECASE) for pattern in item["patterns"]),
        )
        for item in config["rules"]
    ]
    return config, rules


def triage_text(text: str, config: dict[str, Any], rules: list[ReviewRule]) -> dict[str, Any]:
    """Return review suggestions; the result is never a training approval."""
    triggered = [rule for rule in rules if rule.matches(text)]
    phishing_rules = [rule.name for rule in triggered if rule.family == "phishing_or_scam"]
    spam_rules = [rule.name for rule in triggered if rule.family == "spam"]
    policy = config["decision_policy"]

    if len(phishing_rules) >= policy["phishing_or_scam_min_distinct_rules"]:
        candidate_group = "likely_phishing_or_scam"
        proposed_label = policy["likely_phishing_or_scam_proposed_label"]
    elif phishing_rules and policy["any_phishing_or_scam_signal_blocks_likely_spam"]:
        candidate_group = "uncertain"
        proposed_label = policy["uncertain_proposed_label"]
    elif len(spam_rules) >= policy["spam_min_distinct_rules"]:
        candidate_group = "likely_spam"
        proposed_label = policy["likely_spam_proposed_label"]
    else:
        candidate_group = "uncertain"
        proposed_label = policy["uncertain_proposed_label"]

    return {
        "candidate_group": candidate_group,
        "triggered_rules": "|".join(rule.name for rule in triggered),
        "proposed_label": proposed_label,
        "review_status": "pending_manual_confirmation",
    }


def _report_text(value: object, max_length: int | None = None) -> str:
    text = "" if pd.isna(value) else " ".join(str(value).split())
    if max_length is not None and len(text) > max_length:
        text = f"{text[:max_length - 1]}…"
    if text.startswith(("=", "+", "-", "@")):
        text = f"'{text}"
    return text


def build_review_pool(
    frame: pd.DataFrame,
    *,
    source: str,
    raw_label: object,
    config: dict[str, Any],
    rules: list[ReviewRule],
    body_excerpt_length: int = 500,
    allow_spam_proposal: bool = False,
) -> pd.DataFrame:
    """Build a review pool while retaining the physical CSV row identifier."""
    selected = frame.loc[frame["label"].eq(raw_label)]
    rows: list[dict[str, Any]] = []
    for index, row in selected.iterrows():
        subject = _report_text(row.get("subject", ""))
        body = _report_text(row.get("body", ""))
        if not body:
            triage = {
                "candidate_group": "uncertain",
                "triggered_rules": "empty_body",
                "proposed_label": "exclude",
                "review_status": "quality_exclusion_candidate",
            }
        else:
            triage = triage_text(f"{subject}\n{body}", config, rules)
            if triage["proposed_label"] == "spam" and not allow_spam_proposal:
                triage["proposed_label"] = "review"
        rows.append(
            {
                "source": source,
                "original_row_id": int(index) + 2,
                "subject": subject,
                "body_excerpt": _report_text(body, body_excerpt_length),
                "raw_label": raw_label,
                **triage,
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def candidate_counts(pool: pd.DataFrame) -> dict[str, int]:
    """Count every configured candidate group, including absent groups."""
    counts = pool["candidate_group"].value_counts().to_dict()
    return {
        name: int(counts.get(name, 0))
        for name in ("likely_spam", "likely_phishing_or_scam", "uncertain")
    }
