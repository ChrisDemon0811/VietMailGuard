"""Conservative spam-candidate preparation and manual-decision helpers."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from vietmailguard.dataset_audit import normalize_for_comparison
from vietmailguard.label_review import ReviewRule

CURATION_COLUMNS = [
    "source",
    "original_row_id",
    "subject",
    "body_excerpt",
    "raw_label",
    "candidate_group",
    "triggered_spam_rules",
    "triggered_phishing_rules",
    "proposed_label",
    "confidence",
    "review_status",
    "review_reason",
]

MANUAL_LABEL_COLUMNS = [
    "source",
    "original_row_id",
    "manual_decision",
    "review_status",
    "reviewed_at_utc",
    "reviewer",
    "note",
]

MANUAL_DECISIONS = {"spam", "phishing_or_scam", "review", "exclude"}


def content_digest(subject: object, body: object, *, template: bool = False) -> str:
    """Hash normalized subject and body without modifying either source value."""
    text = f"{'' if pd.isna(subject) else subject}\n{'' if pd.isna(body) else body}"
    normalized = normalize_for_comparison(text, template=template)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_duplicate_counts(frames: Iterable[pd.DataFrame]) -> tuple[Counter[str], Counter[str]]:
    """Count exact and URL/email-insensitive template hashes across all supplied corpora."""
    exact: Counter[str] = Counter()
    template: Counter[str] = Counter()
    for frame in frames:
        subjects = frame.get("subject", pd.Series("", index=frame.index))
        bodies = frame.get("body", pd.Series("", index=frame.index))
        for subject, body in zip(subjects, bodies):
            exact[content_digest(subject, body)] += 1
            template[content_digest(subject, body, template=True)] += 1
    return exact, template


def split_triggered_rules(text: str, rules: list[ReviewRule]) -> tuple[list[str], list[str]]:
    """Return matching spam and phishing/scam rule names separately."""
    spam: list[str] = []
    phishing: list[str] = []
    for rule in rules:
        if not rule.matches(text):
            continue
        if rule.family == "spam":
            spam.append(rule.name)
        elif rule.family == "phishing_or_scam":
            phishing.append(rule.name)
    return spam, phishing


def safe_report_text(value: object, max_length: int | None = None) -> str:
    """Normalize whitespace, truncate excerpts, and neutralize spreadsheet formulas."""
    text = "" if pd.isna(value) else " ".join(str(value).split())
    if max_length is not None and len(text) > max_length:
        text = f"{text[:max_length - 1]}…"
    if text.startswith(("=", "+", "-", "@")):
        text = f"'{text}"
    return text


def assess_spam_candidate(
    *,
    source: str,
    original_row_id: int,
    raw_label: object,
    subject: object,
    body: object,
    rules: list[ReviewRule],
    policy: dict[str, Any],
    exact_counts: Counter[str],
    template_counts: Counter[str],
) -> dict[str, Any]:
    """Assess a rule candidate conservatively without confirming a ground-truth label."""
    subject_text = safe_report_text(subject)
    body_text = safe_report_text(body)
    spam_rules, phishing_rules = split_triggered_rules(
        f"{subject_text}\n{body_text}", rules
    )
    exact_count = exact_counts[content_digest(subject, body)]
    template_count = template_counts[content_digest(subject, body, template=True)]
    body_words = len(body_text.split())

    failures: list[str] = []
    if len(spam_rules) < policy["minimum_distinct_spam_rules"]:
        failures.append("insufficient_distinct_spam_signals")
    if len(phishing_rules) > policy["maximum_phishing_or_scam_rules"]:
        failures.append("phishing_or_scam_signal_present")
    if len(body_text) < policy["minimum_body_characters"]:
        failures.append("body_too_short")
    if body_words < policy["minimum_body_words"]:
        failures.append("too_few_body_words")
    if not body_text:
        failures.append("empty_body")
    if policy["require_unique_exact_content"] and exact_count > 1:
        failures.append("exact_duplicate")
    if policy["require_unique_template_content"] and template_count > 1:
        failures.append("template_duplicate")

    if failures:
        candidate_group = "likely_spam_needs_review"
        proposed_label = "review"
        confidence = "low" if "phishing_or_scam_signal_present" in failures else "medium"
        review_status = "needs_human_review"
        reason = ";".join(failures)
    else:
        candidate_group = "high_confidence_curated_spam"
        proposed_label = "spam"
        confidence = "high"
        review_status = "auto_candidate"
        reason = (
            f"{len(spam_rules)}_consistent_commercial_signals;"
            "no_phishing_or_fraud_signal;informative_body;unique_content"
        )

    return {
        "source": source,
        "original_row_id": int(original_row_id),
        "subject": subject_text,
        "body_excerpt": safe_report_text(body, 600),
        "raw_label": raw_label,
        "candidate_group": candidate_group,
        "triggered_spam_rules": "|".join(spam_rules),
        "triggered_phishing_rules": "|".join(phishing_rules),
        "proposed_label": proposed_label,
        "confidence": confidence,
        "review_status": review_status,
        "review_reason": reason,
        "_exact_duplicate": exact_count > 1,
        "_template_duplicate": template_count > 1,
    }


def load_manual_labels(path: Path) -> pd.DataFrame:
    """Load saved human decisions, returning an empty typed table if none exist."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=MANUAL_LABEL_COLUMNS)
    frame = pd.read_csv(path, dtype={"source": str, "original_row_id": int}, keep_default_na=False)
    missing = set(MANUAL_LABEL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manual label file is missing columns: {sorted(missing)}")
    invalid = set(frame["manual_decision"]) - MANUAL_DECISIONS
    if invalid:
        raise ValueError(f"Unsupported manual decisions: {sorted(invalid)}")
    return frame[MANUAL_LABEL_COLUMNS]


def apply_manual_labels(candidates: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Overlay genuine human decisions onto candidates using stable source-row keys."""
    if manual.empty:
        return candidates.copy()
    latest = manual.drop_duplicates(["source", "original_row_id"], keep="last")
    decisions = {
        (row.source, int(row.original_row_id)): row.manual_decision
        for row in latest.itertuples(index=False)
    }
    result = candidates.copy()
    for index, row in result.iterrows():
        decision = decisions.get((str(row["source"]), int(row["original_row_id"])))
        if decision is None:
            continue
        if decision in {"spam", "phishing_or_scam"}:
            result.at[index, "proposed_label"] = decision
            result.at[index, "confidence"] = "high"
            result.at[index, "review_status"] = "confirmed"
            result.at[index, "review_reason"] = f"human_confirmed_{decision}"
        elif decision == "exclude":
            result.at[index, "proposed_label"] = "review"
            result.at[index, "confidence"] = "high"
            result.at[index, "review_status"] = "confirmed"
            result.at[index, "review_reason"] = "human_confirmed_exclude"
        else:
            result.at[index, "proposed_label"] = "review"
            result.at[index, "confidence"] = "low"
            result.at[index, "review_status"] = "needs_human_review"
            result.at[index, "review_reason"] = "human_deferred_review"
    return result


def select_pending_candidates(
    candidates: pd.DataFrame, manual: pd.DataFrame, *, revisit_deferred: bool = False
) -> pd.DataFrame:
    """Return unreviewed rows, optionally including decisions deferred as review."""
    latest = manual.drop_duplicates(["source", "original_row_id"], keep="last")
    completed = {
        (str(row.source), int(row.original_row_id)): str(row.manual_decision)
        for row in latest.itertuples(index=False)
    }
    keep = []
    for row in candidates.itertuples(index=False):
        prior = completed.get((str(row.source), int(row.original_row_id)))
        keep.append(prior is None or (revisit_deferred and prior == "review"))
    return candidates.loc[keep].copy()
