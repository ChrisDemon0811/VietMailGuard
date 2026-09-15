from collections import Counter
from pathlib import Path

import pandas as pd

from vietmailguard.label_review import load_review_config
from vietmailguard.spam_curation import (
    MANUAL_LABEL_COLUMNS,
    apply_manual_labels,
    assess_spam_candidate,
    content_digest,
    select_pending_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def _assess(body: str, *, duplicate: bool = False) -> dict[str, object]:
    _, rules = load_review_config(ROOT / "config" / "label_review_rules.json")
    policy = {
        "minimum_distinct_spam_rules": 2,
        "maximum_phishing_or_scam_rules": 0,
        "minimum_body_characters": 120,
        "minimum_body_words": 20,
        "require_unique_exact_content": True,
        "require_unique_template_content": True,
    }
    count = 2 if duplicate else 1
    exact = Counter({content_digest("Offer", body): count})
    template = Counter({content_digest("Offer", body, template=True): count})
    return assess_spam_candidate(
        source="fixture",
        original_row_id=2,
        raw_label=1,
        subject="Offer",
        body=body,
        rules=rules,
        policy=policy,
        exact_counts=exact,
        template_counts=template,
    )


def test_high_confidence_candidate_is_not_confirmed() -> None:
    body = (
        "This commercial advertisement presents a special offer for our life insurance service. "
        "Request a free quote today and compare the available plans. You may unsubscribe from our "
        "mailing list at any time. This message describes product prices and customer options."
    )
    result = _assess(body)

    assert result["candidate_group"] == "high_confidence_curated_spam"
    assert result["proposed_label"] == "spam"
    assert result["review_status"] == "auto_candidate"


def test_duplicate_or_phishing_signal_requires_human_review() -> None:
    commercial = (
        "Commercial advertisement and special offer for life insurance. Request a free quote and "
        "unsubscribe from our mailing list if you do not want future product promotions from us. "
        "This is a detailed description of prices, discounts, coverage, and sales conditions."
    )
    credential = commercial + " Verify your account and enter your password to sign in now."

    assert _assess(commercial, duplicate=True)["proposed_label"] == "review"
    phishing_result = _assess(credential)
    assert phishing_result["proposed_label"] == "review"
    assert "phishing_or_scam_signal_present" in phishing_result["review_reason"]


def test_manual_decisions_are_applied_and_resume_skips_completed() -> None:
    candidates = pd.DataFrame(
        [
            {
                "source": "Ling",
                "original_row_id": 10,
                "proposed_label": "spam",
                "confidence": "high",
                "review_status": "auto_candidate",
                "review_reason": "rule_candidate",
                "_exact_duplicate": False,
                "_template_duplicate": False,
            },
            {
                "source": "Ling",
                "original_row_id": 11,
                "proposed_label": "review",
                "confidence": "medium",
                "review_status": "needs_human_review",
                "review_reason": "short",
                "_exact_duplicate": False,
                "_template_duplicate": False,
            },
        ]
    )
    manual = pd.DataFrame(
        [
            {
                "source": "Ling",
                "original_row_id": 10,
                "manual_decision": "spam",
                "review_status": "confirmed",
                "reviewed_at_utc": "2026-09-11T00:00:00+00:00",
                "reviewer": "tester",
                "note": "read full message",
            }
        ],
        columns=MANUAL_LABEL_COLUMNS,
    )

    applied = apply_manual_labels(candidates, manual)
    pending = select_pending_candidates(candidates, manual)

    assert applied.loc[0, "review_status"] == "confirmed"
    assert applied.loc[0, "review_reason"] == "human_confirmed_spam"
    assert pending["original_row_id"].tolist() == [11]
