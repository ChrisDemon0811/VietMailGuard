import json
from pathlib import Path

import pandas as pd

from vietmailguard.label_review import build_review_pool, load_review_config, triage_text


ROOT = Path(__file__).resolve().parents[1]


def test_review_rules_require_multiple_signals_and_never_approve_training() -> None:
    config, rules = load_review_config(ROOT / "config" / "label_review_rules.json")

    one_signal = triage_text("Special promotion for you", config, rules)
    spam_candidate = triage_text(
        "Special offer: save up to 70 percent. Unsubscribe from our mailing list.", config, rules
    )
    phishing_candidate = triage_text(
        "Verify your account now and enter your password to sign in.", config, rules
    )

    assert one_signal["candidate_group"] == "uncertain"
    assert spam_candidate["candidate_group"] == "likely_spam"
    assert spam_candidate["proposed_label"] == "spam"
    assert phishing_candidate["candidate_group"] == "likely_phishing_or_scam"
    assert phishing_candidate["proposed_label"] == "review"
    assert {one_signal["review_status"], spam_candidate["review_status"]} == {
        "pending_manual_confirmation"
    }


def test_empty_body_is_an_exclusion_candidate() -> None:
    config, rules = load_review_config(ROOT / "config" / "label_review_rules.json")
    frame = pd.DataFrame([{"subject": "Offer", "body": "  ", "label": 1}])

    pool = build_review_pool(
        frame, source="fixture", raw_label=1, config=config, rules=rules
    )

    assert pool.loc[0, "original_row_id"] == 2
    assert pool.loc[0, "proposed_label"] == "exclude"
    assert pool.loc[0, "review_status"] == "quality_exclusion_candidate"


def test_phishing_signal_blocks_spam_proposal() -> None:
    config, rules = load_review_config(ROOT / "config" / "label_review_rules.json")
    result = triage_text(
        "Special offer and free quote. Enter your password to continue.", config, rules
    )

    assert result["candidate_group"] == "uncertain"
    assert result["proposed_label"] == "review"


def test_every_observed_raw_label_has_exactly_one_policy_status() -> None:
    datasets = json.loads((ROOT / "config" / "datasets.json").read_text(encoding="utf-8"))
    allowed = set(datasets["policy"]["raw_label_statuses"])
    observed = pd.read_csv(ROOT / "reports" / "dataset_labels.csv", dtype=str)

    for filename, group in observed.groupby("file"):
        configured = datasets["datasets"][filename]["raw_label_policy"]
        assert set(group["raw_label"]) == set(configured)
        assert all(item["status"] in allowed for item in configured.values())
        assert all(
            item["training_eligible"] == (item["status"] in {"normal", "spam", "phishing"})
            for item in configured.values()
        )
