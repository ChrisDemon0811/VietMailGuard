from pathlib import Path

import pandas as pd

from vietmailguard.spam_curation import content_digest


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_eligibility_has_only_approved_provenance() -> None:
    eligible = pd.read_csv(ROOT / "data" / "processed" / "baseline_eligibility.csv")

    assert list(eligible.columns) == [
        "original_row_id",
        "source",
        "subject",
        "body",
        "label",
        "label_provenance",
        "language",
        "training_eligible",
    ]
    assert set(eligible["label"]) == {"normal", "spam", "phishing"}
    assert set(eligible["label_provenance"]) == {
        "verified_legitimate_source",
        "nazario_phishing",
        "high_confidence_curated_spam",
    }
    assert eligible["training_eligible"].all()
    assert len(eligible.loc[eligible["label"].eq("spam")]) == 1310


def test_validation_sample_is_stratified_and_preserves_blank_decisions() -> None:
    sample = pd.read_csv(
        ROOT / "reports" / "spam_validation_sample.csv", keep_default_na=False
    )

    assert sample.groupby("source").size().to_dict() == {
        "CEAS_08": 20,
        "Enron": 80,
        "Ling": 15,
        "SpamAssassin": 35,
    }
    assert set(sample["human_decision"]).issubset(
        {"", "spam", "phishing_or_scam", "review", "exclude"}
    )
    assert not sample.duplicated(["source", "original_row_id"]).any()
    eligible = pd.read_csv(ROOT / "data" / "processed" / "baseline_eligibility.csv")
    selected = sample.merge(
        eligible.loc[eligible["label"].eq("spam")],
        on=["source", "original_row_id"],
        how="left",
        validate="one_to_one",
    )
    exact = selected.apply(lambda row: content_digest(row["subject_y"], row["body"]), axis=1)
    template = selected.apply(
        lambda row: content_digest(row["subject_y"], row["body"], template=True), axis=1
    )
    assert exact.is_unique
    assert template.is_unique


def test_sampling_manifest_is_balanced_and_duplicate_free() -> None:
    eligible = pd.read_csv(ROOT / "data" / "processed" / "baseline_eligibility.csv")
    manifest = pd.read_csv(ROOT / "reports" / "baseline_sampling_manifest.csv")

    assert manifest["label"].value_counts().to_dict() == {
        "normal": 1250,
        "spam": 1250,
        "phishing": 1250,
    }
    assert not manifest.duplicated(["source", "original_row_id"]).any()
    selected = manifest.merge(
        eligible,
        on=["source", "original_row_id", "label", "label_provenance"],
        how="left",
        validate="one_to_one",
    )
    assert selected["body"].notna().all()
    exact = selected.apply(lambda row: content_digest(row["subject"], row["body"]), axis=1)
    template = selected.apply(
        lambda row: content_digest(row["subject"], row["body"], template=True), axis=1
    )
    assert exact.is_unique
    assert template.is_unique


def test_outstanding_review_is_not_in_eligibility() -> None:
    outstanding = pd.read_csv(
        ROOT / "reports" / "spam_curation_needs_human_review.csv"
    )
    eligible = pd.read_csv(
        ROOT / "data" / "processed" / "baseline_eligibility.csv",
        usecols=["source", "original_row_id"],
    )

    assert len(outstanding) == 36
    overlap = outstanding.merge(eligible, on=["source", "original_row_id"], how="inner")
    assert overlap.empty
