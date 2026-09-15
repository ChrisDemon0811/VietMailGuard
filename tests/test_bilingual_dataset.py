"""Tests for Version 2 bilingual grouping and split integrity."""

from copy import deepcopy

import pandas as pd
import pytest

from vietmailguard.bilingual_dataset import (
    BilingualLeakageError,
    assign_bilingual_splits,
    build_final_groups,
    prepare_bilingual_rows,
    split_frames,
    validate_bilingual_splits,
)


def _row(
    row_id: str,
    label: str,
    *,
    group_id: str,
    parent_id: str = "",
    language: str = "en",
    source: str = "Enron",
    train_only: bool = False,
) -> dict[str, object]:
    return {
        "id": row_id,
        "parent_id": parent_id,
        "translation_source_id": "translator:" + parent_id if parent_id else "",
        "source": source,
        "parent_source": "Enron" if parent_id else "",
        "language": language,
        "data_origin": "translated" if language == "vi" else "",
        "augmentation_type": "controlled_translation" if train_only else "",
        "subject": f"Subject {row_id}",
        "body": f"Unique body for {row_id}",
        "label": label,
        "raw_label": "1" if label != "normal" else "0",
        "label_status": "evidence_confirmed" if language == "vi" else "eligible",
        "label_provenance": "trusted",
        "content_hash": f"hash-{row_id}",
        "group_id": group_id,
        "training_eligible": True,
        "split_constraint": "train_only" if train_only else "flexible",
    }


def _manual_split_row(
    row_id: str,
    label: str,
    split: str,
    *,
    parent_id: str = "",
    suffix: str | None = None,
) -> dict[str, object]:
    token = suffix or row_id
    return {
        **_row(row_id, label, group_id=f"group-{token}", parent_id=parent_id),
        "final_group_id": f"final-{token}",
        "translation_group_id": f"translation-{token}" if parent_id else "",
        "normalized_content_hash": f"normalized-{token}",
        "template_hash": f"template-{token}",
        "split": split,
    }


def _valid_splits() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.DataFrame([_manual_split_row("e1", "normal", "train")]),
        "validation": pd.DataFrame([_manual_split_row("e2", "spam", "validation")]),
        "test": pd.DataFrame([_manual_split_row("e3", "phishing", "test")]),
    }


def test_parent_translation_and_template_duplicates_share_final_group() -> None:
    rows = pd.DataFrame(
        [
            _row("e1", "normal", group_id="template-parent"),
            _row("e1-copy", "normal", group_id="template-parent"),
            _row(
                "vi1", "normal", group_id="vi-template", parent_id="e1",
                language="vi", source="data_vi"
            ),
        ]
    )
    grouped, audit = build_final_groups(rows)

    assert grouped["final_group_id"].nunique() == 1
    assert grouped.loc[grouped["id"].isin(["e1", "vi1"]), "translation_group_id"].nunique() == 1
    assert audit["parent_translation_edges"] == 1
    assert audit["source_group_edges"] == 1


def test_controlled_translation_and_parent_are_forced_to_train() -> None:
    rows = [_row("spam-parent", "spam", group_id="spam-parent")]
    rows.append(
        _row(
            "spam-vi", "spam", group_id="spam-parent", parent_id="spam-parent",
            language="vi", source="controlled_translation", train_only=True
        )
    )
    for label in ("normal", "spam", "phishing"):
        for index in range(20):
            rows.append(_row(f"{label}-{index}", label, group_id=f"{label}-{index}"))
    grouped, _ = build_final_groups(pd.DataFrame(rows))
    assigned, _ = assign_bilingual_splits(
        grouped, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=42
    )
    splits = split_frames(assigned, seed=42)

    assert set(assigned.loc[assigned["id"].isin(["spam-parent", "spam-vi"]), "split"]) == {"train"}
    assert validate_bilingual_splits(splits)["passed"] is True


def test_final_group_cannot_cross_splits() -> None:
    splits = _valid_splits()
    splits["validation"].loc[0, "final_group_id"] = splits["train"].loc[0, "final_group_id"]
    with pytest.raises(BilingualLeakageError, match="final_group_id"):
        validate_bilingual_splits(splits)


def test_parent_translation_pair_cannot_cross_splits() -> None:
    splits = _valid_splits()
    translation = _manual_split_row("vi1", "normal", "validation", parent_id="e1")
    splits["validation"] = pd.DataFrame([translation])
    with pytest.raises(BilingualLeakageError, match="Parent/translation"):
        validate_bilingual_splits(splits)


def test_content_hash_cannot_cross_splits() -> None:
    splits = _valid_splits()
    splits["validation"].loc[0, "content_hash"] = splits["train"].loc[0, "content_hash"]
    with pytest.raises(BilingualLeakageError, match="content_hash"):
        validate_bilingual_splits(splits)


def test_normalized_duplicate_cannot_cross_splits() -> None:
    splits = _valid_splits()
    splits["validation"].loc[0, "normalized_content_hash"] = splits["train"].loc[
        0, "normalized_content_hash"
    ]
    with pytest.raises(BilingualLeakageError, match="normalized_content_hash"):
        validate_bilingual_splits(splits)


def test_recomputed_template_hash_cannot_cross_splits() -> None:
    splits = _valid_splits()
    splits["validation"].loc[0, "template_hash"] = splits["train"].loc[0, "template_hash"]
    with pytest.raises(BilingualLeakageError, match="template_hash"):
        validate_bilingual_splits(splits)


def test_known_template_group_cannot_cross_splits() -> None:
    splits = _valid_splits()
    splits["validation"].loc[0, "group_id"] = splits["train"].loc[0, "group_id"]
    with pytest.raises(BilingualLeakageError, match="group_id"):
        validate_bilingual_splits(splits)


@pytest.mark.parametrize("invalid_label", ["review", "exclude", "unknown"])
def test_only_three_valid_labels_can_enter_splits(invalid_label: str) -> None:
    splits = _valid_splits()
    splits["train"].loc[0, "label"] = invalid_label
    with pytest.raises(BilingualLeakageError, match="Invalid labels"):
        validate_bilingual_splits(splits)


def test_ineligible_row_cannot_enter_splits() -> None:
    splits = deepcopy(_valid_splits())
    splits["test"].loc[0, "training_eligible"] = False
    with pytest.raises(BilingualLeakageError, match="non-training-eligible"):
        validate_bilingual_splits(splits)


def test_trusted_manifest_link_is_authoritative_when_curation_status_is_blank() -> None:
    english = pd.DataFrame(
        [{
            "id": "e1", "original_row_id": "1", "subject": "Hello", "body": "Message",
            "label": "normal", "raw_label": "0", "source": "Enron", "language": "en",
            "content_hash": "eh1", "group_id": "eg1",
            "label_provenance": "verified_legitimate_source",
        }]
    )
    vietnamese = pd.DataFrame(
        [{
            "id": "vi1", "original_row_id": "1", "parent_id": "e1",
            "translation_source_id": "link:e1", "source": "data_vi", "subject": "",
            "body": "Tin nhắn", "label": "normal", "raw_label": "0",
            "label_status": "eligible", "label_provenance": "translated_dataset_verified_normal",
            "training_eligible": True, "content_hash": "vh1", "group_id": "vg1",
            "language": "vi", "data_origin": "translated", "parent_link_status": "",
            "linked_parent_source": "",
        }]
    )
    augmentation = pd.DataFrame(columns=[
        "id", "parent_id", "translation_source_id", "parent_source", "parent_label",
        "subject", "body", "label", "raw_label", "language", "data_origin",
        "augmentation_type", "content_hash", "group_id", "label_provenance",
        "training_eligible", "translation_quality_status", "protected_token_validation",
    ])
    overlap = pd.DataFrame([{
        "english_id": "e1", "vietnamese_id": "vi1",
        "review_status": "linked_high_confidence_evidence",
    }])
    config = {
        "valid_labels": ["normal", "spam", "phishing"],
        "english_label_provenance": {
            "normal": ["verified_legitimate_source"], "spam": [], "phishing": []
        },
        "trusted_cross_language_statuses": ["linked_high_confidence_evidence"],
        "augmentation_policy": {
            "required_quality_status": "accepted",
            "required_protected_token_validation": "passed",
            "source_name": "controlled_translation",
        },
    }

    selected, audit = prepare_bilingual_rows(
        english, vietnamese, augmentation, overlap, config
    )

    assert set(selected["id"]) == {"e1", "vi1"}
    assert selected.loc[selected["id"].eq("vi1"), "parent_source"].iat[0] == "Enron"
    assert audit["trusted_link_pairs"] == 1
