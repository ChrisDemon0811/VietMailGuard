"""Tests for controlled Vietnamese translation augmentation."""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from vietmailguard.translation_augmentation import (
    TranslationResult,
    apply_duplicate_translation_flags,
    build_augmented_row,
    load_augmentation_config,
    protect_text,
    quality_flags,
    restore_protected_text,
    select_parent_rows,
    validate_parent_split_safety,
    validation_sample,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return load_augmentation_config(ROOT / "config" / "translation_augmentation.json")


def _parents() -> pd.DataFrame:
    rows = [
        {
            "id": "Enron:1",
            "subject": "Meeting with John Smith",
            "body": "Please email john.smith@example.com on 2026-09-13.",
            "label": "normal",
            "raw_label": "0",
            "source": "Enron",
            "language": "en",
            "group_id": "group-normal",
            "label_provenance": "verified_legitimate_source",
        },
        {
            "id": "CEAS:2",
            "subject": "Same group copy",
            "body": "A duplicate group representation.",
            "label": "normal",
            "raw_label": "0",
            "source": "CEAS_08",
            "language": "en",
            "group_id": "group-normal",
            "label_provenance": "verified_legitimate_source",
        },
        {
            "id": "Ling:3",
            "subject": "Discount software",
            "body": "Buy software at https://shop.example.com for $20.",
            "label": "spam",
            "raw_label": "1",
            "source": "Ling",
            "language": "en",
            "group_id": "group-spam",
            "label_provenance": "high_confidence_curated_spam",
        },
        {
            "id": "Enron:4",
            "subject": "Uncertain positive",
            "body": "This label remains under review.",
            "label": "spam",
            "raw_label": "1",
            "source": "Enron",
            "language": "en",
            "group_id": "group-weak",
            "label_provenance": "weak_rule_candidate",
        },
        {
            "id": "Nazario:5",
            "subject": "Verify account",
            "body": "Log in to verify your account.",
            "label": "phishing",
            "raw_label": "1",
            "source": "Nazario",
            "language": "en",
            "group_id": "group-phishing",
            "label_provenance": "nazario_phishing",
        },
    ]
    return pd.DataFrame(rows)


def test_protected_values_restore_exactly() -> None:
    source = (
        "John Smith sent ABC-2026-42 to admin@example.com on 2026-09-13. "
        "Pay $1,250 at https://secure.example.com/login."
    )
    protected = protect_text(source)
    restored, missing = restore_protected_text(protected.masked_text, protected)

    assert restored == source
    assert missing == []
    assert "admin@example.com" in protected.placeholders.values()
    assert "https://secure.example.com/login" in protected.placeholders.values()
    assert "John Smith" in protected.placeholders.values()


def test_lost_url_and_email_are_quality_failures() -> None:
    source = "Contact admin@example.com and visit https://example.com/login today."
    protected = protect_text(source)
    flags = quality_flags(
        source,
        "Hãy liên hệ quản trị viên và truy cập trang đăng nhập hôm nay.",
        protected,
        _config(),
    )

    assert "broken_urls" in flags
    assert "lost_email_addresses" in flags
    assert "lost_protected_values" in flags


def test_extra_domain_like_token_does_not_claim_source_domain_was_lost() -> None:
    source = "Please review the attached document."
    protected = protect_text(source)
    flags = quality_flags(
        source,
        "Vui lòng xem tài liệu đính kèm từ win32net.Net.",
        protected,
        _config(),
    )

    assert "lost_domains" not in flags


def test_domain_attached_to_preserved_url_does_not_create_false_loss() -> None:
    source = "Account suspension.www.example.com/update"
    protected = protect_text(source)
    flags = quality_flags(
        source,
        "Tài khoản bị đình chỉ. www.example.com/update",
        protected,
        _config(),
    )

    assert "lost_domains" not in flags


def test_parent_selection_uses_only_trusted_train_provenance_and_unique_groups() -> None:
    config = deepcopy(_config())
    config["target_per_class"] = 10
    existing = pd.DataFrame([{"parent_id": "Ling:3"}])
    selected, audit = select_parent_rows(_parents(), config, existing)

    assert set(selected["id"]) == {"Enron:1", "Nazario:5"}
    assert selected["group_id"].is_unique
    assert audit["existing_parent_overlap_removed"] == 1
    assert "Enron:4" not in set(selected["id"])


def test_augmented_row_keeps_parent_group_and_split() -> None:
    config = _config()
    parent = _parents().iloc[0]
    subject = TranslationResult("Cuộc họp với John Smith", 1, True, False)
    body = TranslationResult(
        "Vui lòng gửi email john.smith@example.com vào 2026-09-13.",
        2,
        True,
        False,
    )
    row = build_augmented_row(
        parent,
        subject,
        body,
        protect_text(parent["subject"]),
        protect_text(parent["body"]),
        config,
    )
    frame = pd.DataFrame([row])

    assert row["group_id"] == row["parent_group_id"] == "group-normal"
    assert row["parent_split"] == "train"
    assert row["language"] == "vi"
    assert row["data_origin"] == "translated"
    assert row["augmentation_type"] == "controlled_translation"
    validate_parent_split_safety(frame)


def test_duplicate_translations_are_flagged_and_ineligible() -> None:
    config = _config()
    parent = _parents().iloc[0]
    subject = TranslationResult("Thông báo", 0, True, False)
    body = TranslationResult("Đây là nội dung tiếng Việt đầy đủ để kiểm tra.", 0, True, False)
    first = build_augmented_row(
        parent, subject, body, protect_text(parent["subject"]), protect_text(parent["body"]), config
    )
    second = dict(first)
    second.update(
        {
            "id": "aug_vi:other",
            "parent_id": "Other:2",
            "parent_group_id": "other-group",
            "group_id": "other-group",
        }
    )

    checked = apply_duplicate_translation_flags(pd.DataFrame([first, second]), config)

    assert checked["quality_flags"].str.contains("duplicate_translated_output").all()
    assert not checked["training_eligible"].astype(bool).any()


def test_review_sample_leaves_human_fields_blank() -> None:
    config = _config()
    parent = _parents().iloc[0]
    subject = TranslationResult("Cuộc họp với John Smith", 1, True, False)
    body = TranslationResult(
        "Vui lòng gửi email john.smith@example.com vào 2026-09-13.", 2, True, False
    )
    row = build_augmented_row(
        parent,
        subject,
        body,
        protect_text(parent["subject"]),
        protect_text(parent["body"]),
        config,
    )
    sample = validation_sample(pd.DataFrame([row]), _parents(), 1, 42)

    assert sample.loc[0, "translation_quality_decision"] == ""
    assert sample.loc[0, "note"] == ""


def test_split_validator_rejects_group_mismatch() -> None:
    frame = pd.DataFrame(
        [
            {
                "parent_id": "Enron:1",
                "parent_split": "train",
                "parent_group_id": "parent-group",
                "group_id": "different-group",
            }
        ]
    )
    with pytest.raises(ValueError, match="parent_group_id"):
        validate_parent_split_safety(frame)
