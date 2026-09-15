"""Tests for Vietnamese-specific standardization behavior."""

import unicodedata
from pathlib import Path

from vietmailguard.dataset_standardizer import (
    content_hash,
    extract_malformed_urls,
    load_dataset_config,
    normalize_text,
    obfuscated_template_hash,
    parse_prefixed_subject,
)


ROOT = Path(__file__).resolve().parents[1]


def test_vietnamese_unicode_is_normalized_to_nfc() -> None:
    decomposed = "Vie\u0323\u0302t Nam"
    normalized = normalize_text(decomposed)

    assert normalized == "Việt Nam"
    assert unicodedata.is_normalized("NFC", normalized)


def test_subject_parsing_requires_reliable_line_boundary() -> None:
    subject, body, parsed = parse_prefixed_subject(
        "Chủ đề: Xác nhận cuộc họp\nNội dung cuộc họp lúc 9 giờ.",
        ["Chủ đề", "Tiêu đề"],
    )

    assert parsed is True
    assert subject == "Xác nhận cuộc họp"
    assert body == "Nội dung cuộc họp lúc 9 giờ."


def test_subject_prefix_without_boundary_keeps_complete_body() -> None:
    raw = "Chủ đề: Xác nhận cuộc họp Nội dung tiếp tục trên cùng một dòng."

    subject, body, parsed = parse_prefixed_subject(raw, ["Chủ đề", "Tiêu đề"])

    assert parsed is False
    assert subject == ""
    assert body == raw


def test_accented_and_unaccented_text_are_not_conflated() -> None:
    accented = normalize_text("Tài khoản của bạn")
    unaccented = normalize_text("Tai khoan cua ban")

    assert accented == "Tài khoản của bạn"
    assert unaccented == "Tai khoan cua ban"
    assert accented != unaccented
    assert content_hash("", accented) != content_hash("", unaccented)


def test_vietnamese_duplicate_hashing_handles_nfc_and_whitespace() -> None:
    first = "Tài khoản   của bạn\nđã bị khóa."
    second = "Ta\u0300i khoa\u0309n của bạn đã bị khóa."

    assert content_hash("", first) == content_hash("", second)


def test_obfuscated_url_detection_and_template_hashing() -> None:
    first = "Nhấp http : / / example . com / login để xác minh."
    second = "Nhấp https : / / other . net / account để xác minh."

    assert extract_malformed_urls(first) == ["http : / / example . com / login"]
    assert obfuscated_template_hash("", first) == obfuscated_template_hash("", second)


def test_vietnamese_mixed_positive_label_remains_review() -> None:
    config = load_dataset_config(ROOT / "config" / "datasets.json")
    policy = config["policy"]["vietnamese_standardization"]["dataset"][
        "raw_label_policy"
    ]

    assert policy["0"] == {
        "status": "normal",
        "label_status": "eligible",
        "training_eligible": True,
        "label_provenance": "translated_dataset_verified_normal",
    }
    assert policy["1"] == {
        "status": "review",
        "label_status": "review",
        "training_eligible": False,
        "label_provenance": "translated_dataset_mixed_positive",
    }
