"""Focused tests for the read-only Vietnamese corpus audit helpers."""

import pandas as pd

from scripts.audit_vietnamese_dataset import (
    digest_text,
    duplicate_summary,
    normalize_scalar,
    relaxed_template_text,
)


def test_normalize_scalar_handles_unicode_whitespace_and_missing_values():
    assert normalize_scalar("  Xin\n  chào  ") == "Xin chào"
    assert normalize_scalar(pd.NA) == ""


def test_relaxed_template_normalization_handles_spaced_urls():
    normal_url = relaxed_template_text("Xem http://example.com/offer ngay")
    spaced_url = relaxed_template_text("Xem http : / / example . com / deal ngay")

    assert normal_url == spaced_url == "xem <url> ngay"


def test_duplicate_summary_detects_conflicting_labels():
    hashes = pd.Series([digest_text("same"), digest_text("same"), digest_text("other")])
    labels = pd.Series([0, 1, 0])

    result = duplicate_summary(hashes, labels)

    assert result == {
        "groups": 1,
        "participating_rows": 2,
        "redundant_rows_beyond_first": 1,
        "conflicting_label_groups": 1,
        "conflicting_label_rows": 2,
    }
