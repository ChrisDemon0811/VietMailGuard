from __future__ import annotations

from copy import deepcopy

from app.i18n import (
    action_label,
    class_label,
    load_translations,
    risk_label,
    t,
    translation_key_difference,
)


def test_english_and_vietnamese_catalogs_have_the_same_keys() -> None:
    assert translation_key_difference() == {
        "missing_in_en": set(),
        "missing_in_vi": set(),
    }


def test_required_translation_values_are_available() -> None:
    assert t("email_analyzer_title", "en") == "Quick Analyzer"
    assert t("email_analyzer_title", "vi") == "Phân tích nhanh"
    assert class_label("phishing", "en") == "Phishing"
    assert class_label("phishing", "vi") == "Lừa đảo"
    assert action_label("QUARANTINE", "vi") == "Cách ly Email"
    assert risk_label("CRITICAL", "vi") == "NGHIÊM TRỌNG"
    assert t("support_supported", "en") == "Supported"
    assert t("support_experimental", "vi") == "Hỗ trợ ở mức thử nghiệm"
    assert "không có mẫu phishing" in t("no_vietnamese_phishing_examples", "vi")
    assert "Native Vietnamese benchmark" in t("native_benchmark_unavailable", "en")


def test_missing_translation_key_falls_back_safely() -> None:
    assert t("key_that_does_not_exist", "vi") == "key_that_does_not_exist"
    assert t("email_analyzer_title", "unsupported") == "Quick Analyzer"


def test_localizing_result_does_not_change_internal_prediction() -> None:
    result = {"prediction": "spam", "recommended_action": "MOVE_TO_SPAM"}
    original = deepcopy(result)
    assert class_label(result["prediction"], "en") == "Spam"
    assert class_label(result["prediction"], "vi") == "Thư rác"
    assert result == original
    assert result["prediction"] in {"normal", "spam", "phishing"}


def test_translation_catalog_contains_only_strings() -> None:
    catalog = load_translations()
    assert all(isinstance(value, str) for language in catalog.values() for value in language.values())
