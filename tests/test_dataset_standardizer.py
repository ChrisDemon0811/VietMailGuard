from pathlib import Path

from vietmailguard.dataset_standardizer import (
    content_hash,
    extract_urls,
    is_container_artifact,
    is_empty_or_placeholder,
    load_dataset_config,
    normalize_field,
    normalize_text,
    resolve_label,
    template_hash,
)


ROOT = Path(__file__).resolve().parents[1]


def test_text_normalization_is_unicode_safe_and_collapses_whitespace() -> None:
    decomposed = "Vie\u0302\u0323t   Mail\n\tGuard"

    assert normalize_text(decomposed) == "Việt Mail Guard"


def test_label_mapping_requires_policy_or_curated_row_override() -> None:
    config = load_dataset_config(ROOT / "config" / "datasets.json")
    ceas = config["datasets"]["CEAS_08.csv"]
    override = config["policy"]["standardization"]["row_level_spam_override"]

    normal = resolve_label(
        0,
        ceas["raw_label_policy"],
        source="CEAS_08",
        original_row_id=2,
        curated_spam_keys=set(),
        spam_override=override,
    )
    unresolved = resolve_label(
        1,
        ceas["raw_label_policy"],
        source="CEAS_08",
        original_row_id=2,
        curated_spam_keys=set(),
        spam_override=override,
    )
    curated = resolve_label(
        1,
        ceas["raw_label_policy"],
        source="CEAS_08",
        original_row_id=2,
        curated_spam_keys={("CEAS_08", 2)},
        spam_override=override,
    )

    assert (normal.label, normal.provenance) == ("normal", "verified_legitimate_source")
    assert unresolved.label is None
    assert unresolved.policy_status == "review"
    assert (curated.label, curated.provenance) == (
        "spam",
        "high_confidence_curated_spam",
    )


def test_duplicate_hashing_groups_different_urls_as_one_template() -> None:
    first = "Buy now at https://offers.example/a"
    second = "Buy now at https://another.example/b"

    assert content_hash("Sale", first) != content_hash("Sale", second)
    assert template_hash("Sale", first) == template_hash("Sale", second)


def test_empty_and_placeholder_detection() -> None:
    placeholders = {"empty", "none", "null", "nan", "n/a", "na"}

    assert is_empty_or_placeholder(None, placeholders)
    assert is_empty_or_placeholder(" \n EMPTY\t", placeholders)
    assert not is_empty_or_placeholder("The body says empty but has content.", placeholders)
    assert normalize_field(" none ", placeholders) == ""
    assert is_container_artifact(
        "DON'T DELETE THIS MESSAGE",
        "This text is part of the internal format of your mail folder, and is not a real message.",
        ["this text is part of the internal format of your mail folder, and is not a real message"],
    )


def test_url_extraction_counts_distinct_occurrences_and_strips_punctuation() -> None:
    text = "See https://example.com/login, then visit www.example.org/path)."

    assert extract_urls(text) == ["https://example.com/login", "www.example.org/path"]
