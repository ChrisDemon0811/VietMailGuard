"""Tests for evidence-only English/Vietnamese overlap linking."""

from copy import deepcopy
from pathlib import Path

import pandas as pd

from vietmailguard.cross_language_overlap import (
    apply_cross_language_split_groups,
    audit_cross_language_overlap,
    extract_loose_emails,
    extract_url_anchors,
)
from vietmailguard.dataset_standardizer import load_dataset_config


ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict[str, object]:
    config = load_dataset_config(ROOT / "config" / "datasets.json")
    return config["policy"]["cross_language_overlap"]


def _english_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "Enron:10",
                "source": "Enron",
                "sender": "vince.kaminski@enron.com",
                "receiver": "angelika@example.org",
                "date": "2001-09-04",
                "subject": "Powerisk 2001 invitation",
                "body": (
                    "Angelika invited Vince Kaminski to Powerisk in London. "
                    "Contact vince.kaminski@enron.com and visit "
                    "https://powerisk.example.org/event/2001 reference PX-2001."
                ),
                "label": "normal",
                "group_id": "english_group_10",
            },
            {
                "id": "Enron:11",
                "source": "Enron",
                "sender": "other@enron.com",
                "receiver": "team@enron.com",
                "date": "2000-01-01",
                "subject": "Routine meeting",
                "body": "General project meeting notes.",
                "label": "normal",
                "group_id": "english_group_11",
            },
        ]
    )


def _vietnamese_frame(raw_label: int = 0) -> pd.DataFrame:
    text = (
        "Chủ đề: lời mời Powerisk 2001 Angelika mời Vince Kaminski tới London. "
        "Liên hệ vince . kaminski @ enron . com và xem "
        "https : / / powerisk . example . org / event / 2001 mã PX-2001."
    )
    return pd.DataFrame(
        [
            {
                "id": "data_vi:10",
                "source": "data_vi",
                "body": text,
                "original_text": text,
                "raw_label": raw_label,
                "parent_id": "",
                "translation_source_id": "",
                "group_id": "vietnamese_group_10",
            }
        ]
    )


def test_obfuscated_email_and_url_anchors_are_comparable() -> None:
    assert extract_loose_emails("vince . kaminski @ enron . com") == frozenset(
        {"vince.kaminski@enron.com"}
    )
    assert extract_url_anchors(
        "https : / / powerisk . example . org / event / 2001"
    ) == frozenset({"https://powerisk.example.org/event/2001"})


def test_strong_independent_evidence_creates_parent_link() -> None:
    result = audit_cross_language_overlap(
        _english_frame(), _vietnamese_frame(), _policy()
    )

    assert result["high_confidence_links"] == 1
    linked = result["updated_vietnamese"].iloc[0]
    assert linked["parent_id"] == "Enron:10"
    assert str(linked["translation_source_id"]).startswith("xlang_evidence_")


def test_label_incompatibility_prevents_automatic_parent_link() -> None:
    result = audit_cross_language_overlap(
        _english_frame(), _vietnamese_frame(raw_label=1), _policy()
    )

    assert result["high_confidence_links"] == 0
    assert result["updated_vietnamese"].iloc[0]["parent_id"] == ""


def test_future_split_group_is_shared_by_linked_pair() -> None:
    result = audit_cross_language_overlap(
        _english_frame(), _vietnamese_frame(), _policy()
    )
    english, vietnamese = apply_cross_language_split_groups(
        _english_frame(), result["updated_vietnamese"], result["overlap"]
    )

    english_group = english.loc[english["id"].eq("Enron:10"), "cross_language_group_id"].iat[0]
    vietnamese_group = vietnamese.loc[
        vietnamese["id"].eq("data_vi:10"), "cross_language_group_id"
    ].iat[0]
    assert english_group == vietnamese_group


def test_rerun_clears_stale_evidence_generated_link() -> None:
    initial = audit_cross_language_overlap(
        _english_frame(), _vietnamese_frame(), _policy()
    )
    stricter_policy = deepcopy(_policy())
    stricter_policy["high_confidence_points"] = 999.0

    rerun = audit_cross_language_overlap(
        _english_frame(), initial["updated_vietnamese"], stricter_policy
    )

    updated = rerun["updated_vietnamese"].iloc[0]
    assert updated["parent_id"] == ""
    assert updated["translation_source_id"] == ""
