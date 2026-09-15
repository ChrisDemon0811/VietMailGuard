from __future__ import annotations

import pandas as pd
import pytest

from vietmailguard.robustness_evaluation import (
    TRANSFORMATION_TYPES,
    apply_transformation,
    build_robustness_variants,
    build_short_form_challenge,
    challenge_contamination_audit,
    contamination_audit,
    length_bucket,
    remove_accents,
)


def _seed_row() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": "data_vi:1",
            "parent_id": "Enron:1",
            "source": "data_vi",
            "label": "normal",
            "language": "vi",
            "subject": "Cập nhật cuộc họp",
            "body": "Tài khoản dự án không thay đổi. Cuộc họp diễn ra ngày mai.",
            "training_eligible": True,
            "content_hash": "h1",
            "group_id": "g1",
            "normalized_content_hash": "n1",
            "template_hash": "t1",
            "final_group_id": "f1",
        }
    ])


def test_vietnamese_transformations_preserve_lineage() -> None:
    variants = build_robustness_variants(_seed_row())
    assert len(variants) == len(TRANSFORMATION_TYPES)
    assert variants["parent_sample_id"].nunique() == 1
    assert variants["parent_final_group_id"].nunique() == 1
    assert set(variants["transformation_type"]) == set(TRANSFORMATION_TYPES)
    assert variants["robustness_id"].is_unique


def test_accent_and_obfuscation_transformations() -> None:
    assert remove_accents("Tài khoản đã bị khóa") == "Tai khoan da bi khoa"
    assert "t@i kh0an" in apply_transformation(
        "Tài khoản đã bị khóa", "character_substitution"
    ).casefold()
    spaced = apply_transformation("Tài khoản đã bị khóa", "spacing_obfuscation")
    assert "T à i" in spaced
    assert "account" in apply_transformation(
        "Tài khoản cần xác minh", "mixed_language"
    ).casefold()


def test_contamination_audit_rejects_known_group() -> None:
    training = _seed_row().copy()
    with pytest.raises(RuntimeError, match="overlaps production training"):
        contamination_audit(_seed_row(), training)


def test_contamination_audit_accepts_disjoint_lineage() -> None:
    training = _seed_row().assign(
        id="data_vi:2", parent_id="Enron:2", content_hash="h2", group_id="g2",
        normalized_content_hash="n2", template_hash="t2", final_group_id="f2",
    )
    assert not any(contamination_audit(_seed_row(), training).values())


def test_short_challenge_is_balanced_and_bucketed() -> None:
    challenge = build_short_form_challenge()
    assert len(challenge) == 90
    assert challenge["challenge_id"].is_unique
    assert challenge["label"].value_counts().to_dict() == {
        "normal": 30, "spam": 30, "phishing": 30
    }
    assert challenge["language"].value_counts().to_dict() == {"en": 45, "vi": 45}
    assert challenge["length_bucket"].value_counts().to_dict() == {
        "very_short": 30, "short": 30, "normal_length": 30
    }
    assert all(
        length_bucket(int(row.token_count)) == row.length_bucket
        for row in challenge.itertuples()
    )


def test_short_challenge_has_no_matching_training_hash() -> None:
    challenge = build_short_form_challenge()
    training = pd.DataFrame({"content_hash": ["different"], "template_hash": ["different"]})
    assert challenge_contamination_audit(challenge, training) == {
        "content_hash": 0, "template_hash": 0
    }
