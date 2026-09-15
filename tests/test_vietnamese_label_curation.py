"""Tests for conservative Vietnamese parent-assisted label curation."""

from pathlib import Path

from vietmailguard.vietnamese_label_curation import (
    ParentEvidence,
    curate_positive_row,
    load_curation_rules,
    parent_label_is_trusted,
)


ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict[str, object]:
    return load_curation_rules(ROOT / "config" / "vietnamese_label_rules.json")


def _row(body: str) -> dict[str, str]:
    return {"subject": "", "body": body, "raw_label": "1"}


def test_trusted_spam_parent_propagates_without_conflict() -> None:
    parent = ParentEvidence(
        parent_id="Enron:10",
        source="Enron",
        label="spam",
        label_provenance="high_confidence_curated_spam",
        link_confidence="high",
        link_status="linked_high_confidence_evidence",
    )
    result = curate_positive_row(
        _row(
            "Ưu đãi phần mềm dành cho doanh nghiệp. Mua ngay với giá chỉ 20 USD "
            "và được giảm giá miễn phí vận chuyển."
        ),
        parent,
        _policy(),
    )

    assert result["label"] == "spam"
    assert result["label_status"] == "evidence_confirmed"
    assert result["training_eligible"] is True
    assert result["label_provenance"] == (
        "translated_from_high_confidence_curated_spam_parent"
    )


def test_trusted_phishing_parent_propagates_without_conflict() -> None:
    parent = ParentEvidence(
        parent_id="Nazario:7",
        source="Nazario",
        label="phishing",
        label_provenance="nazario_phishing",
        link_confidence="high",
        link_status="linked_exact_provenance",
    )
    result = curate_positive_row(
        _row(
            "Tài khoản của bạn đã bị khóa. Đăng nhập và xác minh tài khoản "
            "bằng mật khẩu ngay lập tức."
        ),
        parent,
        _policy(),
    )

    assert result["label"] == "phishing"
    assert result["label_status"] == "evidence_confirmed"
    assert result["training_eligible"] is True


def test_untrusted_parent_label_does_not_propagate() -> None:
    parent = ParentEvidence(
        parent_id="Enron:11",
        source="Enron",
        label="spam",
        label_provenance="raw_label_review",
        link_confidence="high",
        link_status="linked_high_confidence_evidence",
    )
    result = curate_positive_row(
        _row("Mua phần mềm ngay hôm nay với chương trình giảm giá đặc biệt."),
        parent,
        _policy(),
    )

    assert parent_label_is_trusted(parent, _policy()) is False
    assert result["label"] == "review"
    assert result["training_eligible"] is False


def test_parent_that_is_still_review_never_propagates() -> None:
    parent = ParentEvidence(
        parent_id="Enron:13",
        source="Enron",
        label="review",
        label_provenance="translated_dataset_mixed_positive",
        link_confidence="high",
        link_status="linked_high_confidence_evidence",
    )
    result = curate_positive_row(
        _row("Mua dịch vụ ngay hôm nay với chương trình giảm giá đặc biệt."),
        parent,
        _policy(),
    )

    assert result["label"] == "review"
    assert result["label_status"] == "review"
    assert result["training_eligible"] is False


def test_strong_phishing_conflict_blocks_trusted_spam_parent() -> None:
    parent = ParentEvidence(
        parent_id="Enron:12",
        source="Enron",
        label="spam",
        label_provenance="high_confidence_curated_spam",
        link_confidence="high",
        link_status="linked_high_confidence_evidence",
    )
    result = curate_positive_row(
        _row(
            "Tài khoản ngân hàng đã bị khóa. Khẩn cấp đăng nhập bằng tên đăng nhập "
            "và mật khẩu để xác minh tài khoản."
        ),
        parent,
        _policy(),
    )

    assert result["evidence_group"] in {
        "credential_phishing",
        "account_impersonation",
    }
    assert result["label"] == "review"
    assert result["training_eligible"] is False


def test_single_keyword_is_only_review_evidence() -> None:
    result = curate_positive_row(
        _row("Bản tin nội bộ có hướng dẫn unsubscribe dành cho quản trị viên."),
        ParentEvidence(),
        _policy(),
    )

    assert result["label"] == "review"
    assert result["proposed_label"] == "review"
    assert result["training_eligible"] is False
    assert result["human_decision"] == ""
