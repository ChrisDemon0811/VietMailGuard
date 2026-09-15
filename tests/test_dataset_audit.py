from pathlib import Path

import pandas as pd

from vietmailguard.dataset_audit import audit_repository, normalize_for_comparison


def test_template_normalization_masks_urls_and_emails() -> None:
    first = "Contact Alice@example.com at https://example.com/login NOW"
    second = "contact bob@example.org at http://other.test/login now"
    assert normalize_for_comparison(first, template=True) == normalize_for_comparison(second, template=True)


def test_audit_preserves_raw_file_and_writes_reports(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    reports_dir = tmp_path / "reports"
    raw_dir.mkdir()
    source = raw_dir / "sample.csv"
    source.write_text("subject,body,label\nHello,World,unknown\n", encoding="utf-8")
    before = source.read_bytes()

    result = audit_repository(raw_dir, reports_dir, sample_size=1)

    assert source.read_bytes() == before
    assert result["datasets"]["sample.csv"]["label_mapping"] == {}
    assert (reports_dir / "dataset_audit.csv").exists()
    labels = pd.read_csv(reports_dir / "dataset_labels.csv")
    assert labels.loc[0, "mapping_status"] == "unmapped_pending_provenance_review"

