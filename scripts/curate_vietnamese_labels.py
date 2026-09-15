"""Curate translated Vietnamese positive rows using audited evidence only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.dataset_standardizer import load_dataset_config  # noqa: E402
from vietmailguard.file_utils import sha256_file  # noqa: E402
from vietmailguard.vietnamese_label_curation import (  # noqa: E402
    curate_vietnamese_dataset,
    load_curation_rules,
)


def raw_hashes(raw_dir: Path) -> dict[str, str]:
    """Hash all raw files to verify that curation is non-destructive."""
    return {
        path.name: sha256_file(path)
        for path in sorted(raw_dir.iterdir())
        if path.is_file()
    }


def _markdown_distribution(series: pd.Series, first_header: str) -> list[str]:
    counts = series.fillna("").astype(str).replace("", "(blank)").value_counts()
    lines = [f"| {first_header} | Rows |", "| --- | ---: |"]
    lines.extend(f"| {key} | {value:,} |" for key, value in counts.items())
    return lines


def render_report(
    curated: pd.DataFrame,
    *,
    raw_positive_rows: int,
    standardized_positive_rows: int,
    exact_duplicate_positive_rows_removed: int,
    raw_unchanged: bool,
    input_hashes_unchanged: bool,
) -> str:
    """Render the factual curation report from generated artifacts."""
    positives = curated.loc[curated["raw_label"].astype(str).eq("1")].copy()
    linked = positives.loc[positives["parent_id"].astype(str).ne("")]
    evidence_spam = positives.loc[
        positives["label"].eq("spam")
        & positives["label_status"].eq("evidence_confirmed")
    ]
    evidence_phishing = positives.loc[
        positives["label"].eq("phishing")
        & positives["label_status"].eq("evidence_confirmed")
    ]
    needs_review = positives.loc[positives["label"].eq("review")]
    excluded = positives.loc[positives["label"].eq("exclude")]
    unresolved_fraud = needs_review.loc[
        needs_review["evidence_group"].isin(
            {"financial_fraud", "lottery_scam", "advance_fee_fraud"}
        )
    ]
    human_confirmed = positives["label_status"].eq("human_confirmed").sum()
    rule_only_spam = needs_review["proposed_label"].eq("spam").sum()
    rule_only_phishing = needs_review["proposed_label"].eq("phishing").sum()

    lines = [
        "# Vietnamese label curation report",
        "",
        "## Kết luận",
        "",
        (
            f"Nguồn thô có **{raw_positive_rows:,}** rows `raw_label=1`; sau bước "
            f"standardization/deduplication còn **{standardized_positive_rows:,}** rows positive để curate. "
            f"Có **{len(evidence_spam):,}** spam và **{len(evidence_phishing):,}** phishing "
            "được xác nhận bằng evidence parent; các row này không được gọi là manually confirmed."
        ),
        (
            f"Còn **{len(needs_review):,}** rows cần human review và **{len(excluded):,}** rows bị loại "
            "theo tiêu chí chất lượng khách quan. Rule-assisted proposal không được dùng làm ground truth."
        ),
        "",
        "Dataset hiện vẫn là **translated Vietnamese**, không phải native Vietnamese ground truth.",
        "",
        "## Summary",
        "",
        "| Measure | Rows |",
        "| --- | ---: |",
        f"| Raw positive rows in `data_vi.csv` | {raw_positive_rows:,} |",
        f"| Positive exact-duplicate copies removed during prior standardization | {exact_duplicate_positive_rows_removed:,} |",
        f"| Standardized positive rows curated | {standardized_positive_rows:,} |",
        f"| Linked to audited English parent | {len(linked):,} |",
        f"| Evidence-confirmed spam | {len(evidence_spam):,} |",
        f"| Evidence-confirmed phishing | {len(evidence_phishing):,} |",
        f"| Human-confirmed decisions present | {int(human_confirmed):,} |",
        f"| Needs human review | {len(needs_review):,} |",
        f"| Excluded | {len(excluded):,} |",
        f"| Unresolved fraud/scam | {len(unresolved_fraud):,} |",
        f"| Rule-only spam proposals still under review | {int(rule_only_spam):,} |",
        f"| Rule-only phishing proposals still under review | {int(rule_only_phishing):,} |",
        "",
        "## Evidence hierarchy and decisions",
        "",
        "1. Parent labels are transferable only when both the cross-language link status and the English label provenance are allowlisted.",
        "2. A trusted spam parent is not propagated when Vietnamese content has strong credential, impersonation, lottery, or advance-fee evidence that conflicts with spam.",
        "3. Content rules create evidence groups and proposals only. A keyword or rule proposal never makes a row training-eligible.",
        "4. Blank `human_decision` means no human confirmation exists. Reruns preserve nonblank human decisions from the review file.",
        "5. Version 2 model predictions and held-out test predictions were not used.",
        "",
        "## Linked positives by English parent source",
        "",
    ]
    if linked.empty:
        lines.extend(["| English source | Rows |", "| --- | ---: |", "| None | 0 |"])
    else:
        lines.extend(_markdown_distribution(linked["linked_parent_source"], "English source"))

    lines.extend(["", "## Positive label provenance distribution", ""])
    lines.extend(_markdown_distribution(positives["label_provenance"], "Label provenance"))
    lines.extend(["", "## Evidence-group distribution", ""])
    lines.extend(_markdown_distribution(positives["evidence_group"], "Evidence group"))
    lines.extend(["", "## Final positive-label distribution", ""])
    lines.extend(_markdown_distribution(positives["label"], "Label"))

    lines.extend(
        [
            "",
            "## Training eligibility",
            "",
            (
                "Only `label_status=evidence_confirmed` rows transferred from a trusted English parent, "
                "or future `human_confirmed` spam/phishing rows, are eligible among raw positives. "
                "Every `review` and `exclude` row has `training_eligible=false`."
            ),
            "",
            "The current Vietnamese data is **not yet sufficient for bilingual three-class training**: "
            "the evidence-confirmed positive set is small and has no independently curated Vietnamese phishing class. "
            "Human review and/or a dedicated, provenance-documented Vietnamese phishing corpus is required.",
            "",
            "## Integrity",
            "",
            f"- All `data/raw/` hashes unchanged during curation: **{str(raw_unchanged).lower()}**.",
            f"- English and standardized Vietnamese input hashes unchanged: **{str(input_hashes_unchanged).lower()}**.",
            "- No model was trained and no Version 1 artifact was modified.",
            "- The curated output retains original standardized text, identifiers, parent links, and translation linkage fields.",
            "",
            "## Output artifacts",
            "",
            "- `data/curated/v2/vietnamese_positive_review.csv`",
            "- `data/curated/v2/Vietnamese_Curated_Dataset.csv`",
            "- `config/vietnamese_label_rules.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    config_path = PROJECT_ROOT / "config" / "datasets.json"
    config = load_dataset_config(config_path)
    settings = config["policy"]["vietnamese_label_curation"]

    vietnamese_path = PROJECT_ROOT / settings["vietnamese_input"]
    english_path = PROJECT_ROOT / settings["english_input"]
    overlap_path = PROJECT_ROOT / settings["overlap_input"]
    rules_path = PROJECT_ROOT / settings["rules"]
    review_path = PROJECT_ROOT / settings["review_output"]
    curated_path = PROJECT_ROOT / settings["curated_output"]
    report_path = PROJECT_ROOT / settings["report"]

    required_paths = [vietnamese_path, english_path, overlap_path, rules_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required curation inputs are missing: {missing}")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_before = raw_hashes(raw_dir)
    input_hashes_before = {
        "vietnamese": sha256_file(vietnamese_path),
        "english": sha256_file(english_path),
        "overlap": sha256_file(overlap_path),
    }

    vietnamese = pd.read_csv(vietnamese_path, keep_default_na=False)
    english = pd.read_csv(english_path, keep_default_na=False)
    overlap = pd.read_csv(overlap_path, keep_default_na=False)
    policy = load_curation_rules(rules_path)
    existing_review = None
    if settings["preserve_existing_human_decisions"] and review_path.exists():
        existing_review = pd.read_csv(review_path, keep_default_na=False, dtype=str)

    curated, review = curate_vietnamese_dataset(
        vietnamese, english, overlap, policy, existing_review
    )

    review_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    curated.to_csv(curated_path, index=False, encoding="utf-8-sig")

    raw_source = pd.read_csv(
        PROJECT_ROOT / config["policy"]["vietnamese_standardization"]["input"],
        keep_default_na=False,
    )
    raw_label_column = config["policy"]["vietnamese_standardization"]["dataset"][
        "field_mapping"
    ]["raw_label"]
    raw_positive_rows = int(raw_source[raw_label_column].astype(str).eq("1").sum())
    standardized_positive_rows = int(vietnamese["raw_label"].astype(str).eq("1").sum())
    removed_path = PROJECT_ROOT / config["policy"]["vietnamese_standardization"][
        "removed_rows_report"
    ]
    removed = pd.read_csv(removed_path, keep_default_na=False)
    exact_duplicate_positive_rows_removed = int(
        (
            removed["raw_label"].astype(str).eq("1")
            & removed["reason"].eq("exact_duplicate")
        ).sum()
    )

    raw_after = raw_hashes(raw_dir)
    input_hashes_after = {
        "vietnamese": sha256_file(vietnamese_path),
        "english": sha256_file(english_path),
        "overlap": sha256_file(overlap_path),
    }
    raw_unchanged = raw_before == raw_after
    input_hashes_unchanged = input_hashes_before == input_hashes_after
    if not raw_unchanged:
        raise RuntimeError("A raw dataset changed during Vietnamese label curation")
    if not input_hashes_unchanged:
        raise RuntimeError("A curation input changed during Vietnamese label curation")

    report_path.write_text(
        render_report(
            curated,
            raw_positive_rows=raw_positive_rows,
            standardized_positive_rows=standardized_positive_rows,
            exact_duplicate_positive_rows_removed=exact_duplicate_positive_rows_removed,
            raw_unchanged=raw_unchanged,
            input_hashes_unchanged=input_hashes_unchanged,
        ),
        encoding="utf-8",
    )

    positives = curated[curated["raw_label"].astype(str).eq("1")]
    print("Vietnamese positive-label curation complete")
    print(f"Raw positive rows: {raw_positive_rows:,}")
    print(f"Standardized positive rows: {standardized_positive_rows:,}")
    print(f"Linked positives: {positives['parent_id'].astype(str).ne('').sum():,}")
    print("Final positive labels:")
    print(json.dumps(positives["label"].value_counts().to_dict(), ensure_ascii=False))
    print("Positive label statuses:")
    print(json.dumps(positives["label_status"].value_counts().to_dict(), ensure_ascii=False))
    print(f"Raw files unchanged: {raw_unchanged}")
    print(review_path.relative_to(PROJECT_ROOT).as_posix())
    print(curated_path.relative_to(PROJECT_ROOT).as_posix())
    print(report_path.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
