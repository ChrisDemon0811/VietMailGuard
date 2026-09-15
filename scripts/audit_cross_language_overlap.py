"""Audit English/Vietnamese translation overlap without training a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.cross_language_overlap import audit_cross_language_overlap  # noqa: E402
from vietmailguard.dataset_standardizer import load_dataset_config  # noqa: E402


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def raw_hashes(raw_dir: Path) -> dict[str, str]:
    """Hash every raw file so the audit can prove immutability."""
    return {
        path.name: file_sha256(path)
        for path in sorted(raw_dir.iterdir())
        if path.is_file()
    }


def _reason_counts(overlap: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in overlap["overlap_reason"]:
        counts.update(item for item in str(value).split(";") if item)
    return counts


def render_report(
    result: dict[str, object],
    english: pd.DataFrame,
    vietnamese: pd.DataFrame,
    english_hash: str,
    vietnamese_hash_before: str,
    vietnamese_hash_after: str,
    raw_unchanged: bool,
) -> str:
    """Render a factual leakage-audit report."""
    overlap = result["overlap"]
    linked = overlap.loc[
        overlap["review_status"].isin(
            {"linked_exact_provenance", "linked_high_confidence_evidence"}
        )
    ]
    high_source_counts = linked["english_source"].value_counts().to_dict()
    reasons = _reason_counts(overlap)
    lines = [
        "# Cross-language leakage audit",
        "",
        "## Kết luận",
        "",
        f"Audit đã tạo **{result['candidate_rows']:,} candidate pairs** cho **{result['vietnamese_rows_with_candidates']:,} / {len(vietnamese):,}** Vietnamese rows.",
        f"Có **{result['high_confidence_vietnamese_rows']:,}** Vietnamese rows được liên kết bằng evidence mạnh. Có **{result['medium_only_vietnamese_rows']:,}** rows chỉ có candidate mức medium, **{result['uncertain_only_vietnamese_rows']:,}** rows chỉ có candidate mức thấp và **{result['vietnamese_rows_without_candidates']:,}** rows không có candidate đủ ngưỡng.",
        "",
        "Không row nào bị xóa. Similarity/evidence không được dùng làm ground-truth label.",
        "",
        "## Provenance trước matching",
        "",
        f"- English Master: **{len(english):,}** rows; sources: `{json.dumps(english['source'].value_counts().to_dict(), ensure_ascii=False)}`.",
        f"- Vietnamese Translated Master: **{len(vietnamese):,}** rows; `source=data_vi`, `language=vi`, `data_origin=translated`.",
        f"- Exact parent/provenance links có sẵn trước audit: **{result['exact_provenance_links']:,}**.",
        "- Vietnamese master ban đầu không có preserved source ID đủ để nối trực tiếp với English Master. Vì vậy link mới không được gọi là exact provenance.",
        "",
        "## Candidate results",
        "",
        "| Category | Pair rows | Distinct Vietnamese rows | Interpretation |",
        "| --- | ---: | ---: | --- |",
        f"| Exact provenance link | {result['exact_provenance_links']:,} | {result['exact_provenance_links']:,} | Preserved parent metadata; none observed in the initial input |",
        f"| High-confidence evidence link | {result['high_confidence_links']:,} | {result['high_confidence_vietnamese_rows']:,} | `parent_id` and `translation_source_id` populated |",
        f"| Medium candidate pairs | {result['medium_candidates']:,} | {result['medium_only_vietnamese_rows']:,} | Needs review; no parent link |",
        f"| Low/uncertain candidate pairs | {result['uncertain_candidates']:,} | {result['uncertain_only_vietnamese_rows']:,} | Insufficient evidence; no parent link |",
        "",
        "### High-confidence links by English source",
        "",
        "| English source | Links |",
        "| --- | ---: |",
    ]
    for source, count in high_source_counts.items():
        lines.append(f"| {source} | {count:,} |")
    if not high_source_counts:
        lines.append("| None | 0 |")

    lines.extend(
        [
            "",
            "### Evidence signals across candidate pairs",
            "",
            "| Signal | Candidate pairs |",
            "| --- | ---: |",
        ]
    )
    for reason, count in reasons.most_common():
        lines.append(f"| {reason} | {count:,} |")

    lines.extend(
        [
            "",
            "## Method",
            "",
            "Candidate generation dùng language-invariant anchors: normalized email addresses, standard/obfuscated URLs, preserved alphanumeric IDs, rare ASCII proper nouns/tokens, numbers/dates, subject-prefix region và Enron identity. Common anchors bị giới hạn theo document frequency để tránh ghép mọi email chỉ vì boilerplate Enron.",
            "",
            "Một link chỉ được tự động tạo khi English candidate thuộc source Enron đã được provenance audit hỗ trợ, đứng hạng 1, vượt ngưỡng điểm cao, cách candidate thứ hai đủ xa, có ít nhất hai loại evidence, có length ratio hợp lý và raw-label compatibility không mâu thuẫn với English class. Label compatibility chỉ là safety check, không phải bằng chứng parent relation. Candidate từ source khác vẫn được báo cáo nhưng không tự động điền parent.",
            "",
            "Không chạy multilingual semantic embedding: config không chỉ định local model và repository không có provenance-preserving embedding artifact. Thêm model tải từ mạng sẽ làm audit không còn offline/reproducible. Nếu sau này dùng semantic similarity, kết quả vẫn chỉ được flag để review.",
            "",
            "## Cross-language grouping và split policy",
            "",
            "Các link high-confidence được ghi vào `parent_id` và `translation_source_id` của Vietnamese master. ID nhóm có prefix `xlang_evidence_`, thể hiện rõ đây là evidence-derived link chứ không phải exact provenance.",
            "",
            "Khi tạo bilingual master/split, phải tạo `cross_language_group_id` cho cả English parent và Vietnamese translation từ link manifest, rồi dùng cột đó làm group constraint. Không được split trực tiếp bằng `group_id` nội ngôn ngữ cũ.",
            "",
            "Medium/uncertain candidates phải được review trước split. Nếu chưa review, không dùng Vietnamese row đó trong final Vietnamese test; cách bảo thủ là giữ khỏi evaluation hoặc gộp cùng candidate English parent khi xây split thử nghiệm.",
            "",
            "## Input integrity",
            "",
            f"- English Master SHA-256: `{english_hash}`.",
            f"- Vietnamese Master SHA-256 before link population: `{vietnamese_hash_before}`.",
            f"- Vietnamese Master SHA-256 after link population: `{vietnamese_hash_after}`.",
            f"- All `data/raw/` hashes unchanged: **{str(raw_unchanged).lower()}**.",
            "",
            "Vietnamese processed hash thay đổi có chủ đích vì chỉ hai cột linkage được điền cho high-confidence rows. Nội dung, label và provenance gốc không bị sửa.",
            "",
            "Không có training, merge hoặc xóa dữ liệu trong bước này.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "datasets.json"
    )
    parser.add_argument(
        "--no-populate-links",
        action="store_true",
        help="Write reports without updating Vietnamese parent/link columns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_dataset_config(args.config)
    policy = config["policy"]["cross_language_overlap"]
    english_path = PROJECT_ROOT / policy["english_input"]
    vietnamese_path = PROJECT_ROOT / policy["vietnamese_input"]
    output_path = PROJECT_ROOT / policy["output"]
    report_path = PROJECT_ROOT / policy["report"]

    english_hash = file_sha256(english_path)
    vietnamese_hash_before = file_sha256(vietnamese_path)
    raw_before = raw_hashes(PROJECT_ROOT / "data" / "raw")
    english = pd.read_csv(english_path, keep_default_na=False)
    vietnamese = pd.read_csv(vietnamese_path, keep_default_na=False)
    result = audit_cross_language_overlap(english, vietnamese, policy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["overlap"].to_csv(output_path, index=False, encoding="utf-8")
    if policy["populate_high_confidence_parent_links"] and not args.no_populate_links:
        temporary_path = vietnamese_path.with_suffix(".links.tmp.csv")
        result["updated_vietnamese"].to_csv(
            temporary_path, index=False, encoding="utf-8"
        )
        temporary_path.replace(vietnamese_path)

    vietnamese_hash_after = file_sha256(vietnamese_path)
    raw_after = raw_hashes(PROJECT_ROOT / "data" / "raw")
    raw_unchanged = raw_before == raw_after
    if not raw_unchanged:
        raise RuntimeError("A raw dataset changed during cross-language audit")
    if file_sha256(english_path) != english_hash:
        raise RuntimeError("English Master changed during cross-language audit")

    report_path.write_text(
        render_report(
            result,
            english,
            vietnamese,
            english_hash,
            vietnamese_hash_before,
            vietnamese_hash_after,
            raw_unchanged,
        ),
        encoding="utf-8",
    )

    print("Cross-language overlap audit complete")
    print(f"Candidate pairs: {result['candidate_rows']:,}")
    print(
        "Vietnamese rows with candidates: "
        f"{result['vietnamese_rows_with_candidates']:,}"
    )
    print(f"Exact provenance links: {result['exact_provenance_links']:,}")
    print(f"High-confidence evidence links: {result['high_confidence_links']:,}")
    print(f"Medium candidate pairs: {result['medium_candidates']:,}")
    print(f"Uncertain candidate pairs: {result['uncertain_candidates']:,}")
    print(f"Raw files unchanged: {raw_unchanged}")
    print(output_path.relative_to(PROJECT_ROOT).as_posix())
    print(report_path.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
