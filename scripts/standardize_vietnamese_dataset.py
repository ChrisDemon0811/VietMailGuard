"""Standardize the audited translated Vietnamese dataset for VietMailGuard V2."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.dataset_standardizer import (  # noqa: E402
    load_dataset_config,
    standardize_vietnamese_dataset,
)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for an immutable raw input."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def render_report(result: dict[str, object], raw_hash: str) -> str:
    """Render the factual standardization summary as Markdown."""
    raw_labels = result["input_raw_label_distribution"]
    output_labels = result["output_label_distribution"]
    output_statuses = result["output_label_status_distribution"]
    lines = [
        "# Vietnamese translated dataset standardization report",
        "",
        "## Kết quả",
        "",
        f"- Raw input rows: **{result['input_rows']:,}**",
        f"- Rows after empty checks: **{result['candidate_rows']:,}**",
        f"- Final standardized rows: **{result['output_rows']:,}**",
        f"- Empty/placeholder rows removed: **{result['empty_rows_removed']:,}**",
        f"- Exact normalized duplicate copies removed: **{result['exact_duplicates_removed']:,}**",
        f"- Conflicting-template rows removed: **{result['conflicting_rows_removed']:,}**",
        f"- Subject rows parsed with an unambiguous line boundary: **{result['subject_rows_parsed']:,}**",
        "",
        "Không có training hoặc merge với English Master Dataset trong bước này.",
        "",
        "## Label policy",
        "",
        "| Raw label | Input rows | Standard label | Label status | Training eligible | Provenance |",
        "| --- | ---: | --- | --- | --- | --- |",
        f"| 0 | {raw_labels.get('0', 0):,} | normal | eligible | true | translated_dataset_verified_normal |",
        f"| 1 | {raw_labels.get('1', 0):,} | review | review | false | translated_dataset_mixed_positive |",
        "",
        "Raw label 1 vẫn là `review`; pipeline không tự động đổi thành spam hoặc phishing.",
        "",
        "### Final distribution",
        "",
        "| Label | Rows |",
        "| --- | ---: |",
    ]
    for label, count in output_labels.items():
        lines.append(f"| {label} | {count:,} |")
    lines.extend(["", "| Label status | Rows |", "| --- | ---: |"]) 
    for status, count in output_statuses.items():
        lines.append(f"| {status} | {count:,} |")

    lines.extend(
        [
            "",
            "## Schema và provenance",
            "",
            "Output là superset tương thích với English Master Dataset. Các cột chung `id`, `sender`, `receiver`, `date`, `subject`, `body`, `label`, `raw_label`, `source`, `language`, `url_count`, `content_hash` và `group_id` được giữ nguyên ý nghĩa.",
            "",
            "Các cột V2 bổ sung gồm `data_origin`, `label_status`, `label_provenance`, `training_eligible`, `original_text`, `normalized_text`, `parent_id`, `translation_source_id`, `extracted_urls`, `malformed_url_count` và `malformed_urls`.",
            "",
            "- `language = vi` cho mọi row.",
            "- `data_origin = translated` cho mọi row theo audit V2.1 và AGENTS.md.",
            "- `source = data_vi`; `raw_label` được giữ nguyên.",
            "- `sender`, `receiver` và `date` để trống vì raw dataset không có các trường này.",
            "- `parent_id` và `translation_source_id` để trống cho đến khi V2.3 có evidence linkage.",
            "- `original_text` giữ nguyên chuỗi nguồn. `body` là canonical NFC/whitespace-normalized text khi không thể tách subject chắc chắn.",
            "- `normalized_text` chỉ là representation dẫn xuất cho duplicate detection; không loại dấu tiếng Việt, tên riêng, tổ chức, email, URL hoặc số khỏi `original_text`/`body`.",
            "",
            "Dataset không có newline phân tách subject/body đáng tin cậy. Vì vậy prefix `Chủ đề:` hoặc `Tiêu đề:` một mình không đủ để đoán điểm kết thúc subject; pipeline giữ toàn bộ nội dung trong `body` thay vì cắt sai.",
            "",
            "## URL extraction",
            "",
            f"- Rows có URL chuẩn: **{result['standard_url_rows']:,}**",
            f"- Rows có URL malformed/obfuscated được flag: **{result['malformed_url_rows']:,}**",
            "",
            "URL được trích từ `original_text` trước normalization. Các URL obfuscated được lưu riêng; pipeline không sửa hoặc xóa chúng khỏi original/canonical text.",
            "",
            "## Duplicate handling",
            "",
            "| Representation | Duplicate groups | Action |",
            "| --- | ---: | --- |",
            f"| Raw text exact | {result['raw_exact_duplicate_groups']:,} | Report only |",
            f"| V1-compatible normalized content hash | {result['exact_duplicate_groups']:,} | Keep first canonical; report removed copies |",
            f"| V1-compatible URL/email template hash | {result['template_duplicate_groups']:,} | Keep distinct rows in one group |",
            f"| Extended obfuscated URL/email template hash | {result['obfuscated_template_duplicate_groups']:,} | Keep distinct rows in one group |",
            f"| Conflicting-label template groups | {result['conflicting_template_groups']:,} | Exclude and report if present |",
            "",
            "`group_id` ưu tiên extended template hash khi có nhiều row cùng template; nếu không, nó dùng content hash. Cách này giữ nguyên nguyên tắc V1 và bổ sung khả năng nhóm URL/email bị chèn khoảng trắng.",
            "",
            "## Raw-data integrity",
            "",
            f"SHA-256 trước và sau pipeline: `{raw_hash}`. Raw input không bị ghi lại.",
            "",
            "## Limitations và bước bắt buộc tiếp theo",
            "",
            "- Đây là translated Vietnamese data, không phải native Vietnamese benchmark.",
            "- Raw label 1 vẫn cần row-level curation trước khi dùng cho training.",
            "- `parent_id`/`translation_source_id` chưa được điền; chưa được tạo bilingual split cho đến khi hoàn tất cross-language linkage.",
            "- Template grouping là deterministic audit aid, không chứng minh hai email có cùng nguồn dịch.",
            "- Không được tuyên bố Vietnamese real-world accuracy từ dataset này.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "datasets.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_dataset_config(args.config)
    policy = config["policy"]["vietnamese_standardization"]
    source_path = PROJECT_ROOT / policy["input"]
    before_hash = file_sha256(source_path)
    result = standardize_vietnamese_dataset(source_path, config)

    output_path = PROJECT_ROOT / policy["output"]
    removed_path = PROJECT_ROOT / policy["removed_rows_report"]
    duplicate_path = PROJECT_ROOT / policy["duplicate_report"]
    report_path = PROJECT_ROOT / policy["report"]
    for path in (output_path, removed_path, duplicate_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    result["master"].to_csv(output_path, index=False, encoding="utf-8")
    result["removed"].to_csv(removed_path, index=False, encoding="utf-8")
    result["duplicates"].to_csv(duplicate_path, index=False, encoding="utf-8")

    after_hash = file_sha256(source_path)
    if before_hash != after_hash:
        raise RuntimeError("Raw Vietnamese dataset changed during standardization")
    report_path.write_text(render_report(result, before_hash), encoding="utf-8")

    print("Vietnamese standardization complete")
    print(f"Rows before: {result['input_rows']:,}")
    print(f"Rows after: {result['output_rows']:,}")
    print(f"Empty rows removed: {result['empty_rows_removed']:,}")
    print(f"Exact duplicate copies removed: {result['exact_duplicates_removed']:,}")
    print(f"Subject rows parsed: {result['subject_rows_parsed']:,}")
    print(f"Class distribution: {result['output_label_distribution']}")
    print(f"Raw exact duplicate groups: {result['raw_exact_duplicate_groups']:,}")
    print(f"Normalized duplicate groups: {result['exact_duplicate_groups']:,}")
    print(f"Template duplicate groups: {result['template_duplicate_groups']:,}")
    print(
        "Obfuscated-template duplicate groups: "
        f"{result['obfuscated_template_duplicate_groups']:,}"
    )
    print(f"Conflicting template groups: {result['conflicting_template_groups']:,}")
    print(f"Raw SHA-256 unchanged: {before_hash == after_hash}")
    print(output_path.relative_to(PROJECT_ROOT).as_posix())
    print(report_path.relative_to(PROJECT_ROOT).as_posix())
    print(removed_path.relative_to(PROJECT_ROOT).as_posix())
    print(duplicate_path.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
