"""Create VietMailGuard V2 bilingual master data and leakage-safe splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.bilingual_dataset import (  # noqa: E402
    assign_bilingual_splits,
    build_final_groups,
    load_bilingual_config,
    prepare_bilingual_rows,
    split_frames,
    validate_bilingual_splits,
)
from vietmailguard.file_utils import sha256_file  # noqa: E402
from vietmailguard.split_builder import SPLIT_ORDER  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "bilingual_dataset.json",
    )
    return parser.parse_args()


def _hash_files(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _counts(series: pd.Series, *, blank: str = "not_declared") -> dict[str, int]:
    values = series.fillna("").astype(str).replace("", blank)
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def _markdown_table(counts: dict[str, int], first_column: str) -> list[str]:
    lines = [f"| {first_column} | Rows |", "| --- | ---: |"]
    if not counts:
        return [*lines, "| None | 0 |"]
    return [*lines, *(f"| {key} | {value:,} |" for key, value in counts.items())]


def _cross_table(
    frame: pd.DataFrame, row_column: str, column_column: str
) -> list[str]:
    table = pd.crosstab(frame[row_column], frame[column_column])
    columns = [str(value) for value in table.columns]
    lines = [
        f"| {row_column} | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for index, row in table.iterrows():
        lines.append(
            f"| {index} | " + " | ".join(f"{int(row[column]):,}" for column in table.columns) + " |"
        )
    return lines


def _render_dataset_report(
    master: pd.DataFrame,
    selection: dict[str, Any],
    grouping: dict[str, int],
    config: dict[str, Any],
    integrity: dict[str, bool],
) -> str:
    lines = [
        "# Bilingual dataset report",
        "",
        "## Kết luận",
        "",
        f"Bilingual Master chứa **{len(master):,}** rows có label hợp lệ và đủ điều kiện đưa vào split.",
        "Dữ liệu translated vẫn được ghi rõ nguồn gốc; không row nào được gọi là native Vietnamese.",
        "",
        "## Input và eligibility",
        "",
    ]
    lines.extend(_markdown_table(selection["input_rows"], "Input artifact"))
    lines.extend(["", "### Included rows", ""])
    lines.extend(_markdown_table(selection["included_rows"], "Eligibility source"))
    lines.extend(["", "### Excluded before final master", ""])
    lines.extend(_markdown_table(selection["excluded_rows"], "Reason"))
    lines.extend(
        [
            "",
            "Vietnamese curated rows chỉ được nhận khi `training_eligible=true`, label thuộc ba lớp đích, và pair xuất hiện trong cross-language manifest với trusted high-confidence status. Những row label hợp lệ nhưng linkage medium/low/không có candidate bị giữ ngoài mọi split để không tạo semantic leakage chưa kiểm soát.",
            "",
            "## Master distribution",
            "",
            "### Class",
            "",
        ]
    )
    lines.extend(_markdown_table(_counts(master["label"]), "Class"))
    lines.extend(["", "### Language", ""])
    lines.extend(_markdown_table(_counts(master["language"]), "Language"))
    lines.extend(["", "### Data origin", ""])
    lines.extend(_markdown_table(_counts(master["data_origin"]), "Data origin"))
    lines.extend(["", "### Source", ""])
    lines.extend(_markdown_table(_counts(master["source"]), "Source"))
    lines.extend(["", "### Label provenance", ""])
    lines.extend(_markdown_table(_counts(master["label_provenance"]), "Label provenance"))
    lines.extend(
        [
            "",
            "## Leakage grouping",
            "",
            f"- Final leakage groups: **{grouping['final_groups']:,}**.",
            f"- Known translation groups: **{grouping['translation_groups']:,}**.",
            f"- Parent/translation edges: **{grouping['parent_translation_edges']:,}**.",
            f"- Source duplicate/template edges: **{grouping['source_group_edges']:,}**.",
            f"- Exact content-hash edges: **{grouping['content_hash_edges']:,}**.",
            f"- Recomputed normalized-duplicate edges: **{grouping['normalized_duplicate_edges']:,}**.",
            f"- Recomputed template edges: **{grouping['template_duplicate_edges']:,}**.",
            "",
            "`final_group_id` là connected component của source `group_id`, exact `content_hash`, recomputed normalized/template hashes và confirmed parent/translation links. Hai row chỉ có semantic candidate medium/low không được tự động gọi là parent relation.",
            "",
            "## Unified schema",
            "",
            "Master giữ toàn bộ trường bắt buộc cùng `normalized_content_hash`, `template_hash`, `final_group_id`, `split_constraint`, `eligibility_basis`, `input_artifact` và split assignment để audit lại được.",
            "",
            "English Version 1 artifacts không có trường `data_origin`. Giá trị này được để trống thay vì tự gán `native`; đây là chủ ý tuân thủ yêu cầu không invent metadata. Mọi Vietnamese row được sử dụng đều có `data_origin=translated`.",
            "",
            "## Limitations",
            "",
            "- Không có native Vietnamese data hoặc native Vietnamese benchmark.",
            "- Vietnamese curated high-confidence links đều nối về English source Enron.",
            "- Controlled translations chỉ được dùng trong train và chưa hoàn tất human review sample.",
            "- Vietnamese phishing hiện chỉ đến từ controlled translation; vì vậy validation/test không đo translated-Vietnamese phishing.",
            "- Class normal lớn hơn đáng kể spam và phishing; bước này không downsample hoặc oversample.",
            "- Medium/low semantic overlap candidates không được dùng cho final split trước human review.",
            "",
            "## Integrity",
            "",
            f"- Raw files unchanged during build: **{str(integrity['raw_unchanged']).lower()}**.",
            f"- Input artifacts unchanged during build: **{str(integrity['inputs_unchanged']).lower()}**.",
            "- Không fit TF-IDF, không train classifier và không evaluate model.",
            "",
            f"Master output: `{Path(config['outputs']['master']).as_posix()}`; policy: `config/bilingual_dataset.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_split_report(
    master: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    assignment: dict[str, Any],
    validation: dict[str, Any],
    config: dict[str, Any],
) -> str:
    lines = [
        "# Bilingual split report",
        "",
        "## Split summary",
        "",
        "Target là 70/15/15 với seed 42. Group integrity, parent linkage và train-only augmentation có ưu tiên cao hơn ratio chính xác.",
        "",
        "| Split | Rows | Actual ratio | Final groups |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in SPLIT_ORDER:
        frame = splits[name]
        lines.append(
            f"| {name} | {len(frame):,} | {100 * len(frame) / len(master):.4f}% | {frame['final_group_id'].nunique():,} |"
        )
    lines.extend(["", "### Class distribution", ""])
    lines.extend(_cross_table(master, "split", "label"))
    lines.extend(["", "### Language distribution", ""])
    lines.extend(_cross_table(master, "split", "language"))
    lines.extend(["", "### Data-origin distribution", ""])
    display = master.copy()
    display["data_origin"] = display["data_origin"].replace("", "not_declared")
    lines.extend(_cross_table(display, "split", "data_origin"))
    lines.extend(["", "### Source distribution", ""])
    lines.extend(_cross_table(master, "split", "source"))
    lines.extend(
        [
            "",
            "## Controlled-translation constraint",
            "",
            f"- Train-only final groups: **{assignment['forced_train_groups']:,}**.",
            f"- Rows inside train-only groups, including English parents and related duplicates: **{assignment['forced_train_rows']:,}**.",
            "- Controlled augmentation rows and their English parents are all in train. Không augmentation row nào nằm trong validation/test.",
            "",
            "## Leakage validation",
            "",
            f"- Overall result: **{str(validation['passed']).lower()}**.",
            f"- Parent/translation pairs checked: **{validation['parent_translation_pairs_checked']:,}**.",
            f"- Train-only rows checked: **{validation['train_only_rows_checked']:,}**.",
        ]
    )
    for key, count in validation["cross_split_overlap_counts"].items():
        lines.append(f"- `{key}` cross-split overlaps: **{count:,}**.")
    lines.extend(
        [
            "",
            "## Evaluation subset manifest",
            "",
            f"Machine-readable definitions: `{config['outputs']['evaluation_manifest']}`.",
            "Các subset được định nghĩa cho English, translated Vietnamese, combined, per class, per data_origin và per source. Không có native Vietnamese subset hay metric.",
            "",
            "## Suitability",
            "",
            "Dataset đủ để bắt đầu experimental bilingual three-class training: cả ba lớp có dữ liệu trong train và leakage checks đều pass. Tuy nhiên nó chưa đủ để tuyên bố Vietnamese three-class generalization vì validation/test không có Vietnamese phishing và không có native Vietnamese benchmark.",
            "",
            "Không fit TF-IDF, không train model và không evaluate model trong bước này.",
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluation_manifest(master: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split_files": {
            name: config["outputs"][name] for name in ("train", "validation", "test")
        },
        "subsets": {
            "combined": {"filters": {}},
            "english": {"filters": {"language": ["en"]}},
            "vietnamese_translated": {
                "filters": {"language": ["vi"], "data_origin": ["translated"]},
                "claim_name": "Translated Vietnamese",
            },
            "per_class": {"field": "label", "values": sorted(master["label"].unique())},
            "per_data_origin": {
                "field": "data_origin",
                "values": sorted(master["data_origin"].unique()),
                "blank_value_meaning": "not_declared_in_version_1_english_artifact",
            },
            "per_source": {"field": "source", "values": sorted(master["source"].unique())},
        },
        "unavailable_subsets": {
            "vietnamese_native": "No native Vietnamese rows exist in the final master."
        },
        "evaluation_rules": [
            "Apply filters only within the requested split.",
            "Do not call translated Vietnamese results native Vietnamese results.",
            "Do not use test subsets before final model selection.",
        ],
    }


def _write_csvs_atomically(
    master: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    outputs: dict[str, Path],
) -> None:
    frames = {"master": master, **splits}
    temporary: dict[str, Path] = {}
    try:
        for name, frame in frames.items():
            path = outputs[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary[name] = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(temporary[name], index=False, encoding="utf-8")
        reloaded_splits = {
            name: pd.read_csv(temporary[name], keep_default_na=False, low_memory=False)
            for name in SPLIT_ORDER
        }
        validate_bilingual_splits(reloaded_splits)
        reloaded_master = pd.read_csv(temporary["master"], keep_default_na=False, low_memory=False)
        if len(reloaded_master) != sum(len(frame) for frame in reloaded_splits.values()):
            raise RuntimeError("Serialized master and split row totals differ")
        for name, path in outputs.items():
            temporary[name].replace(path)
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()


def main() -> int:
    args = parse_args()
    config = load_bilingual_config(args.config)
    input_paths = {
        name: PROJECT_ROOT / relative for name, relative in config["inputs"].items()
    }
    for path in input_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Required V2 artifact does not exist: {path}")

    raw_before = _hash_files(PROJECT_ROOT / "data" / "raw")
    input_before = {str(path): sha256_file(path) for path in input_paths.values()}
    english = pd.read_csv(input_paths["english_master"], keep_default_na=False, low_memory=False)
    vietnamese = pd.read_csv(input_paths["vietnamese_curated"], keep_default_na=False, low_memory=False)
    augmentation = pd.read_csv(
        input_paths["vietnamese_augmentation"], keep_default_na=False, low_memory=False
    )
    overlap = pd.read_csv(
        input_paths["cross_language_overlap"], keep_default_na=False, low_memory=False
    )

    rows, selection_audit = prepare_bilingual_rows(
        english, vietnamese, augmentation, overlap, config
    )
    grouped, grouping_audit = build_final_groups(rows)
    ratios = {
        name: float(config["splitting"]["ratios"][name]) for name in SPLIT_ORDER
    }
    seed = int(config["splitting"]["seed"])
    assigned, assignment_audit = assign_bilingual_splits(grouped, ratios, seed)
    master = assigned.sort_values(
        ["split", "label", "language", "source", "id"], kind="stable"
    ).reset_index(drop=True)
    splits = split_frames(master, seed)
    validation = validate_bilingual_splits(splits)

    output_paths = {
        name: PROJECT_ROOT / config["outputs"][name]
        for name in ("master", "train", "validation", "test")
    }
    _write_csvs_atomically(master, splits, output_paths)

    manifest_path = PROJECT_ROOT / config["outputs"]["evaluation_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_evaluation_manifest(master, config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    raw_after = _hash_files(PROJECT_ROOT / "data" / "raw")
    input_after = {str(path): sha256_file(path) for path in input_paths.values()}
    integrity = {
        "raw_unchanged": raw_before == raw_after,
        "inputs_unchanged": input_before == input_after,
    }
    if not all(integrity.values()):
        raise RuntimeError("An immutable raw/input artifact changed during the V2 build")

    dataset_report_path = PROJECT_ROOT / config["outputs"]["dataset_report"]
    split_report_path = PROJECT_ROOT / config["outputs"]["split_report"]
    dataset_report_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_report_path.write_text(
        _render_dataset_report(master, selection_audit, grouping_audit, config, integrity),
        encoding="utf-8",
    )
    split_report_path.write_text(
        _render_split_report(master, splits, assignment_audit, validation, config),
        encoding="utf-8",
    )

    print("VietMailGuard V2 bilingual dataset build complete")
    print(f"master: rows={len(master):,}, final_groups={master['final_group_id'].nunique():,}")
    for name in SPLIT_ORDER:
        frame = splits[name]
        print(
            f"{name}: rows={len(frame):,}, ratio={100 * len(frame) / len(master):.4f}%, "
            f"groups={frame['final_group_id'].nunique():,}, "
            f"classes={frame['label'].value_counts().sort_index().to_dict()}, "
            f"languages={frame['language'].value_counts().sort_index().to_dict()}"
        )
    print(f"leakage_check_passed: {validation['passed']}")
    print(f"raw_unchanged: {integrity['raw_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
