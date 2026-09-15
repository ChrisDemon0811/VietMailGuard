"""Freeze the Version 1 baseline eligibility and sampling plan without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.spam_curation import build_duplicate_counts, content_digest  # noqa: E402
from vietmailguard.dataset_standardizer import is_container_artifact  # noqa: E402

ELIGIBILITY_COLUMNS = [
    "original_row_id",
    "source",
    "subject",
    "body",
    "label",
    "label_provenance",
    "language",
    "training_eligible",
]

VALIDATION_COLUMNS = [
    "source",
    "original_row_id",
    "subject",
    "body_excerpt",
    "proposed_label",
    "curation_reason",
    "human_decision",
    "human_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen V1 baseline data plan.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def _stable_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) < count:
        raise ValueError(f"Requested {count} rows but only {len(frame)} are available")
    ranked = frame.assign(
        _rank=frame.apply(
            lambda row: hashlib.sha256(
                f"{seed}|{row['source']}|{row['original_row_id']}".encode("utf-8")
            ).hexdigest(),
            axis=1,
        )
    ).sort_values("_rank")
    return ranked.head(count).drop(columns="_rank")


def _preserve_validation_decisions(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        sample["human_decision"] = ""
        sample["human_note"] = ""
        return sample
    existing = pd.read_csv(path, keep_default_na=False)
    required = {"source", "original_row_id", "human_decision", "human_note"}
    if not required.issubset(existing.columns):
        raise ValueError(f"Existing validation file lacks columns: {sorted(required - set(existing))}")
    decisions = existing[list(required)].drop_duplicates(
        ["source", "original_row_id"], keep="last"
    )
    merged = sample.merge(decisions, on=["source", "original_row_id"], how="left")
    merged[["human_decision", "human_note"]] = merged[
        ["human_decision", "human_note"]
    ].fillna("")
    return merged


def _append_eligible_rows(
    rows: list[dict[str, object]],
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    source: str,
    label: str,
    provenance: str,
) -> None:
    for index, row in frame.loc[mask].iterrows():
        rows.append(
            {
                "original_row_id": int(index) + 2,
                "source": source,
                "subject": "" if pd.isna(row.get("subject")) else row.get("subject"),
                "body": "" if pd.isna(row.get("body")) else row.get("body"),
                "label": label,
                "label_provenance": provenance,
                "language": "en",
                "training_eligible": True,
            }
        )


def _write_plan_report(
    path: Path,
    eligibility: pd.DataFrame,
    manifest: pd.DataFrame,
    validation: pd.DataFrame,
    uncertain: pd.DataFrame,
    duplicate_exclusions: dict[str, int],
    target: int,
) -> None:
    available_by_class = eligibility["label"].value_counts().to_dict()
    available_by_source = eligibility.groupby(["label", "source"]).size().to_dict()
    proposed_by_source = manifest.groupby(["label", "source"]).size().to_dict()
    validation_by_source = validation["source"].value_counts().to_dict()
    validation_completed = int(validation["human_decision"].astype(str).str.strip().ne("").sum())

    lines = [
        "# VietMailGuard V1 — Baseline Dataset Plan",
        "",
        "Báo cáo này đóng băng chính sách dữ liệu trước Dataset Standardization Pipeline. Không có model nào được train và không có metric ML.",
        "",
        "## Thuật ngữ và quality gates",
        "",
        "- Spam vượt bộ lọc bảo thủ được gọi là `high_confidence_curated_spam`; thuật ngữ này mô tả provenance của curation, không khẳng định đã được con người xác nhận từng email.",
        "- `training_eligible=true` là điều kiện ở cấp dòng. Training vẫn bị chặn ở cấp dataset cho đến khi human-validation sample được hoàn tất và duplicate grouping được kiểm tra lại trong pipeline chính thức.",
        "- Toàn bộ review/uncertain rows bị loại khỏi eligibility table.",
        "",
        "## Available eligible samples after duplicate guards",
        "",
        "| Class | Available |",
        "|---|---:|",
        f"| normal | {int(available_by_class.get('normal', 0)):,} |",
        f"| spam | {int(available_by_class.get('spam', 0)):,} |",
        f"| phishing | {int(available_by_class.get('phishing', 0)):,} |",
        "",
        "| Class | Source | Available | Excluded as exact/template duplicate |",
        "|---|---|---:|---:|",
    ]
    source_order = ["CEAS_08", "Enron", "Ling", "SpamAssassin", "Nazario"]
    for label in ("normal", "spam", "phishing"):
        for source in source_order:
            available = int(available_by_source.get((label, source), 0))
            excluded = int(duplicate_exclusions.get(f"{label}|{source}", 0))
            if available or excluded:
                lines.append(f"| {label} | {source} | {available:,} | {excluded:,} |")

    lines.extend(
        [
            "",
            "## Proposed balanced baseline",
            "",
            f"Target: normal = {target:,}, spam = {target:,}, phishing = {target:,}; total = {3 * target:,}. No oversampling is used.",
            "",
            "| Class | Source | Proposed rows |",
            "|---|---|---:|",
        ]
    )
    for label in ("normal", "spam", "phishing"):
        for source in source_order:
            count = int(proposed_by_source.get((label, source), 0))
            if count:
                lines.append(f"| {label} | {source} | {count:,} |")

    lines.extend(
        [
            "",
            "Normal và spam đều giữ nhiều nguồn. Spam sử dụng toàn bộ nguồn nhỏ hơn trước, rồi lấy phần còn thiếu từ Enron; không source nào bị che giấu. Phishing hiện chỉ đến từ Nazario, đây là hạn chế source diversity của V1.",
            "",
            "## Human validation sample",
            "",
            f"Tổng sample: **{len(validation):,}**; đã có human decision: **{validation_completed:,}**; còn trống: **{len(validation) - validation_completed:,}**.",
            "",
            "| Source | Validation rows |",
            "|---|---:|",
        ]
    )
    for source in ("SpamAssassin", "Ling", "Enron", "CEAS_08"):
        lines.append(f"| {source} | {int(validation_by_source.get(source, 0)):,} |")

    lines.extend(
        [
            "",
            "Validation sample chỉ lấy `high_confidence_curated_spam` unique theo exact và template hash. `human_decision` để trống mặc định; không có quyết định nào được fabricate.",
            "",
            "## Outstanding review",
            "",
            f"Có **{len(uncertain):,}** spam-candidate rows vẫn cần human review và được lưu riêng. Chúng không xuất hiện trong eligibility dataset hoặc sampling manifest.",
            "",
            "## Duplicate/template policy",
            "",
            "- Nội dung so sánh được Unicode-normalize, lowercase và collapse whitespace.",
            "- Template comparison thay URL và địa chỉ email bằng placeholder.",
            "- Nếu exact hoặc template hash xuất hiện nhiều hơn một lần trong toàn bộ sáu raw corpora, mọi thành viên của group bị loại khỏi eligibility/sampling ở bước freeze này.",
            "- Pipeline chính thức vẫn phải tạo `group_id`, ghi duplicate report và bảo đảm group không đi qua nhiều split.",
            "",
            "## Readiness decision",
            "",
            "**Đủ về số lượng và chính sách để chuyển sang Dataset Standardization Pipeline.** Có tối thiểu 1.250 unique rows cho mỗi class theo kế hoạch, không cần oversampling.",
            "",
            "**Chưa được phép train.** Human-validation sample vẫn phải được review, và standardization phải xác nhận lại cleaning, provenance, hashes và leakage-safe grouping trước khi tạo split.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    config = json.loads((root / "config" / "datasets.json").read_text(encoding="utf-8"))
    freeze = config["policy"]["baseline_freeze_v1"]
    frames = {
        filename: pd.read_csv(root / dataset["path"], low_memory=False)
        for filename, dataset in config["datasets"].items()
    }
    exact_counts, template_counts = build_duplicate_counts(frames.values())

    eligible_rows: list[dict[str, object]] = []
    duplicate_exclusions: dict[str, int] = {}
    for filename, dataset in config["datasets"].items():
        frame = frames[filename]
        body_ok = ~frame["body"].fillna("").astype(str).str.strip().eq("")
        artifact_ok = pd.Series(
            [
                not is_container_artifact(
                    row.get("subject", ""),
                    row.get("body", ""),
                    config["policy"]["standardization"]["container_artifact_patterns"],
                )
                for _, row in frame.iterrows()
            ],
            index=frame.index,
        )
        unique = pd.Series(
            [
                exact_counts[content_digest(row.get("subject"), row.get("body"))] == 1
                and template_counts[
                    content_digest(row.get("subject"), row.get("body"), template=True)
                ]
                == 1
                for _, row in frame.iterrows()
            ],
            index=frame.index,
        )
        for raw_label, decision in dataset["raw_label_policy"].items():
            if decision["status"] not in {"normal", "phishing"}:
                continue
            selected = frame["label"].astype(str).eq(raw_label) & body_ok & artifact_ok
            accepted = selected & unique
            label = decision["status"]
            provenance = (
                "verified_legitimate_source" if label == "normal" else "nazario_phishing"
            )
            duplicate_exclusions[f"{label}|{dataset['source_name']}"] = int(
                (selected & ~unique).sum()
            )
            _append_eligible_rows(
                eligible_rows,
                frame,
                accepted,
                source=dataset["source_name"],
                label=label,
                provenance=provenance,
            )

    curation = pd.read_csv(root / config["policy"]["spam_curation"]["candidate_report"], keep_default_na=False)
    curated_spam = curation.loc[
        curation["candidate_group"].eq(freeze["spam_terminology"])
        & curation["proposed_label"].eq("spam")
    ].copy()
    source_to_filename = {
        dataset["source_name"]: filename for filename, dataset in config["datasets"].items()
    }
    for item in curated_spam.itertuples(index=False):
        filename = source_to_filename[item.source]
        frame = frames[filename]
        row = frame.loc[int(item.original_row_id) - 2]
        _append_eligible_rows(
            eligible_rows,
            frame,
            frame.index.to_series().eq(int(item.original_row_id) - 2),
            source=item.source,
            label="spam",
            provenance="high_confidence_curated_spam",
        )
    for source in curated_spam["source"].unique():
        duplicate_exclusions[f"spam|{source}"] = 0

    eligibility = pd.DataFrame(eligible_rows, columns=ELIGIBILITY_COLUMNS)
    eligibility_path = root / freeze["eligibility_dataset"]
    eligibility_path.parent.mkdir(parents=True, exist_ok=True)
    eligibility.to_csv(eligibility_path, index=False, encoding="utf-8")

    validation_parts = []
    for source, count in freeze["validation_sample"].items():
        source_rows = curated_spam.loc[curated_spam["source"].eq(source)]
        validation_parts.append(_stable_sample(source_rows, int(count), int(freeze["random_seed"])))
    validation = pd.concat(validation_parts, ignore_index=True).rename(
        columns={"review_reason": "curation_reason"}
    )[
        ["source", "original_row_id", "subject", "body_excerpt", "proposed_label", "curation_reason"]
    ]
    validation_path = root / freeze["validation_report"]
    validation = _preserve_validation_decisions(validation_path, validation)
    validation[VALIDATION_COLUMNS].to_csv(validation_path, index=False, encoding="utf-8")

    uncertain = curation.loc[~curation["candidate_group"].eq(freeze["spam_terminology"])]
    uncertain_path = root / freeze["outstanding_review_report"]
    uncertain.to_csv(uncertain_path, index=False, encoding="utf-8")

    manifest_parts = []
    plans = {
        "normal": freeze["normal_sampling_plan"],
        "spam": freeze["spam_sampling_plan"],
        "phishing": freeze["phishing_sampling_plan"],
    }
    for label, source_plan in plans.items():
        for source, count in source_plan.items():
            pool = eligibility.loc[
                eligibility["label"].eq(label) & eligibility["source"].eq(source)
            ]
            manifest_parts.append(_stable_sample(pool, int(count), int(freeze["random_seed"])))
    manifest = pd.concat(manifest_parts, ignore_index=True)
    manifest_path = root / freeze["sampling_manifest"]
    manifest[["original_row_id", "source", "label", "label_provenance"]].to_csv(
        manifest_path, index=False, encoding="utf-8"
    )

    _write_plan_report(
        root / freeze["plan_report"],
        eligibility,
        manifest,
        validation,
        uncertain,
        duplicate_exclusions,
        int(freeze["target_per_class"]),
    )
    print(f"{eligibility_path.relative_to(root).as_posix()}: {len(eligibility):,} rows")
    print(f"{validation_path.relative_to(root).as_posix()}: {len(validation):,} rows")
    print(f"{uncertain_path.relative_to(root).as_posix()}: {len(uncertain):,} rows")
    print(f"{manifest_path.relative_to(root).as_posix()}: {len(manifest):,} rows")
    print((root / freeze["plan_report"]).relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
