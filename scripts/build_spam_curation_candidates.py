"""Build a conservative spam curation queue and its status report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.label_review import load_review_config  # noqa: E402
from vietmailguard.spam_curation import (  # noqa: E402
    CURATION_COLUMNS,
    apply_manual_labels,
    assess_spam_candidate,
    build_duplicate_counts,
    load_manual_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the spam curation candidate queue.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def _review_pool_path(dataset: dict[str, object], root: Path, raw_label: str) -> Path:
    value = dataset["raw_label_policy"][raw_label].get("review_pool")
    if not value:
        raise ValueError(f"Missing review_pool for {dataset['source_name']} raw label {raw_label}")
    return root / str(value)


def _write_report(
    path: Path,
    candidates: pd.DataFrame,
    manual: pd.DataFrame,
    *,
    available_phishing: int,
    desired_cap: int,
    desired_spam: int,
) -> None:
    latest = manual.drop_duplicates(["source", "original_row_id"], keep="last")
    decisions = latest["manual_decision"].value_counts().to_dict()
    reviewed = len(latest)
    confirmed_spam = int(decisions.get("spam", 0))
    confirmed_phishing = int(decisions.get("phishing_or_scam", 0))
    excluded = int(decisions.get("exclude", 0))
    deferred = int(decisions.get("review", 0))
    still_review = len(candidates) - confirmed_spam - confirmed_phishing - excluded

    spam_confirmed_rows = candidates.loc[
        candidates["review_status"].eq("confirmed")
        & candidates["proposed_label"].eq("spam")
    ]
    usable_confirmed_spam = int(
        (~spam_confirmed_rows["_exact_duplicate"] & ~spam_confirmed_rows["_template_duplicate"]).sum()
    )
    baseline_n = min(usable_confirmed_spam, available_phishing, desired_cap)
    source_counts = spam_confirmed_rows["source"].value_counts().to_dict()

    exact_duplicate_rows = int(candidates["_exact_duplicate"].sum())
    template_duplicate_rows = int(candidates["_template_duplicate"].sum())
    any_duplicate_rows = int(
        (candidates["_exact_duplicate"] | candidates["_template_duplicate"]).sum()
    )
    auto_candidates = int(candidates["candidate_group"].eq("high_confidence_curated_spam").sum())
    needs_review = int(candidates["review_status"].eq("needs_human_review").sum())

    lines = [
        "# Báo cáo spam curation",
        "",
        "Báo cáo này không phải kết quả training. Rule chỉ tạo candidate; chỉ quyết định được nhập qua CLI review mới có thể mang trạng thái `confirmed`.",
        "",
        "## Trạng thái review thủ công",
        "",
        "| Chỉ số | Số lượng |",
        "|---|---:|",
        f"| Tổng candidate ưu tiên | {len(candidates):,} |",
        f"| Candidate đã có quyết định thủ công | {reviewed:,} |",
        f"| Confirmed spam | {confirmed_spam:,} |",
        f"| Confirmed phishing/scam | {confirmed_phishing:,} |",
        f"| Review later đã ghi nhận | {deferred:,} |",
        f"| Vẫn ở review | {still_review:,} |",
        f"| Human-confirmed exclude | {excluded:,} |",
        f"| Confirmed spam còn dùng được sau duplicate guard | {usable_confirmed_spam:,} |",
        "",
        "## Conservative candidate filter",
        "",
        f"- `high_confidence_curated_spam` chưa xác nhận thủ công: **{auto_candidates:,}**.",
        f"- Candidates cần xem xét kỹ hơn: **{needs_review:,}**.",
        "- Điều kiện candidate nghiêm ngặt: ít nhất hai tín hiệu commercial/spam độc lập, không có phishing/fraud rule, body tối thiểu, và unique theo cả exact hash lẫn URL/email-insensitive template hash.",
        "- `auto_candidate` không phải ground truth và không được dùng để train.",
        "",
        "## Duplicate statistics",
        "",
        "| Chỉ số | Candidate rows |",
        "|---|---:|",
        f"| Exact duplicate | {exact_duplicate_rows:,} |",
        f"| Template duplicate | {template_duplicate_rows:,} |",
        f"| Exact hoặc template duplicate | {any_duplicate_rows:,} |",
        "",
        "## Confirmed spam theo source",
        "",
        "| Source | Confirmed spam |",
        "|---|---:|",
    ]
    for source in ("SpamAssassin", "Ling", "Enron", "CEAS_08"):
        lines.append(f"| {source} | {int(source_counts.get(source, 0)):,} |")

    lines.extend(
        [
            "",
            "## Đề xuất baseline cân bằng",
            "",
            f"Theo công thức `N = min(confirmed_spam_usable, available_phishing, desired_cap)`, hiện tại `N = min({usable_confirmed_spam:,}, {available_phishing:,}, {desired_cap:,}) = {baseline_n:,}`.",
            f"Mục tiêu curation là ít nhất {desired_spam:,} confirmed spam hợp lệ. "
            + (
                "Mục tiêu đã đạt về số lượng, nhưng vẫn phải hoàn tất duplicate grouping trước khi lấy mẫu."
                if usable_confirmed_spam >= desired_spam
                else "Mục tiêu chưa đạt; không được hạ tiêu chuẩn hoặc dùng duplicate để bù số lượng."
            ),
            "Khi đủ dữ liệu, lấy tối đa N unique groups cho mỗi lớp normal/spam/phishing và giữ toàn bộ cùng template group trong một split.",
            "",
            "## Kết luận",
            "",
            (
                "Số spam thủ công đã đủ cho bước chuẩn bị baseline, nhưng chưa cho phép train trước khi standardization và leakage-safe grouping hoàn tất."
                if usable_confirmed_spam >= desired_spam
                else "Dataset chưa đủ để bắt đầu baseline ba lớp: chưa có ít nhất 1.000 spam samples được xác nhận và qua duplicate guard. Cần tiếp tục human curation hoặc bổ sung dedicated spam dataset có provenance rõ."
            ),
            "Nigerian Fraud không được dùng để xây spam class và vẫn tách riêng chờ quyết định taxonomy.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    datasets_config = json.loads((root / "config" / "datasets.json").read_text(encoding="utf-8"))
    _, rules = load_review_config(root / datasets_config["policy"]["review_rules"])
    policy = datasets_config["policy"]["spam_curation"]

    frames = {
        filename: pd.read_csv(root / dataset["path"], low_memory=False)
        for filename, dataset in datasets_config["datasets"].items()
    }
    exact_counts, template_counts = build_duplicate_counts(frames.values())

    assessed: list[dict[str, object]] = []
    for priority_item in policy["source_priority"]:
        filename, raw_label = priority_item.rsplit(":", 1)
        dataset = datasets_config["datasets"][filename]
        pool = pd.read_csv(_review_pool_path(dataset, root, raw_label), keep_default_na=False)
        pool = pool.loc[pool["candidate_group"].eq(policy["candidate_input_group"])]
        frame = frames[filename]
        for item in pool.itertuples(index=False):
            source_index = int(item.original_row_id) - 2
            raw_row = frame.loc[source_index]
            if str(raw_row["label"]) != raw_label:
                raise ValueError(f"Raw label mismatch at {filename} row {item.original_row_id}")
            assessed.append(
                assess_spam_candidate(
                    source=dataset["source_name"],
                    original_row_id=int(item.original_row_id),
                    raw_label=raw_row["label"],
                    subject=raw_row.get("subject", ""),
                    body=raw_row.get("body", ""),
                    rules=rules,
                    policy=policy,
                    exact_counts=exact_counts,
                    template_counts=template_counts,
                )
            )

    candidates = pd.DataFrame(assessed)
    manual_path = root / policy["manual_labels"]
    manual = load_manual_labels(manual_path)
    candidates = apply_manual_labels(candidates, manual)

    candidate_path = root / policy["candidate_report"]
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidates[CURATION_COLUMNS].to_csv(candidate_path, index=False, encoding="utf-8")

    nazario = frames["Nazario.csv"]
    available_phishing = int(
        (nazario["label"].eq(1) & ~nazario["body"].fillna("").astype(str).str.strip().eq("")).sum()
    )
    _write_report(
        root / policy["curation_report"],
        candidates,
        manual,
        available_phishing=available_phishing,
        desired_cap=int(policy["desired_initial_cap"]),
        desired_spam=int(policy["desired_confirmed_spam"]),
    )
    print(f"{candidate_path.relative_to(root).as_posix()}: {len(candidates)} candidates")
    print((root / policy["curation_report"]).relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
