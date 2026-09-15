"""Build review-only pools and a high-confidence label availability report."""

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

from vietmailguard.label_review import build_review_pool, candidate_counts, load_review_config  # noqa: E402

REVIEW_DATASETS = {
    "CEAS_08.csv": "review_ceas_positive.csv",
    "Enron.csv": "review_enron_positive.csv",
    "Ling.csv": "review_ling_positive.csv",
    "SpamAssasin.csv": "review_spamassassin_positive.csv",
    "Nigerian_Fraud.csv": "review_nigerian_fraud.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create rule-assisted label review pools.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument(
        "--datasets-config", type=Path, default=PROJECT_ROOT / "config" / "datasets.json"
    )
    parser.add_argument(
        "--rules-config", type=Path, default=PROJECT_ROOT / "config" / "label_review_rules.json"
    )
    return parser.parse_args()


def _nonempty_body(frame: pd.DataFrame) -> pd.Series:
    return ~frame["body"].fillna("").astype(str).str.strip().eq("")


def _write_high_confidence_report(
    path: Path,
    source_counts: list[dict[str, int | str]],
    pool_counts: dict[str, dict[str, int]],
    pools: dict[str, pd.DataFrame],
) -> None:
    totals = {
        status: sum(int(row[status]) for row in source_counts)
        for status in ("normal", "spam", "phishing", "review", "exclude")
    }
    lines = [
        "# High-confidence label availability report",
        "",
        "This is an eligibility report, not a merged dataset and not a training result.",
        "Counts apply the approved raw-label policy plus the deterministic empty-body exclusion only; duplicate/template grouping has not yet been applied.",
        "The row-reference manifest is `reports/high_confidence_label_inventory.csv`; it intentionally contains no email content and is not a trainable master dataset.",
        "",
        "## Approved counts",
        "",
        "| Class/status | Rows | Training eligible now |",
        "|---|---:|---|",
        f"| normal | {totals['normal']:,} | yes, after future duplicate grouping/splitting |",
        f"| spam | {totals['spam']:,} | no approved spam rows yet |",
        f"| phishing | {totals['phishing']:,} | yes, after future duplicate grouping/splitting |",
        f"| review | {totals['review']:,} | no |",
        f"| exclude | {totals['exclude']:,} | no |",
        "",
        "## Distribution by source",
        "",
        "| Source | normal | spam | phishing | review | exclude |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in source_counts:
        lines.append(
            f"| {row['source']} | {row['normal']:,} | {row['spam']:,} | "
            f"{row['phishing']:,} | {row['review']:,} | {row['exclude']:,} |"
        )

    lines.extend(
        [
            "",
            "## Rule-assisted review pools",
            "",
            "Candidate groups are triage hints only. `likely_spam` requires at least two distinct spam-rule signals, while phishing/scam signals take precedence. Every non-empty row remains `pending_manual_confirmation`; no candidate is admitted to training automatically.",
            "",
            "| Source | likely_spam | likely_phishing_or_scam | uncertain |",
            "|---|---:|---:|---:|",
        ]
    )
    for source, counts in pool_counts.items():
        lines.append(
            f"| {source} | {counts['likely_spam']:,} | "
            f"{counts['likely_phishing_or_scam']:,} | {counts['uncertain']:,} |"
        )

    lines.extend(["", "## Deeper review of Ling and SpamAssassin positives", ""])
    for source in ("Ling.csv", "SpamAssasin.csv"):
        pool = pools[source]
        counts = pool_counts[source]
        total = len(pool)
        display_name = "SpamAssassin" if source == "SpamAssasin.csv" else "Ling"
        lines.extend(
            [
                f"### {display_name} label 1",
                "",
                f"- `{counts['likely_spam']:,}`/{total:,} ({100 * counts['likely_spam'] / total:.1f}%) meet the conservative multi-signal spam-candidate rule.",
                f"- `{counts['likely_phishing_or_scam']:,}`/{total:,} ({100 * counts['likely_phishing_or_scam'] / total:.1f}%) trigger at least two phishing/scam rule families.",
                f"- `{counts['uncertain']:,}`/{total:,} ({100 * counts['uncertain'] / total:.1f}%) remain uncertain. This large group is expected because one keyword is deliberately insufficient and older/obfuscated advertisements often lack explicit modern marketing phrases.",
                "- A phishing/scam signal blocks a `likely_spam` proposal even when spam signals are also present.",
                "",
                "Representative triage candidates:",
                "",
                "| Candidate group | Original row id | Subject | Triggered rules |",
                "|---|---:|---|---|",
            ]
        )
        for group in ("likely_spam", "likely_phishing_or_scam", "uncertain"):
            examples = pool.loc[pool["candidate_group"].eq(group)].head(3)
            for _, row in examples.iterrows():
                subject = str(row["subject"]).replace("|", "\\|") or "(empty subject)"
                triggered = str(row["triggered_rules"]).replace("|", ", ") or "none"
                lines.append(
                    f"| {group} | {int(row['original_row_id'])} | {subject} | {triggered} |"
                )
        lines.extend(
            [
                "",
                "`proposed_label=spam` in this pool is a queueing suggestion only; `review_status=pending_manual_confirmation` prevents it from becoming a training label.",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Balanced baseline recommendation",
            "",
            "Do not build the three-class baseline yet because the approved spam count is zero.",
            f"The first defensible pilot target is **300 unique duplicate/template groups per class**. There are currently {pool_counts['Ling.csv']['likely_spam'] + pool_counts['SpamAssasin.csv']['likely_spam']:,} conservative Ling/SpamAssassin spam candidates before manual confirmation and duplicate grouping, so 300 leaves room for rejected and duplicated candidates without oversampling.",
            "For a stronger later baseline, target 1,200–1,500 unique groups per class only after manual curation expands the confirmed spam pool beyond the strict candidates. Select normal examples across CEAS, Enron, Ling, and SpamAssassin; use cleaned Nazario phishing; and obtain spam only from manually confirmed Ling/SpamAssassin rows.",
            "Do not oversample, copy rows, or let one duplicate/template group cross a split. Reserve enough unique groups for the later 70/15/15 split before fixing the final sample size.",
            "",
            "## Decision",
            "",
            "The current corpus is not yet ready for scientifically defensible three-class baseline training. Normal and Nazario phishing are available provisionally, but no spam row has completed manual confirmation. CEAS/Enron positives remain excluded from the baseline; Nigerian Fraud remains a separate 419/advance-fee review pool pending a future taxonomy decision.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_config = json.loads(args.datasets_config.read_text(encoding="utf-8"))
    review_config, rules = load_review_config(args.rules_config)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for filename in dataset_config["datasets"]:
        frames[filename] = pd.read_csv(args.raw_dir / filename, low_memory=False)

    pool_counts: dict[str, dict[str, int]] = {}
    pools: dict[str, pd.DataFrame] = {}
    for filename, output_name in REVIEW_DATASETS.items():
        pool = build_review_pool(
            frames[filename],
            source=dataset_config["datasets"][filename]["source_name"],
            raw_label=1,
            config=review_config,
            rules=rules,
            allow_spam_proposal=dataset_config["datasets"][filename]["raw_label_policy"]["1"].get(
                "rule_assisted_spam_proposal_enabled", False
            ),
        )
        pool.to_csv(args.reports_dir / output_name, index=False, encoding="utf-8")
        pools[filename] = pool
        pool_counts[filename] = candidate_counts(pool)
        print(f"{output_name}: {len(pool)} rows")

    source_counts: list[dict[str, int | str]] = []
    inventory_rows: list[dict[str, int | str]] = []
    for filename, frame in frames.items():
        policy = dataset_config["datasets"][filename]["raw_label_policy"]
        counts: dict[str, int | str] = {
            "source": dataset_config["datasets"][filename]["source_name"],
            "normal": 0,
            "spam": 0,
            "phishing": 0,
            "review": 0,
            "exclude": 0,
        }
        body_ok = _nonempty_body(frame)
        for raw_label, decision in policy.items():
            selected = frame["label"].astype(str).eq(str(raw_label))
            counts["exclude"] = int(counts["exclude"]) + int((selected & ~body_ok).sum())
            counts[decision["status"]] = int(counts[decision["status"]]) + int(
                (selected & body_ok).sum()
            )
            if decision["training_eligible"]:
                for index in frame.index[selected & body_ok]:
                    inventory_rows.append(
                        {
                            "source": dataset_config["datasets"][filename]["source_name"],
                            "original_row_id": int(index) + 2,
                            "raw_label": raw_label,
                            "approved_label": decision["status"],
                            "eligibility_status": "pending_cleaning_and_duplicate_grouping",
                        }
                    )
        source_counts.append(counts)

    pd.DataFrame(inventory_rows).to_csv(
        args.reports_dir / "high_confidence_label_inventory.csv", index=False, encoding="utf-8"
    )
    report_path = args.reports_dir / "high_confidence_label_report.md"
    _write_high_confidence_report(report_path, source_counts, pool_counts, pools)
    print(f"high_confidence_label_inventory.csv: {len(inventory_rows)} row references")
    print(report_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
