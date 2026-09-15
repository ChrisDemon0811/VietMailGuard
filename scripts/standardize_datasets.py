"""Run the VietMailGuard Dataset Standardization Pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.dataset_audit import audit_repository  # noqa: E402
from vietmailguard.dataset_standardizer import (  # noqa: E402
    load_dataset_config,
    standardize_datasets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize approved VietMailGuard raw datasets.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "datasets.json")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_dataset_config(args.config)
    policy = config["policy"]["standardization"]

    audit_repository(args.raw_dir, args.reports_dir, sample_size=3)
    result = standardize_datasets(args.raw_dir, config)

    outputs = {
        PROJECT_ROOT / policy["output"]: result["master"],
        PROJECT_ROOT / policy["removed_rows_report"]: result["removed"],
        PROJECT_ROOT / policy["duplicate_report"]: result["duplicates"],
        PROJECT_ROOT / policy["review_rows_report"]: result["review"],
        PROJECT_ROOT / policy["summary_report"]: result["summary"],
    }
    for path, frame in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8")

    print("\nPer-source standardization summary")
    print(result["summary"].to_string(index=False))
    print("\nMaster class distribution")
    print(result["master"]["label"].value_counts().reindex(["normal", "spam", "phishing"], fill_value=0).to_string())
    print(f"\nExact duplicate groups: {result['exact_duplicate_groups']:,}")
    print(f"Template duplicate groups: {result['template_duplicate_groups']:,}")
    print(f"Conflicting-label template groups: {result['conflicting_template_groups']:,}")
    print(f"Master rows: {len(result['master']):,}")
    for path in outputs:
        print(path.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
