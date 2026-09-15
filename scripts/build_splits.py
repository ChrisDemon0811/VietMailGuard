"""Create deterministic, leakage-safe train/validation/test splits."""

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

from vietmailguard.dataset_standardizer import load_dataset_config  # noqa: E402
from vietmailguard.split_builder import (  # noqa: E402
    build_split_report,
    build_splits,
    write_splits_atomically,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VietMailGuard group-aware data splits.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "datasets.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_dataset_config(args.config)
    policy = config["policy"]["splitting"]
    master_path = PROJECT_ROOT / policy["input"]
    if not master_path.exists():
        raise FileNotFoundError(f"Standardized master dataset does not exist: {master_path}")
    frame = pd.read_csv(master_path, keep_default_na=False, low_memory=False)
    ratios = {name: float(policy["ratios"][name]) for name in ("train", "validation", "test")}
    seed = int(policy["seed"])

    splits = build_splits(frame, ratios, seed=seed)
    report = build_split_report(splits, ratios, seed)
    outputs = {name: PROJECT_ROOT / path for name, path in policy["outputs"].items()}
    write_splits_atomically(splits, outputs)

    report_path = PROJECT_ROOT / policy["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Split summary")
    for name in ("train", "validation", "test"):
        details = report["splits"][name]
        classes = {label: values["count"] for label, values in details["class_distribution"].items()}
        print(
            f"{name}: rows={details['rows']:,}, groups={details['groups']:,}, "
            f"ratio={100 * details['actual_ratio']:.4f}%, classes={classes}"
        )
    print(f"Leakage check passed: {report['leakage_check']['passed']}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT).as_posix()}")
    for name, path in outputs.items():
        print(f"{name}: {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
