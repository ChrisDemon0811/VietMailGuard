"""Command-line entry point for the read-only raw dataset audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.dataset_audit import audit_repository  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw VietMailGuard datasets without modifying them.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "datasets.json")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Replace datasets.json with newly observed structure. Existing reviewed mappings are preserved by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_repository(args.raw_dir, args.reports_dir, sample_size=args.sample_size)
    config = {
        "schema_version": 1,
        "policy": {
            "allowed_internal_labels": ["normal", "spam", "phishing"],
            "unmapped_label_action": "preserve_for_review_and_exclude_from_training",
            "mapping_rule": "A raw label may be mapped only after its semantics and corpus provenance are verified.",
        },
        "datasets": result["datasets"],
    }
    config_written = args.write_config or not args.config.exists()
    if config_written:
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(result['datasets'])} dataset(s).")
    for report in result["report_files"]:
        report_path = args.reports_dir / report
        print(report_path.resolve().relative_to(PROJECT_ROOT).as_posix())
    if config_written:
        print(args.config.resolve().relative_to(PROJECT_ROOT).as_posix())
    else:
        print("Skipped config/datasets.json (use --write-config to replace it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
