"""Audit PhishVN ZIPs and build offline URL-only train/val feature parts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.v21.paths import resolve_v21_paths  # noqa: E402
from vietmailguard.v21.phishvn_urls import (  # noqa: E402
    _save_json, audit_archives, process_development_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, help="Override VIETMAILGUARD_V21_DATA_DIR for this run")
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()
    paths = resolve_v21_paths()
    source_dir = args.dataset_dir or paths.dataset_dir
    open_zip = source_dir / "PhishVN_v3.1.0_open.zip"
    bundle = source_dir / "PhishVN_v3.0.0_results_bundle.zip"
    rules_path = paths.project_root / "config" / "risk_rules.json"
    audit = audit_archives(open_zip, bundle, paths.cache / "phishvn_audit.sqlite3", rules_path)
    _save_json(paths.reports / "phishvn_url_audit.json", audit)
    critical = {
        key: value for key, value in audit["issues"].items()
        if key in {"invalid_split", "unknown_label", "empty_url", "duplicate_id",
                   "registered_domain_cross_split_rows", "split_header_mismatch",
                   "split_id_missing_from_core", "split_row_mismatch", "split_count_mismatch",
                   "companion_url_absent_from_core", "companion_label_or_split_mismatch",
                   "abuse_type_not_linked_to_positive_core"}
        and value
    }
    if critical:
        raise ValueError(f"URL split/data integrity failed; no feature export: {critical}")
    state = process_development_features(
        open_zip, paths.processed / "phishvn_url_features",
        paths.reports / "phishvn_url_feature_checkpoint.json", rules_path,
        chunk_size=args.chunk_size,
    )
    print(json.dumps({
        "core_rows": audit["core_rows"], "labels": audit["distributions"]["label"],
        "official_splits": audit["distributions"]["split"],
        "feature_rows": state["rows"], "test_feature_rows_exported": 0,
        "checkpoint_complete": state["complete"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
