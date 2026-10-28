"""Stream the configured original Enron corpus into review-only V2.1 CSV parts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.v21.enron_loader import discover_enron_source, load_enron  # noqa: E402
from vietmailguard.v21.paths import resolve_v21_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Original maildir, tar/tar.gz, or ZIP; otherwise auto-discover")
    parser.add_argument("--v2-reference", type=Path, help="Read-only Enron.csv comparison source")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-member-bytes", type=int, default=10_000_000)
    args = parser.parse_args()
    paths = resolve_v21_paths()
    source = args.input or discover_enron_source(paths.dataset_dir)
    reference = args.v2_reference or paths.project_root / "data" / "raw" / "Enron.csv"
    if not reference.is_file():
        reference = None
    result = load_enron(
        source,
        paths.processed / "enron_original_v1",
        paths.reports / "enron_original_loader_report.json",
        v2_reference=reference,
        batch_size=args.batch_size,
        max_member_bytes=args.max_member_bytes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
