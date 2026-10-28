"""Load only indexed TREC06C originals from 2025-123.zip into review-only CSV parts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.v21.paths import resolve_v21_paths  # noqa: E402
from vietmailguard.v21.trec06c import PARSER_VERSION, load_trec06c  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-member-bytes", type=int, default=10_000_000)
    args = parser.parse_args()
    paths = resolve_v21_paths()
    report = load_trec06c(
        paths.dataset_dir / "2025-123.zip",
        paths.processed / f"trec06c_mime_v{PARSER_VERSION}",
        paths.reports / "trec06c_loader_report.json",
        batch_size=args.batch_size,
        max_member_bytes=args.max_member_bytes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
