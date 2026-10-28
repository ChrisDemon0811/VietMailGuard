"""Prepare a pending, metadata-only Chinese benchmark candidate manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vietmailguard.v21.chinese_benchmark import prepare_candidate_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        help="Explicit local CSV with documented research rights; omitted for an empty pending manifest",
    )
    args = parser.parse_args()
    try:
        summary = prepare_candidate_manifest(args.input)
    except (FileNotFoundError, ValueError) as error:
        print(f"Benchmark preparation blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
