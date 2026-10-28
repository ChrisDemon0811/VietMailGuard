"""Run evidence-gated V2.1 label curation; never train or split here."""

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.v21.label_curation import run_curation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, help="Local human decision ledger CSV")
    args = parser.parse_args()
    print(json.dumps(run_curation(decisions_path=args.decisions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
