"""Run the checkpointed V2.1 email audit without merging/training."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.v21.dedup_audit import run_audit  # noqa: E402


if __name__ == "__main__":
    run_audit()
