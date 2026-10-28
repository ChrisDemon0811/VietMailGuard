"""Create a read-only V2.1 source-label approval dossier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.v21.source_approval import run_source_approval_audit  # noqa: E402


def main() -> None:
    print(json.dumps(run_source_approval_audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
