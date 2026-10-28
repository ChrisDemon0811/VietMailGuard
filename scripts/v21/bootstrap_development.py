"""Build a V2.1 development-only dataset from V2 Train/Validation."""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.v21.bootstrap_development import build_development_bootstrap  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(build_development_bootstrap(), ensure_ascii=False, indent=2))
