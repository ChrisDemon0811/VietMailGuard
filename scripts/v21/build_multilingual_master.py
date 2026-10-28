"""Prepare full V2.1 splits, or inspect the isolated development bootstrap."""

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.v21.master_split import PreparationBlocked, build_master_split  # noqa: E402
from vietmailguard.v21.development_dataset import (  # noqa: E402
    DevelopmentDatasetBlocked,
    build_development_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "development-bootstrap"),
                        default="full", help="Development mode preserves V2 Train/Validation and creates no Test")
    args = parser.parse_args(argv)
    try:
        if args.mode == "development-bootstrap":
            manifest = build_development_dataset()
            if manifest.get("final_evaluation_ready") is not False or manifest.get("v2_test_rows_exported") != 0:
                raise DevelopmentDatasetBlocked("Development dataset must not expose a final Test")
            print(json.dumps({"status": manifest["status"], "mode": manifest["mode"],
                              "output_rows": manifest["output_rows"],
                              "development_training_ready": manifest["development_training_ready"],
                              "final_evaluation_ready": False}, indent=2))
        else:
            build_master_split()
    except (PreparationBlocked, DevelopmentDatasetBlocked) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
