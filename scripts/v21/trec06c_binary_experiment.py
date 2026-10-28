"""Prepare and validate a separate TREC06C Ham/Spam experiment.

Run ``prepare`` before ``train``. ``train`` never reads the frozen test CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vietmailguard.v21.trec_binary_experiment import (  # noqa: E402
    prepare_experiment, train_validation, train_embedding_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "embedding"))
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare_experiment()
        print(json.dumps({key: result[key] for key in
                          ("raw_labels", "quality_accepted", "quality_rejected", "split_counts",
                           "group_stats", "leakage", "test_frozen", "test_evaluated")}, indent=2))
    else:
        result = train_validation() if args.stage == "train" else train_embedding_validation()
        print(json.dumps({"recommended_candidate": result["recommended_candidate"],
                          "embedding_status": result["embedding_status"],
                          "test_evaluated": result["test_evaluated"],
                          "candidates": [{"name": x["candidate"], "metrics": x["metrics"],
                                           "fit_seconds": x["fit_seconds"],
                                           "prediction_ms_per_email": x["prediction_ms_per_email"]}
                                          for x in result["candidates"]]}, indent=2))


if __name__ == "__main__":
    main()
