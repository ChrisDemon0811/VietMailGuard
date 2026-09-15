"""Load the saved production pipeline and run an offline prediction smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.modeling import confidence_for_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the serialized production model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "production_pipeline.joblib",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = joblib.load(args.model)
    sample = pd.DataFrame(
        [
            {
                "subject": "Project meeting notes",
                "body": "Hello team, attached are the notes and action items from today's meeting.",
            }
        ]
    )
    predictions, probabilities = confidence_for_rows(pipeline, sample)
    classes = [str(value) for value in pipeline.named_steps["classifier"].classes_]
    probability_map = {
        label: float(probability)
        for label, probability in zip(classes, probabilities[0], strict=True)
    }
    print(f"prediction={predictions[0]}")
    print(f"probabilities={probability_map}")
    print(f"probability_sum={probabilities[0].sum():.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
