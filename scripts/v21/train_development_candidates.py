"""Train V2.1 development candidates without opening a final test set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vietmailguard.v21.development_training import train_development_candidates  # noqa: E402


if __name__ == "__main__":
    result = train_development_candidates()
    print(json.dumps({
        "train_rows": result["train_rows"],
        "validation_rows": result["validation_rows"],
        "development_recommendation": result["development_recommendation"],
        "embedding_skip_reason": result["embedding_skip_reason"],
        "xlm_roberta": result["xlm_roberta"],
        "candidate_metrics": [
            {"candidate": item["candidate"],
             "macro_f1": item["metrics"]["macro_f1"],
             "phishing_recall": item["metrics"]["phishing_recall"]}
            for item in result["candidates"]
        ],
    }, indent=2))
