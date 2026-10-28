"""Run the isolated Chinese synthetic source audit or A/B research training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vietmailguard.v21.research_synthetic_zh import (  # noqa: E402
    ResearchExperimentBlocked,
    audit_and_prepare,
    compare_current_svm,
    train_research_candidates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "train", "current_svm"))
    args = parser.parse_args()
    try:
        if args.stage == "audit":
            result = audit_and_prepare()
            summary = {
                key: result[key] for key in (
                    "source_rows", "source_raw_label_counts", "research_usable_rows",
                    "screened_out_rows", "synthetic_train_rows", "synthetic_challenge_rows",
                    "synthetic_duplicate_groups", "synthetic_conflicting_label_groups",
                    "cross_corpus_fingerprint_overlap_rows", "cross_split_group_count",
                )
            }
        elif args.stage == "train":
            result = train_research_candidates()
            summary = {
                "bootstrap_train_rows": result["bootstrap_train_rows"],
                "validation_rows": result["validation_rows"],
                "synthetic_train_rows": result["synthetic_train_rows"],
                "synthetic_challenge_rows": result["synthetic_challenge_rows"],
                "candidates": [
                    {"experiment": row["experiment"], "classifier": row["classifier"],
                     "macro_f1": row["validation_metrics"]["macro_f1"],
                     "phishing_recall": row["validation_metrics"]["phishing_recall"],
                     "fit_seconds": row["fit_seconds"]}
                    for row in result["candidates"]
                ],
            }
        else:
            result = compare_current_svm()
            summary = {
                "bootstrap_train_rows": result["bootstrap_train_rows"],
                "synthetic_train_rows": result["synthetic_train_rows"],
                "validation_rows": result["validation_rows"],
                "A_current_macro_f1": result["A_current_candidate"]["validation_metrics"]["macro_f1"],
                "B_synthetic_macro_f1": result["B_synthetic_added"]["validation_metrics"]["macro_f1"],
                "macro_f1_delta_B_minus_A": result["validation_macro_f1_delta_B_minus_A"],
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (ResearchExperimentBlocked, OSError, ValueError) as error:
        print(f"Research stage blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
