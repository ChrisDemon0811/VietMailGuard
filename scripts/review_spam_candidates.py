"""Interactive, resumable review of conservative spam candidates."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.spam_curation import (  # noqa: E402
    MANUAL_LABEL_COLUMNS,
    load_manual_labels,
    select_pending_candidates,
)

CHOICES = {
    "s": "spam",
    "p": "phishing_or_scam",
    "r": "review",
    "x": "exclude",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review spam candidates and resume saved progress.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "reports" / "spam_curation_candidates.csv",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=PROJECT_ROOT / "data" / "curated" / "spam_manual_labels.csv",
    )
    parser.add_argument("--source", choices=["SpamAssassin", "Ling", "Enron", "CEAS_08"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reviewer", default="interactive_user")
    parser.add_argument("--revisit-deferred", action="store_true")
    return parser.parse_args()


def _save_decision(path: Path, decisions: pd.DataFrame, row: dict[str, object]) -> pd.DataFrame:
    key = decisions["source"].eq(row["source"]) & decisions["original_row_id"].eq(
        row["original_row_id"]
    )
    decisions = decisions.loc[~key]
    decisions = pd.concat([decisions, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    decisions[MANUAL_LABEL_COLUMNS].to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)
    return decisions


def main() -> int:
    args = parse_args()
    candidates = pd.read_csv(args.candidates, keep_default_na=False)
    decisions = load_manual_labels(args.decisions)
    if args.source:
        candidates = candidates.loc[candidates["source"].eq(args.source)]

    pending = select_pending_candidates(
        candidates, decisions, revisit_deferred=args.revisit_deferred
    )
    pending_rows = pending.to_dict(orient="records")
    if args.limit is not None:
        pending_rows = pending_rows[: args.limit]

    if not pending_rows:
        print("No pending candidates for the selected options.")
        return 0

    for position, row in enumerate(pending_rows, start=1):
        print("\n" + "=" * 78)
        print(f"Candidate {position}/{len(pending_rows)}")
        print(f"Source: {row['source']} | Original row: {row['original_row_id']}")
        print(f"Subject:\n{row['subject'] or '(empty subject)'}")
        print(f"\nBody:\n{row['body_excerpt'] or '(empty body)'}")
        print(f"\nSuggested: {row['proposed_label']} ({row['confidence']})")
        print(f"Reason: {row['review_reason']}")
        print(f"Spam rules: {row['triggered_spam_rules'] or 'none'}")
        print(f"Phishing/scam rules: {row['triggered_phishing_rules'] or 'none'}")

        while True:
            choice = input("\n[S] Spam  [P] Phishing/scam  [R] Review later  [X] Exclude  [Q] Quit: ").strip().lower()
            if choice == "q":
                print(f"Progress saved in {args.decisions}")
                return 0
            if choice in CHOICES:
                break
            print("Invalid choice. Enter S, P, R, X, or Q.")

        decision = CHOICES[choice]
        note = input("Optional note: ").strip()
        decisions = _save_decision(
            args.decisions,
            decisions,
            {
                "source": row["source"],
                "original_row_id": int(row["original_row_id"]),
                "manual_decision": decision,
                "review_status": "needs_human_review" if decision == "review" else "confirmed",
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                "reviewer": args.reviewer,
                "note": note,
            },
        )
        print(f"Saved: {decision}")

    print(f"Completed selected queue. Progress saved in {args.decisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
