"""Local TREC06C review queue and explicit individual human decisions."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from vietmailguard.v21.paths import resolve_v21_paths
from vietmailguard.v21.trec_curation import (
    _read_decisions, build_review_queue, load_full_message, record_human_decision,
)
from vietmailguard.v21.trec_review_plan import prepare_deep_review, summarize_deep_review


QUEUE_FILES = {
    "stratified": "review_queue.csv",
    "ham-probability": "ham_probability_sample.csv",
    "spam-focus": "spam_focus_queue.csv",
}


def _queue(paths, name):
    path = paths.curated / "trec06c" / QUEUE_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"Review queue missing: {path}; run prepare or prepare-deep first")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Build local stratified queue; no labels are promoted")
    prepare.add_argument("--per-stratum", type=int, default=3)
    sub.add_parser("prepare-deep", help="Freeze expanded Ham/Spam review frames; no bulk approval")
    sub.add_parser("summarize", help="Report observed human decisions and uncertainty without relabeling")
    review = sub.add_parser("review", help="Review queued full messages interactively")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--method", default="full_message_manual_review")
    review.add_argument("--id", help="Review one queued ID")
    review.add_argument("--queue", choices=tuple(QUEUE_FILES), default="ham-probability")
    review.add_argument("--raw-label", choices=("ham", "spam"))
    review.add_argument("--limit", type=int, default=20)
    sub.add_parser("status", help="Show counts only, never message text")
    args = parser.parse_args()
    paths = resolve_v21_paths()
    if args.command == "prepare":
        result = build_review_queue(paths, per_stratum=args.per_stratum)
        print(json.dumps({"raw_labels": result["raw_labels"],
                          "queue_rows": result["queue_rows"],
                          "existing_decisions": result["existing_decisions"],
                          "eligible": result["eligible"]}, ensure_ascii=False))
        return 0
    if args.command == "prepare-deep":
        result = prepare_deep_review(paths)
        print(json.dumps({
            "ham_conditional_pool_pending_approval": sum(result["ham_frame_by_length"].values()),
            "ham_sample_rows": result["ham_sample_rows"],
            "spam_sample_rows": result["spam_sample_rows"],
            "bulk_rule_active": result["bulk_rule_active"],
        }, ensure_ascii=False))
        return 0
    if args.command == "summarize":
        result = summarize_deep_review(paths)
        print(json.dumps({key: result[key] for key in (
            "human_decisions", "eligible_by_class", "weak_label_unreviewed",
            "blocked_class_decisions", "ham_conditional_pool_pending_approval",
            "proposed_bulk_eligible_now",
        )}, ensure_ascii=False))
        return 0
    if args.command == "status":
        summary = paths.reports / "trec06c_curation_summary.json"
        if not summary.is_file():
            parser.error("Run prepare first")
        result = json.loads(summary.read_text(encoding="utf-8"))
        decisions = _read_decisions(paths.curated / "trec06c/decisions.csv")
        print(json.dumps({"raw_labels": result["raw_labels"],
                          "queue_rows": result["queue_rows"],
                          "decisions": len(decisions),
                          "eligible": sum(x["label_status"] == "eligible" for x in decisions.values())},
                         ensure_ascii=False))
        return 0
    if not sys.stdin.isatty():
        parser.error("Interactive review requires a local terminal")
    previous = _read_decisions(paths.curated / "trec06c/decisions.csv")
    if args.limit < 1:
        parser.error("--limit must be positive")
    chosen = [row for row in _queue(paths, args.queue)
              if row["id"] not in previous and (not args.id or row["id"] == args.id)
              and (not args.raw_label or row["raw_label"] == args.raw_label)][:args.limit]
    if args.id and not chosen:
        parser.error("ID not in pending queue or already reviewed")
    for item in chosen:
        email = load_full_message(paths, item)
        # Full content is shown only in the local interactive terminal, not logged.
        print(f"\nID: {email['id']} | TREC raw label: {email['raw_label']} | Path: {email['original_path']}")
        print(f"Review cohort: {item.get('review_cohort', args.queue)} | stratum: {item.get('review_stratum', item.get('quality_stratum', ''))}")
        print(f"Quality flags: {item.get('quality_flags') or 'none'} | duplicate: {item.get('duplicate_stratum')}")
        print(f"Subject: {email['subject']}\nBody:\n{email['body']}")
        print("[N] Normal  [S] Spam  [P] Phishing  [R] Unresolved review  [X] Exclude  [K] Skip  [Q] Quit")
        choice = input("Decision: ").strip().casefold()
        mapping = {"n": "normal", "s": "spam", "p": "phishing", "r": "review", "x": "exclude"}
        if choice == "q":
            break
        if choice == "k":
            continue
        if choice not in mapping:
            print("Invalid choice; no decision stored.")
            continue
        evidence = input("Evidence/reason (required; do not paste sensitive text): ").strip()
        try:
            result = record_human_decision(paths, email, decision=mapping[choice],
                                           reviewer=args.reviewer, method=args.method,
                                           evidence=evidence,
                                           review_cohort=item.get("review_cohort") or args.queue,
                                           review_stratum=item.get("review_stratum") or item.get("quality_stratum", ""))
        except ValueError as exc:
            print(f"Not stored: {exc}")
            continue
        print(f"Saved {result['label_status']} decision for {result['id']}")
        if result.get("eligibility_blocker"):
            print(f"Training eligibility blocked: {result['eligibility_blocker']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
