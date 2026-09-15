"""Create deterministic, review-only samples for raw-label semantics analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
)


def representative_indices(group: pd.DataFrame, sample_size: int) -> list[int]:
    """Select deterministic samples spread across source order plus seeded random rows."""
    if len(group) <= sample_size:
        return [int(index) for index in group.index]
    spread_count = min(5, sample_size)
    positions = {
        round(position * (len(group) - 1) / max(spread_count - 1, 1))
        for position in range(spread_count)
    }
    selected = {int(group.index[position]) for position in positions}
    remaining = sample_size - len(selected)
    if remaining > 0:
        pool = group.drop(index=list(selected), errors="ignore")
        selected.update(int(index) for index in pool.sample(n=remaining, random_state=42).index)
    return sorted(selected)


def compact(value: object, limit: int = 700) -> str | None:
    """Make a field reviewable without copying complete email bodies into the report."""
    if pd.isna(value):
        return None
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit]}…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "label_review_samples.json",
    )
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    evidence: dict[str, dict[str, list[dict[str, object]]]] = {}
    for filename in DEFAULT_DATASETS:
        frame = pd.read_csv(args.raw_dir / filename, encoding="utf-8-sig", low_memory=False)
        evidence[filename] = {}
        for raw_label, group in frame.groupby("label", dropna=False, sort=True):
            label_key = "<MISSING>" if pd.isna(raw_label) else str(raw_label)
            rows: list[dict[str, object]] = []
            for index in representative_indices(group, args.sample_size):
                row = frame.loc[index]
                rows.append(
                    {
                        "source_row_number": int(index) + 2,
                        "sender": compact(row.get("sender"), 180),
                        "subject": compact(row.get("subject"), 240),
                        "body_excerpt": compact(row.get("body"), 700),
                        "urls_raw": compact(row.get("urls"), 80),
                    }
                )
            evidence[filename][label_key] = rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve().relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

