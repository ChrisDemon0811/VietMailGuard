from pathlib import Path

import pandas as pd
import pytest

from vietmailguard.dataset_standardizer import content_hash
from vietmailguard.split_builder import LeakageError, build_splits, check_cross_split_leakage


ROOT = Path(__file__).resolve().parents[1]
RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def _row(row_id: int, group_id: str, label: str, source: str = "source_a") -> dict[str, object]:
    subject = f"Subject {row_id}"
    body = f"Unique body content for row {row_id} in group {group_id}."
    return {
        "id": f"id-{row_id}",
        "original_row_id": row_id,
        "sender": "",
        "receiver": "",
        "date": "",
        "subject": subject,
        "body": body,
        "label": label,
        "raw_label": "1",
        "source": source,
        "language": "en",
        "url_count": 0,
        "content_hash": content_hash(subject, body),
        "group_id": group_id,
        "label_provenance": "fixture",
    }


def test_group_id_never_crosses_generated_splits() -> None:
    rows = []
    row_id = 1
    for label in ("normal", "spam", "phishing"):
        for group_number in range(20):
            size = 2 if group_number in {0, 1} else 1
            for _ in range(size):
                rows.append(_row(row_id, f"{label}-group-{group_number}", label))
                row_id += 1
    splits = build_splits(pd.DataFrame(rows), RATIOS, seed=42)

    group_sets = {name: set(frame["group_id"]) for name, frame in splits.items()}
    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])


def test_leakage_check_raises_on_reused_group() -> None:
    train = pd.DataFrame([_row(1, "shared-group", "normal")])
    validation = pd.DataFrame([_row(2, "validation-group", "normal")])
    test = pd.DataFrame([_row(3, "shared-group", "normal")])

    with pytest.raises(LeakageError):
        check_cross_split_leakage(
            {"train": train, "validation": validation, "test": test}
        )


def test_generated_repository_splits_have_disjoint_groups() -> None:
    paths = {
        name: ROOT / "data" / "splits" / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("Repository splits have not been generated yet")
    splits = {
        name: pd.read_csv(path, keep_default_na=False, low_memory=False)
        for name, path in paths.items()
    }

    result = check_cross_split_leakage(splits)

    assert result["passed"] is True
