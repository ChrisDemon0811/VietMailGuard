"""Small source-only regression tests for the isolated TREC binary protocol."""

from __future__ import annotations

import csv
import sqlite3
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vietmailguard.v21.trec_binary_experiment import (
    _load_development, assign_groups, binary_metrics, prepare_experiment, quality_reason,
    verify_no_leakage,
)
from vietmailguard.v21.paths import resolve_v21_paths


def _row(label: str, exact: str, normalized: str, template: str = "") -> dict[str, str]:
    return {"raw_label": label, "exact_hash": exact, "normalized_hash": normalized,
            "template_hash": template}


def test_quality_gate_keeps_source_labels_not_three_class() -> None:
    row = {"id": "trec06c:1", "source": "TREC06C", "raw_label": "spam",
           "status": "ok", "empty": "False", "missing_content": "False",
           "decode_error": "False", "contains_chinese": "True", "body": "中文内容",
           "original_path": "data/1", "content_hash": "abc"}
    assert quality_reason(row) == ""
    assert quality_reason({**row, "raw_label": "phishing"}) == "unexpected_source_or_label"
    assert quality_reason({**row, "decode_error": "True"}) == "decode_warning"
    assert quality_reason({**row, "contains_chinese": "False"}) == "no_han_character"


def test_group_assignment_transitive_template_and_no_leakage() -> None:
    rows = {f"id{i}": _row("ham" if i < 18 else "spam", f"e{i}", f"n{i}")
            for i in range(36)}
    rows["id0"]["exact_hash"] = rows["id1"]["exact_hash"]
    rows["id1"]["template_hash"] = rows["id2"]["template_hash"] = "shared_template"
    assigned, stats = assign_groups(rows)
    assert stats["conflicting_groups"] == 0
    assert assigned["id0"] == assigned["id1"] == assigned["id2"]
    assert verify_no_leakage(rows, assigned)["passed"]
    assert set(value.split("|")[0] for value in assigned.values()) == {"train", "validation", "test"}
    assert assigned == assign_groups(rows)[0]


def test_conflicting_duplicate_group_is_not_admitted() -> None:
    rows = {"ham": _row("ham", "same", "n1"),
            "spam": _row("spam", "same", "n2")}
    assigned, stats = assign_groups(rows)
    assert assigned == {}
    assert stats["conflicting_rows"] == 2


def test_leakage_check_rejects_cross_split_template() -> None:
    rows = {"a": _row("ham", "a", "a", "template"),
            "b": _row("spam", "b", "b", "template")}
    with pytest.raises(ValueError, match="leakage"):
        verify_no_leakage(rows, {"a": "train|g1", "b": "test|g2"})


def test_binary_metrics_false_positive_definition() -> None:
    metrics = binary_metrics(np.array(["ham", "ham", "spam", "spam"]),
                             np.array(["ham", "spam", "ham", "spam"]))
    assert metrics["false_positives_ham_as_spam"] == 1
    assert metrics["false_negatives_spam_as_ham"] == 1
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_training_loader_does_not_open_test_split(tmp_path: Path) -> None:
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    split_dir = paths.splits / "trec06c_binary"
    split_dir.mkdir(parents=True)
    hashes = {}
    for name in ("train", "validation"):
        path = split_dir / f"{name}.csv"
        pd.DataFrame([
            {"id": f"{name}-1", "subject": "中文", "body": "正常内容",
             "raw_label": "ham", "final_group_id": f"{name}-g1", "content_hash": f"{name}-h1"},
            {"id": f"{name}-2", "subject": "中文", "body": "广告内容",
             "raw_label": "spam", "final_group_id": f"{name}-g2", "content_hash": f"{name}-h2"},
        ]).to_csv(path, index=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"split_sha256": hashes}
    # Test is deliberately absent. The development loader must still succeed.
    train, validation = _load_development(split_dir, manifest)
    assert len(train) == len(validation) == 2
    assert not (split_dir / "test.csv").exists()


def test_prepare_small_audited_parts_and_freeze_test(tmp_path: Path) -> None:
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    part_dir = paths.processed / "trec06c_mime_v2"
    part_dir.mkdir(parents=True)
    part = part_dir / "part-00001.csv"
    fields = ("id", "source", "raw_label", "status", "empty", "missing_content",
              "decode_error", "contains_chinese", "subject", "body", "original_path",
              "content_hash", "language_candidate")
    with part.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(20):
            writer.writerow({"id": f"trec06c:{i:03}", "source": "TREC06C",
                "raw_label": "ham" if i < 10 else "spam", "status": "ok",
                "empty": "False", "missing_content": "False", "decode_error": "False",
                "contains_chinese": "True", "subject": f"主题{i}", "body": f"中文内容{i}",
                "original_path": f"data/{i}", "content_hash": f"content{i}",
                "language_candidate": "zh"})
    paths.cache.mkdir(parents=True)
    connection = sqlite3.connect(paths.cache / "dedup_audit.sqlite3")
    connection.execute("CREATE TABLE rows (dataset TEXT, original_id TEXT, exact_hash TEXT, normalized_hash TEXT, template_hash TEXT)")
    connection.execute("CREATE TABLE checkpoints (path TEXT, size_bytes INTEGER, mtime_ns INTEGER)")
    for i in range(20):
        connection.execute("INSERT INTO rows VALUES (?,?,?,?,?)",
                           ("trec06c", f"trec06c:{i:03}", f"exact{i}", f"normal{i}", ""))
    stat = part.stat()
    connection.execute("INSERT INTO checkpoints VALUES (?,?,?)",
                       (str(part.resolve()), stat.st_size, stat.st_mtime_ns))
    connection.commit()
    connection.close()
    manifest = prepare_experiment(paths)
    assert manifest["quality_accepted"] == 20
    assert manifest["test_frozen"] and not manifest["test_evaluated"]
    assert manifest["leakage"]["passed"]
    test_path = paths.splits / "trec06c_binary" / "test.csv"
    original_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
    assert prepare_experiment(paths) == manifest
    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == original_hash
    test_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen test split changed"):
        prepare_experiment(paths)
