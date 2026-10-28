"""Research synthetic source tests; never access production or held-out files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from vietmailguard.v21 import research_synthetic_zh as research
from vietmailguard.v21.paths import resolve_v21_paths


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    raw = paths.raw / "research_synthetic_zh"
    raw.mkdir(parents=True)
    source = raw / "CN_Phishing_Email_dataset.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["content", "label"])
        for label in ("0", "1"):
            for index in range(8):
                text = "这是一封用于本地研究测试的中文邮件，内容完整且包含足够汉字。"
                if label == "1" and index == 7:
                    text += "钓鱼邮件示例"
                writer.writerow([text + f"编号{label}{index}并请核查原始通知。", label])
    config = {
        "schema_version": 1, "scope": "research_synthetic_zh_validation_only",
        "source_repo": "fixture/research", "source_revision": "fixture-revision",
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "declared_license": "fixture-only",
        "raw_label_mapping_research_only": {"0": "normal", "1": "phishing"},
        "seed": 42, "synthetic_challenge_fraction": 0.2,
        "minimum_han_characters": 5, "max_input_characters": 200,
        "minimum_free_ram_gib": 0,
        "tfidf": {
            "word": {"ngram_range": [1, 2], "min_df": 1, "max_features": 200},
            "character": {"analyzer": "char_wb", "ngram_range": [2, 3], "min_df": 1, "max_features": 200},
        },
        "classifiers": {
            "linear_svm": {"C": 1.0, "max_iter": 1000},
            "logistic_regression": {"C": 1.0, "solver": "lbfgs", "max_iter": 100},
        },
    }
    paths.config.mkdir(parents=True)
    (paths.config / "research_synthetic_zh.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    monkeypatch.setattr(research, "_read_reference_hashes", lambda _paths, _rows: 0)
    return paths, source


def test_screening_flags_not_labels() -> None:
    assert "empty_content" in research.screening_reasons("  ", "1", 5)
    assert "insufficient_chinese_text" in research.screening_reasons("English only", "0", 5)
    assert "explicit_label_or_example_marker" in research.screening_reasons("钓鱼邮件示例" * 3, "1", 5)
    assert "html_or_active_markup" in research.screening_reasons("中文正文" * 5 + "<script>x</script>", "0", 5)
    assert "invalid_source_label" in research.screening_reasons("中文正文" * 5, "spam", 5)
    assert research.screening_reasons("正常中文通知内容" * 4, "0", 5) == []


def test_duplicate_label_conflict_stays_out_of_research_train() -> None:
    content = "重复中文邮件内容" * 10
    exact, normalized, template = research.dedup_hashes("", content)
    rows = [
        {"raw_label": label, "label": mapped, "content_hash": exact,
         "normalized_content_hash": normalized, "template_hash": template, "_reasons": []}
        for label, mapped in (("0", "normal"), ("1", "phishing"))
    ]
    duplicate_count, conflict_count = research._group_rows(rows)
    assert (duplicate_count, conflict_count) == (1, 1)
    assert all("duplicate_group_label_conflict" in row["_reasons"] for row in rows)
    assert rows[0]["final_group_id"] == rows[1]["final_group_id"]


def test_audit_is_isolated_idempotent_and_excludes_label_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, source = _fixture(tmp_path, monkeypatch)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = research.audit_and_prepare(paths)
    assert manifest["source_rows"] == 16
    assert manifest["screened_out_rows"] == 1
    assert manifest["source_raw_label_counts"] == {"0": 8, "1": 8}
    assert manifest["synthetic_train_rows"] + manifest["synthetic_challenge_rows"] == 15
    assert manifest["cross_split_group_count"] == 0
    assert manifest["official_training_eligible_rows"] == 0
    assert research.audit_and_prepare(paths) == manifest
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    processed = paths.processed / "research_synthetic_zh"
    with (processed / "screening_manifest.csv").open(encoding="utf-8", newline="") as handle:
        screening = list(csv.DictReader(handle))
    assert len(screening) == 16
    assert not any("body" in row or "content" in row for row in screening)
    assert sum(row["screening_status"] == "exclude" for row in screening) == 1
    assert sum(row["screening_reasons"] == "explicit_label_or_example_marker" for row in screening) == 1


def test_train_and_challenge_groups_cannot_cross() -> None:
    row = {"id": "one", "final_group_id": "group", "content_hash": "hash",
           "normalized_content_hash": "normalized", "template_hash": "",
           "label": "normal", "research_training_eligible": "true"}
    train = pd.DataFrame([row, {**row, "id": "two", "final_group_id": "second",
                                "content_hash": "second", "normalized_content_hash": "second",
                                "label": "phishing"}])
    challenge = pd.DataFrame([{**row, "id": "three", "research_training_eligible": "false"},
                              {**row, "id": "four", "final_group_id": "four",
                               "content_hash": "four", "normalized_content_hash": "four",
                               "research_training_eligible": "false", "label": "phishing"}])
    with pytest.raises(research.ResearchExperimentBlocked, match="leakage"):
        research.validate_research_partitions(train, challenge)


def test_small_end_to_end_training_never_reads_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _fixture(tmp_path, monkeypatch)
    source = research.audit_and_prepare(paths)
    paths.splits.joinpath("development_bootstrap").mkdir(parents=True)
    (paths.splits / "development_bootstrap" / "manifest.json").write_text("{}", encoding="utf-8")
    forbidden = tmp_path / "data" / "splits" / "v2" / "test.csv"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("DO NOT READ", encoding="utf-8")
    bootstrap = pd.DataFrame([
        {"id": f"en-{i}", "subject": f"subject {label}",
         "body": f"english example text {label} {i}", "label": label,
         "language": "en", "data_origin": "not_declared", "source": "fixture"}
        for i, label in enumerate(["normal", "spam", "phishing"] * 3)
    ])
    validation = bootstrap.copy().assign(id=lambda frame: "val-" + frame["id"])
    monkeypatch.setattr(
        research, "verify_and_load_development",
        lambda _paths, _max_characters: (
            {"output_rows": {"train": len(bootstrap), "validation": len(validation)},
             "leakage": {"passed": True}}, bootstrap, validation,
        ),
    )
    monkeypatch.setattr(research, "_memory_snapshot", lambda _paths: {
        "free_ram_gib": 4, "free_disk_gib": 4,
    })
    result = research.train_research_candidates(paths)
    assert len(result["candidates"]) == 4
    assert result["synthetic_train_rows"] == source["synthetic_train_rows"]
    assert result["final_test_accessed"] is False
    assert forbidden.read_text(encoding="utf-8") == "DO NOT READ"
    for item in result["candidates"]:
        assert set(item["validation_metrics"]["per_class"]) == {"normal", "spam", "phishing"}
        assert item["synthetic_challenge_metrics"]["per_class"]["spam"]["support"] == 0
        assert Path(paths.project_root / item["artifact"]).is_file()


def test_tampered_synthetic_split_blocks_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _fixture(tmp_path, monkeypatch)
    research.audit_and_prepare(paths)
    train_path = paths.processed / "research_synthetic_zh" / "train.csv"
    train_path.write_text("changed", encoding="utf-8")
    with pytest.raises(research.ResearchExperimentBlocked, match="checksum"):
        research.train_research_candidates(paths)
