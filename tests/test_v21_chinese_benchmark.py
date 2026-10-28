"""No network, model, or held-out Test access in benchmark preparation tests."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from vietmailguard.v21 import chinese_benchmark as benchmark
from vietmailguard.v21.dedup_audit import dedup_hashes
from vietmailguard.v21.paths import resolve_v21_paths


def _row(candidate_id: str, body: str) -> dict[str, str]:
    row = {name: "" for name in benchmark.INPUT_FIELDS}
    row.update({
        "candidate_id": candidate_id, "source": "independent_local_source",
        "source_record_id": candidate_id, "source_version": "fixture-v1",
        "source_url": "https://example.org/source-card", "license_status": "verified_research_use",
        "permission_evidence": "local-rights-record", "data_origin": "original",
        "language_claim": "zh", "raw_label": "phish-reviewed",
        "proposed_label": "phishing", "label_evidence": "review-note-id",
        "reviewer_1_id": "reviewer-a", "reviewer_1_label": "phishing",
        "reviewer_2_id": "reviewer-b", "reviewer_2_label": "phishing",
        "body": body,
    })
    return row


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark.INPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_empty_prepare_is_pending_and_idempotent(tmp_path: Path) -> None:
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    first = benchmark.prepare_candidate_manifest(paths=paths)
    second = benchmark.prepare_candidate_manifest(paths=paths)
    assert first == second
    assert first["candidate_rows"] == first["accepted_benchmark_rows"] == 0
    assert first["benchmark_status"] == "pending"
    assert not first["reference_fingerprints_checked"]
    assert not first["final_evaluation_ready"]
    assert not first["held_out_test_opened"]
    with (paths.evaluation / "chinese_real_world_benchmark/candidate_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert list(csv.DictReader(handle)) == []


def test_intake_flags_duplicate_and_existing_overlap_without_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    input_path = tmp_path / "private_candidates.csv"
    chinese = "\u8fd9\u662f\u4e00\u5c01\u4e2d\u6587\u7535\u5b50\u90ae\u4ef6\u5185\u5bb9\u7528\u4e8e\u672c\u5730\u5b89\u5168\u68c0\u67e5" * 3
    other = chinese + "\u53e6\u4e00\u6761\u72ec\u7acb\u901a\u77e5"
    _write_rows(input_path, [_row("a", chinese), _row("b", chinese), _row("c", other)])
    existing_hash = dedup_hashes("", other)[0]

    def reference(_paths):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE rows (dataset TEXT, exact_hash TEXT, normalized_hash TEXT, template_hash TEXT)")
        conn.execute("INSERT INTO rows VALUES (?,?,?,?)", ("trec06c", existing_hash, "", ""))
        return conn

    monkeypatch.setattr(benchmark, "_reference_connection", reference)
    monkeypatch.setattr(benchmark, "_synthetic_hashes", lambda _paths: set())
    result = benchmark.prepare_candidate_manifest(input_path, paths=paths)
    assert result["candidate_rows"] == 3
    assert result["internal_duplicate_rows"] == 2
    assert result["internal_duplicate_hash_groups"] == {"exact": 1, "normalized": 1, "template": 0}
    assert result["reference_overlap_rows"] == 1
    assert result["accepted_benchmark_rows"] == 0
    assert not result["final_evaluation_ready"]
    with (paths.evaluation / "chinese_real_world_benchmark/candidate_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    assert rows["a"]["internal_duplicate_kinds"] == "exact;normalized"
    assert rows["b"]["benchmark_status"] == "pending"
    assert rows["c"]["reference_overlap"] == "trec06c"
    assert "body" not in rows["a"] and chinese not in str(rows)
    assert benchmark.prepare_candidate_manifest(input_path, paths=paths) == result
    with pytest.raises(ValueError, match="empty run"):
        benchmark.prepare_candidate_manifest(paths=paths)
    _write_rows(input_path, [_row("changed", chinese)])
    with pytest.raises(ValueError, match="changed since screening checkpoint"):
        benchmark.prepare_candidate_manifest(input_path, paths=paths)


def test_synthetic_or_trec_cannot_be_silently_admitted() -> None:
    text = "\u4e2d\u6587\u7535\u5b50\u90ae\u4ef6\u6d4b\u8bd5\u5185\u5bb9" * 4
    row = _row("synthetic", text)
    row.update({"source": "trec06c", "data_origin": "synthetic"})
    flags = benchmark._flags(row, dedup_hashes("", text))
    assert "already_used_or_synthetic_source" in flags
    assert "not_original_email" in flags


def test_translation_or_ai_label_is_not_human_ground_truth() -> None:
    text = "\u4e2d\u6587\u7535\u5b50\u90ae\u4ef6\u6d4b\u8bd5\u5185\u5bb9" * 4
    row = _row("translated", text)
    row.update({"data_origin": "translated", "reviewer_2_id": "reviewer-a"})
    flags = benchmark._flags(row, dedup_hashes("", text))
    assert "not_original_email" in flags
    assert "independent_chinese_review_unverified" in flags


def test_reference_fingerprint_coverage_must_include_frozen_v2_rows(tmp_path: Path) -> None:
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    paths.cache.mkdir(parents=True)
    index = paths.cache / "dedup_audit.sqlite3"
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE rows (dataset TEXT)")
        connection.execute("INSERT INTO rows VALUES ('v2_reference')")
    with pytest.raises(ValueError, match="coverage is incomplete"):
        benchmark._reference_connection(paths)
