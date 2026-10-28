"""TREC-specific review policy: no binary-label bulk promotion."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from vietmailguard.v21.dedup_audit import FIELDS, _connect, _finalize_groups, _row
from vietmailguard.v21.paths import resolve_v21_paths
from vietmailguard.v21.trec_curation import (
    _read_decisions, build_review_queue, load_full_message, record_human_decision,
)
from vietmailguard.v21.trec_review_plan import _focus, prepare_deep_review, summarize_deep_review
from vietmailguard.v21.trec_review_assist import (
    GeminiTranslator, TrecReviewStore, TRANSLATION_CHUNK_CHARS,
    MAX_TRANSLATION_REQUESTS, TrecGeminiSuggestionProvider,
    _protect, _restore, _translation_chunks,
)
from vietmailguard.v21.llm_reviewer import ReviewRequest
from scripts.v21.trec06c_review_app import load_expert_queue, neighbor_id, visible_queue


def _source_row(id_: str, label: str, body: str, *, decode_error: bool = False) -> dict[str, str]:
    return {
        "id": id_, "source": "TREC06C", "raw_label": label,
        "candidate_label": label, "label": "review", "subject": "邮件通知",
        "body": body, "original_path": f"../data/{id_.split(':')[1]}",
        "language_candidate": "zh", "status": "ok", "empty": "False",
        "missing_content": "False", "decode_error": str(decode_error),
        "mime_recovered": "False", "mime_defects": "", "html_only": "False",
        "contains_chinese": "True", "content_hash": hashlib.sha256(id_.encode()).hexdigest(),
    }


@pytest.fixture
def setup_trec(tmp_path: Path):
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    paths.config.mkdir(parents=True)
    (paths.config / "trec06c_curation_policy.json").write_text(json.dumps({
        "schema_version": 1, "source": "TREC06C", "bulk_approval": False,
        "automatic_phishing_from_spam": False,
        "individual_review_rule_id": "trec06c_individual_human_content_review_v1",
        "allowed_manual_decisions": ["normal", "spam", "phishing", "review", "exclude"],
    }), encoding="utf-8")
    part = paths.processed / "trec06c_mime_v2/part-00001.csv"
    part.parent.mkdir(parents=True)
    rows = [
        _source_row("trec06c:000/001", "ham", "明天举行项目会议，请参加。"),
        _source_row("trec06c:000/002", "spam", "商品限时销售，欢迎了解优惠。"),
        _source_row("trec06c:000/003", "ham", "相同内容用于冲突检查。"),
        _source_row("trec06c:000/004", "spam", "相同内容用于冲突检查。"),
        _source_row("trec06c:000/005", "spam", "损坏内容需要人工核对。", decode_error=True),
    ]
    with part.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    db = paths.cache / "dedup_audit.sqlite3"
    conn = _connect(db)
    try:
        conn.executemany(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})",
                         [_row("trec06c", row, part) for row in rows])
        stat = part.stat()
        conn.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)",
                     (str(part.resolve()), stat.st_size, stat.st_mtime_ns, len(rows), 1))
        _finalize_groups(conn)
        conn.commit()
    finally:
        conn.close()
    return paths, rows


def _decision(paths, row, decision):
    return record_human_decision(
        paths, row, decision=decision, reviewer="reviewer-1",
        method="full_message_manual_review", evidence="Reviewed full message and taxonomy",
        reviewed_at_utc="2026-10-21T00:00:00+00:00",
    )


def test_queue_is_stratified_local_and_resumable(setup_trec) -> None:
    paths, rows = setup_trec
    frozen = paths.project_root / "models/v2_bilingual/production_pipeline.joblib"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"frozen-V2-sentinel")
    original_hash = hashlib.sha256(frozen.read_bytes()).hexdigest()
    first = build_review_queue(paths, per_stratum=2)
    assert first["raw_labels"] == {"ham": 2, "spam": 3}
    assert first["eligible"] == 0
    assert first["conflicting_groups"]["normalized_group_id"] == 1
    assert (paths.curated / "trec06c/review_queue.csv").is_file()
    assert _read_decisions(paths.curated / "trec06c/decisions.csv") == {}
    _decision(paths, rows[0], "normal")
    assert TrecReviewStore(paths).qualified_label(
        rows[0]["id"], _read_decisions(paths.curated / "trec06c/decisions.csv")[rows[0]["id"]]
    ) is None
    second = build_review_queue(paths, per_stratum=2)
    assert second["existing_decisions"] == 1
    assert second["eligible"] == 1
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == original_hash
    with (paths.curated / "trec06c/review_queue.csv").open(encoding="utf-8", newline="") as handle:
        assert rows[0]["id"] not in {r["id"] for r in csv.DictReader(handle)}


def test_only_individually_reviewed_singletons_become_eligible(setup_trec) -> None:
    paths, rows = setup_trec
    assert _decision(paths, rows[0], "normal")["label_status"] == "eligible"
    assert _decision(paths, rows[1], "spam")["label_provenance"] == "trec06c_individual_human_review"
    manifest = _read_decisions(paths.curated / "trec06c/decisions.csv")
    assert len(manifest) == 2
    assert all(manifest[r["id"]]["final_group_id"] for r in rows[:2])
    with pytest.raises(ValueError, match="already has a decision"):
        _decision(paths, rows[0], "normal")
    assert len(_read_decisions(paths.curated / "trec06c/decisions.csv")) == 2


def test_duplicate_conflict_and_decode_error_cannot_enter_eligible(setup_trec) -> None:
    paths, rows = setup_trec
    for row in rows[2:4]:
        result = _decision(paths, row, "spam")
        assert result["label_status"] == "review"
        assert result["eligibility_blocker"] == "duplicate_label_conflict"
    result = _decision(paths, rows[4], "spam")
    assert result["label_status"] == "review"
    assert result["eligibility_blocker"] == "content_quality_review"
    assert not any(x["label_status"] == "eligible" for x in _read_decisions(paths.curated / "trec06c/decisions.csv").values())


def test_no_ml_llm_method_and_queue_provenance_guard(setup_trec) -> None:
    paths, rows = setup_trec
    with pytest.raises(ValueError, match="required"):
        record_human_decision(paths, rows[0], decision="normal", reviewer="bot",
                              method="model_prediction", evidence="score")
    build_review_queue(paths)
    with (paths.curated / "trec06c/review_queue.csv").open(encoding="utf-8", newline="") as handle:
        queued = next(csv.DictReader(handle))
    assert load_full_message(paths, queued)["id"] == queued["id"]
    tampered = dict(queued, original_path="../data/999/999")
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_full_message(paths, tampered)


def test_manifest_rejects_unapproved_eligible_rule(setup_trec) -> None:
    paths, rows = setup_trec
    _decision(paths, rows[0], "normal")
    manifest = paths.curated / "trec06c/decisions.csv"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace("trec06c_individual_human_content_review_v1", "source_label_guess"), encoding="utf-8")
    with pytest.raises(ValueError, match="approved rule"):
        _read_decisions(manifest)


def test_deep_review_is_inactive_stratified_and_resumable(setup_trec) -> None:
    paths, rows = setup_trec
    plan = prepare_deep_review(paths)
    assert plan["source_counts"] == {"ham": 2, "spam": 3}
    assert sum(plan["ham_frame_by_length"].values()) == 1
    assert plan["ham_sample_rows"] == 1
    assert plan["spam_sample_rows"] == 1
    assert plan["bulk_rule_active"] is False
    with (paths.curated / "trec06c/ham_conditional_disposition.csv").open(
            encoding="utf-8", newline="") as handle:
        dispositions = list(csv.DictReader(handle))
    assert len(dispositions) == 2
    assert {row["proposed_status"] for row in dispositions} == {"conditional_candidate", "review"}
    with (paths.curated / "trec06c/spam_review_disposition.csv").open(
            encoding="utf-8", newline="") as handle:
        spam_dispositions = list(csv.DictReader(handle))
    assert len(spam_dispositions) == 3
    assert all(row["proposed_status"] == "review" for row in spam_dispositions)
    assert prepare_deep_review(paths)["ham_sample_id_sha256"] == plan["ham_sample_id_sha256"]
    before = summarize_deep_review(paths)
    assert before["strata"][0]["observed_discordance_rate"] is None
    assert before["proposed_bulk_eligible_now"] == 0
    _decision(paths, rows[0], "spam")  # synthetic human observation contradicts raw Ham
    manifest = paths.curated / "trec06c/decisions.csv"
    decision_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert prepare_deep_review(paths)["ham_sample_id_sha256"] == plan["ham_sample_id_sha256"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == decision_hash
    after = summarize_deep_review(paths)
    populated = [stratum for stratum in after["strata"] if stratum["human_reviewed"]]
    assert len(populated) == 1
    assert populated[0]["observed_discordance_rate"] == 1.0
    assert after["proposed_bulk_eligible_now"] == 0
    assert populated[0]["uncertainty_status"] == "complete_conclusive"


def test_deep_review_rejects_bulk_policy_activation(setup_trec) -> None:
    paths, _ = setup_trec
    policy = paths.config / "trec06c_curation_policy.json"
    text = policy.read_text(encoding="utf-8")
    policy.write_text(text.replace('"bulk_approval": false', '"bulk_approval": true'), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe or inconsistent"):
        prepare_deep_review(paths)


def test_spam_focus_is_review_routing_not_ground_truth() -> None:
    assert _focus("", "账户密码请立即验证") == "credential_or_account_review"
    assert _focus("", "商品优惠，欢迎购买") == "commercial_review"
    assert _focus("", "您好，明天会议") == "other_review"


class _OfflineTranslator:
    name = "test-local"
    is_external = False

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, subject: str, body: str) -> tuple[str, str]:
        self.calls += 1
        return "Tiêu đề " + subject, "Bản dịch " + body


class _MockExternalTranslator(_OfflineTranslator):
    name = "test-external"
    is_external = True


class _MockSuggestionProvider:
    name = "test-gemini"
    is_external = True

    def __init__(self) -> None:
        self.calls = 0

    def review(self, request, *, timeout_seconds):
        self.calls += 1
        assert request.external_enabled and request.explicit_email_sharing_consent
        assert "test@example.com" not in request.body
        return {"assessment": "review", "summary": "Evidence is insufficient.",
                "evidence": [], "suspicious_indicators": []}


def test_translation_cache_privacy_and_progress(setup_trec) -> None:
    paths, rows = setup_trec
    store = TrecReviewStore(paths)
    row = dict(rows[0], body=rows[0]["body"] + " test@example.com https://example.org/a")
    translator = _OfflineTranslator()
    first = store.translation(row, translator)
    second = store.translation(row, translator)
    assert first["cached"] is False and second["cached"] is True
    assert translator.calls == 1
    assert "test@example.com" in first["body_vi"]
    assert "https://example.org/a" in first["body_vi"]
    assert store.cached_translation(row, translator.name) == dict(second)
    assert store.progress()["translations_cached"] == 1
    with pytest.raises(PermissionError):
        store.translation(row, _MockExternalTranslator(), consent=False)
    assert store.progress()["translations_cached"] == 1


def test_protected_tokens_must_survive_translation() -> None:
    masked, tokens = _protect("请访问 https://example.org 或 a@example.org")
    assert "example.org" not in masked
    assert _restore(masked, tokens) == "请访问 https://example.org 或 a@example.org"
    with pytest.raises(ValueError, match="protected"):
        _restore("Đã bỏ token", tokens)


def test_standalone_domain_is_protected() -> None:
    original = "请访问 example.org 或 a@example.org"
    masked, tokens = _protect(original)
    assert "example.org" not in masked
    assert _restore(masked, tokens) == original


def test_gemini_translation_uses_mock_transport_and_validates_json(monkeypatch) -> None:
    import vietmailguard.v21.trec_review_assist as assist

    class Client:
        def close(self):
            pass

    calls = []
    monkeypatch.setattr(assist, "project_environment", lambda: {
        "GEMINI_API_KEY": "test-only-key", "GEMINI_MODEL": "gemini-2.5-flash",
    })
    monkeypatch.setattr(assist, "gemini_model", lambda variables=None: "gemini-2.5-flash")
    monkeypatch.setattr(assist, "_client", lambda key, timeout: Client())

    def fake_generate(client, *, model, contents, config):
        calls.append((model, json.loads(contents), config["tools"]))
        return json.dumps({"subject_vi": "Tiêu đề", "body_vi": "Nội dung"})

    monkeypatch.setattr(assist, "_safe_generate", fake_generate)
    with pytest.raises(PermissionError, match="consent"):
        GeminiTranslator().translate("邮件通知", "会议")
    assert calls == []
    assert GeminiTranslator().translate("邮件通知", "会议", consent=True) == ("Tiêu đề", "Nội dung")
    assert calls[0][2] == []
    assert calls[0][1]["untrusted_email_data"]["body_chunk"] == "会议"
    assert calls[0][1]["untrusted_email_data"]["chunk_count"] == 1
    monkeypatch.setattr(assist, "_safe_generate", lambda *args, **kwargs: '{"wrong": 1}')
    with pytest.raises(ValueError, match="invalid fields"):
        GeminiTranslator().translate("邮件通知", "会议", consent=True)
    monkeypatch.setattr(assist, "_safe_generate", lambda *args, **kwargs: 'not JSON')
    with pytest.raises(ValueError, match="not valid JSON"):
        GeminiTranslator().translate("邮件通知", "会议", consent=True)


def test_gemini_long_email_chunking_and_request_cap(monkeypatch) -> None:
    import vietmailguard.v21.trec_review_assist as assist

    body = "中文句子。" * 1600
    chunks = _translation_chunks(body)
    assert "".join(chunks) == body
    assert len(chunks) > 1
    assert all(len(chunk) <= TRANSLATION_CHUNK_CHARS for chunk in chunks)
    calls: list[dict] = []

    class Client:
        def close(self):
            pass

    monkeypatch.setattr(assist, "project_environment", lambda: {
        "GEMINI_API_KEY": "test-only-key", "GEMINI_MODEL": "gemini-2.5-flash",
    })
    monkeypatch.setattr(assist, "gemini_model", lambda variables=None: "gemini-2.5-flash")
    monkeypatch.setattr(assist, "_client", lambda key, timeout: Client())

    def fake_generate(client, *, model, contents, config):
        payload = json.loads(contents)["untrusted_email_data"]
        calls.append(payload)
        assert config["tools"] == []
        return json.dumps({"subject_vi": "Tiêu đề" if payload["chunk_index"] == 1 else "",
                           "body_vi": f"Đoạn {payload['chunk_index']}"})

    monkeypatch.setattr(assist, "_safe_generate", fake_generate)
    translated_subject, translated_body = GeminiTranslator().translate("主题", body, consent=True)
    assert translated_subject == "Tiêu đề"
    assert translated_body.splitlines() == [f"Đoạn {i}" for i in range(1, len(chunks) + 1)]
    assert len(calls) == len(chunks) == GeminiTranslator.estimate_requests("主题", body)
    assert all(call["chunk_count"] == len(chunks) for call in calls)

    calls.clear()
    with pytest.raises(ValueError, match="request budget"):
        GeminiTranslator().translate("主题", "中" * (TRANSLATION_CHUNK_CHARS * MAX_TRANSLATION_REQUESTS + 1), consent=True)
    assert calls == []


def test_gemini_translation_failure_is_not_cached(setup_trec, monkeypatch) -> None:
    import vietmailguard.v21.trec_review_assist as assist

    paths, rows = setup_trec
    row = dict(rows[0], body="中文句子。" * 1000)
    store = TrecReviewStore(paths)
    calls = 0

    class Client:
        def close(self):
            pass

    monkeypatch.setattr(assist, "project_environment", lambda: {
        "GEMINI_API_KEY": "test-only-key", "GEMINI_MODEL": "gemini-2.5-flash",
    })
    monkeypatch.setattr(assist, "gemini_model", lambda variables=None: "gemini-2.5-flash")
    monkeypatch.setattr(assist, "_client", lambda key, timeout: Client())

    def failing_second_chunk(client, *, model, contents, config):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("synthetic timeout")
        return json.dumps({"subject_vi": "Tiêu đề", "body_vi": "Nội dung"})

    monkeypatch.setattr(assist, "_safe_generate", failing_second_chunk)
    with pytest.raises(TimeoutError, match="synthetic timeout"):
        store.translation(row, GeminiTranslator(), consent=True)
    assert calls == 2
    assert store.progress()["translations_cached"] == 0
    with pytest.raises(PermissionError):
        store.translation(row, GeminiTranslator(), consent=False)
    assert calls == 2


def test_gemini_cache_preserves_protected_values_and_model_version(setup_trec, monkeypatch) -> None:
    import vietmailguard.v21.trec_review_assist as assist

    paths, rows = setup_trec
    store = TrecReviewStore(paths)
    row = dict(rows[0], body="请访问 https://example.org/a 与 mail.example.net, 联系 a@example.org")
    model = ["gemini-2.5-flash"]
    calls = []

    class Client:
        def close(self):
            pass

    monkeypatch.setattr(assist, "project_environment", lambda: {
        "GEMINI_API_KEY": "test-only-key", "GEMINI_MODEL": model[0],
    })
    monkeypatch.setattr(assist, "gemini_model", lambda variables=None: model[0])
    monkeypatch.setattr(assist, "_client", lambda key, timeout: Client())

    def echo_protected(client, *, model, contents, config):
        payload = json.loads(contents)["untrusted_email_data"]
        calls.append(payload)
        return json.dumps({"subject_vi": "Tiêu đề", "body_vi": payload["body_chunk"]})

    monkeypatch.setattr(assist, "_safe_generate", echo_protected)
    translator = GeminiTranslator()
    with pytest.raises(PermissionError):
        store.translation(row, translator, consent=False)
    assert calls == []
    first = store.translation(row, translator, consent=True)
    second = store.translation(row, translator, consent=True)
    assert first["cached"] is False and second["cached"] is True
    assert len(calls) == 1
    assert "https://example.org/a" in first["body_vi"]
    assert "mail.example.net" in first["body_vi"]
    assert "a@example.org" in first["body_vi"]
    assert "example.org" not in calls[0]["body_chunk"]
    assert store.progress()["eligible"] == 0
    model[0] = "gemini-2.5-flash-lite"
    third = store.translation(row, GeminiTranslator(), consent=True)
    assert third["cached"] is False and len(calls) == 2


def test_trec_gemini_suggestion_validates_chinese_quote_and_stays_weak(
        setup_trec, monkeypatch) -> None:
    import vietmailguard.v21.trec_review_assist as assist

    paths, rows = setup_trec
    row = dict(rows[0], body="请核对您的账户。")
    store = TrecReviewStore(paths)
    calls = []

    class Client:
        def close(self):
            pass

    monkeypatch.setattr(assist, "gemini_model", lambda variables=None: "gemini-2.5-flash")
    monkeypatch.setattr(assist, "_api_key", lambda: "test-only-key")
    monkeypatch.setattr(assist, "_client", lambda key, timeout: Client())

    def fake_generate(client, *, model, contents, config):
        calls.append(json.loads(contents))
        assert config["tools"] == []
        return json.dumps({
            "assessment": "review", "summary": "Chưa đủ bằng chứng.",
            "evidence": [{"field": "body", "quote": "请核对您的账户",
                          "explanation": "Yêu cầu kiểm tra tài khoản."}],
            "suspicious_indicators": ["Yêu cầu kiểm tra tài khoản"],
        }, ensure_ascii=False)

    monkeypatch.setattr(assist, "_safe_generate", fake_generate)
    provider = TrecGeminiSuggestionProvider()
    with pytest.raises(PermissionError):
        provider.review(ReviewRequest(sender="", subject="", body=row["body"]),
                        timeout_seconds=15)
    assert calls == []
    result = store.suggestion(row, consent=True, provider=provider)
    assert result["ai_suggested_label"] == "review"
    assert result["label_status"] == "weak_label"
    assert result["evidence"][0]["quote"] == "请核对您的账户"
    assert len(calls) == 1
    assert store.progress()["eligible"] == 0


def test_ai_suggestion_is_weak_and_separate_from_human(setup_trec) -> None:
    paths, rows = setup_trec
    store = TrecReviewStore(paths)
    provider = _MockSuggestionProvider()
    with pytest.raises(PermissionError):
        store.suggestion(rows[0], consent=False, provider=provider)
    assert provider.calls == 0
    first = store.suggestion(rows[0], consent=True, provider=provider)
    second = store.suggestion(rows[0], consent=True, provider=provider)
    assert provider.calls == 1
    assert first["ai_suggested_label"] == second["ai_suggested_label"] == "review"
    assert first["label_status"] == "weak_label"
    ai_path, human_path = store.export_manifests()
    assert len(list(csv.DictReader(ai_path.open(encoding="utf-8", newline="")))) == 1
    assert list(csv.DictReader(human_path.open(encoding="utf-8", newline=""))) == []
    assert store.progress()["eligible"] == 0


def test_translated_only_cannot_write_verified_decision(setup_trec) -> None:
    paths, rows = setup_trec
    store = TrecReviewStore(paths)
    build_review_queue(paths)
    with (store.directory / "review_queue.csv").open(encoding="utf-8", newline="") as handle:
        queued = list(csv.DictReader(handle))
    target = next(item for item in queued if item["id"] == rows[0]["id"])
    with pytest.raises(PermissionError, match="Translated-only"):
        store.record_qualified_decision(
            target, decision="normal", reviewer="vi-reader", evidence="translation only",
            qualified_chinese_reader=False, original_chinese_read=False,
        )
    assert not _read_decisions(store.directory / "decisions.csv")
    result = store.record_qualified_decision(
        target, decision="normal", reviewer="qualified-zh-reader",
        evidence="Original Chinese text describes an ordinary project meeting",
        qualified_chinese_reader=True, original_chinese_read=True,
    )
    assert result["label_status"] == "eligible"
    assert store.progress()["human_verified"] == 1
    assert store.progress()["eligible"] == 1
    assert store.qualified_label(rows[0]["id"], _read_decisions(store.directory / "decisions.csv")[rows[0]["id"]]) == "normal"
    _, human_path = store.export_manifests()
    with human_path.open(encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 1
    assert exported[0]["qualification_status"] == "self_attested_chinese_reader"
    assert exported[0]["original_chinese_checked"] == "true"


def test_review_progress_and_queue_navigation(setup_trec) -> None:
    paths, rows = setup_trec
    store = TrecReviewStore(paths)
    build_review_queue(paths)
    with (store.directory / "review_queue.csv").open(encoding="utf-8", newline="") as handle:
        queue = list(csv.DictReader(handle))
    assert neighbor_id(queue, queue[0]["id"], -1) == queue[0]["id"]
    assert neighbor_id(queue, queue[0]["id"], 1) == queue[1]["id"]
    store.update_state(queue[0]["id"], viewed=True, needs_verification=True,
                       expert=True, note="Cần người đọc tiếng Trung")
    reopened = TrecReviewStore(paths)
    assert reopened.state(queue[0]["id"])["chuyen_chuyen_gia"] == 1
    assert reopened.progress()["expert_queue"] == 1
    assert {row["id"] for row in load_expert_queue(store.directory, reopened.expert_ids())} == {queue[0]["id"]}
    assert len(visible_queue(queue, {}, True)) == len(queue)
    assert len(visible_queue(queue, {queue[0]["id"]: {"decision": "review"}}, True)) == len(queue) - 1


@pytest.mark.trec06c_real
def test_real_trec_curation_keeps_v2_frozen() -> None:
    if os.environ.get("VIETMAILGUARD_RUN_TREC06C_CURATION_INTEGRATION") != "1":
        pytest.skip("Opt in to the local TREC curation integration test")
    paths = resolve_v21_paths()
    if not (paths.processed / "trec06c_mime_v2/part-00001.csv").is_file():
        pytest.skip("TREC authoritative processed parts are not available")
    protected = [
        paths.project_root / "models/v2_bilingual/production_pipeline.joblib",
        paths.project_root / "config/v2_production_frozen.json",
        paths.project_root / "results/v2_bilingual/final_test_metrics.json",
    ]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in protected]
    summary = build_review_queue(paths)
    deep = prepare_deep_review(paths)
    assert sum(deep["ham_disposition"].values()) == summary["raw_labels"]["ham"]
    assert sum(deep["spam_disposition"].values()) == summary["raw_labels"]["spam"]
    assert deep["bulk_rule_active"] is False
    loader = json.loads((paths.reports / "trec06c_loader_report.json").read_text(encoding="utf-8"))
    assert summary["raw_labels"] == loader["raw_label_counts"]
    assert sum(summary["raw_labels"].values()) == loader["processed"]
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in protected] == before


@pytest.mark.trec06c_real
def test_real_trec_review_streamlit_starts_without_api_call() -> None:
    if os.environ.get("VIETMAILGUARD_RUN_TREC06C_CURATION_INTEGRATION") != "1":
        pytest.skip("Opt in to the local TREC review UI integration test")
    paths = resolve_v21_paths()
    if not (paths.curated / "trec06c/ham_probability_sample.csv").is_file():
        pytest.skip("Real TREC review queue is unavailable")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(paths.project_root / "scripts/v21/trec06c_review_app.py"), default_timeout=30
    ).run()
    assert len(app.exception) == 0
    assert len(app.title) == 1
    assert len(app.selectbox) >= 2
    assert "Gửi và dịch email này" not in {button.label for button in app.button}
    assert any("Gemini" in toggle.label for toggle in app.toggle)
