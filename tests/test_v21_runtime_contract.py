"""V2.1 preprocessing and artifact boundaries; no final evaluation or network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vietmailguard.file_utils import sha256_file
from vietmailguard.v21 import runtime_inference as runtime
from vietmailguard.v21.development_training import make_tfidf_pipeline


@pytest.fixture
def training_config() -> dict:
    return json.loads((runtime.PROJECT_ROOT / "config/v21/development_training.json").read_text(
        encoding="utf-8"))


@pytest.fixture
def synthetic_model(training_config: dict):
    # Fitting a tiny synthetic test double is not a research/model-training run.
    model = make_tfidf_pipeline(training_config)
    frame = pd.DataFrame([
        {"subject": "Team meeting", "body": "meeting agenda tomorrow project planning"},
        {"subject": "Team schedule", "body": "meeting agenda tomorrow project planning"},
        {"subject": "Discount offer", "body": "discount offer buy sale products today"},
        {"subject": "Discount sale", "body": "discount offer buy sale products today"},
        {"subject": "Password verify", "body": "verify password locked account login now"},
        {"subject": "Account verify", "body": "verify password locked account login now"},
    ])
    return model.fit(frame, ["normal", "normal", "spam", "spam", "phishing", "phishing"])


def test_pipeline_matches_verified_training_configuration(synthetic_model, training_config: dict) -> None:
    runtime._validate_pipeline(synthetic_model, training_config)
    synthetic_model.named_steps["tfidf"].transformer_list[0][1].set_params(ngram_range=(1, 1))
    with pytest.raises(runtime.V21ModelUnavailable, match="vectorizer differs"):
        runtime._validate_pipeline(synthetic_model, training_config)


def test_runtime_rejects_changed_composer_columns(synthetic_model) -> None:
    synthetic_model.named_steps["compose_text"].subject_column = "alternate"
    with pytest.raises(runtime.V21ModelUnavailable, match="input columns"):
        runtime._validate_pipeline(synthetic_model)


def test_model_and_explanation_share_training_bound_but_security_reads_full_text(synthetic_model) -> None:
    engine = runtime.V21DevelopmentInference(synthetic_model, "2.1-test", maximum_characters=50)
    subject = "Team meeting"
    prefix = "meeting agenda tomorrow project planning " * 3
    body = prefix + " verify password now at http://192.0.2.12/login"
    seen: list[pd.DataFrame] = []
    predict = synthetic_model.predict

    def capture(frame: pd.DataFrame):
        seen.append(frame.copy())
        return predict(frame)

    synthetic_model.predict = capture
    result = engine.analyze_email(subject=subject, body=body)
    assert seen[0].iloc[0]["body"] == body[:50]
    assert result["urls"] == ["http://192.0.2.12/login"]
    assert result["preprocessing"]["truncated_fields"] == ["body"]
    assert result["preprocessing"]["security_analysis_scope"] == "full_email_text"
    expected = engine.explainer.explain_structured(
        pd.DataFrame([{"subject": subject, "body": body[:50]}]), result["prediction"],
        max_features=engine.config["explainability"]["max_model_features"],
        include_character_features=engine.config["explainability"]["include_character_features"],
        minimum_positive_contribution=engine.config["explainability"]["minimum_positive_contribution"])
    assert result["model_explanation"]["features"] == expected["features"]
    assert result["confidence"] is None
    assert result["class_probabilities"] == {}


def test_parsed_eml_chinese_detection_does_not_treat_url_as_english(synthetic_model) -> None:
    engine = runtime.V21DevelopmentInference(synthetic_model, "2.1-test")
    result = engine.analyze_email(parsed_eml={
        "subject": "账户通知", "body": "请查看账户通知。https://example.com/login",
        "sender": "sender@example.com"})
    assert result["detected_language"] == "zh"
    assert result["language_support_status"]["status"] == "unvalidated"
    assert result["model_scope"] == "development_candidate_only"


def test_chinese_english_mixed_remains_unvalidated(synthetic_model) -> None:
    engine = runtime.V21DevelopmentInference(synthetic_model, "2.1-test")
    result = engine.analyze_email(subject="账户通知", body="Your account 请查看通知。")
    assert result["detected_language"] == "mixed"
    assert result["language_support_status"]["status"] == "unvalidated"


def test_runtime_bound_is_safe_for_missing_subject_and_unicode() -> None:
    original = pd.DataFrame([{"subject": None, "body": "Tiếng Việt 中文 😀"}])
    bounded = runtime._bounded_model_input(original, 12)
    assert bounded.iloc[0]["subject"] == ""
    assert bounded.iloc[0]["body"] == "Tiếng Việt 中文 😀"[:12]
    assert original.iloc[0]["subject"] is None


def test_training_manifest_digest_and_preprocessing_must_agree(tmp_path: Path) -> None:
    candidate = tmp_path / "models/v21_multilingual/candidates/candidate.joblib"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"synthetic test artifact, never deserialized")
    metrics = tmp_path / "results/v21_multilingual/metrics.json"
    metrics.parent.mkdir(parents=True)
    training = tmp_path / "config/v21/training.json"
    training.parent.mkdir(parents=True)
    training.write_text(json.dumps({"scope": "development_bootstrap_validation_only",
                                   "classes": ["normal", "spam", "phishing"],
                                   "max_input_characters": 4096}), encoding="utf-8")
    metrics.write_text(json.dumps({
        "scope": "development_validation_only", "config_sha256": sha256_file(training),
        "development_leakage_preflight": {"passed": True},
        "candidates": [{"candidate": "candidate", "artifact": candidate.relative_to(tmp_path).as_posix()}],
    }), encoding="utf-8")
    config = tmp_path / "runtime.json"
    configuration = {
        "schema_version": 1, "scope": "development_candidate_only",
        "probability_status": "unavailable_uncalibrated",
        "supported_classes": ["normal", "spam", "phishing"],
        "candidate_id": "candidate", "model_version": "2.1-test",
        "artifact": candidate.relative_to(tmp_path).as_posix(), "artifact_sha256": sha256_file(candidate),
        "validation_metrics": metrics.relative_to(tmp_path).as_posix(),
        "validation_metrics_sha256": sha256_file(metrics),
        "training_config": training.relative_to(tmp_path).as_posix(),
        "training_config_sha256": sha256_file(training),
    }
    config.write_text(json.dumps(configuration), encoding="utf-8")
    verified, _, _ = runtime._verified_paths(config, tmp_path)
    assert verified["verified_training_config"]["max_input_characters"] == 4096
    training.write_text(training.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(runtime.V21ModelUnavailable):
        runtime._verified_paths(config, tmp_path)


def test_real_local_candidate_has_matching_sha_preprocessing_and_no_fake_probabilities() -> None:
    artifact = runtime.PROJECT_ROOT / "models/v21_multilingual/candidates/tfidf_word_character_linear_svm.joblib"
    if not artifact.is_file():
        pytest.skip("Local V2.1 development candidate is not distributed with source")
    # Once present, verification failure is a failure, not a silently skipped test.
    engine = runtime.load_v21_engine()
    result = engine.analyze_email(subject="Project meeting", body="Please meet at three tomorrow.")
    assert result["model_artifact"]["artifact_sha256"] == sha256_file(artifact)
    assert result["model_artifact"]["candidate_id"] == "tfidf_word_character_linear_svm"
    assert result["preprocessing"]["maximum_characters_per_field"] == 4096
    assert result["confidence"] is None
    assert result["class_probabilities"] == {}
    assert result["probability_source"] == "unavailable_uncalibrated"
    frame = pd.DataFrame([{"subject": "Project meeting", "body": "Please meet at three tomorrow."}])
    np.testing.assert_array_equal(engine.model.predict(frame), [result["prediction"]])


def test_engine_cache_invalidates_after_security_configuration_changes(
    synthetic_model, training_config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "candidate.joblib"
    configuration = {
        "artifact_sha256": "artifact-digest", "model_version": "2.1-test",
        "training_config_sha256": "training-digest",
        "verified_training_config": training_config,
    }
    monkeypatch.setattr(runtime, "_verified_paths", lambda *_: (configuration, artifact, tmp_path))
    security_digest = ["original-security-config"]
    monkeypatch.setattr(runtime, "sha256_file", lambda path: (
        "artifact-digest" if path == artifact else security_digest[0]))
    loads: list[Path] = []

    def load(path):
        loads.append(path)
        return synthetic_model

    monkeypatch.setattr(runtime.joblib, "load", load)
    runtime._load_verified.cache_clear()
    try:
        first = runtime.load_v21_engine()
        assert runtime.load_v21_engine() is first
        assert len(loads) == 1
        security_digest[0] = "changed-security-config"
        assert runtime.load_v21_engine() is not first
        assert len(loads) == 2
    finally:
        runtime._load_verified.cache_clear()


def test_missing_v21_never_falls_back_to_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    from vietmailguard.v21 import runtime_modes

    monkeypatch.setattr(runtime_modes, "model_availability", lambda: (False, "Missing V2.1 artifact"))
    for mode in (runtime_modes.V21_ML_MODE, runtime_modes.V21_HYBRID_MODE):
        with pytest.raises(runtime.V21ModelUnavailable, match="Missing V2.1 artifact"):
            runtime_modes.analyzer_for_mode(mode)
