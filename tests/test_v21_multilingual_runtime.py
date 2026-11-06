"""Functional synthetic checks, never a multilingual accuracy benchmark."""

import copy
import json
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pytest

from vietmailguard.mail_service import MailService
from vietmailguard.v21 import multilingual_runtime as runtime
from vietmailguard.v21.llm_reviewer import MockReviewerProvider, ReviewerPrivacyError
from vietmailguard.v21.runtime_modes import (
    V21_HYBRID_MODE, V21_MULTILINGUAL_MODE, analyzer_for_mode,
)

pytestmark = pytest.mark.usefixtures("no_external_network")


def test_no_activation_with_missing_config(tmp_path):
    available, code = runtime.multilingual_availability(tmp_path / "missing.json", tmp_path)
    assert not available and code == "multilingual_missing_artifact"


def test_tampered_local_file_rejected_before_deserialization(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"synthetic fixture")
    with pytest.raises(runtime.V21ModelUnavailable, match="hash_mismatch"):
        runtime._verified_file(path, "0" * 64)


def test_mode_selection_never_falls_back_or_combines(monkeypatch):
    monkeypatch.setattr(runtime, "multilingual_availability", lambda: (False, "multilingual_missing_artifact"))
    with pytest.raises(runtime.V21ModelUnavailable):
        analyzer_for_mode(V21_MULTILINGUAL_MODE)
    monkeypatch.setattr(runtime, "multilingual_availability", lambda: (True, "multilingual_verified"))
    assert analyzer_for_mode(V21_MULTILINGUAL_MODE) is runtime.analyze_multilingual_email
    assert analyzer_for_mode(V21_HYBRID_MODE, hybrid_base=V21_MULTILINGUAL_MODE) is runtime.analyze_multilingual_email
    with pytest.raises(ValueError):
        analyzer_for_mode(V21_HYBRID_MODE, hybrid_base="v2")


@pytest.fixture(scope="module")
def real_engine():
    # Clean source clones do not contain experiment binaries/optional encoder.
    if not (runtime.PROJECT_ROOT / "models/v21_multilingual/experiments/multilingual_dl_01/C_minilm_mlp.joblib").is_file():
        pytest.skip("Local Experiment 01 MLP not supplied")
    available, reason = runtime.multilingual_availability()
    assert available, reason  # present-but-invalid artifacts must FAIL, not skip
    return runtime.load_multilingual_engine()


SAMPLES = [
    ("en", "Team meeting", "The meeting is at three tomorrow. Bring the project agenda."),
    ("vi", "Lịch họp nhóm", "Cuộc họp diễn ra vào chiều mai. Vui lòng mang tài liệu dự án."),
    ("zh", "会议通知", "明天下午三点开会，请带上会议资料。"),
]


@pytest.mark.production_artifact
@pytest.mark.parametrize("language,subject,body", SAMPLES)
def test_real_encoder_mlp_languages_and_original_security(real_engine, language, subject, body):
    result = real_engine.analyze_email(sender="fixture@example.com", subject=subject, body=body)
    assert result["prediction"] in runtime.CLASSES
    assert result["confidence"] is None and result["class_probabilities"] == {}
    assert result["probability_source"] == "unavailable_uncalibrated"
    assert result["model_scope"] == "multilingual_experimental"
    assert result["model_artifact"]["candidate_id"] == "C_minilm_mlp"
    assert result["detected_language"] == language
    assert result["language_support_status"]["status"] == ("unvalidated" if language == "zh" else "experimental")
    assert result["model_explanation"]["features"] == []
    assert not result["model_explanation_supported"]


@pytest.mark.production_artifact
def test_real_preprocessing_matches_experiment_and_security_sees_original(real_engine):
    import pandas as pd
    from vietmailguard.embedding_modeling import compose_email_texts
    from vietmailguard.v21.runtime_inference import _bounded_model_input

    body = "项目会议。" * 1100 + " http://192.0.2.25/login"
    frame = pd.DataFrame([{"subject": "会议", "body": body}])
    values = real_engine.encoder.encode(compose_email_texts(_bounded_model_input(frame,4096)).tolist(),
        batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, device="cpu")
    expected = str(real_engine.model.predict(values)[0])
    result = real_engine.analyze_email(subject="会议", body=body)
    assert result["prediction"] == expected
    assert result["urls"] == ["http://192.0.2.25/login"]
    assert result["preprocessing"]["truncated_fields"] == ["body"]
    assert result["preprocessing"]["translated"] is False
    with pytest.raises(ValueError):
        real_engine.analyze_email(body=" ")


@pytest.mark.production_artifact
def test_real_eml_sqlite_restart_model_switch_and_reviewer(real_engine, tmp_path):
    calls = []
    def counted(**fields):
        calls.append(fields)
        return real_engine.analyze_email(**fields)
    db = tmp_path / "多语言 email.db"
    snapshots = []
    with MailService.from_database_path(db, analyzer=counted) as service:
        for language, subject, body in SAMPLES:
            message = EmailMessage()
            message["From"] = "fixture@example.com"
            message["To"] = "reader@example.org"
            message["Subject"] = subject
            message.set_content(body)
            data = message.as_bytes()
            received = service.import_eml_bytes(data)
            email_id = received["email"]["id"]
            duplicate = service.import_eml_bytes(data)
            assert duplicate["email"]["id"] == email_id
            opened = service.open_email(email_id)
            assert opened["email"]["subject"] == subject
            assert opened["email"]["body"].strip() == body
            assert opened["analysis"]["result"]["model_artifact"]["candidate_id"] == "C_minilm_mlp"
            original = copy.deepcopy(opened["analysis"])
            prediction = original["prediction"]
            assert opened["email"]["folder"] == {
                "normal":"hop_thu_den", "spam":"thu_rac", "phishing":"cach_ly"}[prediction]
            for folder in ("thu_rac", "cach_ly"):
                service.move_email(email_id, folder)
                current = service.open_email(email_id)
                assert current["email"]["folder"] == folder and current["analysis"] == original
            reviewed = service.review_stored_email(email_id, provider=MockReviewerProvider({
                "assessment": "review", "summary":"Synthetic offline review.",
                "evidence":[], "suspicious_indicators":[]}))
            after = service.open_email(email_id)
            assert after["analysis"] == original and after["email"]["folder"] == "cach_ly"
            class External(MockReviewerProvider):
                is_external = True
            with pytest.raises(ReviewerPrivacyError):
                service.review_stored_email(email_id, provider=External(), allow_external=True,
                                             explicit_email_sharing_consent=False)
            snapshots.append((email_id, original))
    assert len(calls) == 3
    def never(**fields):
        raise AssertionError("Opening persisted mail must not infer with a different model")
    with MailService.from_database_path(db, analyzer=never) as service:
        for email_id, original in snapshots:
            opened = service.open_email(email_id)
            assert opened["analysis"] == original
            assert opened["email"]["folder"] == "cach_ly"
            assert opened["llm_review"] is not None


@pytest.mark.production_artifact
def test_encoder_contract_missing_dependency_and_hash_changes(real_engine, tmp_path, monkeypatch):
    config = json.loads(runtime.DEFAULT_CONFIG.read_text())
    changed = copy.deepcopy(config)
    changed["encoder_files"]["config.json"] = "0" * 64
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(changed))
    assert runtime.multilingual_availability(path)[1] == "multilingual_hash_mismatch"
    spec = runtime.importlib.util.find_spec
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda p: None if p == "sentence_transformers" else spec(p))
    assert runtime.multilingual_availability()[1] == "multilingual_missing_dependency"


@pytest.mark.production_artifact
def test_head_rejects_label_reordering_and_encoder_revision(real_engine):
    config = json.loads(runtime.DEFAULT_CONFIG.read_text())
    artifact = runtime._head(str(runtime.PROJECT_ROOT / config["artifact"]),
                             config["artifact_sha256"], json.dumps(config, sort_keys=True))
    altered = copy.copy(artifact)
    altered.label_order = ["normal", "spam", "phishing"]
    with pytest.raises(runtime.V21ModelUnavailable, match="invalid_classes"):
        runtime.validate_head(altered, config)
    altered = copy.copy(artifact)
    altered.model_revision = "unverified"
    with pytest.raises(runtime.V21ModelUnavailable, match="invalid_head"):
        runtime.validate_head(altered, config)


@pytest.mark.production_artifact
def test_invalid_embedding_fails_without_fake_result(real_engine, monkeypatch):
    monkeypatch.setattr(real_engine.encoder, "encode", lambda *a,**kw: np.full((1,384), np.nan))
    with pytest.raises(runtime.V21ModelUnavailable, match="invalid_embedding"):
        real_engine.analyze_email(body="Synthetic message.")


@pytest.mark.production_artifact
def test_streamlit_real_multilingual_analyze_language_draft_hybrid_and_save(
    real_engine, tmp_path, monkeypatch, streamlit_loopback,
):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("VIETMAILGUARD_MAIL_DB", str(tmp_path / "ui.db"))
    app = AppTest.from_file(str(runtime.PROJECT_ROOT / "app/app.py"), default_timeout=120).run()
    def select(key):
        return next(item for item in app.selectbox if item.key == key)
    def button(key):
        return next(item for item in app.button if item.key == key)
    assert app.session_state["analysis_mode"] == "v2"
    select("analysis_mode_selector").set_value("v21_multilingual").run()
    app.switch_page("views/email_analyzer.py").run()
    for language, subject, body in SAMPLES:
        next(item for item in app.text_input if item.key == "input_quick_subject").set_value(subject)
        next(item for item in app.text_area if item.key == "input_quick_body").set_value(body)
        button("quick_analyze_button").click().run()
        assert not app.exception
        payload = app.session_state["vietmailguard_analysis_result"]
        assert payload["result"]["detected_language"] == language
        assert payload["result"]["model_scope"] == "multilingual_experimental"
    original = copy.deepcopy(app.session_state["vietmailguard_analysis_result"]["result"])
    select("language_selector").set_value("vi").run()
    assert app.session_state["vietmailguard_analysis_result"]["result"] == original
    assert next(item for item in app.text_area if item.key == "input_quick_body").value == SAMPLES[-1][2]
    button("quick_save_to_mailbox").click().run()
    assert not app.exception
    email_id = app.session_state["vietmailguard_analysis_result"]["saved_email_id"]
    select("analysis_mode_selector").set_value("v21_hybrid").run()
    select("hybrid_classifier_selector").set_value("v21_multilingual").run()
    button("quick_analyze_button").click().run()
    assert not app.exception
    before = copy.deepcopy(app.session_state["vietmailguard_analysis_result"]["result"])
    button("quick_llm_review_button").click().run()
    assert not app.exception
    assert app.session_state["vietmailguard_analysis_result"]["result"] == before
    assert app.session_state["vietmailguard_llm_review"]["reviewer"]["provider"] == "mock"
    app.switch_page("views/mail_client.py").run()
    assert select("hybrid_classifier_selector").value == "v21_multilingual"
    app.switch_page("views/email_analyzer.py").run()
    assert select("hybrid_classifier_selector").value == "v21_multilingual"
    select("hybrid_classifier_selector").set_value("v21_ml").run()
    assert not any(item.key == "quick_save_to_mailbox" for item in app.button)
    with MailService.from_database_path(tmp_path / "ui.db") as service:
        assert service.open_email(email_id)["analysis"]["result"] == original
