"""Saved-artifact dashboards: scope, error semantics and offline presentation."""

from __future__ import annotations

import io
import json
import socket
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vietmailguard.v21.dashboard_artifacts import (
    ARTIFACT_PATHS, artifact_signature, candidate_rows, error_directions,
    load_research_artifacts, paired_macro_delta,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dashboard must not access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def write_artifact(root: Path, name: str, document: dict) -> Path:
    path = root / ARTIFACT_PATHS[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_missing_sections_are_absent_not_zero_or_success(tmp_path):
    data = load_research_artifacts(tmp_path)
    assert data.documents == {}
    assert set(data.issues) == set(ARTIFACT_PATHS)
    assert set(data.issues.values()) == {"missing"}


@pytest.mark.parametrize("document", [[], None, {"scope": "final_test"},
                                    {"scope": "development_validation_only", "candidates": [42]}])
def test_invalid_artifact_isolated_without_losing_valid_sections(tmp_path, document):
    write_artifact(tmp_path, "development", document)
    write_artifact(tmp_path, "audit", {"input_rows": 13})
    data = load_research_artifacts(tmp_path)
    assert data.issues["development"] == "invalid"
    assert "development" not in data.documents
    assert data.documents["audit"]["input_rows"] == 13


@pytest.mark.parametrize("payload", ['{"scope": ', '{"scope":"development_validation_only","value":NaN}'])
def test_corrupt_or_non_finite_json_is_not_displayed(tmp_path, payload):
    path = write_artifact(tmp_path, "development", {})
    path.write_text(payload, encoding="utf-8")
    assert load_research_artifacts(tmp_path).issues["development"] == "invalid"


def test_cache_signature_changes_after_result_changes(tmp_path):
    before = artifact_signature(tmp_path)
    path = write_artifact(tmp_path, "audit", {"input_rows": 11})
    created = artifact_signature(tmp_path)
    assert before != created
    path.write_text('{"input_rows": 123456}', encoding="utf-8")
    assert created != artifact_signature(tmp_path)


def test_explicit_false_alarm_and_miss_directions_preserve_saved_json():
    metrics = {"normal_false_positives": 20, "non_normal_false_positives": 3}
    assert error_directions(metrics) == {"normal_to_other": 3, "other_to_normal": 20}
    assert metrics == {"normal_false_positives": 20, "non_normal_false_positives": 3}
    assert error_directions({}) == {"normal_to_other": None, "other_to_normal": None}


def test_embedding_latency_never_claims_classifier_timing_is_end_to_end():
    items = [{"candidate": "synthetic_embedding", "family": "embedding", "metrics": {
        "prediction_ms_per_email": 0.001, "end_to_end_smoke_ms_per_email": 18}},
        {"candidate": "synthetic_missing_encoder", "family": "embedding", "metrics": {
            "prediction_ms_per_email": 0.001}},
        {"candidate": "synthetic_tfidf", "metrics": {"prediction_ms_per_email": 2}}]
    rows = candidate_rows(items)
    assert [row["prediction_ms_per_email"] for row in rows] == [18, None, 2]


def test_binary_rows_have_source_label_errors_and_no_phishing_metric():
    rows = candidate_rows([{"candidate": "synthetic_binary", "metrics": {
        "macro_f1": .7, "false_positives_ham_as_spam": 2,
        "false_negatives_spam_as_ham": 4}}], binary=True)
    assert rows[0]["ham_to_spam"] == 2
    assert rows[0]["spam_to_ham"] == 4
    assert "phishing_recall" not in rows[0]


def test_matched_pair_delta_uses_saved_values_and_missing_stays_unavailable():
    assert paired_macro_delta({}) is None
    assert paired_macro_delta({"A_current_candidate": {"validation_metrics": {"macro_f1": .9}},
                               "B_synthetic_added": {"validation_metrics": {"macro_f1": .8}}}) == pytest.approx(-.1)


def test_only_exact_aggregate_allowlist_is_read(tmp_path, monkeypatch):
    write_artifact(tmp_path, "audit", {"input_rows": 5})
    original = io.open
    reads = []

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            resolved = Path(file).resolve()
            if resolved.is_relative_to(tmp_path):
                relative = resolved.relative_to(tmp_path).as_posix()
                assert relative in ARTIFACT_PATHS.values()
                reads.append(relative)
        return original(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded)
    assert load_research_artifacts(tmp_path).documents["audit"]["input_rows"] == 5
    assert reads == [ARTIFACT_PATHS["audit"]]


def test_curation_status_counts_have_strict_aggregate_columns(tmp_path):
    path = tmp_path / ARTIFACT_PATHS["curation_counts"]
    path.parent.mkdir(parents=True)
    path.write_text("dataset,label_status,rows\nsynthetic,weak_label,12\n", encoding="utf-8")
    data = load_research_artifacts(tmp_path)
    assert data.documents["curation_counts"]["rows"][0]["rows"] == 12
    path.write_text("dataset,body,rows\nsynthetic,private,12\n", encoding="utf-8")
    assert load_research_artifacts(tmp_path).issues["curation_counts"] == "invalid"
    path.write_text("dataset,label_status,rows\nsynthetic,eligible\n", encoding="utf-8")
    assert load_research_artifacts(tmp_path).issues["curation_counts"] == "invalid"


@pytest.mark.parametrize("language", ["en", "vi"])
@pytest.mark.parametrize("page", ["dataset_dashboard", "model_dashboard"])
def test_actual_pages_render_offline_without_email_rows_or_model_loads(monkeypatch, language, page):
    original = io.open

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            path = Path(file).resolve()
            if path.is_relative_to(ROOT / "data"):
                assert path.name == "manifest.json", "Dashboard must use aggregate manifests only"
            assert path.suffix.casefold() not in {".joblib", ".db", ".eml"}
            assert path.name != ".env"
        return original(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded)
    script = (
        "import sys\nimport runpy\nimport streamlit as st\n"
        f"sys.path.insert(0, {str(ROOT / 'app')!r})\n"
        f"st.session_state['interface_language'] = {language!r}\n"
        f"runpy.run_path({str(ROOT / 'app' / 'views' / (page + '.py'))!r})\n"
    )
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    from i18n import t

    captions = "\n".join(str(item.value) for item in app.caption)
    assert t("rd_saved_only", language) in captions
    assert t("rd_binary_notice", language) in "\n".join(
        str(item.value) for group in (app.info, app.warning) for item in group)
    tables = [item.value for item in app.dataframe]
    assert tables
    assert not any(str(column).startswith("rd_") for frame in tables for column in frame.columns)
    assert not any(str(value).startswith(("language_", "email_language_"))
                   for frame in tables for value in frame.astype(str).to_numpy().flat)
    if page == "model_dashboard":
        label = t("rd_normal_to_other", language)
        if (ROOT / ARTIFACT_PATHS["development"]).exists():
            row = next(frame for frame in tables if label in frame and t("rd_candidate", language) in frame)
            actual = next(value for _, value in row.iterrows()
                          if value[t("rd_candidate", language)] == "tfidf_word_character_linear_svm")
            assert actual[label] == 3
            assert actual[t("rd_other_to_normal", language)] == 20


@pytest.mark.parametrize("renderer", ["render_dataset_research", "render_model_research"])
def test_source_only_missing_artifacts_still_render(tmp_path, renderer):
    script = (
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'app')!r})\n"
        f"from research_dashboard_ui import {renderer}\n"
        f"{renderer}(Path({str(tmp_path)!r}), 'vi')\n"
    )
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.info
    assert len(app.metric) == 0
