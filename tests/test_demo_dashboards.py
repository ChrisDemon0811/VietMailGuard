"""Compact dashboards remain read-only, honest about missing metrics and offline."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Resolve the application package before adding Streamlit's script import root;
# otherwise app/app.py can shadow the app namespace during suite collection.
import app.i18n

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import demo_dashboard_ui as ui
from i18n import t

pytestmark = pytest.mark.usefixtures("no_external_network")


def write_result(root, **changes):
    data = {"scope": "development_validation_only", "candidates": [
        {"candidate": name, "status": "PASS", "metrics": {
            "macro_f1": score, "phishing_recall": recall, "normal_to_spam_or_phishing": errors}}
        for name, score, recall, errors in [("A_tfidf_svc", .91, .88, 2), ("C_minilm_mlp", .82, .80, 4)]
    ], **changes}
    path = root / ui.RESULT_PATHS["models"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_comparison_uses_saved_candidates_not_hardcoded_scores(tmp_path):
    write_result(tmp_path)
    rows = ui.comparison_rows(ui.saved_result(tmp_path, "models"), "vi")
    assert [row["Macro F1"] for row in rows] == [.91, .82]
    assert [row["Phishing Recall"] for row in rows] == [.88, .80]
    assert [row[t("demo_false_positives", "vi")] for row in rows] == [2, 4]
    write_result(tmp_path, candidates=[])
    assert all(row["Macro F1"] is None for row in ui.comparison_rows(ui.saved_result(tmp_path, "models"), "vi"))


@pytest.mark.parametrize("content", ['{', '[]', '{"scope":"final_test","candidates":[]}',
                                      '{"scope":"development_validation_only","candidates":[42]}',
                                      '{"scope":"development_validation_only","candidates":[],"x":NaN}'])
def test_invalid_aggregate_is_unavailable(tmp_path, content):
    path = write_result(tmp_path)
    path.write_text(content, encoding="utf-8")
    assert ui.saved_result(tmp_path, "models") == {}


def test_aggregate_loader_reads_only_allowlisted_results(tmp_path, monkeypatch):
    write_result(tmp_path)
    ui._read_result.clear()
    original = io.open
    reads = []

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).resolve().is_relative_to(tmp_path):
            relative = Path(file).resolve().relative_to(tmp_path).as_posix()
            assert relative in ui.RESULT_PATHS.values()
            reads.append(relative)
        return original(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded)
    assert ui.saved_result(tmp_path, "models")
    assert reads == [ui.RESULT_PATHS["models"]]
    assert ui.saved_result(tmp_path, "reduction") == {}


@pytest.mark.parametrize("value", [True, -1, 2, float("nan"), "0.9", None])
def test_invalid_score_never_becomes_a_chart_value(value):
    data = {"candidates": [{"candidate": "A_tfidf_svc", "status": "PASS", "metrics": {"macro_f1": value}}]}
    assert ui.comparison_rows(data, "vi")[0]["Macro F1"] is None


def test_overview_is_compact_and_limits_are_visible(streamlit_loopback):
    script = f"import sys\nsys.path.insert(0, {str(ROOT / 'app')!r})\nfrom demo_dashboard_ui import render_overview\nrender_overview('vi')"
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.title[0].value == t("demo_overview_title", "vi")
    assert [item.value for item in app.metric] == ["3", "3", "3"]
    assert any("Email → Preprocessing → Classifier → Security Analysis → Routing" in item.value for item in app.markdown)
    assert t("demo_limits", "vi") in [item.value for item in app.markdown]
    assert not app.selectbox


def test_summary_has_one_two_bar_chart_and_no_fake_missing_scores(tmp_path, streamlit_loopback):
    write_result(tmp_path)
    script = (f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(ROOT / 'app')!r})\n"
              f"from demo_dashboard_ui import render_model_summary\nrender_model_summary(Path({str(tmp_path)!r}), 'vi')")
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert len(app.dataframe) == 1 and len(app.get("plotly_chart")) == 1
    assert app.dataframe[0].value["Macro F1"].tolist() == [.91, .82]
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["data"][0]["type"] == "bar"
    assert spec["data"][0]["text"] == ["0.910000", "0.820000"]
    assert t("demo_conclusion", "vi") in [item.value for item in app.markdown]
    write_result(tmp_path, candidates=[])
    app.run()
    assert not app.exception and not app.get("plotly_chart")
    assert app.dataframe[0].value["Macro F1"].isna().all()
    assert any(t("demo_missing_comparison", "vi") == item.value for item in app.info)


def test_research_remains_collapsed_inside_model_page(tmp_path, monkeypatch, streamlit_loopback):
    write_result(tmp_path)
    original = ui.saved_result
    monkeypatch.setattr(ui, "saved_result", lambda _root, section: original(tmp_path, section))
    script = (f"import sys,runpy\nimport streamlit as st\nsys.path.insert(0, {str(ROOT / 'app')!r})\n"
              f"st.session_state['interface_language']='vi'\nrunpy.run_path({str(ROOT / 'app/views/model_dashboard.py')!r})")
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    details = next(item for item in app.expander if item.label == t("demo_research_details", "vi"))
    assert details.proto.expanded is False
    assert any(t("demo_reduction_title", "vi") == item.value for item in details.subheader)
    assert len(app.get("plotly_chart")) - len(details.get("plotly_chart")) == 1
    assert details.tabs  # Existing V2/research evidence is retained under disclosure.
