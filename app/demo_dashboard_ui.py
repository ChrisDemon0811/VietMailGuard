"""Compact demo dashboards using saved aggregate results, never inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from i18n import t


RESULT_PATHS = {
    "models": "results/v21_multilingual/experiments/multilingual_dl_01/validation_metrics.json",
    "reduction": "results/v21_multilingual/experiments/dimensionality_reduction_02/validation_metrics.json",
}


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite aggregate")


@st.cache_data(show_spinner=False)
def _read_result(path_text: str, signature: tuple[int, int]) -> dict[str, Any]:
    del signature  # Cache key changes when saved results change.
    try:
        path = Path(path_text)
        if path.stat().st_size > 2_000_000:
            return {}
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        if (not isinstance(data, dict) or data.get("scope") != "development_validation_only"
                or not isinstance(data.get("candidates"), list)
                or not all(isinstance(item, dict) for item in data["candidates"])):
            return {}
        return data
    except (OSError, UnicodeError, ValueError, RecursionError):
        return {}


def saved_result(root: Path, section: str) -> dict[str, Any]:
    """Read only the two allowlisted aggregate files; missing means unavailable."""
    path = root / RESULT_PATHS[section]
    try:
        stat = path.stat()
    except OSError:
        return {}
    return _read_result(str(path), (stat.st_mtime_ns, stat.st_size))


def _number(value: Any, *, probability: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or (probability and value > 1):
        return None
    return float(value)


def comparison_rows(document: dict[str, Any], language: str) -> list[dict[str, Any]]:
    """Select the actual traditional and MLP candidates from the same experiment."""
    rows = []
    for candidate_id, name, architecture in (
        ("A_tfidf_svc", "Traditional ML", "TF-IDF + LinearSVC"),
        ("C_minilm_mlp", "Multilingual Experimental", "MiniLM + MLP"),
    ):
        candidates = [item for item in document.get("candidates", [])
                      if item.get("candidate") == candidate_id and item.get("status") == "PASS"]
        metrics = candidates[0].get("metrics", {}) if len(candidates) == 1 else {}
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append({t("demo_model", language): name,
                     t("demo_architecture", language): architecture,
                     "Macro F1": _number(metrics.get("macro_f1"), probability=True),
                     "Phishing Recall": _number(metrics.get("phishing_recall"), probability=True),
                     t("demo_false_positives", language): _number(metrics.get("normal_to_spam_or_phishing"))})
    return rows


def render_overview(language: str) -> None:
    st.title(t("demo_overview_title", language))
    st.write(t("demo_overview_intro", language))
    for column, key in zip(st.columns(3), ("classes", "languages", "modes"), strict=True):
        column.metric(t("demo_" + key, language), "3")
    st.caption(t("demo_counts_note", language))
    st.subheader(t("demo_pipeline_title", language))
    with st.container(border=True):
        st.markdown("**Email → Preprocessing → Classifier → Security Analysis → Routing**")
    for column, key, title in zip(st.columns(3), ("traditional", "multilingual", "hybrid"),
                                  ("Traditional ML", "Multilingual Experimental", "Hybrid Reviewer"), strict=True):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(t("demo_" + key + "_status", language))
            st.write(t("demo_" + key + "_body", language))
    st.subheader(t("demo_limits_title", language))
    st.write(t("demo_limits", language))


def render_model_summary(root: Path, language: str) -> None:
    st.title(t("demo_comparison_title", language))
    st.caption(t("demo_validation_only", language))
    document = saved_result(root, "models")
    rows = comparison_rows(document, language)
    first, second = [row["Macro F1"] for row in rows]
    with st.container(border=True):
        st.markdown(f"**{t('demo_recommendation', language)}**")
        if first is not None and second is not None and first > second:
            st.write("TF-IDF + LinearSVC · Traditional ML")
            st.caption(t("demo_recommendation_scope", language))
        else:
            st.write(t("demo_missing_comparison", language))
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        "Macro F1": st.column_config.NumberColumn(format="%.6f"),
        "Phishing Recall": st.column_config.NumberColumn(format="%.6f"),
        t("demo_false_positives", language): st.column_config.NumberColumn(format="%d"),
    })
    valid = frame.dropna(subset=["Macro F1"])
    if len(valid) == 2:
        figure = go.Figure(go.Bar(
            x=valid[t("demo_model", language)], y=valid["Macro F1"],
            marker_color=["#2563eb", "#64748b"],
            text=[f"{value:.6f}" for value in valid["Macro F1"]], textposition="outside",
            hovertemplate="%{x}<br>Macro F1: %{y:.6f}<extra></extra>",
        ))
        figure.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10),
                             yaxis=dict(range=[0, 1.08], title="Macro F1"), showlegend=False)
        st.plotly_chart(figure, width="stretch")
        st.write(t("demo_conclusion", language) if first > second
                 else t("demo_no_recommendation", language))
    else:
        st.info(t("demo_missing_comparison", language))
    st.caption(t("demo_limits", language))


def render_reduction_summary(root: Path, language: str) -> None:
    st.subheader(t("demo_reduction_title", language))
    document = saved_result(root, "reduction")
    candidates = []
    for item in document.get("candidates", []):
        metrics = item.get("metrics")
        if item.get("status") == "PASS" and isinstance(metrics, dict):
            candidates.append({t("demo_model", language): item.get("candidate"),
                               "Macro F1": _number(metrics.get("macro_f1"), probability=True),
                               "Phishing Recall": _number(metrics.get("phishing_recall"), probability=True)})
    if candidates:
        st.dataframe(pd.DataFrame(candidates), hide_index=True, width="stretch")
        st.caption(t("demo_reduction_note", language))
    else:
        st.info(t("demo_missing_reduction", language))
