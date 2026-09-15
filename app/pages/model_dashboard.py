"""Model selection and held-out evaluation dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from i18n import class_label, t
from styles import render_page_header
from vietmailguard.dashboard_data import ModelDashboardData, load_model_dashboard_data
from vietmailguard.inference import model_display_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def _dashboard_data() -> ModelDashboardData:
    return load_model_dashboard_data(PROJECT_ROOT)


def _language() -> str:
    return str(st.session_state.get("interface_language", "en"))


def _show_artifact_errors(errors: list[str], language: str) -> None:
    if not errors:
        return
    with st.expander(t("artifact_error_title", language)):
        for error in errors:
            st.code(error, language=None)


def _format_metric(value: Any, language: str) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return t("not_available", language)


def _render_selected_model(data: ModelDashboardData, language: str) -> None:
    st.subheader(t("selected_production_model", language))
    selected = data.selected_model
    if selected is None:
        st.warning(t("missing_model_selection", language))
        return
    name_column, classifier_column, features_column = st.columns(3, gap="large")
    name_column.metric(t("model_version", language), model_display_name(selected))
    classifier_column.metric(
        t("classifier", language), str(selected.get("classifier", t("not_available", language)))
    )
    features_column.metric(
        t("feature_representation", language),
        str(selected.get("representation", t("not_available", language))),
    )
    st.caption(t("selection_basis", language))


def _render_test_metrics(data: ModelDashboardData, language: str) -> None:
    st.subheader(t("held_out_test_metrics", language))
    metrics = data.test_metrics
    if metrics is None:
        st.warning(t("missing_model_metrics", language))
        return
    phishing = metrics.get("per_class", {}).get("phishing", {})
    test_rows = (data.final_evaluation or {}).get("test_data", {}).get("rows")
    metric_specs = [
        ("test_rows", test_rows),
        ("accuracy", metrics.get("accuracy")),
        ("macro_precision", metrics.get("macro_precision")),
        ("macro_recall", metrics.get("macro_recall")),
        ("macro_f1", metrics.get("macro_f1")),
        ("weighted_f1", metrics.get("weighted_f1")),
        ("phishing_recall", phishing.get("recall")),
    ]
    columns = list(st.columns(4)) + list(st.columns(3))
    for column, (key, value) in zip(columns, metric_specs, strict=True):
        display = (
            f"{int(value):,}"
            if key == "test_rows" and value is not None
            else _format_metric(value, language)
        )
        column.metric(t(key, language), display)
    st.caption(t("test_evaluation_note", language))

    st.markdown(f"**{t('per_class_metrics', language)}**")
    per_class = data.per_class_metrics.copy()
    if {"evaluation_scope", "row_type"}.issubset(per_class.columns):
        per_class = per_class.loc[
            per_class["evaluation_scope"].eq("combined")
            & per_class["row_type"].eq("class_metric")
        ].copy()
    if per_class.empty and isinstance(metrics.get("per_class"), dict):
        per_class = pd.DataFrame.from_dict(metrics["per_class"], orient="index")
        per_class.index.name = "class"
        per_class = per_class.reset_index()
    if not per_class.empty:
        per_class["class"] = per_class["class"].map(
            lambda value: class_label(str(value), language)
        )
        renamed = per_class.rename(
            columns={
                "class": t("class", language),
                "precision": t("precision", language),
                "recall": t("recall", language),
                "f1": t("f1", language),
                "support": t("support", language),
            }
        )
        percentage_columns = [
            t("precision", language),
            t("recall", language),
            t("f1", language),
        ]
        st.dataframe(
            renamed.style.format({column: "{:.4f}" for column in percentage_columns}),
            hide_index=True,
            width="stretch",
        )


def _render_translated_vietnamese(data: ModelDashboardData, language: str) -> None:
    st.subheader(t("translated_vietnamese_test", language))
    metrics = data.translated_vietnamese_metrics
    if metrics is None:
        st.info(t("not_evaluated_yet", language))
        return
    per_class = metrics.get("per_class", {})
    columns = st.columns(4)
    columns[0].metric(t("rows", language), f"{int(metrics.get('rows', 0)):,}")
    columns[1].metric(
        class_label("normal", language),
        f"{int(per_class.get('normal', {}).get('support', 0)):,}",
    )
    columns[2].metric(
        class_label("spam", language),
        f"{int(per_class.get('spam', {}).get('support', 0)):,}",
    )
    columns[3].metric(
        class_label("phishing", language),
        f"{int(per_class.get('phishing', {}).get('support', 0)):,}",
    )
    st.metric(
        t("represented_macro_f1", language),
        _format_metric(metrics.get("macro_f1_supported_classes"), language),
    )
    st.warning(t("no_vietnamese_phishing_examples", language))
    st.info(t("native_benchmark_unavailable", language))


def _render_confusion_matrix(data: ModelDashboardData, language: str) -> None:
    matrix = data.confusion_matrix
    if matrix.empty or matrix.shape[1] < 2:
        return
    st.subheader(t("confusion_matrix", language))
    raw_actual = matrix.iloc[:, 0].astype(str).tolist()
    raw_predicted = [str(value) for value in matrix.columns[1:]]
    actual = [class_label(value, language) for value in raw_actual]
    predicted = [class_label(value, language) for value in raw_predicted]
    values = matrix.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    figure = go.Figure(
        data=go.Heatmap(
            z=values,
            x=predicted,
            y=actual,
            colorscale=[[0.0, "#eff6ff"], [1.0, "#1d4ed8"]],
            text=values.astype(int),
            texttemplate="%{text}",
            hovertemplate="%{y} → %{x}: %{z}<extra></extra>",
            showscale=False,
        )
    )
    figure.update_layout(
        xaxis_title=t("predicted_class_axis", language),
        yaxis_title=t("actual_class", language),
        height=420,
        margin=dict(l=30, r=20, t=20, b=40),
    )
    st.plotly_chart(figure, width="stretch")


def _render_comparison(data: ModelDashboardData, language: str) -> None:
    comparison = data.comparison.copy()
    if comparison.empty:
        return
    st.subheader(t("model_comparison", language))
    st.caption(t("comparison_note", language))

    selected_column = (
        "selected" if "selected" in comparison.columns else "recommended_tfidf_candidate"
    )
    if {"experiment_id", "macro_f1", selected_column}.issubset(comparison.columns):
        colors = [
            "#06b6d4" if bool(value) else "#64748b"
            for value in comparison[selected_column]
        ]
        figure = go.Figure(
            data=go.Bar(
                x=comparison["macro_f1"],
                y=comparison["experiment_id"],
                orientation="h",
                marker_color=colors,
                text=comparison["macro_f1"].map(lambda value: f"{float(value):.4f}"),
                textposition="outside",
                hovertemplate=(
                    f"%{{y}}<br>{t('macro_f1', language)}: %{{x:.6f}}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            height=440,
            margin=dict(l=10, r=65, t=10, b=35),
            xaxis_title=t("macro_f1", language),
            yaxis_title="",
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(figure, width="stretch")

    visible_columns = [
        "validation_rank",
        "experiment_id",
        "feature_set",
        "classifier",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "phishing_recall",
        selected_column,
    ]
    visible_columns = [column for column in visible_columns if column in comparison.columns]
    display = comparison.loc[:, visible_columns].rename(
        columns={
            "validation_rank": t("validation_rank", language),
            "experiment_id": t("experiment", language),
            "feature_set": t("feature_set", language),
            "classifier": t("classifier", language),
            "accuracy": t("accuracy", language),
            "macro_precision": t("macro_precision", language),
            "macro_recall": t("macro_recall", language),
            "macro_f1": t("macro_f1", language),
            "weighted_f1": t("weighted_f1", language),
            "phishing_recall": t("phishing_recall", language),
            selected_column: t("selected", language),
        }
    )
    metric_columns = [
        t("accuracy", language),
        t("macro_precision", language),
        t("macro_recall", language),
        t("macro_f1", language),
        t("weighted_f1", language),
        t("phishing_recall", language),
    ]
    present_metric_columns = [column for column in metric_columns if column in display.columns]
    st.dataframe(
        display.style.format({column: "{:.4f}" for column in present_metric_columns}),
        hide_index=True,
        width="stretch",
        height=385,
    )


def _render_robustness(data: ModelDashboardData, language: str) -> None:
    st.subheader(t("robustness_title", language))
    st.caption(t("robustness_intro", language))
    if data.robustness_metrics.empty:
        st.info(t("not_evaluated_yet", language))
    else:
        columns = [
            "scope_value",
            "rows",
            "accuracy",
            "macro_f1_supported_classes",
            "prediction_flip_rate",
            "mean_confidence_drop_vs_clean",
            "mean_risk_score_change_vs_clean",
        ]
        frame = data.robustness_metrics.loc[
            data.robustness_metrics["scope_value"].ne("all_transformed"),
            [column for column in columns if column in data.robustness_metrics.columns],
        ].copy()
        frame["scope_value"] = frame["scope_value"].map(
            lambda value: t(f"robustness_{value}", language)
        )
        frame = frame.rename(
            columns={
                "scope_value": t("robustness_variant", language),
                "rows": t("rows", language),
                "accuracy": t("accuracy", language),
                "macro_f1_supported_classes": t("represented_macro_f1", language),
                "prediction_flip_rate": t("prediction_flip_rate", language),
                "mean_confidence_drop_vs_clean": t("confidence_drop", language),
                "mean_risk_score_change_vs_clean": t("risk_score_change", language),
            }
        )
        st.dataframe(frame, hide_index=True, width="stretch")

    st.markdown(f"**{t('short_form_challenge_title', language)}**")
    if data.short_form_metrics.empty:
        st.info(t("not_evaluated_yet", language))
    else:
        rows = data.short_form_metrics
        focus = rows.loc[
            ((rows["scope_type"].eq("overall")) & (rows["scope_value"].eq("all")))
            | rows["scope_value"].isin(["very_short", "short", "normal_length"])
        ]
        visible = [
            "scope_type",
            "scope_value",
            "rows",
            "accuracy",
            "macro_f1_supported_classes",
            "failure_rate",
            "ml_normal_with_promotional_signal",
        ]
        focus = focus.loc[:, [column for column in visible if column in focus.columns]].copy()
        focus["scope_value"] = focus["scope_value"].map(
            lambda value: t(f"challenge_{value}", language)
        )
        focus = focus.rename(
            columns={
                "scope_type": t("scope", language),
                "scope_value": t("challenge_subset", language),
                "rows": t("rows", language),
                "accuracy": t("accuracy", language),
                "macro_f1_supported_classes": t("represented_macro_f1", language),
                "failure_rate": t("failure_rate", language),
                "ml_normal_with_promotional_signal": t(
                    "normal_with_promotion_signal", language
                ),
            }
        )
        st.dataframe(
            focus,
            hide_index=True,
            width="stretch",
        )


def render() -> None:
    """Render the Model Dashboard page."""
    language = _language()
    render_page_header(
        t("model_dashboard_title", language), t("model_dashboard_intro", language)
    )
    data = _dashboard_data()
    _show_artifact_errors(data.errors, language)
    _render_selected_model(data, language)
    st.divider()
    _render_test_metrics(data, language)
    _render_confusion_matrix(data, language)
    st.divider()
    _render_translated_vietnamese(data, language)
    st.divider()
    _render_comparison(data, language)
    st.divider()
    _render_robustness(data, language)


render()
