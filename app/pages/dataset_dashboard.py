"""Dataset cleaning, provenance, and leakage-safe split dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from i18n import class_label, t
from styles import render_page_header
from vietmailguard.dashboard_data import (
    DatasetDashboardData,
    aggregate_split_distribution,
    load_dataset_dashboard_data,
    split_summary_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def _dashboard_data() -> DatasetDashboardData:
    return load_dataset_dashboard_data(PROJECT_ROOT)


def _language() -> str:
    return str(st.session_state.get("interface_language", "en"))


def _show_artifact_errors(errors: list[str], language: str) -> None:
    if not errors:
        return
    with st.expander(t("artifact_error_title", language)):
        for error in errors:
            st.code(error, language=None)


def _bar_chart(
    frame: pd.DataFrame,
    *,
    labels: list[str],
    color: str,
    x_title: str,
    y_title: str,
) -> go.Figure:
    figure = go.Figure(
        data=go.Bar(
            x=labels,
            y=frame["count"],
            marker_color=color,
            text=frame["count"],
            textposition="outside",
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
    )
    figure.update_layout(
        height=350,
        margin=dict(l=20, r=15, t=15, b=35),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return figure


def _render_overview(data: DatasetDashboardData, language: str) -> None:
    report = data.split_report
    if report is None:
        st.warning(t("missing_dataset_report", language))
        return
    leakage = report.get("leakage_check", {})
    leakage_passed = bool(leakage.get("passed", False))
    columns = st.columns(4)
    columns[0].metric(t("master_samples", language), f"{int(report.get('total_rows', 0)):,}")
    columns[1].metric(t("groups", language), f"{int(report.get('total_groups', 0)):,}")
    columns[2].metric(
        t("final_leakage_groups", language),
        f"{int(data.duplicate_stats.get('final_groups', 0)):,}",
    )
    columns[3].metric(
        t("leakage_check", language),
        t("passed", language) if leakage_passed else t("failed", language),
    )
    if leakage_passed:
        st.success(t("leakage_pass_note", language))
    else:
        st.error(t("leakage_fail_note", language))


def _render_distributions(data: DatasetDashboardData, language: str) -> None:
    report = data.split_report
    if report is None:
        return
    class_distribution = aggregate_split_distribution(report, "class_distribution")
    source_distribution = aggregate_split_distribution(report, "source_distribution")
    language_distribution = aggregate_split_distribution(report, "language_distribution")
    origin_distribution = aggregate_split_distribution(report, "data_origin_distribution")

    class_column, source_column = st.columns(2, gap="large")
    with class_column:
        st.subheader(t("class_distribution", language))
        class_labels = [
            class_label(str(value), language) for value in class_distribution["category"]
        ]
        st.plotly_chart(
            _bar_chart(
                class_distribution,
                labels=class_labels,
                color="#2563eb",
                x_title=t("class", language),
                y_title=t("sample_count", language),
            ),
            width="stretch",
        )
    with source_column:
        st.subheader(t("source_distribution", language))
        st.plotly_chart(
            _bar_chart(
                source_distribution,
                labels=source_distribution["category"].astype(str).tolist(),
                color="#0891b2",
                x_title=t("source", language),
                y_title=t("sample_count", language),
            ),
            width="stretch",
        )

    language_column, origin_column = st.columns(2, gap="large")
    with language_column:
        st.subheader(t("language_distribution", language))
        language_display = language_distribution.rename(
            columns={
                "category": t("language", language),
                "count": t("sample_count", language),
            }
        )
        st.dataframe(language_display, hide_index=True, width="stretch")
    with origin_column:
        st.subheader(t("data_origin_distribution", language))
        origin_display = origin_distribution.rename(
            columns={
                "category": t("data_origin", language),
                "count": t("sample_count", language),
            }
        )
        st.dataframe(origin_display, hide_index=True, width="stretch")


def _render_eligibility(data: DatasetDashboardData, language: str) -> None:
    st.subheader(t("eligibility_title", language))
    summary = data.summary
    if not summary:
        st.warning(t("missing_dataset_report", language))
        return
    first_row = st.columns(4)
    first_row[0].metric(
        t("english_samples", language), f"{int(summary.get('english_samples', 0)):,}"
    )
    first_row[1].metric(
        t("translated_vi_samples", language),
        f"{int(summary.get('translated_vietnamese_samples', 0)):,}",
    )
    first_row[2].metric(
        t("review_rows", language), f"{int(summary.get('review_rows', 0)):,}"
    )
    first_row[3].metric(
        t("excluded_rows", language), f"{int(summary.get('excluded_rows', 0)):,}"
    )
    second_row = st.columns(3)
    second_row[0].metric(
        t("augmentation_rows", language),
        f"{int(summary.get('augmentation_rows', 0)):,}",
    )
    second_row[1].metric(
        t("augmentation_eligible", language),
        f"{int(summary.get('augmentation_eligible', 0)):,}",
    )
    second_row[2].metric(
        t("augmentation_failed", language),
        f"{int(summary.get('augmentation_failed', 0)):,}",
    )
    st.caption(t("eligibility_note", language))


def _render_duplicates(data: DatasetDashboardData, language: str) -> None:
    st.subheader(t("duplicate_statistics", language))
    stats = data.duplicate_stats
    columns = st.columns(4)
    columns[0].metric(
        t("final_leakage_groups", language), f"{int(stats.get('final_groups', 0)):,}"
    )
    columns[1].metric(
        t("translation_groups", language),
        f"{int(stats.get('translation_groups', 0)):,}",
    )
    columns[2].metric(
        t("exact_duplicate_groups", language),
        f"{int(stats.get('exact_duplicate_groups', 0)):,}",
    )
    columns[3].metric(
        t("template_duplicate_groups", language),
        f"{int(stats.get('template_duplicate_groups', 0)):,}",
    )
    st.caption(t("duplicate_policy_note", language))


def _render_splits(data: DatasetDashboardData, language: str) -> None:
    report = data.split_report
    if report is None:
        return
    st.subheader(t("split_statistics", language))
    st.caption(t("split_integrity_note", language))
    splits = split_summary_frame(report)
    if splits.empty:
        st.warning(t("missing_dataset_report", language))
        return
    display = splits.rename(
        columns={
            "split": t("split", language),
            "rows": t("rows", language),
            "groups": t("groups", language),
            "actual_ratio": t("actual_ratio", language),
            "target_ratio": t("target_ratio", language),
        }
    )
    ratio_columns = [t("actual_ratio", language), t("target_ratio", language)]
    st.dataframe(
        display.style.format({column: "{:.2%}" for column in ratio_columns}),
        hide_index=True,
        width="stretch",
    )


def render() -> None:
    """Render the Dataset Dashboard page."""
    language = _language()
    render_page_header(
        t("dataset_dashboard_title", language), t("dataset_dashboard_intro", language)
    )
    data = _dashboard_data()
    _show_artifact_errors(data.errors, language)
    _render_overview(data, language)
    st.divider()
    _render_distributions(data, language)
    st.divider()
    _render_eligibility(data, language)
    st.divider()
    _render_duplicates(data, language)
    st.divider()
    _render_splits(data, language)


render()
