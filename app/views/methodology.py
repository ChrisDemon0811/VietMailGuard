"""Bilingual methodology and limitations page."""

from __future__ import annotations

import streamlit as st

from i18n import t
from styles import render_page_header


def _language() -> str:
    return str(st.session_state.get("interface_language", "en"))


def render() -> None:
    """Render the Methodology / About page."""
    language = _language()
    render_page_header(t("methodology_title", language), t("methodology_intro", language))
    st.info(t("model_scope_notice", language))
    st.warning(t("confidence_risk_distinction", language))

    sections = [
        ("method_version_title", "method_version_body"),
        ("method_task_title", "method_task_body"),
        ("method_tfidf_title", "method_tfidf_body"),
        ("method_models_title", "method_models_body"),
        ("method_calibration_title", "method_calibration_body"),
        ("method_evaluation_title", "method_evaluation_body"),
        ("method_risk_title", "method_risk_body"),
        ("method_explainability_title", "method_explainability_body"),
        ("method_limitations_title", "method_limitations_body"),
        ("method_future_title", "method_future_body"),
        ("provenance_title", "provenance_body"),
    ]
    left_column, right_column = st.columns(2, gap="large")
    for index, (title_key, body_key) in enumerate(sections):
        column = left_column if index % 2 == 0 else right_column
        with column:
            with st.container(border=True):
                st.subheader(t(title_key, language))
                st.markdown(t(body_key, language))


render()
