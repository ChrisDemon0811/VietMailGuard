"""Dataset dashboards from saved reports; no raw email or held-out row reads."""

from pathlib import Path

import streamlit as st

from i18n import t
from research_dashboard_ui import render_dataset_research
from styles import render_page_header


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_REPORTS = (
    "bilingual_dataset_report.md",
    "bilingual_split_report.md",
    "vietnamese_label_curation_report.md",
    "translated_augmentation_report.md",
)


def render() -> None:
    """Present development, audit and research scopes alongside frozen V2 reports."""
    language = str(st.session_state.get("interface_language", "en"))
    render_page_header(t("dataset_dashboard_title", language), t("rd_dataset_intro", language))
    research, production = st.tabs([t("rd_research", language), t("rd_v2", language)])
    with research:
        render_dataset_research(PROJECT_ROOT, language)
    with production:
        st.info(t("rd_v2_dataset_notice", language))
        for name in V2_REPORTS:
            with st.expander(name, expanded=name == V2_REPORTS[0]):
                try:
                    report = (PROJECT_ROOT / "reports" / "v2" / name).read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    st.info(t("rd_section_unavailable", language))
                else:
                    st.markdown(report)


render()
