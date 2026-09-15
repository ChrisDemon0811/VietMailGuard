"""Single bilingual Streamlit entry point for VietMailGuard."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
APP_ROOT = Path(__file__).resolve().parent
for import_root in (PROJECT_ROOT, SOURCE_ROOT, APP_ROOT):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

import streamlit as st

from i18n import SUPPORTED_LANGUAGES, t
from styles import apply_global_styles

st.set_page_config(
    page_title="VietMailGuard",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_styles()

if "interface_language" not in st.session_state:
    st.session_state["interface_language"] = "en"

with st.sidebar:
    language = st.selectbox(
        t("interface_language", st.session_state["interface_language"]),
        options=list(SUPPORTED_LANGUAGES),
        index=list(SUPPORTED_LANGUAGES).index(st.session_state["interface_language"]),
        format_func=lambda code: t("language_name", code),
        key="language_selector",
    )
    st.session_state["interface_language"] = language
    st.markdown(f"### {t('app_title', language)}")
    st.caption(t("app_subtitle", language))
    st.caption(t("sidebar_scope", language))

navigation = st.navigation(
    [
        st.Page(
            "pages/email_analyzer.py",
            title=t("nav_email_analyzer", language),
            icon=":material/mail:",
            url_path="email-analyzer",
            default=True,
        ),
        st.Page(
            "pages/model_dashboard.py",
            title=t("nav_model_dashboard", language),
            icon=":material/model_training:",
            url_path="model-dashboard",
        ),
        st.Page(
            "pages/dataset_dashboard.py",
            title=t("nav_dataset_dashboard", language),
            icon=":material/dataset:",
            url_path="dataset-dashboard",
        ),
        st.Page(
            "pages/methodology.py",
            title=t("nav_methodology", language),
            icon=":material/menu_book:",
            url_path="methodology",
        ),
    ]
)
navigation.run()
