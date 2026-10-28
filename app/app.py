"""Single bilingual Streamlit entry point for VietMailGuard."""

from __future__ import annotations

import importlib
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

# A long-running Streamlit process can retain a pre-update UI module even when
# the page scripts are new. Refresh only this side-effect-free presentation
# module before pages import their required helpers; keep models/caches and
# session state untouched.
import runtime_ui

if not callable(getattr(runtime_ui, "reviewer_request_options", None)):
    runtime_ui = importlib.reload(runtime_ui)

from i18n import SUPPORTED_LANGUAGES, t
from mailbox_ui import sync_mailbox_navigation
from runtime_ui import configured_mailbox_path, render_reviewer_selector
from styles import apply_global_styles
from vietmailguard.mail_service import MailService
from vietmailguard.v21.runtime_inference import model_availability
from vietmailguard.v21.runtime_modes import V2_MODE, V21_ML_MODE, V21_HYBRID_MODE

st.set_page_config(
    page_title="VietMailGuard Mail",
    page_icon=str(APP_ROOT / "assets" / "mail.svg"),
    layout="wide",
    initial_sidebar_state="auto",
)
apply_global_styles()

if "interface_language" not in st.session_state:
    st.session_state["interface_language"] = "en"

language = str(st.session_state["interface_language"])
mail_page = st.Page(
    "views/mail_client.py",
    title=t("nav_mail_client", language),
    icon=":material/security:",
    url_path="mail",
    default=True,
)
quick_analyzer_page = st.Page(
    "views/email_analyzer.py",
    title=t("nav_email_analyzer", language),
    icon=":material/troubleshoot:",
    url_path="quick-analyzer",
)
model_page = st.Page(
    "views/model_dashboard.py",
    title=t("nav_model_dashboard", language),
    icon=":material/model_training:",
    url_path="model-dashboard",
)
dataset_page = st.Page(
    "views/dataset_dashboard.py",
    title=t("nav_dataset_dashboard", language),
    icon=":material/dataset:",
    url_path="dataset-dashboard",
)
methodology_page = st.Page(
    "views/methodology.py",
    title=t("nav_methodology", language),
    icon=":material/menu_book:",
    url_path="methodology",
)
navigation = st.navigation(
    [mail_page, quick_analyzer_page, model_page, dataset_page, methodology_page],
    position="hidden",
)

with st.sidebar:
    st.markdown(f"### {t('nav_mail_client', language)}")
    try:
        with MailService.from_database_path(configured_mailbox_path()) as mail_service:
            mailbox_counts = mail_service.mailbox_counts()
    except Exception:  # pragma: no cover - environment-dependent UI boundary
        mailbox_counts = None
        st.caption(t("mailbox_database_error", language))

    def count_for(view: str) -> int | None:
        if mailbox_counts is None:
            return None
        if view == "gan_sao":
            return int(mailbox_counts["starred"])
        if view == "security":
            return int(mailbox_counts["security_total"])
        return int(mailbox_counts["folders"].get(view, 0))

    def link_label(key: str, view: str) -> str:
        count = count_for(view)
        label = t(key, language)
        if count is None:
            return label
        if view == "hop_thu_den" and mailbox_counts is not None:
            unread = int(mailbox_counts["unread"].get(view, 0))
            return f"{label}  {count} · {unread} {t('unread', language).lower()}"
        return f"{label}  {count}"

    mailbox_links = (
        ("folder_inbox", "hop_thu_den", ":material/inbox:"),
        ("folder_starred", "gan_sao", ":material/star:"),
        ("folder_spam", "thu_rac", ":material/report:"),
        ("folder_quarantine", "cach_ly", ":material/gpp_bad:"),
        ("folder_deleted", "da_xoa", ":material/delete:"),
    )
    for label_key, view, icon in mailbox_links:
        if st.button(
            link_label(label_key, view),
            icon=icon,
            key=f"mail_nav_{view}",
            type="tertiary",
            width="stretch",
        ):
            sync_mailbox_navigation(st.session_state, view, "list")
            st.switch_page(mail_page, query_params={"view": view, "mode": "list"})
    st.markdown('<div class="vmg-sidebar-rule"></div>', unsafe_allow_html=True)
    if st.button(
        link_label("security_view", "security"),
        icon=":material/shield:",
        key="mail_nav_security",
        type="tertiary",
        width="stretch",
    ):
        sync_mailbox_navigation(st.session_state, "security", "list")
        st.switch_page(mail_page, query_params={"view": "security", "mode": "list"})
    st.markdown('<div class="vmg-sidebar-rule"></div>', unsafe_allow_html=True)
    st.caption(t("mail_tools", language))
    st.page_link(
        quick_analyzer_page,
        label=t("nav_email_analyzer", language),
        icon=":material/troubleshoot:",
        width="stretch",
    )
    st.page_link(model_page, label=t("nav_model_dashboard", language), width="stretch")
    st.page_link(dataset_page, label=t("nav_dataset_dashboard", language), width="stretch")
    st.page_link(methodology_page, label=t("nav_methodology", language), width="stretch")
    st.markdown('<div class="vmg-sidebar-rule"></div>', unsafe_allow_html=True)
    language = st.selectbox(
        t("interface_language", language),
        options=list(SUPPORTED_LANGUAGES),
        index=list(SUPPORTED_LANGUAGES).index(st.session_state["interface_language"]),
        format_func=lambda code: t("language_name", code),
        key="language_selector",
    )
    st.session_state["interface_language"] = language
    v21_available, _ = model_availability()
    modes = [V2_MODE]
    if v21_available:
        modes.extend([V21_ML_MODE, V21_HYBRID_MODE])
    selected_mode = st.session_state.get(
        "analysis_mode_selector", st.session_state.get("analysis_mode", V2_MODE)
    )
    if selected_mode not in modes:
        selected_mode = V2_MODE
    st.session_state["analysis_mode_selector"] = selected_mode
    st.selectbox(
        t("analysis_mode", language), options=modes,
        format_func=lambda value: t(f"mode_{value}", language),
        key="analysis_mode_selector",
    )
    st.session_state["analysis_mode"] = st.session_state["analysis_mode_selector"]
    if not v21_available:
        st.caption(t("mode_v21_unavailable", language))
    if st.session_state["analysis_mode"] == V21_HYBRID_MODE:
        render_reviewer_selector(language)

navigation.run()
