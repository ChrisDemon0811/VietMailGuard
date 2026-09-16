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
from vietmailguard.mail_service import MailService

st.set_page_config(
    page_title="VietMailGuard Mail",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_styles()

if "interface_language" not in st.session_state:
    st.session_state["interface_language"] = "en"

language = str(st.session_state["interface_language"])
mail_page = st.Page(
    "pages/mail_client.py",
    title=t("nav_mail_client", language),
    icon=":material/security:",
    url_path="mail",
    default=True,
)
quick_analyzer_page = st.Page(
    "pages/email_analyzer.py",
    title=t("nav_email_analyzer", language),
    icon=":material/troubleshoot:",
    url_path="quick-analyzer",
)
model_page = st.Page(
    "pages/model_dashboard.py",
    title=t("nav_model_dashboard", language),
    icon=":material/model_training:",
    url_path="model-dashboard",
)
dataset_page = st.Page(
    "pages/dataset_dashboard.py",
    title=t("nav_dataset_dashboard", language),
    icon=":material/dataset:",
    url_path="dataset-dashboard",
)
methodology_page = st.Page(
    "pages/methodology.py",
    title=t("nav_methodology", language),
    icon=":material/menu_book:",
    url_path="methodology",
)
navigation = st.navigation(
    [mail_page, quick_analyzer_page, model_page, dataset_page, methodology_page],
    position="hidden",
)

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
    try:
        with MailService.from_database_path() as mail_service:
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

    st.divider()
    st.caption(t("mailbox_navigation", language).upper())
    mailbox_links = (
        ("folder_inbox", "hop_thu_den", ":material/inbox:"),
        ("folder_starred", "gan_sao", ":material/star:"),
        ("folder_spam", "thu_rac", ":material/report:"),
        ("folder_quarantine", "cach_ly", ":material/gpp_bad:"),
        ("folder_deleted", "da_xoa", ":material/delete:"),
        ("security_view", "security", ":material/shield:"),
    )
    for label_key, view, icon in mailbox_links:
        st.page_link(
            mail_page,
            label=link_label(label_key, view),
            icon=icon,
            query_params={"view": view, "mode": "list"},
            width="stretch",
        )
    st.page_link(
        quick_analyzer_page,
        label=t("nav_email_analyzer", language),
        icon=":material/troubleshoot:",
        width="stretch",
    )
    st.divider()
    st.page_link(model_page, label=t("nav_model_dashboard", language), width="stretch")
    st.page_link(dataset_page, label=t("nav_dataset_dashboard", language), width="stretch")
    st.page_link(methodology_page, label=t("nav_methodology", language), width="stretch")

navigation.run()
