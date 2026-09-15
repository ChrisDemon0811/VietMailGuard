"""Three-area local mailbox UI backed exclusively by MailService."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from i18n import action_label, class_label, risk_label, t
from mailbox_ui import (
    FOLDER_VIEW_KEYS,
    display_timestamp,
    message_preview,
    normalize_mailbox_view,
)
from vietmailguard.mail_database import DEFAULT_DATABASE_PATH
from vietmailguard.mail_service import MailService


FOLDER_VIEWS = {"hop_thu_den", "thu_rac", "cach_ly", "da_xoa"}


def _language() -> str:
    return str(st.session_state.get("interface_language", "en"))


def _view_from_query() -> str:
    return normalize_mailbox_view(st.query_params.get("view", "hop_thu_den"))


def _messages_for_view(
    service: MailService,
    view: str,
    query: str,
    filters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    query_filters = dict(filters)
    query_filters["limit"] = 250
    if view in FOLDER_VIEWS:
        query_filters["folder"] = view
    if view == "gan_sao":
        query_filters["starred"] = True
    if view == "security":
        return service.list_security_view(query=query, **query_filters)
    return service.search_email(query, **query_filters)


def _render_search_filters(view: str, language: str) -> tuple[dict[str, Any], bool]:
    """Return validated UI filter values; repository performs all SQL filtering."""

    with st.expander(t("mail_filters", language), expanded=view == "security"):
        columns = st.columns(6)
        if view in FOLDER_VIEWS:
            folder = columns[0].selectbox(
                t("filter_folder", language),
                options=[view],
                format_func=lambda value: t(FOLDER_VIEW_KEYS[value], language),
                disabled=True,
                key=f"mail_filter_folder_{view}",
            )
        else:
            folder = columns[0].selectbox(
                t("filter_folder", language),
                options=[""] + sorted(FOLDER_VIEWS),
                format_func=lambda value: (
                    t("filter_all", language)
                    if not value
                    else t(FOLDER_VIEW_KEYS[value], language)
                ),
                key=f"mail_filter_folder_{view}",
            )

        read_choice = columns[1].selectbox(
            t("filter_read_state", language),
            options=["all", "unread", "read"],
            format_func=lambda value: {
                "all": t("filter_all", language),
                "unread": t("unread", language),
                "read": t("read", language),
            }[value],
            key=f"mail_filter_read_{view}",
        )
        if view == "gan_sao":
            starred_choice = columns[2].selectbox(
                t("filter_starred_state", language),
                options=["starred"],
                format_func=lambda _: t("folder_starred", language),
                disabled=True,
                key=f"mail_filter_starred_{view}",
            )
        else:
            starred_choice = columns[2].selectbox(
                t("filter_starred_state", language),
                options=["all", "starred", "unstarred"],
                format_func=lambda value: {
                    "all": t("filter_all", language),
                    "starred": t("folder_starred", language),
                    "unstarred": t("unstarred", language),
                }[value],
                key=f"mail_filter_starred_{view}",
            )
        prediction = columns[3].selectbox(
            t("filter_prediction", language),
            options=["", "normal", "spam", "phishing"],
            format_func=lambda value: (
                t("filter_all", language)
                if not value
                else class_label(value, language)
            ),
            key=f"mail_filter_prediction_{view}",
        )
        risk_level = columns[4].selectbox(
            t("filter_risk_level", language),
            options=["", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            format_func=lambda value: (
                t("filter_all", language)
                if not value
                else risk_label(value, language)
            ),
            key=f"mail_filter_risk_{view}",
        )
        email_language = columns[5].selectbox(
            t("filter_email_language", language),
            options=["", "en", "vi", "mixed", "unknown"],
            format_func=lambda value: (
                t("filter_all", language)
                if not value
                else t(f"email_language_{value}", language)
            ),
            key=f"mail_filter_language_{view}",
        )

    values: dict[str, Any] = {
        "folder": folder or None,
        "unread": (
            None
            if read_choice == "all"
            else read_choice == "unread"
        ),
        "starred": (
            None
            if starred_choice == "all"
            else starred_choice == "starred"
        ),
        "prediction": prediction or None,
        "risk_level": risk_level or None,
        "language": email_language or None,
    }
    active = any(
        value is not None
        for key, value in values.items()
        if not (key == "folder" and view in FOLDER_VIEWS)
        and not (key == "starred" and view == "gan_sao")
    )
    return values, active


def _prediction_badge(analysis: Mapping[str, Any] | None, language: str) -> str:
    if not analysis:
        return ""
    prediction = str(analysis.get("prediction", ""))
    label = escape(class_label(prediction, language).upper())
    return f'<span class="vmg-mail-badge vmg-{escape(prediction)}">{label}</span>'


def _render_message_list(
    service: MailService,
    messages: list[dict[str, Any]],
    language: str,
    *,
    is_search: bool,
) -> None:
    st.markdown(f"### {t('message_list', language)} · {len(messages)}")
    if not messages:
        st.info(
            t("mailbox_no_search_results", language)
            if is_search
            else t("mailbox_empty", language)
        )
        if not is_search:
            st.caption(t("mailbox_empty_help", language))
        return

    for message in messages:
        email_id = int(message["id"])
        analysis = message.get("analysis")
        with st.container(border=True):
            header, status, star = st.columns([5.2, 1.6, 0.7], vertical_alignment="center")
            sender = str(message.get("sender") or t("unknown_sender", language))
            subject = str(message.get("subject") or t("no_subject", language))
            selected = st.session_state.get("mailbox_selected_id") == email_id
            with header:
                if st.button(
                    f"{'● ' if not message.get('is_read') else ''}{sender} · {subject}",
                    key=f"mail_select_{email_id}",
                    type="primary" if selected else "tertiary",
                    width="stretch",
                ):
                    st.session_state["mailbox_selected_id"] = email_id
                    st.rerun()
            with status:
                st.markdown(_prediction_badge(analysis, language), unsafe_allow_html=True)
                if isinstance(analysis, Mapping):
                    st.caption(
                        f"{risk_label(str(analysis.get('risk_level', 'LOW')), language)} "
                        f"· {float(analysis.get('risk_score', 0.0)):.0f}/100"
                    )
            with star:
                if st.button(
                    "★" if message.get("is_starred") else "☆",
                    key=f"mail_star_{email_id}",
                    help=t(
                        "remove_star" if message.get("is_starred") else "add_star",
                        language,
                    ),
                ):
                    service.toggle_star(email_id)
                    st.rerun()
            st.caption(
                f"{display_timestamp(message.get('sent_at') or message.get('created_at'))}  ·  "
                f"{message_preview(message.get('body'))}"
            )


def _render_security_message_list(
    messages: list[dict[str, Any]],
    language: str,
    *,
    is_search: bool,
) -> None:
    """Render stored security results ordered by repository risk score."""

    st.markdown(f"### {t('security_view', language)} · {len(messages)}")
    if not messages:
        st.info(
            t("mailbox_no_search_results", language)
            if is_search
            else t("mailbox_empty", language)
        )
        return

    for message in messages:
        email_id = int(message["id"])
        analysis = message.get("analysis")
        if not isinstance(analysis, Mapping):
            continue
        with st.container(border=True):
            identity, model, risk = st.columns(
                [4.8, 2.0, 2.1], vertical_alignment="center"
            )
            with identity:
                sender = str(message.get("sender") or t("unknown_sender", language))
                subject = str(message.get("subject") or t("no_subject", language))
                selected = st.session_state.get("mailbox_selected_id") == email_id
                if st.button(
                    f"{sender} · {subject}",
                    key=f"security_select_{email_id}",
                    type="primary" if selected else "tertiary",
                    width="stretch",
                ):
                    st.session_state["mailbox_selected_id"] = email_id
                    st.rerun()
            with model:
                st.markdown(_prediction_badge(analysis, language), unsafe_allow_html=True)
                st.caption(
                    f"{t('model_confidence', language)} "
                    f"{float(analysis.get('confidence', 0.0)):.1%}"
                )
            with risk:
                st.markdown(
                    f"**{float(analysis.get('risk_score', 0.0)):.0f}/100 · "
                    f"{risk_label(str(analysis.get('risk_level', 'LOW')), language)}**"
                )
                folder = str(message.get("folder", "hop_thu_den"))
                folder_label = t(FOLDER_VIEW_KEYS.get(folder, "folder_inbox"), language)
                st.caption(
                    f"{t('security_finding_count', language)}: "
                    f"{int(analysis.get('security_finding_count', 0))} · "
                    f"{t('current_folder', language)}: {folder_label}"
                )


def _run_mail_action(action: Any, language: str) -> None:
    try:
        action()
        st.session_state["mailbox_flash"] = ("success", t("mail_action_success", language))
        st.rerun()
    except Exception:  # pragma: no cover - final UI boundary
        st.error(t("mail_action_error", language))


def _render_analysis(
    analysis: Mapping[str, Any] | None,
    routing: Mapping[str, Any] | None,
    language: str,
) -> None:
    st.markdown(f"### {t('stored_analysis', language)}")
    if not analysis:
        st.info(t("no_stored_analysis", language))
        return

    prediction = str(analysis["prediction"])
    model_column, risk_column = st.columns(2)
    with model_column.container(border=True):
        st.caption(t("analysis_model", language))
        st.markdown(_prediction_badge(analysis, language), unsafe_allow_html=True)
        st.metric(
            t("model_confidence", language),
            f"{float(analysis['confidence']):.1%}",
        )
    with risk_column.container(border=True):
        st.caption(t("analysis_risk", language))
        st.metric(
            t("risk_score", language),
            f"{float(analysis['risk_score']):.0f}/100",
        )
        st.markdown(
            f"**{t('risk_level', language)}:** "
            f"{risk_label(str(analysis.get('risk_level', 'LOW')), language)}"
        )
    st.caption(t("confidence_risk_distinction", language))

    if routing and routing.get("has_warning"):
        severity = str(routing.get("warning_severity", analysis.get("risk_level", "LOW")))
        warning_text = t(
            "security_warning_banner",
            language,
            severity=risk_label(severity, language),
            prediction=class_label(prediction, language),
        )
        if severity in {"HIGH", "CRITICAL"}:
            st.error(warning_text)
        else:
            st.warning(warning_text)
    result = analysis.get("result") if isinstance(analysis.get("result"), Mapping) else {}
    with st.container(border=True):
        st.markdown(f"#### {t('analysis_model_explanation', language)}")
        model_explanation = result.get("model_explanation", {})
        features = (
            model_explanation.get("features", [])
            if isinstance(model_explanation, Mapping)
            else []
        )
        if not features:
            st.caption(t("mail_no_findings", language))
        for feature in features[:8]:
            if isinstance(feature, Mapping):
                st.markdown(
                    f"- `{feature.get('feature', '')}` · {float(feature.get('contribution', 0.0)):.4f}"
                )
        if isinstance(model_explanation, Mapping) and model_explanation.get("limitation"):
            st.caption(str(model_explanation["limitation"]))

    with st.container(border=True):
        st.markdown(f"#### {t('analysis_security_findings', language)}")
        content_findings = result.get("content_findings", [])
        sender_findings = result.get("sender_findings", [])
        content_column, sender_column = st.columns(2)
        for column, heading, findings in (
            (content_column, t("analysis_content_findings", language), content_findings),
            (sender_column, t("analysis_sender_findings", language), sender_findings),
        ):
            with column:
                st.markdown(f"**{heading}**")
                if not findings:
                    st.caption(t("mail_no_findings", language))
                for finding in findings:
                    if isinstance(finding, Mapping):
                        st.markdown(
                            f"- **{escape(str(finding.get('code', 'indicator')))}** — "
                            f"{escape(str(finding.get('description', '')))}"
                        )

    with st.container(border=True):
        st.markdown(f"#### {t('analysis_url_findings', language)}")
        url_findings = result.get("url_findings", [])
        if not url_findings:
            st.caption(t("mail_no_findings", language))
        elif any(
            isinstance(item, Mapping) and bool(item.get("findings"))
            for item in url_findings
        ):
            st.warning(t("suspicious_link_warning", language))
        for url in url_findings:
            if isinstance(url, Mapping):
                st.code(str(url.get("url", "")), language=None)
                for finding in url.get("findings", []):
                    if isinstance(finding, Mapping):
                        st.caption(str(finding.get("description") or finding.get("code", "")))

    with st.container(border=True):
        st.markdown(f"#### {t('analysis_recommendation', language)}")
        st.markdown(
            f"**{action_label(str(analysis['recommended_action']), language)}**"
        )
        st.caption(t("recommendation_note", language))

    with st.container(border=True):
        st.markdown(f"#### {t('analysis_limitations', language)}")
        limitations = result.get("limitations", [])
        if not limitations:
            st.caption(t("analysis_no_limitations", language))
        for limitation in limitations:
            st.markdown(f"- {escape(str(limitation))}")


def _render_action_buttons(
    service: MailService,
    message: Mapping[str, Any],
    selected_id: int,
    language: str,
) -> None:
    """Render applicable actions; MailService persists every state transition."""

    folder = str(message.get("folder", ""))
    actions: list[tuple[str, str, Any]] = [
        (
            "star",
            t("remove_star" if message.get("is_starred") else "add_star", language),
            lambda: service.toggle_star(selected_id),
        ),
        (
            "read",
            t("mark_unread" if message.get("is_read") else "mark_read", language),
            lambda: service.mark_read(selected_id, not bool(message.get("is_read"))),
        ),
    ]
    if folder != "hop_thu_den":
        actions.append(
            (
                "inbox",
                t("move_inbox", language),
                lambda: service.move_email(selected_id, "hop_thu_den"),
            )
        )
    if folder == "thu_rac":
        actions.append(
            (
                "not_spam",
                t("not_spam", language),
                lambda: service.record_feedback(
                    selected_id, action="khong_phai_thu_rac"
                ),
            )
        )
    else:
        actions.append(
            (
                "spam",
                t("mark_spam", language),
                lambda: service.record_feedback(
                    selected_id, action="danh_dau_thu_rac"
                ),
            )
        )
    if folder != "cach_ly":
        actions.extend(
            [
                (
                    "report",
                    t("report_phishing", language),
                    lambda: service.record_feedback(
                        selected_id, action="bao_cao_lua_dao"
                    ),
                ),
                (
                    "quarantine",
                    t("move_quarantine", language),
                    lambda: service.move_email(selected_id, "cach_ly"),
                ),
            ]
        )
    if folder != "da_xoa":
        actions.append(
            (
                "delete",
                t("delete_email", language),
                lambda: service.move_email(selected_id, "da_xoa"),
            )
        )

    for start in range(0, len(actions), 3):
        columns = st.columns(3)
        for column, (action_key, label, callback) in zip(
            columns, actions[start : start + 3], strict=False
        ):
            if column.button(
                label,
                key=f"detail_{action_key}_{selected_id}",
                width="stretch",
            ):
                _run_mail_action(callback, language)


def _render_message_detail(
    service: MailService,
    selected_id: int | None,
    language: str,
) -> None:
    st.markdown(f"### {t('message_detail', language)}")
    if selected_id is None:
        st.info(t("mailbox_empty", language))
        return
    payload = service.open_email(selected_id)
    message = payload.get("email")
    if not isinstance(message, Mapping):
        st.session_state["mailbox_selected_id"] = None
        st.info(t("mailbox_empty", language))
        return

    analysis = payload.get("analysis")
    prediction = (
        str(analysis.get("prediction", ""))
        if isinstance(analysis, Mapping)
        else ""
    )
    if prediction == "phishing":
        st.error(
            f"**{t('phishing_banner_title', language)}**\n\n"
            f"{t('phishing_banner_body', language)}",
            icon="🚨",
        )
    elif prediction == "spam":
        st.warning(t("spam_banner", language), icon="📨")

    st.markdown(
        f"**{t('mail_from', language)}:** {escape(str(message.get('sender') or t('unknown_sender', language)))}  \n"
        f"**{t('mail_to', language)}:** {escape(str(message.get('receiver') or '—'))}  \n"
        f"**{t('mail_sent', language)}:** {escape(display_timestamp(message.get('sent_at')))}  \n"
        f"**{t('mail_header_subject', language)}:** "
        f"{escape(str(message.get('subject') or t('no_subject', language)))}"
    )
    if message.get("cc"):
        st.caption(f"{t('mail_cc', language)}: {escape(str(message['cc']))}")
    st.code(str(message.get("body") or ""), language=None, wrap_lines=True)
    _render_action_buttons(service, message, selected_id, language)

    st.divider()
    _render_analysis(payload.get("analysis"), payload.get("routing"), language)
    with st.expander(t("folder_history", language)):
        history = payload.get("folder_history", [])
        if not history:
            st.caption(t("folder_history_empty", language))
        for event in history:
            st.markdown(
                f"- `{event['previous_folder']}` → `{event['new_folder']}` · "
                f"{event['reason']} · {display_timestamp(event['changed_at'])}"
            )


def render() -> None:
    """Render the mailbox without invoking inference for stored messages."""

    language = _language()
    view = _view_from_query()
    previous_view = st.session_state.get("mailbox_current_view")
    if previous_view != view:
        st.session_state["mailbox_selected_id"] = None
        st.session_state["mailbox_current_view"] = view

    st.markdown(
        '<div class="vmg-mail-header">'
        f'<div><div class="vmg-eyebrow">AI-POWERED BILINGUAL EMAIL SECURITY CLIENT</div>'
        f'<h1>{escape(t("mail_client_title", language))}</h1>'
        f'<p>{escape(t("mail_client_intro", language))}</p></div></div>',
        unsafe_allow_html=True,
    )
    st.caption(t("demo_data_notice", language))

    try:
        service = MailService.from_database_path(DEFAULT_DATABASE_PATH)
    except Exception:  # pragma: no cover - environment-dependent UI boundary
        st.error(t("mailbox_database_error", language))
        return

    try:
        flash = st.session_state.pop("mailbox_flash", None)
        if isinstance(flash, tuple) and len(flash) == 2:
            getattr(st, str(flash[0]), st.info)(str(flash[1]))

        import_column, search_column = st.columns([1, 2.2], vertical_alignment="bottom")
        with import_column:
            uploaded = st.file_uploader(
                t("mail_import", language),
                type=["eml"],
                help=t("mail_import_help", language),
                key="mailbox_eml_upload",
            )
            if st.button(
                t("mail_import", language),
                disabled=uploaded is None,
                key="mailbox_import_button",
                width="stretch",
            ):
                try:
                    with st.spinner(t("mail_importing", language)):
                        imported = service.import_eml_bytes(uploaded.getvalue())
                    st.session_state["mailbox_selected_id"] = imported["email"]["id"]
                    message = t(
                        "mail_import_duplicate" if imported["duplicate"] else "mail_import_success",
                        language,
                    )
                    st.session_state["mailbox_flash"] = (
                        "info" if imported["duplicate"] else "success",
                        message,
                    )
                    st.rerun()
                except Exception:  # pragma: no cover - final UI boundary
                    st.error(t("mail_import_error", language))
        with search_column:
            query = st.text_input(
                t("mail_search", language),
                placeholder=t("mail_search_placeholder", language),
                key="mailbox_search_query",
            )

        filters, has_active_filter = _render_search_filters(view, language)
        if view == "security":
            counts = service.mailbox_counts()
            st.caption(
                t(
                    "security_summary",
                    language,
                    inbox=int(counts["folders"]["hop_thu_den"]),
                    spam=int(counts["folders"]["thu_rac"]),
                    quarantine=int(counts["folders"]["cach_ly"]),
                    unread=sum(int(value) for value in counts["unread"].values()),
                    high_critical=int(counts["high_critical"]),
                )
            )

        messages = _messages_for_view(service, view, query, filters)
        selected_id = st.session_state.get("mailbox_selected_id")
        available_ids = {int(message["id"]) for message in messages}
        if selected_id is not None and int(selected_id) not in available_ids:
            selected_id = None
            st.session_state["mailbox_selected_id"] = None

        list_column, detail_column = st.columns([0.92, 1.25], gap="medium")
        with list_column:
            st.markdown(f"#### {t(FOLDER_VIEW_KEYS[view], language)}")
            is_search = bool(query.strip()) or has_active_filter
            if view == "security":
                _render_security_message_list(
                    messages,
                    language,
                    is_search=is_search,
                )
            else:
                _render_message_list(
                    service,
                    messages,
                    language,
                    is_search=is_search,
                )
        with detail_column:
            _render_message_detail(
                service,
                None if selected_id is None else int(selected_id),
                language,
            )
    finally:
        service.close()


render()
