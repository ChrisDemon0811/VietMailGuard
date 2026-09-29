"""Full-width list/detail local mailbox UI backed exclusively by MailService."""

from __future__ import annotations

from html import escape
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from i18n import action_label, category_label, class_label, risk_label, t
from mailbox_ui import (
    FOLDER_VIEW_KEYS,
    display_timestamp,
    message_preview,
    resolve_mailbox_mode,
    resolve_mailbox_view,
    return_to_mailbox_list,
    select_mailbox_message,
    sync_mailbox_navigation,
)
from styles import apply_mailbox_styles
from vietmailguard.mail_database import DEFAULT_DATABASE_PATH
from vietmailguard.mail_service import MailService


FOLDER_VIEWS = {"hop_thu_den", "thu_rac", "cach_ly", "da_xoa"}


def _language() -> str:
    return str(st.session_state.get("interface_language", "en"))


def _view_from_query() -> str:
    return resolve_mailbox_view(
        st.query_params.get("view"),
        st.session_state,
    )


def _mode_from_query() -> str:
    return resolve_mailbox_mode(
        st.query_params.get("mode"),
        st.session_state,
    )


def _open_detail(email_id: int, view: str) -> None:
    select_mailbox_message(st.session_state, email_id, view)
    st.query_params.update({"view": view, "mode": "detail"})
    st.rerun()


def _back_to_list() -> None:
    view = return_to_mailbox_list(st.session_state)
    st.query_params.update({"view": view, "mode": "list"})
    st.rerun()


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

    with st.container():
        read_choice = st.selectbox(
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
            starred_choice = st.selectbox(
                t("filter_starred_state", language),
                options=["starred"],
                format_func=lambda _: t("folder_starred", language),
                disabled=True,
                key=f"mail_filter_starred_{view}",
            )
        else:
            starred_choice = st.selectbox(
                t("filter_starred_state", language),
                options=["all", "starred", "unstarred"],
                format_func=lambda value: {
                    "all": t("filter_all", language),
                    "starred": t("folder_starred", language),
                    "unstarred": t("unstarred", language),
                }[value],
                key=f"mail_filter_starred_{view}",
            )
        prediction = st.selectbox(
            t("filter_prediction", language),
            options=["", "normal", "spam", "phishing"],
            format_func=lambda value: (
                t("filter_all", language)
                if not value
                else class_label(value, language)
            ),
            key=f"mail_filter_prediction_{view}",
        )
        risk_level = st.selectbox(
            t("filter_risk_level", language),
            options=["", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            format_func=lambda value: (
                t("filter_all", language)
                if not value
                else risk_label(value, language)
            ),
            key=f"mail_filter_risk_{view}",
        )
        email_language = st.selectbox(
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
        if not (key == "starred" and view == "gan_sao")
    )
    return values, active


def _prediction_badge(analysis: Mapping[str, Any] | None, language: str) -> str:
    if not analysis:
        return ""
    prediction = str(analysis.get("prediction", ""))
    label = escape(class_label(prediction, language).upper())
    return f'<span class="vmg-mail-badge vmg-{escape(prediction)}">{label}</span>'


def _mail_list_heading(message: Mapping[str, Any], language: str) -> str:
    """Keep long or malformed headers legible inside a compact open button."""

    sender = message_preview(
        message.get("sender") or t("unknown_sender", language), maximum=42
    )
    subject = message_preview(
        message.get("subject") or t("no_subject", language), maximum=100
    )
    return f"{sender}  ·  {subject}"


def _render_message_list(
    service: MailService,
    messages: list[dict[str, Any]],
    language: str,
    *,
    current_view: str,
    is_search: bool,
) -> None:
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
        read_state = "read" if message.get("is_read") else "unread"
        with st.container(key=f"mail_row_{read_state}_{email_id}"):
            star, identity, meta = st.columns(
                [0.6, 6.6, 2.4], vertical_alignment="center", gap="small"
            )
            with star:
                if st.button(
                    "★" if message.get("is_starred") else "☆",
                    key=f"mail_star_{email_id}",
                    help=t(
                        "remove_star" if message.get("is_starred") else "add_star",
                        language,
                    ),
                    type="tertiary",
                ):
                    service.toggle_star(email_id)
                    st.rerun()
            with identity:
                if st.button(
                    f"{'●  ' if not message.get('is_read') else ''}"
                    f"{_mail_list_heading(message, language)}",
                    key=f"mail_select_{email_id}",
                    type="tertiary",
                    width="stretch",
                    help=t("open_email", language),
                ):
                    _open_detail(email_id, current_view)
                st.markdown(
                    f'<div class="vmg-mail-preview">'
                    f'{escape(message_preview(message.get("body"), maximum=130))}'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with meta:
                st.caption(
                    display_timestamp(message.get("sent_at") or message.get("created_at"))
                )
                if isinstance(analysis, Mapping) and str(
                    analysis.get("prediction")
                ) in {"spam", "phishing"}:
                    st.markdown(
                        _prediction_badge(analysis, language),
                        unsafe_allow_html=True,
                    )
                elif isinstance(analysis, Mapping) and analysis.get("has_warning"):
                    st.caption(t("mail_security_notice", language))


def _render_security_message_list(
    messages: list[dict[str, Any]],
    language: str,
    *,
    is_search: bool,
) -> None:
    """Render stored security results ordered by repository risk score."""

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
        read_state = "read" if message.get("is_read") else "unread"
        with st.container(key=f"mail_row_{read_state}_{email_id}"):
            identity, meta = st.columns([7.2, 2.4], vertical_alignment="center")
            with identity:
                if st.button(
                    f"{'●  ' if not message.get('is_read') else ''}"
                    f"{_mail_list_heading(message, language)}",
                    key=f"security_select_{email_id}",
                    type="tertiary",
                    width="stretch",
                    help=t("open_email", language),
                ):
                    _open_detail(email_id, "security")
                st.markdown(
                    f'<div class="vmg-mail-preview">'
                    f'{escape(message_preview(message.get("body"), maximum=130))}'
                    '</div>',
                    unsafe_allow_html=True,
                )
                folder = str(message.get("folder", "hop_thu_den"))
                folder_label = t(FOLDER_VIEW_KEYS.get(folder, "folder_inbox"), language)
                st.caption(
                    f"{folder_label} · {t('model_confidence', language)}: "
                    f"{float(analysis.get('confidence', 0.0)):.1%} · "
                    f"{t('security_finding_count', language)}: "
                    f"{int(analysis.get('security_finding_count', 0))}"
                )
            with meta:
                st.markdown(_prediction_badge(analysis, language), unsafe_allow_html=True)
                st.caption(
                    f"{t('risk_score', language)}: "
                    f"{float(analysis.get('risk_score', 0.0)):.0f}/100"
                )
                st.caption(display_timestamp(message.get("sent_at") or message.get("created_at")))


def _run_mail_action(
    action: Any,
    language: str,
    *,
    return_to_list: bool = False,
) -> None:
    try:
        action()
        st.session_state["mailbox_flash"] = ("success", t("mail_action_success", language))
        if return_to_list:
            view = return_to_mailbox_list(st.session_state)
            st.query_params.update({"view": view, "mode": "list"})
        st.rerun()
    except Exception:  # pragma: no cover - final UI boundary
        st.error(t("mail_action_error", language))


def _stored_number(value: object) -> float | None:
    """Format saved numeric evidence without deriving or changing a score."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _render_stored_finding(finding: Mapping[str, Any], language: str) -> None:
    """Display only the rule description and evidence actually saved for this mail."""

    code = str(finding.get("code") or "indicator")
    category = category_label(code, language)
    evidence = finding.get("evidence", [])
    if isinstance(evidence, (list, tuple)):
        evidence_text = ", ".join(str(item) for item in evidence if str(item).strip())
    else:
        evidence_text = str(evidence or "").strip()
    description = str(finding.get("description") or "").strip()
    lines = [f"<strong>{escape(category)}</strong>"]
    if evidence_text:
        lines.append(f"{escape(t('evidence', language))}: {escape(evidence_text)}")
    if description:
        lines.append(escape(description))
    st.markdown(
        '<div class="vmg-stored-finding">' + "<br>".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


def _render_model_risk_details(
    analysis: Mapping[str, Any],
    result: Mapping[str, Any],
    language: str,
) -> None:
    """Show the saved probability distribution and risk breakdown, if present."""

    probabilities = result.get("class_probabilities")
    st.markdown(f"**{t('class_probabilities', language)}**")
    probability_lines = []
    if isinstance(probabilities, Mapping):
        for label in ("normal", "spam", "phishing"):
            probability = _stored_number(probabilities.get(label))
            if probability is not None and 0 <= probability <= 1:
                probability_lines.append(
                    f"- {class_label(label, language)}: **{probability:.2%}**"
                )
    if probability_lines:
        st.markdown("\n".join(probability_lines))
    else:
        st.caption(t("stored_probabilities_unavailable", language))

    st.markdown(f"**{t('stored_risk_components', language)}**")
    components = result.get("risk_components")
    component_lines = []
    if isinstance(components, Mapping):
        for key, label_key in (
            ("ml_probability_and_prediction", "risk_component_ml"),
            ("url_indicators", "risk_component_url"),
            ("content_indicators", "risk_component_content"),
            ("sender_indicators", "risk_component_sender"),
        ):
            amount = _stored_number(components.get(key))
            if amount is not None:
                component_lines.append(f"- {t(label_key, language)}: **{amount:.2f}**")
    if component_lines:
        st.markdown("\n".join(component_lines))
        st.caption(t("stored_risk_components_note", language))
    else:
        st.caption(t("stored_risk_components_unavailable", language))

    reasons = result.get("recommendation_reasons")
    if isinstance(reasons, list) and reasons:
        st.markdown(f"**{t('stored_recommendation_reasons', language)}**")
        for reason in reasons:
            if str(reason).strip():
                st.markdown(
                    f'<div class="vmg-stored-finding">{escape(str(reason))}</div>',
                    unsafe_allow_html=True,
                )

    model_version = str(analysis.get("model_version") or "").strip()
    analyzed_at = str(analysis.get("analyzed_at") or "").strip()
    if model_version:
        st.caption(f"{t('model_version', language)}: {model_version}")
    if analyzed_at:
        st.caption(
            f"{t('stored_analysis_time', language)}: {display_timestamp(analyzed_at)}"
        )
    st.caption(t("confidence_risk_distinction", language))


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
    st.markdown(
        '<div class="vmg-analysis-summary">'
        f'{_prediction_badge(analysis, language)}'
        f'<span>{escape(t("model_confidence", language))}: '
        f'<strong>{float(analysis["confidence"]):.1%}</strong></span>'
        f'<span>{escape(t("risk_score", language))}: '
        f'<strong>{float(analysis["risk_score"]):.0f}/100</strong> '
        f'({escape(risk_label(str(analysis.get("risk_level", "LOW")), language))})</span>'
        f'<span>{escape(t("recommended_action", language))}: '
        f'<strong>{escape(action_label(str(analysis["recommended_action"]), language))}</strong></span>'
        '</div>',
        unsafe_allow_html=True,
    )

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
    with st.expander(t("analysis_model_risk_details", language)):
        _render_model_risk_details(analysis, result, language)
    with st.expander(t("analysis_model_explanation", language)):
        model_explanation = result.get("model_explanation", {})
        features = (
            model_explanation.get("features", [])
            if isinstance(model_explanation, Mapping)
            else []
        )
        if not features:
            st.caption(t("no_model_explanation", language))
        else:
            target = str(model_explanation.get("target_class") or prediction)
            st.caption(
                t("model_feature_target", language, label=class_label(target, language))
            )
        for feature in features[:8]:
            if isinstance(feature, Mapping):
                contribution = _stored_number(feature.get("contribution"))
                if contribution is not None:
                    st.markdown(
                        f'<div class="vmg-stored-finding">'
                        f'<strong>{escape(str(feature.get("feature", "")))}</strong>'
                        f' · {contribution:+.4f}</div>',
                        unsafe_allow_html=True,
                    )
        if isinstance(model_explanation, Mapping) and model_explanation.get("limitation"):
            st.caption(str(model_explanation["limitation"]))

    with st.expander(t("analysis_security_findings", language)):
        content_findings = result.get("content_findings", [])
        if not content_findings:
            st.caption(t("no_security_findings", language))
        for finding in content_findings:
            if isinstance(finding, Mapping):
                _render_stored_finding(finding, language)

    with st.expander(t("analysis_url_findings", language)):
        url_findings = result.get("url_findings", [])
        if not url_findings:
            st.caption(t("no_urls", language))
        elif any(
            isinstance(item, Mapping) and bool(item.get("findings"))
            for item in url_findings
        ):
            st.warning(t("suspicious_link_warning", language))
        for url in url_findings:
            if isinstance(url, Mapping):
                st.code(str(url.get("url", "")), language=None)
                findings = url.get("findings", [])
                if not findings:
                    st.caption(t("no_url_findings", language))
                for finding in findings:
                    if isinstance(finding, Mapping):
                        _render_stored_finding(finding, language)
        if url_findings:
            st.caption(t("suspicious_indicator_note", language))

    with st.expander(t("analysis_sender_findings", language)):
        sender_findings = result.get("sender_findings", [])
        if not sender_findings:
            st.caption(t("no_security_findings", language))
        for finding in sender_findings:
            if isinstance(finding, Mapping):
                _render_stored_finding(finding, language)

    with st.expander(t("analysis_limitations", language)):
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
    actions: list[tuple[str, str, Any, bool]] = [
        (
            "star",
            t("remove_star" if message.get("is_starred") else "add_star", language),
            lambda: service.toggle_star(selected_id),
            False,
        ),
        (
            "read",
            t("mark_unread" if message.get("is_read") else "mark_read", language),
            lambda: service.mark_read(selected_id, not bool(message.get("is_read"))),
            False,
        ),
    ]
    if folder != "hop_thu_den":
        actions.append(
            (
                "inbox",
                t("move_inbox", language),
                lambda: service.move_email(selected_id, "hop_thu_den"),
                True,
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
                True,
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
                True,
            )
        )
    actions.append(
        (
            "report",
            t("report_phishing", language),
            lambda: service.record_feedback(selected_id, action="bao_cao_lua_dao"),
            True,
        )
    )
    if folder != "cach_ly":
        actions.append(
            (
                "quarantine",
                t("move_quarantine", language),
                lambda: service.move_email(selected_id, "cach_ly"),
                True,
            )
        )
    if folder != "da_xoa":
        actions.append(
            (
                "delete",
                t("delete_email", language),
                lambda: service.move_email(selected_id, "da_xoa"),
                True,
            )
        )

    icons = {
        "star": ":material/star:",
        "read": ":material/mark_email_unread:",
        "inbox": ":material/inbox:",
        "not_spam": ":material/move_to_inbox:",
        "spam": ":material/report:",
        "report": ":material/flag:",
        "quarantine": ":material/gpp_bad:",
        "delete": ":material/delete:",
    }
    with st.container(horizontal=True, wrap=True, gap="small"):
        for action_key, label, callback, leaves_detail in actions:
            if st.button(
                label,
                key=f"detail_{action_key}_{selected_id}",
                icon=icons[action_key],
                type="tertiary",
            ):
                _run_mail_action(
                    callback,
                    language,
                    return_to_list=leaves_detail,
                )


def _render_message_detail(
    service: MailService,
    selected_id: int | None,
    language: str,
) -> None:
    return_view = resolve_mailbox_view(
        st.session_state.get("mailbox_return_view"), st.session_state
    )
    if st.button(
        f"← {t('back_to_folder', language, folder=t(FOLDER_VIEW_KEYS[return_view], language))}",
        key="mailbox_back",
        type="tertiary",
    ):
        _back_to_list()
    if selected_id is None:
        st.warning(t("mail_detail_unavailable", language))
        return
    payload = service.open_email(selected_id)
    message = payload.get("email")
    if not isinstance(message, Mapping):
        st.error(t("mail_detail_unavailable", language))
        return

    analysis = payload.get("analysis")
    prediction = (
        str(analysis.get("prediction", ""))
        if isinstance(analysis, Mapping)
        else ""
    )
    folder = str(message.get("folder", ""))
    if folder == "cach_ly" or prediction == "phishing":
        st.error(
            f"**{t('phishing_banner_title' if prediction == 'phishing' else 'quarantine_banner_title', language)}**\n\n"
            f"{t('phishing_banner_body' if prediction == 'phishing' else 'quarantine_banner_body', language)}",
            icon=":material/gpp_bad:",
        )
    elif folder == "thu_rac":
        st.warning(t("spam_banner", language), icon=":material/report:")

    subject_column, badge_column = st.columns([7.0, 2.5], vertical_alignment="center")
    subject_column.markdown(
        f'<h1 class="vmg-detail-subject">'
        f'{escape(str(message.get("subject") or t("no_subject", language)))}</h1>',
        unsafe_allow_html=True,
    )
    if isinstance(analysis, Mapping):
        badge_column.markdown(
            _prediction_badge(analysis, language),
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="vmg-mail-meta">'
        f'<strong>{escape(str(message.get("sender") or t("unknown_sender", language)))}</strong><br>'
        f'{escape(t("mail_to", language))}: {escape(str(message.get("receiver") or "—"))}'
        f' · {escape(display_timestamp(message.get("sent_at")))}'
        '</div>',
        unsafe_allow_html=True,
    )
    if message.get("cc"):
        st.caption(f"{t('mail_cc', language)}: {escape(str(message['cc']))}")
    _render_action_buttons(service, message, selected_id, language)

    with st.container(key="mail_body"):
        st.text(str(message.get("body") or ""))

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


def _render_demo_compose(service: MailService, language: str) -> None:
    """Receive a locally composed demo message through the existing service."""

    with st.container(key="mailbox_demo_compose"):
        st.markdown(f"### {t('mail_demo_compose', language)}")
        st.caption(t("mail_demo_notice", language))
        with st.form("mailbox_demo_compose_form"):
            sender_column, recipient_column = st.columns(2)
            with sender_column:
                sender = st.text_input(t("mail_demo_sender", language))
            with recipient_column:
                recipient = st.text_input(t("mail_demo_recipient", language))
            subject = st.text_input(t("mail_demo_subject", language))
            body = st.text_area(t("mail_demo_body", language), height=170)
            submitted = st.form_submit_button(t("mail_demo_send", language))

        if not submitted:
            return
        if not sender.strip() or not (subject.strip() or body.strip()):
            st.warning(t("mail_demo_required", language))
            return
        try:
            with st.spinner(t("mail_demo_processing", language)):
                received = service.receive_email(
                    sender=sender.strip(),
                    receiver=recipient.strip(),
                    subject=subject,
                    body=body,
                    source="demo_compose",
                )
            email = received["email"]
            routed_folder = str(email["folder"])
            select_mailbox_message(st.session_state, int(email["id"]), routed_folder)
            st.session_state["mailbox_demo_compose_open"] = False
            st.query_params.update({"view": routed_folder, "mode": "detail"})
            folder_label = t(FOLDER_VIEW_KEYS[routed_folder], language)
            st.session_state["mailbox_flash"] = (
                "success",
                f"{t('mail_demo_success', language)} "
                f"{t('mail_routed_to', language, folder=folder_label)}",
            )
            st.rerun()
        except Exception:  # pragma: no cover - final UI boundary
            st.error(t("mail_demo_error", language))


def render() -> None:
    """Render the mailbox without invoking inference for stored messages."""

    apply_mailbox_styles()

    language = _language()
    view = _view_from_query()
    mode = sync_mailbox_navigation(st.session_state, view, _mode_from_query())

    try:
        service = MailService.from_database_path(DEFAULT_DATABASE_PATH)
    except Exception:  # pragma: no cover - environment-dependent UI boundary
        st.error(t("mailbox_database_error", language))
        return

    try:
        flash = st.session_state.pop("mailbox_flash", None)
        if isinstance(flash, tuple) and len(flash) == 2:
            getattr(st, str(flash[0]), st.info)(str(flash[1]))

        if mode == "detail":
            selected_id = st.session_state.get("mailbox_selected_id")
            _render_message_detail(
                service,
                None if selected_id is None else int(selected_id),
                language,
            )
            return

        counts = service.mailbox_counts()
        title_column, refresh_column = st.columns([8.5, 1.0], vertical_alignment="center")
        with title_column:
            st.markdown(f"## {t(FOLDER_VIEW_KEYS[view], language)}")
        with refresh_column:
            if st.button(
                t("refresh_mailbox", language),
                key="mailbox_refresh",
                icon=":material/refresh:",
                type="tertiary",
            ):
                st.rerun()
        if view in FOLDER_VIEWS:
            total = int(counts["folders"][view])
            unread = int(counts["unread"][view])
        elif view == "gan_sao":
            total = int(counts["starred"])
            unread = None
        else:
            total = int(counts["security_total"])
            unread = None
        st.caption(
            t("mailbox_count_unread", language, count=total, unread=unread)
            if unread is not None
            else t("mailbox_count", language, count=total)
        )

        search_column, filter_column, compose_column, import_column = st.columns(
            [4.7, 1.2, 1.7, 1.4], vertical_alignment="bottom", gap="small"
        )
        with search_column:
            query = st.text_input(
                t("mail_search", language),
                placeholder=t("mail_search_placeholder", language),
                key="mailbox_search_query",
                label_visibility="collapsed",
            )
        with filter_column:
            with st.popover(
                t("mail_filters", language),
                icon=":material/filter_list:",
                width="stretch",
            ):
                filters, has_active_filter = _render_search_filters(view, language)
        with compose_column:
            if st.button(
                t("mail_demo_compose", language),
                icon=":material/edit_square:",
                key="mailbox_demo_compose_toggle",
                width="stretch",
            ):
                st.session_state["mailbox_demo_compose_open"] = not bool(
                    st.session_state.get("mailbox_demo_compose_open", False)
                )
        with import_column:
            with st.popover(
                t("mail_import", language),
                icon=":material/upload_file:",
                width="stretch",
            ):
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
                ):
                    try:
                        with st.spinner(t("mail_importing", language)):
                            imported = service.import_eml_bytes(uploaded.getvalue())
                        imported_email = imported["email"]
                        routed_folder = str(imported_email["folder"])
                        select_mailbox_message(
                            st.session_state,
                            int(imported_email["id"]),
                            routed_folder,
                        )
                        st.query_params.update({"view": routed_folder, "mode": "detail"})
                        message = t(
                            "mail_import_duplicate"
                            if imported["duplicate"]
                            else "mail_import_success",
                            language,
                        )
                        folder_label = t(FOLDER_VIEW_KEYS[routed_folder], language)
                        st.session_state["mailbox_flash"] = (
                            "info" if imported["duplicate"] else "success",
                            f"{message} {t('mail_routed_to', language, folder=folder_label)}",
                        )
                        st.rerun()
                    except Exception:  # pragma: no cover - final UI boundary
                        st.error(t("mail_import_error", language))

        if st.session_state.get("mailbox_demo_compose_open", False):
            _render_demo_compose(service, language)

        messages = _messages_for_view(service, view, query, filters)
        is_search = bool(query.strip()) or has_active_filter
        with st.container(key="mail_list"):
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
                    current_view=view,
                    is_search=is_search,
                )
    finally:
        service.close()


render()
