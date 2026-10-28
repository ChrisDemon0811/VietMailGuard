"""Interactive email analysis page backed only by the shared inference API."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from i18n import action_label, category_label, class_label, risk_label, t
from mailbox_ui import select_mailbox_message
from runtime_ui import (
    configured_mailbox_path, create_selected_reviewer, draft_input,
    render_reviewer_result, reviewer_controls, reviewer_request_options,
)
from styles import render_page_header
from vietmailguard.eml_parser import parse_eml_bytes
from vietmailguard.mail_service import MailService
from vietmailguard.inference import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RISK_CONFIG_PATH,
    DEFAULT_SECURITY_CONFIG_PATH,
    load_inference_engine,
)
from vietmailguard.v21.runtime_modes import V2_MODE, V21_HYBRID_MODE
from vietmailguard.v21.runtime_inference import load_v21_engine
from vietmailguard.v21.llm_reviewer import review_email

_RESULT_STATE_KEY = "vietmailguard_analysis_result"


class _EmlParseError(ValueError):
    """The uploaded message cannot provide a usable parsed email."""


def _artifact_signature() -> tuple[tuple[str, int, int], ...]:
    paths = (
        DEFAULT_MODEL_PATH,
        DEFAULT_METADATA_PATH,
        DEFAULT_RISK_CONFIG_PATH,
        DEFAULT_SECURITY_CONFIG_PATH,
    )
    return tuple(
        (str(Path(path).resolve()), Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
        for path in paths
    )


@st.cache_resource(show_spinner=False)
def _cached_production_engine(
    artifact_signature: tuple[tuple[str, int, int], ...],
) -> Any:
    """Cache the shared engine while invalidating when an artifact changes."""
    del artifact_signature
    return load_inference_engine()


def _production_engine() -> Any:
    return _cached_production_engine(_artifact_signature())


def _selected_engine() -> Any:
    if st.session_state.get("analysis_mode", V2_MODE) == V2_MODE:
        return _production_engine()
    return load_v21_engine()


def _language() -> str:
    return str(st.session_state.get("interface_language", "en"))


def _evidence_text(finding: dict[str, Any]) -> str:
    evidence = finding.get("evidence", [])
    if isinstance(evidence, list):
        return ", ".join(str(value) for value in evidence if str(value).strip())
    return str(evidence or "")


def _submit_analysis(
    sender: str,
    subject: str,
    body: str,
    eml_bytes: bytes | None,
) -> dict[str, Any]:
    parsed: dict[str, str] | None = None
    if eml_bytes is not None:
        try:
            parsed = parse_eml_bytes(eml_bytes)
        except (TypeError, ValueError, UnicodeError, LookupError) as exc:
            raise _EmlParseError from exc
        if not any(str(parsed.get(field, "")).strip() for field in ("sender", "subject", "body")):
            raise _EmlParseError
    # An attached EML is one complete input, not a patch for an older draft.
    # Never analyze a hybrid of yesterday's manual body and today's MIME headers.
    if parsed is not None:
        effective_sender = parsed.get("sender", "")
        effective_subject = parsed.get("subject", "")
        effective_body = parsed.get("body", "")
    else:
        effective_sender, effective_subject, effective_body = sender.strip(), subject.strip(), body.strip()
    result = _selected_engine().analyze_email(
        sender=effective_sender,
        subject=effective_subject,
        body=effective_body,
    )
    return {
        "result": result,
        "parsed_eml": parsed,
        "input_source": "eml" if parsed is not None else "manual",
        "analysis_mode": str(st.session_state.get("analysis_mode", V2_MODE)),
        "email_fields": {
            "sender": effective_sender, "subject": effective_subject, "body": effective_body,
        },
    }


def _render_prediction_summary(result: dict[str, Any], language: str) -> None:
    prediction = str(result["prediction"])
    confidence = result.get("confidence")
    risk_score = int(result["risk_score"])
    risk_level = str(result["risk_level"])
    action = str(result["recommended_action"])

    st.subheader(t("analysis_complete", language))
    support = result.get("language_support_status", {})
    support_status = str(support.get("status", "unknown")) if isinstance(support, dict) else "unknown"
    detected = str(result.get("detected_language", "unknown"))
    metadata_columns = st.columns(3)
    metadata_columns[0].metric(
        t("model_version", language),
        str(result.get("model_name") or result.get("model_version", t("not_available", language))),
    )
    metadata_columns[1].metric(
        t("detected_email_language", language),
        t(f"email_language_{detected}", language),
    )
    metadata_columns[2].metric(
        t("language_support_status", language),
        t(f"support_{support_status}", language),
    )
    st.markdown(
        f'<span class="vmg-status vmg-{escape(prediction)}">'
        f'{escape(class_label(prediction, language).upper())}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    confidence_column, risk_column = st.columns(2, gap="large")
    with confidence_column:
        with st.container(border=True):
            st.markdown(
                f'<p class="vmg-confidence-kicker">{escape(t("model_output_kicker" if confidence is not None else "model_output_uncalibrated_kicker", language))}</p>',
                unsafe_allow_html=True,
            )
            display_confidence = (
                f"{float(confidence):.1%}"
                if confidence is not None
                else t("not_available", language)
            )
            st.metric(t("model_confidence", language), display_confidence)
            if confidence is not None:
                st.progress(float(confidence))
            st.caption(t("model_confidence_help" if confidence is not None
                         else "confidence_unavailable", language))
    with risk_column:
        with st.container(border=True):
            st.markdown(
                f'<p class="vmg-risk-kicker">{escape(t("risk_output_kicker", language))}</p>',
                unsafe_allow_html=True,
            )
            st.metric(t("risk_score", language), f"{risk_score}/100")
            st.progress(risk_score)
            st.caption(t("risk_score_help" if confidence is not None else "risk_score_uncalibrated_help", language))

    risk_column, action_column = st.columns(2, gap="large")
    risk_column.metric(t("risk_level", language), risk_label(risk_level, language))
    action_column.metric(t("recommended_action", language), action_label(action, language))
    st.caption(t("recommendation_note", language))

    if action != str(result.get("base_recommended_action", action)):
        st.warning(t("prediction_security_disagreement", language))
        st.caption(t("classifier_prediction_unchanged", language))

    probabilities = result.get("class_probabilities", {})
    if isinstance(probabilities, dict) and probabilities:
        probability_rows = [
            {
                t("class", language): class_label(str(label), language),
                t("probability", language): float(probability),
            }
            for label, probability in probabilities.items()
        ]
        with st.expander(t("class_probabilities", language)):
            st.dataframe(
                pd.DataFrame(probability_rows).style.format(
                    {t("probability", language): "{:.2%}"}
                ),
                hide_index=True,
                width="stretch",
            )


def _render_security_findings(result: dict[str, Any], language: str) -> None:
    content_findings = result.get("content_findings", [])
    sender_findings = result.get("sender_findings", [])
    if not content_findings and not sender_findings:
        st.info(t("no_security_findings", language))
        return

    content_column, sender_column = st.columns(2, gap="large")
    with content_column:
        st.markdown(f"**{t('content_findings', language)}**")
        if not content_findings:
            st.caption(t("not_available", language))
        for finding in content_findings:
            category = category_label(str(finding.get("code", "")), language)
            evidence = _evidence_text(finding) or t("not_available", language)
            st.markdown(
                f'<div class="vmg-signal"><strong>{escape(category)}</strong><br>'
                f'{escape(evidence)}</div>',
                unsafe_allow_html=True,
            )
    with sender_column:
        st.markdown(f"**{t('sender_findings', language)}**")
        if not sender_findings:
            st.caption(t("not_available", language))
        for finding in sender_findings:
            category = category_label(str(finding.get("code", "")), language)
            evidence = _evidence_text(finding) or t("not_available", language)
            st.markdown(
                f'<div class="vmg-signal"><strong>{escape(category)}</strong><br>'
                f'{escape(evidence)}</div>',
                unsafe_allow_html=True,
            )
    st.caption(t("suspicious_indicator_note", language))


def _render_explanation(result: dict[str, Any], language: str) -> None:
    reasons = result.get("reasons", [])
    model_reasons = [
        reason
        for reason in reasons
        if isinstance(reason, dict) and reason.get("source") == "model"
    ]
    if not model_reasons:
        st.info(t("no_model_explanation", language))
    for reason in model_reasons:
        feature = str(reason.get("feature", ""))
        contribution = float(reason.get("contribution", 0.0))
        sentence = t(
            "model_feature_reason",
            language,
            feature=feature,
            label=class_label(str(result["prediction"]), language),
            contribution=f"{contribution:.4f}",
        )
        st.markdown(f"- {sentence}")


def _render_urls(result: dict[str, Any], language: str) -> None:
    analyses = result.get("url_findings", [])
    if not analyses:
        st.info(t("no_urls", language))
        return
    for number, analysis in enumerate(analyses, start=1):
        with st.expander(t("url_number", language, number=number), expanded=number == 1):
            st.code(str(analysis.get("url", "")), language=None)
            details = pd.DataFrame(
                [
                    {"field": t("scheme", language), "value": str(analysis.get("scheme", ""))},
                    {"field": t("hostname", language), "value": str(analysis.get("hostname", ""))},
                    {
                        "field": t("registered_domain", language),
                        "value": str(analysis.get("registered_domain", "")),
                    },
                    {
                        "field": t("https", language),
                        "value": t("yes", language)
                        if analysis.get("is_https")
                        else t("no", language),
                    },
                    {
                        "field": t("ip_host", language),
                        "value": t("yes", language)
                        if analysis.get("is_ip_host")
                        else t("no", language),
                    },
                    {
                        "field": t("url_length", language),
                        "value": str(analysis.get("length", 0)),
                    },
                    {
                        "field": t("subdomain_depth", language),
                        "value": str(analysis.get("subdomain_depth", 0)),
                    },
                    {
                        "field": t("suspicious_tokens", language),
                        "value": ", ".join(analysis.get("suspicious_tokens", [])),
                    },
                    {
                        "field": t("punycode", language),
                        "value": t("yes", language)
                        if analysis.get("has_punycode")
                        else t("no", language),
                    },
                    {
                        "field": t("shortener", language),
                        "value": t("yes", language)
                        if analysis.get("is_shortener")
                        else t("no", language),
                    },
                ]
            ).rename(columns={"field": t("field", language), "value": t("value", language)})
            st.dataframe(details, hide_index=True, width="stretch")
            findings = analysis.get("findings", [])
            if findings:
                for finding in findings:
                    category = category_label(str(finding.get("code", "")), language)
                    st.markdown(f"- **{category}:** {_evidence_text(finding)}")
            else:
                st.success(t("no_url_findings", language))
    st.caption(t("suspicious_indicator_note", language))


def _render_parsed_metadata(payload: dict[str, Any], language: str) -> None:
    parsed = payload.get("parsed_eml")
    if not isinstance(parsed, dict):
        return
    with st.expander(t("parsed_eml_title", language)):
        metadata = pd.DataFrame(
            [
                {"field": t("sender", language), "value": parsed.get("sender", "")},
                {"field": t("receiver", language), "value": parsed.get("receiver", "")},
                {"field": t("date", language), "value": parsed.get("date", "")},
                {"field": t("subject", language), "value": parsed.get("subject", "")},
            ]
        ).rename(columns={"field": t("field", language), "value": t("value", language)})
        st.dataframe(metadata, hide_index=True, width="stretch")


def render() -> None:
    """Render the Email Analyzer page."""
    language = _language()
    is_v2 = st.session_state.get("analysis_mode", V2_MODE) == V2_MODE
    render_page_header(
        t("email_analyzer_title", language),
        t("email_analyzer_intro" if is_v2 else "email_analyzer_v21_intro", language),
    )
    if st.session_state.get("analysis_mode", V2_MODE) == V2_MODE:
        st.info(t("model_scope_notice", language))
    else:
        st.info(t("v21_development_notice", language))
    with st.container(border=True):
        st.markdown(f"**{t('known_limitations_title', language)}**")
        st.markdown(t("known_limitations_compact" if is_v2 else "known_limitations_v21", language))

    st.caption(t("quick_draft_notice", language))
    with st.container(border=True):
        sender = draft_input(
            "quick", "sender",
            t("sender", language), placeholder=t("sender_placeholder", language)
        )
        subject = draft_input(
            "quick", "subject",
            t("subject", language), placeholder=t("subject_placeholder", language)
        )
        body = draft_input(
            "quick", "body",
            t("email_body", language),
            multiline=True,
            placeholder=t("body_placeholder", language),
            height=210,
        )
        uploaded = st.file_uploader(
            t("eml_upload", language), type=["eml"], help=t("eml_help", language),
            key="quick_eml_upload",
        )
        if uploaded is not None:
            st.caption(t("eml_selected_input", language))
        submitted = st.button(
            t("analyze_email", language), type="primary", width="stretch",
            key="quick_analyze_button",
        )

    if submitted:
        # A failed new submission must not display a previous mail's prediction.
        st.session_state.pop(_RESULT_STATE_KEY, None)
        st.session_state.pop("vietmailguard_llm_review", None)
        st.session_state["quick_llm_consent"] = False
        try:
            with st.spinner(t("analyzing", language)):
                payload = _submit_analysis(
                    sender,
                    subject,
                    body,
                    uploaded.getvalue() if uploaded is not None else None,
                )
            st.session_state[_RESULT_STATE_KEY] = payload
        except _EmlParseError:
            st.error(t("eml_parse_error", language))
        except ValueError:
            st.error(t("empty_input_error", language))
        except (FileNotFoundError, OSError):
            st.error(t("production_artifact_error", language))
        except Exception:  # pragma: no cover - final UI boundary
            st.error(t("inference_error", language))

    payload = st.session_state.get(_RESULT_STATE_KEY)
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        return
    if payload.get("analysis_mode", V2_MODE) != st.session_state.get("analysis_mode", V2_MODE):
        st.info(t("analyze_email", language))
        return
    _render_parsed_metadata(payload, language)
    result = payload["result"]
    _render_prediction_summary(result, language)
    st.caption(t("quick_analyzed_input", language))
    save_column, open_column = st.columns(2)
    with save_column:
        if st.button(t("quick_save_to_mailbox", language), key="quick_save_to_mailbox"):
            try:
                fields = payload["email_fields"]
                parsed = payload.get("parsed_eml") or {}
                with MailService.from_database_path(configured_mailbox_path()) as service:
                    saved = service.save_analyzed_email(
                        analysis_result=result, sender=fields["sender"],
                        subject=fields["subject"], body=fields["body"],
                        receiver=parsed.get("receiver", ""),
                        sent_at=parsed.get("date") or None,
                        llm_review=st.session_state.get("vietmailguard_llm_review"),
                    )
                payload["saved_email_id"] = saved["email"]["id"]
                payload["saved_folder"] = saved["email"]["folder"]
                st.success(t("quick_saved", language))
            except Exception:
                st.error(t("quick_save_error", language))
    with open_column:
        if payload.get("saved_email_id") and st.button(
            t("quick_open_saved", language), key="quick_open_saved",
        ):
            folder = payload["saved_folder"]
            select_mailbox_message(st.session_state, payload["saved_email_id"], folder)
            st.switch_page("views/mail_client.py", query_params={"view": folder, "mode": "detail"})
    st.divider()
    st.subheader(t("why_title", language))
    st.markdown(f"### A. {t('model_explanation_section', language)}")
    _render_explanation(result, language)
    st.markdown(f"### B. {t('security_findings', language)}")
    _render_security_findings(result, language)
    st.markdown(f"### C. {t('url_findings_section', language)}")
    _render_urls(result, language)
    if st.session_state.get("analysis_mode", V2_MODE) == V21_HYBRID_MODE:
        st.divider()
        st.subheader(t("llm_reviewer_title", language))
        can_review, external = reviewer_controls(language, consent_key="quick_llm_consent")
        if st.button(t("llm_reviewer_run" if external else "reviewer_mock_run", language), disabled=not can_review,
                     key="quick_llm_review_button"):
            try:
                fields = payload["email_fields"]
                provider = create_selected_reviewer()
                with st.spinner(t("reviewer_running", language)):
                    if payload.get("saved_email_id"):
                        with MailService.from_database_path(configured_mailbox_path()) as service:
                            reviewed = service.review_stored_email(
                                payload["saved_email_id"], provider=provider,
                                allow_external=external,
                                explicit_email_sharing_consent=bool(external and can_review),
                                **reviewer_request_options(),
                            )["result"]
                    else:
                        reviewed = review_email(
                            sender=fields["sender"], subject=fields["subject"], body=fields["body"],
                            ml_result=result, provider=provider,
                            allow_external=external,
                            explicit_email_sharing_consent=bool(external and can_review),
                            **reviewer_request_options(),
                        )
                st.session_state["vietmailguard_llm_review"] = reviewed
            except Exception:
                st.error(t("llm_reviewer_error", language))
        reviewed = st.session_state.get("vietmailguard_llm_review")
        if isinstance(reviewed, dict):
            render_reviewer_result(reviewed, language)


render()
