"""Interactive email analysis page backed only by the shared inference API."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from i18n import action_label, category_label, class_label, risk_label, t
from styles import render_page_header
from vietmailguard.eml_parser import parse_eml_bytes
from vietmailguard.inference import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RISK_CONFIG_PATH,
    DEFAULT_SECURITY_CONFIG_PATH,
    load_inference_engine,
)

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
        except (TypeError, ValueError, UnicodeError) as exc:
            raise _EmlParseError from exc
        if not any(str(parsed.get(field, "")).strip() for field in ("sender", "subject", "body")):
            raise _EmlParseError
    effective_sender = sender.strip() or (parsed or {}).get("sender", "")
    effective_subject = subject.strip() or (parsed or {}).get("subject", "")
    effective_body = body.strip() or (parsed or {}).get("body", "")
    result = _production_engine().analyze_email(
        sender=effective_sender,
        subject=effective_subject,
        body=effective_body,
    )
    return {
        "result": result,
        "parsed_eml": parsed,
        "input_source": "eml" if parsed is not None else "manual",
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
                f'<p class="vmg-confidence-kicker">{escape(t("model_output_kicker", language))}</p>',
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
            st.caption(t("model_confidence_help", language))
    with risk_column:
        with st.container(border=True):
            st.markdown(
                f'<p class="vmg-risk-kicker">{escape(t("risk_output_kicker", language))}</p>',
                unsafe_allow_html=True,
            )
            st.metric(t("risk_score", language), f"{risk_score}/100")
            st.progress(risk_score)
            st.caption(t("risk_score_help", language))

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
    render_page_header(
        t("email_analyzer_title", language), t("email_analyzer_intro", language)
    )
    st.info(t("model_scope_notice", language))
    with st.container(border=True):
        st.markdown(f"**{t('known_limitations_title', language)}**")
        st.markdown(t("known_limitations_compact", language))

    with st.form("email_analysis_form", clear_on_submit=False):
        sender = st.text_input(
            t("sender", language), placeholder=t("sender_placeholder", language)
        )
        subject = st.text_input(
            t("subject", language), placeholder=t("subject_placeholder", language)
        )
        body = st.text_area(
            t("email_body", language),
            placeholder=t("body_placeholder", language),
            height=210,
        )
        uploaded = st.file_uploader(
            t("eml_upload", language), type=["eml"], help=t("eml_help", language)
        )
        submitted = st.form_submit_button(
            t("analyze_email", language), type="primary", width="stretch"
        )

    if submitted:
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
    _render_parsed_metadata(payload, language)
    result = payload["result"]
    _render_prediction_summary(result, language)
    st.divider()
    st.subheader(t("why_title", language))
    st.markdown(f"### A. {t('model_explanation_section', language)}")
    _render_explanation(result, language)
    st.markdown(f"### B. {t('security_findings', language)}")
    _render_security_findings(result, language)
    st.markdown(f"### C. {t('url_findings_section', language)}")
    _render_urls(result, language)


render()
