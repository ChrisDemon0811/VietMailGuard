"""Local application wiring; no classifier, SQL, or external requests on render."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

from i18n import t
from vietmailguard.mail_database import DEFAULT_DATABASE_PATH


def configured_mailbox_path() -> Path:
    """Allow a separate local demo database while keeping the existing default."""
    value = os.environ.get("VIETMAILGUARD_MAIL_DB", "").strip()
    return Path(value).expanduser().resolve() if value else DEFAULT_DATABASE_PATH


def _remember_draft(namespace: str, field: str) -> None:
    st.session_state[f"draft_{namespace}_{field}"] = st.session_state.get(
        f"input_{namespace}_{field}", ""
    )


def draft_input(namespace: str, field: str, label: str, *, multiline: bool = False,
                **kwargs: Any) -> str:
    """Keep draft data independent from translated labels/widget cleanup."""
    widget_key = f"input_{namespace}_{field}"
    data_key = f"draft_{namespace}_{field}"
    st.session_state.setdefault(data_key, st.session_state.get(widget_key, ""))
    # Hydrate only when a widget returns after navigation. Reassigning its value
    # on every rerun can erase a newer frontend edit while another field blurs.
    st.session_state.setdefault(widget_key, st.session_state[data_key])
    widget = st.text_area if multiline else st.text_input
    return widget(label, key=widget_key, on_change=_remember_draft,
                  args=(namespace, field), **kwargs)


def selected_reviewer() -> str:
    value = st.session_state.get("reviewer_provider", "mock")
    return value if value in {"mock", "gemini"} else "mock"


def _reset_reviewer_consent() -> None:
    selected = st.session_state.get("reviewer_provider_selector", "mock")
    st.session_state["reviewer_provider"] = selected
    for key in list(st.session_state):
        if key == "quick_llm_consent" or key.startswith("llm_consent_"):
            st.session_state[key] = False


def render_reviewer_selector(language: str) -> None:
    st.session_state.setdefault("reviewer_provider", "mock")
    st.session_state["reviewer_provider_selector"] = selected_reviewer()
    st.selectbox(
        t("reviewer_provider", language), options=["mock", "gemini"],
        format_func=lambda value: t(f"reviewer_provider_{value}", language),
        key="reviewer_provider_selector", on_change=_reset_reviewer_consent,
        help=t("reviewer_provider_help", language),
    )
    if selected_reviewer() == "mock":
        st.caption(t("reviewer_mock_notice", language))
    else:
        from vietmailguard.v21.gemini_reviewer import configuration_status

        status = configuration_status()
        st.caption(f"{t('reviewer_model', language)}: {status['model'] or t('not_available', language)}")
        st.caption(t("reviewer_access_not_checked", language))


def reviewer_controls(language: str, *, consent_key: str) -> tuple[bool, bool]:
    """Disclose external data flow before consent; local mock is explicit."""
    if selected_reviewer() == "mock":
        st.info(t("reviewer_mock_notice", language))
        return True, False
    from vietmailguard.v21.gemini_reviewer import configuration_status

    st.warning(t("reviewer_external_disclosure", language))
    status = configuration_status()
    compatible = all(key in status for key in (
        "review_timeout_seconds", "review_timeout_valid", "model_access_status"))
    ready = bool(compatible and status["api_key_configured"] and status["sdk_version"]
                 and status["model_syntax_valid"] and not status["configuration_error"])
    st.caption(f"{t('reviewer_model', language)}: {status['model'] or t('not_available', language)}")
    st.caption(t("reviewer_access_not_checked", language))
    if ready:
        from vietmailguard.v21.llm_reviewer import MAX_EMAIL_FIELD_CHARS

        st.caption(t("reviewer_request_policy", language).format(
            seconds=status["review_timeout_seconds"], characters=MAX_EMAIL_FIELD_CHARS))
    consent = st.checkbox(t("llm_reviewer_consent", language), key=consent_key)
    if not compatible:
        st.warning(t("reviewer_restart_required", language))
    elif not ready:
        st.warning(t("mode_hybrid_unavailable", language))
    return bool(ready and consent), True


def reviewer_request_options() -> dict[str, float | int]:
    """No hidden resend: each explicit UI review click permits one attempt."""
    if selected_reviewer() == "gemini":
        from vietmailguard.v21.gemini_reviewer import gemini_review_options

        return gemini_review_options()
    return {"timeout_seconds": 5.0, "max_retries": 0}


def create_selected_reviewer() -> Any:
    """Construct a provider only inside an explicit review button handler."""
    if selected_reviewer() == "mock":
        from vietmailguard.v21.llm_reviewer import MockReviewerProvider

        return MockReviewerProvider()
    from vietmailguard.v21.gemini_reviewer import GeminiReviewerProvider

    return GeminiReviewerProvider()


def render_reviewer_result(reviewed: dict[str, Any], language: str) -> None:
    reviewer = reviewed.get("reviewer", {})
    if not isinstance(reviewer, dict):
        st.warning(t("llm_reviewer_error", language))
        return
    provider = str(reviewer.get("provider", ""))
    st.caption(f"{t('reviewer_provider', language)}: {provider}")
    if provider == "mock":
        st.info(t("reviewer_mock_notice", language))
    model = reviewer.get("model")
    if isinstance(model, str) and re.fullmatch(r"gemini-[a-z0-9][a-z0-9._-]{0,119}", model):
        st.caption(f"{t('reviewer_model', language)}: {model}")
    prompt_version = reviewer.get("prompt_version")
    if isinstance(prompt_version, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", prompt_version):
        st.caption(f"{t('reviewer_prompt_version', language)}: {prompt_version}")
    elif provider == "gemini":
        st.caption(t("reviewer_legacy_prompt_notice", language))
    assessment = reviewer.get("assessment")
    assessment_text = (t("reviewer_assessment_phishing", language) if assessment == "phishing"
                       else t(f"class_{assessment}", language)
                       if assessment in {"normal", "spam"}
                       else "REVIEW" if assessment == "review" else t("not_available", language))
    st.write(f"{t('llm_reviewer_assessment', language)}: {assessment_text}")
    status_text = (t("reviewer_status_ok", language) if reviewer.get("status") == "ok"
                   else reviewer.get("status") or t("not_available", language))
    st.write(f"{t('llm_reviewer_status', language)}: {status_text}")
    if provider == "gemini":
        st.caption(t("reviewer_content_only_notice", language))
    _render_reviewer_failure(reviewer, language)
    if reviewer.get("summary"):
        st.text(str(reviewer["summary"]))
    for evidence in reviewer.get("evidence", []):
        if isinstance(evidence, dict):
            st.code(str(evidence.get("quote", "")), language=None)
            st.text(str(evidence.get("explanation", "")))
    for indicator in reviewer.get("suspicious_indicators", []):
        st.text(str(indicator))
    if reviewer.get("input_truncated"):
        st.warning(t("reviewer_input_truncated", language))
    if reviewer.get("human_review_recommended"):
        st.warning(t("llm_reviewer_disagreement" if reviewer.get("agrees_with_ml") is False
                     else "llm_reviewer_review", language))


def _render_reviewer_failure(reviewer: dict[str, Any], language: str) -> None:
    """Only render known diagnostic codes, never raw SDK exceptions/messages."""
    status = reviewer.get("status")
    if status in {"ok", "mock_only", "disabled", "empty_input"}:
        return
    detail = reviewer.get("diagnostic", {})
    category = detail.get("category") if isinstance(detail, dict) else None
    allowed = {
        "timeout", "connection", "rate_or_quota", "authentication", "permission",
        "model_or_endpoint_not_found", "provider_service", "api_request",
        "schema_or_configuration", "configuration", "unknown",
    }
    if status == "invalid_response":
        message = "reviewer_error_invalid_response"
    elif status == "deadline_exceeded":
        message = "reviewer_error_timeout"
    else:
        message = f"reviewer_error_{category}" if category in allowed else "reviewer_error_unknown"
    st.error(t(message, language))
    http = detail.get("http_status") if isinstance(detail, dict) else None
    if isinstance(http, int) and not isinstance(http, bool) and 100 <= http <= 599:
        st.caption(f"HTTP {http}")
    st.caption(t("reviewer_retry_notice", language))
