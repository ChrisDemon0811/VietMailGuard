"""Optional Google Gemini transport for the independent V2.1 email reviewer.

Importing this module does not load credentials, construct a client, or call an
API. The public review gate in ``llm_reviewer.review_email`` must run first.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from .llm_reviewer import (
    REVIEW_RESPONSE_SCHEMA,
    ProviderTimeoutError,
    ReviewRequest,
    ReviewerConfigurationError,
    ReviewerPrivacyError,
    RetryableProviderError,
    build_review_messages,
    review_policy_metadata,
    validated_provider_diagnostic,
)


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_MODEL_ID = re.compile(r"^gemini-[a-z0-9][a-z0-9._-]*$")
_LISTED_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,119}$")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DOTENV_PATH = _PROJECT_ROOT / ".env"


def project_environment(override: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """Merge root .env and process environment without changing os.environ."""
    if override is not None:
        return override
    if _DOTENV_PATH.is_file():
        try:
            from dotenv import dotenv_values
        except ImportError as error:
            if not os.environ.get("GEMINI_API_KEY"):
                raise ReviewerConfigurationError(
                    "python-dotenv is required to load the project .env"
                ) from error
        else:
            merged = {key: value for key, value in dotenv_values(_DOTENV_PATH).items()
                      if value is not None}
            merged.update(os.environ)
            return merged
    return os.environ


def gemini_model(environ: Mapping[str, str] | None = None) -> str:
    model = project_environment(environ).get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    if not _MODEL_ID.fullmatch(model):
        raise ReviewerConfigurationError("GEMINI_MODEL must be a Gemini text model ID")
    return model


def _api_key(environ: Mapping[str, str] | None = None) -> str:
    key = project_environment(environ).get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ReviewerConfigurationError("GEMINI_API_KEY is not configured")
    return key


def _sdk_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def gemini_review_options(environ: Mapping[str, str] | None = None) -> dict[str, float | int]:
    """Explicit interactive limits; no automatic replay of potentially billed email.

    Timeout controls the caller's wait, not billing cancellation. A timed-out
    remote request may still consume quota. Retrying must be a separate user
    action rather than an invisible application retry.
    """
    raw_timeout = project_environment(environ).get("GEMINI_REVIEW_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as error:
        raise ReviewerConfigurationError(
            "GEMINI_REVIEW_TIMEOUT_SECONDS must be a number greater than 0 and at most 60"
        ) from error
    if not 0 < timeout <= 60:
        raise ReviewerConfigurationError(
            "GEMINI_REVIEW_TIMEOUT_SECONDS must be a number greater than 0 and at most 60")
    return {"timeout_seconds": timeout, "max_retries": 0}


def configuration_status(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Offline configuration inspection; never return or print the API key."""
    try:
        variables = project_environment(environ)
        env_error = None
    except ReviewerConfigurationError:
        variables = os.environ if environ is None else environ
        env_error = "python-dotenv is unavailable for the project .env"
    try:
        model = gemini_model(variables)
        model_valid = True
    except ReviewerConfigurationError:
        model = None
        model_valid = False
    try:
        review_options = gemini_review_options(variables)
        review_timeout = review_options["timeout_seconds"]
        timeout_valid = True
    except ReviewerConfigurationError:
        review_timeout = None
        timeout_valid = False
        env_error = ("GEMINI_REVIEW_TIMEOUT_SECONDS must be a number greater than 0 "
                     "and at most 60")
    return {
        "dotenv_path": str(_DOTENV_PATH),
        "dotenv_exists": _DOTENV_PATH.is_file(),
        "dotenv_version": _sdk_version("python-dotenv"),
        "sdk_version": _sdk_version("google-genai"),
        "api_key_configured": bool(variables.get("GEMINI_API_KEY", "").strip()),
        "model": model,
        "model_syntax_valid": model_valid,
        "model_access_status": "not_checked_offline",
        "review_timeout_seconds": review_timeout,
        "review_timeout_valid": timeout_valid,
        "configuration_error": env_error,
        "network_called": False,
    }


def _provider_schema(value: Any) -> Any:
    """Remove length keywords unsupported by Gemini; local validation keeps them."""
    if isinstance(value, dict):
        return {key: _provider_schema(item) for key, item in value.items()
                if key not in {"minLength", "maxLength"}}
    if isinstance(value, list):
        return [_provider_schema(item) for item in value]
    return value


def _client(api_key: str, timeout_seconds: float) -> Any:
    try:
        from google import genai
    except ImportError as error:
        raise ReviewerConfigurationError(
            "google-genai is not installed; install requirements-gemini.txt"
        ) from error
    # The SDK timeout is milliseconds; disable its internal retries so that
    # review_email's bounded retry policy remains the only retry authority.
    return genai.Client(
        api_key=api_key,
        http_options={"timeout": max(100, int(timeout_seconds * 800)),
                      "retry_options": {"attempts": 1}},
    )


def _is_timeout(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    try:
        from httpx import TimeoutException
    except ImportError:
        return False
    return isinstance(error, TimeoutException)


def safe_gemini_diagnostic(error: Exception) -> dict[str, str | int | None]:
    """Expose only allowlisted exception metadata; never its message or request."""
    preserved = getattr(error, "gemini_diagnostic", None)
    if isinstance(preserved, dict):
        return validated_provider_diagnostic(preserved)
    kind = type(error).__name__
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", kind):
        kind = "UnknownError"
    code = getattr(error, "code", None)
    if not isinstance(code, int) or isinstance(code, bool) or not 100 <= code <= 599:
        code = getattr(getattr(error, "response", None), "status_code", None)
    http_status = code if isinstance(code, int) and not isinstance(code, bool) and 100 <= code <= 599 else None
    status = getattr(error, "status", None)
    api_code = status if isinstance(status, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", status) else None
    if _is_timeout(error) or isinstance(error, ProviderTimeoutError):
        category = "timeout"
    elif isinstance(error, ConnectionError):
        category = "connection"
    else:
        try:
            from httpx import ConnectError, NetworkError
            is_network = isinstance(error, (ConnectError, NetworkError))
        except ImportError:
            is_network = False
        if is_network:
            category = "connection"
        elif http_status == 429 or api_code in {"RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED", "RATE_LIMIT_EXCEEDED"}:
            category = "rate_or_quota"
        elif http_status == 401:
            category = "authentication"
        elif http_status == 403:
            category = "permission"
        elif http_status == 404:
            category = "model_or_endpoint_not_found"
        elif http_status is not None and 500 <= http_status <= 599:
            category = "provider_service"
        elif http_status is not None:
            category = "api_request"
        elif isinstance(error, ValueError):
            category = "schema_or_configuration"
        else:
            category = "unknown"
    return validated_provider_diagnostic({
        "exception_type": kind, "http_status": http_status,
        "api_code": api_code, "category": category})


def _redacted_error(error: Exception, replacement: Exception) -> Exception:
    replacement.gemini_diagnostic = safe_gemini_diagnostic(error)
    return replacement


def list_generate_content_models(*, environ: Mapping[str, str] | None = None,
                                 client_factory: Callable[[str, float], Any] | None = None
                                 ) -> list[str]:
    """Read model metadata only. Never call generateContent or print the API key."""
    key = _api_key(environ)
    client = (client_factory or _client)(key, 15.0)
    try:
        available: set[str] = set()
        for model in client.models.list():
            if "generateContent" not in (model.supported_actions or []):
                continue
            name = model.name or ""
            if name.startswith("models/"):
                name = name.removeprefix("models/")
            if _LISTED_MODEL_ID.fullmatch(name):
                available.add(name)
        return sorted(available)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _safe_generate(client: Any, *, model: str, contents: str,
                   config: Mapping[str, Any]) -> str:
    """Return response text and replace provider exceptions with redacted ones."""
    try:
        response = client.models.generate_content(
            model=model, contents=contents, config=dict(config)
        )
        return response.text or ""
    except Exception as error:
        if _is_timeout(error):
            raise _redacted_error(error, ProviderTimeoutError("Gemini transport timeout")) from None
        code = getattr(error, "code", None)
        if isinstance(code, int) and (code in {408, 429} or 500 <= code <= 599):
            raise _redacted_error(error, RetryableProviderError("Gemini temporary API failure")) from None
        # SDK error messages may echo request data or credentials.
        raise _redacted_error(error, RuntimeError("Gemini API request failed")) from None


class GeminiReviewerProvider:
    """External adapter; caller must explicitly enable and consent per email."""

    name = "gemini"
    is_external = True

    def __init__(self, *, environ: Mapping[str, str] | None = None,
                 client_factory: Callable[[str, float], Any] | None = None) -> None:
        self._environ = environ
        self._client_factory = client_factory or _client
        self.model_used: str | None = None
        self.prompt_version_used: str | None = None
        self.prompt_sha256_used: str | None = None

    def review(self, request: ReviewRequest, *, timeout_seconds: float) -> str:
        # A provider can be reused. Do not attribute a failed preflight to the
        # model or prompt from an earlier request.
        self.model_used = None
        self.prompt_version_used = None
        self.prompt_sha256_used = None
        if not (request.external_enabled and request.explicit_email_sharing_consent):
            raise ReviewerPrivacyError("External email sharing requires per-call consent")
        model = gemini_model(self._environ)
        key = _api_key(self._environ)
        messages = build_review_messages(request)
        system_prompt = messages[0]["content"]
        policy = review_policy_metadata(system_prompt)
        client = self._client_factory(key, timeout_seconds)
        try:
            self.model_used = model
            self.prompt_version_used = policy["prompt_version"]
            self.prompt_sha256_used = policy["prompt_sha256"]
            return _safe_generate(
                client, model=model, contents=messages[1]["content"],
                config={
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                    "response_json_schema": _provider_schema(REVIEW_RESPONSE_SCHEMA),
                    "tools": [],
                    "temperature": 0,
                },
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def live_connection_check(*, confirm_synthetic_request: bool = False,
                          timeout_seconds: float = 5.0,
                          environ: Mapping[str, str] | None = None,
                          client_factory: Callable[[str, float], Any] | None = None) -> bool:
    """Opt-in, potentially billed synthetic request; never sends email text."""
    if not confirm_synthetic_request:
        raise ReviewerConfigurationError("Live test requires explicit confirmation")
    model = gemini_model(environ)
    key = _api_key(environ)
    client = (client_factory or _client)(key, timeout_seconds)
    try:
        return bool(_safe_generate(client, model=model,
                                   contents="Reply with the single word OK.",
                                   config={"max_output_tokens": 64, "tools": []}).strip())
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
