"""All live-branch checks use fake transports; no network request is made."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from scripts.v21 import check_gemini
from vietmailguard.v21 import gemini_reviewer
from vietmailguard.v21.llm_reviewer import ProviderTimeoutError


STATUS = {
    "dotenv_exists": True,
    "api_key_configured": True,
    "sdk_version": "test-sdk",
    "dotenv_version": "test-dotenv",
    "model": "gemini-2.5-flash",
    "model_syntax_valid": True,
    "configuration_error": None,
    "review_timeout_seconds": 30.0,
    "model_access_status": "not_checked_offline",
}


class SyntheticQuotaError(RuntimeError):
    code = 429
    status = "RESOURCE_EXHAUSTED"


def test_offline_and_unconfirmed_flags_never_create_a_client(monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_gemini, "configuration_status", lambda: STATUS)

    def unexpected_client():
        raise AssertionError("No Gemini transport should be created")

    monkeypatch.setattr(check_gemini, "GeminiTranslator", unexpected_client)
    monkeypatch.setattr(check_gemini, "TrecGeminiSuggestionProvider", unexpected_client)
    assert check_gemini.main(["--trec-translation-smoke"]) == 0
    offline_output = capsys.readouterr().out
    assert "no network request was made" in offline_output
    assert "Configured model access/quota: NOT CHECKED" in offline_output
    assert "30.0s; automatic UI retries: 0" in offline_output
    assert check_gemini.main(["--live", "--trec-translation-smoke"]) == 2
    assert "blocked" in capsys.readouterr().out


def test_confirmed_branch_uses_exactly_one_synthetic_sentence_and_two_calls(
        monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_gemini, "configuration_status", lambda: STATUS)
    calls = []

    class FakeTranslator:
        def translate(self, subject, body, *, consent):
            calls.append("translate")
            assert subject == ""
            assert body == check_gemini.SYNTHETIC_CHINESE_SENTENCE
            assert consent is True
            return "", "Ngày mai chúng ta họp lúc ba giờ chiều."

    class FakeReviewer:
        def review(self, request, *, timeout_seconds):
            calls.append("review")
            assert request.subject == "" and request.body == check_gemini.SYNTHETIC_CHINESE_SENTENCE
            assert request.external_enabled and request.explicit_email_sharing_consent
            assert timeout_seconds == 15.0
            return json.dumps({
                "assessment": "normal", "summary": "Câu thông báo họp thông thường.",
                "evidence": [{"field": "body", "quote": "开会",
                              "explanation": "Nội dung đề cập cuộc họp."}],
                "suspicious_indicators": [],
            }, ensure_ascii=False)

    monkeypatch.setattr(check_gemini, "GeminiTranslator", FakeTranslator)
    monkeypatch.setattr(check_gemini, "TrecGeminiSuggestionProvider", FakeReviewer)
    assert check_gemini.main([
        "--live", "--confirm-synthetic-request", "--trec-translation-smoke"
    ]) == 0
    assert calls == ["translate", "review"]
    output = capsys.readouterr().out
    assert "Vietnamese detected" in output and "schema valid" in output
    assert "No TREC dataset email" in output


@pytest.mark.parametrize(
    "failure,expected",
    [
        (ProviderTimeoutError("secret test key"), "category=timeout"),
        (SyntheticQuotaError("secret test key"), "category=rate_or_quota"),
        (RuntimeError("secret test key"), "category=unknown"),
    ],
)
def test_transport_failures_are_redacted_and_never_retried(
        monkeypatch, capsys, failure, expected) -> None:
    monkeypatch.setattr(check_gemini, "configuration_status", lambda: STATUS)
    calls = []

    class FailingTranslator:
        def translate(self, subject, body, *, consent):
            calls.append("translate")
            raise failure

    monkeypatch.setattr(check_gemini, "GeminiTranslator", FailingTranslator)
    assert check_gemini.main([
        "--live", "--confirm-synthetic-request", "--trec-translation-smoke"
    ]) == 1
    assert calls == ["translate"]
    output = capsys.readouterr().out
    assert expected in output and "no automatic retry" in output.lower()
    assert "secret test key" not in output


def test_invalid_json_or_non_vietnamese_output_fails_closed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_gemini, "configuration_status", lambda: STATUS)
    calls = []

    class FakeTranslator:
        def translate(self, subject, body, *, consent):
            calls.append("translate")
            return "", "Ngày mai họp lúc ba giờ chiều."

    class BadReviewer:
        def review(self, request, *, timeout_seconds):
            calls.append("review")
            return "not JSON"

    monkeypatch.setattr(check_gemini, "GeminiTranslator", FakeTranslator)
    monkeypatch.setattr(check_gemini, "TrecGeminiSuggestionProvider", BadReviewer)
    flags = ["--live", "--confirm-synthetic-request", "--trec-translation-smoke"]
    assert check_gemini.main(flags) == 1
    assert calls == ["translate", "review"]
    assert "category=schema_or_configuration" in capsys.readouterr().out

    class Untranslated(FakeTranslator):
        def translate(self, subject, body, *, consent):
            calls.append("translate")
            return "", body

    calls.clear()
    monkeypatch.setattr(check_gemini, "GeminiTranslator", Untranslated)
    assert check_gemini.main(flags) == 1
    assert calls == ["translate"]
    assert "category=schema_or_configuration" in capsys.readouterr().out


def test_model_listing_is_metadata_only_and_recommends_listed_flash(monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_gemini, "configuration_status", lambda: STATUS)
    calls = []

    def fake_list():
        calls.append("list")
        return ["gemini-3-flash", "gemini-embedding-001"]

    monkeypatch.setattr(check_gemini, "list_generate_content_models", fake_list)
    assert check_gemini.main(["--list-models"]) == 0
    assert calls == ["list"]
    output = capsys.readouterr().out
    assert "gemini-2.5-flash: NOT LISTED" in output
    assert "Flash alternative candidate: gemini-3-flash" in output
    assert check_gemini.main(["--list-models", "--live"]) == 2
    assert calls == ["list"]


def test_model_listing_error_reports_safe_code_only(monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_gemini, "configuration_status", lambda: STATUS)

    class SDKError(RuntimeError):
        code = 401
        status = "UNAUTHENTICATED"

    def fail_list():
        raise SDKError("secret-key Authorization: Bearer secret-key")

    monkeypatch.setattr(check_gemini, "list_generate_content_models", fail_list)
    assert check_gemini.main(["--list-models"]) == 1
    output = capsys.readouterr().out
    assert "category=authentication" in output
    assert "http=401" in output and "api_code=UNAUTHENTICATED" in output
    assert "secret-key" not in output and "Authorization" not in output


def test_list_models_filters_actions_and_closes_client() -> None:
    class Model:
        def __init__(self, name, actions):
            self.name, self.supported_actions = name, actions

    class Models:
        def __init__(self):
            self.list_calls = 0

        def list(self):
            self.list_calls += 1
            return [Model("models/gemini-3-flash", ["generateContent"]),
                    Model("models/text-embedding-004", ["embedContent"]),
                    Model("models/gemma-3-4b-it", ["generateContent"])]

        def generate_content(self, **kwargs):
            raise AssertionError("Listing must not generate content")

    class Client:
        def __init__(self):
            self.models = Models()
            self.closed = False

        def close(self):
            self.closed = True

    client = Client()
    names = gemini_reviewer.list_generate_content_models(
        environ={"GEMINI_API_KEY": "synthetic-key"},
        client_factory=lambda key, timeout: client,
    )
    assert names == ["gemini-3-flash", "gemma-3-4b-it"]
    assert client.models.list_calls == 1 and client.closed


def test_safe_api_error_diagnostic_never_contains_raw_message_or_url() -> None:
    class SDKError(RuntimeError):
        code = 403
        status = "PERMISSION_DENIED"

    error = SDKError("secret-key https://example.invalid/?key=secret-key email body")
    detail = gemini_reviewer.safe_gemini_diagnostic(error)
    assert detail == {"exception_type": "SDKError", "http_status": 403,
                      "api_code": "PERMISSION_DENIED", "category": "permission"}
    assert "secret-key" not in str(detail)

    class FakeModels:
        def generate_content(self, **kwargs):
            raise error

    client = type("Client", (), {"models": FakeModels()})()
    with pytest.raises(RuntimeError) as caught:
        gemini_reviewer._safe_generate(client, model="gemini-test",
                                        contents="synthetic body", config={})
    assert "secret-key" not in str(caught.value)
    assert gemini_reviewer.safe_gemini_diagnostic(caught.value) == detail


def test_sdk_client_disables_implicit_retries_without_network(monkeypatch) -> None:
    fake_google = ModuleType("google")
    fake_genai = ModuleType("google.genai")
    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return object()

    fake_genai.Client = fake_client
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    gemini_reviewer._client("test-only-key", 20.0)
    assert len(calls) == 1
    assert calls[0]["http_options"]["retry_options"] == {"attempts": 1}
    assert 0 < calls[0]["http_options"]["timeout"] <= 20_000
