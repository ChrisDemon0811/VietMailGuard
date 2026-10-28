"""Offline Gemini configuration check; live request requires two explicit flags."""

from __future__ import annotations

import argparse

from vietmailguard.language_detection import detect_language
from vietmailguard.v21.gemini_reviewer import (
    configuration_status,
    list_generate_content_models,
    live_connection_check,
    safe_gemini_diagnostic,
)
from vietmailguard.v21.llm_reviewer import (
    ReviewRequest, validate_review_response,
)
from vietmailguard.v21.trec_review_assist import GeminiTranslator, TrecGeminiSuggestionProvider


SYNTHETIC_CHINESE_SENTENCE = "明天下午三点开会。"


def _show_safe_failure(stage: str, error: Exception) -> None:
    detail = safe_gemini_diagnostic(error)
    print(
        f"{stage}: category={detail['category']}; "
        f"exception={detail['exception_type']}; "
        f"http={detail['http_status'] if detail['http_status'] is not None else 'N/A'}; "
        f"api_code={detail['api_code'] or 'N/A'}."
    )
    if detail["category"] == "rate_or_quota":
        print("Check the project's rate limits and quota in Google AI Studio; no automatic retry.")
    elif detail["category"] == "timeout":
        print("Request timed out; no automatic retry. A timed-out request may still consume quota.")
    elif detail["category"] == "connection":
        print("Connection failed; check network/proxy/DNS. No automatic retry.")
    elif detail["category"] == "schema_or_configuration":
        print("Application response/configuration validation failed; no automatic retry.")
    else:
        print("Check model access, API permissions, request compatibility, and provider status. No automatic retry.")


def _flash_candidate(names: list[str], selected: str) -> str | None:
    compatible = [name for name in names
                  if name.startswith("gemini-") and "flash" in name
                  and not any(token in name for token in ("image", "audio", "tts", "live"))
                  and name != selected]
    stable = [name for name in compatible if "preview" not in name and "exp" not in name]
    return next(iter(sorted(stable or compatible, reverse=True)), None)


def _run_trec_synthetic_smoke() -> None:
    """Make at most one translation and one reviewer request; no dataset input."""
    subject, body = "", SYNTHETIC_CHINESE_SENTENCE
    translated_subject, translated_body = GeminiTranslator().translate(
        subject, body, consent=True
    )
    if translated_subject.strip() or not translated_body.strip():
        raise ValueError("Synthetic translation added a subject or omitted the body")
    if detect_language(translated_body) != "vi":
        raise ValueError("Synthetic translation is not recognizably Vietnamese")
    request = ReviewRequest(sender="", subject=subject, body=body,
                            external_enabled=True,
                            explicit_email_sharing_consent=True)
    raw = TrecGeminiSuggestionProvider().review(request, timeout_seconds=15.0)
    assessed = validate_review_response(raw, request)
    print("Synthetic translation: Vietnamese detected; inspect meaning manually.")
    print(f"Synthetic reviewer JSON: schema valid; weak suggestion = {assessed['assessment']}.")
    print("No TREC dataset email was sent or labeled.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Permit one potentially billed synthetic Gemini request")
    parser.add_argument("--confirm-synthetic-request", action="store_true",
                        help="Confirm that the optional live test may use the network")
    parser.add_argument("--trec-translation-smoke", action="store_true",
                        help="With --live and confirmation, test synthetic Chinese-to-Vietnamese translation and JSON suggestion (up to 2 API requests)")
    parser.add_argument("--list-models", action="store_true",
                        help="List accessible generateContent models; metadata request only, no generation")
    args = parser.parse_args(argv)
    status = configuration_status()
    print(f"Project .env: {'present' if status['dotenv_exists'] else 'absent'}")
    print(f"GEMINI_API_KEY: {'configured' if status['api_key_configured'] else 'missing'}")
    print(f"google-genai SDK: {status['sdk_version'] or 'not installed'}")
    print(f"python-dotenv: {status['dotenv_version'] or 'not installed'}")
    print(f"GEMINI_MODEL: {status['model'] or 'invalid'}")
    if status["configuration_error"]:
        print(f"Configuration error: {status['configuration_error']}")
    if args.list_models and (args.live or args.confirm_synthetic_request or args.trec_translation_smoke):
        print("--list-models is separate from all live smoke-test flags; no request was made.")
        return 2
    if not args.live and not args.list_models:
        print("Offline configuration check only; no network request was made.")
        print("Configured model access/quota: NOT CHECKED (requires a separately authorized API request).")
        print(f"App reviewer wait limit: {status.get('review_timeout_seconds', 'N/A')}s; automatic UI retries: 0.")
    if not all((status["api_key_configured"], status["sdk_version"],
                status["model_syntax_valid"], not status["configuration_error"])):
        return 1
    if args.list_models:
        try:
            available = list_generate_content_models()
        except Exception as error:
            _show_safe_failure("Model listing failed", error)
            return 1
        print(f"Accessible model IDs reporting generateContent: {len(available)}")
        for name in available:
            print(f"  {name}")
        selected = status["model"]
        if selected in available:
            print(f"Configured model {selected}: LISTED with generateContent; generation/quota not yet tested.")
        else:
            print(f"Configured model {selected}: NOT LISTED for generateContent with this key/API version.")
            candidate = _flash_candidate(available, selected)
            if candidate:
                print(f"Flash alternative candidate: {candidate} (metadata only; JSON/quota not yet tested).")
            else:
                print("No Flash alternative with generateContent was listed; do not change the model blindly.")
        return 0
    if args.live:
        if not args.confirm_synthetic_request:
            print("Live check blocked: add --confirm-synthetic-request.")
            return 2
        try:
            if args.trec_translation_smoke:
                _run_trec_synthetic_smoke()
                return 0
            succeeded = live_connection_check(confirm_synthetic_request=True)
        except Exception as error:
            _show_safe_failure("Live synthetic request failed", error)
            return 1
        print("Live synthetic request: PASS" if succeeded else
              "Live synthetic request: empty response")
        return 0 if succeeded else 1
    if args.trec_translation_smoke:
        print("TREC live smoke not run; add --live --confirm-synthetic-request.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
