"""Execute the real analyzer input handler with safe MIME and an engine spy.

Only the handler AST is compiled: no Streamlit page is rendered, no model or
network is used, and the production MIME parser remains unchanged.
"""

from __future__ import annotations

import ast
from contextlib import nullcontext
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vietmailguard.eml_parser import parse_eml_bytes


PAGE_PATH = Path(__file__).resolve().parents[1] / "app" / "views" / "email_analyzer.py"


class EngineSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def analyze_email(self, **fields: str) -> dict[str, Any]:
        self.calls.append(fields)
        if not fields["subject"].strip() and not fields["body"].strip():
            raise ValueError("No analyzable content")
        return {"prediction": "normal", "confidence": None,
                "model_version": "synthetic-test-double"}


def _handler() -> tuple[dict[str, Any], EngineSpy, ast.If]:
    tree = ast.parse(PAGE_PATH.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if (
        isinstance(node, ast.ClassDef) and node.name == "_EmlParseError"
    ) or (isinstance(node, ast.FunctionDef) and node.name == "_submit_analysis")]
    engine = EngineSpy()
    ui = SimpleNamespace(session_state={"analysis_mode": "v21_ml"}, errors=[])
    ui.spinner = lambda *_args: nullcontext()
    ui.error = ui.errors.append
    scope = {
        "Any": Any, "parse_eml_bytes": parse_eml_bytes,
        "_selected_engine": lambda: engine, "st": ui, "V2_MODE": "v2",
    }
    exec(compile(ast.Module(definitions, type_ignores=[]), str(PAGE_PATH), "exec"), scope)
    render = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "render")
    submission = next(node for node in render.body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Name) and node.test.id == "submitted")
    return scope, engine, submission


def _eml(subject: str, body: str, *, sender: str | None = "demo@example.com") -> bytes:
    message = EmailMessage()
    if sender:
        message["From"] = sender
    message["To"] = "local@example.org"
    if subject:
        message["Subject"] = subject
    message.set_content(body)
    return message.as_bytes()


@pytest.mark.parametrize(("subject", "body"), [
    ("Meeting", "Synthetic meeting at three."),
    ("Lịch họp", "Thông báo họp thử nghiệm, không phải thư thật."),
    ("会议通知", "这是一封人工创建的测试邮件。"),
])
def test_uploaded_unicode_mime_replaces_all_stale_manual_fields(subject, body):
    scope, engine, _ = _handler()
    payload = scope["_submit_analysis"](
        "stale@example.net", "Stale subject", "Stale manual body",
        _eml(subject, body),
    )
    assert engine.calls == [{"sender": "demo@example.com", "subject": subject, "body": body}]
    assert payload["input_source"] == "eml"
    assert payload["email_fields"] == engine.calls[0]
    assert payload["parsed_eml"]["receiver"] == "local@example.org"


def test_absent_eml_headers_never_fall_back_to_stale_manual_content():
    scope, engine, _ = _handler()
    payload = scope["_submit_analysis"](
        "stale@example.net", "Stale subject", "Stale body",
        _eml("", "Only this MIME body should be analyzed.", sender=None),
    )
    assert engine.calls[0] == {"sender": "", "subject": "", "body": "Only this MIME body should be analyzed."}
    assert payload["email_fields"] == engine.calls[0]


def test_html_is_plain_text_with_link_not_scripts_tracking_image_or_network(monkeypatch):
    import socket

    def no_network(*_args, **_kwargs):
        raise AssertionError("MIME parsing and input composition must be offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    message = EmailMessage()
    message["From"] = "demo@example.com"
    message["Subject"] = "Synthetic HTML"
    message.set_content(
        '<p>Thông báo thử nghiệm</p><script>DO_NOT_EXECUTE</script>'
        '<img src="https://example.com/tracking-pixel.png">'
        '<a href="https://example.org/account">Tài khoản</a>',
        subtype="html",
    )
    scope, engine, _ = _handler()
    scope["_submit_analysis"]("old", "old", "old", message.as_bytes())
    body = engine.calls[0]["body"]
    assert "Thông báo thử nghiệm" in body
    assert "https://example.org/account" in body
    assert "DO_NOT_EXECUTE" not in body
    assert "tracking-pixel" not in body
    assert "<script" not in body
    assert "<img" not in body


@pytest.mark.parametrize("raw", [
    b"",
    b"From: demo@example.com\r\nContent-Type: text/plain; charset=not-a-real-charset\r\n\r\n\xff\xfe",
])
def test_empty_or_unknown_charset_clears_stale_results_and_fails_without_inference(raw):
    scope, engine, submission = _handler()
    ui = scope["st"]
    ui.session_state.update({
        "vietmailguard_analysis_result": {"result": {"prediction": "spam"}},
        "vietmailguard_llm_review": {"reviewer": {"assessment": "phishing"}},
        "quick_llm_consent": True,
    })
    scope.update({
        "submitted": True, "sender": "stale@example.net", "subject": "Stale subject",
        "body": "Stale body", "uploaded": SimpleNamespace(getvalue=lambda: raw),
        "language": "en", "_RESULT_STATE_KEY": "vietmailguard_analysis_result",
        "t": lambda key, _language: key,
    })
    exec(compile(ast.Module([submission], type_ignores=[]), str(PAGE_PATH), "exec"), scope)
    assert engine.calls == []
    assert ui.errors == ["eml_parse_error"]
    assert "vietmailguard_analysis_result" not in ui.session_state
    assert "vietmailguard_llm_review" not in ui.session_state
    assert ui.session_state["quick_llm_consent"] is False


def test_manual_submit_uses_only_trimmed_manual_snapshot():
    scope, engine, _ = _handler()
    payload = scope["_submit_analysis"](" demo@example.com ", " Synthetic ", " Only manual text ", None)
    assert payload["parsed_eml"] is None
    assert payload["input_source"] == "manual"
    assert engine.calls == [{"sender": "demo@example.com", "subject": "Synthetic", "body": "Only manual text"}]
