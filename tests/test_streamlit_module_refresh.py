"""Regression for updated pages running with an older cached UI module."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app/app.py"


@pytest.fixture
def stale_ui(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "app"))
    current = importlib.import_module("runtime_ui")
    old = ModuleType("runtime_ui")
    old.__dict__.update(vars(current))
    old.__dict__.pop("reviewer_request_options")
    monkeypatch.setitem(sys.modules, "runtime_ui", old)
    return old


@pytest.mark.parametrize("page", ["mail_client.py", "email_analyzer.py"])
def test_old_cached_module_reproduces_reported_page_import_error(stale_ui, page):
    """Disk can contain the function while Python's loaded module lacks it."""
    source = (ROOT / "app/views" / page).read_text(encoding="utf-8")
    imports = [node for node in ast.parse(source).body
               if isinstance(node, ast.ImportFrom) and node.module == "runtime_ui"]
    assert len(imports) == 1
    assert "def reviewer_request_options(" in (ROOT / "app/runtime_ui.py").read_text(encoding="utf-8")
    with pytest.raises(ImportError, match="reviewer_request_options"):
        exec(compile(ast.Module(body=imports, type_ignores=[]), str(page), "exec"), {})


@pytest.mark.parametrize("cached_value", ["missing", "not_callable"])
def test_entry_refreshes_only_stale_ui_and_preserves_text_draft(
    stale_ui, cached_value, tmp_path, monkeypatch,
):
    from vietmailguard.v21 import gemini_reviewer, runtime_inference

    if cached_value == "not_callable":
        stale_ui.reviewer_request_options = None
    monkeypatch.setenv("VIETMAILGUARD_MAIL_DB", str(tmp_path / "mail.db"))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(runtime_inference, "model_availability", lambda: (False, "source-only test"))
    provider_calls = []

    def forbidden(*args, **kwargs):
        provider_calls.append(True)
        raise AssertionError("Refreshing UI must not construct a Gemini client")

    monkeypatch.setattr(gemini_reviewer, "_client", forbidden)
    backend_identity = id(gemini_reviewer)
    model_loader = runtime_inference.load_v21_engine
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["draft_quick_subject"] = "Bản nháp chưa gửi"
    app.session_state["draft_quick_body"] = "Nội dung tiếng Việt được giữ trong session."
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    assert sys.modules["runtime_ui"] is stale_ui
    assert callable(stale_ui.reviewer_request_options)
    assert id(gemini_reviewer) == backend_identity
    assert runtime_inference.load_v21_engine is model_loader
    app.switch_page("views/email_analyzer.py").run()
    assert not app.exception
    assert next(item for item in app.text_input if item.key == "input_quick_subject").value == "Bản nháp chưa gửi"
    assert next(item for item in app.text_area if item.key == "input_quick_body").value.startswith("Nội dung tiếng Việt")
    assert "vietmailguard_analysis_result" not in app.session_state
    assert provider_calls == []


def test_old_reviewer_backend_shows_restart_notice_instead_of_keyerror(
    tmp_path, monkeypatch,
):
    from vietmailguard.v21 import gemini_reviewer

    monkeypatch.syspath_prepend(str(ROOT / "app"))
    monkeypatch.setattr(gemini_reviewer, "configuration_status", lambda: {
        "api_key_configured": True, "sdk_version": "synthetic-version",
        "model": "gemini-test", "model_syntax_valid": True,
        "configuration_error": None,
    })
    script = tmp_path / "old_backend_ui.py"
    # Synthetic AppTest entry, never email content or a real provider call.
    script.write_text(
        "import streamlit as st\n"
        "from runtime_ui import reviewer_controls\n"
        "st.session_state['reviewer_provider'] = 'gemini'\n"
        "ready, external = reviewer_controls('en', consent_key='test_consent')\n"
        "st.text(str((ready, external)))\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script)).run()
    assert not app.exception
    assert any("older Reviewer code" in item.value for item in app.warning)
    next(item for item in app.checkbox if item.key == "test_consent").check().run()
    assert not app.exception
    assert app.text[0].value == "(False, True)"
