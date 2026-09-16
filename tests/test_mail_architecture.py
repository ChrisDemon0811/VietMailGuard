from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIL_PAGE = PROJECT_ROOT / "app" / "pages" / "mail_client.py"
MAIL_SERVICE = PROJECT_ROOT / "src" / "vietmailguard" / "mail_service.py"


def test_mailbox_ui_has_no_sql_or_direct_inference_logic() -> None:
    source = MAIL_PAGE.read_text(encoding="utf-8")

    assert "from vietmailguard.mail_service import MailService" in source
    assert "sqlite3" not in source
    assert ".execute(" not in source
    assert "analyze_email" not in source
    assert ".predict(" not in source
    assert ".predict_proba(" not in source


def test_mail_service_is_the_inference_and_repository_boundary() -> None:
    source = MAIL_SERVICE.read_text(encoding="utf-8")

    assert "from vietmailguard.inference import analyze_email" in source
    assert "from vietmailguard.mail_repository import" in source
    assert "route_analysis(analysis)" in source
    assert ".predict(" not in source
    assert ".predict_proba(" not in source


def test_mail_detail_uses_plain_body_and_does_not_create_clickable_urls() -> None:
    source = MAIL_PAGE.read_text(encoding="utf-8")

    assert 'st.text(str(message.get("body") or ""))' in source
    assert 'message.get("body_html")' not in source
    assert "st.link_button" not in source
    assert "st.markdown(str(url" not in source


def test_mailbox_uses_separate_full_width_list_and_detail_modes() -> None:
    source = MAIL_PAGE.read_text(encoding="utf-8")

    assert 'if mode == "detail":' in source
    assert "_render_message_detail(" in source
    assert "_render_message_list(" in source
    assert "st.columns([0.92, 1.25]" not in source
    assert "available_ids" not in source


def test_all_list_views_open_the_shared_detail_renderer() -> None:
    source = MAIL_PAGE.read_text(encoding="utf-8")

    assert "_open_detail(email_id, current_view)" in source
    assert '_open_detail(email_id, "security")' in source
    assert source.count("def _render_message_detail(") == 1
