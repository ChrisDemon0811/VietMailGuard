from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "app.py"
pytestmark = pytest.mark.production_artifact

DATASET_DASHBOARD_ARTIFACTS = [
    PROJECT_ROOT / "data" / "splits" / "v2" / f"{name}.csv"
    for name in ("train", "validation", "test")
] + [
    PROJECT_ROOT / "data" / "curated" / "v2" / "Vietnamese_Curated_Dataset.csv",
    PROJECT_ROOT / "data" / "processed" / "Vietnamese_Translated_Augmentation.csv",
]


def test_all_streamlit_pages_render_saved_artifacts() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
    assert not app.exception
    assert [button.label for button in app.button] == ["Analyze Email"]

    app.switch_page("pages/model_dashboard.py").run(timeout=90)
    assert not app.exception
    saved_metrics = json.loads(
        (
            PROJECT_ROOT / "results" / "v2_bilingual" / "final_test_metrics.json"
        ).read_text(encoding="utf-8")
    )
    overall = saved_metrics["overall_test_metrics"]
    displayed_metrics = {metric.label: metric.value for metric in app.metric}
    assert displayed_metrics["Test rows"] == "6,704"
    assert displayed_metrics["Macro F1"] == f"{overall['macro_f1']:.6f}"
    assert displayed_metrics["Phishing Recall"] == (
        f"{overall['per_class']['phishing']['recall']:.6f}"
    )
    assert displayed_metrics["Macro F1 for represented classes"] == "1.000000"
    assert len(app.get("plotly_chart")) == 2
    assert any(
        "No Vietnamese phishing examples" in warning.value for warning in app.warning
    )

    app.switch_page("pages/dataset_dashboard.py").run(timeout=90)
    assert not app.exception
    if all(path.exists() for path in DATASET_DASHBOARD_ARTIFACTS):
        displayed_metrics = {metric.label: metric.value for metric in app.metric}
        assert displayed_metrics["Master samples"] == "44,717"
        assert displayed_metrics["English training samples"] == "42,112"
        assert displayed_metrics["Translated Vietnamese training samples"] == "2,605"
        assert displayed_metrics["Review rows"] == "1,330"
        assert displayed_metrics["Cross-split leakage check"] == "PASSED"
    else:
        assert app.warning
        assert any("not available" in warning.value.casefold() for warning in app.warning)

    app.switch_page("pages/methodology.py").run(timeout=90)
    assert not app.exception
    assert "Risk engine" in [subheader.value for subheader in app.subheader]
    assert "Known limitations" in [subheader.value for subheader in app.subheader]
    assert "Version 1 and Version 2" in [subheader.value for subheader in app.subheader]
    assert "Probability calibration" in [subheader.value for subheader in app.subheader]


def test_language_switch_preserves_inference_result() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=180).run()
    app.text_input[0].set_value("project.manager@example.org")
    app.text_input[1].set_value("Project meeting notes")
    app.text_area[0].set_value(
        "Hello team, attached are the notes and action items from today's meeting."
    )
    app.button[0].click().run(timeout=180)
    before = dict(app.session_state["vietmailguard_analysis_result"]["result"])

    app.selectbox[0].set_value("vi").run(timeout=90)
    after = dict(app.session_state["vietmailguard_analysis_result"]["result"])

    assert not app.exception
    assert before["prediction"] == after["prediction"] == "normal"
    assert before["confidence"] == after["confidence"]
    assert [button.label for button in app.button] == ["Phân tích Email"]
    displayed_metrics = {metric.label: metric.value for metric in app.metric}
    assert displayed_metrics["Hành động đề xuất"] == "Cho phép"


def test_analyzer_shows_v2_metadata_and_separate_explanations() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=180).run()
    app.text_input[1].set_value("Account alert")
    app.text_area[0].set_value(
        "Your account is locked. Verify your password at http://192.0.2.10/login."
    )
    app.button[0].click().run(timeout=180)
    assert not app.exception
    displayed_metrics = {metric.label: metric.value for metric in app.metric}
    assert displayed_metrics["Model Version"] == "VietMailGuard V2 Bilingual"
    assert displayed_metrics["Detected Email Language"] == "English"
    assert displayed_metrics["Language Support Status"] == "Supported"
    markdown_values = [value.value for value in app.markdown]
    assert any("A. Model explanation" in value for value in markdown_values)
    assert any("B. Security Findings" in value for value in markdown_values)
    assert any("C. URL findings" in value for value in markdown_values)


def test_required_v2_ui_language_and_email_scenarios() -> None:
    scenarios = [
        (
            "en",
            "Meeting update",
            "Hello team, the project meeting moved to 3 PM tomorrow.",
            "en",
            "normal",
            "ALLOW",
        ),
        (
            "en",
            "Urgent account verification",
            "Your account is suspended. Enter your password at http://192.0.2.10/login now.",
            "en",
            "phishing",
            "QUARANTINE",
        ),
        (
            "vi",
            "Limited time sale",
            "Discount sale promotion. Buy now and save 70 percent. Special offer. Unsubscribe here.",
            "en",
            "spam",
            "MOVE_TO_SPAM",
        ),
        (
            "vi",
            "Lịch họp dự án",
            "Chào cả nhóm, cuộc họp dự án bắt đầu lúc mười giờ sáng mai.",
            "vi",
            "normal",
            "ALLOW",
        ),
        (
            "vi",
            "Khuyến mãi phần mềm cuối tuần",
            "Giảm giá 40 phần trăm cho gói dịch vụ năm. Xem ưu đãi hoặc hủy đăng ký nhận tin.",
            "vi",
            "normal",
            "REVIEW",
        ),
        (
            "vi",
            "Yêu cầu xác minh tài khoản khẩn cấp",
            "Tài khoản của bạn đã bị khóa. Hãy đăng nhập tại http://192.0.2.10/login và nhập mật khẩu để xác minh.",
            "vi",
            "phishing",
            "QUARANTINE",
        ),
        (
            "en",
            "Account cần xác minh",
            "Your account cần được xác minh ngay. Please đăng nhập để verify your account.",
            "mixed",
            None,
            None,
        ),
    ]
    app = AppTest.from_file(str(APP_PATH), default_timeout=180).run()
    for interface, subject, body, detected, prediction, action in scenarios:
        app.selectbox[0].set_value(interface).run(timeout=90)
        app.text_input[0].set_value("sender@example.com")
        app.text_input[1].set_value(subject)
        app.text_area[0].set_value(body)
        app.button[0].click().run(timeout=180)
        assert not app.exception
        result = app.session_state["vietmailguard_analysis_result"]["result"]
        assert result["detected_language"] == detected
        if prediction is not None:
            assert result["prediction"] == prediction
        if action is not None:
            assert result["recommended_action"] == action
