from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from vietmailguard.inference import INFERENCE_RESULT_FIELDS, load_inference_engine


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.production_artifact


@pytest.fixture(scope="module")
def engine():
    return load_inference_engine()


CASES = {
    "english_normal": {
        "sender": "manager@example.com",
        "subject": "Project meeting agenda",
        "body": "Hello team, please review the project agenda for tomorrow.",
    },
    "english_spam": {
        "sender": "offers@shop.example",
        "subject": "Limited time sale",
        "body": "Discount sale promotion. Buy now and save 70 percent. Special offer. Unsubscribe here.",
    },
    "english_phishing": {
        "sender": "Security <notice@account-alert.example>",
        "subject": "Urgent account verification",
        "body": "Your account is suspended. Sign in now at http://192.0.2.10/login and enter your password to verify your account.",
    },
    "vietnamese_normal": {
        "sender": "quanly@example.com",
        "subject": "Lịch họp dự án",
        "body": "Chào cả nhóm, cuộc họp dự án bắt đầu lúc mười giờ sáng mai.",
    },
    "vietnamese_promotion": {
        "sender": "uudai@example.com",
        "subject": "Khuyến mãi phần mềm cuối tuần",
        "body": "Giảm giá 40 phần trăm cho gói dịch vụ năm. Xem ưu đãi hoặc hủy đăng ký nhận tin.",
    },
    "vietnamese_phishing": {
        "sender": "Hỗ trợ <canhbao@xac-minh.example>",
        "subject": "Yêu cầu xác minh tài khoản khẩn cấp",
        "body": "Tài khoản của bạn đã bị khóa. Hãy đăng nhập tại http://192.0.2.10/login và nhập mật khẩu để xác minh.",
    },
    "vietnamese_unaccented": {
        "sender": "canhbao@example.com",
        "subject": "Xac minh tai khoan",
        "body": "Tai khoan cua ban da bi khoa. Hay dang nhap ngay va nhap mat khau de xac minh.",
    },
    "mixed": {
        "sender": "support@example.com",
        "subject": "Account cần xác minh",
        "body": "Your account cần được xác minh ngay. Please đăng nhập để verify your account.",
    },
}


@pytest.mark.parametrize(
    ("case_name", "language"),
    [
        ("english_normal", "en"),
        ("english_spam", "en"),
        ("english_phishing", "en"),
        ("vietnamese_normal", "vi"),
        ("vietnamese_promotion", "vi"),
        ("vietnamese_phishing", "vi"),
        ("vietnamese_unaccented", "vi"),
        ("mixed", "mixed"),
    ],
)
def test_v2_examples_return_stable_schema(engine, case_name: str, language: str) -> None:
    result = engine.analyze_email(**CASES[case_name])
    assert INFERENCE_RESULT_FIELDS <= set(result)
    assert result["model_name"] == "VietMailGuard V2 Bilingual"
    assert result["model_version"] == "2.0.0"
    assert result["detected_language"] == language
    assert result["prediction"] in {"normal", "spam", "phishing"}
    assert set(result["class_probabilities"]) == {"normal", "spam", "phishing"}
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0)
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0 <= result["risk_score"] <= 100


def test_representative_english_predictions(engine) -> None:
    assert engine.analyze_email(**CASES["english_normal"])["prediction"] == "normal"
    assert engine.analyze_email(**CASES["english_spam"])["prediction"] == "spam"
    assert engine.analyze_email(**CASES["english_phishing"])["prediction"] == "phishing"


def test_vietnamese_support_is_experimental_and_honest(engine) -> None:
    result = engine.analyze_email(**CASES["vietnamese_normal"])
    assert result["language_support_status"]["status"] == "experimental"
    limitations = " ".join(result["limitations"])
    assert "translated" in limitations
    assert "no phishing examples" in limitations
    assert "Native Vietnamese benchmark is not available." in limitations


def test_short_advertisement_failure_keeps_ml_prediction_and_requests_review(engine) -> None:
    result = engine.analyze_email(**CASES["vietnamese_promotion"])
    assert result["prediction"] == "normal"
    assert result["base_recommended_action"] == "ALLOW"
    assert result["recommended_action"] == "REVIEW"
    assert any(value["code"] == "commercial_promotion" for value in result["content_findings"])
    assert any("promotional" in value for value in result["recommendation_reasons"])


def test_short_english_advertisement_failure_also_preserves_ml_prediction(engine) -> None:
    result = engine.analyze_email(
        subject="Weekend sale on office software",
        body="Save 40 percent on our annual plan this weekend. View the offer or unsubscribe.",
    )
    assert result["prediction"] == "normal"
    assert result["recommended_action"] == "REVIEW"
    assert any(value["code"] == "commercial_promotion" for value in result["content_findings"])


def test_vietnamese_phishing_style_has_real_rules_and_model_explanation(engine) -> None:
    result = engine.analyze_email(**CASES["vietnamese_phishing"])
    codes = {value["code"] for value in result["content_findings"]}
    assert {"urgency", "credential_request", "account_suspension"} <= codes
    assert result["model_explanation"]["supported"]
    assert result["model_explanation"]["features"]
    assert "underlying linear classifier" in result["model_explanation"]["limitation"]


def test_mixed_language_is_not_used_to_overwrite_prediction(engine) -> None:
    result = engine.analyze_email(**CASES["mixed"])
    assert result["detected_language"] == "mixed"
    assert result["language_support_status"]["status"] == "experimental_not_independently_benchmarked"
    assert result["prediction"] in {"normal", "spam", "phishing"}


def test_very_short_content_is_unknown_but_still_analyzable(engine) -> None:
    result = engine.analyze_email(subject="Hi", body="Thanks")
    assert result["detected_language"] == "unknown"
    assert result["language_support_status"]["status"] == "unknown"


def test_empty_content_is_rejected(engine) -> None:
    with pytest.raises(ValueError, match="subject or body"):
        engine.analyze_email(sender="sender@example.com", subject=" ", body="\n")


def test_suspicious_and_benign_urls_are_distinguished_without_verdict(engine) -> None:
    suspicious = engine.analyze_email(
        subject="Verify account",
        body="Open hxxp : / / 192 . 0 . 2 . 10 / login to continue.",
    )
    suspicious_codes = {
        finding["code"]
        for analysis in suspicious["url_findings"]
        for finding in analysis["findings"]
    }
    assert {"obfuscated_url", "ip_host", "suspicious_token"} <= suspicious_codes
    benign = engine.analyze_email(
        subject="Documentation",
        body="Read https://www.example.com/about for the project documentation.",
    )
    assert benign["url_findings"][0]["findings"] == []


def test_probability_order_comes_from_classifier(engine) -> None:
    result = engine.analyze_email(**CASES["english_normal"])
    classes = [str(value) for value in engine.model.named_steps["classifier"].classes_]
    assert list(result["class_probabilities"]) == classes
    assert result["probability_source"] == "calibrated_predict_proba"


def test_eml_can_enter_the_same_stable_api(engine) -> None:
    raw = (
        b"From: manager@example.com\r\n"
        b"Subject: Project meeting\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Hello team, please review the project agenda tomorrow.\r\n"
    )
    result = engine.analyze_eml_bytes(raw)
    assert INFERENCE_RESULT_FIELDS <= set(result)
    assert result["detected_language"] == "en"


def test_artifact_fingerprinted_cache_does_not_serve_stale_config(tmp_path: Path) -> None:
    model_path = tmp_path / "production_pipeline.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    risk_path = tmp_path / "risk_rules_v2.json"
    security_path = tmp_path / "security_rules_v2.json"
    for source, target in [
        (ROOT / "models/v2_bilingual/production_pipeline.joblib", model_path),
        (ROOT / "models/v2_bilingual/model_metadata.json", metadata_path),
        (ROOT / "config/risk_rules_v2.json", risk_path),
        (ROOT / "config/security_rules_v2.json", security_path),
    ]:
        shutil.copy2(source, target)
    first = load_inference_engine(model_path, metadata_path, risk_path, security_path)
    security_path.write_text(
        security_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    second = load_inference_engine(model_path, metadata_path, risk_path, security_path)
    assert second is not first
