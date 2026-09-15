"""Run deterministic Version 2 inference smoke examples without using split data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.inference import load_inference_engine  # noqa: E402


EXAMPLES = [
    {
        "scenario": "English normal",
        "sender": "manager@example.com",
        "subject": "Project meeting agenda",
        "body": "Hello team, please review the project agenda for tomorrow.",
    },
    {
        "scenario": "English spam",
        "sender": "offers@shop.example",
        "subject": "Limited time sale",
        "body": "Discount sale promotion. Buy now and save 70 percent. Special offer. Unsubscribe here.",
    },
    {
        "scenario": "English phishing",
        "sender": "Security <notice@account-alert.example>",
        "subject": "Urgent account verification",
        "body": "Your account is suspended. Sign in now at http://192.0.2.10/login and enter your password to verify your account.",
    },
    {
        "scenario": "Vietnamese normal",
        "sender": "quanly@example.com",
        "subject": "Lịch họp dự án",
        "body": "Chào cả nhóm, cuộc họp dự án bắt đầu lúc mười giờ sáng mai.",
    },
    {
        "scenario": "Vietnamese promotional",
        "sender": "uudai@example.com",
        "subject": "Khuyến mãi phần mềm cuối tuần",
        "body": "Giảm giá 40 phần trăm cho gói dịch vụ năm. Xem ưu đãi hoặc hủy đăng ký nhận tin.",
    },
    {
        "scenario": "Vietnamese phishing-style",
        "sender": "Hỗ trợ <canhbao@xac-minh.example>",
        "subject": "Yêu cầu xác minh tài khoản khẩn cấp",
        "body": "Tài khoản của bạn đã bị khóa. Hãy đăng nhập tại http://192.0.2.10/login và nhập mật khẩu để xác minh.",
    },
    {
        "scenario": "Mixed English/Vietnamese",
        "sender": "support@example.com",
        "subject": "Account cần xác minh",
        "body": "Your account cần được xác minh ngay. Please đăng nhập để verify your account.",
    },
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    engine = load_inference_engine()
    output = []
    for example in EXAMPLES:
        result = engine.analyze_email(
            sender=example["sender"],
            subject=example["subject"],
            body=example["body"],
        )
        output.append(
            {
                "scenario": example["scenario"],
                "model_version": result["model_version"],
                "detected_language": result["detected_language"],
                "support_status": result["language_support_status"]["status"],
                "prediction": result["prediction"],
                "confidence": round(float(result["confidence"]), 6),
                "class_probabilities": {
                    key: round(float(value), 6)
                    for key, value in result["class_probabilities"].items()
                },
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "recommended_action": result["recommended_action"],
                "content_finding_codes": [
                    value["code"] for value in result["content_findings"]
                ],
                "url_finding_codes": [
                    finding["code"]
                    for analysis in result["url_findings"]
                    for finding in analysis["findings"]
                ],
                "model_features": [
                    value["feature"]
                    for value in result["model_explanation"]["features"]
                ],
                "recommendation_reasons": result["recommendation_reasons"],
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
