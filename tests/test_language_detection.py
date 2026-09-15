from __future__ import annotations

import pytest

from vietmailguard.language_detection import detect_language, detect_language_details


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello team, please review the project meeting agenda tomorrow.", "en"),
        ("Chào cả nhóm, cuộc họp dự án sẽ bắt đầu vào sáng mai.", "vi"),
        ("Tai khoan cua ban da bi khoa. Hay dang nhap ngay.", "vi"),
        ("Your account cần được xác minh ngay. Please đăng nhập để verify.", "mixed"),
        ("Hi", "unknown"),
        ("123 !!!", "unknown"),
    ],
)
def test_lightweight_language_detection(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_language_detection_exposes_evidence_not_a_class_feature() -> None:
    result = detect_language_details("Tài khoản của bạn cần xác minh ngay")
    assert result.language == "vi"
    assert result.vietnamese_score > result.english_score
    assert result.token_count > 0
