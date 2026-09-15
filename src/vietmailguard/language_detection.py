"""Lightweight email-language detection kept separate from classification."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


_TOKEN_PATTERN = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_VIETNAMESE_DIACRITICS = frozenset(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
_VIETNAMESE_WORDS = frozenset(
    {
        "bạn", "các", "của", "cho", "chúng", "đã", "đang", "được", "giờ",
        "hãy", "không", "khi", "là", "một", "ngày", "người", "nhận", "những",
        "này", "sẽ", "tài", "thông", "tôi", "trong", "và", "vào", "với", "xác",
        "xin", "yêu", "cần", "khẩn", "mật", "khoản", "đăng", "nhập", "chuyển",
        "thanh", "toán", "khuyến", "mãi", "giảm", "giá", "ưu", "đãi",
    }
)
_VIETNAMESE_UNACCENTED_WORDS = frozenset(
    {
        "ban", "cac", "cua", "cho", "chung", "da", "dang", "duoc", "gio",
        "hay", "khong", "khi", "la", "mot", "ngay", "nguoi", "nhan", "nhung",
        "nay", "se", "tai", "thong", "toi", "trong", "va", "vao", "voi", "xac",
        "xin", "yeu", "can", "khan", "mat", "khoan", "dang", "nhap", "chuyen",
        "thanh", "toan", "khuyen", "mai", "giam", "gia", "uu", "dai",
    }
)
_ENGLISH_WORDS = frozenset(
    {
        "a", "account", "and", "are", "at", "be", "click", "email", "for", "from",
        "hello", "in", "is", "login", "meeting", "of", "on", "or", "our", "password",
        "please", "project", "the", "this", "to", "update", "verify", "we", "with",
        "you", "your", "team", "tomorrow", "message", "review", "offer", "sale",
    }
)


@dataclass(frozen=True)
class LanguageDetection:
    """Language decision and transparent lexical evidence."""

    language: str
    token_count: int
    vietnamese_score: float
    english_score: float
    vietnamese_evidence: tuple[str, ...]
    english_evidence: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["vietnamese_evidence"] = list(self.vietnamese_evidence)
        value["english_evidence"] = list(self.english_evidence)
        return value


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(character for character in decomposed if not unicodedata.combining(character)).replace("đ", "d")


def detect_language_details(text: object) -> LanguageDetection:
    """Detect ``en``, ``vi``, ``mixed`` or ``unknown`` without affecting ML input."""
    normalized = unicodedata.normalize("NFC", "" if text is None else str(text)).casefold()
    tokens = _TOKEN_PATTERN.findall(normalized)
    if not tokens:
        return LanguageDetection("unknown", 0, 0.0, 0.0, (), (), "no alphabetic tokens")

    vietnamese_hits: list[str] = []
    english_hits: list[str] = []
    vietnamese_score = 0.0
    english_score = 0.0
    for token in tokens:
        unaccented = _strip_accents(token)
        has_vietnamese_mark = any(character in _VIETNAMESE_DIACRITICS for character in token)
        if has_vietnamese_mark:
            vietnamese_score += 2.0
            vietnamese_hits.append(token)
        elif token in _VIETNAMESE_WORDS:
            vietnamese_score += 1.25
            vietnamese_hits.append(token)
        elif unaccented in _VIETNAMESE_UNACCENTED_WORDS:
            vietnamese_score += 1.0
            vietnamese_hits.append(token)
        if token in _ENGLISH_WORDS:
            english_score += 1.0
            english_hits.append(token)

    # One or two generic words are intentionally not enough to assert a language.
    if len(tokens) < 3 and max(vietnamese_score, english_score) < 2.0:
        language = "unknown"
        reason = "content is too short for a reliable lexical decision"
    elif vietnamese_score >= 2.0 and english_score >= 2.0:
        lower = min(vietnamese_score, english_score)
        upper = max(vietnamese_score, english_score)
        if lower / upper >= 0.25:
            language = "mixed"
            reason = "both English and Vietnamese evidence are substantial"
        else:
            language = "vi" if vietnamese_score > english_score else "en"
            reason = "one language has clearly stronger lexical evidence"
    elif vietnamese_score >= 2.0:
        language = "vi"
        reason = "Vietnamese lexical or diacritic evidence was detected"
    elif english_score >= 2.0:
        language = "en"
        reason = "English lexical evidence was detected"
    else:
        language = "unknown"
        reason = "lexical evidence is insufficient"

    return LanguageDetection(
        language=language,
        token_count=len(tokens),
        vietnamese_score=vietnamese_score,
        english_score=english_score,
        vietnamese_evidence=tuple(dict.fromkeys(vietnamese_hits))[:8],
        english_evidence=tuple(dict.fromkeys(english_hits))[:8],
        reason=reason,
    )


def detect_language(text: object) -> str:
    """Return only the stable language identifier."""
    return detect_language_details(text).language
