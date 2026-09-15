"""Centralized English/Vietnamese presentation translations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATION_PATH = PROJECT_ROOT / "config" / "translations.json"
SUPPORTED_LANGUAGES = ("en", "vi")
DEFAULT_LANGUAGE = "en"


@lru_cache(maxsize=1)
def load_translations(path: Path = TRANSLATION_PATH) -> dict[str, dict[str, str]]:
    """Load and validate the translation catalog once per process."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Translation catalog must be a JSON object")
    catalog: dict[str, dict[str, str]] = {}
    for language in SUPPORTED_LANGUAGES:
        values = raw.get(language)
        if not isinstance(values, dict):
            raise ValueError(f"Missing translation language: {language}")
        catalog[language] = {str(key): str(value) for key, value in values.items()}
    return catalog


def t(key: str, language: str = DEFAULT_LANGUAGE, **values: Any) -> str:
    """Translate one key, falling back to English and finally the key itself."""
    catalog = load_translations()
    selected = language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    template = catalog[selected].get(key, catalog[DEFAULT_LANGUAGE].get(key, key))
    try:
        return template.format(**values)
    except (KeyError, ValueError):
        return template


def class_label(class_id: str, language: str) -> str:
    """Translate a supported internal class without changing the identifier."""
    return t(f"class_{class_id}", language)


def action_label(action_id: str, language: str) -> str:
    """Translate a recommended-action identifier for display only."""
    return t(f"action_{action_id}", language)


def risk_label(level_id: str, language: str) -> str:
    """Translate a risk-level identifier for display only."""
    return t(f"risk_{level_id}", language)


def category_label(category_id: str, language: str) -> str:
    """Translate a security category while preserving unknown codes."""
    return t(f"category_{category_id}", language)


def translation_key_difference() -> dict[str, set[str]]:
    """Return language-specific missing keys for automated validation."""
    catalog = load_translations()
    english = set(catalog["en"])
    vietnamese = set(catalog["vi"])
    return {
        "missing_in_en": vietnamese - english,
        "missing_in_vi": english - vietnamese,
    }
