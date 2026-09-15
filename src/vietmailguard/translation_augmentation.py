"""Controlled English-to-Vietnamese augmentation with parent provenance."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import pandas as pd

from vietmailguard.dataset_standardizer import (
    EMAIL_PATTERN,
    TRAILING_URL_PUNCTUATION,
    URL_PATTERN,
    content_hash,
    extract_urls,
    normalize_text,
)


DOMAIN_PATTERN = re.compile(
    r"(?i)(?<![@\w.-])(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|biz|info|vn|uk|us|de|fr|au|ca)(?![\w.-])"
)
DATE_PATTERN = re.compile(
    r"(?i)\b(?:\d{1,4}[/-]\d{1,2}[/-]\d{1,4}|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b"
)
AMOUNT_PATTERN = re.compile(
    r"(?i)(?:[$€£¥]\s?\d[\d,.]*(?:\s?(?:usd|eur|gbp))?|"
    r"\b\d[\d,.]*\s?(?:usd|eur|gbp|dollars?|euros?|pounds?)\b)"
)
ACCOUNT_ID_PATTERN = re.compile(
    r"(?i)\b(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9]+(?:[-_/][a-z0-9]+)+\b"
)
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,:/-]\d+)*(?!\w)")
MULTIWORD_PROPER_NOUN_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z'’-]{2,})(?:\s+(?:[A-Z][A-Za-z'’-]{2,})){1,4}\b"
)
KNOWN_NAME_PATTERN = re.compile(
    r"(?i)\b(?:Enron|PayPal|Microsoft|Apple|Amazon|eBay|Google|Yahoo|"
    r"Mastercard|Visa|Citibank|Wells Fargo|Bank of America|SpamAssassin|"
    r"Nazario)\b"
)
ALL_CAPS_PATTERN = re.compile(r"\b[A-Z]{2,}(?:[A-Z0-9_-]+)?\b")
PLACEHOLDER_PATTERN = re.compile(r"__VMGPROTECTED\d{5}__")

ENGLISH_COMMON_WORDS = {
    "the", "and", "your", "you", "this", "that", "from", "with", "for",
    "have", "will", "please", "account", "email", "message", "our", "are",
    "was", "not", "can", "all", "has", "but", "would", "should", "been",
}
VIETNAMESE_COMMON_WORDS = {
    "và", "của", "bạn", "cho", "là", "được", "không", "trong", "này", "với",
    "chúng", "tôi", "một", "có", "đã", "sẽ", "vui", "lòng", "tài", "khoản",
    "thông", "tin", "để", "những", "các", "khi", "từ", "về", "người", "theo",
}


@dataclass(frozen=True)
class ProtectedText:
    """Masked text plus exact values that must survive translation."""

    masked_text: str
    placeholders: dict[str, str]
    kinds: dict[str, str]


@dataclass(frozen=True)
class TranslationResult:
    """One restored translation and reproducibility metadata."""

    text: str
    protected_count: int
    protected_values_restored: bool
    protected_segment_strategy_used: bool


class BatchTranslator(Protocol):
    """Minimal interface used by the augmentation pipeline and tests."""

    model_name: str
    revision: str
    device_name: str

    def translate_batch(self, texts: Sequence[str]) -> list[TranslationResult]:
        """Translate texts deterministically while restoring protected values."""


class MarianOfflineTranslator:
    """Local deterministic MarianMT backend loaded from a pinned revision."""

    def __init__(
        self,
        model_config: dict[str, Any],
        *,
        local_files_only: bool = False,
        force_device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Translation dependencies are missing. Install requirements-translation.txt."
            ) from error

        self._torch = torch
        self.model_name = str(model_config["name"])
        self.revision = str(model_config["revision"])
        self.target_prefix = str(model_config["target_prefix"])
        self.max_input_tokens = int(model_config["max_input_tokens"])
        self.batch_size = int(model_config["batch_size"])
        self.generation = dict(model_config["generation"])
        self.tokenizer = MarianTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            local_files_only=local_files_only,
        )
        self.model = MarianMTModel.from_pretrained(
            self.model_name,
            revision=self.revision,
            local_files_only=local_files_only,
        )
        if force_device:
            device = force_device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_name = device
        self.model.to(device)
        self.model.eval()

    def _token_length(self, text: str) -> int:
        return len(
            self.tokenizer.encode(
                f"{self.target_prefix} {text}", add_special_tokens=True
            )
        )

    def _split_oversized_piece(self, piece: str) -> list[str]:
        words = piece.split()
        if not words:
            return []
        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            if self._token_length(word) > self.max_input_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                remaining = word
                while remaining:
                    low, high = 1, len(remaining)
                    best = 1
                    while low <= high:
                        middle = (low + high) // 2
                        if self._token_length(remaining[:middle]) <= self.max_input_tokens:
                            best = middle
                            low = middle + 1
                        else:
                            high = middle - 1
                    chunks.append(remaining[:best])
                    remaining = remaining[best:]
                continue
            candidate = " ".join([*current, word])
            if current and self._token_length(candidate) > self.max_input_tokens:
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _chunks(self, text: str) -> list[str]:
        normalized = normalize_text(text, "NFC")
        if not normalized:
            return []
        pieces = re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized)
        expanded: list[str] = []
        for piece in pieces:
            if not piece.strip():
                continue
            if (
                len(piece) <= self.max_input_tokens * 2
                and self._token_length(piece) <= self.max_input_tokens
            ):
                expanded.append(piece.strip())
            else:
                expanded.extend(self._split_oversized_piece(piece))

        chunks: list[str] = []
        current = ""
        for piece in expanded:
            candidate = f"{current} {piece}".strip()
            if current and self._token_length(candidate) > self.max_input_tokens:
                chunks.append(current)
                current = piece
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _translate_chunks(self, chunks: Sequence[str]) -> list[str]:
        if not chunks:
            return []
        outputs: list[str] = []
        torch = self._torch
        for start in range(0, len(chunks), self.batch_size):
            batch = [f"{self.target_prefix} {item}" for item in chunks[start:start + self.batch_size]]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.device_name)
            with torch.inference_mode():
                generated = self.model.generate(**encoded, **self.generation)
            outputs.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return [normalize_text(value, "NFC") for value in outputs]

    def _translate_masked_many(self, masked_texts: Sequence[str]) -> list[str]:
        owner_chunks: list[list[int]] = []
        flattened: list[str] = []
        for text in masked_texts:
            indexes: list[int] = []
            for chunk in self._chunks(text):
                indexes.append(len(flattened))
                flattened.append(chunk)
            owner_chunks.append(indexes)
        translations = self._translate_chunks(flattened)
        return [
            normalize_text(" ".join(translations[index] for index in indexes), "NFC")
            for indexes in owner_chunks
        ]

    def translate_batch(self, texts: Sequence[str]) -> list[TranslationResult]:
        """Translate locally while never sending placeholders through generation."""
        protected_texts = [protect_text(text) for text in texts]
        layouts: list[list[tuple[str, int | str]]] = []
        translatable: list[str] = []
        for protected in protected_texts:
            layout: list[tuple[str, int | str]] = []
            parts = re.split(f"({PLACEHOLDER_PATTERN.pattern})", protected.masked_text)
            for part in parts:
                if not part:
                    continue
                if PLACEHOLDER_PATTERN.fullmatch(part):
                    layout.append(("placeholder", part))
                elif part.strip():
                    layout.append(("translation", len(translatable)))
                    translatable.append(part)
            layouts.append(layout)
        translated_segments = self._translate_masked_many(translatable)

        results: list[TranslationResult] = []
        for protected, layout in zip(protected_texts, layouts, strict=True):
            rebuilt = " ".join(
                str(value) if kind == "placeholder" else translated_segments[int(value)]
                for kind, value in layout
            )
            restored, missing = restore_protected_text(rebuilt, protected)
            results.append(
                TranslationResult(
                    text=restored,
                    protected_count=len(protected.placeholders),
                    protected_values_restored=not missing,
                    protected_segment_strategy_used=bool(protected.placeholders),
                )
            )
        return results


PROTECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("url", URL_PATTERN),
    ("email", EMAIL_PATTERN),
    ("domain", DOMAIN_PATTERN),
    ("amount", AMOUNT_PATTERN),
    ("date", DATE_PATTERN),
    ("account_identifier", ACCOUNT_ID_PATTERN),
    ("number", NUMBER_PATTERN),
    ("proper_noun", MULTIWORD_PROPER_NOUN_PATTERN),
    ("proper_noun", KNOWN_NAME_PATTERN),
    ("proper_noun", ALL_CAPS_PATTERN),
)


def load_augmentation_config(path: Path) -> dict[str, Any]:
    """Load and validate the controlled-translation configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "input", "output", "seed", "target_per_class", "allowed_parent_labels",
        "allowed_parent_provenance", "required_parent_split", "model", "quality",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Augmentation config is missing: {sorted(missing)}")
    return config


def _protected_spans(text: str) -> list[tuple[int, int, str]]:
    occupied: list[tuple[int, int]] = []
    selected: list[tuple[int, int, str]] = []
    for kind, pattern in PROTECTION_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if kind == "url":
                while end > start and text[end - 1] in TRAILING_URL_PUNCTUATION:
                    end -= 1
                if end == start:
                    continue
            if any(start < other_end and end > other_start for other_start, other_end in occupied):
                continue
            occupied.append((start, end))
            selected.append((start, end, kind))
    return sorted(selected)


def protect_text(text: object) -> ProtectedText:
    """Replace immutable values with deterministic placeholders."""
    source = "" if pd.isna(text) else unicodedata.normalize("NFC", str(text))
    placeholders: dict[str, str] = {}
    kinds: dict[str, str] = {}
    parts: list[str] = []
    cursor = 0
    for index, (start, end, kind) in enumerate(_protected_spans(source)):
        placeholder = f"__VMGPROTECTED{index:05d}__"
        parts.append(source[cursor:start])
        parts.append(placeholder)
        placeholders[placeholder] = source[start:end]
        kinds[placeholder] = kind
        cursor = end
    parts.append(source[cursor:])
    return ProtectedText("".join(parts), placeholders, kinds)


def restore_protected_text(
    translated: str, protected: ProtectedText
) -> tuple[str, list[str]]:
    """Restore exact values and return any placeholders lost by translation."""
    restored = translated
    missing: list[str] = []
    for placeholder, original in protected.placeholders.items():
        if placeholder not in restored:
            missing.append(placeholder)
            continue
        restored = restored.replace(placeholder, original)
    return unicodedata.normalize("NFC", restored), missing


def _stable_order_key(seed: int, row_id: str) -> str:
    return hashlib.sha256(f"{seed}|{row_id}".encode("utf-8")).hexdigest()


def select_parent_rows(
    train: pd.DataFrame,
    config: dict[str, Any],
    existing_vietnamese: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select unique high-confidence train parents with source diversity."""
    required = {
        "id", "subject", "body", "label", "raw_label", "source", "language",
        "group_id", "label_provenance",
    }
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Training split is missing columns: {sorted(missing)}")

    allowed_labels = set(config["allowed_parent_labels"])
    frame = train.loc[
        train["label"].isin(allowed_labels) & train["language"].eq("en")
    ].copy()
    provenance_mask = pd.Series(False, index=frame.index)
    for label, provenances in config["allowed_parent_provenance"].items():
        provenance_mask |= frame["label"].eq(label) & frame["label_provenance"].isin(
            provenances
        )
    frame = frame.loc[provenance_mask]
    frame = frame.loc[
        frame["id"].astype(str).str.strip().ne("")
        & (
            frame["subject"].astype(str).str.strip().ne("")
            | frame["body"].astype(str).str.strip().ne("")
        )
    ]
    frame = frame.drop_duplicates("id", keep="first")

    existing_parent_ids: set[str] = set()
    existing_parent_overlap_removed = 0
    if (
        config.get("exclude_existing_translated_parents", True)
        and existing_vietnamese is not None
        and "parent_id" in existing_vietnamese.columns
    ):
        existing_parent_ids = set(
            existing_vietnamese["parent_id"].astype(str).str.strip().replace("", pd.NA).dropna()
        )
        existing_parent_overlap_removed = int(
            frame["id"].astype(str).isin(existing_parent_ids).sum()
        )
        frame = frame.loc[~frame["id"].astype(str).isin(existing_parent_ids)]

    seed = int(config["seed"])
    frame["_stable_order"] = frame["id"].astype(str).map(
        lambda value: _stable_order_key(seed, value)
    )
    frame = frame.sort_values("_stable_order", kind="stable")
    before_group_filter = frame.groupby("label").size().to_dict()
    if config.get("one_parent_per_group", True):
        group_key = frame["group_id"].astype(str).where(
            frame["group_id"].astype(str).str.strip().ne(""), frame["id"].astype(str)
        )
        frame = frame.assign(_parent_group_key=group_key).drop_duplicates(
            "_parent_group_key", keep="first"
        )

    target = int(config["target_per_class"])
    selections: list[pd.DataFrame] = []
    available_after_filters: dict[str, int] = {}
    for label in config["allowed_parent_labels"]:
        label_frame = frame.loc[frame["label"].eq(label)].copy()
        available_after_filters[label] = len(label_frame)
        queues: dict[str, deque[int]] = {
            str(source): deque(group.index.tolist())
            for source, group in label_frame.groupby("source", sort=True)
        }
        selected_indexes: list[int] = []
        sources = sorted(queues)
        while len(selected_indexes) < target and any(queues.values()):
            for source in sources:
                if queues[source] and len(selected_indexes) < target:
                    selected_indexes.append(queues[source].popleft())
        selections.append(label_frame.loc[selected_indexes])

    selected = pd.concat(selections, ignore_index=True) if selections else frame.iloc[0:0]
    selected = selected.drop(columns=["_stable_order", "_parent_group_key"], errors="ignore")
    selected = selected.sort_values(["label", "source", "id"], kind="stable").reset_index(drop=True)
    audit = {
        "eligible_before_group_filter": before_group_filter,
        "eligible_after_filters": available_after_filters,
        "known_existing_translated_parent_ids": len(existing_parent_ids),
        "existing_parent_overlap_removed": existing_parent_overlap_removed,
        "selected_by_class": selected["label"].value_counts().to_dict(),
        "selected_by_class_source": {
            f"{label}|{source}": int(count)
            for (label, source), count in selected.groupby(["label", "source"]).size().items()
        },
    }
    return selected, audit


def _exact_counter(values: Sequence[str], text: str) -> Counter[str]:
    return Counter({value: text.count(value) for value in values})


def _counter_contains(observed: Counter[str], expected: Counter[str]) -> bool:
    return all(observed[value] >= count for value, count in expected.items())


def quality_flags(
    source_text: str,
    translated_text: str,
    protected: ProtectedText,
    config: dict[str, Any],
) -> list[str]:
    """Return deterministic translation-quality flags."""
    flags: list[str] = []
    target = unicodedata.normalize("NFC", translated_text)
    if not target.strip():
        flags.append("empty_output")
        return flags
    if "\ufffd" in target or not unicodedata.is_normalized("NFC", translated_text):
        flags.append("broken_unicode")

    source_urls = Counter(extract_urls(source_text))
    target_urls = Counter(extract_urls(target))
    if source_urls != target_urls:
        flags.append("broken_urls")
    source_emails = Counter(EMAIL_PATTERN.findall(source_text))
    target_emails = Counter(EMAIL_PATTERN.findall(target))
    if not _counter_contains(target_emails, source_emails):
        flags.append("lost_email_addresses")
    # URLs and email addresses are validated in full above. Only compare
    # standalone domains selected by the non-overlapping protection pass here;
    # otherwise malformed source prose such as ``suspension.www.example.com``
    # can be mistaken for a lost domain while its preserved URL is intact.
    source_domains = Counter(
        protected.placeholders[placeholder]
        for placeholder, kind in protected.kinds.items()
        if kind == "domain"
    )
    target_domains = _exact_counter(list(source_domains), target)
    if not _counter_contains(target_domains, source_domains):
        flags.append("lost_domains")

    originals = list(protected.placeholders.values())
    expected = Counter(originals)
    observed = _exact_counter(list(expected), target)
    if any(observed[value] < count for value, count in expected.items()):
        flags.append("lost_protected_values")

    source_length = max(len(normalize_text(source_text)), 1)
    ratio = len(normalize_text(target)) / source_length
    quality = config["quality"]
    if ratio < float(quality["minimum_length_ratio"]):
        flags.append("translation_too_short")
    if ratio > float(quality["maximum_length_ratio"]):
        flags.append("translation_excessively_long")

    target_words = re.findall(r"(?u)\b\w+\b", target.casefold())
    english_hits = sum(word in ENGLISH_COMMON_WORDS for word in target_words)
    vietnamese_hits = sum(word in VIETNAMESE_COMMON_WORDS for word in target_words)
    has_vietnamese_diacritics = any(
        character in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        for character in target.casefold()
    )
    if (
        normalize_text(source_text).casefold() == normalize_text(target).casefold()
        or (
            english_hits >= int(quality["untranslated_english_min_hits"])
            and english_hits >= float(quality["untranslated_english_dominance"])
            * max(vietnamese_hits, 1)
            and not has_vietnamese_diacritics
        )
    ):
        flags.append("untranslated_english_body")
    return flags


def _flag_is_critical(flag: str, critical_flags: set[str]) -> bool:
    return flag.split(":", maxsplit=1)[0] in critical_flags


def deterministic_translation_id(parent_id: str, model_revision: str) -> str:
    """Build a stable augmentation ID from parent and model revision."""
    digest = hashlib.sha256(f"{model_revision}|{parent_id}".encode("utf-8")).hexdigest()[:20]
    return f"aug_vi:{digest}"


def translation_config_fingerprint(config: dict[str, Any]) -> str:
    """Hash every setting that materially affects generated output or eligibility."""
    material = {
        "model": config["model"],
        "quality": config["quality"],
        "output_metadata": config["output_metadata"],
    }
    serialized = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_augmented_row(
    parent: pd.Series,
    subject_result: TranslationResult,
    body_result: TranslationResult,
    subject_protected: ProtectedText,
    body_protected: ProtectedText,
    config: dict[str, Any],
    *,
    translation_error: str = "",
    translation_device: str = "",
) -> dict[str, Any]:
    """Build one provenance-preserving augmentation row and its quality flags."""
    subject = normalize_text(subject_result.text, "NFC")
    body = normalize_text(body_result.text, "NFC")
    flags = quality_flags(
        str(parent.get("body", "")), body, body_protected, config
    )
    subject_flags = quality_flags(
        str(parent.get("subject", "")), subject, subject_protected, config
    ) if str(parent.get("subject", "")).strip() else []
    for flag in subject_flags:
        scoped = f"subject_{flag}"
        if flag in {"broken_unicode", "broken_urls", "lost_email_addresses", "lost_domains", "lost_protected_values"}:
            flags.append(flag)
        elif flag == "untranslated_english_body":
            flags.append("untranslated_english_subject")
        else:
            flags.append(scoped)
    if not subject_result.protected_values_restored or not body_result.protected_values_restored:
        flags.append("lost_protected_values")
    if translation_error:
        flags.append(f"translation_error:{translation_error}")
    flags = list(dict.fromkeys(flags))

    critical_flags = set(config["quality"]["critical_flags"])
    accepted = not any(_flag_is_critical(flag, critical_flags) for flag in flags)
    parent_id = str(parent["id"])
    revision = str(config["model"]["revision"])
    parent_group = str(parent.get("group_id", "")).strip() or parent_id
    translation_source_id = f"opus-mt-en-vi@{revision}:{parent_id}"
    protected_count = subject_result.protected_count + body_result.protected_count
    segment_strategy_used = (
        subject_result.protected_segment_strategy_used
        or body_result.protected_segment_strategy_used
    )
    combined_source_length = max(
        len(normalize_text(f"{parent.get('subject', '')} {parent.get('body', '')}")), 1
    )
    combined_target_length = len(normalize_text(f"{subject} {body}"))
    return {
        "id": deterministic_translation_id(parent_id, revision),
        "parent_id": parent_id,
        "translation_source_id": translation_source_id,
        "parent_source": str(parent["source"]),
        "parent_label": str(parent["label"]),
        "parent_group_id": parent_group,
        "parent_split": str(config["required_parent_split"]),
        "parent_label_provenance": str(parent["label_provenance"]),
        "subject": subject,
        "body": body,
        "label": str(parent["label"]),
        "raw_label": str(parent["raw_label"]),
        "language": str(config["output_metadata"]["language"]),
        "data_origin": str(config["output_metadata"]["data_origin"]),
        "augmentation_type": str(config["output_metadata"]["augmentation_type"]),
        "content_hash": content_hash(subject, body),
        "group_id": parent_group,
        "label_provenance": f"controlled_translation_from_{parent['label_provenance']}",
        "translation_model": str(config["model"]["name"]),
        "translation_model_revision": revision,
        "translation_device": translation_device,
        "translation_config_hash": translation_config_fingerprint(config),
        "protected_token_count": protected_count,
        "protected_token_validation": "passed" if not any(
            flag in flags for flag in {
                "lost_email_addresses", "lost_domains", "lost_protected_values"
            }
        ) else "failed",
        "translation_length_ratio": round(combined_target_length / combined_source_length, 6),
        "translation_quality_status": "accepted" if accepted else "failed",
        "quality_flags": ";".join(flags),
        "protected_segment_strategy_used": segment_strategy_used,
        "training_eligible": accepted,
    }


def apply_duplicate_translation_flags(
    augmented: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Flag all duplicate translations and make them ineligible."""
    result = augmented.copy()
    duplicate_mask = result["content_hash"].astype(str).ne("") & result[
        "content_hash"
    ].duplicated(keep=False)
    for index in result.index[duplicate_mask]:
        flags = [item for item in str(result.at[index, "quality_flags"]).split(";") if item]
        if "duplicate_translated_output" not in flags:
            flags.append("duplicate_translated_output")
        result.at[index, "quality_flags"] = ";".join(flags)
        result.at[index, "translation_quality_status"] = "failed"
        result.at[index, "training_eligible"] = False
    return result


def revalidate_existing_translations(
    augmented: pd.DataFrame,
    parents: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Recompute quality metadata without calling the translation model."""
    parent_by_id = parents.set_index(parents["id"].astype(str), drop=False)
    refreshed: list[dict[str, Any]] = []
    for row in augmented.itertuples(index=False):
        parent_id = str(row.parent_id)
        if parent_id not in parent_by_id.index:
            raise ValueError(f"Augmentation parent missing from selection: {parent_id}")
        parent = parent_by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):
            raise ValueError(f"Selected parent id is not unique: {parent_id}")
        subject_protected = protect_text(parent["subject"])
        body_protected = protect_text(parent["body"])
        old_flags = [item for item in str(row.quality_flags).split(";") if item]
        translation_error = next(
            (
                item.split(":", maxsplit=1)[1]
                for item in old_flags
                if item.startswith("translation_error:")
            ),
            "",
        )
        refreshed.append(
            build_augmented_row(
                parent,
                TranslationResult(
                    str(row.subject),
                    len(subject_protected.placeholders),
                    True,
                    str(row.protected_segment_strategy_used).casefold() == "true",
                ),
                TranslationResult(
                    str(row.body),
                    len(body_protected.placeholders),
                    True,
                    str(row.protected_segment_strategy_used).casefold() == "true",
                ),
                subject_protected,
                body_protected,
                config,
                translation_error=translation_error,
                translation_device=str(getattr(row, "translation_device", "")),
            )
        )
    return pd.DataFrame(refreshed)


def validate_parent_split_safety(
    augmented: pd.DataFrame, required_split: str = "train"
) -> None:
    """Reject missing parents, duplicate parents, or split/group mismatches."""
    if augmented["parent_id"].astype(str).duplicated().any():
        raise ValueError("A parent_id appears more than once in the augmentation set")
    if not augmented["parent_split"].astype(str).eq(required_split).all():
        raise ValueError("An augmentation row is not assigned to its required parent split")
    if not augmented["group_id"].astype(str).eq(
        augmented["parent_group_id"].astype(str)
    ).all():
        raise ValueError("Augmentation group_id does not match parent_group_id")


def validation_sample(
    augmented: pd.DataFrame,
    parents: pd.DataFrame,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Create a deterministic, source-diverse human review sample."""
    accepted = augmented.loc[augmented["translation_quality_status"].eq("accepted")]
    parent_text = parents.set_index(parents["id"].astype(str)).apply(
        lambda row: normalize_text(f"{row.get('subject', '')}\n{row.get('body', '')}")[:800],
        axis=1,
    )
    rows: list[tuple[int, pd.Series]] = []
    for label in sorted(accepted["label"].unique()):
        label_frame = accepted.loc[accepted["label"].eq(label)].copy()
        label_frame["_order"] = label_frame["parent_id"].astype(str).map(
            lambda value: _stable_order_key(seed + 1, value)
        )
        label_frame = label_frame.sort_values("_order", kind="stable")
        queues = {
            str(source): deque(group.index.tolist())
            for source, group in label_frame.groupby("parent_source", sort=True)
        }
        chosen: list[int] = []
        while len(chosen) < per_class and any(queues.values()):
            for source in sorted(queues):
                if queues[source] and len(chosen) < per_class:
                    chosen.append(queues[source].popleft())
        rows.extend(label_frame.loc[chosen].iterrows())

    records: list[dict[str, str]] = []
    for _, row in rows:
        translation = normalize_text(f"{row['subject']}\n{row['body']}")[:800]
        records.append(
            {
                "parent English text excerpt": str(parent_text.get(str(row["parent_id"]), "")),
                "Vietnamese translation excerpt": translation,
                "class": str(row["label"]),
                "source": str(row["parent_source"]),
                "translation_quality_decision": "",
                "note": "",
            }
        )
    return pd.DataFrame(records)


def count_quality_flags(frame: pd.DataFrame) -> Counter[str]:
    """Count semicolon-delimited quality flags."""
    counts: Counter[str] = Counter()
    for value in frame["quality_flags"].astype(str):
        counts.update(item for item in value.split(";") if item)
    return counts
