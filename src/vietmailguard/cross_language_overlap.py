"""Evidence-based English/Vietnamese translation-overlap auditing."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from vietmailguard.dataset_standardizer import extract_malformed_urls, extract_urls


TOKEN_PATTERN = re.compile(r"(?u)\b[\w.-]{4,}\b")
NUMBER_PATTERN = re.compile(r"\b\d{2,}\b")
IDENTIFIER_PATTERN = re.compile(
    r"(?i)\b(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9_-]{4,}\b"
)
LOOSE_EMAIL_PATTERN = re.compile(
    r"(?i)\b([a-z0-9_%+-]+(?:\s*\.\s*[a-z0-9_%+-]+)*)"
    r"\s*@\s*([a-z0-9-]+(?:\s*\.\s*[a-z0-9-]+)*)"
)
ENRON_PATTERN = re.compile(r"(?i)\b(?:enron|ect|vince kaminski|jeff dasovich)\b")

STOPWORDS = {
    "about",
    "after",
    "again",
    "been",
    "before",
    "being",
    "between",
    "could",
    "email",
    "from",
    "have",
    "into",
    "message",
    "more",
    "other",
    "please",
    "subject",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "with",
    "would",
    "your",
    "bang",
    "chung",
    "chu de",
    "cung",
    "dang",
    "duoc",
    "email",
    "khong",
    "mot",
    "nhung",
    "phai",
    "rằng",
    "sau",
    "theo",
    "thong",
    "trong",
    "truoc",
    "vien",
}


@dataclass(frozen=True)
class EvidenceProfile:
    """Language-invariant evidence extracted from one email."""

    text_length: int
    tokens: frozenset[str]
    subject_tokens: frozenset[str]
    identifiers: frozenset[str]
    numbers: frozenset[str]
    emails: frozenset[str]
    urls: frozenset[str]
    has_enron_identity: bool


def _ascii_token(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token)
    return "".join(char for char in normalized if not unicodedata.combining(char)).casefold()


def _shared_token_candidates(text: str) -> frozenset[str]:
    """Keep original ASCII anchors; do not transliterate Vietnamese prose."""
    tokens: set[str] = set()
    for match in TOKEN_PATTERN.finditer(text):
        raw = match.group(0).strip("._-")
        if not raw or not raw.isascii():
            continue
        token = _ascii_token(raw)
        if len(token) < 4 or token in STOPWORDS or token.isdigit():
            continue
        tokens.add(token)
    return frozenset(tokens)


def extract_loose_emails(text: object) -> frozenset[str]:
    """Extract standard and whitespace-obfuscated email addresses."""
    if pd.isna(text):
        return frozenset()
    emails = {
        f"{re.sub(r'\s+', '', match.group(1)).casefold()}@"
        f"{re.sub(r'\s+', '', match.group(2)).casefold()}"
        for match in LOOSE_EMAIL_PATTERN.finditer(str(text))
    }
    return frozenset(email for email in emails if len(email) >= 6)


def normalize_url_anchor(value: str) -> str:
    """Normalize URL spacing and common hxxp obfuscation for comparison only."""
    anchor = re.sub(r"\s+", "", value).casefold().rstrip(".,;:!?)]}>")
    if anchor.startswith("hxxps://"):
        anchor = "https://" + anchor[8:]
    elif anchor.startswith("hxxp://"):
        anchor = "http://" + anchor[7:]
    return anchor


def extract_url_anchors(text: object) -> frozenset[str]:
    """Extract comparable standard and malformed URL anchors."""
    values = [*extract_urls(text), *extract_malformed_urls(text)]
    return frozenset(normalize_url_anchor(value) for value in values if value)


def build_evidence_profile(text: object, subject: object = "") -> EvidenceProfile:
    """Build deterministic, language-invariant evidence for overlap matching."""
    full_text = "" if pd.isna(text) else str(text)
    subject_text = "" if pd.isna(subject) else str(subject)
    tokens = _shared_token_candidates(full_text)
    subject_tokens = _shared_token_candidates(subject_text or full_text[:240])
    return EvidenceProfile(
        text_length=len(full_text),
        tokens=tokens,
        subject_tokens=subject_tokens,
        identifiers=frozenset(
            match.group(0).casefold() for match in IDENTIFIER_PATTERN.finditer(full_text)
        ),
        numbers=frozenset(NUMBER_PATTERN.findall(full_text)),
        emails=extract_loose_emails(full_text),
        urls=extract_url_anchors(full_text),
        has_enron_identity=bool(ENRON_PATTERN.search(full_text)),
    )


def _document_frequency(profiles: list[EvidenceProfile], field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for profile in profiles:
        counter.update(getattr(profile, field))
    return counter


def _inverted_index(
    profiles: list[EvidenceProfile], field: str, maximum_df: int
) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_index, profile in enumerate(profiles):
        for value in getattr(profile, field):
            index[value].append(row_index)
    return {
        value: rows
        for value, rows in index.items()
        if len(rows) <= maximum_df
    }


def _overlap_score(
    english: EvidenceProfile,
    vietnamese: EvidenceProfile,
    english_df: dict[str, Counter[str]],
) -> tuple[float, dict[str, Any], list[str]]:
    shared_emails = sorted(english.emails & vietnamese.emails)
    shared_urls = sorted(english.urls & vietnamese.urls)
    shared_identifiers = sorted(english.identifiers & vietnamese.identifiers)
    shared_numbers = sorted(english.numbers & vietnamese.numbers)
    shared_tokens = sorted(english.tokens & vietnamese.tokens)
    shared_subject_tokens = sorted(english.subject_tokens & vietnamese.subject_tokens)

    points = 0.0
    if shared_emails:
        points += sum(
            5.0 if english_df["emails"][value] <= 3 else 3.0
            for value in shared_emails[:3]
        )
    if shared_urls:
        points += sum(
            6.0 if english_df["urls"][value] <= 3 else 4.0
            for value in shared_urls[:3]
        )
    if shared_identifiers:
        points += min(6.0, 2.0 * len(shared_identifiers))
    if shared_tokens:
        token_points = 0.0
        for value in shared_tokens[:10]:
            frequency = english_df["tokens"][value]
            token_points += 2.5 if frequency == 1 else 1.75 if frequency <= 3 else 1.0
        points += min(10.0, token_points)
    if shared_numbers:
        number_points = sum(
            1.5 if english_df["numbers"][value] == 1 else 0.75
            for value in shared_numbers[:8]
        )
        points += min(5.0, number_points)
    if shared_subject_tokens:
        points += min(4.0, 1.25 * len(shared_subject_tokens))

    length_ratio = (
        vietnamese.text_length / english.text_length if english.text_length else 0.0
    )
    if 0.55 <= length_ratio <= 2.6:
        points += 0.5

    reasons: list[str] = []
    if shared_emails:
        reasons.append("matching_email_addresses")
    if shared_urls:
        reasons.append("matching_urls")
    if shared_identifiers:
        reasons.append("preserved_identifiers")
    if shared_tokens:
        reasons.append("uncommon_shared_proper_nouns_or_tokens")
    if shared_numbers:
        reasons.append("matching_numbers_or_dates")
    if shared_subject_tokens:
        reasons.append("matching_subject_structure_anchors")
    if english.has_enron_identity and vietnamese.has_enron_identity:
        reasons.append("enron_identity")

    evidence = {
        "score": round(min(1.0, points / 15.0), 6),
        "points": round(points, 3),
        "shared_emails": shared_emails[:5],
        "shared_urls": shared_urls[:5],
        "shared_identifiers": shared_identifiers[:8],
        "shared_tokens": shared_tokens[:12],
        "shared_subject_tokens": shared_subject_tokens[:8],
        "shared_numbers": shared_numbers[:10],
        "length_ratio": round(length_ratio, 4),
    }
    return points, evidence, reasons


def _full_text(frame: pd.DataFrame, language: str) -> pd.Series:
    if language == "vi" and "original_text" in frame:
        return frame["original_text"].fillna("").astype(str)
    fields = [field for field in ("sender", "receiver", "date", "subject", "body") if field in frame]
    result = pd.Series("", index=frame.index, dtype="object")
    for field in fields:
        result = result + " " + frame[field].fillna("").astype(str)
    return result.str.strip()


def _translation_group_id(english_id: str) -> str:
    digest = hashlib.sha256(english_id.encode("utf-8")).hexdigest()[:20]
    return f"xlang_evidence_{digest}"


def audit_cross_language_overlap(
    english: pd.DataFrame,
    vietnamese: pd.DataFrame,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Flag candidate overlaps and link only uniquely strong evidence matches."""
    required_english = {"id", "source", "subject", "body"}
    required_vietnamese = {
        "id",
        "source",
        "body",
        "parent_id",
        "translation_source_id",
    }
    if missing := required_english - set(english.columns):
        raise ValueError(f"English master is missing columns: {sorted(missing)}")
    if missing := required_vietnamese - set(vietnamese.columns):
        raise ValueError(f"Vietnamese master is missing columns: {sorted(missing)}")

    english_text = _full_text(english, "en")
    vietnamese_text = _full_text(vietnamese, "vi")
    english_profiles = [
        build_evidence_profile(text, english.iloc[index].get("subject", ""))
        for index, text in enumerate(english_text)
    ]
    vietnamese_profiles = [
        build_evidence_profile(text, str(text)[:240]) for text in vietnamese_text
    ]

    fields_and_limits = {
        "tokens": int(policy["rare_token_max_english_df"]),
        "identifiers": int(policy["rare_token_max_english_df"]),
        "numbers": int(policy["number_max_english_df"]),
        "urls": int(policy["url_max_english_df"]),
        "emails": int(policy["email_max_english_df"]),
    }
    indexes = {
        field: _inverted_index(english_profiles, field, limit)
        for field, limit in fields_and_limits.items()
    }
    english_df = {
        field: _document_frequency(english_profiles, field)
        for field in fields_and_limits
    }

    minimum_points = float(policy["minimum_candidate_points"])
    medium_points = float(policy["medium_confidence_points"])
    high_points = float(policy["high_confidence_points"])
    high_margin = float(policy["high_confidence_min_margin"])
    max_candidates = int(policy["maximum_candidates_per_vietnamese_row"])
    records: list[dict[str, object]] = []
    high_links: dict[str, tuple[str, str]] = {}
    exact_provenance_links = 0

    english_id_to_index = {
        str(row_id): index for index, row_id in enumerate(english["id"])
    }
    for vi_index, vi_row in vietnamese.iterrows():
        existing_parent = str(vi_row.get("parent_id", "")).strip()
        vi_id = str(vi_row["id"])
        existing_group = str(vi_row.get("translation_source_id", "")).strip()
        generated_evidence_link = existing_group.startswith("xlang_evidence_")
        if (
            existing_parent
            and existing_parent in english_id_to_index
            and not generated_evidence_link
        ):
            en_index = english_id_to_index[existing_parent]
            en_row = english.iloc[en_index]
            evidence = {
                "score": 1.0,
                "points": None,
                "existing_parent_id": existing_parent,
            }
            records.append(
                {
                    "english_id": existing_parent,
                    "vietnamese_id": vi_id,
                    "english_source": str(en_row["source"]),
                    "vietnamese_source": str(vi_row["source"]),
                    "similarity/evidence": json.dumps(evidence, ensure_ascii=False),
                    "overlap_reason": "existing_parent_id_provenance",
                    "confidence": "high",
                    "review_status": "linked_exact_provenance",
                }
            )
            high_links[vi_id] = (
                existing_parent,
                existing_group or _translation_group_id(existing_parent),
            )
            exact_provenance_links += 1
            continue

        profile = vietnamese_profiles[vi_index]
        candidate_indexes: set[int] = set()
        for field in fields_and_limits:
            for value in getattr(profile, field):
                candidate_indexes.update(indexes[field].get(value, []))

        scored: list[tuple[float, int, dict[str, Any], list[str]]] = []
        for en_index in candidate_indexes:
            points, evidence, reasons = _overlap_score(
                english_profiles[en_index], profile, english_df
            )
            if points >= minimum_points:
                scored.append((points, en_index, evidence, reasons))
        scored.sort(key=lambda item: (-item[0], str(english.iloc[item[1]]["id"])))
        if not scored:
            continue

        top_points = scored[0][0]
        second_points = scored[1][0] if len(scored) > 1 else 0.0
        top_margin = top_points - second_points
        for rank, (points, en_index, evidence, reasons) in enumerate(
            scored[:max_candidates], start=1
        ):
            en_row = english.iloc[en_index]
            evidence["candidate_rank"] = rank
            evidence["top_candidate_margin_points"] = round(top_margin, 3)
            raw_label = str(vi_row.get("raw_label", "")).strip()
            english_label = str(en_row.get("label", "")).strip()
            label_compatible = (
                (raw_label == "0" and english_label == "normal")
                or (raw_label == "1" and english_label in {"spam", "phishing"})
            )
            evidence["label_compatibility_check"] = label_compatible
            evidence_types = sum(
                bool(evidence[key])
                for key in (
                    "shared_emails",
                    "shared_urls",
                    "shared_identifiers",
                    "shared_tokens",
                    "shared_numbers",
                    "shared_subject_tokens",
                )
            )
            length_ratio_is_plausible = 0.55 <= evidence["length_ratio"] <= 2.6
            direct_anchor_combo = bool(
                (evidence["shared_emails"] or evidence["shared_urls"])
                and (
                    evidence["shared_tokens"]
                    or evidence["shared_identifiers"]
                    or len(evidence["shared_numbers"]) >= 2
                )
            )
            structural_anchor_combo = bool(
                len(evidence["shared_tokens"]) >= 5
                and len(evidence["shared_subject_tokens"]) >= 1
                and len(evidence["shared_numbers"]) >= 2
            )
            source_supported_combo = bool(
                str(en_row["source"]) == "Enron"
                and len(evidence["shared_tokens"]) >= 6
                and len(evidence["shared_numbers"]) >= 3
            )
            uniquely_strong = (
                rank == 1
                and points >= high_points
                and top_margin >= high_margin
                and evidence_types >= 2
                and length_ratio_is_plausible
                and label_compatible
                and str(en_row["source"]) == "Enron"
                and (
                    direct_anchor_combo
                    or structural_anchor_combo
                    or source_supported_combo
                )
            )
            if uniquely_strong:
                confidence = "high"
                review_status = "linked_high_confidence_evidence"
                english_id = str(en_row["id"])
                high_links[vi_id] = (english_id, _translation_group_id(english_id))
            elif points >= medium_points:
                confidence = "medium"
                review_status = "candidate_needs_review"
            else:
                confidence = "low"
                review_status = "uncertain"
            records.append(
                {
                    "english_id": str(en_row["id"]),
                    "vietnamese_id": vi_id,
                    "english_source": str(en_row["source"]),
                    "vietnamese_source": str(vi_row["source"]),
                    "similarity/evidence": json.dumps(evidence, ensure_ascii=False),
                    "overlap_reason": ";".join(reasons),
                    "confidence": confidence,
                    "review_status": review_status,
                }
            )

    overlap = pd.DataFrame(
        records,
        columns=[
            "english_id",
            "vietnamese_id",
            "english_source",
            "vietnamese_source",
            "similarity/evidence",
            "overlap_reason",
            "confidence",
            "review_status",
        ],
    )
    updated_vietnamese = vietnamese.copy()
    generated_link_mask = updated_vietnamese["translation_source_id"].astype(str).str.startswith(
        "xlang_evidence_"
    )
    updated_vietnamese.loc[generated_link_mask, ["parent_id", "translation_source_id"]] = ""
    for vi_id, (parent_id, translation_source_id) in high_links.items():
        mask = updated_vietnamese["id"].astype(str).eq(vi_id)
        updated_vietnamese.loc[mask, "parent_id"] = parent_id
        updated_vietnamese.loc[mask, "translation_source_id"] = translation_source_id

    status_counts = overlap["review_status"].value_counts().to_dict()
    high_ids = set(
        overlap.loc[
            overlap["review_status"].isin(
                {"linked_exact_provenance", "linked_high_confidence_evidence"}
            ),
            "vietnamese_id",
        ]
    )
    medium_ids = set(
        overlap.loc[
            overlap["review_status"].eq("candidate_needs_review"), "vietnamese_id"
        ]
    ) - high_ids
    uncertain_ids = set(
        overlap.loc[overlap["review_status"].eq("uncertain"), "vietnamese_id"]
    ) - high_ids - medium_ids
    return {
        "overlap": overlap,
        "updated_vietnamese": updated_vietnamese,
        "exact_provenance_links": exact_provenance_links,
        "high_confidence_links": len(high_links) - exact_provenance_links,
        "medium_candidates": int(status_counts.get("candidate_needs_review", 0)),
        "uncertain_candidates": int(status_counts.get("uncertain", 0)),
        "candidate_rows": len(overlap),
        "vietnamese_rows_with_candidates": int(overlap["vietnamese_id"].nunique()),
        "high_confidence_vietnamese_rows": len(high_ids),
        "medium_only_vietnamese_rows": len(medium_ids),
        "uncertain_only_vietnamese_rows": len(uncertain_ids),
        "vietnamese_rows_without_candidates": len(vietnamese)
        - int(overlap["vietnamese_id"].nunique()),
        "semantic_model_used": False,
    }


def apply_cross_language_split_groups(
    english: pd.DataFrame,
    vietnamese: pd.DataFrame,
    overlap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign one future split group to every evidence-linked translation pair."""
    linked_statuses = {
        "linked_exact_provenance",
        "linked_high_confidence_evidence",
    }
    links = overlap.loc[overlap["review_status"].isin(linked_statuses)].copy()
    english_grouped = english.copy()
    vietnamese_grouped = vietnamese.copy()
    english_grouped["cross_language_group_id"] = english_grouped["group_id"].astype(str)
    vietnamese_grouped["cross_language_group_id"] = vietnamese_grouped[
        "group_id"
    ].astype(str)
    for link in links.itertuples(index=False):
        english_id = str(link.english_id)
        vietnamese_id = str(link.vietnamese_id)
        cross_group = _translation_group_id(english_id)
        english_grouped.loc[
            english_grouped["id"].astype(str).eq(english_id),
            "cross_language_group_id",
        ] = cross_group
        vietnamese_grouped.loc[
            vietnamese_grouped["id"].astype(str).eq(vietnamese_id),
            "cross_language_group_id",
        ] = cross_group
    return english_grouped, vietnamese_grouped
