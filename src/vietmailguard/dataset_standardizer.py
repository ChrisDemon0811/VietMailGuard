"""Configuration-driven standardization for the VietMailGuard email corpora."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vietmailguard.dataset_audit import read_dataset

URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
TRAILING_URL_PUNCTUATION = ".,;:!?)]}>"
MALFORMED_URL_CANDIDATE_PATTERN = re.compile(
    r"(?i)\b(?:hxxps?|https?)\s*:\s*/\s*/\s*[a-z0-9-]+"
    r"(?:\s*\.\s*[a-z0-9-]+)+"
    r"(?:\s*/\s*[a-z0-9._~!$&()*+=:@%-]+)*"
    r"|\bwww\s*\.\s*[a-z0-9-]+(?:\s*\.\s*[a-z0-9-]+)+"
    r"(?:\s*/\s*[a-z0-9._~!$&()*+=:@%-]+)*",
)
SPACED_EMAIL_PATTERN = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+\s*@\s*[A-Z0-9.-]+(?:\s*\.\s*[A-Z]{2,})\b"
)


@dataclass(frozen=True)
class LabelDecision:
    """Resolved training decision for one raw row."""

    label: str | None
    provenance: str
    policy_status: str
    reason: str


def normalize_text(value: object, unicode_form: str = "NFC") -> str:
    """Normalize Unicode and collapse all whitespace without lowercasing display text."""
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize(unicode_form, str(value)).split())


def is_empty_or_placeholder(value: object, placeholders: set[str]) -> bool:
    """Return true for null, whitespace-only, or configured whole-field placeholders."""
    normalized = normalize_text(value).casefold()
    return not normalized or normalized in {item.casefold() for item in placeholders}


def normalize_field(value: object, placeholders: set[str], unicode_form: str = "NFC") -> str:
    """Normalize a source field and represent missing placeholders as an empty value."""
    if is_empty_or_placeholder(value, placeholders):
        return ""
    return normalize_text(value, unicode_form)


def is_container_artifact(subject: object, body: object, patterns: list[str]) -> bool:
    """Detect explicitly configured non-message container records."""
    text = normalize_text(f"{'' if pd.isna(subject) else subject} {'' if pd.isna(body) else body}").casefold()
    return any(pattern.casefold() in text for pattern in patterns)


def extract_urls(text: object) -> list[str]:
    """Extract URL-like strings before content cleaning."""
    if pd.isna(text):
        return []
    return [match.rstrip(TRAILING_URL_PUNCTUATION) for match in URL_PATTERN.findall(str(text))]


def extract_malformed_urls(text: object) -> list[str]:
    """Return URL-like spans that are obfuscated or broken by whitespace."""
    if pd.isna(text):
        return []
    findings: list[str] = []
    for match in MALFORMED_URL_CANDIDATE_PATTERN.finditer(str(text)):
        candidate = match.group(0).rstrip(TRAILING_URL_PUNCTUATION)
        if URL_PATTERN.fullmatch(candidate):
            continue
        if candidate not in findings:
            findings.append(candidate)
    return findings


def normalized_duplicate_text(subject: object, body: object, *, template: bool = False) -> str:
    """Build a case-insensitive exact or URL/email-insensitive comparison representation."""
    text = normalize_text(f"{'' if pd.isna(subject) else subject}\n{'' if pd.isna(body) else body}", "NFKC")
    text = text.casefold()
    if template:
        text = URL_PATTERN.sub(" <url> ", text)
        text = EMAIL_PATTERN.sub(" <email> ", text)
        text = normalize_text(text, "NFKC")
    return text


def content_hash(subject: object, body: object) -> str:
    """Return the deterministic SHA-256 hash of normalized subject and body."""
    return hashlib.sha256(normalized_duplicate_text(subject, body).encode("utf-8")).hexdigest()


def template_hash(subject: object, body: object) -> str:
    """Hash content after replacing URLs and email addresses with placeholders."""
    return hashlib.sha256(
        normalized_duplicate_text(subject, body, template=True).encode("utf-8")
    ).hexdigest()


def obfuscated_template_hash(subject: object, body: object) -> str:
    """Hash a template after replacing whitespace-obfuscated URLs and emails."""
    text = normalized_duplicate_text(subject, body, template=True)
    text = MALFORMED_URL_CANDIDATE_PATTERN.sub(" <url> ", text)
    text = SPACED_EMAIL_PATTERN.sub(" <email> ", text)
    return hashlib.sha256(normalize_text(text, "NFKC").encode("utf-8")).hexdigest()


def parse_prefixed_subject(
    text: object,
    prefixes: list[str] | tuple[str, ...],
) -> tuple[str, str, bool]:
    """Split a subject header only when a line boundary makes it unambiguous."""
    if pd.isna(text):
        return "", "", False
    raw_text = unicodedata.normalize("NFC", str(text))
    prefix_pattern = "|".join(re.escape(prefix) for prefix in prefixes)
    match = re.match(
        rf"^\s*(?:{prefix_pattern})\s*:\s*([^\r\n]+)[\r\n]+([\s\S]+)$",
        raw_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return "", normalize_text(raw_text, "NFC"), False
    subject = normalize_text(match.group(1), "NFC")
    body = normalize_text(match.group(2), "NFC")
    if not subject or not body:
        return "", normalize_text(raw_text, "NFC"), False
    return subject, body, True


def normalize_raw_label(value: object) -> str:
    """Convert scalar labels to stable config keys without guessing their meaning."""
    if pd.isna(value):
        return "<MISSING>"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_curated_spam_keys(path: Path, override: dict[str, Any]) -> set[tuple[str, int]]:
    """Load only rows authorized by the frozen conservative spam policy."""
    if not path.exists():
        raise FileNotFoundError(f"Required curated spam report does not exist: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"source", "original_row_id", "candidate_group", "proposed_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Curated spam report is missing columns: {sorted(missing)}")
    selected = frame.loc[
        frame["candidate_group"].eq(override["candidate_group"])
        & frame["proposed_label"].eq(override["required_proposed_label"])
    ]
    return {(str(row.source), int(row.original_row_id)) for row in selected.itertuples(index=False)}


def resolve_label(
    raw_label: object,
    raw_label_policy: dict[str, dict[str, Any]],
    *,
    source: str,
    original_row_id: int,
    curated_spam_keys: set[tuple[str, int]],
    spam_override: dict[str, Any],
) -> LabelDecision:
    """Resolve an explicit source policy or a row-level curated-spam authorization."""
    key = normalize_raw_label(raw_label)
    decision = raw_label_policy.get(key)
    if decision is None:
        return LabelDecision(None, "", "review", "raw_label_missing_from_policy")
    status = decision["status"]
    if status in {"normal", "spam", "phishing"} and decision["training_eligible"]:
        return LabelDecision(
            status,
            str(decision["label_provenance"]),
            status,
            "verified_raw_label_policy",
        )
    if status == "review" and (source, original_row_id) in curated_spam_keys:
        return LabelDecision(
            str(spam_override["mapped_label"]),
            str(spam_override["label_provenance"]),
            "spam",
            "authorized_row_level_curation_override",
        )
    return LabelDecision(None, "", status, f"raw_label_policy_{status}")


def _field_value(row: pd.Series, source_column: str | None) -> object:
    return "" if source_column is None else row.get(source_column, "")


def _removed_record(
    row: dict[str, Any], reason: str, details: str = ""
) -> dict[str, Any]:
    return {
        "id": row.get("id", ""),
        "source": row.get("source", ""),
        "original_row_id": row.get("original_row_id", ""),
        "raw_label": row.get("raw_label", ""),
        "mapped_label": row.get("label", ""),
        "reason": reason,
        "details": details,
        "subject_excerpt": normalize_text(row.get("subject", ""))[:200],
        "content_hash": row.get("content_hash", ""),
        "group_id": row.get("group_id", ""),
    }


def standardize_datasets(
    raw_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Load, map, clean, deduplicate, and group all configured raw datasets."""
    standardization = config["policy"]["standardization"]
    spam_override = standardization["row_level_spam_override"]
    project_root = raw_dir.resolve().parents[1]
    curated_spam_keys = load_curated_spam_keys(project_root / spam_override["source_report"], spam_override)
    placeholders = set(standardization["placeholder_values"])
    unicode_form = str(standardization["unicode_form"])

    candidates: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, int | str]] = {}

    for filename, dataset in config["datasets"].items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Configured raw dataset does not exist: {path}")
        frame, _ = read_dataset(path)
        source = str(dataset["source_name"])
        field_mapping = dataset["field_mapping"]
        summary: dict[str, int | str] = {
            "source": source,
            "input_rows": len(frame),
            "mapped_candidates": 0,
            "review_rows": 0,
            "policy_excluded_rows": 0,
            "empty_rows_removed": 0,
            "artifact_rows_removed": 0,
            "conflict_rows_removed": 0,
            "exact_duplicates_removed": 0,
            "output_rows": 0,
            "normal": 0,
            "spam": 0,
            "phishing": 0,
        }
        raw_label_column = field_mapping["raw_label"]
        if raw_label_column is None or raw_label_column not in frame.columns:
            raise ValueError(f"No configured raw label column for {filename}")

        for index, raw_row in frame.iterrows():
            original_row_id = int(index) + 2
            raw_label = raw_row[raw_label_column]
            decision = resolve_label(
                raw_label,
                dataset["raw_label_policy"],
                source=source,
                original_row_id=original_row_id,
                curated_spam_keys=curated_spam_keys,
                spam_override=spam_override,
            )
            subject_raw = _field_value(raw_row, field_mapping.get("subject"))
            body_raw = _field_value(raw_row, field_mapping.get("body"))
            if decision.label is None:
                if decision.policy_status == "exclude":
                    summary["policy_excluded_rows"] = int(summary["policy_excluded_rows"]) + 1
                else:
                    summary["review_rows"] = int(summary["review_rows"]) + 1
                    review.append(
                        {
                            "source": source,
                            "original_row_id": original_row_id,
                            "raw_label": normalize_raw_label(raw_label),
                            "policy_status": decision.policy_status,
                            "reason": decision.reason,
                            "subject": normalize_text(subject_raw, unicode_form),
                            "body_excerpt": normalize_text(body_raw, unicode_form)[:500],
                        }
                    )
                continue

            normalized_fields = {
                name: normalize_field(
                    _field_value(raw_row, field_mapping.get(name)), placeholders, unicode_form
                )
                for name in ("sender", "receiver", "date", "subject", "body")
            }
            base = {
                "id": f"{source}:{original_row_id}",
                "original_row_id": original_row_id,
                **normalized_fields,
                "label": decision.label,
                "raw_label": normalize_raw_label(raw_label),
                "source": source,
                "language": str(dataset["language"]),
                "label_provenance": decision.provenance,
                "_order": len(candidates),
            }
            if standardization["body_is_required"] and is_empty_or_placeholder(body_raw, placeholders):
                summary["empty_rows_removed"] = int(summary["empty_rows_removed"]) + 1
                removed.append(_removed_record(base, "empty_or_placeholder_body"))
                continue
            if is_container_artifact(
                subject_raw, body_raw, list(standardization["container_artifact_patterns"])
            ):
                summary["artifact_rows_removed"] = int(summary["artifact_rows_removed"]) + 1
                removed.append(_removed_record(base, "container_artifact"))
                continue

            raw_content = f"{'' if pd.isna(subject_raw) else subject_raw}\n{'' if pd.isna(body_raw) else body_raw}"
            base["url_count"] = len(extract_urls(raw_content))
            base["content_hash"] = content_hash(base["subject"], base["body"])
            base["_template_hash"] = template_hash(base["subject"], base["body"])
            candidates.append(base)
            summary["mapped_candidates"] = int(summary["mapped_candidates"]) + 1
        summaries[source] = summary

    working = pd.DataFrame(candidates)
    exact_counts = Counter(working["content_hash"])
    template_counts = Counter(working["_template_hash"])
    working["group_id"] = working.apply(
        lambda row: (
            f"template_{row['_template_hash'][:20]}"
            if template_counts[row["_template_hash"]] > 1
            else f"content_{row['content_hash'][:20]}"
        ),
        axis=1,
    )
    group_label_counts = working.groupby("group_id")["label"].nunique()
    conflicting_groups = set(group_label_counts[group_label_counts > 1].index)

    duplicate_rows: list[dict[str, Any]] = []
    canonical_ids: dict[str, str] = {}
    for digest, group in working.groupby("content_hash", sort=False):
        canonical_ids[digest] = str(group.sort_values("_order").iloc[0]["id"])

    exact_remove_indices: set[int] = set()
    conflict_indices: set[int] = set()
    for index, row in working.iterrows():
        exact_duplicate = exact_counts[row["content_hash"]] > 1
        template_duplicate = template_counts[row["_template_hash"]] > 1
        conflict = row["group_id"] in conflicting_groups
        is_canonical = row["id"] == canonical_ids[row["content_hash"]]
        if conflict:
            action = "exclude_conflicting_template_labels"
            conflict_indices.add(index)
        elif exact_duplicate and not is_canonical:
            action = "remove_exact_duplicate_keep_canonical"
            exact_remove_indices.add(index)
        elif exact_duplicate:
            action = "keep_exact_canonical"
        elif template_duplicate:
            action = "keep_template_grouped"
        else:
            action = "unique"
        if exact_duplicate or template_duplicate:
            duplicate_rows.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "original_row_id": row["original_row_id"],
                    "label": row["label"],
                    "content_hash": row["content_hash"],
                    "template_hash": row["_template_hash"],
                    "group_id": row["group_id"],
                    "duplicate_type": "exact" if exact_duplicate else "template",
                    "exact_group_size": exact_counts[row["content_hash"]],
                    "template_group_size": template_counts[row["_template_hash"]],
                    "label_conflict": conflict,
                    "canonical_id": canonical_ids[row["content_hash"]],
                    "is_canonical": is_canonical,
                    "action": action,
                }
            )

    for index in sorted(conflict_indices):
        row = working.loc[index].to_dict()
        removed.append(
            _removed_record(row, "conflicting_labels_in_template_group", row["group_id"])
        )
        summaries[row["source"]]["conflict_rows_removed"] = int(
            summaries[row["source"]]["conflict_rows_removed"]
        ) + 1
    for index in sorted(exact_remove_indices - conflict_indices):
        row = working.loc[index].to_dict()
        removed.append(
            _removed_record(row, "exact_duplicate", f"canonical_id={canonical_ids[row['content_hash']]}")
        )
        summaries[row["source"]]["exact_duplicates_removed"] = int(
            summaries[row["source"]]["exact_duplicates_removed"]
        ) + 1

    master = working.drop(index=sorted(conflict_indices | exact_remove_indices)).copy()
    output_schema = standardization["output_schema"]
    master = master[output_schema].sort_values(["source", "original_row_id"]).reset_index(drop=True)
    for source, group in master.groupby("source"):
        summaries[source]["output_rows"] = len(group)
        counts = group["label"].value_counts()
        for label in ("normal", "spam", "phishing"):
            summaries[source][label] = int(counts.get(label, 0))

    return {
        "master": master,
        "removed": pd.DataFrame(
            removed,
            columns=[
                "id", "source", "original_row_id", "raw_label", "mapped_label", "reason",
                "details", "subject_excerpt", "content_hash", "group_id",
            ],
        ),
        "duplicates": pd.DataFrame(
            duplicate_rows,
            columns=[
                "id", "source", "original_row_id", "label", "content_hash", "template_hash",
                "group_id", "duplicate_type", "exact_group_size", "template_group_size",
                "label_conflict", "canonical_id", "is_canonical", "action",
            ],
        ),
        "review": pd.DataFrame(
            review,
            columns=[
                "source", "original_row_id", "raw_label", "policy_status", "reason",
                "subject", "body_excerpt",
            ],
        ),
        "summary": pd.DataFrame(summaries.values()),
        "exact_duplicate_groups": sum(count > 1 for count in exact_counts.values()),
        "template_duplicate_groups": sum(count > 1 for count in template_counts.values()),
        "conflicting_template_groups": len(conflicting_groups),
    }


def standardize_vietnamese_dataset(
    source_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Standardize the audited translated Vietnamese dataset without training it."""
    policy = config["policy"]["vietnamese_standardization"]
    dataset = policy["dataset"]
    frame, _ = read_dataset(source_path)
    field_mapping = dataset["field_mapping"]
    label_column = str(field_mapping["raw_label"])
    text_column = str(field_mapping["text"])
    required_columns = {label_column, text_column}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Vietnamese dataset is missing configured columns: {sorted(missing_columns)}"
        )

    placeholders = set(policy["placeholder_values"])
    unicode_form = str(policy["unicode_form"])
    source = str(dataset["source_name"])
    candidates: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    subject_rows_parsed = 0

    for index, raw_row in frame.iterrows():
        original_row_id = int(index) + 2
        raw_label = normalize_raw_label(raw_row[label_column])
        raw_text_value = raw_row[text_column]
        raw_text = "" if pd.isna(raw_text_value) else str(raw_text_value)
        canonical_text = normalize_text(raw_text, unicode_form)
        label_policy = dataset["raw_label_policy"].get(raw_label)
        if label_policy is None:
            label = "review"
            label_status = "review"
            label_provenance = "translated_dataset_unmapped_raw_label"
            training_eligible = False
        else:
            label = str(label_policy["status"])
            label_status = str(label_policy["label_status"])
            label_provenance = str(label_policy["label_provenance"])
            training_eligible = bool(label_policy["training_eligible"])

        base = {
            "id": f"{source}:{original_row_id}",
            "original_row_id": original_row_id,
            "sender": "",
            "receiver": "",
            "date": "",
            "subject": "",
            "body": canonical_text,
            "original_text": raw_text,
            "normalized_text": "",
            "label": label,
            "raw_label": raw_label,
            "label_status": label_status,
            "label_provenance": label_provenance,
            "training_eligible": training_eligible,
            "source": source,
            "language": str(dataset["language"]),
            "data_origin": str(dataset["data_origin"]),
            "parent_id": "",
            "translation_source_id": "",
        }
        if is_empty_or_placeholder(raw_text_value, placeholders):
            removed.append(
                _removed_record(base, "empty_or_placeholder_text", "raw text is required")
            )
            continue

        subject, body, parsed = parse_prefixed_subject(
            raw_text_value, list(policy["subject_prefixes"])
        )
        if parsed:
            subject_rows_parsed += 1
            base["subject"] = subject
            base["body"] = body

        extracted_urls = extract_urls(raw_text)
        malformed_urls = extract_malformed_urls(raw_text)
        base["url_count"] = len(extracted_urls)
        base["extracted_urls"] = json.dumps(extracted_urls, ensure_ascii=False)
        base["malformed_url_count"] = len(malformed_urls)
        base["malformed_urls"] = json.dumps(malformed_urls, ensure_ascii=False)
        base["normalized_text"] = normalized_duplicate_text(
            base["subject"], base["body"]
        )
        base["raw_content_hash"] = hashlib.sha256(
            raw_text.encode("utf-8")
        ).hexdigest()
        base["content_hash"] = content_hash(base["subject"], base["body"])
        base["_template_hash"] = template_hash(base["subject"], base["body"])
        base["_obfuscated_template_hash"] = obfuscated_template_hash(
            base["subject"], base["body"]
        )
        base["_order"] = len(candidates)
        candidates.append(base)

    working = pd.DataFrame(candidates)
    raw_counts = Counter(working["raw_content_hash"])
    exact_counts = Counter(working["content_hash"])
    template_counts = Counter(working["_template_hash"])
    obfuscated_template_counts = Counter(working["_obfuscated_template_hash"])
    working["group_id"] = working.apply(
        lambda row: (
            f"template_{row['_obfuscated_template_hash'][:20]}"
            if obfuscated_template_counts[row["_obfuscated_template_hash"]] > 1
            else f"content_{row['content_hash'][:20]}"
        ),
        axis=1,
    )

    group_label_counts = working.groupby("group_id")["label"].nunique()
    conflicting_groups = set(group_label_counts[group_label_counts > 1].index)
    canonical_ids: dict[str, str] = {}
    for digest, group in working.groupby("content_hash", sort=False):
        canonical_ids[digest] = str(group.sort_values("_order").iloc[0]["id"])

    exact_remove_indices: set[int] = set()
    conflict_indices: set[int] = set()
    duplicate_rows: list[dict[str, Any]] = []
    for index, row in working.iterrows():
        exact_duplicate = exact_counts[row["content_hash"]] > 1
        template_duplicate = template_counts[row["_template_hash"]] > 1
        obfuscated_template_duplicate = (
            obfuscated_template_counts[row["_obfuscated_template_hash"]] > 1
        )
        conflict = row["group_id"] in conflicting_groups
        is_canonical = row["id"] == canonical_ids[row["content_hash"]]
        if conflict:
            action = "exclude_conflicting_template_labels"
            conflict_indices.add(index)
        elif exact_duplicate and not is_canonical:
            action = "remove_exact_duplicate_keep_canonical"
            exact_remove_indices.add(index)
        elif exact_duplicate:
            action = "keep_exact_canonical"
        elif obfuscated_template_duplicate:
            action = "keep_template_grouped"
        else:
            action = "unique"
        if exact_duplicate or template_duplicate or obfuscated_template_duplicate:
            duplicate_rows.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "original_row_id": row["original_row_id"],
                    "raw_label": row["raw_label"],
                    "label": row["label"],
                    "label_status": row["label_status"],
                    "raw_content_hash": row["raw_content_hash"],
                    "content_hash": row["content_hash"],
                    "template_hash": row["_template_hash"],
                    "obfuscated_template_hash": row["_obfuscated_template_hash"],
                    "group_id": row["group_id"],
                    "raw_exact_group_size": raw_counts[row["raw_content_hash"]],
                    "exact_group_size": exact_counts[row["content_hash"]],
                    "template_group_size": template_counts[row["_template_hash"]],
                    "obfuscated_template_group_size": obfuscated_template_counts[
                        row["_obfuscated_template_hash"]
                    ],
                    "label_conflict": conflict,
                    "canonical_id": canonical_ids[row["content_hash"]],
                    "is_canonical": is_canonical,
                    "action": action,
                }
            )

    for index in sorted(conflict_indices):
        row = working.loc[index].to_dict()
        removed.append(
            _removed_record(
                row, "conflicting_labels_in_template_group", str(row["group_id"])
            )
        )
    for index in sorted(exact_remove_indices - conflict_indices):
        row = working.loc[index].to_dict()
        removed.append(
            _removed_record(
                row,
                "exact_duplicate",
                f"canonical_id={canonical_ids[row['content_hash']]}",
            )
        )

    output_schema = list(policy["output_schema"])
    master = working.drop(index=sorted(conflict_indices | exact_remove_indices)).copy()
    master = master[output_schema].sort_values(
        ["source", "original_row_id"]
    ).reset_index(drop=True)
    duplicate_columns = [
        "id",
        "source",
        "original_row_id",
        "raw_label",
        "label",
        "label_status",
        "raw_content_hash",
        "content_hash",
        "template_hash",
        "obfuscated_template_hash",
        "group_id",
        "raw_exact_group_size",
        "exact_group_size",
        "template_group_size",
        "obfuscated_template_group_size",
        "label_conflict",
        "canonical_id",
        "is_canonical",
        "action",
    ]
    return {
        "master": master,
        "removed": pd.DataFrame(
            removed,
            columns=[
                "id",
                "source",
                "original_row_id",
                "raw_label",
                "mapped_label",
                "reason",
                "details",
                "subject_excerpt",
                "content_hash",
                "group_id",
            ],
        ),
        "duplicates": pd.DataFrame(duplicate_rows, columns=duplicate_columns),
        "input_rows": len(frame),
        "candidate_rows": len(working),
        "output_rows": len(master),
        "subject_rows_parsed": subject_rows_parsed,
        "empty_rows_removed": sum(
            record["reason"] == "empty_or_placeholder_text" for record in removed
        ),
        "exact_duplicates_removed": len(exact_remove_indices - conflict_indices),
        "conflicting_rows_removed": len(conflict_indices),
        "raw_exact_duplicate_groups": sum(count > 1 for count in raw_counts.values()),
        "exact_duplicate_groups": sum(count > 1 for count in exact_counts.values()),
        "template_duplicate_groups": sum(
            count > 1 for count in template_counts.values()
        ),
        "obfuscated_template_duplicate_groups": sum(
            count > 1 for count in obfuscated_template_counts.values()
        ),
        "conflicting_template_groups": len(conflicting_groups),
        "standard_url_rows": int(working["url_count"].gt(0).sum()),
        "malformed_url_rows": int(working["malformed_url_count"].gt(0).sum()),
        "input_raw_label_distribution": {
            str(key): int(value)
            for key, value in frame[label_column].value_counts(dropna=False).items()
        },
        "output_label_distribution": {
            str(key): int(value)
            for key, value in master["label"].value_counts(dropna=False).items()
        },
        "output_label_status_distribution": {
            str(key): int(value)
            for key, value in master["label_status"].value_counts(dropna=False).items()
        },
    }


def load_dataset_config(path: Path) -> dict[str, Any]:
    """Read the machine-readable dataset policy."""
    return json.loads(path.read_text(encoding="utf-8"))
