"""Conservative evidence-based curation for translated Vietnamese positives."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vietmailguard.dataset_standardizer import normalize_text


ALLOWED_HUMAN_DECISIONS = {"spam", "phishing", "review", "exclude"}


@dataclass(frozen=True)
class ParentEvidence:
    """Audited English parent evidence available for one translated row."""

    parent_id: str = ""
    source: str = ""
    label: str = ""
    label_provenance: str = ""
    link_confidence: str = ""
    link_status: str = ""


def load_curation_rules(path: Path) -> dict[str, Any]:
    """Load and minimally validate the machine-readable curation policy."""
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "trusted_parent_label_provenance",
        "trusted_parent_link_statuses",
        "minimum_distinct_rules_for_proposal",
        "rules",
        "evidence_groups",
    }
    missing = required - set(policy)
    if missing:
        raise ValueError(f"Vietnamese curation policy is missing: {sorted(missing)}")
    return policy


def detect_triggered_rules(text: object, policy: dict[str, Any]) -> list[str]:
    """Return all distinct rule names supported by the text."""
    normalized = normalize_text(text, "NFC").casefold()
    triggered: list[str] = []
    for name, patterns in policy["rules"].items():
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
            triggered.append(str(name))
    return triggered


def _group_rule_count(
    group: str, triggered: set[str], policy: dict[str, Any]
) -> int:
    return len(triggered & set(policy["evidence_groups"][group]))


def choose_evidence_group(
    triggered_rules: list[str], policy: dict[str, Any]
) -> tuple[str, int]:
    """Choose one review group using conservative multi-rule thresholds."""
    triggered = set(triggered_rules)
    minimum = int(policy["minimum_distinct_rules_for_proposal"])
    priority = (
        "credential_phishing",
        "account_impersonation",
        "advance_fee_fraud",
        "lottery_scam",
        "financial_fraud",
        "commercial_spam",
    )
    scored = [(group, _group_rule_count(group, triggered, policy)) for group in priority]
    eligible = [item for item in scored if item[1] >= minimum]
    if not eligible:
        return "ambiguous", max((score for _, score in scored), default=0)
    priority_index = {name: index for index, name in enumerate(priority)}
    return max(eligible, key=lambda item: (item[1], -priority_index[item[0]]))


def _is_low_information(text: str, policy: dict[str, Any]) -> bool:
    meaningful = re.findall(r"(?u)\b\w+\b", text)
    return (
        len(text.strip()) < int(policy["minimum_meaningful_characters"])
        or len(meaningful) < int(policy["minimum_meaningful_tokens"])
    )


def parent_label_is_trusted(parent: ParentEvidence, policy: dict[str, Any]) -> bool:
    """Return true only for an allowlisted label provenance and link status."""
    allowed = policy["trusted_parent_label_provenance"].get(parent.label, [])
    return (
        parent.link_status in set(policy["trusted_parent_link_statuses"])
        and parent.label_provenance in set(allowed)
        and bool(parent.parent_id)
    )


def curate_positive_row(
    row: dict[str, Any] | pd.Series,
    parent: ParentEvidence,
    policy: dict[str, Any],
    *,
    human_decision: str = "",
    human_note: str = "",
) -> dict[str, Any]:
    """Curate one raw-positive row without treating rule output as ground truth."""
    decision = str(human_decision).strip().casefold()
    if decision and decision not in ALLOWED_HUMAN_DECISIONS:
        raise ValueError(f"Unsupported human decision: {human_decision}")

    subject = normalize_text(row.get("subject", ""), "NFC")
    body = normalize_text(row.get("body", ""), "NFC")
    text = normalize_text(f"{subject}\n{body}", "NFC")
    triggered = detect_triggered_rules(text, policy)
    evidence_group, group_score = choose_evidence_group(triggered, policy)
    low_information = _is_low_information(text, policy)

    if low_information:
        proposed_label = "exclude"
        proposal_confidence = "high"
    elif evidence_group == "commercial_spam":
        proposed_label = "spam"
        proposal_confidence = "medium"
    elif evidence_group in {"credential_phishing", "account_impersonation"}:
        proposed_label = "phishing"
        proposal_confidence = "medium"
    else:
        proposed_label = "review"
        proposal_confidence = "medium" if group_score >= 2 else "low"

    label = "review"
    label_status = "review"
    training_eligible = False
    label_provenance = "translated_dataset_mixed_positive"

    if low_information:
        label = "exclude"
        label_status = "exclude"
        label_provenance = "excluded_low_information_content"
    elif parent_label_is_trusted(parent, policy):
        spam_conflicts = set(policy["strong_conflict_groups_for_spam_parent"])
        parent_conflicts = parent.label == "spam" and evidence_group in spam_conflicts
        phishing_conflicts = parent.label == "phishing" and evidence_group == "commercial_spam"
        if not parent_conflicts and not phishing_conflicts:
            label = parent.label
            proposed_label = parent.label
            proposal_confidence = "high"
            label_status = "evidence_confirmed"
            training_eligible = True
            label_provenance = (
                f"translated_from_{parent.label_provenance}_parent"
            )

    if decision:
        label = decision
        proposed_label = decision
        proposal_confidence = "human_confirmed"
        training_eligible = decision in {"spam", "phishing"}
        label_status = "human_confirmed" if training_eligible else decision
        label_provenance = f"human_confirmed_vietnamese_{decision}"

    return {
        "evidence_group": evidence_group,
        "triggered_rules": ";".join(triggered),
        "proposed_label": proposed_label,
        "proposal_confidence": proposal_confidence,
        "human_decision": decision,
        "human_note": str(human_note).strip(),
        "label": label,
        "label_status": label_status,
        "training_eligible": training_eligible,
        "label_provenance": label_provenance,
    }


def build_parent_evidence_maps(
    english: pd.DataFrame,
    overlap: pd.DataFrame,
) -> dict[str, ParentEvidence]:
    """Build Vietnamese-ID parent evidence from audited links and English rows."""
    english_by_id = english.set_index(english["id"].astype(str), drop=False)
    linked_statuses = {"linked_exact_provenance", "linked_high_confidence_evidence"}
    linked = overlap.loc[overlap["review_status"].isin(linked_statuses)].copy()
    evidence: dict[str, ParentEvidence] = {}
    for item in linked.itertuples(index=False):
        vi_id = str(item.vietnamese_id)
        en_id = str(item.english_id)
        if en_id not in english_by_id.index:
            continue
        parent = english_by_id.loc[en_id]
        if isinstance(parent, pd.DataFrame):
            raise ValueError(f"English parent id is not unique: {en_id}")
        evidence[vi_id] = ParentEvidence(
            parent_id=en_id,
            source=str(parent.get("source", "")),
            label=str(parent.get("label", "")),
            label_provenance=str(parent.get("label_provenance", "")),
            link_confidence=str(getattr(item, "confidence")),
            link_status=str(item.review_status),
        )
    return evidence


def curate_vietnamese_dataset(
    vietnamese: pd.DataFrame,
    english: pd.DataFrame,
    overlap: pd.DataFrame,
    policy: dict[str, Any],
    existing_review: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return full curated dataset and the raw-positive human review table."""
    required = {"id", "raw_label", "source", "subject", "body", "parent_id"}
    missing = required - set(vietnamese.columns)
    if missing:
        raise ValueError(f"Vietnamese master is missing columns: {sorted(missing)}")

    human_by_id: dict[str, tuple[str, str]] = {}
    if existing_review is not None and not existing_review.empty:
        if not {"id", "human_decision", "human_note"}.issubset(existing_review.columns):
            raise ValueError("Existing review file lacks human decision columns")
        for item in existing_review.itertuples(index=False):
            human_by_id[str(item.id)] = (
                str(item.human_decision).strip(),
                str(item.human_note).strip(),
            )

    parent_map = build_parent_evidence_maps(english, overlap)
    curated = vietnamese.copy()
    review_rows: list[dict[str, Any]] = []
    positive_mask = curated["raw_label"].astype(str).eq("1")

    extra_columns = [
        "evidence_group",
        "triggered_rules",
        "proposed_label",
        "proposal_confidence",
        "human_decision",
        "human_note",
        "linked_parent_label",
        "linked_parent_source",
        "linked_parent_label_provenance",
        "linked_parent_confidence",
        "parent_link_status",
    ]
    for column in extra_columns:
        if column not in curated.columns:
            curated[column] = ""

    for index, row in curated.loc[positive_mask].iterrows():
        vi_id = str(row["id"])
        parent = parent_map.get(vi_id, ParentEvidence())
        human_decision, human_note = human_by_id.get(vi_id, ("", ""))
        decision = curate_positive_row(
            row,
            parent,
            policy,
            human_decision=human_decision,
            human_note=human_note,
        )
        for column, value in decision.items():
            curated.at[index, column] = value
        curated.at[index, "linked_parent_label"] = parent.label
        curated.at[index, "linked_parent_source"] = parent.source
        curated.at[index, "linked_parent_label_provenance"] = parent.label_provenance
        curated.at[index, "linked_parent_confidence"] = parent.link_confidence
        curated.at[index, "parent_link_status"] = parent.link_status

        review_rows.append(
            {
                "id": vi_id,
                "source": str(row["source"]),
                "parent_id": str(row.get("parent_id", "")),
                "translation_source_id": str(row.get("translation_source_id", "")),
                "subject": str(row.get("subject", "")),
                "body_excerpt": normalize_text(row.get("body", ""), "NFC")[:500],
                "raw_label": str(row["raw_label"]),
                "linked_parent_label": parent.label,
                "linked_parent_source": parent.source,
                "linked_parent_label_provenance": parent.label_provenance,
                "linked_parent_confidence": parent.link_confidence,
                "parent_link_status": parent.link_status,
                **{key: decision[key] for key in (
                    "evidence_group",
                    "triggered_rules",
                    "proposed_label",
                    "proposal_confidence",
                    "human_decision",
                    "human_note",
                    "label_provenance",
                )},
            }
        )

    review = pd.DataFrame(review_rows)
    if not review.empty:
        review = review.sort_values(
            ["label_provenance", "evidence_group", "source", "id"],
            kind="stable",
        ).reset_index(drop=True)
    return curated, review
