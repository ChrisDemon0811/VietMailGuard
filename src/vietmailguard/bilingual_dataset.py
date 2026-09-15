"""Build and validate the leakage-safe VietMailGuard Version 2 dataset."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from vietmailguard.dataset_standardizer import normalized_duplicate_text
from vietmailguard.split_builder import SPLIT_ORDER, integer_targets


class BilingualLeakageError(RuntimeError):
    """Raised when a Version 2 split violates a leakage constraint."""


class UnionFind:
    """Small deterministic disjoint-set implementation for leakage links."""

    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def load_bilingual_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the Version 2 dataset policy."""
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"inputs", "outputs", "valid_labels", "splitting"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Bilingual config is missing keys: {sorted(missing)}")
    ratios = config["splitting"]["ratios"]
    if tuple(ratios) != SPLIT_ORDER:
        raise ValueError(f"Split ratios must be ordered as {SPLIT_ORDER}")
    if abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    return config


def boolean_series(series: pd.Series) -> pd.Series:
    """Parse serialized booleans without treating non-empty strings as true."""
    true_values = {"true", "1", "yes", "y"}
    return series.map(lambda value: str(value).strip().casefold() in true_values)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_key(seed: int, value: str) -> str:
    return _sha256(f"{seed}|{value}")


def _required_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _text_column(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str)


def _trusted_english_mask(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for label, provenances in config["english_label_provenance"].items():
        mask |= frame["label"].eq(label) & frame["label_provenance"].isin(provenances)
    return mask


def _standardize_english(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in ("id", "original_row_id", "subject", "body", "label", "raw_label",
                   "source", "language", "content_hash", "group_id", "label_provenance"):
        output[column] = _text_column(frame, column)
    output["parent_id"] = ""
    output["translation_source_id"] = ""
    output["parent_source"] = ""
    output["data_origin"] = ""
    output["augmentation_type"] = ""
    output["label_status"] = "eligible"
    output["training_eligible"] = True
    output["eligibility_basis"] = "english_high_confidence_policy"
    output["split_constraint"] = "flexible"
    output["input_artifact"] = "english_master"
    return output


def _standardize_curated_vietnamese(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in (
        "id", "original_row_id", "parent_id", "translation_source_id", "source",
        "subject", "body", "label", "raw_label", "label_status", "label_provenance",
        "content_hash", "group_id", "language", "data_origin",
    ):
        output[column] = _text_column(frame, column)
    output["parent_source"] = _text_column(frame, "linked_parent_source")
    output["augmentation_type"] = ""
    output["training_eligible"] = True
    output["eligibility_basis"] = "trusted_cross_language_parent_link"
    output["split_constraint"] = "flexible"
    output["input_artifact"] = "vietnamese_curated"
    return output


def _standardize_augmentation(
    frame: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in (
        "id", "parent_id", "translation_source_id", "parent_source", "subject", "body",
        "label", "raw_label", "language", "data_origin", "augmentation_type",
        "content_hash", "group_id", "label_provenance",
    ):
        output[column] = _text_column(frame, column)
    output["original_row_id"] = ""
    output["source"] = source_name
    output["label_status"] = "evidence_confirmed"
    output["training_eligible"] = True
    output["eligibility_basis"] = "controlled_translation_quality_passed"
    output["split_constraint"] = "train_only"
    output["input_artifact"] = "vietnamese_augmentation"
    return output


def prepare_bilingual_rows(
    english: pd.DataFrame,
    vietnamese: pd.DataFrame,
    augmentation: pd.DataFrame,
    overlap: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select only trusted rows and harmonize their provenance fields."""
    common = {"id", "subject", "body", "label", "raw_label", "source", "language",
              "content_hash", "group_id", "label_provenance"}
    _required_columns(english, common, "English master")
    _required_columns(
        vietnamese,
        common | {"parent_id", "translation_source_id", "label_status", "training_eligible",
                  "data_origin", "parent_link_status", "linked_parent_source"},
        "Vietnamese curated dataset",
    )
    _required_columns(
        augmentation,
        {"id", "parent_id", "translation_source_id", "parent_source", "subject", "body",
         "label", "raw_label", "language", "data_origin", "augmentation_type",
         "content_hash", "group_id", "label_provenance", "training_eligible",
         "translation_quality_status", "protected_token_validation"},
        "Vietnamese controlled augmentation",
    )
    _required_columns(
        overlap, {"english_id", "vietnamese_id", "review_status"},
        "Cross-language overlap manifest",
    )

    valid_labels = set(config["valid_labels"])
    english_mask = (
        english["label"].isin(valid_labels)
        & english["language"].eq("en")
        & _trusted_english_mask(english, config)
        & (english["subject"].astype(str).str.strip().ne("")
           | english["body"].astype(str).str.strip().ne(""))
    )
    selected_english = english.loc[english_mask].copy()
    english_by_id = selected_english.set_index(selected_english["id"].astype(str), drop=False)

    trusted_statuses = set(config["trusted_cross_language_statuses"])
    trusted_links = overlap.loc[overlap["review_status"].isin(trusted_statuses), [
        "english_id", "vietnamese_id", "review_status"
    ]].drop_duplicates()
    links_per_vi = trusted_links.groupby("vietnamese_id")["english_id"].nunique()
    ambiguous_links = set(links_per_vi.index[links_per_vi.ne(1)].astype(str))
    trusted_links = trusted_links.loc[~trusted_links["vietnamese_id"].astype(str).isin(ambiguous_links)]
    trusted_pair = {
        str(row.vietnamese_id): str(row.english_id)
        for row in trusted_links.itertuples(index=False)
    }

    vi_base_eligible = boolean_series(vietnamese["training_eligible"]) & vietnamese["label"].isin(valid_labels)
    vi_parent_from_manifest = vietnamese["id"].astype(str).map(trusted_pair).fillna("")
    vi_linked = (
        vi_base_eligible
        & vietnamese["parent_id"].astype(str).eq(vi_parent_from_manifest)
        & vi_parent_from_manifest.isin(english_by_id.index.astype(str))
    )
    selected_vietnamese = vietnamese.loc[vi_linked].copy()
    for row in selected_vietnamese.itertuples(index=False):
        parent = english_by_id.loc[str(row.parent_id)]
        if str(row.label) != str(parent["label"]):
            raise ValueError(f"Vietnamese label conflicts with trusted parent: {row.id}")
        selected_vietnamese.loc[
            selected_vietnamese["id"].astype(str).eq(str(row.id)),
            "linked_parent_source",
        ] = str(parent["source"])

    augmentation_policy = config["augmentation_policy"]
    augmentation_base = (
        boolean_series(augmentation["training_eligible"])
        & augmentation["label"].isin(valid_labels)
        & augmentation["translation_quality_status"].eq(
            augmentation_policy["required_quality_status"]
        )
        & augmentation["protected_token_validation"].eq(
            augmentation_policy["required_protected_token_validation"]
        )
        & augmentation["parent_id"].astype(str).isin(english_by_id.index.astype(str))
    )
    selected_augmentation = augmentation.loc[augmentation_base].copy()
    for row in selected_augmentation.itertuples(index=False):
        parent = english_by_id.loc[str(row.parent_id)]
        if str(row.label) != str(parent["label"]):
            raise ValueError(f"Augmentation label conflicts with trusted parent: {row.id}")
        if str(row.parent_label) != str(parent["label"]):
            raise ValueError(f"Augmentation parent_label mismatch: {row.id}")

    combined = pd.concat(
        [
            _standardize_english(selected_english),
            _standardize_curated_vietnamese(selected_vietnamese),
            _standardize_augmentation(
                selected_augmentation, augmentation_policy["source_name"]
            ),
        ],
        ignore_index=True,
    )
    if combined["id"].duplicated().any():
        duplicates = combined.loc[combined["id"].duplicated(keep=False), "id"].head().tolist()
        raise ValueError(f"Unified IDs are not unique: {duplicates}")

    unresolved = vi_base_eligible & ~vi_linked
    audit = {
        "input_rows": {
            "english_master": len(english),
            "vietnamese_curated": len(vietnamese),
            "vietnamese_augmentation": len(augmentation),
        },
        "included_rows": {
            "english_master": len(selected_english),
            "vietnamese_curated_trusted_link": len(selected_vietnamese),
            "vietnamese_augmentation_quality_passed": len(selected_augmentation),
        },
        "excluded_rows": {
            "english_not_high_confidence_or_invalid": int((~english_mask).sum()),
            "vietnamese_review": int(vietnamese["label"].eq("review").sum()),
            "vietnamese_exclude": int(vietnamese["label"].eq("exclude").sum()),
            "vietnamese_eligible_but_unresolved_parent_link": int(unresolved.sum()),
            "vietnamese_other_ineligible": int(
                ((~vi_base_eligible)
                 & ~vietnamese["label"].isin({"review", "exclude"})).sum()
            ),
            "augmentation_failed_quality": int((~augmentation_base).sum()),
        },
        "trusted_link_pairs": len(selected_vietnamese),
        "trusted_link_parents": selected_vietnamese["parent_id"].nunique(),
        "ambiguous_trusted_link_rows": len(ambiguous_links),
    }
    return combined, audit


def _union_by_value(frame: pd.DataFrame, column: str, union_find: UnionFind) -> int:
    edge_count = 0
    for value, rows in frame.loc[frame[column].astype(str).str.strip().ne("")].groupby(column):
        ids = rows["id"].astype(str).tolist()
        for row_id in ids[1:]:
            union_find.union(ids[0], row_id)
            edge_count += 1
    return edge_count


def build_final_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Connect duplicate, template, parent, translation and derivative relations."""
    required = {"id", "subject", "body", "label", "content_hash", "group_id", "parent_id"}
    _required_columns(frame, required, "Unified bilingual rows")
    if frame["id"].astype(str).str.strip().eq("").any() or frame["id"].duplicated().any():
        raise ValueError("Every unified row must have one unique non-empty id")

    result = frame.copy()
    result["normalized_content_hash"] = result.apply(
        lambda row: _sha256(normalized_duplicate_text(row["subject"], row["body"])), axis=1
    )
    result["template_hash"] = result.apply(
        lambda row: _sha256(
            normalized_duplicate_text(row["subject"], row["body"], template=True)
        ),
        axis=1,
    )
    union_find = UnionFind(result["id"].astype(str))
    edges = {
        "source_group_edges": _union_by_value(result, "group_id", union_find),
        "content_hash_edges": _union_by_value(result, "content_hash", union_find),
        "normalized_duplicate_edges": _union_by_value(
            result, "normalized_content_hash", union_find
        ),
        "template_duplicate_edges": _union_by_value(result, "template_hash", union_find),
        "parent_translation_edges": 0,
    }

    ids = set(result["id"].astype(str))
    translation_group_by_id: dict[str, str] = {}
    for row in result.loc[result["parent_id"].astype(str).str.strip().ne("")].itertuples(index=False):
        row_id = str(row.id)
        parent_id = str(row.parent_id)
        if parent_id not in ids:
            raise ValueError(f"Included translation has no included English parent: {row_id}")
        union_find.union(parent_id, row_id)
        edges["parent_translation_edges"] += 1
        translation_id = f"translation_{_sha256(parent_id)[:20]}"
        translation_group_by_id[row_id] = translation_id
        translation_group_by_id[parent_id] = translation_id

    members: dict[str, list[str]] = defaultdict(list)
    for row_id in sorted(ids):
        members[union_find.find(row_id)].append(row_id)
    final_by_id: dict[str, str] = {}
    for component in members.values():
        final_id = f"v2group_{_sha256('|'.join(sorted(component)))[:20]}"
        for row_id in component:
            final_by_id[row_id] = final_id

    result["translation_group_id"] = result["id"].astype(str).map(
        translation_group_by_id
    ).fillna("")
    result["final_group_id"] = result["id"].astype(str).map(final_by_id)
    conflicts = result.groupby("final_group_id")["label"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(
            "Final leakage groups span labels: "
            f"{conflicts.index[conflicts.gt(1)].tolist()[:5]}"
        )
    edges["final_groups"] = result["final_group_id"].nunique()
    edges["translation_groups"] = result.loc[
        result["translation_group_id"].ne(""), "translation_group_id"
    ].nunique()
    return result, edges


def _allocate_groups(
    groups: list[tuple[str, int]], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    targets = integer_targets(sum(size for _, size in groups), ratios)
    counts = {name: 0 for name in SPLIT_ORDER}
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=lambda item: (-item[1], _stable_key(seed, item[0])))
    assignments: dict[str, str] = {}
    for group_id, size in shuffled:
        fitting = [name for name in SPLIT_ORDER if counts[name] + size <= targets[name]]
        if fitting:
            chosen = max(
                fitting,
                key=lambda name: (
                    (targets[name] - counts[name]) / max(targets[name], 1),
                    -SPLIT_ORDER.index(name),
                ),
            )
        else:
            chosen = min(
                SPLIT_ORDER,
                key=lambda name: (
                    (counts[name] + size - targets[name]) / max(targets[name], 1),
                    SPLIT_ORDER.index(name),
                ),
            )
        assignments[group_id] = chosen
        counts[chosen] += size
    return assignments


def assign_bilingual_splits(
    frame: pd.DataFrame, ratios: dict[str, float], seed: int = 42
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign final groups with augmentation groups forced to train."""
    required = {"final_group_id", "label", "source", "language", "split_constraint"}
    _required_columns(frame, required, "Bilingual master")
    grouped = frame.groupby("final_group_id", sort=False).agg(
        size=("id", "size"),
        label_count=("label", "nunique"),
        label=("label", "first"),
        forced_train=("split_constraint", lambda values: (values == "train_only").any()),
        source_profile=("source", lambda values: "+".join(sorted(set(values)))),
        language_profile=("language", lambda values: "+".join(sorted(set(values)))),
    )
    if grouped["label_count"].gt(1).any():
        raise ValueError("A final_group_id spans multiple labels")

    assignments: dict[str, str] = {
        str(group_id): "train"
        for group_id in grouped.index[grouped["forced_train"]]
    }
    class_targets: dict[str, dict[str, int]] = {}
    for label_index, label in enumerate(sorted(frame["label"].unique())):
        label_rows = frame.loc[frame["label"].eq(label)]
        targets = integer_targets(len(label_rows), ratios)
        label_groups = grouped.loc[grouped["label"].eq(label)]
        forced = label_groups.loc[label_groups["forced_train"]]
        forced_rows = int(forced["size"].sum())
        if forced_rows > targets["train"]:
            raise ValueError(
                f"Train-only {label} rows exceed the 70% class target: "
                f"{forced_rows} > {targets['train']}"
            )
        remaining_targets = dict(targets)
        remaining_targets["train"] -= forced_rows
        remaining_total = sum(remaining_targets.values())
        remaining_ratios = {
            name: remaining_targets[name] / remaining_total
            for name in SPLIT_ORDER
        }
        flexible = label_groups.loc[~label_groups["forced_train"]].copy()
        flexible["stratum"] = (
            flexible["source_profile"].astype(str)
            + "|" + flexible["language_profile"].astype(str)
        )
        for stratum_index, (_, stratum) in enumerate(flexible.groupby("stratum", sort=True)):
            units = [
                (str(group_id), int(row["size"]))
                for group_id, row in stratum.iterrows()
            ]
            assignments.update(
                _allocate_groups(
                    units,
                    remaining_ratios,
                    seed + label_index * 1000 + stratum_index,
                )
            )
        class_targets[str(label)] = {
            **{f"target_{name}": targets[name] for name in SPLIT_ORDER},
            "forced_train_rows": forced_rows,
        }

    result = frame.copy()
    result["split"] = result["final_group_id"].map(assignments)
    if result["split"].isna().any():
        raise RuntimeError("At least one final leakage group has no split")
    return result, {
        "class_targets": class_targets,
        "forced_train_groups": int(grouped["forced_train"].sum()),
        "forced_train_rows": int(grouped.loc[grouped["forced_train"], "size"].sum()),
    }


def split_frames(master: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Return deterministic row order for each assigned split."""
    outputs: dict[str, pd.DataFrame] = {}
    for offset, name in enumerate(SPLIT_ORDER):
        frame = master.loc[master["split"].eq(name)].copy()
        frame["_order"] = frame["id"].astype(str).map(
            lambda value: _stable_key(seed + offset, value)
        )
        outputs[name] = frame.sort_values("_order", kind="stable").drop(
            columns="_order"
        ).reset_index(drop=True)
    return outputs


def validate_bilingual_splits(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Reject group, duplicate, template, parent, or label leakage."""
    if set(splits) != set(SPLIT_ORDER):
        raise ValueError(f"Expected splits {SPLIT_ORDER}, got {sorted(splits)}")
    combined = pd.concat(
        [frame.assign(_actual_split=name) for name, frame in splits.items()],
        ignore_index=True,
    )
    valid_labels = {"normal", "spam", "phishing"}
    invalid_labels = sorted(set(combined["label"]) - valid_labels)
    if invalid_labels:
        raise BilingualLeakageError(f"Invalid labels entered splits: {invalid_labels}")
    if not boolean_series(combined["training_eligible"]).all():
        raise BilingualLeakageError("A non-training-eligible row entered a split")
    if combined["label"].isin({"review", "exclude"}).any():
        raise BilingualLeakageError("Review or exclude row entered a split")
    if combined["id"].duplicated().any():
        raise BilingualLeakageError("A row id appears in more than one split")

    checked_columns = (
        "final_group_id", "content_hash", "normalized_content_hash", "template_hash",
        "group_id", "translation_group_id",
    )
    overlaps: dict[str, int] = {}
    for column in checked_columns:
        values = combined.loc[combined[column].astype(str).str.strip().ne("")]
        counts = values.groupby(column)["_actual_split"].nunique()
        overlap_count = int(counts.gt(1).sum())
        overlaps[column] = overlap_count
        if overlap_count:
            examples = counts.index[counts.gt(1)].astype(str).tolist()[:5]
            raise BilingualLeakageError(
                f"{column} crosses splits ({overlap_count} groups): {examples}"
            )

    split_by_id = dict(zip(combined["id"].astype(str), combined["_actual_split"], strict=True))
    parent_rows = combined.loc[combined["parent_id"].astype(str).str.strip().ne("")]
    for row in parent_rows.to_dict("records"):
        parent_id = str(row["parent_id"])
        if parent_id not in split_by_id:
            raise BilingualLeakageError(f"Translation parent absent from splits: {parent_id}")
        if split_by_id[parent_id] != str(row["_actual_split"]):
            raise BilingualLeakageError(
                f"Parent/translation split mismatch: {parent_id} and {row['id']}"
            )
    train_only = combined["split_constraint"].eq("train_only")
    if not combined.loc[train_only, "_actual_split"].eq("train").all():
        raise BilingualLeakageError("A train-only controlled translation left train")
    return {
        "passed": True,
        "checked_identifiers": list(checked_columns),
        "cross_split_overlap_counts": overlaps,
        "parent_translation_pairs_checked": len(parent_rows),
        "train_only_rows_checked": int(train_only.sum()),
    }
