"""Build provenance-preserving Vietnamese translations from English train parents."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.file_utils import sha256_file  # noqa: E402
from vietmailguard.translation_augmentation import (  # noqa: E402
    MarianOfflineTranslator,
    TranslationResult,
    apply_duplicate_translation_flags,
    build_augmented_row,
    count_quality_flags,
    load_augmentation_config,
    protect_text,
    revalidate_existing_translations,
    select_parent_rows,
    translation_config_fingerprint,
    validate_parent_split_safety,
    validation_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "translation_augmentation.json",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the pinned model revision to exist in the local cache.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Override automatic CPU/CUDA selection.",
    )
    return parser.parse_args()


def raw_hashes(raw_dir: Path) -> dict[str, str]:
    """Hash every immutable raw file."""
    return {
        path.name: sha256_file(path)
        for path in sorted(raw_dir.iterdir())
        if path.is_file()
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _write_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _completed_rows(path: Path, selected_ids: set[str], config: dict[str, Any]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    existing = pd.read_csv(path, keep_default_na=False)
    required = {
        "parent_id", "translation_model", "translation_model_revision",
        "translation_config_hash", "subject", "body"
    }
    if not required.issubset(existing.columns):
        return pd.DataFrame()
    model = config["model"]
    valid = existing.loc[
        existing["parent_id"].astype(str).isin(selected_ids)
        & existing["translation_model"].eq(model["name"])
        & existing["translation_model_revision"].eq(model["revision"])
        & existing["translation_config_hash"].eq(
            translation_config_fingerprint(config)
        )
    ].copy()
    return valid.drop_duplicates("parent_id", keep="last")


def _empty_result(text: str) -> TranslationResult:
    protected = protect_text(text)
    return TranslationResult(
        text="",
        protected_count=len(protected.placeholders),
        protected_values_restored=not protected.placeholders,
        protected_segment_strategy_used=False,
    )


def translate_parent_batch(
    parents: pd.DataFrame,
    translator: MarianOfflineTranslator,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Translate a parent batch, falling back to row-level failure records."""
    subjects = parents["subject"].astype(str).tolist()
    bodies = parents["body"].astype(str).tolist()
    try:
        subject_results = translator.translate_batch(subjects)
        body_results = translator.translate_batch(bodies)
        errors = [""] * len(parents)
    except Exception:
        subject_results = []
        body_results = []
        errors = []
        for subject, body in zip(subjects, bodies, strict=True):
            try:
                subject_results.append(translator.translate_batch([subject])[0])
                body_results.append(translator.translate_batch([body])[0])
                errors.append("")
            except Exception as error:  # preserve a visible failure row
                subject_results.append(_empty_result(subject))
                body_results.append(_empty_result(body))
                errors.append(type(error).__name__)

    rows: list[dict[str, Any]] = []
    for (_, parent), subject_result, body_result, error in zip(
        parents.iterrows(), subject_results, body_results, errors, strict=True
    ):
        rows.append(
            build_augmented_row(
                parent,
                subject_result,
                body_result,
                protect_text(parent["subject"]),
                protect_text(parent["body"]),
                config,
                translation_error=error,
                translation_device=translator.device_name,
            )
        )
    return rows


def _table_from_counts(counts: dict[Any, int], first_column: str) -> list[str]:
    lines = [f"| {first_column} | Rows |", "| --- | ---: |"]
    if not counts:
        return [*lines, "| None | 0 |"]
    lines.extend(f"| {key} | {value:,} |" for key, value in counts.items())
    return lines


def render_report(
    augmented: pd.DataFrame,
    selected: pd.DataFrame,
    selection_audit: dict[str, Any],
    config: dict[str, Any],
    *,
    device_name: str,
    reused_rows: int,
    raw_unchanged: bool,
    inputs_unchanged: bool,
) -> str:
    """Render a factual augmentation report."""
    accepted = augmented.loc[augmented["translation_quality_status"].eq("accepted")]
    failed = augmented.loc[augmented["translation_quality_status"].eq("failed")]
    created = augmented["body"].astype(str).str.strip().ne("").sum()
    duplicates = augmented.loc[
        augmented["quality_flags"].astype(str).str.contains(
            "duplicate_translated_output", regex=False
        )
    ]
    duplicate_groups = duplicates["content_hash"].nunique()
    protected_failed = augmented["protected_token_validation"].eq("failed").sum()
    flags = count_quality_flags(augmented)
    class_counts = augmented["label"].value_counts().to_dict()
    accepted_counts = accepted["label"].value_counts().to_dict()
    source_counts = augmented.groupby(["label", "parent_source"]).size().to_dict()
    model = config["model"]

    lines = [
        "# Translated Vietnamese augmentation report",
        "",
        "## Kết quả",
        "",
        f"Đã chọn **{len(selected):,}** English train parents và tạo **{int(created):,}** bản dịch có body không rỗng.",
        f"Có **{len(accepted):,}** rows qua quality gate và **{len(failed):,}** rows bị flag `failed`; failed rows không training-eligible.",
        "",
        "Toàn bộ output có `language=vi`, `data_origin=translated`, `augmentation_type=controlled_translation`. Đây không phải native Vietnamese hay Vietnamese real-world dataset.",
        "",
        "## Parent selection",
        "",
        f"- Input duy nhất: `{config['input']}`; tất cả parent thuộc split `train`.",
        "- Chỉ label/provenance allowlist được chọn. Review, uncertain, exclude và weak labels không được dùng.",
        f"- Một parent tối đa một lần; một parent group tối đa một lần: `{str(config['one_parent_per_group']).lower()}`.",
        f"- Known translated parents excluded: **{selection_audit['existing_parent_overlap_removed']:,}** eligible rows.",
        f"- Target: **{config['target_per_class']:,}** unique parents/class; không oversample.",
        "",
        "### Selected parents by class",
        "",
    ]
    lines.extend(_table_from_counts(class_counts, "Class"))
    lines.extend(["", "### Selected parents by class and English source", ""])
    lines.extend(
        _table_from_counts(
            {f"{label} / {source}": int(count) for (label, source), count in source_counts.items()},
            "Class / source",
        )
    )
    lines.extend(["", "### Accepted translations by class", ""])
    lines.extend(_table_from_counts(accepted_counts, "Class"))

    lines.extend(
        [
            "",
            "## Parent split safety",
            "",
            "- `parent_id`, `parent_source`, `parent_label`, `parent_group_id` and `translation_source_id` are preserved.",
            "- Output `group_id` equals `parent_group_id`; future bilingual splitting must treat this as a hard group constraint.",
            "- Because every parent came from English train, these translations may enter train only. They must not be moved to validation/test.",
            "- Existing known Vietnamese translations of selected English parents were excluded before sampling.",
            "",
            "## Translation method",
            "",
            f"- Model: `{model['name']}`.",
            f"- Pinned revision: `{model['revision']}`.",
            f"- Model license: `{model['license']}`.",
            f"- Direction: `{model['source_language']} → {model['target_language']}`; target prefix `{model['target_prefix']}`.",
            f"- Device used: `{device_name}`.",
            f"- Transformers: `{_package_version('transformers')}`; PyTorch: `{_package_version('torch')}`; SentencePiece: `{_package_version('sentencepiece')}`.",
            f"- Generation parameters: `{json.dumps(model['generation'], sort_keys=True)}`.",
            f"- Translation config hash: `{translation_config_fingerprint(config)}`.",
            f"- Max input tokens/chunk: `{model['max_input_tokens']}`; batch size: `{model['batch_size']}`.",
            "- The public model weights may be downloaded once. Email text is translated locally and is not sent to an online translation API.",
            f"- Rows reused from a matching pinned-revision output on this run: **{reused_rows:,}**.",
            "",
            "## Protected-token validation",
            "",
            "URLs, email addresses, standalone domains, amounts, dates, numbers, account-like identifiers and observable proper nouns are replaced by stable placeholders before translation and restored exactly afterward.",
            f"- Protected-token validation passed: **{len(augmented) - int(protected_failed):,}** rows.",
            f"- Protected-token validation failed: **{int(protected_failed):,}** rows.",
            f"- Rows using protected segment translation: **{int(augmented['protected_segment_strategy_used'].sum()):,}**.",
            "",
            "## Quality checks",
            "",
            f"- Empty/translation failures: **{int(flags.get('empty_output', 0) + sum(value for key, value in flags.items() if key.startswith('translation_error:'))):,}** flags.",
            f"- Duplicate translated output groups: **{duplicate_groups:,}** involving **{len(duplicates):,}** rows.",
            f"- Training-eligible after all checks: **{int(augmented['training_eligible'].sum()):,}** rows.",
            "",
            "### Quality flags",
            "",
        ]
    )
    lines.extend(_table_from_counts(dict(flags.most_common()), "Flag"))
    lines.extend(
        [
            "",
            "## Human review sample",
            "",
            f"A deterministic source-diverse sample targets **{config['review_sample_per_class']}** accepted translations per class. Human decision and note fields remain blank.",
            "",
            "## Limitations",
            "",
            "- Machine translation may alter tone, fluency, or security semantics even when protected tokens survive.",
            "- Proper-noun protection is deterministic and conservative, not a full named-entity recognizer.",
            "- Phishing parents currently come from Nazario only, so source diversity is unavailable for that class.",
            "- Translated augmentation may support training but cannot establish native Vietnamese generalization.",
            "- A native Vietnamese benchmark must remain independent and outside this augmentation.",
            "- Human review of the sample is still outstanding.",
            "",
            "## Integrity",
            "",
            f"- All `data/raw/` hashes unchanged: **{str(raw_unchanged).lower()}**.",
            f"- English train and existing Vietnamese input hashes unchanged: **{str(inputs_unchanged).lower()}**.",
            "- No classifier was trained and no model-selection or test prediction was used.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = load_augmentation_config(args.config)
    input_path = PROJECT_ROOT / config["input"]
    existing_vi_path = PROJECT_ROOT / config["existing_vietnamese_input"]
    output_path = PROJECT_ROOT / config["output"]
    sample_path = PROJECT_ROOT / config["review_sample_output"]
    report_path = PROJECT_ROOT / config["report"]
    checkpoint_path = output_path.with_suffix(".checkpoint.csv")

    for path in (input_path, existing_vi_path):
        if not path.exists():
            raise FileNotFoundError(f"Required augmentation input does not exist: {path}")

    raw_before = raw_hashes(PROJECT_ROOT / "data" / "raw")
    input_hashes_before = {
        str(input_path): sha256_file(input_path),
        str(existing_vi_path): sha256_file(existing_vi_path),
    }
    train = pd.read_csv(input_path, keep_default_na=False)
    existing_vi = pd.read_csv(existing_vi_path, keep_default_na=False)
    selected, selection_audit = select_parent_rows(train, config, existing_vi)
    selected_ids = set(selected["id"].astype(str))

    resume_path = checkpoint_path if checkpoint_path.exists() else output_path
    completed = _completed_rows(resume_path, selected_ids, config)
    completed_by_parent = {
        str(row.parent_id): row._asdict() for row in completed.itertuples(index=False)
    }
    missing = selected.loc[~selected["id"].astype(str).isin(completed_by_parent)]
    translator: MarianOfflineTranslator | None = None
    device_name = "not_loaded"
    generated: list[dict[str, Any]] = list(completed_by_parent.values())

    if not missing.empty:
        translator = MarianOfflineTranslator(
            config["model"], local_files_only=args.offline, force_device=args.device
        )
        device_name = translator.device_name
        row_batch_size = max(1, int(config["model"]["batch_size"]) // 2)
        total = len(missing)
        for start in range(0, total, row_batch_size):
            batch = missing.iloc[start:start + row_batch_size]
            generated.extend(translate_parent_batch(batch, translator, config))
            partial = pd.DataFrame(generated).drop_duplicates("parent_id", keep="last")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_checkpoint(partial, checkpoint_path)
            print(f"Translated {min(start + row_batch_size, total):,}/{total:,} new parents", flush=True)
    elif not completed.empty:
        device_name = "reused_existing_output"

    augmented = pd.DataFrame(generated)
    selected_order = {parent_id: index for index, parent_id in enumerate(selected["id"].astype(str))}
    augmented["_selection_order"] = augmented["parent_id"].astype(str).map(selected_order)
    augmented = augmented.sort_values("_selection_order", kind="stable").drop(
        columns="_selection_order"
    ).reset_index(drop=True)
    if len(augmented) != len(selected):
        raise RuntimeError(
            f"Expected {len(selected)} augmentation rows, generated {len(augmented)}"
        )
    # Outputs produced before runtime-device metadata was added can be migrated
    # without translating again. Passing --device records the known device that
    # produced those completed translations.
    if "translation_device" not in augmented.columns:
        augmented["translation_device"] = args.device or ""
    augmented = revalidate_existing_translations(augmented, selected, config)
    augmented = apply_duplicate_translation_flags(augmented, config)
    validate_parent_split_safety(augmented, str(config["required_parent_split"]))

    required_labels = set(config["allowed_parent_labels"])
    if not set(augmented["label"]).issubset(required_labels):
        raise RuntimeError("Unexpected label entered translated augmentation")
    if augmented.loc[
        augmented["translation_quality_status"].eq("failed"), "training_eligible"
    ].astype(bool).any():
        raise RuntimeError("A failed translation is marked training eligible")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(augmented, checkpoint_path)
    checkpoint_path.replace(output_path)

    review = validation_sample(
        augmented,
        selected,
        int(config["review_sample_per_class"]),
        int(config["seed"]),
    )
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(sample_path, index=False, encoding="utf-8-sig")

    raw_after = raw_hashes(PROJECT_ROOT / "data" / "raw")
    input_hashes_after = {
        str(input_path): sha256_file(input_path),
        str(existing_vi_path): sha256_file(existing_vi_path),
    }
    raw_unchanged = raw_before == raw_after
    inputs_unchanged = input_hashes_before == input_hashes_after
    if not raw_unchanged or not inputs_unchanged:
        raise RuntimeError("An immutable input changed during augmentation")

    recorded_devices = sorted(
        value
        for value in augmented["translation_device"].astype(str).unique()
        if value.strip()
    )
    if recorded_devices:
        device_name = ", ".join(recorded_devices)

    report_path.write_text(
        render_report(
            augmented,
            selected,
            selection_audit,
            config,
            device_name=device_name,
            reused_rows=len(completed),
            raw_unchanged=raw_unchanged,
            inputs_unchanged=inputs_unchanged,
        ),
        encoding="utf-8",
    )
    print("Translated augmentation complete")
    print(f"Parents selected: {len(selected):,}")
    print(f"Translations accepted: {augmented['training_eligible'].sum():,}")
    print(f"Translations failed quality checks: {(~augmented['training_eligible'].astype(bool)).sum():,}")
    print(f"Raw files unchanged: {raw_unchanged}")
    print(output_path.relative_to(PROJECT_ROOT).as_posix())
    print(sample_path.relative_to(PROJECT_ROOT).as_posix())
    print(report_path.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
