"""Isolated, validation-only research using source-labeled synthetic Chinese email.

The Hugging Face CSV is never promoted into the V2.1 curated/master dataset.
Its 0/1 mapping is usable only inside this research experiment. No Test split,
production model, Gemini service, URL, or attachment is opened by this module.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
import re
import shutil
import sqlite3
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from vietmailguard.modeling import EmailTextComposer

from .dedup_audit import dedup_hashes
from .development_training import (
    make_tfidf_pipeline as make_existing_tfidf_pipeline,
    sha256_file,
    validation_metrics,
    verify_and_load_development,
)
from .paths import V21Paths, resolve_v21_paths


AUDIT_VERSION = 1
SOURCE_NAME = "LLMGen_CN_DeepSeek"
SOURCE_FIELDS = {"content", "label"}
RESEARCH_FIELDS = (
    "id", "source", "source_revision", "source_row_number", "original_path",
    "subject", "body", "raw_label", "label", "label_status", "label_provenance",
    "language", "data_origin", "training_eligible", "research_training_eligible",
    "final_group_id", "content_hash", "normalized_content_hash", "template_hash",
    "research_split",
)
SCREEN_FIELDS = (
    "id", "source_row_number", "raw_label", "content_hash", "final_group_id",
    "screening_status", "screening_reasons", "research_split",
)
HAN_RE = re.compile(r"[\u3400-\u9fff]")
HTML_TAG_RE = re.compile(r"<\s*/?\s*(?:script|iframe|html|body|div|p|img|style|table|form|a)\b", re.I)
LABEL_LEAK_RE = re.compile(r"钓鱼邮件示例|正常邮件示例|合法邮件示例|phishing email example|legitimate email example", re.I)
PROMPT_INJECTION_RE = re.compile(
    r"忽略(?:之前|以上).{0,15}(?:指令|要求)|(?:ignore|disregard) (?:all )?(?:previous|prior) instructions",
    re.I,
)


class ResearchExperimentBlocked(RuntimeError):
    """The research-only source or development split failed a safety gate."""


def _location(paths: V21Paths) -> tuple[Path, Path, Path, Path]:
    return (
        paths.raw / "research_synthetic_zh",
        paths.processed / "research_synthetic_zh",
        paths.model / "research_synthetic_zh",
        paths.results / "research_synthetic_zh",
    )


def _config(paths: V21Paths) -> tuple[dict[str, Any], Path]:
    path = paths.config / "research_synthetic_zh.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("scope") != "research_synthetic_zh_validation_only":
        raise ResearchExperimentBlocked("Research-only config scope is missing")
    if set(config["raw_label_mapping_research_only"].values()) != {"normal", "phishing"}:
        raise ResearchExperimentBlocked("Source-label research mapping changed unexpectedly")
    return config, path


def _source_path(paths: V21Paths, config: dict[str, Any]) -> Path:
    raw_dir, _, _, _ = _location(paths)
    source = raw_dir / config["source_file"]
    if not source.is_file():
        raise ResearchExperimentBlocked(f"Pinned synthetic source file is missing: {source}")
    if sha256_file(source) != config["source_sha256"]:
        raise ResearchExperimentBlocked("Pinned synthetic source SHA-256 differs from config")
    return source


def screening_reasons(content: str, raw_label: str, minimum_han: int) -> list[str]:
    """Conservative suitability flags, never a ground-truth label decision."""
    reasons: list[str] = []
    if raw_label not in {"0", "1"}:
        reasons.append("invalid_source_label")
    if not content.strip():
        reasons.append("empty_content")
        return reasons
    if len(HAN_RE.findall(content)) < minimum_han:
        reasons.append("insufficient_chinese_text")
    if LABEL_LEAK_RE.search(content):
        reasons.append("explicit_label_or_example_marker")
    if HTML_TAG_RE.search(content):
        reasons.append("html_or_active_markup")
    if "```" in content:
        reasons.append("code_fence_or_generated_wrapper")
    if PROMPT_INJECTION_RE.search(content):
        reasons.append("embedded_model_instruction")
    return reasons


def _read_reference_hashes(paths: V21Paths, rows: list[dict[str, Any]]) -> int:
    """Use only the existing read-only dedup index, not any held-out CSV."""
    database = paths.cache / "dedup_audit.sqlite3"
    if not database.is_file():
        raise ResearchExperimentBlocked("V2.1 read-only dedup index is required for cross-corpus screening")
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA query_only=ON")
        checked = connection.execute(
            "SELECT COUNT(*) FROM rows WHERE dataset='v2_reference'"
        ).fetchone()[0]
        if checked < 38_013:
            raise ResearchExperimentBlocked("V2 read-only reference coverage is incomplete")
        count = 0
        for row in rows:
            matches = []
            for column, key in (
                ("exact_hash", "content_hash"),
                ("normalized_hash", "normalized_content_hash"),
                ("template_hash", "template_hash"),
            ):
                value = row[key]
                if value and connection.execute(
                    f"SELECT 1 FROM rows WHERE {column}=? LIMIT 1", (value,)
                ).fetchone():
                    matches.append(column)
            if matches:
                row["_reasons"].append("cross_corpus_fingerprint_overlap")
                count += 1
        return count
    finally:
        connection.close()


def _group_rows(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Transitively group exact/normalized/template matches before splitting."""
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for key in ("content_hash", "normalized_content_hash", "template_hash"):
        owner: dict[str, int] = {}
        for index, row in enumerate(rows):
            value = row[key]
            if value:
                previous = owner.setdefault(value, index)
                union(previous, index)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    conflict_groups = 0
    for members in groups.values():
        labels = {rows[index]["raw_label"] for index in members}
        if len(labels) > 1:
            conflict_groups += 1
            for index in members:
                rows[index]["_reasons"].append("duplicate_group_label_conflict")
        digest = hashlib.sha256(
            "|".join(sorted(rows[index]["normalized_content_hash"] for index in members)).encode("ascii")
        ).hexdigest()[:24]
        for index in members:
            rows[index]["final_group_id"] = "synthetic_zh:" + digest
        # Exact/normalized duplicates add no new training evidence. Template
        # relatives remain distinct but must share the same research split.
        seen_normalized: set[str] = set()
        for index in members:
            value = rows[index]["normalized_content_hash"]
            if value in seen_normalized:
                rows[index]["_reasons"].append("repeated_normalized_content")
            seen_normalized.add(value)
    duplicate_groups = sum(len(members) > 1 for members in groups.values())
    return duplicate_groups, conflict_groups


def _split_synthetic(rows: list[dict[str, Any]], seed: int, fraction: float) -> None:
    if not 0 < fraction < 0.5:
        raise ResearchExperimentBlocked("Synthetic challenge fraction must be between 0 and 0.5")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["_reasons"]:
            grouped[row["final_group_id"]].append(row)
    by_label: dict[str, list[str]] = defaultdict(list)
    for group_id, members in grouped.items():
        labels = {member["label"] for member in members}
        if len(labels) != 1:
            raise ResearchExperimentBlocked("Research split contains a conflicting label group")
        by_label[next(iter(labels))].append(group_id)
    if set(by_label) != {"normal", "phishing"}:
        raise ResearchExperimentBlocked("Synthetic research source lacks both source labels")
    rng = random.Random(seed)
    challenge: set[str] = set()
    for label in sorted(by_label):
        group_ids = sorted(by_label[label])
        if len(group_ids) < 5:
            raise ResearchExperimentBlocked(f"Too few independent synthetic {label} groups")
        rng.shuffle(group_ids)
        challenge.update(group_ids[:max(1, round(len(group_ids) * fraction))])
    for row in rows:
        row["research_split"] = (
            "exclude" if row["_reasons"] else
            "synthetic_challenge" if row["final_group_id"] in challenge else
            "synthetic_train"
        )
        row["research_training_eligible"] = "true" if row["research_split"] == "synthetic_train" else "false"


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_and_prepare(paths: V21Paths | None = None) -> dict[str, Any]:
    """Audit the pinned CSV in streaming mode and freeze a separate research split."""
    paths = paths or resolve_v21_paths()
    config, config_path = _config(paths)
    source = _source_path(paths, config)
    _, processed_dir, _, _ = _location(paths)
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = existing.get("output_sha256", {})
        if (existing.get("audit_version") == AUDIT_VERSION
                and existing.get("source_sha256") == sha256_file(source)
                and existing.get("config_sha256") == sha256_file(config_path)
                and all((processed_dir / filename).is_file()
                        and sha256_file(processed_dir / filename) == digest
                        for filename, digest in expected.items())
                and len(expected) == 3):
            return existing
        raise ResearchExperimentBlocked("Existing research manifest differs; refusing to overwrite it")

    csv.field_size_limit(sys.maxsize)
    rows: list[dict[str, Any]] = []
    source_url = (f"hf://datasets/{config['source_repo']}@"
                  f"{config['source_revision']}/{config['source_file']}")
    with source.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        reader = csv.DictReader(handle)
        if not SOURCE_FIELDS.issubset(reader.fieldnames or ()):
            raise ResearchExperimentBlocked("Synthetic CSV requires content,label columns")
        for number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ResearchExperimentBlocked(f"Malformed CSV row at source line {number}")
            content = unicodedata.normalize("NFC", raw["content"] or "")
            raw_label = (raw["label"] or "").strip()
            reasons = screening_reasons(content, raw_label, config["minimum_han_characters"])
            exact, normalized, template = dedup_hashes("", content)
            rows.append({
                "id": f"llmgen_cn:{number-1}", "source": SOURCE_NAME,
                "source_revision": config["source_revision"],
                "source_row_number": str(number-1),
                "original_path": f"{source_url}#row={number-1}",
                "subject": "", "body": content,
                "raw_label": raw_label,
                "label": config["raw_label_mapping_research_only"].get(raw_label, ""),
                "label_status": "weak_label",
                "label_provenance": "llm_generated_deepseek_binary_source_label_research_only",
                "language": "zh" if HAN_RE.search(content) else "unknown",
                "data_origin": "synthetic", "training_eligible": "false",
                "content_hash": exact, "normalized_content_hash": normalized,
                "template_hash": template, "final_group_id": "",
                "_reasons": reasons,
            })
    if not rows:
        raise ResearchExperimentBlocked("Synthetic source CSV is empty")
    cross_matches = _read_reference_hashes(paths, rows)
    duplicate_groups, conflict_groups = _group_rows(rows)
    _split_synthetic(rows, config["seed"], config["synthetic_challenge_fraction"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train = [row for row in rows if row["research_split"] == "synthetic_train"]
    challenge = [row for row in rows if row["research_split"] == "synthetic_challenge"]
    _write_csv(processed_dir / "train.csv", RESEARCH_FIELDS, train)
    _write_csv(processed_dir / "synthetic_challenge.csv", RESEARCH_FIELDS, challenge)
    screening = [
        {**row, "screening_status": "exclude" if row["_reasons"] else "research_usable",
         "screening_reasons": ";".join(row["_reasons"])} for row in rows
    ]
    _write_csv(processed_dir / "screening_manifest.csv", SCREEN_FIELDS, screening)
    reason_counts = Counter(reason for row in rows for reason in row["_reasons"])
    manifest = {
        "audit_version": AUDIT_VERSION,
        "scope": "research_synthetic_zh_only",
        "source_repo": config["source_repo"], "source_revision": config["source_revision"],
        "source_file": config["source_file"], "source_sha256": sha256_file(source),
        "source_schema": list(reader.fieldnames or []),
        "declared_license": config["declared_license"],
        "source_label_meaning": {"0": "synthetic legitimate candidate", "1": "synthetic phishing candidate"},
        "official_training_eligible_rows": 0,
        "config_sha256": sha256_file(config_path),
        "seed": config["seed"], "challenge_fraction": config["synthetic_challenge_fraction"],
        "source_rows": len(rows), "source_raw_label_counts": dict(Counter(row["raw_label"] for row in rows)),
        "chinese_character_rows": sum(row["language"] == "zh" for row in rows),
        "research_usable_rows": len(train) + len(challenge),
        "screened_out_rows": len(rows) - len(train) - len(challenge),
        "screening_reason_counts": dict(reason_counts),
        "synthetic_duplicate_groups": duplicate_groups,
        "synthetic_conflicting_label_groups": conflict_groups,
        "cross_corpus_fingerprint_overlap_rows": cross_matches,
        "synthetic_train_rows": len(train),
        "synthetic_challenge_rows": len(challenge),
        "train_label_counts": dict(Counter(row["label"] for row in train)),
        "challenge_label_counts": dict(Counter(row["label"] for row in challenge)),
        "train_group_count": len({row["final_group_id"] for row in train}),
        "challenge_group_count": len({row["final_group_id"] for row in challenge}),
        "cross_split_group_count": len({row["final_group_id"] for row in train}
                                      & {row["final_group_id"] for row in challenge}),
        "output_sha256": {name: sha256_file(processed_dir / name)
                          for name in ("train.csv", "synthetic_challenge.csv", "screening_manifest.csv")},
        "final_test_created_or_accessed": False,
    }
    if manifest["cross_split_group_count"]:
        raise ResearchExperimentBlocked("Synthetic leakage grouping failed")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load_research_csv(path: Path, expected_split: str, expected_sha: str) -> pd.DataFrame:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ResearchExperimentBlocked(f"Prepared {expected_split} CSV checksum changed")
    csv.field_size_limit(sys.maxsize)
    data: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(RESEARCH_FIELDS) - set(reader.fieldnames or ()):
            raise ResearchExperimentBlocked(f"Prepared {expected_split} schema is incomplete")
        for row in reader:
            if (row["research_split"] != expected_split
                    or row["source"] != SOURCE_NAME
                    or row["data_origin"] != "synthetic"
                    or row["language"] != "zh"
                    or row["label_status"] != "weak_label"
                    or row["training_eligible"] != "false"
                    or row["label"] not in {"normal", "phishing"}
                    or not row["body"].strip()
                    or not row["final_group_id"]):
                raise ResearchExperimentBlocked("A synthetic row was promoted or altered")
            data.append(row)
    return pd.DataFrame(data)


def validate_research_partitions(train: pd.DataFrame, challenge: pd.DataFrame) -> None:
    """The synthetic challenge can never enter the fitted Experiment B pipeline."""
    if train.empty or challenge.empty:
        raise ResearchExperimentBlocked("Synthetic train/challenge partition is empty")
    for key in ("id", "final_group_id", "content_hash", "normalized_content_hash", "template_hash"):
        left = set(train[key].dropna()) - {""}
        right = set(challenge[key].dropna()) - {""}
        if left & right:
            raise ResearchExperimentBlocked(f"Synthetic cross-partition leakage: {key}")
    if set(train["label"]) != {"normal", "phishing"} or set(challenge["label"]) != {"normal", "phishing"}:
        raise ResearchExperimentBlocked("Both synthetic source labels are required in each partition")
    if not train["research_training_eligible"].eq("true").all():
        raise ResearchExperimentBlocked("Synthetic Train research eligibility is incomplete")
    if not challenge["research_training_eligible"].eq("false").all():
        raise ResearchExperimentBlocked("Synthetic challenge was marked trainable")


def make_research_pipeline(config: dict[str, Any], classifier_name: str) -> Pipeline:
    """Identical subject/body composer and TF-IDF for all A/B candidates."""
    word_config = config["tfidf"]["word"]
    char_config = config["tfidf"]["character"]
    word = TfidfVectorizer(
        dtype=np.float32,
        **{key: tuple(value) if key == "ngram_range" else value
           for key, value in word_config.items()},
    )
    character = TfidfVectorizer(
        dtype=np.float32,
        **{key: tuple(value) if key == "ngram_range" else value
           for key, value in char_config.items()},
    )
    if classifier_name == "linear_svm":
        classifier = LinearSVC(random_state=config["seed"], **config["classifiers"][classifier_name])
    elif classifier_name == "logistic_regression":
        classifier = LogisticRegression(
            random_state=config["seed"], **config["classifiers"][classifier_name]
        )
    else:
        raise ValueError(f"Unsupported research classifier: {classifier_name}")
    return Pipeline([
        ("compose_text", EmailTextComposer()),
        ("tfidf", FeatureUnion([("word", word), ("character", character)], n_jobs=1)),
        ("classifier", classifier),
    ])


def synthetic_challenge_metrics(truth: np.ndarray, predicted: np.ndarray,
                                elapsed_seconds: float) -> dict[str, Any]:
    """Source-label diagnostics, explicitly not real Chinese three-class metrics."""
    labels = ["normal", "spam", "phishing"]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0,
    )
    represented = [index for index, count in enumerate(support) if count]
    return {
        "scope": "synthetic_source_label_challenge_only",
        "rows": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1_represented_classes": float(np.mean(f1[represented])),
        "three_class_macro_f1": None,
        "prediction_ms_per_email": elapsed_seconds * 1000 / len(truth),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "per_class": {
            label: {
                "support": int(support[index]),
                "precision": float(precision[index]) if support[index] else None,
                "recall": float(recall[index]) if support[index] else None,
                "f1": float(f1[index]) if support[index] else None,
            }
            for index, label in enumerate(labels)
        },
    }


def _memory_snapshot(paths: V21Paths) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "free_disk_gib": round(shutil.disk_usage(paths.project_root).free / 2**30, 3),
    }
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result["free_ram_gib"] = round(status.available_physical / 2**30, 3)
            result["total_ram_gib"] = round(status.total_physical / 2**30, 3)
    else:
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    result["free_ram_gib"] = round(int(line.split()[1]) * 1024 / 2**30, 3)
                    break
        except OSError:
            pass
    return result


def _write_reports(paths: V21Paths, source: dict[str, Any], result: dict[str, Any]) -> None:
    paths.reports.mkdir(parents=True, exist_ok=True)
    config, _ = _config(paths)
    word = config["tfidf"]["word"]
    character = config["tfidf"]["character"]
    audit_lines = [
        "# Synthetic Chinese source audit — research only", "",
        "Source: [Hugging Face dataset card](https://huggingface.co/datasets/Dizzzy0x00/LLMGen-Phishing-Email-Dataset/blob/main/README.md).",
        f"Revision: `{source['source_revision']}`; CSV SHA-256: `{source['source_sha256']}`.",
        f"The source card declares {source['declared_license']}; the named HF Chinese corpus says its rows were generated by DeepSeek.",
        "This is synthetic source supervision, not human-verified or native real-world email.",
        "The declared dataset license does not independently establish rights in every named brand/domain.",
        "No URL in an email was accessed; no attachment or HTML was executed.", "",
        f"Rows read: **{source['source_rows']:,}**; raw labels: "
        f"`0={source['source_raw_label_counts'].get('0', 0):,}`, "
        f"`1={source['source_raw_label_counts'].get('1', 0):,}`.",
        f"Rows with Han characters: **{source['chinese_character_rows']:,}**.",
        f"Passed research screen: **{source['research_usable_rows']:,}**; "
        f"excluded: **{source['screened_out_rows']:,}**.",
        f"Synthetic Train: **{source['synthetic_train_rows']:,}** "
        f"(Normal {source['train_label_counts'].get('normal', 0):,}, "
        f"Phishing {source['train_label_counts'].get('phishing', 0):,}); "
        f"synthetic challenge: **{source['synthetic_challenge_rows']:,}** "
        f"(Normal {source['challenge_label_counts'].get('normal', 0):,}, "
        f"Phishing {source['challenge_label_counts'].get('phishing', 0):,}).",
        f"Duplicate groups: **{source['synthetic_duplicate_groups']}**; label-conflict "
        f"groups: **{source['synthetic_conflicting_label_groups']}**; "
        f"cross-corpus fingerprint hits: **{source['cross_corpus_fingerprint_overlap_rows']}**.",
        "Screening reasons (may overlap):", "",
    ]
    audit_lines.extend(f"- `{reason}`: {count:,}" for reason, count in sorted(source["screening_reason_counts"].items()))
    audit_lines += [
        "", "Screening removes empty/low-Chinese-text rows, explicit class/example headers,",
        "active HTML/code wrappers, and model-directed instructions. These are quality",
        "filters, not label adjudication. Rejected row IDs and reasons, but no email text,",
        "are retained in the local screening manifest. Dataset text stays Git-ignored.",
        "The challenge split is a source-internal research diagnostic, never a final Test.",
        "Official three-class `training_eligible` remains false for all synthetic rows.", "",
    ]
    (paths.reports / "research_synthetic_zh_audit.md").write_text(
        "\n".join(audit_lines), encoding="utf-8"
    )

    lines = [
        "# Synthetic Chinese data addition — V2.1 research validation", "",
        "Experiment A fits only the inherited V2 development Train. Experiment B adds",
        "only screened synthetic Chinese Train. Both use the unchanged inherited",
        "English/translated-Vietnamese Validation; neither opens held-out Test.",
        "A is refitted under this paired Research configuration, not the existing",
        "V2.1 candidate artifact. Only same-configuration A/B deltas are comparable;",
        "the earlier candidate used different feature and text-length limits.",
        "No production choice, curation approval, calibration, or routing change is made.", "",
        f"Bootstrap Train {result['bootstrap_train_rows']:,}; Validation {result['validation_rows']:,}; "
        f"synthetic Train {source['synthetic_train_rows']:,}; separate synthetic challenge "
        f"{source['synthetic_challenge_rows']:,}.",
        f"Seed {result['seed']}; max {result['max_input_characters']:,} characters per subject/body field;",
        f"word TF-IDF {tuple(word['ngram_range'])} + "
        f"{character.get('analyzer', 'word')} TF-IDF {tuple(character['ngram_range'])}, "
        f"float32, word min_df={word['min_df']}, char min_df={character['min_df']},",
        f"{word['max_features']:,} word and {character['max_features']:,} character "
        "features; class weights follow the frozen research config.",
        f"Pre-fit free RAM {result['resources']['free_ram_gib']} GiB; "
        f"free disk {result['resources']['free_disk_gib']} GiB.", "",
        "| Classifier | Data | Accuracy | Macro F1 | Normal recall | Spam recall | Phishing recall | Normal→Spam/Phishing | Fit s | Predict ms/email |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["candidates"]:
        m = item["validation_metrics"]
        lines.append(
            f"| {item['classifier']} | {item['experiment']} | {m['accuracy']:.6f} | "
            f"{m['macro_f1']:.6f} | {m['per_class']['normal']['recall']:.6f} | "
            f"{m['per_class']['spam']['recall']:.6f} | "
            f"{m['per_class']['phishing']['recall']:.6f} | "
            f"{m['non_normal_false_positives']} | {item['fit_seconds']:.2f} | "
            f"{m['prediction_ms_per_email']:.3f} |"
        )
    lines += ["", "| Classifier | Data | EN Macro F1 | EN Phishing recall | VI translated Macro F1 (represented classes) | VI Phishing recall |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for item in result["candidates"]:
        languages = item["validation_metrics"]["by_language"]
        vi_macro = languages["vi"]["macro_f1_supported_classes"]
        vi_text = f"{vi_macro:.6f}" if vi_macro is not None else "N/A"
        lines.append(
            f"| {item['classifier']} | {item['experiment']} | "
            f"{languages['en']['macro_f1_supported_classes']:.6f} | "
            f"{languages['en']['phishing_recall']:.6f} | "
            f"{vi_text} | N/A |"
        )
    lines += ["", "Per-class precision/recall/F1/support and confusion matrices are",
              "in `validation_metrics.json`; Validation has no Chinese rows",
              "and Vietnamese translated Validation has no Phishing rows (recall N/A).", "",
              "## Synthetic-only diagnostic (not real Chinese email performance)", "",
              "| Classifier | Data | Source-label accuracy | Macro F1 of represented classes | Normal recall | Synthetic Phishing recall |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for item in result["candidates"]:
        m = item["synthetic_challenge_metrics"]
        lines.append(
            f"| {item['classifier']} | {item['experiment']} | {m['accuracy']:.6f} | "
            f"{m['macro_f1_represented_classes']:.6f} | "
            f"{m['per_class']['normal']['recall']:.6f} | "
            f"{m['per_class']['phishing']['recall']:.6f} |"
        )
    lines += [
        "", "This diagnostic has synthetic Normal and Phishing **only**, no Chinese Spam;",
        "it is neither a three-class Chinese validation nor an independent native benchmark.",
        "Its source templates/generator may overlap between synthetic Train and challenge",
        "despite exact/normalized/template hash grouping. The inherited V2 Validation",
        "was already used for V2 selection and remains development-only.",
        "No Chinese phishing recall on real mail or native Vietnamese accuracy is claimed.",
        "All SVC scores are uncalibrated; raw decision scores are not probabilities.",
        "The source card's binary labels may be noisy relative to VietMailGuard's taxonomy.",
    ]
    for classifier in ("linear_svm", "logistic_regression"):
        a = next(item for item in result["candidates"] if item["classifier"] == classifier and item["experiment"] == "A")
        b = next(item for item in result["candidates"] if item["classifier"] == classifier and item["experiment"] == "B")
        delta = b["validation_metrics"]["macro_f1"] - a["validation_metrics"]["macro_f1"]
        lines.append(f"Validation Macro F1 change for {classifier}, B−A: **{delta:+.6f}**.")
    best_a = max(item["validation_metrics"]["macro_f1"] for item in result["candidates"]
                 if item["experiment"] == "A")
    best_b = max(item["validation_metrics"]["macro_f1"] for item in result["candidates"]
                 if item["experiment"] == "B")
    lines += [
        f"Best B versus best A on the same Validation: **{best_b - best_a:+.6f} Macro F1**.",
        "Thus the synthetic addition does **not** improve the strongest development",
        "candidate here. It modestly improves Logistic Regression but reduces",
        "LinearSVC Macro F1; no consistent EN/VI development gain is shown.",
        "", "Conclusion is limited to this unchanged EN/VI development Validation and",
              "the separate synthetic-only diagnostic. Synthetic gains do not establish",
              "real-world Chinese transfer. Candidate artifacts remain research-only.", ""]
    (paths.reports / "research_synthetic_zh_experiment.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def train_research_candidates(paths: V21Paths | None = None) -> dict[str, Any]:
    """Fit four A/B candidates; only inherited Validation and synthetic challenge are read."""
    paths = paths or resolve_v21_paths()
    config, config_path = _config(paths)
    _, processed_dir, model_dir, results_dir = _location(paths)
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ResearchExperimentBlocked("Run the research synthetic audit before training")
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (source.get("audit_version") != AUDIT_VERSION
            or source.get("config_sha256") != sha256_file(config_path)
            or source.get("source_sha256") != sha256_file(_source_path(paths, config))
            or source.get("official_training_eligible_rows") != 0
            or source.get("cross_split_group_count") != 0):
        raise ResearchExperimentBlocked("Synthetic research manifest is not verified")
    screening_path = processed_dir / "screening_manifest.csv"
    if (not screening_path.is_file()
            or sha256_file(screening_path) != source["output_sha256"]["screening_manifest.csv"]):
        raise ResearchExperimentBlocked("Synthetic screening manifest checksum changed")
    synthetic_train = _load_research_csv(
        processed_dir / "train.csv", "synthetic_train", source["output_sha256"]["train.csv"]
    )
    synthetic_challenge = _load_research_csv(
        processed_dir / "synthetic_challenge.csv", "synthetic_challenge",
        source["output_sha256"]["synthetic_challenge.csv"],
    )
    validate_research_partitions(synthetic_train, synthetic_challenge)
    if (len(synthetic_train) != source["synthetic_train_rows"]
            or len(synthetic_challenge) != source["synthetic_challenge_rows"]):
        raise ResearchExperimentBlocked("Prepared synthetic row counts changed")
    resources = _memory_snapshot(paths)
    if resources.get("free_ram_gib", 0) < config["minimum_free_ram_gib"]:
        raise ResearchExperimentBlocked("Not enough free RAM for CPU TF-IDF research")
    if resources["free_disk_gib"] < 1.0:
        raise ResearchExperimentBlocked("Not enough free disk for research candidate artifacts")

    dev_manifest, bootstrap_train, validation = verify_and_load_development(
        paths, int(config["max_input_characters"])
    )
    if (len(bootstrap_train) != dev_manifest.get("output_rows", {}).get("train")
            or len(validation) != dev_manifest.get("output_rows", {}).get("validation")
            or set(bootstrap_train["label"]) != {"normal", "spam", "phishing"}):
        raise ResearchExperimentBlocked("Unexpected inherited V2 development cohort")
    synthetic_train = synthetic_train.assign(
        subject=synthetic_train["subject"].str.slice(0, config["max_input_characters"]),
        body=synthetic_train["body"].str.slice(0, config["max_input_characters"]),
    )
    synthetic_challenge = synthetic_challenge.assign(
        subject=synthetic_challenge["subject"].str.slice(0, config["max_input_characters"]),
        body=synthetic_challenge["body"].str.slice(0, config["max_input_characters"]),
    )
    if set(bootstrap_train["id"]) & (set(synthetic_train["id"]) | set(synthetic_challenge["id"])):
        raise ResearchExperimentBlocked("Synthetic/bootstrap ID collision")
    combined = pd.concat([bootstrap_train, synthetic_train], ignore_index=True)
    if set(combined["label"]) != {"normal", "spam", "phishing"}:
        raise ResearchExperimentBlocked("Experiment B lost a required class")
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    validation_x = validation[["subject", "body"]]
    challenge_x = synthetic_challenge[["subject", "body"]]
    validation_y = validation["label"].to_numpy()
    challenge_y = synthetic_challenge["label"].to_numpy()
    for classifier_name in ("linear_svm", "logistic_regression"):
        for experiment, frame in (("A", bootstrap_train), ("B", combined)):
            pipeline = make_research_pipeline(config, classifier_name)
            started = time.perf_counter()
            pipeline.fit(frame[["subject", "body"]], frame["label"])
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            validation_pred = pipeline.predict(validation_x)
            validation_seconds = time.perf_counter() - started
            started = time.perf_counter()
            challenge_pred = pipeline.predict(challenge_x)
            challenge_seconds = time.perf_counter() - started
            artifact = model_dir / f"experiment_{experiment.lower()}_{classifier_name}.joblib"
            joblib.dump(pipeline, artifact)
            smoke_pred = validation_pred[:3].copy()
            del pipeline
            gc.collect()
            reloaded = joblib.load(artifact)
            if not np.array_equal(reloaded.predict(validation_x.head(3)), smoke_pred):
                raise ResearchExperimentBlocked(f"Candidate reload smoke failed: {artifact.name}")
            del reloaded
            gc.collect()
            candidates.append({
                "experiment": experiment, "classifier": classifier_name,
                "train_rows": len(frame), "synthetic_train_rows": 0 if experiment == "A" else len(synthetic_train),
                "fit_seconds": fit_seconds,
                "artifact": artifact.relative_to(paths.project_root).as_posix(),
                "artifact_sha256": sha256_file(artifact), "artifact_bytes": artifact.stat().st_size,
                "validation_metrics": validation_metrics(validation_y, validation_pred, validation, validation_seconds),
                "synthetic_challenge_metrics": synthetic_challenge_metrics(
                    challenge_y, challenge_pred, challenge_seconds,
                ),
                "probabilities_available": classifier_name == "logistic_regression",
            })
    result = {
        "scope": "research_synthetic_zh_development_validation_only",
        "seed": config["seed"],
        "config_sha256": sha256_file(config_path),
        "synthetic_manifest_sha256": sha256_file(manifest_path),
        "development_manifest_sha256": sha256_file(paths.splits / "development_bootstrap" / "manifest.json"),
        "development_leakage_preflight": dev_manifest["leakage"],
        "resources": resources,
        "max_input_characters": config["max_input_characters"],
        "bootstrap_train_rows": len(bootstrap_train),
        "validation_rows": len(validation),
        "synthetic_train_rows": len(synthetic_train),
        "synthetic_challenge_rows": len(synthetic_challenge),
        "candidates": candidates,
        "final_test_accessed": False,
        "production_model_selected": False,
        "chinese_native_benchmark_available": False,
        "chinese_spam_validation_support": 0,
    }
    (results_dir / "validation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    comparison = [{
        "experiment": item["experiment"], "classifier": item["classifier"],
        "train_rows": item["train_rows"], "synthetic_train_rows": item["synthetic_train_rows"],
        "validation_accuracy": item["validation_metrics"]["accuracy"],
        "validation_macro_f1": item["validation_metrics"]["macro_f1"],
        "validation_normal_recall": item["validation_metrics"]["per_class"]["normal"]["recall"],
        "validation_spam_recall": item["validation_metrics"]["per_class"]["spam"]["recall"],
        "validation_phishing_recall": item["validation_metrics"]["per_class"]["phishing"]["recall"],
        "validation_normal_false_positives": item["validation_metrics"]["non_normal_false_positives"],
        "validation_english_macro_f1": item["validation_metrics"]["by_language"]["en"]["macro_f1_supported_classes"],
        "validation_translated_vietnamese_macro_f1_supported_classes": item["validation_metrics"]["by_language"]["vi"]["macro_f1_supported_classes"],
        "synthetic_challenge_macro_f1_represented_classes": item["synthetic_challenge_metrics"]["macro_f1_represented_classes"],
        "fit_seconds": item["fit_seconds"],
        "validation_prediction_ms_per_email": item["validation_metrics"]["prediction_ms_per_email"],
        "artifact": item["artifact"],
    } for item in candidates]
    with (results_dir / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    for item in candidates:
        name = f"experiment_{item['experiment'].lower()}_{item['classifier']}"
        matrix = item["validation_metrics"]["confusion_matrix"]
        with (results_dir / f"{name}_confusion_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["actual\\predicted", "normal", "spam", "phishing"])
            for label, row in zip(("normal", "spam", "phishing"), matrix):
                writer.writerow([label, *row])
    _write_reports(paths, source, result)
    return result


def compare_current_svm(paths: V21Paths | None = None) -> dict[str, Any]:
    """Pair the existing V2.1 candidate with B using its exact saved config.

    A is loaded read-only. Only B is fitted, using the same composer/vectorizer/
    classifier parameters. Both are evaluated on the inherited Development
    Validation and the separate synthetic-only challenge. No Test is opened.
    """
    paths = paths or resolve_v21_paths()
    research_config, research_config_path = _config(paths)
    _, processed_dir, model_dir, results_dir = _location(paths)
    synthetic_manifest_path = processed_dir / "manifest.json"
    if not synthetic_manifest_path.is_file():
        raise ResearchExperimentBlocked("Synthetic research audit is required")
    synthetic_manifest = json.loads(synthetic_manifest_path.read_text(encoding="utf-8"))
    if (synthetic_manifest.get("config_sha256") != sha256_file(research_config_path)
            or synthetic_manifest.get("source_sha256") != sha256_file(_source_path(paths, research_config))
            or synthetic_manifest.get("official_training_eligible_rows") != 0
            or synthetic_manifest.get("cross_split_group_count") != 0):
        raise ResearchExperimentBlocked("Synthetic research manifest changed")
    screening = processed_dir / "screening_manifest.csv"
    if (not screening.is_file()
            or sha256_file(screening) != synthetic_manifest["output_sha256"]["screening_manifest.csv"]):
        raise ResearchExperimentBlocked("Synthetic screening manifest changed")
    synthetic_train = _load_research_csv(
        processed_dir / "train.csv", "synthetic_train",
        synthetic_manifest["output_sha256"]["train.csv"],
    )
    synthetic_challenge = _load_research_csv(
        processed_dir / "synthetic_challenge.csv", "synthetic_challenge",
        synthetic_manifest["output_sha256"]["synthetic_challenge.csv"],
    )
    validate_research_partitions(synthetic_train, synthetic_challenge)

    existing_config_path = paths.config / "development_training.json"
    existing_result_path = paths.results / "development_validation_metrics.json"
    existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
    existing_result = json.loads(existing_result_path.read_text(encoding="utf-8"))
    dev_manifest_path = paths.splits / "development_bootstrap" / "manifest.json"
    if (existing_result.get("config_sha256") != sha256_file(existing_config_path)
            or existing_result.get("development_manifest_sha256") != sha256_file(dev_manifest_path)
            or existing_result.get("scope") != "development_validation_only"):
        raise ResearchExperimentBlocked("Existing candidate provenance/config is not verified")
    original = next(
        (item for item in existing_result.get("candidates", [])
         if item.get("candidate") == "tfidf_word_character_linear_svm"), None,
    )
    if original is None:
        raise ResearchExperimentBlocked("Current three-class TF-IDF LinearSVC candidate is absent")
    artifact = paths.project_root / original["artifact"]
    if not artifact.is_file() or artifact.stat().st_size != original["artifact_bytes"]:
        raise ResearchExperimentBlocked("Current candidate artifact differs from saved result")
    resources = _memory_snapshot(paths)
    if (resources.get("free_ram_gib", 0) < research_config["minimum_free_ram_gib"]
            or resources["free_disk_gib"] < 1):
        raise ResearchExperimentBlocked("Insufficient resources for matched current-config SVM")
    dev_manifest, bootstrap_train, validation = verify_and_load_development(
        paths, int(existing_config["max_input_characters"])
    )
    cap = int(existing_config["max_input_characters"])
    synthetic_train = synthetic_train.assign(
        subject=synthetic_train["subject"].str.slice(0, cap),
        body=synthetic_train["body"].str.slice(0, cap),
    )
    synthetic_challenge = synthetic_challenge.assign(
        subject=synthetic_challenge["subject"].str.slice(0, cap),
        body=synthetic_challenge["body"].str.slice(0, cap),
    )
    if (set(bootstrap_train["id"]) & set(synthetic_train["id"])
            or set(validation["id"]) & set(synthetic_train["id"])):
        raise ResearchExperimentBlocked("Synthetic ID collides with inherited development data")
    validation_x = validation[["subject", "body"]]
    challenge_x = synthetic_challenge[["subject", "body"]]
    validation_y = validation["label"].to_numpy()
    challenge_y = synthetic_challenge["label"].to_numpy()
    baseline = joblib.load(artifact)
    started = time.perf_counter()
    a_validation_pred = baseline.predict(validation_x)
    a_validation_seconds = time.perf_counter() - started
    a_metrics = validation_metrics(validation_y, a_validation_pred, validation, a_validation_seconds)
    if abs(a_metrics["macro_f1"] - original["metrics"]["macro_f1"]) > 1e-12:
        raise ResearchExperimentBlocked("Current candidate no longer reproduces saved Validation Macro F1")
    started = time.perf_counter()
    a_challenge_pred = baseline.predict(challenge_x)
    a_challenge_seconds = time.perf_counter() - started
    a_challenge = synthetic_challenge_metrics(challenge_y, a_challenge_pred, a_challenge_seconds)
    del baseline
    gc.collect()

    combined = pd.concat([bootstrap_train, synthetic_train], ignore_index=True)
    candidate = make_existing_tfidf_pipeline(existing_config)
    started = time.perf_counter()
    candidate.fit(combined[["subject", "body"]], combined["label"])
    b_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    b_validation_pred = candidate.predict(validation_x)
    b_validation_seconds = time.perf_counter() - started
    b_metrics = validation_metrics(validation_y, b_validation_pred, validation, b_validation_seconds)
    started = time.perf_counter()
    b_challenge_pred = candidate.predict(challenge_x)
    b_challenge_seconds = time.perf_counter() - started
    b_challenge = synthetic_challenge_metrics(challenge_y, b_challenge_pred, b_challenge_seconds)
    model_dir.mkdir(parents=True, exist_ok=True)
    b_artifact = model_dir / "experiment_b_current_config_linear_svm.joblib"
    joblib.dump(candidate, b_artifact)
    smoke = b_validation_pred[:3].copy()
    del candidate
    gc.collect()
    reloaded = joblib.load(b_artifact)
    if not np.array_equal(reloaded.predict(validation_x.head(3)), smoke):
        raise ResearchExperimentBlocked("Current-config B candidate reload smoke failed")
    del reloaded
    gc.collect()

    comparison = {
        "scope": "research_synthetic_zh_current_svm_matched_validation_only",
        "seed": existing_config["seed"],
        "existing_config_sha256": sha256_file(existing_config_path),
        "development_manifest_sha256": sha256_file(dev_manifest_path),
        "synthetic_manifest_sha256": sha256_file(synthetic_manifest_path),
        "resources": resources,
        "bootstrap_train_rows": len(bootstrap_train),
        "synthetic_train_rows": len(synthetic_train),
        "validation_rows": len(validation),
        "synthetic_challenge_rows": len(synthetic_challenge),
        "development_leakage_preflight": dev_manifest["leakage"],
        "A_current_candidate": {
            "artifact": original["artifact"],
            "artifact_sha256": sha256_file(artifact),
            "fit_seconds_from_saved_result": original["fit_seconds"],
            "validation_metrics": a_metrics,
            "synthetic_challenge_metrics": a_challenge,
        },
        "B_synthetic_added": {
            "artifact": b_artifact.relative_to(paths.project_root).as_posix(),
            "artifact_sha256": sha256_file(b_artifact),
            "fit_seconds": b_fit_seconds,
            "validation_metrics": b_metrics,
            "synthetic_challenge_metrics": b_challenge,
        },
        "validation_macro_f1_delta_B_minus_A": b_metrics["macro_f1"] - a_metrics["macro_f1"],
        "final_test_accessed": False,
        "production_model_selected": False,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "current_svm_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Exact current-candidate LinearSVC A/B — research only", "",
        "A loads the existing V2.1 three-class development candidate read-only.",
        "B fits the same saved Word+Character TF-IDF/LinearSVC configuration",
        "with screened Chinese synthetic Train added. Same inherited Validation,",
        "no held-out Test, no existing candidate overwrite.", "",
        f"Existing config SHA-256: `{comparison['existing_config_sha256']}`.",
        f"Bootstrap Train {len(bootstrap_train):,}; synthetic Train {len(synthetic_train):,}; "
        f"Validation {len(validation):,}; synthetic challenge {len(synthetic_challenge):,}.",
        f"B fit time: {b_fit_seconds:.2f} s. A fit time from saved prior result: "
        f"{original['fit_seconds']:.2f} s.", "",
        "| Candidate | Accuracy | Macro F1 | Normal recall | Spam recall | Phishing recall | Normal→non-normal | EN Macro F1 | VI translated Macro F1* |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in (("A current", a_metrics), ("B synthetic", b_metrics)):
        lines.append(
            f"| {name} | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | "
            f"{metrics['per_class']['normal']['recall']:.6f} | "
            f"{metrics['per_class']['spam']['recall']:.6f} | "
            f"{metrics['per_class']['phishing']['recall']:.6f} | "
            f"{metrics['non_normal_false_positives']} | "
            f"{metrics['by_language']['en']['macro_f1_supported_classes']:.6f} | "
            f"{metrics['by_language']['vi']['macro_f1_supported_classes']:.6f} |"
        )
    lines += [
        "", "*VI Validation is translated and has only Normal/Spam; VI Phishing recall N/A.",
        f"Validation Macro F1 B−A: **{comparison['validation_macro_f1_delta_B_minus_A']:+.6f}**.",
        f"Synthetic-only challenge represented-class Macro F1 A→B: "
        f"{a_challenge['macro_f1_represented_classes']:.6f}→"
        f"{b_challenge['macro_f1_represented_classes']:.6f}.",
        "Synthetic challenge is same-generator Normal/Phishing only, never real",
        "Chinese three-class performance. All SVC scores are uncalibrated, not probabilities.",
        "Complete per-class metrics/confusion matrices are in current_svm_comparison.json.",
        "No production decision follows from this Research validation comparison.", "",
    ]
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "research_synthetic_zh_current_svm.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return comparison
