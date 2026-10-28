"""Isolated TREC06C source-label binary experiment (ham/spam, never VMG classes).

The preparation stage freezes a group-safe test split. The training stage has
no test-file input and selects a candidate using validation only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from vietmailguard.modeling import EmailTextComposer
from .paths import V21Paths, resolve_v21_paths


SEED = 42
LABELS = ("ham", "spam")
SOURCE_DIR = "trec06c_mime_v2"
EXPERIMENT = "trec06c_binary"
MAX_TEXT_CHARACTERS = 2000
FIELDS = (
    "id", "source", "original_path", "raw_label", "subject", "body",
    "language_candidate", "content_hash", "exact_hash", "normalized_hash",
    "template_hash", "final_group_id", "split",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(paths: V21Paths) -> tuple[Path, Path, Path]:
    return (paths.splits / EXPERIMENT, paths.results / EXPERIMENT,
            paths.model / EXPERIMENT)


def _parts(paths: V21Paths) -> list[Path]:
    parts = sorted((paths.processed / SOURCE_DIR).glob("part-*.csv"))
    if not parts:
        raise FileNotFoundError("Authoritative TREC06C MIME-v2 parts are missing")
    return parts


def _audit_rows(paths: V21Paths) -> tuple[dict[str, tuple[str, str, str]], sqlite3.Connection]:
    audit_path = paths.cache / "dedup_audit.sqlite3"
    if not audit_path.is_file():
        raise FileNotFoundError("TREC dedup audit index is required")
    conn = sqlite3.connect(f"file:{audit_path.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    rows = {
        identifier: (exact or "", normalized or "", template or "")
        for identifier, exact, normalized, template in conn.execute(
            "SELECT original_id,exact_hash,normalized_hash,template_hash "
            "FROM rows WHERE dataset='trec06c'"
        )
    }
    return rows, conn


def quality_reason(row: dict[str, str]) -> str:
    """Conservative text-quality gate, not a three-class label approval."""
    if row.get("raw_label") not in LABELS or row.get("source") != "TREC06C":
        return "unexpected_source_or_label"
    if row.get("status") != "ok" or row.get("empty", "").lower() == "true":
        return "parse_or_empty"
    if row.get("missing_content", "").lower() == "true" or not row.get("body", "").strip():
        return "missing_body"
    if row.get("decode_error", "").lower() == "true":
        return "decode_warning"
    if row.get("contains_chinese", "").lower() != "true":
        return "no_han_character"
    if not row.get("id") or not row.get("original_path") or not row.get("content_hash"):
        return "missing_provenance"
    return ""


def _root(parent: dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: dict[str, str], left: str, right: str) -> None:
    a, b = _root(parent, left), _root(parent, right)
    if a != b:
        parent[max(a, b)] = min(a, b)


def assign_groups(records: dict[str, dict[str, str]], seed: int = SEED) -> tuple[dict[str, str], dict]:
    """Union exact/normalized/conservative template links, then split groups."""
    parent = {identifier: identifier for identifier in records}
    fingerprints: dict[tuple[str, str], str] = {}
    for identifier, row in records.items():
        for key in ("exact_hash", "normalized_hash", "template_hash"):
            value = row.get(key, "")
            if value:
                previous = fingerprints.setdefault((key, value), identifier)
                _union(parent, identifier, previous)
    groups: dict[str, list[str]] = defaultdict(list)
    for identifier in records:
        groups[_root(parent, identifier)].append(identifier)
    conflicts = {root for root, ids in groups.items()
                 if len({records[identifier]["raw_label"] for identifier in ids}) > 1}
    rng = random.Random(seed)
    ordered = [(rng.random(), root, ids) for root, ids in groups.items() if root not in conflicts]
    ordered.sort(key=lambda item: (-len(item[2]), item[0]))
    totals = Counter(records[ids[0]]["raw_label"] for _, _, ids in ordered for _ in ids)
    targets = {label: {"train": totals[label] * .70, "validation": totals[label] * .15,
                       "test": totals[label] * .15} for label in LABELS}
    actual: dict[str, Counter] = {label: Counter() for label in LABELS}
    assignment: dict[str, str] = {}
    for _, root, ids in ordered:
        label = records[ids[0]]["raw_label"]
        # Largest remaining class-specific deficit; deterministic tie order.
        split = max(("train", "validation", "test"),
                    key=lambda name: (targets[label][name] - actual[label][name],
                                      {"train": 2, "validation": 1, "test": 0}[name]))
        actual[label][split] += len(ids)
        group_id = "trec_" + hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:24]
        for identifier in ids:
            assignment[identifier] = split + "|" + group_id
    return assignment, {
        "groups_total": len(groups), "groups_assigned": len(ordered),
        "conflicting_groups": len(conflicts),
        "conflicting_rows": sum(len(groups[root]) for root in conflicts),
        "class_targets": targets,
    }


def verify_no_leakage(records: dict[str, dict[str, str]], assignment: dict[str, str]) -> dict:
    owners: dict[tuple[str, str], str] = {}
    checked = Counter()
    for identifier, value in assignment.items():
        split, group = value.split("|", 1)
        for key in ("final_group_id", "exact_hash", "normalized_hash", "template_hash"):
            fingerprint = group if key == "final_group_id" else records[identifier].get(key, "")
            if not fingerprint:
                continue
            owner = owners.setdefault((key, fingerprint), split)
            if owner != split:
                raise ValueError(f"Cross-split leakage: {key}")
            checked[key] += 1
    return {"passed": True, "checked_rows": len(assignment),
            "checked_fingerprints": dict(checked),
            "groups": len({value.split("|", 1)[1] for value in assignment.values()})}


def prepare_experiment(paths: V21Paths | None = None) -> dict:
    """Freeze splits atomically before any fitting; re-runs cannot reassign Test."""
    paths = paths or resolve_v21_paths()
    split_dir, result_dir, _ = _paths(paths)
    parts = _parts(paths)
    manifest_path = split_dir / "manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("input_sha256") != {str(p.resolve()): sha256_file(p) for p in parts}:
            raise RuntimeError("Frozen TREC input changed; refusing to reassign Test")
        for name in ("train", "validation", "test"):
            if sha256_file(split_dir / f"{name}.csv") != old["split_sha256"][name]:
                raise RuntimeError(f"Frozen {name} split changed")
        return old
    audit, conn = _audit_rows(paths)
    csv.field_size_limit(100_000_000)
    records: dict[str, dict[str, str]] = {}
    read = Counter()
    rejected = Counter()
    try:
        for part in parts:
            stat = part.stat()
            checkpoint = conn.execute("SELECT size_bytes,mtime_ns FROM checkpoints WHERE path=?",
                                      (str(part.resolve()),)).fetchone()
            if checkpoint != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Part differs from dedup audit checkpoint: {part.name}")
            with part.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    read[row.get("raw_label", "")] += 1
                    reason = quality_reason(row)
                    if reason:
                        rejected[reason] += 1
                        continue
                    identifier = row["id"]
                    if identifier in records or identifier not in audit:
                        raise RuntimeError("Duplicate id or missing dedup audit row")
                    exact, normalized, template = audit[identifier]
                    if not exact or not normalized:
                        raise RuntimeError("Missing audited text fingerprints")
                    records[identifier] = {"raw_label": row["raw_label"],
                                           "exact_hash": exact, "normalized_hash": normalized,
                                           "template_hash": template}
    finally:
        conn.close()
    if sum(read.values()) != len(audit):
        raise RuntimeError("TREC row count differs from dedup audit")
    assignment, group_stats = assign_groups(records)
    leakage = verify_no_leakage(records, assignment)
    split_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = {name: split_dir / f"{name}.csv.tmp" for name in ("train", "validation", "test")}
    handles = {name: path.open("w", encoding="utf-8", newline="") for name, path in temp_paths.items()}
    writers = {name: csv.DictWriter(handles[name], fieldnames=FIELDS, extrasaction="ignore") for name in handles}
    counts = {name: Counter() for name in handles}
    try:
        for writer in writers.values():
            writer.writeheader()
        for part in parts:
            with part.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    value = assignment.get(row["id"])
                    if not value:
                        continue
                    split, group = value.split("|", 1)
                    output = {key: row.get(key, "") for key in FIELDS}
                    output.update(records[row["id"]])
                    output["final_group_id"] = group
                    output["split"] = split
                    writers[split].writerow(output)
                    counts[split][row["raw_label"]] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if any(set(counts[name]) != set(LABELS) for name in counts):
        raise RuntimeError("Every split must contain both binary source labels")
    for name, temporary in temp_paths.items():
        temporary.replace(split_dir / f"{name}.csv")
    manifest = {
        "experiment": EXPERIMENT, "task": "TREC06C source Ham/Spam binary only",
        "not_vietmailguard_three_class_ground_truth": True, "seed": SEED,
        "ratio_target": {"train": .70, "validation": .15, "test": .15},
        "input_sha256": {str(p.resolve()): sha256_file(p) for p in parts},
        "raw_labels": dict(read), "quality_rejected": dict(rejected),
        "quality_accepted": len(records), "group_stats": group_stats,
        "split_counts": {name: dict(count) for name, count in counts.items()},
        "split_sha256": {name: sha256_file(split_dir / f"{name}.csv") for name in counts},
        "leakage": leakage, "test_frozen": True, "test_evaluated": False,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    (result_dir / "split_report.json").write_text(json.dumps({
        "raw_labels": manifest["raw_labels"], "quality_accepted": len(records),
        "quality_rejected": dict(rejected), "split_counts": manifest["split_counts"],
        "group_stats": group_stats, "test_evaluated": False,
    }, indent=2), encoding="utf-8")
    (result_dir / "leakage_report.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    return manifest


def _load_development(split_dir: Path, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for name in ("train", "validation"):
        path = split_dir / f"{name}.csv"
        if sha256_file(path) != manifest["split_sha256"][name]:
            raise RuntimeError(f"{name} differs from frozen manifest")
        frame = pd.read_csv(path, usecols=["id", "subject", "body", "raw_label", "final_group_id", "content_hash"],
                            dtype=str, keep_default_na=False)
        if set(frame.raw_label) != set(LABELS) or frame.id.duplicated().any():
            raise RuntimeError(f"Invalid {name} split")
        frames.append(frame)
    if set(frames[0].final_group_id) & set(frames[1].final_group_id):
        raise RuntimeError("Development group leakage")
    return frames[0], frames[1]


def _pipelines() -> dict[str, Pipeline]:
    word = lambda: TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                   max_features=20_000, sublinear_tf=True, dtype=np.float32)
    char = lambda: TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2,
                                   max_features=50_000, sublinear_tf=True, dtype=np.float32)
    return {
        "char_tfidf_linear_svc": Pipeline([
            ("compose", EmailTextComposer()),
            ("bounded_text", FunctionTransformer(bounded_text)), ("tfidf", char()),
            ("classifier", LinearSVC(C=1.0, random_state=SEED)),
        ]),
        "word_char_tfidf_logistic_regression": Pipeline([
            ("compose", EmailTextComposer()),
            ("bounded_text", FunctionTransformer(bounded_text)),
            ("tfidf", FeatureUnion([("word", word()), ("char", char())])),
            ("classifier", LogisticRegression(max_iter=500, random_state=SEED)),
        ]),
    }


def bounded_text(texts: np.ndarray) -> np.ndarray:
    """Use identical bounded representation at fit and future candidate inference."""
    return np.asarray([str(text)[:MAX_TEXT_CHARACTERS] for text in texts], dtype=object)


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=LABELS, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(LABELS)},
        "confusion_matrix_order": list(LABELS), "confusion_matrix": matrix.tolist(),
        "false_positives_ham_as_spam": int(matrix[0, 1]),
        "false_negatives_spam_as_ham": int(matrix[1, 0]),
    }


def train_validation(paths: V21Paths | None = None) -> dict:
    """Never opens test.csv; no final test evaluation or production promotion."""
    paths = paths or resolve_v21_paths()
    split_dir, result_dir, model_dir = _paths(paths)
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("experiment") != EXPERIMENT or not manifest.get("test_frozen")
            or manifest.get("test_evaluated") or not manifest.get("leakage", {}).get("passed")):
        raise RuntimeError("TREC split/test lock not valid")
    result_path = result_dir / "validation_metrics.json"
    if result_path.is_file():
        for name in ("train", "validation"):
            if sha256_file(split_dir / f"{name}.csv") != manifest["split_sha256"][name]:
                raise RuntimeError(f"Frozen {name} split changed")
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("experiment") != EXPERIMENT or existing.get("test_evaluated") is not False:
            raise RuntimeError("Existing binary result has an invalid experiment/test marker")
        for item in existing.get("candidates", []):
            artifact = model_dir / f"{item['candidate']}.joblib"
            if not artifact.is_file() or sha256_file(artifact) != item["artifact_sha256"]:
                raise RuntimeError("Existing candidate differs from saved validation result")
        if existing.get("candidates"):
            return existing
    train, validation = _load_development(split_dir, manifest)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, pipeline in _pipelines().items():
        started = time.perf_counter()
        pipeline.fit(train[["subject", "body"]], train.raw_label)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predicted = pipeline.predict(validation[["subject", "body"]])
        predict_seconds = time.perf_counter() - started
        artifact = model_dir / f"{name}.joblib"
        joblib.dump(pipeline, artifact)
        restored = joblib.load(artifact)
        if restored.predict(validation[["subject", "body"]].iloc[:1]).tolist() != predicted[:1].tolist():
            raise RuntimeError("Saved candidate reload smoke test failed")
        metrics = binary_metrics(validation.raw_label.to_numpy(), predicted)
        outputs.append({"candidate": name, "representation": name.rsplit("_", 2)[0],
                        "train_rows": len(train), "validation_rows": len(validation),
                        "fit_seconds": fit_seconds,
                        "prediction_ms_per_email": predict_seconds * 1000 / len(validation),
                        "artifact_bytes": artifact.stat().st_size,
                        "artifact_sha256": sha256_file(artifact), "metrics": metrics})
    ranked = sorted(outputs, key=lambda row: (-row["metrics"]["macro_f1"],
                                                 row["metrics"]["false_positives_ham_as_spam"],
                                                 row["prediction_ms_per_email"]))
    result = {"experiment": EXPERIMENT, "test_evaluated": False,
              "selection_basis": "Validation macro F1, then Ham false positives, then latency",
              "representation_limit_characters": MAX_TEXT_CHARACTERS,
              "recommended_candidate": ranked[0]["candidate"],
              "embedding_status": "not_run_without_verified_local_model_and_resource_budget",
              "candidates": outputs}
    (result_dir / "validation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (result_dir / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate", "accuracy", "macro_f1",
            "ham_precision", "ham_recall", "spam_precision", "spam_recall",
            "ham_false_positives", "fit_seconds", "prediction_ms_per_email", "artifact_bytes"))
        writer.writeheader()
        for item in outputs:
            m = item["metrics"]
            writer.writerow({"candidate": item["candidate"], "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"], "ham_precision": m["per_class"]["ham"]["precision"],
                "ham_recall": m["per_class"]["ham"]["recall"],
                "spam_precision": m["per_class"]["spam"]["precision"],
                "spam_recall": m["per_class"]["spam"]["recall"],
                "ham_false_positives": m["false_positives_ham_as_spam"],
                "fit_seconds": item["fit_seconds"],
                "prediction_ms_per_email": item["prediction_ms_per_email"],
                "artifact_bytes": item["artifact_bytes"]})
    return result


def train_embedding_validation(paths: V21Paths | None = None) -> dict:
    """Optional local-only multilingual embedding candidate, validation only."""
    paths = paths or resolve_v21_paths()
    split_dir, result_dir, model_dir = _paths(paths)
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("test_frozen") or manifest.get("test_evaluated"):
        raise RuntimeError("TREC test lock invalid")
    train, validation = _load_development(split_dir, manifest)
    result_path = result_dir / "validation_metrics.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    existing = next((item for item in result["candidates"]
                     if item["candidate"] == "embedding_minilm_linear_svc"), None)
    if existing and "end_to_end_prediction_ms_per_email" in existing:
        return result
    from .development_training import resource_snapshot
    resources = resource_snapshot()
    if resources.get("free_ram_gib", 0) < 2 or (
        resources.get("cuda") and resources.get("free_vram_gib", 0) < 1
    ):
        raise RuntimeError("Insufficient RAM/VRAM for local embedding experiment")
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from vietmailguard.embedding_modeling import (
        EmbeddingClassifierArtifact, load_or_encode_embeddings,
    )
    config = json.loads((paths.config / "development_training.json").read_text(encoding="utf-8"))["embedding"]
    # Resolve an existing snapshot only: no network/model download is permitted.
    local_model = snapshot_download(repo_id=config["model_id"],
                                    revision=config["revision"], local_files_only=True)
    device = "cuda" if resources.get("cuda") else "cpu"
    encoder = SentenceTransformer(local_model, local_files_only=True, device=device)
    encoder.max_seq_length = int(config["max_sequence_length"])
    if encoder.get_embedding_dimension() != config["dimension"]:
        raise RuntimeError("Local embedding dimension differs from frozen config")
    cache_root = paths.cache / "tb"
    batch_size = 16 if device == "cuda" else 8
    train_cache = load_or_encode_embeddings(
        frame=train, split_name="trec_binary_train", encoder=encoder,
        model_id=config["model_id"], revision=config["revision"],
        max_sequence_length=encoder.max_seq_length, embedding_dimension=config["dimension"],
        cache_root=cache_root, batch_size=batch_size, device=device)
    validation_cache = load_or_encode_embeddings(
        frame=validation, split_name="trec_binary_validation", encoder=encoder,
        model_id=config["model_id"], revision=config["revision"],
        max_sequence_length=encoder.max_seq_length, embedding_dimension=config["dimension"],
        cache_root=cache_root, batch_size=batch_size, device=device)
    # A classifier-only timing would conceal the encoder's deployment cost.
    sample_texts = EmailTextComposer().transform(validation[["subject", "body"]].iloc[:16])
    started = time.perf_counter()
    encoder.encode(sample_texts.tolist(), batch_size=16, show_progress_bar=False,
                   convert_to_numpy=True, normalize_embeddings=True, device=device)
    encoder_ms_per_email = (time.perf_counter() - started) * 1000 / len(sample_texts)
    classifier = LinearSVC(C=1.0, random_state=SEED)
    started = time.perf_counter()
    classifier.fit(train_cache.embeddings, train.raw_label)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predicted = classifier.predict(validation_cache.embeddings)
    predict_seconds = time.perf_counter() - started
    candidate = "embedding_minilm_linear_svc"
    artifact = EmbeddingClassifierArtifact(
        experiment_id=candidate, classifier=classifier, model_id=config["model_id"],
        model_revision=config["revision"], max_sequence_length=encoder.max_seq_length,
        embedding_dimension=config["dimension"], label_order=list(LABELS),
        training_scope="trec06c_binary_development")
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"{candidate}.joblib"
    joblib.dump(artifact, artifact_path)
    restored = joblib.load(artifact_path)
    if restored.predict_embeddings(validation_cache.embeddings[:1]).tolist() != predicted[:1].tolist():
        raise RuntimeError("Embedding candidate reload smoke test failed")
    entry = {"candidate": candidate, "representation": "multilingual_minilm_embedding",
             "train_rows": len(train), "validation_rows": len(validation),
             "fit_seconds": fit_seconds,
             "embedding_generation_seconds": {
                 "train": train_cache.original_generation_seconds,
                 "validation": validation_cache.original_generation_seconds},
             "prediction_ms_per_email": predict_seconds * 1000 / len(validation),
             "encoder_smoke_ms_per_email": encoder_ms_per_email,
             "end_to_end_prediction_ms_per_email": encoder_ms_per_email + predict_seconds * 1000 / len(validation),
             "validation_embedding_generation_ms_per_email":
                 validation_cache.original_generation_seconds * 1000 / len(validation),
             "artifact_bytes": artifact_path.stat().st_size,
             "artifact_sha256": sha256_file(artifact_path),
             "metrics": binary_metrics(validation.raw_label.to_numpy(), predicted)}
    result["candidates"] = [item for item in result["candidates"] if item["candidate"] != candidate]
    result["candidates"].append(entry)
    result["embedding_status"] = "completed_local_only"
    result["embedding_model_id"] = config["model_id"]
    result["embedding_revision"] = config["revision"]
    result["embedding_device"] = device
    result["resources_before_loading"] = resources
    ranked = sorted(result["candidates"], key=lambda row: (
        -row["metrics"]["macro_f1"], row["metrics"]["false_positives_ham_as_spam"],
        row.get("end_to_end_prediction_ms_per_email", row["prediction_ms_per_email"])))
    result["recommended_candidate"] = ranked[0]["candidate"]
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_path = result_dir / "model_comparison.csv"
    with comparison_path.open("r", encoding="utf-8", newline="") as handle:
        old_rows = [row for row in csv.DictReader(handle) if row["candidate"] != candidate]
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        m = entry["metrics"]
        fields = ("candidate", "accuracy", "macro_f1", "ham_precision", "ham_recall",
                  "spam_precision", "spam_recall", "ham_false_positives", "fit_seconds",
                  "prediction_ms_per_email", "artifact_bytes", "end_to_end_prediction_ms_per_email")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in old_rows:
            writer.writerow(row)
        writer.writerow({"candidate": candidate, "accuracy": m["accuracy"],
                         "macro_f1": m["macro_f1"],
                         "ham_precision": m["per_class"]["ham"]["precision"],
                         "ham_recall": m["per_class"]["ham"]["recall"],
                         "spam_precision": m["per_class"]["spam"]["precision"],
                         "spam_recall": m["per_class"]["spam"]["recall"],
                         "ham_false_positives": m["false_positives_ham_as_spam"],
                         "fit_seconds": entry["fit_seconds"],
                         "prediction_ms_per_email": entry["prediction_ms_per_email"],
                         "artifact_bytes": entry["artifact_bytes"],
                         "end_to_end_prediction_ms_per_email": entry["end_to_end_prediction_ms_per_email"]})
    return result
