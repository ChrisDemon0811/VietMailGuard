"""Development-only V2.1 candidate training; never reads a test split.

The split manifest and CSVs are verified before any fitting. CSV rows are read
incrementally; only bounded text for the 38,013-row development cohort is held
for scikit-learn's batch estimators. Original email text is never rewritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
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

from vietmailguard.embedding_modeling import (
    EmbeddingClassifierArtifact,
    load_or_encode_embeddings,
)
from vietmailguard.modeling import EmailTextComposer
from .paths import V21Paths, resolve_v21_paths


class DevelopmentTrainingBlocked(RuntimeError):
    """A required development-only integrity or resource condition failed."""


REQUIRED = {
    "id", "subject", "body", "source", "language", "data_origin",
    "label", "label_status", "label_provenance", "training_eligible",
    "curation_rule_id", "content_hash", "final_group_id", "parent_id",
    "normalized_content_hash", "template_hash", "split",
}
LEAKAGE_KEYS = (
    "final_group_id", "content_hash", "normalized_content_hash", "template_hash"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_and_load_development(paths: V21Paths, max_characters: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Validate hashes, labels, lineage and leakage without opening any test file."""
    manifest_path = paths.splits / "development_bootstrap" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("mode") != "development_bootstrap"
            or manifest.get("development_training_ready") is not True
            or manifest.get("final_evaluation_ready") is not False
            or manifest.get("final_test_path") is not None
            or manifest.get("v2_test_rows_exported") != 0
            or manifest.get("leakage", {}).get("passed") is not True):
        raise DevelopmentTrainingBlocked("Development manifest is not verified/isolated from final test")
    master_path = paths.processed / "Development_Master_Dataset.csv"
    if (not master_path.is_file()
            or sha256_file(master_path) != manifest.get("output_sha256", {}).get("master")
            or manifest.get("output_rows", {}).get("master") != sum(
                manifest.get("output_rows", {}).get(split, 0)
                for split in ("train", "validation"))):
        raise DevelopmentTrainingBlocked("Development Master is missing or differs from manifest")
    owners: dict[str, dict[str, tuple[str, str]]] = {key: {} for key in LEAKAGE_KEYS}
    ids: dict[str, tuple[str, str, str]] = {}
    rows: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    parents: list[tuple[str, str, str, str]] = []
    csv.field_size_limit(100_000_000)
    for split in ("train", "validation"):
        path = paths.splits / "development_bootstrap" / f"{split}.csv"
        if sha256_file(path) != manifest["output_sha256"][split]:
            raise DevelopmentTrainingBlocked(f"{split} checksum differs from frozen development manifest")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if REQUIRED - set(reader.fieldnames or ()):
                raise DevelopmentTrainingBlocked(f"{split} is missing required columns")
            for row in reader:
                identifier, label = row["id"], row["label"]
                if (not identifier or identifier in ids or row["split"] != split
                        or label not in {"normal", "spam", "phishing"}
                        or row["label_status"] not in {"eligible", "evidence_confirmed"}
                        or row["training_eligible"].casefold() != "true"
                        or not row["label_provenance"] or not row["curation_rule_id"]
                        or row["language"] not in {"en", "vi"}):
                    raise DevelopmentTrainingBlocked(f"Unverified development row: {identifier}")
                if row["language"] == "vi" and row["data_origin"] != "translated":
                    raise DevelopmentTrainingBlocked(f"Unverified Vietnamese origin: {identifier}")
                if not row["body"].strip():
                    raise DevelopmentTrainingBlocked(f"Empty email body: {identifier}")
                for key in LEAKAGE_KEYS:
                    value = row[key]
                    if not value:
                        continue
                    previous = owners[key].setdefault(value, (split, label))
                    if previous != (split, label):
                        raise DevelopmentTrainingBlocked(f"Cross-split or label-conflict {key}: {identifier}")
                ids[identifier] = (split, label, row["final_group_id"])
                if row["parent_id"]:
                    parents.append((identifier, row["parent_id"], split, label))
                rows[split].append({
                    "id": identifier, "subject": row["subject"][:max_characters],
                    "body": row["body"][:max_characters], "label": label,
                    "language": row["language"], "data_origin": row["data_origin"] or "not_declared",
                    "source": row["source"], "content_hash": row["content_hash"],
                })
        if len(rows[split]) != manifest["output_rows"][split]:
            raise DevelopmentTrainingBlocked(f"{split} row count differs from manifest")
        if set(item["label"] for item in rows[split]) != {"normal", "spam", "phishing"}:
            raise DevelopmentTrainingBlocked(f"{split} does not contain all three classes")
    for identifier, parent_id, split, label in parents:
        if ids.get(parent_id) != (split, label, ids[identifier][2]):
            raise DevelopmentTrainingBlocked(f"Parent/translation leakage: {identifier}")
    return manifest, pd.DataFrame(rows["train"]), pd.DataFrame(rows["validation"])


def resource_snapshot() -> dict[str, Any]:
    """Measure available memory/accelerator before loading neural models."""
    import ctypes
    import shutil
    import sys

    result: dict[str, Any] = {"python": sys.version.split()[0], "cuda": False}
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong),
                        ("available_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong),
                        ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong),
                        ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result["free_ram_gib"] = round(status.available_physical / 2**30, 3)
            result["total_ram_gib"] = round(status.total_physical / 2**30, 3)
    result["free_disk_gib"] = round(shutil.disk_usage(Path.cwd()).free / 2**30, 3)
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda"] = bool(torch.cuda.is_available())
        if result["cuda"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["total_vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3)
            result["free_vram_gib"] = round(torch.cuda.mem_get_info(0)[0] / 2**30, 3)
    except ImportError:
        result["torch_version"] = None
    return result


def xlm_resource_decision(resources: dict[str, Any], config: dict) -> tuple[bool, str]:
    """Conservative gate; no model download or training occurs here."""
    if resources.get("free_ram_gib", 0) < config["min_free_ram_gib"]:
        return False, "insufficient free RAM"
    if not resources.get("cuda") or resources.get("total_vram_gib", 0) < config["min_total_vram_gib"]:
        return False, "insufficient GPU VRAM for XLM-RoBERTa fine-tuning"
    try:
        import accelerate  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False, "transformers/accelerate fine-tuning dependencies unavailable"
    if config.get("require_local_model"):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=config["model_id"], local_files_only=True)
        except (OSError, ValueError) as error:
            return False, f"XLM-R model not present in local cache: {error}"
    return True, "resources and dependencies passed"


def validation_metrics(truth: np.ndarray, prediction: np.ndarray, frame: pd.DataFrame,
                       elapsed_seconds: float) -> dict[str, Any]:
    labels = ["normal", "spam", "phishing"]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0)
    overall = {
        "rows": int(len(truth)), "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(np.mean(f1)), "phishing_recall": float(recall[2]),
        "normal_false_positives": int(np.sum((truth != "normal") & (prediction == "normal"))),
        "non_normal_false_positives": int(np.sum((truth == "normal") & (prediction != "normal"))),
        "prediction_ms_per_email": float(elapsed_seconds * 1000 / len(truth)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(labels)},
    }
    subsets = {}
    for language in ("en", "vi", "zh"):
        mask = frame["language"].eq(language).to_numpy()
        n = int(mask.sum())
        if not n:
            subsets[language] = {"rows": 0, "macro_f1_supported_classes": None,
                                  "phishing_recall": None, "normal_false_positives": None}
            continue
        actual = truth[mask]
        predicted = prediction[mask]
        supported = [label for label in labels if np.any(actual == label)]
        _, sub_recall, sub_f1, _ = precision_recall_fscore_support(
            actual, predicted, labels=supported, zero_division=0)
        subsets[language] = {
            "rows": n, "supported_classes": supported,
            "macro_f1_supported_classes": float(np.mean(sub_f1)),
            "phishing_recall": float(sub_recall[supported.index("phishing")])
            if "phishing" in supported else None,
            "normal_false_positives": int(np.sum((actual != "normal") & (predicted == "normal"))),
            "non_normal_false_positives": int(np.sum((actual == "normal") & (predicted != "normal"))),
            "support": dict(Counter(actual)),
        }
    overall["by_language"] = subsets
    overall["by_data_origin"] = {
        origin: {"rows": int(mask.sum()), "macro_f1_supported_classes": float(np.mean(
            precision_recall_fscore_support(truth[mask], prediction[mask],
                labels=[label for label in labels if np.any(truth[mask] == label)],
                zero_division=0)[2]))}
        for origin in sorted(frame["data_origin"].unique())
        for mask in [frame["data_origin"].eq(origin).to_numpy()]
    }
    return overall


def make_tfidf_pipeline(config: dict) -> Pipeline:
    params = config["tfidf"]
    word = TfidfVectorizer(analyzer="word", dtype=np.float32, **{
        key: tuple(value) if key == "ngram_range" else value
        for key, value in params["word"].items() if key != "dtype"})
    character = TfidfVectorizer(dtype=np.float32, **{
        key: tuple(value) if key == "ngram_range" else value
        for key, value in params["character"].items() if key != "dtype"})
    return Pipeline([
        ("compose_text", EmailTextComposer()),
        ("tfidf", FeatureUnion([("word", word), ("character", character)], n_jobs=1)),
        ("classifier", LinearSVC(random_state=config["seed"], **params["svm"])),
    ])


def rank_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Development recommendation, not a frozen production selection."""
    return sorted(rows, key=lambda row: (
        -row["metrics"]["macro_f1"],
        -row["metrics"]["phishing_recall"],
        row["metrics"]["non_normal_false_positives"],
        row["metrics"]["prediction_ms_per_email"],
    ))


def _train_xlmr(train: pd.DataFrame, validation: pd.DataFrame, config: dict,
                candidates: Path, resources: dict[str, Any]) -> dict[str, Any]:
    """Optional local-only fine-tune on the same development Train/Validation.

    This branch is never entered unless the up-front RAM/VRAM/dependency/local
    snapshot gate passes. It does not load a test split or release a model.
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)
    from huggingface_hub import snapshot_download

    model_config = config["xlm_roberta"]
    snapshot = snapshot_download(model_config["model_id"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    label_order = config["classes"]
    label_to_id = {label: i for i, label in enumerate(label_order)}

    class EmailDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            texts = EmailTextComposer().transform(frame[["subject", "body"]]).tolist()
            self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                       max_length=int(model_config["max_length"]))
            self.labels = [label_to_id[label] for label in frame["label"]]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    train_data = EmailDataset(train)
    validation_data = EmailDataset(validation)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, num_labels=3,
        id2label={i: label for i, label in enumerate(label_order)},
        label2id=label_to_id, ignore_mismatched_sizes=True)
    target = candidates / "xlm_roberta_finetuned"
    arguments = TrainingArguments(
        output_dir=str(target / "trainer"), num_train_epochs=1,
        per_device_train_batch_size=4, per_device_eval_batch_size=8,
        gradient_accumulation_steps=4, learning_rate=2e-5,
        save_strategy="no", eval_strategy="no", report_to=[],
        dataloader_num_workers=0, seed=config["seed"], data_seed=config["seed"],
        fp16=bool(resources["cuda"]), disable_tqdm=True)
    trainer = Trainer(model=model, args=arguments, train_dataset=train_data)
    started = time.perf_counter()
    trainer.train()
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    logits = trainer.predict(validation_data).predictions
    prediction_seconds = time.perf_counter() - started
    predicted = np.asarray([label_order[i] for i in np.argmax(logits, axis=1)])
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    reloaded = AutoModelForSequenceClassification.from_pretrained(target, local_files_only=True)
    if list(reloaded.config.id2label.values()) != label_order:
        raise DevelopmentTrainingBlocked("XLM-R candidate reload class order mismatch")
    metrics = validation_metrics(validation["label"].to_numpy(), predicted,
                                 validation, prediction_seconds)
    return {"candidate": "xlm_roberta_finetuned", "family": "xlm_roberta",
            "fit_seconds": fit_seconds, "artifact": str(target.relative_to(candidates.parents[2])),
            "artifact_bytes": sum(path.stat().st_size for path in target.rglob("*") if path.is_file()),
            "metrics": metrics, "confidence_available": False,
            "pretrained_snapshot": str(snapshot)}


def _write_report(output: dict[str, Any], path: Path) -> None:
    resource = output["resources_before_neural_load"]
    lines = ["# V2.1 Development validation experiments", "",
             "These are **development-only** results from inherited V2 Train/Validation.",
             "V2 Validation already participated in V2 selection. No V2.1 final test was",
             "created or evaluated; no V2 artifact or production model was changed.", "",
             f"Train: **{output['train_rows']:,}**; Validation: **{output['validation_rows']:,}**.",
             "All candidates use the same development split and the same subject/body",
             "composer. Text is capped at 4,096 characters per field as a derived",
             "resource-bound representation; original CSV text is unchanged.", "",
             "## Pre-training resource gate", "",
             f"- Free RAM before neural load: {resource.get('free_ram_gib', 'N/A')} GiB; "
             f"total RAM: {resource.get('total_ram_gib', 'N/A')} GiB.",
             f"- GPU: {resource.get('gpu_name', 'none')}; VRAM: "
             f"{resource.get('total_vram_gib', 'N/A')} GiB total, "
             f"{resource.get('free_vram_gib', 'N/A')} GiB free.",
             f"- XLM-R: **{output['xlm_roberta']['status']}** — "
             f"{output['xlm_roberta']['reason']}.", "",
             "## Validation comparison", "",
             "| Candidate | Macro F1 | Phishing recall | EN phishing recall | VI supported-class Macro F1 | Normal emails falsely flagged | Prediction ms/email* |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rank_candidates(output["candidates"]):
        m = row["metrics"]
        vi = m["by_language"]["vi"]["macro_f1_supported_classes"]
        lines.append(f"| {row['candidate']} | {m['macro_f1']:.6f} | "
                     f"{m['phishing_recall']:.6f} | "
                     f"{m['by_language']['en']['phishing_recall']:.6f} | "
                     f"{vi:.6f} | {m['non_normal_false_positives']} | "
                     f"{m['prediction_ms_per_email']:.3f} |")
    lines += ["", "*For embedding classifiers this is classifier-only prediction time;",
              "the JSON also records a shared warm-batch encoder latency separately.",
              "'Normal emails falsely flagged' means true normal emails predicted",
              "spam/phishing, not malicious emails predicted normal.", "",
              f"Development recommendation: **{output['development_recommendation']}**",
              "by Macro F1, then phishing recall, normal false positives, and latency.",
              "This is not a final production selection.", "",
              "## Coverage and limitations", "",
              "- English Validation contains all three classes.",
              "- Vietnamese Validation is translated and has no phishing examples;",
              "  Vietnamese phishing recall and native Vietnamese generalization are N/A.",
              "- Chinese Development/Validation has zero eligible rows, so Chinese",
              "  classification performance is not measured.",
              "- No independent V2.1 Final Test exists. Final evaluation remains blocked.",
              "- LinearSVC decision scores are not probabilities; uncalibrated candidate",
              "  confidence is unavailable. Calibration is outside this stage.",
              "- Embedding classifier joblibs reference a local pretrained encoder",
              "  snapshot; they are not standalone production artifacts.",
              "- Challenge/held-out metrics are not mixed into this report."]
    coverage = {(entry["language"], entry["label"]): entry["rows"]
                for entry in output["validation_language_label_counts"]}
    lines += ["", "| Validation language | Normal | Spam | Phishing | Total |",
              "|---|---:|---:|---:|---:|"]
    for language in ("en", "vi", "zh"):
        counts = [coverage.get((language, label), 0)
                  for label in ("normal", "spam", "phishing")]
        lines.append(f"| {language} | {counts[0]:,} | {counts[1]:,} | "
                     f"{counts[2]:,} | {sum(counts):,} |")
    if output["embedding_skip_reason"]:
        lines += [f"- Embedding experiment skipped: {output['embedding_skip_reason']}."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_development_candidates(paths: V21Paths | None = None) -> dict[str, Any]:
    paths = paths or resolve_v21_paths()
    config_path = paths.config / "development_training.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resources = resource_snapshot()
    manifest, train, validation = verify_and_load_development(
        paths, int(config["max_input_characters"]))
    paths.model.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    candidates = paths.model / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    truth = validation["label"].to_numpy()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    pipeline = make_tfidf_pipeline(config)
    pipeline.fit(train[["subject", "body"]], train["label"])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predicted = pipeline.predict(validation[["subject", "body"]])
    prediction_seconds = time.perf_counter() - started
    artifact_path = candidates / "tfidf_word_character_linear_svm.joblib"
    joblib.dump(pipeline, artifact_path)
    reloaded = joblib.load(artifact_path)
    if not np.array_equal(reloaded.predict(validation[["subject", "body"]].head(3)), predicted[:3]):
        raise DevelopmentTrainingBlocked("TF-IDF candidate reload prediction mismatch")
    results.append({"candidate": "tfidf_word_character_linear_svm", "family": "tfidf",
                    "fit_seconds": fit_seconds,
                    "artifact": artifact_path.relative_to(paths.project_root).as_posix(),
                    "artifact_bytes": artifact_path.stat().st_size,
                    "metrics": validation_metrics(truth, predicted, validation, prediction_seconds),
                    "confidence_available": False})

    embedding_config = config["embedding"]
    embedding_reason = None
    if resources.get("free_ram_gib", 0) < embedding_config["min_free_ram_gib"]:
        embedding_reason = "insufficient free RAM"
    elif resources.get("cuda") and resources.get("free_vram_gib", 0) < embedding_config["min_free_vram_gib"]:
        embedding_reason = "insufficient free GPU memory"
    if embedding_reason is None:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(
                embedding_config["model_id"], revision=embedding_config["revision"],
                local_files_only=True, device="cuda" if resources.get("cuda") else "cpu")
            encoder.max_seq_length = int(embedding_config["max_sequence_length"])
            dimension = int(encoder.get_embedding_dimension())
            if dimension != embedding_config["dimension"]:
                raise DevelopmentTrainingBlocked("Local encoder dimension does not match configuration")
            device = "cuda" if resources.get("cuda") else "cpu"
            batch_size = embedding_config[f"batch_size_{device}"]
            # Keep the generated temporary filename below the Windows MAX_PATH
            # limit; the helper still keys it by model revision and text hashes.
            cache = paths.cache / "e"
            train_cache = load_or_encode_embeddings(
                frame=train, split_name="train", encoder=encoder,
                model_id=embedding_config["model_id"], revision=embedding_config["revision"],
                max_sequence_length=encoder.max_seq_length, embedding_dimension=dimension,
                cache_root=cache, batch_size=batch_size, device=device)
            validation_cache = load_or_encode_embeddings(
                frame=validation, split_name="validation", encoder=encoder,
                model_id=embedding_config["model_id"], revision=embedding_config["revision"],
                max_sequence_length=encoder.max_seq_length, embedding_dimension=dimension,
                cache_root=cache, batch_size=batch_size, device=device)
            embedding_cache = {"train_hit": train_cache.cache_hit,
                               "validation_hit": validation_cache.cache_hit,
                               "train_generation_seconds": train_cache.original_generation_seconds,
                               "validation_generation_seconds": validation_cache.original_generation_seconds}
            smoke_texts = EmailTextComposer().transform(
                validation[["subject", "body"]].head(8)).tolist()
            encoder.encode(smoke_texts, batch_size=8, normalize_embeddings=True,
                           device=device, show_progress_bar=False)
            encode_times = []
            smoke_embeddings = None
            for _ in range(3):
                started = time.perf_counter()
                smoke_embeddings = encoder.encode(
                    smoke_texts, batch_size=8, normalize_embeddings=True,
                    device=device, show_progress_bar=False)
                encode_times.append((time.perf_counter() - started) * 1000 / 8)
            encoder_ms = float(np.median(encode_times))
            for name, classifier in (
                ("logistic_regression", LogisticRegression(C=1.0, class_weight="balanced",
                    max_iter=2000, random_state=config["seed"])),
                ("linear_svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                    random_state=config["seed"])),
            ):
                started = time.perf_counter()
                classifier.fit(train_cache.embeddings, train["label"])
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                predicted = classifier.predict(validation_cache.embeddings)
                classifier_seconds = time.perf_counter() - started
                name_full = f"embedding_minilm_{name}"
                artifact = EmbeddingClassifierArtifact(
                    experiment_id=name_full, classifier=classifier,
                    model_id=embedding_config["model_id"],
                    model_revision=embedding_config["revision"],
                    max_sequence_length=encoder.max_seq_length,
                    embedding_dimension=dimension,
                    label_order=config["classes"], training_scope="v21_development_bootstrap")
                artifact_path = candidates / f"{name_full}.joblib"
                joblib.dump(artifact, artifact_path)
                reloaded = joblib.load(artifact_path)
                if not np.array_equal(reloaded.predict_embeddings(smoke_embeddings),
                                      classifier.predict(smoke_embeddings)):
                    raise DevelopmentTrainingBlocked(f"{name_full} reload prediction mismatch")
                metrics = validation_metrics(truth, predicted, validation, classifier_seconds)
                metrics["encoder_smoke_ms_per_email"] = encoder_ms
                metrics["end_to_end_smoke_ms_per_email"] = (
                    encoder_ms + metrics["prediction_ms_per_email"])
                results.append({"candidate": name_full, "family": "embedding",
                                "fit_seconds": fit_seconds,
                                "artifact": artifact_path.relative_to(paths.project_root).as_posix(),
                                "artifact_bytes": artifact_path.stat().st_size,
                                "metrics": metrics,
                                "confidence_available": name == "logistic_regression"})
        except (ImportError, OSError) as error:
            embedding_reason = f"local encoder/dependency unavailable: {error}"
            embedding_cache = None
    else:
        embedding_cache = None
    xlm_allowed, xlm_reason = xlm_resource_decision(resources, config["xlm_roberta"])
    if xlm_allowed:
        results.append(_train_xlmr(train, validation, config, candidates, resources))
        xlm_status = "trained_on_development"
    else:
        xlm_status = "skipped_resource_gate"
    ranked = rank_candidates(results)
    output = {
        "scope": "development_validation_only", "seed": config["seed"],
        "development_manifest_sha256": sha256_file(paths.splits / "development_bootstrap" / "manifest.json"),
        "config_sha256": sha256_file(config_path),
        "development_leakage_preflight": manifest["leakage"],
        "train_rows": len(train), "validation_rows": len(validation),
        "train_label_counts": dict(Counter(train["label"])),
        "validation_label_counts": dict(Counter(validation["label"])),
        "validation_language_label_counts": [
            {"language": str(language), "label": str(label), "rows": int(count)}
            for (language, label), count in
            validation.groupby(["language", "label"], observed=True).size().items()],
        "resources_before_neural_load": resources,
        "embedding_cache": embedding_cache,
        "embedding_skip_reason": embedding_reason,
        "xlm_roberta": {"status": xlm_status, "reason": xlm_reason},
        "candidates": results,
        "development_recommendation": ranked[0]["candidate"] if ranked else None,
        "final_production_selected": False,
        "final_test_evaluated": False,
        "native_vietnamese_benchmark": False,
        "chinese_validation_rows": 0,
        "v2_validation_previously_used_for_v2_selection": True,
    }
    output_path = paths.results / "development_validation_metrics.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{
        "candidate": row["candidate"], "family": row["family"],
        "macro_f1": row["metrics"]["macro_f1"],
        "phishing_recall": row["metrics"]["phishing_recall"],
        "english_macro_f1": row["metrics"]["by_language"]["en"]["macro_f1_supported_classes"],
        "english_phishing_recall": row["metrics"]["by_language"]["en"]["phishing_recall"],
        "translated_vietnamese_macro_f1_supported_classes": row["metrics"]["by_language"]["vi"]["macro_f1_supported_classes"],
        "translated_vietnamese_phishing_recall": row["metrics"]["by_language"]["vi"]["phishing_recall"],
        "normal_false_positives": row["metrics"]["non_normal_false_positives"],
        "prediction_ms_per_email_excluding_encoder": row["metrics"]["prediction_ms_per_email"],
        "encoder_smoke_ms_per_email": row["metrics"].get("encoder_smoke_ms_per_email"),
        "fit_seconds": row["fit_seconds"], "artifact": row["artifact"],
    } for row in ranked]).to_csv(paths.results / "development_model_comparison.csv", index=False)
    for row in results:
        matrix_path = paths.results / f"{row['candidate']}_confusion_matrix.csv"
        pd.DataFrame(row["metrics"]["confusion_matrix"], index=config["classes"],
                     columns=config["classes"]).to_csv(matrix_path)
    _write_report(output, paths.reports / "development_training_validation_report.md")
    return output
