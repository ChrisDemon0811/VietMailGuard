"""Frozen MiniLM plus neural head research; development data only, offline."""

from __future__ import annotations

import builtins
import gc
import importlib.metadata
import inspect
import io
import json
import os
import socket
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from vietmailguard.embedding_modeling import (
    EmbeddingClassifierArtifact, compose_email_texts, dataset_embedding_key,
    embedding_row_key, model_cache_identifier,
)
from .development_training import resource_snapshot, sha256_file, verify_and_load_development
from .paths import V21Paths, resolve_v21_paths


EXPERIMENT = "multilingual_dl_01"
CLASSES = ["normal", "spam", "phishing"]


class ExperimentBlocked(RuntimeError):
    """A required integrity, capability or resource gate did not pass."""


def load_config(paths: V21Paths) -> dict[str, Any]:
    config = json.loads((paths.config / "experiments" / f"{EXPERIMENT}.json").read_text(encoding="utf-8"))
    if (config.get("experiment") != EXPERIMENT or config.get("scope") != "development_validation_only"
            or config.get("classes") != CLASSES or config.get("download_allowed") is not False
            or config.get("final_test_allowed") is not False
            or config["encoder"].get("frozen") is not True
            or config["encoder"].get("local_files_only") is not True
            or config["mlp"].get("early_stopping") is not False):
        raise ExperimentBlocked("Experiment configuration violates the development-only contract")
    return config


@contextmanager
def offline_boundary(root: Path):
    """Deny all sockets and all held-out/evaluation files during the experiment."""
    original_open = io.open
    environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}

    def no_network(*args, **kwargs):
        raise ExperimentBlocked("Network access is disabled for this experiment")

    def checked_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(file)).resolve()
            if path.name.casefold() in {"test.csv", "final_test.csv"} or any(
                path.is_relative_to(root / prefix) for prefix in ("data/evaluation", "data/v21/evaluation")
            ):
                raise ExperimentBlocked("Held-out/evaluation records are prohibited")
        return original_open(file, *args, **kwargs)

    with patch.dict(os.environ, environment), patch.object(socket.socket, "connect", no_network), \
            patch.object(socket.socket, "connect_ex", no_network), \
            patch.object(socket, "create_connection", no_network), \
            patch.object(socket, "getaddrinfo", no_network), \
            patch.object(socket.socket, "sendto", no_network), \
            patch.object(io, "open", checked_open), patch.object(builtins, "open", checked_open):
        yield


def measured_resources() -> dict[str, Any]:
    resource_snapshot()
    resources = resource_snapshot()
    try:
        import psutil
        resources["process_rss_gib"] = round(psutil.Process().memory_info().rss / 2**30, 4)
    except ImportError:
        resources["process_rss_gib"] = None
    return resources


def encoder_gate(resources: dict, config: dict) -> tuple[bool, str]:
    if resources.get("free_ram_gib", 0) < config["min_free_ram_gib"]:
        return False, "insufficient free RAM for the pinned encoder"
    if resources.get("cuda") and resources.get("free_vram_gib", 0) < config["min_free_vram_gib"]:
        return False, "insufficient free VRAM for the pinned encoder; no device fallback"
    return True, "resource gate passed"


def local_encoder_snapshot(config: dict) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(config["model_id"], revision=config["revision"], local_files_only=True))


def load_encoder(snapshot: Path, config: dict, device: str):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(snapshot), device=device, local_files_only=True, trust_remote_code=False)
    encoder.max_seq_length = config["max_sequence_length"]
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    get_dimension = getattr(encoder, "get_embedding_dimension", None)
    if get_dimension is None:
        get_dimension = encoder.get_sentence_embedding_dimension
    if get_dimension() != config["dimension"]:
        raise ExperimentBlocked("Pinned encoder dimension does not match configuration")
    return encoder


def preflight(paths: V21Paths, config: dict) -> dict:
    resources = measured_resources()
    allowed, reason = encoder_gate(resources, config["encoder"])
    snapshot = None
    try:
        snapshot = local_encoder_snapshot(config["encoder"])
    except (ImportError, OSError, ValueError):
        allowed, reason = False, "pinned encoder/dependency is not available locally"
    versions = {}
    for package in ("scikit-learn", "numpy", "scipy", "joblib", "torch", "sentence-transformers",
                    "transformers", "huggingface-hub", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"resources": resources, "packages": versions, "encoder_allowed": allowed,
            "encoder_reason": reason, "encoder_cached": snapshot is not None,
            "mlp_sample_weight_supported": "sample_weight" in inspect.signature(MLPClassifier.fit).parameters,
            "snapshot": snapshot}


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _embedding_contract(frame: pd.DataFrame, config: dict) -> tuple[np.ndarray, str, str]:
    texts = compose_email_texts(frame)
    identifier = model_cache_identifier(config["model_id"], config["revision"], config["max_sequence_length"])
    keys = [embedding_row_key(model_identifier=identifier, content_hash=str(content_hash), text=str(text))
            for content_hash, text in zip(frame["content_hash"], texts, strict=True)]
    return texts, identifier, dataset_embedding_key(keys)


def _validate_matrix(matrix: np.ndarray, rows: int, dimension: int) -> None:
    if matrix.shape != (rows, dimension) or matrix.dtype != np.float32:
        raise ExperimentBlocked("Embedding cache shape/dtype differs from the ordered data contract")
    for start in range(0, rows, 2048):
        block = np.asarray(matrix[start:start + 2048])
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=2e-3):
            raise ExperimentBlocked("Embedding cache is not finite L2-normalized float32")


def embedding_matrix(frame: pd.DataFrame, split: str, paths: V21Paths, config: dict,
                     encoder: Any, device: str) -> tuple[np.ndarray, dict]:
    """Read verified existing caches or generate separate resumable batches."""
    texts, identifier, key = _embedding_contract(frame, config)
    stem = f"{split}__{key}"
    own_directory = paths.cache / "experiments" / EXPERIMENT
    matches = list((paths.cache / "e").rglob(stem + ".json")) if (paths.cache / "e").exists() else []
    own_manifest = own_directory / (stem + ".json")
    if own_manifest.exists():
        matches.append(own_manifest)
    expected = {"model_identifier": identifier, "dataset_key": key, "rows": len(frame),
                "embedding_dimension": config["dimension"], "dtype": "float32"}
    started = time.perf_counter()
    if matches:
        metadata_path = sorted(matches)[0]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise ExperimentBlocked("Existing embedding metadata does not match ordered development rows")
        matrix_path = metadata_path.with_suffix(".npy")
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        _validate_matrix(matrix, len(frame), config["dimension"])
        cache_hit = True
    else:
        if encoder is None:
            raise ExperimentBlocked("No matching verified cache and encoder resource gate is blocked")
        own_directory.mkdir(parents=True, exist_ok=True)
        matrix_path = own_directory / (stem + ".npy")
        metadata_path = own_manifest
        partial = matrix_path.with_suffix(".partial.npy")
        checkpoint = matrix_path.with_suffix(".progress.json")
        next_row = 0
        if partial.exists() or checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if any(state.get(name) != value for name, value in expected.items()):
                raise ExperimentBlocked("Interrupted embedding checkpoint belongs to different inputs")
            next_row = state["next_row"]
            if not isinstance(next_row, int) or not 0 <= next_row <= len(frame):
                raise ExperimentBlocked("Invalid embedding checkpoint progress")
            matrix = np.load(partial, mmap_mode="r+", allow_pickle=False)
            if matrix.shape != (len(frame), config["dimension"]) or matrix.dtype != np.float32:
                raise ExperimentBlocked("Interrupted embedding matrix shape/dtype mismatch")
        else:
            matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                             shape=(len(frame), config["dimension"]))
        batch = config[f"batch_size_{device}"]
        for start in range(next_row, len(frame), batch):
            stop = min(start + batch, len(frame))
            block = np.asarray(encoder.encode(texts[start:stop].tolist(), batch_size=batch,
                normalize_embeddings=True, convert_to_numpy=True, device=device,
                show_progress_bar=False), dtype=np.float32)
            _validate_matrix(block, stop - start, config["dimension"])
            matrix[start:stop] = block
            matrix.flush()
            _json(checkpoint, {**expected, "next_row": stop})
        del matrix
        partial.replace(matrix_path)
        _json(metadata_path, {**expected, "normalized_embeddings": True,
                              "generation_seconds": time.perf_counter() - started})
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        cache_hit = False
    if encoder is not None:
        sample = np.unique(np.linspace(0, len(frame) - 1, min(4, len(frame)), dtype=int))
        regenerated = np.asarray(encoder.encode(texts[sample].tolist(), normalize_embeddings=True,
            convert_to_numpy=True, device=device, show_progress_bar=False), dtype=np.float32)
        if not np.allclose(matrix[sample], regenerated, atol=2e-4, rtol=2e-3):
            raise ExperimentBlocked("Cached vectors disagree with the pinned local encoder on sampled rows")
    return matrix, {**expected, "cache_hit": cache_hit,
                    "matrix": matrix_path.relative_to(paths.project_root).as_posix(),
                    "sha256": sha256_file(matrix_path), "manifest_sha256": sha256_file(metadata_path),
                    "load_check_seconds": time.perf_counter() - started,
                    "fresh_encoder_sample_verified": encoder is not None,
                    "original_generation_seconds": metadata.get("generation_seconds") if cache_hit else None}


def label_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    supports = {label: int(np.sum(truth == label)) for label in CLASSES}
    if not len(truth):
        return {"rows": 0, "support": supports, "accuracy": None, "macro_f1": None,
                "phishing_recall": None, "per_class": {label: {"support": 0, "precision": None,
                    "recall": None, "f1": None} for label in CLASSES}}
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, labels=CLASSES, zero_division=0)
    represented = np.array([supports[label] > 0 for label in CLASSES])
    return {"rows": len(truth), "support": supports, "accuracy": float(accuracy_score(truth, predicted)),
            "macro_precision": float(precision.mean()) if represented.all() else None,
            "macro_recall": float(recall.mean()) if represented.all() else None,
            "macro_f1": float(f1.mean()) if represented.all() else None,
            "macro_f1_represented_classes": float(f1[represented].mean()),
            "phishing_recall": float(recall[2]) if supports["phishing"] else None,
            "normal_to_spam_or_phishing": int(np.sum((truth == "normal") & (predicted != "normal"))),
            "spam_or_phishing_to_normal": int(np.sum((truth != "normal") & (predicted == "normal"))),
            "confusion_matrix_order": CLASSES,
            "confusion_matrix": confusion_matrix(truth, predicted, labels=CLASSES).tolist(),
            "per_class": {label: {"support": supports[label],
                "precision": float(precision[index]) if supports[label] else None,
                "recall": float(recall[index]) if supports[label] else None,
                "f1": float(f1[index]) if supports[label] else None}
                for index, label in enumerate(CLASSES)}}


def evaluate(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    truth = frame["label"].to_numpy()
    result = label_metrics(truth, predicted)
    result["by_language"] = {language: label_metrics(truth[mask], predicted[mask])
        for language in ("en", "vi", "zh") for mask in [frame["language"].eq(language).to_numpy()]}
    result["by_data_origin"] = {origin: label_metrics(truth[mask], predicted[mask])
        for origin in sorted(frame["data_origin"].unique())
        for mask in [frame["data_origin"].eq(origin).to_numpy()]}
    return result


def make_classifier(name: str, config: dict):
    if name == "B_minilm_svc":
        return LinearSVC(random_state=config["seed"], **config["svc"])
    if name == "C_minilm_mlp":
        params = {**config["mlp"], "hidden_layer_sizes": tuple(config["mlp"]["hidden_layer_sizes"])}
        return Pipeline([("scale_train_only", StandardScaler()),
                         ("classifier", MLPClassifier(random_state=config["seed"], **params))])
    raise ValueError("Unknown experiment candidate")


def _smoke_indices(frame: pd.DataFrame, config: dict) -> np.ndarray:
    generator = np.random.RandomState(config["seed"])
    indices = []
    for language in ("en", "vi", "zh"):
        choices = np.flatnonzero(frame["language"].eq(language).to_numpy())
        if len(choices):
            indices.extend(generator.choice(choices, min(len(choices), config["benchmark_rows_per_language"]), replace=False))
    return np.array(sorted(indices), dtype=int)


def benchmark(predict, sample: pd.DataFrame, repeats: int, device: str) -> dict:
    def synchronize():
        if device == "cuda":
            import torch
            torch.cuda.synchronize()

    predict(sample)
    times = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        predict(sample)
        synchronize()
        times.append((time.perf_counter() - started) * 1000 / len(sample))
    return {"sample_rows": len(sample), "repeats": repeats, "batch_ms_per_email": times,
            "median_ms_per_email": float(np.median(times)), "includes_encoder": device != "tfidf"}


def _save_classifier(name: str, classifier: Any, model_dir: Path, config: dict) -> tuple[Path, Any]:
    encoder = config["encoder"]
    artifact = EmbeddingClassifierArtifact(name, classifier, encoder["model_id"], encoder["revision"],
        encoder["max_sequence_length"], encoder["dimension"],
        [str(value) for value in classifier.classes_], "dl_01_development_bootstrap_only")
    path = model_dir / f"{name}.joblib"
    if path.exists():
        raise ExperimentBlocked("Refusing to overwrite an existing experiment candidate")
    temporary = path.with_suffix(".partial.joblib")
    joblib.dump(artifact, temporary)
    temporary.replace(path)
    return path, joblib.load(path)


def _write_report(output: dict, path: Path) -> None:
    def number(value):
        return "N/A" if value is None else f"{value:.6f}"

    lines = ["# Multilingual Deep Learning Experiment 01", "", "Ngày: 2026-10-29. Chỉ Development Validation.",
             "", "## Dataset và integrity", "",
             f"Train: {output['train_rows']:,}; Validation: {output['validation_rows']:,}. Giữ nguyên assignment bootstrap V2.",
             "Validation kế thừa từng dùng cho lựa chọn model V2; đây không phải Final Test độc lập.",
             "Không dùng TREC06C, Chinese synthetic hoặc Gemini làm ground truth. Không mở held-out Test.",
             "Exact/content, normalized, template, final group và parent/translation kiểm tra lại trước khi fit.",
             "", "| Split | Ngôn ngữ | Normal | Spam | Phishing |", "|---|---|---:|---:|---:|"]
    for split, counts in output["distribution"].items():
        for language in ("en", "vi", "zh"):
            labels = counts.get(language, {})
            lines.append(f"| {split} | {language} | " + " | ".join(str(labels.get(label, 0)) for label in CLASSES) + " |")
    lines += ["", "## Architecture và protocol", "",
              "A: load TF-IDF Word+Character + LinearSVC hiện tại read-only, không fit lại.",
              "B: frozen multilingual MiniLM, L2-normalized 384D embeddings → balanced LinearSVC.",
              "C: cùng embeddings → StandardScaler fit Train → MLP 128-ReLU → 64-ReLU → 3 softmax.",
              "MLP dùng balanced sample weights từ Train; không dùng Validation cho scaler, encoder, optimizer hoặc early stopping.",
              "Early stopping nội bộ random split bị tắt; tối đa 120 epoch, dừng theo training loss nếu hội tụ.",
              "Đầu MLP huấn luyện CPU; GPU chỉ phục vụ encoder. Softmax chưa calibrated, không là model confidence.",
              "Cấu hình và seed được lưu trước khi chạy; không sweep hyperparameter theo Validation.",
              "Subject/body qua EmailTextComposer hiện tại, giới hạn 4096 ký tự mỗi field trong development loader; MiniLM tiếp tục truncate ở 128 tokens. Baseline TF-IDF có thể giữ thông tin dài hơn — đây là giới hạn so sánh cố định này, không phải bằng chứng mọi neural model đều kém hơn.",
              "", "## Tài nguyên và dependency", "", "```json",
              json.dumps(output["preflight"], ensure_ascii=False, indent=2), "```", "",
              "## Validation comparison", "",
              "| Candidate | Status | Accuracy | Macro F1 | Phishing Recall | Normal→Spam/Phishing | Fit s | Full-Val ms/email* | End-to-end smoke ms/email |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for candidate in output["candidates"]:
        metrics = candidate.get("metrics", {})
        lines.append(f"| {candidate['candidate']} | {candidate['status']} | " + " | ".join(number(value) for value in (
            metrics.get("accuracy"), metrics.get("macro_f1"), metrics.get("phishing_recall"),
            metrics.get("normal_to_spam_or_phishing"), candidate.get("fit_seconds"),
            candidate.get("full_validation_ms_per_email"), candidate.get("latency", {}).get("median_ms_per_email"))) + " |")
        if candidate.get("reason"):
            lines.append(f"\n{candidate['candidate']}: {candidate['reason']}\n")
    lines += ["", "| Candidate | RAM trống trước fit (GiB) | RAM trống sau (GiB) | Iterations | Fit warnings |",
              "|---|---:|---:|---:|---|"]
    for candidate in output["candidates"]:
        lines.append(f"| {candidate['candidate']} | {number(candidate.get('resources_before', {}).get('free_ram_gib'))} | {number(candidate.get('resources_after', {}).get('free_ram_gib'))} | {candidate.get('training_iterations', 'N/A')} | {', '.join(candidate.get('warnings', [])) or 'none recorded'} |")
    lines += ["", "RAM ở đây là snapshot khả dụng của hệ thống, không phải peak RSS. psutil không có trong môi trường; không tạo số đo peak memory giả.",
              "Do RAM hạn chế, xác minh encoder/cache, fit heads và benchmark end-to-end chạy thành các process riêng. Thời gian fit head không bao gồm thời gian sinh embeddings; cache metadata ghi riêng thời gian sinh gốc."]
    lines += ["", "*Full-Val: A gồm compose/TF-IDF/classifier; B/C chỉ classifier trên embedding đã cache.",
              "Cột end-to-end dùng cùng mẫu EN/VI, bao gồm encoder cho B/C, warm-up và median các lần lặp.",
              "Độ trễ batch không tương đương request đơn; xem JSON để biết batch size, device và từng lần đo."]
    if output.get("encoder_measurement_blocked"):
        lines += ["", "Encoder load/end-to-end latency: BLOCKED — " + output["encoder_measurement_blocked"],
                  "Classifier vẫn có thể fit trên cache MiniLM cùng revision đã tồn tại. Kiểm tra lại ordered content keys, metadata, kích thước, L2 và SHA-256; không encode lại để vượt resource gate.",
                  "Xem embedding_verification.json nếu encoder đã xác minh vector trong process trước; receipt được ràng buộc config/data/cache SHA-256. Không coi classifier-only latency là end-to-end latency."]
    for candidate in output["candidates"]:
        if "metrics" not in candidate:
            continue
        metrics = candidate["metrics"]
        if candidate.get("warnings"):
            lines += ["", f"Cảnh báo khi fit: {', '.join(candidate['warnings'])}. Không che giấu trường hợp chưa hội tụ."]
        lines += ["", f"### {candidate['candidate']}", "", "| Class | Precision | Recall | F1 | Support |",
                  "|---|---:|---:|---:|---:|"]
        for label, values in metrics["per_class"].items():
            lines.append(f"| {label} | {number(values['precision'])} | {number(values['recall'])} | {number(values['f1'])} | {values['support']} |")
        lines += ["", "Confusion matrix: hàng thật, cột dự đoán, thứ tự normal/spam/phishing.",
                  "```json", json.dumps(metrics["confusion_matrix"]), "```", "",
                  "| Ngôn ngữ | Rows | Accuracy | Macro F1 ba lớp | Macro F1 lớp có mặt | Phishing Recall |",
                  "|---|---:|---:|---:|---:|---:|"]
        for language, values in metrics["by_language"].items():
            lines.append(f"| {language} | {values['rows']} | {number(values['accuracy'])} | {number(values['macro_f1'])} | {number(values.get('macro_f1_represented_classes'))} | {number(values['phishing_recall'])} |")
        lines += ["", "| Ngôn ngữ | Class | Support | Precision | Recall | F1 |", "|---|---|---:|---:|---:|---:|"]
        for language, values in metrics["by_language"].items():
            for label, values in values["per_class"].items():
                lines.append(f"| {language} | {label} | {values['support']} | {number(values['precision'])} | {number(values['recall'])} | {number(values['f1'])} |")
    lines += ["", "## Kết luận và bước nghiên cứu tiếp theo", "", output["conclusion"], "",
              "Chỉ EN và VI dịch có dữ liệu development; VI Validation không có phishing, ZH không có hàng.",
              "Không có native Vietnamese benchmark hoặc benchmark Chinese phishing độc lập; không tuyên bố EN/VI/ZH Production hay Final Evaluation PASS.",
              "XLM-RoBERTa chưa được chạy trong thí nghiệm này. Điều kiện tài nguyên đã lưu:",
              "```json", json.dumps(output["xlmr_gate"], ensure_ascii=False, indent=2), "```",
              "Nên ưu tiên benchmark bản địa và kiểm duyệt nhãn trước một thí nghiệm Transformer lớn. Cần nhiệm vụ riêng, cache/dependency và tài nguyên đạt gate để tiếp tục XLM-R.",
              "", "## Tái lập", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe scripts/v21/multilingual_dl_experiment_01.py --preflight",
              ".\\.venv\\Scripts\\python.exe scripts/v21/multilingual_dl_experiment_01.py --run",
              ".\\.venv\\Scripts\\python.exe scripts/v21/multilingual_dl_experiment_01.py --benchmark", "```",
              "Candidate và metric lưu riêng trong models/results/v21_multilingual/experiments/multilingual_dl_01/.",
              "Không ghi đè candidate/runtime hiện tại. JSON lưu config/hash, cache provenance, resource snapshots, convergence, class ordering và per-language per-class metrics.",
              "Cache mới ghi batch + checkpoint riêng; cache cũ chỉ đọc. Candidate checkpoint cho phép tiếp tục sau gián đoạn, không fit lại candidate đã hoàn tất."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(paths: V21Paths | None = None) -> dict:
    paths = paths or resolve_v21_paths()
    config = load_config(paths)
    model_dir = paths.model / "experiments" / EXPERIMENT
    results_dir = paths.results / "experiments" / EXPERIMENT
    config_path = paths.config / "experiments" / f"{EXPERIMENT}.json"
    with offline_boundary(paths.project_root), threadpool_limits(limits=config["cpu_threads"]):
        check = preflight(paths, config)
        snapshot = check.pop("snapshot")
        print(json.dumps({"stage": "preflight", **check}), flush=True)
        manifest, train, validation = verify_and_load_development(paths, config["max_input_characters"])
        manifest_path = paths.splits / "development_bootstrap" / "manifest.json"
        fingerprint = {"config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(manifest_path),
                       "train_sha256": manifest["output_sha256"]["train"],
                       "validation_sha256": manifest["output_sha256"]["validation"]}
        model_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = results_dir / "checkpoint.json"
        state = {"fingerprint": fingerprint, "candidates": []}
        if checkpoint_path.exists():
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if state.get("fingerprint") != fingerprint:
                raise ExperimentBlocked("Existing experiment has different data/config; refusing overwrite")
        _json(results_dir / "config_snapshot.json", config)
        indices = _smoke_indices(validation, config)
        sample = validation.iloc[indices]
        runtime = json.loads((paths.config / "runtime_model.json").read_text(encoding="utf-8"))
        baseline_path = paths.project_root / runtime["artifact"]
        if sha256_file(baseline_path) != runtime["artifact_sha256"]:
            raise ExperimentBlocked("Existing V2.1 baseline checksum mismatch")
        baseline_sha = runtime["artifact_sha256"]
        state["candidates"] = [item for item in state["candidates"] if item["status"] == "PASS"]
        completed = {item["candidate"] for item in state["candidates"]}
        for item in state["candidates"]:
            artifact = item.get("artifact")
            if artifact and sha256_file(paths.project_root / artifact) != item["artifact_sha256"]:
                raise ExperimentBlocked("Completed candidate checksum mismatch")
        if "A_tfidf_svc" not in completed:
            print(json.dumps({"stage": "A_baseline_validation", "rows": len(validation)}), flush=True)
            started = time.perf_counter()
            baseline = joblib.load(baseline_path)
            prediction_started = time.perf_counter()
            predicted = np.asarray(baseline.predict(validation[["subject", "body"]]))
            prediction_seconds = time.perf_counter() - prediction_started
            elapsed = time.perf_counter() - started
            metrics = evaluate(validation, predicted)
            latency = benchmark(lambda batch: baseline.predict(batch[["subject", "body"]]), sample,
                                config["benchmark_repeats"], "tfidf")
            saved = json.loads((paths.project_root / runtime["validation_metrics"]).read_text(encoding="utf-8"))
            prior = next(item["metrics"] for item in saved["candidates"] if item["candidate"] == runtime["candidate_id"])
            if not np.isclose(metrics["macro_f1"], prior["macro_f1"], atol=1e-12, rtol=0):
                raise ExperimentBlocked("Baseline prediction differs from saved Development Validation")
            state["candidates"].append({"candidate": "A_tfidf_svc", "status": "PASS", "fit_seconds": None,
                "artifact": runtime["artifact"], "artifact_sha256": baseline_sha, "metrics": metrics,
                "load_and_validation_seconds": elapsed, "latency": latency,
                "full_validation_ms_per_email": prediction_seconds * 1000 / len(validation),
                "saved_validation_macro_f1_matches": bool(np.isclose(metrics["macro_f1"], prior["macro_f1"], atol=1e-12, rtol=0)),
                "resources_after": measured_resources()})
            del baseline
            gc.collect()
            _json(checkpoint_path, state)
        encoder = None
        embeddings = {}
        cache_info = {}
        failure = None
        verification_path = results_dir / "embedding_verification.json"
        previous_results = results_dir / "validation_metrics.json"
        # Preserve a successful encoder verification from an interrupted/resource-
        # blocked run, freeing that process before training heads. No refitting or
        # lower resource threshold is needed to resume from the verified vectors.
        if not verification_path.exists() and previous_results.exists():
            previous = json.loads(previous_results.read_text(encoding="utf-8"))
            cache = previous.get("cache", {})
            if previous.get("fingerprint") == fingerprint and all(
                cache.get(split, {}).get("fresh_encoder_sample_verified") for split in ("train", "validation")
            ):
                _json(verification_path, {"fingerprint": fingerprint, "cache": cache})
        verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None
        if verification is not None:
            if verification.get("fingerprint") != fingerprint:
                raise ExperimentBlocked("Encoder verification receipt belongs to different inputs")
            for split in ("train", "validation"):
                embeddings[split], cache_info[split] = embedding_matrix(
                    train if split == "train" else validation, split, paths, config["encoder"], None, "cpu")
                prior_cache = verification["cache"][split]
                if (not prior_cache["fresh_encoder_sample_verified"] or
                        cache_info[split]["sha256"] != prior_cache["sha256"] or
                        cache_info[split]["manifest_sha256"] != prior_cache["manifest_sha256"]):
                    raise ExperimentBlocked("Previously encoder-verified cache was modified")
                cache_info[split]["encoder_verification_receipt_sha256"] = sha256_file(verification_path)
            print(json.dumps({"stage": "reuse_encoder_verified_cache", "encoder_not_allocated": True}), flush=True)
        elif check["encoder_allowed"]:
            try:
                device = "cuda" if check["resources"].get("cuda") else "cpu"
                encoder = load_encoder(snapshot, config["encoder"], device)
                for split, frame in (("train", train), ("validation", validation)):
                    print(json.dumps({"stage": "embeddings", "split": split, "rows": len(frame)}), flush=True)
                    embeddings[split], cache_info[split] = embedding_matrix(frame, split, paths, config["encoder"], encoder, device)
                _json(verification_path, {"fingerprint": fingerprint, "cache": cache_info})
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                failure = f"encoder/cache stage blocked: {type(error).__name__}"
        else:
            failure = check["encoder_reason"]
        encoder_measurement_blocked = failure or ("Encoder verification completed in a previous process; fresh end-to-end timing unavailable in the classifier-only resume." if verification is not None else None)
        if encoder is None and not embeddings:
            # Existing frozen vectors need no encoder allocation. This is the same
            # representation, not a smaller-model/device substitution. Missing or
            # inconsistent caches still block both heads; E2E timing stays blocked.
            try:
                for split, frame in (("train", train), ("validation", validation)):
                    embeddings[split], cache_info[split] = embedding_matrix(
                        frame, split, paths, config["encoder"], None, "cpu")
                failure = None
            except (OSError, ValueError, ExperimentBlocked):
                failure = "encoder blocked and matching validated embedding cache unavailable"
        for name in ("B_minilm_svc", "C_minilm_mlp"):
            if name in completed:
                continue
            resources = measured_resources()
            reason = failure
            if resources.get("free_ram_gib", 0) < config["classifier_min_free_ram_gib"]:
                reason = "insufficient free RAM for classifier training"
            if name == "C_minilm_mlp" and not check["mlp_sample_weight_supported"]:
                reason = "installed MLPClassifier.fit lacks sample_weight support; no substitute classifier"
            if reason:
                state["candidates"].append({"candidate": name, "status": "BLOCKED", "reason": reason})
                continue
            print(json.dumps({"stage": "fit", "candidate": name}), flush=True)
            classifier = make_classifier(name, config)
            fit_options = {}
            if name == "C_minilm_mlp":
                fit_options["classifier__sample_weight"] = compute_sample_weight("balanced", train["label"].to_numpy())
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                classifier.fit(embeddings["train"], train["label"].to_numpy(), **fit_options)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predicted = np.asarray(classifier.predict(embeddings["validation"]))
            prediction_seconds = time.perf_counter() - started
            artifact_path, loaded = _save_classifier(name, classifier, model_dir, config)
            if not np.array_equal(loaded.predict_embeddings(embeddings["validation"][indices]), predicted[indices]):
                raise ExperimentBlocked("Candidate reload changed predictions")
            def end_to_end(batch):
                vectors = encoder.encode(compose_email_texts(batch).tolist(), normalize_embeddings=True,
                    device=device, convert_to_numpy=True, show_progress_bar=False,
                    batch_size=config["encoder"][f"batch_size_{device}"])
                return loaded.predict_embeddings(vectors)
            latency = (benchmark(end_to_end, sample, config["benchmark_repeats"], device)
                       if encoder is not None else {"status": "BLOCKED", "reason": encoder_measurement_blocked})
            trained_head = classifier.named_steps["classifier"] if name == "C_minilm_mlp" else classifier
            state["candidates"].append({"candidate": name, "status": "PASS", "fit_seconds": fit_seconds,
                "artifact": artifact_path.relative_to(paths.project_root).as_posix(),
                "artifact_sha256": sha256_file(artifact_path), "metrics": evaluate(validation, predicted),
                "full_validation_ms_per_email": prediction_seconds * 1000 / len(validation),
                "latency": latency, "resources_before": resources, "resources_after": measured_resources(),
                "warnings": [type(item.message).__name__ for item in caught],
                "training_iterations": int(getattr(trained_head, "n_iter_", 0)),
                "training_loss": float(trained_head.loss_) if hasattr(trained_head, "loss_") else None,
                "classifier_classes": [str(value) for value in classifier.classes_],
                "calibrated_confidence_available": False})
            _json(checkpoint_path, state)
            del classifier, loaded
            gc.collect()
        if sha256_file(baseline_path) != baseline_sha:
            raise ExperimentBlocked("Existing baseline changed during the experiment")
        successful = {item["candidate"]: item for item in state["candidates"] if item["status"] == "PASS"}
        baseline_f1 = successful.get("A_tfidf_svc", {}).get("metrics", {}).get("macro_f1")
        mlp_f1 = successful.get("C_minilm_mlp", {}).get("metrics", {}).get("macro_f1")
        if mlp_f1 is None or baseline_f1 is None:
            conclusion = "Chưa đủ candidate hoàn tất để kết luận MLP so với baseline; xem BLOCKED và lý do thực tế."
        else:
            conclusion = (f"MLP Macro F1 trừ TF-IDF baseline = {mlp_f1 - baseline_f1:+.6f}. "
                + ("MLP cải thiện trên Development Validation này; chưa chứng minh khả năng khái quát độc lập."
                   if mlp_f1 > baseline_f1 else "MLP không cải thiện baseline trên Development Validation này."))
        training_config = json.loads((paths.config / "development_training.json").read_text(encoding="utf-8"))
        from .development_training import xlm_resource_decision
        xlm_allowed, xlm_reason = xlm_resource_decision(measured_resources(), training_config["xlm_roberta"])
        output = {"experiment": EXPERIMENT, "scope": "development_validation_only", "seed": config["seed"],
            "fingerprint": fingerprint, "preflight": check, "config": config, "cache": cache_info,
            "encoder_measurement_blocked": encoder_measurement_blocked,
            "train_rows": len(train), "validation_rows": len(validation), "leakage": manifest["leakage"],
            "distribution": {split: {str(lang): {str(label): int(count) for label, count in frame.loc[
                frame["language"].eq(lang), "label"].value_counts().items()}
                for lang in ("en", "vi", "zh")} for split, frame in (("train", train), ("validation", validation))},
            "candidates": state["candidates"], "conclusion": conclusion,
            "xlmr_gate": {"allowed": xlm_allowed, "reason": xlm_reason, "trained": False},
            "final_test_opened": False, "production_selected": False, "runtime_modified": False}
        if len(successful) == 3 and all(item.get("latency", {}).get("status") != "BLOCKED" for item in successful.values()):
            output["encoder_measurement_blocked"] = None
        _json(results_dir / "validation_metrics.json", output)
        rows = [{"candidate": item["candidate"], "status": item["status"],
                 "accuracy": item.get("metrics", {}).get("accuracy"),
                 "macro_f1": item.get("metrics", {}).get("macro_f1"),
                 "phishing_recall": item.get("metrics", {}).get("phishing_recall"),
                 "fit_seconds": item.get("fit_seconds"),
                 "end_to_end_ms_per_email": item.get("latency", {}).get("median_ms_per_email")}
                for item in state["candidates"]]
        pd.DataFrame(rows).to_csv(results_dir / "model_comparison.csv", index=False)
        for item in state["candidates"]:
            if "metrics" in item:
                pd.DataFrame(item["metrics"]["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(
                    results_dir / f"{item['candidate']}_confusion_matrix.csv")
        _json(checkpoint_path, state)
        _write_report(output, paths.project_root / "MULTILINGUAL_DL_EXPERIMENT_01.md")
        return output


def measure_saved_candidates(paths: V21Paths | None = None) -> dict:
    """Separate-process latency measurement, without fitting any candidate."""
    paths = paths or resolve_v21_paths()
    config = load_config(paths)
    results_dir = paths.results / "experiments" / EXPERIMENT
    output_path = results_dir / "validation_metrics.json"
    with offline_boundary(paths.project_root), threadpool_limits(limits=config["cpu_threads"]):
        output = json.loads(output_path.read_text(encoding="utf-8"))
        expected_config = output["fingerprint"]["config_sha256"]
        if sha256_file(paths.config / "experiments" / f"{EXPERIMENT}.json") != expected_config:
            raise ExperimentBlocked("Benchmark configuration changed after fitting")
        check = preflight(paths, config)
        snapshot = check.pop("snapshot")
        if not check["encoder_allowed"]:
            raise ExperimentBlocked(check["encoder_reason"])
        manifest, train, validation = verify_and_load_development(paths, config["max_input_characters"])
        if any(manifest["output_sha256"][split] != output["fingerprint"][f"{split}_sha256"]
               for split in ("train", "validation")):
            raise ExperimentBlocked("Development inputs changed after fitting")
        del train
        gc.collect()
        device = "cuda" if check["resources"].get("cuda") else "cpu"
        encoder = load_encoder(snapshot, config["encoder"], device)
        sample = validation.iloc[_smoke_indices(validation, config)]
        receipt = {"fingerprint": output["fingerprint"], "preflight": check, "device": device,
                   "fit_performed": False, "candidates": {}}
        for item in output["candidates"]:
            if item["candidate"] == "A_tfidf_svc" or item["status"] != "PASS":
                continue
            artifact_path = paths.project_root / item["artifact"]
            if sha256_file(artifact_path) != item["artifact_sha256"]:
                raise ExperimentBlocked("Candidate changed before latency measurement")
            artifact = joblib.load(artifact_path)
            def predict(batch):
                vectors = encoder.encode(compose_email_texts(batch).tolist(), normalize_embeddings=True,
                    convert_to_numpy=True, device=device, show_progress_bar=False,
                    batch_size=config["encoder"][f"batch_size_{device}"])
                return artifact.predict_embeddings(vectors)
            item["latency"] = benchmark(predict, sample, config["benchmark_repeats"], device)
            receipt["candidates"][item["candidate"]] = {"artifact_sha256": item["artifact_sha256"],
                "latency": item["latency"]}
            del artifact
        output["encoder_measurement_blocked"] = None
        output["separate_process_latency"] = receipt
        checkpoint_path = results_dir / "checkpoint.json"
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != output["fingerprint"]:
            raise ExperimentBlocked("Candidate checkpoint changed before timing save")
        state["candidates"] = output["candidates"]
        _json(checkpoint_path, state)
        _json(results_dir / "encoder_latency.json", receipt)
        _json(output_path, output)
        _write_report(output, paths.project_root / "MULTILINGUAL_DL_EXPERIMENT_01.md")
        # Synchronize the compact comparison without touching historical results.
        comparison = pd.read_csv(results_dir / "model_comparison.csv")
        for item in output["candidates"]:
            comparison.loc[comparison["candidate"].eq(item["candidate"]), "end_to_end_ms_per_email"] = item.get("latency", {}).get("median_ms_per_email")
        comparison.to_csv(results_dir / "model_comparison.csv", index=False)
        return receipt
