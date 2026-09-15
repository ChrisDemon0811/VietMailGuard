"""Run Version 2 multilingual-embedding experiments without opening the test split."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.embedding_modeling import (  # noqa: E402
    EmbeddingClassifierArtifact,
    compose_email_texts,
    english_training_mask,
    load_or_encode_embeddings,
)
from vietmailguard.file_utils import sha256_file  # noqa: E402
from vietmailguard.modeling import (  # noqa: E402
    compute_metrics,
    load_json,
    read_email_split,
    runtime_versions,
    split_summary,
    validate_labels,
    write_json,
)
from vietmailguard.v2_modeling import (  # noqa: E402
    assert_no_cross_split_values,
    assert_only_model_selection_inputs,
    evaluate_validation_subsets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multilingual embedding classifiers on Version 2 train and "
            "validation data only."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "v2_embedding_experiments.json",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Encoder device. 'auto' uses CUDA when available.",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def _metric(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if np.isnan(number) else number


def _recall(metrics: dict[str, Any], label: str) -> float | None:
    return _metric(metrics["per_class"][label]["recall"])


def _build_classifier(name: str, config: dict[str, Any], seed: int) -> Any:
    kind = config["kind"]
    if kind == "logistic_regression":
        return LogisticRegression(
            C=float(config["C"]),
            class_weight=config["class_weight"],
            solver=str(config["solver"]),
            max_iter=int(config["max_iter"]),
            random_state=seed,
        )
    if kind == "linear_svm":
        return LinearSVC(
            C=float(config["C"]),
            class_weight=config["class_weight"],
            max_iter=int(config["max_iter"]),
            random_state=seed,
        )
    raise ValueError(f"Unsupported embedding classifier {name}: {kind}")


def _subset_columns(subsets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    english = subsets["english_original_artifact"]
    translated = subsets["translated_vietnamese"]
    existing = subsets["existing_translated_vietnamese"]
    controlled = subsets["controlled_translated_augmentation"]
    return {
        "english_validation_rows": english["rows"],
        "english_macro_f1": _metric(english["macro_f1_supported_classes"]),
        "english_phishing_recall": _recall(english, "phishing"),
        "translated_vietnamese_validation_rows": translated["rows"],
        "translated_vietnamese_macro_f1_supported_classes": _metric(
            translated["macro_f1_supported_classes"]
        ),
        "translated_vietnamese_three_class_macro_f1": _metric(
            translated["three_class_macro_f1"]
        ),
        "translated_vietnamese_spam_recall": _recall(translated, "spam"),
        "translated_vietnamese_phishing_recall": _recall(translated, "phishing"),
        "existing_translated_vietnamese_rows": existing["rows"],
        "controlled_translation_validation_rows": controlled["rows"],
    }


def _embedding_result_row(
    *,
    experiment_id: str,
    training_scope: str,
    classifier_name: str,
    metrics: dict[str, Any],
    subsets: dict[str, dict[str, Any]],
    fit_seconds: float,
    prediction_seconds: float,
    generation_seconds: float,
    artifact_path: Path,
    encoder_bytes: int,
    confidence_capability: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "experiment_family": "multilingual_embedding",
        "training_scope": training_scope,
        "validation_cohort": "v2_validation",
        "representation": "paraphrase-multilingual-MiniLM-L12-v2 (384d)",
        "classifier": classifier_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "phishing_recall": _recall(metrics, "phishing"),
        **_subset_columns(subsets),
        "fit_seconds": fit_seconds,
        "embedding_generation_seconds": generation_seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_latency_ms_per_row": 1000.0 * prediction_seconds / max(1, subsets["combined"]["rows"]),
        "classifier_artifact_bytes": artifact_path.stat().st_size,
        "encoder_artifact_bytes": encoder_bytes,
        "confidence_capability": confidence_capability,
        "candidate_artifact": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
        "directly_comparable_to_v2": True,
        "notes": "Validation only; translated Vietnamese is not native Vietnamese.",
    }


def _load_reference_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v1_path = project_path(config["references"]["v1_selected_model"])
    if v1_path.exists():
        selected = load_json(v1_path)
        metrics = selected.get("validation_metrics", {})
        pipeline_path = PROJECT_ROOT / str(selected.get("validation_candidate_pipeline", ""))
        rows.append(
            {
                "experiment_id": f"v1::{selected.get('selected_experiment_id', 'saved_model')}",
                "experiment_family": "v1_english_tfidf_reference",
                "training_scope": "V1 English only",
                "validation_cohort": "v1_validation_different_cohort",
                "representation": selected.get("feature_set", "TF-IDF"),
                "classifier": selected.get("classifier", "unknown"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "phishing_recall": metrics.get("phishing_recall"),
                "english_validation_rows": selected.get("data", {}).get("validation", {}).get("rows"),
                "english_macro_f1": metrics.get("macro_f1"),
                "english_phishing_recall": metrics.get("phishing_recall"),
                "translated_vietnamese_validation_rows": None,
                "translated_vietnamese_macro_f1_supported_classes": None,
                "translated_vietnamese_three_class_macro_f1": None,
                "translated_vietnamese_spam_recall": None,
                "translated_vietnamese_phishing_recall": None,
                "existing_translated_vietnamese_rows": None,
                "controlled_translation_validation_rows": None,
                "fit_seconds": None,
                "embedding_generation_seconds": None,
                "prediction_seconds": None,
                "prediction_latency_ms_per_row": None,
                "classifier_artifact_bytes": pipeline_path.stat().st_size if pipeline_path.is_file() else None,
                "encoder_artifact_bytes": None,
                "confidence_capability": "saved_v1_reference",
                "candidate_artifact": selected.get("validation_candidate_pipeline"),
                "directly_comparable_to_v2": False,
                "notes": "Descriptive only: V1 uses a different validation cohort.",
            }
        )

    tfidf_path = project_path(config["references"]["v2_tfidf_metrics"])
    if tfidf_path.exists():
        artifact = load_json(tfidf_path)
        recommended = artifact["recommended_tfidf_candidate"]
        details = artifact["experiments"][recommended["experiment_id"]]
        global_metrics = details["global_validation"]
        subsets = details["validation_subsets"]
        candidate_path = PROJECT_ROOT / recommended["candidate_pipeline"]
        rows.append(
            {
                "experiment_id": f"v2_tfidf::{recommended['experiment_id']}",
                "experiment_family": "v2_bilingual_tfidf_reference",
                "training_scope": "M2 bilingual train",
                "validation_cohort": "v2_validation",
                "representation": recommended["feature_set"],
                "classifier": recommended["classifier"],
                "accuracy": global_metrics["accuracy"],
                "macro_f1": global_metrics["macro_f1"],
                "phishing_recall": _recall(global_metrics, "phishing"),
                **_subset_columns(subsets),
                "fit_seconds": details.get("fit_seconds"),
                "embedding_generation_seconds": None,
                "prediction_seconds": details.get("predict_seconds"),
                "prediction_latency_ms_per_row": (
                    1000.0 * float(details["predict_seconds"]) / max(1, subsets["combined"]["rows"])
                ),
                "classifier_artifact_bytes": candidate_path.stat().st_size if candidate_path.is_file() else None,
                "encoder_artifact_bytes": None,
                "confidence_capability": recommended["confidence_capability"],
                "candidate_artifact": recommended["candidate_pipeline"],
                "directly_comparable_to_v2": True,
                "notes": "Direct V2 validation comparator; no embedding model dependency.",
            }
        )
    return rows


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _format(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{float(value):.6f}"


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(_format(value) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    comparison: pd.DataFrame,
    embedding_rows: pd.DataFrame,
    *,
    model: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    encoder_bytes: int,
    device: str,
    train_cache_hit: bool,
    validation_cache_hit: bool,
    total_generation_seconds: float,
    gpu_peak_bytes: int | None,
    train_matrix_bytes: int,
    validation_matrix_bytes: int,
    smoke_seconds: float,
    smoke_rows: int,
) -> None:
    m1 = embedding_rows[embedding_rows["training_scope"].eq("M1 English train only")]
    m2 = embedding_rows[embedding_rows["training_scope"].eq("M2 bilingual train")]
    best_m1 = m1.sort_values(["macro_f1", "phishing_recall"], ascending=False).iloc[0]
    best_m2 = m2.sort_values(["macro_f1", "phishing_recall"], ascending=False).iloc[0]
    tfidf = comparison[comparison["experiment_family"].eq("v2_bilingual_tfidf_reference")].iloc[0]
    vi_m1 = best_m1["translated_vietnamese_macro_f1_supported_classes"]
    vi_m2 = best_m2["translated_vietnamese_macro_f1_supported_classes"]
    english_delta = float(best_m2["english_macro_f1"]) - float(tfidf["english_macro_f1"])
    deployment_ratio = encoder_bytes / max(1, int(tfidf["classifier_artifact_bytes"]))

    columns = [
        "experiment_id",
        "macro_f1",
        "phishing_recall",
        "english_macro_f1",
        "english_phishing_recall",
        "translated_vietnamese_macro_f1_supported_classes",
        "translated_vietnamese_spam_recall",
        "prediction_latency_ms_per_row",
    ]
    lines = [
        "# Thực nghiệm multilingual embedding Version 2",
        "",
        "## Ranh giới khoa học",
        "",
        "Thực nghiệm chỉ đọc `data/splits/v2/train.csv` và "
        "`data/splits/v2/validation.csv`. Held-out test set không được mở, không được "
        "đánh giá và không tham gia lựa chọn model.",
        "",
        f"- Train: **{len(train):,}** rows; validation: **{len(validation):,}** rows",
        f"- M1 train: **{int(english_training_mask(train).sum()):,} English rows only**",
        f"- M2 train: **{len(train):,} English + translated Vietnamese rows**",
        "- Vietnamese validation chỉ là dữ liệu dịch, không phải native Vietnamese.",
        "- Vietnamese validation không có phishing; phishing recall và three-class Macro F1 "
        "của subset này là N/A.",
        "",
        "## Encoder đã chọn",
        "",
        f"- Model identifier: `{model['model_id']}`",
        f"- Revision: `{model['revision']}`",
        f"- Provider/source: {model['provider']}",
        f"- License: {model['license']}",
        f"- Embedding dimension: **{model['embedding_dimension']}**",
        f"- Model artifact thực đo: **{encoder_bytes / 1024**2:.1f} MiB**",
        "- Model card công bố hỗ trợ 50 ngôn ngữ, bao gồm English và Vietnamese.",
        f"- Thiết bị encode: **{device}**; max sequence length: **{model['max_sequence_length']} tokens**",
        "- Model card: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "- Mỗi input là `subject + newline + body`, dùng chung `EmailTextComposer` với "
        "pipeline TF-IDF. Text vượt 128 tokens bị truncate bởi encoder; đây là limitation.",
        "",
        "## Kết quả validation",
        "",
        _table(comparison, columns),
        "",
        "V1 chỉ là tham chiếu mô tả vì cohort validation khác. Các dòng V2 dùng đúng cùng "
        "validation split nên so sánh trực tiếp được.",
        "",
        "## Trả lời câu hỏi nghiên cứu",
        "",
        f"- **Zero-shot Vietnamese transfer:** M1 tốt nhất đạt translated-Vietnamese Macro F1 "
        f"(trên normal/spam có support) **{_format(vi_m1)}** và spam recall "
        f"**{_format(best_m1['translated_vietnamese_spam_recall'])}**. Đây là evidence trên "
        "translated validation, không phải native Vietnamese.",
        f"- **Ảnh hưởng của Vietnamese training:** M2 tốt nhất đạt translated-Vietnamese "
        f"Macro F1 **{_format(vi_m2)}**, delta so với M1 tốt nhất là "
        f"**{float(vi_m2) - float(vi_m1):+.6f}**.",
        f"- **TF-IDF có còn cạnh tranh:** V2 TF-IDF Macro F1 **{_format(tfidf['macro_f1'])}**; "
        f"M2 embedding tốt nhất **{_format(best_m2['macro_f1'])}**.",
        f"- **English performance:** M2 embedding English Macro F1 "
        f"**{_format(best_m2['english_macro_f1'])}**, chênh **{english_delta:+.6f}** so với "
        "V2 TF-IDF trên cùng English validation subset.",
        f"- **Deployment cost:** encoder khoảng {encoder_bytes / 1024**2:.1f} MiB, lớn gấp "
        f"xấp xỉ **{deployment_ratio:.1f}×** artifact TF-IDF comparator; còn cần Torch/"
        "Transformers và chi phí encode. Không chọn embedding chỉ vì mới hơn.",
        "",
        "## Runtime và cache",
        "",
        f"- Embedding generation time được ghi nhận: **{total_generation_seconds:.2f} s**",
        f"- End-to-end smoke latency ({smoke_rows} emails, encode + predict): "
        f"**{smoke_seconds:.4f} s**, tương đương **{1000.0 * smoke_seconds / smoke_rows:.2f} ms/email** "
        "cho batch nhỏ trên máy audit.",
        "- `prediction_latency_ms_per_row` trong bảng là classifier-only latency trên "
        "toàn validation matrix; không bao gồm encoder.",
        f"- Train cache hit trong lần chạy này: **{train_cache_hit}**",
        f"- Validation cache hit trong lần chạy này: **{validation_cache_hit}**",
        "- Cache key gồm model identifier/revision, content hash và SHA-256 của composed text.",
        "- Cache nằm dưới `data/cache/embeddings/v2/` và bị `.gitignore` loại khỏi Git.",
        f"- Embedding matrices: train **{train_matrix_bytes / 1024**2:.1f} MiB**, "
        f"validation **{validation_matrix_bytes / 1024**2:.1f} MiB** (float32).",
        f"- Peak CUDA tensor memory: **{gpu_peak_bytes / 1024**2:.1f} MiB**"
        if gpu_peak_bytes is not None
        else "- Peak CUDA tensor memory: N/A (CPU encoder).",
        "",
        "## Calibration và artifacts",
        "",
        "Logistic Regression cung cấp probability thật qua `predict_proba`. LinearSVC chưa "
        "calibrate; raw decision score không được gọi là probability. Bốn classifier được lưu "
        "dưới candidate directory, không ghi đè production model.",
        "",
        "## Limitations",
        "",
        "- Không có native Vietnamese validation benchmark.",
        "- Vietnamese validation chỉ có 172 normal và 8 spam, không có phishing.",
        "- Controlled translations chỉ nằm trong train theo leakage policy.",
        "- Encoder truncate sau 128 tokens; nội dung quan trọng ở cuối email có thể bị mất.",
        "- M1 không thấy Vietnamese labeled training rows, nhưng pretrained encoder vốn đã "
        "được huấn luyện đa ngôn ngữ; đây là zero-shot ở tầng classifier, không phải encoder.",
        "- Chưa chọn final production model và chưa đánh giá held-out test set.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    assert_only_model_selection_inputs(config)
    model_config = config["embedding"]
    label_order = [str(value) for value in config["label_order"]]

    train_path = project_path(config["data"]["train"])
    validation_path = project_path(config["data"]["validation"])
    print("Loading Version 2 train and validation only...", flush=True)
    train = read_email_split(train_path)
    validation = read_email_split(validation_path)
    validate_labels(train, label_order, "train")
    validate_labels(validation, label_order, "validation")
    assert_no_cross_split_values(train, validation, config["leakage_columns"])
    print(f"train: {split_summary(train)}", flush=True)
    print(f"validation: {split_summary(validation)}", flush=True)
    print("Train/validation leakage checks: PASS", flush=True)

    try:
        import sentence_transformers
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding dependencies are missing. Run: "
            "python -m pip install -r requirements-embeddings.txt"
        ) from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    print(f"Loading {model_config['model_id']} at pinned revision...", flush=True)
    encoder = SentenceTransformer(
        model_config["model_id"],
        revision=model_config["revision"],
        device=device,
        trust_remote_code=False,
    )
    encoder.max_seq_length = int(model_config["max_sequence_length"])
    dimension = int(encoder.get_embedding_dimension())
    if dimension != int(model_config["embedding_dimension"]):
        raise RuntimeError(
            f"Embedding dimension mismatch: loaded={dimension}, configured="
            f"{model_config['embedding_dimension']}"
        )

    snapshot = Path(
        snapshot_download(
            repo_id=model_config["model_id"],
            revision=model_config["revision"],
            local_files_only=True,
        )
    )
    encoder_bytes = _directory_size(snapshot)
    batch_size = int(
        model_config["batch_size_cuda"] if device == "cuda" else model_config["batch_size_cpu"]
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cache_root = project_path(model_config["cache_directory"])
    cache_started = time.perf_counter()
    train_cache = load_or_encode_embeddings(
        frame=train,
        split_name="train",
        encoder=encoder,
        model_id=model_config["model_id"],
        revision=model_config["revision"],
        max_sequence_length=int(model_config["max_sequence_length"]),
        embedding_dimension=dimension,
        cache_root=cache_root,
        batch_size=batch_size,
        device=device,
    )
    validation_cache = load_or_encode_embeddings(
        frame=validation,
        split_name="validation",
        encoder=encoder,
        model_id=model_config["model_id"],
        revision=model_config["revision"],
        max_sequence_length=int(model_config["max_sequence_length"]),
        embedding_dimension=dimension,
        cache_root=cache_root,
        batch_size=batch_size,
        device=device,
    )
    cache_wall_seconds = time.perf_counter() - cache_started
    total_generation_seconds = (
        train_cache.original_generation_seconds + validation_cache.original_generation_seconds
    )
    memory_probe_rows = 0
    if device == "cuda" and train_cache.cache_hit and validation_cache.cache_hit:
        # A cache-only rerun would otherwise under-report the encoder's batch memory.
        memory_probe_rows = min(batch_size, len(train))
        encoder.encode(
            compose_email_texts(train.iloc[:memory_probe_rows]).tolist(),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
    gpu_peak_bytes = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    print(
        f"Embeddings ready: train={train_cache.embeddings.shape} "
        f"(cache_hit={train_cache.cache_hit}), validation={validation_cache.embeddings.shape} "
        f"(cache_hit={validation_cache.cache_hit}), current_run_seconds={cache_wall_seconds:.2f}",
        flush=True,
    )

    english_mask = english_training_mask(train).to_numpy()
    if int(english_mask.sum()) == len(train):
        raise RuntimeError("M1 audit expected bilingual train data but found no Vietnamese rows")
    scopes = {
        "m1_english_only": (english_mask, "M1 English train only"),
        "m2_bilingual": (np.ones(len(train), dtype=bool), "M2 bilingual train"),
    }
    candidate_directory = project_path(config["outputs"]["candidate_directory"])
    candidate_directory.mkdir(parents=True, exist_ok=True)
    embedding_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    y_validation = validation["label"].astype(str).to_numpy()

    for scope_id, (mask, scope_name) in scopes.items():
        y_train = train.loc[mask, "label"].astype(str).to_numpy()
        observed = set(y_train)
        if observed != set(label_order):
            raise RuntimeError(f"{scope_id} lacks configured classes: {sorted(observed)}")
        for classifier_id, classifier_config in config["classifiers"].items():
            experiment_id = f"embedding__{scope_id}__{classifier_id}"
            print(f"{experiment_id}: fitting {int(mask.sum()):,} rows", flush=True)
            classifier = _build_classifier(classifier_id, classifier_config, int(config["seed"]))
            started = time.perf_counter()
            classifier.fit(train_cache.embeddings[mask], y_train)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predictions = np.asarray(classifier.predict(validation_cache.embeddings), dtype=object)
            prediction_seconds = time.perf_counter() - started
            metrics = compute_metrics(y_validation, predictions, label_order)
            subsets = evaluate_validation_subsets(validation, predictions, label_order)

            artifact = EmbeddingClassifierArtifact(
                experiment_id=experiment_id,
                classifier=classifier,
                model_id=model_config["model_id"],
                model_revision=model_config["revision"],
                max_sequence_length=int(model_config["max_sequence_length"]),
                embedding_dimension=dimension,
                label_order=label_order,
                training_scope=scope_name,
            )
            artifact_path = candidate_directory / f"{experiment_id}.joblib"
            joblib.dump(artifact, artifact_path, compress=3)
            reloaded = joblib.load(artifact_path)
            original_smoke = artifact.predict_embeddings(validation_cache.embeddings[:3])
            reloaded_smoke = reloaded.predict_embeddings(validation_cache.embeddings[:3])
            if not np.array_equal(original_smoke, reloaded_smoke):
                raise RuntimeError(f"Reload smoke failed: {experiment_id}")
            if reloaded.confidence_capability == "native_predict_proba":
                probabilities = reloaded.predict_proba_embeddings(validation_cache.embeddings[:3])
                if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
                    raise RuntimeError(f"Probability smoke failed: {experiment_id}")

            row = _embedding_result_row(
                experiment_id=experiment_id,
                training_scope=scope_name,
                classifier_name=classifier_config["display_name"],
                metrics=metrics,
                subsets=subsets,
                fit_seconds=fit_seconds,
                prediction_seconds=prediction_seconds,
                generation_seconds=total_generation_seconds,
                artifact_path=artifact_path,
                encoder_bytes=encoder_bytes,
                confidence_capability=reloaded.confidence_capability,
            )
            embedding_rows.append(row)
            details[experiment_id] = {
                "training_scope": scope_name,
                "training_rows": int(mask.sum()),
                "training_language_distribution": {
                    str(key): int(value)
                    for key, value in train.loc[mask, "language"].value_counts().sort_index().items()
                },
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
                "global_validation": metrics,
                "validation_subsets": subsets,
                "candidate_artifact": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
                "candidate_sha256": sha256_file(artifact_path),
                "reload_smoke_test": True,
                "confidence_capability": reloaded.confidence_capability,
            }
            manifest.append(
                {
                    "experiment_id": experiment_id,
                    "path": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(artifact_path),
                    "training_scope": scope_name,
                    "confidence_capability": reloaded.confidence_capability,
                    "reload_smoke_test": True,
                }
            )
            print(
                f"{experiment_id}: macro_f1={metrics['macro_f1']:.6f}, "
                f"phishing_recall={metrics['per_class']['phishing']['recall']:.6f}, "
                f"english_macro_f1={subsets['english_original_artifact']['macro_f1_supported_classes']:.6f}, "
                f"translated_vi_macro_f1_supported="
                f"{subsets['translated_vietnamese']['macro_f1_supported_classes']:.6f}",
                flush=True,
            )

    comparison = pd.DataFrame([*_load_reference_rows(config), *embedding_rows])
    comparison.insert(0, "comparison_order", np.arange(1, len(comparison) + 1))
    embedding_frame = comparison[comparison["experiment_family"].eq("multilingual_embedding")]

    smoke_texts = compose_email_texts(validation.iloc[:3])
    started = time.perf_counter()
    smoke_embeddings = encoder.encode(
        smoke_texts.tolist(),
        batch_size=3,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    best_embedding = embedding_frame.sort_values(
        ["macro_f1", "phishing_recall"], ascending=False
    ).iloc[0]
    smoke_artifact = joblib.load(PROJECT_ROOT / best_embedding["candidate_artifact"])
    smoke_predictions = smoke_artifact.predict_embeddings(smoke_embeddings)
    smoke_seconds = time.perf_counter() - started

    output_path = project_path(config["outputs"]["comparison"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False, encoding="utf-8")

    metrics_artifact = {
        "schema_version": 1,
        "experiment_scope": config["experiment_scope"],
        "held_out_data_accessed": False,
        "final_production_model_selected": False,
        "seed": int(config["seed"]),
        "config_sha256": sha256_file(args.config),
        "input_data": {
            "train": {"path": config["data"]["train"], "sha256": sha256_file(train_path), **split_summary(train)},
            "validation": {"path": config["data"]["validation"], "sha256": sha256_file(validation_path), **split_summary(validation)},
        },
        "leakage_check": {"passed": True, "columns": config["leakage_columns"]},
        "embedding": {
            **model_config,
            "actual_embedding_dimension": dimension,
            "device": device,
            "batch_size": batch_size,
            "encoder_artifact_bytes": encoder_bytes,
            "train_cache_hit": train_cache.cache_hit,
            "validation_cache_hit": validation_cache.cache_hit,
            "train_embedding_generation_seconds": train_cache.original_generation_seconds,
            "validation_embedding_generation_seconds": validation_cache.original_generation_seconds,
            "total_embedding_generation_seconds": total_generation_seconds,
            "current_run_cache_wall_seconds": cache_wall_seconds,
            "train_matrix_bytes": int(train_cache.embeddings.nbytes),
            "validation_matrix_bytes": int(validation_cache.embeddings.nbytes),
            "peak_cuda_tensor_bytes": gpu_peak_bytes,
            "cuda_memory_probe_rows": memory_probe_rows,
            "end_to_end_smoke_rows": len(smoke_texts),
            "end_to_end_smoke_seconds": smoke_seconds,
            "end_to_end_smoke_ms_per_row": 1000.0 * smoke_seconds / len(smoke_texts),
        },
        "experiments": details,
        "runtime": {
            **runtime_versions(),
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "calibration_policy": (
            "Logistic Regression exposes native probabilities. LinearSVC is uncalibrated; "
            "decision_function values are not probabilities."
        ),
    }
    write_json(project_path(config["outputs"]["metrics"]), metrics_artifact)
    write_json(
        project_path(config["outputs"]["candidate_manifest"]),
        {
            "schema_version": 1,
            "final_production_models": [],
            "encoder": {
                "model_id": model_config["model_id"],
                "revision": model_config["revision"],
                "artifact_bytes": encoder_bytes,
            },
            "candidates": manifest,
        },
    )
    _write_report(
        project_path(config["outputs"]["report"]),
        comparison,
        embedding_frame,
        model=model_config,
        train=train,
        validation=validation,
        encoder_bytes=encoder_bytes,
        device=device,
        train_cache_hit=train_cache.cache_hit,
        validation_cache_hit=validation_cache.cache_hit,
        total_generation_seconds=total_generation_seconds,
        gpu_peak_bytes=gpu_peak_bytes,
        train_matrix_bytes=int(train_cache.embeddings.nbytes),
        validation_matrix_bytes=int(validation_cache.embeddings.nbytes),
        smoke_seconds=smoke_seconds,
        smoke_rows=len(smoke_texts),
    )
    print(f"End-to-end smoke predictions: {smoke_predictions.tolist()} ({smoke_seconds:.4f}s)")
    print(comparison[["experiment_id", "macro_f1", "phishing_recall", "english_macro_f1", "translated_vietnamese_macro_f1_supported_classes", "translated_vietnamese_spam_recall"]].to_string(index=False))
    print("Final production model selected: False")
    print("Held-out test data accessed: False")
    print(f"Comparison: {output_path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
