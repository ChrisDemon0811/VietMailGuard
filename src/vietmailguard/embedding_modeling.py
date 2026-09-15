"""Multilingual embedding helpers for validation-only Version 2 experiments."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from vietmailguard.modeling import EmailTextComposer


class TextEncoder(Protocol):
    """Small protocol implemented by SentenceTransformer and test doubles."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> np.ndarray: ...


def compose_email_texts(frame: pd.DataFrame) -> np.ndarray:
    """Use the same subject/body composer as every other training pipeline."""
    return EmailTextComposer().transform(frame[["subject", "body"]])


def model_cache_identifier(model_id: str, revision: str, max_sequence_length: int) -> str:
    """Return a stable identifier for one exact embedding configuration."""
    return f"{model_id}@{revision}:max_seq_length={max_sequence_length}:l2_normalized=true"


def embedding_row_key(
    *,
    model_identifier: str,
    content_hash: str,
    text: str,
) -> str:
    """Key a cached vector by model version, content hash, and composed text hash."""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = {
        "model_identifier": model_identifier,
        "content_hash": str(content_hash),
        "text_sha256": text_hash,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def dataset_embedding_key(row_keys: Sequence[str]) -> str:
    """Fingerprint an ordered embedding matrix from its per-row cache keys."""
    digest = hashlib.sha256()
    for value in row_keys:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_model_directory(model_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "__", model_id).strip("._-")
    return cleaned or "embedding_model"


@dataclass(frozen=True)
class EmbeddingCacheResult:
    """One loaded or newly generated embedding cache."""

    embeddings: np.ndarray
    cache_hit: bool
    matrix_path: Path
    manifest_path: Path
    original_generation_seconds: float


def load_or_encode_embeddings(
    *,
    frame: pd.DataFrame,
    split_name: str,
    encoder: TextEncoder,
    model_id: str,
    revision: str,
    max_sequence_length: int,
    embedding_dimension: int,
    cache_root: Path,
    batch_size: int,
    device: str,
    generation_seconds: float | None = None,
) -> EmbeddingCacheResult:
    """Load a validated matrix cache or encode and persist it as float32.

    ``generation_seconds`` exists only for deterministic unit-test encoders. Normal
    callers leave it unset so this function measures actual encoder wall time.
    """
    import time

    if "content_hash" not in frame:
        raise ValueError("Embedding input is missing content_hash")
    texts = compose_email_texts(frame)
    identifier = model_cache_identifier(model_id, revision, max_sequence_length)
    row_keys = [
        embedding_row_key(
            model_identifier=identifier,
            content_hash=content_hash,
            text=str(text),
        )
        for content_hash, text in zip(frame["content_hash"].astype(str), texts)
    ]
    dataset_key = dataset_embedding_key(row_keys)
    directory = cache_root / _safe_model_directory(model_id) / revision
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{split_name}__{dataset_key}"
    matrix_path = directory / f"{stem}.npy"
    manifest_path = directory / f"{stem}.json"

    if matrix_path.exists() and manifest_path.exists():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "model_identifier": identifier,
            "dataset_key": dataset_key,
            "rows": int(len(frame)),
            "embedding_dimension": int(embedding_dimension),
            "dtype": "float32",
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            matrix = np.load(matrix_path, mmap_mode="r")
            if matrix.shape == (len(frame), embedding_dimension) and matrix.dtype == np.float32:
                return EmbeddingCacheResult(
                    embeddings=matrix,
                    cache_hit=True,
                    matrix_path=matrix_path,
                    manifest_path=manifest_path,
                    original_generation_seconds=float(metadata["generation_seconds"]),
                )

    started = time.perf_counter()
    encoded = encoder.encode(
        texts.tolist(),
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    measured_seconds = time.perf_counter() - started
    elapsed = measured_seconds if generation_seconds is None else float(generation_seconds)
    matrix = np.asarray(encoded, dtype=np.float32)
    expected_shape = (len(frame), embedding_dimension)
    if matrix.shape != expected_shape:
        raise ValueError(f"Encoder returned shape {matrix.shape}; expected {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Encoder returned NaN or infinite values")

    temporary_path = matrix_path.with_suffix(".npy.tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    temporary_path.replace(matrix_path)
    metadata = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "model_identifier": identifier,
        "max_sequence_length": int(max_sequence_length),
        "dataset_key": dataset_key,
        "rows": int(len(frame)),
        "embedding_dimension": int(embedding_dimension),
        "dtype": "float32",
        "normalized_embeddings": True,
        "row_key_contract": "sha256(model_identifier + content_hash + composed_text_sha256)",
        "row_keys_sha256": dataset_embedding_key(row_keys),
        "generation_seconds": elapsed,
        "matrix_bytes": int(matrix.nbytes),
    }
    manifest_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return EmbeddingCacheResult(
        embeddings=np.load(matrix_path, mmap_mode="r"),
        cache_hit=False,
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        original_generation_seconds=elapsed,
    )


def english_training_mask(frame: pd.DataFrame) -> pd.Series:
    """Select only English rows for the uncontaminated M1 classifier."""
    if "language" not in frame:
        raise ValueError("Training data is missing language")
    return frame["language"].fillna("").astype(str).eq("en")


@dataclass
class EmbeddingClassifierArtifact:
    """Serializable classifier plus the exact encoder/composer contract."""

    experiment_id: str
    classifier: BaseEstimator
    model_id: str
    model_revision: str
    max_sequence_length: int
    embedding_dimension: int
    label_order: list[str]
    training_scope: str

    def predict_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict(embeddings), dtype=object)

    def predict_proba_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if not hasattr(self.classifier, "predict_proba"):
            raise TypeError("This uncalibrated classifier has no probability output")
        return np.asarray(self.classifier.predict_proba(embeddings), dtype=float)

    @property
    def confidence_capability(self) -> str:
        if hasattr(self.classifier, "predict_proba"):
            return "native_predict_proba"
        return "unavailable_uncalibrated"
