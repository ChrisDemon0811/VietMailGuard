from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from vietmailguard.embedding_modeling import (
    EmbeddingClassifierArtifact,
    compose_email_texts,
    embedding_row_key,
    english_training_mask,
    load_or_encode_embeddings,
    model_cache_identifier,
)


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, sentences: list[str], **_: object) -> np.ndarray:
        self.calls += 1
        return np.asarray(
            [[float(len(text)), float(text.count("\n"))] for text in sentences],
            dtype=np.float32,
        )


def test_composer_reuses_subject_body_contract_and_handles_missing_subject() -> None:
    frame = pd.DataFrame({"subject": ["Xin chào", None], "body": ["Nội dung", "Body"]})
    assert compose_email_texts(frame).tolist() == ["Xin chào\nNội dung", "\nBody"]


def test_embedding_row_key_changes_with_model_revision_content_or_text() -> None:
    first_model = model_cache_identifier("org/model", "revision-a", 128)
    second_model = model_cache_identifier("org/model", "revision-b", 128)
    baseline = embedding_row_key(
        model_identifier=first_model, content_hash="content-a", text="Xin chào"
    )
    assert baseline != embedding_row_key(
        model_identifier=second_model, content_hash="content-a", text="Xin chào"
    )
    assert baseline != embedding_row_key(
        model_identifier=first_model, content_hash="content-b", text="Xin chào"
    )
    assert baseline != embedding_row_key(
        model_identifier=first_model, content_hash="content-a", text="Xin chào!"
    )


def test_embedding_cache_round_trip_does_not_reencode(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "subject": ["A", "B"],
            "body": ["one", "hai"],
            "content_hash": ["hash-a", "hash-b"],
        }
    )
    encoder = FakeEncoder()
    kwargs = {
        "frame": frame,
        "split_name": "unit",
        "encoder": encoder,
        "model_id": "org/test-model",
        "revision": "commit-1",
        "max_sequence_length": 128,
        "embedding_dimension": 2,
        "cache_root": tmp_path,
        "batch_size": 2,
        "device": "cpu",
        "generation_seconds": 0.25,
    }
    first = load_or_encode_embeddings(**kwargs)
    second = load_or_encode_embeddings(**kwargs)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert encoder.calls == 1
    assert np.array_equal(first.embeddings, second.embeddings)
    metadata = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert metadata["generation_seconds"] == 0.25
    assert metadata["model_identifier"].startswith("org/test-model@commit-1")


def test_m1_english_mask_never_selects_vietnamese_rows() -> None:
    frame = pd.DataFrame({"language": ["en", "vi", "en", ""]})
    assert english_training_mask(frame).tolist() == [True, False, True, False]


def test_embedding_artifact_distinguishes_probability_support() -> None:
    X = np.asarray([[0.0], [0.2], [1.0], [1.2], [2.0], [2.2]])
    y = np.asarray(["normal", "normal", "spam", "spam", "phishing", "phishing"])
    logistic = LogisticRegression(max_iter=500).fit(X, y)
    lr_artifact = EmbeddingClassifierArtifact(
        experiment_id="lr",
        classifier=logistic,
        model_id="model",
        model_revision="revision",
        max_sequence_length=128,
        embedding_dimension=1,
        label_order=["normal", "spam", "phishing"],
        training_scope="unit",
    )
    assert lr_artifact.confidence_capability == "native_predict_proba"
    assert np.allclose(lr_artifact.predict_proba_embeddings(X[:2]).sum(axis=1), 1.0)

    svm = LinearSVC().fit(X, y)
    svm_artifact = EmbeddingClassifierArtifact(
        experiment_id="svm",
        classifier=svm,
        model_id="model",
        model_revision="revision",
        max_sequence_length=128,
        embedding_dimension=1,
        label_order=["normal", "spam", "phishing"],
        training_scope="unit",
    )
    assert svm_artifact.confidence_capability == "unavailable_uncalibrated"
    with pytest.raises(TypeError, match="no probability"):
        svm_artifact.predict_proba_embeddings(X[:2])


def test_embedding_experiment_config_has_no_test_input() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "config" / "v2_embedding_experiments.json").read_text(encoding="utf-8")
    )
    assert set(config["data"]) == {"train", "validation"}
    assert all("test" not in value.lower() for value in config["data"].values())
