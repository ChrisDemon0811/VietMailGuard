"""Opt-in, local-only MiniLM + MLP inference. Never fits or downloads anything."""

from __future__ import annotations

import importlib.util
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from vietmailguard.embedding_modeling import EmbeddingClassifierArtifact, compose_email_texts
from vietmailguard.file_utils import sha256_file
from vietmailguard.inference import EmailSecurityInference
from .runtime_inference import (
    PROJECT_ROOT, V21DevelopmentInference, V21ModelUnavailable, _bounded_model_input,
)

DEFAULT_CONFIG = PROJECT_ROOT / "config/v21/runtime_multilingual.json"
CLASSES = {"normal", "spam", "phishing"}


def _verified_file(path: Path, expected: str) -> tuple[str, int, int, str]:
    stat = path.stat()
    signature = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if _file_digest(*signature) != expected:
        raise V21ModelUnavailable("multilingual_hash_mismatch")
    return (*signature, expected)


@lru_cache(maxsize=64)
def _file_digest(path: str, size: int, modified: int) -> str:
    del size, modified
    return sha256_file(Path(path))


def _inside(root: Path, name: str, namespace: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to((root / namespace).resolve()):
        raise V21ModelUnavailable("multilingual_invalid_path")
    return path


def verified_configuration(config_path: Path = DEFAULT_CONFIG,
                           root: Path = PROJECT_ROOT) -> tuple[dict, tuple, Path]:
    """Verify config, head, training receipt and all local encoder inputs first."""
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if (config["schema_version"] != 1 or config["scope"] != "multilingual_experimental"
                or config["probability_status"] != "unavailable_uncalibrated"
                or set(config["classes"]) != CLASSES):
            raise V21ModelUnavailable("multilingual_invalid_configuration")
        signatures = []
        for key, namespace in (("artifact", "models/v21_multilingual/experiments"),
                               ("training_config", "config/v21/experiments"),
                               ("training_receipt", "results/v21_multilingual/experiments")):
            path = _inside(root, config[key], namespace)
            signatures.append(_verified_file(path, config[key + "_sha256"]))
        training = json.loads((root / config["training_config"]).read_text(encoding="utf-8"))
        receipt = json.loads((root / config["training_receipt"]).read_text(encoding="utf-8"))
        if (training["scope"] != "development_validation_only"
                or training["encoder"] != config["encoder"]
                or training["max_input_characters"] != config["max_input_characters"]
                or not any(row["candidate"] == "C_minilm_mlp"
                           and row["artifact"] == config["artifact"]
                           and row["sha256"] == config["artifact_sha256"]
                           for row in receipt["candidates"])):
            raise V21ModelUnavailable("multilingual_invalid_training_contract")
        for package in ("torch", "sentence_transformers", "huggingface_hub", "transformers"):
            if importlib.util.find_spec(package) is None:
                raise V21ModelUnavailable("multilingual_missing_dependency")
        from huggingface_hub import snapshot_download

        encoder = config["encoder"]
        snapshot = Path(snapshot_download(encoder["model_id"], revision=encoder["revision"],
                                         local_files_only=True))
        files = config["encoder_files"]
        required = {"config.json", "modules.json", "sentence_bert_config.json",
                    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                    "config_sentence_transformers.json", "1_Pooling/config.json"}
        if not required.issubset(files) or "pytorch_model.bin" not in files:
            raise V21ModelUnavailable("multilingual_invalid_encoder_manifest")
        for name, digest in sorted(files.items()):
            # HF cache may use symlinks to its own blobs; verify the resolved bytes.
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise V21ModelUnavailable("multilingual_invalid_path")
            signatures.append(_verified_file(snapshot / name, digest))
        return config, tuple(signatures), snapshot
    except V21ModelUnavailable:
        raise
    except FileNotFoundError as error:
        raise V21ModelUnavailable("multilingual_missing_artifact") from error
    except OSError as error:
        raise V21ModelUnavailable("multilingual_encoder_missing") from error
    except (KeyError, TypeError, ValueError, ImportError) as error:
        raise V21ModelUnavailable("multilingual_invalid_configuration") from error


def validate_head(artifact: Any, config: dict) -> None:
    encoder = config["encoder"]
    if (not isinstance(artifact, EmbeddingClassifierArtifact)
            or artifact.experiment_id != "C_minilm_mlp"
            or artifact.training_scope != "dl_01_development_bootstrap_only"
            or (artifact.model_id, artifact.model_revision, artifact.embedding_dimension,
                artifact.max_sequence_length) != (encoder["model_id"], encoder["revision"],
                                                   encoder["dimension"], encoder["max_sequence_length"])):
        raise V21ModelUnavailable("multilingual_invalid_head")
    head = artifact.classifier
    if (not isinstance(head, Pipeline)
            or list(head.named_steps) != ["scale_train_only", "classifier"]
            or not isinstance(head[0], StandardScaler) or not isinstance(head[-1], MLPClassifier)):
        raise V21ModelUnavailable("multilingual_invalid_head")
    check_is_fitted(head[0])
    check_is_fitted(head[-1])
    if (head.n_features_in_ != encoder["dimension"]
            or list(head.classes_) != artifact.label_order or set(artifact.label_order) != CLASSES
            or head[-1].n_outputs_ != 3):
        raise V21ModelUnavailable("multilingual_invalid_classes")


@lru_cache(maxsize=2)
def _head(path: str, digest: str, config_json: str) -> EmbeddingClassifierArtifact:
    if sha256_file(Path(path)) != digest:
        raise V21ModelUnavailable("multilingual_hash_mismatch")
    artifact = joblib.load(path)
    validate_head(artifact, json.loads(config_json))
    return artifact


def multilingual_availability(config_path: Path = DEFAULT_CONFIG,
                             root: Path = PROJECT_ROOT) -> tuple[bool, str]:
    """Preflight does not import Torch or allocate an encoder on sidebar reruns."""
    try:
        config, _, _ = verified_configuration(config_path, root)
        _head(str(root / config["artifact"]), config["artifact_sha256"], json.dumps(config, sort_keys=True))
        return True, "multilingual_verified"
    except V21ModelUnavailable as error:
        return False, str(error)
    except Exception:
        return False, "multilingual_invalid_head"


@lru_cache(maxsize=1)
def _local_encoder(snapshot: str, config_json: str, signatures: tuple):
    del signatures  # invalidate when verified local bytes change
    from sentence_transformers import SentenceTransformer

    config = json.loads(config_json)
    model = SentenceTransformer(snapshot, device="cpu", local_files_only=True,
                                trust_remote_code=False)
    model.max_seq_length = config["max_sequence_length"]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dimension = getattr(model, "get_embedding_dimension", None)
    if dimension is None:
        dimension = model.get_sentence_embedding_dimension
    if dimension() != config["dimension"]:
        raise V21ModelUnavailable("multilingual_invalid_encoder_dimension")
    return model


class MultilingualExperimentalInference(V21DevelopmentInference):
    """Shared security/risk contract; only the classifier representation differs."""

    def __init__(self, artifact: EmbeddingClassifierArtifact, encoder: Any, config: dict) -> None:
        validate_head(artifact, config)
        self.encoder = encoder
        self.encoder_lock = threading.Lock()
        self.maximum_characters = config["max_input_characters"]
        self.artifact_provenance = {
            "candidate_id": artifact.experiment_id, "artifact_sha256": config["artifact_sha256"],
            "encoder_model_id": artifact.model_id, "encoder_revision": artifact.model_revision,
            "training_config_sha256": config["training_config_sha256"],
            "encoder_files": config["encoder_files"], "inference_mode": "v21_multilingual",
        }
        EmailSecurityInference.__init__(self, model=artifact.classifier, metadata={
            "model_version": config["model_version"], "class_names": artifact.label_order,
            "training_languages": ["en", "vi"], "limitations": [
                "MiniLM + MLP experimental: English and translated Vietnamese development only; no Chinese training.",
                "No independent native Vietnamese or three-class Chinese benchmark; Chinese output is unvalidated.",
                "MLP output is not calibrated; no calibrated confidence or probabilities are displayed.",
                "No TF-IDF feature contributions are available for this neural classifier.",
                "Chinese-specific security rules are unavailable; original sender and URLs are analyzed offline.",
            ],
        })

    def _validate_model_contract(self) -> None:
        # The complete head and encoder contract is verified before construction.
        if set(self.model.classes_) != CLASSES:
            raise V21ModelUnavailable("multilingual_invalid_classes")

    def _predict(self, model_input: pd.DataFrame) -> tuple[str, None, dict]:
        texts = compose_email_texts(_bounded_model_input(model_input, self.maximum_characters))
        with self.encoder_lock:
            values = self.encoder.encode(texts.tolist(), batch_size=1, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False, device="cpu")
        matrix = np.asarray(values, dtype=np.float32)
        if (matrix.shape != (1, 384) or not np.isfinite(matrix).all()
                or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-3)):
            raise V21ModelUnavailable("multilingual_invalid_embedding")
        prediction = str(self.model.predict(matrix)[0])
        if prediction not in CLASSES:
            raise V21ModelUnavailable("multilingual_invalid_classes")
        return prediction, None, {}

    def analyze_email(self, **kwargs: Any) -> dict:
        result = super().analyze_email(**kwargs)
        result["model_name"] = "VietMailGuard V2.1 MiniLM + MLP — Multilingual Experimental"
        result["model_scope"] = "multilingual_experimental"
        result["preprocessing"].update({
            "encoder": self.artifact_provenance["encoder_model_id"],
            "encoder_revision": self.artifact_provenance["encoder_revision"],
            "max_sequence_length": self.encoder.max_seq_length,
            "normalize_embeddings": True, "translated": False,
        })
        result["model_explanation"]["limitation"] = (
            "TF-IDF linear feature contributions do not apply to MiniLM + MLP.")
        result["limitations"].append(
            "Encoder tokenization truncates composed text to 128 tokens, matching Experiment 01; "
            "security analysis and stored email use the full original text.")
        return result


@lru_cache(maxsize=1)
def _engine(config_json: str, signatures: tuple, snapshot: str,
            security_signature: tuple) -> MultilingualExperimentalInference:
    del security_signature
    config = json.loads(config_json)
    artifact = _head(signatures[0][0], config["artifact_sha256"], config_json)
    encoder = _local_encoder(snapshot, json.dumps(config["encoder"], sort_keys=True), signatures[3:])
    return MultilingualExperimentalInference(artifact, encoder, config)


def load_multilingual_engine(config_path: Path = DEFAULT_CONFIG,
                             root: Path = PROJECT_ROOT) -> MultilingualExperimentalInference:
    config, signatures, snapshot = verified_configuration(config_path, root)
    try:
        return _engine(json.dumps(config, sort_keys=True), signatures, str(snapshot), tuple(
            sha256_file(root / "config" / name)
            for name in ("risk_rules_v2.json", "security_rules_v2.json")))
    except V21ModelUnavailable:
        raise
    except Exception as error:
        raise V21ModelUnavailable("multilingual_encoder_load_failed") from error


def analyze_multilingual_email(*, sender: object = "", subject: object = "", body: object = "") -> dict:
    return load_multilingual_engine().analyze_email(sender=sender, subject=subject, body=body)
