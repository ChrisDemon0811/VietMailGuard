"""Explicit V2.3 research inference from pinned local artifacts; never trains."""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from vietmailguard.file_utils import sha256_file
from vietmailguard.inference import EmailSecurityInference
from vietmailguard.embedding_modeling import compose_email_texts
from vietmailguard.v21.runtime_inference import PROJECT_ROOT, V21DevelopmentInference, V21ModelUnavailable, _bounded_model_input
from .modeling import HashedTextClassifier, StreamingHashTfidf

V23_ML_MODE = "v23_ml"
V23_LARGE_MODE = "v23_large_scale"
MODES = (V23_ML_MODE, V23_LARGE_MODE)
CLASSES = ["normal", "spam", "phishing"]


@lru_cache(maxsize=64)
def _digest(path: str, size: int, modified: int) -> str:
    return sha256_file(Path(path))


def verified_configuration(mode: str, root: Path = PROJECT_ROOT) -> tuple[dict, tuple]:
    try:
        if mode not in MODES:raise ValueError("mode")
        config = json.loads((root/f"config/v23/runtime_{mode}.json").read_text(encoding="utf-8"))
        if (config["schema_version"] != 1 or config["scope"] != "research_preview"
                or config["mode"] != mode or config["classes"] != CLASSES
                or config["probability_status"] != "unavailable_uncalibrated"):
            raise ValueError("policy")
        signatures = []
        namespaces = {"artifact": "models/v23_multilingual", "metrics": "results/v23_multilingual",
                      "training_config": "config/v23", "frozen_manifest": "results/v23_multilingual",
                      "protocol": "results/v23_multilingual", "encoder_config": "config/v21"}
        for key, spec in config["files"].items():
            path = (root/spec["path"]).resolve()
            if key not in namespaces or not path.is_relative_to((root/namespaces[key]).resolve()):raise ValueError("path")
            stat = path.stat();signature = (str(path), stat.st_size, stat.st_mtime_ns)
            if _digest(*signature) != spec["sha256"]:raise ValueError("hash")
            signatures.append((*signature, spec["sha256"]))
        files = config["files"]
        metrics = json.loads((root/files["metrics"]["path"]).read_text(encoding="utf-8"))
        manifest = json.loads((root/files["frozen_manifest"]["path"]).read_text(encoding="utf-8"))
        entry = manifest["files"][files["artifact"]["path"]]
        if entry["sha256"] != files["artifact"]["sha256"]:raise ValueError("manifest")
        candidate = next(row for row in metrics["candidates"] if row.get("candidate", row.get("name")) == config["candidate"])
        if (candidate["artifact"] != files["artifact"]["path"] or candidate["artifact_sha256"] != files["artifact"]["sha256"]
                or metrics.get("final_test_used", False) or metrics.get("final_test_ready", False)):
            raise ValueError("provenance")
        config["saved_candidate"] = candidate
        if mode == V23_LARGE_MODE:
            from vietmailguard.v21.multilingual_runtime import verified_configuration as encoder_configuration
            encoder_config, encoder_signatures, snapshot = encoder_configuration(root/files["encoder_config"]["path"], root)
            config["verified_encoder"] = encoder_config
            config["encoder_snapshot"] = str(snapshot)
            signatures.extend(encoder_signatures)
        return config, tuple(signatures)
    except Exception as error:
        raise V21ModelUnavailable("V2.3 model/configuration is missing or unverified") from error


def validate_traditional(model) -> None:
    if (not isinstance(model, HashedTextClassifier) or not isinstance(model.representation, StreamingHashTfidf)
            or model.representation.dimensions != 65536 or model.representation.documents_seen_ != 364761
            or not isinstance(model.projection, SelectKBest) or model.projection.k != 60000
            or not isinstance(model.classifier, LinearSVC) or set(model.classifier.classes_) != set(CLASSES)
            or model.classifier.coef_.shape != (3, 60000)
            or np.asarray(model.representation.idf_).shape != (131072,)
            or not np.isfinite(model.representation.idf_).all()):
        raise V21ModelUnavailable("V2.3 TF-IDF feature/class contract is invalid")
    check_is_fitted(model.classifier);check_is_fitted(model.projection)


class V23Inference(V21DevelopmentInference):
    """Reuse established MIME/language/security/risk behavior with a new ML head."""
    def __init__(self, model, config, encoder=None, scaler=None):
        self.maximum_characters = 4096
        self.runtime_config = config
        self.encoder, self.scaler = encoder, scaler
        self.encoder_lock = threading.Lock()
        self.artifact_provenance = {"inference_mode": config["mode"], "candidate_id": config["candidate"],
            "artifact_sha256": config["files"]["artifact"]["sha256"],
            "training_config_sha256": config["files"]["training_config"]["sha256"],
            "validation_metrics_sha256": config["files"]["metrics"]["sha256"]}
        if encoder is not None:
            self.artifact_provenance.update({"encoder_model_id": config["verified_encoder"]["encoder"]["model_id"],
                "encoder_revision": config["verified_encoder"]["encoder"]["revision"]})
        EmailSecurityInference.__init__(self, model=model, metadata={"model_version": config["model_version"],
            "class_names": CLASSES, "training_languages": ["en", "vi"], "limitations": [
                "V2.3 Research Preview: Development Validation only; no independent Final Test.",
                "Vietnamese evaluation is translated; Chinese three-class accuracy is unvalidated.",
                "Weak/unlabeled source records are not Verified reference labels.",
                "Scores are uncalibrated; calibrated confidence is unavailable.",
                "Chinese-specific security rules are unavailable."]})

    def _validate_model_contract(self):
        if self.runtime_config["mode"] == V23_ML_MODE:validate_traditional(self.model)

    def _predict(self, model_input):
        frame = _bounded_model_input(model_input, 4096)
        if self.encoder is None:
            prediction = str(self.model.predict(frame)[0])
        else:
            import torch
            with self.encoder_lock:
                values = self.encoder.encode(compose_email_texts(frame).tolist(), batch_size=1, device="cpu",
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (1, 384) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=.002):
                raise V21ModelUnavailable("V2.3 invalid embedding")
            mean, scale = self.scaler
            values = values.copy()
            values -= mean
            values /= scale
            with torch.inference_mode():
                logits, _ = self.model(torch.from_numpy(values))
                scores = logits.numpy().copy()
            if scores.shape != (1, 3) or not np.isfinite(scores).all():raise V21ModelUnavailable("V2.3 invalid output")
            scores[:, 0] += self.runtime_config["saved_candidate"]["normal_logit_offset"]
            prediction = CLASSES[int(scores.argmax(axis=1)[0])]
        if prediction not in CLASSES:raise V21ModelUnavailable("V2.3 invalid class")
        return prediction, None, {}

    def analyze_email(self, **kwargs):
        result = super().analyze_email(**kwargs)
        result["model_name"] = "VietMailGuard V2.3 — " + ("TF-IDF + Chi-Square + LinearSVC" if self.encoder is None else "Large-scale MiniLM + MLP")
        result["model_scope"] = "v23_research_preview"
        result["model_explanation"]["limitation"] = "Không có giải thích đóng góp từ khóa cho hashing/neural head; không phải xác suất."
        result["preprocessing"]["composer"] = "EmailTextComposer"
        if self.encoder is not None:
            result["preprocessing"].update({"max_sequence_length":128,"normalize_embeddings":True,"translated":False})
            result["limitations"].append("MiniLM input is truncated to 128 tokens; original full text is stored and security-analyzed.")
        return result


@lru_cache(maxsize=2)
def _load(config_json: str, signatures: tuple, security_hashes: tuple):
    config = json.loads(config_json)
    artifact = next(s[0] for s in signatures if Path(s[0]).as_posix().endswith(config["files"]["artifact"]["path"]))
    if sha256_file(Path(artifact)) != config["files"]["artifact"]["sha256"]:raise V21ModelUnavailable("V2.3 changed artifact")
    if config["mode"] == V23_ML_MODE:
        model = joblib.load(artifact);validate_traditional(model)
        return V23Inference(model, config)
    import torch
    from .classifier_expansion import ExpansionHead
    from vietmailguard.v21.multilingual_runtime import _local_encoder
    # Locally pinned and hash-verified research checkpoint contains NumPy scaler arrays.
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    protocol_path = Path(artifact).parents[3]/config["files"]["protocol"]["path"]
    if (payload["classes"] != CLASSES or payload["protocol_sha256"] != sha256_file(protocol_path)
            or payload["encoder_config_sha256"] != config["files"]["encoder_config"]["sha256"]
            or payload["normal_logit_offset"] != config["saved_candidate"]["normal_logit_offset"]
            or payload["max_sequence_length"] != 128 or payload["max_input_characters_per_field"] != 4096
            or payload["source_parameter_update_rows"] != 357013 or payload["unique_verified_rows"] != 7748):
        raise V21ModelUnavailable("V2.3 checkpoint provenance mismatch")
    mean, scale = np.asarray(payload["scaler_mean"]), np.asarray(payload["scaler_scale"])
    if (mean.shape != (384,) or scale.shape != (384,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):raise V21ModelUnavailable("V2.3 scaler invalid")
    model = ExpansionHead();model.load_state_dict(payload["model"], strict=True);model.eval();model.requires_grad_(False)
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise V21ModelUnavailable("V2.3 weights invalid")
    encoder = _local_encoder(config["encoder_snapshot"], json.dumps(config["verified_encoder"]["encoder"],sort_keys=True), signatures)
    return V23Inference(model, config, encoder, (mean, scale))


def load_v23_engine(mode: str = V23_ML_MODE, root: Path = PROJECT_ROOT):
    config, signatures = verified_configuration(mode, root)
    try:
        return _load(json.dumps(config, sort_keys=True), signatures, tuple(
            sha256_file(root/"config"/name) for name in ("risk_rules_v2.json","security_rules_v2.json")))
    except Exception as error:
        raise V21ModelUnavailable("V2.3 model failed to load; no fallback prediction") from error


def model_availability(mode: str = V23_ML_MODE, root: Path = PROJECT_ROOT):
    try:
        # Neural encoder is allocated only after an explicit Analyze/import action.
        if mode == V23_ML_MODE:load_v23_engine(mode, root)
        else:verified_configuration(mode, root)
        return True, "V2.3 Research Preview artifact verified"
    except Exception:
        return False, "V2.3 artifact/configuration/encoder missing or unverified"


def analyze_v23_email(**fields):
    return load_v23_engine(V23_ML_MODE).analyze_email(**fields)


def analyze_v23_large_email(**fields):
    return load_v23_engine(V23_LARGE_MODE).analyze_email(**fields)
