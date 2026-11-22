"""Restricted-local verified-seed pilot; never promotes pseudo-labels or models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
import warnings
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from vietmailguard.embedding_modeling import compose_email_texts
from vietmailguard.modeling import runtime_versions
from vietmailguard.v21.development_training import resource_snapshot
from vietmailguard.v21.multilingual_dl_experiment import (
    _validate_matrix, evaluate, load_encoder, offline_boundary,
)
from vietmailguard.v21.multilingual_runtime import validate_head, verified_configuration
from vietmailguard.v22.expansion import readonly
from vietmailguard.v22.quality import digest


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_seed(config: dict) -> tuple[dict, dict[str, pd.DataFrame]]:
    area = ROOT / config["data"]["directory"]
    manifest = read_json(area / "manifest.json")
    if not manifest["supervised_development_training_ready"] or manifest["final_evaluation_ready"]:
        raise ValueError("Verified development data is not ready")
    for key, name in (("normal", config["data"].get("normal_owner_decision_file", "owner_scope_decision.json")),
                      ("spam", config["data"].get("spam_owner_decision_file", "owner_spam_decision.json"))):
        if digest(area.parent / name) != config["data"][key + "_owner_decision_sha256"]:
            raise ValueError("Owner scope evidence changed")
    frames = {}; owners = {}; identifiers = {}; parents = []
    for split in ("train", "validation"):
        path = area / (split + ".csv")
        if digest(path) != config["data"][split + "_sha256"] or digest(path) != manifest["output_sha256"][split]:
            raise ValueError("Verified subset checksum mismatch")
        records = []
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row["id"] in identifiers or row["split"] != split or not row["label_provenance"]
                        or not row["curation_rule_id"] or row["training_eligible"].lower() != "true"
                        or not row["v22_use_permission_basis"] or not row["v22_owner_decision"]):
                    raise ValueError("Unapproved seed record")
                identifiers[row["id"]] = (split, row["label"], row["final_group_id"])
                for field in ("final_group_id", "content_hash", "normalized_content_hash", "template_hash"):
                    value = row[field]
                    if value and owners.setdefault((field, value), (split, row["label"])) != (split, row["label"]):
                        raise ValueError("Group/fingerprint leakage or label conflict")
                if row["parent_id"]:
                    parents.append((row["id"], row["parent_id"]))
                row["subject"] = row["subject"][:4096]; row["body"] = row["body"][:4096]
                records.append(row)
        frame = pd.DataFrame(records)
        if len(frame) != config["data"]["expected_rows"][split] or set(frame.label) != set(config["data"]["expected_classes"]):
            raise ValueError("Missing a class or wrong seed count")
        frames[split] = frame
    for child, parent in parents:
        if identifiers.get(child) != identifiers.get(parent):
            raise ValueError("Parent/translation split or label conflict")
    return manifest, frames


def selected_embeddings(frame: pd.DataFrame, split: str) -> tuple[np.ndarray, dict]:
    """Index only approved rows from immutable, checksummed frozen-encoder caches."""
    receipt = read_json(ROOT / "results/v21_multilingual/experiments/multilingual_dl_01/embedding_verification.json")
    cache = receipt["cache"][split]
    source = ROOT / "data/v21/splits/development_bootstrap" / (split + ".csv")
    matrix_path = ROOT / cache["matrix"]
    if (digest(source) != receipt["fingerprint"][split + "_sha256"]
            or digest(matrix_path) != cache["sha256"]
            or digest(matrix_path.with_suffix(".json")) != cache["manifest_sha256"]):
        raise ValueError("Embedding cache or ordered source changed")
    wanted = frame.set_index("id"); positions = {}
    with source.open(encoding="utf-8", newline="") as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            if row["id"] not in wanted.index:
                continue
            approved = wanted.loc[row["id"]]
            if (row["content_hash"] != approved.content_hash or row["subject"][:4096] != approved.subject
                    or row["body"][:4096] != approved.body):
                raise ValueError("Cached row differs from approved input")
            positions[row["id"]] = position
    if set(positions) != set(frame.id):
        raise ValueError("Approved embedding rows unavailable")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    _validate_matrix(matrix, cache["rows"], 384)
    selected = np.asarray(matrix[[positions[x] for x in frame.id]], dtype=np.float32)
    _validate_matrix(selected, len(frame), 384)
    return selected, {"source_sha256": digest(source), "matrix_sha256": cache["sha256"],
                      "selected_rows": len(frame), "encoder_fitted": False}


def pilot_samples(config: dict) -> pd.DataFrame:
    area = ROOT / "data/v22/training_unblock"
    scope = read_json(area / "unlabeled_scope_manifest.json")
    if not scope["use_scope_confirmed"]:
        raise ValueError("Unlabeled local use not confirmed")
    membership = ROOT / scope["membership"]
    if digest(membership) != scope["membership_sha256"]:
        raise ValueError("Unlabeled membership changed")
    with (ROOT / "data/v22/phase33/active_review_priority.csv").open(encoding="utf-8", newline="") as handle:
        candidate_ids = {x["audit_id"] for x in csv.DictReader(handle)
                         if x["source"] == "enron_original" and x["language_signal"] == "en"
                         and x["technical_status"] in ("technical_clear_label_pending", "technical_recovered_label_pending")}
    candidates = []
    with membership.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["audit_id"] in candidate_ids:
                candidates.append(row)
    candidates.sort(key=lambda x: x["audit_id"])
    rng = np.random.RandomState(config["seed"]); rng.shuffle(candidates)
    unique = {}; selected = []
    for row in candidates:
        group = row["proposed_constraint_group_id"]
        if group not in unique:
            unique[group] = row["audit_id"]; selected.append(row)
        if len(selected) == config["proposal_pilot"]["maximum_rows"]:
            break
    if len(selected) != config["proposal_pilot"]["maximum_rows"]:
        raise ValueError("Insufficient independent existing review samples")
    records = []
    with readonly(ROOT / "data/v22/phase32/review.sqlite3") as conn:
        for row in selected:
            pack = conn.execute("SELECT noi_dung_nen FROM mau WHERE id_mau=?", (row["audit_id"],)).fetchone()
            if not pack or not pack[0]:
                raise ValueError("Original review content unavailable")
            content = json.loads(zlib.decompress(pack[0]).decode("utf-8"))
            records.append({**row, "subject": content["subject"][:4096], "body": content["body"][:4096]})
    return pd.DataFrame(records)


def run() -> None:
    csv.field_size_limit(100_000_000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/v22/training_unblock_expanded.json")
    config_path = parser.parse_args().config.resolve()
    if not config_path.is_relative_to(ROOT / "config/v22"):
        raise ValueError("Pilot configuration must be inside config/v22")
    config = read_json(config_path)
    if (not config["supervised_fit_allowed"] or any(config[x] for x in
            ("semi_supervised_fit_allowed", "unverified_pool_label_admission_allowed", "final_test_access_allowed", "network_allowed", "gemini_allowed"))):
        raise ValueError("Invalid restricted pilot authority")
    with offline_boundary(ROOT), threadpool_limits(limits=2):
        resources = resource_snapshot()
        if resources["free_ram_gib"] < config["supervised"]["minimum_free_ram_gib"]:
            raise RuntimeError("Insufficient free RAM; no fitting started")
        manifest, frames = load_seed(config)
        matrices = {}; cache_receipts = {}
        for split in ("train", "validation"):
            matrices[split], cache_receipts[split] = selected_embeddings(frames[split], split)
        tfidf_config = read_json(ROOT / config["teachers"]["tfidf"])
        if digest(ROOT / tfidf_config["artifact"]) != tfidf_config["artifact_sha256"]:
            raise ValueError("TF-IDF teacher integrity failed")
        tfidf = joblib.load(ROOT / tfidf_config["artifact"])
        mlp_config, _, snapshot = verified_configuration(ROOT / config["teachers"]["minilm_mlp"], ROOT)
        mlp_artifact = joblib.load(ROOT / mlp_config["artifact"]); validate_head(mlp_artifact, mlp_config)
        teachers = {"tfidf": tfidf, "minilm_mlp": mlp_artifact.classifier}
        result_dir = ROOT / config["outputs"]["results"]; result_dir.mkdir(parents=True, exist_ok=True)
        model_dir = ROOT / config["outputs"]["models"]; model_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = {"config_sha256": digest(config_path), "seed_manifest_sha256": digest(ROOT / config["data"]["directory"] / "manifest.json"),
                       "runner_sha256": digest(Path(__file__))}
        state_path = result_dir / "results.json"
        state = read_json(state_path) if state_path.exists() else {"fingerprint": fingerprint, "resources": resources,
            "versions": runtime_versions(), "cache": cache_receipts, "candidates": [], "evaluation_scope": config["evaluation"]}
        if state["fingerprint"] != fingerprint:
            raise ValueError("Existing results belong to a different locked pilot")
        completed = {x["candidate"] for x in state["candidates"]}
        for name, teacher in teachers.items():
            xtrain = frames["train"] if name == "tfidf" else matrices["train"]
            xval = frames["validation"] if name == "tfidf" else matrices["validation"]
            for role in ("existing_teacher", "new_verified_supervised"):
                candidate = role + "_" + name
                if candidate in completed:
                    continue
                estimator = teacher if role == "existing_teacher" else clone(teacher)
                fit_seconds = None; artifact = None; recorded_warnings = []
                if role == "new_verified_supervised":
                    artifact = model_dir / (candidate + ".joblib")
                    if artifact.exists():
                        raise ValueError("Refusing to overwrite an existing pilot artifact")
                    started = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        if name == "tfidf":
                            estimator.fit(xtrain, frames["train"].label.to_numpy())
                        else:
                            weights = compute_sample_weight("balanced", frames["train"].label.to_numpy())
                            estimator.fit(xtrain, frames["train"].label.to_numpy(), classifier__sample_weight=weights)
                    fit_seconds = time.perf_counter() - started
                    recorded_warnings = [type(x.message).__name__ for x in caught]
                    payload = estimator if name == "tfidf" else {"classifier": estimator,
                        "encoder_config": mlp_config["encoder"], "scope": config["scope"], "calibrated": False}
                    joblib.dump(payload, artifact)
                started = time.perf_counter(); prediction = estimator.predict(xval)
                inference_seconds = time.perf_counter() - started
                state["candidates"].append({"candidate": candidate, "role": role, "metrics": evaluate(frames["validation"], prediction),
                    "fit_seconds": fit_seconds, "prediction_ms_per_email": 1000 * inference_seconds / len(prediction),
                    "encoder_included_in_timing": False, "calibrated_probability_available": False,
                    "artifact": artifact.relative_to(ROOT).as_posix() if artifact else None,
                    "artifact_sha256": digest(artifact) if artifact else None,
                    "artifact_bytes": artifact.stat().st_size if artifact else None, "warnings": recorded_warnings})
                write_json(state_path, state)
                print(json.dumps({"candidate": candidate, "macro_f1": state["candidates"][-1]["metrics"]["macro_f1"],
                                  "fit_seconds": fit_seconds}), flush=True)
        if "proposal_pilot" not in state:
            sample = pilot_samples(config)
            seed_hashes = set(frames["train"].normalized_content_hash) | set(frames["validation"].normalized_content_hash)
            if set(sample.normalized_content_hash) & seed_hashes:
                raise ValueError("Proposal pool overlaps verified development data")
            now = resource_snapshot()
            if now["free_ram_gib"] < config["supervised"]["minimum_free_ram_gib"]:
                state["proposal_pilot"] = {"status": "BLOCKED", "reason": "insufficient free RAM for local encoder", "accepted": 0}
            else:
                import torch
                torch.set_num_threads(2)
                encoder = load_encoder(snapshot, mlp_config["encoder"], "cpu")
                started = time.perf_counter()
                encoded = np.asarray(encoder.encode(compose_email_texts(sample).tolist(), batch_size=8,
                    normalize_embeddings=True, convert_to_numpy=True, device="cpu", show_progress_bar=False), dtype=np.float32)
                encoding_seconds = time.perf_counter() - started
                _validate_matrix(encoded, len(sample), 384)
                svc_labels = tfidf.predict(sample); svc_scores = tfidf.decision_function(sample)
                mlp_labels = mlp_artifact.classifier.predict(encoded)
                softmax = mlp_artifact.classifier.predict_proba(encoded)
                if not np.isfinite(softmax).all() or not np.allclose(softmax.sum(axis=1), 1):
                    raise ValueError("Invalid uncalibrated MLP scores")
                ordered = np.sort(svc_scores, axis=1); gaps = ordered[:, -1] - ordered[:, -2]
                proposal_rows = [{"audit_id": row.audit_id, "proposed_constraint_group_id": row.proposed_constraint_group_id,
                    "normalized_content_hash": row.normalized_content_hash, "source": "enron_original", "language_signal": "en",
                    "language_human_verified": False, "tfidf_suggested_label": str(svc_labels[i]),
                    "minilm_mlp_suggested_label": str(mlp_labels[i]), "model_disagreement": bool(svc_labels[i] != mlp_labels[i]),
                    "svc_margin_gap_uncalibrated": float(gaps[i]), "mlp_softmax_max_uncalibrated": float(softmax[i].max()),
                    "label_status": "pseudo_suggestion_unverified", "ground_truth": "", "training_eligible": False,
                    "review_required": True, "tfidf_artifact_sha256": tfidf_config["artifact_sha256"],
                    "mlp_artifact_sha256": mlp_config["artifact_sha256"]} for i, row in enumerate(sample.itertuples())]
                proposal_rows.sort(key=lambda x: (not x["model_disagreement"], x["svc_margin_gap_uncalibrated"], x["audit_id"]))
                target = ROOT / config["outputs"]["private_proposals"]
                with target.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(proposal_rows[0])); writer.writeheader(); writer.writerows(proposal_rows)
                state["proposal_pilot"] = {"status": "SUGGESTIONS_ONLY_ADMISSION_BLOCKED", "rows": len(sample),
                    "tfidf_labels": dict(Counter(map(str, svc_labels))), "minilm_mlp_labels": dict(Counter(map(str, mlp_labels))),
                    "disagreements": int(np.sum(svc_labels != mlp_labels)), "accepted": 0, "ground_truth_available": False,
                    "precision": None, "encoder_seconds": encoding_seconds, "proposal_sha256": digest(target),
                    "sample_selection_uses_model_scores": False, "source_review_policy_modified": False,
                    "reason": "No independent annotated acceptance sample per source/class/language/origin cell; 64 proposals cannot establish the registered 99% precision lower bound."}
            state["semi_supervised_comparison"] = {"status": "BLOCKED", "fit_performed": False,
                "macro_f1": None, "phishing_recall": None, "reason": "No pseudo-labels passed admission. Supervised metrics are not relabeled as SSL results."}
            write_json(state_path, state)
        write_json(result_dir / "config_snapshot.json", config)
        print(json.dumps({"results": state_path.relative_to(ROOT).as_posix(), "proposal_pilot": state["proposal_pilot"]}), flush=True)


if __name__ == "__main__":
    run()
