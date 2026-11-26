"""Train V2.3 Development candidates, with truthful per-component fit counts.

No final evaluation, pseudo ground truth, production routing, or cloud APIs.
The large unlabeled corpus learns IDF, not invented classifier targets.
"""
from __future__ import annotations
import csv
import gc
import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
import warnings
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from vietmailguard.modeling import EmailTextComposer, runtime_versions
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary, make_classifier
from vietmailguard.v23.dataset import digest
from vietmailguard.v23.modeling import StreamingHashTfidf, HashedTextClassifier, bounded
from vietmailguard.v23.train_guard import reference_rows

AREA = ROOT / "results/v23_multilingual/development_01"
MODELS = ROOT / "models/v23_multilingual/development_01"
LABELS = ["normal", "spam", "phishing"]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def metrics(truth, prediction, languages):
    report = classification_report(truth, prediction, labels=LABELS, output_dict=True, zero_division=0)
    result = {"accuracy": float(accuracy_score(truth, prediction)),
              "macro_f1": float(f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)),
              "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
              "per_class": {key: report[key] for key in LABELS},
              "confusion_matrix_order": LABELS, "confusion_matrix": confusion_matrix(truth, prediction, labels=LABELS).tolist(),
              "normal_to_spam": int(np.sum((truth == "normal") & (prediction == "spam"))),
              "normal_to_phishing": int(np.sum((truth == "normal") & (prediction == "phishing"))),
              "by_language": {}}
    for language in ["en", "vi", "zh"]:
        selected = np.asarray(languages) == language
        if not selected.any():
            result["by_language"][language] = {"rows": 0, "metrics": None}
            continue
        by_class = {}
        for key in LABELS:
            support = int(np.sum(np.asarray(truth)[selected] == key))
            detail = classification_report(truth[selected], prediction[selected], labels=LABELS, output_dict=True, zero_division=0)[key]
            by_class[key] = detail if support else {"support": 0, "precision": None, "recall": None, "f1-score": None}
        result["by_language"][language] = {"rows": int(selected.sum()), "per_class": by_class,
                                         "language_origin": "inherited metadata, including translated VI; not a native-language benchmark"}
    return result


def main():
    csv.field_size_limit(100_000_000)
    config_path = ROOT / "config/v23/development_training.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha = digest(config_path)
    AREA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect((ROOT / "data/v23/intake.sqlite3").resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA cache_size=-32768")
    if not connection.execute("SELECT 1 FROM cau_hinh WHERE ma='development_guard_complete' AND gia_tri='1'").fetchone():
        raise RuntimeError("Train-side leakage audit must finish before training")
    selection = """FROM thu t WHERE NOT EXISTS(SELECT 1 FROM tham_chieu r WHERE r.id_thu=t.id_thu)
        AND NOT EXISTS(SELECT 1 FROM cach_ly q WHERE q.id_thu=t.id_thu)"""
    expanded = connection.execute("SELECT COUNT(*) " + selection).fetchone()[0]
    reference = reference_rows(ROOT)
    frames = {s: pd.DataFrame(reference[s]) for s in ("train", "validation")}
    for frame in frames.values():
        for field in ("subject", "body"):
            frame[field] = frame[field].astype(str).str.slice(0, 4096)
    if expanded + sum(map(len, frames.values())) < config["minimum_unique_used_in_learning_and_validation"]:
        raise RuntimeError("Fewer than 300000 unique qualified experiment inputs; expand sources first")
    ids_hash = hashlib.sha256()
    for (identifier,) in connection.execute("SELECT t.id_thu " + selection + " ORDER BY t.id_thu"):
        ids_hash.update(int(identifier).to_bytes(8, "little"))
    lock = {"scope": config["scope"], "config_sha256": config_sha, "reference_sha256": reference["source_hashes"],
            "train_only_expansion_id_sha256": ids_hash.hexdigest(), "train_only_expansion_rows": expanded,
            "verified_train_rows": len(frames["train"]), "verified_validation_rows": len(frames["validation"]),
            "unique_used_in_learning_and_validation": expanded + sum(map(len, frames.values())),
            "classifier_supervised_fit_rows": len(frames["train"]), "pseudo_label_rows": 0,
            "final_test_rows": 0, "final_test_ready": False, "runtime_versions": runtime_versions()}
    lock_path = AREA / "protocol.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Frozen experiment protocol differs")
    write(lock_path, lock)
    state_path = AREA / "metrics.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {**lock, "candidates": []}
    done = {row["candidate"] for row in state["candidates"]}
    y_train = frames["train"].label.to_numpy()
    y_validation = frames["validation"].label.to_numpy()
    texts = {key: EmailTextComposer().transform(bounded(frame)) for key, frame in frames.items()}

    def save(name, artifact, predictions, fit_seconds, prediction_seconds, extra):
        path = MODELS / f"{name}.joblib"
        if path.exists():
            raise ValueError("Refusing to overwrite completed candidate")
        joblib.dump(artifact, path)
        row = {"candidate": name, "artifact": path.relative_to(ROOT).as_posix(), "artifact_sha256": digest(path),
               "artifact_bytes": path.stat().st_size, "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
               "metrics": metrics(y_validation, np.asarray(predictions), frames["validation"].language.to_numpy()),
               "calibrated_probability_available": False, **extra}
        state["candidates"].append(row)
        write(state_path, state)
        write(AREA / f"{name}_confusion_matrix.json", {"order": LABELS, "matrix": row["metrics"]["confusion_matrix"]})
        print(json.dumps({"trained": name, "validation_macro_f1": row["metrics"]["macro_f1"], "supervised_rows": len(y_train)}), flush=True)

    baseline = StreamingHashTfidf(config["hash_dimensions_per_analyzer"])
    for start in range(0, len(texts["train"]), config["batch_rows"]):
        baseline.partial_fit(texts["train"][start:start+config["batch_rows"]])
    if "supervised_hash_tfidf_svc" not in done:
        begin = time.perf_counter()
        model = LinearSVC(random_state=config["seed"], **config["linear_svc"]).fit(baseline.transform(texts["train"]), y_train)
        elapsed = time.perf_counter()-begin
        artifact = HashedTextClassifier(baseline, model)
        begin = time.perf_counter();predicted = artifact.predict(frames["validation"]);latency = time.perf_counter()-begin
        save("supervised_hash_tfidf_svc", artifact, predicted, elapsed, latency,
             {"idf_fit_rows": baseline.documents_seen_, "timing_includes_encoder": False, "prediction_scope": "bounded composer + hash TF-IDF + classifier"})
    checkpoint = AREA / "idf_checkpoint.joblib"
    if checkpoint.exists():
        cached = joblib.load(checkpoint)
        if cached["protocol_sha256"] != digest(lock_path):
            raise ValueError("IDF checkpoint protocol differs")
        large, last, fit_elapsed = cached["transformer"], cached["last_id"], cached["fit_seconds"]
    else:
        large = baseline
        last, fit_elapsed = 0, 0.0
    processed = 0
    cursor = connection.execute("SELECT t.id_thu,t.van_ban_nen " + selection + " AND t.id_thu>? ORDER BY t.id_thu", (last,))
    begin = time.perf_counter()
    while batch := cursor.fetchmany(config["batch_rows"]):
        records = [json.loads(zlib.decompress(blob)) for _, blob in batch]
        large.partial_fit(EmailTextComposer().transform(bounded(pd.DataFrame(records))))
        last = batch[-1][0];processed += len(batch)
        if processed % 5120 == 0:
            elapsed = fit_elapsed + time.perf_counter()-begin
            temporary = checkpoint.with_suffix(".partial")
            joblib.dump({"protocol_sha256": digest(lock_path), "transformer": large, "last_id": last, "fit_seconds": elapsed}, temporary)
            temporary.replace(checkpoint)
            print(json.dumps({"stage": "learning_large_scale_idf", "documents_actually_fitted": large.documents_seen_}), flush=True)
    fit_elapsed += time.perf_counter()-begin
    if large.documents_seen_ != len(y_train) + expanded:
        raise ValueError("Actual representation fit count does not reconcile")
    # Freeze the complete representation, including the final partial batch.
    joblib.dump({"protocol_sha256": digest(lock_path), "transformer": large, "last_id": last, "fit_seconds": fit_elapsed}, checkpoint)
    X = large.transform(texts["train"])
    for name in ("large_scale_idf_svc", "large_scale_idf_chi2_svc", "large_scale_idf_svd_svc"):
        if name in done:
            continue
        begin = time.perf_counter()
        projection = (SelectKBest(chi2, k=config["chi2_k"]) if "chi2" in name else
                      TruncatedSVD(config["svd_components"], n_iter=3, random_state=config["seed"]) if "svd" in name else None)
        selected = projection.fit_transform(X, y_train) if projection is not None else X
        model = LinearSVC(random_state=config["seed"], **config["linear_svc"]).fit(selected, y_train)
        elapsed = time.perf_counter()-begin
        artifact = HashedTextClassifier(large, model, projection)
        begin = time.perf_counter();predicted = artifact.predict(frames["validation"]);latency = time.perf_counter()-begin
        save(name, artifact, predicted, elapsed, latency,
             {"idf_fit_rows": large.documents_seen_, "idf_fit_seconds": fit_elapsed,
              "selector_or_projection_fit_rows": len(y_train), "classifier_supervised_fit_rows": len(y_train),
              "feature_dimensions_before": 2*config["hash_dimensions_per_analyzer"],
              "feature_dimensions_after": int(selected.shape[1]), "prediction_scope": "bounded composer + hash TF-IDF + projection + classifier"})
        del selected, artifact, model;gc.collect()
    del X, large, baseline;gc.collect()
    if any(name not in done for name in config["candidate_order"] if name.startswith("minilm")):
        # Reuse the existing frozen encoder cache by ID/content identity. The
        # imported helper only reads V2.1 caches; no old experiment is executed.
        spec = importlib.util.spec_from_file_location("v23_frozen_cache_reader", ROOT / "scripts/v22/run_group_aware_pilot.py")
        helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        matrices, cache_proof = helper.selected_cache(frames)
        write(AREA / "encoder_cache_proof.json", cache_proof)
        dl = json.loads((ROOT / "config/v21/experiments/multilingual_dl_01.json").read_text(encoding="utf-8"))
        for name in ("minilm_mlp_supervised", "minilm_pca_svc_supervised", "minilm_lda_svc_supervised"):
            if name in done:
                continue
            begin = time.perf_counter()
            if "mlp" in name:
                model = make_classifier("C_minilm_mlp", dl)
                model.fit(matrices["train"], y_train, classifier__sample_weight=compute_sample_weight("balanced", y_train))
            else:
                reduction = (PCA(config["pca_components"], random_state=config["seed"])
                             if "pca" in name else LinearDiscriminantAnalysis(n_components=2))
                model = Pipeline([("reduction_train_only", reduction),
                    ("classifier", LinearSVC(random_state=config["seed"], **config["linear_svc"]))])
                model.fit(matrices["train"], y_train)
            elapsed = time.perf_counter()-begin
            begin = time.perf_counter();predicted = model.predict(matrices["validation"]);latency = time.perf_counter()-begin
            save(name, model, predicted, elapsed, latency,
                 {"supervised_fit_rows": len(y_train), "encoder_frozen": True, "encoder_dimension": 384,
                  "expansion_rows_used_by_this_candidate": 0, "prediction_scope": "head/projection only on verified cached embeddings; excludes encoder latency"})
    state["status"] = "DEVELOPMENT_COMPLETE_NOT_FINAL_EVALUATED"
    state["selected_candidate"] = max(state["candidates"], key=lambda row: (row["metrics"]["macro_f1"], row["metrics"]["per_class"]["phishing"]["recall"]))["candidate"]
    state["production_integration_allowed"] = False
    write(state_path, state)
    connection.close()


if __name__ == "__main__":
    with offline_boundary(ROOT), threadpool_limits(limits=2):
        main()
