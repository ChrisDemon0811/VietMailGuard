"""Predeclared matched-control MiniLM head experiments; no runtime changes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from vietmailguard.v21.multilingual_dl_experiment import offline_boundary
from vietmailguard.v23.dataset import digest
from vietmailguard.v23.modeling import bounded
from vietmailguard.v23.train_guard import reference_rows
from vietmailguard.v23.classifier_expansion import (
    ExpansionHead, fit_head, ordered_indices, atomic_json, ResourceMonitor)

NAME = "classifier_expansion_02"
AREA = ROOT / "data/v23" / NAME
RESULTS = ROOT / "results/v23_multilingual" / NAME
MODELS = ROOT / "models/v23_multilingual" / NAME
CLASSES = ["normal", "spam", "phishing"]


class ScaledExpansion:
    def __init__(self, matrix, scaler):
        self.matrix, self.scaler = matrix, scaler

    def __getitem__(self, indices):
        return self.scaler.transform(np.asarray(self.matrix[indices]).copy()).astype(np.float32)


def load_references():
    reference = reference_rows(ROOT)
    frames = {key: pd.DataFrame(reference[key]) for key in ("train", "validation")}
    helper_path = ROOT / "scripts/v22/run_group_aware_pilot.py"
    spec = importlib.util.spec_from_file_location("v23_expansion_cache_reader", helper_path)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    bounded_frames = {key: frame.assign(subject=bounded(frame).subject, body=bounded(frame).body)
                      for key, frame in frames.items()}
    matrices, receipt = helper.selected_cache(bounded_frames)
    return frames, matrices, receipt


def state_save(path, value):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def main(phase):
    config_path = ROOT / "config/v23" / (NAME + ".json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    MODELS.mkdir(parents=True, exist_ok=True)
    lock = json.loads((RESULTS/"protocol.json").read_text(encoding="utf-8"))
    assert lock["config_sha256"] == digest(config_path)
    code_paths = [config_path, Path(__file__), ROOT/"src/vietmailguard/v23/classifier_expansion.py",
                  ROOT/"scripts/v22/run_group_aware_pilot.py"]
    code_hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in code_paths}
    code_lock = RESULTS/"training_code_sha256.json"
    if code_lock.exists():
        assert json.loads(code_lock.read_text(encoding="utf-8")) == code_hashes, "Training code changed after lock"
    else:
        atomic_json(code_lock, code_hashes)
    assert digest(ROOT/"data/v23/intake.sqlite3") == lock["intake_sha256"]
    receipt = json.loads((AREA/"embeddings.json").read_text(encoding="utf-8"))
    assert receipt["matrix_sha256"] == digest(AREA/"embeddings.npy")
    assert receipt["protocol_sha256"] == digest(RESULTS/"protocol.json")
    ids = np.load(AREA/"ids.npy", allow_pickle=False)
    weak = np.load(AREA/"weak_mask.npy", allow_pickle=False)
    assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() == lock["expansion_id_sha256"]
    with sqlite3.connect((ROOT/"data/v23/intake.sqlite3").resolve().as_uri()+"?mode=ro&immutable=1", uri=True) as connection:
        flags = dict(connection.execute("SELECT id_thu,trang_thai_nhan FROM thu"))
        assert all(bool(flag) == (flags[int(identifier)] == "weak_label") for identifier, flag in zip(ids, weak, strict=True))
    expanded = np.load(AREA/"embeddings.npy", mmap_mode="r", allow_pickle=False)
    assert expanded.shape == (config["expansion_rows"], 384) and int(weak.sum()) == lock["weak_label_rows"]
    order = ordered_indices(ids, config["seed"])
    manifest = AREA/"budget_assignment.csv"
    if not manifest.exists():
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["rank", "canonical_id", "label_status", "learning_signal"])
            for rank, index in enumerate(order, 1):
                writer.writerow([rank, int(ids[index]), "weak_label" if weak[index] else "unlabeled",
                                 "uncertain_candidate_set_spam_phishing" if weak[index] else "no_reference_class"])
    budget_plan = {"budgets": [{"rows": n, "weak_label": int(weak[order[:n]].sum()),
                                "unlabeled": int((~weak[order[:n]]).sum()),
                                "ordered_id_sha256": hashlib.sha256(ids[order[:n]].astype('<i8').tobytes()).hexdigest()}
                               for n in config["budgets"]], "assignment_sha256": digest(manifest)}
    plan = RESULTS/"budget_plan.json"
    if plan.exists():
        assert json.loads(plan.read_text(encoding="utf-8")) == budget_plan
    else:
        atomic_json(plan, budget_plan)
    frames, matrices, cache_receipt = load_references()
    reference = reference_rows(ROOT)
    assert reference["source_hashes"] == lock["reference_sha256"]
    assert not (set(frames['train'].experiment_group_id) & set(frames['validation'].experiment_group_id))
    scaler = StandardScaler().fit(matrices["train"])
    train = scaler.transform(matrices["train"]).astype(np.float32)
    labels = np.asarray([CLASSES.index(label) for label in frames["train"].label], dtype=np.int64)
    expansion = ScaledExpansion(expanded, scaler)
    run_path = RESULTS/"training_state.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {
        "protocol_sha256": digest(RESULTS/"protocol.json"), "candidates": [], "validation_evaluated": False}
    assert run["protocol_sha256"] == digest(RESULTS/"protocol.json")
    warm_path = MODELS/"verified_initialization.pt"
    if not warm_path.exists():
        if phase == "evaluate":
            raise ValueError("Evaluation cannot initialize or fit a missing model")
        rng = np.random.default_rng(config["seed"])
        model = ExpansionHead(384, tuple(config["hidden"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=config["warmup_learning_rate"], weight_decay=config["weight_decay"])
        frequencies = np.bincount(labels, minlength=3)
        weights = torch.tensor(len(labels)/(3*frequencies), dtype=torch.float32)
        with ResourceMonitor() as monitor:
            for _ in range(config["warmup_epochs"]):
                permutation = rng.permutation(len(train))
                for start in range(0, len(train), config["batch_rows"]):
                    selection = permutation[start:start+config["batch_rows"]]
                    optimizer.zero_grad(set_to_none=True)
                    prediction, _ = model(torch.from_numpy(train[selection]))
                    loss = F.cross_entropy(prediction, torch.from_numpy(labels[selection]), weight=weights)
                    loss.backward();optimizer.step()
        state_save(warm_path, {"model": model.state_dict(), "classes": CLASSES,
                              "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                              "protocol_sha256": run["protocol_sha256"], "supervised_rows": len(train)})
        atomic_json(RESULTS/"warmup_resources.json", monitor.result)
    initial = torch.load(warm_path, map_location="cpu", weights_only=False)
    assert initial["protocol_sha256"] == run["protocol_sha256"]
    done = {row["name"] for row in run["candidates"]}
    if phase != "evaluate":
        if run["validation_evaluated"]:
            raise ValueError("Validation already evaluated; no additional fitting in this locked run")
        for budget in config["budgets"]:
            for method in config["methods"]:
                name = f"{method}_{budget}"
                if name in done:
                    continue
                path = MODELS/(name + ".pt")
                if path.exists():
                    raise ValueError("Refusing to overwrite completed candidate")
                with ResourceMonitor() as monitor:
                    model, fit_receipt = fit_head(train, labels, expansion, weak, order[:budget], config, method,
                        initial["model"], RESULTS/(name + "_checkpoint.pt"), run["protocol_sha256"])
                assert any(not torch.equal(initial["model"][key], model.state_dict()[key]) for key in initial["model"])
                state_save(path, {"model": model.state_dict(), "classes": CLASSES, "hidden": config["hidden"],
                                  "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                                  "encoder_config_sha256": lock["encoder_config_sha256"],
                                  "protocol_sha256": run["protocol_sha256"], "method": method, "budget": budget,
                                  "calibrated_probability": False})
                row = {"name": name, "method": method, "budget": budget,
                       "artifact": path.relative_to(ROOT).as_posix(), "artifact_sha256": digest(path),
                       "artifact_bytes": path.stat().st_size, "resources": monitor.result, **fit_receipt}
                run["candidates"].append(row)
                atomic_json(run_path, run)
                print(json.dumps({"completed_fit": name, "updated_expansion_rows": fit_receipt["expansion_unique_parameter_update_rows"],
                                  "seconds": monitor.result["seconds"]}), flush=True)
    expected = len(config["budgets"])*len(config["methods"])
    if phase == "train":
        assert len(run["candidates"]) == expected
        return
    if run["validation_evaluated"]:
        print(json.dumps({"stage": "already_evaluated", "candidates": expected}));return
    assert len(run["candidates"]) == expected, "Finish every predeclared fit before evaluating Validation"
    validation = torch.from_numpy(scaler.transform(matrices["validation"]).astype(np.float32))
    truth = frames["validation"].label.to_numpy()
    metrics = []
    for row in run["candidates"]:
        path = ROOT/row["artifact"]
        assert digest(path) == row["artifact_sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = ExpansionHead(384, tuple(config["hidden"]))
        model.load_state_dict(saved["model"]);model.eval()
        started = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(validation)
            prediction = np.asarray(CLASSES)[logits.argmax(dim=1).numpy()]
        elapsed = time.perf_counter()-started
        details = classification_report(truth, prediction, labels=CLASSES, output_dict=True, zero_division=0)
        by_language = {}
        for language in ["en", "vi", "zh"]:
            mask = frames["validation"].language.eq(language).to_numpy()
            if mask.any():
                detail = classification_report(truth[mask], prediction[mask], labels=CLASSES, output_dict=True, zero_division=0)
                by_language[language] = {"rows": int(mask.sum()), "per_class": {
                    label: detail[label] if np.any(truth[mask] == label) else {"support": 0, "precision": None, "recall": None, "f1-score": None}
                    for label in CLASSES}, "native_benchmark_verified": False}
            else:
                by_language[language] = {"rows": 0, "metrics": None}
        item = {**row, "metrics": {"accuracy": float(accuracy_score(truth, prediction)),
                    "macro_f1": float(f1_score(truth, prediction, labels=CLASSES, average="macro")),
                    "weighted_f1": float(f1_score(truth, prediction, average="weighted")),
                    "per_class": {label: details[label] for label in CLASSES},
                    "confusion_matrix": confusion_matrix(truth, prediction, labels=CLASSES).tolist(),
                    "confusion_order": CLASSES, "by_language": by_language},
                "prediction_seconds": elapsed, "prediction_scope": "head and train-only scaling; cached encoder features, not end-to-end",
                "verified_training_rows": 7748, "validation_rows": 1938, "encoder_frozen": True,
                "new_human_approvals": 0, "independent_pseudo_label_precision": None}
        metrics.append(item)
    atomic_json(RESULTS/"metrics.json", {"scope": config["scope"], "protocol_sha256": run["protocol_sha256"],
        "candidates": metrics, "final_test_used": False, "production_integration_allowed": False,
        "all_predeclared_fits_finished_before_validation_evaluation": True, "cache_receipt": cache_receipt})
    run["validation_evaluated"] = True
    atomic_json(run_path, run)
    print(json.dumps({"evaluated_candidates": len(metrics), "macro_f1": {r["name"]: r["metrics"]["macro_f1"] for r in metrics}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "evaluate", "all"], default="all")
    args = parser.parse_args()
    with offline_boundary(ROOT):
        main(args.phase)
