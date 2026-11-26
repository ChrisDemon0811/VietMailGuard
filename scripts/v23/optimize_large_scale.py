"""Select a finite large-scale neural protocol using group-aware Train OOF."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import unicodedata
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from vietmailguard.modeling import EmailTextComposer
from vietmailguard.v21.dedup_audit import dedup_hashes
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary
from vietmailguard.v23.dataset import digest
from vietmailguard.v23.modeling import bounded
from vietmailguard.v23.train_guard import ReferenceJoin
from vietmailguard.v23.classifier_expansion import atomic_json, ResourceMonitor, memory_bytes
from vietmailguard.v23.optimization import (
    LABELS, grouped_folds, metric_values, choose_on_oof, offset_predictions, logits_from, train_recipe, reference_fold_bits,
    fingerprint_fold_map)

NAME = "large_scale_optimization_03"
DATA = ROOT/"data/v23"/NAME
RESULTS = ROOT/"results/v23_multilingual"/NAME
MODELS = ROOT/"models/v23_multilingual"/NAME
BASE = ROOT/"data/v23/classifier_expansion_02"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def references(split):
    csv.field_size_limit(100_000_000)
    protocol = read(ROOT/"results/v23_multilingual/development_01/protocol.json")
    path = ROOT/"data/v22/group_aware_supervised_pilot_01/cohort"/(split+".csv")
    assert digest(path) == protocol["reference_sha256"][split]
    frame = pd.read_csv(path, keep_default_na=False)
    spec = importlib.util.spec_from_file_location("optimization_cache", ROOT/"scripts/v22/run_group_aware_pilot.py")
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    matrix, receipt = helper.selected_cache({split: frame.assign(subject=bounded(frame).subject, body=bounded(frame).body)})
    return frame, matrix[split], receipt


class ScaledMatrix:
    def __init__(self, matrix, scaler):self.matrix, self.scaler = matrix, scaler
    def __getitem__(self, index):return self.scaler.transform(np.asarray(self.matrix[index]).copy()).astype(np.float32)


def protocol(config_path):
    for directory in (DATA, RESULTS, MODELS):directory.mkdir(parents=True, exist_ok=True)
    paths = [config_path, Path(__file__), ROOT/"src/vietmailguard/v23/optimization.py"]
    payload = {"source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
               "preserved_v23_02_manifest_sha256": digest(ROOT/"results/v23_multilingual/classifier_expansion_02/frozen_manifest.json"),
               "preserved_v23_01_manifest_sha256": digest(ROOT/"results/v23_multilingual/development_01/frozen_manifest.json"),
               "embedding_sha256": read(BASE/"embeddings.json")["matrix_sha256"],
               "intake_sha256": read(ROOT/"results/v23_multilingual/classifier_expansion_02/protocol.json")["intake_sha256"],
               "outer_validation_used_for_selection": False, "final_test_allowed": False}
    path = RESULTS/"protocol.json"
    if path.exists():assert read(path) == payload, "Locked optimization protocol/source changed"
    else:
        for manifest_path in [ROOT/"results/v23_multilingual/classifier_expansion_02/frozen_manifest.json",
                              ROOT/"results/v23_multilingual/development_01/frozen_manifest.json"]:
            for name, record in read(manifest_path)["files"].items():
                assert digest(ROOT/name) == record["sha256"], "Baseline changed: "+name
        for name, expected in read(ROOT/"results/v23_multilingual/preserved_before.json").items():
            assert digest(ROOT/name) == expected, "Old version artifact changed: "+name
        atomic_json(path, payload)
    assert digest(BASE/"embeddings.npy") == payload["embedding_sha256"]
    assert digest(ROOT/"data/v23/intake.sqlite3") == payload["intake_sha256"]
    return digest(path)


def source_data():
    ids = np.load(BASE/"ids.npy", allow_pickle=False)
    assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() == read(
        ROOT/"results/v23_multilingual/classifier_expansion_02/protocol.json")["expansion_id_sha256"]
    matrix = np.load(BASE/"embeddings.npy", mmap_mode="r", allow_pickle=False)
    db = sqlite3.connect((ROOT/"data/v23/intake.sqlite3").resolve().as_uri()+"?mode=ro&immutable=1", uri=True)
    sources = dict(db.execute("SELECT id_thu,nguon FROM thu"))
    assert all(sources[int(i)] in {"enron_original", "bruce_guenter_2004", "bruce_guenter_2005"} for i in ids)
    codes = np.asarray([1 if sources[int(i)] == "bruce_guenter_2004" else 2 if sources[int(i)] == "bruce_guenter_2005" else 0 for i in ids], dtype=np.int8)
    assert len(ids) == len(matrix) == 357013
    return db, ids, matrix, codes


def prepare(config, protocol_sha):
    frame, features, _ = references("train")
    labels = np.asarray([LABELS.index(x) for x in frame.label], dtype=np.int64)
    folds = grouped_folds(labels, frame.experiment_group_id.to_numpy(), **{
        "n_splits": config["internal_cv"]["folds"], "seed": config["seed"]})
    assignments = np.empty(len(frame), dtype=np.int8)
    for i, (_, val) in enumerate(folds):assignments[val] = i
    np.save(DATA/"verified_fold.npy", assignments, allow_pickle=False)
    with (DATA/"fold_assignment.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream);writer.writerow(["id", "experiment_group_id", "fold"])
        writer.writerows(zip(frame.id, frame.experiment_group_id, map(int, assignments)))
    if (RESULTS/"cv_source_guard.json").exists():
        assert read(RESULTS/"cv_source_guard.json")["protocol_sha256"] == protocol_sha
        return
    db, ids, expanded, _ = source_data()
    bad_path = DATA/"cv_source_fold_bits.npy"
    bad = np.load(bad_path, allow_pickle=False) if bad_path.exists() else np.zeros(len(ids), dtype=np.uint8)
    progress_path = DATA/"fold_guard_progress.json"
    progress = read(progress_path) if progress_path.exists() else {"text_next": 0, "cosine_next": 0}
    full = (frame.subject+"\n"+frame.body).tolist()
    body_hashes = [hashlib.sha256(" ".join(unicodedata.normalize("NFKC", body).casefold().split()).encode()).hexdigest()
                   for body in frame.body]
    template_hashes = [dedup_hashes(subject, body)[2] for subject, body in zip(frame.subject, frame.body)]
    bodies = fingerprint_fold_map(body_hashes, assignments)
    templates = fingerprint_fold_map(template_hashes, assignments)
    joins = ReferenceJoin(full, config["cv_supplement_guard"]["full_word_trigram_jaccard"])
    positions = {int(identifier): i for i, identifier in enumerate(ids)}
    cursor = db.execute("SELECT id_thu,ma_bam_noi_dung,ma_bam_mau,van_ban_nen FROM thu WHERE id_thu>? ORDER BY id_thu",
                        (int(ids[progress["text_next"]-1]) if progress["text_next"] else 0,))
    checked = progress["text_next"]
    with ResourceMonitor() as monitor:
        while batch := cursor.fetchmany(128):
            rows = [r for r in batch if r[0] in positions and positions[r[0]] >= progress["text_next"]]
            if not rows:continue
            texts = []
            for row in rows:
                value = json.loads(zlib.decompress(row[3]));texts.append(value["subject"]+"\n"+value["body"])
            matching = reference_fold_bits(joins, texts, assignments)
            for index, row in enumerate(rows):
                position = positions[row[0]]
                bad[position] |= bodies.get(row[1], 0) | templates.get(row[2], 0) | int(matching[index])
                checked = position+1
            if checked-progress["text_next"] >= 4096 or checked == len(ids):
                np.save(bad_path, bad, allow_pickle=False)
                progress["text_next"] = checked;atomic_json(progress_path, progress)
                print(json.dumps({"stage": "internal_cv_text_guard", "checked": checked, "source_rows": len(ids),
                                  "related_to_any_fold_so_far": int((bad != 0).sum())}), flush=True)
    del joins, cursor, full;gc.collect()
    device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 512*2**20 else "cpu"
    with ResourceMonitor() as cosine_monitor:
        reference = torch.tensor(features, device=device)
        torch.backends.cuda.matmul.allow_tf32 = False
        for start in range(progress["cosine_next"], len(ids), 512):
            stop = min(start+512, len(ids))
            value = torch.tensor(np.asarray(expanded[start:stop]).copy(), device=device)
            for fold in range(3):
                maxima = (value @ reference[assignments == fold].T).max(dim=1).values.cpu().numpy()
                bad[start:stop] |= ((maxima >= config["cv_supplement_guard"]["embedding_cosine"])*(1 << fold)).astype(np.uint8)
            progress["cosine_next"] = stop
            if stop % 4096 == 0 or stop == len(ids):
                np.save(bad_path, bad, allow_pickle=False);atomic_json(progress_path, progress)
        del reference, value
        if device == "cuda":torch.cuda.empty_cache()
    assert progress["text_next"] == len(ids) and progress["cosine_next"] == len(ids)
    atomic_json(RESULTS/"cv_source_guard.json", {"protocol_sha256": protocol_sha, "source_rows": len(ids),
        "per_fold_source_rows_retained": [int(((bad & (1 << fold)) == 0).sum()) for fold in range(3)],
        "related_to_any_fold": int((bad != 0).sum()),
        "exclusion_mask_sha256": digest(bad_path), "verified_assignment_sha256": digest(DATA/"fold_assignment.csv"),
        "scope": "conservative CV split constraint, not verified duplicate identity or a change to original corpus",
        "text_resources": monitor.result, "cosine_resources": cosine_monitor.result,
        "outer_validation_opened": False, "final_training_pool_rows": len(ids)})
    db.close()


def cross_validate(config, protocol_sha):
    if (RESULTS/"selection.json").exists():
        assert read(RESULTS/"selection.json")["protocol_sha256"] == protocol_sha
        return
    frame, matrix, _ = references("train")
    labels = np.asarray([LABELS.index(x) for x in frame.label], dtype=np.int64)
    folds = grouped_folds(labels, frame.experiment_group_id.to_numpy(), n_splits=3, seed=config["seed"])
    db, ids, expansion, codes = source_data();db.close()
    assert digest(DATA/"cv_source_fold_bits.npy") == read(RESULTS/"cv_source_guard.json")["exclusion_mask_sha256"]
    related_folds = np.load(DATA/"cv_source_fold_bits.npy", allow_pickle=False)
    oof = {recipe["name"]: np.full((len(frame), 3), np.nan, dtype=np.float32) for recipe in config["recipes"]}
    receipts = []
    for fold, (fit, holdout) in enumerate(folds):
        raw_indices = np.flatnonzero((related_folds & (1 << fold)) == 0)
        scaler = StandardScaler().fit(matrix[fit])
        train = scaler.transform(matrix[fit]).astype(np.float32)
        validation = scaler.transform(matrix[holdout]).astype(np.float32)
        warm = None
        for recipe in config["recipes"]:
            name = f"cv{fold}_{recipe['name']}"
            path = MODELS/(name+".pt")
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                assert saved["protocol_sha256"] == protocol_sha
                from vietmailguard.v23.classifier_expansion import ExpansionHead
                model = ExpansionHead(384, tuple(config["hidden"]));model.load_state_dict(saved["model"])
            else:
                model, saved = train_recipe(train, labels[fit], ScaledMatrix(expansion, scaler), raw_indices, codes,
                    recipe, config, path, protocol_sha, warm,
                    {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "classes": LABELS,
                     "normal_logit_offset": 0., "calibrated_probability": False})
            if warm is None:warm = saved["warm_state"]
            oof[recipe["name"]][holdout] = logits_from(model, validation)
            receipts.append({"name": name, "fold": fold, "recipe": recipe["name"], "train_rows": len(fit), "holdout_rows": len(holdout),
                             "artifact": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "stages": saved["stages"],
                             "source_parameter_update_rows": saved["source_parameter_update_rows"]})
            atomic_json(RESULTS/"cv_training.json", {"protocol_sha256": protocol_sha, "runs": receipts})
            print(json.dumps({"stage": "internal_cv_fit", "fold": fold, "recipe": recipe["name"],
                              "source_parameter_update_rows": saved["source_parameter_update_rows"]}), flush=True)
    assert all(np.isfinite(value).all() for value in oof.values())
    for name, value in oof.items():np.save(DATA/(name+"_oof_logits.npy"), value, allow_pickle=False)
    selected, grid = choose_on_oof(oof, labels, config["decision_normal_logit_offsets"])
    control, control_grid = choose_on_oof(oof, labels, config["decision_normal_logit_offsets"], include_control=True)
    atomic_json(RESULTS/"selection.json", {"protocol_sha256": protocol_sha, "selected": selected, "control": control,
        "all_predeclared_train_oof_comparisons": grid+control_grid,
        "outer_validation_used": False, "probability_calibrated": False,
        "oof_estimates_are_selection_estimates_not_independent_test_metrics": True})
    print(json.dumps({"stage": "selection_locked", "selected": selected["recipe"], "normal_offset": selected["normal_logit_offset"],
                      "train_oof_macro_f1": selected["metrics"]["macro_f1"]}), flush=True)


def final_fit_and_evaluate(config, protocol_sha):
    if (RESULTS/"metrics.json").exists():return
    selection = read(RESULTS/"selection.json")
    assert selection["protocol_sha256"] == protocol_sha and not selection["outer_validation_used"]
    frame, matrix, _ = references("train")
    labels = np.asarray([LABELS.index(x) for x in frame.label], dtype=np.int64)
    db, ids, expansion, codes = source_data();db.close()
    scaler = StandardScaler().fit(matrix)
    train = scaler.transform(matrix).astype(np.float32)
    fits = [];warm = None
    for chosen in [selection["control"], selection["selected"]]:
        recipe = next(r for r in config["recipes"] if r["name"] == chosen["recipe"])
        path = MODELS/("final_"+recipe["name"]+".pt")
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            assert saved["protocol_sha256"] == protocol_sha
        else:
            _, saved = train_recipe(train, labels, ScaledMatrix(expansion, scaler), np.arange(len(ids)), codes, recipe,
                config, path, protocol_sha, warm,
                {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "classes": LABELS,
                 "normal_logit_offset": chosen["normal_logit_offset"], "calibrated_probability": False,
                 "encoder_reference_config": "config/v21/runtime_multilingual.json",
                 "encoder_config_sha256": digest(ROOT/"config/v21/runtime_multilingual.json"),
                 "max_input_characters_per_field": 4096, "max_sequence_length": 128})
        if warm is None:warm = saved["warm_state"]
        if recipe["name"] != "verified_control":
            assert saved["source_parameter_update_rows"] + len(frame) >= config["minimum_final_parameter_update_rows"]
        fits.append({"name": recipe["name"], "artifact": path.relative_to(ROOT).as_posix(), "artifact_sha256": digest(path),
                     "artifact_bytes": path.stat().st_size, "normal_logit_offset": chosen["normal_logit_offset"],
                     "supervised_verified_rows": len(frame), "source_parameter_update_rows": saved["source_parameter_update_rows"],
                     "total_unique_train_parameter_update_rows": len(frame)+saved["source_parameter_update_rows"],
                     "stages": saved["stages"], "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                     "source_trunk_change_from_warmup": saved["source_trunk_change_from_warmup"],
                     "final_trunk_change_from_warmup": saved["final_trunk_change_from_warmup"]})
    atomic_json(RESULTS/"final_fits_before_validation.json", {"protocol_sha256": protocol_sha,
        "selection_sha256": digest(RESULTS/"selection.json"), "candidates": fits})
    # Outer Development Validation is first used here, after CV selection and both final fits.
    validation_frame, validation_matrix, _ = references("validation")
    truth = np.asarray([LABELS.index(x) for x in validation_frame.label], dtype=np.int64)
    predictions = scaler.transform(validation_matrix).astype(np.float32)
    from vietmailguard.v23.classifier_expansion import ExpansionHead
    for row in fits:
        saved = torch.load(ROOT/row["artifact"], map_location="cpu", weights_only=False)
        model = ExpansionHead(384, tuple(config["hidden"]));model.load_state_dict(saved["model"])
        logits = logits_from(model, predictions)
        predicted = offset_predictions(logits, row["normal_logit_offset"])
        row["metrics"] = metric_values(truth, predicted)
        row["metrics"]["by_language"] = {}
        for language in ["en", "vi", "zh"]:
            keep = validation_frame.language.eq(language).to_numpy()
            detail = metric_values(truth[keep], predicted[keep]) if keep.any() else None
            if detail:
                for index, label in enumerate(LABELS):
                    if not np.any(truth[keep] == index):detail["per_class"][label] = {"support": 0, "precision": None, "recall": None, "f1-score": None}
                if len(set(truth[keep])) < 3:detail["macro_f1"] = None
            row["metrics"]["by_language"][language] = detail
        np.save(DATA/(row["name"]+"_outer_predictions.npy"), predicted, allow_pickle=False)
    atomic_json(RESULTS/"metrics.json", {"protocol_sha256": protocol_sha, "selection_sha256": digest(RESULTS/"selection.json"),
        "candidates": fits, "outer_validation_rows": len(truth), "final_test_used": False,
        "selection_used_outer_validation": False, "probability_calibrated": False, "runtime_integration_allowed": False})
    print(json.dumps({"stage": "outer_evaluation_completed", "macro_f1": {r["name"]: r["metrics"]["macro_f1"] for r in fits}}), flush=True)


def main():
    parser = argparse.ArgumentParser();parser.add_argument("--phase", choices=["prepare", "cv", "final", "all"], default="all")
    args = parser.parse_args()
    config_path = ROOT/"config/v23"/(NAME+".json")
    config = read(config_path)
    torch.set_num_threads(config["cpu_threads"]);torch.use_deterministic_algorithms(True)
    protocol_sha = protocol(config_path)
    if args.phase in ("prepare", "all"):prepare(config, protocol_sha)
    if args.phase in ("cv", "all"):cross_validate(config, protocol_sha)
    if args.phase in ("final", "all"):final_fit_and_evaluate(config, protocol_sha)


if __name__ == "__main__":
    with offline_boundary(ROOT):main()
