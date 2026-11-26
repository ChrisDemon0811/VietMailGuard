"""Train-side selection and bounded auxiliary gradients for an isolated study."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from scipy.sparse import csr_matrix

from .classifier_expansion import ExpansionHead, ResourceMonitor, consistency_loss, update_teacher
from .train_guard import features

LABELS = ["normal", "spam", "phishing"]


def fingerprint_fold_map(fingerprints, folds):
    """An absent fingerprint is not a group shared by unrelated messages."""
    result = {}
    for value, fold in zip(fingerprints, folds, strict=True):
        if value:
            result[value] = result.get(value, 0) | (1 << int(fold))
    return result


def reference_fold_bits(join, texts, reference_folds):
    """Exact same sparse Jaccard test, retaining which holdout was matched."""
    indices, offsets, sizes = [], [0], []
    for text in texts:
        values = features(text)
        sizes.append(len(values))
        indices.extend(sorted(join.vocabulary[x] for x in values if x in join.vocabulary))
        offsets.append(len(indices))
    matrix = csr_matrix((np.ones(len(indices), dtype=np.int32), indices, offsets),
                        shape=(len(texts), len(join.vocabulary)))
    intersections = (matrix @ join.transpose).tocoo()
    union = np.asarray(sizes, dtype=np.int64)[intersections.row]+join.sizes[intersections.col]-intersections.data
    good = (union > 0) & (intersections.data >= join.threshold*union)
    result = np.zeros(len(texts), dtype=np.uint8)
    np.bitwise_or.at(result, intersections.row[good],
                     (1 << np.asarray(reference_folds)[intersections.col[good]]).astype(np.uint8))
    return result


def grouped_folds(labels, groups, n_splits=3, seed=42):
    result = []
    assigned = np.zeros(len(labels), dtype=int)
    for train, validation in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(labels)), labels, groups):
        if set(np.asarray(groups)[train]) & set(np.asarray(groups)[validation]):
            raise ValueError("Internal fold group leakage")
        if set(np.asarray(labels)[train]) != {0, 1, 2} or set(np.asarray(labels)[validation]) != {0, 1, 2}:
            raise ValueError("Internal fold lacks a class")
        assigned[validation] += 1
        result.append((train, validation))
    if not np.all(assigned == 1):
        raise ValueError("OOF assignment is incomplete")
    return result


def metric_values(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    detail = classification_report(truth, prediction, labels=[0, 1, 2], output_dict=True, zero_division=0)
    return {"rows": len(truth), "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
            "normal_false_positives": int(np.sum((truth == 0) & (prediction != 0))),
            "normal_support": int(np.sum(truth == 0)),
            "per_class": {label: detail[str(i)] for i, label in enumerate(LABELS)},
            "confusion_order": LABELS, "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist()}


def offset_predictions(logits, offset):
    adjusted = np.asarray(logits).copy()
    adjusted[:, 0] += offset
    return adjusted.argmax(axis=1)


def choose_on_oof(oof, labels, offsets, include_control=False):
    """Finite predeclared grid; no outer Validation or probability calibration."""
    records = []
    for name, logits in oof.items():
        if (name == "verified_control") != include_control:
            continue
        for offset in offsets:
            metric = metric_values(labels, offset_predictions(logits, offset))
            records.append({"recipe": name, "normal_logit_offset": offset,
                            "normal_fpr_gate_passed": metric["normal_false_positives"] / metric["normal_support"] <= .01,
                            "metrics": metric})
    passing = [row for row in records if row["normal_fpr_gate_passed"]]
    if passing:
        chosen = max(passing, key=lambda row: (row["metrics"]["macro_f1"], -row["metrics"]["normal_false_positives"],
                    row["metrics"]["per_class"]["phishing"]["recall"], -row["normal_logit_offset"]))
    else:
        chosen = max(records, key=lambda row: (-row["metrics"]["normal_false_positives"], row["metrics"]["macro_f1"],
                                              -row["normal_logit_offset"]))
    return chosen, records


def assign_guarded_gradients(parameters, supervised, auxiliary, cap=.2):
    """Project conflicting auxiliary gradient; cap its prediction-path norm.

    The decoder has no supervised gradient. Its gradient is scaled by the same
    factor, but it does not inflate the norm used to protect the classifier.
    This is a local gradient safeguard, not an accuracy guarantee.
    """
    primary = torch.autograd.grad(supervised, parameters, retain_graph=auxiliary is not None, allow_unused=True)
    if auxiliary is None:
        for p, grad in zip(parameters, primary):
            p.grad = grad
        return {"source_reached_optimizer": False, "conflict": False, "scale": 0., "aux_ratio": 0.}
    extra = list(torch.autograd.grad(auxiliary, parameters, allow_unused=True))
    norm2 = sum(float(torch.sum(g.detach()**2)) for g in primary if g is not None)
    dot = sum(float(torch.sum(a.detach()*b.detach())) for a, b in zip(primary, extra) if a is not None and b is not None)
    if dot < 0 and norm2 > 0:
        coefficient = dot / norm2
        for i, (a, b) in enumerate(zip(primary, extra)):
            if a is not None:
                extra[i] = (torch.zeros_like(a) if b is None else b) - coefficient*a
    aux2 = sum(float(torch.sum(b.detach()**2)) for a, b in zip(primary, extra) if a is not None and b is not None)
    scale = min(1., cap * math.sqrt(norm2) / max(math.sqrt(aux2), 1e-30)) if norm2 > 0 else 0.
    reached = False
    for p, a, b in zip(parameters, primary, extra):
        if a is None:
            p.grad = b*scale if b is not None else None
            reached |= bool(b is not None and torch.any(p.grad != 0))
        else:
            p.grad = a if b is None else a + scale*b
            reached |= bool(b is not None and torch.any(p.grad != a))
    return {"source_reached_optimizer": reached, "conflict": dot < 0, "scale": scale,
            "aux_ratio": scale * math.sqrt(aux2) / max(math.sqrt(norm2), 1e-30)}


def logits_from(model, matrix, batch=1024):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(matrix), batch):
            logits, _ = model(torch.from_numpy(np.asarray(matrix[start:start+batch]).copy()))
            outputs.append(logits.numpy())
    return np.vstack(outputs)


def train_recipe(train, labels, expansion, raw_indices, source_codes, recipe, config, checkpoint: Path,
                 protocol_sha256, warm_state=None, preprocessing=None):
    """No holdout arrays are accepted by this training function."""
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    model = ExpansionHead(train.shape[1], tuple(config["hidden"]))
    frequencies = np.bincount(labels, minlength=3)
    class_weights = torch.tensor(len(labels)/(3*frequencies), dtype=torch.float32)
    stages = []
    if warm_state is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["warmup_learning_rate"], weight_decay=config["weight_decay"])
        with ResourceMonitor() as monitor:
            for _ in range(config["warmup_epochs"]):
                for selected in np.array_split(rng.permutation(len(train)), math.ceil(len(train)/config["batch_rows"])):
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = model(torch.from_numpy(train[selected]))
                    loss = F.cross_entropy(logits, torch.from_numpy(labels[selected]), weight=class_weights)
                    loss.backward();optimizer.step()
        stages.append({"stage": "verified_warmup", "unique_verified_rows": len(train), "resources": monitor.result})
        warm_state = copy.deepcopy(model.state_dict())
    else:
        model.load_state_dict(warm_state)
    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    teacher = copy.deepcopy(frozen)
    rng = np.random.default_rng(config["seed"]+101)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["source_learning_rate"], weight_decay=config["weight_decay"])
    position, verified_order = 0, rng.permutation(len(train))
    seen, weak_seen, soft_seen = set(), set(), set()
    stats = {"batches": 0, "conflicts": 0, "scale_sum": 0., "max_aux_ratio": 0.}
    weak_counts = np.bincount(source_codes[raw_indices], minlength=3)
    weak_total = int(weak_counts[1:].sum())
    source_weights = {code: weak_total/(2*int(weak_counts[code])) for code in (1, 2) if weak_counts[code]}
    with ResourceMonitor() as monitor:
        for _ in range(config["source_epochs"]):
            indices = rng.permutation(raw_indices)
            for start in range(0, len(indices), config["batch_rows"]):
                if position >= len(verified_order):
                    position, verified_order = 0, rng.permutation(len(train))
                selected_verified = verified_order[position:position+config["batch_rows"]]
                position += len(selected_verified)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(torch.from_numpy(train[selected_verified]))
                supervised = F.cross_entropy(logits, torch.from_numpy(labels[selected_verified]), weight=class_weights)
                selected_raw = indices[start:start+config["batch_rows"]]
                auxiliary = None
                weak = source_codes[selected_raw] > 0
                if recipe["name"] != "verified_control":
                    source = torch.from_numpy(np.asarray(expansion[selected_raw]).copy())
                    noisy = F.dropout(source, p=config["embedding_dropout"], training=True)
                    noisy_logits, reconstructed = model(noisy)
                    auxiliary = torch.tensor(0.)
                    if recipe["reconstruction"]:
                        auxiliary = auxiliary + recipe["reconstruction"] * F.mse_loss(reconstructed, source)
                    if recipe["weak"] and weak.any():
                        clean_logits, _ = model(source)
                        with torch.no_grad():
                            warm_logits, _ = frozen(source)
                            reliability = (1-warm_logits.softmax(1)[:, 0]).square()
                        partial = torch.logsumexp(clean_logits, 1)-torch.logsumexp(clean_logits[:, 1:], 1)
                        weights = torch.tensor([source_weights.get(int(code), 0.) for code in source_codes[selected_raw]])
                        auxiliary = auxiliary + recipe["weak"]*(partial*reliability*weights).sum()/len(selected_raw)
                    if recipe["consistency"]:
                        with torch.no_grad():
                            target, _ = teacher(source)
                        consistent, _ = consistency_loss(noisy_logits, target)
                        auxiliary = auxiliary + recipe["consistency"]*consistent
                if not torch.isfinite(supervised) or auxiliary is not None and not torch.isfinite(auxiliary):
                    raise ValueError("Nonfinite training objective")
                gradient = assign_guarded_gradients(list(model.parameters()), supervised, auxiliary,
                    config["gradient_guard"]["max_prediction_path_aux_to_verified_norm"])
                optimizer.step()
                update_teacher(teacher, model, config["ema_decay"])
                if gradient["source_reached_optimizer"]:
                    seen.update(map(int, selected_raw))
                    if recipe["weak"]:weak_seen.update(map(int, selected_raw[weak]))
                    if recipe["consistency"]:soft_seen.update(map(int, selected_raw))
                stats["batches"] += 1
                stats["conflicts"] += int(gradient["conflict"])
                stats["scale_sum"] += gradient["scale"]
                stats["max_aux_ratio"] = max(stats["max_aux_ratio"], gradient["aux_ratio"])
    stages.append({"stage": "source_training", "unique_verified_rows": len(train), "unique_source_rows": len(seen),
                   "weak_partial_rows": len(weak_seen), "online_soft_target_rows": len(soft_seen),
                   "unlabeled_source_rows": int(sum(source_codes[i] == 0 for i in seen)),
                   "gradient_stats": stats, "resources": monitor.result})
    source_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(list(model.trunk.parameters())+list(model.classifier.parameters()),
        lr=config["recovery_learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(config["seed"]+202)
    with ResourceMonitor() as monitor:
        for _ in range(config["recovery_epochs"]):
            for selected in np.array_split(rng.permutation(len(train)), math.ceil(len(train)/config["batch_rows"])):
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(torch.from_numpy(train[selected]))
                loss = F.cross_entropy(logits, torch.from_numpy(labels[selected]), weight=class_weights)
                loss.backward();optimizer.step()
    stages.append({"stage": "verified_recovery", "unique_verified_rows": len(train), "unique_source_rows": 0,
                   "resources": monitor.result})
    distances = {"source_trunk_change_from_warmup": float(torch.linalg.vector_norm(source_state["trunk.0.weight"]-warm_state["trunk.0.weight"])),
                 "final_trunk_change_from_warmup": float(torch.linalg.vector_norm(model.state_dict()["trunk.0.weight"]-warm_state["trunk.0.weight"]))}
    if recipe["name"] != "verified_control" and len(seen) != len(raw_indices):
        raise ValueError("Not every declared source row reached a nonzero optimizer gradient")
    payload = {"model": model.state_dict(), "warm_state": warm_state, "recipe": recipe, "protocol_sha256": protocol_sha256,
               "source_parameter_update_rows": len(seen), "unique_verified_rows": len(train), "stages": stages, **distances}
    payload.update(preprocessing or {})
    temporary = checkpoint.with_suffix(".partial")
    torch.save(payload, temporary);temporary.replace(checkpoint)
    return model, payload
