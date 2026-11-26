"""Small, isolated partial-label/consistency study on frozen MiniLM features.

Source candidate sets and EMA targets are uncertain training signals, never
Verified labels. The decoder learns from unlabeled features, not invented classes.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import threading
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def memory_bytes():
    """OS memory counters without requiring a new monitoring dependency."""
    if sys.platform == "win32":
        from ctypes import wintypes
        class Status(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in ("total", "available", "total_page", "available_page",
                                                      "total_virtual", "available_virtual", "extended")]
        class Counters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak_rss", "rss", "peak_paged", "paged",
                                                  "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        status = Status();status.length = ctypes.sizeof(status)
        counters = Counters();counters.size = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        handle = kernel.GetCurrentProcess()
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)) or not query(handle, ctypes.byref(counters), counters.size):
            raise OSError("Windows memory counters unavailable")
        return int(counters.rss), int(status.available)
    import os
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024, int(os.sysconf("SC_AVPHYS_PAGES"))*int(os.sysconf("SC_PAGE_SIZE"))


class ExpansionHead(nn.Module):
    def __init__(self, dimension=384, hidden=(128, 64)):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dimension, hidden[0]), nn.ReLU(),
                                   nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.classifier = nn.Linear(hidden[1], 3)
        self.decoder = nn.Linear(hidden[1], dimension)

    def forward(self, inputs):
        latent = self.trunk(inputs)
        return self.classifier(latent), self.decoder(latent)


def partial_label_loss(logits, allowed=(1, 2)):
    """Negative log candidate-set mass, without choosing Spam or Phishing."""
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(logits[:, list(allowed)], dim=1)).mean()


def consistency_loss(student_logits, teacher_logits):
    """Soft-target KL, entropy-weighted; weights are NOT calibrated confidence."""
    target = teacher_logits.detach().softmax(dim=1)
    entropy = -(target * target.clamp_min(1e-8).log()).sum(dim=1)
    weight = (1 - entropy / math.log(3)).clamp_min(.05)
    divergence = F.kl_div(student_logits.log_softmax(dim=1), target, reduction="none").sum(dim=1)
    return (weight * divergence).mean(), weight.mean().detach()


@torch.no_grad()
def update_teacher(teacher, student, decay):
    for target, source in zip(teacher.parameters(), student.parameters(), strict=True):
        target.mul_(decay).add_(source, alpha=1-decay)


def ordered_indices(ids, seed):
    return np.asarray(sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f"{seed}:{int(ids[i])}".encode("ascii")).digest()), dtype=np.int64)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ResourceMonitor:
    """Sample process RSS/available RAM; record CUDA allocator peaks separately."""
    def __init__(self, interval=.25):
        self.interval = interval
        self.stop = threading.Event()
        self.peak_rss = 0
        self.min_available = float("inf")

    def sample(self):
        rss, available = memory_bytes()
        self.peak_rss = max(self.peak_rss, rss)
        self.min_available = min(self.min_available, available)

    def __enter__(self):
        self.start = time.perf_counter()
        self.sample()
        def watch():
            while not self.stop.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()
        self.result = {"seconds": time.perf_counter()-self.start,
                       "sampled_peak_process_rss_gib": self.peak_rss/2**30,
                       "minimum_available_system_ram_gib": self.min_available/2**30,
                       "sampling_interval_seconds": self.interval,
                       "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0,
                       "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0,
                       "vram_scope": "PyTorch allocator in this process, not whole-device usage"}


def fit_head(train, labels, expansion, is_weak, indices, config, method,
             initial_state, checkpoint_path: Path, protocol_sha256: str):
    """Matched verified replay; expansion targets never consult Validation."""
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    model = ExpansionHead(train.shape[1], tuple(config["hidden"]))
    model.load_state_dict(initial_state)
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["expansion_learning_rate"],
                                 weight_decay=config["weight_decay"])
    frequencies = np.bincount(labels, minlength=3)
    weights = torch.tensor(len(labels)/(3*frequencies), dtype=torch.float32)
    order = rng.permutation(len(train))
    position = 0
    epoch_start = 0
    expansion_seen = set()
    weak_seen = set()
    consistency_seen = set()
    steps = 0
    history = []
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["protocol_sha256"] != protocol_sha256 or saved["method"] != method or saved["budget"] != len(indices):
            raise ValueError("Expansion checkpoint protocol differs")
        model.load_state_dict(saved["model"])
        teacher.load_state_dict(saved["teacher"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch_start = saved["next_epoch"]
        position, order = saved["position"], saved["order"]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        steps, history = saved["steps"], saved["history"]
        expansion_seen, weak_seen, consistency_seen = map(set, saved["seen"])
    batch = config["batch_rows"]
    for epoch in range(epoch_start, config["expansion_epochs"]):
        totals = np.zeros(5)
        chosen = rng.permutation(indices)
        for start in range(0, len(chosen), batch):
            if position >= len(order):
                order = rng.permutation(len(train)); position = 0
            verified = order[position:position+batch]
            position += len(verified)
            inputs = torch.from_numpy(np.asarray(train[verified]).copy())
            truth = torch.from_numpy(labels[verified])
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            supervised = F.cross_entropy(logits, truth, weight=weights)
            loss = supervised
            weak_loss = torch.tensor(0.)
            consistency = torch.tensor(0.)
            reconstruction = torch.tensor(0.)
            reliability = torch.tensor(0.)
            selected = chosen[start:start+batch]
            if method != "verified_replay_control":
                source = torch.from_numpy(np.asarray(expansion[selected]).copy())
                weak_mask = np.asarray(is_weak[selected], dtype=bool)
                source_logits, _ = model(source)
                if weak_mask.any():
                    weak_loss = partial_label_loss(source_logits[weak_mask], config["weak_candidate_indices"])
                    loss = loss + config["weak_loss_weight"] * weak_loss
                    weak_seen.update(map(int, selected[weak_mask]))
                    expansion_seen.update(map(int, selected[weak_mask]))
                if method == "weak_partial_consistency_denoise":
                    corrupted = F.dropout(source, p=config["embedding_dropout"], training=True)
                    noisy_logits, decoded = model(corrupted)
                    with torch.no_grad():
                        target, _ = teacher(source)
                    consistency, reliability = consistency_loss(noisy_logits, target)
                    reconstruction = F.mse_loss(decoded, source)
                    loss = loss + config["consistency_loss_weight"]*consistency + config["reconstruction_loss_weight"]*reconstruction
                    consistency_seen.update(map(int, selected))
                    expansion_seen.update(map(int, selected))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite expansion loss; no Validation-based adjustment allowed")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            update_teacher(teacher, model, config["ema_decay"])
            totals += [float(supervised.detach()), float(weak_loss.detach()), float(consistency.detach()),
                       float(reconstruction.detach()), float(reliability)]
            steps += 1
        history.append({"epoch": epoch+1, "mean_batch_losses": (totals/math.ceil(len(indices)/batch)).tolist()})
        temporary = checkpoint_path.with_suffix(".partial")
        torch.save({"protocol_sha256": protocol_sha256, "method": method, "budget": len(indices),
                    "model": model.state_dict(), "teacher": teacher.state_dict(), "optimizer": optimizer.state_dict(),
                    "next_epoch": epoch+1, "steps": steps, "history": history, "position": position, "order": order,
                    "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "seen": [sorted(expansion_seen), sorted(weak_seen), sorted(consistency_seen)]}, temporary)
        temporary.replace(checkpoint_path)
        print(json.dumps({"method": method, "budget": len(indices), "epoch": epoch+1,
                          "expansion_unique_used_for_parameter_updates": len(expansion_seen)}), flush=True)
    return model, {"optimizer_steps": steps, "verified_supervised_rows": len(train),
                   "expansion_unique_parameter_update_rows": len(expansion_seen), "weak_partial_loss_rows": len(weak_seen),
                   "soft_consistency_and_reconstruction_rows": len(consistency_seen), "hard_pseudo_label_rows": 0,
                   "online_soft_pseudo_target_rows": len(consistency_seen),
                   "soft_targets_are_uncalibrated_not_ground_truth": True, "history": history}
