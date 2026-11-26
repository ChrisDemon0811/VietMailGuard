"""Encode the already guarded Train-only corpus with the pinned local MiniLM."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from vietmailguard.modeling import EmailTextComposer
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary, local_encoder_snapshot, load_encoder
from vietmailguard.v23.dataset import digest
from vietmailguard.v23.modeling import bounded
from vietmailguard.v23.classifier_expansion import atomic_json, ResourceMonitor, memory_bytes

NAME = "classifier_expansion_02"
AREA = ROOT / "data/v23" / NAME
RESULTS = ROOT / "results/v23_multilingual" / NAME


def main():
    config_path = ROOT / "config/v23" / (NAME + ".json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    torch.set_num_threads(config["cpu_threads"])
    AREA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    previous = ROOT / "results/v23_multilingual/development_01/frozen_manifest.json"
    old = json.loads(previous.read_text(encoding="utf-8"))
    for path, record in old["files"].items():
        if digest(ROOT / path) != record["sha256"]:
            raise ValueError("Baseline preserved file differs: " + path)
    if shutil.disk_usage(ROOT).free < 5*2**30:
        raise RuntimeError("Need at least 5 GiB disk free")
    database = ROOT / "data/v23/intake.sqlite3"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    selection = """FROM thu t WHERE NOT EXISTS(SELECT 1 FROM tham_chieu r WHERE r.id_thu=t.id_thu)
        AND NOT EXISTS(SELECT 1 FROM cach_ly q WHERE q.id_thu=t.id_thu)"""
    rows = list(connection.execute("SELECT t.id_thu,t.trang_thai_nhan,t.nguon " + selection + " ORDER BY t.id_thu"))
    assert len(rows) == config["expansion_rows"]
    ids = np.asarray([r[0] for r in rows], dtype=np.int64)
    weak = np.asarray([r[1] == "weak_label" for r in rows], dtype=np.bool_)
    assert all(r[1] in {"weak_label", "unlabeled"} for r in rows)
    assert all(r[2].startswith("bruce_guenter_") for r in rows if r[1] == "weak_label")
    hash_ids = hashlib.sha256()
    for identifier in ids:
        hash_ids.update(int(identifier).to_bytes(8, "little"))
    baseline = json.loads((previous.parent / "protocol.json").read_text(encoding="utf-8"))
    assert hash_ids.hexdigest() == baseline["train_only_expansion_id_sha256"]
    encoder_config = json.loads((ROOT/config["encoder_config"]).read_text(encoding="utf-8"))
    snapshot = local_encoder_snapshot(encoder_config["encoder"])
    for path, expected in encoder_config["encoder_files"].items():
        assert digest(snapshot/path) == expected, "Pinned encoder SHA differs"
    lock = {"config_sha256": digest(config_path), "baseline_frozen_manifest_sha256": digest(previous),
            "intake_sha256": digest(database), "expansion_id_sha256": hash_ids.hexdigest(),
            "rows": len(rows), "weak_label_rows": int(weak.sum()), "unlabeled_rows": int((~weak).sum()),
            "encoder_config_sha256": digest(ROOT/config["encoder_config"]), "encoder_revision": encoder_config["encoder"]["revision"],
            "encoder_frozen": True, "reference_sha256": baseline["reference_sha256"],
            "selection_before_validation": True, "final_test_allowed": False}
    protocol_path = RESULTS/"protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == lock, "Locked protocol differs"
    else:
        atomic_json(protocol_path, lock)
    for path, value in [(AREA/"ids.npy", ids), (AREA/"weak_mask.npy", weak)]:
        if path.exists():
            assert np.array_equal(np.load(path, allow_pickle=False), value)
        else:
            np.save(path, value, allow_pickle=False)
    receipt = AREA/"embeddings.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        assert saved["protocol_sha256"] == digest(protocol_path)
        assert digest(AREA/"embeddings.npy") == saved["matrix_sha256"]
        print(json.dumps({"stage": "complete_cache_verified", "rows": len(rows)}), flush=True)
        return
    checkpoint = AREA/"embedding_progress.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {
        "next_row": 0, "seconds": 0., "protocol_sha256": digest(protocol_path)}
    assert state["protocol_sha256"] == digest(protocol_path)
    partial = AREA/"embeddings.partial.npy"
    if partial.exists():
        matrix = np.load(partial, mmap_mode="r+", allow_pickle=False)
        assert matrix.shape == (len(rows), 384) and matrix.dtype == np.float32
    else:
        assert state["next_row"] == 0
        matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(rows), 384))
    device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 1024**3 else "cpu"
    if memory_bytes()[1] < 1.5*2**30:
        raise RuntimeError("At least 1.5 GiB available RAM required for the pinned encoder")
    batch = config["embedding_batch_" + device]
    with ResourceMonitor() as monitor:
        encoder = load_encoder(snapshot, encoder_config["encoder"], device)
        position = state["next_row"]
        cursor = connection.execute("SELECT t.id_thu,t.van_ban_nen " + selection + " AND t.id_thu>? ORDER BY t.id_thu",
                                    (int(ids[position-1]) if position else 0,))
        started = time.perf_counter()
        since_checkpoint = 0
        while values := cursor.fetchmany(batch):
            assert [v[0] for v in values] == list(ids[position:position+len(values)])
            frame = pd.DataFrame([json.loads(zlib.decompress(blob)) for _, blob in values])
            texts = EmailTextComposer().transform(bounded(frame)).tolist()
            try:
                encoded = encoder.encode(texts, batch_size=batch, normalize_embeddings=True,
                    convert_to_numpy=True, device=device, show_progress_bar=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch = max(1, batch//2)
                encoded = encoder.encode(texts, batch_size=batch, normalize_embeddings=True,
                    convert_to_numpy=True, device=device, show_progress_bar=False)
            assert np.isfinite(encoded).all() and np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=.002)
            matrix[position:position+len(values)] = encoded
            position += len(values)
            since_checkpoint += len(values)
            if since_checkpoint >= config["embedding_checkpoint_rows"] or position == len(rows):
                matrix.flush()
                atomic_json(checkpoint, {**state, "next_row": position, "seconds": state["seconds"] + time.perf_counter()-started})
                since_checkpoint = 0
                if position % 5120 == 0 or position == len(rows):
                    print(json.dumps({"stage": "encode", "completed": position, "total": len(rows), "device": device,
                                      "current_batch": batch, "emails_per_second_this_session": (position-state["next_row"])/(time.perf_counter()-started)}), flush=True)
        assert position == len(rows)
    del matrix
    partial.replace(AREA/"embeddings.npy")
    atomic_json(receipt, {**lock, "protocol_sha256": digest(protocol_path), "matrix_sha256": digest(AREA/"embeddings.npy"),
                         "dimension": 384, "dtype": "float32", "normalized": True, "device": device,
                         "max_sequence_length": 128, "max_characters_per_field": 4096,
                         "encoder_training_rows": 0, "classifier_training_rows": 0,
                         "resources_this_session": monitor.result,
                         "encoding_seconds_all_sessions": state["seconds"] + monitor.result["seconds"]})
    connection.close()
    print(json.dumps({"stage": "encoded", "rows": len(rows), "resources": monitor.result}), flush=True)


if __name__ == "__main__":
    with offline_boundary(ROOT):
        main()
