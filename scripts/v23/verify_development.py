"""Verify completed local Development artifacts; never fit or open a Final Test.

Outputs contain identifiers/hashes/aggregates only. Email text stays in the
private intake database. The independent Final Test gate remains closed.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sqlite3
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from vietmailguard.modeling import EmailTextComposer
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary
from vietmailguard.v23.dataset import digest
from vietmailguard.v23.modeling import bounded
from vietmailguard.v23.train_guard import reference_rows

AREA = ROOT / "results/v23_multilingual/development_01"
LABELS = ["normal", "spam", "phishing"]


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    state = json.loads((AREA / "metrics.json").read_text(encoding="utf-8"))
    protocol = json.loads((AREA / "protocol.json").read_text(encoding="utf-8"))
    if state["status"] != "DEVELOPMENT_COMPLETE_NOT_FINAL_EVALUATED":
        raise ValueError("Training has not completed")
    proof = json.loads((AREA / "run_integrity.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "results/v23_multilingual/preserved_before.json").read_text(encoding="utf-8"))
    for path, expected in {**frozen, **proof["source_sha256"]}.items():
        if digest(ROOT / path) != expected:
            raise ValueError("Preserved/source SHA mismatch: " + path)
    reference = reference_rows(ROOT)
    frames = {split: pd.DataFrame(reference[split]) for split in ("train", "validation")}
    if reference["source_hashes"] != protocol["reference_sha256"]:
        raise ValueError("Reference differs from the fitted protocol")
    # final_group_id is inherited provenance. The locked V2.2 experiment's
    # constraint components are recorded separately in experiment_group_id.
    train_groups = set(frames["train"].experiment_group_id)
    validation_groups = set(frames["validation"].experiment_group_id)
    assert not (train_groups & validation_groups)
    for split, frame in frames.items():
        assert frame.experiment_split.eq(split).all()
        assert frame.experiment_group_id.ne("").all()
    train_ids = set(frames["train"].id)
    assert not (train_ids & set(frames["validation"].id))

    spec = importlib.util.spec_from_file_location("v23_verification_cache", ROOT / "scripts/v22/run_group_aware_pilot.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    matrices, _ = helper.selected_cache({key: bounded(frame).assign(
        **{column: frame[column] for column in frame.columns if column not in {"subject", "body"}})
        for key, frame in frames.items()})
    validations = []
    for row in state["candidates"]:
        path = ROOT / row["artifact"]
        if digest(path) != row["artifact_sha256"]:
            raise ValueError("Candidate SHA mismatch: " + row["candidate"])
        model = joblib.load(path)
        inputs = matrices["validation"] if row["candidate"].startswith("minilm") else frames["validation"]
        prediction = model.predict(inputs)
        observed = confusion_matrix(frames["validation"].label, prediction, labels=LABELS).tolist()
        if observed != row["metrics"]["confusion_matrix"]:
            raise ValueError("Reloaded prediction/metric mismatch: " + row["candidate"])
        representation = getattr(model, "representation", None)
        if representation is not None:
            assert representation.documents_seen_ == row["idf_fit_rows"]
        classifier = getattr(model, "classifier", None) if representation is not None else model[-1]
        assert set(classifier.classes_) == set(LABELS)
        validations.append({"candidate": row["candidate"], "sha256": "PASS", "reload_prediction": "PASS"})
    checkpoint = joblib.load(AREA / "idf_checkpoint.joblib")
    assert checkpoint["protocol_sha256"] == digest(AREA / "protocol.json")
    assert checkpoint["transformer"].documents_seen_ == protocol["verified_train_rows"] + protocol["train_only_expansion_rows"]

    database = ROOT / "data/v23/intake.sqlite3"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    selection = """FROM thu t WHERE NOT EXISTS(SELECT 1 FROM tham_chieu r WHERE r.id_thu=t.id_thu)
        AND NOT EXISTS(SELECT 1 FROM cach_ly q WHERE q.id_thu=t.id_thu)"""
    counts = {
        "canonical_all_including_legacy_references": connection.execute("SELECT COUNT(*) FROM thu").fetchone()[0],
        "verified_reference_rows": connection.execute("SELECT COUNT(*) FROM tham_chieu").fetchone()[0],
        "source_input_rows": connection.execute("SELECT COUNT(*) FROM ban_ghi WHERE nguon NOT LIKE 'inherited_verified_%'").fetchone()[0],
        "quarantined_canonical_rows_including_references": connection.execute("SELECT COUNT(DISTINCT id_thu) FROM cach_ly").fetchone()[0],
        "remaining_expansion": connection.execute("SELECT COUNT(*) " + selection).fetchone()[0],
        "expansion_by_source_status_language": list(connection.execute(
            "SELECT nguon,trang_thai_nhan,ngon_ngu_goi_y,COUNT(*) " + selection + " GROUP BY 1,2,3")),
        "all_source_outcomes": list(connection.execute("SELECT nguon,ly_do,COUNT(*) FROM ban_ghi GROUP BY 1,2")),
    }
    assert counts["remaining_expansion"] == protocol["train_only_expansion_rows"]
    assert connection.execute("SELECT COUNT(*) FROM (SELECT id_thu FROM tham_chieu GROUP BY 1 HAVING COUNT(DISTINCT nhan)>1)").fetchone()[0] == 0
    for path, sha, _, complete in connection.execute("SELECT * FROM tien_do"):
        assert complete and digest(ROOT / path) == sha

    # Also measure duplicate model inputs after the documented character bound.
    # Distinct full emails are not claimed to be distinct truncated model inputs.
    hashes = hashlib.sha256()
    input_hashes = set()
    manifest_path = ROOT / "data/v23/train_expansion_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["canonical_id", "normalized_full_content_sha256", "source", "label_status", "split", "learning_role"])
        cursor = connection.execute("SELECT t.id_thu,t.ma_bam_chuan,t.nguon,t.trang_thai_nhan,t.van_ban_nen " + selection + " ORDER BY t.id_thu")
        checked = 0
        while batch := cursor.fetchmany(256):
            texts = EmailTextComposer().transform(bounded(pd.DataFrame([json.loads(zlib.decompress(r[4])) for r in batch])))
            for row, text in zip(batch, texts):
                hashes.update(int(row[0]).to_bytes(8, "little"))
                input_hashes.add(hashlib.sha256(text.encode("utf-8")).digest())
                writer.writerow([*row[:4], "train", "unsupervised_idf_only"])
            checked += len(batch)
            if checked % 51200 == 0:
                print(json.dumps({"verified_expansion_rows": checked}), flush=True)
    assert hashes.hexdigest() == protocol["train_only_expansion_id_sha256"]
    for text in EmailTextComposer().transform(bounded(frames["train"])):
        input_hashes.add(hashlib.sha256(text.encode("utf-8")).digest())
    counts["distinct_bounded_train_model_input_sha256"] = len(input_hashes)
    counts["idf_documents_actually_fitted"] = checkpoint["transformer"].documents_seen_
    counts["bounded_model_input_repetitions"] = counts["idf_documents_actually_fitted"] - len(input_hashes)
    counts["verified_supervised_fit"] = len(frames["train"])
    counts["verified_validation"] = len(frames["validation"])
    counts["unique_full_email_records_used_in_learning_and_validation"] = protocol["unique_used_in_learning_and_validation"]
    counts["by_reference_split_label_language_origin"] = {
        split: [{"label": label, "language": language, "origin": origin, "rows": int(n)}
                for (label, language, origin), n in frame.groupby(["label", "language", "data_origin"]).size().items()]
        for split, frame in frames.items()}
    counts["verified_groups"] = {"train": len(train_groups), "validation": len(validation_groups),
                                "group_column": "experiment_group_id",
                                "validation_spam": int(frames["validation"].loc[frames["validation"].label.eq("spam"), "experiment_group_id"].nunique())}
    connection.close()
    write(AREA / "usage_accounting.json", counts)
    with (AREA / "model_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["candidate", "accuracy", "macro_f1", "weighted_f1", "spam_recall", "phishing_recall",
                   "fit_seconds", "idf_fit_seconds", "prediction_seconds_1938_rows", "prediction_scope", "artifact_bytes"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in state["candidates"]:
            m = row["metrics"]
            writer.writerow({"candidate": row["candidate"], **{k: m[k] for k in ["accuracy", "macro_f1", "weighted_f1"]},
                "spam_recall": m["per_class"]["spam"]["recall"], "phishing_recall": m["per_class"]["phishing"]["recall"],
                "fit_seconds": row["fit_seconds"], "idf_fit_seconds": row.get("idf_fit_seconds", "not_measured_separately"),
                "prediction_seconds_1938_rows": row["prediction_seconds"], "prediction_scope": row["prediction_scope"], "artifact_bytes": row["artifact_bytes"]})
    result = {"frozen_old_artifacts_unchanged": len(frozen), "source_hashes_unchanged": len(proof["source_sha256"]),
              "raw_source_checkpoints_verified": True, "benchmark_sha256_verified": True,
              "candidate_reloads": validations, "expansion_id_hash_verified": True,
              "new_human_approvals": 0, "pseudo_label_rows": 0, "final_test_opened": False,
              "external_network_allowed": False, "scope": "Development only; not independent evaluation or production"}
    write(AREA / "verification.json", result)
    print(json.dumps({"verification": result, "usage": counts}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    with offline_boundary(ROOT), threadpool_limits(limits=2):
        main()
