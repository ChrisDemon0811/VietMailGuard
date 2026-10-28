"""Guard V2.1 development candidate training from test and provenance leakage."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

from vietmailguard.v21.development_training import (
    DevelopmentTrainingBlocked,
    make_tfidf_pipeline,
    rank_candidates,
    sha256_file,
    validation_metrics,
    verify_and_load_development,
    xlm_resource_decision,
)
from vietmailguard.v21.paths import resolve_v21_paths


def _fixture(tmp_path):
    paths = resolve_v21_paths(project_root=tmp_path, environ={})
    folder = paths.splits / "development_bootstrap"
    folder.mkdir(parents=True)
    rows = {}
    for split in ("train", "validation"):
        rows[split] = []
        for label in ("normal", "spam", "phishing"):
            identifier = f"{split}_{label}"
            rows[split].append({
                "id": identifier, "subject": f"Subject {identifier}",
                "body": f"Body of {identifier}", "source": "verified_source",
                "language": "en", "data_origin": "", "label": label,
                "label_status": "eligible", "label_provenance": "verified_source_label",
                "training_eligible": "True", "curation_rule_id": "v2_bootstrap_verified",
                "content_hash": f"content_{identifier}",
                "normalized_content_hash": f"norm_{identifier}",
                "template_hash": f"template_{identifier}",
                "final_group_id": f"group_{identifier}", "parent_id": "", "split": split,
            })
        with (folder / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[split][0]))
            writer.writeheader()
            writer.writerows(rows[split])
    paths.processed.mkdir(parents=True)
    master_path = paths.processed / "Development_Master_Dataset.csv"
    with master_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows["train"][0]))
        writer.writeheader()
        writer.writerows(rows["train"] + rows["validation"])
    manifest = {
        "mode": "development_bootstrap",
        "development_training_ready": True, "final_evaluation_ready": False,
        "final_test_path": None, "v2_test_rows_exported": 0,
        "leakage": {"passed": True},
        "output_rows": {**{split: len(rows[split]) for split in rows}, "master": 6},
        "output_sha256": {**{split: sha256_file(folder / f"{split}.csv") for split in rows},
                          "master": sha256_file(master_path)},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "test.csv").write_text("FORBIDDEN TEST FILE", encoding="utf-8")
    return paths, rows, manifest


def test_verified_development_split_never_opens_test(tmp_path):
    paths, _, _ = _fixture(tmp_path)
    manifest, train, validation = verify_and_load_development(paths, 4096)
    assert len(train) == len(validation) == 3
    assert set(train["label"]) == {"normal", "spam", "phishing"}
    assert manifest["final_test_path"] is None
    assert (paths.splits / "development_bootstrap" / "test.csv").read_text() == "FORBIDDEN TEST FILE"


def test_rejects_cross_split_group_even_with_updated_manifest(tmp_path):
    paths, rows, manifest = _fixture(tmp_path)
    rows["validation"][0]["final_group_id"] = rows["train"][0]["final_group_id"]
    path = paths.splits / "development_bootstrap" / "validation.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows["validation"][0]))
        writer.writeheader()
        writer.writerows(rows["validation"])
    manifest["output_sha256"]["validation"] = sha256_file(path)
    (path.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DevelopmentTrainingBlocked, match="Cross-split"):
        verify_and_load_development(paths, 4096)


def test_rejects_changed_checksum_and_review_rows(tmp_path):
    paths, rows, manifest = _fixture(tmp_path)
    path = paths.splits / "development_bootstrap" / "train.csv"
    rows["train"][0]["label_status"] = "review"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows["train"][0]))
        writer.writeheader()
        writer.writerows(rows["train"])
    with pytest.raises(DevelopmentTrainingBlocked, match="checksum"):
        verify_and_load_development(paths, 4096)
    manifest["output_sha256"]["train"] = sha256_file(path)
    (path.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DevelopmentTrainingBlocked, match="Unverified"):
        verify_and_load_development(paths, 4096)


def test_rejects_missing_or_changed_master(tmp_path):
    paths, _, _ = _fixture(tmp_path)
    master = paths.processed / "Development_Master_Dataset.csv"
    master.write_text("corrupt", encoding="utf-8")
    with pytest.raises(DevelopmentTrainingBlocked, match="Development Master"):
        verify_and_load_development(paths, 4096)


def test_vietnamese_missing_phishing_is_na_not_zero():
    frame = pd.DataFrame({"language": ["en", "en", "en", "vi", "vi"],
                          "data_origin": ["not_declared"] * 3 + ["translated"] * 2})
    actual = np.array(["normal", "spam", "phishing", "normal", "spam"])
    prediction = actual.copy()
    metrics = validation_metrics(actual, prediction, frame, 0.01)
    assert metrics["macro_f1"] == 1.0
    assert metrics["by_language"]["vi"]["phishing_recall"] is None
    assert metrics["by_language"]["zh"]["macro_f1_supported_classes"] is None


def test_tfidf_pipeline_composes_text_and_ranking_is_validation_only():
    config = {"seed": 42, "tfidf": {
        "word": {"ngram_range": [1, 2], "min_df": 1, "max_features": 100,
                 "sublinear_tf": True, "dtype": "float32"},
        "character": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 1,
                      "max_features": 100, "sublinear_tf": True, "dtype": "float32"},
        "svm": {"C": 1.0, "class_weight": "balanced", "max_iter": 5000}}}
    pipeline = make_tfidf_pipeline(config)
    assert list(pipeline.named_steps) == ["compose_text", "tfidf", "classifier"]
    assert [name for name, _ in pipeline.named_steps["tfidf"].transformer_list] == ["word", "character"]
    rows = [{"candidate": "a", "metrics": {"macro_f1": 0.8, "phishing_recall": 0.9,
               "non_normal_false_positives": 2, "prediction_ms_per_email": 1.0}},
            {"candidate": "b", "metrics": {"macro_f1": 0.9, "phishing_recall": 0.5,
               "non_normal_false_positives": 0, "prediction_ms_per_email": 0.1}}]
    assert rank_candidates(rows)[0]["candidate"] == "b"


def test_xlmr_resource_gate_blocks_small_gpu_before_download():
    allowed, reason = xlm_resource_decision(
        {"free_ram_gib": 15, "cuda": True, "total_vram_gib": 4},
        {"min_free_ram_gib": 8, "min_total_vram_gib": 8,
         "model_id": "FacebookAI/xlm-roberta-base", "require_local_model": True})
    assert allowed is False
    assert "VRAM" in reason
