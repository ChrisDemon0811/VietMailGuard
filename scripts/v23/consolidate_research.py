"""Report-only extraction from saved V2.3 results. No model loading or scoring.

This script never fits a model, calls predict, opens email datasets or a Test.
Original frozen files are checked before/after; derived tables/figures go only
to reports/v23/final_research.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
from vietmailguard.v21.multilingual_dl_experiment import offline_boundary

OUT = ROOT/"reports/v23/final_research"
LABELS = ["normal", "spam", "phishing"]
EXPERIMENTS = ["development_01", "classifier_expansion_02", "large_scale_optimization_03"]


def checked_path(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("File outside project boundary")
    if path.name.lower() in {"test.csv", "final_test.csv", ".env"} or any(
            path.is_relative_to(root/prefix) for prefix in ["data/evaluation", "data/v21/evaluation"]):
        raise ValueError("Held-out or secret file is prohibited")
    return path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(relative):
    return json.loads(checked_path(ROOT, relative).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def csv_rows(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def verify_saved_metrics(metric):
    """Arithmetic consistency only; does not evaluate predictions or a dataset."""
    cm = np.asarray(metric["confusion_matrix"], dtype=np.int64)
    if cm.shape != (3, 3) or (cm < 0).any() or cm.sum() != 1938:
        raise ValueError("Unexpected benchmark confusion matrix")
    support, predicted, correct = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(3), where=predicted != 0)
    recall = correct/support
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=precision+recall != 0)
    actual = {"accuracy": float(correct.sum()/cm.sum()), "macro_f1": float(f1.mean()),
              "weighted_f1": float(np.sum(f1*support)/support.sum())}
    for key, value in actual.items():
        if key in metric and not math.isclose(value, metric[key], abs_tol=1e-12):
            raise ValueError("Saved metric inconsistent with saved matrix: "+key)
    for i, label in enumerate(LABELS):
        saved = metric["per_class"][label]
        if saved["support"] != support[i] or not math.isclose(saved["recall"], recall[i], abs_tol=1e-12):
            raise ValueError("Saved class support/recall mismatch")
        for key, value in [("precision", precision[i]), ("f1-score", f1[i])]:
            if key in saved and not math.isclose(saved[key], value, abs_tol=1e-12):
                raise ValueError("Saved class metric inconsistent with saved matrix")


def require_common_reference(references):
    if not references or any(reference != references[0] for reference in references):
        raise ValueError("Different benchmark fingerprints must not be ranked together")


def protected_files():
    expected = read("results/v23_multilingual/preserved_before.json")
    counts = {"older_artifacts": len(expected)}
    for name in EXPERIMENTS:
        relative = f"results/v23_multilingual/{name}/frozen_manifest.json"
        path = checked_path(ROOT, relative)
        declared = path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
        if sha(path) != declared:raise ValueError("Frozen manifest checksum mismatch")
        manifest = read(relative)
        entries = {key: value["sha256"] for key, value in manifest["files"].items()}
        for archive in manifest.get("archived_invalid_attempts", {}).values():
            entries.update(archive["files_sha256"])
        for key, value in entries.items():
            if key in expected and expected[key] != value:raise ValueError("Conflicting frozen checksums")
            expected[key] = value
        expected[relative] = declared
        counts[name] = len(manifest["files"])
    for relative, expected_hash in expected.items():
        if sha(checked_path(ROOT, relative)) != expected_hash:
            raise ValueError("Preserved file changed: "+relative)
    return expected, counts


def collect():
    d = read("results/v23_multilingual/development_01/protocol.json")
    e = read("results/v23_multilingual/classifier_expansion_02/protocol.json")
    require_common_reference([d["reference_sha256"], e["reference_sha256"]])
    current_manifest = read("data/v22/group_aware_supervised_pilot_01/cohort/manifest.json")
    require_common_reference([d["reference_sha256"], current_manifest["output_sha256"]])
    cache = read("data/v23/classifier_expansion_02/embeddings.json")
    require_common_reference([d["reference_sha256"], cache["reference_sha256"]])
    optimization = read("results/v23_multilingual/large_scale_optimization_03/protocol.json")
    if optimization["embedding_sha256"] != cache["matrix_sha256"] or optimization["intake_sha256"] != cache["intake_sha256"]:
        raise ValueError("Optimization has a different data/cache lineage")
    rows, detailed, parameters = [], [], []
    for experiment in EXPERIMENTS:
        source = f"results/v23_multilingual/{experiment}/metrics.json"
        result = read(source)
        for item in result["candidates"]:
            metric = item["metrics"];verify_saved_metrics(metric)
            name = item.get("candidate", item.get("name"))
            source_updates = item.get("expansion_unique_parameter_update_rows", item.get("source_parameter_update_rows", 0))
            fit = item.get("fit_seconds", item.get("resources", {}).get("seconds"))
            if "stages" in item:fit = sum(stage["resources"]["seconds"] for stage in item["stages"])
            row = {"experiment": experiment, "candidate": name, "benchmark": "V22_group9686_val1938",
                   "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"], "weighted_f1": metric["weighted_f1"],
                   **{label+"_recall": metric["per_class"][label]["recall"] for label in LABELS},
                   "normal_false_positives": int(sum(metric["confusion_matrix"][0][1:])), "verified_supervised_rows": 7748,
                   "source_parameter_update_rows": source_updates, "total_classifier_parameter_rows": 7748+source_updates,
                   "classifier_supervised_only": source_updates == 0,
                   "idf_fit_rows": item.get("idf_fit_rows", "N/A"),
                   "soft_teacher_target_rows": item.get("online_soft_pseudo_target_rows", 0),
                   "fit_seconds_recorded": fit, "prediction_seconds": item.get("prediction_seconds", "N/A"),
                   "timing_scope": item.get("prediction_scope", "Stage timings exclude cached encoder; shared warmup charged to control"),
                   "artifact_bytes": item["artifact_bytes"], "artifact": item["artifact"], "artifact_sha256": item["artifact_sha256"],
                   "source_json": source}
            rows.append(row)
            for label in LABELS:
                value = metric["per_class"][label]
                detailed.append({"experiment": experiment, "candidate": name, "language": "all", "class": label,
                    "support": value["support"], "precision": value["precision"], "recall": value["recall"], "f1": value["f1-score"]})
            for language, subset in metric.get("by_language", {}).items():
                for label in LABELS:
                    value = (subset or {}).get("per_class", {}).get(label, {})
                    supported = value.get("support", 0) > 0
                    detailed.append({"experiment": experiment, "candidate": name, "language": language, "class": label,
                        "support": value.get("support", 0), "precision": value.get("precision"), "recall": value.get("recall"),
                        "f1": value.get("f1-score") if supported else None})
            matrix_rows = [{"true_class": label, **dict(zip(LABELS, values))}
                           for label, values in zip(LABELS, metric["confusion_matrix"])]
            csv_rows(OUT/"confusion_matrices"/(experiment+"__"+name+".csv"), matrix_rows)
            config = read("config/v23/"+{"development_01": "development_training.json", "classifier_expansion_02": "classifier_expansion_02.json", "large_scale_optimization_03": "large_scale_optimization_03.json"}[experiment])
            if experiment == "development_01":
                neural = name.startswith("minilm")
                architecture = "Frozen MiniLM 384 → " + ("MLP 128,64" if "mlp" in name else "PCA100 → LinearSVC" if "pca" in name else "LDA2 → LinearSVC") if neural else "Word/Character hash TF-IDF131072 → "+("chi2(60000) → " if "chi2" in name else "SVD100 → " if "svd" in name else "")+"LinearSVC"
                learning = "MLP: Adam lr=.001, max_iter120, alpha=.0001, batch256" if "mlp" in name else "LinearSVC C=1 balanced max_iter5000"
            else:
                architecture = "Frozen MiniLM 384 → MLP 128,64 → 3 classes; decoder 64→384"
                learning = "Warmup30 lr=.001; expansion3 lr=.0001; weak=.25 KL=.05 reconstruction=.1 by method" if experiment == "classifier_expansion_02" else "Warmup30 lr=.001; source1 lr=.0001; recovery20 lr=.0003; prediction-gradient cap=.2; recipe/offset from Train OOF"
            parameters.append({"experiment": experiment, "candidate": name, "architecture": architecture, "optimizer_or_solver": learning,
                "seed": config["seed"], "verified_supervised_rows": 7748, "source_parameter_rows": source_updates,
                "idf_rows": item.get("idf_fit_rows", "N/A"), "max_input_characters_per_field": 4096,
                "encoder_tokens": 128 if "minilm" in name or experiment != "development_01" else "N/A",
                "cpu_threads": 2, "config": "config/v23/"+{"development_01": "development_training.json", "classifier_expansion_02": "classifier_expansion_02.json", "large_scale_optimization_03": "large_scale_optimization_03.json"}[experiment],
                "full_config_json": json.dumps(config, ensure_ascii=False, separators=(",", ":"))})
    if len(rows) != 21:raise ValueError("Unexpected number of valid outer-Validation candidates")
    return rows, detailed, parameters


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected, counts = protected_files()
    write_json(OUT/"integrity_before.json", {"status": "PASS", "source_file_count": len(expected), "groups": counts, "sha256": expected})
    rows, detailed, parameters = collect()
    csv_rows(OUT/"model_comparison.csv", rows)
    csv_rows(OUT/"per_class_language_metrics.csv", detailed)
    csv_rows(OUT/"training_parameters.csv", parameters)
    write_json(OUT/"results_snapshot.json", {name: read(f"results/v23_multilingual/{name}/metrics.json") for name in EXPERIMENTS})
    write_json(OUT/"training_configs.json", {name: read("config/v23/"+file) for name, file in [
        ("development_01", "development_training.json"), ("classifier_expansion_02", "classifier_expansion_02.json"),
        ("large_scale_optimization_03", "large_scale_optimization_03.json")] } | {"sklearn_mlp_reference": read("config/v21/experiments/multilingual_dl_01.json")})
    usage = read("results/v23_multilingual/development_01/usage_accounting.json")
    source_status = Counter()
    source_language = []
    for source, status, language, count in usage["expansion_by_source_status_language"]:
        source_status[status] += count
        source_language.append({"source": source, "label_status": status, "language_signal": language, "rows": count,
                                "language_evidence": "heuristic; not verified native-language reference", "three_class_label": "unassigned"})
    roles = [
        {"scope": "final_training", "status": "verified", "rows": 7748, "role": "supervised ground truth"},
        {"scope": "final_training", "status": "weak_label", "rows": source_status["weak_label"], "role": "uncertain source label; reconstruction and optional weak constraint"},
        {"scope": "final_training", "status": "unlabeled", "rows": source_status["unlabeled"], "role": "unsupervised/consistency, no reference class"},
        {"scope": "development_validation", "status": "verified", "rows": 1938, "role": "fixed historically exposed benchmark"},
        {"scope": "original_admitted_development", "status": "verified", "rows": 9730, "role": "overlaps cohort above; do not add"},
        {"scope": "excluded_from_experiment_not_deleted", "status": "legacy_mask44", "rows": 44, "role": "contained in original 9730"}]
    assert sum(row["rows"] for row in roles if row["scope"] == "final_training") == 364761
    csv_rows(OUT/"data_roles.csv", roles);csv_rows(OUT/"expansion_source_language.csv", source_language)
    distribution = [{"split": split, **row, "origin": row["origin"] or "not_declared"}
                    for split, values in usage["by_reference_split_label_language_origin"].items() for row in values]
    csv_rows(OUT/"verified_split_distribution.csv", distribution)
    write_json(OUT/"data_accounting.json", usage)
    old = next(row for row in rows if row["candidate"] == "weak_partial_consistency_denoise_357013")
    new = next(row for row in rows if row["candidate"] == "guarded_weak_recovery")
    delta = {"macro_f1_absolute": new["macro_f1"]-old["macro_f1"],
             "macro_f1_percentage_points": 100*(new["macro_f1"]-old["macro_f1"]),
             "macro_f1_relative_percent": 100*(new["macro_f1"]-old["macro_f1"])/old["macro_f1"],
             "normal_false_positives_old": old["normal_false_positives"], "normal_false_positives_new": new["normal_false_positives"],
             "normal_fp_reduction_count": old["normal_false_positives"]-new["normal_false_positives"],
             "normal_fp_reduction_percent": 100*(old["normal_false_positives"]-new["normal_false_positives"])/old["normal_false_positives"],
             "spam_recall_change": new["spam_recall"]-old["spam_recall"], "phishing_recall_change": new["phishing_recall"]-old["phishing_recall"]}
    write_json(OUT/"improvement_analysis.json", delta)
    baseline = next(row for row in read("results/v22_multilingual/experiments/group_aware_supervised_pilot_01/metrics.json")["candidates"] if row["candidate"] == "tfidf_linearsvc")
    verify_saved_metrics(baseline["metrics"])
    csv_rows(OUT/"confusion_matrices"/"context_v22_tfidf_baseline.csv", [
        {"true_class": label, **dict(zip(LABELS, values))}
        for label, values in zip(LABELS, baseline["metrics"]["confusion_matrix"])])
    v21 = read("results/v21_multilingual/development_validation_metrics.json")
    v21_svc = next(row for row in v21["candidates"] if row["candidate"] == "tfidf_word_character_linear_svm")
    trec = read("results/v21_multilingual/trec06c_binary/validation_metrics.json")
    trec_svc = next(row for row in trec["candidates"] if row["candidate"] == "char_tfidf_linear_svc")
    context = [
        {"model": "V22 saved TF-IDF+LinearSVC", "validation_rows": 1938, "macro_f1": baseline["metrics"]["macro_f1"],
         "comparable_to_v23": True, "reason": "same frozen group-aware cohort manifest", "source": "results/v22_multilingual/experiments/group_aware_supervised_pilot_01/metrics.json"},
        {"model": "V21 Development TF-IDF+LinearSVC", "validation_rows": v21["validation_rows"], "macro_f1": v21_svc["metrics"]["macro_f1"],
         "comparable_to_v23": False, "reason": "different 6709-row inherited Development Validation", "source": "results/v21_multilingual/development_validation_metrics.json"},
        {"model": "TREC06C binary Character+LinearSVC", "validation_rows": trec_svc["validation_rows"], "macro_f1": trec_svc["metrics"]["macro_f1"],
         "comparable_to_v23": False, "reason": "Chinese Ham/Spam binary taxonomy and different corpus; no three-class Phishing evidence", "source": "results/v21_multilingual/trec06c_binary/validation_metrics.json"}]
    csv_rows(OUT/"benchmark_context.csv", context)
    create_figures(rows, baseline, delta)
    resources = []
    def resource_row(experiment, candidate, stage, values):
        resources.append({"experiment": experiment, "candidate": candidate, "stage": stage,
            "seconds": values.get("seconds"), "sampled_peak_process_rss_gib": values.get("sampled_peak_process_rss_gib"),
            "cuda_peak_allocated_gib": values.get("cuda_peak_allocated_gib"),
            "cuda_peak_reserved_gib": values.get("cuda_peak_reserved_gib"),
            "scope": "sampled process RSS and process PyTorch allocator; not whole-system peak"})
    cache = read("data/v23/classifier_expansion_02/embeddings.json")
    resource_row(EXPERIMENTS[1], "shared_encoder_cache", "encode_source_once", cache["resources_this_session"])
    resource_row(EXPERIMENTS[1], "shared_verified_warmup", "verified_warmup",
                 read("results/v23_multilingual/classifier_expansion_02/warmup_resources.json"))
    first_metrics = read("results/v23_multilingual/development_01/metrics.json")
    idf = next(item for item in first_metrics["candidates"] if item.get("idf_fit_seconds") is not None)
    resource_row(EXPERIMENTS[0], "shared_large_idf", "fit_idf", {"seconds": idf["idf_fit_seconds"]})
    for experiment in EXPERIMENTS:
        for item in read(f"results/v23_multilingual/{experiment}/metrics.json")["candidates"]:
            name = item.get("candidate", item.get("name"))
            if "resources" in item:resource_row(experiment, name, "head_fit_excluding_shared_warmup", item["resources"])
            for stage in item.get("stages", []):resource_row(experiment, name, stage["stage"], stage["resources"])
    csv_rows(OUT/"resource_stages.csv", resources)
    selection = read("results/v23_multilingual/large_scale_optimization_03/selection.json")
    write_json(OUT/"train_oof_selection.json", selection)
    csv_rows(OUT/"train_oof_selection.csv", [{"recipe": r["recipe"], "normal_logit_offset": r["normal_logit_offset"],
        "normal_fpr_gate_passed": r["normal_fpr_gate_passed"], "macro_f1": r["metrics"]["macro_f1"],
        "normal_false_positives": r["metrics"]["normal_false_positives"], "scope": "Train group OOF selection, not outer Validation or Test"}
        for r in selection["all_predeclared_train_oof_comparisons"]])
    write_reports(rows, baseline, delta, expected, context)
    for relative, checksum in expected.items():
        if sha(checked_path(ROOT, relative)) != checksum:raise ValueError("Original changed during consolidation")
    write_json(OUT/"integrity_after.json", {"status": "PASS", "protected_file_count": len(expected), "all_original_sha256_unchanged": True,
        "groups": counts, "metric_matrix_checks": len(rows)+1, "no_training_or_prediction_executed": True,
        "no_email_dataset_or_final_test_opened": True, "valid_v23_outer_validation_candidates": 21})
    write_bundle_manifest()
    print(json.dumps({"valid_candidates": len(rows), "preserved_files": len(expected), "groups": counts, "improvement": delta}), flush=True)


def markdown_table(headers, values):
    def cell(value):
        if value is None:return "N/A"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"---|"*len(headers)] +
                     ["| "+" | ".join(cell(value) for value in row)+" |" for row in values])


def write_bundle_manifest():
    paths = [path for path in OUT.rglob("*") if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}]
    paths.extend([OUT.parent/"V23_FINAL_RESEARCH_REPORT.md", OUT.parent/"V23_RESEARCH_RESULTS.md",
                  Path(__file__), ROOT/"tests/test_v23_research_consolidation.py"])
    payload = {path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
               for path in sorted(paths)}
    destination = OUT/"manifest.json"
    write_json(destination, {"schema_version": 1, "scope": "derived_research_consolidation_only", "date": "2026-11-24",
        "existing_frozen_manifests_modified": False, "new_training_or_evaluation": False,
        "files": payload, "self_hash_policy": "manifest excluded from entries; detached SHA-256 in manifest.sha256"})
    destination.with_suffix(".sha256").write_text(sha(destination)+"  manifest.json\n", encoding="ascii")


def write_reports(rows, baseline, delta, expected, context):
    """Generate the two requested Markdown documents from the saved snapshots."""
    parent = OUT.parent
    data_table = markdown_table(["Vai trò", "Verified", "Weak Label", "Unlabeled", "Tổng"], [
        ["Train tham gia cập nhật MLP large-scale", 7748, 127743, 229270, 364761],
        ["Development Validation cố định", 1938, 0, 0, 1938],
        ["Cohort Verified (Train + Validation)", 9686, 0, 0, 9686],
        ["Development được nhận lại, gồm 44 mẫu mask", 9730, 0, 0, 9730]])
    split_table = markdown_table(["Split", "Normal", "Spam", "Phishing", "EN", "VI dịch", "ZH", "Nhóm"], [
        ["Train", 6256, 130, 1362, 7315, 433, 0, 6944],
        ["Validation", 1565, 33, 340, 1827, 111, 0, 1736]])
    result_table = markdown_table(["Giai đoạn / candidate", "Accuracy", "Macro F1", "Weighted F1", "Spam Recall", "Phishing Recall", "Normal sai", "Classifier rows"],
        [[f"{EXPERIMENTS.index(r['experiment'])+1:02d} / `{r['candidate']}`", *[f"{r[k]:.6f}" for k in
          ["accuracy", "macro_f1", "weighted_f1", "spam_recall", "phishing_recall"]], r["normal_false_positives"],
          r["total_classifier_parameter_rows"]] for r in rows])
    names = [(EXPERIMENTS[1], "weak_partial_consistency_denoise_357013"),
             (EXPERIMENTS[2], "guarded_weak_recovery"), (EXPERIMENTS[2], "verified_control")]
    focal = [next(r for r in rows if r["experiment"] == exp and r["candidate"] == name) for exp, name in names]
    focal_table = markdown_table(["Mô hình", "Macro F1", "Normal Recall", "Spam Recall", "Phishing Recall", "Normal sai"],
        [[r["candidate"], *[f"{r[k]:.6f}" for k in ["macro_f1", "normal_recall", "spam_recall", "phishing_recall"]],
          r["normal_false_positives"]] for r in focal]+[
        ["TF-IDF + LinearSVC lưu từ V2.2", f"{baseline['metrics']['macro_f1']:.6f}",
         *[f"{baseline['metrics']['per_class'][label]['recall']:.6f}" for label in LABELS], 1]])
    method_table = markdown_table(["Nhóm", "Kiến trúc / cấu hình", "Dữ liệu fit"], [
        ["TF-IDF", "Hash Word(1,2) + char_wb(3,5), 65.536 bucket mỗi nhánh; nonnegative, sublinear TF, IDF, L2; LinearSVC C=1 balanced max_iter=5000", "Classifier 7.748 Verified; IDF 7.748 hoặc 364.761"],
        ["Chi-Square", "SelectKBest chi2 k=60.000 → LinearSVC; giữ sparse", "Chi2/classifier chỉ 7.748 Train"],
        ["SVD", "TruncatedSVD 100 chiều, n_iter=3 → LinearSVC", "SVD/classifier chỉ 7.748 Train"],
        ["MiniLM supervised", "Encoder đa ngôn ngữ đóng băng 384 chiều → StandardScaler → MLP (128,64); Adam lr=.001, alpha=.0001, batch256, max_iter120", "Scaler/MLP chỉ 7.748 Train"],
        ["PCA / LDA", "Embedding384 → PCA100 hoặc LDA2 → LinearSVC", "Fit chỉ Train; LDA dùng nhãn Train"],
        ["Weak/SSL giai đoạn 02", "Frozen MiniLM 384 → MLP 128,64 + decoder384; warmup30; source3; Adam lr=.001/.0001, wd=.0001", "Ngân sách 10k/50k/100k/357.013; Verified replay; 3 phương pháp mỗi ngân sách"],
        ["Optimization 03", "Warmup30 → source1 → Verified recovery20; lr .001/.0001/.0003; auxiliary gradient cap .2; CPU2 threads, batch256", "3-fold group Train OOF chọn recipe/offset; sau đó fit trên toàn Train-side"]])
    hashes = markdown_table(["Manifest đã đóng băng", "SHA-256"], [[name, sha(ROOT/f"results/v23_multilingual/{name}/frozen_manifest.json")] for name in EXPERIMENTS])
    report = f"""# VietMailGuard V2.3 — Tổng kết nghiên cứu cuối

Ngày tổng hợp: 24/11/2026. Phạm vi: nghiên cứu Development local, không phải Production hoặc Final Evaluation.

## 1. Kết luận chính

**Đã đạt mục tiêu quy mô học weak/SSL trên 300.000 email:** 364.761 bản ghi email unique theo chuẩn hóa toàn văn thực sự tham gia cập nhật tham số MLP, gồm 7.748 Verified Train, 127.743 Weak Label và 229.270 Unlabeled. **Chưa đạt 300.000 mẫu supervised Verified.** Encoder MiniLM đóng băng; tham số được học là MLP/decoder, không phải encoder.

Tối ưu large-scale nâng Macro F1 từ **0,815811 lên 0,944871**, giảm lỗi Normal từ **92 xuống 15**. Tuy nhiên, **TF-IDF + LinearSVC vẫn là baseline tốt nhất trên benchmark hiện tại**, Macro F1 **0,974976**. V23 hash TF-IDF + Chi-Square + LinearSVC đạt cùng Macro F1 và cùng confusion matrix; không có bằng chứng vượt baseline. Giữ nguyên runtime/model hiện hành.

## 2. Dữ liệu, provenance và quyền sử dụng

{data_table}

Các dòng trong bảng mô tả những phạm vi chồng lặp, không được cộng dồn. Tổng bản ghi tham gia học và Validation là **366.699**, gồm 364.761 Train-side + 1.938 Validation. Không có Final Test mới đủ điều kiện; không mở Final Test cũ.

Luồng dữ liệu được ghi trong `usage_accounting.json`: 743.587 email nguồn đầu vào; 332.801 bản sao normalized toàn văn và 38.415 mẫu lỗi/chất lượng không đạt; còn 372.371 ứng viên nguồn. Kết hợp 9.730 Development cũ, có 19 bản ghi nguồn trùng với Development, tạo catalog 382.082 bản ghi (9.730 Verified, 127.744 Weak, 244.608 Unlabeled). Sau các guard, pool bổ sung thực dùng là 357.013: Bruce Guenter 2004 **62.537 Weak**, 2005 **65.206 Weak**, Enron **229.270 Unlabeled**. Catalog lớn hơn không đồng nghĩa classifier đã học toàn bộ catalog.

44 mẫu Verified được mask khỏi cohort thí nghiệm, không xóa bản gốc. 15.339 bản ghi nguồn catalog không tham gia pool cuối. Không cộng nhầm 9.730 reference bị loại khỏi expansion với các mẫu nguồn bị quarantine.

Nhãn nguồn Bruce là spam nhị phân; không được chuyển thành ground truth Spam ba lớp. Partial label dùng tập ứng viên {{Spam, Phishing}}, vẫn có khả năng sai cả giả định loại Normal. Enron vẫn Unlabeled, không tự gán Normal. Quyền dùng corpus là phạm vi nghiên cứu local đã ghi nhận; quyền tái phân phối email/model không được suy ra từ báo cáo này. Không xuất nội dung email, URL cá nhân hoặc dữ liệu nhận dạng vào bộ tổng hợp.

### Phân bố nhãn, ngôn ngữ và đơn vị độc lập

{split_table}

VI có 544 email dịch, gồm 433 Train và 111 Validation; không phải benchmark tiếng Việt bản địa. Metadata origin trống được giữ là `not_declared`, không tự coi là bản địa đã xác minh. Validation VI không có Normal, vì vậy Macro F1 ba lớp tiếng Việt là N/A. Không có Verified ZH trong benchmark: mọi metric chất lượng ba lớp tiếng Trung là N/A.

Trong pool bổ sung, tín hiệu ngôn ngữ là heuristic: EN 261.407; ZH 3.622; VI 1; mixed 261; other/mixed 4.899; unknown 86.823. Không phải thống kê nhãn bản địa đã kiểm chứng. [Bảng chi tiết theo nguồn](final_research/expansion_source_language.csv).

Unique toàn văn không đồng nghĩa mọi mẫu độc lập về template/ngữ nghĩa. Sau giới hạn input có 364.094 đầu vào khác nhau và 667 lần lặp input; 357.013 embedding nguồn chứa 346.656 vector khác nhau, 10.357 lần lặp vector. Các số này không làm thay đổi số email tham gia cập nhật, nhưng giới hạn mức đa dạng biểu diễn và không được gọi là 364.761 nhóm ngữ nghĩa độc lập.

## 3. Phương pháp và khả năng tái lập

{method_table}

Tất cả seed **42**; encoder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` được đóng băng, embedding 384 chiều, max sequence 128 token, tối đa 4.096 ký tự mỗi trường subject/body. Revision: `e04d38a7d68c60a7a95390045400a555127ab033`. Tham số cấu hình đầy đủ nằm ở [training_parameters.csv](final_research/training_parameters.csv), [training_configs.json](final_research/training_configs.json); phiên bản thư viện/protocol giữ tại các JSON nguồn, không cập nhật môi trường trong lần tổng hợp này.

### Giai đoạn 01 — IDF lớn và supervised đối chứng

Bảy candidate: hashed TF-IDF supervised, IDF mở rộng, Chi 2, SVD, MiniLM + MLP, PCA, LDA. IDF lớn học thống kê từ 364.761 email nhưng cả bảy classifier chỉ học nhãn của 7.748 Verified Train. Không gọi giai đoạn này là 364.761 mẫu supervised hoặc neural parameter learning. PCA/LDA/SVD chỉ fit Train, không fit Validation. Baseline V2.2 dùng vocabulary TF-IDF học được tối đa 140.000 đặc trưng; V23 dùng hashing 131.072, nên không mô tả hai pipeline là một.

### Giai đoạn 02 — Cập nhật tham số bằng weak/SSL

Ngân sách nguồn được khóa theo prefix SHA256(seed:canonical_id):10.000,50.000,100.000,357.013, không dựa vào Validation/prediction. Mỗi ngân sách có verified-replay control, weak partial-label, và weak + consistency + denoising. Control khớp số bước replay nhưng không học gradient nguồn. Weak-only chỉ cập nhật trên phần Weak của ngân sách; full weak-only là 127.743  + 7.748 = 135.491 email. Chỉ full weak/SSL dùng toàn 357.013 nguồn + 7.748 Verified = 364.761.

Weak loss là −log(pSpam+pPhishing), trọng số 0,25; consistency KL 0,05 và reconstruction 0,1; EMA 0,99, dropout 0,1. Reliability entropy là heuristic chưa hiệu chuẩn. Full SSL có 357.013 online soft teacher-target rows, chồng lặp chính các Weak/Unlabeled trên; không phải 357.013 nhãn chuẩn mới. Không tạo hard pseudo-label hoặc Human Verified.

### Giai đoạn 03 — Tối ưu có kiểm soát bằng Train OOF

Ba fold group-aware chỉ từ Verified Train,4 recipe và 3 Normal-logit offsets (0;0,5;1) được khai báo trước. Scaler/class weights chỉ fit fold Train. Loại quan hệ nguồn với internal holdout theo exact body, template không rỗng, word-trigram Jaccard≥0,8125 và encoder cosine≥0,98. Pool nguồn giữ lại theo fold: 356.880 / 356.999 / 356.972; 188 bản ghi liên quan ít nhất một holdout là ràng buộc bảo thủ, không phải phán quyết duplicate của con người. Giữ quan hệ parent/translation.

Chọn `guarded_weak_recovery`, Normal-logit offset 0 bằng pooled Train OOF Macro F1 **0,907763**, với Normal FPR ≤ 1%. OOF control tại offset 0 là 0,898536. Đây là score dùng lựa chọn, không phải ước lượng Test độc lập. Chỉ recipe được chọn và matched control được đánh giá outer Validation. [Toàn bộ12 so sánh Train OOF](final_research/train_oof_selection.csv).

Recipe chọn dùng weak loss 0,005 và reconstruction 0,01, không consistency; cân bằng mass theo nguồn Bruce-year và nhân (1−warm-teacher NormalScore)^2. Score này không calibrated probability. Gradient auxiliary xung đột bị chiếu bỏ phần ngược supervised gradient, rồi giới hạn norm ≤ 0,2 trên prediction path. Sau source pass có 20 epoch recovery Verified; giữ tham số đã học, không reset mô hình. Full 357.013 nguồn vẫn cập nhật tham số, kể cả 556 Weak có hệ số partial bằng 0 vẫn tham gia reconstruction. Không sinh nhãn Verified mới.

## 4. Kết quả thực đo trên cùng Development Validation

[Bảng đủ21 candidate](V23_RESEARCH_RESULTS.md), [CSV metrics](final_research/model_comparison.csv), [precision/recall/F1 theo lớp và ngôn ngữ](final_research/per_class_language_metrics.csv).

{focal_table}

![Toàn bộ kết quả V23](final_research/figures/all_v23_models.png)

### Mức cải thiện và đánh đổi

Macro F1 tăng **{delta['macro_f1_absolute']:.9f}**, tương đương **{delta['macro_f1_percentage_points']:.6f} điểm phần trăm** (tương đối **{delta['macro_f1_relative_percent']:.4f}%**). Normal false positives giảm77/92, tức **{delta['normal_fp_reduction_percent']:.4f}%**. Đây là so sánh cùng benchmark, không phải bằng chứng tổng quát ngoài thực tế.

Normal→Spam giảm50→3; Normal→Phishing giảm42→12. Spam Precision tăng0,382716→0,903226, nhưng Spam Recall giảm31/33→28/33 (−9,0909 điểm phần trăm). Phishing Recall giảm337/340→332/340 (−1,4706 điểm phần trăm). Vì vậy không thể kết luận mọi lớp đều cải thiện.

Candidate tối ưu vẫn thấp hơn matched supervised control **0,019203 Macro F1**, và thấp hơn baseline TF-IDF **0,030105**. Normal sai 15, so với control 7 và baseline 1. Dữ liệu lớn đã được học thật nhưng chưa cho thấy lợi ích chất lượng so với supervised đối chứng.

![F1 và lỗi Normal](final_research/figures/optimization_tradeoff.png)

### Những gì có bằng chứng và những gì chỉ là giả thuyết

**Có bằng chứng:** Train Verified mất cân bằng (Normal 6.256, Spam 130, Phishing 1.362); source weak không tách Spam/Phishing; full source 357.013 lớn gấp nhiều lần Verified 7.748. Protocol cũ tạo 92 Normal FP, recipe mới 15. Guard ghi 1.009/1.395 source batch có gradient auxiliary xung đột; max norm ratio được giới hạn 0,2. Tham số prediction path cuối khác matched control (trunk 0 L2≈1,593375; classifier weight L2≈0,364710), nên kết quả không chỉ là đọc corpus hoặc bỏ toàn bộ ảnh hưởng nguồn.

**Giả thuyết phù hợp nhưng chưa chứng minh nhân quả:** áp lực weak {{Spam,Phishing}} có thể đẩy email Normal về các lớp đe dọa; replay/weighting/recovery có thể hạn chế drift; source imbalance, mismatch miền và mất thông tin do128 token có thể góp phần. Nhiều thay đổi được áp dụng cùng lúc; chưa có ablation đủ để phân bổ mức cải thiện cho từng thành phần. Gradient conflict không tự chứng minh nhãn sai, agreement teacher không chứng minh nhãn đúng. Không kết luận SSL nói chung kém hiệu quả.

## 5. Confusion matrices

Thứ tự hàng nhãn thật/cột dự đoán: **Normal, Spam, Phishing**. [CSV cho từng candidate](final_research/confusion_matrices/). Màu trong hình chuẩn hóa từng hàng; số ghi là count thật.

![Confusion matrices](final_research/figures/confusion_comparison.png)

## 6. Thời gian, bộ nhớ và phần cứng

Máy nghiên cứu khoảng 16 GB RAM, RTX 3050 4 GB. Encoder nguồn 357.013 dòng chạy GPU batch 32, **1.309,224649 giây**, RSS process đo mẫu tối đa 2,281467 GiB; PyTorch peak allocated 0,512734 GiB /reserved 0,560547 GiB, không phải toàn VRAM thiết bị. Encoder không được fine-tune. Cache float 32 được tái sử dụng, không trích xuất lại trong optimization hoặc tổng hợp.

Shared large-IDF fit 299,322637giây. Head cũ full weak/SSL 46,433589giây, sampledRSS1,407486GiB, chưa cộng warmup/cache. Optimization selected source 12,174796giây + recovery 1,149289giây; shared warmup 1,715129giây được hạch toán ở control; peakRSS head ≈ 1,450096GiB. Các head dùng CPU2threads. [Tài nguyên từng stage](final_research/resource_stages.csv), [thời gian/artifact cho21 candidate](final_research/model_comparison.csv).

Không cộng peakRAM của các stage thành RAM tổng; không coi sampledRSS là peak chính xác tuyệt đối. Latency sklearn/PyTorch trên embedding cache chỉ là head/projection, loại thời gian encoder; TF-IDF prediction gồm vectorization theo scope đã lưu. Không dùng hai loại latency này để kết luận tốc độ end-to-end. Những timing/peakRAM không được lưu ghi N/A. Kích thước head artifact không bao gồm encoder/cache; tất cả byte count và SHA nằm trong bảng CSV, không giả định head nhỏ là tổng package nhỏ.

## 7. Benchmark và giới hạn khoa học

V22 TF-IDF baseline và 21 kết quả V23 dùng cùng Train/Validation content hashes, 7.748/1.938. Có thể so sánh thăm dò trên cùng split, nhưng các phương pháp, ngân sách/steps khác nhau và benchmark đã được xem qua nhiều nghiên cứu. Không thể coi lần chọn bằng Train OOF là xóa lịch sử exposure outer Validation.

V21 Macro F1 **{context[1]['macro_f1']:.6f}** dùng Validation 6.709, khác cohort. TREC06C **{context[2]['macro_f1']:.6f}** dùng 8.862 Validation và taxonomy Ham/Spam nhị phân, không đo Phishing ba lớp. Không xếp hạng chung hoặc gọi mức chênh là tăng/giảm chất lượng V23. [Bảng ngữ cảnh benchmark](final_research/benchmark_context.csv).

Validation Spam chỉ 33 mẫu thuộc 13 nhóm. Một lỗi làm Recall đổi 1/33≈3,03 điểm phần trăm; mẫu trong nhóm không độc lập, nên không lấy 33 như 33 thử nghiệm độc lập để khẳng định độ chắc chắn. Còn 342 cảnh báo template thường và 20 liên hệ chưa đủ bằng chứng; đã được phân loại là không đủ để coi chắc chắn leakage, không được tuyên bố đã chứng minh độc lập ngữ nghĩa tuyệt đối. Không thay mask/split trong tổng hợp này.

Chất lượng pseudo/weak chưa có kiểm định nhãn độc lập đủ phạm vi. Không có Final Test mới ba lớp; không đọc Final Test cũ. Không chứng minh năng lực Phishing tiếng Trung thực tế hoặc VI bản địa; TREC binary/synthetic challenge không thay thế benchmark native. Không tuyên bố Production Ready, Final Evaluation PASS hoặc supervised Verified 300.000.

## 8. Integrity và bảo toàn

**{len(expected)} file bảo toàn khớp SHA-256 trước/sau**, gồm 207 artifact cũ, các manifest V23 và tệp hợp lệ/lần thử lỗi đã lưu. [Snapshot trước](final_research/integrity_before.json), [kết quả sau](final_research/integrity_after.json). Số 42/47/50 là entries của từng frozen manifest, không cộng đơn giản thành union 379 vì còn manifest và archive.

{hashes}

Lần CV bị lỗi empty-template được bảo toàn trong `*_aborted_empty_template`, dừng trước outer Validation; không đưa vào 21 candidate hợp lệ hoặc lấy metric từ lần lỗi. Các frozen manifest cũ được giữ nguyên. Báo cáo không deserialize model, không fit/predict, không chạy lại evaluation. Chỉ kiểm tra số học 22 confusion matrices đã lưu (21 V23 + baseline V22), checksum và tạo bảng/hình mới. Manifest mới cho bộ tài liệu tổng hợp tách biệt tại `final_research/manifest.json`, SHA tại `manifest.sha256`.

## 9. Sử dụng cho đồ án

Dùng bảng dữ liệu để mô tả quy mô/nguồn nhãn, bảng 21 candidate để trình bày thực nghiệm, bảng Train OOF để giải thích lựa chọn có kiểm soát, hình tradeoff/confusion để phân tích lỗi. Bản SVG thích hợp chèn báo cáo vector; PNG 180 dpi dùng trình chiếu. Các bảng không chứa email và không tự cấp quyền công bố dữ liệu gốc.

Tái tạo bảng/hình offline từ các artifact local đã lưu:

```powershell
.\\.venv\\Scripts\\python.exe -B scripts/v23/consolidate_research.py
.\\.venv\\Scripts\\python.exe -B -m pytest tests/test_v23_research_consolidation.py -q -p no:cacheprovider
```

Không có bước train, inference, download hoặc gọi API trong lệnh tổng hợp. Script yêu cầu các manifest/results/cache metadata local để xác minh, không thay bằng số liệu hardcode nếu file thiếu. Khóa hiện trạng nghiên cứu; không tối ưu thêm trên Validation này.

Kiểm thử bộ tổng hợp: **3 passed**, không skipped/deselected. Ba test kiểm tra số học metrics/matrix, chặn xếp hạng benchmark khác nhau và chặn đọc đường dẫn Test/secret/ngoài project. Đây là kiểm thử tổng hợp báo cáo, không phải chạy lại regression ứng dụng hoặc kiểm định độ chính xác model. `git diff --check`: PASS; cảnh báo LF/CRLF của `.gitignore` có sẵn không phải lỗi nội dung.

Nguồn kiểm chứng chính:

| Phần | File gốc local |
|---|---|
| Quy mô, chất lượng, nhãn/ngôn ngữ | [usage_accounting.json](../../results/v23_multilingual/development_01/usage_accounting.json) |
| 7 candidate IDF/supervised | [development metrics](../../results/v23_multilingual/development_01/metrics.json) |
| 12 candidate weak/SSL | [expansion metrics](../../results/v23_multilingual/classifier_expansion_02/metrics.json) |
| Chọn bằng Train OOF | [selection.json](../../results/v23_multilingual/large_scale_optimization_03/selection.json) |
| Optimization và đối chứng | [optimization metrics](../../results/v23_multilingual/large_scale_optimization_03/metrics.json) |
| Baseline cùng benchmark | [V22 pilot metrics](../../results/v22_multilingual/experiments/group_aware_supervised_pilot_01/metrics.json) |

Các link local trên trỏ vào aggregate metadata/metrics, không phải email hoặc Final Test. Các hash chi tiết để kiểm chứng toàn bộ nằm trong manifest, không chỉ những link tóm tắt này.

**Trạng thái:** LARGE_SCALE_WEAK_SSL_PARAMETER_LEARNING ≥300.000: YES; SUPERVISED_VERIFIED ≥300.000: NO; INDEPENDENT_FINAL_EVALUATION: NOT READY; RUNTIME_CHANGE: NO.
"""
    (parent/"V23_FINAL_RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    appendix = f"""# VietMailGuard V2.3 — Bảng kết quả nghiên cứu

Ngày 24/11/2026. Chỉ tổng hợp metrics đã lưu; không huấn luyện/đánh giá lại. Xem [báo cáo phương pháp và giới hạn](V23_FINAL_RESEARCH_REPORT.md).

## Dữ liệu tham gia

{data_table}

{split_table}

364.761 là email tham gia cập nhật MLP weak/SSL; chỉ 7.748 có nhãn Verified dùng supervised loss. 357.013 nguồn =127.743 Weak +229.270 Unlabeled, không được tính là Verified. 366.699 gồm cả 1.938 Validation. Không có Final Test độc lập.

## Toàn bộ 21 candidate V23 hợp lệ

Giai đoạn 01:7 candidate;02:12;03:2. Cùng benchmark 1.938; hàng là các thực nghiệm riêng, không cộng classifier rows giữa các hàng. Control giai đoạn 02 chỉ cập nhật Verified dù mang tên ngân sách nguồn. Weak-only chỉ dùng phần Weak. Giai đoạn 01 IDF lớn không làm tăng số mẫu cập nhật classifier.

{result_table}

Baseline TF-IDF + LinearSVC lưu từV22: Accuracy 0,995356; Macro F1 **0,974976**, bằng candidate V23 Chi 2. Giữ baseline; optimizedMLP 0,944871 vẫn thấp hơn. Không so trực tiếp vớiV21 Validation 6.709 hoặcTREC binary.

## Thông số huấn luyện

{method_table}

Seed 42; cấu hình đầy đủ của từng candidate tại [training_parameters.csv](final_research/training_parameters.csv). [Các cấu hình gốc đã chụp](final_research/training_configs.json). Việc thiếu timing/RAM được giữ N/A, không suy đoán.

## Danh mục bảng và hình

| File | Nội dung |
|---|---|
| [model_comparison.csv](final_research/model_comparison.csv) | 21 kết quả, counts cập nhật classifier, thời gian, artifact bytes/SHA |
| [per_class_language_metrics.csv](final_research/per_class_language_metrics.csv) | Precision/Recall/F1 từng lớp EN/VI/ZH, thiếu support là N/A |
| [data_roles.csv](final_research/data_roles.csv) | Verified/Weak/Unlabeled theo vai trò |
| [verified_split_distribution.csv](final_research/verified_split_distribution.csv) | Split × nhãn × ngôn ngữ × origin |
| [expansion_source_language.csv](final_research/expansion_source_language.csv) | Nguồn × trạng thái × tín hiệu ngôn ngữ |
| [data_accounting.json](final_research/data_accounting.json) | Luồng raw/duplicate/quality/cohort thực tế |
| [resource_stages.csv](final_research/resource_stages.csv) | Resource scope theo stage và encodercache |
| [train_oof_selection.csv](final_research/train_oof_selection.csv) | 12 so sánh Train OOF; không phải outer metric |
| [benchmark_context.csv](final_research/benchmark_context.csv) | Các benchmark có/không so trực tiếp được |
| [improvement_analysis.json](final_research/improvement_analysis.json) | F1+0,129060; FP−77; tradeoffRecall |
| [confusion_matrices/](final_research/confusion_matrices/) | 21 ma trận CSV, hàng thật/cột dự đoán Normal/Spam/Phishing |
| [all_v23_models.png](final_research/figures/all_v23_models.png) / [SVG](final_research/figures/all_v23_models.svg) | 21 Macro F1 cùngValidation |
| [optimization_tradeoff.png](final_research/figures/optimization_tradeoff.png) / [SVG](final_research/figures/optimization_tradeoff.svg) | F1 và lỗi Normal |
| [confusion_comparison.png](final_research/figures/confusion_comparison.png) / [SVG](final_research/figures/confusion_comparison.svg) | 4 ma trận trọng tâm |
| [results_snapshot.json](final_research/results_snapshot.json) | Bản sao aggregateJSON của3 experiment, không có email |
| [integrity_after.json](final_research/integrity_after.json) | SHA trước/sau của379 file bảo toàn |
| [manifest.json](final_research/manifest.json) | Checksums bộ tài liệu mới, không thay frozen manifests |

## Kết luận dùng trong đồ án

Đạt quy mô tham số weak/SSL trên 300.000 email; chưa đạt 300.000 supervised Verified. Tối ưu giảm Normal false positives đáng kể so với cấu hình weak/SSL trước nhưng làm giảm Recall Spam/Phishing; matched supervised control vàTF-IDF baseline vẫn tốt hơn. Chỉ có 33 Spam/13 nhóm ởValidation; không suy rộng thành chất lượng production/đa ngôn ngữ bản địa. Không kiểm địnhFinal Test, không có phê duyệt nhãn mới hoặc thay đổi runtime.
"""
    (parent/"V23_RESEARCH_RESULTS.md").write_text(appendix, encoding="utf-8")


def create_figures(rows, baseline, delta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    folder = OUT/"figures";folder.mkdir(exist_ok=True)
    def save(fig, stem):
        fig.savefig(folder/(stem+".png"), dpi=180);fig.savefig(folder/(stem+".svg"));plt.close(fig)
    colors = {EXPERIMENTS[0]: "#64748B", EXPERIMENTS[1]: "#D97706", EXPERIMENTS[2]: "#2563EB"}
    fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    y = np.arange(len(rows));bars = ax.barh(y, [r["macro_f1"] for r in rows], color=[colors[r["experiment"]] for r in rows])
    ax.set_yticks(y, [f"{EXPERIMENTS.index(r['experiment'])+1:02d} | {r['candidate']}" for r in rows]);ax.invert_yaxis()
    ax.set_xlim(0, 1.07);ax.set_xlabel("Development Macro F1; same 1,938-row Validation")
    ax.set_title("All 21 valid V2.3 outer-Validation results; not independent Test scores")
    ax.axvline(baseline["metrics"]["macro_f1"], color="#0F172A", linestyle="--", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, row in zip(bars, rows):ax.text(row["macro_f1"]+.006, bar.get_y()+bar.get_height()/2, f"{row['macro_f1']:.6f}", va="center", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=.5))
    save(fig, "all_v23_models")
    old = next(r for r in rows if r["candidate"] == "weak_partial_consistency_denoise_357013")
    new = next(r for r in rows if r["candidate"] == "guarded_weak_recovery")
    control = next(r for r in rows if r["experiment"] == EXPERIMENTS[2] and r["candidate"] == "verified_control")
    names = ["Previous weak/SSL", "Optimized large-scale", "New matched control", "Saved TF-IDF baseline"]
    f1 = [old["macro_f1"], new["macro_f1"], control["macro_f1"], baseline["metrics"]["macro_f1"]]
    fp = [old["normal_false_positives"], new["normal_false_positives"], control["normal_false_positives"], sum(baseline["metrics"]["confusion_matrix"][0][1:])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, values, title in [(axes[0], f1, "Macro F1"), (axes[1], fp, "Normal → Spam/Phishing")]:
        bars = ax.bar(range(4), values, color=["#D97706", "#2563EB", "#64748B", "#475569"])
        ax.set_xticks(range(4), names, rotation=15, ha="right");ax.set_title(title);ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):ax.annotate(f"{value:.6f}" if ax is axes[0] else str(value), (bar.get_x()+bar.get_width()/2, value), xytext=(0, 4), textcoords="offset points", ha="center")
    axes[0].set_ylim(0, 1.08);axes[1].set_ylim(0, 108)
    fig.suptitle("Same Development Validation; no independent Final Test")
    save(fig, "optimization_tradeoff")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    snapshots = read("results/v23_multilingual/classifier_expansion_02/metrics.json")
    old_m = next(r["metrics"] for r in snapshots["candidates"] if r["name"] == old["candidate"])
    latest = read("results/v23_multilingual/large_scale_optimization_03/metrics.json")
    for ax, title, metric in zip(axes.flat, names,
            [old_m, next(r["metrics"] for r in latest["candidates"] if r["name"] == new["candidate"]),
             next(r["metrics"] for r in latest["candidates"] if r["name"] == "verified_control"), baseline["metrics"]]):
        cm = np.asarray(metric["confusion_matrix"]);rate = cm/cm.sum(1, keepdims=True)
        ax.imshow(rate, vmin=0, vmax=1, cmap="Blues");ax.set_title(title)
        ax.set_xticks(range(3), LABELS);ax.set_yticks(range(3), LABELS);ax.set_xlabel("Predicted");ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if rate[i,j]>.5 else "black")
    fig.suptitle("Row-normalized colors; annotations are actual counts")
    save(fig, "confusion_comparison")


if __name__ == "__main__":
    with offline_boundary(ROOT):main()
