"""Isolated Chi-square research; Train-only selection, fixed Development Validation."""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .development_training import make_tfidf_pipeline, sha256_file, verify_and_load_development
from .dimensionality_experiment import MemoryMonitor, memory_snapshot, latency, sparse_transform
from .multilingual_dl_experiment import ExperimentBlocked, _json, evaluate, offline_boundary
from .paths import resolve_v21_paths

EXPERIMENT = "feature_selection_03"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def context(paths):
    config = read_json(paths.config / "experiments" / f"{EXPERIMENT}.json")
    if config["download_allowed"] or config["final_test_allowed"]:
        raise ExperimentBlocked("Offline Development contract violated")
    runtime = read_json(paths.config / "runtime_model.json")
    for key in ("artifact", "training_config", "validation_metrics"):
        if sha256_file(paths.project_root / runtime[key]) != runtime[key + "_sha256"]:
            raise ExperimentBlocked("Baseline dependency checksum mismatch")
    manifest, train, validation = verify_and_load_development(paths, 4096)
    signature = {"experiment_config": sha256_file(paths.config / "experiments" / f"{EXPERIMENT}.json"),
                 "runtime": sha256_file(paths.config / "runtime_model.json"),
                 "train": manifest["output_sha256"]["train"],
                 "validation": manifest["output_sha256"]["validation"]}
    return config, runtime, train, validation, signature


def select_sparse(matrix, labels, k):
    if (not sparse.issparse(matrix) or not np.isfinite(matrix.data).all()
            or (matrix.data < 0).any() or not 0 < k <= matrix.shape[1]):
        raise ExperimentBlocked("Chi-square requires finite nonnegative sparse TF-IDF and valid k")
    selector = SelectKBest(chi2, k=k)
    selected = selector.fit_transform(matrix, labels)
    if not sparse.issparse(selected):
        raise ExperimentBlocked("Feature selection must remain sparse")
    return selector, selected


def related_groups(records):
    """Union all exact/normalized/template/group/parent relations before Train CV."""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    ids = {row["id"]: i for i, row in enumerate(records)}
    for i, row in enumerate(records):
        for key in ("final_group_id", "content_hash", "normalized_content_hash", "template_hash"):
            if row.get(key):
                token = (key, row[key])
                other = owners.setdefault(token, i)
                parent[find(i)] = find(other)
        if row.get("parent_id"):
            if row["parent_id"] not in ids:
                raise ExperimentBlocked("Train CV parent is missing")
            parent[find(i)] = find(ids[row["parent_id"]])
    return np.array([find(i) for i in range(len(records))])


def group_folds(frame, groups, n_splits, seed):
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        frame, frame.label, groups))
    for left, right in folds:
        if set(groups[left]) & set(groups[right]):
            raise ExperimentBlocked("Related group crossed a CV fold")
        if set(frame.iloc[left].label) != {"normal", "spam", "phishing"} or set(frame.iloc[right].label) != {"normal", "spam", "phishing"}:
            raise ExperimentBlocked("CV fold missing a class")
    return folds


def run_cv(train, paths, config, training):
    resources = memory_snapshot()
    if resources["free_ram_gib"] is None or resources["free_ram_gib"] < config["cv_min_free_ram_gib"]:
        return {"status": "BLOCKED", "reason": "Insufficient free RAM for repeated full fold pipeline fits",
                "resources": resources, "required_free_ram_gib": config["cv_min_free_ram_gib"]}
    records = []
    with (paths.splits / "development_bootstrap/train.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append({key: row[key] for key in ("id", "final_group_id", "content_hash", "normalized_content_hash", "template_hash", "parent_id")})
    if [row["id"] for row in records] != train.id.tolist():
        raise ExperimentBlocked("CV row order differs from verified Train")
    groups = related_groups(records)
    folds = group_folds(train, groups, config["cv_folds"], config["seed"])
    scores = []
    for k in config["k"]:
        for c in config["C"]:
            fold_scores = []
            for left, right in folds:
                # Fresh vectorizers, selector and SVM in every candidate/fold.
                pipeline = make_tfidf_pipeline(training)
                pipeline.set_params(classifier__C=c)
                pipeline.steps.insert(-1, ("select", SelectKBest(chi2, k=k)))
                start = time.perf_counter()
                pipeline.fit(train.iloc[left], train.iloc[left].label)
                prediction = pipeline.predict(train.iloc[right])
                fold_scores.append({"macro_f1": float(f1_score(train.iloc[right].label, prediction, average="macro")),
                                    "phishing_recall": float(recall_score(train.iloc[right].label, prediction, labels=["phishing"], average="macro")),
                                    "seconds": time.perf_counter()-start})
                del pipeline
                gc.collect()
            scores.append({"k": k, "C": c, "folds": fold_scores,
                           "macro_f1": float(np.mean([item["macro_f1"] for item in fold_scores])),
                           "phishing_recall": float(np.mean([item["phishing_recall"] for item in fold_scores]))})
    best = sorted(scores, key=lambda row: (-row["macro_f1"], -row["phishing_recall"], row["k"], row["C"]))[0]
    return {"status": "PASS", "scores": scores, "selected": {"k": best["k"], "C": best["C"]},
            "selection_data": "Train CV only", "related_groups": len(set(groups)), "resources": resources}


def candidate(name, paths=None):
    paths = paths or resolve_v21_paths()
    result_dir = paths.results / "experiments" / EXPERIMENT
    model_dir = paths.model / "experiments" / EXPERIMENT
    with offline_boundary(paths.project_root), threadpool_limits(2):
        config, runtime, train, validation, signature = context(paths)
        result_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / f"{name}.json"
        if output_path.exists():
            previous = read_json(output_path)
            if previous["fingerprint"] != signature:
                raise ExperimentBlocked("Checkpoint config/data changed")
            if previous.get("artifact") and sha256_file(paths.project_root / previous["artifact"]) != previous["artifact_sha256"]:
                raise ExperimentBlocked("Checkpoint artifact checksum mismatch")
            return previous
        base = joblib.load(paths.project_root / runtime["artifact"])
        training = read_json(paths.project_root / runtime["training_config"])
        resources = memory_snapshot()
        record = {"candidate": name, "fingerprint": signature, "resources": resources}
        if name == "B_train_cv":
            cv = run_cv(train, paths, config, training)
            _json(result_dir / "train_cv_selection.json", cv)
            if cv["status"] != "PASS":
                record.update(cv)
                _json(output_path, record)
                return record
            k, c = cv["selected"]["k"], cv["selected"]["C"]
        elif name != "baseline":
            k, c = int(name.removeprefix("A_chi2_")), training["tfidf"]["svm"]["C"]
        if resources["free_ram_gib"] is None or resources["free_ram_gib"] < config["candidate_min_free_ram_gib"]:
            record.update(status="BLOCKED", reason="Insufficient RAM for sparse candidate")
            _json(output_path, record)
            return record
        with MemoryMonitor() as monitor:
            if name == "baseline":
                model = base
                fit_seconds = None
                dimensions = sum(len(vectorizer.vocabulary_) for _, vectorizer in base.named_steps["tfidf"].transformer_list)
                selection = None
            else:
                matrix = sparse_transform(base[:-1], train, config["transform_batch_rows"])
                dimensions = matrix.shape[1]
                start = time.perf_counter()
                selector, reduced = select_sparse(matrix, train.label, k)
                svm = clone(base[-1]).set_params(C=c)
                svm.fit(reduced, train.label)
                fit_seconds = time.perf_counter()-start
                model = Pipeline(base.steps[:-1] + [("select", selector), ("classifier", svm)])
                indices = selector.get_support(indices=True)
                word_count = len(base.named_steps["tfidf"].transformer_list[0][1].vocabulary_)
                scores = selector.scores_[indices]
                finite = scores[np.isfinite(scores)]
                selection = {"word_features": int((indices < word_count).sum()),
                             "character_features": int((indices >= word_count).sum()),
                             "chi2_score_quantiles": np.quantile(finite, [0,.25,.5,.75,1]).tolist(),
                             "selected_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
                artifact = model_dir / f"{name}.joblib"
                if artifact.exists():
                    raise ExperimentBlocked("Orphan artifact exists; refusing overwrite")
                joblib.dump(model, artifact, compress=3)
                record.update(artifact=artifact.relative_to(paths.project_root).as_posix(), artifact_sha256=sha256_file(artifact), C=c, k=k)
                del matrix, reduced
                gc.collect()
            prediction = model.predict(validation)
            metrics = evaluate(validation, prediction)
            sample = validation.groupby("language", sort=True).head(8)
            timing = latency(model.predict, sample, config["latency_repeats"], "subject/body composition + TF-IDF + selection if present + classifier; warm batch")
            if name != "baseline":
                reloaded = joblib.load(paths.project_root / record["artifact"])
                if not np.array_equal(reloaded.predict(sample), model.predict(sample)):
                    raise ExperimentBlocked("Artifact reload prediction mismatch")
            else:
                saved = read_json(paths.project_root / runtime["validation_metrics"])
                expected = next(row["metrics"]["macro_f1"] for row in saved["candidates"] if row["candidate"] == runtime["candidate_id"])
                if not np.isclose(metrics["macro_f1"], expected, atol=1e-12, rtol=0):
                    raise ExperimentBlocked("Reproduced baseline disagrees with saved Validation")
                record.update(artifact=runtime["artifact"], artifact_sha256=runtime["artifact_sha256"])
        record.update(status="PASS", metrics=metrics, input_dimensions=dimensions,
                      output_dimensions=dimensions if name == "baseline" else k,
                      fit_selector_classifier_seconds=fit_seconds, selection=selection, latency=timing,
                      memory=monitor.result, artifact_bytes=(paths.project_root / record["artifact"]).stat().st_size,
                      train_rows=len(train), validation_rows=len(validation), leakage="PASS")
        _json(output_path, record)
        return record


def summarize(paths=None):
    paths = paths or resolve_v21_paths()
    config = read_json(paths.config / "experiments" / f"{EXPERIMENT}.json")
    directory = paths.results / "experiments" / EXPERIMENT
    rows = [read_json(directory / f"{name}.json") for name in ["baseline"] + [f"A_chi2_{k}" for k in config["k"]] + ["B_train_cv"]]
    output = {"scope": "development_validation_only", "candidates": rows, "runtime_changed": False, "heldout_opened": False}
    _json(directory / "validation_metrics.json", output)
    return output


def feature_diagnostics(paths, rows):
    """Aggregate baseline coefficient retention without publishing email tokens."""
    baseline = next(row for row in rows if row["candidate"] == "baseline")
    artifact = paths.project_root / baseline["artifact"]
    if sha256_file(artifact) != baseline["artifact_sha256"]:
        raise ExperimentBlocked("Baseline checksum changed before feature analysis")
    base = joblib.load(artifact)
    weights = np.abs(base[-1].coef_)
    result = {}
    for row in rows:
        if row["status"] != "PASS" or not row.get("selection"):
            continue
        artifact = paths.project_root / row["artifact"]
        if sha256_file(artifact) != row["artifact_sha256"]:
            raise ExperimentBlocked("Candidate checksum changed before feature analysis")
        model = joblib.load(artifact)
        indices = model.named_steps["select"].get_support(indices=True)
        if hashlib.sha256(indices.tobytes()).hexdigest() != row["selection"]["selected_indices_sha256"]:
            raise ExperimentBlocked("Selected feature indices differ from checkpoint")
        result[row["candidate"]] = {
            "word_features": row["selection"]["word_features"],
            "character_features": row["selection"]["character_features"],
            "baseline_absolute_coefficient_mass_retained": {
                str(label): float(weights[i, indices].sum() / weights[i].sum())
                for i, label in enumerate(base[-1].classes_)
            },
        }
        del model
    return result


def write_report(output, root, diagnostics):
    """Render measured checkpoints; never insert expected scores as results."""
    rows = [row for row in output["candidates"] if row["status"] == "PASS"]
    base = next(row for row in rows if row["candidate"] == "baseline")
    cv = next(row for row in output["candidates"] if row["candidate"] == "B_train_cv")
    config = read_json(root / "config/v21/experiments/feature_selection_03.json")
    best = max((row for row in rows if row["candidate"] != "baseline"), key=lambda row: row["metrics"]["macro_f1"])
    delta = best["metrics"]["macro_f1"] - base["metrics"]["macro_f1"]
    lines = [
        "# Experiment 03 — Chi-square Feature Selection",
        "",
        "Thí nghiệm Development riêng, không thay runtime, không đánh giá Final Test.",
        "",
        f"**Kết luận:** Macro F1 tốt nhất của feature selection = {best['metrics']['macro_f1']:.9f}; "
        f"baseline = {base['metrics']['macro_f1']:.9f}; chênh lệch = {delta:+.9f}.",
        "Giữ nguyên mô hình chính. Đây là so sánh quan sát trên Development Validation đã dùng trong phát triển, "
        "không phải kiểm định cải thiện có ý nghĩa thống kê hoặc kết quả Test độc lập.",
        "",
        "## Dữ liệu và protocol",
        "",
        f"- Train: {base['train_rows']:,}; Validation: {base['validation_rows']:,}; assignment kế thừa được giữ nguyên.",
        "- Chỉ đọc Development Master/Train/Validation. Kiểm tra hash, provenance/eligibility, "
        "exact/normalized/template fingerprint, final_group_id và parent/translation: PASS.",
        "- TF-IDF frozen baseline đã fit trên đúng Train: Word (1,2), tối đa 60.000; "
        "char_wb (3,5), tối đa 80.000; min_df=2, sublinear_tf=True, float32.",
        "- Subject và body cắt tối đa 4.096 ký tự mỗi trường theo preprocessing Development hiện tại; "
        "ghép subject + newline + body. Không đổi tiền xử lý giữa baseline và candidate.",
        "- A tái sử dụng TF-IDF baseline chỉ học từ Train để cô lập tác động selection; "
        "chi2 và classifier mới chỉ fit Train. Sparse TF-IDF không chuyển toàn bộ thành dense.",
        f"- LinearSVC: C=1.0, class_weight=balanced, max_iter=5000, random_state={config['seed']}. "
        f"CPU, giới hạn threadpool={config['cpu_threads']}; xử lý transform theo lô {config['transform_batch_rows']} dòng.",
        "- B (nếu qua resource gate): StratifiedGroupKFold 3 folds, union các quan hệ group/hash/template/parent; "
        "k × C∈{0.1,1,10}; fit lại vectorizer, selector và SVM trong từng fold; chọn bằng Macro F1, "
        "tie-break Phishing Recall rồi k/C nhỏ hơn. Chỉ đánh giá candidate đã chốt trên Validation.",
        "- Các mức k của A được cố định trước; không lặp tối ưu bằng Validation. Không có calibration/probability.",
        "",
        "## Kết quả Development Validation",
        "",
        "| Cấu hình | Chiều trước → sau | Accuracy | Macro F1 | Δ Macro F1 | Phishing Recall | Normal → Spam/Phishing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['candidate']} | {row['input_dimensions']:,} → {row['output_dimensions']:,} | "
                     f"{m['accuracy']:.6f} | {m['macro_f1']:.9f} | {m['macro_f1']-base['metrics']['macro_f1']:+.9f} | "
                     f"{m['phishing_recall']:.6f} | {m['normal_to_spam_or_phishing']} |")
    lines += ["", "## Thời gian, RAM và artifact", "",
              "| Cấu hình | Fit selector + SVM (s) | Warm E2E median (ms/email) | Peak RSS lấy mẫu (GiB) | Native process peak (GiB) | Artifact (MiB) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        fit = row["fit_selector_classifier_seconds"]
        lines.append(f"| {row['candidate']} | {'không refit' if fit is None else f'{fit:.3f}'} | "
                     f"{row['latency']['median_ms_per_email']:.3f} | {row['memory']['sampled_peak_rss_gib']:.3f} | "
                     f"{row['memory']['after']['process_peak_rss_gib']:.3f} | {row['artifact_bytes']/2**20:.3f} |")
    lines += ["", "Fit trên chỉ gồm chi2 + SVM; không gồm đọc/kiểm tra dữ liệu, transform TF-IDF, "
              "serialize hoặc chi phí fit TF-IDF baseline trước đây. Không dùng số này làm tổng thời gian huấn luyện từ đầu.",
              "Latency cùng 16 dòng (8 EN + 8 VI), warm-up một lần, 5 lần lặp, threadpool 2: "
              "gồm ghép nội dung → TF-IDF → selection nếu có → SVM; không gồm disk I/O/SQLite/UI, cold model loading hoặc truncation CSV. "
              "Đây là throughput mỗi email trong batch nhỏ, không phải SLA cho email dài bất kỳ.",
              "RSS lấy mẫu 100 ms trong mỗi subprocess riêng; native peak tính cả khởi tạo/đọc dữ liệu. "
              "Baseline không huấn luyện lại nên peak của baseline là reload/Validation; không so nó với peak fit "
              "candidate để kết luận giảm hay tăng RAM huấn luyện baseline. Ứng dụng khác có thể làm nhiễu đo thời gian/RAM. "
              "Artifact candidate nén joblib mức 3, "
              "baseline dùng kích thước file hiện có; chênh lệch kích thước chịu ảnh hưởng serialization.",
              "", "## Đặc trưng được giữ", "",
              "| Cấu hình | Word / 60.000 | Character / 80.000 | Giữ abs(coef) baseline Normal | Spam | Phishing |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, data in diagnostics.items():
        mass = data["baseline_absolute_coefficient_mass_retained"]
        lines.append(f"| {name} | {data['word_features']:,} | {data['character_features']:,} | "
                     f"{mass['normal']:.2%} | {mass['spam']:.2%} | {mass['phishing']:.2%} |")
    lines += ["", "Chi2 xếp hạng liên hệ đơn biến với nhãn trên Train; không đo quan hệ nhân quả hoặc "
              "chất lượng từng lớp. Các tỷ lệ |coef| trên là tổng trị tuyệt đối trọng số baseline còn lại tại "
              "các cột được chọn, không phải tỷ lệ confidence, explained variance hay đóng góp xác suất. "
              "Chi2 score quantiles và SHA-256 indices lưu trong JSON; không xuất token/email thô.",
              "Selector giảm số cột đầu vào SVM nhưng vẫn giữ toàn bộ vocabulary TF-IDF để transform rồi "
              "lọc cột. Vì vậy không kỳ vọng E2E latency giảm tỷ lệ với k; preprocessing vẫn tốn phần lớn công việc.",
              "", "## Per-class và confusion matrix", "",
              "Thứ tự hàng thật/cột dự đoán: normal, spam, phishing."]
    for row in rows:
        lines += ["", f"### {row['candidate']}", "",
                  "| Lớp | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for label, values in row["metrics"]["per_class"].items():
            lines.append(f"| {label} | {values['precision']:.6f} | {values['recall']:.6f} | {values['f1']:.6f} | {values['support']} |")
        lines += ["", "```text", *[str(line) for line in row["metrics"]["confusion_matrix"]], "```"]
    lines += ["", "## Theo ngôn ngữ", "",
              "| Cấu hình | EN Macro F1 (6.529) | VI translated Accuracy (180) | VI Spam Recall (8) | VI Phishing Recall | ZH |",
              "|---|---:|---:|---:|---|---|"]
    for row in rows:
        languages = row["metrics"]["by_language"]
        lines.append(f"| {row['candidate']} | {languages['en']['macro_f1']:.6f} | "
                     f"{languages['vi']['accuracy']:.6f} | {languages['vi']['per_class']['spam']['recall']:.6f} | N/A (0 mẫu) | N/A (0 mẫu) |")
    lines += ["", "VI chỉ có 172 Normal, 8 Spam, 0 Phishing và đều là bản dịch; không có benchmark VI bản địa. "
              "Không báo Macro F1 ba lớp VI hoặc khả năng ZH. JSON giữ per-language/per-origin/support đầy đủ; "
              "origin trống của EN được ghi not_declared, không suy diễn native.",
              "", "## B — Train group-aware CV", "", f"**{cv['status']}**."]
    if cv["status"] == "BLOCKED":
        lines += [f"RAM trống khi kiểm tra: {cv['resources']['free_ram_gib']:.3f} GiB; "
                  f"ngưỡng cấu hình trước chạy: {cv['required_free_ram_gib']:.1f} GiB. "
                  "Không fit fold thật, không có score Train CV, không có candidate B hoặc lựa chọn k/C tối ưu. "
                  "Ngưỡng là headroom bảo thủ cho nhiều lần fit vocabulary trong fold, không phải khẳng định "
                  "mức RAM tối thiểu tuyệt đối của sklearn. Không thay bằng CV dùng vectorizer toàn Train."]
    else:
        lines += ["Kết quả và lựa chọn trước Validation lưu tại train_cv_selection.json."]
    lines += ["", "## Diễn giải và bảo toàn", "",
              "Ở k=60.000 và 100.000, confusion matrix giống nhau: bắt thêm 1 Phishing nhưng bỏ sót thêm "
              "1 Spam so với baseline, làm Macro F1 thấp hơn dù Accuracy bằng nhau. k nhỏ hơn làm mất thêm "
              "tín hiệu phân biệt Spam. Đây là bằng chứng trade-off quan sát; chưa chứng minh nguyên nhân của từng lỗi.",
              "Không có cấu hình A vượt baseline về Macro F1. Không chọn/thay model chính từ những kết quả này. "
              "CV B chưa thực hiện nên không kết luận mọi k/C khác đều thất bại.",
              "V2, V2.1 runtime, Experiment 01/02 và dữ liệu được giữ nguyên; không truy cập Final Test, "
              "không Transformer mới, không download, không Gemini, không commit/push.",
              "", "## Tái lập", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -u scripts/v21/feature_selection_03.py --run",
              ".\\.venv\\Scripts\\python.exe -m pytest tests/test_v21_feature_selection.py -q",
              "```", "",
              "Checkpoint được xác minh hash config/Train/Validation/runtime/artifact, không tự ghi đè model có sẵn. "
              "BLOCKED CV cũng là checkpoint của lần đo này; thử CV ở điều kiện khác cần run namespace/config riêng, "
              "không sửa kết quả đã đo. Chi tiết phiên bản, hash code/artifact: run_manifest.json.", ""]
    lines += ["## File của nhiệm vụ", "",
              "- config/v21/experiments/feature_selection_03.json — grid và resource gates.",
              "- src/vietmailguard/v21/feature_selection_experiment.py — training, CV, kiểm tra và báo cáo.",
              "- scripts/v21/feature_selection_03.py — CLI và subprocess từng candidate.",
              "- tests/test_v21_feature_selection.py — sparse safety, Train-only fitting, folds, reload/resume/integrity.",
              "- FEATURE_SELECTION_EXPERIMENT_03.md — báo cáo này.",
              "- models/v21_multilingual/experiments/feature_selection_03/ — bốn artifact A riêng.",
              "- results/v21_multilingual/experiments/feature_selection_03/ — checkpoints, metrics, diagnostics, manifest và verification.",
              "", "## Kiểm thử và bảo toàn", ""]
    verification = root / "results/v21_multilingual/experiments/feature_selection_03/verification.json"
    if verification.is_file():
        checked = read_json(verification)
        lines += [f"- Unit/integration riêng: {checked['targeted_tests']}.",
                  f"- Regression offline: {checked['regression_tests']}.",
                  "- 3 skipped: real TREC archive/curation/review UI tests cần opt-in; "
                  "3 deselected: các test đọc held-out thật, được loại để tuân thủ không mở Test. "
                  "Không coi đây là full regression bao gồm held-out.",
                  "- 1 ConvergenceWarning của fixture MLP chỉ chạy 5 iterations, không phải candidate Chi-square.",
                  f"- SHA-256 trước/sau: {checked['protected_files_unchanged']}/{checked['protected_files_checked']} file bảo vệ không đổi.",
                  f"- git diff --check: {checked['diff_check']}; không có external-network/held-out access attempt trong regression.",
                  "", "Lệnh thực tế:", "", "```powershell", *checked["commands"], "```", "",
                  "Nhánh local: v21/experiment-feature-selection-03. Các thay đổi có sẵn trước nhiệm vụ được bảo toàn.", ""]
    else:
        lines += ["Kết quả regression và bảo toàn bổ sung chưa được ghi vào verification.json; không tự suy diễn PASS.", ""]
    (root / "FEATURE_SELECTION_EXPERIMENT_03.md").write_text("\n".join(lines), encoding="utf-8")


def finalize(paths=None):
    paths = paths or resolve_v21_paths()
    with offline_boundary(paths.project_root):
        output = summarize(paths)
        diagnostics = feature_diagnostics(paths, output["candidates"])
        directory = paths.results / "experiments" / EXPERIMENT
        _json(directory / "feature_diagnostics.json", diagnostics)
        versions = {name: importlib.metadata.version(name) for name in
                    ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl")}
        files = [paths.project_root / path for path in (
            "config/v21/experiments/feature_selection_03.json",
            "scripts/v21/feature_selection_03.py",
            "src/vietmailguard/v21/feature_selection_experiment.py",
            "tests/test_v21_feature_selection.py",
            "src/vietmailguard/v21/development_training.py",
            "src/vietmailguard/v21/dimensionality_experiment.py",
            "src/vietmailguard/v21/multilingual_dl_experiment.py",
            "src/vietmailguard/modeling.py",
            "src/vietmailguard/v21/paths.py",
        )]
        _json(directory / "run_manifest.json", {
            "experiment": EXPERIMENT, "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
            "fingerprint": output["candidates"][0]["fingerprint"],
            "source_sha256": {path.relative_to(paths.project_root).as_posix(): sha256_file(path) for path in files},
            "result_sha256": {path.name: sha256_file(path) for path in sorted(directory.glob("*.json"))
                              if path.name not in {"run_manifest.json", "verification.json"}},
            "artifact_sha256": {row["artifact"]: row["artifact_sha256"] for row in output["candidates"] if row.get("artifact")},
            "final_test_accessed": False, "runtime_changed": False, "publication_approval": "NO",
        })
        write_report(output, paths.project_root, diagnostics)
        return output
