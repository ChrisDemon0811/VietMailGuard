"""Run model selection using only train.csv and validation.csv."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.modeling import (  # noqa: E402
    assert_disjoint_splits,
    build_experiment_pipeline,
    evaluate_pipeline,
    iter_experiment_specs,
    load_json,
    metrics_row,
    plot_model_comparison,
    rank_comparison,
    read_email_split,
    runtime_versions,
    sha256_file,
    split_summary,
    validate_labels,
    write_json,
    write_metric_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a VietMailGuard model using train and validation only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "model_experiments.json",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a small result table without an optional markdown dependency."""
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, record in frame[columns].iterrows():
        values = [
            f"{record[column]:.6f}" if isinstance(record[column], float) else str(record[column])
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    label_order = [str(value) for value in config["label_order"]]
    train_path = project_path(config["data"]["train"])
    validation_path = project_path(config["data"]["validation"])

    print("Loading model-selection data (train + validation only)...", flush=True)
    train = read_email_split(train_path)
    validation = read_email_split(validation_path)
    validate_labels(train, label_order, "train")
    validate_labels(validation, label_order, "validation")
    assert_disjoint_splits(train, validation, left_name="train", right_name="validation")
    print(f"train: {split_summary(train)}", flush=True)
    print(f"validation: {split_summary(validation)}", flush=True)

    X_train = train[["subject", "body"]]
    y_train = train["label"].astype(str)
    X_validation = validation[["subject", "body"]]
    y_validation = validation["label"].astype(str)

    validation_directory = project_path(config["outputs"]["validation_directory"])
    cache_root = project_path(config["outputs"]["cache_directory"])
    validation_directory.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    best_pipeline = None
    best_experiment_id = ""
    best_sort_key: tuple[float, float, float, float, str] | None = None
    specs = iter_experiment_specs(config)
    print(f"Running {len(specs)} validation experiments...", flush=True)

    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.experiment_id}: fitting", flush=True)
        feature_cache = cache_root / spec.feature_name
        pipeline = build_experiment_pipeline(spec, config, cache_directory=feature_cache)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        metrics, _, predict_seconds = evaluate_pipeline(
            pipeline, X_validation, y_validation, label_order
        )
        row = metrics_row(
            spec,
            metrics,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
        )
        rows.append(row)
        write_metric_artifacts(validation_directory, spec.experiment_id, metrics)
        sort_key = (
            -float(metrics[config["selection"]["primary_metric"]]),
            -float(metrics["per_class"]["phishing"]["recall"]),
            -float(metrics["weighted_f1"]),
            -float(metrics["accuracy"]),
            spec.experiment_id,
        )
        if best_sort_key is None or sort_key < best_sort_key:
            best_sort_key = sort_key
            best_pipeline = pipeline
            best_experiment_id = spec.experiment_id
        print(
            f"[{index}/{len(specs)}] {spec.experiment_id}: "
            f"macro_f1={metrics['macro_f1']:.6f}, "
            f"phishing_recall={metrics['per_class']['phishing']['recall']:.6f}, "
            f"fit_seconds={fit_seconds:.2f}",
            flush=True,
        )

    comparison = rank_comparison(pd.DataFrame(rows), config["selection"])
    comparison_path = project_path(config["outputs"]["comparison"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    plot_model_comparison(comparison, project_path(config["outputs"]["comparison_plot"]))

    selected_row = comparison.iloc[0].to_dict()
    if best_pipeline is None or best_experiment_id != selected_row["experiment_id"]:
        raise RuntimeError("In-memory winner does not match the persisted comparison ranking")
    candidate_path = project_path(config["outputs"]["validation_candidate_pipeline"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    best_pipeline.memory = None
    joblib.dump(best_pipeline, candidate_path)

    selected_artifact = {
        "selection_complete": True,
        "test_data_used": False,
        "selected_experiment_id": str(selected_row["experiment_id"]),
        "feature_set": str(selected_row["feature_set"]),
        "classifier": str(selected_row["classifier"]),
        "ranking_rule": config["selection"],
        "validation_metrics": {
            key: float(selected_row[key])
            for key in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "phishing_precision",
                "phishing_recall",
                "phishing_f1",
            )
        },
        "data": {
            "train": {
                "path": config["data"]["train"],
                "sha256": sha256_file(train_path),
                **split_summary(train),
            },
            "validation": {
                "path": config["data"]["validation"],
                "sha256": sha256_file(validation_path),
                **split_summary(validation),
            },
        },
        "experiment_config_sha256": sha256_file(args.config),
        "runtime_versions": runtime_versions(),
        "validation_candidate_pipeline": config["outputs"]["validation_candidate_pipeline"],
    }
    write_json(project_path(config["outputs"]["selected_model"]), selected_artifact)

    report_path = project_path(config["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_columns = [
        "validation_rank",
        "feature_set",
        "classifier",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "phishing_recall",
    ]
    report_lines = [
        "# Báo cáo lựa chọn model VietMailGuard Version 1",
        "",
        "## Phương pháp lựa chọn",
        "",
        "Model được chọn chỉ bằng `train.csv` và `validation.csv`. TF-IDF được fit bên trong "
        "sklearn Pipeline trên training split. Held-out test chưa được đọc trong bước lựa chọn.",
        "",
        f"Tiêu chí chính: `{config['selection']['primary_metric']}`. Tiêu chí phụ: "
        f"`{config['selection']['secondary_metric']}`.",
        "",
        "## Kết quả validation",
        "",
        markdown_table(comparison, table_columns),
        "",
        "## Production candidate đã khóa",
        "",
        f"- Experiment: `{selected_row['experiment_id']}`",
        f"- Feature: {selected_row['feature_set']}",
        f"- Classifier: {selected_row['classifier']}",
        f"- Validation Macro F1: {selected_row['macro_f1']:.6f}",
        f"- Validation phishing recall: {selected_row['phishing_recall']:.6f}",
        "- Linear SVM, nếu được chọn, dùng sigmoid calibration nội bộ; raw decision score không được dùng làm xác suất.",
        "",
        "Model này thắng vì có validation Macro F1 cao nhất trong 9 thí nghiệm. Phishing recall "
        "của nó cũng được dùng làm tiêu chí phụ khi Macro F1 hòa.",
        "",
        "## Held-out test",
        "",
        "Chưa đánh giá. Chạy `python scripts/evaluate_final_model.py` đúng một lần sau khi kiểm tra selection artifact.",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    resolved_cache = cache_root.resolve()
    resolved_results = (PROJECT_ROOT / "results").resolve()
    if resolved_results in resolved_cache.parents and resolved_cache.name == ".training_cache":
        shutil.rmtree(resolved_cache)

    print("Validation ranking complete.", flush=True)
    print(comparison[table_columns].to_string(index=False), flush=True)
    print(f"Selected: {selected_row['experiment_id']}", flush=True)
    print("Held-out test used: False", flush=True)
    print(f"Comparison: {comparison_path.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
