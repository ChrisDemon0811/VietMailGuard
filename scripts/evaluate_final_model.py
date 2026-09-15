"""Finalize the locked candidate and evaluate the held-out test once."""

from __future__ import annotations

import argparse
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
        description="Refit the locked candidate and evaluate held-out test exactly once."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "final_evaluation.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing final test result. Not used in the normal one-shot workflow.",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def main() -> int:
    args = parse_args()
    final_config = load_json(args.config)
    metrics_path = project_path(final_config["outputs"]["metrics"])
    if metrics_path.exists() and not args.force:
        raise FileExistsError(
            f"Final test result already exists: {metrics_path}. "
            "Refusing to evaluate the held-out test again without --force."
        )

    experiment_config_path = project_path(final_config["experiment_config"])
    experiment_config = load_json(experiment_config_path)
    selection_path = project_path(final_config["selected_model"])
    selection = load_json(selection_path)
    if not selection.get("selection_complete") or selection.get("test_data_used"):
        raise ValueError("Selection artifact is not a clean train/validation-only decision")
    if selection["experiment_config_sha256"] != sha256_file(experiment_config_path):
        raise ValueError("Experiment config changed after model selection")

    label_order = [str(value) for value in experiment_config["label_order"]]
    train_path = project_path(final_config["development_data"]["train"])
    validation_path = project_path(final_config["development_data"]["validation"])
    train = read_email_split(train_path)
    validation = read_email_split(validation_path)
    validate_labels(train, label_order, "train")
    validate_labels(validation, label_order, "validation")
    assert_disjoint_splits(train, validation, left_name="train", right_name="validation")
    if sha256_file(train_path) != selection["data"]["train"]["sha256"]:
        raise ValueError("train.csv changed after model selection")
    if sha256_file(validation_path) != selection["data"]["validation"]["sha256"]:
        raise ValueError("validation.csv changed after model selection")

    selected_id = selection["selected_experiment_id"]
    specs = {spec.experiment_id: spec for spec in iter_experiment_specs(experiment_config)}
    if selected_id not in specs:
        raise ValueError(f"Selected experiment no longer exists in config: {selected_id}")

    development = pd.concat([train, validation], ignore_index=True)
    X_development = development[["subject", "body"]]
    y_development = development["label"].astype(str)
    pipeline = build_experiment_pipeline(specs[selected_id], experiment_config)
    print(
        f"Locked candidate: {selected_id}. Fitting on train + validation "
        f"({len(development):,} rows) before opening held-out test...",
        flush=True,
    )
    started = time.perf_counter()
    pipeline.fit(X_development, y_development)
    fit_seconds = time.perf_counter() - started
    production_path = project_path(final_config["production_pipeline"])
    production_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, production_path)
    print(f"Final refit complete in {fit_seconds:.2f}s. Production pipeline saved.", flush=True)

    # This is deliberately the first point where the held-out test path is opened.
    test_path = project_path(final_config["held_out_test"])
    test = read_email_split(test_path)
    validate_labels(test, label_order, "test")
    assert_disjoint_splits(development, test, left_name="train+validation", right_name="test")
    print(f"Held-out test opened after final refit: {split_summary(test)}", flush=True)

    metrics, predictions, predict_seconds = evaluate_pipeline(
        pipeline,
        test[["subject", "body"]],
        test["label"].astype(str),
        label_order,
    )
    final_metrics = {
        "evaluation": "held_out_test",
        "test_evaluation_count_for_this_run": 1,
        "selected_experiment_id": selected_id,
        "selection_metric": experiment_config["selection"],
        "selection_validation_metrics": selection["validation_metrics"],
        "final_refit_data": {
            "splits": ["train", "validation"],
            "summary": split_summary(development),
            "fit_seconds": fit_seconds,
        },
        "test_data": {
            "path": final_config["held_out_test"],
            "sha256": sha256_file(test_path),
            **split_summary(test),
        },
        "test_metrics": metrics,
        "test_predict_seconds": predict_seconds,
        "runtime_versions": runtime_versions(),
        "probability_policy": (
            "Native predict_proba for MultinomialNB/LogisticRegression; sigmoid-calibrated "
            "probabilities for LinearSVC. Raw SVM decision scores are never treated as probabilities."
        ),
    }

    metrics_directory = metrics_path.parent
    write_metric_artifacts(metrics_directory, "final_test", metrics)
    write_json(metrics_path, final_metrics)

    prediction_path = project_path(final_config["outputs"]["prediction_records"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame = test[["id", "group_id", "source", "language", "label"]].copy()
    prediction_frame["predicted_label"] = predictions
    prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8")

    report_path = project_path(final_config["outputs"]["report"])
    report = report_path.read_text(encoding="utf-8")
    marker = "## Held-out test"
    prefix = report.split(marker, maxsplit=1)[0].rstrip()
    per_class_lines = [
        f"| {label} | {values['precision']:.6f} | {values['recall']:.6f} | "
        f"{values['f1']:.6f} | {values['support']} |"
        for label, values in metrics["per_class"].items()
    ]
    confusion_lines = [
        "| " + label + " | " + " | ".join(str(value) for value in row) + " |"
        for label, row in zip(
            metrics["confusion_matrix_label_order"],
            metrics["confusion_matrix"],
            strict=True,
        )
    ]
    final_section = [
        "",
        marker,
        "",
        "Production candidate đã được refit trên `train + validation`, rồi mới mở và đánh giá "
        "held-out test đúng một lần trong lần chạy này.",
        "",
        f"- Test rows: {len(test):,}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Macro Precision: {metrics['macro_precision']:.6f}",
        f"- Macro Recall: {metrics['macro_recall']:.6f}",
        f"- Macro F1: {metrics['macro_f1']:.6f}",
        f"- Weighted F1: {metrics['weighted_f1']:.6f}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
        *per_class_lines,
        "",
        "Confusion matrix, theo thứ tự `normal`, `spam`, `phishing`:",
        "",
        "| Actual / Predicted | normal | spam | phishing |",
        "|---|---:|---:|---:|",
        *confusion_lines,
        "",
        "## Giới hạn diễn giải",
        "",
        "Phishing hiện chỉ đến từ Nazario trong tập dữ liệu đủ điều kiện. Spam mang provenance "
        "`high_confidence_curated_spam`, không phải nhãn đã được con người xác nhận từng email. "
        "Class và source vì vậy còn có thể bị confound; metric cao trên split hiện tại không chứng "
        "minh khả năng tổng quát sang corpus mới. Cần external-corpus evaluation hoặc bổ sung source "
        "diversity ở bước sau.",
        "",
        "Nếu production candidate là Linear SVM, confidence đến từ sigmoid calibration CV=3; "
        "không dùng raw decision score như xác suất.",
        "",
    ]
    report_path.write_text(prefix + "\n" + "\n".join(final_section), encoding="utf-8")

    print("Final held-out test metrics", flush=True)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        print(f"{key}: {metrics[key]:.6f}", flush=True)
    for label in label_order:
        values = metrics["per_class"][label]
        print(
            f"{label}: precision={values['precision']:.6f}, recall={values['recall']:.6f}, "
            f"f1={values['f1']:.6f}, support={values['support']}",
            flush=True,
        )
    print("Held-out test evaluations in this run: 1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
