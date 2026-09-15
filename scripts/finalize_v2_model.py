"""Refit the frozen Version 2 model and evaluate its held-out test exactly once."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.file_utils import sha256_file  # noqa: E402
from vietmailguard.modeling import (  # noqa: E402
    REQUIRED_DATA_COLUMNS,
    compute_metrics,
    load_json,
    plot_confusion_matrix,
    read_email_split,
    runtime_versions,
    split_summary,
    validate_labels,
    write_json,
)
from vietmailguard.v2_finalization import (  # noqa: E402
    OneShotGuard,
    assert_one_shot_not_started,
    build_frozen_pipeline,
    build_group_calibration_splits,
    classification_report_frame,
    create_one_shot_journal,
    evaluate_final_subsets,
    validate_controlled_test_descendants,
    verify_frozen_artifacts,
)
from vietmailguard.v2_modeling import assert_no_cross_split_values  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Version 2 refit and one-shot held-out evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "v2_production_frozen.json",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def _read_held_out_once(path: Path) -> tuple[pd.DataFrame, str]:
    """Read the held-out file bytes once and parse/hash that in-memory snapshot."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    frame = pd.read_csv(io.BytesIO(raw), keep_default_na=False, low_memory=False)
    missing = REQUIRED_DATA_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Held-out test is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Held-out test contains no rows")
    if frame["body"].astype(str).str.strip().eq("").any():
        raise ValueError("Held-out test contains an empty email body")
    return frame, digest


def _write_confusion_artifacts(
    output_csv: Path, output_png: Path, metrics: dict[str, Any]
) -> None:
    labels = metrics["confusion_matrix_label_order"]
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    matrix.index.name = "actual\\predicted"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_csv, encoding="utf-8")
    plot_confusion_matrix(
        matrix.to_numpy(),
        labels,
        output_png,
        title="VietMailGuard V2 held-out test confusion matrix",
    )


def _origin_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame["data_origin"].fillna("").astype(str)
    language = frame["language"].fillna("").astype(str)
    labels = values.mask(values.eq("") & language.eq("en"), "original_english_artifact")
    labels = labels.mask(labels.eq(""), "unspecified")
    return {str(key): int(value) for key, value in labels.value_counts().sort_index().items()}


def _smoke_examples() -> list[dict[str, str]]:
    return [
        {
            "scenario": "English normal example",
            "subject": "Project meeting agenda",
            "body": "Hello team, our weekly project meeting is at 10 AM tomorrow. Regards, Alex.",
        },
        {
            "scenario": "English spam example",
            "subject": "Weekend sale on office software",
            "body": "Save 40 percent on our annual plan this weekend. View the offer or unsubscribe.",
        },
        {
            "scenario": "English phishing example",
            "subject": "Urgent account verification required",
            "body": "Your account is suspended. Sign in now at http://192.0.2.10/login to verify your password.",
        },
        {
            "scenario": "Vietnamese normal example",
            "subject": "Lịch họp nhóm dự án",
            "body": "Chào cả nhóm, cuộc họp hằng tuần sẽ bắt đầu lúc 10 giờ sáng mai.",
        },
        {
            "scenario": "Vietnamese spam-like example",
            "subject": "Khuyến mãi phần mềm cuối tuần",
            "body": "Giảm giá 40 phần trăm cho gói dịch vụ năm. Xem ưu đãi hoặc hủy đăng ký nhận tin.",
        },
        {
            "scenario": "Vietnamese phishing-like example",
            "subject": "Yêu cầu xác minh tài khoản khẩn cấp",
            "body": "Tài khoản của bạn đã bị khóa. Hãy đăng nhập tại http://192.0.2.10/login và nhập mật khẩu để xác minh.",
        },
    ]


def _run_reload_smoke(pipeline_path: Path) -> list[dict[str, Any]]:
    pipeline = joblib.load(pipeline_path)
    examples = _smoke_examples()
    frame = pd.DataFrame(examples)[["subject", "body"]]
    predictions = np.asarray(pipeline.predict(frame), dtype=object)
    probabilities = np.asarray(pipeline.predict_proba(frame), dtype=float)
    classifier = pipeline.named_steps["classifier"]
    classes = np.asarray(classifier.classes_, dtype=object)
    if probabilities.shape != (len(examples), len(classes)):
        raise RuntimeError("Production probability output has an invalid shape")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-7
    ):
        raise RuntimeError("Production pipeline returned invalid calibrated probabilities")
    output: list[dict[str, Any]] = []
    for example, prediction, row in zip(examples, predictions, probabilities, strict=True):
        output.append(
            {
                "scenario": example["scenario"],
                "prediction": str(prediction),
                "confidence": float(row[int(np.flatnonzero(classes == prediction)[0])]),
                "class_probabilities": {
                    str(label): float(value)
                    for label, value in zip(classes, row, strict=True)
                },
            }
        )
    return output


def _v1_comparison_context(test: pd.DataFrame) -> dict[str, Any]:
    """Evaluate V1 on exact V2 subsets only when contamination can be ruled out."""
    v1_metrics_path = PROJECT_ROOT / "results" / "final_test_metrics.json"
    v1_model_path = PROJECT_ROOT / "models" / "production_pipeline.joblib"
    v1_train_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
    v1_validation_path = PROJECT_ROOT / "data" / "splits" / "validation.csv"
    context: dict[str, Any] = {
        "v1_saved_test_metrics": None,
        "v1_saved_test_metrics_note": (
            "Version 1 uses its own English held-out cohort; comparison with V2 is descriptive."
        ),
        "exact_v2_english_subset": None,
        "exact_v2_translated_vietnamese_subset": None,
    }
    if v1_metrics_path.is_file():
        saved = load_json(v1_metrics_path)
        context["v1_saved_test_metrics"] = saved.get("test_metrics")
        context["v1_saved_test_rows"] = saved.get("test_data", {}).get("rows")
    if not (v1_model_path.is_file() and v1_train_path.is_file() and v1_validation_path.is_file()):
        context["exact_subset_limitation"] = "Version 1 model or development splits are unavailable."
        return context

    v1_development = pd.concat(
        [read_email_split(v1_train_path), read_email_split(v1_validation_path)],
        ignore_index=True,
    )
    v1_ids = set(v1_development["id"].astype(str))
    v1_hashes = set(v1_development["content_hash"].astype(str))
    v1_groups = set(v1_development["group_id"].astype(str))
    english = test[test["language"].astype(str).eq("en")]
    english_overlap = (
        english["id"].astype(str).isin(v1_ids)
        | english["content_hash"].astype(str).isin(v1_hashes)
        | english["group_id"].astype(str).isin(v1_groups)
    )
    context["v2_english_rows"] = int(len(english))
    context["v2_english_rows_seen_in_v1_development"] = int(english_overlap.sum())

    translated = test[
        test["language"].astype(str).eq("vi")
        & test["data_origin"].astype(str).eq("translated")
    ]
    parent_ids = translated["parent_id"].fillna("").astype(str)
    known_parent = parent_ids.str.strip().ne("")
    parent_overlap = known_parent & parent_ids.isin(v1_ids)
    context["v2_translated_vietnamese_rows"] = int(len(translated))
    context["v2_translated_rows_with_known_parent"] = int(known_parent.sum())
    context["v2_translated_parent_rows_seen_in_v1_development"] = int(parent_overlap.sum())

    v1_pipeline: Any | None = None
    label_order = ["normal", "spam", "phishing"]
    if len(english) and not english_overlap.any():
        v1_pipeline = joblib.load(v1_model_path)
        predictions = v1_pipeline.predict(english[["subject", "body"]])
        english_copy = english.reset_index(drop=True)
        context["exact_v2_english_subset"] = evaluate_final_subsets(
            english_copy, predictions, label_order
        )["english_test"]
    else:
        context["exact_v2_english_limitation"] = (
            "V1 cannot be evaluated fairly on the exact V2 English test subset because "
            "rows/groups from that subset occur in V1 development data."
        )

    translated_is_auditable = (
        len(translated) > 0
        and bool(known_parent.all())
        and not bool(parent_overlap.any())
        and not translated["content_hash"].astype(str).isin(v1_hashes).any()
        and not translated["group_id"].astype(str).isin(v1_groups).any()
    )
    if translated_is_auditable:
        if v1_pipeline is None:
            v1_pipeline = joblib.load(v1_model_path)
        predictions = v1_pipeline.predict(translated[["subject", "body"]])
        translated_copy = translated.reset_index(drop=True)
        context["exact_v2_translated_vietnamese_subset"] = evaluate_final_subsets(
            translated_copy, predictions, label_order
        )["translated_vietnamese_test"]
    else:
        context["exact_v2_translated_vietnamese_limitation"] = (
            "A leakage-safe V1 comparison on the exact translated-Vietnamese subset "
            "cannot be established because parent provenance is incomplete or a known "
            "parent/content/group appears in V1 development data."
        )
    return context


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def _recall(metrics: dict[str, Any], label: str) -> str:
    return _format(metrics["per_class"][label]["recall"])


def _write_evaluation_report(
    path: Path,
    *,
    overall: dict[str, Any],
    subsets: dict[str, dict[str, Any]],
    development: pd.DataFrame,
    test: pd.DataFrame,
    fit_seconds: float,
    predict_seconds: float,
    calibration_folds: int,
    descendant_check: dict[str, int],
    v1_context: dict[str, Any],
    smoke: list[dict[str, Any]],
) -> None:
    per_class = [
        f"| {label} | {_format(values['precision'])} | {_format(values['recall'])} | "
        f"{_format(values['f1'])} | {values['support']} |"
        for label, values in overall["per_class"].items()
    ]
    subset_rows = []
    for name in (
        "combined",
        "english_test",
        "translated_vietnamese_test",
        "original_english",
        "existing_translated_vietnamese",
        "controlled_translation_augmentation_descendants",
    ):
        metrics = subsets[name]
        subset_rows.append(
            f"| {name} | {metrics['rows']} | {_format(metrics['accuracy'])} | "
            f"{_format(metrics['macro_f1_supported_classes'])} | "
            f"{_recall(metrics, 'normal')} | {_recall(metrics, 'spam')} | "
            f"{_recall(metrics, 'phishing')} |"
        )
    confusion_rows = [
        "| " + label + " | " + " | ".join(str(value) for value in row) + " |"
        for label, row in zip(
            overall["confusion_matrix_label_order"],
            overall["confusion_matrix"],
            strict=True,
        )
    ]
    smoke_rows = [
        f"| {item['scenario']} | {item['prediction']} | {item['confidence']:.6f} |"
        for item in smoke
    ]
    v1_saved = v1_context.get("v1_saved_test_metrics")
    v1_lines = []
    if v1_saved:
        v1_lines.extend(
            [
                f"- V1 own English held-out Macro F1: {_format(v1_saved.get('macro_f1'))}",
                f"- V1 own English held-out phishing recall: "
                f"{_format(v1_saved.get('per_class', {}).get('phishing', {}).get('recall'))}",
                "- V1 and V2 held-out figures are descriptive, not a paired comparison, "
                "because the held-out cohorts differ.",
            ]
        )
    else:
        v1_lines.append("- Saved Version 1 held-out metrics were unavailable.")
    v1_lines.extend(
        [
            f"- V2 English test rows overlapping V1 development: "
            f"{v1_context.get('v2_english_rows_seen_in_v1_development', 'N/A')} / "
            f"{v1_context.get('v2_english_rows', 'N/A')}.",
            f"- {v1_context.get('exact_v2_english_limitation', 'Exact V2 English comparison was leakage-safe and is stored in final metrics.')}",
            f"- Vietnamese translated rows with known parents: "
            f"{v1_context.get('v2_translated_rows_with_known_parent', 'N/A')} / "
            f"{v1_context.get('v2_translated_vietnamese_rows', 'N/A')}.",
            f"- {v1_context.get('exact_v2_translated_vietnamese_limitation', 'Exact translated-Vietnamese comparison was leakage-safe and is stored in final metrics.')}",
        ]
    )
    lines = [
        "# Version 2 final model evaluation",
        "",
        "## Locked selection and final refit",
        "",
        "The production approach was selected and documented in "
        "`reports/v2/model_selection_rationale.md` before the held-out test was opened.",
        "",
        "- Representation: Word TF-IDF, word n-grams (1, 2), 60,000 maximum features.",
        "- Classifier: LinearSVC with C=1.0 and balanced class weights.",
        f"- Confidence: sigmoid CalibratedClassifierCV with {calibration_folds} "
        "StratifiedGroupKFold folds and ensemble disabled.",
        f"- Refit data: train + validation, {len(development):,} rows.",
        f"- Refit time: {fit_seconds:.2f} seconds.",
        "- Test rows were not used to fit TF-IDF, LinearSVC, or calibration.",
        "",
        "## One-shot combined held-out test",
        "",
        f"- Test rows: {len(test):,}",
        f"- Accuracy: {_format(overall['accuracy'])}",
        f"- Macro Precision: {_format(overall['macro_precision'])}",
        f"- Macro Recall: {_format(overall['macro_recall'])}",
        f"- Macro F1: {_format(overall['macro_f1'])}",
        f"- Weighted F1: {_format(overall['weighted_f1'])}",
        f"- Prediction time: {predict_seconds:.3f} seconds.",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
        *per_class,
        "",
        "## Test by language and origin",
        "",
        "| Scope | Rows | Accuracy | Macro F1* | Normal recall | Spam recall | Phishing recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *subset_rows,
        "",
        "*Subset Macro F1 covers supported classes only. Missing-class metrics are N/A.",
        "",
        "Native Vietnamese benchmark is not available.",
        "",
        f"Controlled translated test descendants: {descendant_check['controlled_test_rows']}; "
        f"verified test parents: {descendant_check['verified_test_parent_rows']}.",
        "",
        "## Confusion matrix",
        "",
        "| Actual / predicted | normal | spam | phishing |",
        "|---|---:|---:|---:|",
        *confusion_rows,
        "",
        "## Version 1 comparison boundary",
        "",
        *v1_lines,
        "",
        "## Reload smoke test",
        "",
        "These handcrafted examples test artifact loading and output shape. They are not metrics.",
        "",
        "| Scenario | Prediction | Calibrated confidence |",
        "|---|---|---:|",
        *smoke_rows,
        "",
        "## Limitations and claim boundary",
        "",
        "- Vietnamese evaluation uses translated data only. It does not establish native "
        "Vietnamese real-world performance.",
        "- Any class absent from a subset has N/A metrics rather than an invented zero score.",
        "- The phishing corpus remains source-concentrated, so source/class confounding is possible.",
        "- Spam labels retain curated provenance and are not all individually human-confirmed.",
        "- Mixed-language email has not been independently benchmarked.",
        "- No post-test tuning or repeat held-out evaluation is permitted. A serious issue "
        "requires a Version 2.1 protocol and a new test set.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    outputs = config["outputs"]
    guard = OneShotGuard(
        started_path=PROJECT_ROOT / "results" / "v2_bilingual" / ".final_test_evaluation_started.json",
        final_metrics_path=project_path(outputs["final_metrics"]),
    )
    assert_one_shot_not_started(guard)
    verify_frozen_artifacts(config, PROJECT_ROOT)
    print("Frozen validation selection verified. Held-out test has not been opened.", flush=True)

    label_order = [str(value) for value in config["label_order"]]
    train_path = project_path(config["data"]["train"])
    validation_path = project_path(config["data"]["validation"])
    train = read_email_split(train_path)
    validation = read_email_split(validation_path)
    validate_labels(train, label_order, "train")
    validate_labels(validation, label_order, "validation")
    assert_no_cross_split_values(train, validation, config["leakage_columns"])
    development = pd.concat([train, validation], ignore_index=True)
    labels = development["label"].astype(str)
    groups = development[config["calibration"]["group_column"]].astype(str)
    calibration_splits = build_group_calibration_splits(
        labels,
        groups,
        n_splits=int(config["calibration"]["n_splits"]),
        seed=int(config["calibration"]["random_state"]),
    )
    pipeline = build_frozen_pipeline(config, calibration_splits)
    print(
        f"Fitting frozen pipeline on train + validation only ({len(development):,} rows)...",
        flush=True,
    )
    started = time.perf_counter()
    pipeline.fit(development[["subject", "body"]], labels)
    fit_seconds = time.perf_counter() - started
    production_path = project_path(outputs["production_pipeline"])
    _atomic_joblib_dump(pipeline, production_path)
    smoke = _run_reload_smoke(production_path)
    print(f"Production artifact saved and reload smoke passed in {fit_seconds:.2f}s.", flush=True)

    journal_payload = {
        "schema_version": 1,
        "status": "started",
        "started_at": _utc_now(),
        "process_id": os.getpid(),
        "frozen_config": args.config.relative_to(PROJECT_ROOT).as_posix(),
        "frozen_config_sha256": sha256_file(args.config),
        "selection_rationale_sha256": sha256_file(
            project_path(config["selection_artifacts"]["rationale"])
        ),
        "policy": "Do not reopen this held-out test, including after a partial failure.",
    }
    create_one_shot_journal(guard.started_path, journal_payload)

    # First and only read of the Version 2 held-out CSV occurs here, after refit.
    test_path = project_path(config["data"]["held_out_test"])
    test, test_sha256 = _read_held_out_once(test_path)
    validate_labels(test, label_order, "test")
    assert_no_cross_split_values(development, test, config["leakage_columns"])
    development_ids = set(development["id"].astype(str))
    test_parent_ids = {
        value for value in test["parent_id"].fillna("").astype(str) if value.strip()
    }
    parent_overlap = test_parent_ids & development_ids
    if parent_overlap:
        raise ValueError(
            f"Test translation parent appears in development: {sorted(parent_overlap)[:5]}"
        )
    descendant_check = validate_controlled_test_descendants(development, test)
    print(f"Held-out test opened once after refit: {split_summary(test)}", flush=True)
    print("Development/test leakage checks: PASS", flush=True)

    started = time.perf_counter()
    predictions = np.asarray(
        pipeline.predict(test[["subject", "body"]]), dtype=object
    )
    predict_seconds = time.perf_counter() - started
    overall = compute_metrics(test["label"].astype(str), predictions, label_order)
    subsets = evaluate_final_subsets(test, predictions, label_order)
    report_frame = classification_report_frame(subsets, label_order)
    classification_path = project_path(outputs["classification_report"])
    classification_path.parent.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(classification_path, index=False, encoding="utf-8")
    _write_confusion_artifacts(
        project_path(outputs["confusion_matrix_csv"]),
        project_path(outputs["confusion_matrix_png"]),
        overall,
    )

    v1_context = _v1_comparison_context(test)
    evaluation_timestamp = _utc_now()
    metadata = {
        "schema_version": 1,
        "model_version": config["model_version"],
        "artifact": outputs["production_pipeline"],
        "artifact_sha256": sha256_file(production_path),
        "classifier": "LinearSVC with sigmoid CalibratedClassifierCV",
        "representation": "Word TF-IDF (1, 2)",
        "embedding_model": None,
        "preprocessing": config["text_composer"],
        "calibration": config["calibration"],
        "training_languages": sorted(development["language"].astype(str).unique().tolist()),
        "supported_content_languages": {
            "en": {"status": "supported", "evaluation": "independent English test subset"},
            "vi": {
                "status": "experimental",
                "evaluation": "translated Vietnamese test subset only",
            },
            "mixed": {"status": "experimental_not_independently_benchmarked"},
        },
        "dataset_sources": sorted(development["source"].astype(str).unique().tolist()),
        "training_data_origins": _origin_counts(development),
        "class_names": label_order,
        "training_rows": int(len(development)),
        "test_metrics": overall,
        "test_subset_metrics": subsets,
        "native_benchmark_status": "Native Vietnamese benchmark is not available.",
        "limitations": [
            "Vietnamese evaluation is translated, not native Vietnamese.",
            "Translated Vietnamese test may lack one or more classes; absent metrics are N/A.",
            "Phishing data remains source-concentrated.",
            "Mixed-language content has not been independently benchmarked.",
            "No post-test tuning or repeat held-out evaluation is allowed.",
        ],
        "evaluated_at": evaluation_timestamp,
        "smoke_tests": smoke,
    }
    write_json(project_path(outputs["model_metadata"]), metadata)
    _write_evaluation_report(
        project_path(outputs["evaluation_report"]),
        overall=overall,
        subsets=subsets,
        development=development,
        test=test,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        calibration_folds=len(calibration_splits),
        descendant_check=descendant_check,
        v1_context=v1_context,
        smoke=smoke,
    )

    final_metrics = {
        "schema_version": 1,
        "evaluation": "v2_held_out_test_one_shot",
        "held_out_test_evaluation_count": 1,
        "selection_completed_before_test_access": True,
        "post_test_tuning_allowed": False,
        "selected_experiment_id": config["selected_experiment_id"],
        "frozen_config": args.config.relative_to(PROJECT_ROOT).as_posix(),
        "frozen_config_sha256": sha256_file(args.config),
        "selection_rationale": config["selection_artifacts"]["rationale"],
        "selection_rationale_sha256": sha256_file(
            project_path(config["selection_artifacts"]["rationale"])
        ),
        "final_refit": {
            "input_splits": ["train", "validation"],
            "rows": int(len(development)),
            "summary": split_summary(development),
            "fit_seconds": fit_seconds,
            "calibration_folds": len(calibration_splits),
            "calibration_group_column": config["calibration"]["group_column"],
        },
        "test_data": {
            "path": config["data"]["held_out_test"],
            "sha256_from_single_read_snapshot": test_sha256,
            **split_summary(test),
            "origin_distribution": _origin_counts(test),
        },
        "leakage_checks": {
            "passed": True,
            "columns": config["leakage_columns"],
            "test_parent_ids_in_development": 0,
            **descendant_check,
        },
        "overall_test_metrics": overall,
        "test_subset_metrics": subsets,
        "v1_comparison": v1_context,
        "production_artifact": {
            "path": outputs["production_pipeline"],
            "sha256": sha256_file(production_path),
            "reload_smoke_test": True,
            "calibrated_probability_smoke_test": True,
            "smoke_predictions": smoke,
        },
        "prediction_seconds": predict_seconds,
        "native_vietnamese_benchmark": {
            "available": False,
            "statement": "Native Vietnamese benchmark is not available.",
        },
        "runtime_versions": {
            **runtime_versions(),
            "platform": platform.platform(),
        },
        "completed_at": evaluation_timestamp,
    }
    # The metrics file is the permanent completion guard and is deliberately written last.
    write_json(guard.final_metrics_path, final_metrics)
    journal_payload.update(
        {
            "status": "completed",
            "completed_at": evaluation_timestamp,
            "final_metrics": outputs["final_metrics"],
            "final_metrics_sha256": sha256_file(guard.final_metrics_path),
        }
    )
    write_json(guard.started_path, journal_payload)

    print("Final one-shot test metrics", flush=True)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        print(f"{key}: {overall[key]:.6f}", flush=True)
    for label in label_order:
        values = overall["per_class"][label]
        print(
            f"{label}: precision={values['precision']:.6f}, "
            f"recall={values['recall']:.6f}, f1={values['f1']:.6f}, "
            f"support={values['support']}",
            flush=True,
        )
    for name in ("english_test", "translated_vietnamese_test"):
        values = subsets[name]
        print(
            f"{name}: rows={values['rows']}, accuracy={_format(values['accuracy'])}, "
            f"macro_f1_supported={_format(values['macro_f1_supported_classes'])}, "
            f"phishing_recall={_recall(values, 'phishing')}",
            flush=True,
        )
    for item in smoke:
        print(
            f"smoke | {item['scenario']} | {item['prediction']} | "
            f"confidence={item['confidence']:.6f}",
            flush=True,
        )
    print("Held-out test evaluation count: 1", flush=True)
    print("Post-test tuning performed: False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
