"""Train Version 2 TF-IDF candidates with train and validation data only."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.modeling import (  # noqa: E402
    build_experiment_pipeline,
    compute_metrics,
    iter_experiment_specs,
    load_json,
    metrics_row,
    plot_model_comparison,
    read_email_split,
    runtime_versions,
    sha256_file,
    split_summary,
    validate_labels,
    write_json,
    write_metric_artifacts,
)
from vietmailguard.v2_modeling import (  # noqa: E402
    assert_no_cross_split_values,
    assert_only_model_selection_inputs,
    classifier_confidence_capability,
    evaluate_validation_subsets,
    rank_bilingual_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Version 2 bilingual TF-IDF models using validation data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "v2_tfidf_experiments.json",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    """Resolve one repository-relative configured path."""
    return PROJECT_ROOT / value


def _value(value: float | None) -> float | None:
    return None if value is None else float(value)


def _recall(metrics: dict[str, Any], label: str) -> float | None:
    return _value(metrics["per_class"][label]["recall"])


def _load_v1_reference(config: dict[str, Any]) -> dict[str, Any] | None:
    selected_path = project_path(config["v1_reference"]["selected_model"])
    if not selected_path.exists():
        return None
    selected = load_json(selected_path)
    metrics = selected.get("validation_metrics", {})
    return {
        "artifact": selected_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_sha256": sha256_file(selected_path),
        "selected_experiment_id": selected.get("selected_experiment_id"),
        "feature_set": selected.get("feature_set"),
        "classifier": selected.get("classifier"),
        "validation_rows": selected.get("data", {}).get("validation", {}).get("rows"),
        "validation_macro_f1": metrics.get("macro_f1"),
        "validation_phishing_recall": metrics.get("phishing_recall"),
        "translated_vietnamese_metrics": None,
        "translated_vietnamese_metrics_reason": (
            "Version 1 was trained and validated on English email only."
        ),
        "comparison_limit": (
            "Descriptive only: Version 1 and Version 2 use different leakage-safe "
            "validation cohorts."
        ),
    }


def _comparison_row(
    spec: Any,
    global_metrics: dict[str, Any],
    subsets: dict[str, dict[str, Any]],
    *,
    fit_seconds: float,
    predict_seconds: float,
    confidence_capability: str,
    deployability_rank: int,
    candidate_path: Path,
    v1_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    row = metrics_row(
        spec,
        global_metrics,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )
    english = subsets["english_original_artifact"]
    translated = subsets["translated_vietnamese"]
    existing = subsets["existing_translated_vietnamese"]
    controlled = subsets["controlled_translated_augmentation"]
    english_macro = _value(english["macro_f1_supported_classes"])
    translated_macro = _value(translated["macro_f1_supported_classes"])
    row.update(
        {
            "confidence_capability": confidence_capability,
            "english_validation_rows": english["rows"],
            "english_macro_f1": english_macro,
            "english_normal_recall": _recall(english, "normal"),
            "english_spam_recall": _recall(english, "spam"),
            "english_phishing_recall": _recall(english, "phishing"),
            "translated_vietnamese_validation_rows": translated["rows"],
            "translated_vietnamese_macro_f1_supported_classes": translated_macro,
            "translated_vietnamese_three_class_macro_f1": _value(
                translated["three_class_macro_f1"]
            ),
            "translated_vietnamese_normal_recall": _recall(translated, "normal"),
            "translated_vietnamese_spam_recall": _recall(translated, "spam"),
            "translated_vietnamese_phishing_recall": _recall(translated, "phishing"),
            "existing_translated_vietnamese_rows": existing["rows"],
            "existing_translated_vietnamese_macro_f1_supported_classes": _value(
                existing["macro_f1_supported_classes"]
            ),
            "controlled_translation_validation_rows": controlled["rows"],
            "controlled_translation_macro_f1": _value(
                controlled["macro_f1_supported_classes"]
            ),
            "language_balance_gap": abs(english_macro - translated_macro)
            if english_macro is not None and translated_macro is not None
            else None,
            "normal_false_positive_rate": 1.0
            - float(global_metrics["per_class"]["normal"]["recall"]),
            "deployability_rank": int(deployability_rank),
            "candidate_pipeline": candidate_path.relative_to(PROJECT_ROOT).as_posix(),
            "candidate_pipeline_sha256": sha256_file(candidate_path),
        }
    )
    if v1_reference is None:
        row["v1_reference_macro_f1"] = None
        row["english_macro_f1_delta_vs_v1_reference"] = None
        row["v1_reference_phishing_recall"] = None
        row["english_phishing_recall_delta_vs_v1_reference"] = None
    else:
        v1_macro = _value(v1_reference["validation_macro_f1"])
        v1_phishing = _value(v1_reference["validation_phishing_recall"])
        row["v1_reference_macro_f1"] = v1_macro
        row["english_macro_f1_delta_vs_v1_reference"] = (
            english_macro - v1_macro
            if english_macro is not None and v1_macro is not None
            else None
        )
        english_phishing = _recall(english, "phishing")
        row["v1_reference_phishing_recall"] = v1_phishing
        row["english_phishing_recall_delta_vs_v1_reference"] = (
            english_phishing - v1_phishing
            if english_phishing is not None and v1_phishing is not None
            else None
        )
    return row


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{float(value):.6f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, record in frame[columns].iterrows():
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, (float, np.floating)):
                values.append(_format_metric(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    path: Path,
    comparison: pd.DataFrame,
    recommendation: dict[str, Any],
    v1_reference: dict[str, Any] | None,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    columns = [
        "validation_rank",
        "feature_set",
        "classifier",
        "macro_f1",
        "phishing_recall",
        "english_macro_f1",
        "translated_vietnamese_macro_f1_supported_classes",
        "normal_false_positive_rate",
    ]
    lines = [
        "# Version 2 bilingual TF-IDF validation experiments",
        "",
        "## Scientific boundary",
        "",
        "The experiment fitted every TF-IDF vocabulary and classifier on the Version 2 "
        "training split only. Validation was used for model comparison. The locked "
        "held-out split was not read or evaluated.",
        "",
        f"- Train rows: **{len(train):,}**",
        f"- Validation rows: **{len(validation):,}**",
        "- Random seed: **42**",
        "- Required experiments completed: **9**",
        "",
        "## Validation ranking",
        "",
        _markdown_table(comparison, columns),
        "",
        "Ranking uses Macro F1 first, phishing recall second, then cross-language "
        "balance, normal false-positive rate, and deployability. The Vietnamese "
        "validation subset contains no phishing rows, so its three-class Macro F1 and "
        "phishing recall are N/A. The reported Vietnamese Macro F1 is calculated only "
        "over supported classes (normal and spam).",
        "",
        "## Recommended classical candidate",
        "",
        f"- Experiment: `{recommendation['experiment_id']}`",
        f"- Representation: {recommendation['feature_set']}",
        f"- Classifier: {recommendation['classifier']}",
        f"- Combined validation Macro F1: {_format_metric(recommendation['macro_f1'])}",
        f"- Combined phishing recall: {_format_metric(recommendation['phishing_recall'])}",
        f"- English Macro F1: {_format_metric(recommendation['english_macro_f1'])}",
        "- This is a TF-IDF recommendation, not the final Version 2 production model. "
        "The multilingual embedding experiment remains pending.",
        "",
        "## Confidence and calibration",
        "",
        "Multinomial Naive Bayes and Logistic Regression expose native probabilities. "
        "Linear SVM candidates remain uncalibrated and do not expose model confidence. "
        "Raw `decision_function` values are not reported as probabilities. Calibration "
        "is deferred until final model selection makes it necessary.",
        "",
        "## Evaluation by origin",
        "",
        f"- Original English artifact rows in validation: **{int(recommendation['english_validation_rows']):,}**",
        f"- Existing translated Vietnamese rows in validation: **{int(recommendation['existing_translated_vietnamese_rows']):,}**",
        f"- Controlled translated augmentation rows in validation: **{int(recommendation['controlled_translation_validation_rows']):,}**",
        "- Controlled augmentation is train-only by leakage policy, so its validation metrics are N/A.",
        "- No native Vietnamese validation benchmark exists.",
        "",
        "## Version 1 reference",
        "",
    ]
    if v1_reference is None:
        lines.append("Version 1 saved validation results were unavailable; no comparison was fabricated.")
    else:
        lines.extend(
            [
                f"- V1 selected experiment: `{v1_reference['selected_experiment_id']}`",
                f"- V1 saved validation Macro F1: {_format_metric(v1_reference['validation_macro_f1'])}",
                f"- V1 saved validation phishing recall: {_format_metric(v1_reference['validation_phishing_recall'])}",
                f"- V2 candidate English Macro F1 delta: {_format_metric(recommendation['english_macro_f1_delta_vs_v1_reference'])}",
                f"- V2 candidate English phishing-recall delta: {_format_metric(recommendation['english_phishing_recall_delta_vs_v1_reference'])}",
                "- V1 translated-Vietnamese validation metrics: N/A (V1 was English-only).",
                "- These deltas are descriptive, not a paired controlled comparison, because the "
                "V1 and V2 leakage-safe validation cohorts differ.",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Vietnamese evaluation uses translated data only.",
            "- Vietnamese validation has normal and spam but no phishing examples.",
            "- Controlled translations occur only in training and cannot measure generalization.",
            "- Class imbalance remains substantial; no duplicate oversampling was introduced.",
            "- Final production selection and held-out evaluation remain blocked until the "
            "multilingual embedding comparison is complete.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    assert_only_model_selection_inputs(config)
    label_order = [str(value) for value in config["label_order"]]
    train_path = project_path(config["data"]["train"])
    validation_path = project_path(config["data"]["validation"])

    print("Loading Version 2 train and validation only...", flush=True)
    train = read_email_split(train_path)
    validation = read_email_split(validation_path)
    validate_labels(train, label_order, "train")
    validate_labels(validation, label_order, "validation")
    assert_no_cross_split_values(train, validation, config["leakage_columns"])
    print(f"train: {split_summary(train)}", flush=True)
    print(f"validation: {split_summary(validation)}", flush=True)
    print("Train/validation leakage checks: PASS", flush=True)

    X_train = train[["subject", "body"]]
    y_train = train["label"].astype(str)
    X_validation = validation[["subject", "body"]]
    y_validation = validation["label"].astype(str)

    validation_directory = project_path(config["outputs"]["validation_directory"])
    candidate_directory = project_path(config["outputs"]["candidate_directory"])
    validation_directory.mkdir(parents=True, exist_ok=True)
    candidate_directory.mkdir(parents=True, exist_ok=True)
    v1_reference = _load_v1_reference(config)

    comparison_rows: list[dict[str, Any]] = []
    experiment_results: dict[str, Any] = {}
    candidate_manifest: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    specs = iter_experiment_specs(config)
    print(f"Running {len(specs)} configured validation experiments...", flush=True)

    with tempfile.TemporaryDirectory(prefix="vietmailguard_v2_tfidf_") as cache_root:
        for index, spec in enumerate(specs, start=1):
            feature_config = config["features"][spec.feature_name]
            classifier_config = config["classifiers"][spec.classifier_name]
            if (
                classifier_config["kind"] == "multinomial_nb"
                and not bool(feature_config.get("nonnegative", False))
            ):
                skipped.append(
                    {
                        "experiment_id": spec.experiment_id,
                        "reason": "MultinomialNB requires nonnegative features",
                    }
                )
                continue

            print(f"[{index}/{len(specs)}] {spec.experiment_id}: fitting", flush=True)
            pipeline = build_experiment_pipeline(
                spec,
                config,
                cache_directory=Path(cache_root) / spec.feature_name,
            )
            started = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started

            started = time.perf_counter()
            predictions = np.asarray(pipeline.predict(X_validation), dtype=object)
            predict_seconds = time.perf_counter() - started
            global_metrics = compute_metrics(y_validation, predictions, label_order)
            subsets = evaluate_validation_subsets(validation, predictions, label_order)
            confidence_capability = classifier_confidence_capability(pipeline)

            if confidence_capability == "native_predict_proba":
                probability_sample = np.asarray(
                    pipeline.predict_proba(X_validation.iloc[:5]), dtype=float
                )
                if not np.allclose(probability_sample.sum(axis=1), 1.0, atol=1e-7):
                    raise RuntimeError(f"Invalid probabilities for {spec.experiment_id}")

            candidate_path = candidate_directory / f"{spec.experiment_id}.joblib"
            pipeline.memory = None
            joblib.dump(pipeline, candidate_path, compress=3)
            reloaded = joblib.load(candidate_path)
            smoke_original = np.asarray(pipeline.predict(X_validation.iloc[:3]), dtype=object)
            smoke_reloaded = np.asarray(reloaded.predict(X_validation.iloc[:3]), dtype=object)
            if not np.array_equal(smoke_original, smoke_reloaded):
                raise RuntimeError(f"Reload smoke test failed for {spec.experiment_id}")

            write_metric_artifacts(
                validation_directory,
                spec.experiment_id,
                global_metrics,
            )
            row = _comparison_row(
                spec,
                global_metrics,
                subsets,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                confidence_capability=confidence_capability,
                deployability_rank=int(classifier_config["deployability_rank"]),
                candidate_path=candidate_path,
                v1_reference=v1_reference,
            )
            comparison_rows.append(row)
            experiment_results[spec.experiment_id] = {
                "feature_set": spec.feature_display_name,
                "classifier": spec.classifier_display_name,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "confidence_capability": confidence_capability,
                "global_validation": global_metrics,
                "validation_subsets": subsets,
                "candidate_pipeline": candidate_path.relative_to(PROJECT_ROOT).as_posix(),
                "candidate_pipeline_sha256": sha256_file(candidate_path),
                "reload_smoke_test": True,
            }
            candidate_manifest.append(
                {
                    "experiment_id": spec.experiment_id,
                    "path": candidate_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(candidate_path),
                    "confidence_capability": confidence_capability,
                    "reload_smoke_test": True,
                }
            )
            print(
                f"[{index}/{len(specs)}] {spec.experiment_id}: "
                f"macro_f1={global_metrics['macro_f1']:.6f}, "
                f"phishing_recall={global_metrics['per_class']['phishing']['recall']:.6f}, "
                f"english_macro_f1={subsets['english_original_artifact']['macro_f1_supported_classes']:.6f}, "
                f"translated_vi_macro_f1_supported="
                f"{subsets['translated_vietnamese']['macro_f1_supported_classes']:.6f}, "
                f"fit_seconds={fit_seconds:.2f}",
                flush=True,
            )

    comparison = rank_bilingual_comparison(pd.DataFrame(comparison_rows))
    comparison_path = project_path(config["outputs"]["comparison"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    plot_frame = comparison.copy()
    plot_frame["selected"] = plot_frame["recommended_tfidf_candidate"]
    plot_model_comparison(plot_frame, project_path(config["outputs"]["comparison_plot"]))

    recommendation = comparison.iloc[0].to_dict()
    metrics_artifact = {
        "schema_version": 1,
        "experiment_scope": config["experiment_scope"],
        "held_out_data_accessed": False,
        "selection_complete_for_tfidf_candidates": True,
        "final_production_model_selected": False,
        "pending_experiment": config["selection"]["pending_experiment"],
        "seed": int(config["seed"]),
        "experiment_config_sha256": sha256_file(args.config),
        "label_order": label_order,
        "input_data": {
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
        "leakage_checks": {
            "passed": True,
            "columns": config["leakage_columns"],
        },
        "ranking_policy": config["selection"],
        "v1_reference": v1_reference,
        "experiments": experiment_results,
        "skipped_experiments": skipped,
        "recommended_tfidf_candidate": {
            key: recommendation[key]
            for key in (
                "experiment_id",
                "feature_set",
                "classifier",
                "macro_f1",
                "phishing_recall",
                "english_macro_f1",
                "translated_vietnamese_macro_f1_supported_classes",
                "translated_vietnamese_three_class_macro_f1",
                "normal_false_positive_rate",
                "confidence_capability",
                "candidate_pipeline",
                "candidate_pipeline_sha256",
            )
        },
        "calibration_policy": (
            "Linear SVM candidates are uncalibrated. Raw decision scores are not "
            "probabilities. Calibrate only if final selection later requires it."
        ),
        "runtime_versions": runtime_versions(),
    }
    write_json(project_path(config["outputs"]["metrics"]), metrics_artifact)
    write_json(
        project_path(config["outputs"]["candidate_manifest"]),
        {
            "schema_version": 1,
            "final_production_models": [],
            "candidates": candidate_manifest,
        },
    )
    _write_report(
        project_path(config["outputs"]["report"]),
        comparison,
        recommendation,
        v1_reference,
        train,
        validation,
    )

    print("Validation ranking complete.", flush=True)
    print(
        comparison[
            [
                "validation_rank",
                "experiment_id",
                "macro_f1",
                "phishing_recall",
                "english_macro_f1",
                "translated_vietnamese_macro_f1_supported_classes",
                "confidence_capability",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Recommended TF-IDF candidate: {recommendation['experiment_id']}", flush=True)
    print("Final production model selected: False", flush=True)
    print("Held-out data accessed: False", flush=True)
    print(f"Comparison: {comparison_path.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
