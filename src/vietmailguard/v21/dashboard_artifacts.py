"""Read-only V2.1 development artifacts for the existing dashboards."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ARTIFACT_PATHS = {
    "development": "results/v21_multilingual/development_validation_metrics.json",
    "bootstrap": "data/v21/splits/development_bootstrap/manifest.json",
    "audit": "reports/v21/dedup_audit_summary.json",
    "curation": "reports/v21/label_curation_summary.json",
    "approval": "reports/v21/source_label_approval_summary.json",
    "trec_curation": "reports/v21/trec06c_curation_summary.json",
    "trec_review": "reports/v21/trec06c_human_review_summary.json",
    "trec_split": "results/v21_multilingual/trec06c_binary/split_report.json",
    "trec_leakage": "results/v21_multilingual/trec06c_binary/leakage_report.json",
    "trec_validation": "results/v21_multilingual/trec06c_binary/validation_metrics.json",
    "synthetic_data": "data/v21/processed/research_synthetic_zh/manifest.json",
    "synthetic_validation": "results/v21_multilingual/research_synthetic_zh/validation_metrics.json",
    "synthetic_current": "results/v21_multilingual/research_synthetic_zh/current_svm_comparison.json",
    "benchmark": "data/v21/evaluation/chinese_real_world_benchmark/manifest.json",
    "runtime": "config/v21/runtime_model.json",
    "training": "config/v21/development_training.json",
    "phishvn": "reports/v21/phishvn_url_audit.json",
    "curation_counts": "reports/v21/label_curation_counts.csv",
}

EXPECTED_SCOPES = {
    "development": ("scope", "development_validation_only"),
    "bootstrap": ("mode", "development_bootstrap"),
    "trec_validation": ("experiment", "trec06c_binary"),
    "synthetic_data": ("scope", "research_synthetic_zh_only"),
    "synthetic_validation": ("scope", "research_synthetic_zh_development_validation_only"),
    "synthetic_current": ("scope", "research_synthetic_zh_current_svm_matched_validation_only"),
    "benchmark": ("scope", "chinese_real_world_benchmark_preparation_only"),
    "runtime": ("scope", "development_candidate_only"),
    "phishvn": ("modality", "url_not_email"),
}


@dataclass
class ResearchDashboardArtifacts:
    """Aggregate JSON only; never load email rows, model binaries or credentials."""

    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues: dict[str, str] = field(default_factory=dict)


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def load_research_artifacts(root: Path) -> ResearchDashboardArtifacts:
    """Load a bounded allowlist, isolating missing or invalid sections."""
    result = ResearchDashboardArtifacts()
    for name, relative in ARTIFACT_PATHS.items():
        path = Path(root) / relative
        try:
            if not path.is_file():
                result.issues[name] = "missing"
                continue
            if path.stat().st_size > 2_000_000:
                raise ValueError("Aggregate artifact is too large")
            if name == "curation_counts":
                with path.open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames != ["dataset", "label_status", "rows"]:
                        raise ValueError("Unexpected aggregate CSV schema")
                    rows = []
                    for row in reader:
                        count = int(row["rows"])
                        if count < 0 or row["label_status"] not in {"eligible", "review", "weak_label", "exclude"}:
                            raise ValueError("Invalid status count")
                        rows.append({"dataset": row["dataset"], "label_status": row["label_status"], "rows": count})
                    document = {"rows": rows}
            else:
                document = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            if not isinstance(document, dict):
                raise ValueError("Expected JSON object")
            expected = EXPECTED_SCOPES.get(name)
            if expected and document.get(expected[0]) != expected[1]:
                raise ValueError("Unexpected evaluation scope")
            for key in ("candidates", "by_split_language_label", "by_split_language_origin", "by_split_source"):
                if key in document and (not isinstance(document[key], list)
                                        or not all(isinstance(row, dict) for row in document[key])):
                    raise ValueError("Invalid aggregate rows")
            result.documents[name] = document
        except (OSError, UnicodeError, ValueError, TypeError, csv.Error, RecursionError):
            result.issues[name] = "invalid"
    return result


def artifact_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Invalidate presentation cache when a saved artifact changes."""
    entries = []
    for relative in ARTIFACT_PATHS.values():
        try:
            stat = (Path(root) / relative).stat()
            entries.append((relative, stat.st_mtime_ns, stat.st_size))
        except OSError:
            entries.append((relative, -1, -1))
    return tuple(entries)


def error_directions(metrics: dict[str, Any]) -> dict[str, int | None]:
    """Use explicit error directions from the saved development metric schema."""
    return {
        "normal_to_other": metrics.get("non_normal_false_positives"),
        "other_to_normal": metrics.get("normal_false_positives"),
    }


def candidate_rows(candidates: list[dict[str, Any]], *, binary: bool = False) -> list[dict[str, Any]]:
    """Build comparable rows without substituting classifier-only embedding cost."""
    rows = []
    for candidate in candidates:
        metrics = candidate.get("metrics", candidate.get("validation_metrics", {}))
        if not isinstance(metrics, dict):
            continue
        row = {
            "candidate": candidate.get("candidate", candidate.get("classifier")),
            "experiment": candidate.get("experiment"),
            "train_rows": candidate.get("train_rows"),
            "validation_rows": metrics.get("rows", candidate.get("validation_rows")),
            "accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("macro_f1"),
            "fit_seconds": candidate.get("fit_seconds", candidate.get("fit_seconds_from_saved_result")),
        }
        embedding = candidate.get("family") == "embedding" or "embedding" in str(candidate.get("representation", ""))
        end_to_end = metrics.get("end_to_end_smoke_ms_per_email", candidate.get("end_to_end_prediction_ms_per_email"))
        row["prediction_ms_per_email"] = end_to_end if embedding else metrics.get(
            "prediction_ms_per_email", candidate.get("prediction_ms_per_email"))
        if binary:
            row["ham_to_spam"] = metrics.get("false_positives_ham_as_spam")
            row["spam_to_ham"] = metrics.get("false_negatives_spam_as_ham")
        else:
            row["phishing_recall"] = metrics.get("phishing_recall")
            row.update(error_directions(metrics))
        rows.append(row)
    return rows


def paired_macro_delta(document: dict[str, Any]) -> float | None:
    """Subtract already saved comparable Validation values, without evaluation."""
    try:
        first = document["A_current_candidate"]["validation_metrics"]["macro_f1"]
        second = document["B_synthetic_added"]["validation_metrics"]["macro_f1"]
        if isinstance(first, bool) or isinstance(second, bool):
            return None
        delta = float(second) - float(first)
        return delta if math.isfinite(delta) else None
    except (KeyError, TypeError, ValueError):
        return None


def load_development_validation(root: Path) -> dict[str, Any] | None:
    path = root / "results" / "v21_multilingual" / "development_validation_metrics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("scope") != "development_validation_only":
        raise ValueError("Unexpected V2.1 validation scope")
    return data


def load_development_dataset(root: Path) -> dict[str, Any] | None:
    path = root / "data" / "v21" / "splits" / "development_bootstrap" / "manifest.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("mode") != "development_bootstrap":
        raise ValueError("Unexpected V2.1 dataset scope")
    return data
