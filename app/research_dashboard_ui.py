"""Presentation of saved research aggregates; no evaluation or email-row access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from i18n import class_label, t
from vietmailguard.v21.dashboard_artifacts import (
    ARTIFACT_PATHS, ResearchDashboardArtifacts, artifact_signature,
    candidate_rows, load_research_artifacts, paired_macro_delta,
)


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def records(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


@st.cache_data(show_spinner=False)
def _cached_artifacts(root: str, signature: tuple) -> ResearchDashboardArtifacts:
    return load_research_artifacts(Path(root))


def research_artifacts(root: Path, language: str) -> ResearchDashboardArtifacts:
    data = _cached_artifacts(str(root), artifact_signature(root))
    if data.issues:
        with st.expander(t("rd_artifact_issues", language)):
            for name, issue in data.issues.items():
                st.write(f"{ARTIFACT_PATHS[name]} — {t('rd_' + issue, language)}")
    st.caption(t("rd_saved_only", language))
    return data


def document(data: ResearchDashboardArtifacts, key: str, language: str) -> dict[str, Any]:
    value = data.documents.get(key, {})
    if not value:
        st.info(t("rd_section_unavailable", language))
    else:
        st.caption(f"{t('rd_artifact', language)}: {ARTIFACT_PATHS[key]}")
        if value.get("completed_at_utc") or value.get("created_at_utc"):
            st.caption(str(value.get("completed_at_utc", value.get("created_at_utc"))))
    return value


def table(rows: list[dict[str, Any]], language: str) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame = frame.dropna(axis=1, how="all")
    if "label" in frame:
        frame["label"] = frame["label"].map(lambda value: class_label(str(value), language)
                                           if value in {"normal", "spam", "phishing"} else value)
    if "language" in frame:
        frame["language"] = frame["language"].map(lambda value: t("email_language_" + str(value), language)
                                                  if value in {"en", "vi", "zh"} else value)
    metric_columns = {"accuracy", "macro_f1", "phishing_recall", "precision", "recall", "f1",
                      "represented_macro_f1", "macro_f1_represented_classes", "fit_seconds",
                      "prediction_ms_per_email"}
    formats = {t("rd_" + column, language): "{:.6f}" for column in frame.columns
               if column in metric_columns and pd.api.types.is_numeric_dtype(frame[column])}
    frame = frame.rename(columns={column: t("rd_" + column, language) for column in frame.columns})
    st.dataframe(frame.style.format(formats, na_rep="N/A"), hide_index=True, width="stretch")


def counts(values: dict[str, Any], language: str) -> None:
    table([{"measure": t("rd_" + key, language), "rows": value}
           for key, value in values.items() if value is not None], language)


def render_bootstrap(data: ResearchDashboardArtifacts, language: str) -> None:
    manifest = document(data, "bootstrap", language)
    if not manifest:
        return
    st.info(t("rd_bootstrap_notice", language))
    counts(mapping(manifest.get("output_rows")), language)
    for title, key in (("class_language", "by_split_language_label"),
                       ("origin", "by_split_language_origin"), ("sources", "by_split_source")):
        st.subheader(t("rd_" + title, language))
        table(records(manifest.get(key)), language)
    with st.expander(t("rd_group_integrity", language)):
        table([{"fingerprint": key, **mapping(value)} for key, value in mapping(
            manifest.get("duplicate_groups")).items()], language)
        table([mapping(manifest.get("leakage"))], language)
        counts({"parent_linked_rows": manifest.get("parent_linked_rows")}, language)
    table([{"development_training_ready": manifest.get("development_training_ready"),
            "final_evaluation_ready": manifest.get("final_evaluation_ready")}], language)
    st.caption(t("rd_language_limits", language))


def render_audit(data: ResearchDashboardArtifacts, language: str) -> None:
    audit = document(data, "audit", language)
    if audit:
        counts({key: audit.get(key) for key in ("input_rows", "v21_source_rows",
                "read_only_v2_reference_rows", "conflict_rows", "physically_removed_rows")}, language)
        st.caption(t("rd_audit_scope", language))
        table([{"dataset": source, "rows": number}
               for source, number in mapping(audit.get("rows_per_dataset")).items()], language)
        with st.expander(t("rd_group_integrity", language)):
            table([{"fingerprint": name, **mapping(value)} for name, value in mapping(
                audit.get("duplicate_groups")).items()], language)
            st.caption(t("rd_overlap_notice", language))
        with st.expander(t("rd_raw_languages", language)):
            table([{"language": name, "rows": value} for name, value in mapping(
                audit.get("rows_per_language")).items()], language)
    st.subheader(t("rd_curation", language))
    approval = document(data, "approval", language)
    if approval:
        counts(mapping(approval.get("status_counts")), language)
        counts({key: approval.get(key) for key in ("new_eligible_from_source_approval",
                "source_rules_active", "source_rules_pending", "source_rules_blocked")}, language)
    curation = document(data, "curation", language)
    if curation:
        table([{"language": lang, "label": label, "rows": number}
               for lang, values in mapping(curation.get("eligible_by_language_label")).items()
               for label, number in mapping(values).items()], language)
    table(records(data.documents.get("curation_counts", {}).get("rows")), language)
    st.caption(t("rd_status_notice", language))


def render_trec_data(data: ResearchDashboardArtifacts, language: str) -> None:
    split = document(data, "trec_split", language)
    if split:
        st.info(t("rd_binary_notice", language))
        counts(mapping(split.get("raw_labels")), language)
        counts({"quality_accepted": split.get("quality_accepted")}, language)
        table([{"reason": key, "rows": value} for key, value in mapping(
            split.get("quality_rejected")).items()], language)
        table([{"split": name, **mapping(value)} for name, value in mapping(
            split.get("split_counts")).items()], language)
        st.caption(t("rd_sealed_test", language))
        table([{key: value for key, value in mapping(split.get("group_stats")).items()
                if key != "class_targets"}], language)
        leakage = data.documents.get("trec_leakage", {})
        table([{key: leakage.get(key) for key in ("passed", "checked_rows", "groups")}], language)
    review = document(data, "trec_review", language)
    if review:
        counts({key: review.get(key) for key in ("human_decisions", "weak_label_unreviewed",
                "ham_conditional_pool_pending_approval", "proposed_bulk_eligible_now")}, language)
        table([{"label": key, "rows": value} for key, value in mapping(
            review.get("eligible_by_class")).items()], language)
        st.caption(t("rd_trec_approval_notice", language))


def render_synthetic_data(data: ResearchDashboardArtifacts, language: str) -> None:
    manifest = document(data, "synthetic_data", language)
    if not manifest:
        return
    st.warning(t("rd_synthetic_notice", language))
    counts({key: manifest.get(key) for key in ("source_rows", "research_usable_rows",
            "screened_out_rows", "synthetic_train_rows", "synthetic_challenge_rows",
            "official_training_eligible_rows", "synthetic_duplicate_groups",
            "cross_split_group_count", "cross_corpus_fingerprint_overlap_rows")}, language)
    table([{"split": split, "label": label, "rows": value}
           for split, key in (("train", "train_label_counts"), ("challenge", "challenge_label_counts"))
           for label, value in mapping(manifest.get(key)).items()], language)
    with st.expander(t("rd_screening", language)):
        table([{"reason": key, "rows": value} for key, value in mapping(
            manifest.get("screening_reason_counts")).items()], language)
        st.caption(t("rd_overlapping_flags", language))


def render_benchmark(data: ResearchDashboardArtifacts, language: str) -> None:
    benchmark = document(data, "benchmark", language)
    if benchmark:
        table([{key: benchmark.get(key) for key in ("benchmark_status", "candidate_rows",
                "accepted_benchmark_rows", "final_evaluation_ready", "reference_fingerprints_checked",
                "held_out_test_opened", "model_evaluated")}], language)
    st.warning(t("rd_benchmark_notice", language))
    urls = data.documents.get("phishvn", {})
    if urls:
        st.subheader(t("rd_url_dataset", language))
        st.caption(t("rd_url_notice", language))
        counts({"core_rows": urls.get("core_rows")}, language)
        for key in ("label", "split"):
            table([{key: name, "rows": value} for name, value in mapping(
                mapping(urls.get("distributions")).get(key)).items()], language)


def render_dataset_research(root: Path, language: str) -> None:
    data = research_artifacts(root, language)
    tabs = st.tabs([t("rd_" + key, language) for key in (
        "bootstrap", "audit", "trec", "synthetic", "benchmarks")])
    for tab, renderer in zip(tabs, (render_bootstrap, render_audit, render_trec_data,
                                  render_synthetic_data, render_benchmark), strict=True):
        with tab:
            renderer(data, language)


def render_metric_details(metrics: dict[str, Any], language: str, *, binary: bool = False) -> None:
    table([{"label": label, **{key: values.get(key) for key in ("precision", "recall", "f1", "support")}}
           for label, values in mapping(metrics.get("per_class")).items() if isinstance(values, dict)], language)
    matrix = metrics.get("confusion_matrix")
    order = metrics.get("confusion_matrix_order", ["ham", "spam"] if binary
                        else ["normal", "spam", "phishing"])
    if (isinstance(matrix, list) and isinstance(order, list) and len(matrix) == len(order)
            and all(isinstance(row, list) and len(row) == len(order) for row in matrix)):
        st.caption(t("rd_matrix_axes", language))
        names = [class_label(label, language) if label in {"normal", "spam", "phishing"}
                 else str(label) for label in order]
        st.dataframe(pd.DataFrame(matrix, index=names, columns=names), width="stretch")
    if not binary:
        table([{"language": lang, "rows": subset.get("rows"),
                "represented_macro_f1": subset.get("macro_f1_supported_classes"),
                "phishing_recall": subset.get("phishing_recall"),
                "normal_to_other": subset.get("non_normal_false_positives"),
                **{f"{label}_support": mapping(subset.get("support")).get(label)
                   for label in ("normal", "spam", "phishing")}}
               for lang, subset in mapping(metrics.get("by_language")).items()
               if isinstance(subset, dict)], language)
        table([{"data_origin": origin, "rows": subset.get("rows"),
                "represented_macro_f1": subset.get("macro_f1_supported_classes")}
               for origin, subset in mapping(metrics.get("by_data_origin")).items()
               if isinstance(subset, dict)], language)


def render_candidate_details(candidates: list[dict[str, Any]], language: str, *, binary: bool = False) -> None:
    for candidate in candidates:
        name = candidate.get("candidate", candidate.get("classifier", ""))
        label = f"{candidate.get('experiment', '')} {name}".strip()
        with st.expander(label):
            render_metric_details(mapping(candidate.get("metrics", candidate.get("validation_metrics"))),
                                  language, binary=binary)
            challenge = mapping(candidate.get("synthetic_challenge_metrics"))
            if challenge:
                st.caption(t("rd_challenge_notice", language))
                render_metric_details(challenge, language)


def render_development_models(data: ResearchDashboardArtifacts, language: str) -> None:
    development = document(data, "development", language)
    if not development:
        return
    st.info(t("rd_bootstrap_notice", language))
    runtime = data.documents.get("runtime", {})
    counts({key: development.get(key) for key in ("train_rows", "validation_rows")}, language)
    table([{"configured_candidate": runtime.get("candidate_id"), "model_version": runtime.get("model_version"),
            "recommended_candidate": development.get("development_recommendation"),
            "probability_status": runtime.get("probability_status")}], language)
    st.caption(t("rd_configured_notice", language))
    candidates = records(development.get("candidates"))
    table(candidate_rows(candidates), language)
    st.caption(t("rd_latency_notice", language))
    st.caption(t("rd_error_directions", language))
    st.caption(t("rd_language_limits", language))
    render_candidate_details(candidates, language)
    with st.expander(t("rd_configuration", language)):
        training = data.documents.get("training", {})
        table([{"seed": development.get("seed"), "max_input_characters": training.get("max_input_characters"),
                "final_test_evaluated": development.get("final_test_evaluated"),
                "final_production_selected": development.get("final_production_selected")}], language)
        table([mapping(development.get("development_leakage_preflight"))], language)
        table([mapping(development.get("xlm_roberta"))], language)
        st.json(mapping(training.get("tfidf")), expanded=False)
        st.caption(t("rd_resource_gate", language))


def render_trec_models(data: ResearchDashboardArtifacts, language: str) -> None:
    validation = document(data, "trec_validation", language)
    if not validation:
        return
    st.warning(t("rd_binary_notice", language))
    table([{"recommended_candidate": validation.get("recommended_candidate"),
            "test_evaluated": validation.get("test_evaluated")}], language)
    candidates = records(validation.get("candidates"))
    table(candidate_rows(candidates, binary=True), language)
    st.caption(t("rd_latency_notice", language))
    render_candidate_details(candidates, language, binary=True)


def render_synthetic_models(data: ResearchDashboardArtifacts, language: str) -> None:
    st.warning(t("rd_synthetic_notice", language))
    paired = document(data, "synthetic_current", language)
    if paired:
        st.subheader(t("rd_current_pair", language))
        candidates = []
        for key, name in (("A_current_candidate", "A"), ("B_synthetic_added", "B")):
            candidate = mapping(paired.get(key))
            if candidate:
                candidates.append({**candidate, "experiment": name, "classifier": "LinearSVC"})
        table(candidate_rows(candidates), language)
        delta = paired_macro_delta(paired)
        if delta is not None:
            st.write(t("rd_delta", language, value=f"{delta:+.6f}"))
        st.caption(t("rd_pair_notice", language))
        render_candidate_details(candidates, language)
    secondary = document(data, "synthetic_validation", language)
    if secondary:
        st.subheader(t("rd_secondary_pair", language))
        candidates = records(secondary.get("candidates"))
        table(candidate_rows(candidates), language)
        st.caption(t("rd_secondary_notice", language))
        render_candidate_details(candidates, language)
    with st.expander(t("rd_synthetic_challenge", language)):
        st.caption(t("rd_challenge_notice", language))
        groups = []
        for key, name in (("A_current_candidate", "A current"), ("B_synthetic_added", "B current")):
            candidate = mapping(paired.get(key))
            if candidate:
                groups.append({**candidate, "experiment": name, "classifier": "LinearSVC"})
        groups.extend(records(secondary.get("candidates")))
        table([{"experiment": candidate.get("experiment"), "candidate": candidate.get("classifier"),
                **{key: mapping(candidate.get("synthetic_challenge_metrics")).get(key)
                   for key in ("rows", "accuracy", "macro_f1_represented_classes")}}
               for candidate in groups], language)


def render_model_research(root: Path, language: str) -> None:
    data = research_artifacts(root, language)
    tabs = st.tabs([t("rd_" + key, language) for key in ("development", "trec", "synthetic", "benchmarks")])
    for tab, renderer in zip(tabs, (render_development_models, render_trec_models,
                                  render_synthetic_models, render_benchmark), strict=True):
        with tab:
            renderer(data, language)
