"""Inactive TREC06C approval proposal and reproducible local human-review samples."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from .paths import V21Paths, resolve_v21_paths
from .trec_curation import (
    LABELS, QUEUE_FIELDS, SEED, _audit_connection, _conflict_groups, _eligible_gate,
    _length_bucket, _parts, _read_decisions, _validate_part_checkpoint,
    _validate_policy, _write_csv, quality_flags,
)


PLAN_VERSION = 1
HAM_PER_LENGTH = 120
SPAM_PER_FOCUS = 50
DEEP_QUEUE_FIELDS = QUEUE_FIELDS + (
    "review_cohort", "review_stratum", "review_focus", "sampling_frame_size",
)
DISPOSITION_FIELDS = (
    "id", "source", "raw_label", "original_path", "proposed_status",
    "reason", "quality_flags", "length_bucket", "exact_group_id",
    "normalized_group_id", "template_group_id", "content_hash",
)
SPAM_DISPOSITION_FIELDS = DISPOSITION_FIELDS + ("review_focus",)


def _hash_ids(rows: list[dict[str, str]]) -> str:
    ids = "\n".join(sorted(row["id"] for row in rows))
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()


def _offer(bucket: list[tuple[str, dict[str, str]]], row: dict[str, str], quota: int) -> None:
    priority = hashlib.sha256(f"trec11b1:{SEED}:{row['id']}".encode()).hexdigest()
    bucket.append((priority, row))
    bucket.sort(key=lambda item: item[0])
    if len(bucket) > quota:
        bucket.pop()


def _focus(subject: str, body: str) -> str:
    """Two-cue review routing only; never returns a class label."""
    text = (subject + " " + body).casefold()
    credential = bool(re.search(r"账户|账号|登录|密碼|密码|验证码|驗證碼|身份验证", text))
    account_action = bool(re.search(r"点击|點擊|立即|链接|連結|核实|验证|驗證|更新", text))
    financial = bool(re.search(r"转账|轉帳|汇款|匯款|银行卡|銀行卡|中奖|中獎|奖金|獎金|投资|投資|手续费|手續費", text))
    financial_action = bool(re.search(r"支付|领取|領取|提供|联系|聯繫|账户|帳戶|付款", text))
    commercial = bool(re.search(r"商品|产品|產品|课程|課程|优惠|優惠|折扣|广告|廣告|招聘|销售|銷售", text))
    call_to_action = bool(re.search(r"购买|購買|报名|報名|咨询|諮詢|联系|聯繫|订购|訂購|促销|促銷", text))
    if credential and account_action:
        return "credential_or_account_review"
    if financial and financial_action:
        return "financial_or_fraud_review"
    if commercial and call_to_action:
        return "commercial_review"
    return "other_review"


def _queue_row(raw: dict[str, str], audit, part: Path, paths: V21Paths,
               *, cohort: str, stratum: str, focus: str = "") -> dict[str, str]:
    flags = quality_flags(raw)
    return {
        "id": raw["id"], "source": raw["source"], "raw_label": raw["raw_label"],
        "original_path": raw["original_path"],
        "part_path": str(part.relative_to(paths.project_root)),
        "subject": raw["subject"], "body_excerpt": raw["body"][:300],
        "length_bucket": _length_bucket(raw["subject"], raw["body"]),
        "quality_stratum": "clean" if not flags else "flagged",
        "duplicate_stratum": "singleton", "quality_flags": ";".join(flags),
        "audit_reason": audit["audit_reason"],
        "exact_group_id": audit["exact_group_id"],
        "normalized_group_id": audit["normalized_group_id"],
        "template_group_id": audit["template_group_id"],
        "human_decision": "", "human_note": "",
        "review_cohort": cohort, "review_stratum": stratum,
        "review_focus": focus, "sampling_frame_size": "",
    }


def prepare_deep_review(paths: V21Paths | None = None) -> dict:
    """One streaming pass. Write private samples and inactive candidate dispositions.

    Does not write eligible labels or activate any source-level rule.
    Existing plan IDs are immutable so future review statistics keep their frame.
    """
    paths = paths or resolve_v21_paths()
    _validate_policy(paths)
    local = paths.curated / "trec06c"
    local.mkdir(parents=True, exist_ok=True)
    plan_path = local / "deep_review_plan.json"
    conn = _audit_connection(paths)
    csv.field_size_limit(100_000_000)
    ham_buckets: dict[str, list[tuple[str, dict[str, str]]]] = {
        name: [] for name in ("short", "medium", "long")
    }
    spam_buckets: dict[str, list[tuple[str, dict[str, str]]]] = {
        name: [] for name in (
            "credential_or_account_review", "financial_or_fraud_review",
            "commercial_review", "other_review",
        )
    }
    frame_counts = Counter()
    disposition_counts = Counter()
    source_counts = Counter()
    temporary = local / "ham_conditional_disposition.csv.tmp"
    spam_temporary = local / "spam_review_disposition.csv.tmp"
    try:
        conflicts = _conflict_groups(conn)
        with (temporary.open("w", encoding="utf-8", newline="") as handle,
              spam_temporary.open("w", encoding="utf-8", newline="") as spam_handle):
            writer = csv.DictWriter(handle, fieldnames=DISPOSITION_FIELDS)
            writer.writeheader()
            spam_writer = csv.DictWriter(spam_handle, fieldnames=SPAM_DISPOSITION_FIELDS)
            spam_writer.writeheader()
            for part in _parts(paths):
                _validate_part_checkpoint(conn, part)
                with part.open("r", encoding="utf-8-sig", newline="") as source:
                    for raw in csv.DictReader(source):
                        audit = conn.execute(
                            "SELECT * FROM rows WHERE audit_id=?", (f"trec06c|{raw['id']}",)
                        ).fetchone()
                        if (audit is None or audit["original_id"] != raw["id"]
                                or audit["raw_label"] != raw["raw_label"]
                                or audit["source_path"] != raw["original_path"]):
                            raise RuntimeError("TREC review plan provenance differs from audited source")
                        source_counts[raw["raw_label"]] += 1
                        flags = quality_flags(raw)
                        length = _length_bucket(raw["subject"], raw["body"])
                        duplicate = any(audit[col] for col in (
                            "exact_group_id", "normalized_group_id", "template_group_id"
                        ))
                        if raw["raw_label"] == "ham":
                            blocker = _eligible_gate(conn, audit, flags, conflicts)
                            # Narrowest prospective source-rule frame. MIME recovery,
                            # defects and HTML-only are held for separate review.
                            if not flags and blocker is None:
                                status, reason = "conditional_candidate", "clean_chinese_singleton"
                                frame_counts[("ham", length)] += 1
                                item = _queue_row(raw, audit, part, paths,
                                                  cohort="ham_probability", stratum=length)
                                _offer(ham_buckets[length], item, HAM_PER_LENGTH)
                            else:
                                status = "review"
                                reason = blocker or "mime_or_html_quality_review"
                            writer.writerow({
                                "id": raw["id"], "source": raw["source"],
                                "raw_label": raw["raw_label"],
                                "original_path": raw["original_path"],
                                "proposed_status": status, "reason": reason,
                                "quality_flags": ";".join(flags), "length_bucket": length,
                                "exact_group_id": audit["exact_group_id"],
                                "normalized_group_id": audit["normalized_group_id"],
                                "template_group_id": audit["template_group_id"],
                                "content_hash": raw["content_hash"],
                            })
                            disposition_counts[(status, reason)] += 1
                        elif raw["raw_label"] == "spam":
                            focus = _focus(raw["subject"], raw["body"])
                            reason = ("content_quality_review" if flags
                                      else "duplicate_group_needs_joint_review" if duplicate
                                      else "binary_spam_taxonomy_unverified")
                            spam_writer.writerow({
                                "id": raw["id"], "source": raw["source"],
                                "raw_label": raw["raw_label"],
                                "original_path": raw["original_path"],
                                "proposed_status": "review", "reason": reason,
                                "quality_flags": ";".join(flags), "length_bucket": length,
                                "exact_group_id": audit["exact_group_id"],
                                "normalized_group_id": audit["normalized_group_id"],
                                "template_group_id": audit["template_group_id"],
                                "content_hash": raw["content_hash"],
                                "review_focus": focus,
                            })
                            disposition_counts[("spam_review", reason)] += 1
                            if not flags and not duplicate and audit["language"] == "zh":
                                frame_counts[("spam", focus)] += 1
                                item = _queue_row(raw, audit, part, paths,
                                                  cohort="spam_focus", stratum=focus, focus=focus)
                                _offer(spam_buckets[focus], item, SPAM_PER_FOCUS)
                        else:
                            raise RuntimeError("Unexpected TREC raw label")
        ham_sample = [row for bucket in ham_buckets.values() for _, row in bucket]
        spam_sample = [row for bucket in spam_buckets.values() for _, row in bucket]
        for row in ham_sample:
            row["sampling_frame_size"] = str(frame_counts[("ham", row["review_stratum"])])
        for row in spam_sample:
            row["sampling_frame_size"] = str(frame_counts[("spam", row["review_stratum"])])
        ham_sample.sort(key=lambda row: (row["review_stratum"], row["id"]))
        spam_sample.sort(key=lambda row: (row["review_stratum"], row["id"]))
        if plan_path.is_file():
            previous = json.loads(plan_path.read_text(encoding="utf-8"))
            if (previous.get("plan_version") != PLAN_VERSION
                    or previous.get("ham_sample_id_sha256") != _hash_ids(ham_sample)
                    or previous.get("spam_sample_id_sha256") != _hash_ids(spam_sample)):
                raise RuntimeError("Frozen TREC review sample changed; do not overwrite review frame")
        _write_csv(local / "ham_probability_sample.csv", DEEP_QUEUE_FIELDS, ham_sample)
        _write_csv(local / "spam_focus_queue.csv", DEEP_QUEUE_FIELDS, spam_sample)
        temporary.replace(local / "ham_conditional_disposition.csv")
        spam_temporary.replace(local / "spam_review_disposition.csv")
        plan = {
            "plan_version": PLAN_VERSION, "seed": SEED,
            "ham_per_length": HAM_PER_LENGTH, "spam_per_focus": SPAM_PER_FOCUS,
            "source_counts": dict(source_counts),
            "ham_frame_by_length": {name: frame_counts[("ham", name)] for name in ham_buckets},
            "spam_focus_frame": {name: frame_counts[("spam", name)] for name in spam_buckets},
            "ham_disposition": {"|".join(key): value for key, value in sorted(disposition_counts.items())
                                if key[0] != "spam_review"},
            "spam_disposition": {key[1]: value for key, value in sorted(disposition_counts.items())
                                 if key[0] == "spam_review"},
            "ham_sample_rows": len(ham_sample), "spam_sample_rows": len(spam_sample),
            "ham_sample_id_sha256": _hash_ids(ham_sample),
            "spam_sample_id_sha256": _hash_ids(spam_sample),
            "bulk_rule_active": False, "human_review_completed": False,
        }
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return plan
    finally:
        conn.close()
        if temporary.is_file():
            temporary.unlink()
        if spam_temporary.is_file():
            spam_temporary.unlink()


def _wilson_95(errors: int, reviewed: int) -> tuple[float, float] | None:
    if reviewed == 0:
        return None
    z = 1.959963984540054
    p = errors / reviewed
    denominator = 1 + z * z / reviewed
    center = (p + z * z / (2 * reviewed)) / denominator
    radius = z * math.sqrt(p * (1 - p) / reviewed + z * z / (4 * reviewed * reviewed)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_deep_review(paths: V21Paths | None = None) -> dict:
    """Observed sample disagreements only; never infer a source-wide rate from no reviews."""
    paths = paths or resolve_v21_paths()
    local = paths.curated / "trec06c"
    plan = json.loads((local / "deep_review_plan.json").read_text(encoding="utf-8"))
    decisions = _read_decisions(local / "decisions.csv")
    samples = {}
    for name in ("ham_probability_sample", "spam_focus_queue"):
        with (local / f"{name}.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if _hash_ids(rows) != plan[f"{'ham' if name.startswith('ham') else 'spam'}_sample_id_sha256"]:
            raise RuntimeError("TREC sampling frame changed since the review plan")
        samples[name] = rows
    by_stratum = []
    for cohort, frames in (("ham_probability", plan["ham_frame_by_length"]),
                           ("spam_focus", plan["spam_focus_frame"])):
        queue_name = "ham_probability_sample" if cohort == "ham_probability" else "spam_focus_queue"
        for stratum, population in frames.items():
            selected = [row for row in samples[queue_name] if row["review_stratum"] == stratum]
            reviewed = [decisions[row["id"]] for row in selected if row["id"] in decisions]
            confirmed = [row for row in reviewed if row["decision"] in LABELS]
            discordant = sum(row["decision"] != ("normal" if cohort == "ham_probability" else "spam") for row in confirmed)
            complete_conclusive = len(confirmed) == len(selected) and len(selected) > 0
            interval = (_wilson_95(discordant, len(confirmed))
                        if cohort == "ham_probability" and complete_conclusive else None)
            by_stratum.append({
                "cohort": cohort, "stratum": stratum, "population_rows": population,
                "sample_rows": len(selected), "human_reviewed": len(reviewed),
                "conclusive_class_decisions": len(confirmed),
                "discordant_class_decisions": discordant,
                "unresolved_or_excluded_decisions": len(reviewed) - len(confirmed),
                "observed_discordance_rate": discordant / len(confirmed) if confirmed else None,
                "wilson_95": list(interval) if interval else None,
                "sampling_complete": len(reviewed) == len(selected),
                "uncertainty_status": (
                    "not_probability_sample" if cohort == "spam_focus"
                    else "complete_conclusive" if complete_conclusive
                    else "pending_or_unresolved_human_review"
                ),
            })
    class_counts = Counter(row["decision"] for row in decisions.values() if row["label_status"] == "eligible")
    status_counts = Counter(row["label_status"] for row in decisions.values())
    summary = {
        "human_decisions": len(decisions),
        "eligible_by_class": {label: class_counts[label] for label in ("normal", "spam", "phishing")},
        "human_review_status": dict(status_counts),
        "weak_label_unreviewed": sum(plan["source_counts"].values()) - len(decisions),
        "blocked_class_decisions": sum(row["decision"] in LABELS and row["label_status"] == "review"
                                       for row in decisions.values()),
        "ham_conditional_pool_pending_approval": sum(plan["ham_frame_by_length"].values()),
        "proposed_bulk_eligible_now": 0,
        "strata": by_stratum,
        "bulk_rule_active": False,
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "trec06c_human_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# TREC06C human review and conditional approval (V2.1 Prompt 11B.1)",
        "",
        "**No source-level bulk rule is active.** This report never turns a proposal or source label into ground truth.",
        "Source: [TREC 2006 Spam Track overview](https://trec.nist.gov/pubs/trec15/papers/SPAM06.OVERVIEW.pdf). Ham is mailing-list mail and Spam is spam-trap mail in a binary benchmark; the content was modified for the benchmark. These labels do not directly establish VietMailGuard's three classes.",
        "",
        "## Counts and prospective rule",
        "",
        f"- Indexed: {plan['source_counts']['ham']:,} Ham; {plan['source_counts']['spam']:,} Spam; {sum(plan['source_counts'].values()):,} total.",
        f"- Actual human decisions: {summary['human_decisions']:,}; eligible Normal {summary['eligible_by_class']['normal']:,}, Spam {summary['eligible_by_class']['spam']:,}, Phishing {summary['eligible_by_class']['phishing']:,}.",
        f"- Unreviewed weak labels: {summary['weak_label_unreviewed']:,}; human Review {status_counts['review']:,}; human Exclude {status_counts['exclude']:,}; class decisions held in Review by eligibility gates {summary['blocked_class_decisions']:,}.",
        f"- Narrow Ham clean-Chinese singleton pool: **{summary['ham_conditional_pool_pending_approval']:,} proposed candidates**, not eligible. Proposed batch eligible now: **0**.",
        f"- Ham outside this pool: {plan['source_counts']['ham'] - summary['ham_conditional_pool_pending_approval']:,} retained for quality/duplicate review; Spam: all {plan['source_counts']['spam']:,} remain source-level Review.",
        "- Per-record disposition lists are local under `data/v21/curated/trec06c/`; no raw email text is included here.",
        f"- Explicit duplicate-group blocker: Ham {plan['ham_disposition'].get('review|duplicate_group_needs_joint_review', 0):,}; Spam {plan['spam_disposition'].get('duplicate_group_needs_joint_review', 0):,}. These are mutually exclusive disposition counts after earlier quality gates, not the total number of duplicate rows.",
        "- Prior full-corpus dedup audit found 5,008 exact, 5,005 normalized, and 3,274 template duplicate groups; zero within-TREC raw-label conflicts. Group counts overlap and are not additive (see `trec06c_curation_report.md`).",
        "",
        "## Observed human discordance by review stratum",
        "",
        "Ham discordance means a conclusive human class other than Normal; Spam discordance means a conclusive human class other than commercial Spam. `review`/`exclude` decisions are counted separately as unresolved/excluded, not silently treated as correct. A Wilson 95% interval is shown only when an entire Ham probability-sample stratum has conclusive human classes. It does not prove whole-corpus purity. Partial review and the targeted Spam queue do not support a population-wide rate or interval.",
        "",
        "| Cohort | Stratum | Frame | Sample | Human reviewed | Conclusive | Discordant | Observed rate | Wilson 95% | Complete |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in by_stratum:
        rate = "N/A" if item["observed_discordance_rate"] is None else f"{item['observed_discordance_rate']:.3%}"
        interval = "N/A" if item["wilson_95"] is None else f"{item['wilson_95'][0]:.3%}–{item['wilson_95'][1]:.3%}"
        lines.append(
            f"| {item['cohort']} | {item['stratum']} | {item['population_rows']:,} | {item['sample_rows']:,} | "
            f"{item['human_reviewed']:,} | {item['conclusive_class_decisions']:,} | "
            f"{item['discordant_class_decisions']:,} | {rate} | {interval} | "
            f"{'yes' if item['sampling_complete'] else 'no'} |"
        )
    lines += [
        "",
        "## Review and approval boundary",
        "",
        "The Ham sample is deterministic, length-stratified random-by-hash within the narrow singleton frame (short/medium/long). The separate 221-row exception queue and Spam focus queue are deliberately enriched and cannot estimate corpus prevalence. Spam focus uses multiple content cues only to prioritize review; it never assigns a class. A Chinese-reading human must inspect complete mail and record evidence. Duplicate/quality-blocked class decisions remain `review` even when a human opinion is recorded.",
        "",
        "The inactive proposal is `config/v21/trec06c_conditional_approval_proposal.json`. Activation requires completion of the frozen Ham sample, adjudication of anomalies, an explicit user-chosen tolerance for observed discordance/uncertainty, a later user approval, and fresh group/V2-fingerprint checks. No Spam or Phishing bulk rule is proposed. TREC06C Phishing eligible remains limited to individually reviewed, gate-passing messages; no Chinese phishing coverage claim follows from this corpus.",
        "",
        "Local PowerShell review commands (from repository root):",
        "",
        "```powershell",
        '.venv\\Scripts\\python.exe -m scripts.v21.curate_trec06c review --queue ham-probability --reviewer "your-id" --limit 20',
        '.venv\\Scripts\\python.exe -m scripts.v21.curate_trec06c review --queue stratified --raw-label ham --reviewer "your-id" --limit 20',
        '.venv\\Scripts\\python.exe -m scripts.v21.curate_trec06c review --queue spam-focus --reviewer "your-id" --limit 20',
        ".venv\\Scripts\\python.exe -m scripts.v21.curate_trec06c summarize",
        "```",
        "",
        "`K` skips without a decision; `R` records unresolved Review. The CLI displays full email only in the local terminal and never contacts Gemini or another API. Local review queue/decision CSVs are Git-ignored and must not be published. Reviewer fields are self-attested, not cryptographic identity verification.",
        "",
        "No V2 model/test/metric, V2.1 candidate, Validation, or Final Test was used for label selection or changed. No model was trained and Prompt 07/08 was not run.",
    ]
    (paths.reports / "trec06c_human_review_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary
