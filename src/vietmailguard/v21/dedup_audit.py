"""Disk-backed V2.1 email audit. No source row is deleted or relabeled in place.

Hash groups are evidence for review, not a training merge or a fuzzy matcher.
The V2 bilingual master is a read-only provenance reference, never an output.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .paths import V21Paths, resolve_v21_paths


AUDIT_VERSION = 1
DATASETS = (
    ("merged_dataset", "csv_sources/merged_dataset"),
    ("balanced_dataset", "csv_sources/balanced_dataset"),
    ("meajor", "csv_sources/meajor"),
    ("trec06c", "trec06c_mime_v2"),
    ("enron_original", "enron_original_v1"),
)
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")
SPACE_RE = re.compile(r"\s+")
MISSING = {"", "empty", "<empty>", "nan", "none", "null", "n/a"}
FIELDS = (
    "audit_id", "dataset", "source", "source_path", "original_id", "language",
    "data_origin", "raw_label", "candidate_label", "label", "label_provenance",
    "label_status", "training_eligible", "audit_reason", "parent_id", "v2_group_id",
    "exact_hash", "normalized_hash", "template_hash", "exact_group_id",
    "normalized_group_id", "template_group_id", "provenance_group_id",
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean(value: str | None) -> str:
    value = unicodedata.normalize("NFC", value or "")
    return "" if value.strip().casefold() in MISSING else value


def dedup_hashes(subject: str | None, body: str | None) -> tuple[str, str, str]:
    """Strict, normalized and conservative URL/email-template fingerprints.

    Template normalization never modifies stored source text. Short/generic
    messages are deliberately not assigned a template hash.
    """
    subject, body = _clean(subject), _clean(body)
    if not subject.strip() and not body.strip():
        return "", "", ""
    original = subject + "\x1f" + body
    normalized = SPACE_RE.sub(" ", unicodedata.normalize("NFKC", original).casefold()).strip()
    template, url_count = URL_RE.subn(" <url> ", normalized)
    template, email_count = EMAIL_RE.subn(" <email> ", template)
    template = SPACE_RE.sub(" ", template).strip()
    non_placeholders = template.replace("<url>", "").replace("<email>", "").strip()
    cjk_chars = sum('\u3400' <= char <= '\u9fff' for char in non_placeholders)
    enough_context = len(non_placeholders) >= 80 and (len(non_placeholders.split()) >= 10 or cjk_chars >= 30)
    return (
        _digest(original),
        _digest(normalized),
        _digest(template) if (url_count or email_count) and enough_context else "",
    )


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rows (
            audit_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, source TEXT,
            source_path TEXT, original_id TEXT, language TEXT, data_origin TEXT,
            raw_label TEXT, candidate_label TEXT, label TEXT, label_provenance TEXT,
            label_status TEXT NOT NULL, training_eligible INTEGER NOT NULL,
            audit_reason TEXT, parent_id TEXT, v2_group_id TEXT,
            exact_hash TEXT, normalized_hash TEXT, template_hash TEXT,
            exact_group_id TEXT, normalized_group_id TEXT,
            template_group_id TEXT, provenance_group_id TEXT
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            path TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL, row_count INTEGER NOT NULL,
            audit_version INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_rows_exact ON rows(exact_hash);
        CREATE INDEX IF NOT EXISTS idx_rows_normal ON rows(normalized_hash);
        CREATE INDEX IF NOT EXISTS idx_rows_template ON rows(template_hash);
        CREATE INDEX IF NOT EXISTS idx_rows_parent ON rows(parent_id);
        CREATE INDEX IF NOT EXISTS idx_rows_dataset ON rows(dataset);
    """)
    return conn


def _parts(paths: V21Paths, *, include_v2_reference: bool) -> list[tuple[str, Path]]:
    inputs: list[tuple[str, Path]] = []
    for dataset, relative in DATASETS:
        directory = paths.processed / relative
        parts = sorted(directory.glob("part-*.csv"))
        if not parts:
            raise FileNotFoundError(f"Missing standardized {dataset} parts: {directory}")
        inputs.extend((dataset, part) for part in parts)
    if include_v2_reference:
        reference = paths.project_root / "data/processed/Bilingual_Master_Dataset.csv"
        if not reference.is_file():
            raise FileNotFoundError(f"Missing read-only V2 reference: {reference}")
        inputs.append(("v2_reference", reference))
    return inputs


def _row(dataset: str, raw: dict[str, str], path: Path) -> tuple[str, ...]:
    original_id = raw.get("id", "")
    if not original_id:
        raise ValueError(f"Missing stable row id in {path}")
    audit_id = f"{dataset}|{original_id}"
    subject, body = raw.get("subject", ""), raw.get("body", "")
    exact, normalized, template = dedup_hashes(subject, body)
    source = raw.get("source", "") or dataset
    raw_label = raw.get("raw_label", "") or ""
    candidate = raw.get("candidate_label", "") or ""
    label = raw.get("label", "") or "review"
    provenance = raw.get("label_provenance", "") or ""
    parent_id = raw.get("parent_id", "") or ""
    v2_group = raw.get("final_group_id", "") or raw.get("translation_group_id", "") or ""
    language = raw.get("language", "") or raw.get("language_candidate", "") or "unknown"
    origin = raw.get("data_origin", "") or ("original" if dataset in {"enron_original", "trec06c"} else "unknown")
    source_path = (raw.get("original_path", "") or raw.get("source_path", "")
                   or raw.get("source_file", "") or raw.get("input_artifact", "") or str(path))
    enron_row = raw.get("v2_enron_csv_row", "") or ""
    provenance_group = (
        f"v2_enron_row_{enron_row}" if dataset == "enron_original" and enron_row
        else f"v2_enron_row_{raw.get('original_row_id', '')}"
        if dataset == "v2_reference" and source == "Enron" and raw.get("original_row_id", "")
        else f"v2_{v2_group}" if v2_group else ""
    )
    reason = ""
    if dataset == "v2_reference":
        eligible = label in {"normal", "spam", "phishing"} and raw.get("training_eligible", "").casefold() == "true"
        status = "eligible" if eligible else "review"
        candidate = label if eligible else ""
        reason = "frozen_v2_reference_only" if eligible else "v2_reference_not_eligible"
    else:
        eligible = False
        status = "review"
        reason = "unverified_v21_label"
        if dataset in {"merged_dataset", "balanced_dataset"}:
            reason = "related_corpora_unverified_binary_label"
        if dataset == "meajor":
            reason = "source_specific_label_semantics_unverified"
        if dataset == "enron_original":
            reason = "original_enron_has_no_verified_normal_label"
        if dataset == "trec06c":
            reason = "trec_ham_spam_candidate_not_vmg_three_class_label"
        if not normalized or raw.get("empty", "").casefold() == "true":
            status, reason = "exclude", "empty_content"
        elif raw.get("status", "ok") not in {"", "ok"}:
            status, reason = "review", "parse_or_source_warning"
    return (
        audit_id, dataset, source, source_path, original_id, language, origin,
        raw_label, candidate, label, provenance, status, int(eligible), reason,
        parent_id, v2_group, exact, normalized, template,
        "", "", "", provenance_group,
    )


def _ingest(conn: sqlite3.Connection, dataset: str, path: Path) -> tuple[int, bool]:
    resolved = str(path.resolve())
    stat = path.stat()
    old = conn.execute("SELECT size_bytes, mtime_ns, row_count, audit_version FROM checkpoints WHERE path=?", (resolved,)).fetchone()
    if old:
        if old[:2] != (stat.st_size, stat.st_mtime_ns) or old[3] != AUDIT_VERSION:
            raise RuntimeError(f"Input changed since checkpoint: {path}")
        return old[2], True
    csv.field_size_limit(100_000_000)
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle, conn:
        reader = csv.DictReader(handle)
        required = {"id", "subject", "body", "raw_label", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid standardized schema: {path}")
        batch: list[tuple[str, ...]] = []
        for raw in reader:
            batch.append(_row(dataset, raw, path))
            count += 1
            if len(batch) >= 1000:
                conn.executemany(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})", batch)
                batch.clear()
        if batch:
            conn.executemany(f"INSERT INTO rows VALUES ({','.join('?' for _ in FIELDS)})", batch)
        conn.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)", (resolved, stat.st_size, stat.st_mtime_ns, count, AUDIT_VERSION))
    return count, False


def _finalize_groups(conn: sqlite3.Connection) -> None:
    # A template link is a review lead; it does not erase source identity.
    # This audit never grants V2.1 training admission to frozen V2 reference
    # rows (which include the old held-out split). 'eligible' below denotes
    # only that an already-verified source label survived the audit.
    conn.execute("UPDATE rows SET training_eligible=0 WHERE dataset='v2_reference'")
    conn.execute("""UPDATE rows SET data_origin='unknown'
        WHERE dataset IN ('merged_dataset','balanced_dataset','meajor')
        AND data_origin='original'""")
    conn.execute("""UPDATE rows SET audit_reason='candidate_mapping_inconsistent',
        label_status='review', training_eligible=0
        WHERE label_status!='exclude' AND (
            (dataset='trec06c' AND ((raw_label='ham' AND candidate_label!='ham')
                OR (raw_label='spam' AND candidate_label!='spam')
                OR raw_label NOT IN ('ham','spam')))
            OR (dataset IN ('merged_dataset','balanced_dataset','meajor')
                AND ((raw_label IN ('0','0.0') AND candidate_label!='normal_candidate')
                  OR (raw_label IN ('1','1.0') AND candidate_label!='spam_or_scam_or_phishing_candidate')
                  OR (raw_label NOT IN ('','0','0.0','1','1.0') AND candidate_label!='unknown')))
            OR (dataset='enron_original' AND raw_label!='')
        )""")
    for column, group_column, prefix in (
        ("exact_hash", "exact_group_id", "exact"),
        ("normalized_hash", "normalized_group_id", "normalized"),
        ("template_hash", "template_group_id", "template"),
    ):
        conn.execute(f"""UPDATE rows SET {group_column}='{prefix}_' || substr({column},1,20)
            WHERE {column} IN (SELECT {column} FROM rows WHERE {column}!=''
            GROUP BY {column} HAVING count(*)>1)""")
    # Explicit Enron source-row pointers and frozen V2 final groups are kept
    # separately from text hashes. They never transfer a training label.
    conn.commit()


def _apply_conflicts(conn: sqlite3.Connection) -> int:
    conn.execute("DROP TABLE IF EXISTS temp.conflicts")
    conn.execute("""CREATE TEMP TABLE conflicts AS
        SELECT normalized_hash AS hash FROM rows WHERE normalized_hash!=''
        GROUP BY normalized_hash
        HAVING count(DISTINCT CASE WHEN candidate_label IN ('normal','normal_candidate','ham') THEN 'normal'
            WHEN candidate_label='spam' THEN 'spam' WHEN candidate_label='phishing' THEN 'phishing' END)>1
        OR (sum(CASE WHEN dataset IN ('merged_dataset','balanced_dataset','meajor')
            AND raw_label IN ('0','0.0') THEN 1 ELSE 0 END)>0
            AND sum(CASE WHEN dataset IN ('merged_dataset','balanced_dataset','meajor')
            AND raw_label IN ('1','1.0') THEN 1 ELSE 0 END)>0)""")
    conn.execute("CREATE INDEX temp.idx_conflict_hash ON conflicts(hash)")
    count = conn.execute("SELECT count(*) FROM rows WHERE normalized_hash IN (SELECT hash FROM conflicts)").fetchone()[0]
    conn.execute("""UPDATE rows SET label_status='review', training_eligible=0,
        audit_reason='duplicate_label_conflict' WHERE label_status!='exclude'
        AND normalized_hash IN (SELECT hash FROM conflicts)""")
    conn.commit()
    return count


def _write_csv(path: Path, columns: tuple[str, ...], rows: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _reports(conn: sqlite3.Connection, paths: V21Paths, input_counts: dict[str, int], resumed: int, conflict_rows: int) -> dict[str, object]:
    reports = paths.reports
    reports.mkdir(parents=True, exist_ok=True)
    distribution = conn.execute("""SELECT dataset, language, raw_label, candidate_label, label_status,
        count(*) FROM rows GROUP BY dataset, language, raw_label, candidate_label, label_status
        ORDER BY dataset, language, raw_label, candidate_label, label_status""")
    _write_csv(reports / "dedup_distribution.csv", ("dataset", "language", "raw_label", "candidate_label", "label_status", "rows"), distribution)
    _write_csv(reports / "audit_reason_distribution.csv", ("dataset", "audit_reason", "rows"),
        conn.execute("""SELECT dataset, audit_reason, count(*) FROM rows
            GROUP BY dataset, audit_reason ORDER BY dataset, audit_reason"""))
    duplicate_summary: dict[str, dict[str, int]] = {}
    for name, column in (("exact", "exact_hash"), ("normalized", "normalized_hash"), ("template", "template_hash")):
        groups, rows = conn.execute(f"""SELECT count(*), coalesce(sum(n),0) FROM
            (SELECT count(*) n FROM rows WHERE {column}!='' GROUP BY {column} HAVING n>1)""").fetchone()
        duplicate_summary[name] = {"groups": groups, "rows_in_groups": rows, "extra_rows": rows - groups}
    _write_csv(reports / "duplicate_groups.csv",
        ("kind", "group_id", "rows", "datasets", "candidate_labels"),
        (
            (kind, f"{kind}_{hash_value[:20]}", n, datasets, candidates)
            for kind, column in (("exact", "exact_hash"), ("normalized", "normalized_hash"), ("template", "template_hash"))
            for hash_value, n, datasets, candidates in conn.execute(f"""
                SELECT {column}, count(*), group_concat(DISTINCT dataset),
                group_concat(DISTINCT candidate_label) FROM rows
                WHERE {column}!='' GROUP BY {column} HAVING count(*)>1
                ORDER BY count(*) DESC""")
        ))
    _write_csv(reports / "conflicting_labels.csv",
        ("normalized_group_id", "audit_id", "dataset", "raw_label", "candidate_label", "label", "label_provenance"),
        conn.execute("""SELECT normalized_group_id, audit_id, dataset, raw_label,
            candidate_label, label, label_provenance FROM rows
            WHERE audit_reason='duplicate_label_conflict' ORDER BY normalized_group_id, audit_id"""))
    provenance_query = """SELECT group_key, count(*), group_concat(DISTINCT dataset)
        FROM (SELECT provenance_group_id AS group_key, dataset FROM rows WHERE provenance_group_id!=''
              UNION ALL SELECT 'v2_final_' || v2_group_id, dataset FROM rows
              WHERE dataset='v2_reference' AND v2_group_id!='')
        GROUP BY group_key HAVING count(*)>1 ORDER BY count(*) DESC"""
    _write_csv(reports / "provenance_groups.csv",
        ("provenance_group_id", "rows", "datasets"), conn.execute(provenance_query))
    cross: list[tuple[str, str, str, int]] = []
    for kind, column in (("exact", "exact_hash"), ("normalized", "normalized_hash"), ("template", "template_hash")):
        cross_counts: Counter[tuple[str, str]] = Counter()
        current_hash = ""
        members: list[tuple[str, int]] = []
        def add_pairs(group: list[tuple[str, int]]) -> None:
            for index, (left, left_count) in enumerate(group):
                for right, right_count in group[index + 1:]:
                    cross_counts[(left, right)] += left_count * right_count
        for hash_value, dataset, count in conn.execute(f"""SELECT {column}, dataset, count(*)
            FROM rows WHERE {column}!='' GROUP BY {column}, dataset
            ORDER BY {column}, dataset"""):
            if current_hash and hash_value != current_hash:
                add_pairs(members)
                members = []
            current_hash = hash_value
            members.append((dataset, count))
        add_pairs(members)
        cross.extend((kind, a, b, count) for (a, b), count in cross_counts.most_common())
    _write_csv(reports / "cross_source_overlap.csv", ("evidence_kind", "dataset_a", "dataset_b", "matching_pairs"), cross)
    source_cross: Counter[tuple[str, str, str, str, str]] = Counter()
    for kind, column in (("exact", "exact_hash"), ("normalized", "normalized_hash")):
        current_hash = ""
        members_by_source: list[tuple[str, str, int]] = []
        def add_source_pairs(group: list[tuple[str, str, int]]) -> None:
            for index, (left_dataset, left_source, left_count) in enumerate(group):
                for right_dataset, right_source, right_count in group[index + 1:]:
                    if (left_dataset, left_source) != (right_dataset, right_source):
                        source_cross[(kind, left_dataset, left_source, right_dataset, right_source)] += left_count * right_count
        for hash_value, dataset, source, count in conn.execute(f"""SELECT {column}, dataset, source, count(*)
            FROM rows WHERE {column}!='' GROUP BY {column}, dataset, source
            ORDER BY {column}, dataset, source"""):
            if current_hash and hash_value != current_hash:
                add_source_pairs(members_by_source)
                members_by_source = []
            current_hash = hash_value
            members_by_source.append((dataset, source, count))
        add_source_pairs(members_by_source)
    _write_csv(reports / "cross_corpus_source_overlap.csv",
        ("evidence_kind", "dataset_a", "source_a", "dataset_b", "source_b", "matching_pairs"),
        ((*key, count) for key, count in source_cross.most_common()))
    statuses = dict(conn.execute("SELECT label_status, count(*) FROM rows GROUP BY label_status"))
    datasets = dict(conn.execute("SELECT dataset, count(*) FROM rows GROUP BY dataset"))
    training_eligible_rows = conn.execute("SELECT count(*) FROM rows WHERE training_eligible=1").fetchone()[0]
    mapping_warning_rows = conn.execute("SELECT count(*) FROM rows WHERE audit_reason='candidate_mapping_inconsistent'").fetchone()[0]
    languages = dict(conn.execute("SELECT language, count(*) FROM rows GROUP BY language"))
    provenance_groups = conn.execute(f"SELECT count(*) FROM ({provenance_query})").fetchone()[0]
    total = sum(datasets.values())
    summary = {
        "audit_version": AUDIT_VERSION, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": total, "output_rows": total, "physically_removed_rows": 0,
        "v21_source_rows": total - datasets.get("v2_reference", 0),
        "read_only_v2_reference_rows": datasets.get("v2_reference", 0),
        "input_parts": len(input_counts), "resumed_parts": resumed,
        "rows_per_dataset": datasets, "label_status": statuses,
        "rows_per_language": languages, "v21_training_eligible_rows": training_eligible_rows,
        "candidate_mapping_warning_rows": mapping_warning_rows,
        "duplicate_groups": duplicate_summary, "conflict_rows": conflict_rows,
        "provenance_groups": provenance_groups,
        "cross_source_pairs": len(cross),
        "cross_corpus_source_pairs": len(source_cross),
        "scope": "V2.1 standardized email parts plus read-only V2 bilingual provenance reference",
        "excluded_input_representations": [
            "trec06c older parser output (same ZIP)", "English_Master_Dataset.csv (contained in Bilingual Master)",
            "Phishing_validation_emails.csv (challenge-only)", "PhishVN ZIPs (URL-only, not emails)",
        ],
    }
    (reports / "dedup_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V2.1 Dataset Audit & Deduplication", "",
        f"Completed: {summary['completed_at_utc']}", "",
        "No training merge, model training, or source deletion occurred. V2 artifacts are read-only.", "",
        f"Rows before audit: **{total:,}**; rows after audit: **{total:,}**; physically removed: **0**.", "",
        f"V2.1 standardized source rows: **{summary['v21_source_rows']:,}**; frozen V2 read-only reference rows: **{summary['read_only_v2_reference_rows']:,}**.", "",
        "## Input rows and audit status", "",
        "| Dataset | Rows |", "|---|---:|",
    ]
    lines += [f"| {name} | {count:,} |" for name, count in datasets.items()]
    lines += ["", "| Status | Rows |", "|---|---:|"]
    lines += [f"| {name} | {count:,} |" for name, count in statuses.items()]
    lines += ["", f"Rows admitted to V2.1 training by this audit: **{training_eligible_rows:,}** (expected zero before curation).",
              f"Rows with inconsistent source candidate mapping: **{mapping_warning_rows:,}** (kept for review).", "",
              "Language counts are in `dedup_audit_summary.json`; full dataset × language × raw/candidate/status counts are in `dedup_distribution.csv`."]
    lines += ["", "## Duplicate evidence", "", "| Kind | Groups | Rows in groups | Extra rows (not deleted) |", "|---|---:|---:|---:|"]
    lines += [f"| {kind} | {v['groups']:,} | {v['rows_in_groups']:,} | {v['extra_rows']:,} |" for kind,v in duplicate_summary.items()]
    lines += ["", "## Cross-corpus source evidence", "",
              "Top strict normalized-text matches (matching row pairs, not distinct emails):", "",
              "| Source A | Source B | Pairs |", "|---|---|---:|"]
    normalized_source_matches = [
        (left_dataset, left_source, right_dataset, right_source, count)
        for (kind, left_dataset, left_source, right_dataset, right_source), count in source_cross.most_common()
        if kind == "normalized"
    ]
    lines += [f"| {left_dataset}/{left_source} | {right_dataset}/{right_source} | {count:,} |"
              for left_dataset, left_source, right_dataset, right_source, count in normalized_source_matches[:12]]
    nazario = [
        (kind, left_dataset, left_source, right_dataset, right_source, count)
        for (kind, left_dataset, left_source, right_dataset, right_source), count in source_cross.items()
        if "Nazario" in (left_source, right_source)
    ]
    if nazario:
        lines += ["", "Nazario-specific cross-source matches (reference only, not label transfer):"]
        lines += [f"- {kind}: {left_dataset}/{left_source} ↔ {right_dataset}/{right_source}: {count:,} matching pairs"
                  for kind, left_dataset, left_source, right_dataset, right_source, count in sorted(nazario)]
    if not any("trec06c" in (left_dataset, right_dataset)
               for _, left_dataset, _, right_dataset, _ in normalized_source_matches):
        lines += ["", "No cross-dataset normalized-text match to TREC06C was observed; this does not prove the corpora are disjoint."]
    lines += [
        "", f"Rows in normalized-hash groups with contradictory candidate/binary raw labels: **{conflict_rows:,}**; audit status `review`.",
        f"Explicit provenance groups with more than one row: **{provenance_groups:,}**.",
        "", "`duplicate_groups.csv` and `cross_source_overlap.csv` are evidence, not instructions to delete or transfer labels. Cross-source counts are matching row pairs, not distinct emails; template links are less certain than exact/normalized links.",
        "Template hashes replace only URL/email values, require substantial remaining text, and are review leads. Similar wording alone creates no match.",
        "", "## Label and provenance policy", "",
        "- Merged/Balanced are related corpus representations, not independent datasets. Their binary positive labels remain unverified.",
        "- MeAJOR labels remain source-specific review candidates. TREC ham/spam remains a two-class candidate taxonomy, not VietMailGuard phishing ground truth.",
        "- Enron original has no automatic normal label. V2 Bilingual Master is a frozen read-only reference including Nazario and translated descendants; its existing eligibility is reported, not transferred.",
        "- No LLM predictions are used. `eligible` applies only to previously verified V2 reference labels without detected conflicts; reference rows have `training_eligible=0` for V2.1 because the frozen V2 held-out split is also present. New V2.1 rows remain `review` or `exclude` pending curation.",
        "- Exact and normalized hashes may identify corpus overlap; template groups require human review. No fuzzy similarity-based removal.",
        "- Candidate links across languages require explicit parent metadata or separately validated evidence; translated content alone is not an exact-text match.",
        "", "## Scope and limitations", "",
        "- V2 frozen test rows in the reference are present solely to detect overlap; they are not V2.1 training-eligible by this report.",
        "- Phishing validation challenge and PhishVN URL data are not email-audit inputs. The older TREC parser output and English Master are duplicate representations and are not counted twice.",
        "- Hash equality can indicate a shared template, not necessarily the same original message. Subject/body extraction differences may cause false negatives.",
        "- The local audit index contains hashes and provenance metadata, not full email bodies. Derived reports may contain source paths/identifiers and remain local.",
        "", "Detailed distributions: `dedup_distribution.csv`; duplicate groups: `duplicate_groups.csv`; conflicts: `conflicting_labels.csv`; explicit links: `provenance_groups.csv`; cross-dataset pairs: `cross_source_overlap.csv`; corpus/source detail including Nazario and MeAJOR subcorpora: `cross_corpus_source_overlap.csv`.",
        "Language values are source metadata, not independently verified language detection; `unknown` and MeAJOR multi-code values are retained rather than guessed.",
        "The resumable row-level audit index is `data/v21/cache/dedup_audit.sqlite3` (local only); statuses and hash groups can be queried without copying sensitive message bodies.",
    ]
    (reports / "dedup_audit_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def run_audit(paths: V21Paths | None = None, *, include_v2_reference: bool = True) -> dict[str, object]:
    """Process each immutable CSV part atomically; resume from SQLite checkpoints."""
    paths = paths or resolve_v21_paths()
    inputs = _parts(paths, include_v2_reference=include_v2_reference)
    conn = _connect(paths.cache / "dedup_audit.sqlite3")
    try:
        counts: dict[str, int] = {}
        resumed = 0
        for dataset, part in inputs:
            count, skipped = _ingest(conn, dataset, part)
            counts[str(part)] = count
            resumed += int(skipped)
            print(f"{dataset}: {part.name}: {count:,} rows{' (checkpoint)' if skipped else ''}", flush=True)
        _finalize_groups(conn)
        conflict_rows = _apply_conflicts(conn)
        summary = _reports(conn, paths, counts, resumed, conflict_rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        conn.close()
