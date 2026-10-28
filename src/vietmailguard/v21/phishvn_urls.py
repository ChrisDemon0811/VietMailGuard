"""Offline PhishVN URL audit and train/val feature extraction for V2.1.

This is not an email adapter. Official URL test rows are audited for aggregate
split integrity but are never exported as development features.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

from vietmailguard.url_analyzer import analyze_url


VERSION = 1
DATA_MEMBER = "data/dataset_url.csv"
SPLITS = ("train", "val", "test")
REQUIRED = {"id", "url", "url_norm", "label", "source", "tier", "split", "channel", "lang"}
FEATURE_FIELDS = (
    "id", "dataset", "dataset_version", "archive_member", "source_row",
    "url", "raw_label", "candidate_label", "url_experiment_status",
    "email_training_eligible", "split", "source", "tier", "channel",
    "lang", "label_source", "collected_at", "domain", "registered_domain",
    "scheme", "hostname", "is_ip_host", "is_https_observed",
    "url_length", "subdomain_depth", "has_punycode", "is_shortener",
    "is_obfuscated", "suspicious_tokens", "finding_codes", "letter_count",
    "digit_count", "special_count", "path_length", "query_length",
    "parse_warning",
)


def _reader(archive: ZipFile, member: str):
    with archive.open(member) as handle:
        yield from csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline=""))


def _header(archive: ZipFile, member: str) -> list[str]:
    with archive.open(member) as handle:
        return next(csv.reader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")), [])


def verify_manifest(archive: ZipFile, prefix: str = "") -> dict:
    """Hash every listed member in a stream; never extract or load joblib."""
    member = prefix + "MANIFEST.txt"
    lines = archive.read(member).decode("utf-8-sig").splitlines()
    checked = 0
    for line in lines:
        match = re.match(r"^([0-9a-f]{64})\s+(.+?)\s+(?:\(\d+ bytes\)|\d+)$", line)
        if not match:
            continue
        expected, relative = match.groups()
        target = prefix + relative
        if target not in archive.namelist():
            raise ValueError(f"Manifest member missing: {relative}")
        digest = hashlib.sha256()
        with archive.open(target) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Manifest SHA-256 mismatch: {relative}")
        checked += 1
    if not checked:
        raise ValueError("Manifest contains no verifiable members")
    return {"verified_members": checked, "sha256_mismatches": 0}


def _rules(config_path: Path) -> tuple[dict, str]:
    raw = config_path.read_bytes()
    return json.loads(raw)["url_rules"], hashlib.sha256(raw).hexdigest()


def derive_url_features(raw: dict[str, str], rules: dict) -> dict[str, str]:
    """Reuse frozen V2 offline URL analyzer without altering its behavior."""
    url = raw.get("url", "") or ""
    analysis = analyze_url(url, rules)
    try:
        parsed = urlsplit(analysis.analysis_url if analysis.scheme else f"http://{analysis.analysis_url}")
        path_length = len(parsed.path)
        query_length = len(parsed.query)
    except ValueError:
        path_length = query_length = 0
    text = analysis.analysis_url
    label = raw.get("label", "")
    tier = raw.get("tier", "")
    result = {key: "" for key in FEATURE_FIELDS}
    result.update({
        "id": raw.get("id", ""), "dataset": "phishvn_url",
        "dataset_version": "3.1.0", "archive_member": DATA_MEMBER,
        "url": url, "raw_label": label,
        "candidate_label": {"phishing": "url_abuse_candidate", "benign": "benign_url_candidate"}.get(label, "review"),
        "url_experiment_status": "core_candidate" if tier in {"gold", "silver"} else "review_bronze",
        "email_training_eligible": "false", "split": raw.get("split", ""),
        "source": raw.get("source", ""), "tier": tier,
        "channel": raw.get("channel", ""), "lang": raw.get("lang", ""),
        "label_source": raw.get("label_source", ""),
        "collected_at": raw.get("collected_at", ""),
        "domain": raw.get("domain", ""),
        "registered_domain": analysis.registered_domain,
        "scheme": analysis.scheme, "hostname": analysis.hostname,
        "is_ip_host": str(analysis.is_ip_host).lower(),
        "is_https_observed": str(analysis.is_https).lower(),
        "url_length": str(analysis.length),
        "subdomain_depth": str(analysis.subdomain_depth),
        "has_punycode": str(analysis.has_punycode).lower(),
        "is_shortener": str(analysis.is_shortener).lower(),
        "is_obfuscated": str(analysis.is_obfuscated).lower(),
        "suspicious_tokens": json.dumps(analysis.suspicious_tokens),
        "finding_codes": json.dumps([finding.code for finding in analysis.findings]),
        "letter_count": str(sum(char.isalpha() for char in text)),
        "digit_count": str(sum(char.isdigit() for char in text)),
        "special_count": str(sum(not char.isalnum() for char in text)),
        "path_length": str(path_length),
        "query_length": str(query_length),
        "parse_warning": str(not bool(analysis.hostname)).lower(),
    })
    return result


def audit_archives(open_zip: Path, results_zip: Path, db_path: Path, config_path: Path) -> dict:
    """Inspect provenance, labels, manifests and split integrity; no scoring."""
    rules, rule_sha = _rules(config_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(open_zip) as data, ZipFile(results_zip) as bundle:
        manifest = verify_manifest(data)
        bundle_manifest = verify_manifest(bundle, "p1a_results_bundle/")
        header = _header(data, DATA_MEMBER)
        if not REQUIRED.issubset(header):
            raise ValueError(f"PhishVN core schema missing {sorted(REQUIRED - set(header))}")
        distributions = {key: Counter() for key in ("label", "split", "source", "tier", "channel", "lang", "label_source")}
        missing: Counter[str] = Counter()
        issues: Counter[str] = Counter()
        with sqlite3.connect(db_path) as connection:
            connection.execute("DROP TABLE IF EXISTS url_records")
            connection.execute("DROP TABLE IF EXISTS url_groups")
            connection.execute("CREATE TABLE url_records (id TEXT PRIMARY KEY, split TEXT, label TEXT, url_hash TEXT, norm_hash TEXT)")
            connection.execute("CREATE TABLE url_groups (registered_domain TEXT PRIMARY KEY, split TEXT)")
            total = 0
            for row in _reader(data, DATA_MEMBER):
                total += 1
                for key in distributions:
                    distributions[key][row.get(key, "")] += 1
                for key in REQUIRED:
                    if not (row.get(key) or "").strip():
                        missing[key] += 1
                if row.get("split") not in SPLITS:
                    issues["invalid_split"] += 1
                if row.get("label") not in {"phishing", "benign"}:
                    issues["unknown_label"] += 1
                url = row.get("url", "")
                if not url.strip():
                    issues["empty_url"] += 1
                record_id = row.get("id", "")
                url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
                norm_hash = hashlib.sha256((row.get("url_norm", "") or "").encode("utf-8")).hexdigest()
                try:
                    connection.execute("INSERT INTO url_records VALUES (?, ?, ?, ?, ?)",
                                       (record_id, row.get("split", ""), row.get("label", ""), url_hash, norm_hash))
                except sqlite3.IntegrityError:
                    issues["duplicate_id"] += 1
                if url:
                    group = analyze_url(url, rules).registered_domain
                    if group:
                        prior = connection.execute(
                            "SELECT split FROM url_groups WHERE registered_domain = ?", (group,)
                        ).fetchone()
                        if prior and prior[0] != row.get("split", ""):
                            issues["registered_domain_cross_split_rows"] += 1
                        elif not prior:
                            connection.execute("INSERT INTO url_groups VALUES (?, ?)", (group, row.get("split", "")))
            connection.execute("CREATE INDEX IF NOT EXISTS idx_phishvn_url_hash ON url_records(url_hash)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_phishvn_norm_hash ON url_records(norm_hash)")
            unique_urls = connection.execute("SELECT COUNT(DISTINCT url_hash) FROM url_records").fetchone()[0]
            conflicting_url_labels = connection.execute(
                "SELECT COUNT(*) FROM (SELECT url_hash FROM url_records GROUP BY url_hash HAVING COUNT(DISTINCT label) > 1)"
            ).fetchone()[0]
            split_files = {}
            for split in SPLITS:
                member = f"data/splits/url_{split}.csv"
                if _header(data, member) != header:
                    issues["split_header_mismatch"] += 1
                count = 0
                for row in _reader(data, member):
                    count += 1
                    original = connection.execute("SELECT split, label, url_hash FROM url_records WHERE id = ?", (row.get("id", ""),)).fetchone()
                    if original is None:
                        issues["split_id_missing_from_core"] += 1
                    elif (original[0] != split or original[1] != row.get("label") or
                          original[2] != hashlib.sha256((row.get("url", "") or "").encode("utf-8")).hexdigest()):
                        issues["split_row_mismatch"] += 1
                split_files[split] = count
                if count != distributions["split"][split]:
                    issues["split_count_mismatch"] += 1
            groups = connection.execute("SELECT COUNT(*) FROM url_groups").fetchone()[0]
            companion_rows = abuse_rows = 0
            companion_labels: Counter[str] = Counter()
            abuse_types: Counter[str] = Counter()
            for row in _reader(data, "data/vn_compphish.csv"):
                companion_rows += 1
                companion_labels[row.get("label", "")] += 1
                match = connection.execute(
                    "SELECT label, split FROM url_records WHERE norm_hash = ? LIMIT 1",
                    (hashlib.sha256(row.get("url", "").encode("utf-8")).hexdigest(),),
                ).fetchone()
                if not match:
                    issues["companion_url_absent_from_core"] += 1
                elif (row.get("label") != ("1" if match[0] == "phishing" else "0") or
                      row.get("split") != match[1]):
                    issues["companion_label_or_split_mismatch"] += 1
            for row in _reader(data, "data/abuse_type.csv"):
                abuse_rows += 1
                abuse_types[row.get("abuse_type", "")] += 1
                match = connection.execute(
                    "SELECT label FROM url_records WHERE url_hash = ? LIMIT 1",
                    (hashlib.sha256(row.get("url", "").encode("utf-8")).hexdigest(),),
                ).fetchone()
                if not match or match[0] != "phishing":
                    issues["abuse_type_not_linked_to_positive_core"] += 1
        benchmark_prefix = "p1a_results_bundle/"
        benchmark_rows = list(_reader(bundle, benchmark_prefix + "per_seed_metrics.csv"))
        benchmark_models = Counter(row.get("model", "") for row in benchmark_rows)
        bundle_members = bundle.namelist()
        return {
            "version": "3.1.0", "modality": "url_not_email", "core_rows": total,
            "core_header": header, "distributions": {key: dict(value) for key, value in distributions.items()},
            "missing_required": dict(missing), "issues": dict(issues),
            "registered_domain_groups": groups, "official_split_file_rows": split_files,
            "unique_exact_urls": unique_urls, "exact_duplicate_url_rows": total - unique_urls,
            "exact_url_conflicting_label_groups": conflicting_url_labels,
            "companion_representation_rows": companion_rows,
            "companion_binary_labels": dict(companion_labels),
            "abuse_type_rows": abuse_rows, "abuse_types": dict(abuse_types),
            "open_manifest": manifest, "benchmark_manifest": bundle_manifest,
            "benchmark": {
                "version": "3.0.0", "metrics_rows": len(benchmark_rows),
                "models": dict(benchmark_models),
                "reported_n_test_values": sorted({row.get("n_test", "") for row in benchmark_rows}),
                "prediction_files": sum(member.startswith(benchmark_prefix + "predictions/") and member.endswith(".csv") for member in bundle_members),
                "joblib_files_present_not_loaded": sum(member.endswith(".joblib") for member in bundle_members),
                "results_are_external_reference_only": True,
            },
            "v2_url_rules_sha256": rule_sha,
            "test_policy": "aggregate_integrity_audit_only; no test feature export, scoring or tuning",
        }


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _write_part(rows: list[dict[str, str]], part: Path) -> None:
    part.parent.mkdir(parents=True, exist_ok=True)
    temp = part.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, part)


def process_development_features(
    open_zip: Path, output_dir: Path, checkpoint_path: Path, config_path: Path,
    *, chunk_size: int = 5000,
) -> dict:
    """Export offline URL features for official train/val only, resumably."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    rules, rule_sha = _rules(config_path)
    stat = open_zip.stat()
    identity = {"archive_size": stat.st_size, "archive_mtime_ns": stat.st_mtime_ns,
                "rule_sha256": rule_sha, "chunk_size": chunk_size, "parser_version": VERSION}
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in identity.items()):
            raise ValueError("Source ZIP, URL rules or chunk size changed since checkpoint")
        if state["complete"]:
            for split in ("train", "val"):
                if len(list((output_dir / split).glob("part-*.csv"))) != state["parts"][split]:
                    raise ValueError("Feature checkpoint/parts mismatch")
            return state
    else:
        if any(any((output_dir / split).glob("part-*.csv")) for split in ("train", "val")):
            raise ValueError("Feature parts exist without a checkpoint")
        state = {**identity, "last_ordinal": 0, "parts": {"train": 0, "val": 0},
                 "rows": {"train": 0, "val": 0}, "parse_warning_rows": 0,
                 "complete": False}
    with ZipFile(open_zip) as archive:
        batch = []
        for ordinal, raw in enumerate(_reader(archive, DATA_MEMBER), 1):
            if ordinal <= state["last_ordinal"]:
                continue
            batch.append((ordinal, raw))
            if len(batch) == chunk_size:
                _commit_batch(batch, rules, output_dir, checkpoint_path, state)
                batch.clear()
        if batch:
            _commit_batch(batch, rules, output_dir, checkpoint_path, state)
    state["complete"] = True
    _save_json(checkpoint_path, state)
    return state


def _commit_batch(batch, rules, output_dir, checkpoint_path, state) -> None:
    by_split = {"train": [], "val": []}
    warnings = 0
    for ordinal, raw in batch:
        split = raw.get("split", "")
        if split not in by_split:
            continue  # The official test is intentionally not exported.
        feature = derive_url_features(raw, rules)
        feature["source_row"] = str(ordinal)
        by_split[split].append(feature)
        warnings += feature["parse_warning"] == "true"
    for split, rows in by_split.items():
        if rows:
            number = state["parts"][split] + 1
            _write_part(rows, output_dir / split / f"part-{number:06d}.csv")
            state["parts"][split] = number
            state["rows"][split] += len(rows)
    state["parse_warning_rows"] += warnings
    state["last_ordinal"] = batch[-1][0]
    _save_json(checkpoint_path, state)
