"""Synthetic PhishVN ZIP fixtures; no network or private corpus required."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

import vietmailguard.v21.phishvn_urls as phishvn
from vietmailguard.v21.phishvn_urls import (
    DATA_MEMBER, audit_archives, derive_url_features,
    process_development_features, verify_manifest,
)


def _csv(fields: list[str], rows: list[list[str]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle)
    writer.writerow(fields)
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _zip(path: Path, payloads: dict[str, bytes], *, prefix: str = "") -> None:
    manifest = [f"{hashlib.sha256(data).hexdigest()}  {name.removeprefix(prefix)}  {len(data)}"
                for name, data in payloads.items()]
    with ZipFile(path, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr(prefix + "MANIFEST.txt", "\n".join(manifest).encode("utf-8"))


def _fixtures(tmp_path: Path, *, leak: bool = False) -> tuple[Path, Path, Path]:
    fields = ["id", "url", "url_norm", "label", "source", "tier", "split", "channel", "lang", "label_source"]
    test_url = "https://test.example.net/path" if not leak else "https://safe.example.org/other"
    rows = [
        ["a", "https://safe.example.org/home", "https://safe.example.org/home", "benign", "tranco", "gold", "train", "url", "vi", "feed"],
        ["b", "http://127.0.0.1/login", "http://127.0.0.1/login", "phishing", "tinnhiemmang", "silver", "val", "url", "vi", "feed"],
        ["c", test_url, test_url, "phishing", "chongluadao", "bronze", "test", "url", "mixed", "community"],
    ]
    payloads = {DATA_MEMBER: _csv(fields, rows)}
    for split in ("train", "val", "test"):
        payloads[f"data/splits/url_{split}.csv"] = _csv(fields, [row for row in rows if row[6] == split])
    payloads["data/vn_compphish.csv"] = _csv(["url", "label", "split"], [[row[2], "1" if row[3] == "phishing" else "0", row[6]] for row in rows])
    payloads["data/abuse_type.csv"] = _csv(["url", "abuse_type", "rule"], [[rows[1][1], "unknown", ""], [rows[2][1], "unknown", ""]])
    open_zip = tmp_path / "PhishVN_v3.1.0_open.zip"
    _zip(open_zip, payloads)
    prefix = "p1a_results_bundle/"
    bundle = tmp_path / "PhishVN_v3.0.0_results_bundle.zip"
    _zip(bundle, {prefix + "per_seed_metrics.csv": _csv(["model", "seed", "n_test"], [["LogReg", "0", "1"]]),
                  prefix + "models/LogReg_seed0.joblib": b"not-loaded"}, prefix=prefix)
    rules_path = tmp_path / "risk_rules.json"
    rules_path.write_text(json.dumps({"url_rules": {
        "long_url_length": 120, "excessive_subdomain_depth": 3,
        "suspicious_tokens": ["login", "verify"],
        "shortener_domains": [], "domain_patterns": [],
    }}), encoding="utf-8")
    return open_zip, bundle, rules_path


def test_phishvn_audit_manifest_provenance_and_splits(tmp_path: Path) -> None:
    open_zip, bundle, rules = _fixtures(tmp_path)
    result = audit_archives(open_zip, bundle, tmp_path / "cache.sqlite3", rules)
    assert result["core_rows"] == 3
    assert result["modality"] == "url_not_email"
    assert result["issues"] == {}
    assert result["official_split_file_rows"] == {"train": 1, "val": 1, "test": 1}
    assert result["benchmark"]["joblib_files_present_not_loaded"] == 1


def test_phishvn_features_reuse_offline_analyzer_and_exclude_test(tmp_path: Path) -> None:
    open_zip, _, rules_path = _fixtures(tmp_path)
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.json"
    state = process_development_features(open_zip, output, checkpoint, rules_path, chunk_size=2)
    assert state["complete"] is True
    assert state["rows"] == {"train": 1, "val": 1}
    assert not (output / "test").exists()
    with (output / "val" / "part-000001.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["candidate_label"] == "url_abuse_candidate"
    assert row["email_training_eligible"] == "false"
    assert row["is_ip_host"] == "true"
    assert "ip_host" in json.loads(row["finding_codes"])
    assert row["scheme"] == "http"
    assert process_development_features(open_zip, output, checkpoint, rules_path, chunk_size=2) == state


def test_phishvn_manifest_tamper_rejected(tmp_path: Path) -> None:
    open_zip, _, _ = _fixtures(tmp_path)
    with ZipFile(open_zip) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents[DATA_MEMBER] = b"tampered"
    tampered = tmp_path / "tampered.zip"
    with ZipFile(tampered, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with ZipFile(tampered) as archive, pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest(archive)


def test_phishvn_registered_domain_leakage_is_reported(tmp_path: Path) -> None:
    open_zip, bundle, rules = _fixtures(tmp_path, leak=True)
    result = audit_archives(open_zip, bundle, tmp_path / "cache.sqlite3", rules)
    assert result["issues"]["registered_domain_cross_split_rows"] == 1


def test_phishvn_benign_url_not_declared_malicious() -> None:
    rules = {"long_url_length": 120, "excessive_subdomain_depth": 3,
             "suspicious_tokens": ["login"], "shortener_domains": [], "domain_patterns": []}
    row = derive_url_features({"id": "safe", "url": "https://example.org/home", "label": "benign",
                               "tier": "gold", "split": "train"}, rules)
    assert row["candidate_label"] == "benign_url_candidate"
    assert json.loads(row["finding_codes"]) == []
    assert row["email_training_eligible"] == "false"


def test_phishvn_checkpoint_resumes_without_duplicate_parts(tmp_path: Path, monkeypatch) -> None:
    open_zip, _, rules = _fixtures(tmp_path)
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.json"
    original = phishvn._commit_batch
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        original(*args)

    monkeypatch.setattr(phishvn, "_commit_batch", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        process_development_features(open_zip, output, checkpoint, rules, chunk_size=2)
    monkeypatch.setattr(phishvn, "_commit_batch", original)
    result = process_development_features(open_zip, output, checkpoint, rules, chunk_size=2)
    assert result["rows"] == {"train": 1, "val": 1}
    assert result["parts"] == {"train": 1, "val": 1}
    assert result["last_ordinal"] == 3
