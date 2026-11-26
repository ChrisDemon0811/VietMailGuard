"""Resumable, private V2.3 intake using existing parsing/deduplication code.

This intake does not approve labels, assign evaluation splits, or train models.
Full email text is compressed locally; console output is aggregate metadata only.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
import sqlite3
import subprocess
import unicodedata
import zlib
from pathlib import Path, PurePosixPath

from vietmailguard.eml_parser import parse_eml_bytes
from vietmailguard.v21.dedup_audit import dedup_hashes
from vietmailguard.v22.quality import language_signal


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path.resolve().as_uri(), uri=True)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA cache_size=-32768")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS cau_hinh(ma TEXT PRIMARY KEY, gia_tri TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS thu(
            id_thu INTEGER PRIMARY KEY, id_nguon TEXT UNIQUE NOT NULL,
            nguon TEXT NOT NULL, nhan_goc TEXT, nhan_ung_vien TEXT,
            trang_thai_nhan TEXT NOT NULL, ngon_ngu_goi_y TEXT,
            duong_dan_goc TEXT, ma_bam_chuan TEXT UNIQUE NOT NULL,
            ma_bam_noi_dung TEXT, ma_bam_mau TEXT, id_cha TEXT,
            van_ban_nen BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS ban_ghi(
            id_nguon TEXT PRIMARY KEY, nguon TEXT NOT NULL, id_thu INTEGER,
            ly_do TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tien_do(
            duong_dan TEXT PRIMARY KEY, ma_bam TEXT NOT NULL,
            so_dong INTEGER NOT NULL, hoan_tat INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS chi_muc_noi_dung ON thu(ma_bam_noi_dung);
        CREATE INDEX IF NOT EXISTS chi_muc_mau ON thu(ma_bam_mau);
        CREATE TABLE IF NOT EXISTS tham_chieu(
            id_goc TEXT PRIMARY KEY, id_thu INTEGER NOT NULL, nhan TEXT NOT NULL,
            ngon_ngu TEXT, nhom_goc TEXT, id_cha TEXT, split_goc TEXT,
            can_cu_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS chi_muc_tham_chieu_thu ON tham_chieu(id_thu);
        CREATE TABLE IF NOT EXISTS bam_bao_ve(
            ma_bam TEXT NOT NULL, loai TEXT NOT NULL,
            PRIMARY KEY(ma_bam,loai));
        CREATE TABLE IF NOT EXISTS cach_ly(
            id_thu INTEGER NOT NULL, ly_do TEXT NOT NULL,
            PRIMARY KEY(id_thu,ly_do));
    """)
    return connection


def add_email(connection: sqlite3.Connection, source: str, row: dict,
              policy: dict, *, minimum: int = 8) -> str:
    """Keep original normalized Unicode content; no keyword-based label mapping."""
    identifier = source + ":" + str(row["id"])
    if connection.execute("SELECT 1 FROM ban_ghi WHERE id_nguon=?", (identifier,)).fetchone():
        return "already_processed"
    subject = unicodedata.normalize("NFC", str(row.get("subject") or ""))
    body = unicodedata.normalize("NFC", str(row.get("body") or ""))
    reason = ""
    if sum(1 for _ in itertools.islice(re.finditer(r"[^\W_]", body), minimum)) < minimum:
        reason = "excluded_empty_or_insufficient_body"
    elif body.count("\ufffd") >= max(3, len(body) // 100):
        reason = "excluded_decoding_damage"
    if reason:
        connection.execute("INSERT INTO ban_ghi VALUES (?,?,NULL,?)", (identifier, source, reason))
        return reason
    _, normalized, template = dedup_hashes(subject, body)
    # Equivalent to dedup_hashes('', body)[1], without its second full template
    # analysis. Full text is still used; this is not a prefix approximation.
    normalized_body = hashlib.sha256(
        " ".join(unicodedata.normalize("NFKC", body).casefold().split()).encode("utf-8")
    ).hexdigest()
    existing = connection.execute("SELECT id_thu FROM thu WHERE ma_bam_chuan=?", (normalized,)).fetchone()
    if existing:
        mail_id, reason = existing[0], "duplicate_normalized_full_text"
    else:
        payload = {"subject": subject, "body": body, "sender": row.get("sender", ""),
                   "receiver": row.get("receiver", ""), "date": row.get("date", ""),
                   "source_id": row["id"], "source_path": row.get("source_path", ""),
                   "raw_content_sha256": row.get("raw_content_sha256", ""),
                   "data_origin": row.get("data_origin", "native"),
                   "language_original": row.get("language_candidate", ""),
                   "rights_evidence": policy["rights_evidence"],
                   "quality_flags": {k: row[k] for k in ("mime_defects", "decode_warning", "is_forwarded") if k in row}}
        compressed = zlib.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"), level=1)
        cursor = connection.execute("INSERT INTO thu VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)", (
            identifier, source, row.get("raw_label", policy.get("raw_label", "")),
            policy["candidate_label"], policy["label_status"], language_signal(subject + "\n" + body),
            str(row.get("source_path", "")), normalized, normalized_body, template,
            row.get("parent_id", ""), compressed))
        mail_id, reason = cursor.lastrowid, "unique_quality_candidate"
    connection.execute("INSERT INTO ban_ghi VALUES (?,?,?,?)", (identifier, source, mail_id, reason))
    return reason


def ingest_csv(connection: sqlite3.Connection, root: Path, source: str,
               policy: dict, config: dict) -> None:
    csv.field_size_limit(100_000_000)
    for path in sorted((root / policy["parts"]).glob("part-*.csv")):
        key, fingerprint = path.relative_to(root).as_posix(), digest(path)
        old = connection.execute("SELECT ma_bam,so_dong,hoan_tat FROM tien_do WHERE duong_dan=?", (key,)).fetchone()
        if old and old[0] != fingerprint:
            raise ValueError("Input part changed; refusing incompatible resume")
        if old and old[2]:
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            count = 0
            for count, row in enumerate(csv.DictReader(stream), 1):
                if old and count <= old[1]:
                    continue
                add_email(connection, source, row, policy, minimum=config["min_body_alphanumeric"])
                if count % config["batch_rows"] == 0:
                    connection.execute("INSERT OR REPLACE INTO tien_do VALUES (?,?,?,0)", (key, fingerprint, count))
                    connection.commit()
            connection.execute("INSERT OR REPLACE INTO tien_do VALUES (?,?,?,1)", (key, fingerprint, count))
            connection.commit()
        if int(path.stem.split("-")[-1]) % 25 == 0:
            print(json.dumps({"source": source, "completed_part": path.name}), flush=True)


def archive_members(executable: Path, archive: Path):
    """7-Zip's archive-order metadata; never print member names/email headers."""
    process = subprocess.Popen([str(executable), "l", "-slt", "-ba", "-sccUTF-8", str(archive)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, encoding="utf-8")
    current = {}
    def is_file(member):
        return ("Size" in member and "Path" in member
                and member.get("Folder") != "+"
                and not member.get("Attributes", "").startswith("D"))
    try:
        for line in process.stdout:
            if not line.strip():
                if is_file(current):
                    yield current
                current = {}
            elif " = " in line:
                key, _, value = line.rstrip("\r\n").partition(" = ")
                current[key] = value
        if is_file(current):
            yield current
        if process.wait() != 0:
            raise ValueError("7-Zip metadata listing failed")
    finally:
        if process.poll() is None:
            process.kill()
        process.stdout.close()


def ingest_archive(connection: sqlite3.Connection, root: Path, source: str,
                   policy: dict, config: dict, executable: Path) -> None:
    path = root / policy["archive"]
    key, fingerprint = path.relative_to(root).as_posix(), digest(path)
    old = connection.execute("SELECT ma_bam,so_dong,hoan_tat FROM tien_do WHERE duong_dan=?", (key,)).fetchone()
    if old and old[0] != fingerprint:
        raise ValueError("Archive changed; refusing incompatible resume")
    if old and old[2]:
        return
    process = subprocess.Popen([str(executable), "x", "-so", "-bd", "-y", str(path)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        count = 0
        for count, member in enumerate(archive_members(executable, path), 1):
            remaining, crc, pieces = int(member["Size"]), 0, []
            retain = remaining <= config["max_email_bytes"]
            while remaining:
                chunk = process.stdout.read(min(remaining, 65536))
                if not chunk:
                    raise ValueError("Archive stream ended early")
                crc = zlib.crc32(chunk, crc)
                remaining -= len(chunk)
                if retain:
                    pieces.append(chunk)
            if member.get("CRC") and crc != int(member["CRC"], 16):
                raise ValueError("Archive-order/CRC mismatch; refusing misattributed email")
            if old and count <= old[1]:
                continue
            relative = PurePosixPath(member["Path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive member path")
            identifier = fingerprint + ":" + str(relative)
            if retain:
                try:
                    row = parse_eml_bytes(b"".join(pieces))
                    row.update(id=identifier, source_path=str(relative),
                               raw_content_sha256=hashlib.sha256(b"".join(pieces)).hexdigest())
                    add_email(connection, source, row, policy, minimum=config["min_body_alphanumeric"])
                except (ValueError, UnicodeError, LookupError):
                    connection.execute("INSERT OR IGNORE INTO ban_ghi VALUES (?,?,NULL,?)", (
                        source + ":" + identifier, source, "excluded_mime_parse_failure"))
            else:
                connection.execute("INSERT OR IGNORE INTO ban_ghi VALUES (?,?,NULL,?)", (
                    source + ":" + identifier, source, "excluded_oversized_email"))
            if count % config["batch_rows"] == 0:
                connection.execute("INSERT OR REPLACE INTO tien_do VALUES (?,?,?,0)", (key, fingerprint, count))
                connection.commit()
            if count % 10000 == 0:
                print(json.dumps({"source": source, "processed": count}), flush=True)
        if process.wait() != 0:
            raise ValueError("7-Zip extraction stream failed")
        connection.execute("INSERT OR REPLACE INTO tien_do VALUES (?,?,?,1)", (key, fingerprint, count))
        connection.commit()
    finally:
        if process.poll() is None:
            process.kill()
        process.stdout.close()


def summary(connection: sqlite3.Connection) -> dict:
    return {
        "input_records": connection.execute("SELECT COUNT(*) FROM ban_ghi").fetchone()[0],
        "unique_quality_candidates": connection.execute("SELECT COUNT(*) FROM thu").fetchone()[0],
        "by_source_outcome": list(connection.execute("SELECT nguon,ly_do,COUNT(*) FROM ban_ghi GROUP BY nguon,ly_do")),
        "unique_source_status_language": list(connection.execute("SELECT nguon,trang_thai_nhan,ngon_ngu_goi_y,COUNT(*) FROM thu GROUP BY nguon,trang_thai_nhan,ngon_ngu_goi_y")),
        "training_rows": 0, "validation_rows": 0, "test_rows": 0,
        "training_gate_passed": False,
        "limitations": ["Quality/normalized-unique candidates only; near-duplicate and protected-group audit pending.",
                        "Enron remains unlabeled; source spam remains ambiguous weak label, never evaluation ground truth."]}


def inherited_references(connection: sqlite3.Connection, root: Path) -> dict:
    """Reuse only the already admitted 9,730, not the unapproved 38,013 wholesale.

    Historical development exposure is recorded, not disguised as a fresh Test.
    Existing labels and their permission/curation evidence are preserved verbatim.
    """
    config = json.loads((root / "config/v22/approved_development.json").read_text(encoding="utf-8"))
    csv.field_size_limit(100_000_000)
    for split in ("train", "validation"):
        path = root / config[split]
        if digest(path) != config[split + "_sha256"]:
            raise ValueError("Previously admitted reference changed")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["label"] not in {"normal", "spam", "phishing"}:
                    raise ValueError("Reference has unsupported label")
                required = ("label_provenance", "curation_rule_id", "v22_owner_decision", "v22_use_permission_basis")
                if not all(row.get(key) for key in required):
                    raise ValueError("Previously admitted reference lacks evidence")
                policy = {"label_status": "verified", "candidate_label": row["label"],
                          "rights_evidence": row["v22_use_permission_basis"]}
                source = "inherited_verified_" + row["source"]
                add_email(connection, source, row, policy, minimum=0)
                mail = connection.execute("SELECT id_thu FROM ban_ghi WHERE id_nguon=?", (source + ":" + row["id"],)).fetchone()
                if mail is None or mail[0] is None:
                    continue
                evidence = {key: row[key] for key in required}
                evidence.update(historical_role="previous_development_not_independent_test",
                                new_human_review=False, input_sha256=config[split + "_sha256"],
                                data_origin=row.get("data_origin", "unknown"))
                connection.execute("INSERT OR IGNORE INTO tham_chieu VALUES (?,?,?,?,?,?,?,?)", (
                    row["id"], mail[0], row["label"], row["language"], row["final_group_id"],
                    row.get("parent_id", ""), row["split"], json.dumps(evidence)))
        connection.commit()
    return {"rows": connection.execute("SELECT COUNT(*) FROM tham_chieu").fetchone()[0],
            "unique_mail": connection.execute("SELECT COUNT(DISTINCT id_thu) FROM tham_chieu").fetchone()[0],
            "by_original_split_label_language": list(connection.execute(
                "SELECT split_goc,nhan,ngon_ngu,COUNT(*) FROM tham_chieu GROUP BY 1,2,3")),
            "conflicting_verified_content": connection.execute(
                "SELECT COUNT(*) FROM (SELECT id_thu FROM tham_chieu GROUP BY id_thu HAVING COUNT(DISTINCT nhan)>1)").fetchone()[0],
            "new_independent_reference_test_rows": 0}


def protected_metadata_guard(connection: sqlite3.Connection, root: Path) -> dict:
    """Read only old *audit fingerprints*, never a frozen Test CSV/text/label.

    Conservatively protect all V2 reference fingerprints, including development,
    rather than inspecting Test membership or outcomes to choose new samples.
    This exact/template guard does not certify full near-duplicate coverage.
    """
    path = root / "data/v21/cache/dedup_audit.sqlite3"
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as old:
        protected = set()
        groups = set()
        fields = "normalized_hash,template_hash,normalized_group_id,template_group_id,provenance_group_id"
        for row in old.execute("SELECT " + fields + " FROM rows WHERE dataset='v2_reference'"):
            protected.update((value, kind) for value, kind in zip(row[:2], ("normalized", "template")) if value)
            groups.update(value for value in row[2:] if value)
        for row in old.execute("SELECT " + fields + " FROM rows WHERE dataset<>'v2_reference'"):
            if any(value in groups for value in row[2:] if value):
                protected.update((value, kind) for value, kind in zip(row[:2], ("normalized", "template")) if value)
    connection.executemany("INSERT OR IGNORE INTO bam_bao_ve VALUES (?,?)", sorted(protected))
    connection.execute("""INSERT OR IGNORE INTO cach_ly
        SELECT t.id_thu,'legacy_reference_exact_or_template_constraint'
        FROM thu t WHERE EXISTS(SELECT 1 FROM bam_bao_ve b WHERE
        (b.loai='normalized' AND b.ma_bam=t.ma_bam_chuan) OR
        (b.loai='template' AND b.ma_bam=t.ma_bam_mau))""")
    connection.commit()
    return {"protected_hashes": len(protected),
            "quarantined_unique_mail": connection.execute("SELECT COUNT(DISTINCT id_thu) FROM cach_ly").fetchone()[0],
            "legacy_test_file_opened": False,
            "near_duplicate_and_translation_coverage_certified": False}
