"""Streaming, resumable extraction of the labelled TREC06C ZIP corpus.

The index's ham/spam values are *candidates*, never training labels here.
Only ``data/`` members named by the index are read; ``data_cut/`` is ignored.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from bs4 import BeautifulSoup


INDEX_MEMBER = "trec06c-utf8/label/index"
PARSER_VERSION = "2"
INDEX_PATH = re.compile(r"^\.\./data/(\d{3}/\d{3})$")
CHINESE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
FIELDS = (
    "id", "source", "archive_name", "index_path", "index_line", "original_path",
    "zip_member", "raw_label", "candidate_label", "label", "label_status",
    "training_eligible", "subject", "body", "sender", "receiver", "date",
    "language_candidate", "content_hash", "normalized_text_hash", "duplicate_raw_of",
    "duplicate_text_of", "has_html", "html_only", "missing_content", "empty",
    "decode_error", "charset_fallback", "charsets_used", "contains_chinese",
    "mime_recovered", "mime_defects", "body_source",
    "byte_length", "member_crc", "status", "error",
)
COUNTER_FIELDS = (
    "processed", "ok", "failed", "empty", "missing_content", "decode_error",
    "charset_fallback", "has_html", "html_only", "contains_chinese",
    "mime_recovered", "mime_defects", "exact_duplicates", "normalized_duplicates", "missing_member",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def _decode_part(part: Message) -> tuple[str, str, bool, bool]:
    payload = part.get_payload(decode=True)
    if payload is None:
        content = part.get_payload()
        return str(content if isinstance(content, str) else ""), "already_text", False, False
    declared = part.get_content_charset()
    # The source is explicitly the UTF-8 variant. Its body is sometimes
    # transcoded while a legacy gb2312/gbk header survives unchanged.
    utf8_transcoded = not payload.isascii()
    if utf8_transcoded:
        try:
            payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            utf8_transcoded = False
    candidates = (["utf-8", declared] if utf8_transcoded else [declared, "utf-8"]) + ["gb18030", "big5"]
    tried: set[str] = set()
    for charset in candidates:
        if not charset or charset.casefold() in tried:
            continue
        tried.add(charset.casefold())
        try:
            return payload.decode(charset, errors="strict"), charset, False, bool(declared and charset != declared)
        except (UnicodeDecodeError, LookupError):
            continue
    return payload.decode("utf-8", errors="replace"), "utf-8-replace", True, True


def _html_text(value: str) -> str:
    soup = BeautifulSoup(value, "html.parser")
    for tag in soup.find_all(("script", "style", "noscript")):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    links = [str(tag.get("href")).strip() for tag in soup.find_all(href=True)]
    return "\n".join([text, *(link for link in links if link)]).strip()


def parse_message(raw: bytes) -> dict[str, object]:
    """Decode one MIME email, retaining subject/body and diagnostic flags."""
    message = BytesParser(policy=policy.default).parsebytes(raw)
    plain: list[str] = []
    html: list[str] = []
    charsets: list[str] = []
    decode_error = False
    fallback = False
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        media_type = part.get_content_type()
        if media_type not in ("text/plain", "text/html"):
            continue
        value, charset, error, used_fallback = _decode_part(part)
        charsets.append(charset)
        decode_error |= error
        fallback |= used_fallback
        (html if media_type == "text/html" else plain).append(value)
    subject = _clean(message.get("subject", ""))
    decode_error |= "\ufffd" in subject
    body = "\n".join(_clean(text) for text in plain if _clean(text))
    if not body:
        html_texts = [_clean(_html_text(text)) for text in html]
        body = "\n".join(text for text in html_texts if text)
    body_source = "mime"
    recovered = False
    raw_body_match = re.search(rb"\r?\n\r?\n", raw)
    if raw_body_match:
        raw_body = raw[raw_body_match.end():]
        transfer_encoding = str(message.get("Content-Transfer-Encoding", "")).casefold()
        has_bad_boundary = any(
            type(defect).__name__ in ("StartBoundaryNotFoundDefect", "MultipartInvariantViolationDefect")
            for defect in message.defects
        )
        # Only bypass MIME when the archive contains actual UTF-8 non-ASCII
        # body bytes inconsistent with a stale transfer header, or MIME
        # parsing demonstrably dropped the textual payload.
        if not raw_body.isascii() and (
            transfer_encoding in ("base64", "quoted-printable") or has_bad_boundary or not body
        ):
            try:
                recovered_body = raw_body.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                recovered_body = ""
            if recovered_body and (
                not body or len(CHINESE.findall(recovered_body)) > len(CHINESE.findall(body))
            ):
                if message.get_content_type() == "text/html" or "<html" in recovered_body.casefold():
                    recovered_body = _html_text(recovered_body)
                body = _clean(recovered_body)
                body_source = "utf8_raw_recovery"
                recovered = True
                fallback = True
    body = unicodedata.normalize("NFC", body)
    decode_error |= "\ufffd" in body
    return {
        "subject": subject,
        "body": body,
        "sender": _clean(message.get("from", "")),
        "receiver": _clean(message.get("to", "")),
        "date": _clean(message.get("date", "")),
        "has_html": bool(html),
        "html_only": bool(html and not plain),
        "decode_error": decode_error,
        "charset_fallback": fallback,
        "charsets_used": ",".join(dict.fromkeys(charsets)),
        "missing_content": not bool(body),
        "empty": not bool(subject or body),
        "contains_chinese": bool(CHINESE.search(subject + " " + body)),
        "mime_recovered": recovered,
        "mime_defects": ",".join(dict.fromkeys(type(defect).__name__ for defect in message.defects)),
        "body_source": body_source,
    }


def _entries(archive: ZipFile):
    with archive.open(INDEX_MEMBER) as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.decode("utf-8-sig").strip()
            parts = line.split()
            if len(parts) != 2 or parts[0] not in ("ham", "spam"):
                raise ValueError(f"Invalid label index line {line_number}: {line!r}")
            match = INDEX_PATH.fullmatch(parts[1])
            if match is None:
                raise ValueError(f"Unsafe/non-original index path at line {line_number}: {parts[1]!r}")
            yield line_number, parts[0], parts[1], f"trec06c-utf8/data/{match.group(1)}"


def _connect(db_path: Path, archive_sha256: str, max_member_bytes: int) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS hashes (kind TEXT NOT NULL, hash TEXT NOT NULL, first_id TEXT NOT NULL, PRIMARY KEY(kind, hash))")
    connection.execute("CREATE TABLE IF NOT EXISTS parts (number INTEGER PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL, rows INTEGER NOT NULL)")
    previous = connection.execute("SELECT value FROM state WHERE key='archive_sha256'").fetchone()
    if previous and previous[0] != archive_sha256:
        connection.close()
        raise ValueError("Archive changed since checkpoint; refusing to mix outputs")
    if not previous:
        connection.execute("INSERT INTO state VALUES ('archive_sha256', ?)", (archive_sha256,))
        connection.execute("INSERT INTO state VALUES ('parser_version', ?)", (PARSER_VERSION,))
        connection.execute("INSERT INTO state VALUES ('max_member_bytes', ?)", (str(max_member_bytes),))
        connection.execute("INSERT INTO state VALUES ('next_line', '1')")
        connection.execute("INSERT INTO state VALUES ('counts', ?)", (json.dumps({}),))
        connection.commit()
    elif connection.execute("SELECT value FROM state WHERE key='parser_version'").fetchone() != (PARSER_VERSION,):
        connection.close()
        raise ValueError("Parser version changed since checkpoint; use a new output directory")
    else:
        saved_limit = connection.execute("SELECT value FROM state WHERE key='max_member_bytes'").fetchone()
        if saved_limit is None and max_member_bytes == 10_000_000:
            # Early parser-v2 checkpoints used this fixed default before the
            # value was explicitly frozen in checkpoint metadata.
            connection.execute("INSERT INTO state VALUES ('max_member_bytes', '10000000')")
            connection.commit()
        elif saved_limit != (str(max_member_bytes),):
            connection.close()
            raise ValueError("Member size limit changed since checkpoint; use a new output directory")
    for _, filename, expected, _ in connection.execute("SELECT number, filename, sha256, rows FROM parts"):
        part = db_path.parent / filename
        if not part.is_file() or _sha256_file(part) != expected:
            connection.close()
            raise ValueError(f"Committed output part missing/changed: {part}")
    return connection


def _state(connection: sqlite3.Connection, key: str) -> str:
    row = connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    assert row is not None
    return row[0]


def _make_row(
    archive: ZipFile, archive_name: str, line_number: int, raw_label: str,
    original_path: str, member: str, max_member_bytes: int,
) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update({
        "id": f"trec06c:{member.removeprefix('trec06c-utf8/data/')}",
        "source": "TREC06C", "archive_name": archive_name,
        "index_path": INDEX_MEMBER, "index_line": line_number,
        "original_path": original_path, "zip_member": member,
        "raw_label": raw_label, "candidate_label": raw_label,
        "label": "review", "label_status": "review", "training_eligible": False,
        "language_candidate": "zh", "status": "ok",
    })
    try:
        info = archive.getinfo(member)
    except KeyError:
        row.update(status="missing_member", error="Index member not found in ZIP")
        return row
    row["byte_length"] = info.file_size
    row["member_crc"] = f"{info.CRC:08x}"
    if info.file_size > max_member_bytes:
        row.update(status="oversize", error=f"Member exceeds {max_member_bytes} bytes")
        return row
    try:
        with archive.open(info) as handle:
            raw = handle.read(max_member_bytes + 1)
        if len(raw) > max_member_bytes:
            row.update(status="oversize", error=f"Member exceeds {max_member_bytes} bytes")
            return row
        row["content_hash"] = hashlib.sha256(raw).hexdigest()
        parsed = parse_message(raw)
        row.update(parsed)
        normalized = _clean(parsed["subject"] + " " + parsed["body"]).casefold()
        if normalized:
            row["normalized_text_hash"] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        row["status"] = "empty" if parsed["empty"] else "ok"
    except (BadZipFile, OSError, ValueError, TypeError, UnicodeError) as error:
        row.update(status="parse_error", error=f"{type(error).__name__}: {error}"[:300])
    return row


def _count_row(counts: Counter, row: dict[str, object]) -> None:
    counts["processed"] += 1
    counts["ok" if row["status"] in ("ok", "empty") else "failed"] += 1
    for field in ("empty", "missing_content", "decode_error", "charset_fallback", "has_html", "html_only", "contains_chinese", "mime_recovered"):
        counts[field] += int(bool(row[field]))
    counts["mime_defects"] += int(bool(row["mime_defects"]))
    counts["exact_duplicates"] += int(bool(row["duplicate_raw_of"]))
    counts["normalized_duplicates"] += int(bool(row["duplicate_text_of"]))
    counts["missing_member"] += int(row["status"] == "missing_member")


def _write_report(report_path: Path, payload: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)


def load_trec06c(
    archive_path: Path, output_dir: Path, report_path: Path, *,
    batch_size: int = 1000, max_member_bytes: int = 10_000_000,
    max_batches: int | None = None,
) -> dict[str, object]:
    """Write bounded CSV parts; resume atomically from SQLite checkpoint."""
    archive_path = Path(archive_path)
    if batch_size < 1 or max_member_bytes < 1:
        raise ValueError("batch_size and max_member_bytes must be positive")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_hash = _sha256_file(archive_path)
    connection = _connect(output_dir / "checkpoint.sqlite", archive_hash, max_member_bytes)
    try:
        with ZipFile(archive_path) as archive:
            labels = Counter(raw_label for _, raw_label, _, _ in _entries(archive))
            total = sum(labels.values())
            member_names = archive.namelist()
            archive_counts = {
                "all_members": len(member_names),
                "data_members": sum(name.startswith("trec06c-utf8/data/") for name in member_names),
                "data_cut_members": sum(name.startswith("trec06c-utf8/data_cut/") for name in member_names),
            }
            del member_names
            next_line = int(_state(connection, "next_line"))
            counts = Counter(json.loads(_state(connection, "counts")))
            parts_done = connection.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
            batches_this_run = 0
            entry_iter = iter(_entries(archive))
            while next_line <= total and (max_batches is None or batches_this_run < max_batches):
                rows: list[dict[str, object]] = []
                for line_number, raw_label, original_path, member in entry_iter:
                    if line_number < next_line:
                        continue
                    row = _make_row(archive, archive_path.name, line_number, raw_label, original_path, member, max_member_bytes)
                    rows.append(row)
                    if len(rows) == batch_size:
                        break
                if not rows:
                    break
                part_number = parts_done + 1
                filename = f"part-{part_number:05d}.csv"
                final_path = output_dir / filename
                temp_path: Path | None = None
                try:
                    connection.execute("BEGIN")
                    for row in rows:
                        for kind, hash_field, duplicate_field in (
                            ("raw", "content_hash", "duplicate_raw_of"),
                            ("text", "normalized_text_hash", "duplicate_text_of"),
                        ):
                            if not row[hash_field]:
                                continue
                            existing = connection.execute(
                                "SELECT first_id FROM hashes WHERE kind=? AND hash=?",
                                (kind, row[hash_field]),
                            ).fetchone()
                            if existing:
                                row[duplicate_field] = existing[0]
                            else:
                                connection.execute(
                                    "INSERT INTO hashes VALUES (?, ?, ?)",
                                    (kind, row[hash_field], row["id"]),
                                )
                        _count_row(counts, row)
                    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".tmp", dir=output_dir, delete=False) as handle:
                        temp_path = Path(handle.name)
                        writer = csv.DictWriter(handle, fieldnames=FIELDS)
                        writer.writeheader()
                        writer.writerows(rows)
                    part_hash = _sha256_file(temp_path)
                    os.replace(temp_path, final_path)
                    connection.execute("INSERT INTO parts VALUES (?, ?, ?, ?)", (part_number, filename, part_hash, len(rows)))
                    next_line = int(rows[-1]["index_line"]) + 1
                    connection.execute("UPDATE state SET value=? WHERE key='next_line'", (str(next_line),))
                    connection.execute("UPDATE state SET value=? WHERE key='counts'", (json.dumps(dict(counts)),))
                    connection.commit()
                except Exception:
                    connection.rollback()
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
                    raise
                parts_done += 1
                batches_this_run += 1
                payload = {
                    "source": "TREC06C", "archive_name": archive_path.name,
                    "archive_sha256": archive_hash, "parser_version": PARSER_VERSION,
                    "index_path": INDEX_MEMBER,
                    "index_rows": total, "archive_members": archive_counts,
                    "processed": counts["processed"], "complete": next_line > total,
                    "next_index_line": next_line, "csv_parts": parts_done,
                    "counts": {field: counts[field] for field in COUNTER_FIELDS},
                    "raw_label_counts": dict(labels),
                    "policy": "ham/spam are candidate_label only; all rows remain review and training_eligible=false",
                }
                _write_report(report_path, payload)
            # Also produce a report for an already-complete resumed run.
            payload = {
                "source": "TREC06C", "archive_name": archive_path.name,
                "archive_sha256": archive_hash, "parser_version": PARSER_VERSION,
                "index_path": INDEX_MEMBER,
                "index_rows": total, "archive_members": archive_counts,
                "processed": counts["processed"], "complete": next_line > total,
                "next_index_line": next_line, "csv_parts": parts_done,
                "counts": {field: counts[field] for field in COUNTER_FIELDS},
                "raw_label_counts": dict(labels),
                "policy": "ham/spam are candidate_label only; all rows remain review and training_eligible=false",
            }
            _write_report(report_path, payload)
            return payload
    finally:
        connection.close()
