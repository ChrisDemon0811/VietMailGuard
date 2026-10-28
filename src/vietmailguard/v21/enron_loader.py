"""Stream original Enron maildir archives into review-only V2.1 CSV parts.

The raw corpus has no verified per-message normal/spam/phishing labels. Reading
or matching a V2 representation never makes a row training-eligible.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import tarfile
import tempfile
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup


PARSER_VERSION = "1"
FIELDS = (
    "id", "source", "input_format", "source_archive", "source_path",
    "source_ordinal", "message_id", "raw_label", "candidate_label", "label",
    "label_status", "training_eligible", "language_candidate", "data_origin",
    "sender", "receiver", "cc", "date", "subject", "body",
    "raw_sender", "raw_receiver", "raw_date", "raw_subject",
    "raw_content_sha256", "normalized_text_hash", "duplicate_raw_of",
    "duplicate_text_of", "v2_enron_csv_row", "v2_overlap_type",
    "is_forwarded", "has_html", "html_only", "has_attachment",
    "mime_defects", "decode_warning", "charsets_used", "missing_body",
    "empty", "byte_length", "status", "error",
)
COUNT_FLAGS = (
    "empty", "missing_body", "has_html", "html_only", "has_attachment",
    "mime_defects", "decode_warning", "is_forwarded", "duplicate_raw_of",
    "duplicate_text_of", "v2_enron_csv_row",
)
FORWARD_SUBJECT = re.compile(r"^(?:fw|fwd)\s*:", re.IGNORECASE)
FORWARD_BODY = re.compile(r"(?:^-{2,}\s*forwarded message|^-{2,}\s*original message)", re.IGNORECASE | re.MULTILINE)
V2_COLUMNS = {"subject", "body", "label"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def _strict_key(subject: str, body: str) -> str:
    return _clean(unicodedata.normalize("NFKC", subject + "\n" + body)).casefold()


def _compact_key(subject: str, body: str) -> str:
    # Supports candidate tracing to preprocessed Enron.csv; not proof of identity.
    return "".join(char for char in unicodedata.normalize("NFKC", subject + body).casefold() if char.isalnum())


def _body_key(body: str) -> str:
    return _clean(unicodedata.normalize("NFKC", body)).casefold()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text_part(part: Message) -> tuple[str, str, bool]:
    payload = part.get_payload(decode=True)
    if payload is None:
        try:
            content = part.get_content()
        except (LookupError, UnicodeError):
            content = part.get_payload()
        return str(content if isinstance(content, str) else ""), "already_text", False
    declared = part.get_content_charset()
    seen: set[str] = set()
    for charset in (declared, "utf-8", "windows-1252", "latin-1"):
        if not charset or charset.casefold() in seen:
            continue
        seen.add(charset.casefold())
        try:
            return payload.decode(charset, errors="strict"), charset, bool(declared and charset != declared)
        except (UnicodeError, LookupError):
            continue
    return payload.decode("utf-8", errors="replace"), "utf-8-replace", True


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(("script", "style", "noscript")):
        tag.decompose()
    visible = soup.get_text(" ", strip=True)
    links = [str(tag.get("href")).strip() for tag in soup.find_all(href=True)]
    return "\n".join([visible, *(link for link in links if link)]).strip()


def parse_enron_message(raw: bytes) -> dict[str, object]:
    """Extract MIME text without fetching URLs or reading attachment payloads."""
    message = BytesParser(policy=policy.default).parsebytes(raw)
    plain: list[str] = []
    html: list[str] = []
    charsets: list[str] = []
    defects: list[str] = []
    decode_warning = False
    has_attachment = False
    for part in message.walk():
        defects.extend(type(item).__name__ for item in part.defects)
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            has_attachment = True
            continue
        media = part.get_content_type()
        if media not in ("text/plain", "text/html"):
            if part.get_filename():
                has_attachment = True
            continue
        content, charset, warning = _text_part(part)
        charsets.append(charset)
        decode_warning |= warning
        (html if media == "text/html" else plain).append(content)
    body = "\n".join(_clean(text) for text in plain if _clean(text))
    if not body:
        rendered = [_clean(_html_to_text(value)) for value in html]
        body = "\n".join(value for value in rendered if value)
    raw_subject = str(message.get("subject", ""))
    raw_sender = str(message.get("from", ""))
    raw_receiver = str(message.get("to", ""))
    raw_date = str(message.get("date", ""))
    subject = _clean(raw_subject)
    decode_warning |= "\ufffd" in subject or "\ufffd" in body
    return {
        "message_id": _clean(message.get("message-id", "")),
        "sender": _clean(raw_sender), "receiver": _clean(raw_receiver),
        "cc": _clean(", ".join(str(value) for value in message.get_all("cc", []))),
        "date": _clean(raw_date), "subject": subject, "body": unicodedata.normalize("NFC", body),
        "raw_sender": raw_sender, "raw_receiver": raw_receiver,
        "raw_date": raw_date, "raw_subject": raw_subject,
        "is_forwarded": bool(FORWARD_SUBJECT.search(subject) or FORWARD_BODY.search(body)),
        "has_html": bool(html), "html_only": bool(html and not plain),
        "has_attachment": has_attachment, "mime_defects": ",".join(dict.fromkeys(defects)),
        "decode_warning": decode_warning, "charsets_used": ",".join(dict.fromkeys(charsets)),
        "missing_body": not bool(body), "empty": not bool(subject or body),
    }


def detect_enron_format(path: Path) -> str:
    """Identify the actual supplied container, independently of its suffix."""
    path = Path(path)
    if path.is_dir():
        return "maildir"
    if not path.is_file():
        raise FileNotFoundError(path)
    if zipfile.is_zipfile(path):
        return "zip"
    if tarfile.is_tarfile(path):
        return "tar"
    with path.open("rb") as handle:
        prefix = handle.read(4096)
    if b"," in prefix and (b"subject" in prefix.lower() or b"body" in prefix.lower()):
        return "csv"
    raise ValueError(f"Unsupported Enron input format: {path}")


def discover_enron_source(dataset_dir: Path) -> Path:
    candidates = sorted(
        path for path in Path(dataset_dir).iterdir()
        if "enron" in path.name.casefold() or (path.is_dir() and path.name.casefold() == "maildir")
    )
    originals = [path for path in candidates if detect_enron_format(path) != "csv"]
    if len(originals) != 1:
        raise ValueError(f"Expected one original Enron source in {dataset_dir}; found {len(originals)}. Specify --input.")
    return originals[0]


@dataclass(frozen=True)
class _SourceItem:
    ordinal: int
    path: str
    size: int
    data: bytes | None
    error: str = ""


def _iter_maildir(root: Path, start: int, limit: int) -> Iterator[_SourceItem]:
    ordinal = 0
    for directory, subdirs, files in os.walk(root):
        subdirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            ordinal += 1
            if ordinal < start:
                continue
            relative = path.relative_to(root).as_posix()
            try:
                size = path.stat().st_size
                if size > limit:
                    yield _SourceItem(ordinal, relative, size, None, "oversize")
                else:
                    yield _SourceItem(ordinal, relative, size, path.read_bytes())
            except OSError as error:
                yield _SourceItem(ordinal, relative, 0, None, f"read_error: {error}"[:300])


def _iter_tar(path: Path, start: int, limit: int) -> Iterator[_SourceItem]:
    ordinal = 0
    with tarfile.open(path, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            ordinal += 1
            if ordinal < start:
                continue
            if member.size > limit:
                yield _SourceItem(ordinal, member.name, member.size, None, "oversize")
                continue
            try:
                stream = archive.extractfile(member)
                if stream is None:
                    raise OSError("Tar member cannot be opened")
                data = stream.read(limit + 1)
                if len(data) > limit:
                    yield _SourceItem(ordinal, member.name, member.size, None, "oversize")
                else:
                    yield _SourceItem(ordinal, member.name, member.size, data)
            except (OSError, tarfile.TarError) as error:
                yield _SourceItem(ordinal, member.name, member.size, None, f"read_error: {error}"[:300])


def _iter_zip(path: Path, start: int, limit: int) -> Iterator[_SourceItem]:
    ordinal = 0
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            ordinal += 1
            if ordinal < start:
                continue
            if member.file_size > limit:
                yield _SourceItem(ordinal, member.filename, member.file_size, None, "oversize")
                continue
            try:
                with archive.open(member) as handle:
                    data = handle.read(limit + 1)
                if len(data) > limit:
                    yield _SourceItem(ordinal, member.filename, member.file_size, None, "oversize")
                else:
                    yield _SourceItem(ordinal, member.filename, member.file_size, data)
            except (OSError, zipfile.BadZipFile) as error:
                yield _SourceItem(ordinal, member.filename, member.file_size, None, f"read_error: {error}"[:300])


def _iter_source(path: Path, kind: str, start: int, limit: int) -> Iterator[_SourceItem]:
    if kind == "maildir":
        yield from _iter_maildir(path, start, limit)
    elif kind == "tar":
        yield from _iter_tar(path, start, limit)
    elif kind == "zip":
        yield from _iter_zip(path, start, limit)
    else:
        raise ValueError("CSV is a derived representation, not the original MIME corpus")


def _source_fingerprint(path: Path, kind: str) -> str:
    if kind != "maildir":
        return _sha256_file(path)
    # Bound memory while fingerprinting the directory inventory. Maildir input
    # must remain immutable while processing; per-message raw hashes are kept.
    digest = hashlib.sha256()
    for directory, subdirs, files in os.walk(path):
        subdirs.sort()
        for name in sorted(files):
            member = Path(directory) / name
            if member.is_symlink() or not member.is_file():
                continue
            stat = member.stat()
            digest.update(f"{member.relative_to(path).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _open_checkpoint(output_dir: Path, source_fingerprint: str, kind: str, limit: int, v2_fingerprint: str) -> sqlite3.Connection:
    connection = sqlite3.connect(output_dir / "checkpoint.sqlite")
    connection.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS hashes (kind TEXT NOT NULL, hash TEXT NOT NULL, first_id TEXT NOT NULL, PRIMARY KEY(kind, hash))")
    connection.execute("CREATE TABLE IF NOT EXISTS parts (number INTEGER PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL, rows INTEGER NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS v2_keys (kind TEXT NOT NULL, hash TEXT NOT NULL, row_number INTEGER NOT NULL, raw_label TEXT NOT NULL, PRIMARY KEY(kind, hash, row_number))")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_v2_key ON v2_keys(kind, hash)")
    settings = {
        "source_fingerprint": source_fingerprint, "input_format": kind,
        "max_member_bytes": str(limit), "v2_fingerprint": v2_fingerprint,
        "parser_version": PARSER_VERSION,
    }
    exists = connection.execute("SELECT value FROM state WHERE key='next_ordinal'").fetchone()
    if exists:
        for key, value in settings.items():
            actual = connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
            if actual != (value,):
                connection.close()
                raise ValueError(f"Checkpoint input/configuration changed: {key}")
    else:
        connection.executemany("INSERT INTO state VALUES (?, ?)", settings.items())
        connection.execute("INSERT INTO state VALUES ('next_ordinal', '1')")
        connection.execute("INSERT INTO state VALUES ('counts', '{}')")
        connection.execute("INSERT INTO state VALUES ('complete', '0')")
        connection.commit()
    for _, filename, expected, _ in connection.execute("SELECT number, filename, sha256, rows FROM parts"):
        part = output_dir / filename
        if not part.is_file() or _sha256_file(part) != expected:
            connection.close()
            raise ValueError(f"Committed output part missing/changed: {part}")
    return connection


def _state(connection: sqlite3.Connection, key: str) -> str:
    row = connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    if row is None:
        raise ValueError(f"Missing checkpoint state: {key}")
    return row[0]


def _build_v2_index(connection: sqlite3.Connection, path: Path | None) -> None:
    if connection.execute("SELECT 1 FROM state WHERE key='v2_index_ready'").fetchone():
        return
    if path is None:
        connection.execute("INSERT INTO state VALUES ('v2_index_ready', 'unavailable')")
        connection.commit()
        return
    csv.field_size_limit(20_000_000)
    connection.execute("BEGIN")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not V2_COLUMNS <= set(reader.fieldnames or []):
                raise ValueError("V2 Enron.csv lacks subject/body/label columns")
            for number, row in enumerate(reader, 1):
                subject, body = row["subject"] or "", row["body"] or ""
                keys = {"strict_full": _strict_key(subject, body), "compact_full": _compact_key(subject, body)}
                if len(_body_key(body)) >= 80:
                    keys["strict_body"] = _body_key(body)
                for kind, value in keys.items():
                    if value:
                        connection.execute(
                            "INSERT OR IGNORE INTO v2_keys VALUES (?, ?, ?, ?)",
                            (kind, _hash(value), number, row["label"] or ""),
                        )
        connection.execute("INSERT INTO state VALUES ('v2_index_ready', 'ready')")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _row(item: _SourceItem, kind: str, archive_name: str) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update({
        "id": "enron:" + hashlib.sha256(item.path.encode("utf-8")).hexdigest()[:24],
        "source": "Enron_original", "input_format": kind,
        "source_archive": archive_name, "source_path": item.path,
        "source_ordinal": item.ordinal, "raw_label": "", "candidate_label": "",
        "label": "review", "label_status": "review", "training_eligible": False,
        "language_candidate": "en", "data_origin": "original",
        "byte_length": item.size, "status": "ok",
    })
    if item.data is None:
        row.update(status=item.error.split(":", 1)[0], error=item.error)
        return row
    row["raw_content_sha256"] = hashlib.sha256(item.data).hexdigest()
    try:
        parsed = parse_enron_message(item.data)
        row.update(parsed)
        normalized = _strict_key(str(row["subject"]), str(row["body"]))
        if normalized:
            row["normalized_text_hash"] = _hash(normalized)
        row["status"] = "empty" if row["empty"] else "ok"
    except (ValueError, TypeError, UnicodeError, LookupError, OSError) as error:
        row.update(status="parse_error", error=f"{type(error).__name__}: {error}"[:300])
    return row


def _mark_duplicates_and_overlap(connection: sqlite3.Connection, row: dict[str, object]) -> None:
    for kind, hash_field, match_field in (
        ("raw", "raw_content_sha256", "duplicate_raw_of"),
        ("text", "normalized_text_hash", "duplicate_text_of"),
    ):
        value = row[hash_field]
        if not value:
            continue
        existing = connection.execute("SELECT first_id FROM hashes WHERE kind=? AND hash=?", (kind, value)).fetchone()
        if existing:
            row[match_field] = existing[0]
        else:
            connection.execute("INSERT INTO hashes VALUES (?, ?, ?)", (kind, value, row["id"]))
    if not row["body"]:
        return
    subject, body = str(row["subject"]), str(row["body"])
    keys = [("strict_full", _strict_key(subject, body)), ("compact_full", _compact_key(subject, body))]
    if len(_body_key(body)) >= 80:
        keys.append(("strict_body", _body_key(body)))
    for kind, value in keys:
        if not value:
            continue
        match = connection.execute(
            "SELECT row_number FROM v2_keys WHERE kind=? AND hash=? ORDER BY row_number LIMIT 1",
            (kind, _hash(value)),
        ).fetchone()
        if match:
            row["v2_enron_csv_row"] = match[0]
            row["v2_overlap_type"] = kind
            break


def _count(counts: Counter, row: dict[str, object]) -> None:
    counts["processed"] += 1
    counts["parsed" if row["status"] in ("ok", "empty") else "failed"] += 1
    for flag in COUNT_FLAGS:
        counts[flag] += int(bool(row[flag]))
    counts[f"status_{row['status']}"] += 1
    if row["v2_overlap_type"]:
        counts[f"v2_{row['v2_overlap_type']}"] += 1


def _report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_enron(
    input_path: Path, output_dir: Path, report_path: Path, *,
    v2_reference: Path | None = None, batch_size: int = 1000,
    max_member_bytes: int = 10_000_000, max_batches: int | None = None,
) -> dict[str, object]:
    """Read original MIME messages one at a time and checkpoint CSV batches."""
    input_path, output_dir, report_path = Path(input_path), Path(output_dir), Path(report_path)
    if batch_size < 1 or max_member_bytes < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("Batch/member limits must be positive")
    kind = detect_enron_format(input_path)
    if kind == "csv":
        raise ValueError("CSV is a derived representation; original MIME maildir/tar/ZIP is required")
    if v2_reference is not None:
        v2_reference = Path(v2_reference)
        if not v2_reference.is_file():
            raise FileNotFoundError(v2_reference)
    source_fingerprint = _source_fingerprint(input_path, kind)
    v2_fingerprint = _sha256_file(v2_reference) if v2_reference else "unavailable"
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = _open_checkpoint(output_dir, source_fingerprint, kind, max_member_bytes, v2_fingerprint)
    try:
        _build_v2_index(connection, v2_reference)
        next_ordinal = int(_state(connection, "next_ordinal"))
        counts = Counter(json.loads(_state(connection, "counts")))
        parts_done = connection.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
        complete = _state(connection, "complete") == "1"
        batches_done = 0
        if not complete:
            iterator = iter(_iter_source(input_path, kind, next_ordinal, max_member_bytes))
            while max_batches is None or batches_done < max_batches:
                rows = []
                for item in iterator:
                    rows.append(_row(item, kind, input_path.name))
                    if len(rows) >= batch_size:
                        break
                if not rows:
                    complete = True
                    connection.execute("UPDATE state SET value='1' WHERE key='complete'")
                    connection.commit()
                    break
                part_number = parts_done + 1
                filename = f"part-{part_number:05d}.csv"
                temp_path: Path | None = None
                try:
                    connection.execute("BEGIN")
                    for row in rows:
                        _mark_duplicates_and_overlap(connection, row)
                        _count(counts, row)
                    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".tmp", dir=output_dir, delete=False) as handle:
                        temp_path = Path(handle.name)
                        writer = csv.DictWriter(handle, fieldnames=FIELDS)
                        writer.writeheader()
                        writer.writerows(rows)
                    checksum = _sha256_file(temp_path)
                    os.replace(temp_path, output_dir / filename)
                    next_ordinal = int(rows[-1]["source_ordinal"]) + 1
                    connection.execute("INSERT INTO parts VALUES (?, ?, ?, ?)", (part_number, filename, checksum, len(rows)))
                    connection.execute("UPDATE state SET value=? WHERE key='next_ordinal'", (str(next_ordinal),))
                    connection.execute("UPDATE state SET value=? WHERE key='counts'", (json.dumps(dict(counts)),))
                    connection.commit()
                except Exception:
                    connection.rollback()
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
                    raise
                parts_done += 1
                batches_done += 1
                _report(report_path, {
                    "source": "Enron_original", "input_format": kind,
                    "input_name": input_path.name, "input_sha256_or_inventory": source_fingerprint,
                    "v2_reference_name": v2_reference.name if v2_reference else None,
                    "v2_reference_sha256": v2_fingerprint,
                    "parser_version": PARSER_VERSION, "complete": False,
                    "processed": counts["processed"], "next_ordinal": next_ordinal,
                    "csv_parts": parts_done, "counts": dict(counts),
                    "label_policy": "no verified raw label; all rows review and training_eligible=false",
                })
        payload = {
            "source": "Enron_original", "input_format": kind,
            "input_name": input_path.name, "input_sha256_or_inventory": source_fingerprint,
            "v2_reference_name": v2_reference.name if v2_reference else None,
            "v2_reference_sha256": v2_fingerprint,
            "parser_version": PARSER_VERSION, "complete": complete,
            "processed": counts["processed"], "next_ordinal": next_ordinal,
            "csv_parts": parts_done, "counts": dict(counts),
            "label_policy": "no verified raw label; all rows review and training_eligible=false",
        }
        _report(report_path, payload)
        return payload
    finally:
        connection.close()
