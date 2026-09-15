"""Parse .eml messages with Python's standard email package."""

from __future__ import annotations

import unicodedata
from email import policy
from email.message import EmailMessage, Message
from email.parser import BytesParser
from pathlib import Path
from bs4 import BeautifulSoup


def _clean_header(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def _part_content(part: Message) -> str:
    try:
        content = part.get_content()
    except (LookupError, UnicodeDecodeError):
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        content = payload.decode(charset, errors="replace")
    if isinstance(content, bytes):
        charset = part.get_content_charset() or "utf-8"
        return content.decode(charset, errors="replace")
    return str(content)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for unsafe_element in soup.find_all(("script", "style", "noscript")):
        unsafe_element.decompose()
    visible_text = soup.get_text(" ", strip=True)
    hrefs = [
        str(element.get("href")).strip()
        for element in soup.find_all(href=True)
        if str(element.get("href", "")).strip()
    ]
    combined = "\n".join([visible_text, *hrefs])
    return combined.strip()


def _message_body(message: EmailMessage) -> str:
    if not message.is_multipart():
        content = _part_content(message)
        return _html_to_text(content) if message.get_content_type() == "text/html" else content

    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(_part_content(part))
        elif content_type == "text/html":
            html_parts.append(_html_to_text(_part_content(part)))
    selected = plain_parts if plain_parts else html_parts
    return "\n".join(value.strip() for value in selected if value.strip())


def parse_eml_bytes(data: bytes) -> dict[str, str]:
    """Parse raw .eml bytes into fields accepted by the inference API."""
    if not isinstance(data, bytes):
        raise TypeError(".eml input must be bytes")
    message = BytesParser(policy=policy.default).parsebytes(data)
    recipients = [
        _clean_header(value)
        for header in ("to", "cc")
        for value in message.get_all(header, [])
        if _clean_header(value)
    ]
    return {
        "sender": _clean_header(message.get("from", "")),
        "receiver": ", ".join(recipients),
        "date": _clean_header(message.get("date", "")),
        "subject": _clean_header(message.get("subject", "")),
        "body": unicodedata.normalize("NFC", _message_body(message)).strip(),
    }


def parse_eml_file(path: Path) -> dict[str, str]:
    """Read and parse one local .eml file."""
    return parse_eml_bytes(path.read_bytes())
