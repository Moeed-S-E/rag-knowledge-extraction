"""Text cleaning and preprocessing."""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


_WHITESPACE = re.compile(r"\s+")
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def clean_text(text: object, max_characters: int = 20_000) -> str:
    """Normalize a text value without deleting non-Latin scripts."""
    if text is None:
        return ""
    value = html.unescape(str(text))
    if BeautifulSoup:
        value = BeautifulSoup(value, "html.parser").get_text(" ")
    else:
        value = re.sub(r"<[^>]+>", " ", value)
    value = _CONTROL.sub(" ", value)
    value = unicodedata.normalize("NFKC", value)
    value = _WHITESPACE.sub(" ", value).strip()
    if len(value) > max_characters:
        value = value[:max_characters].rsplit(" ", 1)[0].strip() or value[:max_characters]
    return value


def clean_record(record: dict, max_characters: int = 20_000) -> dict:
    cleaned = dict(record)
    cleaned["title"] = clean_text(record.get("title", ""), max_characters=500)
    cleaned["text"] = clean_text(record.get("text", ""), max_characters=max_characters)
    cleaned["source"] = clean_text(record.get("source", "local"), max_characters=500)
    return cleaned


def clean_records(records: Iterable[dict], max_characters: int = 20_000) -> tuple[list[dict], dict]:
    output: list[dict] = []
    dropped = 0
    for record in records:
        cleaned = clean_record(record, max_characters=max_characters)
        if not cleaned.get("text", "").strip():
            dropped += 1
            continue
        output.append(cleaned)
    total = len(output) + dropped
    return output, {
        "input_records": total,
        "output_records": len(output),
        "dropped_records": dropped,
        "drop_percentage": round((dropped / total) * 100, 2) if total else 0.0,
        "max_characters": max_characters,
    }
