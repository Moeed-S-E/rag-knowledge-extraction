"""Dataset acquisition and validation utilities."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = ("id", "title", "text")


def demo_documents(count: int = 50) -> list[dict[str, str]]:
    """Create a reproducible corpus for demos and tests."""
    topics = [
        ("rag", "Retrieval augmented generation combines document retrieval with language model generation. A retriever finds relevant passages, and the generator writes an answer grounded in those passages."),
        ("embeddings", "Text embeddings map words, sentences, or documents into vectors. Similar meanings tend to have nearby vectors, enabling semantic search and clustering."),
        ("evaluation", "A retrieval system can be evaluated with precision at k and recall at k. Ground-truth relevant documents make it possible to compare chunking and ranking choices."),
        ("cleaning", "Text cleaning removes markup, normalizes whitespace, and handles missing or extremely long values before documents are split into chunks."),
        ("api", "FastAPI exposes typed HTTP endpoints for health checks, search, and question answering. Structured request logs make latency and failures observable."),
    ]
    documents: list[dict[str, str]] = []
    for index in range(max(0, count)):
        topic, text = topics[index % len(topics)]
        documents.append({
            "id": f"demo-{index + 1:05d}",
            "title": f"{topic.title()} Notes {index + 1}",
            "text": f"{text} This demonstration document belongs to the {topic} topic and contains repeatable reference material for the RAG pipeline.",
            "source": "local-demo",
        })
    return documents


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
            except json.JSONDecodeError:
                records.append({"_line": line_number, "_malformed": True, "text": line})
    return records


def normalize_record(record: dict, index: int) -> dict[str, str]:
    """Map common dataset column names into the project schema."""
    identifier = record.get("id") or record.get("document_id") or record.get("url") or f"record-{index + 1:06d}"
    title = record.get("title") or record.get("name") or "Untitled document"
    text = record.get("text") or record.get("content") or record.get("body") or ""
    return {
        "id": str(identifier),
        "title": str(title),
        "text": str(text),
        "source": str(record.get("source") or record.get("url") or "local"),
    }


def quality_report(records: list[dict]) -> dict:
    normalized = [normalize_record(record, index) for index, record in enumerate(records)]
    missing = Counter()
    duplicate_ids = [item_id for item_id, count in Counter(item["id"] for item in normalized).items() if count > 1]
    malformed = sum(1 for record in records if record.get("_malformed"))
    for item in normalized:
        for field in REQUIRED_FIELDS:
            if not item[field].strip():
                missing[field] += 1
    encoding_issues = 0
    for item in normalized:
        try:
            item["text"].encode("utf-8")
        except UnicodeError:
            encoding_issues += 1
    total = len(normalized)
    return {
        "total_records": total,
        "missing_fields": dict(missing),
        "missing_field_percentage": {field: round((missing[field] / total) * 100, 2) if total else 0.0 for field in REQUIRED_FIELDS},
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_rate_percentage": round((len(duplicate_ids) / total) * 100, 2) if total else 0.0,
        "malformed_json_lines": malformed,
        "encoding_issue_count": encoding_issues,
        "average_text_characters": round(sum(len(item["text"]) for item in normalized) / total, 2) if total else 0.0,
        "empty_text_count": sum(not item["text"].strip() for item in normalized),
    }


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def looks_like_supported_text(value: str) -> bool:
    return bool(re.search(r"\w", value or ""))
