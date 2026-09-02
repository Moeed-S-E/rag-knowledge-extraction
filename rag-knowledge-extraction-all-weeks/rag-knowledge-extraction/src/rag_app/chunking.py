"""Deterministic text chunking utilities."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    id: str
    document_id: str
    title: str
    text: str
    source: str
    position: int
    topic: str = ""
    sentiment_label: str = ""
    sentiment_score: int = 0
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _split_recursive(text: str, max_length: int) -> list[str]:
    if len(text) <= max_length:
        return [text.strip()] if text.strip() else []
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    for separator in separators:
        pieces = [piece.strip() for piece in text.split(separator) if piece.strip()]
        if len(pieces) > 1 and max(len(piece) for piece in pieces) <= max_length * 1.5:
            chunks: list[str] = []
            current = ""
            for piece in pieces:
                candidate = f"{current}{separator}{piece}" if current else piece
                if len(candidate) <= max_length:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    current = piece
            if current:
                chunks.append(current.strip())
            return [chunk for chunk in chunks if chunk]
    return [text[start : start + max_length].strip() for start in range(0, len(text), max_length) if text[start : start + max_length].strip()]


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = re.sub(r"\s+", " ", text or "").strip()
    base = _split_recursive(text, chunk_size)
    if not base:
        return []
    if overlap == 0:
        return base
    output: list[str] = []
    for index, item in enumerate(base):
        if index == 0:
            output.append(item)
            continue
        prefix = base[index - 1][-overlap:]
        output.append(f"{prefix} {item}".strip())
    return output


def chunk_documents(records: list[dict], chunk_size: int = 700, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in records:
        pieces = chunk_text(record.get("text", ""), chunk_size=chunk_size, overlap=overlap)
        for position, piece in enumerate(pieces):
            chunks.append(Chunk(
                id=f"{record.get('id', 'document')}-chunk-{position:04d}",
                document_id=str(record.get("id", "document")),
                title=str(record.get("title", "Untitled")),
                text=piece,
                source=str(record.get("source", "local")),
                position=position,
            ))
    return chunks
