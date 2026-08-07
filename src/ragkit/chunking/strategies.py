"""Text chunking strategies: Fixed-size, Recursive Character, and Sentence-based Semantic chunking."""

from __future__ import annotations

import re
from typing import List, Optional

# Default separators for recursive character splitting
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def fixed_size_chunk(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> List[str]:
    """
    Split text into fixed character length chunks with specified overlap.

    Args:
        text: Input text string.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


def recursive_character_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Recursively split text by primary separators (paragraphs, newlines, sentences, spaces)
    to keep semantic blocks intact while respecting chunk_size.

    Args:
        text: Input text string.
        chunk_size: Maximum character limit per chunk.
        overlap: Overlap characters between chunks.
        separators: Hierarchical list of separator strings.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    if separators is None:
        separators = DEFAULT_SEPARATORS

    # Select appropriate separator
    separator = separators[-1]
    new_separators = []

    for i, sep in enumerate(separators):
        if sep == "":
            separator = ""
            break
        if sep in text:
            separator = sep
            new_separators = separators[i + 1 :]
            break

    # Split text using chosen separator
    if separator != "":
        splits = text.split(separator)
    else:
        splits = list(text)

    final_chunks = []
    current_chunk: List[str] = []
    current_len = 0

    for split in splits:
        piece = split if separator == "" else split + separator
        piece_len = len(piece)

        if piece_len > chunk_size:
            # If a single piece exceeds chunk_size, recurse with finer separators
            if current_chunk:
                merged = "".join(current_chunk).strip()
                if merged:
                    final_chunks.append(merged)
                current_chunk = []
                current_len = 0

            if new_separators:
                sub_chunks = recursive_character_chunk(
                    split, chunk_size, overlap, new_separators
                )
                final_chunks.extend(sub_chunks)
            else:
                sub_chunks = fixed_size_chunk(split, chunk_size, overlap)
                final_chunks.extend(sub_chunks)
            continue

        if current_len + piece_len > chunk_size:
            merged = "".join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

            # Handle overlap
            if overlap > 0 and current_chunk:
                overlap_buffer = ""
                for p in reversed(current_chunk):
                    if len(overlap_buffer) + len(p) <= overlap:
                        overlap_buffer = p + overlap_buffer
                    else:
                        break
                current_chunk = [overlap_buffer, piece] if overlap_buffer else [piece]
                current_len = sum(len(p) for p in current_chunk)
            else:
                current_chunk = [piece]
                current_len = piece_len
        else:
            current_chunk.append(piece)
            current_len += piece_len

    if current_chunk:
        merged = "".join(current_chunk).strip()
        if merged:
            final_chunks.append(merged)

    return final_chunks if final_chunks else [text]


def semantic_sentence_chunk(
    text: str, max_chunk_size: int = 500, overlap_sentences: int = 1
) -> List[str]:
    """
    Split text into sentence-aware semantic chunks. Handles edge cases like text
    without sentence punctuation gracefully.

    Args:
        text: Input text string.
        max_chunk_size: Maximum character threshold per chunk.
        overlap_sentences: Number of trailing sentences to include in next chunk.

    Returns:
        List of semantically coherent text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= max_chunk_size:
        return [text]

    # Sentence boundary regex (handles . ! ? followed by space/newline)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Edge case: text without clear sentence boundaries
    if len(sentences) <= 1:
        return recursive_character_chunk(text, chunk_size=max_chunk_size)

    chunks = []
    current_sentences: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # Handle sentence larger than max_chunk_size
        if sent_len > max_chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_len = 0

            sub_chunks = recursive_character_chunk(sent, chunk_size=max_chunk_size)
            chunks.extend(sub_chunks)
            continue

        if current_len + sent_len + 1 > max_chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))

            if overlap_sentences > 0 and len(current_sentences) >= overlap_sentences:
                overlap_sents = current_sentences[-overlap_sentences:]
                current_sentences = overlap_sents + [sent]
                current_len = sum(len(s) + 1 for s in current_sentences)
            else:
                current_sentences = [sent]
                current_len = sent_len
        else:
            current_sentences.append(sent)
            current_len += sent_len + 1

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
