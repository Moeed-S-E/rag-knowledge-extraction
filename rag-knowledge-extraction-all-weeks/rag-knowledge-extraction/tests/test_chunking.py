import pytest

from rag_app.chunking import chunk_text


def test_short_text_stays_in_one_chunk():
    assert chunk_text("short text", chunk_size=50, overlap=5) == ["short text"]


def test_long_text_without_sentence_boundaries_is_split():
    chunks = chunk_text("x" * 125, chunk_size=50, overlap=0)
    assert len(chunks) == 3
    assert all(chunk for chunk in chunks)


def test_overlap_repeats_previous_tail():
    chunks = chunk_text("alpha beta gamma delta epsilon", chunk_size=18, overlap=5)
    assert len(chunks) > 1
    assert any("delta" in chunk or "gamma" in chunk for chunk in chunks[1:])


def test_invalid_chunk_configuration_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=10, overlap=10)
