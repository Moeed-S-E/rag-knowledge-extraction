"""Unit tests for text chunking strategies and edge cases."""

import unittest
from ragkit.chunking.strategies import (
    fixed_size_chunk,
    recursive_character_chunk,
    semantic_sentence_chunk,
)


class TestChunkingStrategies(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(fixed_size_chunk(""), [])
        self.assertEqual(recursive_character_chunk("   "), [])
        self.assertEqual(semantic_sentence_chunk(None), [])

    def test_very_short_text(self):
        short_text = "Short text example."
        self.assertEqual(fixed_size_chunk(short_text, chunk_size=100), [short_text])
        self.assertEqual(
            recursive_character_chunk(short_text, chunk_size=100), [short_text]
        )
        self.assertEqual(
            semantic_sentence_chunk(short_text, max_chunk_size=100), [short_text]
        )

    def test_text_without_clear_sentence_boundaries(self):
        # Continuous long string without periods, newlines, or sentence punctuation
        no_punctuation_text = "word" * 200  # 800 characters without punctuation
        chunks_rec = recursive_character_chunk(no_punctuation_text, chunk_size=200)
        chunks_sem = semantic_sentence_chunk(no_punctuation_text, max_chunk_size=200)

        self.assertGreater(len(chunks_rec), 1)
        self.assertGreater(len(chunks_sem), 1)
        for chunk in chunks_rec:
            self.assertLessEqual(len(chunk), 250)
        for chunk in chunks_sem:
            self.assertLessEqual(len(chunk), 250)

    def test_recursive_character_chunk_overlap(self):
        text = (
            "Paragraph one has detailed explanation of AI.\n\n"
            "Paragraph two delves into vector databases and similarity metrics.\n\n"
            "Paragraph three covers large language model embeddings."
        )
        chunks = recursive_character_chunk(text, chunk_size=80, overlap=20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 120)

    def test_semantic_sentence_chunking(self):
        text = (
            "Sentence one introduces RAG systems. Sentence two discusses vector databases. "
            "Sentence three explains dense retrieval mechanics. Sentence four covers LLM prompt augmented generation."
        )
        chunks = semantic_sentence_chunk(text, max_chunk_size=100, overlap_sentences=1)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(any(s in c for s in ["Sentence", "RAG", "vector", "LLM"]))


if __name__ == "__main__":
    unittest.main()
