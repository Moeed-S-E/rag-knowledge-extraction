"""Embedding providers with an offline deterministic fallback."""
from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np


_TOKEN = re.compile(r"[\w\u0080-\uffff]+", re.UNICODE)


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    provider: str
    elapsed_seconds: float


class EmbeddingProvider:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimensions: int = 384):
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = None
        self._idf: dict[str, float] = {}
        self._vocabulary: dict[str, int] = {}
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self.provider = f"sentence-transformers:{model_name}"
        except Exception:
            self.provider = "tfidf-hashing-fallback"

    def fit(self, texts: list[str]) -> None:
        if self._model is not None:
            return
        document_count = max(1, len(texts))
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(_TOKEN.findall(text.lower())))
        self._idf = {token: float(np.log((1 + document_count) / (1 + frequency)) + 1.0) for token, frequency in document_frequency.items()}
        self._vocabulary = {token: index for index, token in enumerate(sorted(self._idf))}

    def encode(self, texts: list[str]) -> EmbeddingResult:
        started = time.perf_counter()
        if not texts:
            return EmbeddingResult(np.empty((0, self.dimensions), dtype=np.float32), self.provider, 0.0)
        if self._model is not None:
            vectors = np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        else:
            if not self._idf:
                self.fit(texts)
            vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
            for row, text in enumerate(texts):
                counts = Counter(_TOKEN.findall(text.lower()))
                for token, count in counts.items():
                    digest = hashlib.sha256(token.encode("utf-8")).digest()
                    index = int.from_bytes(digest[:8], "big") % self.dimensions
                    vectors[row, index] += float(count) * self._idf.get(token, 1.0)
                norm = np.linalg.norm(vectors[row])
                if norm:
                    vectors[row] /= norm
        return EmbeddingResult(vectors, self.provider, time.perf_counter() - started)
