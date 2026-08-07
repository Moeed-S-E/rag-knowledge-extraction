"""Sentence-transformers embedding wrapper and indexing benchmark logger."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np


class Encoder:
    """Wrapper around sentence-transformers model for embedding generation and timing benchmarks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate dense vector embeddings for input text or list of text chunks.

        Args:
            texts: Single string or list of text chunk strings.
            batch_size: Batch size for model inference.
            show_progress_bar: Whether to display progress bar.

        Returns:
            Tuple of (numpy array of embeddings, total generation time in seconds).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([]), 0.0

        model = self._load_model()
        start_time = time.perf_counter()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

        elapsed_time = time.perf_counter() - start_time
        return embeddings, elapsed_time

    def benchmark_indexing(
        self, chunks: List[str], total_dataset_chunks: int = 10000
    ) -> Dict[str, Any]:
        """
        Encode a sample batch of chunks, log per-chunk metrics, and project total indexing time.

        Args:
            chunks: List of text chunk strings.
            total_dataset_chunks: Projected total chunk count in full dataset.

        Returns:
            Dictionary containing timing metrics and projected indexing duration.
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "total_time_sec": 0.0,
                "time_per_chunk_ms": 0.0,
                "throughput_chunks_per_sec": 0.0,
                "projected_total_indexing_sec": 0.0,
            }

        embeddings, elapsed = self.encode(chunks)
        num_chunks = len(chunks)

        time_per_chunk_sec = elapsed / num_chunks if num_chunks > 0 else 0.0
        time_per_chunk_ms = round(time_per_chunk_sec * 1000, 3)
        throughput = round(num_chunks / elapsed, 2) if elapsed > 0 else 0.0

        projected_total_sec = round(time_per_chunk_sec * total_dataset_chunks, 2)

        metrics = {
            "model_name": self.model_name,
            "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim > 1 else 0,
            "sample_chunk_count": num_chunks,
            "total_sample_time_sec": round(elapsed, 4),
            "avg_time_per_chunk_ms": time_per_chunk_ms,
            "throughput_chunks_per_sec": throughput,
            "projected_dataset_size": total_dataset_chunks,
            "projected_total_indexing_time_sec": projected_total_sec,
            "projected_total_indexing_time_min": round(projected_total_sec / 60, 2),
        }

        return metrics
