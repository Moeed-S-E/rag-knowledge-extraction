"""Vector store abstraction with optional ChromaDB support."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .chunking import Chunk


class LocalVectorStore:
    """A small persistent cosine-similarity store used when ChromaDB is absent."""
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path = directory / "chunks.json"
        self.vectors_path = directory / "vectors.npy"
        self.chunks: list[dict] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if self.metadata_path.exists():
            self.chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.vectors_path.exists():
            self.vectors = np.load(self.vectors_path)

    def reset(self) -> None:
        self.chunks = []
        self.vectors = np.empty((0, 0), dtype=np.float32)
        for path in (self.metadata_path, self.vectors_path):
            if path.exists():
                path.unlink()

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = [chunk.to_dict() for chunk in chunks]
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.metadata_path.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(self.vectors_path, self.vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.vectors.size == 0 or not self.chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.size != self.vectors.shape[1]:
            raise ValueError("query vector dimension does not match the index")
        query_norm = np.linalg.norm(query)
        matrix_norm = np.linalg.norm(self.vectors, axis=1)
        scores = (self.vectors @ query) / np.maximum(matrix_norm * query_norm, 1e-12)
        indices = np.argsort(-scores)[:top_k]
        return [{**self.chunks[int(index)], "score": float(scores[int(index)])} for index in indices]


class VectorStore:
    def __init__(self, directory: Path, collection_name: str = "rag_chunks"):
        self.directory = directory
        self.collection_name = collection_name
        self.backend = "local"
        self.local = LocalVectorStore(directory)
        self.collection = None
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=str(directory / "chroma"))
            self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
            self.backend = "chromadb"
        except Exception:
            self.client = None

    def reset(self) -> None:
        self.local.reset()
        if self.collection is not None:
            try:
                ids = self.collection.get().get("ids", [])
                if ids:
                    self.collection.delete(ids=ids)
            except Exception:
                pass

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.local.add(chunks, vectors)
        if self.collection is not None and chunks:
            self.collection.upsert(
                ids=[chunk.id for chunk in chunks],
                embeddings=np.asarray(vectors).tolist(),
                documents=[chunk.text for chunk in chunks],
                metadatas=[{"document_id": chunk.document_id, "title": chunk.title, "source": chunk.source, "position": chunk.position, "topic": chunk.topic, "sentiment_label": chunk.sentiment_label, "sentiment_score": chunk.sentiment_score, "entities": ",".join(chunk.entities)} for chunk in chunks],
            )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[list[dict], float]:
        started = time.perf_counter()
        if self.collection is not None:
            try:
                if self.collection.count() == 0:
                    return [], time.perf_counter() - started
                result = self.collection.query(query_embeddings=[np.asarray(query_vector).tolist()], n_results=top_k, include=["documents", "metadatas", "distances"])
                rows = []
                for index, text in enumerate(result["documents"][0]):
                    metadata = result["metadatas"][0][index]
                    metadata["entities"] = [value for value in str(metadata.get("entities", "")).split(",") if value]
                    rows.append({"id": result["ids"][0][index], "text": text, **metadata, "score": 1.0 - float(result["distances"][0][index])})
                return rows, time.perf_counter() - started
            except Exception:
                pass
        rows = self.local.search(query_vector, top_k=top_k)
        return rows, time.perf_counter() - started
