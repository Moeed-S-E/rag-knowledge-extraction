"""End-to-end retrieval-augmented generation service."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .analysis import entities, sentiment, top_terms
from .chunking import Chunk, chunk_documents
from .config import Settings, get_settings
from .embeddings import EmbeddingProvider
from .llm import LLMClient
from .vector_store import VectorStore


_WORDS = re.compile(r"[\w\u0080-\uffff]+", re.UNICODE)


class RAGService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = EmbeddingProvider()
        self.store = VectorStore(self.settings.index_dir)
        self.llm = LLMClient(self.settings.llm_api_key, self.settings.llm_base_url, self.settings.llm_model, self.settings.llm_timeout_seconds)
        self.chunks: list[Chunk] = []
        self.ready = bool(self.store.local.chunks)

    def build(self, records: list[dict], reset: bool = True) -> dict:
        chunks = chunk_documents(records, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("No non-empty chunks were produced")
        for chunk in chunks:
            chunk_terms = top_terms([chunk.text], limit=1)
            chunk.topic = chunk_terms[0]["term"] if chunk_terms else "uncategorized"
            chunk_sentiment = sentiment(chunk.text)
            chunk.sentiment_label = chunk_sentiment["label"]
            chunk.sentiment_score = chunk_sentiment["score"]
            chunk.entities = entities(chunk.text)
        self.embedder.fit([chunk.text for chunk in chunks])
        embedded = self.embedder.encode([chunk.text for chunk in chunks])
        if reset:
            self.store.reset()
        self.store.add(chunks, embedded.vectors)
        self.chunks = chunks
        self.ready = True
        return {
            "documents": len(records),
            "chunks": len(chunks),
            "embedding_provider": embedded.provider,
            "embedding_seconds": round(embedded.elapsed_seconds, 4),
            "estimated_seconds_per_chunk": round(embedded.elapsed_seconds / len(chunks), 6),
            "vector_store": self.store.backend,
        }

    def _ensure_loaded(self) -> None:
        if not self.ready:
            self.chunks = [Chunk(**item) for item in self.store.local.chunks]
            self.ready = bool(self.chunks)

    def search(self, query: str, top_k: int | None = None, topic: str | None = None, entity: str | None = None, positive_boost: bool = False) -> dict:
        query = (query or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        self._ensure_loaded()
        if not self.ready:
            return {"query": query, "results": [], "retrieval_latency_seconds": 0.0}
        vector = self.embedder.encode([query]).vectors[0]
        requested_k = top_k or self.settings.top_k
        candidate_k = min(max(requested_k * 5, requested_k), 50)
        results, latency = self.store.search(vector, candidate_k)
        if topic:
            results = [item for item in results if str(item.get("topic", "")).lower() == topic.lower()]
        if entity:
            results = [item for item in results if entity.lower() in {str(value).lower() for value in item.get("entities", [])}]
        if positive_boost:
            results.sort(key=lambda item: float(item.get("score", 0.0)) + (0.05 if item.get("sentiment_label") == "positive" else 0.0), reverse=True)
        return {"query": query, "results": results[:requested_k], "retrieval_latency_seconds": round(latency, 6), "filters": {"topic": topic, "entity": entity, "positive_boost": positive_boost}}

    @staticmethod
    def _support_score(answer: str, context: list[dict]) -> float:
        answer_words = {word.lower() for word in _WORDS.findall(answer) if len(word) > 2}
        context_words = {word.lower() for item in context for word in _WORDS.findall(item.get("text", ""))}
        if not answer_words:
            return 0.0
        return round(len(answer_words & context_words) / len(answer_words), 4)

    def query(self, question: str, top_k: int | None = None, topic: str | None = None, entity: str | None = None, positive_boost: bool = False) -> dict:
        started = time.perf_counter()
        retrieval = self.search(question, top_k=top_k, topic=topic, entity=entity, positive_boost=positive_boost)
        context = retrieval["results"]
        generation = self.llm.generate(question, context)
        if generation.answer:
            answer = generation.answer
            citations = [citation for citation in generation.citations if any(item.get("id") == citation for item in context)]
            provider = generation.provider
            generation_error = generation.error
        else:
            if context:
                best = context[0]
                answer = f"Based on {best.get('title', 'the retrieved source')}: {best.get('text', '')}"
                citations = [str(best.get("id"))]
            else:
                answer = "I don't know based on the indexed documents."
                citations = []
            provider = "offline-extractive-fallback"
            generation_error = generation.error
        support_score = self._support_score(answer, context)
        unsupported = support_score < 0.25 if context else True
        result = {
            "query": question,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": context,
            "support_score": support_score,
            "hallucination_flag": unsupported,
            "provider": provider,
            "retrieval_latency_seconds": retrieval["retrieval_latency_seconds"],
            "generation_latency_seconds": round(generation.latency_seconds, 6),
            "total_latency_seconds": round(time.perf_counter() - started, 6),
            "generation_error": generation_error,
        }
        self._log(result)
        return result

    def _log(self, result: dict) -> None:
        self.settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    def metadata(self) -> dict:
        self._ensure_loaded()
        return {
            "ready": self.ready,
            "chunk_count": len(self.chunks),
            "vector_store": self.store.backend,
            "embedding_provider": self.embedder.provider,
            "llm_enabled": self.llm.enabled,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
        }
