"""Run a 30-case end-to-end RAG evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_app.config import get_settings
from rag_app.data import read_jsonl
from rag_app.rag import RAGService


CASES = [
    ("How does retrieval augmented generation work?", ["retrieval", "generation"]),
    ("Why does RAG use retrieved passages?", ["retrieved", "passages"]),
    ("What is a retriever?", ["retriever"]),
    ("What is a generator in RAG?", ["generator"]),
    ("How can RAG reduce hallucinations?", ["grounded"]),
    ("What are text embeddings?", ["embeddings", "vectors"]),
    ("Why are nearby vectors useful?", ["similar", "vectors"]),
    ("What can embeddings enable?", ["search", "clustering"]),
    ("How are sentences represented for semantic search?", ["vectors"]),
    ("Which embedding model is preferred in this project?", ["MiniLM"]),
    ("How do we measure retrieval quality?", ["precision", "recall"]),
    ("What is Precision at k?", ["precision"]),
    ("What is Recall at k?", ["recall"]),
    ("Why use ground truth chunks?", ["ground-truth", "relevant"]),
    ("What can be changed during retrieval experiments?", ["chunking", "ranking"]),
    ("Why clean text before indexing?", ["cleaning", "chunks"]),
    ("What does text cleaning remove?", ["markup", "whitespace"]),
    ("How are missing values handled?", ["missing"]),
    ("What happens to extremely long text?", ["long"]),
    ("Does the cleaner preserve non-Latin scripts?", ["scripts"]),
    ("What does FastAPI expose?", ["endpoints"]),
    ("Which API endpoint checks health?", ["health"]),
    ("Why are request logs useful?", ["latency", "failures"]),
    ("How does the API validate requests?", ["typed", "validation"]),
    ("What kind of service wraps the RAG system?", ["FastAPI"]),
    ("What is the purpose of this demonstration corpus?", ["demo", "pipeline"]),
    ("Where are generated reports stored?", ["artifacts"]),
    ("What does the offline fallback use?", ["TF-IDF", "NumPy"]),
    ("What happens when the LLM key is missing?", ["extractive", "fallback"]),
    ("How can another person run the project?", ["README", "requirements"]),
]


def main() -> int:
    service = RAGService(get_settings())
    records = read_jsonl(service.settings.clean_data)
    if not service.store.local.chunks:
        service.build(records)
    rows = []
    for query, expected_terms in CASES:
        started = time.perf_counter()
        result = service.query(query, top_k=3)
        answer = result["answer"].lower()
        matched = sum(term.lower() in answer for term in expected_terms)
        quality = matched / len(expected_terms)
        rows.append({
            "query": query,
            "expected_terms": expected_terms,
            "matched_terms": matched,
            "generation_quality_score": round(quality, 4),
            "has_citation": bool(result["citations"]),
            "support_score": result["support_score"],
            "retrieval_latency_seconds": result["retrieval_latency_seconds"],
            "generation_latency_seconds": result["generation_latency_seconds"],
            "total_latency_seconds": round(time.perf_counter() - started, 6),
        })
    report = {
        "case_count": len(rows),
        "mean_generation_quality": round(sum(row["generation_quality_score"] for row in rows) / len(rows), 4),
        "citation_rate": round(sum(row["has_citation"] for row in rows) / len(rows), 4),
        "mean_support_score": round(sum(row["support_score"] for row in rows) / len(rows), 4),
        "mean_retrieval_latency_seconds": round(sum(row["retrieval_latency_seconds"] for row in rows) / len(rows), 6),
        "mean_generation_latency_seconds": round(sum(row["generation_latency_seconds"] for row in rows) / len(rows), 6),
        "known_limitations": [
            "The included demo corpus is small and synthetic; replace it with a real 5,000+ document corpus for production benchmarking.",
            "The offline generator is extractive and is not equivalent to a hosted language model.",
            "Lexical support checks can miss paraphrases and do not prove factual correctness.",
            "Transformer and ChromaDB quality depends on optional dependencies and available compute.",
        ],
        "cases": rows,
    }
    Path("artifacts/system_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
