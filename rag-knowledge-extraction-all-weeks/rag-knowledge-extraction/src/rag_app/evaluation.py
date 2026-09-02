"""Retrieval evaluation metrics and evaluation harness."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationCase:
    query: str
    relevant_ids: set[str]


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    selected = retrieved_ids[:k]
    return sum(item in relevant_ids for item in selected) / len(selected) if selected else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return sum(item in relevant_ids for item in retrieved_ids[:k]) / len(relevant_ids)


def evaluate(service, cases: list[EvaluationCase], k: int = 5) -> dict:
    rows = []
    for case in cases:
        result = service.search(case.query, top_k=k)
        ids = [str(item.get("id")) for item in result["results"]]
        rows.append({
            "query": case.query,
            "relevant_ids": sorted(case.relevant_ids),
            "retrieved_ids": ids,
            "precision_at_k": round(precision_at_k(ids, case.relevant_ids, k), 4),
            "recall_at_k": round(recall_at_k(ids, case.relevant_ids, k), 4),
            "latency_seconds": result["retrieval_latency_seconds"],
        })
    return {
        "case_count": len(rows),
        "k": k,
        "mean_precision_at_k": round(sum(row["precision_at_k"] for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_recall_at_k": round(sum(row["recall_at_k"] for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_latency_seconds": round(sum(row["latency_seconds"] for row in rows) / len(rows), 6) if rows else 0.0,
        "cases": rows,
    }
