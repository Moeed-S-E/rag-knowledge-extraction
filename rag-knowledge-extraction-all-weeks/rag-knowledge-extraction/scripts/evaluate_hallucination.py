"""Evaluate citation grounding and off-topic refusal behavior."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_app.config import get_settings
from rag_app.data import read_jsonl
from rag_app.rag import RAGService


def main() -> int:
    service = RAGService(get_settings())
    records = read_jsonl(service.settings.clean_data)
    if not service.store.local.chunks:
        service.build(records)
    cases = [
        {"query": "What is retrieval augmented generation?", "grounded": True},
        {"query": "What are text embeddings used for?", "grounded": True},
        {"query": "How is retrieval quality measured?", "grounded": True},
        {"query": "What is the capital of France and today's weather?", "grounded": False},
        {"query": "Who won the latest World Cup?", "grounded": False},
        {"query": "Give me medical advice for a serious illness.", "grounded": False},
    ]
    rows = []
    for case in cases:
        result = service.query(case["query"])
        refusal = "don't know" in result["answer"].lower() or not result["citations"]
        passed = bool(result["citations"]) and result["support_score"] >= 0.25 if case["grounded"] else refusal
        rows.append({
            "query": case["query"],
            "expected_grounded": case["grounded"],
            "citations": result["citations"],
            "support_score": result["support_score"],
            "hallucination_flag": result["hallucination_flag"],
            "refusal_detected": refusal,
            "passed": passed,
        })
    report = {
        "case_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "pass_rate": round(sum(row["passed"] for row in rows) / len(rows), 4) if rows else 0.0,
        "cases": rows,
        "notes": "Lexical support and citation checks are guardrails, not formal factuality proofs.",
    }
    Path("artifacts/hallucination_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
