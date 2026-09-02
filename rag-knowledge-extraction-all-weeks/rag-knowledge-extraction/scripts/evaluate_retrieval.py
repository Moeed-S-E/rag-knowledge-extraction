"""Evaluate retrieval on labeled queries."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_app.config import get_settings
from rag_app.data import read_jsonl
from rag_app.evaluation import EvaluationCase, evaluate
from rag_app.rag import RAGService


def main() -> int:
    service = RAGService(get_settings())
    records = read_jsonl(service.settings.clean_data)
    if not service.store.local.chunks:
        service.build(records)
    cases = []
    topic_to_id = {}
    for record in records:
        title = str(record.get("title", ""))
        topic_to_id.setdefault(title.split()[0].lower(), record.get("id"))
    queries = [
        ("How does retrieval augmented generation work?", "rag"),
        ("What are text embeddings used for?", "embeddings"),
        ("How do we measure retrieval quality?", "evaluation"),
        ("How should text be cleaned?", "cleaning"),
        ("What does FastAPI expose?", "api"),
    ]
    for query, topic in queries:
        if topic in topic_to_id:
            cases.append(EvaluationCase(query, {f"{topic_to_id[topic]}-chunk-0000"}))
    report = evaluate(service, cases, k=5)
    Path("artifacts/retrieval_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
