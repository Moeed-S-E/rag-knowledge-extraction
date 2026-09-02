# 12-Week Roadmap Coverage

This project is an all-weeks implementation of the supplied Parallax Labs AI/ML Engineer Intern Roadmap. It is intentionally packaged as a standalone Python project and does not require a GitHub repository.

| Week | Requirement | Deliverable in this project | Main files |
| --- | --- | --- | --- |
| 1 | Environment verification, dataset acquisition, validation, data quality report | Reproducible local dataset flow with JSONL schema checks and a quality report | `scripts/verify_environment.py`, `scripts/acquire_dataset.py`, `rag_app.data` |
| 2 | Cleaning, edge cases, spaCy subset option, unit tests, dropped-data logging | Markup removal, Unicode normalization, whitespace handling, length limits, mixed-language preservation, and tests | `rag_app.cleaning`, `tests/test_cleaning.py`, `scripts/clean_dataset.py` |
| 3 | Chunking, embeddings, timing, strategy documentation | Recursive chunking with overlap, Sentence Transformers preference, deterministic fallback, timing metadata | `rag_app.chunking`, `rag_app.embeddings`, `scripts/build_index.py` |
| 4 | ChromaDB, semantic search, latency tests, empty/malformed cases | ChromaDB adapter with local NumPy/JSON fallback and safe empty-query behavior | `rag_app.vector_store`, `rag_app.rag` |
| 5 | 20-query retrieval evaluation, Precision@K, Recall@K, tuning | Evaluation utilities and reproducible labeled retrieval benchmark | `rag_app.evaluation`, `scripts/evaluate_retrieval.py` |
| 6 | LLM integration, prompting, errors, latency, CLI | OpenAI-compatible client, JSON output prompt, retries, timeouts, CLI, structured logs | `rag_app.llm`, `rag_app.cli`, `rag_app.rag` |
| 7 | Hallucination mitigation, refusal, source citations, structured output | Citation filtering, lexical support score, hallucination flag, off-topic evaluation | `scripts/evaluate_hallucination.py`, `rag_app.rag` |
| 8 | Topic modeling, visualization, manual review, topic-filtered retrieval | Corpus topics, distribution plot, sample documents per topic, topic metadata and filters | `rag_app.analysis`, `scripts/analyze_corpus.py` |
| 9 | Sentiment or NER, evaluation, metadata enrichment, advanced retrieval | Rule-based sentiment and entity extraction stored on chunks, entity filters, sentiment boosting | `rag_app.analysis`, `rag_app.rag`, `rag_app.vector_store` |
| 10 | FastAPI, query/metadata/health endpoints, logs, HTTP errors, concurrency-ready service | Typed API with `/health`, `/metadata`, `/search`, `/query`, validation and structured request logging | `rag_app.api` |
| 11 | 30 Q&A evaluation, retrieval/generation/latency report, limitations, API tests, typing/docstrings | Automated 30-case evaluation and endpoint/unit test suite | `scripts/evaluate_system.py`, `tests/` |
| 12 | README, architecture, setup, examples, benchmarks, demo, presentation | Comprehensive README, demo guide, final report, presentation outline, generated artifacts | `README.md`, `docs/DEMO_GUIDE.md`, `docs/FINAL_REPORT.md`, `docs/PRESENTATION_OUTLINE.md` |

## Full execution sequence

```bash
export PYTHONPATH="$PWD/src"
python scripts/verify_environment.py
python scripts/acquire_dataset.py --count 50
python scripts/clean_dataset.py
python scripts/build_index.py
python scripts/evaluate_retrieval.py
python scripts/evaluate_hallucination.py
python scripts/analyze_corpus.py
python scripts/evaluate_system.py
pytest -q
```

The demo corpus uses 50 local records by default. For the Week 1 scale requirement, provide a real JSONL export with at least 5,000 records using `scripts/acquire_dataset.py --input your_file.jsonl --count 5000`.
