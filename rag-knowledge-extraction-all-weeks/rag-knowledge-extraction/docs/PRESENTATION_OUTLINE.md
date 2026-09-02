# Final Presentation Outline

## Slide 1 — Project title

**RAG Knowledge Extraction System**. An offline-first Python pipeline for grounded knowledge retrieval and question answering.

## Slide 2 — Problem and motivation

Domain-specific questions require reliable access to a document corpus. A retrieval-augmented system can provide answers tied to source chunks instead of relying only on model memory.

## Slide 3 — End-to-end architecture

Raw documents flow through validation, cleaning, chunking, embeddings, vector search, retrieval, generation, support checks, and delivery through CLI or FastAPI.

## Slide 4 — Data engineering

The system accepts normalized JSONL records, reports missing and duplicate data, handles malformed input, and emits cleaned records with a measured drop rate.

## Slide 5 — Chunking and embeddings

Recursive splitting favors readable boundaries and overlap. Sentence Transformers is the enhanced provider, while a deterministic local fallback keeps the project reproducible.

## Slide 6 — Vector retrieval

ChromaDB is supported for persistent vector search. A NumPy/JSON store provides a portable fallback. Search returns chunk IDs, metadata, similarity scores, and latency.

## Slide 7 — Evaluation

Precision@K, Recall@K, retrieval latency, citation rate, lexical support, and generation quality are written to machine-readable artifacts for repeatable comparison.

## Slide 8 — Hallucination mitigation

The prompt asks the optional LLM to use only supplied context, cite chunk IDs, and say it does not know when context is insufficient. A post-generation support check flags weakly grounded answers.

## Slide 9 — NLP enrichment

Topic keywords, sentiment labels, and named entities are stored with chunks. Retrieval can filter by topic/entity or boost positive-sentiment chunks.

## Slide 10 — API and CLI

FastAPI exposes health, metadata, search, and query endpoints. The CLI supports index building, metadata inspection, retrieval, and cited answers.

## Slide 11 — Testing and limitations

Unit tests cover cleaning, chunking, service behavior, API validation, and empty or malformed inputs. Limitations include demo-corpus size, heuristic support scoring, and optional-model resource requirements.

## Slide 12 — Lessons and next steps

The next production steps are to ingest a real 5,000+ document corpus, add stronger human-labeled evaluation, benchmark under concurrency, add authentication and rate limiting, and monitor provider costs and quality.
