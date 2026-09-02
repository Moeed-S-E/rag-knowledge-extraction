# Demo Guide

This guide is designed for a two-to-three-minute project demonstration.

## 1. Prepare the system

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## 2. Build the local index

```bash
python scripts/acquire_dataset.py --count 50
python scripts/clean_dataset.py
python scripts/build_index.py
```

Explain that the pipeline validates raw records, cleans text, splits documents into overlapping chunks, enriches chunks with topic/sentiment/entity metadata, embeds them, and persists the index.

## 3. Demonstrate the CLI

```bash
python -m rag_app.cli metadata
python -m rag_app.cli search "What are text embeddings used for?"
python -m rag_app.cli query "How does retrieval augmented generation work?"
python -m rag_app.cli query "Who won the latest World Cup?"
```

The grounded answer includes a chunk citation. The off-topic query returns a conservative response when the indexed context does not provide a reliable answer.

## 4. Demonstrate the API

In one terminal:

```bash
uvicorn rag_app.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata
curl -X POST http://127.0.0.1:8000/search \\
  -H 'Content-Type: application/json' \\
  -d '{"query":"What is retrieval augmented generation?","top_k":3,"topic":"retrieval"}'
curl -X POST http://127.0.0.1:8000/query \\
  -H 'Content-Type: application/json' \\
  -d '{"query":"What does FastAPI expose?","top_k":3}'
```

Show the `retrieved_chunks`, `citations`, `support_score`, and latency fields in the response. Request logs are appended to `logs/rag_requests.jsonl`.

## 5. Demonstrate evaluation

```bash
python scripts/evaluate_retrieval.py
python scripts/evaluate_hallucination.py
python scripts/analyze_corpus.py
python scripts/evaluate_system.py
pytest -q
```

The output artifacts are JSON reports in `artifacts/`. The topic distribution is saved as `artifacts/topic_distribution.png` when Matplotlib is installed.

## Suggested spoken narrative

The system is offline-first: it can search and answer locally without an API key. For a stronger production configuration, Sentence Transformers and ChromaDB are used when installed, and an OpenAI-compatible provider can generate structured, cited answers. The service still handles missing dependencies, empty indexes, malformed queries, provider timeouts, rate limits, and unsupported questions without failing silently.
