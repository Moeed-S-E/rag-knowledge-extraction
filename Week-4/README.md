# Week 4: Embedding & Vector Database Ingestion

## Objectives

- Wrap an embedding model (sentence-transformers)
- Embed all chunks and ingest into ChromaDB
- Benchmark embedding throughput and index size
- Implement top-K semantic search
- Handle edge cases (empty DB, malformed query)

## Approach

This week, we focused on setting up a local ChromaDB instance to store the text chunks generated in Week 3, along with their dense vector embeddings. 

**Key Implementations:**
- **`ChromaStore` (`src/ragkit/vectorstore/chroma_client.py`)**: A class that wraps `chromadb.PersistentClient`, managing the initialization, collection creation (`get_or_create_collection`), document ingestion (`add_chunks`), and semantic search (`search`).
- **Semantic Search Benchmark (`Week-4/scripts/run_vector_db_search.py`)**: A script that loads the chunks from Week 3, re-computes full embeddings using `sentence-transformers/all-MiniLM-L6-v2`, ingests them into a `chromadb` collection, and benchmarks retrieval latency across 10 diverse test queries.
- **Edge Cases Handled**: Added robust checks for when a collection is empty, or when a malformed (empty) query is passed to the search function.

## What's in this folder

- `scripts/run_vector_db_search.py` — The main runnable script that performs ingestion and benchmarking.

## How to Run

```bash
# from the project root
uv sync
python Week-4/scripts/run_vector_db_search.py
```

## Results

The benchmarking script successfully ingested the chunks into the ChromaDB collection using `cosine` similarity distance space. The retrieval speeds are highly optimal, taking just a few milliseconds per query for top-3 retrieval. Edge cases (such as querying an empty database or malformed queries) gracefully output warnings and return empty result sets without crashing the application.

## Notes / Known Issues

- ChromaDB's telemetry has been disabled to prevent unnecessary warnings or background network requests during development.
- Currently, embeddings are generated inline during ingestion for simplicity. For massive datasets, this should ideally be separated into a pre-computation pipeline to avoid redundant re-encoding.
