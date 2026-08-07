# Week 3: Chunking Strategies & Sentence Embeddings

## Objectives

- Implement chunking strategies: Fixed-size, Recursive Character, and Semantic Sentence chunking.
- Handle edge cases in chunking (very short text, text without clear sentence boundaries, empty input).
- Integrate `sentence-transformers` (`all-MiniLM-L6-v2`) to generate dense 384-dimensional embeddings.
- Benchmark embedding generation time per chunk and calculate projected full dataset indexing duration.
- Document chunking strategy decisions and model selection rationale.

## Folder Structure

- `src/ragkit/chunking/strategies.py`: Implements `fixed_size_chunk`, `recursive_character_chunk`, and `semantic_sentence_chunk`.
- `src/ragkit/embeddings/encoder.py`: `Encoder` class for embedding generation and indexing benchmarking.
- `tests/test_chunking.py`: Unit tests for chunking edge cases.
- `Week-3/scripts/run_chunking_embeddings.py`: Execution pipeline that chunks cleaned data, computes embeddings, logs timing, and exports results.
- `Week-3/reports/chunking_embedding_report.md`: Detailed benchmark report and strategy decision documentation.

## How to Run

```bash
# Run unit tests
uv run python -m unittest discover -s tests

# Run chunking & embedding pipeline
uv run python Week-3/scripts/run_chunking_embeddings.py
```

## Benchmarks & Model Selection

- **Selected Strategy**: Recursive Character Chunking (`chunk_size=300`, `overlap=30`)
- **Selected Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-D)
- **Avg Inference Time**: ~8.6 ms per chunk
- **Throughput**: ~115.8 chunks/sec
- **Projected 50,000 Chunks Indexing Time**: ~431.8 seconds (~7.2 minutes)
