# Week 3: Chunking Strategy & Embedding Benchmark Report

## 1. Executive Summary

- **Primary Chunking Strategy**: **Recursive Character Chunking** (`chunk_size=300`, `overlap=30`)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dense Embedding Dimension**: `384`
- **Avg Encoding Speed per Chunk**: `8.637 ms`
- **Encoding Throughput**: `115.78 chunks/second`
- **Projected 50,000 Chunk Indexing Time**: `431.86 sec` (~`7.2 minutes`)

---

## 2. Chunking Strategy Rationale & Decision

### Evaluated Strategies:
1. **Fixed-Size Chunking**: Simple character slicing.
   - *Drawback*: Cuts words and sentences in half, causing context fragmentation and loss of semantic meaning at boundaries.
2. **Recursive Character Chunking** (Chosen Primary Strategy):
   - *Rationale*: Recursively splits by structural boundaries (`\n\n`, `\n`, `. `, ` `, `""`).
   - Keeps paragraphs and sentences intact while guaranteeing strict upper bounds on token window sizes.
3. **Semantic Sentence Chunking**:
   - *Rationale*: Groups full sentences together until maximum character limit is reached.
   - Ideal for structured prose; falls back to recursive splitting for unpunctuated text walls.

---

## 3. Embedding Model Selection Rationale

Selected Model: **`sentence-transformers/all-MiniLM-L6-v2`**

### Why this model?
- **Speed & Efficiency**: Lightweight 6-layer Transformer architecture with extremely fast inference (~8.637 ms/chunk on CPU).
- **Optimal Vector Dimension (384-D)**: Provides a high quality-to-memory ratio compared to 768-D or 1536-D models, drastically reducing vector store RAM footprint in ChromaDB.
- **Strong Benchmark Performance**: Trained specifically on sentence-pair datasets using contrastive learning, excelling at semantic search, cosine distance matching, and retrieval tasks.

---

## 4. Indexing Performance Benchmarks

| Metric | Measured Benchmark Value |
|---|---|
| **Model Architecture** | `all-MiniLM-L6-v2` |
| **Vector Dimension** | `384` |
| **Processed Sample Chunks** | `6` |
| **Batch Time** | `0.0518 s` |
| **Time per Chunk** | `8.637 ms` |
| **Throughput** | `115.78 chunks/sec` |
| **Projected 50,000 Chunk Index Time** | **`431.86 s` (~`7.2 min`)** |

---

## Status
✅ Chunking strategy implemented, benchmarked, and ready for vector storage indexing in ChromaDB.
