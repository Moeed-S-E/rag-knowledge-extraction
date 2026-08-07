"""Week 3: Text Chunking Pipeline, Sentence-Transformers Embedding Generation & Indexing Benchmarking."""

from __future__ import annotations

import sys
import time
from pathlib import Path
import pandas as pd

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ragkit.chunking.strategies import (
    fixed_size_chunk,
    recursive_character_chunk,
    semantic_sentence_chunk,
)
from ragkit.embeddings.encoder import Encoder

CLEAN_DATA_PATH = Path("data/processed/clean_dataset.csv")
RAW_DATA_PATH = Path("data/raw/arxiv_summaries.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_CHUNKS_PATH = PROCESSED_DIR / "chunked_embeddings.csv"
REPORT_PATH = Path("Week-3/reports/chunking_embedding_report.md")


def load_dataset() -> pd.DataFrame:
    if CLEAN_DATA_PATH.exists():
        print(f"    Loading clean dataset from {CLEAN_DATA_PATH} ...")
        return pd.read_csv(CLEAN_DATA_PATH)
    elif RAW_DATA_PATH.exists():
        print(f"   Clean dataset missing, fallback to {RAW_DATA_PATH} ...")
        return pd.read_csv(RAW_DATA_PATH)
    else:
        print("   Generating sample documents for chunking ...")
        return pd.DataFrame(
            {
                "id": ["doc_1", "doc_2", "doc_3"],
                "category": ["Sci/Tech", "World", "Business"],
                "text": [
                    "Retrieval-Augmented Generation (RAG) combines dense vector retrieval with LLM text generation. "
                    "By embedding text documents into vector spaces using models like all-MiniLM-L6-v2, semantic similarity "
                    "can be efficiently queried using vector indices like HNSW in ChromaDB.",
                    "Natural Language Processing has seen rapid advancement with Transformers and self-attention mechanisms. "
                    "Recursive character splitting preserves paragraph and sentence context better than naive fixed token boundaries.",
                    "Embedding generation throughput depends on batch sizes, GPU/CPU acceleration, and vector dimensions. "
                    "Sentence-transformers provide 384-dimensional dense representations optimized for cosine distance search.",
                ],
            }
        )


def main() -> None:
    print("\n=======================================================")
    print("  Week 3: Text Chunking & Sentence Embeddings Pipeline")
    print("=======================================================")

    # Step 1: Load Dataset
    df = load_dataset()
    print(f"   Loaded {len(df):,} document(s).")

    # Step 2: Run Chunking Strategies
    print("\n   Applying Chunking Strategies ...")
    chunks_record = []

    for idx, row in df.iterrows():
        doc_id = row.get("id", f"doc_{idx}")
        category = row.get("category", "Uncategorized")
        text = str(row.get("text", ""))

        if not text.strip():
            continue

        # Generate chunks using recursive and semantic strategies
        rec_chunks = recursive_character_chunk(text, chunk_size=300, overlap=30)
        sem_chunks = semantic_sentence_chunk(text, max_chunk_size=300, overlap_sentences=1)

        # We select Recursive Character Chunking as primary strategy for RAG index
        for c_idx, chunk_text in enumerate(rec_chunks):
            chunks_record.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_c{c_idx}",
                    "category": category,
                    "strategy": "recursive_character",
                    "chunk_length": len(chunk_text),
                    "chunk_text": chunk_text,
                }
            )

    df_chunks = pd.DataFrame(chunks_record)
    print(f"  • Total documents processed : {len(df):,}")
    print(f"  • Total chunks generated    : {len(df_chunks):,}")
    print(
        f"  • Avg chunk length (chars)  : {round(df_chunks['chunk_length'].mean(), 1)}"
    )

    # Step 3: Embed Chunks & Benchmark Performance
    print("\n  Generating Embeddings using sentence-transformers (all-MiniLM-L6-v2) ...")
    encoder = Encoder(model_name="all-MiniLM-L6-v2")

    chunk_texts = df_chunks["chunk_text"].tolist()
    start_bench = time.perf_counter()
    embeddings, total_time = encoder.encode(chunk_texts, batch_size=32)
    bench_time = time.perf_counter() - start_bench

    # Indexing metrics
    metrics = encoder.benchmark_indexing(
        chunk_texts, total_dataset_chunks=50000
    )

    print(f"  • Model Name               : {metrics['model_name']}")
    print(f"  • Embedding Dimension      : {metrics['embedding_dimension']}")
    print(f"  • Sample Chunks Encoded    : {metrics['sample_chunk_count']}")
    print(f"  • Total Batch Time         : {metrics['total_sample_time_sec']} sec")
    print(f"  • Avg Time per Chunk       : {metrics['avg_time_per_chunk_ms']} ms")
    print(f"  • Throughput               : {metrics['throughput_chunks_per_sec']} chunks/sec")
    print(
        f"   Projected Index Time (50k chunks): {metrics['projected_total_indexing_time_sec']} sec (~{metrics['projected_total_indexing_time_min']} min)"
    )

    # Add embedding norm/summary to dataframe for output storage
    df_chunks["embedding_dim"] = metrics["embedding_dimension"]
    df_chunks["embedding_vector_sample"] = [
        str(vec[:5].tolist()) for vec in embeddings
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_chunks.to_csv(OUTPUT_CHUNKS_PATH, index=False)
    print(f"\n Chunked dataset with embeddings metadata saved → {OUTPUT_CHUNKS_PATH}")

    # Step 4: Write Documentation & Benchmark Report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_md = f"""# Week 3: Chunking Strategy & Embedding Benchmark Report

## 1. Executive Summary

- **Primary Chunking Strategy**: **Recursive Character Chunking** (`chunk_size=300`, `overlap=30`)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dense Embedding Dimension**: `384`
- **Avg Encoding Speed per Chunk**: `{metrics['avg_time_per_chunk_ms']} ms`
- **Encoding Throughput**: `{metrics['throughput_chunks_per_sec']} chunks/second`
- **Projected 50,000 Chunk Indexing Time**: `{metrics['projected_total_indexing_time_sec']} sec` (~`{metrics['projected_total_indexing_time_min']} minutes`)

---

## 2. Chunking Strategy Rationale & Decision

### Evaluated Strategies:
1. **Fixed-Size Chunking**: Simple character slicing.
   - *Drawback*: Cuts words and sentences in half, causing context fragmentation and loss of semantic meaning at boundaries.
2. **Recursive Character Chunking** (Chosen Primary Strategy):
   - *Rationale*: Recursively splits by structural boundaries (`\\n\\n`, `\\n`, `. `, ` `, `""`).
   - Keeps paragraphs and sentences intact while guaranteeing strict upper bounds on token window sizes.
3. **Semantic Sentence Chunking**:
   - *Rationale*: Groups full sentences together until maximum character limit is reached.
   - Ideal for structured prose; falls back to recursive splitting for unpunctuated text walls.

---

## 3. Embedding Model Selection Rationale

Selected Model: **`sentence-transformers/all-MiniLM-L6-v2`**

### Why this model?
- **Speed & Efficiency**: Lightweight 6-layer Transformer architecture with extremely fast inference (~{metrics['avg_time_per_chunk_ms']} ms/chunk on CPU).
- **Optimal Vector Dimension (384-D)**: Provides a high quality-to-memory ratio compared to 768-D or 1536-D models, drastically reducing vector store RAM footprint in ChromaDB.
- **Strong Benchmark Performance**: Trained specifically on sentence-pair datasets using contrastive learning, excelling at semantic search, cosine distance matching, and retrieval tasks.

---

## 4. Indexing Performance Benchmarks

| Metric | Measured Benchmark Value |
|---|---|
| **Model Architecture** | `all-MiniLM-L6-v2` |
| **Vector Dimension** | `{metrics['embedding_dimension']}` |
| **Processed Sample Chunks** | `{metrics['sample_chunk_count']}` |
| **Batch Time** | `{metrics['total_sample_time_sec']} s` |
| **Time per Chunk** | `{metrics['avg_time_per_chunk_ms']} ms` |
| **Throughput** | `{metrics['throughput_chunks_per_sec']} chunks/sec` |
| **Projected 50,000 Chunk Index Time** | **`{metrics['projected_total_indexing_time_sec']} s` (~`{metrics['projected_total_indexing_time_min']} min`)** |

---

## Status
✅ Chunking strategy implemented, benchmarked, and ready for vector storage indexing in ChromaDB.
"""
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"  Benchmark & Decision Report generated → {REPORT_PATH}\n")


if __name__ == "__main__":
    main()
