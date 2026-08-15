"""Week 4: ChromaDB Setup, Data Ingestion, and Semantic Search Benchmark."""

import sys
import time
from pathlib import Path
import pandas as pd

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ragkit.embeddings.encoder import Encoder
from ragkit.vectorstore.chroma_client import ChromaStore

CHUNKS_PATH = Path("data/processed/chunked_embeddings.csv")
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_papers"

# 10 Diverse Test Queries
TEST_QUERIES = [
    "What is Retrieval-Augmented Generation?",
    "How do Transformers use self-attention?",
    "Optimizing vector database indexing performance",
    "Comparison of text chunking strategies for RAG",
    "Financial news on global markets",  # Testing something out of domain
    "Machine learning applications in healthcare",
    "Understanding cosine similarity in dense vectors",
    "Natural language processing advances",
    "What are the benefits of recursive character chunking?",
    "How to reduce embedding latency?",
]

def load_data() -> pd.DataFrame:
    if not CHUNKS_PATH.exists():
        print(f"Error: Could not find chunks at {CHUNKS_PATH}. Please run Week 3 pipeline first.")
        sys.exit(1)
    print(f"Loading data from {CHUNKS_PATH}...")
    return pd.read_csv(CHUNKS_PATH)

def main():
    print("=======================================================")
    print(" Week 4: ChromaDB Setup & Semantic Search Benchmark")
    print("=======================================================\n")

    # Step 1: Initialize ChromaDB
    print(f"Initializing ChromaDB in {PERSIST_DIR}...")
    store = ChromaStore(persist_dir=PERSIST_DIR)
    
    # Check Edge Case: Empty DB
    print("\n--- Testing Edge Case: Querying Empty Database ---")
    store.get_or_create_collection("temp_empty_collection")
    encoder = Encoder(model_name="all-MiniLM-L6-v2")
    empty_q_emb, _ = encoder.encode(["Test query"])
    empty_res = store.search(query_embeddings=empty_q_emb.tolist(), k=2)
    print(f"Result from empty db: {empty_res}\n")

    print(f"Creating/getting collection: {COLLECTION_NAME}")
    store.get_or_create_collection(COLLECTION_NAME)

    # Step 2: Load Data & Re-compute Embeddings
    df = load_data()
    print(f"Loaded {len(df)} chunks.")
    
    chunk_texts = df["chunk_text"].tolist()
    ids = df["chunk_id"].tolist()
    
    # Extract metadata
    metadatas = []
    for _, row in df.iterrows():
        meta = {
            "doc_id": str(row.get("doc_id", "")),
            "category": str(row.get("category", "")),
            "strategy": str(row.get("strategy", "")),
        }
        metadatas.append(meta)

    print("\nGenerating embeddings for all chunks...")
    embeddings_np, enc_time = encoder.encode(chunk_texts, batch_size=32, show_progress_bar=False)
    embeddings = embeddings_np.tolist()
    print(f"Encoded {len(embeddings)} chunks in {enc_time:.2f} seconds.")

    # Step 3: Ingest Data
    print("\nIngesting data into ChromaDB...")
    start_ingest = time.perf_counter()
    store.add_chunks(ids=ids, documents=chunk_texts, embeddings=embeddings, metadatas=metadatas)
    ingest_time = time.perf_counter() - start_ingest
    print(f"Ingested {len(ids)} chunks in {ingest_time:.2f} seconds.")

    # Step 4: Run Queries & Benchmark
    print("\n--- Running Semantic Search Benchmark ---")
    
    # Check Edge Case: Malformed Query (Empty String)
    print("\nTesting Edge Case: Malformed Query")
    try:
        malformed_emb, _ = encoder.encode([""])
        if len(malformed_emb) == 0 or not any(malformed_emb[0]):
            store.search(query_embeddings=[[]]) # pass empty to trigger warning
        else:
            # If the encoder handles empty strings, we can pass an empty list directly
            store.search(query_embeddings=[[]])
    except Exception as e:
        print(f"Caught expected exception or warning: {e}")

    total_latency = 0
    print("\nBenchmark Results:")
    for i, query in enumerate(TEST_QUERIES, 1):
        start_q = time.perf_counter()
        
        # Embed query
        q_emb, _ = encoder.encode([query])
        
        # Search
        results = store.search(query_embeddings=q_emb.tolist(), k=3)
        
        latency = (time.perf_counter() - start_q) * 1000  # in ms
        total_latency += latency
        
        print(f"\n[{i}/10] Query: '{query}'")
        print(f"       Latency: {latency:.2f} ms")
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            dists = results["distances"][0] if "distances" in results and results["distances"] else [0]*len(docs)
            for j, (doc, dist) in enumerate(zip(docs, dists)):
                # Print just a snippet of the document
                snippet = doc[:80].replace("\n", " ") + "..." if len(doc) > 80 else doc
                print(f"       Rank {j+1} (Dist: {dist:.4f}): {snippet}")
        else:
            print("       No results found.")

    avg_latency = total_latency / len(TEST_QUERIES)
    print(f"\n--- Benchmark Summary ---")
    print(f"Total Queries: {len(TEST_QUERIES)}")
    print(f"Average Latency: {avg_latency:.2f} ms per query")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
