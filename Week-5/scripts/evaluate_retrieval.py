"""Week 5: Retrieval Evaluation Script."""

import sys
import time
from pathlib import Path

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ragkit.embeddings.encoder import Encoder
from ragkit.vectorstore.chroma_client import ChromaStore
from ragkit.evaluation.metrics import precision_at_k, recall_at_k

PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_papers"

# 20 Manual Test Queries mapped to expected chunk IDs
EVAL_DATASET = [
    # doc_1_c0: "Deep Learning in Computer Vision has advanced exponentially using Transformers & CNNs."
    ("What technologies are advancing computer vision?", ["doc_1_c0"]),
    ("How is deep learning used in image processing?", ["doc_1_c0"]),
    ("Tell me about CNNs and transformers.", ["doc_1_c0"]),
    
    # doc_5_c0: "Natural Language Processing (NLP) enables machines to understand human language with high precision."
    ("How do machines understand human text?", ["doc_5_c0"]),
    ("What is NLP used for?", ["doc_5_c0"]),
    ("Can computers comprehend human language accurately?", ["doc_5_c0"]),
    ("Natural Language Processing overview.", ["doc_5_c0"]),

    # doc_6_c0: "Multilingual Text: Hello! Bonjour! こんにちは! مرحبا! Mixed script data processing test."
    ("How to handle text in different languages?", ["doc_6_c0"]),
    ("Testing French and Arabic strings.", ["doc_6_c0"]),
    ("Mixed script and multilingual data.", ["doc_6_c0"]),
    ("Hello in various languages.", ["doc_6_c0"]),

    # doc_7_c0: "Control chars: Sanitized text removing non-printable control symbols!"
    ("How to clean text of invisible characters?", ["doc_7_c0"]),
    ("Removing control symbols from data.", ["doc_7_c0"]),
    ("Text sanitization and non-printable characters.", ["doc_7_c0"]),

    # doc_8_c0: "alert('xss') Clean text with HTML tags and 'escaped entities'."
    ("How to handle HTML tags in text?", ["doc_8_c0"]),
    ("Escaped entities and cross-site scripting examples.", ["doc_8_c0"]),
    ("XSS alerts and HTML cleaning.", ["doc_8_c0"]),

    # doc_10_c0: "Vector databases like ChromaDB enable efficient similarity search over dense embeddings."
    ("What are vector databases used for?", ["doc_10_c0"]),
    ("How to search dense embeddings?", ["doc_10_c0"]),
    ("Tell me about ChromaDB and similarity search.", ["doc_10_c0"]),
]


def main():
    print("=======================================================")
    print(" Week 5: Retrieval Evaluation (Precision & Recall) ")
    print("=======================================================\n")

    store = ChromaStore(persist_dir=PERSIST_DIR)
    store.get_or_create_collection(COLLECTION_NAME)
    
    encoder = Encoder(model_name="all-MiniLM-L6-v2")

    # Experiment with different K values
    k_values = [1, 2, 3]

    for k in k_values:
        print(f"\n--- Evaluating for Top-K = {k} ---")
        
        total_precision = 0.0
        total_recall = 0.0
        
        for query, expected_chunks in EVAL_DATASET:
            # Embed query
            q_emb, _ = encoder.encode([query])
            
            # Search
            results = store.search(query_embeddings=q_emb.tolist(), k=k)
            
            # Extract retrieved chunk IDs
            retrieved_chunks = results.get("ids", [[]])[0] if results and "ids" in results and results["ids"] else []
            
            # Calculate metrics
            p = precision_at_k(retrieved_chunks, expected_chunks, k)
            r = recall_at_k(retrieved_chunks, expected_chunks, k)
            
            total_precision += p
            total_recall += r
            
        avg_precision = total_precision / len(EVAL_DATASET)
        avg_recall = total_recall / len(EVAL_DATASET)
        
        print(f"Average Precision@{k}: {avg_precision:.4f}")
        print(f"Average Recall@{k}:    {avg_recall:.4f}")

    print("\n=======================================================\n")

if __name__ == "__main__":
    main()
