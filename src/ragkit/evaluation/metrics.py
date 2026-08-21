"""RAG evaluation metrics: precision, recall, hallucination detection."""
from typing import List, Set

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@K:
    Of the top K retrieved documents, what fraction are relevant?
    """
    if not retrieved or k <= 0:
        return 0.0
    
    top_k_retrieved = retrieved[:k]
    relevant_set = set(relevant)
    
    hits = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    return hits / len(top_k_retrieved)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@K:
    Of all relevant documents, what fraction were found in the top K retrieved?
    """
    if not relevant or k <= 0:
        return 0.0
        
    top_k_retrieved = retrieved[:k]
    relevant_set = set(relevant)
    
    hits = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    return hits / len(relevant_set)
