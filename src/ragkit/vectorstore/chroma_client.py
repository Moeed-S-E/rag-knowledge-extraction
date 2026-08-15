"""ChromaDB collection management: create, upsert, query."""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class ChromaStore:
    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.persist_dir = persist_dir
        # Ensure the persistence directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = None

    def get_or_create_collection(self, collection_name: str):
        """Create a new collection or get an existing one."""
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Using cosine distance for semantic search
        )
        return self.collection

    def add_chunks(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add documents and their embeddings to the collection."""
        if not self.collection:
            raise ValueError("Collection not initialized. Call get_or_create_collection first.")
        
        if not ids or not documents or not embeddings:
            print("Warning: Attempted to add empty lists to ChromaDB.")
            return

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query_embeddings: List[List[float]], k: int = 5) -> Dict[str, Any]:
        """Retrieve top-K chunks for the given query embeddings."""
        if not self.collection:
            raise ValueError("Collection not initialized. Call get_or_create_collection first.")

        # Edge case: Empty DB
        if self.collection.count() == 0:
            print("Warning: Querying an empty database.")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        # Edge case: Malformed/Empty query embeddings
        if not query_embeddings or not any(query_embeddings):
            print("Warning: Malformed or empty query embeddings provided.")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        # Handle top-k being larger than collection count
        k = min(k, self.collection.count())
        
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=k
        )
        return results
