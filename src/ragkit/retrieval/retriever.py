"""Top-k retrieval, reranking, metadata filters."""


class Retriever:
    def retrieve(self, query: str, k: int = 5):
        raise NotImplementedError
