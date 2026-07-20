"""ChromaDB collection management: create, upsert, query."""


class ChromaStore:
    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.persist_dir = persist_dir
