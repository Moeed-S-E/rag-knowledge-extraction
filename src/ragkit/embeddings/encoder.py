"""sentence-transformers embedding wrapper."""


class Encoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def encode(self, texts):
        raise NotImplementedError
