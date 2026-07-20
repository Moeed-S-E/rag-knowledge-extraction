"""Dataset acquisition (ArXiv / Reddit / Wikipedia sources)."""


def load_dataset(source: str, **kwargs):
    """Load a raw dataset by source name. Implement per-source loaders here."""
    raise NotImplementedError
