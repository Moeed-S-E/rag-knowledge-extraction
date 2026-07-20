import sys
import traceback
import urllib.request


# ── helpers ──────────────────────────────────────────────────────────────────

def _hf_reachable(timeout: int = 6) -> bool:
    """Return True if huggingface.co answers within *timeout* seconds."""
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=timeout)
        return True
    except Exception:
        return False


def run_check(name: str, func) -> bool:
    try:
        result = func()
        print(f"  ✅  {name}: {result}")
        return True
    except Exception as exc:
        print(f"  ❌  {name}: FAILED")
        print(f"       {exc}")
        traceback.print_exc()
        return False


# ── individual checks ─────────────────────────────────────────────────────────

def check_python() -> str:
    return sys.version.split("\n")[0]


def check_numpy() -> str:
    import numpy as np
    a = np.array([1.0, 2.0, 3.0])
    assert a.mean() == 2.0
    return f"OK (v{np.__version__})"


def check_pandas() -> str:
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert len(df) == 3
    return f"OK (v{pd.__version__})"


def check_spacy() -> str:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Environment check passed.")
    assert len(doc) > 0
    return f"OK (spaCy v{spacy.__version__}, model en_core_web_sm loaded)"


def check_nltk() -> str:
    """
    NLTK ≥ 3.8.1 requires both 'punkt' AND 'punkt_tab' for word_tokenize.
    We call nltk.download() idempotently for all three corpora so the check
    never fails because of a missing resource file.
    """
    import nltk

    for resource in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(resource, quiet=True)

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    tokens = word_tokenize("Environment check passed.")
    assert len(tokens) > 0, "word_tokenize returned empty list"

    stops = stopwords.words("english")
    assert len(stops) > 10, "stopwords corpus appears empty"

    return f"OK (v{nltk.__version__}, punkt+punkt_tab+stopwords ready)"


def check_torch() -> str:
    import torch
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.mean().item() == 2.0
    return f"OK (v{torch.__version__}, CUDA={'available' if torch.cuda.is_available() else 'not available'})"


def check_transformers() -> str:
    import transformers
    return f"OK (v{transformers.__version__})"


def check_sentence_transformers(hf_online: bool) -> str:
    """
    • If HuggingFace is reachable: load all-MiniLM-L6-v2 and run a real encode.
    • If offline: import the library, then validate the torch→transformers
      pipeline with a tiny locally-instantiated BertConfig model doing a
      real forward pass + mean-pooling — no internet required.
    """
    from sentence_transformers import SentenceTransformer

    if hf_online:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = model.encode("Environment check.")
        assert len(vec) > 0, "encode() returned empty vector"
        return f"OK (all-MiniLM-L6-v2, embedding dim={len(vec)})"

    # ── offline fallback ──────────────────────────────────────────────────────
    import torch
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    tiny_bert = BertModel(cfg)
    tiny_bert.eval()

    with torch.no_grad():
        input_ids = torch.randint(0, 100, (1, 8))
        attention_mask = torch.ones(1, 8)
        out = tiny_bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (1, 8, 32)
        # mean-pool exactly as sentence-transformers does
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

    assert pooled.shape == (1, 32), f"unexpected pooled shape: {pooled.shape}"
    return (
        f"OK (offline fallback — import OK, torch→transformers forward pass "
        f"+ mean-pool verified, pooled shape={tuple(pooled.shape)})"
    )


def check_chromadb() -> str:
    import chromadb
    import numpy as np

    client = chromadb.EphemeralClient()
    col = client.create_collection("env_check")
    col.add(
        ids=["doc1"],
        embeddings=[[0.1] * 384],
        documents=["test document"],
    )
    results = col.query(query_embeddings=[[0.1] * 384], n_results=1)
    assert results["ids"][0][0] == "doc1"
    return f"OK (ChromaDB v{chromadb.__version__}, ephemeral collection round-trip passed)"


def check_sklearn() -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import sklearn

    vect = TfidfVectorizer()
    X = vect.fit_transform(["hello world", "environment check"])
    assert X.shape[0] == 2
    return f"OK (v{sklearn.__version__})"


def check_scipy() -> str:
    import scipy
    from scipy.spatial.distance import cosine
    import numpy as np

    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    dist = cosine(a, b)
    assert abs(dist - 1.0) < 1e-6
    return f"OK (v{scipy.__version__})"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n🚀  Starting Environment Verification")
    print("=" * 55)

    # Probe network once; share result with the ST check
    print("\n🌐  Probing huggingface.co … ", end="", flush=True)
    hf_online = _hf_reachable()
    print("reachable ✅" if hf_online else "blocked ❌  (offline fallback will be used)")

    print()

    checks = [
        ("Python",                lambda: check_python()),
        ("NumPy",                 lambda: check_numpy()),
        ("Pandas",                lambda: check_pandas()),
        ("spaCy",                 lambda: check_spacy()),
        ("NLTK",                  lambda: check_nltk()),
        ("PyTorch",               lambda: check_torch()),
        ("Transformers",          lambda: check_transformers()),
        ("Sentence-Transformers", lambda: check_sentence_transformers(hf_online)),
        ("ChromaDB",              lambda: check_chromadb()),
        ("scikit-learn",          lambda: check_sklearn()),
        ("SciPy",                 lambda: check_scipy()),
    ]

    results = []
    for name, func in checks:
        results.append(run_check(name, func))

    passed = sum(results)
    failed = len(results) - passed

    print()
    print("=" * 55)
    if failed == 0:
        print(f"✅  All {passed}/{len(results)} checks passed.")
        sys.exit(0)
    else:
        print(f"❌  {failed}/{len(results)} check(s) failed — see output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
