# Week 1 — Environment & Data Acquisition

> **Parall Internship · Week 1**  
> Goal: stand up a reproducible NLP environment, acquire a real-world text dataset (5,000+ documents), validate it, and produce an auditable data-quality report.

---

## Project Structure

```
Week-1/
├── data/
│   └── raw/
│       ├── arxiv_summaries.csv       # 10,000-row AG News sample (id | category | text)
│       └── data_quality_report.md    # Auto-generated validation report
├── scripts/
│   ├── verify_environment.py         # 11-point library health check
│   └── validate_data.py              # Dataset acquisition + validation pipeline
├── pyproject.toml                    # uv project root anchor
├── requirements.txt                  # Pinned dependency list
└── uv.lock                           # Reproducible lock file
```

---

## Stack

| Layer | Package | Version |
|---|---|---|
| Core | numpy | 1.26.4 |
| Core | pandas | 2.2.2 |
| NLP | spaCy + en_core_web_sm | 3.8.14 / 3.8.0 |
| NLP | NLTK | 3.8.1 |
| NLP | datasets (HuggingFace) | 5.0.0 |
| Embeddings | sentence-transformers | 3.0.1 |
| Embeddings | transformers | 4.57.6 |
| Embeddings | torch | 2.13.0 |
| Vector DB | chromadb | 0.5.3 |
| ML utils | scikit-learn | 1.7.2 |
| ML utils | scipy | 1.15.3 |
| Utilities | tqdm | 4.69.0 |

---

## Quick Start

> **Prerequisite**: [`uv`](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

### 1 — Clone & set up the environment

```bash
git clone <repo-url>
cd Week-1

# Create venv and install all dependencies in one shot
uv venv
uv pip install -r requirements.txt
```

### 2 — Verify the environment

```bash
# From project root (preferred)
uv run scripts/verify_environment.py

# Or from inside scripts/
cd scripts/
uv run python verify_environment.py
```

Expected output:
```
🚀  Starting Environment Verification
=======================================================
🌐  Probing huggingface.co … reachable ✅

  ✅  Python: 3.10.x
  ✅  NumPy: OK (v1.26.4)
  ✅  Pandas: OK (v2.2.2)
  ✅  spaCy: OK (v3.8.14, model en_core_web_sm loaded)
  ✅  NLTK: OK (v3.8.1, punkt+punkt_tab+stopwords ready)
  ✅  PyTorch: OK (v2.13.0, CUDA=not available)
  ✅  Transformers: OK (v4.57.6)
  ✅  Sentence-Transformers: OK (all-MiniLM-L6-v2, embedding dim=384)
  ✅  ChromaDB: OK (v0.5.3, ephemeral collection round-trip passed)
  ✅  scikit-learn: OK (v1.7.2)
  ✅  SciPy: OK (v1.15.3)

=======================================================
✅  All 11/11 checks passed.
```

### 3 — Acquire & validate the dataset

```bash
uv run scripts/validate_data.py
```

This will:
1. Probe `huggingface.co` for reachability
2. Download **AG News** (120k articles, 4 categories) via HuggingFace `datasets`
3. Sample **10,000 rows** with `random_state=42`
4. Save to `data/raw/arxiv_summaries.csv`
5. Run 7 validation checks and write `data/raw/data_quality_report.md`

---

## Scripts

### `scripts/verify_environment.py`

Runs 11 import + functional checks across every dependency. Key design decisions:

- **NLTK `punkt_tab` fix** — NLTK ≥ 3.8.1 requires both `punkt` *and* `punkt_tab` for `word_tokenize`. The script downloads both idempotently via `nltk.download()` rather than relying on `nltk.data.find()`.
- **Sentence-Transformers online/offline branching** — probes `huggingface.co` first. If reachable, loads `all-MiniLM-L6-v2` and runs a real encode. If blocked, validates the full `torch → transformers` pipeline with a locally-instantiated tiny BertModel doing a real forward pass + mean-pooling — no internet required.
- **Non-zero exit on failure** — CI-friendly; exits `1` if any single check fails.

### `scripts/validate_data.py`

End-to-end acquisition and validation pipeline. Validation checks:

| Check | What it catches |
|---|---|
| Total record count | Confirms dataset loaded fully |
| Null / missing per column | Gaps in id, category, or text |
| Duplicate rate (text column) | Copy-pasted or repeated documents |
| Empty-string texts | Blank rows that aren't technically null |
| UTF-8 round-trip | Encoding corruption |
| Non-printable char ratio | Binary garbage or control characters |
| Text length min / avg / max | Degenerate one-word or truncated rows |
| Schema match | Exact `id \| category \| text` column check |

**Acquisition fallback chain:**

```
huggingface.co reachable?
  ├── YES → load ag_news (HuggingFace datasets) → sample 10,000 rows
  │          └── download error (auth/network)? → fallback ↓
  └── NO  → NLTK Reuters-21578 corpus (~10,788 articles, local download)
```

---

## Dataset

| Field | Detail |
|---|---|
| **Source** | [AG News](https://huggingface.co/datasets/ag_news) — public, no auth required |
| **Full size** | 120,000 training articles |
| **Sampled** | 10,000 rows (`random_state=42`) |
| **Categories** | World · Sports · Business · Sci/Tech |
| **Schema** | `id` (str) · `category` (str) · `text` (str) |
| **File** | `data/raw/arxiv_summaries.csv` |
| **Encoding** | UTF-8, no BOM |

### Data Quality Summary (last run)

| Metric | Result |
|---|---|
| Total records | 10,000 |
| Duplicates | 0 (0.0%) |
| Nulls (any column) | 0 (0.0%) |
| Bad UTF-8 rows | 0 (0.0%) |
| Non-printable flagged | 0 (0.0%) |
| Text length min / median / max | 100 / 231 / 1,009 chars |
| Schema validation | ✅ Passed |

Full report: [`data/raw/data_quality_report.md`](data/raw/data_quality_report.md)

---

## Known Quirks

**ChromaDB telemetry warnings**
```
Failed to send telemetry event ...: capture() takes 1 positional argument but 3 were given
```
This is a cosmetic bug in ChromaDB v0.5.3's telemetry client. All collection operations work correctly — safe to ignore.

**`uv run` must resolve to project root**  
Always run scripts from the `Week-1/` root, or from any subdirectory — `pyproject.toml` anchors the project so `uv` always picks up `.venv` correctly. If you run `uv run` from a directory *above* `Week-1/` without a `pyproject.toml` there, it will spin up an empty environment.

---

## Adding New Dependencies

```bash
# Install and auto-update uv.lock
uv add <package>

# Then sync requirements.txt manually if needed
uv pip freeze > requirements.txt
```
