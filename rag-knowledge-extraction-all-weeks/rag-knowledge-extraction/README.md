# RAG Knowledge Extraction System

A self-contained Python implementation of the Parallax Labs AI/ML Engineer Intern roadmap. It provides a practical retrieval-augmented generation (RAG) pipeline with dataset acquisition, cleaning, chunking, embeddings, vector search, retrieval evaluation, optional LLM generation, hallucination checks, lightweight NLP analysis, a CLI, and a FastAPI service.

The project is intentionally usable without GitHub and is designed to run in two modes:

| Mode | What it uses | Internet/API required |
| --- | --- | --- |
| **Offline fallback** | TF-IDF embeddings, local NumPy vector search, extractive answers, rule-based NLP | No |
| **Enhanced** | Sentence Transformers, ChromaDB, spaCy, scikit-learn, OpenAI-compatible LLM endpoint | Only for the selected integrations |

The fallback mode makes the project testable on a clean Python installation. Optional integrations are detected at runtime instead of making startup fail.

## Features

The implementation covers the roadmap requirements as follows:

| Roadmap area | Implementation |
| --- | --- |
| Data acquisition and validation | `scripts/acquire_dataset.py`, `rag_app.data` |
| Text cleaning and edge cases | `rag_app.cleaning` with unit tests |
| Chunking and embedding | `rag_app.chunking`, `rag_app.embeddings` |
| Vector database and semantic search | `rag_app.vector_store` with ChromaDB adapter and local fallback |
| Retrieval evaluation | `scripts/evaluate_retrieval.py`, `rag_app.evaluation` |
| LLM integration | `rag_app.llm` using any OpenAI-compatible endpoint |
| Hallucination mitigation | `rag_app.rag` with citation checks and “I don't know” behavior |
| Topic, sentiment, and entity analysis | `rag_app.analysis` |
| API and CLI | `rag_app.api`, `rag_app.cli` |
| Automated testing | `tests/` |

## Quick start

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1

python -m pip install -r requirements.txt
cp .env.example .env

python scripts/verify_environment.py
python scripts/acquire_dataset.py --count 50
python scripts/clean_dataset.py
python scripts/build_index.py
python -m rag_app.cli search "What is retrieval augmented generation?"
```

The `src` layout requires either installing the package or setting `PYTHONPATH`:

```bash
python -m pip install -e .
# or, without installation:
export PYTHONPATH="$PWD/src"
```

On Windows PowerShell, use `$env:PYTHONPATH = "$PWD\\src"`.

## Optional LLM configuration

The system works offline by default. To enable generated answers, set these values in `.env` or in the shell:

```text
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=deepseek/deepseek-chat-v3-0324
```

Any OpenAI-compatible endpoint is supported. The client uses a short timeout, retries transient failures, handles rate limits, and validates the response shape. If the key is missing or the endpoint fails, the application returns a grounded extractive answer instead of crashing.

## Dataset options

The acquisition script creates a reproducible local demonstration corpus by default. For a real corpus, pass a JSONL file containing objects with `id`, `title`, `text`, and optional `source` fields:

```bash
python scripts/acquire_dataset.py --input /path/to/documents.jsonl --count 5000
```

The validation report is written to `artifacts/data_quality_report.json`. The cleaned corpus is written to `data/processed/clean_documents.jsonl`, and the cleaning report is written to `artifacts/cleaning_report.json`.

## API

Start the service with:

```bash
uvicorn rag_app.api:app --host 127.0.0.1 --port 8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/query \\
  -H 'Content-Type: application/json' \\
  -d '{"query":"What is RAG?","top_k":3}'
```

Available endpoints are `GET /health`, `GET /metadata`, `POST /query`, and `POST /search`.

## Evaluation and analysis

```bash
python scripts/evaluate_retrieval.py
python scripts/analyze_corpus.py
pytest -q
```

Evaluation results are stored in `artifacts/retrieval_evaluation.json`. NLP analysis is stored in `artifacts/nlp_analysis.json` and, when possible, `artifacts/topic_distribution.png`.

## Architecture

```text
Raw JSONL/CSV
    |
    v
Validation --> data quality report
    |
    v
Cleaning --> normalized documents
    |
    v
Chunking --> overlapping text chunks
    |
    v
Embedding provider (SentenceTransformer or TF-IDF fallback)
    |
    v
Vector store (ChromaDB or local NumPy fallback)
    |
    v
Retriever --> top-k chunks + metadata
    |
    +--> optional OpenAI-compatible LLM --> grounded answer + citations
    |
    +--> extractive fallback answer + support score
    |
    +--> FastAPI / CLI
```

## Design decisions

The default chunker is a deterministic recursive character splitter. It favors paragraph, line, sentence, and whitespace boundaries before using a hard character cut. This keeps chunks readable while enforcing a predictable maximum size and overlap.

The preferred embedding model is `all-MiniLM-L6-v2`, which is small enough for local experimentation. When that dependency or model is unavailable, TF-IDF vectors provide a deterministic and fully local fallback. ChromaDB is used when installed; otherwise a JSON/NumPy local index is used so the API remains operational.

The answer generator is deliberately conservative. It instructs an external model to cite source IDs and refuse unsupported questions. A lexical support check then compares the answer with the retrieved context and marks weakly supported responses. This is a guardrail, not a formal proof of factuality.

## Project layout

```text
rag-knowledge-extraction/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── src/rag_app/
├── scripts/
├── tests/
├── data/raw/
├── data/processed/
├── artifacts/
└── logs/
```

## Important limitation

A 5,000-document corpus and transformer embeddings can require substantial RAM, disk, and download time. The included demo corpus is intentionally small and reproducible. Replace it with a larger JSONL export for a full internship-scale run, then benchmark on the target machine.

## License

This educational project is provided for experimentation and portfolio use. Add the license required by your organization before redistribution.

## Complete 12-week implementation

The ZIP now includes every roadmap week. The detailed mapping is in `docs/12_WEEK_COVERAGE.md`, while `docs/DEMO_GUIDE.md`, `docs/FINAL_REPORT.md`, and `docs/PRESENTATION_OUTLINE.md` provide the Week 12 demonstration and reporting materials.

Run the complete roadmap sequence with:

```bash
export PYTHONPATH="$PWD/src"
python scripts/verify_environment.py
python scripts/acquire_dataset.py --count 50
python scripts/clean_dataset.py
python scripts/spacy_subset.py
python scripts/build_index.py
python scripts/evaluate_retrieval.py
python scripts/evaluate_hallucination.py
python scripts/analyze_corpus.py
python scripts/evaluate_system.py
pytest -q
```

The added deliverables include per-chunk topic, sentiment, and entity metadata; topic and entity retrieval filters; optional positive-sentiment boosting; a 6-case grounded/off-topic hallucination check; an automated 30-case end-to-end benchmark; optional LDA topic assignments; a spaCy tokenization/lemmatization subset run; and final documentation suitable for a demo or presentation.
