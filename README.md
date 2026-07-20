# RAG-Powered Knowledge Extraction System

12-week AI/ML Engineer Internship project (Track A). Builds a complete
Retrieval-Augmented Generation pipeline: data ingestion -> cleaning ->
chunking -> embedding -> ChromaDB -> retrieval -> LLM generation ->
evaluation -> NLP analysis -> FastAPI/CLI.

## Project Structure

- `src/ragkit/` — the installable package; each submodule corresponds to
  one stage of the pipeline (see `docs/architecture.md`).
- `Week-1/` .. `Week-12/` — weekly deliverables (scripts + reports),
  each calling into `src/ragkit/` rather than duplicating logic.
- `data/` — raw, processed, and vector-store data (raw files kept out of
  git; see `.gitignore`).
- `configs/` — YAML configs for chunking, embeddings, and LLM settings.
- `tests/` — unit tests for the `ragkit` package.

## Setup

```bash
uv sync
cp .env.example .env   # fill in DEEPSEEK_API_KEY / OPENROUTER_API_KEY
```

## Running weekly deliverables

Each `Week-N/` folder has its own README.md with that week's objectives,
approach, and run instructions.
