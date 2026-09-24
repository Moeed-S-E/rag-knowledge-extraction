# Week 10: RAG API Development (FastAPI)

## Objectives
- Wrap the entire RAG system in a production-ready FastAPI application.
- Create endpoints for querying (`/query`), checking health (`/health`), and fetching metadata (`/metadata`).
- Implement request logging to a structured log file.
- Add HTTP error handling for edge cases.
- Test the API interface under concurrent load.

## Architecture

The system has been transformed from an interactive CLI to a stateless API service. 
To optimize for latency, the heavily memory-intensive models (like the SentenceTransformer embeddings) and the database connections (ChromaDB Persistent Client) are loaded strictly once during the FastAPI `lifespan` startup event, rather than instantiated on every request.

### Endpoints

1. **`GET /health`**
   - Returns a simple `{"status": "ok"}` to verify the server is active.
2. **`GET /metadata`**
   - Returns the amount of documents actively stored in the ChromaDB vector store.
3. **`POST /query`**
   - The primary RAG route. It accepts a JSON payload:
     ```json
     {
       "question": "What is SpaceX?",
       "k": 3,
       "topic": "SPACE-GALAXY",
       "entity_boost": "SpaceX"
     }
     ```
   - It executes the full pipeline (Embedding generation, DB Search with optional Topic/Entity filters, LLM Structured Generation, and LLM Hallucination checking) and returns the JSON result.

### Structured Logging
Every query request, including the number of retrieved chunks, boolean success status of the hallucination check, and all latencies are logged directly to `Week-10/logs/api.log` using standard python `logging`.

## Testing

A concurrent testing script (`Week-10/scripts/test_api_concurrent.py`) was created utilizing Python's `ThreadPoolExecutor` to blast the `/query` endpoint with multiple simultaneous requests. 
The FastAPI event loop and the OpenRouter API connection both remained stable without dropping 500 errors.

## Frontend Architecture

A React Single-Page Application (SPA) was added to interface beautifully with the RAG API. 
Located in the `frontend/` directory, it was built using:
- **Vite** for lightning-fast compilation
- **Material-UI (MUI)** for premium UI components
- **Tailwind CSS v4** for utility styling and micro-animations
- **TypeScript** for strict type safety matching the API's Pydantic schemas

## How to Run

**1. Start the API Server:**
```bash
source .venv/bin/activate
uvicorn Week-10.api.main:app --host 127.0.0.1 --port 8000
```
*(Wait until you see `RAG system fully initialized.` in the console)*

**2. Start the React Frontend:**
Open a new terminal window:
```bash
cd frontend
bun run dev
```
Then open `http://localhost:5173` in your browser.

**3. Test the API Concurrently (Optional):**
Open a new terminal and run the test script:
```bash
source .venv/bin/activate
python Week-10/scripts/test_api_concurrent.py
```
