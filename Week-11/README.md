# Week 11: End-to-End Evaluation & Code Quality

## Objectives
- Develop an automated evaluation script covering 30 Q&A pairs to benchmark the RAG pipeline.
- Identify and document known architectural limitations and system bottlenecks.
- Implement robust FastAPI unit tests using `pytest` and `fastapi.testclient.TestClient`.
- Add docstrings and type hints across the entire codebase for final cleanup.

## System Architecture Limitations & Known Bottlenecks
While our RAG system is fully functional, scalable, and includes advanced features like Entity Boosting and LLM-as-a-Judge hallucination checks, the current architecture has several limitations:

### 1. Sequential Pipeline Latency
The Hallucination verification phase operates entirely *after* the generation phase completes. This means we wait for a full round-trip to the LLM to generate the answer, and then wait for a *second* full round-trip to verify it. 
**Impact**: Doubles the perceived latency for end-users. 
**Solution**: We could implement streaming responses where the answer streams immediately, and the hallucination flag updates as a background task.

### 2. Synchronous Database Calls
The `ChromaDB` search call in the `/query` endpoint blocks the FastAPI event loop. 
**Impact**: Under extremely high concurrent load, this synchronous CPU-bound task will throttle the async event loop, reducing throughput. 
**Solution**: Run `store.search` inside a `ThreadPoolExecutor` using `asyncio.to_thread`.

### 3. Context Window Bloat
We allow `k` up to 20, but we simply concatenate the raw retrieved chunks into the prompt context.
**Impact**: We risk overflowing the context window of smaller local LLMs, and drastically inflate token costs/latency for cloud LLMs.
**Solution**: Implement an intermediate "Reranking" or "Context Compression" step before sending to the LLM.

### 4. Embedding Model Cold Starts
While mitigated in the API by loading the `SentenceTransformer` during the `lifespan` event, any horizontally scaled worker node spinning up must re-load this massive model into memory. 
**Impact**: Kubernetes autoscaling will experience multi-second delays before a pod becomes healthy. 
**Solution**: Decouple embeddings into a dedicated GPU microservice (e.g., using NVIDIA Triton or Text Embeddings Inference).

## Running Tests
Unit tests have been added using `pytest`. They mock the heavy ML models to ensure lightning-fast CI/CD pipeline execution.
```bash
pytest Week-11/tests/test_api.py -v
```

## Running the E2E Evaluation
The E2E evaluation actually executes the real models and prompts against 30 pre-defined queries.
```bash
python Week-11/scripts/evaluate_e2e.py
```
The script will output a comprehensive `evaluation_report.md` in the `reports` directory detailing average latencies and hallucination rates.
