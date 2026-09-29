# E2E RAG Evaluation Report

## Overview
- **Total Queries Executed**: 30
- **Successful Requests**: 30 (100.0%)
- **Failed Requests**: 0

## Generation Quality
- **Hallucinations Detected (LLM-as-a-Judge)**: 0 / 30 (0.0%)
- **Average Chunks Retrieved per Query**: 3.0

## Performance Latency (Average)
- **Retrieval Phase**: 47.5 ms
- **Generation Phase**: 2960.8 ms
- **Total End-to-End Pipeline**: 3106.8 ms

## Known System Limitations & Bottlenecks
1. **Model Cold Starts**: The OpenRouter API and local SentenceTransformers introduce high variance in latency if cold-started. We solved local cold starts via FastAPI lifespan, but network IO to the LLM remains a bottleneck.
2. **Synchronous DB Calls**: ChromaDB `search` blocks the main thread in our API. Under extremely high concurrency, this could throttle the event loop.
3. **Sequential Pipeline**: The Hallucination verification runs *after* generation. We must wait for the LLM to generate the answer, then wait again for the LLM to verify it. This doubles the generation latency.
4. **Context Window Limits**: We hard limit `k=3` (or up to 20) but we do not actively summarize long documents, which might overflow the context window of smaller models or bloat token costs.
