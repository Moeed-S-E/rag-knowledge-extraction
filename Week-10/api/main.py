"""Week 10: RAG FastAPI Application."""

import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from ragkit.embeddings.encoder import Encoder
from ragkit.vectorstore.chroma_client import ChromaStore
from ragkit.generation.llm_client import LLMClient
from ragkit.generation.prompts import (
    SYSTEM_PROMPT, 
    RAG_USER_PROMPT_TEMPLATE,
    HALLUCINATION_SYSTEM_PROMPT,
    HALLUCINATION_USER_PROMPT_TEMPLATE
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)

# File handler with structured format
file_handler = logging.FileHandler(log_dir / "api.log")
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_papers"

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events to load heavy components once at startup."""
    try:
        logger.info("Initializing RAG system components...")
        encoder = Encoder(model_name="all-MiniLM-L6-v2")
        encoder._load_model()
        
        store = ChromaStore(persist_dir=PERSIST_DIR)
        store.get_or_create_collection(COLLECTION_NAME)
        
        llm = LLMClient(provider="openrouter")
        
        state["encoder"] = encoder
        state["store"] = store
        state["llm"] = llm
        
        logger.info("RAG system fully initialized.")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise e
    finally:
        state.clear()
        logger.info("RAG system shut down.")

from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Application & Models
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Knowledge Extraction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question to the RAG system")
    k: int = Field(3, description="Number of context chunks to retrieve", ge=1, le=20)
    topic: Optional[str] = Field(None, description="Topic cluster to filter by")
    entity_boost: Optional[str] = Field(None, description="Entity to aggressively prioritize in retrieval")

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    is_supported: bool
    hallucination_reason: str
    retrieved_chunks: int
    latencies: Dict[str, float]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.get("/metadata")
def get_metadata():
    """Return database metadata."""
    try:
        collection = state["store"].collection
        count = collection.count()
        return {"status": "ok", "document_count": count, "collection_name": COLLECTION_NAME}
    except Exception as e:
        logger.error(f"Metadata error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal database error")

@app.post("/query", response_model=QueryResponse)
def execute_query(req: QueryRequest):
    """Execute a full RAG pipeline query."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty or just whitespace.")
        
    start_time = time.perf_counter()
    logger.info(f"Received query: {req.question}")

    try:
        encoder = state["encoder"]
        store = state["store"]
        llm = state["llm"]
        
        # 1. Retrieval Phase
        retrieval_start = time.perf_counter()
        q_emb, _ = encoder.encode([req.question])
        
        where_filter = {"topic": req.topic} if req.topic else None
        fetch_k = req.k * 3 if req.entity_boost else req.k
            
        results = store.search(query_embeddings=q_emb.tolist(), k=fetch_k, where=where_filter)
        
        retrieved_docs = results.get("documents", [[]])[0]
        retrieved_ids = results.get("ids", [[]])[0]
        retrieved_metadatas = results.get("metadatas", [[]])[0]
        retrieved_distances = results.get("distances", [[]])[0]

        if retrieved_docs and req.entity_boost:
            boosted_results = []
            for i in range(len(retrieved_docs)):
                meta = retrieved_metadatas[i] or {}
                ent_str = meta.get("entities", "")
                dist = retrieved_distances[i]
                
                if req.entity_boost.lower() in ent_str.lower():
                    dist -= 0.5
                    
                boosted_results.append({
                    "id": retrieved_ids[i],
                    "doc": retrieved_docs[i],
                    "meta": meta,
                    "dist": dist
                })
                
            boosted_results = sorted(boosted_results, key=lambda x: x["dist"])
            top_results = boosted_results[:req.k]
            
            retrieved_docs = [r["doc"] for r in top_results]
            retrieved_ids = [r["id"] for r in top_results]

        context = ""
        if retrieved_docs:
            chunks_fmt = []
            for i, doc in enumerate(retrieved_docs):
                chunks_fmt.append(f"--- Chunk {retrieved_ids[i]} ---\n{doc}")
            context = "\n\n".join(chunks_fmt)
            
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

        # 2. Generation Phase
        generation_start = time.perf_counter()
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context, question=req.question)
        
        response_json = llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        
        answer = response_json.get("answer", "Error: No answer generated.")
        citations = response_json.get("citations", [])
        
        generation_latency = (time.perf_counter() - generation_start) * 1000

        # 3. Hallucination Check Phase
        hallucination_latency = 0.0
        is_supported = True
        hallucination_reason = ""

        if answer.strip() != "I don't know." and not answer.startswith("API Request Failed:"):
            hallucination_start = time.perf_counter()
            check_prompt = HALLUCINATION_USER_PROMPT_TEMPLATE.format(context=context, answer=answer)
            check_result = llm.check_hallucination(system_prompt=HALLUCINATION_SYSTEM_PROMPT, user_prompt=check_prompt)
            
            is_supported = check_result.get("is_supported", True)
            hallucination_reason = check_result.get("reason", "No reason provided.")
            hallucination_latency = (time.perf_counter() - hallucination_start) * 1000

        total_latency = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Query processed | retrieved_chunks={len(retrieved_docs)} | is_supported={is_supported} "
            f"| total_latency_ms={total_latency:.1f}"
        )

        return QueryResponse(
            answer=answer,
            citations=[str(c) for c in citations],
            is_supported=is_supported,
            hallucination_reason=hallucination_reason,
            retrieved_chunks=len(retrieved_docs),
            latencies={
                "retrieval_ms": round(retrieval_latency, 1),
                "generation_ms": round(generation_latency, 1),
                "hallucination_ms": round(hallucination_latency, 1),
                "total_ms": round(total_latency, 1)
            }
        )

    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during pipeline execution.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
