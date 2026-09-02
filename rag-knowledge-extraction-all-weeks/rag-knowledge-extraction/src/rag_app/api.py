"""FastAPI application for the RAG service."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import RAGService


app = FastAPI(title="RAG Knowledge Extraction API", version="1.0.0")
service = RAGService()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=50)
    topic: str | None = Field(default=None, max_length=100)
    entity: str | None = Field(default=None, max_length=100)
    positive_boost: bool = False


class QueryRequest(SearchRequest):
    pass


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", **service.metadata()}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return service.metadata()


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    try:
        return service.search(request.query, request.top_k, request.topic, request.entity, request.positive_boost)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Search failed") from exc


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    try:
        return service.query(request.query, request.top_k, request.topic, request.entity, request.positive_boost)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Query failed") from exc
