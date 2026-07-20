"""FastAPI app exposing the RAG system."""

from fastapi import FastAPI

app = FastAPI(title="ragkit API")


@app.get("/health")
def health():
    return {"status": "ok"}
