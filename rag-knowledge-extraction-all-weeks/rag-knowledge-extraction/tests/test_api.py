from pathlib import Path

from fastapi.testclient import TestClient

from rag_app.config import Settings
from rag_app.data import demo_documents
from rag_app.rag import RAGService


def test_service_build_and_search(tmp_path: Path):
    settings = Settings(index_dir=tmp_path / "index", raw_data=tmp_path / "raw.jsonl", clean_data=tmp_path / "clean.jsonl", log_file=tmp_path / "requests.jsonl")
    service = RAGService(settings)
    service.build(demo_documents(5))
    result = service.query("What is retrieval augmented generation?")
    assert result["answer"]
    assert result["citations"]


def test_health_endpoint():
    from rag_app import api
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_empty_query_is_rejected():
    from rag_app import api
    response = TestClient(api.app).post("/query", json={"query": ""})
    assert response.status_code in (400, 422)


def test_enrichment_filtering(tmp_path: Path):
    settings = Settings(index_dir=tmp_path / "index", raw_data=tmp_path / "raw.jsonl", clean_data=tmp_path / "clean.jsonl", log_file=tmp_path / "requests.jsonl")
    service = RAGService(settings)
    service.build(demo_documents(10))
    result = service.search("retrieval", top_k=5, topic="retrieval")
    assert result["results"]
    assert all(item.get("topic") == "retrieval" for item in result["results"])
