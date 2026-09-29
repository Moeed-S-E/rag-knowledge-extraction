"""Unit tests for the FastAPI RAG endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
import importlib.util
from pathlib import Path

# Dynamically import Week-10/api/main.py since it has a hyphen in the folder name
project_root = Path(__file__).resolve().parent.parent.parent
main_py_path = project_root / "Week-10" / "api" / "main.py"
spec = importlib.util.spec_from_file_location("main_api", main_py_path)
main_module = importlib.util.module_from_spec(spec)
sys.modules["main_api"] = main_module
spec.loader.exec_module(main_module)

app = main_module.app
state = main_module.state
COLLECTION_NAME = main_module.COLLECTION_NAME

client = TestClient(app)

@pytest.fixture
def mock_state():
    """Mock the global state to avoid loading real models during testing."""
    with patch.dict(state, {}, clear=True):
        mock_encoder = MagicMock()
        # Return a dummy embedding (list of floats)
        mock_encoder.encode.return_value = ([[0.1] * 384], None)
        
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_store.collection = mock_collection
        # Return dummy search results
        mock_store.search.return_value = {
            "documents": [["Test document content"]],
            "ids": [["chunk_1"]],
            "metadatas": [[{"entities": "Test Entity"}]],
            "distances": [[0.5]]
        }
        
        mock_llm = MagicMock()
        # Mock LLM generation
        mock_llm.generate.return_value = {
            "answer": "This is a mocked answer.",
            "citations": ["chunk_1"]
        }
        # Mock LLM hallucination check
        mock_llm.check_hallucination.return_value = {
            "is_supported": True,
            "reason": "Supported."
        }
        
        state["encoder"] = mock_encoder
        state["store"] = mock_store
        state["llm"] = mock_llm
        
        yield state

def test_health_check():
    """Test the /health endpoint returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_metadata(mock_state):
    """Test the /metadata endpoint returns collection statistics."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["document_count"] == 42
    assert data["collection_name"] == COLLECTION_NAME

def test_execute_query_success(mock_state):
    """Test a successful RAG query execution."""
    payload = {
        "question": "What is the test document?",
        "k": 1,
        "topic": "test",
        "entity_boost": "Test Entity"
    }
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a mocked answer."
    assert data["citations"] == ["chunk_1"]
    assert data["is_supported"] is True
    assert "retrieval_ms" in data["latencies"]
    assert "generation_ms" in data["latencies"]

def test_execute_query_empty_question():
    """Test querying with an empty question returns a 400 Bad Request."""
    payload = {
        "question": "   ",
        "k": 1
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

def test_execute_query_missing_field():
    """Test querying with missing required field (question) returns 422 Unprocessable Entity."""
    payload = {
        "k": 1
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422
