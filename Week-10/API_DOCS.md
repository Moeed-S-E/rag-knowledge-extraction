# RAG API Documentation

The RAG Knowledge Extraction system exposes a RESTful API powered by FastAPI. 

> **Interactive Swagger UI:** When the server is running, you can view the auto-generated interactive documentation and test the endpoints directly from your browser by navigating to:
> **http://127.0.0.1:8000/docs**

---

## Base URL
`http://127.0.0.1:8000`

---

## Endpoints

### 1. Health Check
Checks if the API server is up and running.

**Request:**
`GET /health`

**Response:**
`200 OK`
```json
{
  "status": "ok"
}
```

---

### 2. Collection Metadata
Retrieves basic metadata about the loaded ChromaDB vector collection.

**Request:**
`GET /metadata`

**Response:**
`200 OK`
```json
{
  "status": "ok",
  "document_count": 120,
  "collection_name": "arxiv_papers"
}
```
*Note: Returns `500 Internal Server Error` if the database connection fails.*

---

### 3. Execute Query (RAG)
The core endpoint. Retrieves relevant chunks from the vector database, formats a context string, and prompts the LLM to generate an answer. It also runs a subsequent LLM verification step to check for hallucinations.

**Request:**
`POST /query`

**Headers:**
`Content-Type: application/json`

**Body Parameters:**
| Parameter      | Type   | Required | Default | Description |
|----------------|--------|----------|---------|-------------|
| `question`     | string | Yes      | -       | The user's query for the RAG system. Cannot be empty. |
| `k`            | int    | No       | 3       | Number of document chunks to retrieve (max 20). |
| `topic`        | string | No       | null    | Filter retrieval strictly to this topic cluster. |
| `entity_boost` | string | No       | null    | Aggressively boost chunks containing this exact entity in their metadata. |

**Example Request Body:**
```json
{
  "question": "What is the impact of deep learning?",
  "k": 3,
  "entity_boost": "CNNs"
}
```

**Response:**
`200 OK`
```json
{
  "answer": "Deep learning architectures like CNNs and Transformers have significantly advanced image and text processing.",
  "citations": ["12", "45"],
  "is_supported": true,
  "hallucination_reason": "The generated answer is directly supported by the context provided in Chunk 12 and 45.",
  "retrieved_chunks": 3,
  "latencies": {
    "retrieval_ms": 12.5,
    "generation_ms": 1850.2,
    "hallucination_ms": 1400.1,
    "total_ms": 3262.8
  }
}
```

**Error Responses:**
- `400 Bad Request`: If the `question` field is empty or purely whitespace.
- `422 Unprocessable Entity`: If the JSON payload types are invalid (handled natively by FastAPI/Pydantic).
- `500 Internal Server Error`: If the RAG pipeline fails (e.g. OpenRouter API is down, or ChromaDB errors out).
