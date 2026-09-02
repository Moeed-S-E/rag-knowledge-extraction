"""Configuration for the RAG application."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")


def _path(value: str, default: str) -> Path:
    candidate = Path(os.getenv(value, default))
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@dataclass(frozen=True)
class Settings:
    raw_data: Path = _path("RAG_RAW_DATA", "data/raw/documents.jsonl")
    clean_data: Path = _path("RAG_CLEAN_DATA", "data/processed/clean_documents.jsonl")
    index_dir: Path = _path("RAG_INDEX_DIR", "artifacts/index")
    log_file: Path = _path("RAG_LOG_FILE", "logs/rag_requests.jsonl")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))

    def ensure_directories(self) -> None:
        for path in (self.raw_data, self.clean_data, self.index_dir, self.log_file):
            path.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
