"""OpenAI-compatible generation client with conservative fallbacks."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


@dataclass
class GenerationResult:
    answer: str
    citations: list[str]
    provider: str
    latency_seconds: float
    error: str | None = None


class LLMClient:
    def __init__(self, api_key: str = "", base_url: str = "", model: str = "", timeout_seconds: float = 30):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model and requests)

    def generate(self, question: str, context: list[dict]) -> GenerationResult:
        started = time.perf_counter()
        if not self.enabled:
            return GenerationResult("", [], "offline", time.perf_counter() - started, "LLM is not configured")
        formatted_context = "\n\n".join(f"[{item.get('id')}] {item.get('text', '')}" for item in context)
        system = (
            "You answer questions using only the supplied context. If the context does not support an answer, "
            "say exactly that you do not know. Return valid JSON with keys answer and citations, where citations "
            "is an array of source chunk IDs. Do not invent citations."
        )
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {question}"},
            ],
        }
        error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    error = f"provider returned HTTP {response.status_code}"
                    time.sleep(0.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                body: Any = response.json()
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content) if isinstance(content, str) else content
                answer = str(parsed.get("answer", "")).strip()
                citations = [str(value) for value in parsed.get("citations", []) if value]
                return GenerationResult(answer, citations, f"llm:{self.model}", time.perf_counter() - started)
            except Exception as exc:
                error = str(exc)
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        return GenerationResult("", [], "llm-error", time.perf_counter() - started, error)
