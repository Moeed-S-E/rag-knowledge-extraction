"""DeepSeek / OpenRouter API wrapper."""

import os
from openai import OpenAI, APIConnectionError, RateLimitError, APIError

class LLMClient:
    def __init__(self, provider: str = "openrouter"):
        self.provider = provider
        
        # Configure API Keys and Base URLs
        if self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not api_key:
            raise ValueError(f"API key for {self.provider} not found in environment variables.")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        self.model = "deepseek/deepseek-chat"

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate a structured JSON answer from the LLM."""
        import json
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"answer": f"API Request Failed: {type(e).__name__} - {str(e)}", "citations": []}

    def check_hallucination(self, system_prompt: str, user_prompt: str) -> dict:
        """Check if the generated answer hallucinated beyond the provided context."""
        import json
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"is_supported": False, "reason": f"Hallucination check failed due to API error: {str(e)}"}
