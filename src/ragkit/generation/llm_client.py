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

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate answer with robust error handling."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0, # low temp for RAG
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API Request Failed: {type(e).__name__} - {str(e)}"
