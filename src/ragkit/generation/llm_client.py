"""DeepSeek / OpenRouter API wrapper."""


class LLMClient:
    def __init__(self, provider: str = "deepseek"):
        self.provider = provider

    def generate(self, prompt: str) -> str:
        raise NotImplementedError
