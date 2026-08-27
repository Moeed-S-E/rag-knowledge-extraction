import os
from openai import OpenAI

api_key = "your_key_here"
base_url = "https://openrouter.ai/api/v1"

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

try:
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10
    )
    print(response)
except Exception as e:
    print(f"Exception Type: {type(e)}")
    print(f"Exception Msg: {e}")
