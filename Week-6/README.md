# Week 6: End-to-End RAG System & LLM Integration

## Objectives
- Integrate the OpenRouter API to generate answers using the retrieved context.
- Implement robust prompt engineering techniques (System Prompt, Context Injection).
- Add error handling for external API calls (rate limits, timeouts, bad formats).
- Measure and log end-to-end latency (retrieval + generation).
- Wrap the entire pipeline in an interactive Command Line Interface (CLI).

## Implementation Details

### LLM Client (`src/ragkit/generation/llm_client.py`)
We implemented `LLMClient` wrapping the standard `openai` SDK to interact with OpenAI-compatible endpoints like **OpenRouter** (`https://openrouter.ai/api/v1`).
- The model used by default is `deepseek/deepseek-chat`.
- **Robust Error Handling**: The `generate()` method gracefully catches:
  - `RateLimitError`
  - `APIConnectionError`
  - `APIError` (for malformed responses)
  and returns a safe string error rather than crashing the application.

### Prompts (`src/ragkit/generation/prompts.py`)
- Added a `SYSTEM_PROMPT` containing specific guidelines instructing the model to remain factual, concise, and to reject answering if the answer isn't within the context (preventing hallucinations).
- Added `RAG_USER_PROMPT_TEMPLATE` for cleanly separating `{context}` and `{question}`.

### CLI Application (`Week-6/scripts/rag_cli.py`)
Developed a `typer` based interactive loop that:
1. Prompts the user for a query.
2. Embeds the query and queries ChromaDB (Retrieval Phase).
3. Injects the results into the prompt templates.
4. Generates an answer via the LLM (Generation Phase).
5. **Logs latency separately** for both retrieval and generation, providing real-time performance insights.

## How to Run

1. Ensure your virtual environment is active:
   ```bash
   source .venv/bin/activate
   ```
2. Set up your `.env` file:
   - Create a `.env` file in the root of the project (you can copy from `.env.example`).
   - Add your API key:
     ```
     OPENROUTER_API_KEY="your-api-key-here"
     ```
3. Start the CLI:
   ```bash
   uv run python Week-6/scripts/rag_cli.py
   ```
   You can optionally specify a custom Top-K retrieval window:
   ```bash
   uv run python Week-6/scripts/rag_cli.py --top-k 2
   ```

Type your questions, and type `exit` or `quit` to leave the interactive prompt.
