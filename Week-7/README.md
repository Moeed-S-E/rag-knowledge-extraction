# Week 7: Hallucination Mitigation & Structured Output

## Objectives
- Introduce an automated hallucination check to verify claims against retrieved context.
- Adjust prompts to explicitly output structured JSON with citations.
- Adjust prompts to forcefully reject off-topic questions with "I don't know."
- Test the system with off-topic and on-topic queries.
- Create a new CLI `rag_cli_v2.py` implementing the dual-step generation and verification.

## Implementation Details

### Structured Output Generation
We modified `LLMClient` to utilize OpenRouter's JSON Object mode (`response_format={"type": "json_object"}`).
The System Prompt now strictly enforces the output format:
```json
{
  "answer": "Generated answer based on the context",
  "citations": [1, 2] // The IDs of the chunks used
}
```

### Hallucination Checker (LLM-as-a-Judge)
We implemented a secondary verification pass using a distinct `HALLUCINATION_SYSTEM_PROMPT`.
- If the first generation step returns an answer (and not an admission of ignorance), `rag_cli_v2.py` triggers `LLMClient.check_hallucination()`.
- The judge LLM is given both the raw context and the generated answer. It verifies that **all claims** in the generated answer are explicitly supported by the context.
- It returns a structured JSON:
  ```json
  {
    "is_supported": true,
    "reason": "Explanation of verification"
  }
  ```
- The CLI automatically surfaces a `⚠️ HALLUCINATION DETECTED` warning if `is_supported` is `false`.

### Mitigation Strategies & Efficacy
1. **The "I don't know" rule**: 
   - We updated the System Prompt to say: `If the context does not contain the answer, your answer MUST be exactly "I don't know."`
   - **Testing**: When asked "What is the capital of France?", the model reliably output `I don't know.`, avoiding the generation phase entirely and mitigating arbitrary generation.
2. **Two-Step Verification**:
   - For valid on-topic questions (e.g. "What is NLP?"), the model generates a precise answer and cites `[Chunk 1]`.
   - The hallucination checker runs immediately after, verifying the answer against Chunk 1, ensuring no auxiliary unsourced information was injected.

## How to Run

1. Ensure your virtual environment is active:
   ```bash
   source .venv/bin/activate
   ```
2. Your `.env` file should contain your OpenRouter API key.
3. Run the new V2 CLI:
   ```bash
   uv run python Week-7/scripts/rag_cli_v2.py
   ```
