"""Week 7: Hallucination-Resistant RAG CLI with Structured Outputs."""

import sys
import time
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Load environment variables from .env file directly in the app
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from ragkit.embeddings.encoder import Encoder
from ragkit.vectorstore.chroma_client import ChromaStore
from ragkit.generation.llm_client import LLMClient
from ragkit.generation.prompts import (
    SYSTEM_PROMPT, 
    RAG_USER_PROMPT_TEMPLATE,
    HALLUCINATION_SYSTEM_PROMPT,
    HALLUCINATION_USER_PROMPT_TEMPLATE
)

console = Console()

PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_papers"

def init_rag_system():
    console.print("[yellow]Initializing embeddings and Vector DB... (this may take a few seconds)[/yellow]")
    encoder = Encoder(model_name="all-MiniLM-L6-v2")
    # Pre-load the model so the first query doesn't hang
    encoder._load_model()
    
    store = ChromaStore(persist_dir=PERSIST_DIR)
    store.get_or_create_collection(COLLECTION_NAME)
    
    console.print("[yellow]Initializing LLM client...[/yellow]")
    llm = LLMClient(provider="openrouter")
    
    return encoder, store, llm

def main(
    k: int = typer.Option(3, "--top-k", "-k", help="Number of chunks to retrieve"),
):
    """Start an interactive chat with the Hallucination-Resistant RAG system."""
    console.print(Panel.fit("[bold green]Welcome to the RAG CLI v2 (Hallucination Checked)[/bold green]\nType 'exit' or 'quit' to stop."))
    
    try:
        encoder, store, llm = init_rag_system()
    except Exception as e:
        console.print(f"[bold red]Failed to initialize RAG system: {e}[/bold red]")
        sys.exit(1)

    while True:
        question = typer.prompt("You")
        if question.lower() in ("exit", "quit"):
            console.print("[bold blue]Goodbye![/bold blue]")
            break
            
        if not question.strip():
            continue

        start_time = time.perf_counter()

        # 1. Retrieval Phase
        console.print("[dim italic]Retrieving context...[/dim italic]")
        retrieval_start = time.perf_counter()
        q_emb, _ = encoder.encode([question])
        results = store.search(query_embeddings=q_emb.tolist(), k=k)
        
        retrieved_docs = results.get("documents", [[]])[0]
        # In Week 4, chunks were added with IDs like "id_1", "id_2" etc.
        retrieved_ids = results.get("ids", [[]])[0]

        if not retrieved_docs:
            context = "No relevant context found."
        else:
            context = "\n\n".join([f"--- Chunk {retrieved_ids[i]} ---\n{doc}" for i, doc in enumerate(retrieved_docs)])
            
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

        # Print the context being sent to the API without markup parsing
        console.print(Panel(Text(context), title="[dim]Retrieved Context (Sent to API)[/dim]", border_style="dim"))

        # 2. Generation Phase
        generation_start = time.perf_counter()
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)
        
        console.print("[dim italic]Generating structured answer...[/dim italic]")
        response_json = llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        
        answer = response_json.get("answer", "Error: No answer generated.")
        citations = response_json.get("citations", [])
        
        generation_latency = (time.perf_counter() - generation_start) * 1000

        # 3. Hallucination Check Phase
        hallucination_latency = 0
        is_supported = True
        hallucination_reason = ""

        # We only check for hallucinations if the model actually provided an answer instead of admitting ignorance.
        if answer.strip() != "I don't know." and not answer.startswith("API Request Failed:"):
            console.print("[dim italic]Running hallucination check...[/dim italic]")
            hallucination_start = time.perf_counter()
            
            check_prompt = HALLUCINATION_USER_PROMPT_TEMPLATE.format(context=context, answer=answer)
            check_result = llm.check_hallucination(system_prompt=HALLUCINATION_SYSTEM_PROMPT, user_prompt=check_prompt)
            
            is_supported = check_result.get("is_supported", True)
            hallucination_reason = check_result.get("reason", "No reason provided.")
            
            hallucination_latency = (time.perf_counter() - hallucination_start) * 1000

        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000

        # Output Results
        if citations:
            footer = f"\n\n[dim]Sources Cited: {citations}[/dim]"
        else:
            footer = ""

        final_answer_text = answer + footer
        
        if is_supported:
            console.print(Panel(final_answer_text, title="[bold cyan]RAG System[/bold cyan]", border_style="cyan"))
        else:
            # Highlight hallucinated output
            console.print(Panel(f"[red]⚠️ HALLUCINATION DETECTED[/red]\n{hallucination_reason}\n\n[dim]Original Answer:[/dim]\n{final_answer_text}", title="[bold red]RAG System (Rejected)[/bold red]", border_style="red"))
        
        # Log Latencies
        console.print(f"[dim]Latencies -> Retrieval: {retrieval_latency:.1f}ms | Generation: {generation_latency:.1f}ms | Hallucination Check: {hallucination_latency:.1f}ms | Total: {total_latency:.1f}ms[/dim]\n")

if __name__ == "__main__":
    typer.run(main)
