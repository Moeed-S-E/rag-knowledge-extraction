"""Week 6: End-to-End RAG CLI with Typer."""

import sys
import time
from pathlib import Path
# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Load environment variables from .env file directly in the app
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from ragkit.embeddings.encoder import Encoder
from ragkit.vectorstore.chroma_client import ChromaStore
from ragkit.generation.llm_client import LLMClient
from ragkit.generation.prompts import SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE

app = typer.Typer(help="Interactive CLI for the RAG Knowledge Extraction System.")
console = Console()

PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_papers"

# Initialize components lazily to speed up CLI startup if needed, 
# but for an interactive loop, initializing once is fine.
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
    """Start an interactive chat with the RAG system."""
    console.print(Panel.fit("[bold green]Welcome to the RAG CLI[/bold green]\nType 'exit' or 'quit' to stop."))
    
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
        if not retrieved_docs:
            context = "No relevant context found."
        else:
            context = "\n\n".join([f"--- Chunk {i+1} ---\n{doc}" for i, doc in enumerate(retrieved_docs)])
            
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

        from rich.text import Text
        # Print the context being sent to the API without markup parsing
        console.print(Panel(Text(context), title="[dim]Retrieved Context (Sent to API)[/dim]", border_style="dim"))

        # 2. Generation Phase
        generation_start = time.perf_counter()
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)
        
        console.print("[dim italic]Generating answer...[/dim italic]")
        response = llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        if isinstance(response, dict):
            answer = response.get("answer", str(response))
        else:
            answer = str(response)
        generation_latency = (time.perf_counter() - generation_start) * 1000
        
        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000

        # Output Results
        console.print(Panel(answer, title="[bold cyan]RAG System[/bold cyan]", border_style="cyan"))
        
        # Log Latencies
        console.print(f"[dim]Latencies -> Retrieval: {retrieval_latency:.1f}ms | Generation: {generation_latency:.1f}ms | Total: {total_latency:.1f}ms[/dim]\n")

if __name__ == "__main__":
    typer.run(main)

