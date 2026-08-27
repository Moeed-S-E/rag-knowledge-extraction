from rich.console import Console
from rich.panel import Panel
console = Console()
try:
    console.print(Panel("This is a test [1] and [citation]", title="Title"))
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {type(e)} - {e}")
