import sys
import pexpect

def run():
    p = pexpect.spawn("uv run python Week-6/scripts/rag_cli.py", encoding='utf-8')
    p.expect("You: ")
    p.sendline("what is GENAI")
    p.expect("Generating answer...")
    print(p.before)
    p.sendline("exit")

if __name__ == "__main__":
    run()
