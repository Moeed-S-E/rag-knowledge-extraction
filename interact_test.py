import pexpect

child = pexpect.spawn("uv run python Week-6/scripts/rag_cli.py", encoding='utf-8')
child.expect("You:")
child.sendline("hi")
child.expect("Generating answer...")
print(child.before)
child.sendline("exit")
