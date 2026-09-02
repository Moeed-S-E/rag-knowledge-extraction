import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv(".env")

from ragkit.generation.llm_client import LLMClient
from ragkit.generation.prompts import HALLUCINATION_SYSTEM_PROMPT, HALLUCINATION_USER_PROMPT_TEMPLATE

llm = LLMClient()

context = "Cats are small, carnivorous mammals. They are often kept as indoor pets."
answer = "Cats are small mammals. They have wings and can fly."

check_prompt = HALLUCINATION_USER_PROMPT_TEMPLATE.format(context=context, answer=answer)
print(llm.check_hallucination(HALLUCINATION_SYSTEM_PROMPT, check_prompt))
