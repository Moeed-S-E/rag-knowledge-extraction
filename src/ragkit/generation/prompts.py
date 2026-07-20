"""Prompt templates for grounded answer generation."""

RAG_PROMPT_TEMPLATE = """Answer the question using only the context below.

Context:
{context}

Question: {question}
"""
