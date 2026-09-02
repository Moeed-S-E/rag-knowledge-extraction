"""Command-line interface for the RAG system."""
from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .data import read_jsonl
from .rag import RAGService


def _service() -> RAGService:
    return RAGService(get_settings())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the RAG knowledge extraction system")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("metadata", help="show index metadata")
    build = subparsers.add_parser("build", help="build the index from the cleaned JSONL corpus")
    build.add_argument("--data", default=None)
    search = subparsers.add_parser("search", help="retrieve relevant chunks")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=None)
    query = subparsers.add_parser("query", help="answer a question with citations")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    service = _service()
    try:
        if args.command == "metadata":
            output = service.metadata()
        elif args.command == "build":
            path = args.data or str(service.settings.clean_data)
            records = read_jsonl(__import__("pathlib").Path(path))
            output = service.build(records)
        elif args.command == "search":
            output = service.search(args.query, args.top_k)
        else:
            output = service.query(args.question, args.top_k)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
