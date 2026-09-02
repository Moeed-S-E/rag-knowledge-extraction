"""Build the persistent RAG index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_app.config import get_settings
from rag_app.data import read_jsonl
from rag_app.rag import RAGService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    service = RAGService(get_settings())
    records = read_jsonl(args.data or service.settings.clean_data)
    result = service.build(records)
    Path("artifacts/index_build_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
