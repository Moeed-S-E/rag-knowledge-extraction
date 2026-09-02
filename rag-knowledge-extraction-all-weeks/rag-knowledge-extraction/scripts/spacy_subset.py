"""Run optional spaCy tokenization and lemmatization on a corpus subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_app.config import get_settings
from rag_app.data import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    settings = get_settings()
    records = read_jsonl(args.input or settings.clean_data)[: max(0, args.limit)]
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
        output = []
        for record in records:
            doc = nlp(record.get("text", ""))
            output.append({"id": record.get("id"), "tokens": [token.text for token in doc], "lemmas": [token.lemma_ or token.text.lower() for token in doc]})
        report = {"provider": "spacy", "records": output}
    except Exception as exc:
        report = {"provider": "unavailable", "error": str(exc), "records": []}
    Path("artifacts/spacy_subset.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"provider": report["provider"], "record_count": len(report["records"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
