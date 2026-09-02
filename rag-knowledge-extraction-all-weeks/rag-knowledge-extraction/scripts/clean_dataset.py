"""Clean the acquired dataset and write a preprocessing report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_app.cleaning import clean_records
from rag_app.config import get_settings
from rag_app.data import read_jsonl, save_report, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    settings = get_settings()
    records = read_jsonl(args.input or settings.raw_data)
    cleaned, report = clean_records(records)
    write_jsonl(cleaned, settings.clean_data)
    save_report(report, Path("artifacts/cleaning_report.json"))
    print(json.dumps({"output": str(settings.clean_data), "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
