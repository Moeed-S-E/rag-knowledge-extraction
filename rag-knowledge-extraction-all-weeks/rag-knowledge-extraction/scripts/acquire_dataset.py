"""Acquire a local demo or normalize an external JSONL dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_app.data import demo_documents, normalize_record, quality_report, read_jsonl, save_report, write_jsonl
from rag_app.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="optional input JSONL file")
    parser.add_argument("--count", type=int, default=50, help="demo records when --input is omitted")
    args = parser.parse_args()
    settings = get_settings()
    if args.input:
        records = [normalize_record(record, index) for index, record in enumerate(read_jsonl(args.input))]
    else:
        records = demo_documents(args.count)
    written = write_jsonl(records, settings.raw_data)
    report = quality_report(records)
    save_report(report, Path("artifacts/data_quality_report.json"))
    print(json.dumps({"output": str(settings.raw_data), "written": written, "quality_report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
