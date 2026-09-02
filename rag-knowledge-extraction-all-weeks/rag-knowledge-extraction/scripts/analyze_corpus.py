"""Run lightweight NLP corpus analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_app.analysis import analyze_documents, save_topic_plot
from rag_app.topic_model import fit_topics
from rag_app.config import get_settings
from rag_app.data import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    settings = get_settings()
    records = read_jsonl(args.input or settings.clean_data)
    report = analyze_documents(records)
    report["lda_topic_model"] = fit_topics(records)
    Path("artifacts/nlp_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_created = save_topic_plot(report, Path("artifacts/topic_distribution.png"))
    print(json.dumps({"report": report, "plot_created": plot_created}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
