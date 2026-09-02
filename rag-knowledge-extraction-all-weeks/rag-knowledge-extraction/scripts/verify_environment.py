"""Verify core and optional Python dependencies."""
from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path


PACKAGES = ["numpy", "pydantic", "fastapi", "pytest", "requests", "pandas", "sklearn", "spacy", "sentence_transformers", "chromadb"]


def main() -> int:
    results = {}
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            results[package] = {"installed": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:
            results[package] = {"installed": False, "error": str(exc)}
    report = {"python": sys.version, "platform": platform.platform(), "packages": results}
    output = Path("artifacts/environment_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
