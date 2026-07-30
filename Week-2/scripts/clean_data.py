"""Week 2: Data Cleaning, Edge Case Handling, spaCy Processing & Quality Metrics Reporting."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Add src directory to path if not installed in editable mode
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ragkit.ingestion.cleaning import (
    clean_dataframe,
    process_spacy_nlp,
)

RAW_DATA_PATH = Path("data/raw/arxiv_summaries.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_CLEAN_PATH = PROCESSED_DIR / "clean_dataset.csv"
REPORT_PATH = Path("Week-2/reports/cleaning_report.md")


def load_or_generate_raw_data() -> pd.DataFrame:
    """Load raw dataset from Week 1 or generate a sample dataset with synthetic edge cases if missing."""
    if RAW_DATA_PATH.exists():
        print(f"  📥  Loading raw dataset from {RAW_DATA_PATH} ...")
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"  ✅  Loaded {len(df):,} records.")
        return df

    print("  ⚠️   Raw dataset not found at data/raw/arxiv_summaries.csv.")
    print("  🔨  Generating synthetic dataset with edge cases for demonstration ...")
    
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_data = {
        "id": [f"doc_{i}" for i in range(1, 11)],
        "category": ["Sci/Tech", "World", "Sports", "Business"] * 2 + ["Sci/Tech", "World"],
        "text": [
            "<div><p>Deep Learning in Computer Vision has advanced exponentially using Transformers &amp; CNNs.</p></div>",
            "   ",  # Empty text edge case
            "Too short",  # Short text edge case
            "<div><p>Deep Learning in Computer Vision has advanced exponentially using Transformers &amp; CNNs.</p></div>",  # Duplicate edge case
            "<b>Natural Language Processing</b> (NLP) enables machines to understand human language with high precision.",
            "Multilingual Text: Hello! Bonjour! こんにちは! مرحبا! Mixed script data processing test.",
            "Control chars: \x00\x08Sanitized text removing non-printable control symbols!\x1f",
            "<script>alert('xss')</script> Clean text with HTML tags and &quot;escaped entities&quot;.",
            "word " * 3000,  # Extremely long text edge case (> 10k chars)
            "Vector databases like ChromaDB enable efficient similarity search over dense embeddings.",
        ],
    }
    df = pd.DataFrame(sample_data)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"  💾  Saved synthetic raw dataset to {RAW_DATA_PATH} ({len(df)} records).")
    return df


def main() -> None:
    print("\n=======================================================")
    print("🚀  Week 2: Data Cleaning & Text Preprocessing Pipeline")
    print("=======================================================")

    # Step 1: Load Raw Dataset
    df_raw = load_or_generate_raw_data()

    # Step 2: Perform Data Cleaning & Edge Case Filtering
    print("\n🧹  Cleaning dataset and handling edge cases ...")
    df_clean, stats = clean_dataframe(
        df_raw,
        text_column="text",
        min_length=15,
        max_length=10000,
        truncate_long=False,
    )

    print(f"  • Initial records     : {stats['initial_count']:,}")
    print(f"  • Final clean records : {stats['final_count']:,}")
    print(f"  • Total dropped       : {stats['dropped_count']:,}")
    print(f"  • Invalid/Empty dropped: {stats['dropped_null_or_invalid']:,}")
    print(f"  • Duplicates dropped  : {stats['dropped_duplicates']:,}")
    print(f"  📊 Drop percentage    : {stats['drop_percentage']}%")

    # Step 3: Run spaCy NLP Analysis on a Subset
    print("\n🔤  Running spaCy tokenization & lemmatization on subset ...")
    sample_texts = df_clean["text"].head(5).tolist()
    nlp_results = []
    try:
        nlp_results = process_spacy_nlp(sample_texts)
        print(f"  ✅  Processed spaCy NLP statistics for {len(nlp_results)} sample texts.")
    except Exception as exc:
        print(f"  ⚠️   spaCy processing skipped or fallback used: {exc}")

    # Step 4: Export Clean Dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_CLEAN_PATH, index=False, encoding="utf-8")
    print(f"\n💾  Clean dataset saved → {OUTPUT_CLEAN_PATH}")

    # Step 5: Generate Cleaning Quality Report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    spacy_sample_md = ""
    if nlp_results:
        spacy_sample_md = "\n### spaCy Subset Tokenization / Lemmatization Sample\n\n"
        for idx, res in enumerate(nlp_results):
            spacy_sample_md += (
                f"- **Doc {idx+1}**: {res['num_tokens']} tokens, {res['num_lemmas']} unique lemmas\n"
                f"  - *Sample Tokens*: `{res['sample_tokens'][:5]}`\n"
                f"  - *Sample Lemmas*: `{res['sample_lemmas'][:5]}`\n"
            )

    report_content = f"""# Week 2: Data Cleaning & Preprocessing Report

**Dataset Input** : `{RAW_DATA_PATH}`  
**Dataset Output** : `{OUTPUT_CLEAN_PATH}`  

---

## Cleaning Summary Metrics

| Metric | Value |
|---|---|
| **Initial Records** | {stats['initial_count']:,} |
| **Clean Records** | {stats['final_count']:,} |
| **Total Dropped** | {stats['dropped_count']:,} |
| **Invalid / Empty / Out-of-Bounds Dropped** | {stats['dropped_null_or_invalid']:,} |
| **Duplicates Dropped** | {stats['dropped_duplicates']:,} |
| **Data Drop Percentage** | **{stats['drop_percentage']}%** |

---

## Operations Performed

1. **HTML Tag & Entity Stripping**: Removed tags (`<p>`, `<div>`, `<script>`) and unescaped HTML entities (`&amp;`, `&quot;`).
2. **Control Character & Special Character Cleaning**: Filtered non-printable ASCII control bytes (`\\x00-\\x1f`) and normalized Unicode to NFKC format.
3. **Whitespace Normalization**: Collapsed consecutive spaces, tabs, and newlines into single spaces.
4. **Edge Case Validation**:
   - Filtered empty and whitespace-only text inputs.
   - Filtered text under 15 characters (too short for meaningful embedding).
   - Filtered extremely long documents (> 10,000 characters).
   - Preserved mixed-language and multilingual scripts (e.g. CJK, Arabic, French).
5. **Deduplication**: Removed exact duplicate clean text strings.
{spacy_sample_md}
---

## Status
✅ Dataset is clean, validated, and ready for vector embedding in Week 3.
"""
    REPORT_PATH.write_text(report_content, encoding="utf-8")
    print(f"📄  Quality Report generated → {REPORT_PATH}\n")


if __name__ == "__main__":
    main()
