from __future__ import annotations

import os
import re
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
OUT_CSV = RAW_DIR / "arxiv_summaries.csv"
REPORT_PATH = RAW_DIR / "data_quality_report.md"

EXPECTED_COLUMNS = {"id", "category", "text"}
SAMPLE_SIZE = 10_000
NONPRINTABLE_THRESHOLD = 0.05   # flag rows where >5 % of chars are non-printable


# ── network probe ─────────────────────────────────────────────────────────────

def _hf_reachable(timeout: int = 8) -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=timeout)
        return True
    except Exception:
        return False


# ── acquisition helpers ───────────────────────────────────────────────────────

def _acquire_hf() -> tuple[pd.DataFrame, str]:
    """
    Download AG News from HuggingFace Hub (public, no auth required).
    Schema: id | category | text
    AG News has 120,000 train rows across 4 news categories.
    We sample SAMPLE_SIZE rows for headroom above the 5,000 threshold.
    """
    from datasets import load_dataset

    print("  📥  Loading AG News from HuggingFace Hub (public dataset) …")
    ds = load_dataset("ag_news", split="train", trust_remote_code=False)
    print(f"  ✅  Loaded {len(ds):,} rows.")

    # AG News label map
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    rows = [
        {
            "id": str(i),
            "category": label_names[row["label"]],
            "text": row["text"],
        }
        for i, row in enumerate(ds)
    ]
    df = pd.DataFrame(rows)[["id", "category", "text"]]

    # Sample down to SAMPLE_SIZE
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    source_note = (
        "**Primary source used**: HuggingFace Hub — "
        "`ag_news` (AG News corpus, 4-category news classification dataset, "
        "120k training articles). "
        f"Sampled to {SAMPLE_SIZE:,} rows (random_state=42)."
    )
    return df, source_note


def _acquire_reuters() -> tuple[pd.DataFrame, str]:
    """Fallback: NLTK Reuters-21578 corpus (≈10,788 news articles)."""
    import nltk

    print("  📥  HuggingFace unreachable — using NLTK Reuters-21578 corpus …")
    nltk.download("reuters", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    from nltk.corpus import reuters

    rows = []
    for fid in reuters.fileids():
        cats = ", ".join(reuters.categories(fid))
        text = reuters.raw(fid).strip()
        rows.append({"id": fid, "category": cats, "text": text})

    df = pd.DataFrame(rows)[["id", "category", "text"]]
    print(f"  ✅  Loaded {len(df):,} Reuters documents.")

    source_note = (
        "**Fallback source used**: NLTK Reuters-21578 corpus "
        f"({len(df):,} news articles). "
        "Primary HuggingFace source was unreachable at acquisition time."
    )
    return df, source_note


def acquire_dataset() -> tuple[pd.DataFrame, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🌐  Probing huggingface.co … ", end="", flush=True)
    online = _hf_reachable()
    print("reachable ✅" if online else "blocked ❌")

    if online:
        try:
            df, note = _acquire_hf()
        except Exception as exc:
            print(f"  ⚠️   HuggingFace download failed ({exc.__class__.__name__}: {exc})")
            print("  ↩️   Falling back to NLTK Reuters-21578 corpus …")
            df, note = _acquire_reuters()
    else:
        df, note = _acquire_reuters()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"  💾  Saved → {OUT_CSV}  ({len(df):,} rows)")
    return df, note


# ── validation helpers ────────────────────────────────────────────────────────

def _nonprintable_ratio(s: str) -> float:
    """Fraction of characters in *s* that are non-printable (control chars)."""
    if not s:
        return 0.0
    count = sum(
        1 for ch in s
        if unicodedata.category(ch).startswith("C")   # Cc, Cf, Cs, Co, Cn
    )
    return count / len(s)


def _utf8_roundtrip_ok(s: str) -> bool:
    try:
        return s.encode("utf-8").decode("utf-8") == s
    except Exception:
        return False


def validate(df: pd.DataFrame, source_note: str) -> dict:
    print("\n🔎  Running data-quality validation …")

    total = len(df)
    text_col = "text"

    # ── schema ────────────────────────────────────────────────────────────────
    actual_cols = set(df.columns)
    schema_ok = EXPECTED_COLUMNS.issubset(actual_cols)
    missing_cols = EXPECTED_COLUMNS - actual_cols

    # ── nulls ─────────────────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_pct = (null_counts / total * 100).round(2)

    # ── duplicates ────────────────────────────────────────────────────────────
    dup_count = int(df.duplicated(subset=[text_col]).sum())
    dup_rate = round(dup_count / total * 100, 2)

    # ── empty strings (not null) ──────────────────────────────────────────────
    empty_count = int((df[text_col].fillna("").str.strip() == "").sum())

    # ── text length stats ─────────────────────────────────────────────────────
    lengths = df[text_col].dropna().str.len()
    len_min  = int(lengths.min())
    len_max  = int(lengths.max())
    len_mean = round(float(lengths.mean()), 1)
    len_p25  = int(lengths.quantile(0.25))
    len_p50  = int(lengths.quantile(0.50))
    len_p75  = int(lengths.quantile(0.75))

    # ── encoding checks ───────────────────────────────────────────────────────
    text_series = df[text_col].dropna().astype(str)

    # UTF-8 round-trip
    bad_encoding = int((~text_series.apply(_utf8_roundtrip_ok)).sum())
    encoding_rate = round(bad_encoding / total * 100, 2)

    # Non-printable character ratio
    np_ratios = text_series.apply(_nonprintable_ratio)
    np_flagged = int((np_ratios > NONPRINTABLE_THRESHOLD).sum())
    np_rate = round(np_flagged / total * 100, 2)

    # ── missing-field rate (any column) ──────────────────────────────────────
    any_null_count = int(df.isnull().any(axis=1).sum())
    missing_field_rate = round(any_null_count / total * 100, 2)

    stats = {
        "total": total,
        "schema_ok": schema_ok,
        "missing_cols": missing_cols,
        "null_counts": null_counts,
        "null_pct": null_pct,
        "dup_count": dup_count,
        "dup_rate": dup_rate,
        "empty_count": empty_count,
        "len_min": len_min,
        "len_max": len_max,
        "len_mean": len_mean,
        "len_p25": len_p25,
        "len_p50": len_p50,
        "len_p75": len_p75,
        "bad_encoding": bad_encoding,
        "encoding_rate": encoding_rate,
        "np_flagged": np_flagged,
        "np_rate": np_rate,
        "missing_field_rate": missing_field_rate,
        "source_note": source_note,
        "actual_cols": list(df.columns),
    }

    # Print summary to stdout
    print(f"  Total records       : {total:,}")
    print(f"  Schema OK           : {schema_ok}  (cols: {list(df.columns)})")
    print(f"  Duplicate texts     : {dup_count:,}  ({dup_rate}%)")
    print(f"  Empty-string texts  : {empty_count:,}")
    print(f"  Null in any column  : {any_null_count:,}  ({missing_field_rate}%)")
    print(f"  Bad UTF-8 rows      : {bad_encoding:,}  ({encoding_rate}%)")
    print(f"  High non-printable  : {np_flagged:,}  ({np_rate}%)")
    print(f"  Text length min/avg/max : {len_min} / {len_mean} / {len_max}")

    return stats


# ── report writer ─────────────────────────────────────────────────────────────

def write_report(stats: dict) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nc = stats["null_counts"]
    np_ = stats["null_pct"]

    col_null_rows = "\n".join(
        f"| `{col}` | {nc[col]:,} | {np_[col]:.2f}% |"
        for col in stats["actual_cols"]
    )

    schema_status = "✅ Passed" if stats["schema_ok"] else (
        f"❌ Failed — missing columns: {', '.join(stats['missing_cols'])}"
    )

    report = f"""# Data Quality Report

**Generated** : {ts}  
**Dataset file** : `data/raw/arxiv_summaries.csv`  
**Script** : `scripts/validate_data.py`

---

## Summary Metrics

| Metric | Value |
|---|---|
| **Total records** | {stats['total']:,} |
| **Duplicate rate** (text column) | {stats['dup_rate']}% ({stats['dup_count']:,} rows) |
| **Empty-string texts** | {stats['empty_count']:,} |
| **Missing-field rate** (any column) | {stats['missing_field_rate']}% ({stats['total'] - stats['empty_count']:,} clean rows) |
| **Bad UTF-8 encoding rows** | {stats['bad_encoding']:,} ({stats['encoding_rate']}%) |
| **High non-printable char rows** (>{NONPRINTABLE_THRESHOLD*100:.0f}% threshold) | {stats['np_flagged']:,} ({stats['np_rate']}%) |

---

## Per-Column Null / Missing Breakdown

| Column | Null Count | Null % |
|---|---|---|
{col_null_rows}

---

## Text Length Statistics (`text` column)

| Stat | Characters |
|---|---|
| Minimum | {stats['len_min']:,} |
| 25th percentile | {stats['len_p25']:,} |
| Median | {stats['len_p50']:,} |
| Mean | {stats['len_mean']:,} |
| 75th percentile | {stats['len_p75']:,} |
| Maximum | {stats['len_max']:,} |

---

## Schema Validation

**Expected columns** : `id`, `category`, `text`  
**Actual columns** : {', '.join(f'`{c}`' for c in stats['actual_cols'])}  
**Status** : {schema_status}

---

## Encoding Check

- **UTF-8 round-trip** : {stats['bad_encoding']} row(s) failed encode → decode round-trip  
- **Non-printable character scan** : {stats['np_flagged']} row(s) exceeded {NONPRINTABLE_THRESHOLD*100:.0f}% non-printable threshold  
- Both checks used Python's built-in `unicodedata` module; no external tools required

---

## Notes

{stats['source_note']}

Validation run with Python {sys.version.split()[0]} on {ts}.  
All metrics computed in-memory using pandas; the CSV was written in UTF-8 with no BOM.
"""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n📄  Report saved → {REPORT_PATH}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n🚀  Dataset Acquisition & Validation")
    print("=" * 55)

    df, source_note = acquire_dataset()
    stats = validate(df, source_note)
    write_report(stats)

    print("\n✅  Done.")


if __name__ == "__main__":
    main()