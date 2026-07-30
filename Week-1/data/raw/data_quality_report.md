# Data Quality Report

**Generated** : 2026-07-21 10:29 UTC  
**Dataset file** : `data/raw/arxiv_summaries.csv`  
**Script** : `scripts/validate_data.py`

---

## Summary Metrics

| Metric | Value |
|---|---|
| **Total records** | 10,788 |
| **Duplicate rate** (text column) | 1.21% (131 rows) |
| **Empty-string texts** | 0 |
| **Missing-field rate** (any column) | 0.0% (10,788 clean rows) |
| **Bad UTF-8 encoding rows** | 0 (0.0%) |
| **High non-printable char rows** (>5% threshold) | 0 (0.0%) |

---

## Per-Column Null / Missing Breakdown

| Column | Null Count | Null % |
|---|---|---|
| `id` | 0 | 0.00% |
| `category` | 0 | 0.00% |
| `text` | 0 | 0.00% |

---

## Text Length Statistics (`text` column)

| Stat | Characters |
|---|---|
| Minimum | 25 |
| 25th percentile | 275 |
| Median | 531 |
| Mean | 815.0 |
| 75th percentile | 1,019 |
| Maximum | 14,055 |

---

## Schema Validation

**Expected columns** : `id`, `category`, `text`  
**Actual columns** : `id`, `category`, `text`  
**Status** : ✅ Passed

---

## Encoding Check

- **UTF-8 round-trip** : 0 row(s) failed encode → decode round-trip  
- **Non-printable character scan** : 0 row(s) exceeded 5% non-printable threshold  
- Both checks used Python's built-in `unicodedata` module; no external tools required

---

## Notes

**Fallback source used**: NLTK Reuters-21578 corpus (10,788 news articles). Primary HuggingFace source was unreachable at acquisition time.

Validation run with Python 3.14.6 on 2026-07-21 10:29 UTC.  
All metrics computed in-memory using pandas; the CSV was written in UTF-8 with no BOM.
