# Week 2: Data Cleaning & Preprocessing Report

**Dataset Input** : `data/raw/arxiv_summaries.csv`  
**Dataset Output** : `data/processed/clean_dataset.csv`  

---

## Cleaning Summary Metrics

| Metric | Value |
|---|---|
| **Initial Records** | 10 |
| **Clean Records** | 6 |
| **Total Dropped** | 4 |
| **Invalid / Empty / Out-of-Bounds Dropped** | 3 |
| **Duplicates Dropped** | 1 |
| **Data Drop Percentage** | **40.0%** |

---

## Operations Performed

1. **HTML Tag & Entity Stripping**: Removed tags (`<p>`, `<div>`, `<script>`) and unescaped HTML entities (`&amp;`, `&quot;`).
2. **Control Character & Special Character Cleaning**: Filtered non-printable ASCII control bytes (`\x00-\x1f`) and normalized Unicode to NFKC format.
3. **Whitespace Normalization**: Collapsed consecutive spaces, tabs, and newlines into single spaces.
4. **Edge Case Validation**:
   - Filtered empty and whitespace-only text inputs.
   - Filtered text under 15 characters (too short for meaningful embedding).
   - Filtered extremely long documents (> 10,000 characters).
   - Preserved mixed-language and multilingual scripts (e.g. CJK, Arabic, French).
5. **Deduplication**: Removed exact duplicate clean text strings.

### spaCy Subset Tokenization / Lemmatization Sample

- **Doc 1**: 13 tokens, 11 unique lemmas
  - *Sample Tokens*: `['Deep', 'Learning', 'in', 'Computer', 'Vision']`
  - *Sample Lemmas*: `['Deep', 'Learning', 'in', 'Computer', 'Vision']`
- **Doc 2**: 16 tokens, 13 unique lemmas
  - *Sample Tokens*: `['Natural', 'Language', 'Processing', '(', 'NLP']`
  - *Sample Lemmas*: `['Natural', 'Language', 'Processing', 'NLP', 'enable']`
- **Doc 3**: 17 tokens, 11 unique lemmas
  - *Sample Tokens*: `['Multilingual', 'Text', ':', 'Hello', '!']`
  - *Sample Lemmas*: `['multilingual', 'text', 'hello', 'bonjour', 'こんにちは']`
- **Doc 4**: 12 tokens, 8 unique lemmas
  - *Sample Tokens*: `['Control', 'chars', ':', 'Sanitized', 'text']`
  - *Sample Lemmas*: `['control', 'char', 'sanitize', 'text', 'remove']`
- **Doc 5**: 14 tokens, 9 unique lemmas
  - *Sample Tokens*: `["alert('xss", "'", ')', 'Clean', 'text']`
  - *Sample Lemmas*: `["alert('xss", 'clean', 'text', 'with', 'HTML']`

---

## Status
✅ Dataset is clean, validated, and ready for vector embedding in Week 3.
