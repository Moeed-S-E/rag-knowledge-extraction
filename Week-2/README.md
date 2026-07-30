# Week 2: Data Cleaning & Preprocessing

## Objectives

- Implement robust text cleaning functions (removing HTML tags, control/special characters, normalizing whitespace).
- Handle edge cases in text data: empty/whitespace text, extremely long documents, and mixed/multilingual scripts.
- Perform spaCy tokenization and lemmatization on a subset of the dataset.
- Write unit tests covering text cleaning edge cases.
- Output clean dataset ready for embedding (`data/processed/clean_dataset.csv`) and log the percentage of data dropped during cleaning.

## Structure

- `src/ragkit/ingestion/cleaning.py`: Core cleaning utilities and DataFrame cleaning functions.
- `tests/test_cleaning.py`: Comprehensive unit tests covering HTML stripping, whitespace normalization, special char filtering, and edge cases.
- `Week-2/scripts/clean_data.py`: Main processing script that loads raw data, executes cleaning, runs spaCy NLP subset analysis, exports clean dataset, and logs data drop metrics.
- `Week-2/reports/cleaning_report.md`: Detailed quality report and drop percentage metrics.

## How to Run

```bash
# Run unit tests
uv run python -m unittest discover -s tests

# Run data cleaning pipeline
uv run python Week-2/scripts/clean_data.py
```

## Results

- **Output Clean Dataset**: `data/processed/clean_dataset.csv`
- **Data Drop Metrics**: Logged and formatted in `Week-2/reports/cleaning_report.md`.
