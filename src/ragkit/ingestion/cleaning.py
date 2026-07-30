"""Text cleaning and normalization utilities for document preprocessing."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Regular expressions for cleaning
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTIPLE_SPACES_RE = re.compile(r"\s+")
# Remove unwanted control characters except newlines/tabs when desired
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def remove_html_tags(text: str) -> str:
    """Strip HTML/XML tags from text and unescape HTML entities."""
    if not text:
        return ""
    # Remove HTML tags first
    text = HTML_TAG_RE.sub(" ", text)
    # Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
    text = html.unescape(text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespaces, tabs, and newlines into single spaces."""
    if not text:
        return ""
    return MULTIPLE_SPACES_RE.sub(" ", text).strip()


def remove_special_characters(
    text: str, keep_punctuation: bool = True
) -> str:
    """Remove control characters and optionally non-alphanumeric special characters."""
    if not text:
        return ""
    # Remove control characters
    text = CONTROL_CHAR_RE.sub("", text)
    # Normalize unicode (NFKC normalizes compatibility characters)
    text = unicodedata.normalize("NFKC", text)

    if not keep_punctuation:
        # Retain alphanumeric characters and spaces
        text = re.sub(r"[^\w\s]", "", text)

    return text


def clean_text(
    text: Optional[str],
    min_length: int = 10,
    max_length: Optional[int] = 10000,
    keep_punctuation: bool = True,
    truncate_long: bool = False,
) -> Optional[str]:
    """
    Clean a text string handling edge cases: empty input, HTML, whitespace, 
    special characters, and length constraints.

    Args:
        text: Input string to clean.
        min_length: Minimum character length required after cleaning.
        max_length: Maximum character length allowed.
        keep_punctuation: Whether to retain standard punctuation.
        truncate_long: If True, truncate text exceeding max_length instead of discarding.

    Returns:
        Cleaned text string, or None if the text fails edge case validation (e.g., empty or out of bounds).
    """
    if text is None or not isinstance(text, str):
        return None

    # Step 1: Remove HTML tags & unescape entities
    cleaned = remove_html_tags(text)

    # Step 2: Remove control chars & normalize unicode
    cleaned = remove_special_characters(cleaned, keep_punctuation=keep_punctuation)

    # Step 3: Normalize whitespace
    cleaned = normalize_whitespace(cleaned)

    # Step 4: Handle empty text edge case
    if not cleaned:
        return None

    # Step 5: Handle length edge cases
    if len(cleaned) < min_length:
        return None

    if max_length and len(cleaned) > max_length:
        if truncate_long:
            cleaned = cleaned[:max_length].rsplit(" ", 1)[0]
        else:
            return None

    return cleaned


def process_spacy_nlp(
    texts: List[str],
    model_name: str = "en_core_web_sm",
    batch_size: int = 50,
) -> List[Dict[str, Any]]:
    """
    Run spaCy tokenization and lemmatization on a list of texts.

    Args:
        texts: List of clean text strings.
        model_name: spaCy model to use.
        batch_size: Processing batch size.

    Returns:
        List of dicts containing tokens, lemmas, and entity counts per text.
    """
    try:
        import spacy
    except ImportError:
        raise ImportError("spaCy is required for process_spacy_nlp. Please install spacy.")

    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Fallback to blank model if full model disabled/missing
        nlp = spacy.blank("en")

    results = []
    docs = nlp.pipe(texts, batch_size=batch_size)

    for doc in docs:
        tokens = [token.text for token in doc if not token.is_space]
        lemmas = [token.lemma_ for token in doc if not token.is_space and not token.is_punct]
        num_sentences = len(list(doc.sents)) if doc.has_annotation("SENT_START") else 1

        results.append(
            {
                "num_tokens": len(tokens),
                "num_lemmas": len(set(lemmas)),
                "sample_tokens": tokens[:10],
                "sample_lemmas": lemmas[:10],
                "num_sentences": num_sentences,
            }
        )

    return results


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    min_length: int = 10,
    max_length: Optional[int] = 10000,
    truncate_long: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean a pandas DataFrame text column, drop invalid/duplicate rows, and return statistics.

    Returns:
        Tuple of (cleaned DataFrame, summary dictionary containing drop percentages and counts).
    """
    initial_count = len(df)
    if initial_count == 0:
        return df, {
            "initial_count": 0,
            "final_count": 0,
            "dropped_count": 0,
            "drop_percentage": 0.0,
        }

    df_clean = df.copy()

    # Apply text cleaning
    df_clean["cleaned_text"] = df_clean[text_column].apply(
        lambda t: clean_text(
            t,
            min_length=min_length,
            max_length=max_length,
            truncate_long=truncate_long,
        )
    )

    # Filter out rows where cleaned_text is None
    null_after_clean = df_clean["cleaned_text"].isnull().sum()
    df_clean = df_clean.dropna(subset=["cleaned_text"])

    # Deduplicate based on cleaned_text
    duplicates_count = df_clean.duplicated(subset=["cleaned_text"]).sum()
    df_clean = df_clean.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)

    # Overwrite original column or keep cleaned_text as text
    df_clean[text_column] = df_clean["cleaned_text"]
    df_clean = df_clean.drop(columns=["cleaned_text"])

    final_count = len(df_clean)
    dropped_count = initial_count - final_count
    drop_percentage = round((dropped_count / initial_count) * 100, 2)

    stats = {
        "initial_count": initial_count,
        "final_count": final_count,
        "dropped_count": dropped_count,
        "dropped_null_or_invalid": int(null_after_clean),
        "dropped_duplicates": int(duplicates_count),
        "drop_percentage": drop_percentage,
    }

    return df_clean, stats
