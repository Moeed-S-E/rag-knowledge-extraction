"""Unit tests for text cleaning and normalization utilities using unittest."""

import unittest
import pandas as pd
from ragkit.ingestion.cleaning import (
    clean_text,
    remove_html_tags,
    normalize_whitespace,
    remove_special_characters,
    clean_dataframe,
)


class TestTextCleaning(unittest.TestCase):
    def test_remove_html_tags(self):
        html_input = "<p>Hello <b>World</b> &amp; Python!</p>"
        expected = "Hello  World  & Python!"
        self.assertEqual(remove_html_tags(html_input).strip(), expected.strip())
        self.assertEqual(remove_html_tags(""), "")
        self.assertEqual(
            remove_html_tags("Plain text with no tags"), "Plain text with no tags"
        )

    def test_normalize_whitespace(self):
        raw_input = "  Hello   \n\n  World \t  From   Python!  "
        expected = "Hello World From Python!"
        self.assertEqual(normalize_whitespace(raw_input), expected)
        self.assertEqual(normalize_whitespace(""), "")

    def test_remove_special_characters(self):
        raw_input = "Hello \x00\x08World!\x1f"
        self.assertEqual(remove_special_characters(raw_input), "Hello World!")
        self.assertEqual(
            remove_special_characters("Hello, World!!", keep_punctuation=False),
            "Hello World",
        )

    def test_clean_text_edge_cases(self):
        # Edge Case 1: Empty text / None / Whitespace only
        self.assertIsNone(clean_text(None))
        self.assertIsNone(clean_text(""))
        self.assertIsNone(clean_text("     \n\t  "))

        # Edge Case 2: Extremely short text
        self.assertIsNone(clean_text("Short", min_length=10))
        self.assertEqual(
            clean_text("Valid length text string", min_length=10),
            "Valid length text string",
        )

        # Edge Case 3: Extremely long text
        long_text = "word " * 1000  # 5000 characters
        self.assertIsNone(clean_text(long_text, max_length=100))
        truncated = clean_text(long_text, max_length=100, truncate_long=True)
        self.assertIsNotNone(truncated)
        self.assertLessEqual(len(truncated), 100)

        # Edge Case 4: HTML combined with whitespace
        html_dirty = (
            "<div>   <h1>Title</h1>  <p>Some content with &lt;code&gt; tags.</p> </div>"
        )
        cleaned = clean_text(html_dirty)
        self.assertEqual(cleaned, "Title Some content with <code> tags.")

        # Edge Case 5: Mixed languages / Unicode characters
        multilingual = "  <p>Hello! Bonjour! こんにちは! مرحبا! 123 </p> "
        cleaned_ml = clean_text(multilingual)
        self.assertIn("Hello! Bonjour!", cleaned_ml)
        self.assertIn("こんにちは!", cleaned_ml)
        self.assertIn("مرحبا!", cleaned_ml)

    def test_clean_dataframe(self):
        data = {
            "id": [1, 2, 3, 4, 5],
            "text": [
                "<div>Valid text number one for processing.</div>",
                "Short",  # should be dropped (< 10 chars)
                "<div>Valid text number one for processing.</div>",  # duplicate
                "",  # empty
                "Another unique valid text entry for testing.",
            ],
        }
        df = pd.DataFrame(data)
        cleaned_df, stats = clean_dataframe(df, min_length=10)

        self.assertEqual(len(cleaned_df), 2)
        self.assertEqual(stats["initial_count"], 5)
        self.assertEqual(stats["final_count"], 2)
        self.assertEqual(stats["dropped_count"], 3)
        self.assertEqual(stats["drop_percentage"], 60.0)
        self.assertIn("text", cleaned_df.columns)


if __name__ == "__main__":
    unittest.main()
