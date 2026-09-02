from rag_app.cleaning import clean_records, clean_text


def test_clean_text_removes_markup_and_normalizes_whitespace():
    assert clean_text("<p>Hello&nbsp; world</p>\n\nnext") == "Hello world next"


def test_clean_text_preserves_mixed_language_text():
    value = clean_text("English 中文 العربية")
    assert "中文" in value and "العربية" in value


def test_clean_text_handles_empty_and_long_values():
    assert clean_text(None) == ""
    result = clean_text("word " * 100, max_characters=100)
    assert len(result) <= 100


def test_clean_records_drops_empty_text():
    cleaned, report = clean_records([{"id": "1", "text": "ok"}, {"id": "2", "text": "   "}])
    assert len(cleaned) == 1
    assert report["dropped_records"] == 1
