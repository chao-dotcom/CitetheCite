"""Tests for utility functions."""

import pytest
from main import extract_wikipedia_title


def test_extract_wikipedia_title_en():
    """Test extracting title from English Wikipedia URL."""
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    title, lang = extract_wikipedia_title(url)
    assert title == "Python (programming language)"
    assert lang == "en"


def test_extract_wikipedia_title_es():
    """Test extracting title from Spanish Wikipedia URL."""
    url = "https://es.wikipedia.org/wiki/Python"
    title, lang = extract_wikipedia_title(url)
    assert title == "Python"
    assert lang == "es"


def test_extract_wikipedia_title_with_underscores():
    """Test extracting title with underscores."""
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    title, lang = extract_wikipedia_title(url)
    assert title == "Artificial intelligence"
    assert lang == "en"


def test_extract_wikipedia_title_invalid_url():
    """Test that invalid URLs raise ValueError."""
    with pytest.raises(ValueError, match="Invalid Wikipedia URL"):
        extract_wikipedia_title("not-a-url")

    with pytest.raises(ValueError, match="Invalid Wikipedia URL"):
        extract_wikipedia_title("https://example.com/wiki/Test")

