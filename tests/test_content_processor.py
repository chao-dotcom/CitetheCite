"""Tests for ContentProcessor."""

from main import ContentProcessor


def test_extract_paragraphs():
    """Test paragraph extraction from HTML."""
    html_text = "<p>First paragraph with enough text.</p><p>Second paragraph.</p>"
    processor = ContentProcessor()

    result = processor.extract_paragraphs(html_text, max_paragraphs=2)

    assert len(result) > 0
    assert "First paragraph" in result[0]


def test_create_citation():
    """Test citation creation from page data."""
    page_data = {
        "title": "Test Article",
        "revid": "12345",
        "displaytitle": "Test Article",
    }
    processor = ContentProcessor()

    citation = processor.create_citation(page_data, lang="en")

    assert citation.title == "Test Article"
    assert citation.revision_id == "12345"
    assert "wikipedia.org" in citation.url
    assert citation.source_id.startswith("wiki_")

