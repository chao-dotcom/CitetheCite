"""Tests for CitationFormatter."""

import pytest
from main import CitationFormatter, Citation


def test_format_apa(sample_citation):
    """Test APA citation formatting."""
    formatter = CitationFormatter()
    result = formatter.format_apa(sample_citation)

    assert "Wikipedia contributors" in result
    assert "Test Article" in result
    assert "2025" in result
    assert sample_citation.url in result


def test_format_mla(sample_citation):
    """Test MLA citation formatting."""
    formatter = CitationFormatter()
    result = formatter.format_mla(sample_citation)

    assert "Test Article" in result
    assert "Wikipedia" in result
    assert sample_citation.url in result


def test_format_bibliography_apa(sample_citation):
    """Test bibliography formatting in APA style."""
    formatter = CitationFormatter()
    citations = [sample_citation]

    result = formatter.format_bibliography(citations, "APA")

    assert len(result) == 1
    assert "Wikipedia contributors" in result[0]


def test_format_bibliography_mla(sample_citation):
    """Test bibliography formatting in MLA style."""
    formatter = CitationFormatter()
    citations = [sample_citation]

    result = formatter.format_bibliography(citations, "MLA")

    assert len(result) == 1
    assert "Test Article" in result[0]


def test_format_bibliography_defaults_to_apa(sample_citation):
    """Test that unknown style defaults to APA."""
    formatter = CitationFormatter()
    citations = [sample_citation]

    result = formatter.format_bibliography(citations, "UNKNOWN")

    assert len(result) == 1
    assert "Wikipedia contributors" in result[0]  # APA format

