"""Tests for WikipediaClient."""

import pytest
from main import WikipediaClient


@pytest.mark.asyncio
async def test_extract_related_links_filters_meta_pages():
    """Test that meta pages are filtered from related links."""
    page_data = {
        "links": [
            {"title": "Python", "exists": True, "ns": 0},
            {"title": "Wikipedia:About", "exists": True, "ns": 4},  # Meta
            {"title": "JavaScript", "exists": True, "ns": 0},
        ]
    }
    client = WikipediaClient()

    result = client.extract_related_links(page_data, max_links=3)

    assert len(result) == 2
    assert "Python" in result
    assert "JavaScript" in result
    assert "Wikipedia:About" not in result


@pytest.mark.asyncio
async def test_extract_related_links_respects_max_links():
    """Test that max_links limit is respected."""
    page_data = {
        "links": [
            {"title": f"Article{i}", "exists": True, "ns": 0}
            for i in range(10)
        ]
    }
    client = WikipediaClient()

    result = client.extract_related_links(page_data, max_links=3)

    assert len(result) == 3


@pytest.mark.asyncio
async def test_extract_related_links_filters_nonexistent():
    """Test that non-existent pages are filtered."""
    page_data = {
        "links": [
            {"title": "Python", "exists": True, "ns": 0},
            {"title": "Nonexistent", "exists": False, "ns": 0},
        ]
    }
    client = WikipediaClient()

    result = client.extract_related_links(page_data, max_links=5)

    assert "Python" in result
    assert "Nonexistent" not in result


@pytest.mark.asyncio
async def test_wikipedia_client_close():
    """Test that client can be closed."""
    client = WikipediaClient()
    await client.close()  # Should not raise

