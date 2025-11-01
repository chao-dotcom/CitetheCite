"""Shared pytest fixtures."""

import pytest
from main import Citation


@pytest.fixture
def sample_citation():
    """Reusable citation fixture."""
    return Citation(
        source_id="wiki_test",
        title="Test Article",
        url="https://en.wikipedia.org/wiki/Test?oldid=123",
        revision_id="123",
        accessed_date="2025-10-31T00:00:00Z",
    )


@pytest.fixture
def mock_wikipedia_response():
    """Mock Wikipedia API response."""
    return {
        "parse": {
            "title": "Python (programming language)",
            "text": {"*": "<p>Python is a programming language...</p>"},
            "revid": 123456,
            "displaytitle": "Python (programming language)",
            "links": [
                {"title": "Programming", "exists": True, "ns": 0},
                {"title": "Wikipedia:About", "exists": True, "ns": 4},
            ],
        }
    }

