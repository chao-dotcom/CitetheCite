"""Tests for ReportGenerator."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from main import ReportGenerator, GenerateRequest, HTTPException


@pytest.mark.asyncio
async def test_generate_basic_flow():
    """Test basic report generation flow."""
    # Mock all components
    generator = ReportGenerator(openai_api_key="test-key")
    
    # Mock Wikipedia client
    mock_page_data = {
        "title": "Python (programming language)",
        "text": {"*": "<p>Python is a programming language.</p>"},
        "revid": "123456",
        "displaytitle": "Python (programming language)",
        "links": [
            {"title": "Programming", "exists": True, "ns": 0},
        ],
    }
    
    generator.wiki_client.fetch_page = AsyncMock(return_value=mock_page_data)
    generator.wiki_client.extract_related_links = MagicMock(return_value=[])
    
    # Mock LLM generator
    generator.llm_generator.generate_summary = AsyncMock(
        return_value="Python is a programming language [wiki_python_programming_language]."
    )
    
    # Mock content processor
    generator.content_processor.extract_paragraphs = MagicMock(
        return_value=["Python is a programming language."]
    )
    
    request = GenerateRequest(
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        citation_style="APA",
        max_related_links=0,
        report_length="short",
    )
    
    response = await generator.generate(request)
    
    assert response.success is True
    assert "Python" in response.title
    assert len(response.sources) >= 1
    assert len(response.bibliography) >= 1
    assert response.processing_time_seconds > 0
    
    # Cleanup
    await generator.close()


@pytest.mark.asyncio
async def test_generate_with_related_links():
    """Test report generation with related links."""
    generator = ReportGenerator(openai_api_key="test-key")
    
    # Mock main page
    mock_main_page = {
        "title": "Python",
        "text": {"*": "<p>Python is a language.</p>"},
        "revid": "123",
        "displaytitle": "Python",
        "links": [
            {"title": "Programming", "exists": True, "ns": 0},
        ],
    }
    
    # Mock related page
    mock_related_page = {
        "title": "Programming",
        "text": {"*": "<p>Programming is writing code.</p>"},
        "revid": "456",
        "displaytitle": "Programming",
        "links": [],
    }
    
    generator.wiki_client.fetch_page = AsyncMock(
        side_effect=[mock_main_page, mock_related_page]
    )
    generator.wiki_client.extract_related_links = MagicMock(
        return_value=["Programming"]
    )
    
    generator.llm_generator.generate_summary = AsyncMock(
        return_value="Summary [wiki_python] [wiki_programming]"
    )
    
    generator.content_processor.extract_paragraphs = MagicMock(
        side_effect=[
            ["Python is a language."],  # Main page
            ["Programming is writing code."],  # Related page
        ]
    )
    
    request = GenerateRequest(
        url="https://en.wikipedia.org/wiki/Python",
        citation_style="MLA",
        max_related_links=1,
        report_length="medium",
    )
    
    response = await generator.generate(request)
    
    assert response.success is True
    assert len(response.sources) >= 2  # Main + related
    assert len(response.bibliography) >= 2
    
    await generator.close()


@pytest.mark.asyncio
async def test_generate_invalid_url():
    """Test that invalid URLs raise errors."""
    generator = ReportGenerator(openai_api_key="test-key")
    
    request = GenerateRequest(
        url="https://invalid-url.com/page",
        citation_style="APA",
    )
    
    # Should fail at URL validation before even calling generate
    # But if it gets through, should fail in extract_wikipedia_title
    with pytest.raises((ValueError, HTTPException)):
        await generator.generate(request)
    
    await generator.close()


@pytest.mark.asyncio
async def test_generate_page_not_found():
    """Test handling of non-existent Wikipedia pages."""
    generator = ReportGenerator(openai_api_key="test-key")
    
    generator.wiki_client.fetch_page = AsyncMock(
        side_effect=HTTPException(status_code=404, detail="Page not found")
    )
    
    request = GenerateRequest(
        url="https://en.wikipedia.org/wiki/NonexistentPage12345",
        citation_style="APA",
    )
    
    with pytest.raises(HTTPException) as exc_info:
        await generator.generate(request)
    
    assert exc_info.value.status_code == 404
    
    await generator.close()


@pytest.mark.asyncio
async def test_close_cleanup():
    """Test that close() properly cleans up resources."""
    generator = ReportGenerator(openai_api_key="test-key")
    
    generator.wiki_client.close = AsyncMock()
    generator.llm_generator.close = AsyncMock()
    
    await generator.close()
    
    # Both clients should have close called
    generator.wiki_client.close.assert_called_once()
    # LLMGenerator.close might not be implemented, so we check if it exists
    if hasattr(generator.llm_generator, "close"):
        generator.llm_generator.close.assert_called_once()

