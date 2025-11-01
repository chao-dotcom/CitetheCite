"""Tests for LLMGenerator."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from main import LLMGenerator, HTTPException


@pytest.mark.asyncio
async def test_generate_summary_success():
    """Test successful summary generation."""
    # Mock OpenAI client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a generated summary [wiki_test]"
    
    generator = LLMGenerator(api_key="test-key")
    generator.client = mock_client
    generator.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    paragraphs = ["Python is a programming language.", "It was created by Guido van Rossum."]
    source_ids = ["wiki_python", "wiki_guido"]
    
    result = await generator.generate_summary(paragraphs, source_ids, length="medium")
    
    assert result == "This is a generated summary [wiki_test]"
    assert generator.client.chat.completions.create.called


@pytest.mark.asyncio
async def test_generate_summary_length_mismatch():
    """Test that mismatched paragraph and source_id lengths raise ValueError."""
    generator = LLMGenerator(api_key="test-key")
    
    paragraphs = ["Paragraph 1", "Paragraph 2"]
    source_ids = ["wiki_one"]  # Mismatched length
    
    with pytest.raises(ValueError, match="must have the same length"):
        await generator.generate_summary(paragraphs, source_ids, length="short")


@pytest.mark.asyncio
async def test_generate_summary_timeout():
    """Test that timeout errors are handled."""
    generator = LLMGenerator(api_key="test-key")
    generator.client = AsyncMock()
    
    # Simulate timeout
    import asyncio
    generator.client.chat.completions.create = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )
    
    paragraphs = ["Test paragraph"]
    source_ids = ["wiki_test"]
    
    with pytest.raises(HTTPException) as exc_info:
        await generator.generate_summary(paragraphs, source_ids, length="medium")
    
    assert exc_info.value.status_code == 504
    assert "timeout" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_generate_summary_api_error():
    """Test that API errors are handled."""
    generator = LLMGenerator(api_key="test-key")
    generator.client = AsyncMock()
    
    # Simulate API error
    generator.client.chat.completions.create = AsyncMock(
        side_effect=Exception("API error")
    )
    
    paragraphs = ["Test paragraph"]
    source_ids = ["wiki_test"]
    
    with pytest.raises(HTTPException) as exc_info:
        await generator.generate_summary(paragraphs, source_ids, length="medium")
    
    assert exc_info.value.status_code == 500
    assert "LLM generation failed" in exc_info.value.detail

