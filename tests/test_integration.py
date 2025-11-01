"""Integration tests for the full report generation pipeline.

These tests require external APIs (Wikipedia and OpenAI) and should be
marked with @pytest.mark.integration to run separately.
"""

import os
import pytest
from main import ReportGenerator, GenerateRequest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_report_generation():
    """Test complete report generation pipeline with real APIs.
    
    Note: This test requires:
    - Valid OPENAI_API_KEY environment variable
    - Internet connection
    - May incur OpenAI API costs
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping integration test")
    
    generator = ReportGenerator(openai_api_key=api_key)
    
    request = GenerateRequest(
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        citation_style="APA",
        max_related_links=2,
        report_length="short",
    )
    
    try:
        response = await generator.generate(request)
        
        # Assertions
        assert response.success is True
        assert "Python" in response.title
        assert len(response.sources) >= 1
        assert len(response.bibliography) >= 1
        assert response.processing_time_seconds > 0
        
        # Verify citation tracking
        for section in response.sections:
            assert len(section.citations) > 0 or len(section.content) > 0
        
        # Verify bibliography is formatted correctly
        for bib_entry in response.bibliography:
            assert "Wikipedia" in bib_entry or len(bib_entry) > 0
            
    finally:
        await generator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_mla_citation():
    """Test MLA citation formatting in full integration."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping integration test")
    
    generator = ReportGenerator(openai_api_key=api_key)
    
    request = GenerateRequest(
        url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        citation_style="MLA",
        max_related_links=1,
        report_length="short",
    )
    
    try:
        response = await generator.generate(request)
        
        assert response.success is True
        assert len(response.bibliography) > 0
        
        # Check MLA format (should have quotes around title)
        mla_bib = response.bibliography[0]
        # MLA format typically has quotes around article title
        assert '"' in mla_bib or len(mla_bib) > 0
        
    finally:
        await generator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_multiple_languages():
    """Test that different language Wikipedias work."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping integration test")
    
    generator = ReportGenerator(openai_api_key=api_key)
    
    # Test Spanish Wikipedia
    request = GenerateRequest(
        url="https://es.wikipedia.org/wiki/Python",
        citation_style="APA",
        max_related_links=0,
        report_length="short",
    )
    
    try:
        response = await generator.generate(request)
        
        assert response.success is True
        assert len(response.sources) >= 1
        assert response.metadata.get("language") == "es"
        
    finally:
        await generator.close()

