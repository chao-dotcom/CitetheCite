"""
Wikipedia Report Generator API
A FastAPI service that generates citation-backed reports from Wikipedia articles.
"""

import asyncio
import functools
import hashlib
import html
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================


class Config:
    """Configuration constants for the application."""

    # API Settings
    SERVICE_NAME = os.getenv("SERVICE_NAME", "WikiReportAPI")
    VERSION = "1.0.0"
    CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "contact@example.com")

    # Rate Limiting
    WIKIPEDIA_CALLS_PER_SECOND = 1.0
    CLIENT_REQUESTS_PER_MINUTE = 10

    # Timeouts (seconds)
    WIKIPEDIA_TIMEOUT = 15.0
    LLM_TIMEOUT = 30.0
    TOTAL_REQUEST_TIMEOUT = 60.0

    # LLM Settings
    OPENAI_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 2000  # Note: Guide shows 200, but increased for better summaries

    # Content Limits
    MAX_PARAGRAPH_LENGTH = 500
    MAX_RELATED_LINKS = 5


# ============================================================================
# Data Models
# ============================================================================


class Citation(BaseModel):
    """Citation model for Wikipedia sources."""

    source_id: str
    title: str
    url: str
    revision_id: str
    accessed_date: str


class ReportSection(BaseModel):
    """Section of the generated report."""

    heading: str
    content: str
    citations: List[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    """Request model for report generation."""

    url: str
    citation_style: Literal["APA", "MLA"] = "APA"
    max_related_links: int = Field(default=3, ge=0, le=5)
    report_length: Literal["short", "medium"] = "medium"

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate Wikipedia URL format."""
        pattern = r"https?://([a-z]{2,3})\.wikipedia\.org/wiki/([^?#]+)"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid Wikipedia URL format. "
                "Expected: https://en.wikipedia.org/wiki/Article_Title"
            )
        return v


class GenerateResponse(BaseModel):
    """Response model for generated reports."""

    success: bool
    title: str
    summary: str
    sections: List[ReportSection] = Field(default_factory=list)
    sources: List[Citation] = Field(default_factory=list)
    bibliography: List[str] = Field(default_factory=list)
    generated_at: str
    processing_time_seconds: float
    metadata: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict] = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: str
    config: Dict


# ============================================================================
# Utilities
# ============================================================================


def rate_limiter(calls_per_second: float):
    """Rate limiting decorator for async functions."""

    def decorator(func):
        last_called = [0.0]

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            min_interval = 1.0 / calls_per_second
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            last_called[0] = time.time()
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def extract_wikipedia_title(url: str) -> Tuple[str, str]:
    """
    Extract title and language from Wikipedia URL.

    Args:
        url: Wikipedia URL

    Returns:
        Tuple of (title, language_code)

    Raises:
        ValueError: If URL format is invalid
    """
    pattern = r"https?://([a-z]{2,3})\.wikipedia\.org/wiki/([^?#]+)"
    match = re.match(pattern, url)
    if not match:
        raise ValueError("Invalid Wikipedia URL format")

    lang = match.group(1)
    title = match.group(2).replace("_", " ")
    return title, lang


# ============================================================================
# Core Components
# ============================================================================


class WikipediaClient:
    """Client for interacting with Wikipedia MediaWiki API."""

    def __init__(self):
        self.session = httpx.AsyncClient(
            timeout=Config.WIKIPEDIA_TIMEOUT,
            headers={
                "User-Agent": f"{Config.SERVICE_NAME}/{Config.VERSION} ({Config.CONTACT_EMAIL})"
            },
        )
        self._rate_limited_fetch = rate_limiter(Config.WIKIPEDIA_CALLS_PER_SECOND)(
            self._fetch_api
        )

    async def _fetch_api(self, url: str, params: dict) -> dict:
        """Make API request to Wikipedia."""
        try:
            response = await self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            logger.error(f"Wikipedia API timeout: {url}")
            raise HTTPException(status_code=504, detail="Wikipedia API timeout")
        except httpx.HTTPStatusError as e:
            logger.error(f"Wikipedia API error: {e}")
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Page not found")
            raise HTTPException(status_code=502, detail=f"Wikipedia API error: {e}")

    @rate_limiter(Config.WIKIPEDIA_CALLS_PER_SECOND)
    async def fetch_page(self, title: str, lang: str = "en") -> dict:
        """
        Fetch Wikipedia page content via MediaWiki API.

        Args:
            title: Wikipedia page title
            lang: Language code (default: "en")

        Returns:
            Parsed page data including content and metadata
        """
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "parse",
            "page": title,
            "format": "json",
            "prop": "text|revid|displaytitle|links",
        }

        logger.debug(f"Fetching page: {title} (lang: {lang})")
        data = await self._rate_limited_fetch(api_url, params)

        if "error" in data:
            raise HTTPException(
                status_code=404, detail=f"Page not found: {title}"
            )

        parse_data = data.get("parse", {})
        return {
            "title": parse_data.get("title", title),
            "text": parse_data.get("text", {}).get("*", ""),
            "revid": str(parse_data.get("revid", "")),
            "displaytitle": parse_data.get("displaytitle", title),
            "links": parse_data.get("links", []),
        }

    def extract_related_links(
        self, page_data: dict, max_links: int = 3
    ) -> List[str]:
        """
        Extract related article links from page data.

        Args:
            page_data: Page data from fetch_page
            max_links: Maximum number of links to return

        Returns:
            List of article titles (filtered to exclude meta pages)
        """
        links = page_data.get("links", [])
        related = []

        for link in links:
            # Filter: only article namespace (ns=0), exists, not meta pages
            if (
                link.get("ns") == 0
                and link.get("exists")
                and not link.get("title", "").startswith("Wikipedia:")
                and link.get("title") not in related
            ):
                related.append(link.get("title"))

            if len(related) >= max_links:
                break

        return related

    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()


class ContentProcessor:
    """Processor for extracting and cleaning Wikipedia content."""

    @staticmethod
    def extract_paragraphs(html_text: str, max_paragraphs: int = 5) -> List[str]:
        """
        Extract clean paragraphs from Wikipedia HTML.

        Args:
            html_text: Raw HTML from Wikipedia
            max_paragraphs: Maximum number of paragraphs to extract

        Returns:
            List of cleaned paragraph strings
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "lxml")
        paragraphs = []

        # Extract paragraph tags
        for p in soup.find_all("p")[:max_paragraphs]:
            text = p.get_text(strip=True)
            # Clean up: remove extra whitespace
            text = " ".join(text.split())
            # Filter out very short paragraphs (likely stubs)
            if len(text) > 50:
                # Truncate if too long
                if len(text) > Config.MAX_PARAGRAPH_LENGTH:
                    text = text[: Config.MAX_PARAGRAPH_LENGTH] + "..."
                paragraphs.append(text)

        return paragraphs

    @staticmethod
    def create_citation(page_data: dict, lang: str = "en") -> Citation:
        """
        Create a Citation object from page data.

        Args:
            page_data: Page data from WikipediaClient
            lang: Language code

        Returns:
            Citation object
        """
        title = page_data.get("title", "")
        revid = page_data.get("revid", "")
        url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}?oldid={revid}"
        source_id = f"wiki_{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}"

        return Citation(
            source_id=source_id,
            title=page_data.get("displaytitle", title),
            url=url,
            revision_id=revid,
            accessed_date=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )


class LLMGenerator:
    """Generator for LLM-powered summaries with citations."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = Config.OPENAI_MODEL

    async def generate_summary(
        self,
        paragraphs: List[str],
        source_ids: List[str],
        length: str = "medium",
    ) -> str:
        """
        Generate a summary with inline citations using LLM.

        Args:
            paragraphs: List of text paragraphs to summarize
            source_ids: Corresponding source IDs for citations
            length: Report length ("short" or "medium")

        Returns:
            Generated summary text with [source_id] citation markers
        """
        if len(paragraphs) != len(source_ids):
            raise ValueError("paragraphs and source_ids must have the same length")

        # Build prompt
        content_text = "\n\n".join(paragraphs)
        word_limit = 150 if length == "short" else 300

        prompt = f"""Generate a concise, factual summary based on the following Wikipedia content. 
Include inline citations using [source_id] format where you reference information.

Content:
{content_text}

Requirements:
- Length: approximately {word_limit} words
- Include citations in brackets like [wiki_python] when referencing information
- Be factual and objective
- Focus on key points

Summary:"""

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that creates academic-style summaries with proper citations.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=Config.LLM_TEMPERATURE,
                    max_tokens=Config.LLM_MAX_TOKENS,
                ),
                timeout=Config.LLM_TIMEOUT,
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except asyncio.TimeoutError:
            logger.error("LLM API timeout")
            raise HTTPException(status_code=504, detail="LLM generation timeout")
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise HTTPException(
                status_code=500, detail=f"LLM generation failed: {str(e)}"
            )

    async def close(self):
        """Close the OpenAI client."""
        # OpenAI client doesn't need explicit close, but good practice
        pass


class CitationFormatter:
    """Formatter for citations in different styles."""

    @staticmethod
    def format_apa(citation: Citation) -> str:
        """
        Format citation in APA style.

        Args:
            citation: Citation object

        Returns:
            Formatted APA citation string
        """
        date = datetime.fromisoformat(citation.accessed_date.replace("Z", "+00:00"))
        date_str = date.strftime("%B %d, %Y")

        return (
            f'Wikipedia contributors. ({date.year}, {date.strftime("%B %d")}). '
            f'"{citation.title}." Wikipedia, The Free Encyclopedia. '
            f'Retrieved {date_str}, from {citation.url}'
        )

    @staticmethod
    def format_mla(citation: Citation) -> str:
        """
        Format citation in MLA style.

        Args:
            citation: Citation object

        Returns:
            Formatted MLA citation string
        """
        date = datetime.fromisoformat(citation.accessed_date.replace("Z", "+00:00"))
        date_str = date.strftime("%d %b. %Y")

        return (
            f'"{citation.title}." Wikipedia, The Free Encyclopedia, '
            f'{date_str}, {citation.url}.'
        )

    @classmethod
    def format_bibliography(
        cls, citations: List[Citation], style: str
    ) -> List[str]:
        """
        Format a list of citations as a bibliography.

        Args:
            citations: List of Citation objects
            style: Citation style ("APA" or "MLA")

        Returns:
            List of formatted citation strings
        """
        formatters = {"APA": cls.format_apa, "MLA": cls.format_mla}
        formatter = formatters.get(style, cls.format_apa)
        return [formatter(cite) for cite in citations]


class ReportGenerator:
    """Main orchestrator for report generation."""

    def __init__(self, openai_api_key: str):
        self.wiki_client = WikipediaClient()
        self.content_processor = ContentProcessor()
        self.llm_generator = LLMGenerator(openai_api_key)
        self.citation_formatter = CitationFormatter()

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate a complete report from a Wikipedia URL.

        Args:
            request: GenerateRequest with URL and parameters

        Returns:
            GenerateResponse with complete report data
        """
        start_time = time.time()

        try:
            # Extract title and language
            title, lang = extract_wikipedia_title(request.url)

            # Fetch main page
            logger.info(f"Fetching main page: {title}")
            main_page = await self.wiki_client.fetch_page(title, lang)
            main_citation = self.content_processor.create_citation(main_page, lang)

            # Extract related links
            related_titles = self.wiki_client.extract_related_links(
                main_page, max_links=request.max_related_links
            )

            # Fetch related pages in parallel
            logger.info(f"Fetching {len(related_titles)} related pages")
            related_pages = await asyncio.gather(
                *[
                    self.wiki_client.fetch_page(title, lang)
                    for title in related_titles
                ],
                return_exceptions=True,
            )

            # Process all pages
            all_citations = [main_citation]
            all_paragraphs = []
            all_source_ids = []

            # Process main page
            paragraphs = self.content_processor.extract_paragraphs(
                main_page["text"], max_paragraphs=5
            )
            for para in paragraphs:
                all_paragraphs.append(para)
                all_source_ids.append(main_citation.source_id)

            # Process related pages
            for page_data in related_pages:
                if isinstance(page_data, Exception):
                    logger.warning(f"Failed to fetch related page: {page_data}")
                    continue

                citation = self.content_processor.create_citation(page_data, lang)
                all_citations.append(citation)

                paragraphs = self.content_processor.extract_paragraphs(
                    page_data["text"], max_paragraphs=3
                )
                for para in paragraphs:
                    all_paragraphs.append(para)
                    all_source_ids.append(citation.source_id)

            # Generate summary with LLM
            logger.info("Generating summary with LLM")
            summary = await self.llm_generator.generate_summary(
                all_paragraphs, all_source_ids, length=request.report_length
            )

            # Format bibliography
            bibliography = self.citation_formatter.format_bibliography(
                all_citations, request.citation_style
            )

            # Extract citations from summary
            citation_pattern = r"\[(\w+)\]"
            used_citations = list(set(re.findall(citation_pattern, summary)))

            # Create sections (simple version - can be enhanced)
            sections = [
                ReportSection(
                    heading="Summary",
                    content=summary,
                    citations=used_citations,
                )
            ]

            processing_time = time.time() - start_time

            return GenerateResponse(
                success=True,
                title=main_page.get("displaytitle", title),
                summary=summary,
                sections=sections,
                sources=all_citations,
                bibliography=bibliography,
                generated_at=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                processing_time_seconds=round(processing_time, 2),
                metadata={
                    "main_page_revision": main_page.get("revid", ""),
                    "related_pages_count": str(len([p for p in related_pages if not isinstance(p, Exception)])),
                    "total_sources": str(len(all_citations)),
                    "language": lang,
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Report generation failed: {str(e)}"
            )

    async def close(self):
        """Close all client connections."""
        await self.wiki_client.close()
        await self.llm_generator.close()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=Config.SERVICE_NAME,
    version=Config.VERSION,
    description="Generate citation-backed reports from Wikipedia articles",
)


# Global generator instance
generator: Optional[ReportGenerator] = None


@app.on_event("startup")
async def startup():
    """Initialize application on startup."""
    global generator
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    generator = ReportGenerator(api_key)
    logger.info(f"{Config.SERVICE_NAME} v{Config.VERSION} started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on application shutdown."""
    global generator
    if generator:
        await generator.close()
    logger.info("Application shutdown complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": Config.SERVICE_NAME,
        "version": Config.VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        config={
            "max_related_links": str(Config.MAX_RELATED_LINKS),
            "wikipedia_rate_limit": f"{Config.WIKIPEDIA_CALLS_PER_SECOND}/s",
            "llm_model": Config.OPENAI_MODEL,
            "version": Config.VERSION,
        },
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_report(request: GenerateRequest):
    """
    Generate a citation-backed report from a Wikipedia URL.

    Args:
        request: GenerateRequest with URL and parameters

    Returns:
        GenerateResponse with complete report
    """
    if not generator:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        async with asyncio.timeout(Config.TOTAL_REQUEST_TIMEOUT):
            response = await generator.generate(request)
            return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

