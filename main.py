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
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
# Use absolute path to ensure .env is found regardless of working directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

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
        if not isinstance(v, str) or not v.strip():
            raise ValueError("URL must be a non-empty string")
        
        # Normalize: remove trailing slash, handle URL encoding
        v = v.strip().rstrip('/')
        
        # Pattern: https?://(language).wikipedia.org/wiki/(article)
        # Language code: 2-3 lowercase letters
        # Article: any characters except ? and # (query/fragment)
        pattern = r"https?://([a-z]{2,3})\.wikipedia\.org/wiki/([^?#]+)"
        
        if not re.match(pattern, v, re.IGNORECASE):
            raise ValueError(
                f"Invalid Wikipedia URL format: '{v}'. "
                "Expected format: https://en.wikipedia.org/wiki/Article_Title "
                "(or https://[lang].wikipedia.org/wiki/Article_Title)"
            )
        
        # Normalize to lowercase for language code
        return re.sub(r"(https?://)([a-z]{2,3})(\.wikipedia\.org)", 
                     lambda m: m.group(1) + m.group(2).lower() + m.group(3), 
                     v, flags=re.IGNORECASE)


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
            "prop": "text|revid|displaytitle|links|sections",
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
            "sections": parse_data.get("sections", []),
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
    def extract_references(html_text: str) -> List[Dict[str, str]]:
        """
        Extract references/citations from Wikipedia article HTML.
        
        Args:
            html_text: Raw HTML from Wikipedia article
            
        Returns:
            List of reference dictionaries with id, content, and url (deduplicated and filtered)
        """
        from bs4 import BeautifulSoup
        import hashlib
        
        soup = BeautifulSoup(html_text, "lxml")
        references = []
        seen_urls = set()
        seen_content_hashes = set()
        
        # Find the References section (usually in <ol class="references"> or <div class="reflist">)
        ref_sections = soup.find_all(["ol", "div"], class_=lambda x: x and ("reference" in x.lower() or "reflist" in x.lower()))
        
        # Extract individual references
        for ref_section in ref_sections:
            # Find <li> tags within reference sections
            for li in ref_section.find_all("li", recursive=True):
                ref_id = li.get("id", "")
                # Extract reference content
                ref_text = li.get_text(strip=True)
                
                if not ref_text or len(ref_text) < 10:
                    continue
                
                # Try to extract URL if present - prioritize external links
                url = ""
                # First try to find external link
                link = li.find("a", class_=lambda x: x and "external" in x.lower() if x else False)
                if not link:
                    # Fallback to any link
                    link = li.find("a", href=True)
                
                if link:
                    url = link.get("href", "")
                    # Make absolute URL if relative
                    if url.startswith("//"):
                        url = "https:" + url
                    elif url.startswith("/"):
                        url = "https://en.wikipedia.org" + url
                
                # Filter out unwanted URLs
                if url:
                    # Skip anchor links (internal Wikipedia references)
                    if url.startswith("#"):
                        url = ""  # Clear invalid URL but keep the reference
                    # Skip Wikipedia/Wikidata URLs (we want original sources, not Wikipedia)
                    elif any(domain in url.lower() for domain in [
                        "wikipedia.org", "wikidata.org", "wikimedia.org"
                    ]):
                        url = ""  # Clear Wikipedia URL but keep the reference
                    # Skip if we've seen this URL before (duplicate)
                    elif url in seen_urls:
                        continue
                    else:
                        seen_urls.add(url)
                
                # Create content hash for deduplication
                content_hash = hashlib.md5(ref_text.encode()).hexdigest()
                if content_hash in seen_content_hashes:
                    continue
                seen_content_hashes.add(content_hash)
                
                references.append({
                    "id": ref_id,
                    "content": ref_text,
                    "url": url,
                })
        
        # Sort references: prioritize those with external URLs
        references.sort(key=lambda x: (not x["url"] or x["url"].startswith("#"), len(x["content"])))
        
        # Limit to top 30 unique, high-quality references
        references = references[:30]
        
        return references

    @staticmethod
    def create_citation_from_reference(ref: Dict[str, str], ref_index: int) -> Citation:
        """
        Create a Citation object from a Wikipedia reference.
        
        Args:
            ref: Reference dictionary with id, content, and url
            ref_index: Index of the reference (for source_id)
            
        Returns:
            Citation object
        """
        import re
        
        # Generate source_id from reference
        source_id = f"ref_{ref_index + 1}"
        
        # Extract and clean title from reference content
        ref_content = ref.get("content", "")
        
        # Clean Wikipedia markup characters (^, ab, abcd, etc.)
        # Remove patterns like "^ab", "^abcd", "^According to", etc.
        cleaned_content = re.sub(r'\^[a-z]*', '', ref_content)  # Remove ^ab, ^abcd, etc.
        cleaned_content = re.sub(r'\^[A-Z][a-z]*\s*', '', cleaned_content)  # Remove ^According to, etc.
        cleaned_content = cleaned_content.strip()
        
        # Try to extract a meaningful title
        # Look for common citation patterns:
        # 1. Author names followed by year: "Smith (2020)" or "Smith, J. (2020)"
        # 2. Book/article titles in quotes: "Title of Book"
        # 3. Journal names: "Journal Name"
        
        title = ""
        
        # Pattern 1: Author (Year) or Author, Name (Year)
        author_year_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+)*)\s*\((\d{4})\)', cleaned_content)
        if author_year_match:
            author = author_year_match.group(1)
            year = author_year_match.group(2)
            title = f"{author} ({year})"
        else:
            # Pattern 2: Look for quoted titles
            quoted_match = re.search(r'"([^"]+)"', cleaned_content)
            if quoted_match:
                title = quoted_match.group(1)
            else:
                # Pattern 3: Look for italicized titles (common in citations)
                italic_match = re.search(r'<i>([^<]+)</i>', ref_content)
                if italic_match:
                    title = italic_match.group(1)
                else:
                    # Pattern 4: First sentence or first 80 characters
                    # Remove common prefixes
                    title = cleaned_content.split('.')[0]
                    title = title.split(',')[0]
                    # Remove common citation prefixes
                    title = re.sub(r'^(According to|See|Cf\.|e\.g\.|i\.e\.)\s*', '', title, flags=re.IGNORECASE)
                    title = title.strip()
        
        # Clean up title
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        if len(title) > 100:
            title = title[:100] + "..."
        if not title or len(title) < 3:
            title = f"Reference {ref_index + 1}"
        
        # Get URL
        url = ref.get("url", "")
        if not url or url.startswith("#"):
            # If no valid URL, try to extract from content
            url_match = re.search(r'https?://[^\s\)]+', ref_content)
            if url_match:
                url = url_match.group(0)
            else:
                url = ""  # No URL available
        
        return Citation(
            source_id=source_id,
            title=title,
            url=url,
            revision_id="",  # References don't have revision IDs
            accessed_date=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

    @staticmethod
    def create_citation(page_data: dict, lang: str = "en") -> Citation:
        """
        Create a Citation object from page data (DEPRECATED - use references instead).
        This is kept for backward compatibility but should not be used for final citations.

        Args:
            page_data: Page data from WikipediaClient
            lang: Language code

        Returns:
            Citation object
        """
        title = page_data.get("title", "")
        revid = page_data.get("revid", "")
        url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}?oldid={revid}"
        # Generate source_id: keep colons for namespace pages, replace other special chars
        source_id = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        # Ensure it starts with 'wiki_' prefix
        if not source_id.startswith('wiki_'):
            source_id = f"wiki_{source_id}"

        # Strip HTML from displaytitle
        displaytitle = page_data.get("displaytitle", title)
        if displaytitle and "<" in displaytitle:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(displaytitle, "html.parser")
            displaytitle = soup.get_text(strip=True)

        return Citation(
            source_id=source_id,
            title=displaytitle,
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
        source_mapping: Dict[str, int],
        citations: List[Citation],
        length: str = "medium",
    ) -> str:
        """
        Generate a summary with inline citations using LLM.

        Args:
            paragraphs: List of text paragraphs to summarize
            source_ids: Corresponding source IDs for citations
            source_mapping: Mapping from source_id to citation number (1, 2, 3, ...)
            citations: List of Citation objects to show in prompt
            length: Report length ("short" or "medium")

        Returns:
            Generated summary text with sequential citation numbers [1], [2], [3]...
        """
        if len(paragraphs) != len(source_ids):
            raise ValueError("paragraphs and source_ids must have the same length")

        # Build prompt
        content_text = "\n\n".join(paragraphs)
        word_limit = 150 if length == "short" else 300

        # Build source info for the prompt - show citation titles so AI knows what to cite
        # Use source_mapping to get the correct order (this matches source_order from generate())
        # Reverse the mapping to get citation number -> source_id
        num_to_source = {v: k for k, v in source_mapping.items()}
        num_sources = len(source_mapping)
        
        # Create a mapping from source_id to citation title
        source_id_to_citation = {cite.source_id: cite for cite in citations}
        
        # Build source list with titles in the correct order (matching citation numbers)
        source_list_items = []
        for citation_num in sorted(num_to_source.keys()):
            source_id = num_to_source[citation_num]
            citation = source_id_to_citation.get(source_id)
            if citation:
                title = citation.title[:80] + "..." if len(citation.title) > 80 else citation.title
                source_list_items.append(f"  [{citation_num}] {title}")
            else:
                source_list_items.append(f"  [{citation_num}] {source_id}")
        
        source_list = "\n".join(source_list_items)
        
        # Calculate minimum citations to use (aim for at least 3-5 if we have many sources)
        min_citations = min(5, max(2, num_sources // 6)) if num_sources > 5 else num_sources
        
        # Build example showing how to use multiple citations
        example_citations = ", ".join([f"[{i+1}]" for i in range(min(min_citations, 5))])
        
        prompt = f"""Generate a concise, factual summary based on the following Wikipedia content. 
Include inline citations using sequential numbers [1], [2], [3], etc. in the order they first appear in your summary.

CRITICAL REQUIREMENT: You have {num_sources} source(s) available. You MUST use at least {min_citations} DIFFERENT citation numbers (e.g., {example_citations}) throughout your summary. DO NOT use only [1] for everything.

Available sources with their titles (use these citation numbers):
{source_list}

Content:
{content_text}

Requirements:
- Length: approximately {word_limit} words
- Include citations using sequential numbers in brackets: [1], [2], [3], etc.
- MANDATORY: You MUST use at least {min_citations} different citation numbers from [1] through [{num_sources}]
- DO NOT use [1] for every sentence - vary your citations
- Number citations in the order they FIRST appear in your summary (first citation = [1], second new citation = [2], etc.)
- Use multiple citations together when appropriate: [1][2][3] or [1][2] to support claims
- Each citation number MUST correspond to one of the {num_sources} source(s) listed above
- Cite different sources for different facts/claims - match sources to relevant content
- Example: If discussing origins, use [1]. If discussing applications, use [2] or [3]. If discussing the trick itself, use [4] or [5].
- Be factual and objective
- Focus on key points
- Vary your citations - use different sources throughout the summary

IMPORTANT: Your summary should contain citations like [1], [2], [3], [4], etc. - NOT just [1] repeated.

Summary:"""

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an academic research assistant that creates well-cited summaries. CRITICAL: You MUST use multiple different citations (e.g., [1], [2], [3], [4]) throughout your summary. NEVER use only [1] for everything. Each different fact or claim should cite a different source when possible. Use multiple citations together like [1][2] when multiple sources support the same claim. Vary your citations - if you have 10 sources, use at least 5-7 different citation numbers.",
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
        
        # If this is a reference citation (not Wikipedia), format it differently
        if citation.source_id.startswith("ref_"):
            # For references, use the title and URL directly
            if citation.url and not citation.url.startswith("#"):
                # Clean title - remove any remaining Wikipedia markup
                clean_title = citation.title.replace("^", "").strip()
                return f"{clean_title}. Retrieved {date_str}, from {citation.url}"
            else:
                clean_title = citation.title.replace("^", "").strip()
                return f"{clean_title}. (n.d.)."
        
        # Original Wikipedia citation format (fallback)
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
        
        # If this is a reference citation (not Wikipedia), format it differently
        if citation.source_id.startswith("ref_"):
            # For references, use the title and URL directly
            clean_title = citation.title.replace("^", "").strip()
            if citation.url and not citation.url.startswith("#"):
                return f'"{clean_title}." {date_str}, {citation.url}.'
            else:
                return f'"{clean_title}."'
        
        # Original Wikipedia citation format (fallback)
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
            
            # Extract references from the main page (these are the actual sources to cite)
            logger.info("Extracting references from Wikipedia article")
            references = self.content_processor.extract_references(main_page["text"])
            
            if not references:
                logger.warning("No references found in Wikipedia article. Falling back to Wikipedia citation.")
                # Fallback: if no references found, use Wikipedia as citation
                main_citation = self.content_processor.create_citation(main_page, lang)
                all_citations = [main_citation]
            else:
                # Create citations from references
                all_citations = [
                    self.content_processor.create_citation_from_reference(ref, idx)
                    for idx, ref in enumerate(references)
                ]
                logger.info(f"Extracted {len(all_citations)} references from Wikipedia article")

            # Extract related links
            related_titles = self.wiki_client.extract_related_links(
                main_page, max_links=request.max_related_links
            )

            # Fetch related pages in parallel (for additional content, but we'll use their references too)
            logger.info(f"Fetching {len(related_titles)} related pages")
            related_pages = await asyncio.gather(
                *[
                    self.wiki_client.fetch_page(title, lang)
                    for title in related_titles
                ],
                return_exceptions=True,
            )

            # Process all pages
            all_paragraphs = []
            all_source_ids = []

            # Process main page
            paragraphs = self.content_processor.extract_paragraphs(
                main_page["text"], max_paragraphs=5
            )
            # Distribute paragraphs across different citations to encourage multiple citations
            for idx, para in enumerate(paragraphs):
                all_paragraphs.append(para)
                # Map each paragraph to a different citation (round-robin distribution)
                if all_citations:
                    # Use modulo to cycle through citations
                    citation_idx = idx % len(all_citations)
                    all_source_ids.append(all_citations[citation_idx].source_id)
                else:
                    all_source_ids.append("ref_1")

            # Process related pages - extract their references too
            for page_data in related_pages:
                if isinstance(page_data, Exception):
                    logger.warning(f"Failed to fetch related page: {page_data}")
                    continue

                # Extract references from related page (but don't add duplicates)
                related_refs = self.content_processor.extract_references(page_data["text"])
                if related_refs:
                    # Add references from related pages, but skip duplicates
                    start_idx = len(all_citations)
                    existing_urls = {c.url for c in all_citations if c.url}
                    existing_titles = {c.title for c in all_citations}
                    
                    for idx, ref in enumerate(related_refs):
                        ref_url = ref.get("url", "")
                        # Skip if URL already exists or if it's a duplicate title
                        if ref_url and ref_url in existing_urls:
                            continue
                        citation = self.content_processor.create_citation_from_reference(ref, start_idx + idx)
                        if citation.title not in existing_titles:
                            all_citations.append(citation)
                            if citation.url:
                                existing_urls.add(citation.url)
                            existing_titles.add(citation.title)
                # Note: We don't fallback to Wikipedia citation for related pages
                # Only use references, not Wikipedia itself

                paragraphs = self.content_processor.extract_paragraphs(
                    page_data["text"], max_paragraphs=3
                )
                # Distribute paragraphs across different citations
                for idx, para in enumerate(paragraphs):
                    all_paragraphs.append(para)
                    # Use round-robin to distribute across all citations
                    if all_citations:
                        # Start from where we left off, cycle through all citations
                        citation_idx = (len(all_paragraphs) - 1) % len(all_citations)
                        all_source_ids.append(all_citations[citation_idx].source_id)
                    else:
                        all_source_ids.append("ref_1")

            # Get unique sources in order of first appearance in content
            source_order = []
            seen_sources = set()
            for source_id in all_source_ids:
                if source_id not in seen_sources:
                    source_order.append(source_id)
                    seen_sources.add(source_id)
            
            # Create mapping: source_id -> citation_number (for reference, but AI will number in order of appearance)
            source_mapping = {source_id: idx + 1 for idx, source_id in enumerate(source_order)}
            
            # Generate summary with LLM - AI will use [1], [2], [3] in order of citation appearance
            logger.info("Generating summary with LLM")
            summary = await self.llm_generator.generate_summary(
                all_paragraphs, all_source_ids, source_mapping, all_citations, length=request.report_length
            )

            # Extract citations from summary - match numeric citations [1], [2], [3], etc.
            citation_pattern = r"\[(\d+)\]"
            found_citation_nums = [int(m) for m in re.findall(citation_pattern, summary)]
            
            # Filter out invalid citation numbers (those exceeding available sources)
            max_valid_citation = len(source_order)
            valid_citation_nums = [num for num in found_citation_nums if 1 <= num <= max_valid_citation]
            
            if not valid_citation_nums:
                logger.warning("No valid citations found in summary, or all citations exceed available sources")
                # If no valid citations, use all sources
                valid_citation_nums = list(range(1, len(source_order) + 1))
            
            # Get unique citation numbers in order of first appearance in summary
            used_citation_nums = []
            seen_nums = set()
            for num in valid_citation_nums:
                if num not in seen_nums:
                    used_citation_nums.append(num)
                    seen_nums.add(num)
            
            # Log warning if AI used invalid citation numbers
            invalid_nums = [num for num in found_citation_nums if num > max_valid_citation]
            if invalid_nums:
                logger.warning(f"AI used invalid citation numbers {invalid_nums} (max valid: {max_valid_citation}). Filtered out.")
            
            # Map citation numbers to sources
            # Citation [1] maps to first source in source_order, [2] to second, etc.
            ordered_citations = []
            citation_to_source = {}  # Map citation number to source_id
            
            # Map each valid citation number to its corresponding source
            # ONLY include citations that were actually used in the summary
            for num in sorted(used_citation_nums):
                if 1 <= num <= len(source_order):
                    source_id = source_order[num - 1]  # num-1 because list is 0-indexed
                    citation_to_source[num] = source_id
                    citation = next((c for c in all_citations if c.source_id == source_id), None)
                    if citation and citation not in ordered_citations:
                        ordered_citations.append(citation)
            
            # Log if some citations weren't used (for debugging)
            unused_count = len(all_citations) - len(ordered_citations)
            if unused_count > 0:
                logger.info(f"Summary used {len(ordered_citations)} citations out of {len(all_citations)} available. {unused_count} citations not included in bibliography.")
            
            bibliography = self.citation_formatter.format_bibliography(
                ordered_citations, request.citation_style
            )

            # Create sections (simple version - can be enhanced)
            # Convert citation numbers to strings for the citations field
            sections = [
                ReportSection(
                    heading="Summary",
                    content=summary,
                    citations=[str(num) for num in used_citation_nums],
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

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title=Config.SERVICE_NAME,
    version=Config.VERSION,
    description="Generate citation-backed reports from Wikipedia articles",
)

# Serve static files (HTML interface)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serve the main interface."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "API is running. Visit /docs for API documentation."}


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

