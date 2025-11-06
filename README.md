# Cite The Cite, Wikipedia Report Generator API

Wikipedia Report Generator API, A FastAPI service that generates citation-backed reports from Wikipedia articles using MediaWiki API and LLM summarization.

## Demo & Assets

![Cite the Cite Demo](asset/banner.png)

# **Demo Video:** [Watch on YouTube](https://youtu.be/HFS2E9ivCEM)


---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)

---

## Overview

This service accepts a Wikipedia URL and generates a structured, citation-backed report. The process includes:

1. **Fetching** the main Wikipedia page via MediaWiki API
2. **Extracting** external references from Wikipedia articles (not Wikipedia itself)
3. **Deduplicating** and filtering references (removes Wikipedia/Wikidata links, limits to top 30)
4. **Processing** content and creating proper citations from external sources
5. **Generating** an LLM-powered summary with multiple inline citation markers [1], [2], [3]...
6. **Formatting** bibliography in APA or MLA style (only cited sources)
7. **Returning** structured JSON response with web interface support

**Use cases:**
- Academic research assistance
- Content creation with proper attribution
- Automated literature review generation
- Educational tools

---

## Features

- **MediaWiki API Integration** - Uses the official API, no HTML scraping
- **External Reference Extraction** - Cites original sources, not Wikipedia
- **Smart Reference Processing** - Deduplication, filtering, and quality control
- **Multiple Citations** - AI uses [1], [2], [3]... throughout summary
- **Multiple Citation Styles** - Supports APA and MLA
- **Web Interface** - Beautiful UI for easy interaction
- **Related Article Discovery** - Automatically finds relevant sources
- **Rate Limiting** - Respects Wikipedia's infrastructure
- **Reproducible Results** - Tracks Wikipedia revision IDs
- **OpenAPI Documentation** - Auto-generated interactive docs
- **Async/Await** - Non-blocking HTTP operations
- **Error Handling** - Graceful degradation with detailed error messages

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Web Interface (Browser)                         │
│                    ┌──────────────────────────────────┐                 │
│                    │   static/index.html              │                 │
│                    │   - URL Input Form               │                 │
│                    │   - Citation Style Selector      │                 │
│                    │   - Results Display               │                 │
│                    └──────────────┬───────────────────┘                 │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │ HTTP POST /generate
                                    │ {url, citation_style, report_length}
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application (main.py)                     │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ReportGenerator                              │   │
│  │                  (Main Orchestrator)                            │   │
│  └──┬──────────────────────┬──────────────────────┬───────────────┘   │
│     │                      │                      │                    │
│     ▼                      ▼                      ▼                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │WikipediaClient│   │ContentProcessor│  │   LLMGenerator        │    │
│  │              │   │              │   │   (OpenAI API)         │    │
│  │ - fetch_page │   │ - extract_   │   │   - generate_summary   │    │
│  │ - extract_   │   │   references │   │   - Multiple citations │    │
│  │   related    │   │ - extract_   │   │   - Citation mapping   │    │
│  │   links      │   │   paragraphs │   │                        │    │
│  └──────┬───────┘   │ - create_    │   └───────────┬────────────┘    │
│         │           │   citation   │               │                   │
│         │           │   from_ref   │               │                   │
│         │           └──────┬───────┘               │                   │
│         │                  │                       │                   │
│         ▼                  ▼                       ▼                   │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │              CitationFormatter                               │     │
│  │  - format_apa()  - format_mla()  - format_bibliography()     │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Wikipedia  │    │  Wikipedia   │    │   OpenAI     │
│  MediaWiki   │    │  References  │    │   GPT API   │
│  API         │    │  (External)  │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   JSON Response      │
                    │   {                  │
                    │     title,           │
                    │     summary,         │
                    │     sources,         │
                    │     bibliography    │
                    │   }                  │
                    └──────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CiteWiki

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Set OpenAI API key
export OPENAI_API_KEY='sk-...'

# Optional: Configure settings
export SERVICE_NAME='WikiReportAPI'
export CONTACT_EMAIL='your-email@example.com'
```

### Running Locally

```bash
# Development server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit:
- **Web Interface:** http://localhost:8000
- **API:** http://localhost:8000/generate
- **Interactive Docs:** http://localhost:8000/docs
- **OpenAPI Schema:** http://localhost:8000/openapi.json

### First Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "citation_style": "APA",
    "max_related_links": 3,
    "report_length": "medium"
  }'
```

---

## API Documentation

### Endpoint: `POST /generate`

Generate a citation-backed report from a Wikipedia URL.

**Request Body:**

```json
{
  "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
  "citation_style": "APA",
  "max_related_links": 3,
  "report_length": "medium"
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "title": "Artificial intelligence",
  "summary": "Artificial intelligence (AI) is intelligence demonstrated by machines... [wiki_artificial_intelligence]",
  "sections": [...],
  "sources": [...],
  "bibliography": [...],
  "generated_at": "2025-10-31T22:00:00Z",
  "processing_time_seconds": 8.5
}
```

### Endpoint: `GET /health`

Check service health and configuration.

---

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=main --cov-report=html
```

### Code Style

The project follows PEP 8 with type hints and docstrings. Use `black` for formatting.

---

## Deployment

### Docker

```bash
# Build image
docker build -t wiki-report-api .

# Run container
docker run -p 8000:8000 -e OPENAI_API_KEY='sk-...' wiki-report-api
```

### Docker Compose

```bash
# Set environment variables
export OPENAI_API_KEY='sk-...'

# Start services
docker-compose up -d
```

---

## License

This project is licensed under the MIT License.

---

**Last Updated:** October 31, 2025 | **Version:** 1.0.0

