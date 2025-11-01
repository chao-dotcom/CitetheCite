"""Tests for FastAPI endpoints."""

import os
from fastapi.testclient import TestClient
from main import app


def test_root_endpoint():
    """Test root endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


def test_health_endpoint():
    """Test /health endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "config" in data
    assert "timestamp" in data


def test_generate_endpoint_validation():
    """Test request validation."""
    client = TestClient(app)

    # Invalid URL
    response = client.post("/generate", json={"url": "not-a-wikipedia-url"})
    assert response.status_code == 422  # Validation error

    # Missing required field
    response = client.post("/generate", json={})
    assert response.status_code == 422

    # Valid URL format but will fail without OpenAI key
    response = client.post(
        "/generate",
        json={
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "citation_style": "APA",
        },
    )
    # Will fail due to missing OpenAI key or service not initialized
    assert response.status_code in [422, 500, 503]

