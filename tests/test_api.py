"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from veritas_crag.api.app import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "apple_silicon_mps_available" in data
    assert data["project"] == "VeritasCRAG"


def test_query_endpoint():
    payload = {
        "query": "What is the Docling latency on M4?",
        "top_k": 2,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["action"] == "CORRECT"
    assert data["confidence"] >= 0.65
    assert len(data["citations"]) >= 1
    assert data["latency_seconds"] > 0.0


def test_ingest_endpoint_unsupported_format():
    files = {"file": ("malicious.exe", b"binarycontent", "application/octet-stream")}
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
