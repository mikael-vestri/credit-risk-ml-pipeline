"""
Test API routes and endpoints.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_root_endpoint():
    """Test root endpoint returns correct structure."""
    from api.app import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint."""
    from api.app import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_model_info_endpoint():
    """Test model info endpoint (may fail if model not loaded)."""
    from api.app import app

    client = TestClient(app)
    response = client.get("/model/info")

    # Should return 503 if model not loaded, or 200 if loaded
    assert response.status_code in [200, 503]
