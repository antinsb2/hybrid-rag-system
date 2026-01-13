"""
Tests for API endpoints.
"""

from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rag.api.server import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "capabilities" in data
    assert data["status"] == "healthy"
    
    print(f"✅ Health check: {data}")


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "name" in data
    assert "endpoints" in data
    
    print(f"✅ Root endpoint: {data['name']}")


def test_stats_endpoint():
    """Test stats endpoint."""
    response = client.get("/stats")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "is_indexed" in data
    
    print(f"✅ Stats: indexed={data['is_indexed']}")


def test_query_validation():
    """Test query request validation."""
    # Empty query should fail
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422  # Validation error
    
    # Invalid top_k should fail
    response = client.post("/query", json={"query": "test", "top_k": 100})
    assert response.status_code == 422
    
    print("✅ Query validation working")


if __name__ == "__main__":
    test_health_endpoint()
    test_root_endpoint()
    test_stats_endpoint()
    test_query_validation()
    print("\n✅ All API tests passed!")
