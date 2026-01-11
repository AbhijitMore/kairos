from fastapi.testclient import TestClient
import pytest
import numpy as np
from app.main import app, get_engine
from src.kairos.core.policy import KairosPolicy

# Mocking the engine for API tests
class MockEngine:
    def predict_calibrated(self, X):
        return np.array([0.8] * len(X))

@pytest.fixture
def client():
    # Setup: Dependency Overrides are the standard/robust way for FastAPI testing
    mock_engine = MockEngine()
    mock_policy = KairosPolicy(tau_low=0.3, tau_high=0.7)
    
    def override_get_engine():
        return mock_engine, mock_policy
        
    app.dependency_overrides[get_engine] = override_get_engine
    with TestClient(app) as c:
        yield c
    # Teardown
    app.dependency_overrides.clear()

def test_predict_endpoint_success(client):
    payload = {
        "instances": [
            {
                "age": 39,
                "workclass": "State-gov",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    # 0.8 is >= 0.7 (ACCEPT)
    assert data[0]["decision"] == "ACCEPT"

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
