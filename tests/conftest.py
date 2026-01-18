import pytest
import numpy as np
from fastapi.testclient import TestClient
from kairos.api.main import app
from kairos.api.dependencies import get_inference_deps
from kairos.core.policy import KairosPolicy


# Mock Engine for consistent API testing without loading 500MB of artifacts
class MockEngine:
    def predict_calibrated(self, X):
        return np.array([0.8] * len(X))


@pytest.fixture
def mock_engine():
    return MockEngine()


@pytest.fixture
def mock_policy():
    return KairosPolicy(tau_low=0.3, tau_high=0.7)


@pytest.fixture
def client(mock_engine, mock_policy):
    """
    Standardized FastAPI TestClient with dependency overrides.
    Ensures tests don't attempt to load real models from disk.
    """

    def override_get_engine():
        return mock_engine, mock_policy

    app.dependency_overrides[get_inference_deps] = override_get_engine
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    """Valid API Key headers."""
    return {"X-API-KEY": "kairos_dev_key_2026"}


@pytest.fixture
def sample_instance():
    """A single standardized Pydantic-compatible instance for prediction."""
    return {
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
        "native_country": "United-States",
    }
