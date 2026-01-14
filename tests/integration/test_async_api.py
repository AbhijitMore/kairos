from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@patch("app.main.predict_batch_task.delay")
def test_predict_batch_async_endpoint(mock_delay, client):
    # Mock celery task trigger
    mock_task = MagicMock()
    mock_task.id = "fake-task-id"
    mock_delay.return_value = mock_task

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
                "native_country": "United-States",
            }
        ]
    }
    headers = {"X-API-KEY": "kairos_dev_key_2026"}

    response = client.post("/predict/batch/async", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "fake-task-id"
    assert data["status"] == "PENDING"
    assert "poll_url" in data
    mock_delay.assert_called_once()


@patch("app.main.AsyncResult")
def test_get_task_status_endpoint(mock_async_result, client):
    # Mock AsyncResult behavior
    mock_result = MagicMock()
    mock_result.status = "SUCCESS"
    mock_result.ready.return_value = True
    mock_result.successful.return_value = True
    mock_result.result = {"predictions": [0.8], "count": 1}
    mock_async_result.return_value = mock_result

    headers = {"X-API-KEY": "kairos_dev_key_2026"}
    response = client.get("/predict/status/fake-task-id", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["result"]["predictions"] == [0.8]
