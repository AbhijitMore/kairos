import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.integration
@patch("app.main.predict_batch_task.delay")
def test_predict_batch_async_endpoint(
    mock_delay, client, sample_instance, auth_headers
):
    # Mock celery task trigger
    mock_task = MagicMock()
    mock_task.id = "fake-task-id"
    mock_delay.return_value = mock_task

    payload = {"instances": [sample_instance]}
    response = client.post("/predict/batch/async", json=payload, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "fake-task-id"
    assert data["status"] == "PENDING"
    assert "poll_url" in data
    mock_delay.assert_called_once()


@pytest.mark.integration
@patch("app.main.AsyncResult")
def test_get_task_status_endpoint(mock_async_result, client, auth_headers):
    # Mock AsyncResult behavior
    mock_result = MagicMock()
    mock_result.status = "SUCCESS"
    mock_result.ready.return_value = True
    mock_result.successful.return_value = True
    mock_result.result = {"predictions": [0.8], "count": 1}
    mock_async_result.return_value = mock_result

    response = client.get("/predict/status/fake-task-id", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["result"]["predictions"] == [0.8]
