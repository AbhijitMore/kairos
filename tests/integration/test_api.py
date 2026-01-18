import pytest


@pytest.mark.integration
def test_predict_endpoint_success(client, sample_instance, auth_headers):
    payload = {"instances": [sample_instance]}
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    # Mock engine returns 0.8, which is >= 0.7 (ACCEPT)
    assert data[0]["decision"] == "ACCEPT"


@pytest.mark.integration
def test_predict_endpoint_unauthorized(client):
    payload = {"instances": []}
    # No header
    response = client.post("/predict", json=payload)
    assert response.status_code == 403

    # Invalid Key
    headers = {"X-API-KEY": "wrong_key"}
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 403


@pytest.mark.integration
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
