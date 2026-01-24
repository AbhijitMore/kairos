import pytest


@pytest.mark.integration
def test_predict_adult_success(client, sample_instance, auth_headers):
    # Sample instance needs discriminator
    tagged_instance = sample_instance.copy()
    tagged_instance["dataset_type"] = "adult"
    payload = {"dataset": "adult", "instances": [tagged_instance]}
    response = client.post("/api/v1/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["decision"] in ["ACCEPT", "REJECT", "ABSTAIN"]


@pytest.mark.integration
def test_predict_home_credit_success(client, auth_headers):
    # Minimal valid Home Credit instance
    home_credit_instance = {
        "dataset_type": "home_credit",
        "AMT_INCOME_TOTAL": 200000.0,
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "REGION_RATING_CLIENT": 2,
        "DAYS_BIRTH": -15000,
        "DAYS_EMPLOYED": -2000,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
        "NAME_EDUCATION_TYPE": "Secondary",
        "NAME_INCOME_TYPE": "Working",
        "OCCUPATION_TYPE": "Laborers",
    }
    payload = {"dataset": "home_credit", "instances": [home_credit_instance]}
    response = client.post("/api/v1/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["decision"] in ["ACCEPT", "REJECT", "ABSTAIN"]


@pytest.mark.integration
def test_predict_endpoint_unauthorized(client):
    payload = {"instances": []}
    # No header
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 403

    # Invalid Key
    headers = {"X-API-KEY": "wrong_key"}
    response = client.post("/api/v1/predict", json=payload, headers=headers)
    assert response.status_code == 403


@pytest.mark.integration
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
