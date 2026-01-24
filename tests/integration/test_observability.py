import pytest
from unittest.mock import patch
from kairos.api.dependencies import APIState


@pytest.mark.integration
def test_metrics_endpoint_availability(client):
    """Verifies Prometheus metrics endpoint is online."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "kairos_decisions_total" in response.text


@pytest.mark.integration
def test_metric_increment_with_dataset_labels(
    client, sample_instance, auth_headers, mock_engine, mock_policy
):
    """
    Verifies that calling predict increments the decision counter
    with correct dataset labels.
    """

    def get_count(text, dataset):
        count = 0.0
        for line in text.splitlines():
            if "kairos_decisions_total" in line and f'dataset="{dataset}"' in line:
                try:
                    count += float(line.split()[-1])
                except (ValueError, IndexError):
                    continue
        return count

    # 1. Get initial metric state
    resp_init = client.get("/metrics")
    adult_init = get_count(resp_init.text, "adult")
    hc_init = get_count(resp_init.text, "home_credit")

    # 2. Trigger predictions with mocked engine
    with patch.object(APIState, "get_engine", return_value=(mock_engine, mock_policy)):
        # Trigger adult prediction
        adult_inst_with_tag = sample_instance.copy()
        adult_inst_with_tag["dataset_type"] = "adult"
        payload_adult = {"dataset": "adult", "instances": [adult_inst_with_tag]}
        resp_adult = client.post(
            "/api/v1/predict", json=payload_adult, headers=auth_headers
        )
        assert resp_adult.status_code == 200, resp_adult.text

        # Trigger home_credit prediction
        home_credit_instance = {
            "dataset_type": "home_credit",
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 406597.5,
            "AMT_ANNUITY": 24700.5,
            "AMT_GOODS_PRICE": 351000.0,
            "REGION_RATING_CLIENT": 2,
            "DAYS_BIRTH": -9461,
            "DAYS_EMPLOYED": -637,
            "EXT_SOURCE_1": 0.08,
            "EXT_SOURCE_2": 0.26,
            "EXT_SOURCE_3": 0.13,
            "NAME_EDUCATION_TYPE": "Secondary",
            "NAME_INCOME_TYPE": "Working",
            "OCCUPATION_TYPE": "Laborers",
        }
        payload_hc = {"dataset": "home_credit", "instances": [home_credit_instance]}
        resp_hc = client.post("/api/v1/predict", json=payload_hc, headers=auth_headers)
        assert resp_hc.status_code == 200, resp_hc.text

    # 3. Verify increments
    resp_final = client.get("/metrics")
    adult_final = get_count(resp_final.text, "adult")
    hc_final = get_count(resp_final.text, "home_credit")

    # Print metrics on failure for easier debugging
    msg = f"Adult: {adult_init} -> {adult_final}, HC: {hc_init} -> {hc_final}\nMetrics Output:\n{resp_final.text}"
    assert adult_final > adult_init, msg
    assert hc_final > hc_init, msg
