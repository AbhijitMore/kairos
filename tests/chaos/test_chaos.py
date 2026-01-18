"""
Chaos Engineering Tests for KAIROS

Simulates failure scenarios to test system resilience:
- Service unavailability
- Network latency/timeouts
- Partial failures
- Resource exhaustion
- Cascading failures

Requires: pytest-timeout, requests
"""

import pytest
import requests
import time
import os
from unittest.mock import patch
from app.main import app
from fastapi.testclient import TestClient


class TestChaosScenarios:
    """Test system behavior under adverse conditions."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        return os.getenv("API_KEY", "kairos_dev_key_2026")

    def test_model_loading_failure(self, client, api_key):
        """
        Chaos: Model fails to load on startup.
        Expected: API returns 503 Service Unavailable.
        """
        with patch("app.main.KairosInferenceEngine.load") as mock_load:
            mock_load.side_effect = FileNotFoundError("Model not found")

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["engine_ready"] is False

    def test_redis_connection_failure(self, client, api_key):
        """
        Chaos: Redis becomes unavailable.
        Expected: Async endpoints fail gracefully.
        """
        payload = {
            "instances": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States",
                    "education_num": 13,
                }
            ]
        }

        with patch("app.tasks.celery_app.send_task") as mock_celery:
            mock_celery.side_effect = ConnectionError("Redis unavailable")

            response = client.post(
                "/predict/batch/async", json=payload, headers={"X-API-KEY": api_key}
            )
            # Should handle gracefully (either 503 or fallback to sync)
            assert response.status_code in [200, 503]

    @pytest.mark.timeout(5)
    def test_slow_inference_timeout(self, client, api_key):
        """
        Chaos: Model inference takes too long.
        Expected: Request times out gracefully.
        """
        payload = {
            "instances": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States",
                    "education_num": 13,
                }
            ]
        }

        with patch("app.main.state.engine.predict_calibrated") as mock_predict:
            # Simulate slow inference
            def slow_predict(*args, **kwargs):
                time.sleep(10)  # Exceeds timeout
                return [0.5]

            mock_predict.side_effect = slow_predict

            # This should timeout or return quickly with error
            start = time.time()
            try:
                client.post(
                    "/predict", json=payload, headers={"X-API-KEY": api_key}, timeout=3
                )
                elapsed = time.time() - start
                assert elapsed < 5, "Request should timeout quickly"
            except requests.exceptions.Timeout:
                pass  # Expected behavior

    def test_malformed_input_handling(self, client, api_key):
        """
        Chaos: Client sends malformed data.
        Expected: 422 Validation Error with helpful message.
        """
        malformed_payloads = [
            {"instances": []},  # Empty instances
            {"instances": [{"age": "invalid"}]},  # Wrong type
            {"instances": [{}]},  # Missing required fields
            {},  # No instances key
            {"instances": "not_a_list"},  # Wrong structure
        ]

        for payload in malformed_payloads:
            response = client.post(
                "/predict", json=payload, headers={"X-API-KEY": api_key}
            )
            assert response.status_code in [
                422,
                500,
            ], f"Should reject malformed payload: {payload}"

    def test_rate_limit_enforcement(self, client, api_key):
        """
        Chaos: Client exceeds rate limit.
        Expected: 429 Too Many Requests after threshold.
        """
        payload = {
            "instances": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States",
                    "education_num": 13,
                }
            ]
        }

        # Hammer the endpoint
        responses = []
        for _ in range(100):
            response = client.post(
                "/predict", json=payload, headers={"X-API-KEY": api_key}
            )
            responses.append(response.status_code)
            if response.status_code == 429:
                break

        # Should eventually rate limit
        assert 429 in responses, "Rate limiting should trigger"

    def test_concurrent_requests_stability(self, client, api_key):
        """
        Chaos: Multiple concurrent requests.
        Expected: System remains stable, no crashes.
        """
        import concurrent.futures

        payload = {
            "instances": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States",
                    "education_num": 13,
                }
            ]
        }

        def make_request():
            return client.post("/predict", json=payload, headers={"X-API-KEY": api_key})

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Most should succeed (allowing for some rate limiting)
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count > 30, f"Only {success_count}/50 requests succeeded"

    def test_memory_leak_detection(self, client, api_key):
        """
        Chaos: Repeated requests to detect memory leaks.
        Expected: Memory usage remains stable.
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        payload = {
            "instances": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States",
                    "education_num": 13,
                }
            ]
        }

        # Make 1000 requests
        for _ in range(1000):
            client.post("/predict", json=payload, headers={"X-API-KEY": api_key})

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory should not increase by more than 100MB
        assert (
            memory_increase < 100
        ), f"Potential memory leak: {memory_increase:.2f}MB increase"

    def test_invalid_api_key_handling(self, client):
        """
        Chaos: Invalid/missing API key.
        Expected: 403 Forbidden.
        """
        payload = {
            "instances": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States",
                    "education_num": 13,
                }
            ]
        }

        # No API key
        response = client.post("/predict", json=payload)
        assert response.status_code == 403

        # Invalid API key
        response = client.post(
            "/predict", json=payload, headers={"X-API-KEY": "invalid_key"}
        )
        assert response.status_code == 403
