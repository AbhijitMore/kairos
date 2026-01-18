"""
Load Testing Suite for KAIROS API

This module uses Locust to simulate high-traffic scenarios and measure:
- Throughput (RPS)
- Latency percentiles (p50, p95, p99)
- Error rates under load
- System stability at 10K+ RPS target

Run with: locust -f tests/load/test_load.py --host=http://localhost:8000
"""

import json
import random
from locust import HttpUser, task, between, events
import logging

logger = logging.getLogger(__name__)


class KairosAPIUser(HttpUser):
    """
    Simulates a user making predictions against the KAIROS API.
    """

    wait_time = between(0.1, 0.5)  # Wait 100-500ms between requests

    def on_start(self):
        """Called when a simulated user starts."""
        self.api_key = "kairos_dev_key_2026"  # From .env
        self.headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        # Sample test data
        self.sample_instances = [
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
            },
            {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 13,
                "native_country": "United-States",
                "education_num": 13,
            },
            {
                "age": 28,
                "workclass": "Private",
                "marital_status": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Wife",
                "race": "Black",
                "sex": "Female",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "Cuba",
                "education_num": 11,
            },
        ]

    @task(10)
    def predict_single(self):
        """Test single prediction endpoint (most common use case)."""
        instance = random.choice(self.sample_instances)
        payload = {"instances": [instance]}

        with self.client.post(
            "/predict",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/predict [single]",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if len(data) == 1 and "decision" in data[0]:
                        response.success()
                    else:
                        response.failure("Invalid response structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def predict_batch(self):
        """Test batch prediction endpoint."""
        batch_size = random.randint(5, 20)
        instances = [random.choice(self.sample_instances) for _ in range(batch_size)]
        payload = {"instances": instances}

        with self.client.post(
            "/predict",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name=f"/predict [batch-{batch_size}]",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if len(data) == batch_size:
                        response.success()
                    else:
                        response.failure(
                            f"Expected {batch_size} results, got {len(data)}"
                        )
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def predict_async(self):
        """Test async batch prediction endpoint."""
        instances = [random.choice(self.sample_instances) for _ in range(10)]
        payload = {"instances": instances}

        # Submit async job
        with self.client.post(
            "/predict/batch/async",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/predict/batch/async [submit]",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    task_id = data.get("task_id")
                    if task_id:
                        response.success()
                        # Poll for results
                        self.poll_async_result(task_id)
                    else:
                        response.failure("No task_id in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")

    def poll_async_result(self, task_id):
        """Poll async task status."""
        with self.client.get(
            f"/predict/status/{task_id}",
            headers=self.headers,
            catch_response=True,
            name="/predict/status/{task_id} [poll]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Test health endpoint."""
        with self.client.get(
            "/health", catch_response=True, name="/health"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure("Service unhealthy")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    logger.info("🚀 Starting KAIROS load test...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops - print summary."""
    logger.info("✅ Load test completed")

    stats = environment.stats
    logger.info(f"\n{'=' * 60}")
    logger.info("LOAD TEST SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total Requests: {stats.total.num_requests}")
    logger.info(f"Total Failures: {stats.total.num_failures}")
    logger.info(f"Failure Rate: {stats.total.fail_ratio * 100:.2f}%")
    logger.info(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    logger.info(
        f"Median Response Time (p50): {stats.total.get_response_time_percentile(0.5):.2f}ms"
    )
    logger.info(
        f"95th Percentile (p95): {stats.total.get_response_time_percentile(0.95):.2f}ms"
    )
    logger.info(
        f"99th Percentile (p99): {stats.total.get_response_time_percentile(0.99):.2f}ms"
    )
    logger.info(f"Requests/sec: {stats.total.total_rps:.2f}")
    logger.info(f"{'=' * 60}\n")
