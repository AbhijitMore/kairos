# KAIROS Testing Infrastructure

Comprehensive test suite covering unit, integration, load, chaos, and end-to-end testing.

## Test Coverage Overview

| Test Type             | Location             | Purpose                      | Coverage    |
| --------------------- | -------------------- | ---------------------------- | ----------- |
| **Unit Tests**        | `tests/unit/`        | Individual component testing | 82%+        |
| **Integration Tests** | `tests/integration/` | API endpoint validation      | Full API    |
| **Load Tests**        | `tests/load/`        | Performance benchmarking     | 10K+ RPS    |
| **Chaos Tests**       | `tests/chaos/`       | Resilience validation        | 9 scenarios |
| **E2E Tests**         | `tests/e2e/`         | Full user workflows          | Dashboard   |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run All Tests

```bash
./scripts/run_tests.sh
```

### 3. Run Specific Test Suites

```bash
# Unit + Integration tests
pytest tests/unit/ tests/integration/ -v

# Load testing (requires running API)
./scripts/run_load_tests.sh

# Chaos engineering
pytest tests/chaos/ -v --timeout=30

# End-to-end UI tests (requires running dashboard)
./scripts/run_e2e_tests.sh
```

---

## Load Testing

### Prerequisites

- API must be running: `docker compose up api`
- Locust installed: `pip install locust`

### Run Load Test

```bash
# Default: 100 users, 60 seconds
./scripts/run_load_tests.sh

# Custom configuration
USERS=1000 RUN_TIME=5m ./scripts/run_load_tests.sh

# Interactive mode (Web UI at http://localhost:8089)
locust -f tests/load/test_load.py --host=http://localhost:8000
```

### Load Test Scenarios

- **Single Prediction** (70% of traffic): Individual instance predictions
- **Batch Prediction** (20% of traffic): 5-20 instances per request
- **Async Prediction** (10% of traffic): Background job submission + polling

### Performance Targets

| Metric            | Target     | Measured |
| ----------------- | ---------- | -------- |
| **Throughput**    | 10,000 RPS | TBD      |
| **Latency (p50)** | < 50ms     | TBD      |
| **Latency (p95)** | < 200ms    | TBD      |
| **Latency (p99)** | < 500ms    | TBD      |
| **Error Rate**    | < 0.1%     | TBD      |

---

## Chaos Engineering

### Test Scenarios

1. **Model Loading Failure**: Model artifact missing/corrupted
2. **Redis Connection Failure**: Async queue unavailable
3. **Slow Inference Timeout**: Model takes too long to respond
4. **Malformed Input Handling**: Invalid data validation
5. **Rate Limit Enforcement**: DDoS protection
6. **Concurrent Request Stability**: Thread safety validation
7. **Memory Leak Detection**: Resource exhaustion prevention
8. **Invalid API Key Handling**: Authentication bypass attempts

### Run Chaos Tests

```bash
pytest tests/chaos/ -v --timeout=30
```

### Expected Behavior

- **Graceful Degradation**: System returns 503/500 with helpful errors
- **No Crashes**: Service remains available for valid requests
- **Resource Cleanup**: No memory leaks or zombie processes

---

## End-to-End UI Testing

### Prerequisites

- Dashboard must be running: `docker compose up dashboard`
- Chrome/Chromium installed
- Selenium + webdriver-manager: `pip install selenium webdriver-manager`

### Run E2E Tests

```bash
# Headless mode (default)
./scripts/run_e2e_tests.sh

# Visible browser (for debugging)
HEADLESS=false ./scripts/run_e2e_tests.sh

# Specific test
pytest tests/e2e/test_ui.py::TestKairosDashboard::test_submit_prediction_success -v
```

### Test Coverage

- ✅ Page loading and rendering
- ✅ Form field validation
- ✅ Successful prediction submission
- ✅ Error handling (invalid input)
- ✅ Multiple sequential predictions
- ✅ Responsive design (mobile/tablet/desktop)
- ✅ API error handling
- ✅ Basic accessibility (ARIA, labels)
- ✅ Performance metrics (< 5s load time)

---

## Fixing Common Issues

### Permission Error with `.coverage`

```bash
# Remove locked coverage file
rm -f .coverage

# Re-run tests
pytest --cov=src --cov=app tests/
```

### Load Test: API Not Responding

```bash
# Check if API is running
curl http://localhost:8000/health

# Start API if needed
docker compose up api -d
```

### E2E Test: Dashboard Not Responding

```bash
# Check if dashboard is running
curl http://localhost:5000

# Start dashboard if needed
docker compose up dashboard -d
```

### Selenium WebDriver Issues

```bash
# Auto-install/update drivers
pip install --upgrade webdriver-manager

# Manual Chrome driver install (macOS)
brew install chromedriver
```

---

## CI/CD Integration

### GitHub Actions

The test suite is integrated into `.github/workflows/ci.yml`:

```yaml
- name: Run Unit & Integration Tests
  run: pytest --cov=src --cov=app --cov-fail-under=75 tests/unit/ tests/integration/

- name: Run Chaos Tests
  run: pytest tests/chaos/ --timeout=30
  continue-on-error: true # Some failures expected

- name: Run Load Tests
  run: ./scripts/run_load_tests.sh
  env:
    USERS: 50
    RUN_TIME: 30s
```

### Coverage Gates

- **Minimum Coverage**: 75% (enforced in CI)
- **Current Coverage**: 82%
- **Coverage Report**: `htmlcov/index.html`

---

## Test Data

### Sample Instances

Test data is based on UCI Adult dataset:

```python
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
    "education_num": 13
}
```

---

## Best Practices

### Writing Tests

1. **Isolation**: Each test should be independent
2. **Cleanup**: Use fixtures for setup/teardown
3. **Assertions**: Clear, specific assertions with helpful messages
4. **Mocking**: Mock external dependencies (Redis, MLflow)
5. **Timeouts**: Set reasonable timeouts for async operations

### Running Tests Locally

```bash
# Fast feedback loop (unit tests only)
pytest tests/unit/ -v

# Full validation before commit
./scripts/run_tests.sh

# Specific test file
pytest tests/unit/test_calibration.py -v

# Specific test function
pytest tests/unit/test_calibration.py::test_isotonic_calibration -v
```

---

## Troubleshooting

### Tests Hanging

```bash
# Add timeout to all tests
pytest --timeout=10 tests/

# Kill stuck processes
pkill -f pytest
```

### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use editable install
pip install -e .
```

### Flaky Tests

```bash
# Re-run failed tests
pytest --lf  # Last failed

# Run tests multiple times
pytest --count=3 tests/unit/test_policy.py
```

---

## Metrics & Reporting

### Coverage Report

```bash
# Generate HTML report
pytest --cov=src --cov=app --cov-report=html tests/

# Open in browser
open htmlcov/index.html
```

### Load Test Report

```bash
# Run load test (generates HTML report)
./scripts/run_load_tests.sh

# Open report
open outputs/load_test_report.html
```

### E2E Test Report

```bash
# Run E2E tests (generates HTML report)
./scripts/run_e2e_tests.sh

# Open report
open outputs/e2e_test_report.html
```

---

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Maintain coverage** (>75%)
3. **Update this README** if adding new test types
4. **Run full test suite** before committing

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Locust Documentation](https://docs.locust.io/)
- [Selenium Documentation](https://www.selenium.dev/documentation/)
- [Chaos Engineering Principles](https://principlesofchaos.org/)
