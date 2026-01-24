# KAIROS Development Guide

Welcome to the KAIROS development ecosystem. This project follows high-standard production ML practices.

## 🚀 Quick Start

1. **Initialize Environment**:

   ```bash
   make setup
   source venv/bin/activate
   ```

2. **Verify Installation**:
   ```bash
   make test-unit
   ```

## 🛠️ Automated Workflow (`Makefile`)

We use a `Makefile` to unify development tasks.

| Command                 | Description                                                    |
| :---------------------- | :------------------------------------------------------------- |
| `make setup`            | Create venv and install all dependencies (including dev tools) |
| `make test`             | Run the full test suite with coverage                          |
| `make test-unit`        | Run fast internal logic tests                                  |
| `make test-integration` | Run API and component interaction tests                        |
| `make lint`             | Format and lint code using Ruff                                |
| `make type-check`       | Run static analysis using Mypy                                 |
| `make train`            | Run the model training and HPO pipeline                        |
| `make api`              | Start the FastAPI service (auto-reload enabled)                |
| `make web`              | Start the dashboard (Flask)                                    |
| `make docker-up`        | Build and start the entire stack in Docker                     |

## 🧪 Testing Protocol

- **New code requires tests**: Add unit tests in `tests/unit/` for logic and integration tests in `tests/integration/` for API changes.
- **Coverage Gate**: CI will fail if total coverage drops below **75%**.
- **Run before you push**:
  ```bash
  make lint
  make type-check
  make test
  ```

## 📁 Directory Structure

- `src/kairos/api`: Multi-Engine FastAPI node & schema contracts.
- `src/kairos/web`: Decision Dashboard (Flask/CSS).
- `src/kairos/core`: Decision logic, hybrid ensembles, and calibration.
- `src/kairos/data`: Feature engineering and loaders (UCI Adult/Home Credit).
- `scripts/`: Operational tools (alerts verification, test scripts).
- `tests/`: Multi-tier testing suite (Metric Integration, Unit).
