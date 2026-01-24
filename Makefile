# KAIROS Development Makefile
# Standardized commands for FAANG-grade workflow

.PHONY: setup test test-unit test-integration lint type-check train api web clean docker-up docker-down alert-check

# --- Installation ---
setup:
	@echo "🚀 Setting up development environment..."
	./scripts/install_dev.sh

# --- Testing ---
test:
	@echo "🧪 Running full test suite..."
	PYTHONPATH=src pytest tests/unit tests/integration

test-unit:
	@echo "📦 Running unit tests..."
	PYTHONPATH=src pytest -m unit --no-cov

test-integration:
	@echo "🔗 Running integration tests..."
	PYTHONPATH=src pytest -m integration --no-cov

# --- Code Quality ---
lint:
	@echo "🧹 Linting with Ruff..."
	ruff check .
	ruff format .

type-check:
	@echo "🔎 Type checking with Mypy..."
	PYTHONPATH=src mypy --explicit-package-bases src/kairos

alert-check:
	@echo "🛡️ Verifying AlertManager Rules..."
	python scripts/verify_alerts.py

# --- Execution ---
train:
	@echo "🏋️ Training KAIROS model..."
	PYTHONPATH=src python train.py --hpo --trials 10

api:
	@echo "📡 Starting FastAPI Service..."
	PYTHONPATH=src uvicorn kairos.api.main:app --host 0.0.0.0 --port 8001 --reload

web:
	@echo "📊 Starting Dashboard (Flask)..."
	PYTHONPATH=src API_URL=http://localhost:8001/api/v1/predict flask --app src/kairos/web/app.py run --port 5001

# --- Infrastructure ---
docker-up:
	@echo "🐳 Starting Docker stack..."
	docker-compose up --build -d

docker-down:
	@echo "🛑 Stopping Docker stack..."
	docker-compose down

# --- Cleanup ---
clean:
	@echo "🧹 Cleaning up artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f .coverage coverage.xml
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
