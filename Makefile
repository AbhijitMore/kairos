# KAIROS Development Makefile
# Standardized commands for FAANG-grade workflow

.PHONY: setup test test-unit test-integration lint type-check train api web clean docker-up docker-down

# --- Installation ---
setup:
	@echo "🚀 Setting up development environment..."
	./scripts/install_dev.sh

# --- Testing ---
test:
	@echo "🧪 Running full test suite..."
	./scripts/run_tests.sh

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
	mypy --explicit-package-bases src/kairos

# --- Execution ---
train:
	@echo "🏋️ Training KAIROS model..."
	PYTHONPATH=src python train.py --hpo --trials 10

api:
	@echo "📡 Starting FastAPI Service..."
	PYTHONPATH=src uvicorn kairos.api.main:app --host 0.0.0.0 --port 8000 --reload

web:
	@echo "📊 Starting Dashboard (Flask)..."
	PYTHONPATH=src python src/kairos/web/app.py

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
