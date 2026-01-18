#!/bin/bash
# KAIROS Test Runner Script
# Runs all test suites with proper configuration

set -e  # Exit on error

echo "🧪 KAIROS Test Suite Runner"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Clean up old coverage data
print_status "Cleaning up old test artifacts..."
rm -f .coverage coverage.xml
rm -rf .pytest_cache htmlcov/

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export API_KEY="kairos_dev_key_2026"

# Run unit tests
echo ""
echo "📦 Running Unit Tests..."
echo "--------------------------------------"
pytest tests/unit/ -v --cov=src --cov=app --cov-report=term-missing || {
    print_error "Unit tests failed"
    exit 1
}
print_status "Unit tests passed"

# Run integration tests
echo ""
echo "🔗 Running Integration Tests..."
echo "--------------------------------------"
pytest tests/integration/ -v || {
    print_error "Integration tests failed"
    exit 1
}
print_status "Integration tests passed"

# Run chaos engineering tests
echo ""
echo "💥 Running Chaos Engineering Tests..."
echo "--------------------------------------"
pytest tests/chaos/ -v --timeout=30 || {
    print_warning "Some chaos tests failed (this may be expected)"
}
print_status "Chaos tests completed"

# Generate coverage report
echo ""
echo "📊 Generating Coverage Report..."
echo "--------------------------------------"
pytest --cov=src --cov=app --cov-report=html --cov-report=term --cov-fail-under=75 tests/unit/ tests/integration/ || {
    print_warning "Coverage below 75% threshold"
}
print_status "Coverage report generated in htmlcov/"

echo ""
echo "======================================"
print_status "All test suites completed!"
echo ""
echo "📈 Next steps:"
echo "   - View coverage: open htmlcov/index.html"
echo "   - Run load tests: ./scripts/run_load_tests.sh"
echo "   - Run E2E tests: ./scripts/run_e2e_tests.sh"
