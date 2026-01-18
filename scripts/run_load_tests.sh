#!/bin/bash
# KAIROS Load Testing Script
# Runs Locust load tests against the API

set -e

echo "🚀 KAIROS Load Testing Suite"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
API_HOST="${API_HOST:-http://localhost:8000}"
USERS="${USERS:-100}"
SPAWN_RATE="${SPAWN_RATE:-10}"
RUN_TIME="${RUN_TIME:-60s}"

echo -e "${GREEN}Configuration:${NC}"
echo "  API Host: $API_HOST"
echo "  Users: $USERS"
echo "  Spawn Rate: $SPAWN_RATE users/sec"
echo "  Run Time: $RUN_TIME"
echo ""

# Check if API is running
echo "Checking API availability..."
if ! curl -s -f "$API_HOST/health" > /dev/null; then
    echo -e "${YELLOW}[!] API not responding at $API_HOST${NC}"
    echo "    Start the API with: docker compose up api"
    exit 1
fi
echo -e "${GREEN}[✓] API is running${NC}"
echo ""

# Check if locust is installed
if ! command -v locust &> /dev/null; then
    echo -e "${YELLOW}[!] Locust not installed${NC}"
    echo "    Install with: pip install locust"
    exit 1
fi

# Run load test
echo "Starting load test..."
echo "======================================"
locust \
    -f tests/load/test_load.py \
    --host="$API_HOST" \
    --users="$USERS" \
    --spawn-rate="$SPAWN_RATE" \
    --run-time="$RUN_TIME" \
    --headless \
    --html=outputs/load_test_report.html \
    --csv=outputs/load_test_results

echo ""
echo "======================================"
echo -e "${GREEN}[✓] Load test completed!${NC}"
echo ""
echo "📊 Reports generated:"
echo "   - HTML: outputs/load_test_report.html"
echo "   - CSV: outputs/load_test_results_*.csv"
echo ""
echo "💡 Tips:"
echo "   - Run with more users: USERS=1000 ./scripts/run_load_tests.sh"
echo "   - Run longer: RUN_TIME=5m ./scripts/run_load_tests.sh"
echo "   - Interactive mode: locust -f tests/load/test_load.py --host=$API_HOST"
