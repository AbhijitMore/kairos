#!/bin/bash
# KAIROS End-to-End UI Testing Script
# Runs Selenium tests against the dashboard

set -e

echo "🎭 KAIROS End-to-End UI Testing"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:5000}"
HEADLESS="${HEADLESS:-true}"

echo -e "${GREEN}Configuration:${NC}"
echo "  Dashboard URL: $DASHBOARD_URL"
echo "  Headless Mode: $HEADLESS"
echo ""

# Check if dashboard is running
echo "Checking dashboard availability..."
if ! curl -s -f "$DASHBOARD_URL" > /dev/null; then
    echo -e "${YELLOW}[!] Dashboard not responding at $DASHBOARD_URL${NC}"
    echo "    Start the dashboard with: docker compose up dashboard"
    exit 1
fi
echo -e "${GREEN}[✓] Dashboard is running${NC}"
echo ""

# Check dependencies
if ! python -c "import selenium" 2>/dev/null; then
    echo -e "${YELLOW}[!] Selenium not installed${NC}"
    echo "    Install with: pip install selenium webdriver-manager"
    exit 1
fi

# Set environment variables
export DASHBOARD_URL
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run E2E tests
echo "Starting E2E tests..."
echo "======================================"

if [ "$HEADLESS" = "true" ]; then
    pytest tests/e2e/ -v --headless --html=outputs/e2e_test_report.html --self-contained-html
else
    pytest tests/e2e/ -v --html=outputs/e2e_test_report.html --self-contained-html
fi

echo ""
echo "======================================"
echo -e "${GREEN}[✓] E2E tests completed!${NC}"
echo ""
echo "📊 Report generated: outputs/e2e_test_report.html"
echo ""
echo "💡 Tips:"
echo "   - Run with visible browser: HEADLESS=false ./scripts/run_e2e_tests.sh"
echo "   - Test specific scenario: pytest tests/e2e/test_ui.py::TestKairosDashboard::test_submit_prediction_success -v"
