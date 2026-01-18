#!/bin/bash
# KAIROS Development Setup Script

set -e

echo "🚀 Starting KAIROS Development Environment Setup..."

# 1. Environment Check
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.12+"
    exit 1
fi

# 2. Virtual Environment Setup
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# 3. Dependency Installation
echo "pip upgrading..."
pip install --upgrade pip

echo "🛠️ Installing core and development dependencies..."
pip install -e ".[dev]"

# 4. System Checks
echo "🔍 Checking infrastructure..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping | grep -q "PONG"; then
        echo "✅ Redis is running."
    else
        echo "⚠️  Redis is installed but not running. Async tasks will fail."
    fi
else
    echo "⚠️  Redis is not installed. Use 'docker-compose up -d redis' if needed."
fi

echo ""
echo "✨ Setup complete!"
echo "👉 To start developing, run: 'source venv/bin/activate'"
echo "👉 Run 'make test' to verify your installation."
