#!/bin/bash

# run_all_tests.sh - Automated test runner for Santa Fe Double Auction
# 
# This script provides one-click testing capability for the entire codebase
# Runs unit tests, integration tests, and generates coverage reports

set -e  # Exit on any error

echo "==============================================="
echo "Santa Fe Double Auction - Automated Test Suite"
echo "==============================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "main.py" ] && [ ! -f "src_code/main.py" ] && [ ! -f "code/main.py" ]; then
    echo "Error: Please run this script from the repository root directory"
    exit 1
fi

# Navigate to correct directory if needed
if [ -f "src_code/main.py" ] || [ -f "code/main.py" ]; then
    echo "Setting up environment from repository root..."
else
    echo "Already in correct directory..."
fi

# Check Python and dependencies
echo "🐍 Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if pytest is available or set up virtual environment
USING_VENV=false
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "⚠️  pytest not found. Setting up virtual environment..."
    echo ""
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    USING_VENV=true
    
    # Install all requirements including pytest
    echo "Installing requirements including pytest and coverage..."
    pip install -r requirements.txt
    
    echo "✅ pytest installed in virtual environment"
else
    echo "✅ pytest is available"
fi

echo ""
echo "🧪 Running comprehensive test suite..."
echo "----------------------------------------"

# Run all tests with coverage (use proper python command based on environment)
if [ "$USING_VENV" = true ]; then
    # In venv, use direct python/pytest
    python -m pytest tests/ \
        --verbose \
        --tb=short \
        --strict-markers \
        --disable-warnings \
        --cov=src_code \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        --cov-branch \
        --cov-fail-under=40
else
    # Use python3 for system install
    python3 -m pytest tests/ \
        --verbose \
        --tb=short \
        --strict-markers \
        --disable-warnings \
        --cov=src_code \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        --cov-branch \
        --cov-fail-under=40
fi

TEST_EXIT_CODE=$?

echo ""
echo "----------------------------------------"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed successfully!"
    echo ""
    echo "📊 Coverage Report:"
    echo "   - Terminal output: See above"
    echo "   - HTML report: htmlcov/index.html"
    echo ""
    echo "🎉 One-click testing complete!"
else
    echo "❌ Some tests failed!"
    echo "   Exit code: $TEST_EXIT_CODE"
    echo "   Check the output above for details"
    echo ""
fi

# Optional: Run quick smoke test of main auction
echo ""
echo "🔥 Running quick smoke test of auction system..."
echo "------------------------------------------------"

# Use the appropriate python command
PYTHON_CMD="python3"
if [ "$USING_VENV" = true ]; then
    PYTHON_CMD="python"
fi

if $PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, 'src_code')
from auction import Auction
from traders.registry import get_trader_class

# Quick auction with ZIC vs ZIP
config = {
    'experiment_name': 'smoke_test',
    'num_rounds': 1,
    'num_periods': 1, 
    'num_steps': 5,
    'num_buyers': 1,
    'num_sellers': 1,
    'num_tokens': 1,
    'min_price': 1,
    'max_price': 200,
    'gametype': 0,
    'buyers': [{'class': get_trader_class('zic', is_buyer=True)}],
    'sellers': [{'class': get_trader_class('zip', is_buyer=False)}],
    'rng_seed_auction': 42,
    'rng_seed_values': 123,
}

auction = Auction(config)
auction.run_auction()
print('✅ Smoke test passed - auction system working!')
"; then
    echo "✅ Auction smoke test passed!"
else
    echo "⚠️  Auction smoke test failed - check main system"
fi

echo ""
echo "==============================================="
echo "Test run completed!"
echo "==============================================="

exit $TEST_EXIT_CODE