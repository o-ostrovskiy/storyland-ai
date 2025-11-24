# StoryLand AI - Makefile
# Common commands for development and demo

.PHONY: help install test test-cov lint run run-dev eval clean db-reset

# Default target
help:
	@echo "StoryLand AI - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies"
	@echo "  make install-dev   Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all unit tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make test-agents   Run agent tests only"
	@echo "  make eval          Run ADK evaluation (single test)"
	@echo ""
	@echo "Running:"
	@echo "  make run           Run demo (Pride and Prejudice)"
	@echo "  make run-1984      Run demo (1984 by George Orwell)"
	@echo "  make run-gatsby    Run demo (The Great Gatsby)"
	@echo "  make run-dev       Start ADK Web UI"
	@echo "  make run-verbose   Run with verbose logging"
	@echo ""
	@echo "Database:"
	@echo "  make db-reset      Delete SQLite database"
	@echo "  make db-show       Show recent sessions"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Clean cache files"
	@echo "  make check-env     Verify environment setup"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# =============================================================================
# Testing
# =============================================================================

test:
	.venv/bin/pytest tests/unit/ -v

test-cov:
	.venv/bin/pytest tests/unit/ --cov=. --cov-report=term-missing

test-agents:
	.venv/bin/pytest tests/unit/test_agents.py -v

test-models:
	.venv/bin/pytest tests/unit/test_models.py -v

test-tools:
	.venv/bin/pytest tests/unit/test_tools.py -v

test-services:
	.venv/bin/pytest tests/unit/test_services.py -v

# =============================================================================
# ADK Evaluation
# =============================================================================

eval:
	.venv/bin/adk eval agents/storyland single_test \
		--config_file_path tests/evaluation/eval_config.json \
		--print_detailed_results

# =============================================================================
# Running
# =============================================================================

run:
	.venv/bin/python main.py "Pride and Prejudice" --author "Jane Austen"

run-1984:
	.venv/bin/python main.py "1984" --author "George Orwell"

run-gatsby:
	.venv/bin/python main.py "The Great Gatsby" --author "F. Scott Fitzgerald"

run-nightingale:
	.venv/bin/python main.py "The Nightingale" --author "Kristin Hannah"

run-luxury:
	.venv/bin/python main.py "Pride and Prejudice" --author "Jane Austen" \
		--budget luxury --pace relaxed --museums

run-family:
	.venv/bin/python main.py "Harry Potter" --author "J.K. Rowling" \
		--with-kids --budget moderate

run-verbose:
	.venv/bin/python main.py "Pride and Prejudice" --author "Jane Austen" -v

run-db:
	.venv/bin/python main.py "Pride and Prejudice" --author "Jane Austen" --database

run-dev:
	.venv/bin/python main.py --dev

# =============================================================================
# Database
# =============================================================================

db-reset:
	rm -f storyland_sessions.db
	@echo "Database deleted"

db-show:
	@sqlite3 -column -header storyland_sessions.db \
		"SELECT id, user_id, json_extract(state, '\$$.book_title') as book, create_time FROM sessions ORDER BY create_time DESC LIMIT 5;" \
		2>/dev/null || echo "No database found"

db-users:
	@sqlite3 storyland_sessions.db \
		"SELECT user_id, COUNT(*) as sessions FROM sessions GROUP BY user_id;" \
		2>/dev/null || echo "No database found"

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "Cache cleaned"

check-env:
	@echo "Checking environment..."
	@.venv/bin/python -c "import os; from dotenv import load_dotenv; load_dotenv(); key = os.getenv('GOOGLE_API_KEY'); print('GOOGLE_API_KEY:', 'OK' if key else 'MISSING')"
	@.venv/bin/python -c "import google.adk; print('google-adk:', 'OK')"
	@.venv/bin/python -c "import google.genai; print('google-genai:', 'OK')"
	@echo "Environment OK"

# =============================================================================
# Jupyter
# =============================================================================

notebook:
	jupyter notebook

lab:
	jupyter lab
