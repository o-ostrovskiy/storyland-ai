# StoryLand AI - Makefile
# Common commands for development and demo

.PHONY: help install install-dev test test-cov test-agents test-models test-tools test-services test-api test-integration test-integration-live test-integration-vcr-record test-all eval eval-setup eval-setup-one eval-run eval-summary eval-report eval-export run run-api run-1984 run-gatsby run-nightingale run-luxury run-family run-verbose run-db run-dev db-reset db-show db-users clean check-env notebook lab

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
	@echo "  make test-api        Run API unit tests"
	@echo "  make test-integration        Run integration tests (VCR)"
	@echo "  make test-integration-live   Run live integration tests (real API/network)"
	@echo "  make test-integration-vcr-record  Re-record VCR cassettes"
	@echo "  make test-all      Run all tests (unit + integration)"
	@echo "  make eval          Run ADK evaluation (single test)"
	@echo ""
	@echo "Evaluation (Langfuse):"
	@echo "  make eval-setup              Create Langfuse datasets from all evalsets"
	@echo "  make eval-setup-one EVALSET_FILE=<path>  Register a single evalset file"
	@echo "  make eval-run      Run scheduled evaluations"
	@echo "  make eval-report   Generate evaluation trend report"
	@echo "  make eval-summary  Show evaluation summary"
	@echo ""
	@echo "Running:"
	@echo "  make run-api       Start FastAPI SSE server"
	@echo "  make run           Run demo (Pride and Prejudice)"
	@echo "  make run-1984      Run demo (1984 by George Orwell)"
	@echo "  make run-gatsby    Run demo (The Great Gatsby)"
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

test-api:
	.venv/bin/pytest tests/unit/test_api.py -v

test-integration:
	.venv/bin/pytest tests/integration/ -v -m "integration and not real_api"

test-integration-vcr-record:
	.venv/bin/pytest tests/integration/ -v --vcr-record=all

test-integration-live:
	.venv/bin/pytest tests/integration/ -v -m "integration and real_api"

test-all:
	.venv/bin/pytest tests/ -v

# =============================================================================
# Evaluation Pipeline
# =============================================================================

eval-setup:
	.venv/bin/python evaluation/tools/langfuse_eval.py --create-datasets --evalset-dir evaluation
	@echo "✅ Langfuse datasets created. View at your Langfuse dashboard."

eval-setup-one:
	.venv/bin/python evaluation/tools/langfuse_eval.py --create-datasets --evalset-file $(EVALSET_FILE)
	@echo "✅ Dataset registered. View at your Langfuse dashboard."

eval-run:
	PYTHONIOENCODING=utf-8 .venv/bin/python evaluation/tools/run_scheduled_eval.py --output-dir evaluation/results --max-cases 10

eval-summary:
	.venv/bin/python evaluation/tools/eval_dashboard.py --action summary --days 7

eval-report:
	.venv/bin/python evaluation/tools/eval_dashboard.py --action report --days 30
	@echo "✅ Trend report generated at evaluation/trend_report.md"

eval-export:
	.venv/bin/python evaluation/tools/eval_dashboard.py --action export --days 30
	@echo "✅ Metrics exported to evaluation/metrics.json"

# =============================================================================
# Running
# =============================================================================

run-api:
	.venv/bin/uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8080 --reload


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
	rm -rf build/ storyland_ai.egg-info/ .coverage htmlcov/ 2>/dev/null || true
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
