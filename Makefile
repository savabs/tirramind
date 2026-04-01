# TirraMind — Development Automation
# Usage: make <target>
# Run `make help` to see all targets.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: dev
dev: ## Install package in editable mode with dev + quant extras
	pip install -e ".[dev,quant]"

.PHONY: hooks
hooks: ## Install pre-commit hooks
	pre-commit install

.PHONY: setup
setup: dev hooks ## Full dev setup (install + hooks)

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Run ruff linter
	ruff check agent/ tests/

.PHONY: format
format: ## Run ruff formatter
	ruff format agent/ tests/

.PHONY: format-check
format-check: ## Check formatting without modifying files
	ruff format --check agent/ tests/

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run all tests
	pytest tests/ -v --tb=short

.PHONY: test-fast
test-fast: ## Run tests excluding slow/live markers
	pytest tests/ -v --tb=short -m "not slow and not live and not integration"

.PHONY: test-file
test-file: ## Run a specific test file (usage: make test-file F=tests/test_foo.py)
	pytest $(F) -v --tb=short

.PHONY: coverage
coverage: ## Run tests with coverage report
	pytest tests/ --cov=agent --cov-report=term-missing --tb=short

# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

.PHONY: check
check: lint test ## Lint + test (CI equivalent)

.PHONY: ci
ci: lint format-check test ## Full CI check (lint + format + test)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove build/cache artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
