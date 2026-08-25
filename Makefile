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

.PHONY: cursor-rules
cursor-rules: ## Regenerate .cursor/rules from .instructions.md files
	python3 scripts/generate_cursor_rules.py

.PHONY: cursor-skills
cursor-skills: ## Migrate .github/prompts to .cursor/skills
	python3 scripts/migrate_prompts_to_skills.py

.PHONY: cursor-setup
cursor-setup: cursor-rules cursor-skills ## Regenerate Cursor rules + skills

.PHONY: mcp-setup
mcp-setup: ## Create/sync .cursor/mcp.json from example + .env keys
	python3 scripts/setup_mcp.py

.PHONY: awos-hooks
awos-hooks: ## Install AWOS git hooks (post-commit/post-merge/pre-push scan)
	python3 -m agent.awos.cli install-hooks

.PHONY: setup
setup: dev hooks cursor-setup mcp-setup awos-hooks ## Full dev setup (hooks, cursor, MCP, AWOS)

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

.PHONY: vault-lint
vault-lint: ## Obsidian + fact drift linters (matches CI vault-lint job)
	python3 scripts/obsidian_lint.py --strict --no-stale
	python3 scripts/fact_lint.py

.PHONY: quality-gate
quality-gate: ## Pre-completion gate: pytest + vault lint + active task steps
	python3 scripts/quality_gate.py

.PHONY: quality-gate-fast
quality-gate-fast: ## Quality gate without pytest (vault lint + task steps only)
	python3 scripts/quality_gate.py --skip-tests

.PHONY: ghost-pattern-daily
ghost-pattern-daily: ## MP-1 loop: refresh sensors, scan chains, resolve alerts
	python3 scripts/ghost_pattern_daily.py

.PHONY: ghost-readout-backfill
ghost-readout-backfill: ## Backfill CL=F/BZ=F/NG=F daily bars for alert resolution
	python3 scripts/backfill_readout_prices.py

.PHONY: ghost-ais-backfill
ghost-ais-backfill: ## Backfill AIS daily activity (port-call proxy + live snapshot)
	python3 scripts/backfill_ais_daily.py

.PHONY: ghost-archive-publish
ghost-archive-publish: ## Push briefs + alerts to public GitHub archive repo
	bash scripts/publish_ghost_archive.sh

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run all tests
	pytest tests/ -v --tb=short

.PHONY: test-fast
test-fast: ## Run tests excluding slow/live markers
	pytest tests/ -v --tb=short -m "not slow and not live and not integration"

.PHONY: run
run: ## Refresh data + build + deliver the Intelligence Brief (no serve)
	.venv/bin/tirra-engine --once

.PHONY: serve
serve: ## Serve the Intelligence Brief over HTTP
	.venv/bin/tirra-serve

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

.PHONY: kaggle-loop
kaggle-loop: ## Bootstrap V64+ and run autonomous Kaggle watch→analyze→push loop
	python3 scripts/kaggle_loop.py --bootstrap-v64 --loop --max-runs 4 --interval 90

.PHONY: kaggle-analyze
kaggle-analyze: ## Analyze latest Kaggle logs and print next-config decision
	python3 scripts/kaggle_loop.py --once

.PHONY: honest-baseline
honest-baseline: ## Phase A honest baseline audit (smoke, IC-only)
	python3 scripts/honest_baseline_audit.py --smoke

.PHONY: honest-baseline-full
honest-baseline-full: ## Full honest baseline audit (IC-only, canonical labels)
	python3 scripts/honest_baseline_audit.py --out .tirra_pipeline/honest_baseline_audit_full.json

.PHONY: data-label-audit
data-label-audit: ## DATA_FIX: label leakage + backfill shift audit
	python3 scripts/data_label_audit.py --smoke

.PHONY: stage1-ssl-train
stage1-ssl-train: ## Phase B.1: local SSL-only HetTGN training (smoke 2ep)
	python3 scripts/retrain_gnn.py --preset phase50_stage1_ssl --epochs 2 --skip-eval --hidden-dim 128 --num-layers 2 --num-heads 4

.PHONY: export-embeddings
export-embeddings: ## Phase B.2: export GNN embeddings per fold (smoke)
	python3 scripts/export_gnn_embeddings.py --checkpoint .tirra_pipeline/gnn_model_phase50.pt --weights-from-epoch .tirra_pipeline/checkpoints/phase50/epoch_090.pt --smoke

.PHONY: stage2-eval
stage2-eval: ## Phase B.3: Stage2 Ridge ranker vs Momentum (smoke)
	python3 scripts/stage2_ranker_eval.py --checkpoint .tirra_pipeline/gnn_model_phase50.pt --weights-from-epoch .tirra_pipeline/checkpoints/phase50/epoch_090.pt --smoke

.PHONY: stage2-eval-full
stage2-eval-full: ## Phase B.3 full 40-fold Stage2 ranker eval
	python3 scripts/stage2_ranker_eval.py --checkpoint .tirra_pipeline/gnn_model_phase50.pt --weights-from-epoch .tirra_pipeline/checkpoints/phase50/epoch_090.pt

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
