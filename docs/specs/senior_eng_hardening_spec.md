---
title: "Spec: senior_eng_hardening"
tags:
  - doc/spec
  - layer/llm-support
  - topic/engineering
---

# Spec: senior_eng_hardening

## Goal

Transform TirraMind from a well-documented but manually enforced workflow into a mechanically hardened, production-grade development process that a senior engineer would recognize as their own shop.

## Files Affected

### Create
- `Makefile` — build automation (test, lint, format, check, dev-setup)
- `tests/conftest.py` — shared pytest fixtures
- `.pre-commit-config.yaml` — commit-time quality gates
- `.github/workflows/ci.yml` — GitHub Actions CI pipeline
- `README.md` — project overview, setup, usage
- `tasks/done/.gitkeep` — archive directory
- `agent/pipeline/.instructions.md` — Layer 7 pipeline conventions
- `agent/config/.instructions.md` — config module conventions
- `agent/core/.instructions.md` — orchestrator module conventions
- `.github/prompts/spec-to-task.prompt.md` — convert spec → task file
- `.github/prompts/debug.prompt.md` — structured debug workflow
- `.github/prompts/post-mortem.prompt.md` — incident/failure retrospective
- `.github/agents/architect.agent.md` — architecture decision agent
- `[[TEMPLATE]]` — ADR template
- `[[0001-pipeline-no-llm]]` — first ADR (captures existing decision)

### Modify
- `pyproject.toml` — add quant deps, pytest/ruff/mypy config
- `AGENTS.md` — add architect agent reference

## Implementation Steps

### Step 1: pyproject.toml — declare truth about dependencies and tools
- Add quant optional dependency group (numpy, scipy, statsmodels, hmmlearn, filterpy, numpyro, pgmpy, cvxpy)
- Add [tool.pytest.ini_options] — testpaths, markers, addopts, filterwarnings
- Add [tool.ruff] — target-version, line-length, select rules
- Add [tool.ruff.lint] — rule selection (E, F, W, I, UP, B, SIM, S)
- Add [tool.ruff.format] — quote-style, indent-style

### Step 2: Makefile — one command to do anything
- test: run pytest
- lint: run ruff check
- format: run ruff format
- typecheck: run pyright/mypy (if available, soft-fail)
- check: lint + test combined
- dev: install editable with dev+quant extras
- clean: remove __pycache__, .pytest_cache, etc.

### Step 3: tests/conftest.py — shared test infrastructure
- mock_httpx_client fixture (patched httpx.AsyncClient)
- mock_cache fixture (in-memory DataCache)
- tmp_db fixture (temp SQLite for pipeline tests)
- sample_tool_result fixture
- sample_time_series fixture (numpy array with known properties)
- auto-register markers

### Step 4: .pre-commit-config.yaml — commit-time quality gates
- ruff check (fast linter)
- ruff format (fast formatter)
- trailing-whitespace, end-of-file-fixer, check-yaml, check-toml
- No slow hooks (no mypy, no full test suite — those run in CI)

### Step 5: .github/workflows/ci.yml — automated testing on push/PR
- Trigger: push to main, all PRs
- Matrix: Python 3.11, 3.12
- Steps: checkout, setup-python, install deps, ruff check, pytest
- Cache pip for speed

### Step 6: README.md — onboarding in 5 minutes
- One-paragraph description
- Architecture diagram (text-based)
- Quick start (clone, install, test)
- Project structure table
- Development workflow summary
- Link to project_memory.md for deep dive

### Step 7: tasks/done/.gitkeep — archive for completed tasks

### Step 8: Module .instructions.md files
- agent/pipeline/.instructions.md — DAG conventions, operator patterns, no LLM, SQLite
- agent/config/.instructions.md — env-var config, frozen dataclasses, TIRRA_ prefix
- agent/core/.instructions.md — orchestrator conventions, pipeline ordering

### Step 9: New prompts
- spec-to-task.prompt.md — reads spec, generates task file from TASK_TEMPLATE.md
- debug.prompt.md — structured diagnosis: reproduce → instrument → capture → hypothesis → fix → regress
- post-mortem.prompt.md — after a hard bug or failed approach, capture: what happened, root cause, what we learned, what changes

### Step 10: Architect agent + ADR system
- architect.agent.md — read-only agent that writes ADR docs
- [[TEMPLATE]] — standard ADR format (Status, Context, Decision, Consequences)
- [[0001-pipeline-no-llm]] — first real ADR capturing the existing pipeline decision

### Step 11: Update AGENTS.md — add architect agent

## Edge Cases

- Pre-commit hooks must not break if ruff is not installed (CI installs it)
- CI must not fail on missing optional quant deps in basic test run
- Makefile must work on Linux and macOS
- conftest.py fixtures must not conflict with existing test mocking patterns

## Testing Plan

- Run `make check` after implementation to validate Makefile + pytest + ruff
- Verify pre-commit config is valid YAML
- Verify CI workflow is valid YAML (act or manual inspection)
- Run existing test suite to ensure conftest.py doesn't break anything

---

## Related

- [[senior_eng_hardening|Research: Senior Eng Hardening]]
