---
title: "Feature: Senior Engineering Process Hardening"
tags:
  - doc/research
  - topic/engineering
---

# Feature: Senior Engineering Process Hardening

## Goal

Close every gap between "well-documented workflow" and "mechanically enforced, production-grade development process." Make TirraMind's development feel like a senior engineer's shop: creative but reliable, fast but disciplined.

## Current Architecture

### What exists (strong)
- 3-tier instruction hierarchy: copilot-instructions.md → AGENTS.md → RulesForAI.md
- 3 folder-level .instructions.md (tools, quant, tests)
- 6 prompts (brainstorm-to-spec, full-pipeline, next-step, research, review-quant, sprint)
- 3 custom agents (code-reviewer, quant-researcher, test-writer)
- RESEARCH_TEMPLATE.md, TASK_TEMPLATE.md
- project_memory.md (5000+ lines of DNA)
- 48+ test files, 3000+ tests passing

### What's missing (gaps)
1. **No build automation** — no Makefile/justfile. Developers must know all commands.
2. **No CI/CD** — no GitHub Actions. Breaking changes merge silently.
3. **No pre-commit hooks** — code quality not gated at commit time.
4. **No conftest.py** — shared fixtures duplicated across 48 test files.
5. **No pyproject.toml tooling config** — pytest, ruff, mypy not configured.
6. **Quant dependencies missing** — numpy, scipy, etc. not declared.
7. **No README** — no entry point for onboarding.
8. **No tasks/done/** — completed tasks accumulate in active/.
9. **Missing .instructions.md** files for pipeline/, config/, core/, memory/, learning/ modules.
10. **No spec-to-task prompt** — referenced as next step but never built.
11. **No debug prompt** — Rule 6 says "debug before guessing" but no structured workflow.
12. **No post-mortem/retrospective prompt** — senior engineers reflect on what went wrong.
13. **No architectural decision record (ADR) system** — architecture decisions buried in task files.
14. **No dependency audit** — chromadb declared but unused, quant libs missing.

## Observations

### A senior engineer's shop includes:
1. **One command to do anything** — `make test`, `make lint`, `make check`
2. **Guardrails at commit time** — pre-commit hooks catch problems before they land
3. **CI runs on every push** — tests, lint, type-check. Green = safe to merge.
4. **Shared test infrastructure** — conftest.py with reusable fixtures (mock HTTP, temp dirs, sample data)
5. **Configured tools in pyproject.toml** — pytest markers, ruff rules, test paths
6. **Complete dependency list** — if it imports, it's declared
7. **README that gets you running in 5 minutes** — setup, test, contribute
8. **Completed work is archived** — tasks/done/ with history
9. **Every module has conventions** — .instructions.md for every non-trivial directory
10. **Decision records** — why architecture X was chosen over Y, captured once and referenced forever
11. **Debug workflow** — structured approach to diagnosing problems, not random guessing
12. **Post-mortem culture** — after a hard bug or failed approach, capture what was learned

## Risks

- Over-engineering: adding too much process for a solo/small team. Keep configs minimal.
- CI costs: GitHub Actions free tier (2000 min/month) should be plenty.
- Pre-commit friction: must be fast (<5s) or developers will skip them.
- README maintenance: keep it short and auto-derivable from existing docs.

## Implementation Intent

Create all missing infrastructure in one cohesive pass:
1. Makefile (build automation)
2. pyproject.toml tooling section (pytest + ruff config)
3. conftest.py (shared fixtures)
4. .pre-commit-config.yaml (commit-time quality gates)
5. GitHub Actions CI workflow (test + lint on push)
6. Fix dependency declarations
7. README.md (quick start)
8. tasks/done/ directory
9. Module .instructions.md files (pipeline, config, core)
10. New prompts: spec-to-task, debug, post-mortem
11. New agent: architect (ADR writer)
12. ADR template + directory

---

## Related

- [[senior_eng_hardening_spec|Spec: Senior Eng Hardening]]
