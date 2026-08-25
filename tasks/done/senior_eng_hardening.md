---
title: "Task: senior_eng_hardening"
tags:
  - doc/task
  - status/done
  - topic/engineering
---

# Task: senior_eng_hardening

Status: completed
Research: [[senior_eng_hardening]]
Spec: [[senior_eng_hardening_spec]]

## Goal

Close every gap between documented workflow and mechanically enforced process. Make development feel like a senior engineer's shop.

## Steps

- [x] 1: pyproject.toml — add quant deps + tool config (pytest, ruff)
- [x] 2: Makefile — build automation targets
- [x] 3: tests/conftest.py — shared fixtures
- [x] 4: .pre-commit-config.yaml — commit hooks
- [x] 5: .github/workflows/ci.yml — CI pipeline
- [x] 6: README.md — onboarding
- [x] 7: tasks/done/.gitkeep — archive directory
- [x] 8: Module .instructions.md files (pipeline, config, core)
- [x] 9: New prompts (spec-to-task, debug, post-mortem)
- [x] 10: Architect agent + ADR system
- [x] 11: Update AGENTS.md

## Completion Checklist

- [x] `make help` shows all targets
- [x] Existing test suite still passes (127 pipeline, 122 lobbying — 2 pre-existing count fails only)
- [x] All new files validated (YAML, TOML syntax verified)
- [x] Checkpoint written

---

## Related

- [[senior_eng_hardening|Research: Senior Eng Hardening]]
- [[senior_eng_hardening_spec|Spec: Senior Eng Hardening]]
