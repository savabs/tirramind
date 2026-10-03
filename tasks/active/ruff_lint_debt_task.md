---
title: "Task: Ruff Lint Debt Remediation"
tags:
  - doc/task
  - status/active
  - topic/tooling
  - topic/ci
---

# Task: Ruff Lint Debt Remediation

Status: active
Research: `docs/research/ruff_lint_debt.md`
Spec: `docs/specs/ruff_lint_debt_spec.md`

## Goal

Make the CI `lint` job green — `ruff check agent/ tests/` and
`ruff format --check agent/ tests/` both exit 0 — without changing runtime
behaviour or the test suite result.

## Scope Notes

- Layer: none — this is repo hygiene, not feature work. It touches files across
  layers 1–7 but changes no logic in any of them.
- Main files expected to change: 75 files for the semantic pass, 111 for the
  format pass. None of the four `CLAUDE.md` §9 critical files.
- Non-goals: loosening `pyproject.toml` ignores to hide real findings;
  upgrading the pinned pre-commit ruff (tracked separately below).

## Steps

- [x] 1.1: Record the `ruff check --statistics` baseline
  Verification: 211 errors; distribution captured in the research note
- [x] 1.2: Record the pytest baseline from a pristine worktree at branch HEAD
  Verification: 5 failed, 10938 passed, 15 skipped, 9 deselected (863s)
- [x] 2.1: Rewrite the 22 single-line `datetime.UTC` shims
  Verification: `ruff check --select E702` reports 0
- [x] 3.1: Import `Any` in `cde_encoder.py` (F821)
  Verification: `ruff check --select F821` reports 0
- [x] 3.2: Drop the unused functional import in `heterogeneous_cde_func.py` (F811)
  Verification: `ruff check --select F811` reports 0
- [x] 4.1: Apply `ruff check --fix` (safe fixes only)
  Verification: 165 then 32 fixed across two passes; 0 remain
- [x] 2.3: Delete the 32 `UTC = UTC` self-assignments the fixer leaves behind
  Verification: `grep -rn "UTC = UTC" agent/ tests/` is empty
- [x] 4.5: Import-smoke-test every changed `agent/` module
  Verification: 64/64 import cleanly
- [x] 5.1: Run `ruff format` as a separate commit
  Verification: 92 files reformatted; AST identical for all 92
- [x] 6.2: Re-run the suite and diff against the baseline
  Verification: see Outcome below

## Completion Checklist

- [ ] Research note exists and is current
- [ ] Spec matches the actual implementation plan
- [ ] Each completed step has a verification result
- [ ] Edge-case testing was added and run for code changes
- [ ] Checkpoint written at the end of the session or sub-phase
- [ ] Frontmatter tags and `## Related` section are current

## Follow-up

The underlying cause is a version split: `.pre-commit-config.yaml` pins ruff
`v0.4.8` while CI installs the latest. Cleaning the current errors does not stop
the drift from recurring the next time ruff ships a rule. Aligning the two —
either by pinning CI or bumping pre-commit — is worth a separate task.

## Related

- [[ruff_lint_debt]]
- [[ruff_lint_debt_spec]]
