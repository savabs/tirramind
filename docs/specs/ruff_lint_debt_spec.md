---
title: "Spec: Ruff Lint Debt Remediation"
tags:
  - doc/spec
  - topic/tooling
  - topic/ci
---

# Spec: Ruff Lint Debt Remediation

Goal: `ruff check agent/ tests/` and `ruff format --check agent/ tests/` both
exit 0, with the test suite result unchanged.

Non-goal: changing any runtime behaviour. Every step below is either provably
behaviour-neutral or is justified individually in the research note.

## Ordering principle

Semantic changes land in one commit; the whitespace-only reformat lands in a
second. `ruff format` touches 111 files, so mixing the two would bury the
reviewable diff. The manual fixes precede `--fix` because the `datetime.UTC`
shims otherwise get auto-fixed into `UTC = UTC` self-assignments.

## Steps

### 1. Capture the baseline

- 1.1: Record `ruff check agent/ tests/ --statistics` before any edit.
  Verification: 211 errors, distribution recorded in the research note.
- 1.2: Run `pytest tests/ -m "not live and not slow"` against a pristine
  checkout of the branch HEAD, in a separate worktree so an in-flight edit
  cannot contaminate the count.
  Verification: pass/fail/skip counts recorded, run reaches 100%.

### 2. Remove the obsolete `datetime.UTC` compat shims (54 files)

- 2.1: Rewrite the 22 single-line shims
  `from datetime import X, timezone; UTC = timezone.utc`
  to `from datetime import UTC, X`, dropping `timezone` from the import list.
  Verification: `ruff check --select E702` reports 0; `grep -rn "; UTC = timezone.utc"` is empty.
- 2.2: Confirm `timezone` is unused elsewhere in each of those 22 files before
  dropping it from the import.
  Verification: the only other occurrence repo-wide is
  `BlockingScheduler(timezone="UTC")` in `agent/awos/orchestrator/daemon.py`,
  a keyword argument, not the module.
- 2.3: After `--fix` runs UP017, delete the 32 standalone `UTC = UTC`
  self-assignments it produces, along with the now-false 3.10-compat comment.
  Verification: `grep -rn "UTC = UTC" agent/ tests/` is empty.

### 3. Fix the three findings that are not style

- 3.1: `agent/models/gnn/cde_encoder.py` — import `Any` from `typing` and
  unquote the `id_map` annotation.
  Verification: `ruff check --select F821` reports 0.
- 3.2: `agent/models/gnn/heterogeneous_cde_func.py` — drop the unused
  `import torch.nn.functional as F`; keep the local `F` drift matrix, whose
  name matches the module's documented notation.
  Verification: `ruff check --select F811` reports 0; no importer of this
  module references `F` (checked with grep across `agent/`, `tests/`, `scripts/`).
- 3.3: `agent/quant/microstructure.py` — the deprecated `typing.Dict/List/Tuple`
  imports (UP035) resolve once UP006/UP045 rewrite the usages and F401 drops
  the import. No separate edit needed.
  Verification: `ruff check --select UP035` reports 0 after step 4.

### 4. Apply `ruff check --fix` and audit the diff

- 4.1: Run `ruff check agent/ tests/ --fix` (safe fixes only, no `--unsafe-fixes`).
  Verification: 0 errors remain.
- 4.2: Confirm none of the four `CLAUDE.md` §9 critical files were touched.
  Verification: `git diff --name-only` matches none of `trainer.py`,
  `models/gnn/gnn.py`, `quant/scoring.py`, `pipeline/dag.py`.
- 4.3: Audit every non-import line in the diff by hand.
  Verification: only two semantic rewrites exist — a SIM114 branch merge and a
  SIM300 comparison flip — and both are argued equivalent below.
- 4.4: Confirm each file where UP037 unquoted an annotation carries
  `from __future__ import annotations`, so the unquoted forward reference is
  never evaluated.
  Verification: all 6 such files have the import.
- 4.5: Import every changed `agent/` module in a fresh interpreter.
  Verification: 64/64 import with no error — catches any `NameError` the shim
  removal could have introduced, faster than the full suite.

### 5. Format

- 5.1: Run `ruff format agent/ tests/` as its own commit.
  Verification: `ruff format --check agent/ tests/` exits 0, and
  `git diff --stat` for that commit shows whitespace-only changes.

### 6. Verify

- 6.1: Both CI lint commands exit 0.
- 6.2: Re-run the test suite and diff against the 1.2 baseline.
  Verification: identical pass/fail/skip counts, and the *same* set of test
  ids failing — per `LESSONS.md`, a newly-passing test is as suspicious as a
  newly-failing one and must be explained, not accepted.

## Equivalence arguments for the two semantic rewrites

**SIM114** in `agent/portfolio/constructor.py:_sample_covariance`:

```python
if cov.ndim == 0:      cov = np.array([[float(cov)]])
elif cov.shape == ():  cov = np.array([[float(cov)]])
```
becomes `if cov.ndim == 0 or cov.shape == ():`. For a numpy array
`ndim == 0` and `shape == ()` are the same predicate, so the `elif` was already
unreachable. Both arms were byte-identical. Behaviour is unchanged.

**SIM300** in `agent/quant/microstructure.py`: `if N < self.n_buckets:` becomes
`if self.n_buckets > N:`. Same comparison, operands transposed.

## Related

- [[ruff_lint_debt]]
- [[ruff_lint_debt_task]]
