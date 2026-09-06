---
title: "Research: Ruff Lint Debt in CI"
tags:
  - doc/research
  - topic/tooling
  - topic/ci
---

# Research: Ruff Lint Debt in CI

## Problem

The CI `lint` job (`.github/workflows/ci.yml`) runs two commands, and both fail on `main`:

```
ruff check agent/ tests/          # 211 errors (162 auto-fixable)
ruff format --check agent/ tests/ # 111 files would be reformatted
```

This is pre-existing debt, unrelated to any recent change. It surfaced while
investigating why CI failed on PR #1 — that failure turned out to be a GitHub
account billing lock, not a code problem, but it exposed the lint job as
already-red underneath.

## Root cause

The repo pins two different ruff versions:

| Where | Version | Behaviour |
| --- | --- | --- |
| `.pre-commit-config.yaml` | `v0.4.8` | what contributors run locally |
| `.github/workflows/ci.yml` | `pip install ruff` → latest | what CI enforces |

The `[tool.ruff.lint].ignore` list in `pyproject.toml` was tuned against 0.4.8.
Newer ruff (0.16.x) fires rules that 0.4.8 did not, so CI drifted red while
pre-commit stayed green. Nobody introduced the errors; the enforcement moved.

## Violation distribution (baseline, ruff 0.16.4)

```
55  I001    unsorted-imports
47  F401    unused-import
34  UP017   datetime-timezone-utc
24  UP037   quoted-annotation
22  E702    multiple-statements-on-one-line-semicolon
10  UP045   non-pep604-annotation-optional
 7  UP006   non-pep585-annotation
 6  SIM300  yoda-conditions
 3  UP035   deprecated-import
 1  F811    redefined-while-unused
 1  F821    undefined-name
 1  SIM114  if-with-same-arms
```

## Findings that are not merely cosmetic

### 1. The `datetime.UTC` compat shim is obsolete and blocks the fixer

54 files carry a Python 3.10 compatibility shim in one of two shapes:

```python
from datetime import datetime, timezone; UTC = timezone.utc   # 22 files — the E702 count
# datetime.UTC is Python 3.11+; Kaggle / many dev boxes are still on 3.10.
UTC = timezone.utc                                            # 32 files, on its own line
```

`datetime.UTC` exists from Python 3.11. The repo declares
`requires-python = ">=3.11"`, ruff targets `py311`, and CI tests only 3.11 and
3.12. The comment's premise no longer holds, so the shim is dead weight.

It also actively interferes with `--fix`: UP017 rewrites `timezone.utc` to
`UTC` and adds `from datetime import UTC`, which turns the standalone shim line
into the self-assignment `UTC = UTC`. The shims must be removed rather than
auto-fixed around.

### 2. `F821` is a real latent defect

`agent/models/gnn/cde_encoder.py:165` annotates `id_map: "Any"` without
importing `Any`. It never raises, because the file has
`from __future__ import annotations` and the annotation is never evaluated —
but the name is genuinely undefined, and any runtime introspection
(`typing.get_type_hints`) would fail.

### 3. `F811` shadows a module import with a local tensor

`agent/models/gnn/heterogeneous_cde_func.py` imports
`torch.nn.functional as F`, then binds a local `F` to the CDE drift matrix
inside `forward()`. The module import is never used anywhere in the file —
`F.norm(...)` on line 166 is a method call on the local tensor, not the
functional namespace. The `F` name for the drift matrix is deliberate and
documented throughout the module's docstrings, so the import is the thing to
drop, not the local.

## Auto-fix safety assessment

The stated risk of `--fix` is unused-import removal breaking re-export
patterns, since `agent/` relies on package imports. Checked: **no `__init__.py`
appears among the 46 F401 findings.** All 46 are either stdlib imports in
module bodies or over-broad "import the whole public surface" blocks at the top
of test files. No re-export is at risk.

Two F401 hits are indented and warranted a closer look, in case they were
availability guards rather than dead code:

- `agent/convergence/neural_hawkes.py:87` — top-level `import numpy as np`;
  `np.` appears zero times in the file.
- `agent/models/gnn/signature_path.py:225` — a redundant `import numpy as np`
  inside a loop body that uses only `_iisig.logsig` and `torch.tensor`.

Both are dead. Neither guards an optional dependency.

## Constraints

- `CLAUDE.md` §9 names four critical files to read before editing:
  `trainer.py`, `agent/models/gnn/gnn.py`, `agent/quant/scoring.py`,
  `agent/pipeline/dag.py`.
- `LESSONS.md` records two occasions where a test asserted a bug. A green suite
  is not proof here: the test result must be compared before and after, and a
  test that starts *passing* is as suspicious as one that starts failing.
- The pre-commit workflow guard (`agent/workflow_guard.py`) rejects any commit
  touching non-workflow files without a governing `tasks/active/<name>.md` that
  cites existing `Research:` and `Spec:` files.

## Related

- [[ruff_lint_debt_spec]]
- [[ruff_lint_debt_task]]
