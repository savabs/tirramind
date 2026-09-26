---
title: "Chat Checkpoint — 2026-09-06 — Ruff Lint Debt"
tags:
  - doc/memory
  - topic/tooling
  - topic/ci
---

# Chat Checkpoint — 2026-09-06 — Ruff Lint Debt

## What this session did

Made the CI `lint` job green. Both commands now exit 0:

```
ruff check agent/ tests/
ruff format --check agent/ tests/
```

Baseline was 211 check errors and 111 files ruff format would rewrite, both
pre-existing on `main`.

## What was learned

**The debt was a version split, not accumulated sloppiness.**
`.pre-commit-config.yaml` pinned ruff `v0.4.8`; CI does `pip install ruff` and
gets the latest. The `pyproject.toml` ignore list was tuned against 0.4.8, so CI
went red while local pre-commit stayed green and nobody saw it. The two versions
actively disagree — 0.4.8 enforces UP038, which newer ruff removed, and its
formatter produces different output. Any fix for one was a regression for the
other until the pin was bumped.

**CI is still not pinned.** It installs whatever is latest at run time — 0.16.6
today. The cleanup was verified against 0.16.6 specifically, not just the local
0.16.4, but a future release adding rules can turn the job red again with no
code change. Pinning CI to a version is the durable fix and is left as a
decision for the owner.

**Three findings were real, not cosmetic:**

- `agent/models/gnn/cde_encoder.py` annotated `id_map: "Any"` without importing
  `Any`. Silent only because `from __future__ import annotations` stops the
  annotation being evaluated.
- `agent/models/gnn/heterogeneous_cde_func.py` imported
  `torch.nn.functional as F` and never used it — the `F` in that module is the
  CDE drift matrix, a local tensor.
- 54 files carried a `UTC = timezone.utc` shim for Python 3.10, which
  `requires-python = ">=3.11"` has made dead. Left alone it also breaks the
  fixer: UP017 turns it into the self-assignment `UTC = UTC`.

**Two hooks in the repo's own pre-commit config are not safe on this codebase:**

- `trailing-whitespace` strips the trailing double-spaces in
  `agent/quant/ghost_brief.py`, which are Markdown hard line breaks in the brief
  template. Collapsing them reflows the rendered brief into one paragraph.
  `pyproject.toml` already carves this out for ruff via W291/W293, but the hook
  is a separate tool and never read that config. Now excluded.
- `pre-commit run --all-files` is repo-wide — it touches `scripts/`, notebooks
  and `products/`, not just the `agent/`+`tests/` scope CI lints. It rewrote 421
  things across the repo in one invocation here. Use the commit-time hook, which
  only sees staged files.

## Method notes worth reusing

- The `.venv` lives in the main checkout, so pre-commit hooks (`entry:
  .venv/bin/python3 ...`) fail in a worktree until you symlink it in. `.venv/`
  is gitignored, so the symlink is invisible to git.
- Formatting was proven behaviour-neutral rather than asserted: parse each file
  before and after, normalise per-line trailing whitespace inside string
  constants, compare `ast.dump`. 90 of 92 matched outright; the other two were
  docstring re-indentation, traced to a source line indented 2 spaces instead of
  4 and fixed at the source.
- Test baselines must come from a separate pristine worktree. Running the
  baseline in the working directory races your own edits — pytest imports test
  modules at collection, so an edit landing mid-collection silently contaminates
  the comparison.

## Known issues / next steps

- CI's unpinned `pip install ruff` is the remaining source of drift.
- `.venv/bin/pip` and `.venv/bin/pre-commit` carry a stale shebang pointing at
  `/Users/becmachlean/BACKUP_FROM_LINUX/tirramind_v1/.venv/`. Work around with
  `python -m pip` / `python -m pre_commit`; worth recreating the venv.
- 5 test failures pre-date this work and still fail. Two are `TestLiveNetwork`
  cases in `test_power_grid_edge.py`, which are network-dependent.

## How to resume

Everything is committed on `claude/priceless-carson-478711`:

```
6916f23 fix: stop the whitespace hook eating Markdown hard breaks in ghost_brief
1d437da style: ruff format agent/ and tests/
f2eafee fix: clear all 211 ruff check errors (no behaviour change)
ae6be93 chore: align pre-commit ruff with the version CI enforces
29ca607 docs: research, spec and task for the ruff lint debt in CI
```

Governing artifacts: `tasks/active/ruff_lint_debt_task.md`,
`docs/specs/ruff_lint_debt_spec.md`, `docs/research/ruff_lint_debt.md`.

## Related

- [[ruff_lint_debt]]
- [[ruff_lint_debt_spec]]
- [[ruff_lint_debt_task]]
