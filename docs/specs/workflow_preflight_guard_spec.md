---
title: "Spec: workflow_preflight_guard"
tags:
  - doc/spec
  - topic/workflow
---

# Spec: workflow_preflight_guard

## Goal
Create a mechanical workflow-preflight guard that blocks non-workflow commits unless a selected task file and its linked research/spec artifacts exist.

## Files Affected
- `agent/workflow_guard.py`
- `.pre-commit-config.yaml`
- `pyproject.toml`
- `tests/test_workflow_guard.py`
- `[[workflow_preflight_guard]]`
- `[[workflow_preflight_guard_spec]]`
- `[[workflow_preflight_guard]]`
- `[[checkpoint_archive_2026]]`

## Implementation Steps
1. Create `[[workflow_preflight_guard]]` documenting the enforcement options and why the guard should be a lightweight standalone module.
2. Create `[[workflow_preflight_guard_spec]]` defining the exact validation behavior and explicit task-selection rule.
3. Create `[[workflow_preflight_guard]]` with atomic implementation and validation steps.
4. Implement `agent/workflow_guard.py` with:
   - workflow-file classification
   - task-file parsing for `Research:` and `Spec:` metadata
   - validation that the selected task file exists and references existing research/spec files
   - staged-file collection via git for hook use
   - CLI flags for `--staged`, `--task`, and explicit file paths
5. Add a console-script entry point in `pyproject.toml` for manual use.
6. Add `.pre-commit-config.yaml` with a fast local hook that runs the guard on staged files.
7. Write `tests/test_workflow_guard.py` covering edge cases:
   - workflow-only changes pass without a task
   - non-workflow changes fail without `--task`
   - missing task file fails
   - malformed task metadata fails
   - missing linked research/spec files fail
   - valid task plus implementation file passes
8. Validate the new tests and write a checkpoint.

## Edge Cases
- Commits that only touch workflow artifacts should pass without a selected task.
- Mixed commits containing both workflow and implementation files should still require a selected task.
- The task file may exist but contain missing or malformed `Research:` or `Spec:` lines.
- The guard should return a useful error when run outside a git repo or with no staged files.
- The hook should not import heavy optional dependencies.

## Testing Plan
- Run targeted unit tests for `tests/test_workflow_guard.py`.
- Run `get_errors` on modified files.
- Manually inspect `.pre-commit-config.yaml` for valid structure.
- If feasible, run the guard command against a small synthetic file set via unit tests rather than relying on git state.

---

## Related

- [[workflow_preflight_guard|Research: Workflow Preflight Guard]]
