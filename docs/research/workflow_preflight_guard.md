---
title: "Feature: workflow_preflight_guard"
tags:
  - doc/research
  - topic/workflow
---

# Feature: workflow_preflight_guard

## Goal
- Add a mechanical repository-level guard that blocks implementation commits unless the governing task file and its linked research/spec artifacts exist.
- Provide a manual command developers can run before editing or committing to verify workflow preflight explicitly.

## Search Log
- GitHub keywords searched: none required for this internal workflow tooling change
- Documentation keywords searched: none required for this internal workflow tooling change
- Other search surfaces used: `pyproject.toml`, `agent/cli.py`, `tests/test_pipeline_config_cli.py`, `[[senior_eng_hardening]]`, `[[senior_eng_hardening_spec]]`, workflow instruction files

## External Repositories Reviewed
- Repository: none
	- Why it is relevant: not applicable
	- Useful implementation idea: not applicable
	- License: not applicable
	- Reuse conclusion: rejected

## Documentation Reviewed
- Source title / URL: `[[senior_eng_hardening]]`
	- What it clarified: commit-time guardrails were already identified as a missing mechanical enforcement layer
	- API or concept details to carry forward: keep commit hooks fast and narrow in scope
- Source title / URL: `[[senior_eng_hardening_spec]]`
	- What it clarified: `.pre-commit-config.yaml` is already an approved insertion point for guardrails
	- API or concept details to carry forward: avoid slow checks in pre-commit; put deeper validation elsewhere
- Source title / URL: `pyproject.toml`
	- What it clarified: `pre-commit` is already a dev dependency and a new script entry point can be added cleanly
	- API or concept details to carry forward: expose the guard as a console script for both manual and hook use
- Source title / URL: `agent/cli.py`
	- What it clarified: the main CLI has heavy transitive imports and is a poor host for a lightweight workflow guard
	- API or concept details to carry forward: implement the guard in a standalone lightweight module rather than extending the main agent CLI

## Current Architecture
- Relevant local modules:
	- workflow policy lives in `.github/copilot-instructions.md`, `AGENTS.md`, and `RulesForAI.md`
	- no runtime or commit-time workflow guard currently exists
	- `pyproject.toml` already declares `pre-commit` as a dev dependency
	- no `.pre-commit-config.yaml` currently exists
- Existing patterns to preserve:
	- task files link directly to research and spec files
	- documentation-first workflow for non-trivial work
	- atomic steps and checkpointing
- Correct insertion points:
	- new lightweight guard module under `agent/` so it can be tested without importing heavy quant dependencies
	- `.pre-commit-config.yaml` for commit-time enforcement
	- `pyproject.toml` for a console-script entry point

## Observations
- What already exists:
	- instruction-level fail-closed wording now exists
	- task files encode the research/spec linkage needed for enforcement
	- multiple task files can be active simultaneously
- What is missing:
	- no code checks that the three artifacts exist before implementation changes are committed
	- no explicit way to select which active task governs the current implementation change
- Important constraints:
	- because multiple active tasks are normal in this repo, the guard cannot infer the correct task reliably from `tasks/active/` alone
	- the hook must stay fast and avoid importing the full agent stack
	- the hook should allow workflow-only commits without requiring a task selector

## Risks
- Technical risks:
	- requiring an explicit task selector adds friction, but guessing the active task would be wrong often enough to be worse
	- a commit-time hook cannot literally stop a file from being edited; it can only block the change from being committed
	- if the guard parses task files too loosely, malformed metadata could slip through
- Testing risks:
	- staged-file detection is environment-dependent; pure validation logic should be separated from git subprocess calls
	- hooks must remain deterministic when run outside git or with no staged files

## Data Requirements
- Required inputs or sources:
	- changed file paths
	- selected task file path
	- task file metadata referencing research/spec paths
- What already exists locally:
	- task files with `Research:` and `Spec:` metadata
	- dev dependency on `pre-commit`
- What still needs to be added:
	- a lightweight workflow guard module
	- pre-commit config that invokes it
	- tests covering malformed task files, missing artifacts, workflow-only changes, and multiple-task scenarios

## Math/Algorithm Survey
- Candidate approaches:
	- extend `agent.cli.py` with a workflow subcommand
	- add a standalone lightweight module plus console script
	- use a git hook shell script only
- Why one approach is preferred:
	- a standalone module plus console script gives reusable validation logic, fast startup, and direct unit-test coverage; the pre-commit hook can call that command without importing heavy runtime dependencies
- Complexity or dependency notes:
	- low to moderate complexity; stdlib only

## Implementation Intent
- Concepts approved for implementation:
	- `agent.workflow_guard` module with pure validation helpers and a CLI
	- `tirra-workflow-check` console script entry point
	- `.pre-commit-config.yaml` local hook that runs the guard on staged files
	- explicit `--task` selector for non-workflow commits
	- workflow-only fast path that permits research/spec/task/checkpoint-only commits without a task selector
- Concepts rejected:
	- guessing the active task from multiple active tasks
	- extending the heavy main CLI for this lightweight guard
	- slow commit hooks that run the full test suite

---

## Related

- [[workflow_preflight_guard_spec|Spec: Workflow Preflight Guard]]
