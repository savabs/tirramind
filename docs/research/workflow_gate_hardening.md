---
title: "Feature: workflow_gate_hardening"
tags:
  - doc/research
  - topic/workflow
---

# Feature: workflow_gate_hardening

## Goal
- Convert the repository's research -> spec -> implement preference into an explicit fail-closed gate for non-trivial work.
- Reduce cases where the agent jumps directly to coding because the immediate user request appears concrete.

## Search Log
- GitHub keywords searched: none required for this internal workflow hardening change
- Documentation keywords searched: none required for this internal workflow hardening change
- Other search surfaces used: `.github/copilot-instructions.md`, `AGENTS.md`, `RulesForAI.md`, `[[ai_workflow_rules]]`

## External Repositories Reviewed
- Repository: none
	- Why it is relevant: not applicable
	- Useful implementation idea: not applicable
	- License: not applicable
	- Reuse conclusion: rejected

## Documentation Reviewed
- Source title / URL: `.github/copilot-instructions.md`
	- What it clarified: the repository already defines phased workflow, but it does not enforce a preflight gate before edits
	- API or concept details to carry forward: hardening should live in the primary instruction file because that is the highest-authority repo workflow source
- Source title / URL: `AGENTS.md`
	- What it clarified: all agents are already told not to skip the phased workflow for non-trivial changes
	- API or concept details to carry forward: agent defaults should mirror the hard gate in shorter language
- Source title / URL: `RulesForAI.md`
	- What it clarified: the thin rules file currently says to start with structure, but it reads as advice rather than a mandatory gate
	- API or concept details to carry forward: add an explicit preflight checklist and allowed-pre-gate file scope

## Current Architecture
- Relevant local modules:
	- `.github/copilot-instructions.md` is the primary workflow authority.
	- `AGENTS.md` is the concise agent-default layer.
	- `RulesForAI.md` is the tactical day-to-day checklist.
	- `[[ai_workflow_rules]]` documents the earlier workflow-instructions work and confirms the prior gap was operational rather than architectural.
- Existing patterns to preserve:
	- research -> spec -> implement
	- atomic decomposition
	- task file as source of truth
- Correct insertion points:
	- `.github/copilot-instructions.md` for the hard gate
	- `AGENTS.md` for concise agent-default reinforcement
	- `RulesForAI.md` for the checklist-level preflight wording

## Observations
- What already exists:
	- the repository strongly prefers research and spec work before implementation
	- the workflow already distinguishes planning mode from implementation mode
- What is missing:
	- there is no mandatory preflight checklist requiring a research note, spec, and active task file before non-trivial code edits
	- there is no explicit restriction on what files may be edited before that preflight passes
	- there is no deterministic first-response shape requiring the agent to declare research, spec, or implementation mode
- Important constraints:
	- the hardening should still allow obviously trivial one-file, no-behavior-change requests to proceed without ceremony
	- the language should be concrete enough to survive prompt arbitration against default "just implement" behavior

## Risks
- Licensing or reuse risks:
	- none; this is internal workflow text only
- Technical risks:
	- if the gate is too broad, it will create friction for typo fixes and similarly trivial edits
	- if the gate is too vague, it will continue to lose against stronger act-now instructions
- Testing risks:
	- documentation-only changes cannot prove runtime enforcement; validation is limited to wording clarity, internal consistency, and absence of conflicting instructions

## Data Requirements
- Required inputs or sources:
	- current workflow instruction files
	- prior workflow-gap research and task notes
- What already exists locally:
	- `.github/copilot-instructions.md`, `AGENTS.md`, `RulesForAI.md`, `[[ai_workflow_rules]]`, `[[ai_workflow_rules_spec]]`
- What still needs to be added:
	- a dedicated hard-gate workflow note, spec, task file, and checkpoint

## Math/Algorithm Survey
- Candidate approaches:
	- strengthen wording only in `.github/copilot-instructions.md`
	- reinforce the same gate in all workflow instruction layers
	- add a first-response preflight checklist and a restricted pre-gate file scope
- Why one approach is preferred:
	- a single-file change in the primary instructions is necessary but not sufficient; matching concise wording in `AGENTS.md` and `RulesForAI.md` improves salience during instruction arbitration
- Complexity or dependency notes:
	- low complexity; no code or dependency changes required

## Implementation Intent
- Concepts approved for implementation:
	- mandatory preflight for non-trivial work
	- explicit definition of trivial exceptions
	- explicit limitation that pre-gate edits may only touch research/spec/task/checkpoint artifacts
	- requirement that implementation-mode work cite the task/spec it is following
- Concepts rejected:
	- broad changes to runtime agent code or external tooling for enforcement
	- duplicate manifesto-style instructions across many files

---

## Related

- [[workflow_gate_hardening_spec|Spec: Workflow Gate Hardening]]
