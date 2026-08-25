---
title: "Feature: ai_workflow_rules"
tags:
  - doc/research
  - topic/workflow
---

# Feature: ai_workflow_rules

## Goal
- Add a thin, day-to-day operating rules layer that improves how chat and coding-agent workflows are used in the repository.
- Solve the gap between high-level architectural rules and the practical tactical behaviors needed during active sessions.

## Search Log
- GitHub keywords searched: none required for this internal workflow note
- Documentation keywords searched: none required for this internal workflow note
- Other search surfaces used: repository instruction files and internal workflow guidance

## External Repositories Reviewed
- Repository: none
	- Why it is relevant: not applicable
	- Useful implementation idea: not applicable
	- License: not applicable
	- Reuse conclusion: rejected

## Documentation Reviewed
- Source title / URL: `.github/copilot-instructions.md`
	- What it clarified: the repository already had strong phased workflow and architecture constraints
	- API or concept details to carry forward: use the main instruction file as the primary source of truth
- Source title / URL: `AGENTS.md`
	- What it clarified: the default obligations for all agents and the need to stay aligned with repo-wide rules
	- API or concept details to carry forward: any new tactical guide must complement agent defaults rather than override them
- Source title / URL: `docs/copilot_pro_optimization_guide.md`
	- What it clarified: session-level operating habits and context-efficiency advice worth reflecting in a thinner tactical guide
	- API or concept details to carry forward: practical workflow suggestions belong in a concise checklist, not a second large manifesto

## Current Architecture
- Relevant local modules:
	- `.github/copilot-instructions.md` is the primary repository-level instruction source for agent behavior, workflow phases, and architecture boundaries.
	- `AGENTS.md` defines agent defaults, tool permissions, and the requirement to follow the phased workflow.
	- `docs/copilot_pro_optimization_guide.md` contains operator guidance on session management, prompting, and context efficiency, but it is a human guide rather than an active instruction file.
	- There is no dedicated `RulesForAI.md` or other thin session-level rules artifact for day-to-day chat versus agent usage.
- Existing patterns to preserve:
	- research -> spec -> implement
	- atomic decomposition
	- one source of truth for durable repository rules
- Correct insertion points:
	- `.github/copilot-instructions.md` for enforceable workflow behavior
	- `RulesForAI.md` for a lightweight tactical checklist
	- `AGENTS.md` for agent-default alignment

## Observations
- What already exists:
	- The repository already enforces research -> spec -> implement and atomic decomposition well.
- What is missing:
	- The main missing behaviors are operational rather than architectural: explicit chat-vs-agent boundaries, docs-first behavior for unfamiliar technology, and a standard debug-first fallback when the model loops.
	- There is no thin tactical guide that users can treat as a quick operating checklist.
- Important constraints:
	- Creating a second large instruction document would duplicate `.github/copilot-instructions.md` and increase the risk of drift.
	- A short `RulesForAI.md` can act as a human-readable operating checklist if it stays tactical and points back to the primary instruction file.

## Risks
- Licensing or reuse risks:
	- None specific beyond normal documentation-drift risk because no external code reuse is involved.
- Technical risks:
	- Duplicating permanent rules across multiple files can create contradictory guidance over time.
	- A `RulesForAI.md` file may be mistaken for the primary instruction source unless it explicitly states its scope.
- Testing risks:
	- Overly broad rules could reduce agent autonomy by forcing unnecessary pauses before bounded implementation steps.

## Data Requirements
- Required inputs or sources:
	- existing workflow files and guidance documents
- What already exists locally:
	- `.github/copilot-instructions.md`, `AGENTS.md`, and `docs/copilot_pro_optimization_guide.md`
- What still needs to be added:
	- a thin tactical rules file and aligned workflow wording in the main instruction stack

## Math/Algorithm Survey
- Candidate approaches:
	- add only to the main instruction file
	- create a thin tactical guide and reinforce the main file
- Why one approach is preferred:
	- a thin guide plus main-file reinforcement gives visibility without splitting authority across multiple architecture manifestos
- Complexity or dependency notes:
	- low complexity; no code or dependency changes required

## Implementation Intent
- Concepts approved for implementation:
	- a concise tactical rules file
	- reinforcement of missing operational rules in the main instruction stack
	- clear precedence language so `.github/copilot-instructions.md` remains primary
- Concepts rejected:
	- creating a second large architecture or workflow manifesto
- Notes for the spec:
	- keep the tactical rules short, operational, and explicitly subordinate to the main instruction file

---

## Related

- [[ai_workflow_rules_spec|Spec: Ai Workflow Rules]]
