---
title: "Spec: ai_workflow_rules"
tags:
  - doc/spec
  - topic/workflow
---

# Spec: ai_workflow_rules

## Goal
Add a thin, repo-specific AI operating rules layer that improves day-to-day use of chat and agent workflows without duplicating the repository's core architectural instructions.

## Files Affected
- `.github/copilot-instructions.md` — add a short operational collaboration section
- `RulesForAI.md` — create a concise, human-readable workflow checklist
- `[[ai_workflow_rules]]` — research record
- `[[ai_workflow_rules_spec]]` — implementation spec
- `[[ai_workflow_rules]]` — task tracker

## Implementation Steps
1. Create `[[ai_workflow_rules]]` documenting current workflow coverage and the remaining gaps.
2. Create `[[ai_workflow_rules_spec]]` defining the narrow scope of the new rules layer.
3. Create `[[ai_workflow_rules]]` with atomic implementation and validation steps.
4. Create `RulesForAI.md` as a short tactical checklist covering structure-first work, chat-vs-agent boundaries, one-problem-per-step decomposition, docs-first behavior for unfamiliar technology, debugging mode, tests, commits, and session resets.
5. Update `.github/copilot-instructions.md` with a concise operational collaboration section so the missing rules affect actual agent behavior.
6. Validate the changed markdown/instruction files and write a checkpoint summarizing the work.

## Edge Cases
- `RulesForAI.md` must not become a second architecture manifesto.
- The new instructions must not contradict existing rules about phased workflow, testing, or layer boundaries.
- The collaboration rules should guide behavior without forcing code changes during brainstorming-only requests.

## Testing Plan
- Validate the modified markdown/instruction files with editor diagnostics.
- Manually verify that the new rules do not duplicate existing sections verbatim and that `RulesForAI.md` clearly points back to `.github/copilot-instructions.md` as the primary instruction source.
- Write a checkpoint in `docs/memory/` to make the new workflow artifacts discoverable in later sessions.

---

## Related

- [[ai_workflow_rules|Research: Ai Workflow Rules]]
