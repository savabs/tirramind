---
title: "Task: Documentation Guardrails Workflow"
tags:
  - doc/task
  - status/done
  - topic/workflow
---

# Task: Documentation Guardrails Workflow

Status: completed
Research: [[documentation_guardrails_workflow]]
Spec: [[documentation_guardrails_workflow_spec]]

## Goal
Require trusted documentation or primary-source grounding before applying mathematical concepts in implementation work.

- [x] 1.1: Update `.github/copilot-instructions.md` with trusted-source guardrails for mathematical concepts
- [x] 1.2: Update `AGENTS.md` so subagents inherit the trusted-source requirement
- [x] 1.3: Persist the preference in `/memories/workflow.md`
- [x] 1.4: Verify the updated wording is present

## Notes
- The rule should force explicit source grounding, not vague references to “standard methods”.
- The rule should complement the existing math-explanation and option-comparison instructions.

---

## Related

- [[documentation_guardrails_workflow|Research: Documentation Guardrails Workflow]]
- [[documentation_guardrails_workflow_spec|Spec: Documentation Guardrails Workflow]]
