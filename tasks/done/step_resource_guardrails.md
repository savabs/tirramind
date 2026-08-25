---
title: "Task: Step Resource Guardrails"
tags:
  - doc/task
  - status/done
---

# Task: Step Resource Guardrails

Status: completed
Research: [[step_resource_guardrails]]
Spec: [[step_resource_guardrails_spec]]

## Goal
Make every meaningful implementation step start with a written resource pass for that exact step and its nearby concepts.

- [x] 1.1: Update `.github/copilot-instructions.md` with per-step resource discovery rules
- [x] 1.2: Update `AGENTS.md` so subagents inherit the per-step resource rule
- [x] 1.3: Persist the preference in `/memories/workflow.md`
- [x] 1.4: Verify the updated wording is present

## Notes
- The rule should force resource gathering at step granularity, not only at feature granularity.
- The record should live in the research/spec workflow artifacts so later coding can reference it directly.

---

## Related

- [[step_resource_guardrails|Research: Step Resource Guardrails]]
- [[step_resource_guardrails_spec|Spec: Step Resource Guardrails]]
