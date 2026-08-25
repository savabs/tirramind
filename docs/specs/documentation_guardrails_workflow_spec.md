---
title: "Spec: Documentation Guardrails Workflow"
tags:
  - doc/spec
  - topic/workflow
---

# Spec: Documentation Guardrails Workflow

## Goal
Add explicit instruction guardrails so mathematical concepts and methods must be anchored to trusted documentation or primary sources before implementation.

## Files Affected
- `.github/copilot-instructions.md`
- `AGENTS.md`
- `/memories/workflow.md`
- `[[documentation_guardrails_workflow]]`

## Implementation Steps
1. Add a rule to `.github/copilot-instructions.md` requiring a trusted-source check before applying a mathematical concept or method.
2. Add wording that the agent must explicitly state which source it is relying on and why it is trustworthy.
3. Mirror the core rule in `AGENTS.md` so subagents inherit the same behavior.
4. Persist the preference in `/memories/workflow.md`.
5. Verify the new wording is present.

## Edge Cases
- The rule should not force citations for trivial arithmetic or already-settled local implementation details.
- The rule should activate for substantive mathematical concepts, statistical procedures, estimators, tests, filters, and optimization methods.
- Keep the wording concise enough to remain operational.

## Testing Plan
- Verify the new guardrail text exists in `.github/copilot-instructions.md`.
- Verify the matching agent-level rule exists in `AGENTS.md`.
- Verify `/memories/workflow.md` records the preference.

---

## Related

- [[documentation_guardrails_workflow|Research: Documentation Guardrails Workflow]]
