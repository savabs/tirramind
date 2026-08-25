---
title: "Spec: Step Resource Guardrails"
tags:
  - doc/spec
---

# Spec: Step Resource Guardrails

## Goal
Require a step-level resource discovery pass before implementation starts on any meaningful step, and require the findings to be written down and used during coding.

## Files Affected
- `.github/copilot-instructions.md`
- `AGENTS.md`
- `/memories/workflow.md`
- `[[step_resource_guardrails]]`

## Implementation Steps
1. Add wording to `.github/copilot-instructions.md` requiring a resource scan before each implementation step.
2. Require that the scan include the direct topic, subtopics, and adjacent concepts that could affect the step.
3. Require that those references be recorded in the research or spec artifacts before coding.
4. Mirror the rule in `AGENTS.md` so subagents follow it too.
5. Persist the preference in `/memories/workflow.md`.
6. Verify the wording is present.

## Edge Cases
- Do not require a fresh external search for purely mechanical or trivial edits.
- The rule should trigger for meaningful implementation steps, especially anything mathematical, architectural, or unfamiliar.
- Keep the wording precise and operational rather than vague.

## Testing Plan
- Verify the new step-resource rule exists in `.github/copilot-instructions.md`.
- Verify the matching subagent rule exists in `AGENTS.md`.
- Verify `/memories/workflow.md` contains the persistent preference.

---

## Related

- [[step_resource_guardrails|Research: Step Resource Guardrails]]
