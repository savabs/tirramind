> Legacy reference. Use Cursor skill `/spec-to-task` — canonical: `.cursor/skills/spec-to-task/SKILL.md`

---
description: "Convert a completed spec into an active task file with atomic, verifiable steps."
---

# Spec To Task

You are converting a completed specification into an actionable task file.

## Input Required

The user will provide:
- Path to a completed spec file (e.g., `docs/specs/<feature>_spec.md`)

## Instructions

1. Read the spec file completely.
2. Read `tasks/TASK_TEMPLATE.md` for the standard format.
3. Verify the spec has: Goal, Files Affected, Implementation Steps, Edge Cases, Testing Plan.
4. If the spec is incomplete, **STOP** and list what's missing. Do not generate a task from an incomplete spec.

5. For each Implementation Step in the spec:
   - Break it into the smallest possible atomic sub-steps.
   - Each sub-step changes ONE thing and has ONE verification method.
   - If a step description contains "and", split it into two steps.
   - Name steps as: `<phase>.<step>: <verb> <specific thing>`
   - Add a `Verification:` line under each step.

6. Add a testing step after each implementation sub-phase (group of related steps).

7. Write the task file to `tasks/active/<feature_name>.md`.

## Output Format

```markdown
# Task: <feature_name>

Status: active
Research: docs/research/<feature_name>.md
Spec: docs/specs/<feature_name>_spec.md

## Goal
<one sentence from spec>

## Scope Notes
- Layer: <which layer(s)>
- Main files: <list>
- Non-goals: <what this does NOT include>

## Steps

- [ ] 1.1: <verb> <thing>
  Verification: <how to verify>
- [ ] 1.2: <verb> <thing>
  Verification: <how to verify>
...

## Completion Checklist
- [ ] All steps marked [x]
- [ ] All tests pass
- [ ] Spec fully implemented
- [ ] Checkpoint written

## Notes
```

## Rules
- Do NOT start implementation. This prompt produces a task file only.
- Do NOT invent steps beyond what the spec describes.
- Every step must be independently verifiable.
- Prefer 10 tiny steps over 5 medium steps.
