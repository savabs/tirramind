---
title: "Spec: workflow_templates"
tags:
  - doc/spec
  - topic/workflow
---

# Spec: workflow_templates

## Goal
Add the remaining reusable workflow scaffolding: a prompt that turns brainstorming into a spec-ready plan without coding, and a reusable task template that matches existing task-file conventions.

## Files Affected
- `.github/prompts/brainstorm-to-spec.prompt.md` — new planning-only prompt template
- `tasks/TASK_TEMPLATE.md` — reusable active-task template
- `[[workflow_templates]]` — research record
- `[[workflow_templates_spec]]` — feature spec
- `[[workflow_templates]]` — task tracker

## Implementation Steps
1. Create `[[workflow_templates]]` documenting the missing workflow artifacts and their insertion points.
2. Create `[[workflow_templates_spec]]` defining the narrow scope of the prompt and task template additions.
3. Create `[[workflow_templates]]` with atomic implementation and validation steps.
4. Add `.github/prompts/brainstorm-to-spec.prompt.md` that explicitly stays in planning mode, decomposes the problem, identifies missing docs, drafts a research-and-spec-ready step list, and stops before implementation.
5. Add `tasks/TASK_TEMPLATE.md` with the standard task header, reference fields, atomic checkbox steps, verification placeholders, and status workflow.
6. Validate the new files and write a checkpoint documenting how to use them.

## Edge Cases
- The brainstorming prompt must not generate code or silently transition into implementation.
- The prompt should complement rather than duplicate `full-pipeline.prompt.md`.
- The task template should remain generic enough for code, research, and workflow features.

## Testing Plan
- Validate the new markdown files and prompt file with editor diagnostics.
- Manually verify that the new prompt ends in planning/spec output rather than code generation.
- Manually verify that the task template matches the repository's existing task-file structure and supports atomic steps.

---

## Related

- [[workflow_templates|Research: Workflow Templates]]
