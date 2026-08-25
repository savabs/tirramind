---
title: "Feature: workflow_templates"
tags:
  - doc/research
  - topic/workflow
---

# Feature: workflow_templates

Use this research note to add the remaining reusable workflow scaffolding: a brainstorm-to-spec prompt and a reusable task template.

## Goal
- Add repeatable workflow artifacts that help the user stay in planning mode until a problem is decomposed and ready for implementation.
- Standardize task-file creation so new work starts from the same atomic structure every time.

## Search Log
- GitHub keywords searched: none needed; this is an internal workflow artifact update.
- Documentation keywords searched: none needed beyond repository-local prompt and task conventions.
- Other search surfaces used: repository prompt files, task files, checkpoint files, and workflow docs.

## External Repositories Reviewed
- None. This change is repository-local workflow scaffolding.

## Documentation Reviewed
- Source title / URL:
  - `.github/prompts/next-step.prompt.md`
  - `.github/prompts/research.prompt.md`
  - `.github/prompts/full-pipeline.prompt.md`
  - `.github/prompts/sprint.prompt.md`
  - `RulesForAI.md`
  - `[[RESEARCH_TEMPLATE]]`
  - `[[checkpoint_archive_2026]]` (archived entries for the late-March and early-April workflow checkpoints)
  - What it clarified:
    - prompt files use short YAML frontmatter plus concrete ordered instructions
    - the repo already has research, implementation, and sprint prompts, but no dedicated brainstorming handoff prompt
    - task files are handoff artifacts but there is no reusable task-file template under `tasks/`

## Current Architecture
- Relevant local modules:
  - `.github/prompts/` for repeatable agent workflows
  - `tasks/active/` for active work tracking
  - `[[RESEARCH_TEMPLATE]]` for research note standardization
- Existing patterns to preserve:
  - concise YAML frontmatter in prompt files
  - ordered, imperative instructions
  - task files with `Status`, `Research`, `Spec`, and checkbox steps
- Correct insertion points:
  - add a new prompt under `.github/prompts/`
  - add a reusable task template at `tasks/TASK_TEMPLATE.md`

## Observations
- Existing prompts focus on research-only, next-step execution, sprint execution, and full pipeline automation.
- The workflow still lacks a clean planning prompt that converts a rough idea into an implementation-ready spec boundary without coding.
- Task files are structurally consistent, but the repo does not currently provide a blank template to start from.
- A lightweight task template should be generic enough for docs, tooling, and code features.

## Risks
- If the new prompt is too broad, it will overlap confusingly with `full-pipeline.prompt.md`.
- If the task template is too opinionated, people will ignore it and continue creating ad hoc files.
- The new prompt should stop before implementation; otherwise it undermines the chat-versus-agent boundary we just added.

## Data Requirements
- Required inputs or sources:
  - existing prompt conventions
  - existing task-file conventions
- What already exists locally:
  - all needed examples are already in the repository
- What still needs to be added:
  - brainstorm-to-spec prompt
  - reusable task template

## Math/Algorithm Survey
- Not applicable. This feature concerns workflow scaffolding rather than mathematical or algorithmic logic.

## Implementation Intent
- Concepts approved for implementation:
  - prompt-driven planning handoff
  - reusable task-file template
- Concepts rejected:
  - another full workflow manifesto
  - a prompt that automatically writes code during brainstorming
- Notes for the spec:
  - keep both artifacts lightweight and aligned with existing repo style

---

## Related

- [[workflow_templates_spec|Spec: Workflow Templates]]
