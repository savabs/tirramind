---
title: "Spec: workflow_orientation"
tags:
  - doc/spec
  - topic/workflow
---

# Spec: workflow_orientation

## Goal
Orient the TirraMind codebase to follow the phased workflow defined in `copilot_agent_workflow.md`.

## Files Affected
- `agent/core/orchestrator.py` — add research phase, task file generation
- `.github/copilot-instructions.md` — create with workflow rules
- `[[project_memory]]` — create with architecture knowledge
- `docs/research/.gitkeep` — create directory
- `docs/specs/.gitkeep` — create directory
- `tasks/active/.gitkeep` — create directory

## Implementation Steps
1. Create directory scaffold: `docs/research/`, `docs/specs/`, `docs/memory/`, `tasks/active/`
2. Create `.github/copilot-instructions.md` encoding the workflow rules
3. Create `[[project_memory]]` with current architecture documentation
4. Add `_research()` method to Orchestrator — LLM analysis before planning
5. Update `run()` to call research → plan → execute → synthesize (4 phases)
6. Add `_write_task_file()` and `_slugify()` to Orchestrator for task tracking
7. Dog-food the system: create research doc, spec doc, and task file for this work

## Edge Cases
- Goal slugs could collide (two different goals → same slug). Acceptable for v1.
- Research phase adds latency (one extra LLM call). Acceptable tradeoff for plan quality.
- Task directory may not exist if agent runs from unexpected CWD. Handled with `mkdir(parents=True)`.

## Testing Plan
- Verify Python imports pass: `python -c "from agent.core.orchestrator import Orchestrator"`
- Verify directory structure exists
- Verify copilot instructions load correctly in VS Code

---

## Related

- [[workflow_orientation|Research: Workflow Orientation]]
