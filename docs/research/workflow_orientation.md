---
title: "Feature: workflow_orientation"
tags:
  - doc/research
  - topic/workflow
---

# Feature: workflow_orientation

## Current Architecture
- Orchestrator in `agent/core/orchestrator.py` runs: Goal → Plan → Execute → Synthesize
- No explicit research phase before planning
- No task tracking files produced during runs
- No `docs/` directory structure for workflow artifacts
- `.github/copilot-instructions.md` did not exist

## Observations
- The `copilot_agent_workflow.md` defines a Research → Spec → Implement pipeline
- The orchestrator's planner goes straight to task decomposition without research
- No file-based task tracking — everything lives in memory
- The workflow's directory structure (`docs/research/`, `docs/specs/`, `docs/memory/`, `tasks/active/`) was completely absent

## Risks
- Adding a research phase adds one extra LLM call per run (slight latency increase)
- Task file generation writes to filesystem — needs `tasks/active/` to exist
- Slugification of goals could collide if two goals produce the same slug (acceptable for now)

---

## Related

- [[workflow_orientation_spec|Spec: Workflow Orientation]]
