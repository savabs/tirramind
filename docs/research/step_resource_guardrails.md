---
title: "Feature: Step Resource Guardrails"
tags:
  - doc/research
---

# Feature: Step Resource Guardrails

## Current Architecture
- Workspace-level behavior is controlled by `.github/copilot-instructions.md`.
- Shared agent behavior is controlled by `AGENTS.md`.
- Persistent user collaboration preferences are stored in `/memories/workflow.md`.

## Observations
- The repo already requires research-first and trusted-source grounding for new concepts.
- The remaining gap is step granularity: the agent is not yet forced to collect and record the relevant references for each implementation step before writing code.
- The user wants step-local resource discovery to include the exact topic, nearby subtopics, and any adjacent concepts that might become relevant during implementation.

## Risks
- Without a per-step guardrail, references can stay too high-level and fail to cover the precise concept being implemented in the current step.
- Mathematical or systems code can drift into improvisation when the exact step-level sources are not written down before coding.
- Future sessions and subagents lose continuity if step-relevant references are not recorded in a stable location.

## Data Requirements
- No runtime data changes.
- Only instruction files and user memory need updates.

## Math/Algorithm Survey
- The correct workflow should require a resource pass before each implementation step.
- That pass should cover:
  - the exact method or component being implemented
  - immediate subtopics and neighboring concepts that may affect design
  - authoritative docs, papers, standard references, and implementation docs
  - a written record in the research/spec workflow artifacts before code begins

---

## Related

- [[step_resource_guardrails_spec|Spec: Step Resource Guardrails]]
