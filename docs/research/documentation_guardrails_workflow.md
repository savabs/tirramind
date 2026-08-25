---
title: "Feature: Documentation Guardrails Workflow"
tags:
  - doc/research
  - topic/workflow
---

# Feature: Documentation Guardrails Workflow

## Current Architecture
- Workspace-level behavior is controlled by `.github/copilot-instructions.md`.
- Shared subagent behavior is controlled by `AGENTS.md`.
- Persistent collaboration preferences are stored in `/memories/workflow.md`.

## Observations
- Current rules require research and authoritative documentation before new concepts are coded.
- Current rules require mathematical explanation and option comparison once work becomes quantitative.
- The remaining gap is a hard guardrail requiring the agent to anchor the chosen mathematical concept to a trusted source before implementation, instead of relying on memory or intuition alone.

## Risks
- Without an explicit documentation guardrail, the agent can still apply a mathematically plausible concept without proving it is grounded in a trusted paper, standard reference, or authoritative library documentation.
- Mathematical code becomes harder to audit if the source of the concept, its assumptions, and its implementation rationale are not made explicit.
- Subagents may behave inconsistently unless the same guardrail is reflected in `AGENTS.md`.

## Data Requirements
- No runtime data changes.
- Only instruction files and user memory need updates.

## Math/Algorithm Survey
- The correct mathematical workflow should require:
  - identification of the mathematical objective or test statistic
  - identification of at least one trusted source for the concept
  - a statement of why that source is trusted in this context
  - comparison with the main implementation alternatives
  - justification when a smaller, higher-signal method/tool set is preferred over a broader one

---

## Related

- [[documentation_guardrails_workflow_spec|Spec: Documentation Guardrails Workflow]]
