---
title: "Feature: Math Explanation Workflow"
tags:
  - doc/research
  - topic/workflow
---

# Feature: Math Explanation Workflow

## Current Architecture
- Workspace-level agent behavior is defined in `.github/copilot-instructions.md`.
- Agent-wide discovery and behavior references live in `AGENTS.md`.
- Persistent user preferences are stored in `/memories/workflow.md`.

## Observations
- Existing instructions strongly enforce research-first, spec-first, and math-before-LLM.
- The current instructions do not explicitly require mathematical explanation once work shifts from data collection to modeling, scoring, inference, or statistical control.
- The current instructions do not explicitly require comparing alternative mathematical implementations before choosing one.
- The current instructions do not explicitly prefer a minimal high-signal toolset when adding more tools would make the model harder to reason about or operationally heavier.

## Risks
- Without an explicit rule, implementation can drift into "just code it" mode for mathematical modules, leaving the user unable to evaluate assumptions, null hypotheses, complexity, or numerical tradeoffs.
- Overusing too many tools or methods can degrade interpretability, increase maintenance cost, and weaken causal attribution.
- If only one instruction file is changed, subagents may not consistently inherit the intent unless the shared agent-definition file also reflects it.

## Data Requirements
- No new runtime data needed.
- Only instruction content and persistent preference memory need updates.

## Math/Algorithm Survey
- For math-heavy work, good collaboration requires explicit treatment of:
  - objective function / test statistic
  - null and alternative hypotheses
  - estimator choice and assumptions
  - numerical stability and computational complexity
  - candidate implementation options and why one is preferred
- Tool-selection discipline matters when the marginal value of adding another source or method is lower than the extra complexity it introduces.
- The instruction should prefer the smallest set of high-signal tools that preserves edge and interpretability.

---

## Related

- [[math_explanation_workflow_spec|Spec: Math Explanation Workflow]]
