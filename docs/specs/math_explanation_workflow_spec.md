---
title: "Spec: Math Explanation Workflow"
tags:
  - doc/spec
  - topic/workflow
---

# Spec: Math Explanation Workflow

## Goal
Update workspace instructions so that once work enters the mathematical phase, the agent must explain the mathematics, compare viable implementation options, and justify tool-selection tradeoffs with precision.

## Files Affected
- `.github/copilot-instructions.md`
- `AGENTS.md`
- `/memories/workflow.md`

## Implementation Steps
1. Add a math-collaboration rule to `.github/copilot-instructions.md` requiring explicit explanation of the mathematics for scoring, inference, filtering, optimization, and statistical tests.
2. Add instruction text requiring alternative implementation options and tradeoff discussion before locking in a mathematical approach.
3. Add instruction text preferring a minimal high-signal toolset when more tools increase complexity faster than edge.
4. Mirror the essential rule in `AGENTS.md` so subagents inherit the same expectation.
5. Store the preference in persistent user memory.

## Edge Cases
- Do not require full textbook exposition for trivial arithmetic; the rule should activate for substantive mathematical logic.
- Preserve the repo’s existing research-first workflow and cost-discipline principles.
- Keep the new instruction precise and short enough to remain usable.

## Testing Plan
- Verify the updated instruction text is present in `.github/copilot-instructions.md` and `AGENTS.md`.
- Verify `/memories/workflow.md` includes the new persistent preference.
- No code or runtime tests required.

---

## Related

- [[math_explanation_workflow|Research: Math Explanation Workflow]]
