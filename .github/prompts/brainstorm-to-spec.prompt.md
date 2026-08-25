> Legacy reference. Use Cursor skill `/brainstorm-to-spec` — canonical: `.cursor/skills/brainstorm-to-spec/SKILL.md`

---
description: "Turn a rough idea into a research/spec-ready plan without writing code."
---

# Brainstorm To Spec

Use this prompt when the problem is still fuzzy and should stay in planning mode.

## Objective

Convert a rough request into a bounded implementation plan that is ready for the repository's research -> spec -> task workflow.

## Instructions

1. Restate the user's goal in one or two sentences.
2. Identify the distinct problems hidden inside the request.
3. Split combined requests into atomic sub-problems.
4. For each sub-problem, state whether it belongs in:
   - research
   - specification
   - implementation
   - testing
5. Identify what information is still missing before code should be written:
   - repository files to inspect
   - external documentation to gather
   - assumptions that need confirmation
6. Propose a minimal file plan:
   - research note path
   - spec path
   - task file path
   - likely implementation files, if already obvious
7. Draft an atomic step list where each step changes one thing and can be verified independently.
8. Include a short risk list covering scope confusion, missing docs, and testing obligations.
9. Stop before implementation.

## Output Format

Return exactly these sections:

```markdown
## Problem Breakdown
- ...

## Missing Inputs
- ...

## Proposed Files
- ...

## Atomic Steps
1. ...

## Risks
- ...

## Recommended Next Prompt
- ...
```

## Rules

- Do not write code.
- Do not propose multi-problem implementation steps.
- If unfamiliar technology is involved, require docs and OSS research before implementation.
- Keep the result compact enough that it can be turned directly into a spec and task file.