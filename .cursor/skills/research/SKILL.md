---
name: research
description: Research a new feature: read relevant files, analyze architecture, write research doc. No code changes. Use for research-only work with no code changes.
disable-model-invocation: true
---

## Instructions

This is a **research-only** task. Do NOT modify any code files.

1. Read only the files relevant to the requested feature/topic.
2. Analyze the project structure and identify:
   - Which layer(s) of the 7-layer stack this belongs to
   - Existing modules that relate to this feature
   - Dependencies and integration points
3. Write the research document to `docs/research/<feature_name>.md` with this structure:

```markdown
# Feature: <name>

## Current Architecture
- (relevant modules, patterns, dependencies)

## Observations
- (what exists, what's missing, what connects to what)

## Risks
- (edge cases, breaking changes, security concerns)

## Data Requirements
- (what data series/sources are needed, what's available, what's missing)

## Math/Algorithm Survey
- (what algorithms apply, what libraries exist vs. build from scratch, complexity)
```

4. Do NOT create specs or task files — that's the next phase.
