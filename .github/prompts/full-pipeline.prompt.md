---
description: "Full pipeline: research → spec → implement → test → review → checkpoint. Automates the entire optimal workflow for a new feature."
---

# Full Feature Pipeline

You are executing the complete TirraMind development pipeline for a new feature.
Follow these phases in exact order.  Do NOT skip phases. Do NOT move to the next phase until the current phase is complete.

## Input Required

The user will provide:
- **Feature name**: A short identifier (e.g., `wikipedia_pageviews`, `vpin_calculation`)
- **Feature description**: What it should do, which layer it belongs to, any specific requirements

---

## Phase 1: Research (NO code changes)

1. Read only the files relevant to this feature area.
2. Identify: which layer (1-7), existing patterns to follow, insertion points, dependencies.
3. Search for free data sources / open-source libraries that apply.
4. Write findings to `docs/research/<feature_name>.md` using this structure:

```
# Feature: <name>
## Current Architecture
## Observations  
## Risks
## Data Requirements
## Math/Algorithm Survey
```

5. **STOP** and show the user a 3-line summary of the research. Ask: "Research complete. Proceed to spec phase?"

---

## Phase 2: Specification (NO code changes)

1. Read `docs/research/<feature_name>.md`.
2. Transform research into a precise implementation plan.
3. Write to `docs/specs/<feature_name>_spec.md`:

```
# Spec: <feature_name>
## Goal
## Files Affected
## Implementation Steps (numbered, atomic — each step changes ONE thing)
## Edge Cases
## Testing Plan
```

4. Create/update the task file at `tasks/active/<feature_name>.md`:

```
# Task: <feature_name>
Status: active
Research: docs/research/<feature_name>.md
Spec: docs/specs/<feature_name>_spec.md

## Steps
- [ ] 1.1: <first step description>
- [ ] 1.2: <second step description>
...
```

5. **STOP** and show the user the numbered step list. Ask: "Spec complete. Proceed to implementation?"

---

## Phase 3: Implementation (one step at a time)

For EACH step in the task file:

1. Read the task file to find the next `[ ]` step.
2. Read the spec for details on that step.
3. Implement ONLY that step — minimum file changes.
4. Write edge case tests for the code you just wrote:
   - Invalid inputs, boundary values, error paths
   - Type mismatches, missing required fields
   - NaN/Inf/empty for numerical code
   - HTTP errors/timeouts for tools
   - Mock all external I/O
5. Run: `pytest tests/ -v --tb=short -x`
6. If tests fail → fix → re-run. Do not proceed until green.
7. Mark the step `[x]` in the task file.
8. Move to next step. Repeat until all steps are done.

After ALL implementation steps complete:

9. **STOP** and show the user a summary of all files changed. Ask: "Implementation complete. Proceed to review?"

---

## Phase 4: Review

Review all changed files for:
- Correctness: matches spec?
- Numerical stability: division by zero, NaN propagation, empty arrays
- Security: no credential leaks, input validation at boundaries
- Layer discipline: code is in the correct layer, doesn't mix concerns
- Test coverage: every public function has edge case tests

Report as:

| Finding | Severity | File | Description |
|---------|----------|------|-------------|
| ... | PASS/WARN/FAIL | ... | ... |

If any FAIL findings: fix them, re-run tests, then continue.

---

## Phase 5: Checkpoint

1. Update the task file: set `Status: completed`.
2. Write a checkpoint to `docs/memory/chat_checkpoint_<today's date>.md`:

```
# Checkpoint: <date>
## Feature: <name>
## What Was Done
- (list of changes)
## Files Changed
- (list)
## Test Results
- (pass/fail summary)  
## Next Steps
- (what to work on next, if anything)
## Open Issues
- (anything unresolved)
```

3. Tell the user: "Feature complete. Recommend starting a new chat session for the next feature."
