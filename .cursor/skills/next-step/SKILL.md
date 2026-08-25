---
name: next-step
description: Execute the next atomic step from the active task file, write edge case tests, and update the task file. Use to execute exactly one unchecked step from the active task file.
disable-model-invocation: true
---

## Instructions

1. Read the active task file at `tasks/active/` to find the current task.
2. Identify the next uncompleted step (the first step not marked `[x]`).
3. Read the corresponding spec in `docs/specs/` if referenced.
4. Implement **only that one step** — change the minimum files necessary.
5. Write edge case tests covering: invalid inputs, boundary values, error paths, type mismatches, and domain-specific edge cases.
6. Run the tests: `pytest tests/ -v --tb=short`
7. If tests pass, mark the step as `[x]` done in the task file.
8. If tests fail, fix the implementation and re-run. Do not move to the next step.
9. Report: what was done, what tests were added, what's next.
