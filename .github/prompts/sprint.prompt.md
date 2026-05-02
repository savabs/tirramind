---
description: "Sprint mode: execute ALL remaining steps in the active task file sequentially without stopping between steps."
---

# Sprint Mode

Execute all remaining uncompleted steps from the active task file without pausing between steps.

## Instructions

1. Read the task file from `tasks/active/` to find the current task.
2. Read the linked spec for full context.
3. Count the remaining `[ ]` steps.
4. For EACH remaining step, in order:
   a. Implement the step (minimum file changes).
   b. Write/update edge case tests.
   c. Run `pytest tests/ -v --tb=short -x`.
   d. If tests fail → fix → re-run.
   e. Mark step `[x]` in task file.
   f. Immediately continue to next step.
5. After ALL steps complete:
   - Run the full test suite: `pytest tests/ -v --tb=short`
   - Show summary: steps completed, files changed, test results
   - Write a checkpoint to `docs/memory/chat_checkpoint_<today>.md`
   - Update task status to `completed`

## Rules
- Do NOT stop between steps to ask for confirmation.
- Do NOT re-read the spec between steps (read it once at the start).
- If a step fails after 3 attempts, skip it and note it as blocked in the task file, then continue.
- If you need to create a new file, follow the conventions in the folder's `.instructions.md`.
