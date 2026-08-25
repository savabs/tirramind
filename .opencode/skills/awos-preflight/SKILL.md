---
name: awos-preflight
description: Pre-execution safety checks before implementing kernel changes
license: MIT
---

# AWOS Preflight Protocol

From AWOS.md §2.5 and `protocols/IMPLEMENTATION_PROTOCOL.md`.

## When to run
Before every non-trivial implementation step.

## Preflight checklist
1. **Research exists** — a `docs/research/<topic>.md` or equivalent
2. **Spec exists** — a `docs/spec/<feature>.md` with acceptance criteria
3. **Task file exists** — `tasks/active/<task-id>.md` with atomic steps
4. **No "and" steps** — each task step is one atomic unit of work
5. **VISION.md alignment** — mapped to a specific stage (1-7), not deferred work
6. **Fact ownership** — any new facts are assigned to `memories/repo/project_structure.md`
7. **Hot-path safety** — no edits planned to `.awosignore`-listed files or live state
8. **Test plan clear** — know which tests to run before and after
9. **Rollback plan** — how to undo if the change breaks things

## Output
Before implementing, state:
- Task file being worked on
- Specific step being executed
- Files that will change
- Tests to run
- Expected observable behavior
