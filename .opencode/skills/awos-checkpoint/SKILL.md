---
name: awos-checkpoint
title: "AWOS Checkpoint Skill"
tags:
  - tool/opencode
description: Standard AWOS checkpoint creation with versioning and single-owner facts
license: MIT
---

# AWOS Checkpoint Protocol

From AWOS.md §2.7 and `docs/memory/checkpoint_*.md`.

## When
- End of every non-trivial coding session
- After completing a task file
- Before switching context

## Format
```markdown
# Checkpoint: YYYY-MM-DD — <summary>

## Version
<incremented version>

## What changed
- File: change description and rationale

## Decisions
- Decision: rationale

## State
- Current phase: <phase>
- Active task: <task-id>
- Test status: <passing/failing/N>

## Next
- [ ] Next step 1
- [ ] Next step 2
```

## Rules
- Use `python scripts/session_checkpoint.py -m "summary"` if available
- Place in `docs/memory/` with filename `checkpoint_YYYY-MM-DD_<slug>.md`
- One owner per fact — link to `memories/repo/project_structure.md`
- No duplicate facts across checkpoints
