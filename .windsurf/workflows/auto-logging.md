---
title: Auto-Logging Workflow
description: Auto-update VERSIONS.md, LESSONS.md, and checkpoint memory every session
tags:
  - doc/wiki
  - topic/workflow
---

# Auto-Logging Workflow (STANDING ORDER)

This workflow runs automatically. Never wait for the user to ask.

## On Every Code Change

1. If it's a bug fix → add entry to `LESSONS.md` Part 1
2. If it's a new architecture concept → add entry to `LESSONS.md` Part 2
3. If it's a new Kaggle push → add row to `VERSIONS.md` kernel versions table

## On Session Start

1. Read `VERSIONS.md` and `LESSONS.md` to load context
2. Read latest checkpoint in `docs/memory/`
3. Update todo_list from active task file

## On Session End / Breakpoint

1. Write checkpoint to `docs/memory/chat_checkpoint_YYYY-MM-DD.md`
2. Update `VERSIONS.md` status columns (running→done, etc.)
3. Update todo_list final state
4. Confirm both `VERSIONS.md` and `LESSONS.md` are committed

## Checkpoint Template

```markdown
---
title: "Checkpoint: YYYY-MM-DD — [one-line summary]"
tags:
  - doc/checkpoint
  - phase/50
  - topic/gnn
---

# Checkpoint: [date]

## What was done
- 

## Decisions made
- 

## Current state
- Active version: Vxx
- Last epoch: xx
- IC: pending / +0.xxx

## Next steps
- 
```
