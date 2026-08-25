---
name: session-start
description: Cold-start a new session: load context from the knowledge graph without re-reading the entire codebase. Use for manual cold-start beyond the automatic sessionStart hook context.
disable-model-invocation: true
---

# Session Start

> **Note:** The `sessionStart` hook already injects active tasks and the latest checkpoint.
> Use this skill for a deeper manual cold-start (triad load, next-step identification).

You are starting a new coding session on TirraMind. Load the minimum context needed to be productive.

## Instructions

Follow these steps in exact order:

### Step 1: Load Project Memory
Read `docs/memory/project_memory.md` for persistent architectural knowledge.

### Step 2: Load Latest Checkpoint
Find and read the most recent `docs/memory/chat_checkpoint_*.md` file (sort by filename, take the last one). This tells you what happened in the previous session.

### Step 3: Identify Active Work
List all files in `tasks/active/` (excluding `.gitkeep`). Read the most relevant active task file.

### Step 4: Load the Triad
From the active task, follow the `Research:` and `Spec:` wiki links to load the research note and specification.

### Step 5: Report

Output this summary:

```
## Session Context

**Last session:** <date and summary from checkpoint>
**Active tasks:** <list of active task names>
**Current task:** <name of the most relevant task>
**Next step:** <the first unchecked step in the task>
**Key context:** <1-2 sentences about what the research/spec says for that step>
```

## Rules
- Do NOT read the entire codebase. Only read the 3-4 files this procedure tells you to.
- Do NOT start implementation until the user confirms the next step.
- If no active tasks exist, say so and ask the user what to work on.
- If the checkpoint mentions blockers or failed approaches, highlight them prominently.
