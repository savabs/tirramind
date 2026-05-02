---
description: "Extract and consolidate learnings from completed tasks and checkpoints into project memory."
---

# Extract Learnings

Mine completed work for reusable patterns and update the project's institutional memory.

## Instructions

### Step 1: Run the extraction script

Run: `python scripts/extract_patterns.py`

Review the output. It scans `tasks/done/` and recent checkpoints for sections titled "What We Learned", "Lessons", "Risks", "Observations", etc.

### Step 2: Identify actionable insights

From the extracted patterns, identify:
- **Workflow improvements** — things that should change in how we work (→ update `copilot-instructions.md`)
- **Architectural lessons** — things learned about the codebase structure (→ update `project_memory.md`)
- **Common pitfalls** — recurring bugs or mistakes (→ add to relevant `.instructions.md` or test templates)
- **Confirmed techniques** — approaches that worked well and should be reused (→ note in project memory)

### Step 3: Update project memory

For each actionable insight:
1. Check if it's already captured in `docs/memory/project_memory.md`. If yes, skip.
2. If new, add it to the appropriate section of `project_memory.md`.
3. If it's a workflow rule change, update `.github/copilot-instructions.md`.

### Step 4: Report

Output a summary of what was found and what was updated.

## Rules
- Do NOT fabricate patterns. Only report what the scripts actually found in the files.
- Do NOT update project memory with session-specific details — only durable knowledge.
- If no patterns are found, say "No new patterns detected" and suggest running this after the next feature is complete.
