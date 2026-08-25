---
title: "Spec: ECC-Inspired Workflow Improvements"
tags:
  - doc/spec
  - topic/workflow
  - topic/agent-infrastructure
---

# Spec: ECC Workflow Improvements

## Goal

Apply 6 concrete improvements inspired by the Everything Claude Code repo and Karpathy's CLAUDE.md learnings to TirraMind's developer workflow. Focus on automation gaps: session context persistence, pattern extraction, quality gates, checkpoint rotation, cold-start procedure, and loop prevention.

## Files Affected

### New Files
| File | Purpose |
|------|---------|
| `scripts/session_checkpoint.py` | Auto-generate checkpoint from git + task state |
| `scripts/rotate_checkpoints.py` | Archive old checkpoints, keep last N active |
| `scripts/quality_gate.py` | Pre-completion quality checks |
| `scripts/extract_patterns.py` | Mine completed tasks for reusable insights |
| `.github/prompts/session-start.prompt.md` | Cold-start prompt |
| `.github/prompts/extract-learnings.prompt.md` | Pattern extraction prompt |

### Modified Files
| File | Change |
|------|--------|
| `.github/copilot-instructions.md` | Add loop prevention rule, checkpoint rotation policy, reference new tools |
| `AGENTS.md` | Add new prompts to the Available Prompts table |

## Implementation Steps

### Step 1: Create `scripts/session_checkpoint.py`
Auto-generates a checkpoint markdown file from:
- Current date/time
- Active task files (glob `tasks/active/*.md`, exclude `.gitkeep`)
- Latest git log (last 5 commits)
- Changed files since last checkpoint
- Proper Obsidian frontmatter with `doc/checkpoint` tag

Usage: `python scripts/session_checkpoint.py` → writes `docs/memory/chat_checkpoint_<date>.md`

### Step 2: Create `scripts/rotate_checkpoints.py`
- Scans `docs/memory/chat_checkpoint_*.md`
- Keeps the N most recent (default N=15)
- Archives older ones: extracts title + date + "What Was Done" section, appends to `docs/memory/checkpoint_archive_<year>.md`
- Deletes the original archived files
- Prints summary of rotated/kept counts

Usage: `python scripts/rotate_checkpoints.py [--keep 15]`

### Step 3: Create `scripts/quality_gate.py`
Pre-completion checks:
1. All `tests/test_*.py` files pass (`pytest --tb=short -q`)
2. `python scripts/obsidian_lint.py` returns no FM01/FM02/LK01 errors
3. No `[ ]` (unchecked) steps remain in the active task file being closed
4. Exit code 0 = all pass, 1 = failures (with details)

Usage: `python scripts/quality_gate.py [--task [[foo]]]`

### Step 4: Create `scripts/extract_patterns.py`
Scans `tasks/done/` and recent checkpoints for:
- Recurring "What We Learned" / "Lessons" / "Risks realized" sections
- Groups by topic tag
- Outputs a markdown summary

Usage: `python scripts/extract_patterns.py` → prints summary to stdout

### Step 5: Create `session-start.prompt.md`
Codifies the cold-start procedure as a prompt:
1. Read `[[project_memory]]`
2. Read the most recent `docs/memory/chat_checkpoint_*.md`
3. List active tasks in `tasks/active/`
4. For the most relevant active task, follow `[[links]]` to its research + spec
5. Report: current state, what's next, any blockers

### Step 6: Create `extract-learnings.prompt.md`
Agent-driven pattern extraction:
1. Run `python scripts/extract_patterns.py`
2. Review the output for actionable insights
3. Update `[[project_memory]]` with new learnings
4. Update `copilot-instructions.md` if a workflow rule needs changing

### Step 7: Update `copilot-instructions.md`
Add to the Operational Collaboration Rules section:
- Loop prevention rule: "After 2 failed attempts at the same fix, mandatory switch to debug mode. Do not attempt a 3rd fix without first completing Steps 1-4 of the debug protocol."
- Checkpoint rotation policy: "Run `python scripts/rotate_checkpoints.py` when checkpoint count exceeds 30."
- Reference new scripts in the Memory System section.

### Step 8: Update `AGENTS.md`
Add `session-start` and `extract-learnings` to the Available Prompts table.

## Edge Cases

- `session_checkpoint.py`: No active tasks → still generates checkpoint with git info only
- `rotate_checkpoints.py`: Fewer than N checkpoints → no rotation, just reports count
- `quality_gate.py`: No active task file specified → checks all active tasks
- `extract_patterns.py`: No completed tasks → outputs "No completed tasks to analyze"

## Testing Plan

- Run each script standalone and verify output format
- `rotate_checkpoints.py`: Test with a temp directory containing fake checkpoints
- `quality_gate.py`: Test with a known-good and known-bad task file
- Validate modified `.md` files with `python scripts/obsidian_lint.py`

## Related

- [[ecc_workflow_improvements]]
- [[copilot-instructions]]
- [[project_memory]]
