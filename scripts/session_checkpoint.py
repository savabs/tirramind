#!/usr/bin/env python3
"""Auto-generate a session checkpoint from git state and active tasks.

Usage:
    python scripts/session_checkpoint.py [--message "optional summary"]

Writes to docs/memory/chat_checkpoint_<YYYY-MM-DD>[_sessionN].md
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = ROOT / "docs" / "memory"
TASKS_DIR = ROOT / "tasks" / "active"


def _git(cmd: str) -> str:
    """Run a git command and return stdout (empty string on failure)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT)] + cmd.split(),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _active_tasks() -> list[str]:
    """Return list of active task filenames (excluding .gitkeep)."""
    if not TASKS_DIR.exists():
        return []
    return sorted(f.stem for f in TASKS_DIR.glob("*.md") if f.name != ".gitkeep")


def _recent_commits(n: int = 5) -> str:
    """Return last N git log entries, one-line format."""
    return _git(f"log --oneline -n {n}")


def _changed_files_today() -> str:
    """Return files changed in the last commit."""
    return _git("diff --name-only HEAD~1 HEAD")


def _next_filename() -> Path:
    """Determine the checkpoint filename, avoiding collisions."""
    today = datetime.date.today().isoformat()
    base = MEMORY_DIR / f"chat_checkpoint_{today}.md"
    if not base.exists():
        return base
    # Find next session number
    session = 2
    while True:
        candidate = MEMORY_DIR / f"chat_checkpoint_{today}_session{session}.md"
        if not candidate.exists():
            return candidate
        session += 1


def generate_checkpoint(message: str = "") -> Path:
    """Generate and write a checkpoint file. Returns the path."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    filepath = _next_filename()
    today = datetime.date.today().isoformat()

    tasks = _active_tasks()
    task_links = "\n".join(f"- [[{t}]]" for t in tasks) if tasks else "- (none)"

    commits = _recent_commits()
    changed = _changed_files_today()

    summary = message if message else "Auto-generated session checkpoint."

    content = f"""---
title: "Checkpoint: {today}"
tags:
  - doc/checkpoint
---

# Checkpoint: {today}

**Generated:** {datetime.datetime.now().isoformat(timespec='minutes')}
**Summary:** {summary}

> **Single-owner rule:** Do NOT copy raw metric values (test counts, node counts,
> ENRICHMENT_DIM, DAG size, failure counts) into this checkpoint.
> Reference the canonical owners instead:
> - Current metrics → [[tirramind_structure]] (`memories/repo/tirramind_structure.md`)
> - Roadmap / next phases → [[quant_training_ground]] (`tasks/active/quant_training_ground.md`)
> This checkpoint is an **append-only historical record**. Never edit it after the session ends.

---

## Active Tasks

{task_links}

## Recent Commits

```
{commits if commits else "(no recent commits)"}
```

## Files Changed (last commit)

```
{changed if changed else "(no changes)"}
```

## Canonical State References

- Current metrics: see [[tirramind_structure]]
- Roadmap / next phases: see [[quant_training_ground]]
- Architecture decisions: see `docs/adr/`

## Related

{task_links}
"""

    filepath.write_text(content)
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a session checkpoint.")
    parser.add_argument("--message", "-m", default="", help="Optional summary message.")
    args = parser.parse_args()

    path = generate_checkpoint(args.message)
    print(f"Checkpoint written: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
