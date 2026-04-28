"""Staleness watcher.

Detects:
- active task files whose last-modified time exceeds ``stale_task_days``
- checkpoint count exceeding a soft cap (>= 30) → recommend rotation
- long-open branches (informational only; left to a later version)
"""

from __future__ import annotations

import time
from pathlib import Path

from agent.awos.events.schema import Event, TriggerCategory
from agent.awos.watchers.base import Watcher


class StalenessWatcher(Watcher):
    name = "staleness"

    def __init__(
        self,
        bus,
        repo_root: Path,
        *,
        stale_task_days: int = 7,
        checkpoint_soft_cap: int = 30,
    ) -> None:
        super().__init__(bus, repo_root)
        self.stale_task_days = int(stale_task_days)
        self.checkpoint_soft_cap = int(checkpoint_soft_cap)

    def scan(self) -> list[Event]:
        events: list[Event] = []
        now = time.time()
        cutoff = now - self.stale_task_days * 86400

        tasks_dir = self.repo_root / "tasks" / "active"
        if tasks_dir.exists():
            stale: list[dict] = []
            for p in tasks_dir.glob("*.md"):
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if mtime < cutoff:
                    stale.append(
                        {
                            "path": str(p.relative_to(self.repo_root)),
                            "age_days": round((now - mtime) / 86400, 1),
                        }
                    )
            if stale:
                events.append(
                    Event(
                        source=self.name,
                        category=TriggerCategory.STALENESS,
                        confidence=0.7,
                        payload={"stale_tasks": stale},
                        rationale=f"{len(stale)} active tasks older than "
                        f"{self.stale_task_days} days",
                    )
                )

        checkpoints_dir = self.repo_root / "docs" / "memory"
        if checkpoints_dir.exists():
            cps = sorted(checkpoints_dir.glob("*checkpoint*.md"))
            if len(cps) >= self.checkpoint_soft_cap:
                events.append(
                    Event(
                        source=self.name,
                        category=TriggerCategory.STALENESS,
                        confidence=0.5,
                        payload={"checkpoint_count": len(cps)},
                        rationale=(
                            f"{len(cps)} checkpoints — consider rotation"
                        ),
                    )
                )

        return events


__all__ = ["StalenessWatcher"]
