"""Base classes shared by periodic watchers."""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from agent.awos.events.bus import EventBus
from agent.awos.events.schema import Event

log = logging.getLogger(__name__)


class Watcher(ABC):
    """A periodic producer of events."""

    name: str

    def __init__(self, bus: EventBus, repo_root: Path) -> None:
        self.bus = bus
        self.repo_root = Path(repo_root)

    @abstractmethod
    def scan(self) -> list[Event]:
        """Return events discovered this tick (may be empty)."""

    def run_once(self) -> int:
        try:
            events = self.scan()
        except Exception as e:
            log.exception("%s watcher failed: %s", self.name, e)
            return 0
        for e in events:
            try:
                self.bus.publish(e)
            except Exception as exc:
                log.exception("%s publish failed: %s", self.name, exc)
        return len(events)


def run_subprocess(
    cmd: list[str], cwd: Path, timeout: float = 30.0
) -> tuple[int, str, str]:
    """Run a subprocess, returning (code, stdout, stderr)."""
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except FileNotFoundError as e:
        return 127, "", str(e)


__all__ = ["Watcher", "run_subprocess"]
