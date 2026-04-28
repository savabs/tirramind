"""Obsidian lint watcher — wraps ``scripts/obsidian_lint.py``.

Emits DRIFT events on FM01/FM02/LK01 findings.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.awos.events.schema import Event, TriggerCategory
from agent.awos.watchers.base import Watcher, run_subprocess


_HARD_CODES = {"FM01", "FM02", "LK01"}


class ObsidianWatcher(Watcher):
    name = "obsidian"

    def __init__(self, bus, repo_root: Path, *, timeout: float = 30.0) -> None:
        super().__init__(bus, repo_root)
        self.timeout = timeout

    def scan(self) -> list[Event]:
        script = self.repo_root / "scripts" / "obsidian_lint.py"
        if not script.exists():
            return []
        code, out, _err = run_subprocess(
            ["python", str(script), "--json"],
            self.repo_root,
            timeout=self.timeout,
        )
        findings = _parse_findings(out)
        hard = [f for f in findings if f.get("code") in _HARD_CODES]
        if not hard:
            return []
        return [
            Event(
                source=self.name,
                category=TriggerCategory.DRIFT,
                confidence=0.8,
                payload={
                    "findings": hard[:50],
                    "total": len(hard),
                    "exit_code": code,
                },
                rationale=f"obsidian_lint: {len(hard)} hard findings",
            )
        ]


def _parse_findings(out: str) -> list[dict]:
    out = out.strip()
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "findings" in data:
        return list(data["findings"])
    return []


__all__ = ["ObsidianWatcher"]
