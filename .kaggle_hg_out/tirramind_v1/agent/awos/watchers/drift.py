"""Drift watcher — wraps ``scripts/fact_lint.py``.

Emits a ``DRIFT`` event when the fact-lint reports FL01/FL03 findings.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.awos.events.schema import Event, TriggerCategory
from agent.awos.watchers.base import Watcher, run_subprocess


class DriftWatcher(Watcher):
    name = "drift"

    def __init__(self, bus, repo_root: Path, *, timeout: float = 30.0) -> None:
        super().__init__(bus, repo_root)
        self.timeout = timeout

    def scan(self) -> list[Event]:
        script = self.repo_root / "scripts" / "fact_lint.py"
        if not script.exists():
            return []
        # prefer JSON output if supported; fall back to exit code
        code, out, err = run_subprocess(
            ["python", str(script), "--json"],
            self.repo_root,
            timeout=self.timeout,
        )
        findings = _parse_findings(out)
        if code == 127 or (code != 0 and not findings):
            # script missing or raw output — still emit a summary if the
            # exit code indicates failure
            if code in (0, 127):
                return []
            return [
                Event(
                    source=self.name,
                    category=TriggerCategory.DRIFT,
                    confidence=0.6,
                    payload={"exit_code": code, "stderr": err[:500]},
                    rationale="fact_lint exited non-zero",
                )
            ]
        if not findings:
            return []
        # filter to FL01/FL03 (canonical-owner violations)
        hard = [f for f in findings if f.get("code") in ("FL01", "FL03")]
        if not hard:
            return []
        return [
            Event(
                source=self.name,
                category=TriggerCategory.DRIFT,
                confidence=0.85,
                payload={"findings": hard[:50], "total": len(hard)},
                rationale=f"fact_lint found {len(hard)} FL01/FL03 findings",
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


__all__ = ["DriftWatcher"]
