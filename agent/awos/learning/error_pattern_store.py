"""ErrorPatternStore — persistent failure-memory for the learning runtime.

Adapted from AWOS into the learning subpackage. Stores verbal critiques of
failed attempts (signal runs / tender matches / analyses) so the policy layer
can advise future runs without repeating the same mistake.

Reference: Reflexion (Shinn et al. 2023) — verbal critique in episodic memory
improves pass@1 without weight updates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = ".awos/error_patterns.jsonl"
_MAX_ERROR_MSG_CHARS = 300
_MAX_CRITIQUE_CHARS = 400


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ErrorPattern:
    """A single persisted failure critique."""

    task_id: str
    signal_name: str
    source_tool: str
    error_type: str
    error_msg: str
    critique: str
    created_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"ErrorPattern(signal={self.signal_name}, src={self.source_tool}, "
            f"type={self.error_type}, ts={self.created_at})"
        )


class ErrorPatternStore:
    """Append-only JSONL store of failure critiques (durable, thread-safe append)."""

    def __init__(self, store_path: str = _DEFAULT_STORE_PATH) -> None:
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, pattern: ErrorPattern) -> None:
        """Append exactly one ErrorPattern. Never raises."""
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(pattern.as_dict(), ensure_ascii=False) + "\n")
                fh.flush()
        except Exception:
            logger.exception("[ErrorPatternStore] save failed")

    def retrieve(self, task_id: str, signal_name: str, top_n: int = 3) -> list[ErrorPattern]:
        """Most recent critiques matching task_id (and signal_name, if given)."""
        matches: list[tuple[int, ErrorPattern]] = []
        for i, rec in enumerate(self._iter_records()):
            if rec.get("task_id") != task_id:
                continue
            if signal_name is not None and rec.get("signal_name") != signal_name:
                continue
            p = self._to_pattern(rec)
            if p is not None:
                matches.append((i, p))
        matches.sort(key=lambda t: (t[1].created_at, t[0]))  # chrono asc
        matches.reverse()  # newest first, tiebroken by latest insertion
        return [p for _, p in matches[:top_n]]

    def retrieve_all(self) -> list[ErrorPattern]:
        out: list[ErrorPattern] = []
        for rec in self._iter_records():
            p = self._to_pattern(rec)
            if p is not None:
                out.append(p)
        out.sort(key=lambda p: p.created_at, reverse=True)
        return out

    # ── Linked helpers for the evolving loop ────────────────────────────────
    def summary(self) -> dict[str, Any]:
        """Count failure critiques grouped by signal + error type."""
        by_signal: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for rec in self._iter_records():
            by_signal[rec.get("signal_name", "?")] = by_signal.get(rec.get("signal_name", "?"), 0) + 1
            by_type[rec.get("error_type", "?")] = by_type.get(rec.get("error_type", "?"), 0) + 1
        return {"count": sum(by_signal.values()), "by_signal": by_signal, "by_error_type": by_type}

    # ── Internals ───────────────────────────────────────────────────────────
    def _iter_records(self) -> Any:
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
        except Exception as exc:
            logger.warning("[ErrorPatternStore] read failed: %s", exc)

    @staticmethod
    def _to_pattern(rec: dict[str, Any]) -> ErrorPattern | None:
        try:
            return ErrorPattern(
                task_id=str(rec.get("task_id", "")),
                signal_name=str(rec.get("signal_name", "")),
                source_tool=str(rec.get("source_tool", "")),
                error_type=str(rec.get("error_type", "")),
                error_msg=str(rec.get("error_msg", ""))[:_MAX_ERROR_MSG_CHARS],
                critique=str(rec.get("critique", ""))[:_MAX_CRITIQUE_CHARS],
                created_at=str(rec.get("created_at", _now_iso())),
                metadata=rec.get("metadata") or {},
            )
        except Exception:
            return None


__all__ = ["ErrorPattern", "ErrorPatternStore"]
