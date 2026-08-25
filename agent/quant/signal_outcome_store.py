"""SignalOutcomeStore — persistent pending/realized signal ledger.

The honest learning loop for the live digest:

  Phase 1 — SURFACE (no reward): the digest flags an anomaly and records a
      "pending" signal here with full context (direction, flagged time, ref value).
  Phase 2 — REALIZE: later, forward data is checked — did the flagged series
      actually move in the flagged direction? Only then is a real outcome
      recorded so the router learns honestly. Signals with no forward data yet
      stay pending (no reward — we do not guess).

This replaces the earlier behavior of awarding success on every surface, which
was exactly the fake-reward disease diagnosed earlier.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PendingSignal:
    """One surfaced anomaly awaiting forward-data realization."""

    status: str  # "pending" | "realized"
    source: str
    observation_type: str
    entity_id: str
    field: str
    direction: float  # +1 (bullish flag) or -1 (bearish flag); sign of z-score
    flagged_ts: float
    ref_value: float
    zscore: float
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    success: bool | None = None  # set only when realized
    realized_ts: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PendingSignal:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class SignalOutcomeStore:
    """Append-only JSONL ledger of pending + realized signal outcomes."""

    def __init__(self, path: str = ".tirra_opportunities/signal_outcomes.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, rec: PendingSignal | dict[str, Any]) -> None:
        data = rec.as_dict() if isinstance(rec, PendingSignal) else rec
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")

    def surface(
        self,
        *,
        source: str,
        observation_type: str,
        entity_id: str,
        field: str,
        direction: float,
        flagged_ts: float,
        ref_value: float,
        zscore: float,
    ) -> PendingSignal:
        """Record a surfaced anomaly as pending (no reward assigned)."""
        rec = PendingSignal(
            status="pending",
            source=source,
            observation_type=observation_type,
            entity_id=entity_id,
            field=field,
            direction=float(direction),
            flagged_ts=float(flagged_ts),
            ref_value=float(ref_value),
            zscore=float(zscore),
        )
        self._append(rec)
        return rec

    def realize(self, signal_id: str, success: bool) -> None:
        """Append the realization record that closes the loop for a signal."""
        self._append(
            {
                "signal_id": signal_id,
                "status": "realized",
                "success": bool(success),
                "realized_ts": _now_ts(),
            }
        )

    def pending(self) -> list[PendingSignal]:
        """Live pending signals not yet realized."""
        realized_ids = self._realized_ids()
        out: list[PendingSignal] = []
        for rec in self._iter_records():
            if rec.get("status") != "pending":
                continue
            sig = rec.get("signal_id")
            if sig in realized_ids:
                continue
            p = PendingSignal.from_dict(rec)
            if p:
                out.append(p)
        return out

    def _realized_ids(self) -> set[str]:
        return {r.get("signal_id") for r in self._iter_records() if r.get("status") == "realized"}

    def _iter_records(self) -> Any:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue


def _now_ts() -> float:
    import time
    return time.time()


__all__ = ["PendingSignal", "SignalOutcomeStore"]
