"""BriefDeliverer — persistent delivery of the Intelligence Brief.

Pure I/O + schedule-friendly delivery layer. Takes a fully-built brief dict
(render function injected, so no layer inversion / no scripts import here),
writes JSON + rendered Markdown to a delivery directory, and appends a
persistent delivery record.

This is the bridge from "the pipeline produces a brief" to "a human/API can
consume the brief repeatedly."
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeliveryRecord:
    """One persisted delivery (row in the delivery log)."""

    delivered_at: float
    json_path: str
    md_path: str
    n_contracts: int
    n_anomalies: int
    duration_ms: float
    checksum: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeliveryRecord:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class BriefDeliverer:
    """Build-free delivery: persists brief JSON + Markdown to out_dir with a log."""

    def __init__(
        self,
        out_dir: str = ".tirra_delivery",
        stem: str = "intelligence_brief",
        render_md: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self._out = Path(out_dir)
        self._stem = stem
        self._render_md = render_md
        self._log_path = self._out / "delivery_log.jsonl"
        self._out.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────
    def deliver(self, brief: dict[str, Any]) -> DeliveryRecord:
        """Persist one brief (JSON + optional Markdown) and append the record."""
        started = time.time()
        json_text = json.dumps(brief, indent=2, ensure_ascii=False)
        checksum = hashlib.sha256(json_text.encode("utf-8")).hexdigest()[:16]

        json_path = self._out / f"{self._stem}.json"
        json_path.write_text(json_text, encoding="utf-8")
        md_path = self._out / f"{self._stem}.md"
        md_text = self._render_md(brief) if self._render_md else ""
        md_path.write_text(md_text, encoding="utf-8")

        record = DeliveryRecord(
            delivered_at=time.time(),
            json_path=str(json_path),
            md_path=str(md_path),
            n_contracts=len(brief.get("contract_opportunities", [])),
            n_anomalies=len(brief.get("live_anomalies", [])),
            duration_ms=round((time.time() - started) * 1000, 2),
            checksum=checksum,
        )
        self._append_record(record)
        return record

    def latest(self) -> DeliveryRecord | None:
        """Most recent delivery record, or None.

        records() returns newest-first, so the newest is the first element.
        """
        rows = self.records()
        return rows[0] if rows else None

    def records(self, limit: int = 200) -> list[DeliveryRecord]:
        """Delivery records, newest first (persisted, crash-readable)."""
        if not self._log_path.exists():
            return []
        out: list[DeliveryRecord] = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(DeliveryRecord.from_dict(json.loads(line)))
            except Exception:
                continue
        return out[-limit:][::-1]

    def status(self) -> dict[str, Any]:
        """Human/API-friendly status object."""
        latest = self.latest()
        return {
            "out_dir": str(self._out),
            "total_deliveries": len(self.records(limit=10_000)),
            "latest": latest.as_dict() if latest else None,
        }

    # ── Internals ───────────────────────────────────────────────────────────
    def _append_record(self, record: DeliveryRecord) -> None:
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


__all__ = ["BriefDeliverer", "DeliveryRecord"]
