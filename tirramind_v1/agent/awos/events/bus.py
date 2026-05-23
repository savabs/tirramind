"""SQLite-backed, WAL-mode event bus.

Design:
- Append-only events table.
- Content-hash dedup within a configurable window.
- Process-safe (WAL + short retries on ``database is locked``).
- Idempotent writes: identical events within the dedup window return the
  original row's id, no duplicate row inserted.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agent.awos.events.schema import Event, EventStatus, TriggerCategory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    source        TEXT NOT NULL,
    category      TEXT NOT NULL,
    confidence    REAL NOT NULL,
    status        TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    rationale     TEXT,
    parent_event_id TEXT,
    dedup_hash    TEXT,
    payload_truncated INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_status   ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
CREATE INDEX IF NOT EXISTS idx_events_dedup_ts ON events(dedup_hash, ts);
"""


class EventBus:
    """Process-safe SQLite event bus.

    Thread-safety: each thread uses its own connection (SQLite's
    ``check_same_thread`` requirement). The class caches per-thread
    connections in a ``threading.local``.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        dedup_window_s: int = 600,
        max_payload_bytes: int = 1_000_000,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dedup_window = timedelta(seconds=dedup_window_s)
        self.max_payload_bytes = max_payload_bytes
        self.busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        self._init_schema()

    # ------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.busy_timeout_ms / 1000.0,
                isolation_level=None,  # autocommit; we manage BEGIN manually
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        with _retry_on_lock():
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    def publish(self, event: Event) -> Event:
        """Insert event unless an identical one exists in the dedup window.

        Returns either the newly-stored event or the pre-existing duplicate.
        """
        payload_json = json.dumps(event.payload, default=_json_default)
        truncated = False
        if len(payload_json) > self.max_payload_bytes:
            payload_json = json.dumps({"truncated": True, "prefix": payload_json[: self.max_payload_bytes]})
            truncated = True

        dedup_hash = event.dedup_hash or _hash_for_dedup(event.source, event.category, payload_json)

        conn = self._conn()
        # transactional lookup + insert to avoid dedup races
        with _retry_on_lock():
            conn.execute("BEGIN IMMEDIATE")
            try:
                cutoff = (datetime.now(UTC) - self.dedup_window).isoformat()
                existing = conn.execute(
                    "SELECT * FROM events WHERE dedup_hash = ? AND ts >= ? ORDER BY ts DESC LIMIT 1",
                    (dedup_hash, cutoff),
                ).fetchone()
                if existing is not None:
                    conn.execute("COMMIT")
                    return _row_to_event(existing)

                conn.execute(
                    "INSERT INTO events "
                    "(id, ts, source, category, confidence, status, "
                    " payload_json, rationale, parent_event_id, dedup_hash, "
                    " payload_truncated) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.id,
                        event.ts.isoformat(),
                        event.source,
                        event.category.value,
                        float(event.confidence),
                        event.status.value,
                        payload_json,
                        event.rationale,
                        event.parent_event_id,
                        dedup_hash,
                        int(truncated),
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        return event.model_copy(update={"dedup_hash": dedup_hash, "payload_truncated": truncated})

    # ------------------------------------------------------------------
    def fetch(
        self,
        *,
        limit: int = 100,
        status: EventStatus | None = None,
        category: TriggerCategory | None = None,
    ) -> list[Event]:
        sql = "SELECT * FROM events"
        conds: list[str] = []
        args: list[Any] = []
        if status is not None:
            conds.append("status = ?")
            args.append(status.value)
        if category is not None:
            conds.append("category = ?")
            args.append(category.value)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))

        conn = self._conn()
        with _retry_on_lock():
            rows = conn.execute(sql, tuple(args)).fetchall()
        return [_row_to_event(r) for r in rows]

    def get(self, event_id: str) -> Event | None:
        conn = self._conn()
        with _retry_on_lock():
            row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return _row_to_event(row) if row else None

    def mark(self, event_id: str, status: EventStatus) -> None:
        conn = self._conn()
        with _retry_on_lock():
            conn.execute(
                "UPDATE events SET status = ? WHERE id = ?",
                (status.value, event_id),
            )

    def bulk_mark(self, event_ids: Iterable[str], status: EventStatus) -> None:
        conn = self._conn()
        ids = list(event_ids)
        if not ids:
            return
        with _retry_on_lock():
            conn.executemany(
                "UPDATE events SET status = ? WHERE id = ?",
                [(status.value, i) for i in ids],
            )

    def count(self, status: EventStatus | None = None) -> int:
        conn = self._conn()
        if status is None:
            sql, args = "SELECT COUNT(*) FROM events", ()
        else:
            sql, args = "SELECT COUNT(*) FROM events WHERE status = ?", (status.value,)
        with _retry_on_lock():
            return int(conn.execute(sql, args).fetchone()[0])

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


# ======================================================================
def _hash_for_dedup(source: str, category: TriggerCategory, payload_json: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode())
    h.update(b"|")
    h.update(category.value.encode())
    h.update(b"|")
    h.update(payload_json.encode())
    return h.hexdigest()


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        ts=datetime.fromisoformat(row["ts"]),
        source=row["source"],
        category=TriggerCategory(row["category"]),
        confidence=row["confidence"],
        status=EventStatus(row["status"]),
        payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
        rationale=row["rationale"],
        parent_event_id=row["parent_event_id"],
        dedup_hash=row["dedup_hash"],
        payload_truncated=bool(row["payload_truncated"]),
    )


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


class _retry_on_lock:
    """Context manager that retries on ``database is locked`` up to N times."""

    def __init__(self, retries: int = 5, initial_delay: float = 0.01) -> None:
        self.retries = retries
        self.initial_delay = initial_delay

    def __enter__(self) -> _retry_on_lock:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # we do not swallow exceptions here; retry happens via __call__
        return False

    def __call__(self, fn):  # pragma: no cover - unused, kept for future
        def inner(*a, **kw):
            delay = self.initial_delay
            for _ in range(self.retries):
                try:
                    return fn(*a, **kw)
                except sqlite3.OperationalError as e:
                    if "locked" not in str(e):
                        raise
                    time.sleep(delay)
                    delay *= 2
            return fn(*a, **kw)

        return inner


__all__ = ["EventBus"]
