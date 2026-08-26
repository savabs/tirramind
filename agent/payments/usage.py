"""Usage metering — per-subscriber API call log.

Tracks every metered API call (which key, which endpoint, which tier) so
usage can be reported back to a subscriber and, eventually, capped per tier.
Deliberately separate from PipelineStore (that's pipeline/tool execution
data) and from SubscriberStore (that's Paddle subscription lifecycle state)
— this is billing-adjacent customer usage data with its own access pattern
(append-heavy, queried by key + time range).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

_DEFAULT_DB_PATH = ".tirra_opportunities/usage.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    tier TEXT,
    requested_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_api_usage_key_time
    ON api_usage(key_id, requested_at);
"""


class UsageStore:
    """SQLite-backed log of metered API calls."""

    def __init__(self, path: str = _DEFAULT_DB_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def log(self, *, key_id: str, endpoint: str, tier: str | None = None) -> None:
        """Record one metered call. Called after successful authorization."""
        self._conn.execute(
            "INSERT INTO api_usage (key_id, endpoint, tier, requested_at) VALUES (?, ?, ?, ?)",
            (key_id, endpoint, tier, time.time()),
        )
        self._conn.commit()

    def count_since(self, key_id: str, since: float) -> int:
        """Total metered calls by this key since a given epoch timestamp."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM api_usage WHERE key_id=? AND requested_at>=?",
            (key_id, since),
        ).fetchone()
        return int(row["n"]) if row else 0

    def usage_by_endpoint(self, key_id: str, since: float | None = None) -> dict[str, int]:
        """Call counts grouped by endpoint, optionally since a timestamp."""
        if since is not None:
            rows = self._conn.execute(
                "SELECT endpoint, COUNT(*) AS n FROM api_usage " "WHERE key_id=? AND requested_at>=? GROUP BY endpoint",
                (key_id, since),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT endpoint, COUNT(*) AS n FROM api_usage WHERE key_id=? GROUP BY endpoint",
                (key_id,),
            ).fetchall()
        return {r["endpoint"]: int(r["n"]) for r in rows}

    def summary(self, key_id: str, since: float | None = None) -> dict[str, Any]:
        """Combined view: total calls + per-endpoint breakdown."""
        by_endpoint = self.usage_by_endpoint(key_id, since=since)
        return {"total": sum(by_endpoint.values()), "by_endpoint": by_endpoint}


__all__ = ["UsageStore"]
