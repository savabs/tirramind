"""
TirraMind — Pipeline Store

SQLite-based persistent storage for pipeline run metadata and structured data.
WAL mode for concurrent read/write safety.

Schema:
    dag_runs       — execution metadata (run_id, dag_name, status, timing)
    pipeline_data  — tool output rows (source, params, data, timestamp)
    signals        — computed signal values (name, value, timestamp, metadata)

Usage:
    store = PipelineStore(Path(".tirra_pipeline/pipeline.db"))
    store.store_data("cftc", {"mode": "latest"}, {...})
    rows = store.query_data("cftc", since=1711270000.0)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_DB_PATH = ".tirra_pipeline/pipeline.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dag_runs (
    run_id TEXT PRIMARY KEY,
    dag_name TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL DEFAULT 'running',
    trigger TEXT NOT NULL DEFAULT 'manual',
    node_results_json TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    params_json TEXT NOT NULL,
    data_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pipeline_data_source
    ON pipeline_data(source, fetched_at);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name TEXT NOT NULL,
    computed_at REAL NOT NULL,
    value REAL NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_name
    ON signals(signal_name, computed_at);
"""


class PipelineStore:
    """SQLite-backed storage for pipeline runs, data, and signals."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = str(db_path)
        self._is_memory = self._db_path == ":memory:"
        if not self._is_memory:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    # ── connection management ──────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> PipelineStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── DAG runs ───────────────────────────────────────────────

    def record_run_start(
        self,
        dag_name: str,
        trigger: str = "manual",
        run_id: str | None = None,
    ) -> str:
        """Create a new dag_run record. Returns run_id."""
        if run_id is None:
            run_id = uuid.uuid4().hex[:12]
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO dag_runs (run_id, dag_name, started_at, status, trigger) "
            "VALUES (?, ?, ?, 'running', ?)",
            (run_id, dag_name, time.time(), trigger),
        )
        conn.commit()
        log.info("Pipeline run started: %s [%s] trigger=%s", dag_name, run_id, trigger)
        return run_id

    def record_run_end(
        self,
        run_id: str,
        status: str,
        node_results: dict[str, Any] | None = None,
    ) -> None:
        """Update a dag_run record with final status and node results."""
        conn = self._get_conn()
        node_json = json.dumps(node_results, default=str) if node_results else None
        conn.execute(
            "UPDATE dag_runs SET finished_at=?, status=?, node_results_json=? "
            "WHERE run_id=?",
            (time.time(), status, node_json, run_id),
        )
        conn.commit()
        log.info("Pipeline run ended: %s status=%s", run_id, status)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a specific run by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM dag_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_runs(
        self,
        dag_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get recent runs, optionally filtered by DAG name."""
        conn = self._get_conn()
        if dag_name:
            rows = conn.execute(
                "SELECT * FROM dag_runs WHERE dag_name=? "
                "ORDER BY started_at DESC LIMIT ?",
                (dag_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM dag_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── pipeline data ──────────────────────────────────────────

    def store_data(
        self,
        source: str,
        params: dict[str, Any],
        data: Any,
    ) -> int:
        """Insert a tool result row. Returns the row ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO pipeline_data (source, fetched_at, params_json, data_json) "
            "VALUES (?, ?, ?, ?)",
            (
                source,
                time.time(),
                json.dumps(params, default=str, sort_keys=True),
                json.dumps(data, default=str),
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        log.debug("Stored data: source=%s row_id=%s", source, row_id)
        return row_id  # type: ignore[return-value]

    def query_data(
        self,
        source: str,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query stored data rows by source and time range."""
        conn = self._get_conn()
        clauses = ["source=?"]
        params: list[Any] = [source]

        if since is not None:
            clauses.append("fetched_at>=?")
            params.append(since)
        if until is not None:
            clauses.append("fetched_at<=?")
            params.append(until)

        where = " AND ".join(clauses)
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM pipeline_data WHERE {where} "  # noqa: S608
            "ORDER BY fetched_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._data_row_to_dict(r) for r in rows]

    # ── signals ────────────────────────────────────────────────

    def store_signal(
        self,
        signal_name: str,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert a computed signal value. Returns the row ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO signals (signal_name, computed_at, value, metadata_json) "
            "VALUES (?, ?, ?, ?)",
            (
                signal_name,
                time.time(),
                value,
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        log.debug("Stored signal: %s=%.6f row_id=%s", signal_name, value, row_id)
        return row_id  # type: ignore[return-value]

    def query_signals(
        self,
        signal_name: str,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query signal values by name and time range."""
        conn = self._get_conn()
        clauses = ["signal_name=?"]
        params: list[Any] = [signal_name]

        if since is not None:
            clauses.append("computed_at>=?")
            params.append(since)
        if until is not None:
            clauses.append("computed_at<=?")
            params.append(until)

        where = " AND ".join(clauses)
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM signals WHERE {where} "  # noqa: S608
            "ORDER BY computed_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._signal_row_to_dict(r) for r in rows]

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if d.get("node_results_json"):
            try:
                d["node_results"] = json.loads(d["node_results_json"])
            except (json.JSONDecodeError, TypeError):
                d["node_results"] = None
        else:
            d["node_results"] = None
        d.pop("node_results_json", None)
        return d

    @staticmethod
    def _data_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        try:
            d["params"] = json.loads(d.pop("params_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            d["params"] = {}
        try:
            d["data"] = json.loads(d.pop("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            d["data"] = {}
        return d

    @staticmethod
    def _signal_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def new_run_id() -> str:
        """Generate a new unique run ID."""
        return uuid.uuid4().hex[:12]
