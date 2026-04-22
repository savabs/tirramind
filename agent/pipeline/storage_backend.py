"""
TirraMind — Storage Backend Abstraction

Defines the backend-neutral interface for database connection management.
PipelineStore delegates connection lifecycle and schema initialization to a
StorageBackend implementation, keeping domain logic independent of the
specific database driver.

Current backends:
    SQLiteBackend — local file or :memory: SQLite with WAL mode.
    PostgresBackend — PostgreSQL via psycopg2 with transparent SQL dialect
        translation (``INSERT OR REPLACE`` → ``ON CONFLICT DO UPDATE``,
        ``?`` → ``%s``, ``AUTOINCREMENT`` → ``SERIAL``, etc.).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Backend-neutral database connection provider.

    Implementations manage connection lifecycle, driver-specific
    configuration, and schema initialization.  The returned connection
    must be DB-API 2.0 compatible with a dict-like row factory
    (rows support ``dict(row)``).
    """

    @abstractmethod
    def get_connection(self) -> Any:
        """Return a DB-API 2.0 compatible connection.

        The connection's row factory must produce rows where
        ``dict(row)`` yields ``{column_name: value}`` mappings.
        Implementations should cache and reuse the connection.
        """
        ...

    @abstractmethod
    def init_schema(self, schema_sql: str) -> None:
        """Initialize database schema from DDL text."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the backend and release resources."""
        ...

    @property
    @abstractmethod
    def db_path(self) -> str:
        """Return the database path or identifier (for backward compat)."""
        ...

    @property
    def is_memory(self) -> bool:
        """Whether this backend uses an ephemeral in-memory store."""
        return False


class SQLiteBackend(StorageBackend):
    """SQLite-backed storage with WAL mode and busy-timeout."""

    def __init__(self, db_path: str | Path = ".tirra_pipeline/pipeline.db") -> None:
        self._db_path = str(db_path)
        self._is_memory = self._db_path == ":memory:"
        if not self._is_memory:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def get_connection(self) -> sqlite3.Connection:
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

    def init_schema(self, schema_sql: str) -> None:
        conn = self.get_connection()
        conn.executescript(schema_sql)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def is_memory(self) -> bool:
        return self._is_memory


# ── PostgreSQL SQL dialect translation ────────────────────────

# Tables whose INSERT OR REPLACE maps to ON CONFLICT ... DO UPDATE SET.
# Keys: table name → tuple of columns forming the UNIQUE/PK conflict target.
_UPSERT_CONFLICT_TARGETS: dict[str, tuple[str, ...]] = {
    "features": ("feature_name", "version", "effective_at"),
    "beliefs": ("variable_name", "version", "effective_at"),
    "pending_rl_transitions": ("date",),
    "portfolio_weights": ("date", "ticker"),
    "paper_trade_pnl": ("date",),
}

# Tables whose INSERT OR IGNORE maps to ON CONFLICT ... DO NOTHING.
_IGNORE_CONFLICT_TARGETS: dict[str, tuple[str, ...]] = {
    "entities": ("entity_id",),
    "entity_aliases": ("source", "external_id"),
    "entity_links": ("entity_id_a", "entity_id_b", "link_type"),
    "discovered_sources": ("source_id",),
    "entity_type_registry": ("type_name",),
}

# Tables with an auto-increment (SERIAL) primary key — RETURNING is
# appended to INSERTs so the adapter can populate ``lastrowid``.
_AUTO_INCREMENT_TABLES: frozenset[str] = frozenset(
    {
        "pipeline_data",
        "signals",
        "features",
        "beliefs",
        "entity_aliases",
        "entity_observations",
        "depth_evaluations",
        "entity_links",
        "entity_alerts",
        "convergence_clusters",
        "rl_transitions",
        "pending_rl_transitions",
        "rl_policy_checkpoints",
        "portfolio_weights",
        "paper_trade_pnl",
        "unresolved_entities",
    }
)

# Non-standard PK column names (default assumed to be ``id``).
_PK_COLUMNS: dict[str, str] = {
    "entity_aliases": "alias_id",
    "entity_links": "link_id",
}

# Pre-compiled patterns for DML rewriting.
_INSERT_OR_REPLACE_RE = re.compile(
    r"INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)\s*\(([^)]+)\)",
    re.IGNORECASE | re.DOTALL,
)
_INSERT_OR_IGNORE_RE = re.compile(
    r"INSERT\s+OR\s+IGNORE\s+INTO\s+(\w+)",
    re.IGNORECASE,
)
_INSERT_TABLE_RE = re.compile(
    r"INSERT\s+INTO\s+(\w+)",
    re.IGNORECASE,
)


def _translate_ddl(sql: str) -> str:
    """Translate SQLite DDL to PostgreSQL-compatible DDL."""
    sql = re.sub(
        r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b",
        "SERIAL PRIMARY KEY",
        sql,
        flags=re.IGNORECASE,
    )
    # SQLite REAL is 8-byte float; PostgreSQL REAL is only 4-byte.
    sql = re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bBLOB\b", "BYTEA", sql, flags=re.IGNORECASE)
    return sql


def _translate_dml(sql: str) -> tuple[str, str | None]:
    """Translate a single SQLite DML statement to PostgreSQL.

    Returns ``(translated_sql, pk_column)`` where *pk_column* is the
    auto-increment primary-key column name when a ``RETURNING`` clause
    should be appended, or ``None`` otherwise.
    """
    pk_col: str | None = None

    # ── INSERT OR REPLACE → INSERT … ON CONFLICT … DO UPDATE ──
    m = _INSERT_OR_REPLACE_RE.search(sql)
    if m:
        table = m.group(1)
        columns = [c.strip() for c in m.group(2).split(",")]
        conflict_cols = _UPSERT_CONFLICT_TARGETS.get(table, ())

        # Remove "OR REPLACE" from the statement text.
        sql = re.sub(
            r"INSERT\s+OR\s+REPLACE\s+INTO",
            "INSERT INTO",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )

        if conflict_cols:
            conflict_clause = ", ".join(conflict_cols)
            update_cols = [c for c in columns if c not in conflict_cols]
            if update_cols:
                set_clause = ", ".join(
                    f"{c} = EXCLUDED.{c}" for c in update_cols
                )
                sql = (
                    sql.rstrip()
                    + f" ON CONFLICT ({conflict_clause})"
                    f" DO UPDATE SET {set_clause}"
                )
            else:
                sql = (
                    sql.rstrip()
                    + f" ON CONFLICT ({conflict_clause}) DO NOTHING"
                )

        if table in _AUTO_INCREMENT_TABLES:
            pk_col = _PK_COLUMNS.get(table, "id")

    else:
        # ── INSERT OR IGNORE → INSERT … ON CONFLICT … DO NOTHING ──
        m = _INSERT_OR_IGNORE_RE.search(sql)
        if m:
            table = m.group(1)
            sql = re.sub(
                r"INSERT\s+OR\s+IGNORE\s+INTO",
                "INSERT INTO",
                sql,
                count=1,
                flags=re.IGNORECASE,
            )
            conflict_cols = _IGNORE_CONFLICT_TARGETS.get(table)
            if conflict_cols:
                conflict_clause = ", ".join(conflict_cols)
                sql = (
                    sql.rstrip()
                    + f" ON CONFLICT ({conflict_clause}) DO NOTHING"
                )
            else:
                sql = sql.rstrip() + " ON CONFLICT DO NOTHING"

            if table in _AUTO_INCREMENT_TABLES:
                pk_col = _PK_COLUMNS.get(table, "id")
        else:
            # ── Plain INSERT — detect table for RETURNING ──
            m = _INSERT_TABLE_RE.search(sql)
            if m:
                table = m.group(1)
                if table in _AUTO_INCREMENT_TABLES:
                    pk_col = _PK_COLUMNS.get(table, "id")

    # Replace ``?`` parameter placeholders with ``%s`` (psycopg2).
    sql = sql.replace("?", "%s")
    return sql, pk_col


class _PostgresCursorAdapter:
    """Thin wrapper around a psycopg2 cursor exposing ``lastrowid``."""

    __slots__ = ("_cursor", "_lastrowid")

    def __init__(self, cursor: Any, lastrowid: int | None = None) -> None:
        self._cursor = cursor
        self._lastrowid = lastrowid

    @property
    def lastrowid(self) -> int | None:
        return self._lastrowid

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cursor.fetchall()

    def __iter__(self) -> Any:
        return iter(self._cursor)


class _PostgresConnectionAdapter:
    """Wraps a psycopg2 connection to translate SQLite SQL on the fly.

    PipelineStore issues SQLite-dialect SQL (``?`` placeholders,
    ``INSERT OR REPLACE``, ``INSERT OR IGNORE``).  This adapter
    rewrites every statement to PostgreSQL syntax before execution,
    keeping the store's 2 400 lines of SQL unchanged.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # noinspection PyUnresolvedReferences
    def execute(
        self,
        sql: str,
        params: tuple[Any, ...] | list[Any] | None = None,
    ) -> _PostgresCursorAdapter:
        from psycopg2.extras import RealDictCursor

        translated, pk_col = _translate_dml(sql)

        if pk_col:
            translated = (
                translated.rstrip().rstrip(";") + f" RETURNING {pk_col}"
            )

        cursor = self._conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(translated, params)

        lastrowid: int | None = None
        if pk_col:
            row = cursor.fetchone()
            if row:
                lastrowid = row[pk_col]

        return _PostgresCursorAdapter(cursor, lastrowid=lastrowid)

    def executescript(self, sql_script: str) -> None:
        """Execute a multi-statement SQL script (for DDL init)."""
        cursor = self._conn.cursor()
        for stmt in sql_script.split(";"):
            stmt = stmt.strip()
            if stmt:
                cursor.execute(stmt)
        cursor.close()

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class PostgresBackend(StorageBackend):
    """PostgreSQL-backed storage via psycopg2.

    Requires ``psycopg2`` or ``psycopg2-binary`` (optional dependency).
    The backend returns a connection adapter that transparently translates
    SQLite SQL dialect to PostgreSQL, so ``PipelineStore`` works without
    any SQL changes.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string, e.g.
        ``"postgresql://user:pass@localhost:5432/tirramind"``.
    schema:
        Optional PostgreSQL schema name.  When set the backend creates
        the schema (if absent) and sets ``search_path`` so all tables
        are isolated inside it.  Useful for test isolation.
    """

    def __init__(self, dsn: str, *, schema: str | None = None) -> None:
        self._dsn = dsn
        self._schema = schema
        self._conn: Any = None
        self._adapter: _PostgresConnectionAdapter | None = None

    def get_connection(self) -> _PostgresConnectionAdapter:  # type: ignore[override]
        if self._adapter is None:
            try:
                import psycopg2
            except ImportError as exc:
                raise ImportError(
                    "psycopg2 is required for PostgresBackend. "
                    "Install it with: pip install psycopg2-binary"
                ) from exc

            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = False

            if self._schema:
                with self._conn.cursor() as cur:
                    # Identifier is validated to be a simple name below.
                    if not re.fullmatch(r"[a-zA-Z_]\w*", self._schema):
                        raise ValueError(
                            f"Invalid schema name: {self._schema!r}"
                        )
                    cur.execute(
                        f"CREATE SCHEMA IF NOT EXISTS {self._schema}"
                    )
                    cur.execute(f"SET search_path TO {self._schema}")
                self._conn.commit()

            self._adapter = _PostgresConnectionAdapter(self._conn)
        return self._adapter

    def init_schema(self, schema_sql: str) -> None:
        translated = _translate_ddl(schema_sql)
        adapter = self.get_connection()
        adapter.executescript(translated)
        adapter.commit()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._adapter = None

    @property
    def db_path(self) -> str:
        return self._dsn
