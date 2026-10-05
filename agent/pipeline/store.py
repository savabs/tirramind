"""
TirraMind — Pipeline Store

SQLite-based persistent storage for pipeline run metadata and structured data.
WAL mode for concurrent read/write safety.

Schema:
    dag_runs             — execution metadata (run_id, dag_name, status, timing)
    pipeline_data        — tool output rows (source, params, data, timestamp)
    signals              — computed signal values (name, value, timestamp, metadata)
    features             — engineered feature records (EngineeredFeature protocol)
    beliefs              — world model posteriors
    entities             — canonical entity registry (cross-source)
    entity_aliases       — mappings from source-specific IDs to canonical entities
    entity_observations  — timestamped entity data points with depth level
    depth_evaluations    — depth measurement metrics (MI gain, KL divergence)
    portfolio_weights    — per-instrument weights emitted by inference DAG
    paper_trade_pnl      — daily paper-trade P&L records

Usage:
    store = PipelineStore(Path(".tirra_pipeline/pipeline.db"))
    store.store_data("cftc", {"mode": "latest"}, {...})
    rows = store.query_data("cftc", since=1711270000.0)
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, NoReturn

from agent.features.protocol import EngineeredFeature, validate_feature
from agent.models.belief import BeliefState, validate_belief
from agent.pipeline.storage_backend import SQLiteBackend, StorageBackend

log = logging.getLogger(__name__)

_DEFAULT_DB_PATH = ".tirra_pipeline/pipeline.db"

# ── entity_observations write-path guards (audit 2026-09-23, P1.1 / P1.4) ──
#
# ``store_entity_observation`` is the single write boundary every collector
# passes through.  Per-collector guards did not generalise — the same
# out-of-range-timestamp class recurred three times — so the bound lives here.

#: Lower bound for a plausible ``observed_at``: 1990-01-01T00:00:00Z.
OBSERVED_AT_MIN_EPOCH = 631_152_000.0

#: ``observed_at`` may lead wall-clock by at most this much (clock skew,
#: timezone slop).  Anything further ahead is a data error, not an observation.
OBSERVED_AT_MAX_AHEAD_SECONDS = 86_400.0

#: The logical uniqueness key of ``entity_observations``.  There is no UNIQUE
#: index backing it on the live database yet (see audit P2.2); until there is,
#: ``store_entity_observation`` enforces it in the write path.
#:
#: ``value_json`` is part of the key *deliberately*, and this differs from the
#: four-column key the 2026-09-23 audit proposes.  Measured on the live
#: database: 198,181 rows collide on the four-column key, but only 121,597 are
#: byte-identical.  The 76,584-row difference is almost entirely GDELT —
#: ``gdelt/geopolitical_event`` has 74,750 four-column collisions of which just
#: 1,822 are byte-identical, because GDELT timestamps are day-granular and a
#: country legitimately has hundreds of distinct events per day (one sampled
#: entity/day holds 255 rows with 255 distinct ``event_id`` values).  Keying
#: without ``value_json`` would collapse ~72,900 real observations into one
#: per entity-day — trading a duplicate-row bug for a data-loss bug.
#:
#: Including ``value_json`` reproduces the audit's 121,597 figure exactly, so
#: it removes precisely the duplicates the audit identified and nothing else.
#:
#: Caveat: the match is on the serialised bytes, so a collector that builds the
#: same payload with a different dict insertion order produces a different
#: ``value_json`` and will not dedup.  ``json.dumps(..., sort_keys=True)`` would
#: fix that, but it would also stop every *new* write matching the 384,285 rows
#: already on disk — that normalisation belongs with the P2.2 migration, not
#: here.
OBSERVATION_UNIQUE_KEY = (
    "entity_id",
    "source_tool",
    "observation_type",
    "observed_at",
    "value_json",
)


class ObservationRejected(ValueError):
    """An entity observation was refused at the write boundary.

    Raised by :meth:`PipelineStore.store_entity_observation` when
    ``observed_at`` is not a finite number inside
    ``[OBSERVED_AT_MIN_EPOCH, now + OBSERVED_AT_MAX_AHEAD_SECONDS]``.

    This is deliberately an exception rather than a skip-and-continue: a
    collector that mis-maps its date field must break loudly at the point of
    the mistake.  Silently clamping the timestamp to ``now`` is what
    ``gov_contracts`` already does, and it converts a visible data error into
    an invisible one.
    """


_PIPELINE_SCHEMA_NAME = "pipeline_store"
_PIPELINE_SCHEMA_VERSION = 1
_PIPELINE_SCHEMA_DESCRIPTION = "Baseline portable schema: epoch timestamps, integer booleans, JSON text payloads"

_SCHEMA_MIGRATIONS_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    schema_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    applied_at REAL NOT NULL,
    description TEXT NOT NULL,
    PRIMARY KEY (schema_name, version)
);

CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied
    ON schema_migrations(schema_name, applied_at);
"""

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

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    effective_at REAL NOT NULL,
    computed_at REAL NOT NULL,
    horizon TEXT NOT NULL,
    value REAL,
    quality REAL NOT NULL,
    missing_reason TEXT,
    source_signals_json TEXT NOT NULL,
    builder TEXT NOT NULL,
    unit TEXT NOT NULL DEFAULT 'raw',
    metadata_json TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_features_unique
    ON features(feature_name, version, effective_at);

CREATE INDEX IF NOT EXISTS idx_features_lookup
    ON features(feature_name, effective_at);

CREATE TABLE IF NOT EXISTS beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    variable_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    effective_at REAL NOT NULL,
    computed_at REAL NOT NULL,
    dist_type TEXT NOT NULL,
    mean REAL,
    variance REAL,
    probabilities_json TEXT,
    evidence_count INTEGER NOT NULL,
    model_graph_hash TEXT NOT NULL,
    confidence REAL NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_beliefs_unique
    ON beliefs(variable_name, version, effective_at);

CREATE INDEX IF NOT EXISTS idx_beliefs_lookup
    ON beliefs(variable_name, effective_at);

-- Entity registry (Phase 10a)

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    created_at REAL NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    UNIQUE(source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_lookup
    ON entity_aliases(source, external_id);

CREATE TABLE IF NOT EXISTS entity_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    source_tool TEXT NOT NULL,
    observed_at REAL NOT NULL,
    ingested_at REAL NOT NULL,
    observation_type TEXT NOT NULL,
    depth_level INTEGER NOT NULL DEFAULT 1,
    value_json TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_entity_obs_lookup
    ON entity_observations(entity_id, source_tool, observed_at);

-- Depth evaluation metrics (Phase 10a)

CREATE TABLE IF NOT EXISTS depth_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    depth_level INTEGER NOT NULL,
    evaluated_at REAL NOT NULL,
    target_variable TEXT NOT NULL,
    mi_gain REAL,
    kl_divergence REAL,
    sharpe_delta REAL,
    sample_size INTEGER NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_depth_eval_lookup
    ON depth_evaluations(tool_name, depth_level, evaluated_at);

-- Entity links (Phase 11a – cross-entity infrastructure)

CREATE TABLE IF NOT EXISTS entity_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id_a TEXT NOT NULL REFERENCES entities(entity_id),
    entity_id_b TEXT NOT NULL REFERENCES entities(entity_id),
    link_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL,
    created_at REAL NOT NULL,
    metadata_json TEXT,
    UNIQUE(entity_id_a, entity_id_b, link_type)
);

CREATE INDEX IF NOT EXISTS idx_entity_links_a
    ON entity_links(entity_id_a, link_type);
CREATE INDEX IF NOT EXISTS idx_entity_links_b
    ON entity_links(entity_id_b, link_type);

-- Entity alerts (Phase 20 – signal fusion)

CREATE TABLE IF NOT EXISTS entity_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    alert_time REAL NOT NULL,
    obs_type_surprise REAL NOT NULL,
    temporal_surprise REAL NOT NULL,
    value_surprise REAL NOT NULL,
    neighborhood_surprise REAL NOT NULL,
    memory_drift REAL NOT NULL,
    cusum_statistic REAL NOT NULL,
    hawkes_intensity REAL NOT NULL,
    event_study_score REAL NOT NULL,
    composite_surprise REAL NOT NULL,
    observation_count INTEGER NOT NULL,
    evidence_sources_json TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_entity_alerts_entity
    ON entity_alerts(entity_id, alert_time);
CREATE INDEX IF NOT EXISTS idx_entity_alerts_time
    ON entity_alerts(alert_time);
CREATE INDEX IF NOT EXISTS idx_entity_alerts_composite
    ON entity_alerts(composite_surprise);

-- Convergence clusters (Phase 20 – signal fusion)

CREATE TABLE IF NOT EXISTS convergence_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id TEXT NOT NULL UNIQUE,
    cluster_time REAL NOT NULL,
    member_entity_ids_json TEXT NOT NULL,
    correlated_surprise_score REAL NOT NULL,
    temporal_span_hours REAL NOT NULL,
    contributing_domains_json TEXT NOT NULL,
    contributing_tools_json TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_convergence_clusters_time
    ON convergence_clusters(cluster_time);
CREATE INDEX IF NOT EXISTS idx_convergence_clusters_score
    ON convergence_clusters(correlated_surprise_score);

CREATE TABLE IF NOT EXISTS rl_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    state_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    reward REAL NOT NULL,
    next_state_json TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_rl_transitions_ts
    ON rl_transitions(timestamp);

CREATE TABLE IF NOT EXISTS pending_rl_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    timestamp REAL NOT NULL,
    state_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    metadata_json TEXT,
    completed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pending_rl_date
    ON pending_rl_transitions(date);

CREATE TABLE IF NOT EXISTS rl_policy_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_at REAL NOT NULL,
    policy_type TEXT NOT NULL,
    config_json TEXT NOT NULL,
    state_dict_blob BLOB NOT NULL,
    metrics_json TEXT,
    is_best INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_rl_checkpoints_type
    ON rl_policy_checkpoints(policy_type, saved_at);

-- Portfolio weights (Phase 24d – inference DAG)

CREATE TABLE IF NOT EXISTS portfolio_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    weight REAL NOT NULL,
    metadata_json TEXT,
    created_at REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_weights_unique
    ON portfolio_weights(date, ticker);

CREATE INDEX IF NOT EXISTS idx_portfolio_weights_date
    ON portfolio_weights(date);

-- Paper trade P&L (Phase 24d – inference DAG)

CREATE TABLE IF NOT EXISTS paper_trade_pnl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    portfolio_return REAL NOT NULL,
    benchmark_return REAL NOT NULL,
    cumulative_return REAL NOT NULL,
    metadata_json TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_paper_trade_pnl_date
    ON paper_trade_pnl(date);

-- Tier 8: Autonomous Discovery tables

CREATE TABLE IF NOT EXISTS discovered_sources (
    source_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    description TEXT,
    format TEXT NOT NULL,
    update_frequency TEXT,
    topic_tags_json TEXT,
    probe_result_json TEXT,
    mi_score REAL,
    status TEXT NOT NULL DEFAULT 'discovered',
    discovered_at REAL NOT NULL,
    promoted_at REAL,
    tool_config_json TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS unresolved_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text TEXT NOT NULL,
    source_tool TEXT NOT NULL,
    context_snippet TEXT,
    observed_at REAL NOT NULL,
    cluster_id INTEGER,
    resolved_type TEXT,
    resolved_at REAL
);

CREATE INDEX IF NOT EXISTS idx_unresolved_cluster
    ON unresolved_entities(cluster_id);

CREATE TABLE IF NOT EXISTS entity_type_registry (
    type_name TEXT PRIMARY KEY,
    parent_type TEXT,
    discovered_at REAL NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    active INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT
);

-- Adversarial flags (Phase 22 – adversarial robustness scan)

CREATE TABLE IF NOT EXISTS adversarial_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flag_type TEXT NOT NULL,
    severity REAL NOT NULL,
    confidence REAL NOT NULL,
    entity_id TEXT,
    signal_name TEXT,
    flagged_at REAL NOT NULL,
    evidence_json TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_adversarial_flags_time
    ON adversarial_flags(flagged_at);
CREATE INDEX IF NOT EXISTS idx_adversarial_flags_entity
    ON adversarial_flags(entity_id, flagged_at);
CREATE INDEX IF NOT EXISTS idx_adversarial_flags_type
    ON adversarial_flags(flag_type, flagged_at);
"""


class PipelineStore:
    """SQLite-backed storage for pipeline runs, data, and signals.

    Accepts an optional ``backend`` (:class:`StorageBackend`) to
    decouple domain logic from the database driver.  When no backend
    is supplied, a :class:`SQLiteBackend` is created from *db_path*.
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB_PATH,
        *,
        backend: StorageBackend | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        else:
            self._backend = SQLiteBackend(db_path)
        # Write-boundary bookkeeping for entity_observations (audit P1.1/P1.4).
        # Counted per "<source_tool>/<observation_type>" so a caller that trips
        # a guard is identifiable without grepping the log.
        self._observations_rejected: dict[str, int] = {}
        self._observations_deduplicated: dict[str, int] = {}
        # entity_observations still has no UNIQUE index (audit P2.2), so the
        # idempotency check is read-then-write and needs to be atomic.  A
        # single PipelineStore is shared across DAGExecutor's worker threads
        # (see _LockingConnection), which locks per *statement*, not per
        # check-then-write pair.
        self._observation_write_lock = threading.RLock()
        self._init_schema()

    # ── backward-compat properties ────────────────────────────

    @property
    def _db_path(self) -> str:
        return self._backend.db_path

    @property
    def _is_memory(self) -> bool:
        return self._backend.is_memory

    @property
    def _conn(self) -> Any:
        """Expose the raw connection for legacy test access.

        Returns ``None`` when the backend connection is closed,
        preserving the behavior tests expect after ``close()``.
        """
        # Access the backend's internal connection state directly
        # so that ``store._conn is None`` works after close().
        return getattr(self._backend, "_conn", None)

    # ── connection management ──────────────────────────────────

    def _get_conn(self) -> Any:
        return self._backend.get_connection()

    def _init_schema(self) -> None:
        """Re-run schema initialization (idempotent)."""
        self._backend.init_schema(_SCHEMA_MIGRATIONS_SQL)
        self._backend.init_schema(_SCHEMA_SQL)
        self._migrate_dag_runs_heartbeat()
        self._record_schema_version()

    def _migrate_dag_runs_heartbeat(self) -> None:
        """Add ``dag_runs.heartbeat_at`` to databases created before it existed.

        ``CREATE TABLE IF NOT EXISTS`` in ``_SCHEMA_SQL`` never adds columns to
        an already-existing table, so any pre-existing ``.tirra_pipeline/*.db``
        needs this column added explicitly. Without it, a run that crashes
        mid-execution (process killed, box rebooted) leaves ``status='running'``
        forever — nothing ever flips it to 'failed'. ``reap_stale_runs()``
        below is what actually reaps those; this just gives it a column to
        distinguish "still alive" from "died silently."
        """
        conn = self._get_conn()
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(dag_runs)").fetchall()}
        if "heartbeat_at" not in cols:
            conn.execute("ALTER TABLE dag_runs ADD COLUMN heartbeat_at REAL")
            conn.commit()

    def _record_schema_version(self) -> None:
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE schema_name=? AND version=?",
            (_PIPELINE_SCHEMA_NAME, _PIPELINE_SCHEMA_VERSION),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO schema_migrations (schema_name, version, applied_at, description) VALUES (?, ?, ?, ?)",
                (
                    _PIPELINE_SCHEMA_NAME,
                    _PIPELINE_SCHEMA_VERSION,
                    time.time(),
                    _PIPELINE_SCHEMA_DESCRIPTION,
                ),
            )
            conn.commit()

    def get_schema_version(self) -> int:
        row = (
            self._get_conn()
            .execute(
                "SELECT MAX(version) AS version FROM schema_migrations WHERE schema_name=?",
                (_PIPELINE_SCHEMA_NAME,),
            )
            .fetchone()
        )
        if row is None or row["version"] is None:
            return 0
        return int(row["version"])

    def query_schema_migrations(self) -> list[dict[str, Any]]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT schema_name, version, applied_at, description "
                "FROM schema_migrations WHERE schema_name=? "
                "ORDER BY version ASC",
                (_PIPELINE_SCHEMA_NAME,),
            )
            .fetchall()
        )
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._backend.close()

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
        now = time.time()
        conn.execute(
            "INSERT INTO dag_runs (run_id, dag_name, started_at, status, trigger, heartbeat_at) "
            "VALUES (?, ?, ?, 'running', ?, ?)",
            (run_id, dag_name, now, trigger, now),
        )
        conn.commit()
        log.info("Pipeline run started: %s [%s] trigger=%s", dag_name, run_id, trigger)
        return run_id

    def heartbeat(self, run_id: str) -> None:
        """Mark *run_id* as still alive. Called periodically by the executor
        while a run is in progress (see ``DAGExecutor.execute``'s per-layer
        heartbeat) so ``reap_stale_runs`` can tell "long-running but alive"
        apart from "process died and nobody will ever call record_run_end."
        """
        conn = self._get_conn()
        conn.execute(
            "UPDATE dag_runs SET heartbeat_at=? WHERE run_id=? AND status='running'",
            (time.time(), run_id),
        )
        conn.commit()

    def reap_stale_runs(self, stale_after_seconds: float = 3600.0) -> list[str]:
        """Mark any run stuck in ``status='running'`` past *stale_after_seconds*
        since its last heartbeat as ``'failed'``.

        This is the fix for the documented failure mode: a process that dies
        mid-run (killed, OOM, box reboot) never reaches ``record_run_end``, so
        without this the row sits at ``status='running'`` forever — indistin-
        guishable, to any later query, from a run that is still legitimately
        in progress. Called opportunistically at the start of
        ``DAGExecutor.execute`` so every entry point (manual trigger,
        scheduler, ``run_chain.py``) self-heals the table before adding a new
        run to it.

        Returns the list of run_ids that were reaped.
        """
        conn = self._get_conn()
        cutoff = time.time() - stale_after_seconds
        rows = conn.execute(
            "SELECT run_id, dag_name, started_at, heartbeat_at FROM dag_runs "
            "WHERE status='running' AND COALESCE(heartbeat_at, started_at) < ?",
            (cutoff,),
        ).fetchall()
        reaped: list[str] = []
        for row in rows:
            run_id = row["run_id"]
            last_alive = row["heartbeat_at"] if row["heartbeat_at"] is not None else row["started_at"]
            age = time.time() - last_alive
            conn.execute(
                "UPDATE dag_runs SET status='failed', finished_at=?, "
                "node_results_json=? WHERE run_id=? AND status='running'",
                (
                    time.time(),
                    json.dumps(
                        {
                            "_reaped": (
                                f"No heartbeat for {age:.0f}s (threshold {stale_after_seconds:.0f}s); "
                                "the process that ran this DAG almost certainly died "
                                "mid-execution without recording a final status."
                            )
                        }
                    ),
                    run_id,
                ),
            )
            reaped.append(run_id)
            log.warning(
                "Reaped stale run %s (dag=%s): no heartbeat for %.0fs — marking failed",
                run_id,
                row["dag_name"],
                age,
            )
        conn.commit()
        return reaped

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
            "UPDATE dag_runs SET finished_at=?, status=?, node_results_json=? WHERE run_id=?",
            (time.time(), status, node_json, run_id),
        )
        conn.commit()
        log.info("Pipeline run ended: %s status=%s", run_id, status)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a specific run by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM dag_runs WHERE run_id=?", (run_id,)).fetchone()
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
                "SELECT * FROM dag_runs WHERE dag_name=? ORDER BY started_at DESC LIMIT ?",
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
            "INSERT INTO pipeline_data (source, fetched_at, params_json, data_json) VALUES (?, ?, ?, ?)",
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

    def list_sources(self) -> list[dict[str, Any]]:
        """Enumerate distinct sources with row count + latest fetch time.

        Used to publish a source catalog to API consumers (Data Platform tier)
        so they don't have to guess valid `source` values for query_data().
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source, COUNT(*) AS rows, MAX(fetched_at) AS last_fetched_at "
            "FROM pipeline_data GROUP BY source ORDER BY source"
        ).fetchall()
        return [{"source": r["source"], "rows": r["rows"], "last_fetched_at": r["last_fetched_at"]} for r in rows]

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
            "INSERT INTO signals (signal_name, computed_at, value, metadata_json) VALUES (?, ?, ?, ?)",
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

    # ── engineered features ─────────────────────────────────────

    def store_feature(self, feature: EngineeredFeature) -> int:
        """Validate and persist a single engineered feature.

        Uses INSERT OR REPLACE on the unique constraint
        ``(feature_name, version, effective_at)`` so duplicate
        recomputation is idempotent.

        Returns the row ID.

        Raises:
            ValueError: if the feature fails validation.
        """
        errors = validate_feature(feature)
        if errors:
            raise ValueError(f"Feature '{feature.feature_name}' failed validation: " + "; ".join(errors))
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT OR REPLACE INTO features "
            "(feature_name, version, effective_at, computed_at, horizon, "
            " value, quality, missing_reason, source_signals_json, builder, "
            " unit, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                feature.feature_name,
                feature.version,
                feature.effective_at,
                feature.computed_at,
                feature.horizon,
                feature.value,
                feature.quality,
                feature.missing_reason,
                json.dumps(list(feature.source_signals)),
                feature.builder,
                feature.unit,
                json.dumps(feature.metadata, default=str) if feature.metadata else None,
            ),
        )
        conn.commit()
        row_id: int = cursor.lastrowid  # type: ignore[assignment]
        log.debug(
            "Stored feature: %s v%d effective_at=%.0f row_id=%s",
            feature.feature_name,
            feature.version,
            feature.effective_at,
            row_id,
        )
        return row_id

    def store_features_batch(self, features: list[EngineeredFeature]) -> list[int]:
        """Validate and persist a batch of features in one transaction.

        Returns a list of row IDs (one per feature).

        Raises:
            ValueError: if *any* feature in the batch fails validation.
                No rows are written when this happens.
        """
        # Validate entire batch up-front so we don't partial-write.
        all_errors: list[str] = []
        for idx, feat in enumerate(features):
            errs = validate_feature(feat)
            if errs:
                all_errors.append(f"[{idx}] {feat.feature_name}: {'; '.join(errs)}")
        if all_errors:
            raise ValueError(f"Batch validation failed ({len(all_errors)} feature(s)): " + " | ".join(all_errors))

        conn = self._get_conn()
        row_ids: list[int] = []
        try:
            for feat in features:
                cursor = conn.execute(
                    "INSERT OR REPLACE INTO features "
                    "(feature_name, version, effective_at, computed_at, horizon, "
                    " value, quality, missing_reason, source_signals_json, builder, "
                    " unit, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        feat.feature_name,
                        feat.version,
                        feat.effective_at,
                        feat.computed_at,
                        feat.horizon,
                        feat.value,
                        feat.quality,
                        feat.missing_reason,
                        json.dumps(list(feat.source_signals)),
                        feat.builder,
                        feat.unit,
                        (json.dumps(feat.metadata, default=str) if feat.metadata else None),
                    ),
                )
                row_ids.append(cursor.lastrowid)  # type: ignore[arg-type]
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        log.debug("Stored %d features in batch", len(row_ids))
        return row_ids

    def query_features(
        self,
        feature_name: str,
        *,
        since: float | None = None,
        until: float | None = None,
        version: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query stored features by name, time range, and optional version."""
        conn = self._get_conn()
        clauses = ["feature_name=?"]
        params: list[Any] = [feature_name]

        if since is not None:
            clauses.append("effective_at>=?")
            params.append(since)
        if until is not None:
            clauses.append("effective_at<=?")
            params.append(until)
        if version is not None:
            clauses.append("version=?")
            params.append(version)

        where = " AND ".join(clauses)
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM features WHERE {where} "  # noqa: S608
            "ORDER BY effective_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._feature_row_to_dict(r) for r in rows]

    def get_latest_feature(
        self,
        feature_name: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the single most-recent feature record, or None."""
        results = self.query_features(feature_name, version=version, limit=1)
        return results[0] if results else None

    # ── beliefs ────────────────────────────────────────────────

    def store_belief(self, belief: BeliefState) -> int:
        """Validate and persist a single belief record.

        Uses INSERT OR REPLACE on the unique constraint
        ``(variable_name, version, effective_at)`` so duplicate
        recomputation is idempotent.

        Returns the row ID.

        Raises:
            ValueError: if the belief fails validation.
        """
        errors = validate_belief(belief)
        if errors:
            raise ValueError(f"Belief '{belief.variable_name}' failed validation: " + "; ".join(errors))
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT OR REPLACE INTO beliefs "
            "(variable_name, version, effective_at, computed_at, dist_type, "
            " mean, variance, probabilities_json, evidence_count, "
            " model_graph_hash, confidence, stale, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                belief.variable_name,
                belief.version,
                belief.effective_at,
                belief.computed_at,
                belief.dist_type,
                belief.mean,
                belief.variance,
                (json.dumps(belief.probabilities) if belief.probabilities is not None else None),
                belief.evidence_count,
                belief.model_graph_hash,
                belief.confidence,
                1 if belief.stale else 0,
                (json.dumps(belief.metadata, default=str) if belief.metadata else None),
            ),
        )
        conn.commit()
        row_id: int = cursor.lastrowid  # type: ignore[assignment]
        log.debug(
            "Stored belief: %s v%d effective_at=%.0f row_id=%s",
            belief.variable_name,
            belief.version,
            belief.effective_at,
            row_id,
        )
        return row_id

    def store_beliefs_batch(self, beliefs: list[BeliefState]) -> list[int]:
        """Validate and persist a batch of beliefs atomically.

        Returns a list of row IDs.

        Raises:
            ValueError: if *any* belief fails validation (no rows written).
        """
        all_errors: list[str] = []
        for idx, b in enumerate(beliefs):
            errs = validate_belief(b)
            if errs:
                all_errors.append(f"[{idx}] {b.variable_name}: {'; '.join(errs)}")
        if all_errors:
            raise ValueError(f"Batch validation failed ({len(all_errors)} belief(s)): " + " | ".join(all_errors))

        conn = self._get_conn()
        row_ids: list[int] = []
        try:
            for b in beliefs:
                cursor = conn.execute(
                    "INSERT OR REPLACE INTO beliefs "
                    "(variable_name, version, effective_at, computed_at, dist_type, "
                    " mean, variance, probabilities_json, evidence_count, "
                    " model_graph_hash, confidence, stale, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        b.variable_name,
                        b.version,
                        b.effective_at,
                        b.computed_at,
                        b.dist_type,
                        b.mean,
                        b.variance,
                        (json.dumps(b.probabilities) if b.probabilities is not None else None),
                        b.evidence_count,
                        b.model_graph_hash,
                        b.confidence,
                        1 if b.stale else 0,
                        (json.dumps(b.metadata, default=str) if b.metadata else None),
                    ),
                )
                row_ids.append(cursor.lastrowid)  # type: ignore[arg-type]
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        log.debug("Stored %d beliefs in batch", len(row_ids))
        return row_ids

    def query_beliefs(
        self,
        variable_name: str,
        *,
        since: float | None = None,
        until: float | None = None,
        version: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query stored beliefs by variable name, time range, and version."""
        conn = self._get_conn()
        clauses = ["variable_name=?"]
        params: list[Any] = [variable_name]

        if since is not None:
            clauses.append("effective_at>=?")
            params.append(since)
        if until is not None:
            clauses.append("effective_at<=?")
            params.append(until)
        if version is not None:
            clauses.append("version=?")
            params.append(version)

        where = " AND ".join(clauses)
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM beliefs WHERE {where} "  # noqa: S608
            "ORDER BY effective_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._belief_row_to_dict(r) for r in rows]

    def get_latest_belief(
        self,
        variable_name: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the single most-recent belief record, or None."""
        results = self.query_beliefs(variable_name, version=version, limit=1)
        return results[0] if results else None

    def query_all_latest_beliefs(self) -> list[dict[str, Any]]:
        """Return the most recent belief for each distinct variable_name.

        Uses a GROUP BY on variable_name with MAX(effective_at) to pick
        the latest record per variable.  Useful for loading the full
        current world-model belief state into downstream consumers
        (e.g. SAC inference).
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT b.* FROM beliefs b "
            "INNER JOIN ("
            "  SELECT variable_name, MAX(effective_at) AS max_ea "
            "  FROM beliefs GROUP BY variable_name"
            ") latest "
            "ON b.variable_name = latest.variable_name "
            "AND b.effective_at = latest.max_ea "
            "ORDER BY b.variable_name",
        ).fetchall()
        return [self._belief_row_to_dict(r) for r in rows]

    # ── edge confidence & component performance (Change 13/14) ─

    _EDGE_CONF_SOURCE = "edge_confidence"
    _COMPONENT_PERF_PREFIX = "component_perf_"

    def store_edge_confidences(
        self,
        as_of: float,
        dag_version: str,
        confidences: dict[str, Any],
    ) -> int:
        """Persist edge confidence scores for a DAG evaluation cycle."""
        return self.store_data(
            self._EDGE_CONF_SOURCE,
            {"as_of": as_of, "dag_version": dag_version},
            confidences,
        )

    def query_edge_confidence_history(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query recent edge confidence evaluations."""
        return self.query_data(self._EDGE_CONF_SOURCE, limit=limit)

    def store_component_performance(
        self,
        component: str,
        as_of: float,
        arm: int,
        reward: float,
        metrics: dict[str, Any] | None = None,
    ) -> int:
        """Persist a component refit outcome for the meta-scheduler."""
        return self.store_data(
            f"{self._COMPONENT_PERF_PREFIX}{component}",
            {"as_of": as_of, "arm": arm, "reward": reward},
            metrics or {},
        )

    def query_component_history(
        self,
        component: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query recent performance records for a component."""
        return self.query_data(f"{self._COMPONENT_PERF_PREFIX}{component}", limit=limit)

    def mark_beliefs_stale(
        self,
        reason: str = "structure_change",
        dag_version: str | None = None,
    ) -> int:
        """Mark all non-stale beliefs as stale. Returns count of rows updated."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE beliefs SET stale=1 WHERE stale=0",
        )
        conn.commit()
        count = cursor.rowcount
        log.info(
            "Marked %d beliefs stale (reason=%s, dag_version=%s)",
            count,
            reason,
            dag_version,
        )
        return count

    # ── entities ───────────────────────────────────────────────

    def register_entity(
        self,
        entity_type: str,
        canonical_name: str,
        entity_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a canonical entity. Returns entity_id.

        Uses INSERT OR IGNORE so re-registration is idempotent.
        """
        if not canonical_name or not canonical_name.strip():
            raise ValueError("canonical_name must be non-empty")
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO entities "
            "(entity_id, entity_type, canonical_name, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                entity_id,
                entity_type,
                canonical_name.strip(),
                time.time(),
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        conn.commit()
        log.debug(
            "Registered entity: %s type=%s name=%s",
            entity_id,
            entity_type,
            canonical_name,
        )
        return entity_id

    def add_entity_alias(
        self,
        entity_id: str,
        source: str,
        external_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Add a source-specific alias for an entity.

        Uses INSERT OR IGNORE so duplicate aliases are idempotent.
        """
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases "
            "(entity_id, source, external_id, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (entity_id, source, external_id, confidence, time.time()),
        )
        conn.commit()

    def resolve_entity(self, source: str, external_id: str) -> str | None:
        """Resolve a source-specific ID to a canonical entity_id.

        Returns None if no alias matches.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE source=? AND external_id=?",
            (source, external_id),
        ).fetchone()
        return row["entity_id"] if row else None

    def query_entity_aliases(self, entity_id: str) -> list[dict[str, Any]]:
        """Return all aliases for a given entity_id."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entity_aliases WHERE entity_id=?",
            (entity_id,),
        ).fetchall()
        return [
            {
                "alias_id": r["alias_id"],
                "entity_id": r["entity_id"],
                "source": r["source"],
                "external_id": r["external_id"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def query_all_entity_aliases(self) -> list[dict[str, Any]]:
        """Return every row in entity_aliases — used by EntityResolver."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM entity_aliases ORDER BY entity_id").fetchall()
        return [
            {
                "alias_id": r["alias_id"],
                "entity_id": r["entity_id"],
                "source": r["source"],
                "external_id": r["external_id"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Get a single entity record by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM entities WHERE entity_id=?", (entity_id,)).fetchone()
        if row is None:
            return None
        return self._entity_row_to_dict(row)

    # ── entity_observations write-path guards ──────────────────

    def _reject_observation(
        self,
        *,
        entity_id: str,
        source_tool: str,
        observation_type: str,
        observed_at: Any,
        reason: str,
        cause: BaseException | None = None,
    ) -> NoReturn:
        """Count, log and raise.  Never returns."""
        key = f"{source_tool}/{observation_type}"
        self._observations_rejected[key] = self._observations_rejected.get(key, 0) + 1
        message = (
            f"Rejected entity observation: {key} entity={entity_id} "
            f"observed_at={observed_at!r} — {reason}. "
            f"Valid range is [{OBSERVED_AT_MIN_EPOCH:.0f}, now+{OBSERVED_AT_MAX_AHEAD_SECONDS:.0f}] "
            f"epoch seconds (1990-01-01 .. tomorrow). "
            f"{self._observations_rejected[key]} rejected from this source in this process."
        )
        log.error(message)
        raise ObservationRejected(message) from cause

    def _validate_observed_at(
        self,
        observed_at: Any,
        *,
        entity_id: str,
        source_tool: str,
        observation_type: str,
    ) -> float:
        """Return *observed_at* as a float, or raise :class:`ObservationRejected`.

        Guards the single write boundary against the class of bug that put
        rows dated 1920-01-01, 1972-03-24 and 2030-01-30 into the live
        database (audit P1.1).  Those rows are not merely wrong in place —
        ``trainer.py`` derives its train/val/test boundaries from
        ``MIN``/``MAX(observed_at)``, so one future-dated row moves the whole
        evaluation window.
        """
        try:
            timestamp = float(observed_at)
        except (TypeError, ValueError) as exc:
            self._reject_observation(
                entity_id=entity_id,
                source_tool=source_tool,
                observation_type=observation_type,
                observed_at=observed_at,
                reason=f"not a number ({type(observed_at).__name__})",
                cause=exc,
            )
        # NaN fails every comparison, so it would slip through a bare range
        # check; inf would poison MAX(observed_at) outright.
        if not math.isfinite(timestamp):
            self._reject_observation(
                entity_id=entity_id,
                source_tool=source_tool,
                observation_type=observation_type,
                observed_at=observed_at,
                reason="not a finite number",
            )
        if timestamp < OBSERVED_AT_MIN_EPOCH:
            self._reject_observation(
                entity_id=entity_id,
                source_tool=source_tool,
                observation_type=observation_type,
                observed_at=observed_at,
                reason="implausibly old (before 1990-01-01)",
            )
        horizon = time.time() + OBSERVED_AT_MAX_AHEAD_SECONDS
        if timestamp > horizon:
            self._reject_observation(
                entity_id=entity_id,
                source_tool=source_tool,
                observation_type=observation_type,
                observed_at=observed_at,
                reason=f"in the future (more than {OBSERVED_AT_MAX_AHEAD_SECONDS:.0f}s ahead of now)",
            )
        return timestamp

    def _note_duplicate_observation(
        self,
        *,
        entity_id: str,
        source_tool: str,
        observation_type: str,
        observed_at: float,
        row_id: int,
    ) -> None:
        key = f"{source_tool}/{observation_type}"
        count = self._observations_deduplicated.get(key, 0) + 1
        self._observations_deduplicated[key] = count
        # Loud on the first hit per source, then every 1000th — a re-run of a
        # backfill must be visible in the log, without 100k identical lines.
        if count == 1 or count % 1000 == 0:
            log.warning(
                "Duplicate entity observation suppressed: %s entity=%s "
                "observed_at=%.3f row_id=%d — %d suppressed from this source in this "
                "process. Re-writing an identical observation is now idempotent; if "
                "you did not expect a re-run, something is writing the same row twice.",
                key,
                entity_id,
                observed_at,
                row_id,
                count,
            )

    def observation_write_stats(self) -> dict[str, dict[str, int]]:
        """Rows this process refused or collapsed at the observation boundary.

        Keys of both sub-dicts are ``"<source_tool>/<observation_type>"``.
        Counters are per-process, not persisted — they exist so a collection
        run can report what it threw away instead of reporting success.
        """
        return {
            "rejected": dict(self._observations_rejected),
            "deduplicated": dict(self._observations_deduplicated),
        }

    def store_entity_observation(
        self,
        entity_id: str,
        source_tool: str,
        observed_at: float,
        observation_type: str,
        value: Any,
        depth_level: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a timestamped observation for an entity. Returns row ID.

        This is the single write boundary for ``entity_observations``; two
        guards live here (audit 2026-09-23, P1.1 and P1.4).

        **Range.** ``observed_at`` must be a finite epoch-seconds value inside
        ``[1990-01-01, now + 1 day]``.  Anything else raises
        :class:`ObservationRejected` — it is not clamped, not skipped, not
        defaulted to ``now``.

        **Idempotency.** :data:`OBSERVATION_UNIQUE_KEY` — ``(entity_id,
        source_tool, observation_type, observed_at, value_json)`` — is treated
        as unique.  Re-writing an observation that is already stored refreshes
        that row (``ingested_at``, ``depth_level``, ``metadata_json``) and
        returns its id instead of appending, so re-running a collector cannot
        silently double-count.  The collapse is counted and logged; see
        :meth:`observation_write_stats`.

        Note that a same-timestamp observation carrying a *different* value is
        a different observation and is still stored — see
        :data:`OBSERVATION_UNIQUE_KEY` for why that matters (GDELT).

        The uniqueness is enforced here in Python because the live table has
        no UNIQUE index yet (121,597 pre-existing duplicates block creating
        one — audit P2.2).  ``self._observation_write_lock`` makes the
        check-then-write atomic within this process, which covers
        ``DAGExecutor``'s worker threads.  It does **not** make it atomic
        across processes; only a UNIQUE index will.

        Raises
        ------
        ObservationRejected
            If ``observed_at`` is out of range or not a finite number.
        """
        timestamp = self._validate_observed_at(
            observed_at,
            entity_id=entity_id,
            source_tool=source_tool,
            observation_type=observation_type,
        )
        value_json = json.dumps(value, default=str)
        metadata_json = json.dumps(metadata, default=str) if metadata else None
        ingested_at = time.time()
        conn = self._get_conn()

        with self._observation_write_lock:
            # ORDER BY matches the keep-rule of the planned dedup migration
            # (P2.2: newest ingested_at wins), so that on a live database that
            # still holds duplicate groups we update the row the migration
            # would keep.
            existing = conn.execute(
                "SELECT id FROM entity_observations "
                "WHERE entity_id=? AND source_tool=? AND observation_type=? "
                "AND observed_at=? AND value_json=? "
                "ORDER BY ingested_at DESC, id DESC LIMIT 1",
                (entity_id, source_tool, observation_type, timestamp, value_json),
            ).fetchone()

            if existing is not None:
                row_id = int(existing["id"])
                # value_json is part of the key, so only the non-key columns
                # move; last write wins on those.
                conn.execute(
                    "UPDATE entity_observations SET ingested_at=?, depth_level=?, metadata_json=? WHERE id=?",
                    (ingested_at, depth_level, metadata_json, row_id),
                )
                conn.commit()
                self._note_duplicate_observation(
                    entity_id=entity_id,
                    source_tool=source_tool,
                    observation_type=observation_type,
                    observed_at=timestamp,
                    row_id=row_id,
                )
                return row_id

            cursor = conn.execute(
                "INSERT INTO entity_observations "
                "(entity_id, source_tool, observed_at, ingested_at, "
                " observation_type, depth_level, value_json, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entity_id,
                    source_tool,
                    timestamp,
                    ingested_at,
                    observation_type,
                    depth_level,
                    value_json,
                    metadata_json,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid

        log.debug(
            "Stored entity observation: entity=%s tool=%s depth=%d row_id=%s",
            entity_id,
            source_tool,
            depth_level,
            row_id,
        )
        return row_id  # type: ignore[return-value]

    def query_entity_observations(
        self,
        entity_id: str,
        *,
        source_tool: str | None = None,
        since: float | None = None,
        until: float | None = None,
        depth_level: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query entity observations with optional filters."""
        conn = self._get_conn()
        clauses = ["entity_id=?"]
        params: list[Any] = [entity_id]

        if source_tool is not None:
            clauses.append("source_tool=?")
            params.append(source_tool)
        if since is not None:
            clauses.append("observed_at>=?")
            params.append(since)
        if until is not None:
            clauses.append("observed_at<=?")
            params.append(until)
        if depth_level is not None:
            clauses.append("depth_level=?")
            params.append(depth_level)

        where = " AND ".join(clauses)
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM entity_observations WHERE {where} "  # noqa: S608
            "ORDER BY observed_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._entity_obs_row_to_dict(r) for r in rows]

    def count_entity_observations(
        self,
        entity_id: str,
        *,
        source_tool: str | None = None,
    ) -> int:
        """Return the number of observations for an entity."""
        conn = self._get_conn()
        clauses = ["entity_id=?"]
        params: list[Any] = [entity_id]
        if source_tool is not None:
            clauses.append("source_tool=?")
            params.append(source_tool)
        where = " AND ".join(clauses)
        row = conn.execute(
            f"SELECT COUNT(*) FROM entity_observations WHERE {where}",  # noqa: S608
            params,
        ).fetchone()
        return row[0] if row else 0

    # ── entity links ───────────────────────────────────────────

    def link_entities(
        self,
        entity_id_a: str,
        entity_id_b: str,
        link_type: str,
        source: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        """Create a typed link between two entities.

        Idempotent via INSERT OR IGNORE.  Returns the link_id on
        insertion, or ``None`` when the (a, b, link_type) tuple
        already exists.

        Raises ``ValueError`` for self-links (entity_id_a == entity_id_b).
        """
        if entity_id_a == entity_id_b:
            raise ValueError("Cannot link an entity to itself")
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT OR IGNORE INTO entity_links "
            "(entity_id_a, entity_id_b, link_type, confidence, "
            " source, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entity_id_a,
                entity_id_b,
                link_type,
                confidence,
                source,
                time.time(),
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        conn.commit()
        if cursor.lastrowid and cursor.rowcount:
            log.debug(
                "Linked entities: %s -> %s type=%s",
                entity_id_a,
                entity_id_b,
                link_type,
            )
            return cursor.lastrowid
        return None  # already existed

    def query_entity_links(
        self,
        entity_id: str,
        *,
        link_type: str | None = None,
        direction: str = "both",
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query links for an entity.

        *direction* controls which side of the link to match:
        ``"outgoing"`` — entity is *entity_id_a* (source),
        ``"incoming"`` — entity is *entity_id_b* (target),
        ``"both"`` (default) — either side.
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(f"direction must be outgoing/incoming/both, got {direction!r}")

        conn = self._get_conn()
        parts: list[str] = []
        params: list[Any] = []

        if direction in ("outgoing", "both"):
            clauses = ["entity_id_a=?", "confidence>=?"]
            p: list[Any] = [entity_id, min_confidence]
            if link_type is not None:
                clauses.append("link_type=?")
                p.append(link_type)
            parts.append(f"SELECT * FROM entity_links WHERE {' AND '.join(clauses)}")  # noqa: S608
            params.extend(p)

        if direction in ("incoming", "both"):
            clauses = ["entity_id_b=?", "confidence>=?"]
            p = [entity_id, min_confidence]
            if link_type is not None:
                clauses.append("link_type=?")
                p.append(link_type)
            parts.append(f"SELECT * FROM entity_links WHERE {' AND '.join(clauses)}")  # noqa: S608
            params.extend(p)

        sql = " UNION ".join(parts) + " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()  # noqa: S608
        return [self._entity_link_row_to_dict(r) for r in rows]

    def query_co_occurrences(
        self,
        entity_id_a: str,
        entity_id_b: str,
        *,
        window_seconds: float = 72 * 3600,
        source_tool_a: str | None = None,
        source_tool_b: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find temporal co-occurrences between two entities.

        Returns observation pairs where
        ``abs(obs_a.observed_at - obs_b.observed_at) <= window_seconds``.

        Each result dict contains ``obs_a``, ``obs_b``, and
        ``time_delta_seconds`` (signed: positive means a comes after b).
        """
        conn = self._get_conn()

        # Build WHERE clauses for each side (params must match SQL
        # placeholder order: all ``a`` clauses first, then ``b``).
        clauses_a = ["a.entity_id=?"]
        clauses_b = ["b.entity_id=?"]
        params_a: list[Any] = [entity_id_a]
        params_b: list[Any] = [entity_id_b]

        if source_tool_a is not None:
            clauses_a.append("a.source_tool=?")
            params_a.append(source_tool_a)
        if source_tool_b is not None:
            clauses_b.append("b.source_tool=?")
            params_b.append(source_tool_b)
        if since is not None:
            clauses_a.append("a.observed_at>=?")
            params_a.append(since)
            clauses_b.append("b.observed_at>=?")
            params_b.append(since)

        where_a = " AND ".join(clauses_a)
        where_b = " AND ".join(clauses_b)

        params: list[Any] = params_a + params_b + [window_seconds, limit]

        sql = (
            "SELECT "
            "  a.id AS a_id, a.entity_id AS a_entity_id, "
            "  a.source_tool AS a_source_tool, a.observed_at AS a_observed_at, "
            "  a.ingested_at AS a_ingested_at, "
            "  a.observation_type AS a_observation_type, "
            "  a.depth_level AS a_depth_level, "
            "  a.value_json AS a_value_json, a.metadata_json AS a_metadata_json, "
            "  b.id AS b_id, b.entity_id AS b_entity_id, "
            "  b.source_tool AS b_source_tool, b.observed_at AS b_observed_at, "
            "  b.ingested_at AS b_ingested_at, "
            "  b.observation_type AS b_observation_type, "
            "  b.depth_level AS b_depth_level, "
            "  b.value_json AS b_value_json, b.metadata_json AS b_metadata_json "
            "FROM entity_observations a "
            "INNER JOIN entity_observations b "
            f"ON ({where_a}) AND ({where_b}) "  # noqa: S608
            "   AND ABS(a.observed_at - b.observed_at) <= ? "
            "ORDER BY ABS(a.observed_at - b.observed_at) ASC "
            "LIMIT ?"
        )

        rows = conn.execute(sql, params).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            obs_a = self._co_occ_half(d, "a")
            obs_b = self._co_occ_half(d, "b")
            results.append(
                {
                    "obs_a": obs_a,
                    "obs_b": obs_b,
                    "time_delta_seconds": obs_a["observed_at"] - obs_b["observed_at"],
                }
            )
        return results

    @staticmethod
    def _co_occ_half(row_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
        """Extract one side of a co-occurrence join row into an obs dict."""
        d: dict[str, Any] = {
            "id": row_dict[f"{prefix}_id"],
            "entity_id": row_dict[f"{prefix}_entity_id"],
            "source_tool": row_dict[f"{prefix}_source_tool"],
            "observed_at": row_dict[f"{prefix}_observed_at"],
            "ingested_at": row_dict[f"{prefix}_ingested_at"],
            "observation_type": row_dict[f"{prefix}_observation_type"],
            "depth_level": row_dict[f"{prefix}_depth_level"],
        }
        try:
            d["value"] = json.loads(row_dict.get(f"{prefix}_value_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            d["value"] = {}
        try:
            d["metadata"] = json.loads(row_dict.get(f"{prefix}_metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    # ── bulk graph queries (Phase 12a) ─────────────────────────

    def query_all_entities(
        self,
        *,
        entity_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return all entities, optionally filtered by type.

        *limit*/*offset* page the result at the SQL level (default: return
        everything, unchanged from prior behaviour — existing training-code
        callers pass neither).
        """
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if entity_type is not None:
            clauses.append("entity_type=?")
            params.append(entity_type)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = f"SELECT * FROM entities {where}ORDER BY created_at"  # noqa: S608
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        return [self._entity_row_to_dict(r) for r in rows]

    def count_all_entities(self, *, entity_type: str | None = None) -> int:
        """Return the total entity count, optionally filtered by type.

        Used for pagination metadata without materializing every row.
        """
        conn = self._get_conn()
        if entity_type is not None:
            row = conn.execute("SELECT COUNT(*) FROM entities WHERE entity_type=?", (entity_type,)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return row[0] if row else 0

    def query_all_observations(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return all observations ordered by observed_at.

        Optional *since* / *until* narrow the time window.
        """
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("observed_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("observed_at <= ?")
            params.append(until)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = conn.execute(
            f"SELECT * FROM entity_observations WHERE {where} "  # noqa: S608
            "ORDER BY observed_at",
            params,
        ).fetchall()
        return [self._entity_obs_row_to_dict(r) for r in rows]

    def query_all_entity_links(
        self,
        *,
        link_type: str | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return all entity links, optionally filtered by type and confidence.

        *limit*/*offset* page the result at the SQL level (default: return
        everything, unchanged from prior behaviour — existing training-code
        callers pass neither).
        """
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if link_type is not None:
            clauses.append("link_type=?")
            params.append(link_type)
        if min_confidence > 0.0:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM entity_links WHERE {where} ORDER BY created_at"  # noqa: S608
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = [*params, limit, offset]
        rows = conn.execute(sql, params).fetchall()
        return [self._entity_link_row_to_dict(r) for r in rows]

    def count_all_entity_links(
        self,
        *,
        link_type: str | None = None,
        min_confidence: float = 0.0,
    ) -> int:
        """Return the total entity-link count matching the given filters.

        Used for pagination metadata without materializing every row.
        """
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if link_type is not None:
            clauses.append("link_type=?")
            params.append(link_type)
        if min_confidence > 0.0:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        where = " AND ".join(clauses) if clauses else "1=1"
        row = conn.execute(
            f"SELECT COUNT(*) FROM entity_links WHERE {where}",  # noqa: S608
            params,
        ).fetchone()
        return row[0] if row else 0

    # ── depth evaluations ──────────────────────────────────────

    def store_depth_evaluation(
        self,
        tool_name: str,
        depth_level: int,
        target_variable: str,
        sample_size: int,
        *,
        mi_gain: float | None = None,
        kl_divergence: float | None = None,
        sharpe_delta: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a depth evaluation result. Returns row ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO depth_evaluations "
            "(tool_name, depth_level, evaluated_at, target_variable, "
            " mi_gain, kl_divergence, sharpe_delta, sample_size, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tool_name,
                depth_level,
                time.time(),
                target_variable,
                mi_gain,
                kl_divergence,
                sharpe_delta,
                sample_size,
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        log.debug(
            "Stored depth evaluation: tool=%s depth=%d target=%s row_id=%s",
            tool_name,
            depth_level,
            target_variable,
            row_id,
        )
        return row_id  # type: ignore[return-value]

    def query_depth_evaluations(
        self,
        tool_name: str | None = None,
        *,
        depth_level: int | None = None,
        target_variable: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query depth evaluation records with optional filters."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []

        if tool_name is not None:
            clauses.append("tool_name=?")
            params.append(tool_name)
        if depth_level is not None:
            clauses.append("depth_level=?")
            params.append(depth_level)
        if target_variable is not None:
            clauses.append("target_variable=?")
            params.append(target_variable)

        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM depth_evaluations WHERE {where} "  # noqa: S608
            "ORDER BY evaluated_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._depth_eval_row_to_dict(r) for r in rows]

    # ── entity alerts (Phase 20) ──────────────────────────────

    def store_entity_alert(
        self,
        entity_id: str,
        entity_type: str,
        entity_name: str,
        alert_time: float,
        *,
        obs_type_surprise: float,
        temporal_surprise: float,
        value_surprise: float,
        neighborhood_surprise: float,
        memory_drift: float,
        cusum_statistic: float,
        hawkes_intensity: float,
        event_study_score: float,
        composite_surprise: float,
        observation_count: int,
        evidence_sources: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store an entity alert. Returns row ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO entity_alerts "
            "(entity_id, entity_type, entity_name, alert_time, "
            "obs_type_surprise, temporal_surprise, value_surprise, "
            "neighborhood_surprise, memory_drift, "
            "cusum_statistic, hawkes_intensity, event_study_score, "
            "composite_surprise, observation_count, "
            "evidence_sources_json, metadata_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                entity_id,
                entity_type,
                entity_name,
                alert_time,
                obs_type_surprise,
                temporal_surprise,
                value_surprise,
                neighborhood_surprise,
                memory_drift,
                cusum_statistic,
                hawkes_intensity,
                event_study_score,
                composite_surprise,
                observation_count,
                json.dumps(list(evidence_sources)),
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def delete_entity_alerts(self, *, since: float, until: float) -> int:
        """Delete entity_alerts rows with ``alert_time`` in ``[since, until)``.

        Used by entity_scoring to make same-day reruns idempotent: a plain
        INSERT with no key here previously meant running the DAG twice in
        one day silently doubled ``entity_alerts`` (9,704 rows for 4,852
        distinct entities after two runs). The caller deletes the current
        scoring day's window before re-inserting, so a rerun replaces that
        day's alerts instead of duplicating them, while alerts from other
        days (the 7-day rolling window `inference` reads) are untouched.

        Returns the number of rows deleted.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM entity_alerts WHERE alert_time >= ? AND alert_time < ?",
            (since, until),
        )
        conn.commit()
        return cursor.rowcount

    def query_entity_alerts(
        self,
        *,
        entity_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        min_composite: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query entity alerts with optional filters."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if entity_id is not None:
            clauses.append("entity_id=?")
            params.append(entity_id)
        if since is not None:
            clauses.append("alert_time >= ?")
            params.append(since)
        if until is not None:
            clauses.append("alert_time <= ?")
            params.append(until)
        if min_composite is not None:
            clauses.append("composite_surprise >= ?")
            params.append(min_composite)
        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM entity_alerts WHERE {where} "  # noqa: S608
            "ORDER BY alert_time DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._entity_alert_row_to_dict(r) for r in rows]

    # ── convergence clusters (Phase 20) ────────────────────────

    def store_convergence_cluster(
        self,
        cluster_id: str,
        cluster_time: float,
        member_entity_ids: list[str],
        correlated_surprise_score: float,
        temporal_span_hours: float,
        *,
        contributing_domains: tuple[str, ...] = (),
        contributing_tools: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a convergence cluster. Returns row ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO convergence_clusters "
            "(cluster_id, cluster_time, member_entity_ids_json, "
            "correlated_surprise_score, temporal_span_hours, "
            "contributing_domains_json, contributing_tools_json, metadata_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                cluster_id,
                cluster_time,
                json.dumps(member_entity_ids),
                correlated_surprise_score,
                temporal_span_hours,
                json.dumps(list(contributing_domains)),
                json.dumps(list(contributing_tools)),
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def query_convergence_clusters(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        min_score: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query convergence clusters with optional filters."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("cluster_time >= ?")
            params.append(since)
        if until is not None:
            clauses.append("cluster_time <= ?")
            params.append(until)
        if min_score is not None:
            clauses.append("correlated_surprise_score >= ?")
            params.append(min_score)
        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM convergence_clusters WHERE {where} "  # noqa: S608
            "ORDER BY cluster_time DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._convergence_cluster_row_to_dict(r) for r in rows]

    # ── adversarial flags (Phase 22) ───────────────────────────

    def store_adversarial_flag(
        self,
        flag_type: str,
        severity: float,
        confidence: float,
        flagged_at: float,
        *,
        entity_id: str | None = None,
        signal_name: str | None = None,
        evidence: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store an adversarial detection flag. Returns row ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO adversarial_flags "
            "(flag_type, severity, confidence, entity_id, signal_name, "
            "flagged_at, evidence_json, metadata_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                flag_type,
                severity,
                confidence,
                entity_id,
                signal_name,
                flagged_at,
                json.dumps(evidence, default=str) if evidence else "{}",
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def query_adversarial_flags(
        self,
        *,
        flag_type: str | None = None,
        entity_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        min_severity: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query adversarial flags with optional filters."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if flag_type is not None:
            clauses.append("flag_type=?")
            params.append(flag_type)
        if entity_id is not None:
            clauses.append("entity_id=?")
            params.append(entity_id)
        if since is not None:
            clauses.append("flagged_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("flagged_at <= ?")
            params.append(until)
        if min_severity is not None:
            clauses.append("severity >= ?")
            params.append(min_severity)
        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM adversarial_flags WHERE {where} "  # noqa: S608
            "ORDER BY flagged_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._adversarial_flag_row_to_dict(r) for r in rows]

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _feature_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["source_signals"] = tuple(json.loads(d.pop("source_signals_json", "[]")))
        except (json.JSONDecodeError, TypeError):
            d["source_signals"] = ()
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
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
    def _data_row_to_dict(row: Any) -> dict[str, Any]:
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
    def _signal_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def _belief_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["probabilities"] = json.loads(d.pop("probabilities_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["probabilities"] = None
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        d["stale"] = bool(d.get("stale", 0))
        return d

    @staticmethod
    def _entity_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def _entity_obs_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["value"] = json.loads(d.pop("value_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            d["value"] = {}
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def _depth_eval_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def _entity_link_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def _entity_alert_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["evidence_sources"] = tuple(json.loads(d.pop("evidence_sources_json", "[]")))
        except (json.JSONDecodeError, TypeError):
            d["evidence_sources"] = ()
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def _convergence_cluster_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["member_entity_ids"] = json.loads(d.pop("member_entity_ids_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            d["member_entity_ids"] = []
        try:
            d["contributing_domains"] = tuple(json.loads(d.pop("contributing_domains_json", "[]")))
        except (json.JSONDecodeError, TypeError):
            d["contributing_domains"] = ()
        try:
            d["contributing_tools"] = tuple(json.loads(d.pop("contributing_tools_json", "[]")))
        except (json.JSONDecodeError, TypeError):
            d["contributing_tools"] = ()
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def _adversarial_flag_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["evidence"] = json.loads(d.pop("evidence_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            d["evidence"] = {}
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    @staticmethod
    def new_run_id() -> str:
        """Generate a new unique run ID."""
        return uuid.uuid4().hex[:12]

    # ── RL transitions ─────────────────────────────────────────

    def store_rl_transition(
        self,
        timestamp: float,
        state: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        next_state: dict[str, Any],
        done: bool,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store one RL transition. Returns the row id."""
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO rl_transitions "
            "(timestamp, state_json, action_json, reward, next_state_json, done, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                json.dumps(state, default=str),
                json.dumps(action, default=str),
                reward,
                json.dumps(next_state, default=str),
                int(done),
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def query_rl_transitions(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query RL transitions within a time range."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if start_time is not None:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM rl_transitions{where} ORDER BY timestamp"  # noqa: S608
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["state"] = json.loads(d.pop("state_json", "{}"))
            d["action"] = json.loads(d.pop("action_json", "{}"))
            d["next_state"] = json.loads(d.pop("next_state_json", "{}"))
            d["done"] = bool(d["done"])
            try:
                d["metadata"] = json.loads(d.pop("metadata_json", "null"))
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = None
            result.append(d)
        return result

    # ── Pending RL transitions ─────────────────────────────────

    def store_pending_transition(
        self,
        date: str,
        timestamp: float,
        state: list[float],
        action: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a pending transition (state + action, reward unknown yet).

        Uses INSERT OR REPLACE so re-runs for the same date are idempotent.
        Returns the row id.
        """
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT OR REPLACE INTO pending_rl_transitions "
            "(date, timestamp, state_json, action_json, metadata_json, completed) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (
                date,
                timestamp,
                json.dumps(state),
                json.dumps(action),
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def query_pending_transition(self, date: str) -> dict[str, Any] | None:
        """Load a pending transition for a specific date, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM pending_rl_transitions WHERE date=? AND completed=0",
            (date,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["state"] = json.loads(d.pop("state_json", "[]"))
        d["action"] = json.loads(d.pop("action_json", "[]"))
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    def complete_pending_transition(
        self,
        date: str,
        reward: float,
        next_state: list[float],
        done: bool = False,
    ) -> bool:
        """Complete a pending transition: write the full transition to rl_transitions.

        Loads the pending row for *date*, combines with reward/next_state,
        writes to rl_transitions, and marks the pending row as completed.

        Returns True if a transition was completed, False if no pending found.
        """
        pending = self.query_pending_transition(date)
        if pending is None:
            return False

        # Validate state/next_state are finite
        import math

        for val in pending["state"]:
            if not math.isfinite(val):
                log.warning("Pending state for %s contains non-finite value, skipping", date)
                return False
        for val in next_state:
            if not math.isfinite(val):
                log.warning("next_state for %s contains non-finite value, skipping", date)
                return False

        # Store the completed transition
        self.store_rl_transition(
            timestamp=pending["timestamp"],
            state=pending["state"],
            action=pending["action"],
            reward=reward,
            next_state=next_state,
            done=done,
            metadata=pending.get("metadata"),
        )

        # Mark pending as completed
        conn = self._get_conn()
        conn.execute(
            "UPDATE pending_rl_transitions SET completed=1 WHERE date=?",
            (date,),
        )
        conn.commit()
        return True

    # ── RL policy checkpoints ──────────────────────────────────

    def store_rl_checkpoint(
        self,
        policy_type: str,
        config: dict[str, Any],
        state_dict_bytes: bytes,
        metrics: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> int:
        """Store a policy checkpoint blob. Returns the row id."""
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO rl_policy_checkpoints "
            "(saved_at, policy_type, config_json, state_dict_blob, metrics_json, is_best) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                policy_type,
                json.dumps(config, default=str),
                state_dict_bytes,
                json.dumps(metrics, default=str) if metrics else None,
                int(is_best),
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def load_best_rl_checkpoint(self, policy_type: str) -> dict[str, Any] | None:
        """Load the best checkpoint for a policy type, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM rl_policy_checkpoints WHERE policy_type=? AND is_best=1 ORDER BY saved_at DESC LIMIT 1",
            (policy_type,),
        ).fetchone()
        if row is None:
            return None
        return self._parse_checkpoint_row(row)

    def load_latest_rl_checkpoint(self, policy_type: str) -> dict[str, Any] | None:
        """Load the most recent checkpoint for a policy type, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM rl_policy_checkpoints WHERE policy_type=? ORDER BY saved_at DESC LIMIT 1",
            (policy_type,),
        ).fetchone()
        if row is None:
            return None
        return self._parse_checkpoint_row(row)

    @staticmethod
    def _parse_checkpoint_row(row: Any) -> dict[str, Any]:
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json", "{}"))
        d["state_dict_bytes"] = d.pop("state_dict_blob")
        try:
            d["metrics"] = json.loads(d.pop("metrics_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metrics"] = None
        d["is_best"] = bool(d["is_best"])
        return d

    # ── portfolio weights (Phase 24d) ─────────────────────────

    def store_portfolio_weights(
        self,
        date: str,
        weights: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> list[int]:
        """Store portfolio weights for a given date.

        One row per ticker. Uses INSERT OR REPLACE so re-runs for the
        same date are idempotent.

        Parameters
        ----------
        date : Calendar date string, e.g. ``"2026-04-14"``.
        weights : ``{ticker: weight}`` mapping.
        metadata : Optional per-date metadata (stored identically on each row).

        Returns
        -------
        List of row IDs for stored rows.
        """
        if not date or not date.strip():
            raise ValueError("date must be non-empty")
        if not weights:
            raise ValueError("weights must be non-empty")
        conn = self._get_conn()
        meta_json = json.dumps(metadata, default=str) if metadata else None
        now = time.time()
        row_ids: list[int] = []
        try:
            for ticker, weight in weights.items():
                cur = conn.execute(
                    "INSERT OR REPLACE INTO portfolio_weights "
                    "(date, ticker, weight, metadata_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (date.strip(), ticker, float(weight), meta_json, now),
                )
                if cur.lastrowid:
                    row_ids.append(cur.lastrowid)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        log.debug("Stored %d portfolio weights for %s", len(row_ids), date)
        return row_ids

    def query_portfolio_weights(self, date: str) -> dict[str, float]:
        """Query portfolio weights for a specific date.

        Returns
        -------
        ``{ticker: weight}`` dict.  Empty dict if no weights for that date.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT ticker, weight FROM portfolio_weights WHERE date=?",
            (date.strip(),),
        ).fetchall()
        return {r["ticker"]: r["weight"] for r in rows}

    # ── paper trade P&L (Phase 24d) ───────────────────────────

    def store_paper_pnl(
        self,
        date: str,
        portfolio_return: float,
        benchmark_return: float,
        cumulative_return: float,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store one day of paper-trade P&L.

        Uses INSERT OR REPLACE keyed on the UNIQUE date column, so
        re-runs for the same date overwrite the previous row.

        Returns
        -------
        Row ID.
        """
        if not date or not date.strip():
            raise ValueError("date must be non-empty")
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT OR REPLACE INTO paper_trade_pnl "
            "(date, portfolio_return, benchmark_return, cumulative_return, "
            " metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                date.strip(),
                float(portfolio_return),
                float(benchmark_return),
                float(cumulative_return),
                json.dumps(metadata, default=str) if metadata else None,
                time.time(),
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        log.debug(
            "Stored paper P&L for %s: port=%.6f bench=%.6f",
            date,
            portfolio_return,
            benchmark_return,
        )
        return row_id  # type: ignore[return-value]

    def query_paper_pnl(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 365,
    ) -> list[dict[str, Any]]:
        """Query paper-trade P&L records.

        Parameters
        ----------
        start_date, end_date : Calendar date strings for range filtering.
        limit : Maximum records to return.

        Returns
        -------
        List of dicts with ``date``, ``portfolio_return``, ``benchmark_return``,
        ``cumulative_return``, and ``metadata`` keys.
        """
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if start_date is not None:
            clauses.append("date >= ?")
            params.append(start_date.strip())
        if end_date is not None:
            clauses.append("date <= ?")
            params.append(end_date.strip())
        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM paper_trade_pnl WHERE {where} "  # noqa: S608
            "ORDER BY date ASC LIMIT ?",
            params,
        ).fetchall()
        return [self._pnl_row_to_dict(r) for r in rows]

    @staticmethod
    def _pnl_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.pop("metadata_json", "null"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = None
        return d

    # ── Tier 8: discovered sources ─────────────────────────────

    def store_discovered_source(
        self,
        source_id: str,
        name: str,
        url: str,
        fmt: str,
        *,
        description: str | None = None,
        update_frequency: str | None = None,
        topic_tags: list[str] | None = None,
        probe_result: dict | list | None = None,
        mi_score: float | None = None,
        status: str = "discovered",
        tool_config: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Store a discovered data source candidate."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO discovered_sources "
            "(source_id, name, url, description, format, update_frequency, "
            "topic_tags_json, probe_result_json, mi_score, status, discovered_at, "
            "tool_config_json, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                name,
                url,
                description,
                fmt,
                update_frequency,
                json.dumps(topic_tags) if topic_tags else None,
                json.dumps(probe_result, default=str) if probe_result else None,
                mi_score,
                status,
                time.time(),
                json.dumps(tool_config, default=str) if tool_config else None,
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        conn.commit()

    def update_source_status(
        self,
        source_id: str,
        status: str,
    ) -> None:
        """Update a discovered source's status."""
        conn = self._get_conn()
        extras = ""
        params: list[Any] = [status]
        if status == "active":
            extras = ", promoted_at=?"
            params.append(time.time())
        params.append(source_id)
        conn.execute(
            f"UPDATE discovered_sources SET status=?{extras} WHERE source_id=?",  # noqa: S608
            params,
        )
        conn.commit()

    def increment_source_failures(self, source_id: str) -> int:
        """Increment consecutive_failures and return new count."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE discovered_sources SET consecutive_failures = consecutive_failures + 1 WHERE source_id=?",
            (source_id,),
        )
        conn.commit()
        row = conn.execute(
            "SELECT consecutive_failures FROM discovered_sources WHERE source_id=?",
            (source_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def reset_source_failures(self, source_id: str) -> None:
        """Reset consecutive failures to 0 on successful fetch."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE discovered_sources SET consecutive_failures=0 WHERE source_id=?",
            (source_id,),
        )
        conn.commit()

    def query_discovered_sources(
        self,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query discovered sources, optionally filtered by status."""
        conn = self._get_conn()
        if status is not None:
            rows = conn.execute(
                "SELECT * FROM discovered_sources WHERE status=? ORDER BY discovered_at",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM discovered_sources ORDER BY discovered_at").fetchall()
        return [self._source_row_to_dict(r) for r in rows]

    @staticmethod
    def _source_row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        for key in (
            "topic_tags_json",
            "probe_result_json",
            "tool_config_json",
            "metadata_json",
        ):
            plain = key.replace("_json", "")
            try:
                d[plain] = json.loads(d.pop(key, "null"))
            except (json.JSONDecodeError, TypeError):
                d[plain] = None
        return d

    # ── Tier 8: unresolved entities ────────────────────────────

    def store_unresolved_entity(
        self,
        raw_text: str,
        source_tool: str,
        context_snippet: str | None = None,
        observed_at: float | None = None,
    ) -> int:
        """Store an unresolved entity mention. Returns row id."""
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO unresolved_entities (raw_text, source_tool, context_snippet, observed_at) VALUES (?, ?, ?, ?)",
            (raw_text, source_tool, context_snippet, observed_at or time.time()),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def query_unresolved_entities(
        self,
        *,
        cluster_id: int | None = None,
        resolved: bool = False,
    ) -> list[dict[str, Any]]:
        """Query unresolved entity mentions."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if not resolved:
            clauses.append("resolved_type IS NULL")
        if cluster_id is not None:
            clauses.append("cluster_id=?")
            params.append(cluster_id)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = conn.execute(
            f"SELECT * FROM unresolved_entities WHERE {where} ORDER BY observed_at",  # noqa: S608
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def update_unresolved_cluster(
        self,
        entity_ids: list[int],
        cluster_id: int,
    ) -> None:
        """Assign cluster_id to a batch of unresolved entities."""
        if not entity_ids:
            return
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in entity_ids)
        conn.execute(
            f"UPDATE unresolved_entities SET cluster_id=? WHERE id IN ({placeholders})",  # noqa: S608
            [cluster_id, *entity_ids],
        )
        conn.commit()

    def resolve_unresolved_entities(
        self,
        cluster_id: int,
        resolved_type: str,
    ) -> int:
        """Mark all entities in a cluster as resolved. Returns count."""
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE unresolved_entities SET resolved_type=?, resolved_at=? "
            "WHERE cluster_id=? AND resolved_type IS NULL",
            (resolved_type, time.time(), cluster_id),
        )
        conn.commit()
        return cur.rowcount

    # ── Tier 8: entity type registry ───────────────────────────

    def register_entity_type(
        self,
        type_name: str,
        *,
        parent_type: str | None = None,
        source: str = "seed",
        confidence: float = 1.0,
        metadata: dict | None = None,
    ) -> bool:
        """Register an entity type. Returns True if newly created."""
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT OR IGNORE INTO entity_type_registry "
            "(type_name, parent_type, discovered_at, source, confidence, active, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, 1, ?)",
            (
                type_name,
                parent_type,
                time.time(),
                source,
                confidence,
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        conn.commit()
        return cur.rowcount > 0

    def query_entity_types(
        self,
        *,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Query registered entity types."""
        conn = self._get_conn()
        if active_only:
            rows = conn.execute("SELECT * FROM entity_type_registry WHERE active=1 ORDER BY type_name").fetchall()
        else:
            rows = conn.execute("SELECT * FROM entity_type_registry ORDER BY type_name").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["metadata"] = json.loads(d.pop("metadata_json", "null"))
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = None
            result.append(d)
        return result

    def deactivate_entity_type(self, type_name: str) -> None:
        """Mark an entity type as inactive (never delete)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE entity_type_registry SET active=0 WHERE type_name=?",
            (type_name,),
        )
        conn.commit()

    def reactivate_entity_type(self, type_name: str) -> None:
        """Re-activate a previously deactivated entity type."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE entity_type_registry SET active=1 WHERE type_name=?",
            (type_name,),
        )
        conn.commit()
