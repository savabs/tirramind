"""Tests for PipelineStore (SQLite persistence layer)."""

from __future__ import annotations

import threading
import time
from datetime import UTC

import pytest

from agent.pipeline.store import PipelineStore

# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def store():
    """In-memory PipelineStore for fast tests."""
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def file_store(tmp_path):
    """File-based PipelineStore for persistence tests."""
    db = tmp_path / "test.db"
    s = PipelineStore(db_path=db)
    yield s, db
    s.close()


# ── Schema initialization ─────────────────────────────────────


class TestSchemaInit:
    def test_tables_created(self, store: PipelineStore):
        conn = store._get_conn()
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "dag_runs" in tables
        assert "pipeline_data" in tables
        assert "signals" in tables
        assert "features" in tables
        assert "schema_migrations" in tables

    def test_indexes_created(self, store: PipelineStore):
        conn = store._get_conn()
        indexes = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()]
        assert "idx_pipeline_data_source" in indexes
        assert "idx_signals_name" in indexes
        assert "idx_features_unique" in indexes
        assert "idx_features_lookup" in indexes

    def test_wal_mode(self, file_store):
        store, _ = file_store
        conn = store._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_idempotent_schema_init(self, store: PipelineStore):
        """Calling _init_schema twice should not error."""
        store._init_schema()
        store._init_schema()
        conn = store._get_conn()
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "dag_runs" in tables
        baseline_count = conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE schema_name='pipeline_store' AND version=1"
        ).fetchone()[0]
        assert baseline_count == 1

    def test_baseline_schema_version_recorded(self, store: PipelineStore):
        assert store.get_schema_version() == 1

    def test_query_schema_migrations(self, store: PipelineStore):
        rows = store.query_schema_migrations()
        assert len(rows) == 1
        assert rows[0]["schema_name"] == "pipeline_store"
        assert rows[0]["version"] == 1
        assert "Baseline portable schema" in rows[0]["description"]

    def test_directory_creation(self, tmp_path):
        """Store creates parent directory if it doesn't exist."""
        deep = tmp_path / "a" / "b" / "c" / "test.db"
        s = PipelineStore(db_path=deep)
        assert deep.parent.exists()
        s.close()


# ── Context manager ────────────────────────────────────────────


class TestContextManager:
    def test_enter_exit(self):
        with PipelineStore(db_path=":memory:") as store:
            store.store_data("test", {}, {"a": 1})
        # Connection should be closed after __exit__
        assert store._conn is None

    def test_close_idempotent(self, store: PipelineStore):
        store.close()
        store.close()  # Should not error


# ── DAG runs ───────────────────────────────────────────────────


class TestDagRuns:
    def test_record_run_start(self, store: PipelineStore):
        run_id = store.record_run_start("test_dag", trigger="manual")
        assert isinstance(run_id, str)
        assert len(run_id) == 12

    def test_record_run_start_custom_id(self, store: PipelineStore):
        run_id = store.record_run_start("test_dag", run_id="custom123")
        assert run_id == "custom123"

    def test_get_run(self, store: PipelineStore):
        run_id = store.record_run_start("test_dag", trigger="scheduled")
        run = store.get_run(run_id)
        assert run is not None
        assert run["dag_name"] == "test_dag"
        assert run["status"] == "running"
        assert run["trigger"] == "scheduled"
        assert run["started_at"] > 0
        assert run["finished_at"] is None

    def test_get_run_not_found(self, store: PipelineStore):
        assert store.get_run("nonexistent") is None

    def test_record_run_end(self, store: PipelineStore):
        run_id = store.record_run_start("test_dag")
        node_results = {"fetch_cftc": {"status": "completed"}}
        store.record_run_end(run_id, "completed", node_results)

        run = store.get_run(run_id)
        assert run["status"] == "completed"
        assert run["finished_at"] is not None
        assert run["finished_at"] >= run["started_at"]
        assert run["node_results"] == node_results

    def test_record_run_end_failed(self, store: PipelineStore):
        run_id = store.record_run_start("test_dag")
        store.record_run_end(run_id, "failed", {"error": "timeout"})
        run = store.get_run(run_id)
        assert run["status"] == "failed"

    def test_record_run_end_no_results(self, store: PipelineStore):
        run_id = store.record_run_start("test_dag")
        store.record_run_end(run_id, "completed")
        run = store.get_run(run_id)
        assert run["node_results"] is None

    def test_get_runs_all(self, store: PipelineStore):
        store.record_run_start("dag_a")
        store.record_run_start("dag_b")
        store.record_run_start("dag_a")
        runs = store.get_runs()
        assert len(runs) == 3

    def test_get_runs_filtered(self, store: PipelineStore):
        store.record_run_start("dag_a")
        store.record_run_start("dag_b")
        store.record_run_start("dag_a")
        runs = store.get_runs(dag_name="dag_a")
        assert len(runs) == 2
        assert all(r["dag_name"] == "dag_a" for r in runs)

    def test_get_runs_limit(self, store: PipelineStore):
        for i in range(10):
            store.record_run_start(f"dag_{i}")
        runs = store.get_runs(limit=3)
        assert len(runs) == 3

    def test_get_runs_order_desc(self, store: PipelineStore):
        id1 = store.record_run_start("dag")
        time.sleep(0.01)
        id2 = store.record_run_start("dag")
        runs = store.get_runs()
        assert runs[0]["run_id"] == id2  # Most recent first
        assert runs[1]["run_id"] == id1

    def test_get_runs_empty(self, store: PipelineStore):
        assert store.get_runs() == []


# ── Pipeline data ──────────────────────────────────────────────


class TestPipelineData:
    def test_store_data_returns_id(self, store: PipelineStore):
        row_id = store.store_data("cftc", {"mode": "latest"}, {"positions": [1, 2, 3]})
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_store_and_query_data(self, store: PipelineStore):
        store.store_data("cftc", {"mode": "latest"}, {"net": 1500})
        rows = store.query_data("cftc")
        assert len(rows) == 1
        assert rows[0]["source"] == "cftc"
        assert rows[0]["data"] == {"net": 1500}
        assert rows[0]["params"] == {"mode": "latest"}
        assert rows[0]["fetched_at"] > 0

    def test_query_data_by_source(self, store: PipelineStore):
        store.store_data("cftc", {}, {"a": 1})
        store.store_data("finra", {}, {"b": 2})
        store.store_data("cftc", {}, {"c": 3})

        cftc = store.query_data("cftc")
        assert len(cftc) == 2
        finra = store.query_data("finra")
        assert len(finra) == 1

    def test_query_data_since(self, store: PipelineStore):
        t_before = time.time() - 1
        store.store_data("src", {}, {"old": True})
        t_after = time.time() + 0.01
        rows = store.query_data("src", since=t_after)
        assert len(rows) == 0

        rows = store.query_data("src", since=t_before)
        assert len(rows) == 1

    def test_query_data_until(self, store: PipelineStore):
        store.store_data("src", {}, {"d": 1})
        t_now = time.time() + 1
        rows = store.query_data("src", until=t_now)
        assert len(rows) == 1

        rows = store.query_data("src", until=0.0)
        assert len(rows) == 0

    def test_query_data_since_and_until(self, store: PipelineStore):
        t_start = time.time()
        store.store_data("src", {}, {"d": 1})
        t_end = time.time() + 1
        rows = store.query_data("src", since=t_start, until=t_end)
        assert len(rows) == 1

    def test_query_data_limit(self, store: PipelineStore):
        for i in range(20):
            store.store_data("src", {}, {"i": i})
        rows = store.query_data("src", limit=5)
        assert len(rows) == 5

    def test_query_data_order_desc(self, store: PipelineStore):
        store.store_data("src", {}, {"first": True})
        time.sleep(0.01)
        store.store_data("src", {}, {"second": True})
        rows = store.query_data("src")
        assert rows[0]["data"]["second"] is True  # Most recent first

    def test_query_data_empty(self, store: PipelineStore):
        assert store.query_data("nonexistent") == []

    def test_store_data_complex_types(self, store: PipelineStore):
        """Nested dicts, lists, None values."""
        data = {
            "records": [{"ticker": "AAPL", "volume": 1e8}],
            "metadata": {"date": "2026-03-25", "count": None},
        }
        store.store_data("complex", {"nested": {"a": 1}}, data)
        rows = store.query_data("complex")
        assert rows[0]["data"]["records"][0]["ticker"] == "AAPL"

    def test_store_data_params_sorted(self, store: PipelineStore):
        """Params JSON should be sorted for determinism."""
        store.store_data("src", {"z": 1, "a": 2}, {})
        conn = store._get_conn()
        row = conn.execute("SELECT params_json FROM pipeline_data").fetchone()
        assert row[0] == '{"a": 2, "z": 1}'


# ── Signals ────────────────────────────────────────────────────


class TestSignals:
    def test_store_signal_returns_id(self, store: PipelineStore):
        row_id = store.store_signal("cftc_net_long", 1500.5)
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_store_and_query_signal(self, store: PipelineStore):
        store.store_signal("momentum", 0.85, metadata={"ticker": "AAPL"})
        rows = store.query_signals("momentum")
        assert len(rows) == 1
        assert rows[0]["signal_name"] == "momentum"
        assert rows[0]["value"] == pytest.approx(0.85)
        assert rows[0]["metadata"] == {"ticker": "AAPL"}

    def test_signal_no_metadata(self, store: PipelineStore):
        store.store_signal("simple", 1.0)
        rows = store.query_signals("simple")
        assert rows[0]["metadata"] is None

    def test_query_signals_by_name(self, store: PipelineStore):
        store.store_signal("alpha", 0.5)
        store.store_signal("beta", 0.7)
        store.store_signal("alpha", 0.6)
        rows = store.query_signals("alpha")
        assert len(rows) == 2

    def test_query_signals_since(self, store: PipelineStore):
        store.store_signal("sig", 1.0)
        rows = store.query_signals("sig", since=time.time() + 1)
        assert len(rows) == 0

    def test_query_signals_until(self, store: PipelineStore):
        store.store_signal("sig", 1.0)
        rows = store.query_signals("sig", until=time.time() + 1)
        assert len(rows) == 1

    def test_query_signals_limit(self, store: PipelineStore):
        for i in range(15):
            store.store_signal("sig", float(i))
        rows = store.query_signals("sig", limit=5)
        assert len(rows) == 5

    def test_signal_precision(self, store: PipelineStore):
        """Verify float precision is maintained."""
        store.store_signal("precise", 0.123456789012345)
        rows = store.query_signals("precise")
        assert rows[0]["value"] == pytest.approx(0.123456789012345, abs=1e-12)

    def test_negative_signal(self, store: PipelineStore):
        store.store_signal("drawdown", -0.15)
        rows = store.query_signals("drawdown")
        assert rows[0]["value"] == pytest.approx(-0.15)

    def test_zero_signal(self, store: PipelineStore):
        store.store_signal("flat", 0.0)
        rows = store.query_signals("flat")
        assert rows[0]["value"] == pytest.approx(0.0)


# ── SQL injection prevention ──────────────────────────────────


class TestSQLInjection:
    def test_source_injection(self, store: PipelineStore):
        """Parameterized queries should prevent injection."""
        malicious = "'; DROP TABLE pipeline_data; --"
        store.store_data(malicious, {}, {"safe": True})
        # Table should still exist
        rows = store.query_data(malicious)
        assert len(rows) == 1

    def test_signal_name_injection(self, store: PipelineStore):
        malicious = "'; DROP TABLE signals; --"
        store.store_signal(malicious, 1.0)
        rows = store.query_signals(malicious)
        assert len(rows) == 1

    def test_params_injection(self, store: PipelineStore):
        store.store_data("src", {"key": "'; DROP TABLE pipeline_data; --"}, {})
        rows = store.query_data("src")
        assert len(rows) == 1

    def test_dag_name_injection(self, store: PipelineStore):
        malicious = "'; DROP TABLE dag_runs; --"
        run_id = store.record_run_start(malicious)
        run = store.get_run(run_id)
        assert run["dag_name"] == malicious


# ── Concurrent access ─────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_writes(self, file_store):
        """Multiple threads writing simultaneously should not error."""
        store, db_path = file_store
        errors = []

        def writer(source: str, count: int):
            try:
                # Each thread gets its own store/connection
                s = PipelineStore(db_path=db_path)
                for i in range(count):
                    s.store_data(source, {"i": i}, {"val": i})
                s.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"src_{i}", 20)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent write errors: {errors}"

        # Verify all data written
        s = PipelineStore(db_path=db_path)
        for i in range(5):
            rows = s.query_data(f"src_{i}", limit=100)
            assert len(rows) == 20, f"src_{i} has {len(rows)} rows, expected 20"
        s.close()

    def test_concurrent_read_write(self, file_store):
        """Reading while writing should work with WAL mode."""
        store, db_path = file_store
        # Pre-populate
        for i in range(10):
            store.store_data("pre", {}, {"i": i})

        errors = []
        read_counts = []

        def reader():
            try:
                s = PipelineStore(db_path=db_path)
                rows = s.query_data("pre", limit=100)
                read_counts.append(len(rows))
                s.close()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                s = PipelineStore(db_path=db_path)
                for i in range(10):
                    s.store_data("new", {}, {"i": i})
                s.close()
            except Exception as e:
                errors.append(e)

        t_read = threading.Thread(target=reader)
        t_write = threading.Thread(target=writer)
        t_read.start()
        t_write.start()
        t_read.join(timeout=10)
        t_write.join(timeout=10)

        assert not errors
        assert read_counts[0] == 10  # Reader should see pre-populated data


# ── File persistence ───────────────────────────────────────────


class TestPersistence:
    def test_data_survives_reopen(self, tmp_path):
        db = tmp_path / "persist.db"
        s1 = PipelineStore(db_path=db)
        s1.store_data("src", {}, {"persisted": True})
        s1.store_signal("sig", 42.0)
        run_id = s1.record_run_start("dag")
        s1.record_run_end(run_id, "completed")
        s1.close()

        # Reopen
        s2 = PipelineStore(db_path=db)
        data_rows = s2.query_data("src")
        assert len(data_rows) == 1
        assert data_rows[0]["data"]["persisted"] is True

        sig_rows = s2.query_signals("sig")
        assert len(sig_rows) == 1
        assert sig_rows[0]["value"] == pytest.approx(42.0)

        run = s2.get_run(run_id)
        assert run["status"] == "completed"
        s2.close()


# ── Edge cases ─────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_data_dict(self, store: PipelineStore):
        store.store_data("src", {}, {})
        rows = store.query_data("src")
        assert rows[0]["data"] == {}

    def test_large_data_blob(self, store: PipelineStore):
        """Store a large-ish JSON blob (~100KB)."""
        big = {"items": [{"k": f"v_{i}", "n": i} for i in range(1000)]}
        store.store_data("big", {}, big)
        rows = store.query_data("big")
        assert len(rows[0]["data"]["items"]) == 1000

    def test_unicode_data(self, store: PipelineStore):
        store.store_data("intl", {}, {"name": "日本語テスト", "emoji": "🚀"})
        rows = store.query_data("intl")
        assert rows[0]["data"]["name"] == "日本語テスト"

    def test_new_run_id_unique(self):
        ids = {PipelineStore.new_run_id() for _ in range(100)}
        assert len(ids) == 100  # All unique

    def test_store_data_non_serializable_falls_back(self, store: PipelineStore):
        """datetime objects should serialize via default=str."""
        from datetime import datetime

        data = {"ts": datetime.now(UTC)}
        store.store_data("dt", {}, data)
        rows = store.query_data("dt")
        assert isinstance(rows[0]["data"]["ts"], str)

    def test_query_data_default_limit(self, store: PipelineStore):
        """Default limit is 100."""
        for i in range(150):
            store.store_data("many", {}, {"i": i})
        rows = store.query_data("many")
        assert len(rows) == 100

    def test_query_signals_default_limit(self, store: PipelineStore):
        for i in range(150):
            store.store_signal("many", float(i))
        rows = store.query_signals("many")
        assert len(rows) == 100

    def test_get_runs_default_limit(self, store: PipelineStore):
        for i in range(30):
            store.record_run_start(f"dag_{i}")
        runs = store.get_runs()
        assert len(runs) == 20

    def test_node_results_with_complex_json(self, store: PipelineStore):
        run_id = store.record_run_start("dag")
        results = {
            "node_1": {"status": "completed", "data": [1, 2, 3]},
            "node_2": {"status": "failed", "error": "timeout", "retries": 2},
        }
        store.record_run_end(run_id, "completed", results)
        run = store.get_run(run_id)
        assert run["node_results"]["node_1"]["data"] == [1, 2, 3]
        assert run["node_results"]["node_2"]["retries"] == 2
