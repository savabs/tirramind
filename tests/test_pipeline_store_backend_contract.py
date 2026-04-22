"""Backend-parity contract tests for PipelineStore.

These tests define behavior that every PipelineStore backend must preserve.
SQLite runs first; PostgreSQL runs when ``TIRRA_TEST_PG_DSN`` is set and
``psycopg2`` is installed — otherwise the postgres tests are skipped.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from agent.pipeline.store import PipelineStore


StoreFactory = Callable[[Path], PipelineStore]


def _sqlite_store_factory(tmp_path: Path) -> PipelineStore:
    return PipelineStore(db_path=tmp_path / "contract.sqlite3")


def _postgres_store_factory(tmp_path: Path) -> PipelineStore:
    """Create a PipelineStore backed by PostgreSQL in an isolated schema."""
    dsn = os.environ.get("TIRRA_TEST_PG_DSN")
    if not dsn:
        pytest.skip("TIRRA_TEST_PG_DSN not set — skipping Postgres tests")

    psycopg2 = pytest.importorskip("psycopg2")

    # Verify the server is reachable.
    try:
        test_conn = psycopg2.connect(dsn)
        test_conn.close()
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL server not reachable")

    from agent.pipeline.storage_backend import PostgresBackend

    schema = f"tirra_test_{uuid.uuid4().hex[:8]}"
    backend = PostgresBackend(dsn, schema=schema)
    store = PipelineStore(backend=backend)

    # Stash cleanup info on the store so the fixture can drop the schema.
    store._test_pg_dsn = dsn  # type: ignore[attr-defined]
    store._test_pg_schema = schema  # type: ignore[attr-defined]
    return store


BACKEND_FACTORIES: dict[str, StoreFactory] = {
    "sqlite": _sqlite_store_factory,
    "postgres": _postgres_store_factory,
}


@pytest.fixture(params=sorted(BACKEND_FACTORIES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> PipelineStore:
    factory = BACKEND_FACTORIES[request.param]
    current_store = factory(tmp_path)
    yield current_store
    current_store.close()

    # Drop the ephemeral test schema if this was a Postgres run.
    pg_dsn = getattr(current_store, "_test_pg_dsn", None)
    pg_schema = getattr(current_store, "_test_pg_schema", None)
    if pg_dsn and pg_schema:
        import psycopg2

        conn = psycopg2.connect(pg_dsn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {pg_schema} CASCADE")
        conn.close()


class TestPipelineStoreBackendContract:
    def test_schema_version_baseline(self, store: PipelineStore) -> None:
        assert store.get_schema_version() == 1

        migrations = store.query_schema_migrations()
        assert len(migrations) == 1
        assert migrations[0]["schema_name"] == "pipeline_store"
        assert migrations[0]["version"] == 1

    def test_run_roundtrip(self, store: PipelineStore) -> None:
        run_id = store.record_run_start("contract_dag", trigger="test")
        store.record_run_end(run_id, "completed", {"step": {"status": "ok"}})

        run = store.get_run(run_id)
        assert run is not None
        assert run["dag_name"] == "contract_dag"
        assert run["status"] == "completed"
        assert run["trigger"] == "test"
        assert run["node_results"] == {"step": {"status": "ok"}}

    def test_pipeline_data_roundtrip(self, store: PipelineStore) -> None:
        row_id = store.store_data("contract_source", {"mode": "test"}, {"value": 7})

        assert isinstance(row_id, int)
        rows = store.query_data("contract_source")
        assert len(rows) == 1
        assert rows[0]["params"] == {"mode": "test"}
        assert rows[0]["data"] == {"value": 7}

    def test_entity_graph_roundtrip(self, store: PipelineStore) -> None:
        entity_a = store.register_entity("company", "Alpha Corp", "ent_alpha")
        entity_b = store.register_entity("company", "Beta Corp", "ent_beta")

        store.add_entity_alias(entity_a, "sec", "0000001")
        observation_id = store.store_entity_observation(
            entity_id=entity_a,
            source_tool="contract_tool",
            observed_at=1700000000.0,
            observation_type="filing_count",
            value={"count": 3},
            depth_level=2,
        )
        link_id = store.link_entities(
            entity_a,
            entity_b,
            "supplier",
            source="contract_test",
        )

        aliases = store.query_entity_aliases(entity_a)
        observations = store.query_entity_observations(entity_a)
        links = store.query_entity_links(entity_a)

        assert aliases[0]["source"] == "sec"
        assert aliases[0]["external_id"] == "0000001"
        assert observation_id > 0
        assert observations[0]["value"] == {"count": 3}
        assert observations[0]["depth_level"] == 2
        assert link_id is not None
        assert any(link["link_type"] == "supplier" for link in links)