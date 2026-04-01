"""
Tests for DAGRegistry + daily_collection DAG (Step 7.7).

Covers: registry CRUD, validation, load_defaults, daily_collection DAG
structure (nodes, schedule, params, topo sort, independence).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.pipeline.dag import DAG
from agent.pipeline.registry import DAGRegistry
from agent.pipeline.dags.daily_collection import build_daily_collection_dag


# ── Helpers ───────────────────────────────────────────────────


def _make_dag(name: str = "test", schedule: str | None = None) -> DAG:
    dag = DAG(name=name, schedule=schedule, description=f"Test: {name}")
    dag.add("n1", operator=lambda p, u: "ok", params={})
    return dag


# ── DAGRegistry CRUD ──────────────────────────────────────────


class TestRegistryCRUD:
    """Basic register/get/list/remove operations."""

    def test_register_and_get(self):
        r = DAGRegistry()
        dag = _make_dag("alpha")
        r.register(dag)
        assert r.get("alpha") is dag

    def test_get_missing_returns_none(self):
        r = DAGRegistry()
        assert r.get("nope") is None

    def test_list_all_empty(self):
        r = DAGRegistry()
        assert r.list_all() == []

    def test_list_all_multiple(self):
        r = DAGRegistry()
        for name in ["charlie", "alpha", "bravo"]:
            r.register(_make_dag(name))
        names = [d.name for d in r.list_all()]
        assert names == ["alpha", "bravo", "charlie"]  # Sorted

    def test_list_names(self):
        r = DAGRegistry()
        r.register(_make_dag("b"))
        r.register(_make_dag("a"))
        assert r.list_names() == ["a", "b"]

    def test_len(self):
        r = DAGRegistry()
        assert len(r) == 0
        r.register(_make_dag("x"))
        assert len(r) == 1
        r.register(_make_dag("y"))
        assert len(r) == 2

    def test_remove_existing(self):
        r = DAGRegistry()
        r.register(_make_dag("d"))
        assert r.remove("d") is True
        assert r.get("d") is None
        assert len(r) == 0

    def test_remove_missing(self):
        r = DAGRegistry()
        assert r.remove("ghost") is False

    def test_overwrite_existing(self):
        r = DAGRegistry()
        r.register(_make_dag("d"))
        dag2 = _make_dag("d")
        r.register(dag2)
        assert r.get("d") is dag2
        assert len(r) == 1


# ── Registry Validation ──────────────────────────────────────


class TestRegistryValidation:
    """Validation behavior on register."""

    def test_reject_invalid_dag(self):
        r = DAGRegistry()
        bad = DAG(name="empty")  # No nodes
        with pytest.raises(ValueError, match="Invalid DAG"):
            r.register(bad)

    def test_reject_dag_with_cycle(self):
        r = DAGRegistry()
        dag = DAG(name="cyclic")
        dag.add("a", operator="x", depends_on=["b"])
        dag.add("b", operator="x", depends_on=["a"])
        with pytest.raises(ValueError, match="Invalid DAG"):
            r.register(dag)

    def test_accept_valid_dag(self):
        r = DAGRegistry()
        r.register(_make_dag())
        assert len(r) == 1

    def test_invalid_dag_not_stored(self):
        r = DAGRegistry()
        try:
            r.register(DAG(name="bad"))
        except ValueError:
            pass
        assert len(r) == 0


# ── load_defaults ─────────────────────────────────────────────


class TestLoadDefaults:
    """Tests for load_defaults() integration with daily_collection."""

    def test_load_defaults_registers_daily_collection(self):
        r = DAGRegistry()
        mock_registry = MagicMock()
        r.load_defaults(mock_registry)
        assert r.get("daily_collection") is not None

    def test_load_defaults_count(self):
        r = DAGRegistry()
        r.load_defaults(MagicMock())
        assert len(r) >= 1  # At least daily_collection

    def test_load_defaults_idempotent(self):
        r = DAGRegistry()
        r.load_defaults(MagicMock())
        count1 = len(r)
        r.load_defaults(MagicMock())  # Overwrites, doesn't duplicate
        assert len(r) == count1

    def test_daily_collection_is_valid_after_load(self):
        r = DAGRegistry()
        r.load_defaults(MagicMock())
        dag = r.get("daily_collection")
        assert dag.validate() == []


# ── daily_collection DAG Structure ────────────────────────────


class TestDailyCollectionStructure:
    """Tests for the daily_collection DAG definition."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    def test_name(self, dag):
        assert dag.name == "daily_collection"

    def test_schedule_weekday_6pm_utc(self, dag):
        assert dag.schedule == "0 18 * * 1-5"

    def test_has_description(self, dag):
        assert dag.description  # Non-empty

    def test_validates_clean(self, dag):
        assert dag.validate() == []

    def test_node_count(self, dag):
        assert len(dag.nodes) == 6

    def test_expected_node_ids(self, dag):
        expected = {
            "fetch_cftc",
            "fetch_finra_scan",
            "fetch_power_demand",
            "fetch_power_fuel",
            "fetch_gdelt",
            "fetch_polymarket",
        }
        assert set(dag.nodes.keys()) == expected

    def test_all_nodes_independent(self, dag):
        """No dependencies between nodes — single parallel layer."""
        for node in dag.nodes.values():
            assert node.depends_on == [], f"Node {node.id} has deps: {node.depends_on}"

    def test_single_parallel_layer(self, dag):
        layers = dag.topo_sort()
        assert len(layers) == 1
        assert len(layers[0]) == 6

    def test_all_roots(self, dag):
        assert len(dag.roots()) == 6


class TestDailyCollectionNodes:
    """Tests for individual node configurations."""

    @pytest.fixture
    def dag(self):
        return build_daily_collection_dag()

    def test_cftc_node(self, dag):
        n = dag.nodes["fetch_cftc"]
        assert n.operator == "cftc"
        assert n.params == {"mode": "latest"}
        assert n.retries >= 2
        assert n.timeout >= 60

    def test_finra_scan_node(self, dag):
        n = dag.nodes["fetch_finra_scan"]
        assert n.operator == "finra_short_volume"
        assert n.params["mode"] == "short_volume"
        assert "ticker" not in n.params  # scan mode: no ticker
        assert n.retries >= 2

    def test_power_demand_node(self, dag):
        n = dag.nodes["fetch_power_demand"]
        assert n.operator == "power_grid"
        assert n.params == {"mode": "demand"}

    def test_power_fuel_node(self, dag):
        n = dag.nodes["fetch_power_fuel"]
        assert n.operator == "power_grid"
        assert n.params == {"mode": "fuel_mix"}

    def test_gdelt_node(self, dag):
        n = dag.nodes["fetch_gdelt"]
        assert n.operator == "gdelt"
        assert n.params["mode"] == "events"
        assert n.params["hours_back"] == 24
        assert n.params["limit"] >= 100

    def test_polymarket_node(self, dag):
        n = dag.nodes["fetch_polymarket"]
        assert n.operator == "polymarket"
        assert n.params["category"] == "all"
        assert n.params["limit"] >= 50

    def test_all_nodes_use_tool_operators(self, dag):
        """All nodes reference tools by string name (not callables)."""
        for node in dag.nodes.values():
            assert isinstance(node.operator, str), f"{node.id} operator is not a str"

    def test_all_nodes_store_results(self, dag):
        for node in dag.nodes.values():
            assert node.store_result is True, f"{node.id} should store results"

    def test_all_timeouts_positive(self, dag):
        for node in dag.nodes.values():
            assert node.timeout > 0, f"{node.id} has non-positive timeout"

    def test_all_retries_at_least_one(self, dag):
        for node in dag.nodes.values():
            assert node.retries >= 1, f"{node.id} has zero retries"


# ── Integration: Registry + Scheduler ─────────────────────────


class TestRegistrySchedulerIntegration:
    """Verify DAGRegistry satisfies PipelineScheduler's DAGProvider protocol."""

    def test_registry_has_get_and_list_all(self):
        r = DAGRegistry()
        assert hasattr(r, "get")
        assert hasattr(r, "list_all")
        assert callable(r.get)
        assert callable(r.list_all)

    def test_registry_works_with_scheduler(self):
        from agent.pipeline.scheduler import PipelineScheduler
        from agent.pipeline.executor import DAGExecutor

        r = DAGRegistry()
        r.load_defaults(MagicMock())
        executor = MagicMock(spec=DAGExecutor)

        # This should not raise — registry satisfies DAGProvider
        scheduler = PipelineScheduler(executor=executor, registry=r)
        assert len(scheduler.list_dags()) >= 1

    def test_scheduler_registers_daily_collection_cron(self):
        from agent.pipeline.scheduler import PipelineScheduler
        from agent.pipeline.executor import DAGExecutor

        r = DAGRegistry()
        r.load_defaults(MagicMock())
        executor = MagicMock(spec=DAGExecutor)

        scheduler = PipelineScheduler(executor=executor, registry=r)
        scheduler.start(blocking=False)
        try:
            jobs = scheduler._scheduler.get_jobs()
            job_ids = {j.id for j in jobs}
            assert "daily_collection" in job_ids
        finally:
            scheduler.stop()
