"""Tests for DAGExecutor (parallel execution, retry, timeout, failure cascading)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from agent.pipeline.dag import DAG
from agent.pipeline.executor import DAGExecutor, DagRun, NodeResult
from agent.pipeline.store import PipelineStore
from agent.tools.base import Tool, ToolRegistry, ToolResult

# ── Mock tools ─────────────────────────────────────────────────


class EchoTool(Tool):
    """Returns params as data."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="ok", data=kwargs)


class SlowTool(Tool):
    """Sleeps for a configurable duration."""

    @property
    def name(self) -> str:
        return "slow"

    @property
    def description(self) -> str:
        return "Slow"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"sleep": {"type": "number"}}}

    def execute(self, **kwargs: Any) -> ToolResult:
        time.sleep(kwargs.get("sleep", 0.5))
        return ToolResult(success=True, output="done", data={"slept": True})


class FailOnceTool(Tool):
    """Fails on first call, succeeds on retry."""

    def __init__(self):
        self._calls = 0

    @property
    def name(self) -> str:
        return "fail_once"

    @property
    def description(self) -> str:
        return "Fails once"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        self._calls += 1
        if self._calls == 1:
            return ToolResult(success=False, output="First attempt fails")
        return ToolResult(success=True, output="ok", data={"attempt": self._calls})


class AlwaysFailTool(Tool):
    @property
    def name(self) -> str:
        return "always_fail"

    @property
    def description(self) -> str:
        return "Always fails"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=False, output="Always fails")


class CounterTool(Tool):
    """Counts how many times it's been called."""

    def __init__(self):
        self.call_count = 0

    @property
    def name(self) -> str:
        return "counter"

    @property
    def description(self) -> str:
        return "Counter"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        self.call_count += 1
        return ToolResult(success=True, output="ok", data={"count": self.call_count})


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(SlowTool())
    reg.register(FailOnceTool())
    reg.register(AlwaysFailTool())
    reg.register(CounterTool())
    return reg


@pytest.fixture
def store():
    return PipelineStore(db_path=":memory:")


# ── Basic execution ────────────────────────────────────────────


class TestBasicExecution:
    def test_single_node_dag(self, registry, store):
        dag = DAG(name="simple")
        dag.add("n1", operator="echo", params={"key": "val"})
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "completed"
        assert run.success
        assert run.node_results["n1"].status == "completed"
        assert run.node_results["n1"].output == {"key": "val"}

    def test_linear_chain(self, registry, store):
        dag = DAG(name="chain")
        dag.add("a", operator="echo", params={"step": "a"})
        dag.add("b", operator="echo", params={"step": "b"}, depends_on=["a"])
        dag.add("c", operator="echo", params={"step": "c"}, depends_on=["b"])
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "completed"
        for nid in ["a", "b", "c"]:
            assert run.node_results[nid].status == "completed"

    def test_diamond_dag(self, registry, store):
        dag = DAG(name="diamond")
        dag.add("root", operator="echo", params={"r": 1})
        dag.add("left", operator="echo", params={"l": 1}, depends_on=["root"])
        dag.add("right", operator="echo", params={"r": 1}, depends_on=["root"])
        dag.add("sink", operator="echo", params={"s": 1}, depends_on=["left", "right"])
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "completed"
        assert all(nr.status == "completed" for nr in run.node_results.values())

    def test_wide_parallel(self, registry, store):
        """10 independent nodes should all complete."""
        dag = DAG(name="wide")
        for i in range(10):
            dag.add(f"n{i}", operator="echo", params={"i": i})
        executor = DAGExecutor(tool_registry=registry, store=store, max_workers=10)
        run = executor.execute(dag)

        assert run.status == "completed"
        assert len(run.node_results) == 10
        assert all(nr.status == "completed" for nr in run.node_results.values())

    def test_function_operator_in_dag(self, store):
        def compute(params, upstream):
            return sum(params.get("values", []))

        dag = DAG(name="func")
        dag.add("sum", operator=compute, params={"values": [1, 2, 3, 4, 5]})
        executor = DAGExecutor(store=store)
        run = executor.execute(dag)

        assert run.status == "completed"
        assert run.node_results["sum"].output == 15


# ── Parallel execution verification ──────────────────────────


class TestParallelExecution:
    def test_parallel_is_faster_than_sequential(self, registry, store):
        """3 slow nodes (0.2s each) in parallel should take ~0.2s, not ~0.6s."""
        dag = DAG(name="par_speed")
        for i in range(3):
            dag.add(f"s{i}", operator="slow", params={"sleep": 0.2})
        executor = DAGExecutor(tool_registry=registry, store=store, max_workers=3)

        start = time.time()
        run = executor.execute(dag)
        elapsed = time.time() - start

        assert run.status == "completed"
        assert elapsed < 0.8  # With parallelism, should be well under 0.6s + overhead


# ── Failure handling ──────────────────────────────────────────


class TestFailureHandling:
    def test_single_node_failure(self, registry, store):
        dag = DAG(name="fail")
        dag.add("bad", operator="always_fail")
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "failed"
        assert not run.success
        assert run.node_results["bad"].status == "failed"
        assert run.node_results["bad"].error is not None

    def test_failure_cascades_to_dependents(self, registry, store):
        dag = DAG(name="cascade")
        dag.add("fail_node", operator="always_fail")
        dag.add("downstream", operator="echo", depends_on=["fail_node"])
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "failed"
        assert run.node_results["fail_node"].status == "failed"
        assert run.node_results["downstream"].status == "skipped"

    def test_partial_failure(self, registry, store):
        """One branch fails, other branch succeeds."""
        dag = DAG(name="partial")
        dag.add("ok_branch", operator="echo", params={"ok": True})
        dag.add("fail_branch", operator="always_fail")
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "failed"  # Any failure = overall failed
        assert run.node_results["ok_branch"].status == "completed"
        assert run.node_results["fail_branch"].status == "failed"

    def test_deep_cascade(self, registry, store):
        """Failure at root cascades through entire chain."""
        dag = DAG(name="deep_fail")
        dag.add("root", operator="always_fail")
        dag.add("l1", operator="echo", depends_on=["root"])
        dag.add("l2", operator="echo", depends_on=["l1"])
        dag.add("l3", operator="echo", depends_on=["l2"])
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.node_results["root"].status == "failed"
        assert run.node_results["l1"].status == "skipped"
        assert run.node_results["l2"].status == "skipped"
        assert run.node_results["l3"].status == "skipped"

    def test_all_nodes_fail(self, registry, store):
        dag = DAG(name="all_fail")
        dag.add("f1", operator="always_fail")
        dag.add("f2", operator="always_fail")
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "failed"
        assert all(nr.status == "failed" for nr in run.node_results.values())


# ── Retry ──────────────────────────────────────────────────────


class TestRetry:
    def test_retry_succeeds_on_second_attempt(self, registry, store):
        dag = DAG(name="retry")
        dag.add("flaky", operator="fail_once", retries=2)
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "completed"
        nr = run.node_results["flaky"]
        assert nr.status == "completed"
        assert nr.retries_used == 1  # Succeeded on 2nd attempt (0-indexed)

    def test_retry_exhaustion(self, registry, store):
        dag = DAG(name="exhaust")
        dag.add("bad", operator="always_fail", retries=3)
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "failed"
        nr = run.node_results["bad"]
        assert nr.status == "failed"
        assert nr.retries_used == 3

    def test_no_retry_by_default(self, registry, store):
        dag = DAG(name="noretry")
        dag.add("bad", operator="always_fail")  # retries=1 by default
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        nr = run.node_results["bad"]
        assert nr.retries_used == 1


# ── Store integration ──────────────────────────────────────────


class TestStoreIntegration:
    def test_run_recorded_in_store(self, registry, store):
        dag = DAG(name="stored")
        dag.add("n1", operator="echo", params={"a": 1})
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        # Check run metadata in store
        stored_run = store.get_run(run.run_id)
        assert stored_run is not None
        assert stored_run["dag_name"] == "stored"
        assert stored_run["status"] == "completed"
        assert stored_run["finished_at"] is not None

    def test_data_stored_for_completed_nodes(self, registry, store):
        dag = DAG(name="persist")
        dag.add("n1", operator="echo", params={"x": 42})
        executor = DAGExecutor(tool_registry=registry, store=store)
        executor.execute(dag)

        # Check data persisted
        rows = store.query_data("n1")
        assert len(rows) == 1
        assert rows[0]["data"] == {"x": 42}

    def test_custom_table_name(self, registry, store):
        dag = DAG(name="custom")
        dag.add("n1", operator="echo", params={"v": 1}, table_name="my_table")
        executor = DAGExecutor(tool_registry=registry, store=store)
        executor.execute(dag)

        rows = store.query_data("my_table")
        assert len(rows) == 1

    def test_failed_node_not_stored(self, registry, store):
        dag = DAG(name="no_store")
        dag.add("bad", operator="always_fail")
        executor = DAGExecutor(tool_registry=registry, store=store)
        executor.execute(dag)

        rows = store.query_data("bad")
        assert len(rows) == 0

    def test_store_result_false(self, registry, store):
        dag = DAG(name="no_persist")
        dag.add("n1", operator="echo", params={"a": 1}, store_result=False)
        executor = DAGExecutor(tool_registry=registry, store=store)
        executor.execute(dag)

        rows = store.query_data("n1")
        assert len(rows) == 0

    def test_execution_without_store(self, registry):
        """Executor works fine without a store."""
        dag = DAG(name="no_store")
        dag.add("n1", operator="echo", params={"a": 1})
        executor = DAGExecutor(tool_registry=registry)
        run = executor.execute(dag)
        assert run.status == "completed"

    def test_failed_run_recorded(self, registry, store):
        dag = DAG(name="fail_rec")
        dag.add("bad", operator="always_fail")
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        stored = store.get_run(run.run_id)
        assert stored["status"] == "failed"


# ── DagRun / NodeResult ───────────────────────────────────────


class TestDataclasses:
    def test_node_result_to_dict(self):
        nr = NodeResult(
            node_id="test",
            status="completed",
            started_at=1000.0,
            finished_at=1001.0,
            output={"data": True},
            retries_used=0,
        )
        d = nr.to_dict()
        assert d["node_id"] == "test"
        assert d["status"] == "completed"
        assert "output" not in d  # to_dict doesn't include output (could be huge)

    def test_dag_run_success(self):
        run = DagRun(run_id="abc", dag_name="test", status="completed")
        assert run.success is True

    def test_dag_run_failure(self):
        run = DagRun(run_id="abc", dag_name="test", status="failed")
        assert run.success is False

    def test_run_timing(self, registry, store):
        dag = DAG(name="timing")
        dag.add("n1", operator="echo")
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.started_at > 0
        assert run.finished_at is not None
        assert run.finished_at >= run.started_at

    def test_trigger_value(self, registry, store):
        dag = DAG(name="trig")
        dag.add("n1", operator="echo")
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag, trigger="scheduled")
        assert run.trigger == "scheduled"


# ── Upstream data flow ────────────────────────────────────────


class TestUpstreamDataFlow:
    def test_function_receives_upstream(self, registry, store):
        """A function operator can access output from upstream tool nodes."""

        def combine(params, upstream):
            return {"sources": list(upstream.keys()), "count": len(upstream)}

        dag = DAG(name="flow")
        dag.add("fetch", operator="echo", params={"data": "fetched"})
        dag.add("process", operator=combine, depends_on=["fetch"])
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "completed"
        assert run.node_results["process"].output == {"sources": ["fetch"], "count": 1}

    def test_upstream_from_multiple_parents(self, registry, store):
        def merge(params, upstream):
            return {k: v for d in upstream.values() for k, v in (d if isinstance(d, dict) else {}).items()}

        dag = DAG(name="multi")
        dag.add("a", operator="echo", params={"from_a": True})
        dag.add("b", operator="echo", params={"from_b": True})
        dag.add("merge", operator=merge, depends_on=["a", "b"])
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.status == "completed"
        merged = run.node_results["merge"].output
        assert merged["from_a"] is True
        assert merged["from_b"] is True


# ── Edge cases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_invalid_dag_raises(self, registry, store):
        dag = DAG(name="")  # No name
        dag.add("a", operator="echo")
        executor = DAGExecutor(tool_registry=registry, store=store)
        with pytest.raises(ValueError):
            executor.execute(dag)

    def test_empty_dag_raises(self, registry, store):
        dag = DAG(name="empty")
        executor = DAGExecutor(tool_registry=registry, store=store)
        with pytest.raises(ValueError):
            executor.execute(dag)

    def test_max_workers_1(self, registry, store):
        """Serial execution with max_workers=1."""
        dag = DAG(name="serial")
        for i in range(5):
            dag.add(f"n{i}", operator="echo", params={"i": i})
        executor = DAGExecutor(tool_registry=registry, store=store, max_workers=1)
        run = executor.execute(dag)
        assert run.status == "completed"
        assert all(nr.status == "completed" for nr in run.node_results.values())

    def test_node_result_has_timing(self, registry, store):
        dag = DAG(name="time")
        dag.add("n1", operator="echo")
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        nr = run.node_results["n1"]
        assert nr.started_at is not None
        assert nr.finished_at is not None
        assert nr.finished_at >= nr.started_at


# ── Timeout reconciliation ──────────────────────────────────────
#
# A node with timeout=1 whose operator actually takes ~2s used to be
# reported "failed: timed out" even though the operator kept running
# in its worker thread, finished, and its DB write landed — the run
# record was left showing a false failure forever. These verify the
# fix: the node's *initial* result is still "failed" (execute() must
# return promptly, bounded by the timeout, not block for the full
# operator duration), but once the background thread actually
# finishes, `_reconcile_timeout` corrects the NodeResult in place and
# re-persists the dag_runs row.


class TestTimeoutReconciliation:
    def test_execute_returns_promptly_not_after_full_duration(self, registry, store):
        """execute() must not block for the operator's full ~2s runtime
        just because its timeout (1s) was exceeded — that would defeat
        the whole point of a node timeout."""
        dag = DAG(name="slow_bounded")
        dag.add("slow", operator="slow", params={"sleep": 2.0}, timeout=1, retries=1)
        executor = DAGExecutor(tool_registry=registry, store=store)

        started = time.time()
        run = executor.execute(dag)
        elapsed = time.time() - started

        assert elapsed < 1.9, f"execute() blocked for {elapsed:.2f}s; timeout should have bounded it near 1s"
        assert run.node_results["slow"].status == "failed"
        assert "timed out" in (run.node_results["slow"].error or "")

    def test_late_success_reconciles_to_completed_and_stores_output(self, registry, store):
        """Once the timed-out operator actually finishes and succeeds, its
        NodeResult flips from 'failed' to 'completed' and its output is
        persisted — the write is not silently dropped nor redone."""
        dag = DAG(name="slow_reconcile")
        dag.add(
            "slow",
            operator="slow",
            params={"sleep": 1.5},
            timeout=1,
            retries=1,
            table_name="slow_output",
        )
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        # Transient state immediately after execute() returns.
        assert run.node_results["slow"].status == "failed"

        # Give the background thread (operator sleeps 1.5s, timeout was 1s)
        # time to finish and for its done-callback to fire.
        deadline = time.time() + 5.0
        while run.node_results["slow"].status != "completed" and time.time() < deadline:
            time.sleep(0.05)

        nr = run.node_results["slow"]
        assert nr.status == "completed", "late success was never reconciled"
        assert nr.error is None
        assert nr.output == {"slept": True}

        # The late write actually landed in the store...
        rows = store.query_data("slow_output", limit=10)
        assert len(rows) == 1
        assert rows[0]["data"] == {"slept": True}

        # ...and the persisted dag_runs record was corrected too, not left
        # showing the transient false failure forever.
        persisted = store.get_run(run.run_id)
        assert persisted["status"] == "completed"

    def test_downstream_of_late_success_was_skipped_not_rerun(self, registry, store):
        """Documents the known, accepted limitation: downstream nodes are
        evaluated synchronously right after the timeout fires, before the
        slow node's reconciliation can land, so they are skipped rather
        than retroactively executed. The fix only corrects the false
        failure report for the timed-out node itself."""
        dag = DAG(name="slow_then_downstream")
        dag.add("slow", operator="slow", params={"sleep": 1.5}, timeout=1, retries=1)
        dag.add("downstream", operator="echo", depends_on=["slow"])
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert run.node_results["downstream"].status == "skipped"

        deadline = time.time() + 5.0
        while run.node_results["slow"].status != "completed" and time.time() < deadline:
            time.sleep(0.05)
        assert run.node_results["slow"].status == "completed"
        # downstream is not retroactively executed by reconciliation.
        assert run.node_results["downstream"].status == "skipped"
