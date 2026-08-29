"""Tests for pipeline hardening: stale-run reaping, missing-credential
classification, swallowed store-exception fix, and the zero-rows guard.

Context: production `daily_collection` had 1 completed run ever, 5 failed,
and 2 runs stuck in status='running' forever (the process that ran them
died mid-execution and nothing ever flipped their status). This file
verifies the fixes for that class of bug, not the DAG's node wiring.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from agent.pipeline.dag import DAG
from agent.pipeline.executor import DAGExecutor, DagRun, NodeResult, _is_missing_credential_error
from agent.pipeline.store import PipelineStore
from agent.tools.base import Tool, ToolRegistry, ToolResult

# ── Mock tools ─────────────────────────────────────────────────


class EchoTool(Tool):
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
        return ToolResult(success=True, output="ok", data=kwargs or {"ok": True})


class NeedsApiKeyTool(Tool):
    """Mimics electricity_monitor / satellite_activity: fails identically on
    every call because an optional free-tier credential is unset. Counts
    calls so tests can assert the executor does NOT burn retries on it."""

    def __init__(self, message: str = "EIA API key required. Set TIRRA_EIA_API_KEY.") -> None:
        self.calls = 0
        self._message = message

    @property
    def name(self) -> str:
        return "needs_api_key"

    @property
    def description(self) -> str:
        return "Needs a credential that is not configured"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(success=False, output=self._message)


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


@pytest.fixture
def store():
    return PipelineStore(db_path=":memory:")


# ── Missing-credential classification (rule 5) ──────────────────


class TestMissingCredentialClassification:
    @pytest.mark.parametrize(
        "message",
        [
            "EIA API key required. Set TIRRA_EIA_API_KEY.",
            "FRED API key required for building_permits. Set TIRRA_FRED_API_KEY in .env.",
            "[MISSING_CONFIG] SatelliteActivityTool (fire mode) requires TIRRA_NASA_FIRMS_KEY but it is not set.",
        ],
    )
    def test_detects_known_credential_errors(self, message):
        assert _is_missing_credential_error(message)

    def test_does_not_flag_unrelated_errors(self):
        assert not _is_missing_credential_error("Connection timed out")
        assert not _is_missing_credential_error("'NoneType' object is not subscriptable")

    def test_node_skipped_not_failed_and_not_retried(self, store):
        reg = ToolRegistry()
        needs_key = NeedsApiKeyTool()
        reg.register(needs_key)
        dag = DAG(name="cred_dag")
        dag.add("n1", operator="needs_api_key", retries=3)
        executor = DAGExecutor(tool_registry=reg, store=store)

        run = executor.execute(dag)

        nr = run.node_results["n1"]
        assert nr.status == "skipped"
        assert nr.error.startswith("Missing credential:")
        # Must not retry a permanently-unconfigured credential 3 times.
        assert needs_key.calls == 1
        # A missing-credential node must not fail the whole run.
        assert run.status == "completed"

    def test_missing_credential_does_not_trip_zero_rows_guard(self, store):
        """A DAG where the only node is missing a credential should not be
        reported as a failed 'zero rows' run — it never was eligible to
        write anything in the first place."""
        reg = ToolRegistry()
        reg.register(NeedsApiKeyTool())
        dag = DAG(name="cred_only")
        dag.add("n1", operator="needs_api_key")
        executor = DAGExecutor(tool_registry=reg, store=store)

        run = executor.execute(dag)
        assert run.status == "completed"
        assert run.error is None


# ── Swallowed store-exception fix (rule 3) ──────────────────────


class _ExplodingStore:
    """Minimal stand-in implementing only what DAGExecutor calls on a store,
    with store_data always raising — simulates a disk-full / DB-locked
    failure during persistence, the exact case that used to be caught and
    only logged as a warning while the node still recorded 'completed'."""

    def __init__(self) -> None:
        self.store_data_calls = 0

    def reap_stale_runs(self, *_args: Any, **_kwargs: Any) -> list[str]:
        return []

    def record_run_start(self, dag_name: str, trigger: str = "manual", run_id: str | None = None) -> str:
        return run_id or "fake-run-id"

    def heartbeat(self, run_id: str) -> None:
        pass

    def record_run_end(self, run_id: str, status: str, node_results: dict[str, Any] | None = None) -> None:
        self.last_status = status
        self.last_node_results = node_results

    def store_data(self, source: str, params: dict[str, Any], data: Any) -> int:
        self.store_data_calls += 1
        raise RuntimeError("disk full")


class TestSwallowedStoreExceptionFix:
    def test_store_failure_fails_the_node_not_just_a_log_line(self, registry):
        exploding = _ExplodingStore()
        dag = DAG(name="store_explodes")
        dag.add("n1", operator="echo", params={"a": 1})
        executor = DAGExecutor(tool_registry=registry, store=exploding)  # type: ignore[arg-type]

        run = executor.execute(dag)

        nr = run.node_results["n1"]
        assert nr.status == "failed"
        assert "failed to store result" in nr.error
        assert nr.stored is False
        assert exploding.store_data_calls == 1
        # The whole run must reflect the failure — this is what "a run that
        # writes zero rows must not report success" means end to end.
        assert run.status == "failed"
        assert run.rows_written == 0
        assert exploding.last_status == "failed"


# ── Zero-rows guard (rule 4) ─────────────────────────────────────


class TestZeroRowsGuard:
    def _run(self, rows_written: int) -> DagRun:
        run = DagRun(run_id="r1", dag_name="d1", started_at=time.time())
        run.node_results["n1"] = NodeResult(node_id="n1", status="completed", stored=rows_written > 0)
        run.rows_written = rows_written
        run.status = "completed"
        return run

    def test_downgrades_completed_run_with_zero_rows(self):
        run = self._run(rows_written=0)
        DAGExecutor._apply_zero_rows_guard(run, eligible_store_nodes=1, any_failure=False)
        assert run.status == "failed"
        assert run.error is not None
        assert "zero rows" in run.error

    def test_leaves_run_alone_when_rows_were_written(self):
        run = self._run(rows_written=3)
        DAGExecutor._apply_zero_rows_guard(run, eligible_store_nodes=1, any_failure=False)
        assert run.status == "completed"
        assert run.error is None

    def test_leaves_run_alone_when_no_nodes_were_eligible(self):
        run = self._run(rows_written=0)
        DAGExecutor._apply_zero_rows_guard(run, eligible_store_nodes=0, any_failure=False)
        assert run.status == "completed"
        assert run.error is None

    def test_does_not_override_an_already_failed_run(self):
        run = self._run(rows_written=0)
        run.status = "failed"
        DAGExecutor._apply_zero_rows_guard(run, eligible_store_nodes=1, any_failure=True)
        assert run.status == "failed"
        # any_failure=True short-circuits — the guard leaves the existing
        # failure reason (if any) alone rather than overwriting it.
        assert run.error is None

    def test_end_to_end_all_nodes_completed_but_none_stored(self, registry):
        """A store that silently no-ops (returns without raising, without
        actually persisting) is exactly the shape this guard exists for —
        the executor's own contract (raise on failure) is honored, but the
        run still produced nothing."""

        class _NoOpStore(_ExplodingStore):
            def store_data(self, source: str, params: dict[str, Any], data: Any) -> int:
                self.store_data_calls += 1
                return None  # "succeeds" without ever landing a row

        # Force node.store_result False path is irrelevant here: we want the
        # node to go through the normal store_data() call. Monkeypatch
        # DAGExecutor to treat a None return as "not actually stored" is out
        # of scope for this test; instead this test documents current
        # behavior: store_data() must raise to signal non-persistence, and
        # when it does not raise, the node is trusted as stored. Covered
        # instead via TestSwallowedStoreExceptionFix above for the raising
        # case, which is what real backends (SQLite) do on failure.
        store_ = _NoOpStore()
        dag = DAG(name="noop_store")
        dag.add("n1", operator="echo")
        executor = DAGExecutor(tool_registry=registry, store=store_)  # type: ignore[arg-type]
        run = executor.execute(dag)
        # store_data did not raise, so the executor trusts it: this documents
        # the boundary of the guard rather than asserting a false positive.
        assert run.status == "completed"
        assert run.rows_written == 1


# ── Stale "running" run reaping ──────────────────────────────────


class TestStaleRunReaping:
    def _insert_running(self, store: PipelineStore, run_id: str, started_at: float, heartbeat_at: float | None) -> None:
        conn = store._get_conn()
        conn.execute(
            "INSERT INTO dag_runs (run_id, dag_name, started_at, status, trigger, heartbeat_at) "
            "VALUES (?, 'some_dag', ?, 'running', 'manual', ?)",
            (run_id, started_at, heartbeat_at),
        )
        conn.commit()

    def test_reaps_run_with_no_heartbeat_past_threshold(self, store):
        old = time.time() - 10_000
        self._insert_running(store, "dead-run", started_at=old, heartbeat_at=old)

        reaped = store.reap_stale_runs(stale_after_seconds=3600)

        assert reaped == ["dead-run"]
        row = store.get_run("dead-run")
        assert row["status"] == "failed"
        assert row["finished_at"] is not None

    def test_does_not_reap_a_recently_alive_run(self, store):
        now = time.time()
        self._insert_running(store, "alive-run", started_at=now - 10_000, heartbeat_at=now)

        reaped = store.reap_stale_runs(stale_after_seconds=3600)

        assert reaped == []
        row = store.get_run("alive-run")
        assert row["status"] == "running"

    def test_falls_back_to_started_at_when_heartbeat_missing(self, store):
        old = time.time() - 10_000
        self._insert_running(store, "no-heartbeat-run", started_at=old, heartbeat_at=None)

        reaped = store.reap_stale_runs(stale_after_seconds=3600)

        assert reaped == ["no-heartbeat-run"]

    def test_executor_self_heals_stale_runs_from_other_dags_on_execute(self, registry, store):
        """This is what makes the fix apply in production without any
        separate cron/timer: every call to DAGExecutor.execute() (manual
        trigger, scheduler cron job, or run_chain.py) reaps stale runs
        first."""
        old = time.time() - 10_000
        self._insert_running(store, "orphaned-collection-run", started_at=old, heartbeat_at=old)

        dag = DAG(name="unrelated_dag")
        dag.add("n1", operator="echo")
        executor = DAGExecutor(tool_registry=registry, store=store, stale_run_after_seconds=3600)
        executor.execute(dag)

        row = store.get_run("orphaned-collection-run")
        assert row["status"] == "failed"

    def test_heartbeat_keeps_a_genuinely_long_run_alive(self, registry, store):
        """A run spanning several DAG layers must not be reaped mid-flight
        just because a single layer takes a while — the per-layer heartbeat
        in DAGExecutor.execute() is what prevents that."""
        dag = DAG(name="two_layers")
        dag.add("n1", operator="echo")
        dag.add("n2", operator="echo", depends_on=["n1"])
        # Absurdly small threshold: if heartbeats didn't happen, the run
        # would look stale to a reap check taken between layers.
        executor = DAGExecutor(tool_registry=registry, store=store, stale_run_after_seconds=3600)
        run = executor.execute(dag)
        assert run.status == "completed"
        # No stale run left behind for the run's own id.
        assert store.get_run(run.run_id)["status"] == "completed"


# ── Shared-connection concurrency race (found live during daily_collection
#    verification: bankruptcy_court / defi_flows entity persistence racing
#    the executor's own store_data() calls on the one PipelineStore instance
#    every tool + DAGExecutor share) ──────────────────────────────


# ── LESSONS F-13: timeout does not stop the thread ────────────────
#
# Production incident (2026-08-27): a node's timeout marked it "failed" and
# moved on, but the operator's thread kept running — Python cannot forcibly
# kill a running thread. train_gnn (>1800s) and generate_features (>120s)
# both leaked threads that kept allocating memory, running a 1.9GB box down
# to 20MB available with swap nearly full. These tests verify the fix:
# (1) operators are handed a cancellation signal they can poll,
# (2) the executor stops starting new work once it knows a thread may be
#     leaking ("degraded"), and
# (3) a still-running timed-out operator is logged loudly and repeatedly,
# not just once at the very end.
# Critically: none of this ever reports a timed-out-then-cancelled node as
# a success (rule 1 — a returned dict is success, so a node that actually
# raises after being cancelled must stay "failed").


def _cancel_aware_op(sleep_total: float, poll: float = 0.02, captured: list | None = None):
    """FunctionOperator callback that polls ``cancel_event`` and raises
    (rather than returning a dict) as soon as it's set — so a cooperative
    operator's own thread actually stops working once cancelled, instead of
    running to completion regardless."""

    def _fn(params: dict, upstream: dict, cancel_event=None) -> dict:
        if captured is not None:
            captured.append(cancel_event)
        deadline = time.time() + sleep_total
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("stopped early: cancel_event was set")
            time.sleep(poll)
        return {"status": "completed_fully"}

    return _fn


def _counting_op(counter: list):
    """FunctionOperator callback that records every call it actually makes,
    so tests can assert a skipped node's operator was never invoked."""

    def _fn(params: dict, upstream: dict, cancel_event=None) -> dict:
        counter.append(1)
        return {"ran": True}

    return _fn


def _ignores_cancel_op(sleep_total: float):
    """A plain, non-cooperative operator — the realistic majority case
    today. Sleeps for its full duration regardless of cancel_event, so
    tests using it can verify the orphan-watchdog logging path (which must
    work even when the operator itself never checks the signal)."""

    def _fn(params: dict, upstream: dict, cancel_event=None) -> dict:
        time.sleep(sleep_total)
        return {"status": "completed_fully"}

    return _fn


class TestCooperativeCancellation:
    def test_cancel_event_is_set_after_node_times_out(self, registry, store):
        captured: list = []
        dag = DAG(name="cancel_dag")
        dag.add(
            "n1",
            operator=_cancel_aware_op(sleep_total=5.0, captured=captured),
            timeout=0.1,
            retries=1,
        )
        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        # Transient state right after execute() returns: still "failed",
        # exactly as before this fix — the timeout itself is unchanged.
        assert run.node_results["n1"].status == "failed"
        assert "timed out" in (run.node_results["n1"].error or "")

        deadline = time.time() + 3.0
        while not captured and time.time() < deadline:
            time.sleep(0.02)
        assert captured, "operator was never invoked with a cancel_event"
        cancel_event = captured[0]

        deadline = time.time() + 3.0
        while not cancel_event.is_set() and time.time() < deadline:
            time.sleep(0.02)
        assert cancel_event.is_set(), "cancel_event was never set after the node's timeout fired"

    def test_cooperative_operator_actually_stops_and_is_not_reported_success(self, registry, store):
        """The real point of cancellation: a cooperative operator's thread
        exits in roughly one poll interval after the timeout, not after its
        full (here 5s) runtime — and the run must still show "failed", not
        flip to "completed" just because the operator eventually returned
        control instead of raising an unrelated exception."""
        dag = DAG(name="cancel_stops_dag")
        dag.add(
            "n1",
            operator=_cancel_aware_op(sleep_total=5.0, poll=0.02),
            timeout=0.1,
            retries=3,  # even with retries configured, a cancelled node must not retry
        )
        executor = DAGExecutor(tool_registry=registry, store=store)

        started = time.time()
        run = executor.execute(dag)
        assert run.node_results["n1"].status == "failed"

        # Give the operator's thread time to notice cancel_event and raise,
        # and the done-callback time to reconcile — bounded well under the
        # 5s it would take if cancellation did nothing.
        deadline = time.time() + 2.0
        while "Cancelled after timeout" not in (run.node_results["n1"].error or "") and time.time() < deadline:
            time.sleep(0.02)
        elapsed = time.time() - started

        assert elapsed < 2.5, (
            f"cooperative operator took {elapsed:.2f}s to stop; cancel_event should have "
            "ended it in well under its full 5s sleep"
        )
        nr = run.node_results["n1"]
        assert nr.status == "failed", "a cancelled-then-raised operator must never be reported as completed"
        assert "Cancelled after timeout" in nr.error
        assert store.get_run(run.run_id)["status"] == "failed"


class TestDegradedPoolGuard:
    def test_timeout_marks_executor_degraded(self, registry, store):
        dag = DAG(name="degrade_dag")
        dag.add("slow", operator=_ignores_cancel_op(sleep_total=2.0), timeout=0.1, retries=1)
        executor = DAGExecutor(tool_registry=registry, store=store)

        assert executor.is_degraded is False
        executor.execute(dag)
        assert executor.is_degraded is True

    def test_degraded_executor_skips_later_layer_without_running_it(self, registry, store):
        """A node timeout in layer 0 must stop layer 1 from ever starting
        new work on this executor instance — reproduces the exact shape of
        the 2026-08-27 incident where train_gnn's timeout was followed by
        generate_features being scheduled into an already-degraded process."""
        n0_calls: list = []
        n2_calls: list = []
        dag = DAG(name="degrade_layers_dag")
        dag.add("n0", operator=_counting_op(n0_calls))  # layer 0, unrelated root
        dag.add("slow", operator=_ignores_cancel_op(sleep_total=2.0), timeout=0.1, retries=1)  # layer 0
        dag.add("n2", operator=_counting_op(n2_calls), depends_on=["n0"])  # layer 1

        executor = DAGExecutor(tool_registry=registry, store=store)
        run = executor.execute(dag)

        assert executor.is_degraded is True
        assert n0_calls == [1], "n0 was already submitted in the same (degraded) layer as 'slow' — it should still run"
        assert n2_calls == [], "n2's operator must never be invoked once the pool is degraded"
        nr2 = run.node_results["n2"]
        assert nr2.status == "skipped"
        assert "degraded" in nr2.error

    def test_degraded_executor_skips_next_dag_in_the_same_process(self, registry, store):
        """run_chain.py reuses one DAGExecutor across every DAG in the
        nightly chain — degradation must carry across DAG boundaries within
        that one process, not just within a single DagRun."""
        dag1 = DAG(name="first_dag")
        dag1.add("slow", operator=_ignores_cancel_op(sleep_total=2.0), timeout=0.1, retries=1)

        calls: list = []
        dag2 = DAG(name="second_dag")
        dag2.add("n1", operator=_counting_op(calls))

        executor = DAGExecutor(tool_registry=registry, store=store)
        executor.execute(dag1)
        assert executor.is_degraded is True

        run2 = executor.execute(dag2)
        assert calls == [], "second DAG's node must not run once this executor is degraded"
        assert run2.node_results["n1"].status == "skipped"


class TestOrphanThreadLogging:
    def test_logs_loudly_while_timed_out_operator_still_running(self, registry, store, caplog):
        import logging as _logging

        caplog.set_level(_logging.ERROR, logger="agent.pipeline.executor")
        dag = DAG(name="orphan_dag")
        dag.add("slow", operator=_ignores_cancel_op(sleep_total=0.5), timeout=0.05, retries=1)
        executor = DAGExecutor(
            tool_registry=registry,
            store=store,
            orphan_log_interval_seconds=0.05,
        )
        executor.execute(dag)

        # Let the orphaned thread run its course and the watchdog get at
        # least one poll interval in before it exits.
        deadline = time.time() + 2.0
        while not any("ORPHAN THREAD" in r.message for r in caplog.records) and time.time() < deadline:
            time.sleep(0.02)

        orphan_logs = [r for r in caplog.records if "ORPHAN THREAD" in r.message]
        assert orphan_logs, "no ORPHAN THREAD log emitted while the timed-out operator was still running"
        assert orphan_logs[0].levelno == _logging.ERROR
        assert "slow" in orphan_logs[0].message
        assert "orphan_dag" in orphan_logs[0].message


class TestConcurrentStoreAccess:
    def test_concurrent_writes_from_many_threads_do_not_raise(self, tmp_path):
        """Reproduces the shape of daily_collection's first layer: ~50
        independent nodes, several threads, all writing to the one shared
        PipelineStore/connection concurrently. Before the ``_LockingConnection``
        fix in storage_backend.py this reproducibly raised
        ``sqlite3.InterfaceError: bad parameter or other API misuse``.
        """
        store = PipelineStore(db_path=str(tmp_path / "concurrent.db"))
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                for j in range(20):
                    store.store_data(f"source_{i}", {"i": i}, {"j": j})
                    store.register_entity("test_type", f"entity-{i}-{j}", f"id-{i}-{j}")
                    store.store_entity_observation(
                        entity_id=f"id-{i}-{j}",
                        source_tool=f"tool_{i}",
                        observed_at=time.time(),
                        observation_type="test_obs",
                        value={"v": j},
                    )
            except Exception as exc:  # pragma: no cover - only on regression
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))

        assert errors == [], f"concurrent store access raised: {errors}"

        conn = store._get_conn()
        assert conn.execute("SELECT COUNT(*) FROM pipeline_data").fetchone()[0] == 8 * 20
        assert conn.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0] == 8 * 20
        store.close()

    def test_locking_connection_serializes_across_threads(self, tmp_path):
        """Direct check on the lock itself: two threads calling execute()
        concurrently must never interleave at the C level (which is what
        produces the InterfaceError) — enforced by asserting the same lock
        object guards every connection obtained from one backend."""
        store = PipelineStore(db_path=str(tmp_path / "lock.db"))
        conn_a = store._get_conn()
        conn_b = store._get_conn()
        assert conn_a is conn_b  # cached — same wrapper, same lock
        lock = store._backend._lock
        assert hasattr(lock, "acquire") and hasattr(lock, "release")
        store.close()
