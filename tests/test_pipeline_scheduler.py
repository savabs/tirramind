"""
Tests for PipelineScheduler (Step 7.6).

Covers: registration, lifecycle (start/stop), trigger, blocking behavior,
scheduled job wiring, error handling, and integration flow.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, PropertyMock

import pytest

from agent.pipeline.dag import DAG
from agent.pipeline.executor import DAGExecutor, DagRun, NodeResult
from agent.pipeline.scheduler import PipelineScheduler


# ── Helpers ───────────────────────────────────────────────────


def _make_dag(
    name: str = "test_dag",
    schedule: str | None = None,
    description: str = "",
) -> DAG:
    """Create a minimal valid DAG with one node."""
    dag = DAG(name=name, schedule=schedule, description=description or f"Test: {name}")
    dag.add("node_a", operator=lambda p, u: "ok", params={})
    return dag


def _make_executor(**overrides) -> MagicMock:
    """Mock DAGExecutor returning a successful DagRun."""
    executor = MagicMock(spec=DAGExecutor)
    executor.execute.return_value = DagRun(
        run_id="run-001",
        dag_name="test_dag",
        started_at=100.0,
        finished_at=101.5,
        status="completed",
        trigger="scheduled",
        node_results={"node_a": NodeResult(node_id="node_a", status="completed")},
    )
    for k, v in overrides.items():
        setattr(executor.execute, k, v)
    return executor


def _make_registry(dags: list[DAG] | None = None) -> MagicMock:
    """Mock a DAGProvider-compatible registry."""
    registry = MagicMock()
    registry.list_all.return_value = dags or []
    return registry


# ── Registration ──────────────────────────────────────────────


class TestRegistration:
    """Tests for register() and list_dags()."""

    def test_register_valid_dag(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag("my_dag"))
        assert len(s.list_dags()) == 1
        assert s.list_dags()[0]["name"] == "my_dag"

    def test_register_dag_with_schedule(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag("cron_dag", schedule="0 18 * * 1-5"))
        info = s.list_dags()[0]
        assert info["schedule"] == "0 18 * * 1-5"
        assert info["nodes"] == 1

    def test_register_invalid_dag_raises_valueerror(self):
        s = PipelineScheduler(executor=_make_executor())
        bad_dag = DAG(name="empty")  # No nodes
        with pytest.raises(ValueError, match="Invalid DAG"):
            s.register(bad_dag)

    def test_register_while_running_raises(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag())
        s.start(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="Cannot register"):
                s.register(_make_dag("another"))
        finally:
            s.stop()

    def test_register_overwrites_same_name(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag("dup", schedule="0 12 * * *"))
        s.register(_make_dag("dup", schedule="0 18 * * *"))
        assert len(s.list_dags()) == 1
        assert s.list_dags()[0]["schedule"] == "0 18 * * *"

    def test_register_multiple_dags(self):
        s = PipelineScheduler(executor=_make_executor())
        for name in ["alpha", "beta", "gamma"]:
            s.register(_make_dag(name))
        names = {d["name"] for d in s.list_dags()}
        assert names == {"alpha", "beta", "gamma"}


class TestRegistryConstructor:
    """Tests for auto-registration from registry in constructor."""

    def test_registry_dags_auto_registered(self):
        dags = [_make_dag("a"), _make_dag("b"), _make_dag("c")]
        registry = _make_registry(dags)
        s = PipelineScheduler(executor=_make_executor(), registry=registry)
        assert len(s.list_dags()) == 3
        registry.list_all.assert_called_once()

    def test_no_registry_means_empty(self):
        s = PipelineScheduler(executor=_make_executor())
        assert s.list_dags() == []

    def test_empty_registry(self):
        registry = _make_registry([])
        s = PipelineScheduler(executor=_make_executor(), registry=registry)
        assert s.list_dags() == []

    def test_registry_invalid_dag_raises(self):
        bad = DAG(name="bad")  # No nodes
        registry = _make_registry([bad])
        with pytest.raises(ValueError, match="Invalid DAG"):
            PipelineScheduler(executor=_make_executor(), registry=registry)


# ── list_dags ─────────────────────────────────────────────────


class TestListDags:
    """Tests for list_dags() output format."""

    def test_empty(self):
        s = PipelineScheduler(executor=_make_executor())
        assert s.list_dags() == []

    def test_fields_present(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag("x", schedule="*/5 * * * *", description="Every 5m"))
        info = s.list_dags()[0]
        assert info["name"] == "x"
        assert info["schedule"] == "*/5 * * * *"
        assert info["description"] == "Every 5m"
        assert info["nodes"] == 1

    def test_manual_dag_schedule_is_none(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag("manual"))
        assert s.list_dags()[0]["schedule"] is None


# ── Lifecycle ─────────────────────────────────────────────────


class TestLifecycle:
    """Tests for start(), stop(), running property."""

    def test_start_stop(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag())
        s.start(blocking=False)
        assert s.running is True
        s.stop()
        assert s.running is False

    def test_running_before_start(self):
        s = PipelineScheduler(executor=_make_executor())
        assert s.running is False

    def test_double_start_raises(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag())
        s.start(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="already running"):
                s.start(blocking=False)
        finally:
            s.stop()

    def test_stop_idempotent(self):
        s = PipelineScheduler(executor=_make_executor())
        s.stop()  # Not started — no-op
        s.stop()  # Still no-op
        assert s.running is False

    def test_stop_after_start(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag())
        s.start(blocking=False)
        s.stop()
        assert s.running is False
        assert s._scheduler is None  # Cleaned up

    def test_restart_after_stop(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag())
        s.start(blocking=False)
        s.stop()
        # Should be able to start again
        s.start(blocking=False)
        assert s.running is True
        s.stop()

    def test_start_with_no_dags(self):
        s = PipelineScheduler(executor=_make_executor())
        s.start(blocking=False)
        try:
            assert s.running is True
            assert s._scheduler.get_jobs() == []
        finally:
            s.stop()

    def test_start_registers_only_scheduled_dags(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag("cron1", schedule="0 18 * * 1-5"))
        s.register(_make_dag("cron2", schedule="0 6 * * *"))
        s.register(_make_dag("manual"))  # No schedule
        s.start(blocking=False)
        try:
            jobs = s._scheduler.get_jobs()
            job_ids = {j.id for j in jobs}
            assert job_ids == {"cron1", "cron2"}
        finally:
            s.stop()

    def test_jobs_cleaned_after_stop(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag("d", schedule="0 0 * * *"))
        s.start(blocking=False)
        s.stop()
        assert s._scheduler is None


# ── Blocking Behavior ─────────────────────────────────────────


class TestBlocking:
    """Tests for blocking start and unblocking via stop()."""

    def test_blocking_start_unblocked_by_stop(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag())
        started = threading.Event()
        finished = threading.Event()

        def run():
            started.set()
            s.start(blocking=True)
            finished.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        started.wait(timeout=3)
        time.sleep(0.1)  # Let start() reach _stop_event.wait()

        assert not finished.is_set(), "start(blocking=True) should still be blocking"
        s.stop()
        finished.wait(timeout=3)
        assert finished.is_set(), "start() should have returned after stop()"
        t.join(timeout=3)

    def test_nonblocking_start_returns_immediately(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag())
        # This must return — if it blocks, the test will timeout
        s.start(blocking=False)
        assert s.running is True
        s.stop()


# ── Trigger ───────────────────────────────────────────────────


class TestTrigger:
    """Tests for trigger() manual DAG execution."""

    def test_trigger_known_dag(self):
        executor = _make_executor()
        s = PipelineScheduler(executor=executor)
        s.register(_make_dag("my_dag"))
        run = s.trigger("my_dag")
        executor.execute.assert_called_once()
        assert run.status == "completed"
        assert run.run_id == "run-001"

    def test_trigger_unknown_dag_raises(self):
        s = PipelineScheduler(executor=_make_executor())
        with pytest.raises(KeyError, match="Unknown DAG"):
            s.trigger("nonexistent")

    def test_trigger_works_without_start(self):
        executor = _make_executor()
        s = PipelineScheduler(executor=executor)
        s.register(_make_dag("d"))
        run = s.trigger("d")
        assert isinstance(run, DagRun)
        assert run.dag_name == "test_dag"

    def test_trigger_works_while_running(self):
        executor = _make_executor()
        s = PipelineScheduler(executor=executor)
        s.register(_make_dag("d"))
        s.start(blocking=False)
        try:
            run = s.trigger("d")
            assert run.status == "completed"
        finally:
            s.stop()

    def test_trigger_passes_manual_trigger_type(self):
        executor = _make_executor()
        s = PipelineScheduler(executor=executor)
        dag = _make_dag("d")
        s.register(dag)
        s.trigger("d")
        call_args = executor.execute.call_args
        assert call_args.kwargs.get("trigger") == "manual" or call_args[1].get("trigger") == "manual"

    def test_trigger_passes_correct_dag(self):
        executor = _make_executor()
        s = PipelineScheduler(executor=executor)
        dag = _make_dag("specific")
        s.register(dag)
        s.trigger("specific")
        passed_dag = executor.execute.call_args[0][0]
        assert passed_dag.name == "specific"
        assert passed_dag is dag

    def test_trigger_propagates_executor_error(self):
        executor = _make_executor()
        executor.execute.side_effect = RuntimeError("executor boom")
        s = PipelineScheduler(executor=executor)
        s.register(_make_dag())
        with pytest.raises(RuntimeError, match="executor boom"):
            s.trigger("test_dag")

    def test_trigger_returns_dagrun(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag())
        result = s.trigger("test_dag")
        assert isinstance(result, DagRun)


# ── _run_dag (scheduled execution) ───────────────────────────


class TestRunDag:
    """Tests for _run_dag() — the method APScheduler calls."""

    def test_run_dag_calls_executor(self):
        executor = _make_executor()
        s = PipelineScheduler(executor=executor)
        s.register(_make_dag("d"))
        run = s._run_dag("d")
        executor.execute.assert_called_once()
        assert run is not None
        assert run.status == "completed"

    def test_run_dag_uses_scheduled_trigger(self):
        executor = _make_executor()
        s = PipelineScheduler(executor=executor)
        s.register(_make_dag("d"))
        s._run_dag("d")
        assert executor.execute.call_args[1]["trigger"] == "scheduled"

    def test_run_dag_unknown_name_returns_none(self):
        s = PipelineScheduler(executor=_make_executor())
        result = s._run_dag("ghost")
        assert result is None

    def test_run_dag_swallows_executor_exception(self):
        executor = _make_executor()
        executor.execute.side_effect = RuntimeError("crash")
        s = PipelineScheduler(executor=executor)
        s.register(_make_dag("d"))
        # Should NOT raise — returns None
        result = s._run_dag("d")
        assert result is None

    def test_run_dag_swallows_any_exception_type(self):
        executor = _make_executor()
        executor.execute.side_effect = ValueError("bad value")
        s = PipelineScheduler(executor=executor)
        s.register(_make_dag("d"))
        result = s._run_dag("d")
        assert result is None


# ── Cron Validation ───────────────────────────────────────────


class TestCronValidation:
    """Tests for cron expression handling."""

    def test_valid_cron_expressions(self):
        s = PipelineScheduler(executor=_make_executor())
        valid_crons = [
            "0 18 * * 1-5",     # weekdays at 6pm
            "*/15 * * * *",     # every 15 minutes
            "0 0 * * *",        # midnight daily
            "30 6 1 * *",       # 6:30 on 1st of month
            "0 */2 * * *",      # every 2 hours
        ]
        for cron in valid_crons:
            dag = _make_dag(f"dag_{cron.replace(' ', '_')}", schedule=cron)
            s.register(dag)  # Should not raise

        s.start(blocking=False)
        try:
            jobs = s._scheduler.get_jobs()
            assert len(jobs) == len(valid_crons)
        finally:
            s.stop()

    def test_invalid_cron_expression_raises_on_start(self):
        s = PipelineScheduler(executor=_make_executor())
        dag = _make_dag("bad_cron", schedule="not a cron")
        s.register(dag)  # register doesn't validate cron
        with pytest.raises(ValueError):
            s.start(blocking=False)
        # Scheduler should not be left in running state
        assert s.running is False


# ── Integration ───────────────────────────────────────────────


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_lifecycle(self):
        """Register → start → trigger → stop."""
        executor = _make_executor()
        s = PipelineScheduler(executor=executor)
        dag = _make_dag("daily", schedule="0 18 * * 1-5")
        s.register(dag)

        s.start(blocking=False)
        assert s.running

        run = s.trigger("daily")
        assert run.status == "completed"
        assert executor.execute.call_count == 1

        s.stop()
        assert not s.running

    def test_multiple_dags_triggered(self):
        executor = _make_executor()
        s = PipelineScheduler(executor=executor)
        for name in ["dag_a", "dag_b", "dag_c"]:
            s.register(_make_dag(name, schedule="0 * * * *"))

        s.start(blocking=False)
        try:
            assert len(s._scheduler.get_jobs()) == 3
            for name in ["dag_a", "dag_b", "dag_c"]:
                s.trigger(name)
            assert executor.execute.call_count == 3
        finally:
            s.stop()

    def test_mixed_scheduled_and_manual_dags(self):
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag("scheduled_dag", schedule="0 0 * * *"))
        s.register(_make_dag("manual_dag"))
        s.start(blocking=False)
        try:
            jobs = s._scheduler.get_jobs()
            assert len(jobs) == 1
            assert jobs[0].id == "scheduled_dag"
            # Both can be triggered manually
            s.trigger("scheduled_dag")
            s.trigger("manual_dag")
        finally:
            s.stop()

    def test_registry_to_scheduler_flow(self):
        """Simulate the CLI flow: registry → scheduler → start → stop."""
        dags = [
            _make_dag("fetch_cftc", schedule="0 18 * * 1-5"),
            _make_dag("fetch_finra", schedule="0 18 * * 1-5"),
            _make_dag("manual_analysis"),
        ]
        registry = _make_registry(dags)
        executor = _make_executor()

        s = PipelineScheduler(executor=executor, registry=registry)
        assert len(s.list_dags()) == 3

        s.start(blocking=False)
        try:
            assert len(s._scheduler.get_jobs()) == 2  # Only scheduled ones
        finally:
            s.stop()

    def test_cli_blocking_pattern(self):
        """Simulate the CLI pattern: start(blocking) → KeyboardInterrupt → stop."""
        s = PipelineScheduler(executor=_make_executor())
        s.register(_make_dag("d", schedule="0 0 * * *"))
        finished = threading.Event()

        def cli_simulation():
            try:
                s.start(blocking=True)
            except KeyboardInterrupt:
                s.stop()
            finished.set()

        t = threading.Thread(target=cli_simulation, daemon=True)
        t.start()
        time.sleep(0.2)  # Let scheduler start and block
        assert s.running

        # Simulate what the CLI does when Ctrl+C is pressed:
        # In practice, the main thread gets SIGINT. In tests, we just stop().
        s.stop()
        finished.wait(timeout=3)
        assert finished.is_set()
        assert not s.running
        t.join(timeout=3)
