"""
TirraMind — Pipeline Scheduler

Thin wrapper around APScheduler BackgroundScheduler. Registers DAGs with
cron triggers, executes them via DAGExecutor, provides start/stop lifecycle
and manual-trigger capability.

Design:
    - start(blocking=True) blocks on a threading.Event; KeyboardInterrupt
      propagates to the caller (CLI catches it, calls stop()).
    - start(blocking=False) returns immediately — for tests.
    - stop() is idempotent.
    - trigger() works whether the scheduler is running or not.
    - register() validates eagerly; rejects while running.
    - _run_dag() swallows exceptions so a bad DAG never kills the scheduler.
    - Constructor accepts optional registry (duck-typed: needs list_all()).

Usage:
    scheduler = PipelineScheduler(executor=executor, registry=dag_registry)
    scheduler.start()            # blocks — CLI wraps in try/except KeyboardInterrupt
    # or
    scheduler.start(blocking=False)   # for tests
    scheduler.trigger("daily")
    scheduler.stop()
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol, runtime_checkable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from agent.pipeline.dag import DAG
from agent.pipeline.executor import DAGExecutor, DagRun

log = logging.getLogger(__name__)


@runtime_checkable
class DAGProvider(Protocol):
    """Minimal interface a DAG registry must satisfy."""

    def get(self, name: str) -> DAG | None: ...
    def list_all(self) -> list[DAG]: ...


class PipelineScheduler:
    """Cron-based DAG scheduler backed by APScheduler."""

    def __init__(
        self,
        executor: DAGExecutor,
        registry: Any = None,
    ) -> None:
        self._executor = executor
        self._dags: dict[str, DAG] = {}
        self._scheduler: BackgroundScheduler | None = None
        self._stop_event = threading.Event()
        self._running = False

        # Auto-register from registry if provided (duck-typed: needs list_all())
        if registry is not None:
            for dag in registry.list_all():
                self.register(dag)

    # ── registration ───────────────────────────────────────────

    def register(self, dag: DAG) -> None:
        """Register a DAG. Must be called before start()."""
        if self._running:
            raise RuntimeError("Cannot register DAGs while scheduler is running")
        errors = dag.validate()
        if errors:
            raise ValueError(f"Invalid DAG {dag.name!r}: {'; '.join(errors)}")
        self._dags[dag.name] = dag
        log.info("Registered DAG: %s (schedule=%s)", dag.name, dag.schedule)

    def list_dags(self) -> list[dict[str, Any]]:
        """Return summary info for every registered DAG."""
        return [
            {
                "name": d.name,
                "schedule": d.schedule,
                "description": d.description,
                "nodes": len(d.nodes),
            }
            for d in self._dags.values()
        ]

    # ── lifecycle ──────────────────────────────────────────────

    def start(self, blocking: bool = True) -> None:
        """Start the scheduler.

        Args:
            blocking: If True (default), blocks on an internal Event until
                      stop() is called from another thread or the process
                      receives KeyboardInterrupt (which propagates to caller).
                      If False, returns immediately (for tests).
        """
        if self._running:
            raise RuntimeError("Scheduler already running")

        sched = BackgroundScheduler(timezone="UTC")
        scheduled = 0

        for name, dag in self._dags.items():
            if dag.schedule:
                sched.add_job(
                    self._run_dag,
                    CronTrigger.from_crontab(dag.schedule, timezone="UTC"),
                    id=name,
                    name=f"DAG: {name}",
                    kwargs={"dag_name": name},
                    max_instances=1,
                    coalesce=True,
                    misfire_grace_time=300,
                )
                scheduled += 1
                log.info("Scheduled DAG %s: %s", name, dag.schedule)
            else:
                log.info("DAG %s has no schedule (manual only)", name)

        sched.start()
        self._scheduler = sched
        self._running = True
        self._stop_event.clear()

        log.info(
            "Scheduler started: %d DAGs registered, %d scheduled",
            len(self._dags),
            scheduled,
        )

        if blocking:
            self._stop_event.wait()  # KeyboardInterrupt propagates to caller

    def stop(self) -> None:
        """Shut down the scheduler gracefully. Idempotent."""
        if not self._running:
            return
        if self._scheduler:
            self._scheduler.shutdown(wait=True)
            self._scheduler = None
        self._running = False
        self._stop_event.set()
        log.info("Scheduler stopped")

    @property
    def running(self) -> bool:
        """Whether the scheduler is currently active."""
        return self._running

    # ── execution ──────────────────────────────────────────────

    def trigger(self, dag_name: str) -> DagRun:
        """Execute a DAG immediately (manual trigger).

        Works whether the scheduler is started or not.
        Raises KeyError if DAG not registered.
        Propagates executor exceptions to caller.
        """
        if dag_name not in self._dags:
            raise KeyError(f"Unknown DAG: {dag_name!r}")
        dag = self._dags[dag_name]
        return self._executor.execute(dag, trigger="manual")

    def _run_dag(self, dag_name: str) -> DagRun | None:
        """Execute a DAG. Called by APScheduler cron jobs.

        Exceptions are caught and logged — a failing DAG must never crash
        the scheduler. APScheduler ignores the return value.
        """
        dag = self._dags.get(dag_name)
        if dag is None:
            log.error("Scheduled DAG %r not found", dag_name)
            return None
        try:
            run = self._executor.execute(dag, trigger="scheduled")
            log.info(
                "DAG %s run %s: %s (%.1fs)",
                dag_name,
                run.run_id,
                run.status,
                (run.finished_at or 0) - run.started_at,
            )
            return run
        except Exception:
            log.exception("DAG %s execution failed", dag_name)
            return None
