"""
TirraMind — DAG Executor

Executes a DAG by topologically sorting nodes into layers, then running
each layer in parallel via ThreadPoolExecutor. Handles retry, timeout,
failure propagation, and result storage.

Usage:
    executor = DAGExecutor(tool_registry=registry, store=store)
    run = executor.execute(dag, trigger="manual")
    print(run.status, run.node_results)
"""

from __future__ import annotations

import functools
import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any

from agent.pipeline.dag import DAG
from agent.pipeline.operators import resolve_operator
from agent.pipeline.store import PipelineStore
from agent.tools.base import ToolRegistry

log = logging.getLogger(__name__)

# Default: a run stuck in status='running' with no heartbeat for longer than
# this is almost certainly a dead process, not slow work — see
# ``PipelineStore.reap_stale_runs``. Sized generously above the largest single
# node timeout in any registered DAG (train_gnn=1800s) plus room for a full
# layered run of several such nodes in sequence.
_DEFAULT_STALE_RUN_SECONDS = 3600.0 * 3

# Matches the credential-missing error strings tools raise when a free-tier
# key (FRED, NASA FIRMS, EIA, ...) is unset — see agent/tools/*.py's own
# "API key required" / "API key not configured" / "requires a FRED API key" /
# "MISSING_CONFIG" messages (surveyed across every tool in agent/tools/ that
# gates on a TIRRA_*_KEY env var). Tool-internal wording is owned by the L1
# data engineers; this pattern only classifies the *executor outcome* (skip
# vs. fail), which is this file's call per this DAG's rule 5. Deliberately
# does NOT match "check API key validity" style messages (a configured-but-
# invalid key is a real, loud failure, not a permanently-missing credential).
_MISSING_CREDENTIAL_RE = re.compile(
    r"api key (?:required|not configured|missing)"
    r"|missing_config"
    r"|requires (?:an?\s+)?[a-z0-9_]*\s*api key"
    r"|requires tirra_\w+"
    r"|set tirra_\w+",
    re.IGNORECASE,
)


def _is_missing_credential_error(message: str) -> bool:
    return bool(_MISSING_CREDENTIAL_RE.search(message or ""))


@dataclass
class NodeResult:
    """Outcome of executing a single node."""

    node_id: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    started_at: float | None = None
    finished_at: float | None = None
    output: Any = None
    error: str | None = None
    retries_used: int = 0
    stored: bool = False  # True once its result actually landed in the store

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "retries_used": self.retries_used,
        }


@dataclass
class DagRun:
    """Full execution record of a DAG run."""

    run_id: str
    dag_name: str
    started_at: float = 0.0
    finished_at: float | None = None
    status: str = "running"  # running | completed | failed
    trigger: str = "manual"
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    rows_written: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "completed"


class DAGExecutor:
    """Executes a DAG: topo sort → parallel layers → retry → store results."""

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        store: PipelineStore | None = None,
        max_workers: int = 4,
        stale_run_after_seconds: float = _DEFAULT_STALE_RUN_SECONDS,
    ) -> None:
        self._registry = tool_registry
        self._store = store
        self._max_workers = max_workers
        self._stale_run_after_seconds = stale_run_after_seconds

    def execute(self, dag: DAG, trigger: str = "manual") -> DagRun:
        """Execute a DAG. Returns a DagRun with all node results."""
        # 1. Validate and topo sort
        layers = dag.topo_sort()  # Raises ValueError if invalid

        # 1b. Self-heal: reap any prior run of *any* DAG left stuck in
        # 'running' by a process that died mid-execution (killed, OOM,
        # reboot) before it ever called record_run_end. Done here so every
        # entry point (manual trigger, scheduler cron job, run_chain.py)
        # gets this for free.
        if self._store:
            try:
                self._store.reap_stale_runs(self._stale_run_after_seconds)
            except Exception:
                log.warning("reap_stale_runs failed; continuing anyway", exc_info=True)

        # 2. Create run record
        run_id = PipelineStore.new_run_id()
        if self._store:
            self._store.record_run_start(dag.name, trigger=trigger, run_id=run_id)

        run = DagRun(
            run_id=run_id,
            dag_name=dag.name,
            started_at=time.time(),
            trigger=trigger,
        )

        # Initialize all node results as pending
        for node in dag.nodes.values():
            run.node_results[node.id] = NodeResult(node_id=node.id)

        upstream_outputs: dict[str, Any] = {}
        any_failure = False
        # Count nodes that were actually eligible to persist something (not
        # disabled, not dependency-skipped) so the zero-rows guard below can
        # tell "nothing ran" apart from "things ran and legitimately wrote
        # nothing" apart from "things ran, claimed success, and stored zero
        # rows" (the F-01..F-04 silent-success pattern applied to collection).
        eligible_store_nodes = 0

        # 3. Execute layer by layer
        for layer in layers:
            layer_results = self._execute_layer(dag, layer, upstream_outputs, run)
            for nid, nr in layer_results.items():
                run.node_results[nid] = nr
                node = dag.nodes[nid]
                if node.store_result and nr.status != "skipped":
                    eligible_store_nodes += 1
                if nr.status == "completed":
                    upstream_outputs[nid] = nr.output
                elif nr.status == "failed":
                    any_failure = True

            # Heartbeat once per layer boundary so a run spanning several
            # layers (e.g. model DAGs) doesn't look dead to reap_stale_runs
            # just because a single layer takes a while.
            if self._store:
                try:
                    self._store.heartbeat(run_id)
                except Exception:
                    log.warning("heartbeat failed for run %s", run_id, exc_info=True)

        run.rows_written = sum(1 for nr in run.node_results.values() if nr.stored)

        # 4. Finalize
        run.finished_at = time.time()
        run.status = "failed" if any_failure else "completed"

        if self._store is not None:
            self._apply_zero_rows_guard(run, eligible_store_nodes, any_failure)

        if self._store:
            result_dicts = {nid: nr.to_dict() for nid, nr in run.node_results.items()}
            if run.error:
                result_dicts["_run_error"] = run.error
            self._store.record_run_end(run_id, run.status, result_dicts)

        log.info(
            "DAG %s run %s finished: %s (%.1fs)",
            dag.name,
            run_id,
            run.status,
            (run.finished_at - run.started_at),
        )
        return run

    @staticmethod
    def _apply_zero_rows_guard(run: DagRun, eligible_store_nodes: int, any_failure: bool) -> None:
        """Downgrade *run* to 'failed' if nothing was actually persisted.

        A run that had at least one node eligible to store data, hit no
        node-level failures, yet ``run.rows_written == 0`` is the exact
        silent-success shape documented in LESSONS.md (F-01..F-04) applied to
        collection: every node "completed" without raising, but nothing
        landed in the store. Mutates ``run.status``/``run.error`` in place;
        does not touch ``dag_runs`` itself (the caller persists afterward).
        """
        if any_failure or eligible_store_nodes == 0 or run.rows_written > 0:
            return
        run.status = "failed"
        run.error = (
            f"{eligible_store_nodes} node(s) completed and were eligible to store "
            "results, but zero rows were written to the store. Treating as a failed "
            "run rather than reporting false success."
        )
        log.error("DAG %s run %s: %s", run.dag_name, run.run_id, run.error)

    def _execute_layer(
        self,
        dag: DAG,
        layer: list[str],
        upstream_outputs: dict[str, Any],
        run: DagRun,
    ) -> dict[str, NodeResult]:
        """Execute all nodes in a layer in parallel."""
        results: dict[str, NodeResult] = {}

        # Check which nodes should be skipped (dependency failed or disabled)
        executable = []
        for nid in layer:
            node = dag.nodes[nid]
            # Change 12: skip nodes disabled by tool routing
            if not node.enabled:
                nr = run.node_results[nid]
                nr.status = "skipped"
                nr.error = "Skipped: disabled by tool router"
                results[nid] = nr
                log.info("Node %s skipped: disabled by tool router", nid)
                continue
            deps_ok = all(
                run.node_results.get(dep, NodeResult(node_id=dep)).status == "completed" for dep in node.depends_on
            )
            if not deps_ok:
                nr = run.node_results[nid]
                nr.status = "skipped"
                nr.error = "Skipped: upstream dependency failed"
                results[nid] = nr
                log.warning("Node %s skipped: upstream dependency failed", nid)
            else:
                executable.append(nid)

        if not executable:
            return results

        # Execute in parallel. NOTE: this pool is deliberately *not* used as
        # a context manager. ``ThreadPoolExecutor.__exit__`` calls
        # ``shutdown(wait=True)``, which blocks until every submitted task
        # finishes — even ones we already gave up on via
        # ``future.result(timeout=...)`` below. That defeated the whole
        # point of a node timeout: a node with timeout=1 whose operator took
        # ~8s was reported "failed: timed out" immediately, yet the layer
        # (and the run) didn't actually move on until that operator's
        # thread finished ~8s later anyway — Python cannot forcibly kill a
        # running thread, so the only lever we have is not waiting on it.
        pool = ThreadPoolExecutor(max_workers=min(self._max_workers, len(executable)))
        try:
            futures: dict[str, Future[NodeResult]] = {}
            for nid in executable:
                node = dag.nodes[nid]
                future = pool.submit(self._execute_node, node, upstream_outputs)
                futures[nid] = future

            for nid, future in futures.items():
                node = dag.nodes[nid]
                try:
                    nr = future.result(timeout=node.timeout)
                except TimeoutError:
                    nr = NodeResult(
                        node_id=nid,
                        status="failed",
                        error=(
                            f"Execution timed out (>{node.timeout}s); the operator "
                            "keeps running in the background since a thread cannot "
                            "be forcibly stopped — if it later completes (and "
                            "commits) this run's record will be corrected rather "
                            "than left showing a false failure."
                        ),
                        started_at=time.time(),
                        finished_at=time.time(),
                    )
                    # The thread behind `future` is still running. Reconcile
                    # the record once it actually finishes instead of
                    # permanently reporting "failed" for work that quietly
                    # succeeds and commits after the deadline.
                    future.add_done_callback(functools.partial(self._reconcile_timeout, run, node, nr))
                except Exception as exc:
                    nr = NodeResult(
                        node_id=nid,
                        status="failed",
                        error=str(exc),
                        started_at=time.time(),
                        finished_at=time.time(),
                    )

                results[nid] = nr

                # Store result if successful
                if nr.status == "completed" and self._store and node.store_result:
                    try:
                        self._store.store_data(
                            source=node.table_name or node.id,
                            params=node.params,
                            data=nr.output,
                        )
                        nr.stored = True
                    except Exception as exc:
                        # Previously only logged a warning and left nr.status as
                        # "completed" — the operator's real output (fetched data)
                        # then simply vanished with no record of a failure
                        # anywhere except this log line. That's the exact
                        # "returned dict is success" shape from a different
                        # angle: the node ran fine, but the thing the node
                        # exists to do (persist its result) silently didn't
                        # happen. Downgrading to "failed" makes it count
                        # toward any_failure and toward the zero-rows guard.
                        nr.status = "failed"
                        nr.error = f"Executed successfully but failed to store result: {exc}"
                        log.error("Failed to store result for %s: %s", nid, exc)
        finally:
            # wait=False: don't block the run on a node that already blew
            # through its timeout. Its thread finishes on its own; see
            # _reconcile_timeout for how its eventual outcome gets recorded.
            pool.shutdown(wait=False)

        return results

    def _reconcile_timeout(
        self,
        run: DagRun,
        node: Any,  # Node type
        nr: NodeResult,
        future: Future[NodeResult],
    ) -> None:
        """Correct a node's recorded outcome once a timed-out operator finishes.

        Runs as a ``Future`` done-callback, potentially from the worker
        thread that just finished running *node*'s operator (possibly well
        after this run's layer — and even the whole DAG run — already
        returned). Mutates *nr* in place: it is the same ``NodeResult``
        instance already sitting in both ``run.node_results`` and the
        ``DagRun`` handed back to the caller, so callers holding onto that
        run see the correction regardless of when it lands. Also re-persists
        the run record so ``dag_runs`` doesn't keep a stale false failure.
        """
        try:
            # NOTE: `future` wraps `self._execute_node`, whose return value is
            # itself a NodeResult (not the raw operator output) — `_execute_node`
            # never lets an operator exception escape (it's caught into a
            # status="failed" NodeResult after retries), so this call only
            # raises for something outside that contract. Pull fields back out
            # of it below rather than assigning it wholesale to `nr.output`:
            # that previously nested a whole NodeResult inside `nr.output`
            # (and inside the stored row's data) instead of the operator's
            # actual output.
            result = future.result()
        except Exception as exc:
            nr.status = "failed"
            nr.error = f"Failed after exceeding its {node.timeout}s timeout: {exc}"
            nr.finished_at = time.time()
            log.warning(
                "Node %s failed after exceeding its %ss timeout: %s",
                node.id,
                node.timeout,
                exc,
            )
        else:
            nr.status = result.status
            nr.output = result.output
            nr.error = result.error
            nr.retries_used = result.retries_used
            nr.finished_at = result.finished_at or time.time()
            if result.status == "completed":
                log.warning(
                    "Node %s exceeded its %ss timeout but completed successfully "
                    "afterward — correcting its recorded status from 'failed' to "
                    "'completed' (its write, if any, already landed and is not "
                    "being redone).",
                    node.id,
                    node.timeout,
                )
                if self._store and node.store_result:
                    try:
                        self._store.store_data(
                            source=node.table_name or node.id,
                            params=node.params,
                            data=result.output,
                        )
                        nr.stored = True
                    except Exception as exc:
                        # Same reasoning as the main-path fix in _execute_layer:
                        # a completed-but-unstored result is not a success.
                        nr.status = "failed"
                        nr.error = f"Completed after timeout but failed to store result: {exc}"
                        log.error("Failed to store late result for %s: %s", node.id, exc)
            else:
                log.warning(
                    "Node %s exceeded its %ss timeout and ultimately failed anyway: %s",
                    node.id,
                    node.timeout,
                    result.error,
                )

        if self._store is not None:
            try:
                any_failure = any(r.status == "failed" for r in run.node_results.values())
                run.rows_written = sum(1 for r in run.node_results.values() if r.stored)
                run.status = "failed" if any_failure else "completed"
                result_dicts = {n: r.to_dict() for n, r in run.node_results.items()}
                self._store.record_run_end(run.run_id, run.status, result_dicts)
            except Exception:
                log.warning(
                    "Failed to persist reconciled status for run %s node %s",
                    run.run_id,
                    node.id,
                    exc_info=True,
                )

    def _execute_node(
        self,
        node: Any,  # Node type
        upstream_outputs: dict[str, Any],
    ) -> NodeResult:
        """Execute a single node with retry logic."""
        nr = NodeResult(node_id=node.id, status="running", started_at=time.time())

        # Build params for the operator
        if isinstance(node.operator, str):
            # Tool operator: inject __tool__ key
            exec_params = {"__tool__": node.operator, **node.params}
        else:
            exec_params = dict(node.params)

        # Resolve operator
        operator = resolve_operator(node.operator, tool_registry=self._registry)

        last_error: str | None = None
        for attempt in range(node.retries):
            try:
                result = operator.execute(exec_params, upstream_results=upstream_outputs)
                nr.status = "completed"
                nr.output = result
                nr.finished_at = time.time()
                nr.retries_used = attempt
                return nr
            except Exception as exc:
                last_error = str(exc)
                nr.retries_used = attempt + 1

                # Rule 5: a source with no credential configured (FRED, NASA
                # FIRMS, EIA, ...) is not a broken node — it's an honest,
                # permanent, known state that will fail identically on every
                # retry and every future run until someone sets the env var.
                # Retrying it burns the node's timeout budget for nothing and
                # marking it "failed" makes a run with an unset optional key
                # look identical to a genuine breakage in the dag_runs table.
                # Skip immediately with the reason preserved verbatim (loud,
                # not swallowed — just classified correctly).
                if _is_missing_credential_error(last_error):
                    nr.status = "skipped"
                    nr.error = f"Missing credential: {last_error}"
                    nr.finished_at = time.time()
                    log.warning(
                        "Node %s skipped: missing credential (%s)",
                        node.id,
                        last_error,
                    )
                    return nr

                if attempt < node.retries - 1:
                    backoff = 0.1 * (2**attempt)  # 0.1, 0.2, 0.4, ...
                    log.warning(
                        "Node %s attempt %d failed: %s. Retrying in %.1fs",
                        node.id,
                        attempt + 1,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    # Final attempt: no retry follows, so the intermediate
                    # branch above never fires for it. Without this, a node
                    # with the default retries=1 (or any node's last attempt)
                    # fails with zero logged diagnostic — the exception is
                    # only visible in the returned NodeResult.error, which
                    # most callers never inspect.
                    log.warning(
                        "Node %s attempt %d failed (no retries left): %s",
                        node.id,
                        attempt + 1,
                        exc,
                    )

        # All retries exhausted
        nr.status = "failed"
        nr.error = last_error
        nr.finished_at = time.time()
        return nr
