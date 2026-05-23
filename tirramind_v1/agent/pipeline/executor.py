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

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any

from agent.pipeline.dag import DAG
from agent.pipeline.operators import resolve_operator
from agent.pipeline.store import PipelineStore
from agent.tools.base import ToolRegistry

log = logging.getLogger(__name__)


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
    ) -> None:
        self._registry = tool_registry
        self._store = store
        self._max_workers = max_workers

    def execute(self, dag: DAG, trigger: str = "manual") -> DagRun:
        """Execute a DAG. Returns a DagRun with all node results."""
        # 1. Validate and topo sort
        layers = dag.topo_sort()  # Raises ValueError if invalid

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

        # 3. Execute layer by layer
        for layer in layers:
            layer_results = self._execute_layer(dag, layer, upstream_outputs, run)
            for nid, nr in layer_results.items():
                run.node_results[nid] = nr
                if nr.status == "completed":
                    upstream_outputs[nid] = nr.output
                elif nr.status == "failed":
                    any_failure = True

        # 4. Finalize
        run.finished_at = time.time()
        run.status = "failed" if any_failure else "completed"

        if self._store:
            result_dicts = {nid: nr.to_dict() for nid, nr in run.node_results.items()}
            self._store.record_run_end(run_id, run.status, result_dicts)

        log.info(
            "DAG %s run %s finished: %s (%.1fs)",
            dag.name,
            run_id,
            run.status,
            (run.finished_at - run.started_at),
        )
        return run

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

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(executable))) as pool:
            futures: dict[str, Future[NodeResult]] = {}
            for nid in executable:
                node = dag.nodes[nid]
                future = pool.submit(self._execute_node, node, upstream_outputs)
                futures[nid] = future

            for nid, future in futures.items():
                node = dag.nodes[nid]
                try:
                    nr = future.result(timeout=node.timeout + 5)  # Extra buffer
                except TimeoutError:
                    nr = NodeResult(
                        node_id=nid,
                        status="failed",
                        error=f"Execution timed out (>{node.timeout}s)",
                        started_at=time.time(),
                        finished_at=time.time(),
                    )
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
                    except Exception as exc:
                        log.warning("Failed to store result for %s: %s", nid, exc)

        return results

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

        # All retries exhausted
        nr.status = "failed"
        nr.error = last_error
        nr.finished_at = time.time()
        return nr
