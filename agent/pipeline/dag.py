"""
TirraMind — DAG Data Model

Defines DAG (Directed Acyclic Graph) and Node for pipeline execution.

A DAG is a set of nodes with dependency relationships. Each node represents
either a tool call (via ToolOperator) or a pure function (via FunctionOperator).
Nodes declare dependencies; the executor runs independent nodes in parallel.

Usage:
    dag = DAG("daily_collection", schedule="0 18 * * 1-5")
    dag.add("fetch_cftc", operator="cftc", params={"mode": "latest"})
    dag.add("fetch_finra", operator="finra_short_volume", params={"mode": "scan"})
    dag.add("compute", operator=my_func, depends_on=["fetch_cftc", "fetch_finra"])
    errors = dag.validate()
    layers = dag.topo_sort()  # [[fetch_cftc, fetch_finra], [compute]]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Node:
    """A single unit of work in a DAG."""

    id: str
    operator: str | Callable[..., Any]  # Tool name (str) or callable
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    retries: int = 1
    timeout: int = 60
    store_result: bool = True
    table_name: str | None = None


@dataclass
class DAG:
    """Directed Acyclic Graph of pipeline nodes."""

    name: str
    schedule: str | None = None  # Cron expression or None for manual
    description: str = ""
    nodes: dict[str, Node] = field(default_factory=dict)

    def add(
        self,
        node_id: str,
        operator: str | Callable[..., Any],
        params: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Add a node to the DAG. Returns the node ID."""
        if node_id in self.nodes:
            raise ValueError(f"Duplicate node ID: {node_id!r}")
        node = Node(
            id=node_id,
            operator=operator,
            params=params or {},
            depends_on=depends_on or [],
            **kwargs,
        )
        self.nodes[node_id] = node
        return node_id

    def roots(self) -> list[str]:
        """Return node IDs with no dependencies (entry points)."""
        return [nid for nid, n in self.nodes.items() if not n.depends_on]

    def validate(self) -> list[str]:
        """Validate the DAG. Returns list of error strings (empty = valid)."""
        errors: list[str] = []

        if not self.nodes:
            errors.append("DAG has no nodes")
            return errors

        if not self.name:
            errors.append("DAG has no name")

        # Check: all depends_on reference valid nodes
        for nid, node in self.nodes.items():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    errors.append(
                        f"Node {nid!r} depends on {dep!r} which does not exist"
                    )
                if dep == nid:
                    errors.append(f"Node {nid!r} has self-dependency")

        # Check: no cycles (Kahn's algorithm)
        if not errors:  # Only check cycles if deps are valid
            cycle_err = self._detect_cycle()
            if cycle_err:
                errors.append(cycle_err)

        return errors

    def _detect_cycle(self) -> str | None:
        """Detect cycles using Kahn's algorithm. Returns error string or None."""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.depends_on:
                # dep -> node (dep is depended upon by node)
                # We need reverse: in_degree counts how many deps a node HAS
                pass
        # Recompute: in_degree[nid] = number of dependencies nid has
        in_degree = {nid: len(n.depends_on) for nid, n in self.nodes.items()}

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        visited = 0

        # adjacency: parent -> children (if A depends on B, then B -> A)
        children: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.depends_on:
                children[dep].append(nid)

        while queue:
            current = queue.popleft()
            visited += 1
            for child in children[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if visited != len(self.nodes):
            # Find nodes in cycle for error message
            in_cycle = [nid for nid, deg in in_degree.items() if deg > 0]
            return f"DAG contains a cycle involving nodes: {in_cycle}"

        return None

    def topo_sort(self) -> list[list[str]]:
        """Topological sort into execution layers.

        Returns a list of layers. Each layer is a list of node IDs that
        can execute in parallel. Layer N depends only on layers 0..N-1.

        Raises ValueError if DAG is invalid (has cycles or other errors).
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid DAG: {'; '.join(errors)}")

        # Layer-by-layer topological sort (Kahn's with level tracking)
        in_degree = {nid: len(n.depends_on) for nid, n in self.nodes.items()}
        children: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.depends_on:
                children[dep].append(nid)

        current_layer = [nid for nid, deg in in_degree.items() if deg == 0]
        layers: list[list[str]] = []

        while current_layer:
            layers.append(sorted(current_layer))  # Sort for determinism
            next_layer: list[str] = []
            for nid in current_layer:
                for child in children[nid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_layer.append(child)
            current_layer = next_layer

        return layers
