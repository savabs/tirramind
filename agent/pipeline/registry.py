"""
TirraMind — DAG Registry

Static registry of all defined DAGs. The scheduler and CLI query this
to discover what DAGs exist.

Interface matches the PipelineScheduler's DAGProvider protocol:
    register(dag), get(name), list_all(), load_defaults(tool_registry)

Usage:
    registry = DAGRegistry()
    registry.load_defaults(tool_registry)
    dag = registry.get("daily_collection")
"""

from __future__ import annotations

import logging
from typing import Any

from agent.pipeline.dag import DAG

log = logging.getLogger(__name__)


class DAGRegistry:
    """Central registry for pipeline DAGs."""

    def __init__(self) -> None:
        self._dags: dict[str, DAG] = {}

    def register(self, dag: DAG) -> None:
        """Register a DAG. Validates before accepting."""
        errors = dag.validate()
        if errors:
            raise ValueError(f"Invalid DAG {dag.name!r}: {'; '.join(errors)}")
        if dag.name in self._dags:
            log.warning("Overwriting existing DAG: %s", dag.name)
        self._dags[dag.name] = dag
        log.info("Registered DAG: %s (%d nodes)", dag.name, len(dag.nodes))

    def get(self, name: str) -> DAG | None:
        """Look up a DAG by name. Returns None if not found."""
        return self._dags.get(name)

    def list_all(self) -> list[DAG]:
        """Return all registered DAGs, sorted by name for determinism."""
        return sorted(self._dags.values(), key=lambda d: d.name)

    def list_names(self) -> list[str]:
        """Return sorted list of all DAG names."""
        return sorted(self._dags.keys())

    def remove(self, name: str) -> bool:
        """Remove a DAG. Returns True if it existed."""
        if name in self._dags:
            del self._dags[name]
            return True
        return False

    def __len__(self) -> int:
        return len(self._dags)

    def load_defaults(self, tool_registry: Any) -> None:
        """Register all built-in DAGs.

        Imports DAG builders from agent.pipeline.dags and registers them.
        tool_registry is passed through so DAG builders can reference tool names.
        """
        from agent.pipeline.dags import get_default_dags

        for dag in get_default_dags(tool_registry):
            self.register(dag)
        log.info("Loaded %d default DAGs", len(self._dags))
