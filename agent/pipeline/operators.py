"""
TirraMind — Pipeline Operators

Operators bridge DAG nodes to actual execution — either calling a Tool
from the ToolRegistry or invoking a pure Python function.

ToolOperator:  Looks up a tool by name, calls tool.execute(**params), returns ToolResult.data
FunctionOperator:  Calls a Python callable with (params, upstream_results), returns its output

Both catch exceptions and return structured error info instead of crashing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from agent.tools.base import ToolRegistry, ToolResult

log = logging.getLogger(__name__)


class Operator(ABC):
    """Base class for pipeline operators."""

    @abstractmethod
    def execute(
        self,
        params: dict[str, Any],
        upstream_results: dict[str, Any] | None = None,
    ) -> Any:
        """Execute the operator. Returns result data or raises."""
        ...


class ToolOperator(Operator):
    """Executes a Tool from the ToolRegistry."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._registry = tool_registry

    def execute(
        self,
        params: dict[str, Any],
        upstream_results: dict[str, Any] | None = None,
    ) -> Any:
        tool_name = params.get("__tool__")
        if tool_name is None:
            raise ValueError("ToolOperator requires '__tool__' in params")

        tool = self._registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool not found in registry: {tool_name!r}")

        # Build execution params without __tool__
        exec_params = {k: v for k, v in params.items() if k != "__tool__"}

        # Resolve upstream references in params: "$upstream.node_id"
        resolved = self._resolve_upstream(exec_params, upstream_results or {})

        log.debug("ToolOperator executing: %s(%s)", tool_name, resolved)
        result: ToolResult = tool.execute(**resolved)

        if not result.success:
            raise RuntimeError(f"Tool {tool_name!r} failed: {result.output}")

        return result.data if result.data is not None else result.output

    @staticmethod
    def _resolve_upstream(
        params: dict[str, Any],
        upstream: dict[str, Any],
    ) -> dict[str, Any]:
        """Replace '$upstream.node_id' strings with actual upstream results."""
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("$upstream."):
                ref_id = v[len("$upstream."):]
                if ref_id not in upstream:
                    raise ValueError(
                        f"Upstream reference '{v}' not found. "
                        f"Available: {list(upstream.keys())}"
                    )
                resolved[k] = upstream[ref_id]
            else:
                resolved[k] = v
        return resolved


class FunctionOperator(Operator):
    """Executes a pure Python callable."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError(f"FunctionOperator requires a callable, got {type(fn)}")
        self._fn = fn

    def execute(
        self,
        params: dict[str, Any],
        upstream_results: dict[str, Any] | None = None,
    ) -> Any:
        log.debug("FunctionOperator executing: %s", self._fn.__name__)
        return self._fn(params, upstream_results or {})


def resolve_operator(
    node_operator: str | Callable[..., Any],
    tool_registry: ToolRegistry | None = None,
) -> Operator:
    """Factory: create the right Operator for a node's operator field.

    - If node_operator is a string: ToolOperator (tool lookup by name)
    - If node_operator is callable: FunctionOperator
    """
    if callable(node_operator):
        return FunctionOperator(node_operator)
    if isinstance(node_operator, str):
        if tool_registry is None:
            raise ValueError(
                "ToolRegistry required for string operator (tool name)"
            )
        return ToolOperator(tool_registry)
    raise TypeError(f"Unsupported operator type: {type(node_operator)}")
