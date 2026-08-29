"""
TirraMind — Pipeline Operators

Operators bridge DAG nodes to actual execution — either calling a Tool
from the ToolRegistry or invoking a pure Python function.

ToolOperator:  Looks up a tool by name, calls tool.execute(**params), returns ToolResult.data
FunctionOperator:  Calls a Python callable with (params, upstream_results), returns its output

Both catch exceptions and return structured error info instead of crashing.
"""

from __future__ import annotations

import inspect
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from agent.tools.base import ToolRegistry, ToolResult

log = logging.getLogger(__name__)


class Operator(ABC):
    """Base class for pipeline operators."""

    @abstractmethod
    def execute(
        self,
        params: dict[str, Any],
        upstream_results: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        """Execute the operator. Returns result data or raises.

        ``cancel_event`` (LESSONS F-13): the executor sets this once the
        node's timeout has already fired, so the *caller* has stopped
        waiting on this call but the thread running it has not — Python
        cannot forcibly kill a running thread. Long-running operators
        SHOULD poll ``cancel_event.is_set()`` between chunks of work and
        return/raise early when set, so a "timed out" node actually stops
        doing work instead of just being ignored by the executor. Operators
        that don't check it behave exactly as before this signal existed —
        this parameter is additive, not a new requirement.
        """
        ...


class ToolOperator(Operator):
    """Executes a Tool from the ToolRegistry."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._registry = tool_registry

    def execute(
        self,
        params: dict[str, Any],
        upstream_results: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        # Tool.execute()'s contract (owned by the L1 data engineers) has no
        # cancellation parameter today — a single HTTP fetch is short enough
        # that this has never been the leak vector; the model/feature-builder
        # FunctionOperators are. Accepting and dropping cancel_event here
        # keeps the Operator interface uniform without forcing a change onto
        # every tool. Revisit only if a specific tool's own timeout becomes
        # the leak (that decision belongs to whoever owns that tool).
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
                ref_id = v[len("$upstream.") :]
                if ref_id not in upstream:
                    raise ValueError(f"Upstream reference '{v}' not found. Available: {list(upstream.keys())}")
                resolved[k] = upstream[ref_id]
            else:
                resolved[k] = v
        return resolved


class FunctionOperator(Operator):
    """Executes a pure Python callable.

    DAG node functions have historically had the signature
    ``fn(params, upstream_results) -> dict``. To wire cooperative
    cancellation (LESSONS F-13) through without breaking every existing
    node function, this operator inspects the callable's signature *once*
    at construction: if it declares a ``cancel_event`` parameter (or takes
    ``**kwargs``), the executor's cancellation ``threading.Event`` is
    forwarded; otherwise the call is made exactly as before. A function
    that ignores this is no worse off than before the parameter existed.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError(f"FunctionOperator requires a callable, got {type(fn)}")
        self._fn = fn
        try:
            sig = inspect.signature(fn)
            self._accepts_cancel_event = "cancel_event" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            # Builtins / C-extension callables without an inspectable
            # signature — fall back to the old, unconditional call shape.
            self._accepts_cancel_event = False

    def execute(
        self,
        params: dict[str, Any],
        upstream_results: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        log.debug("FunctionOperator executing: %s", self._fn.__name__)
        if self._accepts_cancel_event:
            return self._fn(params, upstream_results or {}, cancel_event=cancel_event)
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
            raise ValueError("ToolRegistry required for string operator (tool name)")
        return ToolOperator(tool_registry)
    raise TypeError(f"Unsupported operator type: {type(node_operator)}")
