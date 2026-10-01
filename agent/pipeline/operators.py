"""
TirraMind — Pipeline Operators

Operators bridge DAG nodes to actual execution — either calling a Tool
from the ToolRegistry or invoking a pure Python function.

ToolOperator:  Looks up a tool by name, calls tool.execute(**params), returns ToolResult.data
FunctionOperator:  Calls a Python callable with (params, upstream_results), returns its output

Neither operator swallows exceptions: a failing tool or function raises out of
``execute`` and the executor records a genuinely failed node. What operators
*do* own is the reverse translation — turning a returned *payload* back into a
node status (``classify_payload_status`` below), because a DAG function that
reports "I did nothing" inside its return dict is not a success.
"""

from __future__ import annotations

import inspect
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from agent.tools.base import ToolRegistry, ToolResult

log = logging.getLogger(__name__)

# ── reserved node-param keys ────────────────────────────────────────
# Directives to the executor/operator layer, never arguments to the
# underlying tool or DAG function. The executor injects ``__tool__`` and
# strips every reserved key before calling anything, so a node can carry
# executor metadata without every tool needing to accept it.
TOOL_PARAM_KEY = "__tool__"

#: Names of the domain tables a node claims to write. Read by
#: ``DAGExecutor``'s zero-rows guard, which takes real row-count deltas on
#: exactly these tables. ``pipeline_data`` is refused (see the executor):
#: it holds the per-node result *envelope*, written on every successful
#: node, which is precisely why the old guard could never fire —
#: docs/publications/thirteen_ways_a_pipeline_lies.md, Way 2.
DOMAIN_TABLES_PARAM_KEY = "__domain_tables__"

RESERVED_PARAM_KEYS = frozenset({TOOL_PARAM_KEY, DOMAIN_TABLES_PARAM_KEY})

# ── payload → node status ───────────────────────────────────────────
# Every layer 3-6 node is a FunctionOperator returning a plain dict, and
# the executor used to mark any non-raising operator "completed". So a
# node whose whole output was {"status": "skipped", "reason":
# "no_sac_model"} was recorded as a success. These are the literal status
# strings the DAG functions in agent/pipeline/dags/ actually emit
# (surveyed 2026-09-23: "skipped", "error", "completed", "ready",
# "completed_fully", "insufficient_data").
#
# Deliberately a closed vocabulary rather than "anything that isn't
# 'completed'": "ready" and "completed_fully" are success payloads, and
# silently reclassifying an unknown word would be the same guess-in-the-
# dark this whole class of bug is made of. An unrecognised status stays
# "completed" and keeps whatever meaning its DAG gave it.
_PAYLOAD_SKIP_STATUSES = frozenset({"skipped", "skip"})
_PAYLOAD_FAILURE_STATUSES = frozenset({"failed", "failure", "error"})


def classify_payload_status(payload: Any) -> str:
    """Map an operator's returned payload to a node status.

    Returns one of ``"completed"`` / ``"skipped"`` / ``"failed"``. Only a
    payload that *says* it skipped or failed is reclassified; everything
    else — including any non-mapping return value — stays ``"completed"``.
    """
    if not isinstance(payload, Mapping):
        return "completed"

    # ToolResult-shaped payloads.
    if payload.get("success") is False:
        return "failed"

    # Several DAG helpers use a bare boolean flag instead of a status
    # string (e.g. world_model_update's ``{"skipped": True, "reason": ...}``).
    if payload.get("skipped") is True:
        return "skipped"

    raw = payload.get("status")
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in _PAYLOAD_SKIP_STATUSES:
            return "skipped"
        if normalized in _PAYLOAD_FAILURE_STATUSES:
            return "failed"
    return "completed"


def payload_reason(payload: Any) -> str:
    """Best-effort human reason from a skipped/failed payload.

    Never invents one: returns ``"<no reason given>"`` when the payload
    carries none, so a node that skips without saying why is visibly
    under-reporting rather than looking like it explained itself.
    """
    if isinstance(payload, Mapping):
        for key in ("reason", "error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "<no reason given>"


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
        tool_name = params.get(TOOL_PARAM_KEY)
        if tool_name is None:
            raise ValueError(f"ToolOperator requires {TOOL_PARAM_KEY!r} in params")

        tool = self._registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool not found in registry: {tool_name!r}")

        # Build execution params without any executor directive. Tools take
        # their own kwargs only — a reserved key reaching tool.execute()
        # would be an unexpected-keyword TypeError.
        exec_params = {k: v for k, v in params.items() if k not in RESERVED_PARAM_KEYS}

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
