"""Tests for Pipeline Operators (ToolOperator, FunctionOperator, resolve_operator)."""

from __future__ import annotations

from typing import Any

import pytest

from agent.pipeline.operators import (
    FunctionOperator,
    ToolOperator,
    resolve_operator,
)
from agent.tools.base import Tool, ToolRegistry, ToolResult

# ── Mock tools ─────────────────────────────────────────────────


class MockTool(Tool):
    """A simple mock tool that returns whatever data is passed."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "Mock"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"key": {"type": "string"}}}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="ok", data={"received": kwargs})


class FailingTool(Tool):
    @property
    def name(self) -> str:
        return "failing_tool"

    @property
    def description(self) -> str:
        return "Always fails"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=False, output="Something went wrong")


class ExplodingTool(Tool):
    @property
    def name(self) -> str:
        return "exploding_tool"

    @property
    def description(self) -> str:
        return "Throws exception"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("Boom!")


class DataOnlyTool(Tool):
    """Returns success but data=None (only output text)."""

    @property
    def name(self) -> str:
        return "data_only"

    @property
    def description(self) -> str:
        return "No data field"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="text only", data=None)


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(MockTool())
    reg.register(FailingTool())
    reg.register(ExplodingTool())
    reg.register(DataOnlyTool())
    return reg


# ── ToolOperator ───────────────────────────────────────────────


class TestToolOperator:
    def test_basic_execution(self, registry):
        op = ToolOperator(registry)
        result = op.execute({"__tool__": "mock_tool", "key": "value"})
        assert result == {"received": {"key": "value"}}

    def test_missing_tool_name(self, registry):
        op = ToolOperator(registry)
        with pytest.raises(ValueError, match="__tool__"):
            op.execute({"key": "value"})

    def test_tool_not_in_registry(self, registry):
        op = ToolOperator(registry)
        with pytest.raises(ValueError, match="not found"):
            op.execute({"__tool__": "nonexistent"})

    def test_tool_returns_failure(self, registry):
        op = ToolOperator(registry)
        with pytest.raises(RuntimeError, match="failed"):
            op.execute({"__tool__": "failing_tool"})

    def test_tool_throws_exception(self, registry):
        op = ToolOperator(registry)
        with pytest.raises(RuntimeError, match="Boom!"):
            op.execute({"__tool__": "exploding_tool"})

    def test_tool_returns_none_data(self, registry):
        """If data is None, fall back to output string."""
        op = ToolOperator(registry)
        result = op.execute({"__tool__": "data_only"})
        assert result == "text only"

    def test_upstream_resolution(self, registry):
        op = ToolOperator(registry)
        upstream = {"prev_node": {"positions": [1, 2, 3]}}
        result = op.execute(
            {"__tool__": "mock_tool", "key": "$upstream.prev_node"},
            upstream_results=upstream,
        )
        assert result == {"received": {"key": {"positions": [1, 2, 3]}}}

    def test_upstream_ref_not_found(self, registry):
        op = ToolOperator(registry)
        with pytest.raises(ValueError, match="not found"):
            op.execute(
                {"__tool__": "mock_tool", "key": "$upstream.missing"},
                upstream_results={},
            )

    def test_no_upstream_resolve_for_plain_strings(self, registry):
        op = ToolOperator(registry)
        result = op.execute({"__tool__": "mock_tool", "key": "plain_string"})
        assert result == {"received": {"key": "plain_string"}}

    def test_multiple_params(self, registry):
        op = ToolOperator(registry)
        result = op.execute({"__tool__": "mock_tool", "a": "1", "b": "2"})
        assert result == {"received": {"a": "1", "b": "2"}}


# ── FunctionOperator ──────────────────────────────────────────


class TestFunctionOperator:
    def test_basic_function(self):
        def add_one(params, upstream):
            return params["x"] + 1

        op = FunctionOperator(add_one)
        result = op.execute({"x": 5})
        assert result == 6

    def test_function_with_upstream(self):
        def merge(params, upstream):
            return {**upstream.get("a", {}), **upstream.get("b", {})}

        op = FunctionOperator(merge)
        result = op.execute({}, upstream_results={"a": {"k1": 1}, "b": {"k2": 2}})
        assert result == {"k1": 1, "k2": 2}

    def test_function_returns_none(self):
        def noop(params, upstream):
            return None

        op = FunctionOperator(noop)
        result = op.execute({})
        assert result is None

    def test_function_throws(self):
        def broken(params, upstream):
            raise ValueError("bad input")

        op = FunctionOperator(broken)
        with pytest.raises(ValueError, match="bad input"):
            op.execute({})

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="callable"):
            FunctionOperator("not a function")

    def test_lambda_operator(self):
        op = FunctionOperator(lambda p, u: p.get("val", 0) * 2)
        assert op.execute({"val": 21}) == 42

    def test_empty_params_and_upstream(self):
        def identity(params, upstream):
            return {"params": params, "upstream": upstream}

        op = FunctionOperator(identity)
        result = op.execute({})
        assert result == {"params": {}, "upstream": {}}


# ── resolve_operator ──────────────────────────────────────────


class TestResolveOperator:
    def test_string_returns_tool_operator(self, registry):
        op = resolve_operator("cftc", tool_registry=registry)
        assert isinstance(op, ToolOperator)

    def test_callable_returns_function_operator(self):
        fn = lambda p, u: None  # noqa: E731
        op = resolve_operator(fn)
        assert isinstance(op, FunctionOperator)

    def test_string_without_registry_raises(self):
        with pytest.raises(ValueError, match="ToolRegistry required"):
            resolve_operator("cftc")

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            resolve_operator(42)  # type: ignore[arg-type]

    def test_class_method_as_callable(self):
        class Processor:
            @staticmethod
            def process(params, upstream):
                return params

        op = resolve_operator(Processor.process)
        assert isinstance(op, FunctionOperator)
        assert op.execute({"a": 1}) == {"a": 1}
