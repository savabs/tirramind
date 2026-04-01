"""
Tests for PipelineQueryTool (Step 7.8).

Covers: mode validation, data queries, signal queries, run queries,
relative time parsing, limit clamping, empty results, output formatting,
CLI registration, and integration with PipelineStore.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from agent.pipeline.store import PipelineStore
from agent.tools.pipeline_query import PipelineQueryTool, _parse_relative_time


# ── Helpers ───────────────────────────────────────────────────


@pytest.fixture
def store():
    """In-memory PipelineStore for testing."""
    return PipelineStore(":memory:")


@pytest.fixture
def tool(store):
    return PipelineQueryTool(store=store)


def _seed_data(store: PipelineStore, source: str = "cftc", n: int = 3):
    """Insert n rows of test data for the given source."""
    for i in range(n):
        store.store_data(source, {"mode": "latest", "i": i}, {"value": i * 10})


def _seed_signals(store: PipelineStore, name: str = "momentum", n: int = 3):
    """Insert n signal values."""
    for i in range(n):
        store.store_signal(name, float(i), {"detail": f"test_{i}"})


def _seed_runs(store: PipelineStore, dag_name: str = "daily_collection", n: int = 3):
    """Insert n completed DAG runs."""
    for i in range(n):
        run_id = store.new_run_id()
        store.record_run_start(dag_name, trigger="scheduled", run_id=run_id)
        store.record_run_end(run_id, "completed", {"node_a": {"status": "completed"}})


# ── Tool Metadata ─────────────────────────────────────────────


class TestToolMetadata:
    def test_name(self, tool):
        assert tool.name == "pipeline_query"

    def test_description_nonempty(self, tool):
        assert len(tool.description) > 20

    def test_parameters_has_mode(self, tool):
        props = tool.parameters["properties"]
        assert "mode" in props
        assert props["mode"]["enum"] == ["data", "signals", "runs"]

    def test_parameters_required_mode(self, tool):
        assert "mode" in tool.parameters["required"]

    def test_openai_schema(self, tool):
        schema = tool.to_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "pipeline_query"


# ── Mode Validation ───────────────────────────────────────────


class TestModeValidation:
    def test_invalid_mode(self, tool):
        result = tool.execute(mode="garbage")
        assert not result.success
        assert "Invalid mode" in result.output

    def test_empty_mode(self, tool):
        result = tool.execute(mode="")
        assert not result.success

    def test_valid_modes(self, tool):
        # data mode needs source, signals needs signal_name, runs works as-is
        r = tool.execute(mode="runs")
        assert r.success


# ── Data Mode ─────────────────────────────────────────────────


class TestDataMode:
    def test_requires_source(self, tool):
        result = tool.execute(mode="data")
        assert not result.success
        assert "source" in result.output.lower()

    def test_empty_source(self, tool):
        result = tool.execute(mode="data", source="")
        assert not result.success

    def test_query_with_data(self, store, tool):
        _seed_data(store, "cftc", 5)
        result = tool.execute(mode="data", source="cftc")
        assert result.success
        assert result.data["count"] == 5
        assert len(result.data["rows"]) == 5
        assert "cftc" in result.output

    def test_query_no_data(self, tool):
        result = tool.execute(mode="data", source="nonexistent")
        assert result.success  # Empty is not an error
        assert result.data["count"] == 0
        assert "No data found" in result.output

    def test_limit(self, store, tool):
        _seed_data(store, "gdelt", 50)
        result = tool.execute(mode="data", source="gdelt", limit=10)
        assert result.data["count"] == 10

    def test_limit_clamped_max(self, store, tool):
        _seed_data(store, "x", 3)
        result = tool.execute(mode="data", source="x", limit=9999)
        # Clamped to 500, but only 3 rows exist
        assert result.data["count"] == 3

    def test_limit_clamped_min(self, store, tool):
        _seed_data(store, "x", 3)
        result = tool.execute(mode="data", source="x", limit=-5)
        assert result.data["count"] >= 1  # At least 1

    def test_output_text_shows_preview(self, store, tool):
        _seed_data(store, "cftc", 3)
        result = tool.execute(mode="data", source="cftc")
        assert "Pipeline data for 'cftc'" in result.output

    def test_output_text_truncates(self, store, tool):
        _seed_data(store, "x", 10)
        result = tool.execute(mode="data", source="x")
        assert "more rows" in result.output

    def test_data_contains_structured_rows(self, store, tool):
        store.store_data("test_src", {"k": "v"}, {"payload": 42})
        result = tool.execute(mode="data", source="test_src")
        row = result.data["rows"][0]
        assert row["data"]["payload"] == 42
        assert row["params"]["k"] == "v"

    def test_source_isolation(self, store, tool):
        _seed_data(store, "alpha", 3)
        _seed_data(store, "beta", 5)
        r1 = tool.execute(mode="data", source="alpha")
        r2 = tool.execute(mode="data", source="beta")
        assert r1.data["count"] == 3
        assert r2.data["count"] == 5


# ── Signals Mode ──────────────────────────────────────────────


class TestSignalsMode:
    def test_requires_signal_name(self, tool):
        result = tool.execute(mode="signals")
        assert not result.success
        assert "signal_name" in result.output

    def test_empty_signal_name(self, tool):
        result = tool.execute(mode="signals", signal_name="")
        assert not result.success

    def test_query_with_signals(self, store, tool):
        _seed_signals(store, "momentum", 5)
        result = tool.execute(mode="signals", signal_name="momentum")
        assert result.success
        assert result.data["count"] == 5
        assert "momentum" in result.output

    def test_query_no_signals(self, tool):
        result = tool.execute(mode="signals", signal_name="ghost")
        assert result.success
        assert result.data["count"] == 0

    def test_signal_values_in_rows(self, store, tool):
        store.store_signal("vol", 1.5, {"src": "test"})
        result = tool.execute(mode="signals", signal_name="vol")
        row = result.data["rows"][0]
        assert row["value"] == 1.5
        assert row["metadata"]["src"] == "test"

    def test_limit_signals(self, store, tool):
        _seed_signals(store, "s", 20)
        result = tool.execute(mode="signals", signal_name="s", limit=5)
        assert result.data["count"] == 5


# ── Runs Mode ─────────────────────────────────────────────────


class TestRunsMode:
    def test_query_all_runs(self, store, tool):
        _seed_runs(store, "daily", 3)
        result = tool.execute(mode="runs")
        assert result.success
        assert result.data["count"] == 3

    def test_query_no_runs(self, tool):
        result = tool.execute(mode="runs")
        assert result.success
        assert result.data["count"] == 0
        assert "No pipeline runs" in result.output

    def test_filter_by_dag_name(self, store, tool):
        _seed_runs(store, "daily", 3)
        _seed_runs(store, "weekly", 2)
        result = tool.execute(mode="runs", dag_name="weekly")
        assert result.data["count"] == 2

    def test_runs_output_format(self, store, tool):
        _seed_runs(store, "daily", 1)
        result = tool.execute(mode="runs")
        assert "Pipeline runs:" in result.output
        assert "daily" in result.output

    def test_runs_contain_status(self, store, tool):
        _seed_runs(store, "d", 1)
        result = tool.execute(mode="runs")
        run = result.data["runs"][0]
        assert run["status"] == "completed"

    def test_limit_runs(self, store, tool):
        _seed_runs(store, "d", 10)
        result = tool.execute(mode="runs", limit=3)
        assert result.data["count"] == 3


# ── Relative Time Parsing ────────────────────────────────────


class TestRelativeTimeParsing:
    def test_parse_hours(self):
        ts = _parse_relative_time("24h")
        assert ts is not None
        assert abs(ts - (time.time() - 86400)) < 2

    def test_parse_days(self):
        ts = _parse_relative_time("7d")
        assert ts is not None
        assert abs(ts - (time.time() - 604800)) < 2

    def test_parse_weeks(self):
        ts = _parse_relative_time("2w")
        assert ts is not None
        assert abs(ts - (time.time() - 1209600)) < 2

    def test_parse_empty(self):
        assert _parse_relative_time("") is None

    def test_parse_none_like(self):
        assert _parse_relative_time("   ") is None

    def test_parse_invalid(self):
        assert _parse_relative_time("abc") is None
        assert _parse_relative_time("10x") is None

    def test_parse_no_number(self):
        assert _parse_relative_time("d") is None

    def test_fractional(self):
        ts = _parse_relative_time("1.5h")
        assert ts is not None
        assert abs(ts - (time.time() - 5400)) < 2

    def test_since_filters_data(self, store, tool):
        # Insert old and new data
        store.store_data("src", {}, {"old": True})
        time.sleep(0.05)
        store.store_data("src", {}, {"new": True})
        # Since 1h should include both (they're seconds old)
        result = tool.execute(mode="data", source="src", since="1h")
        assert result.data["count"] == 2


# ── Edge Cases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_extra_kwargs_ignored(self, tool):
        # Extra unknown params don't crash
        result = tool.execute(mode="runs", bogus="whatever")
        assert result.success

    def test_limit_zero_becomes_one(self, store, tool):
        _seed_data(store, "x", 5)
        result = tool.execute(mode="data", source="x", limit=0)
        assert result.data["count"] >= 1

    def test_limit_string_cast(self, store, tool):
        _seed_data(store, "x", 5)
        result = tool.execute(mode="data", source="x", limit="3")
        assert result.data["count"] == 3

    def test_source_whitespace_stripped(self, store, tool):
        _seed_data(store, "cftc", 2)
        result = tool.execute(mode="data", source="  cftc  ")
        assert result.data["count"] == 2

    def test_signal_name_whitespace_stripped(self, store, tool):
        _seed_signals(store, "vol", 2)
        result = tool.execute(mode="signals", signal_name="  vol  ")
        assert result.data["count"] == 2


# ── CLI Registration ──────────────────────────────────────────


class TestCLIRegistration:
    def test_pipeline_query_in_registry(self):
        """Verify pipeline_query tool is registered in build_tool_registry."""
        pytest.importorskip("hmmlearn")
        from agent.config.settings import AgentConfig
        from agent.cli import build_tool_registry

        config = AgentConfig()
        registry = build_tool_registry(config)
        assert "pipeline_query" in registry.list_names()

    def test_pipeline_query_tool_instance(self):
        pytest.importorskip("hmmlearn")
        from agent.config.settings import AgentConfig
        from agent.cli import build_tool_registry

        config = AgentConfig()
        registry = build_tool_registry(config)
        tool = registry.get("pipeline_query")
        assert tool is not None
        assert isinstance(tool, PipelineQueryTool)
