"""Test that build_tool_registry wires PipelineStore to all tools that accept it."""

from __future__ import annotations

import inspect
from unittest.mock import patch

from agent.cli import build_tool_registry

# Tools that do NOT accept pipeline_store in __init__
EXCLUDED_TOOLS = {
    "web_search",
    "web_browse",
    "code_executor",
    "shell_runner",
    "file_read",
    "file_write",
    "list_directory",
    "market_data",
    "macro_data",
    "liquidity_regime",
    "backtest",
    "polymarket_whales",
    "power_grid",
    "weather_alerts",
    "earthquake_proximity",
    "pipeline_query",
    "job_postings",
    "building_permits",
    "satellite_activity",
    "labor_disruptions",
    "energy_supply",
    "treasury_receipts",
    "internet_infrastructure",
}


@patch("agent.pipeline.store.PipelineStore.__init__", return_value=None)
def test_all_tools_with_pipeline_store_param_receive_it(mock_init):
    """Every tool whose __init__ accepts pipeline_store must receive a non-None value."""
    registry = build_tool_registry()

    for name, tool in registry._tools.items():
        if name in EXCLUDED_TOOLS:
            continue
        # Check if the tool class __init__ accepts pipeline_store
        sig = inspect.signature(type(tool).__init__)
        if "pipeline_store" in sig.parameters:
            store_attr = getattr(tool, "_store", None) or getattr(tool, "pipeline_store", None)
            assert store_attr is not None, (
                f"Tool '{name}' ({type(tool).__name__}) accepts pipeline_store "
                f"but received None — wiring bug in build_tool_registry()"
            )


@patch("agent.pipeline.store.PipelineStore.__init__", return_value=None)
def test_all_tools_share_same_pipeline_store(mock_init):
    """All wired tools must share the same PipelineStore instance."""
    registry = build_tool_registry()

    stores = set()
    for name, tool in registry._tools.items():
        if name in EXCLUDED_TOOLS:
            continue
        sig = inspect.signature(type(tool).__init__)
        if "pipeline_store" in sig.parameters:
            store = getattr(tool, "_store", None) or getattr(tool, "pipeline_store", None)
            if store is not None:
                stores.add(id(store))

    assert len(stores) == 1, (
        f"Expected all tools to share one PipelineStore, but found {len(stores)} distinct instances"
    )
