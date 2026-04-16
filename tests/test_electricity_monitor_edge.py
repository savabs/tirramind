"""
Edge-case tests for ElectricityMonitorTool (7b-AD).

Coverage targets:
- Invalid / missing / boundary parameters
- Mode validation (demand, generation, interchange)
- API key handling (missing, present, env var)
- Demand: region, days clamping, hourly aggregation, cache
- Generation: fuel mix proportions, renewable/fossil share
- Interchange: bidirectional flows, net import/export
- HTTP errors, empty responses
- Helper functions (_safe_float, _fetch_eia, _aggregate_hourly, _fuel_mix_proportions)
- Integration: tool count = 47, arm count = 35
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.electricity_monitor import (
    EIA_FUEL_TYPES,
    KNOWN_REGIONS,
    VALID_MODES,
    ElectricityMonitorTool,
    _aggregate_hourly,
    _fetch_eia,
    _fuel_mix_proportions,
    _safe_float,
)
from agent.tools.base import ToolResult


# ── FIXTURES ──


@pytest.fixture
def tool():
    cache = MagicMock()
    cache.get.return_value = None
    t = ElectricityMonitorTool(cache=cache)
    t._api_key = "test-eia-key"
    return t


@pytest.fixture
def tool_no_key():
    cache = MagicMock()
    cache.get.return_value = None
    t = ElectricityMonitorTool(cache=cache)
    t._api_key = None
    return t


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


# ══════════════════════════════════════════════════════════════════════════
# 1. Mode validation
# ══════════════════════════════════════════════════════════════════════════


class TestModeValidation:
    def test_invalid_mode(self, tool):
        r = tool.execute(mode="invalid", region="PJM")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self, tool):
        r = tool.execute(mode="", region="PJM")
        assert not r.success

    def test_none_mode(self, tool):
        r = tool.execute(region="PJM")
        assert not r.success

    def test_valid_modes_constant(self):
        assert VALID_MODES == {"demand", "generation", "interchange"}

    def test_no_api_key(self, tool_no_key):
        r = tool_no_key.execute(mode="demand", region="PJM")
        assert not r.success
        assert "API key" in r.output

    def test_missing_region(self, tool):
        r = tool.execute(mode="demand")
        assert not r.success
        assert "region" in r.output.lower()

    def test_empty_region(self, tool):
        r = tool.execute(mode="demand", region="")
        assert not r.success

    def test_region_uppercased(self, tool):
        with patch("agent.tools.electricity_monitor._fetch_eia", return_value=[]):
            r = tool.execute(mode="demand", region="pjm")
            assert r.success  # region gets uppercased internally


# ══════════════════════════════════════════════════════════════════════════
# 2. Demand mode
# ══════════════════════════════════════════════════════════════════════════


class TestDemandMode:
    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_fetch_failure(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="demand", region="PJM")
        assert not r.success
        assert "Failed" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_empty_records(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="demand", region="PJM")
        assert r.success
        assert "No demand data" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_successful_demand(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {"period": "2024-03-15T10", "value": 35000, "type-name": "Demand"},
            {"period": "2024-03-15T11", "value": 38000, "type-name": "Demand"},
            {"period": "2024-03-15T12", "value": 32000, "type-name": "Demand"},
        ]
        r = tool.execute(mode="demand", region="PJM")
        assert r.success
        assert "Electricity Demand" in r.output
        assert "Peak" in r.output
        assert "38,000" in r.output or "38000" in r.output
        tool._cache.put.assert_called_once()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = "cached demand"
        r = tool.execute(mode="demand", region="PJM")
        assert r.success
        assert r.output == "cached demand"
        mock_fetch.assert_not_called()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_days_clamped_max(self, mock_fetch, tool):
        mock_fetch.return_value = []
        tool.execute(mode="demand", region="PJM", days=99)
        assert mock_fetch.called

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_days_non_numeric(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="demand", region="PJM", days="abc")
        assert r.success

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_known_region_name(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {"period": "2024-03-15T10", "value": 50000, "type-name": "Demand"},
        ]
        r = tool.execute(mode="demand", region="PJM")
        assert "PJM Interconnection" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_unknown_region(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {"period": "2024-03-15T10", "value": 50000, "type-name": "Demand"},
        ]
        r = tool.execute(mode="demand", region="XXXX")
        assert r.success
        assert "XXXX" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_no_type_match(self, mock_fetch, tool):
        """Records without type-name should still be used."""
        mock_fetch.return_value = [
            {"period": "2024-03-15T10", "value": 40000},
        ]
        r = tool.execute(mode="demand", region="PJM")
        assert r.success


# ══════════════════════════════════════════════════════════════════════════
# 3. Generation mode
# ══════════════════════════════════════════════════════════════════════════


class TestGenerationMode:
    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_fetch_failure(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="generation", region="CISO")
        assert not r.success

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_empty_records(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="generation", region="CISO")
        assert r.success
        assert "No generation data" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_successful_generation(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {"fueltype": "NG", "value": 10000},
            {"fueltype": "SUN", "value": 5000},
            {"fueltype": "WND", "value": 3000},
            {"fueltype": "NUC", "value": 7000},
        ]
        r = tool.execute(mode="generation", region="CISO")
        assert r.success
        assert "Generation Mix" in r.output
        assert "Renewable share" in r.output
        assert "Fossil share" in r.output
        tool._cache.put.assert_called_once()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_generation_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = "cached gen"
        r = tool.execute(mode="generation", region="CISO")
        assert r.success
        mock_fetch.assert_not_called()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_generation_fuel_labels(self, mock_fetch, tool):
        mock_fetch.return_value = [
            {"fueltype": "COL", "value": 8000},
            {"fueltype": "WAT", "value": 4000},
        ]
        r = tool.execute(mode="generation", region="PJM")
        assert r.success
        assert "Coal" in r.output
        assert "Hydro" in r.output


# ══════════════════════════════════════════════════════════════════════════
# 4. Interchange mode
# ══════════════════════════════════════════════════════════════════════════


class TestInterchangeMode:
    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_both_fetch_failure(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="interchange", region="PJM")
        assert not r.success
        assert "Failed" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_one_direction_success(self, mock_fetch, tool):
        # First call (from) returns data, second (to) returns None
        mock_fetch.side_effect = [
            [{"toba": "NYIS", "value": 1000}],
            None,
        ]
        r = tool.execute(mode="interchange", region="PJM")
        assert r.success
        assert "Interchange Flows" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_both_directions(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            [  # exports
                {"toba": "NYIS", "value": 5000},
                {"toba": "MISO", "value": 3000},
            ],
            [  # imports
                {"fromba": "CPLE", "value": 4000},
            ],
        ]
        r = tool.execute(mode="interchange", region="PJM")
        assert r.success
        assert "Exports" in r.output or "export" in r.output
        assert "Imports" in r.output or "import" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_net_import(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            [{"toba": "X", "value": 1000}],  # exports
            [{"fromba": "Y", "value": 5000}],  # imports
        ]
        r = tool.execute(mode="interchange", region="PJM")
        assert r.success
        assert "import" in r.output.lower()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_net_export(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            [{"toba": "X", "value": 5000}],
            [{"fromba": "Y", "value": 1000}],
        ]
        r = tool.execute(mode="interchange", region="PJM")
        assert r.success
        assert "export" in r.output.lower()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_interchange_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = "cached interchange"
        r = tool.execute(mode="interchange", region="PJM")
        assert r.success
        mock_fetch.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# 5. Helper functions
# ══════════════════════════════════════════════════════════════════════════


class TestHelpers:
    # -- _safe_float --
    def test_safe_float_normal(self):
        assert _safe_float("42.5") == 42.5

    def test_safe_float_bad(self):
        assert _safe_float("abc") == 0.0

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    # -- _aggregate_hourly --
    def test_aggregate_empty(self):
        result = _aggregate_hourly([])
        assert result["hours"] == 0
        assert result["peak_mw"] == 0

    def test_aggregate_normal(self):
        records = [{"value": 100}, {"value": 200}, {"value": 300}]
        result = _aggregate_hourly(records)
        assert result["peak_mw"] == 300
        assert result["trough_mw"] == 100
        assert result["avg_mw"] == 200
        assert result["hours"] == 3

    def test_aggregate_zero_values(self):
        records = [{"value": 0}, {"value": 0}]
        result = _aggregate_hourly(records)
        assert result["hours"] == 0  # zero values filtered out

    def test_aggregate_mixed(self):
        records = [{"value": 0}, {"value": 500}, {"value": "bad"}, {"value": 1000}]
        result = _aggregate_hourly(records)
        assert result["peak_mw"] == 1000
        assert result["trough_mw"] == 500
        assert result["hours"] == 2

    # -- _fuel_mix_proportions --
    def test_fuel_mix_empty(self):
        assert _fuel_mix_proportions([]) == {}

    def test_fuel_mix_single(self):
        records = [{"fueltype": "SUN", "value": 1000}]
        result = _fuel_mix_proportions(records)
        assert "Solar" in result
        assert result["Solar"]["share_pct"] == 100.0

    def test_fuel_mix_multiple(self):
        records = [
            {"fueltype": "SUN", "value": 3000},
            {"fueltype": "WND", "value": 3000},
            {"fueltype": "NG", "value": 4000},
        ]
        result = _fuel_mix_proportions(records)
        assert "_summary" in result
        assert result["_summary"]["renewable_pct"] == 60.0
        assert result["_summary"]["fossil_pct"] == 40.0

    def test_fuel_mix_zero_values_ignored(self):
        records = [
            {"fueltype": "SUN", "value": 1000},
            {"fueltype": "COL", "value": 0},
        ]
        result = _fuel_mix_proportions(records)
        assert "Coal" not in result

    def test_fuel_mix_unknown_fuel(self):
        records = [{"fueltype": "ZZZ", "value": 1000}]
        result = _fuel_mix_proportions(records)
        assert "ZZZ" in result


# ══════════════════════════════════════════════════════════════════════════
# 6. Fetch EIA function
# ══════════════════════════════════════════════════════════════════════════


class TestFetchEia:
    @patch("agent.tools.electricity_monitor.httpx.get")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response(
            200, json_data={"response": {"data": [{"value": 42}]}}
        )
        result = _fetch_eia("electricity/rto/region-data", None, "key")
        assert result is not None
        assert len(result) == 1

    @patch("agent.tools.electricity_monitor.httpx.get")
    def test_http_error(self, mock_get):
        mock_get.return_value = _mock_response(500)
        result = _fetch_eia("electricity/rto/region-data", None, "key")
        assert result is None

    @patch("agent.tools.electricity_monitor.httpx.get")
    def test_network_error(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.ConnectError("timeout")
        result = _fetch_eia("electricity/rto/region-data", None, "key")
        assert result is None

    @patch("agent.tools.electricity_monitor.httpx.get")
    def test_with_facets(self, mock_get):
        mock_get.return_value = _mock_response(
            200, json_data={"response": {"data": []}}
        )
        result = _fetch_eia("test", {"respondent": ["PJM"]}, "key")
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════
# 7. API key handling
# ══════════════════════════════════════════════════════════════════════════


class TestApiKey:
    def test_key_from_env(self):
        with patch.dict(os.environ, {"TIRRA_EIA_API_KEY": "my-eia-key"}):
            t = ElectricityMonitorTool()
            assert t._api_key == "my-eia-key"

    def test_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TIRRA_EIA_API_KEY", None)
            t = ElectricityMonitorTool()
            assert t._api_key is None

    def test_empty_key(self):
        with patch.dict(os.environ, {"TIRRA_EIA_API_KEY": "  "}):
            t = ElectricityMonitorTool()
            assert t._api_key is None


# ══════════════════════════════════════════════════════════════════════════
# 8. Tool metadata
# ══════════════════════════════════════════════════════════════════════════


class TestToolMetadata:
    def test_name(self):
        t = ElectricityMonitorTool()
        assert t.name == "electricity_monitor"

    def test_description(self):
        t = ElectricityMonitorTool()
        assert "demand" in t.description.lower()
        assert "generation" in t.description.lower()
        assert "interchange" in t.description.lower()

    def test_required_params(self):
        t = ElectricityMonitorTool()
        assert "mode" in t.parameters.get("required", [])
        assert "region" in t.parameters.get("required", [])


# ══════════════════════════════════════════════════════════════════════════
# 9. Integration counts
# ══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_tool_count(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        names = registry.list_names()
        assert len(names) == 60, f"Expected 60, got {len(names)}: {names}"

    def test_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48, f"Expected 48, got {len(DEFAULT_ARMS)}"

    def test_electricity_registered(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert "electricity_monitor" in registry.list_names()

    def test_electricity_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "electricity_demand" in arm_names
