"""
Edge-case tests for CapitalFlowsTool (7b-AC).

Coverage targets:
- Invalid / missing / boundary parameters
- FRED API key requirement
- Empty / malformed API responses
- Cache hit / miss paths
- Holdings: coordinated selling/buying detection, MoM change
- Flows: reversal detection, average calculation
- Reserves: stress detection, drawdown thresholds
- Country filtering (valid/invalid)
- Period validation
- HTTP errors, timeouts
- Mode validation
- Helper functions (_pct_change, _detect_coordinated, _reserve_stress)
- Integration: tool count = 44, arm count = 32
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.capital_flows import (
    VALID_MODES,
    CapitalFlowsTool,
    _detect_coordinated,
    _fetch_fred,
    _latest,
    _pct_change,
    _reserve_stress,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    cache = MagicMock()
    cache.get.return_value = None
    return CapitalFlowsTool(fred_api_key="test-key", cache=cache)


@pytest.fixture
def tool_no_key():
    cache = MagicMock()
    cache.get.return_value = None
    return CapitalFlowsTool(fred_api_key="", cache=cache)


@pytest.fixture
def tool_no_cache():
    return CapitalFlowsTool(fred_api_key="test-key", cache=None)


def _obs(date: str = "2025-01-01", value: str = "1000") -> dict[str, str]:
    return {"date": date, "value": value}


def _fred_response(observations: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {"observations": observations or []}


def _mock_response(data: dict, status: int = 200) -> httpx.Response:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.json.return_value = data
    r.raise_for_status = MagicMock()
    if status >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=r)
    return r


# ═══════════════════════════════════════════════════════════════════════════
# 1. Mode validation
# ═══════════════════════════════════════════════════════════════════════════


class TestModeValidation:
    def test_valid_modes(self, tool):
        assert {"holdings", "flows", "reserves"} == VALID_MODES

    def test_empty_mode(self, tool):
        r = tool.execute(mode="")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_invalid_mode(self, tool):
        r = tool.execute(mode="bogus")
        assert not r.success
        assert "bogus" in r.output

    def test_none_mode(self, tool):
        r = tool.execute(mode=None)
        assert not r.success

    def test_mode_case_insensitive(self, tool):
        # Should normalize to lowercase
        r = tool.execute(mode="HOLDINGS")
        assert "Invalid mode" not in r.output

    def test_mode_whitespace(self, tool):
        r = tool.execute(mode="  holdings  ")
        assert "Invalid mode" not in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 2. FRED API key requirement
# ═══════════════════════════════════════════════════════════════════════════


class TestAPIKeyRequirement:
    def test_holdings_requires_key(self, tool_no_key):
        r = tool_no_key.execute(mode="holdings")
        assert not r.success
        assert "FRED" in r.output

    def test_flows_requires_key(self, tool_no_key):
        r = tool_no_key.execute(mode="flows")
        assert not r.success

    def test_reserves_requires_key(self, tool_no_key):
        r = tool_no_key.execute(mode="reserves")
        assert not r.success


# ═══════════════════════════════════════════════════════════════════════════
# 3. Helper functions
# ═══════════════════════════════════════════════════════════════════════════


class TestHelpers:
    # -- _pct_change --
    def test_pct_change_positive(self):
        assert _pct_change(100, 110) == pytest.approx(10.0)

    def test_pct_change_negative(self):
        assert _pct_change(100, 90) == pytest.approx(-10.0)

    def test_pct_change_zero_old(self):
        assert _pct_change(0, 100) is None

    def test_pct_change_no_change(self):
        assert _pct_change(100, 100) == pytest.approx(0.0)

    def test_pct_change_negative_to_positive(self):
        result = _pct_change(-100, 50)
        assert result is not None

    # -- _latest --
    def test_latest_empty(self):
        assert _latest([]) is None

    def test_latest_single(self):
        obs = [_obs("2025-01-01", "500")]
        assert _latest(obs) == obs[0]

    def test_latest_multiple(self):
        obs = [_obs("2025-01-01", "500"), _obs("2025-02-01", "600")]
        assert _latest(obs)["value"] == "600"

    # -- _detect_coordinated --
    def test_coordinated_selling_detected(self):
        changes = {"Japan": -5.0, "China": -3.0, "UK": 1.0}
        result = _detect_coordinated(changes)
        assert result["coordinated_selling"] is True
        assert len(result["sellers"]) == 2

    def test_coordinated_buying_detected(self):
        changes = {"Japan": 5.0, "China": 3.0, "UK": -1.0}
        result = _detect_coordinated(changes)
        assert result["coordinated_buying"] is True
        assert len(result["buyers"]) == 2

    def test_no_coordination(self):
        changes = {"Japan": -1.0, "China": 1.0, "UK": 0.5}
        result = _detect_coordinated(changes)
        assert result["coordinated_selling"] is False
        assert result["coordinated_buying"] is False

    def test_coordinated_all_none(self):
        changes = {"Japan": None, "China": None}
        result = _detect_coordinated(changes)
        assert result["coordinated_selling"] is False
        assert result["coordinated_buying"] is False
        assert result["sellers"] == []
        assert result["buyers"] == []

    def test_coordinated_custom_threshold(self):
        changes = {"Japan": -1.5, "China": -1.5}
        # With default threshold=-2.0, these don't count
        result = _detect_coordinated(changes, threshold=-2.0)
        assert result["coordinated_selling"] is False
        # With threshold=-1.0, they do
        result = _detect_coordinated(changes, threshold=-1.0)
        assert result["coordinated_selling"] is True

    # -- _reserve_stress --
    def test_reserve_stress_insufficient_data(self):
        result = _reserve_stress([_obs()])
        assert result["stress"] is False
        assert result["drawdown_pct"] is None

    def test_reserve_stress_no_stress(self):
        obs = [
            _obs("2025-01-01", "100"),
            _obs("2025-02-01", "101"),
            _obs("2025-03-01", "102"),
            _obs("2025-04-01", "103"),
        ]
        result = _reserve_stress(obs)
        assert result["stress"] is False

    def test_reserve_stress_detected(self):
        obs = [
            _obs("2025-01-01", "100"),
            _obs("2025-02-01", "95"),
            _obs("2025-03-01", "92"),
            _obs("2025-04-01", "90"),
        ]
        result = _reserve_stress(obs)
        assert result["stress"] is True
        assert result["drawdown_pct"] < -5.0

    def test_reserve_stress_empty(self):
        result = _reserve_stress([])
        assert result["stress"] is False


# ═══════════════════════════════════════════════════════════════════════════
# 4. Holdings mode
# ═══════════════════════════════════════════════════════════════════════════


class TestHoldingsMode:
    @patch("agent.tools.capital_flows._fetch_fred")
    def test_holdings_basic(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _obs("2025-01-01", "1100"),
            _obs("2025-02-01", "1150"),
        ]
        r = tool.execute(mode="holdings")
        assert r.success
        assert r.data["mode"] == "holdings"
        assert "holdings" in r.data

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_holdings_country_filter(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs("2025-01-01", "900")]
        r = tool.execute(mode="holdings", country="japan")
        assert r.success
        assert len(r.data["holdings"]) == 1

    def test_holdings_invalid_country(self, tool):
        r = tool.execute(mode="holdings", country="atlantis")
        assert not r.success
        assert "Unknown country" in r.output

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_holdings_empty_data(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="holdings")
        # No results but still returns success=False (no data)
        assert not r.success or r.data["holdings"] == []

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_holdings_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = [_obs("2025-01-01", "1200")]
        r = tool.execute(mode="holdings", country="japan")
        assert r.success
        mock_fetch.assert_not_called()

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_holdings_coordinated_selling(self, mock_fetch, tool):
        # Return declining values for each country
        def side_effect(series_id, *args, **kwargs):
            return [_obs("2025-01-01", "1000"), _obs("2025-02-01", "900")]

        mock_fetch.side_effect = side_effect
        r = tool.execute(mode="holdings")
        assert r.success
        assert r.data["coordination"] is not None

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_holdings_selling_flag(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _obs("2025-01-01", "1000"),
            _obs("2025-02-01", "900"),
        ]
        r = tool.execute(mode="holdings", country="japan")
        assert r.success
        assert "SELLING" in r.output

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_holdings_buying_flag(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _obs("2025-01-01", "1000"),
            _obs("2025-02-01", "1100"),
        ]
        r = tool.execute(mode="holdings", country="japan")
        assert r.success
        assert "BUYING" in r.output


# ═══════════════════════════════════════════════════════════════════════════
# 5. Flows mode
# ═══════════════════════════════════════════════════════════════════════════


class TestFlowsMode:
    @patch("agent.tools.capital_flows._fetch_fred")
    def test_flows_basic(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs("2025-01-01", "5000")]
        r = tool.execute(mode="flows")
        assert r.success
        assert r.data["mode"] == "flows"

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_flows_reversal_detection(self, mock_fetch, tool):
        # Sign change from positive to negative
        obs = [_obs(f"2025-0{i}-01", str(val)) for i, val in enumerate([100, 200, 300, -50, -100, -200], 1)]
        mock_fetch.return_value = obs
        r = tool.execute(mode="flows")
        assert r.success
        # At least one flow series should show reversal
        flows_data = r.data["flows"]
        has_reversal = any(f["flow_reversal"] for f in flows_data)
        assert has_reversal

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_flows_empty_data(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="flows")
        # No data for any series
        assert not r.success

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_flows_outflow_label(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs("2025-01-01", "-5000")]
        r = tool.execute(mode="flows")
        assert r.success
        assert "outflow" in r.output

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_flows_inflow_label(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs("2025-01-01", "5000")]
        r = tool.execute(mode="flows")
        assert r.success
        assert "inflow" in r.output

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_flows_cache_write(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs()]
        tool.execute(mode="flows")
        tool._cache.put.assert_called()

    def test_flows_cache_hit(self, tool):
        tool._cache.get.return_value = [_obs()]
        r = tool.execute(mode="flows")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 6. Reserves mode
# ═══════════════════════════════════════════════════════════════════════════


class TestReservesMode:
    @patch("agent.tools.capital_flows._fetch_fred")
    def test_reserves_basic(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs("2025-01-01", "3200000000000")]
        r = tool.execute(mode="reserves")
        assert r.success
        assert r.data["mode"] == "reserves"

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_reserves_stress_alert(self, mock_fetch, tool):
        obs = [
            _obs("2025-01-01", "1000"),
            _obs("2025-02-01", "950"),
            _obs("2025-03-01", "920"),
            _obs("2025-04-01", "900"),
        ]
        mock_fetch.return_value = obs
        r = tool.execute(mode="reserves")
        assert r.success
        assert "STRESS" in r.output
        assert len(r.data["stress_alerts"]) > 0

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_reserves_country_filter(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs()]
        r = tool.execute(mode="reserves", country="china")
        assert r.success

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_reserves_empty(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="reserves")
        assert not r.success

    def test_reserves_cache_hit(self, tool):
        tool._cache.get.return_value = [_obs(), _obs("2025-02-01", "1050")]
        r = tool.execute(mode="reserves")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 7. Period validation
# ═══════════════════════════════════════════════════════════════════════════


class TestPeriodValidation:
    @patch("agent.tools.capital_flows._fetch_fred")
    def test_valid_period(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs()]
        r = tool.execute(mode="holdings", country="japan", period="5y")
        assert r.success

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_invalid_period_defaults(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs()]
        r = tool.execute(mode="holdings", country="japan", period="99z")
        assert r.success  # Falls back to "2y"

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_empty_period_defaults(self, mock_fetch, tool):
        mock_fetch.return_value = [_obs()]
        r = tool.execute(mode="holdings", country="japan", period="")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 8. No-cache paths
# ═══════════════════════════════════════════════════════════════════════════


class TestNoCache:
    @patch("agent.tools.capital_flows._fetch_fred")
    def test_holdings_no_cache(self, mock_fetch, tool_no_cache):
        mock_fetch.return_value = [_obs()]
        r = tool_no_cache.execute(mode="holdings", country="japan")
        assert r.success

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_flows_no_cache(self, mock_fetch, tool_no_cache):
        mock_fetch.return_value = [_obs()]
        r = tool_no_cache.execute(mode="flows")
        assert r.success

    @patch("agent.tools.capital_flows._fetch_fred")
    def test_reserves_no_cache(self, mock_fetch, tool_no_cache):
        mock_fetch.return_value = [_obs()]
        r = tool_no_cache.execute(mode="reserves")
        assert r.success


# ═══════════════════════════════════════════════════════════════════════════
# 9. FRED fetch helper
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchFred:
    @patch("agent.tools.capital_flows.httpx.Client")
    def test_fetch_basic(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(_fred_response([_obs("2025-01-01", "1000")]))
        result = _fetch_fred("TEST_SERIES", "key", "2024-01-01", "2025-01-01")
        assert len(result) == 1
        assert result[0]["value"] == "1000"

    @patch("agent.tools.capital_flows.httpx.Client")
    def test_fetch_filters_missing_values(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        obs = [
            {"date": "2025-01-01", "value": "1000"},
            {"date": "2025-02-01", "value": "."},
            {"date": "2025-03-01", "value": ""},
            {"date": "2025-04-01", "value": "1100"},
        ]
        mock_client.get.return_value = _mock_response({"observations": obs})
        result = _fetch_fred("TEST", "key", "2024-01-01", "2025-04-01")
        assert len(result) == 2

    @patch("agent.tools.capital_flows.httpx.Client")
    def test_fetch_http_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response({}, status=500)
        result = _fetch_fred("TEST", "key", "2024-01-01", "2025-01-01")
        assert result == []

    @patch("agent.tools.capital_flows.httpx.Client")
    def test_fetch_timeout(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ReadTimeout("timeout")
        result = _fetch_fred("TEST", "key", "2024-01-01", "2025-01-01")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# 10. Tool metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestToolMetadata:
    def test_tool_name(self, tool):
        assert tool.name == "capital_flows"

    def test_tool_description(self, tool):
        assert "capital" in tool.description.lower()

    def test_parameters_schema(self, tool):
        props = tool.parameters["properties"]
        assert "mode" in props
        assert "period" in props
        assert "country" in props

    def test_required_params(self, tool):
        assert "mode" in tool.parameters["required"]


# ═══════════════════════════════════════════════════════════════════════════
# 11. Integration counts
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_tool_count(self):
        from agent.cli import build_tool_registry

        reg = build_tool_registry()
        # 60 -> 61 on 2026-08-26: nightlight_activity was 100% dead code
        # (constructor kwarg mismatch meant registration TypeErrored and was
        # skipped) — now correctly registered.
        assert len(reg.list_names()) == 61

    def test_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48


# ═══════════════════════════════════════════════════════════════════════════
# Phase 28: L2 capital-flow entity persistence
# ═══════════════════════════════════════════════════════════════════════════


def _make_store_mock():
    """Build a mock PipelineStore for L2 persistence testing."""
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    return store


class TestL2PersistenceGuards:
    """Persistence guards: no store or no entity_id_from_key → no-op."""

    def test_no_store_returns_zeros(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        tool._store = None
        counts = tool._persist_entities({"holdings": []}, "holdings")
        assert counts == {"capital_flow_obs": 0}

    def test_no_entity_id_fn_returns_zeros(self):
        import agent.tools.capital_flows as cf_mod

        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        tool._store = _make_store_mock()
        original = cf_mod._entity_id_from_key
        try:
            cf_mod._entity_id_from_key = None
            counts = tool._persist_entities({"holdings": []}, "holdings")
            assert counts == {"capital_flow_obs": 0}
        finally:
            cf_mod._entity_id_from_key = original

    def test_inner_exception_returns_zeros(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        store.register_entity.side_effect = RuntimeError("DB down")
        tool._store = store
        data = {
            "holdings": [
                {
                    "key": "japan",
                    "country": "Japan",
                    "latest_value_billions": 1100.0,
                    "mom_change_pct": -2.0,
                }
            ]
        }
        counts = tool._persist_entities(data, "holdings")
        assert counts == {"capital_flow_obs": 0}


class TestL2PersistenceHoldings:
    """holdings mode persists per-country capital_flow obs."""

    def test_holdings_persists_mapped_countries(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "holdings": [
                {
                    "key": "japan",
                    "country": "Japan",
                    "latest_value_billions": 1100.0,
                    "mom_change_pct": -2.0,
                },
                {
                    "key": "china",
                    "country": "China",
                    "latest_value_billions": 800.0,
                    "mom_change_pct": 1.5,
                },
            ],
        }
        counts = tool._persist_entities(data, "holdings")
        assert counts["capital_flow_obs"] == 2

    def test_holdings_skips_total(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "holdings": [
                {
                    "key": "total",
                    "country": "Total",
                    "latest_value_billions": 8000.0,
                    "mom_change_pct": 0.5,
                },
            ],
        }
        counts = tool._persist_entities(data, "holdings")
        assert counts["capital_flow_obs"] == 0

    def test_holdings_obs_type_and_depth(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "holdings": [
                {
                    "key": "japan",
                    "country": "Japan",
                    "latest_value_billions": 1100.0,
                    "mom_change_pct": -2.0,
                }
            ]
        }
        tool._persist_entities(data, "holdings")
        obs = store.store_entity_observation.call_args_list[0]
        assert obs.kwargs["observation_type"] == "capital_flow"
        assert obs.kwargs["depth_level"] == 2
        assert obs.kwargs["source_tool"] == "capital_flows"

    def test_holdings_targets_correct_country(self):
        from agent.pipeline.entity import entity_id_from_key

        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "holdings": [
                {
                    "key": "uk",
                    "country": "UK",
                    "latest_value_billions": 700.0,
                    "mom_change_pct": 0.0,
                }
            ]
        }
        tool._persist_entities(data, "holdings")
        gb_eid = entity_id_from_key("country", "GB")
        obs = store.store_entity_observation.call_args_list[0]
        assert obs.kwargs["entity_id"] == gb_eid

    def test_holdings_value_fields(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "holdings": [
                {
                    "key": "japan",
                    "country": "Japan",
                    "latest_value_billions": 1100.0,
                    "mom_change_pct": -2.0,
                }
            ]
        }
        tool._persist_entities(data, "holdings")
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        assert val["flow_type"] == "holdings"
        assert val["latest_value"] == 1100.0
        assert val["mom_change_pct"] == -2.0


class TestL2PersistenceFlows:
    """flows mode persists all to country=US."""

    def test_flows_persists_to_US(self):
        from agent.pipeline.entity import entity_id_from_key

        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "flows": [
                {"series": "net_tic", "latest_value": 50000, "flow_reversal": False},
                {
                    "series": "foreign_net_purchases",
                    "latest_value": -20000,
                    "flow_reversal": True,
                },
            ],
        }
        counts = tool._persist_entities(data, "flows")
        assert counts["capital_flow_obs"] == 2

        us_eid = entity_id_from_key("country", "US")
        for call in store.store_entity_observation.call_args_list:
            assert call.kwargs["entity_id"] == us_eid

    def test_flows_value_has_reversal(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {"flows": [{"series": "net_tic", "latest_value": -5000, "flow_reversal": True}]}
        tool._persist_entities(data, "flows")
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        assert val["flow_type"] == "flows"
        assert val["stress"] is True


class TestL2PersistenceReserves:
    """reserves mode persists per-country with stress info."""

    def test_reserves_persists_mapped_countries(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "reserves": [
                {
                    "key": "china_reserves",
                    "series": "TRESEGCNM052N",
                    "latest_value": 3200e9,
                    "stress": {"stress": True, "drawdown_pct": -6.0},
                },
                {
                    "key": "japan_reserves",
                    "series": "TRESEGJPM052N",
                    "latest_value": 1300e9,
                    "stress": {"stress": False, "drawdown_pct": -1.0},
                },
            ],
            "stress_alerts": ["china_reserves"],
            "errors": [],
        }
        counts = tool._persist_entities(data, "reserves")
        assert counts["capital_flow_obs"] == 2

    def test_reserves_skips_aggregate(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "reserves": [
                {
                    "key": "total_reserves_ex_gold",
                    "series": "X",
                    "latest_value": 9e12,
                    "stress": {},
                }
            ]
        }
        counts = tool._persist_entities(data, "reserves")
        assert counts["capital_flow_obs"] == 0

    def test_reserves_targets_correct_country(self):
        from agent.pipeline.entity import entity_id_from_key

        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "reserves": [
                {
                    "key": "india_reserves",
                    "series": "X",
                    "latest_value": 600e9,
                    "stress": {"stress": False, "drawdown_pct": -0.5},
                }
            ]
        }
        tool._persist_entities(data, "reserves")
        in_eid = entity_id_from_key("country", "IN")
        obs = store.store_entity_observation.call_args_list[0]
        assert obs.kwargs["entity_id"] == in_eid

    def test_reserves_stress_in_value(self):
        tool = CapitalFlowsTool(fred_api_key="test-key", cache=MagicMock())
        store = _make_store_mock()
        tool._store = store
        data = {
            "reserves": [
                {
                    "key": "saudi_reserves",
                    "series": "X",
                    "latest_value": 400e9,
                    "stress": {"stress": True, "drawdown_pct": -8.0},
                }
            ]
        }
        tool._persist_entities(data, "reserves")
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        assert val["flow_type"] == "reserves"
        assert val["stress"] is True
