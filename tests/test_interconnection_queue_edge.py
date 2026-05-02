"""
Edge-case tests for InterconnectionQueueTool (7b-K).

Coverage targets:
- Invalid / missing / boundary parameters
- Mode validation (queue, summary, datacenter)
- API key handling
- Queue: status mapping, state/fuel filters, min_mw filtering, project listing
- Summary: both/planned/construction, aggregation by fuel/state/status
- Datacenter: hyperscaler pattern matching, min_mw default, state concentration
- HTTP errors, empty responses
- Helper functions (_safe_float, _is_datacenter, _status_to_eia,
  _fetch_generators, _summarize_pipeline)
- Integration: tool count = 47, arm count = 35
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.interconnection_queue import (
    VALID_MODES,
    InterconnectionQueueTool,
    _fetch_generators,
    _is_datacenter,
    _safe_float,
    _status_to_eia,
    _summarize_pipeline,
)

# ── FIXTURES ──


@pytest.fixture
def tool():
    cache = MagicMock()
    cache.get.return_value = None
    t = InterconnectionQueueTool(cache=cache)
    t._api_key = "test-eia-key"
    return t


@pytest.fixture
def tool_no_key():
    cache = MagicMock()
    cache.get.return_value = None
    t = InterconnectionQueueTool(cache=cache)
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


def _gen_record(
    name="Solar Farm 1",
    entity="SunPower",
    mw=100,
    fuel="SUN",
    state="TX",
    status="PL",
    tech="Photovoltaic",
):
    """Build a generator record matching EIA response format."""
    return {
        "plantName": name,
        "entityName": entity,
        "nameplate-capacity-mw": mw,
        "energy-source-code": fuel,
        "stateid": state,
        "status": status,
        "technology": tech,
    }


# ══════════════════════════════════════════════════════════════════════════
# 1. Mode validation
# ══════════════════════════════════════════════════════════════════════════


class TestModeValidation:
    def test_invalid_mode(self, tool):
        r = tool.execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self, tool):
        r = tool.execute(mode="")
        assert not r.success

    def test_none_mode(self, tool):
        r = tool.execute()
        assert not r.success

    def test_valid_modes_constant(self):
        assert {"queue", "summary", "datacenter"} == VALID_MODES

    def test_no_api_key(self, tool_no_key):
        r = tool_no_key.execute(mode="queue")
        assert not r.success
        assert "API key" in r.output


# ══════════════════════════════════════════════════════════════════════════
# 2. Queue mode
# ══════════════════════════════════════════════════════════════════════════


class TestQueueMode:
    def test_invalid_status(self, tool):
        r = tool.execute(mode="queue", status="retired")
        assert not r.success
        assert "Invalid status" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_fetch_failure(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="queue")
        assert not r.success
        assert "Failed" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_empty_results(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="queue")
        assert r.success
        assert "No planned" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_successful_queue(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _gen_record("Big Solar", "SunCo", 500, "SUN", "TX"),
            _gen_record("Wind Farm", "WindCo", 300, "WND", "IA"),
        ]
        r = tool.execute(mode="queue")
        assert r.success
        assert "Generator Queue" in r.output
        assert "Big Solar" in r.output
        assert "800" in r.output  # total MW
        tool._cache.put.assert_called_once()

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_queue_construction_status(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _gen_record("Gas Plant", "GasCo", 200, "NG", "PA", status="U"),
        ]
        r = tool.execute(mode="queue", status="construction")
        assert r.success
        assert "Construction" in r.output or "construction" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_queue_state_filter(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _gen_record("TX Solar", "Co", 100, "SUN", "TX"),
        ]
        r = tool.execute(mode="queue", state="TX")
        assert r.success
        assert "TX" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_queue_fuel_filter(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _gen_record("Solar 1", "Co", 100, "SUN", "CA"),
        ]
        r = tool.execute(mode="queue", fuel="SUN")
        assert r.success
        assert "Solar" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_queue_min_mw_filter(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _gen_record("Small", "Co", 5, "SUN", "CA"),
            _gen_record("Big", "Co", 500, "SUN", "CA"),
        ]
        r = tool.execute(mode="queue", min_mw=100)
        assert r.success
        assert "Big" in r.output
        assert "Small" not in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_queue_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = "cached queue"
        r = tool.execute(mode="queue")
        assert r.success
        assert r.output == "cached queue"
        mock_fetch.assert_not_called()

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_queue_many_projects_truncated(self, mock_fetch, tool):
        records = [_gen_record(f"Solar {i}", "Co", 100, "SUN", "TX") for i in range(30)]
        mock_fetch.return_value = records
        r = tool.execute(mode="queue")
        assert r.success
        assert "5 more projects" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_queue_empty_after_min_mw(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _gen_record("Tiny", "Co", 1, "SUN", "CA"),
        ]
        r = tool.execute(mode="queue", min_mw=500)
        assert r.success
        assert "No planned" in r.output


# ══════════════════════════════════════════════════════════════════════════
# 3. Summary mode
# ══════════════════════════════════════════════════════════════════════════


class TestSummaryMode:
    def test_invalid_status(self, tool):
        r = tool.execute(mode="summary", status="retired")
        assert not r.success

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_empty_results(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="summary")
        assert r.success
        assert "No planned" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_both_status(self, mock_fetch, tool):
        """Default 'both' fetches PL and U."""
        mock_fetch.side_effect = [
            [_gen_record("Planned", "Co", 100, "SUN", "TX", "PL")],
            [_gen_record("Under Const", "Co", 200, "WND", "IA", "U")],
        ]
        r = tool.execute(mode="summary")
        assert r.success
        assert "Pipeline Summary" in r.output
        assert "300" in r.output  # total MW

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_summary_planned_only(self, mock_fetch, tool):
        mock_fetch.return_value = [
            _gen_record("Solar", "Co", 100, "SUN", "TX"),
        ]
        r = tool.execute(mode="summary", status="planned")
        assert r.success
        assert "Solar" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_summary_state_filter(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            [_gen_record("TX1", "Co", 100, "SUN", "TX")],
            [],
        ]
        r = tool.execute(mode="summary", state="TX")
        assert r.success
        assert "TX" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_summary_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = "cached summary"
        r = tool.execute(mode="summary")
        assert r.success
        mock_fetch.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# 4. Datacenter mode
# ══════════════════════════════════════════════════════════════════════════


class TestDatacenterMode:
    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_empty_results(self, mock_fetch, tool):
        mock_fetch.return_value = []
        r = tool.execute(mode="datacenter")
        assert r.success
        assert "No planned" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_no_dc_matches(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            [_gen_record("Regular Plant", "Regular Corp", 200, "NG", "TX")],
            [],
        ]
        r = tool.execute(mode="datacenter")
        assert r.success
        assert "No suspected" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_dc_match_found(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            [_gen_record("AWS Data Center VA", "Amazon Web Services", 300, "NG", "VA")],
            [],
        ]
        r = tool.execute(mode="datacenter")
        assert r.success
        assert "Data Center Power" in r.output
        assert "Amazon" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_dc_min_mw_default(self, mock_fetch, tool):
        """Default min_mw for datacenter is 50."""
        mock_fetch.side_effect = [
            [_gen_record("Small DC", "Amazon", 30, "NG", "VA")],
            [],
        ]
        r = tool.execute(mode="datacenter")
        assert r.success
        assert "No suspected" in r.output  # 30 < 50 default

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_dc_custom_min_mw(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            [_gen_record("Small DC", "Amazon", 30, "NG", "VA")],
            [],
        ]
        r = tool.execute(mode="datacenter", min_mw=10)
        assert r.success
        assert "Amazon" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_dc_state_filter(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            [_gen_record("DC VA", "Microsoft Azure", 200, "NG", "VA")],
            [],
        ]
        r = tool.execute(mode="datacenter", state="VA")
        assert r.success
        assert "Microsoft" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_dc_state_concentration(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            [
                _gen_record("DC1", "Amazon", 200, "NG", "VA"),
                _gen_record("DC2", "Google Cloud", 300, "NG", "VA"),
                _gen_record("DC3", "Meta", 100, "NG", "TX"),
            ],
            [],
        ]
        r = tool.execute(mode="datacenter")
        assert r.success
        assert "State concentration" in r.output

    @patch("agent.tools.interconnection_queue._fetch_generators")
    def test_dc_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = "cached dc"
        r = tool.execute(mode="datacenter")
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

    # -- _status_to_eia --
    def test_status_planned(self):
        assert _status_to_eia("planned") == "PL"

    def test_status_construction(self):
        assert _status_to_eia("construction") == "U"

    def test_status_operating(self):
        assert _status_to_eia("operating") == "OP"

    def test_status_unknown(self):
        assert _status_to_eia("nonexistent") is None

    def test_status_case_insensitive(self):
        assert _status_to_eia("Planned") == "PL"

    # -- _is_datacenter --
    def test_dc_amazon(self):
        assert _is_datacenter("Amazon Web Services", "Cloud Farm")

    def test_dc_aws(self):
        assert _is_datacenter("AWS Infrastructure", "Power Plant")

    def test_dc_microsoft(self):
        assert _is_datacenter("Microsoft Corporation", "Azure")

    def test_dc_google(self):
        assert _is_datacenter("Google LLC", "Data Farm")

    def test_dc_meta(self):
        assert _is_datacenter("Meta Platforms", "Server Farm")

    def test_dc_equinix(self):
        assert _is_datacenter("Equinix Inc", "DC Campus")

    def test_dc_digital_realty(self):
        assert _is_datacenter("Digital Realty Trust", "Center")

    def test_dc_plant_name_match(self):
        assert _is_datacenter("Random Corp", "Data Center Operations")

    def test_dc_colocation(self):
        assert _is_datacenter("Random Corp", "Colocation Facility")

    def test_dc_no_match(self):
        assert not _is_datacenter("Solar Power Inc", "Solar Farm 1")

    def test_dc_empty(self):
        assert not _is_datacenter("", "")

    def test_dc_case_insensitive(self):
        assert _is_datacenter("AMAZON", "cloud")

    # -- _summarize_pipeline --
    def test_summarize_empty(self):
        result = _summarize_pipeline([])
        assert result["total_mw"] == 0
        assert result["project_count"] == 0

    def test_summarize_single(self):
        records = [_gen_record("Solar", "Co", 100, "SUN", "TX", "PL")]
        result = _summarize_pipeline(records)
        assert result["total_mw"] == 100
        assert result["project_count"] == 1
        assert "SUN" in result["by_fuel"]

    def test_summarize_multiple_fuels(self):
        records = [
            _gen_record("S1", "Co", 100, "SUN", "TX"),
            _gen_record("W1", "Co", 200, "WND", "IA"),
            _gen_record("G1", "Co", 300, "NG", "PA"),
        ]
        result = _summarize_pipeline(records)
        assert result["total_mw"] == 600
        assert len(result["by_fuel"]) == 3
        # Sorted by MW descending
        fuels = list(result["by_fuel"].keys())
        assert fuels[0] == "NG"  # highest MW

    def test_summarize_state_aggregation(self):
        records = [
            _gen_record("S1", "Co", 100, "SUN", "TX"),
            _gen_record("S2", "Co", 200, "SUN", "TX"),
            _gen_record("W1", "Co", 50, "WND", "IA"),
        ]
        result = _summarize_pipeline(records)
        assert result["by_state"]["TX"] == 300
        assert result["by_state"]["IA"] == 50


# ══════════════════════════════════════════════════════════════════════════
# 6. Fetch generators function
# ══════════════════════════════════════════════════════════════════════════


class TestFetchGenerators:
    @patch("agent.tools.interconnection_queue.httpx.get")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response(200, json_data={"response": {"data": [{"value": 42}]}})
        result = _fetch_generators({"status": ["PL"]}, "key")
        assert result is not None
        assert len(result) == 1

    @patch("agent.tools.interconnection_queue.httpx.get")
    def test_http_error(self, mock_get):
        mock_get.return_value = _mock_response(500)
        result = _fetch_generators({"status": ["PL"]}, "key")
        assert result is None

    @patch("agent.tools.interconnection_queue.httpx.get")
    def test_network_error(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.ConnectError("fail")
        result = _fetch_generators({"status": ["PL"]}, "key")
        assert result is None


# ══════════════════════════════════════════════════════════════════════════
# 7. API key handling
# ══════════════════════════════════════════════════════════════════════════


class TestApiKey:
    def test_key_from_env(self):
        with patch.dict(os.environ, {"TIRRA_EIA_API_KEY": "my-eia-key"}):
            t = InterconnectionQueueTool()
            assert t._api_key == "my-eia-key"

    def test_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TIRRA_EIA_API_KEY", None)
            t = InterconnectionQueueTool()
            assert t._api_key is None

    def test_empty_key(self):
        with patch.dict(os.environ, {"TIRRA_EIA_API_KEY": "  "}):
            t = InterconnectionQueueTool()
            assert t._api_key is None


# ══════════════════════════════════════════════════════════════════════════
# 8. Tool metadata
# ══════════════════════════════════════════════════════════════════════════


class TestToolMetadata:
    def test_name(self):
        t = InterconnectionQueueTool()
        assert t.name == "interconnection_queue"

    def test_description(self):
        t = InterconnectionQueueTool()
        assert "queue" in t.description.lower()
        assert "datacenter" in t.description.lower()

    def test_mode_required(self):
        t = InterconnectionQueueTool()
        assert "mode" in t.parameters.get("required", [])


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

    def test_queue_registered(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert "interconnection_queue" in registry.list_names()

    def test_queue_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "energy_infrastructure_pipeline" in arm_names
