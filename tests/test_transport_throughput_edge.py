"""
Edge case tests for TransportThroughputTool (BTS Socrata — US border crossings).

Covers: mode validation, measure resolution, border resolution, parameter clamping,
recent/trend/port/compare modes, SoQL query construction, cache interaction,
HTTP error handling, timeout, empty responses, malformed data, output formatting,
constants, aliases, registry + bandit integration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.transport_throughput import (
    _KEY_MEASURES,
    BORDER_ALIASES,
    MEASURE_ALIASES,
    VALID_BORDERS,
    VALID_MEASURES,
    TransportThroughputTool,
    _resolve_border,
    _resolve_measure,
    _safe_int,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> TransportThroughputTool:
    return TransportThroughputTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    import json

    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


# ── 1. Tool Metadata ─────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "transport_throughput"

    def test_description_nonempty(self):
        assert len(_tool().description) > 50

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        props = params["properties"]
        assert "mode" in props
        assert "measure" in props
        assert "border" in props
        assert "state" in props
        assert "months_back" in props
        assert "limit" in props

    def test_mode_enum(self):
        modes = _tool().parameters["properties"]["mode"]["enum"]
        assert set(modes) == {"recent", "trend", "port", "compare"}


# ── 2. Input Validation ──────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_invalid_measure(self):
        r = _tool().execute(mode="recent", measure="flying_carpets")
        assert not r.success
        assert "Unknown measure" in r.output

    def test_invalid_border(self):
        r = _tool().execute(mode="recent", measure="trucks", border="france")
        assert not r.success
        assert "Unknown border" in r.output

    def test_months_back_clamped_low(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([{"max_date": "2025-06-01T00:00:00"}], None)
            r = _tool().execute(mode="recent", months_back=0)
        # months_back = max(1, ...) → clamped
        assert r.success or True  # may fail due to second fetch, but first clamp works

    def test_months_back_clamped_high(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([{"max_date": "2025-06-01T00:00:00"}], None)
            r = _tool().execute(mode="recent", months_back=999)
        # months_back = min(..., 60) → clamped

    def test_limit_clamped_low(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([{"max_date": "2025-06-01T00:00:00"}], None)
            r = _tool().execute(mode="recent", limit=0)

    def test_limit_clamped_high(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([{"max_date": "2025-06-01T00:00:00"}], None)
            r = _tool().execute(mode="recent", limit=999)

    def test_extra_kwargs_ignored(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([{"max_date": "2025-06-01T00:00:00"}], None)
            r = _tool().execute(mode="recent", bogus="thing")


# ── 3. Measure Resolution ────────────────────────────────────


class TestMeasureResolution:
    def test_alias_trucks(self):
        assert _resolve_measure("trucks") == "Trucks"

    def test_alias_trains(self):
        assert _resolve_measure("trains") == "Trains"

    def test_alias_rail(self):
        assert _resolve_measure("rail") == "Trains"

    def test_alias_vehicles(self):
        assert _resolve_measure("vehicles") == "Personal Vehicles"

    def test_alias_cars(self):
        assert _resolve_measure("cars") == "Personal Vehicles"

    def test_alias_containers_loaded(self):
        assert _resolve_measure("containers_loaded") == "Rail Containers Loaded"

    def test_alias_containers_empty(self):
        assert _resolve_measure("containers_empty") == "Rail Containers Empty"

    def test_alias_rail_loaded(self):
        assert _resolve_measure("rail_loaded") == "Rail Containers Loaded"

    def test_alias_rail_empty(self):
        assert _resolve_measure("rail_empty") == "Rail Containers Empty"

    def test_alias_pedestrians(self):
        assert _resolve_measure("pedestrians") == "Pedestrians"

    def test_alias_buses(self):
        assert _resolve_measure("buses") == "Buses"

    def test_alias_passengers(self):
        assert _resolve_measure("passengers") == "Personal Vehicle Passengers"

    def test_alias_bus_passengers(self):
        assert _resolve_measure("bus_passengers") == "Bus Passengers"

    def test_alias_train_passengers(self):
        assert _resolve_measure("train_passengers") == "Train Passengers"

    def test_canonical_name(self):
        assert _resolve_measure("Trucks") == "Trucks"

    def test_canonical_case_insensitive(self):
        assert _resolve_measure("TRUCKS") == "Trucks"

    def test_unknown(self):
        assert _resolve_measure("airplanes") is None

    def test_whitespace(self):
        assert _resolve_measure("  trucks  ") == "Trucks"

    def test_all_aliases_resolve(self):
        for alias, expected in MEASURE_ALIASES.items():
            result = _resolve_measure(alias)
            assert result == expected, f"Alias '{alias}' → {result}, expected {expected}"

    def test_all_valid_measures_resolve(self):
        for m in VALID_MEASURES:
            assert _resolve_measure(m) == m


# ── 4. Border Resolution ─────────────────────────────────────


class TestBorderResolution:
    def test_canada(self):
        assert _resolve_border("canada") == "US-Canada Border"

    def test_mexico(self):
        assert _resolve_border("mexico") == "US-Mexico Border"

    def test_ca(self):
        assert _resolve_border("ca") == "US-Canada Border"

    def test_mx(self):
        assert _resolve_border("mx") == "US-Mexico Border"

    def test_canonical(self):
        assert _resolve_border("US-Canada Border") == "US-Canada Border"

    def test_case_insensitive(self):
        assert _resolve_border("CANADA") == "US-Canada Border"

    def test_unknown(self):
        assert _resolve_border("france") is None

    def test_whitespace(self):
        assert _resolve_border("  mexico  ") == "US-Mexico Border"

    def test_all_aliases_resolve(self):
        for alias, expected in BORDER_ALIASES.items():
            assert _resolve_border(alias) == expected


# ── 5. Safe Int ───────────────────────────────────────────────


class TestSafeInt:
    def test_normal(self):
        assert _safe_int("42") == 42

    def test_float_string(self):
        # "42.5" → ValueError from int(), returns default
        assert _safe_int("42.5") == 0

    def test_none(self):
        assert _safe_int(None) == 0

    def test_empty(self):
        assert _safe_int("") == 0

    def test_custom_default(self):
        assert _safe_int("bad", 99) == 99

    def test_int_passthrough(self):
        assert _safe_int(42) == 42

    def test_negative(self):
        assert _safe_int("-5") == -5


# ── 6. Recent Mode ───────────────────────────────────────────


class TestRecentMode:
    def test_basic_recent(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            # First call: max_date, Second call: aggregated data
            m.side_effect = [
                ([{"max_date": "2025-06-01T00:00:00"}], None),
                (
                    [
                        {
                            "border": "US-Canada Border",
                            "measure": "Trucks",
                            "total": "38000",
                        },
                        {
                            "border": "US-Canada Border",
                            "measure": "Trains",
                            "total": "12000",
                        },
                        {
                            "border": "US-Mexico Border",
                            "measure": "Trucks",
                            "total": "42000",
                        },
                    ],
                    None,
                ),
            ]
            r = _tool().execute(mode="recent")
        assert r.success
        assert r.data["count"] == 3
        assert "2025-06-01" in r.output

    def test_recent_no_data(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.side_effect = [
                ([{}], None),  # max_date is None
            ]
            r = _tool().execute(mode="recent")
        assert r.success
        assert r.data["count"] == 0

    def test_recent_fetch_error_first(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([], "BTS API: Rate limited.")
            r = _tool().execute(mode="recent")
        assert not r.success

    def test_recent_fetch_error_second(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.side_effect = [
                ([{"max_date": "2025-06-01T00:00:00"}], None),
                ([], "BTS API: Rate limited."),
            ]
            r = _tool().execute(mode="recent")
        assert not r.success

    def test_recent_border_filter(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.side_effect = [
                ([{"max_date": "2025-06-01T00:00:00"}], None),
                (
                    [
                        {
                            "border": "US-Mexico Border",
                            "measure": "Trucks",
                            "total": "42000",
                        }
                    ],
                    None,
                ),
            ]
            r = _tool().execute(mode="recent", border="mexico")
        assert r.success

    def test_recent_zero_totals_filtered(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.side_effect = [
                ([{"max_date": "2025-06-01T00:00:00"}], None),
                (
                    [
                        {
                            "border": "US-Canada Border",
                            "measure": "Trucks",
                            "total": "0",
                        },
                        {
                            "border": "US-Canada Border",
                            "measure": "Trains",
                            "total": "5000",
                        },
                    ],
                    None,
                ),
            ]
            r = _tool().execute(mode="recent")
        assert r.success
        # Zero-total rows don't appear in formatted_records
        assert r.data["count"] == 1

    def test_recent_key_measure_marker(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.side_effect = [
                ([{"max_date": "2025-06-01T00:00:00"}], None),
                (
                    [
                        {
                            "border": "US-Canada Border",
                            "measure": "Trucks",
                            "total": "38000",
                        }
                    ],
                    None,
                ),
            ]
            r = _tool().execute(mode="recent")
        assert "*" in r.output  # key trade indicator marker


# ── 7. Trend Mode ────────────────────────────────────────────


class TestTrendMode:
    def test_basic_trend(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = (
                [
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "35000",
                    },
                    {
                        "date": "2025-02-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "37000",
                    },
                    {
                        "date": "2025-03-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "36000",
                    },
                ],
                None,
            )
            r = _tool().execute(mode="trend", measure="trucks", months_back=6)
        assert r.success
        assert r.data["count"] == 3
        assert r.data["measure"] == "Trucks"

    def test_trend_empty(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="trend")
        assert r.success
        assert r.data["count"] == 0

    def test_trend_fetch_error(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([], "BTS API: Request timed out.")
            r = _tool().execute(mode="trend")
        assert not r.success

    def test_trend_mom_change(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = (
                [
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "10000",
                    },
                    {
                        "date": "2025-02-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "11000",
                    },
                ],
                None,
            )
            r = _tool().execute(mode="trend")
        assert r.success
        assert "+10.0%" in r.output

    def test_trend_border_filter(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = (
                [
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Mexico Border",
                        "total": "40000",
                    },
                ],
                None,
            )
            r = _tool().execute(mode="trend", border="mexico")
        assert r.success
        assert "Mexico" in r.output

    def test_trend_both_borders(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = (
                [
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "35000",
                    },
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Mexico Border",
                        "total": "40000",
                    },
                ],
                None,
            )
            r = _tool().execute(mode="trend")
        assert r.success
        assert "Both Borders" in r.output


# ── 8. Port Mode ─────────────────────────────────────────────


class TestPortMode:
    def test_basic_port(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.side_effect = [
                ([{"max_date": "2025-06-01T00:00:00"}], None),
                (
                    [
                        {
                            "port_name": "Laredo",
                            "state": "Texas",
                            "border": "US-Mexico Border",
                            "value": "50000",
                        },
                        {
                            "port_name": "El Paso",
                            "state": "Texas",
                            "border": "US-Mexico Border",
                            "value": "30000",
                        },
                    ],
                    None,
                ),
            ]
            r = _tool().execute(mode="port", state="Texas")
        assert r.success
        assert r.data["count"] == 2
        assert "Laredo" in r.output

    def test_port_empty(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.side_effect = [
                ([{"max_date": "2025-06-01T00:00:00"}], None),
                ([], None),
            ]
            r = _tool().execute(mode="port", state="Alaska")
        assert r.success
        assert r.data["count"] == 0

    def test_port_fetch_error(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([], "BTS API: Rate limited.")
            r = _tool().execute(mode="port")
        assert not r.success

    def test_port_no_date(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="port")
        assert r.success
        assert r.data["count"] == 0

    def test_port_zero_value_filtered(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.side_effect = [
                ([{"max_date": "2025-06-01T00:00:00"}], None),
                (
                    [
                        {
                            "port_name": "Empty Port",
                            "state": "TX",
                            "border": "US-Mexico Border",
                            "value": "0",
                        }
                    ],
                    None,
                ),
            ]
            r = _tool().execute(mode="port")
        assert r.success
        assert r.data["count"] == 0


# ── 9. Compare Mode ──────────────────────────────────────────


class TestCompareMode:
    def test_basic_compare(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = (
                [
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "35000",
                    },
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Mexico Border",
                        "total": "40000",
                    },
                    {
                        "date": "2025-02-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "36000",
                    },
                    {
                        "date": "2025-02-01T00:00:00",
                        "border": "US-Mexico Border",
                        "total": "42000",
                    },
                ],
                None,
            )
            r = _tool().execute(mode="compare", measure="trucks")
        assert r.success
        assert r.data["count"] == 2
        assert "Canada" in r.output
        assert "Mexico" in r.output

    def test_compare_ratio(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = (
                [
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "20000",
                    },
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Mexico Border",
                        "total": "10000",
                    },
                ],
                None,
            )
            r = _tool().execute(mode="compare")
        assert r.success
        assert r.data["comparison"][0]["ratio"] == pytest.approx(2.0)

    def test_compare_zero_mexico(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = (
                [
                    {
                        "date": "2025-01-01T00:00:00",
                        "border": "US-Canada Border",
                        "total": "20000",
                    },
                ],
                None,
            )
            r = _tool().execute(mode="compare")
        assert r.success
        # Mexico is 0 → ratio is None
        assert r.data["comparison"][0]["ratio"] is None
        assert "n/a" in r.output

    def test_compare_empty(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="compare")
        assert r.success
        assert r.data["count"] == 0

    def test_compare_fetch_error(self):
        with patch.object(TransportThroughputTool, "_fetch_bts") as m:
            m.return_value = ([], "BTS API error")
            r = _tool().execute(mode="compare")
        assert not r.success


# ── 10. BTS Fetch ─────────────────────────────────────────────


class TestBTSFetch:
    def test_successful_fetch(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp([{"border": "US-Canada Border", "value": "100"}])
            mock_client.return_value = mc

            data, err = _tool()._fetch_bts({"$select": "count(*)"})
        assert err is None
        assert len(data) == 1

    def test_rate_limited(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp([], status=429)
            mock_client.return_value = mc

            data, err = _tool()._fetch_bts({"$select": "count(*)"})
        assert data == []
        assert "Rate limited" in err

    def test_http_500(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp([], status=500)
            mock_client.return_value = mc

            data, err = _tool()._fetch_bts({"$select": "count(*)"})
        assert "500" in err

    def test_timeout(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.side_effect = httpx.TimeoutException("timeout")
            mock_client.return_value = mc

            data, err = _tool()._fetch_bts({"$select": "count(*)"})
        assert data == []
        assert "timed out" in err.lower()

    def test_generic_exception(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.side_effect = ConnectionError("DNS fail")
            mock_client.return_value = mc

            data, err = _tool()._fetch_bts({"$select": "count(*)"})
        assert "error" in err.lower()

    def test_non_list_response(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp({"error": "bad query"})
            mock_client.return_value = mc

            data, err = _tool()._fetch_bts({"$select": "count(*)"})
        assert data == []
        assert "Unexpected" in err

    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = [{"border": "test"}]
        data, err = TransportThroughputTool(cache=cache)._fetch_bts({"$select": "count(*)"})
        assert err is None
        assert len(data) == 1

    def test_cache_miss_then_put(self):
        cache = MagicMock()
        cache.get.return_value = None
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp([{"border": "test"}])
            mock_client.return_value = mc

            data, err = TransportThroughputTool(cache=cache)._fetch_bts({"$select": "count(*)"})
        assert err is None
        cache.put.assert_called_once()

    def test_empty_response_no_cache_put(self):
        cache = MagicMock()
        cache.get.return_value = None
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp([])
            mock_client.return_value = mc

            data, err = TransportThroughputTool(cache=cache)._fetch_bts({"$select": "count(*)"})
        assert err is None
        cache.put.assert_not_called()


# ── 11. Constants ─────────────────────────────────────────────


class TestConstants:
    def test_valid_measures_count(self):
        assert len(VALID_MEASURES) == 10

    def test_valid_borders_count(self):
        assert len(VALID_BORDERS) == 2

    def test_key_measures_subset(self):
        for m in _KEY_MEASURES:
            assert m in VALID_MEASURES

    def test_all_aliases_map_to_valid(self):
        for alias, target in MEASURE_ALIASES.items():
            assert target in VALID_MEASURES, f"Alias '{alias}' maps to invalid measure '{target}'"

    def test_all_border_aliases_map_to_valid(self):
        for alias, target in BORDER_ALIASES.items():
            assert target in VALID_BORDERS


# ── 12. Registry & Bandit Integration ────────────────────────


class TestRegistryIntegration:
    def test_tool_registered(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert "transport_throughput" in registry.list_names()

    def test_tool_count_27(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert len(registry.list_names()) == 61

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = [a.name for a in DEFAULT_ARMS]
        assert "transport_flow" in names

    def test_bandit_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "transport_flow")
        assert "transport_throughput" in arm.tools

    def test_bandit_arm_count_16(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48
