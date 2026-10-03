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
- Completeness: response.total, offset paging, hour coverage, partial persistence

The completeness tests are the load-bearing ones. This collector's entire
failure history is HTTP 200 + success=True + fewer rows than EIA published,
so a test that hands the fake three records and asserts three rows back
certifies nothing. :class:`FakeEIA` below models the real API's *shape* —
server-side facets, sort-then-slice, and a ``total`` larger than the page —
so an under-sized or mis-faceted query produces a measurable shortfall the
tests can fail on.
"""

import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fixture_time import T

from agent.data.cache import DataCache
from agent.pipeline.store import PipelineStore
from agent.tools.electricity_monitor import (
    KNOWN_REGIONS,
    VALID_MODES,
    EIAWindow,
    ElectricityMonitorTool,
    PeriodFormatError,
    _aggregate_hourly,
    _clean_period,
    _expected_hours,
    _fetch_eia,
    _fuel_mix_proportions,
    _safe_float,
)

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


def _eia_period(hours_after_base: int) -> str:
    """An EIA hourly period string anchored on the fixture epoch."""
    return datetime.fromtimestamp(T(hours_after_base * 3600), tz=UTC).strftime("%Y-%m-%dT%H")


def _rec(period, *, type_code=None, type_name=None, value=None, **extra):
    r = {"period": period, "respondent": "PJM", "value-units": "megawatthours"}
    if type_code is not None:
        r["type"] = type_code
    if type_name is not None:
        r["type-name"] = type_name
    if value is not None:
        r["value"] = value
    r.update(extra)
    return r


def _hourly_demand(hours=24, first_hour=0, base_mw=50000):
    """`hours` consecutive hourly actual-demand (type D) records."""
    return [
        _rec(
            _eia_period(first_hour + i),
            type_code="D",
            type_name="Demand",
            value=str(base_mw + i),
        )
        for i in range(hours)
    ]


def _win(records, total=None, complete=True):
    """An EIAWindow standing in for a real fetch."""
    records = list(records)
    return EIAWindow(
        records=records,
        total=len(records) if total is None else total,
        complete=complete,
    )


class FakeEIA:
    """A stand-in that behaves the way api.eia.gov v2 actually behaves.

    The behaviours that matter, each one load-bearing for some assertion:

    * ``value`` is present ONLY when ``data[0]=value`` was requested. Drop
      that selector and every reading parses as absent, not as 0 MW.
    * ``facets[x][]`` filter server-side. Drop ``facets[type][]=D`` from the
      collector and the forecast rows are back in the payload, eating the
      row budget exactly as they do live.
    * rows are sorted by period, THEN sliced by ``length``/``offset``. This
      is what turns an under-sized ``length`` into missing hours rather than
      into an error.
    * ``total`` reports the whole filtered series, not the page — so a caller
      can tell a truncated read from a complete one.
    """

    def __init__(self, rows, status=200):
        self.rows = list(rows)
        self.status = status
        self.calls: list[list[tuple]] = []

    def __call__(self, url, params=None, **kwargs):
        pairs = list(params.items()) if isinstance(params, dict) else list(params or [])
        self.calls.append(pairs)
        if self.status != 200:
            return _mock_response(self.status)

        def values_for(name):
            return [v for k, v in pairs if k == name]

        wants_value = ("data[0]", "value") in pairs

        facets: dict[str, set] = {}
        for key, val in pairs:
            if key.startswith("facets[") and key.endswith("][]"):
                facets.setdefault(key[7:-3], set()).add(val)

        rows = list(self.rows)
        for field, allowed in facets.items():
            rows = [r for r in rows if r.get(field) in allowed]

        start = values_for("start")
        end = values_for("end")
        if start:
            rows = [r for r in rows if r["period"] >= start[0]]
        if end:
            rows = [r for r in rows if r["period"] <= end[0]]

        direction = (values_for("sort[0][direction]") or ["desc"])[0]
        rows.sort(key=lambda r: r["period"], reverse=direction == "desc")

        total = len(rows)
        length = int((values_for("length") or [5000])[0])
        offset = int((values_for("offset") or [0])[0])
        page = rows[offset : offset + length]

        data = [{k: v for k, v in r.items() if k != "value" or wants_value} for r in page]
        return _mock_response(200, json_data={"response": {"total": total, "data": data}})

    def params_of(self, call_index=0) -> list[tuple]:
        return self.calls[call_index]


def _region_data_rows(demand_hours=48):
    """region-data as EIA serves it: four series share the endpoint.

    The day-ahead forecast (DF) rows deliberately run *past* the newest
    actual-demand hour, because that is what makes them dominate a
    sort-desc page and starve the demand budget on the live API.
    """
    rows = []
    for h in range(demand_hours):
        rows.append(_rec(_eia_period(h), type_code="D", type_name="Demand", value=str(70000 + h)))
        rows.append(_rec(_eia_period(h), type_code="NG", type_name="Net generation", value=str(81000 + h)))
        rows.append(_rec(_eia_period(h), type_code="TI", type_name="Total interchange", value=str(500 + h)))
    for h in range(demand_hours // 2, demand_hours + 24):
        rows.append(
            _rec(
                _eia_period(h),
                type_code="DF",
                type_name="Day-ahead demand forecast",
                value=str(85000 + h),
            )
        )
    return rows


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
        assert {"demand", "generation", "interchange"} == VALID_MODES

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

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_region_uppercased(self, mock_fetch, tool):
        mock_fetch.return_value = _win(_hourly_demand(24))
        r = tool.execute(mode="demand", region="pjm")
        assert r.success, r.output
        assert mock_fetch.call_args.args[1]["respondent"] == ["PJM"]


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
    def test_empty_records_is_a_failure(self, mock_fetch, tool):
        """Zero rows from an always-on public source is a bug, not a fact.

        The old code returned success=True here AND cached the emptiness for
        6h, so the next run replayed the zero without asking EIA.
        """
        mock_fetch.return_value = _win([], total=270532)
        r = tool.execute(mode="demand", region="PJM")
        assert not r.success
        assert "No demand data" in r.output
        assert "total=270532" in r.output
        tool._cache.put.assert_not_called()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_successful_demand(self, mock_fetch, tool):
        mock_fetch.return_value = _win(_hourly_demand(24, base_mw=32000))
        r = tool.execute(mode="demand", region="PJM")
        assert r.success, r.output
        assert "Electricity Demand" in r.output
        assert "Peak" in r.output
        assert "32,023" in r.output
        tool._cache.put.assert_called_once()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_short_window_is_red(self, mock_fetch, tool):
        """Three hours where 24 were asked for is a shortfall, not a success."""
        mock_fetch.return_value = _win(_hourly_demand(3))
        r = tool.execute(mode="demand", region="PJM")
        assert not r.success
        assert "only 3 of 24 requested hour(s)" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_truncated_window_is_red(self, mock_fetch, tool):
        """EIA said 500 rows, we got 24: F-16's signature, now visible."""
        mock_fetch.return_value = _win(_hourly_demand(24), total=500, complete=False)
        r = tool.execute(mode="demand", region="PJM")
        assert not r.success
        assert "EIA declared 500 row(s)" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_cache_hit_replays_records(self, mock_fetch, tool):
        records = _hourly_demand(24)
        tool._cache.get.return_value = {"text": "cached demand", "records": records}
        r = tool.execute(mode="demand", region="PJM")
        assert r.success
        assert r.output == "cached demand"
        mock_fetch.assert_not_called()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_legacy_string_cache_entry_refetches(self, mock_fetch, tool):
        """A pre-records cache entry must not serve a green zero-row run."""
        tool._cache.get.return_value = "cached demand"
        mock_fetch.return_value = _win(_hourly_demand(24))
        r = tool.execute(mode="demand", region="PJM")
        assert r.success, r.output
        assert mock_fetch.called
        assert "Electricity Demand" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_recordless_cache_entry_refetches(self, mock_fetch, tool):
        tool._cache.get.return_value = {"text": "cached demand", "records": []}
        mock_fetch.return_value = _win(_hourly_demand(24))
        r = tool.execute(mode="demand", region="PJM")
        assert r.success, r.output
        assert mock_fetch.called

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_days_clamped_max(self, mock_fetch, tool):
        mock_fetch.return_value = _win(_hourly_demand(7 * 24))
        r = tool.execute(mode="demand", region="PJM", days=99)
        assert r.success, r.output
        # days clamps to 7, so the budget is 7*24 hours plus the lag buffer.
        assert mock_fetch.call_args.kwargs["length"] == 7 * 24 + 24

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_days_non_numeric(self, mock_fetch, tool):
        mock_fetch.return_value = _win(_hourly_demand(24))
        r = tool.execute(mode="demand", region="PJM", days="abc")
        assert r.success, r.output
        assert mock_fetch.call_args.kwargs["length"] == 24 + 24

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_known_region_name(self, mock_fetch, tool):
        mock_fetch.return_value = _win(_hourly_demand(24))
        r = tool.execute(mode="demand", region="PJM")
        assert "PJM Interconnection" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_unknown_region(self, mock_fetch, tool):
        mock_fetch.return_value = _win(_hourly_demand(24))
        r = tool.execute(mode="demand", region="XXXX")
        assert r.success, r.output
        assert "XXXX" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_untyped_records_accepted(self, mock_fetch, tool):
        """Untyped rows are accepted — the request facets to type=D server-side."""
        mock_fetch.return_value = _win([_rec(_eia_period(h), value=str(40000 + h)) for h in range(24)])
        r = tool.execute(mode="demand", region="PJM")
        assert r.success, r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_demand_rejects_foreign_series(self, mock_fetch, tool):
        """NG/TI/DF rows are not demand and must not become demand stats."""
        mock_fetch.return_value = _win(
            [_rec(_eia_period(h), type_code="NG", type_name="Net generation", value="81000") for h in range(24)]
        )
        r = tool.execute(mode="demand", region="PJM")
        assert not r.success
        assert "0 of them actual demand" in r.output


# ══════════════════════════════════════════════════════════════════════════
# 3. Generation mode
# ══════════════════════════════════════════════════════════════════════════


def _hourly_generation(hours=24, fuels=("NG", "SUN", "WND", "NUC"), respondent="CISO"):
    rows = []
    for h in range(hours):
        for i, fuel in enumerate(fuels):
            rows.append(
                _rec(
                    _eia_period(h),
                    value=str(1000 * (i + 1)),
                    fueltype=fuel,
                    respondent=respondent,
                )
            )
    return rows


class TestGenerationMode:
    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_fetch_failure(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="generation", region="CISO")
        assert not r.success

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_empty_records_is_a_failure(self, mock_fetch, tool):
        mock_fetch.return_value = _win([], total=541815)
        r = tool.execute(mode="generation", region="CISO")
        assert not r.success
        assert "No generation data" in r.output
        tool._cache.put.assert_not_called()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_successful_generation(self, mock_fetch, tool):
        mock_fetch.return_value = _win(_hourly_generation(24))
        r = tool.execute(mode="generation", region="CISO")
        assert r.success, r.output
        assert "Generation Mix" in r.output
        assert "Renewable share" in r.output
        assert "Fossil share" in r.output
        tool._cache.put.assert_called_once()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_generation_length_not_tied_to_our_fuel_table(self, mock_fetch, tool):
        """CISO publishes GEO and ERCO publishes BAT — EIA's fuel vocabulary is
        wider than EIA_FUEL_TYPES, so sizing `length` by len() under-fetches."""
        from agent.tools.electricity_monitor import EIA_FUEL_TYPES

        mock_fetch.return_value = _win(_hourly_generation(24))
        tool.execute(mode="generation", region="CISO")
        assert mock_fetch.call_args.kwargs["length"] > 24 * len(EIA_FUEL_TYPES)

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_generation_short_window_is_red(self, mock_fetch, tool):
        mock_fetch.return_value = _win(_hourly_generation(4))
        r = tool.execute(mode="generation", region="CISO")
        assert not r.success
        assert "only 4 of 24 requested hour(s)" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_generation_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = {
            "text": "cached gen",
            "records": _hourly_generation(24),
        }
        r = tool.execute(mode="generation", region="CISO")
        assert r.success
        mock_fetch.assert_not_called()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_generation_fuel_labels(self, mock_fetch, tool):
        mock_fetch.return_value = _win(_hourly_generation(24, fuels=("COL", "WAT")))
        r = tool.execute(mode="generation", region="PJM")
        assert r.success, r.output
        assert "Coal" in r.output
        assert "Hydro" in r.output


# ══════════════════════════════════════════════════════════════════════════
# 4. Interchange mode
# ══════════════════════════════════════════════════════════════════════════


def _hourly_interchange(hours=24, *, direction="from", partner="NYIS", value="1000"):
    key = "toba" if direction == "from" else "fromba"
    other = "fromba" if direction == "from" else "toba"
    return [_rec(_eia_period(h), value=value, **{key: partner, other: "PJM"}) for h in range(hours)]


class TestInterchangeMode:
    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_both_fetch_failure(self, mock_fetch, tool):
        mock_fetch.return_value = None
        r = tool.execute(mode="interchange", region="PJM")
        assert not r.success
        assert "Failed" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_one_direction_failure_is_red(self, mock_fetch, tool):
        """A 5xx on one leg used to print "Total imports: 0 MWh" and a net
        position computed from it, then persist the surviving half as
        authoritative history — with success=True."""
        mock_fetch.side_effect = [_win(_hourly_interchange(24, direction="from")), None]
        r = tool.execute(mode="interchange", region="PJM")
        assert not r.success
        assert "imports (toba)" in r.output
        assert "0 MWh" not in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_both_directions(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            _win(_hourly_interchange(24, direction="from", partner="NYIS", value="5000")),
            _win(_hourly_interchange(24, direction="to", partner="CPLE", value="4000")),
        ]
        r = tool.execute(mode="interchange", region="PJM")
        assert r.success, r.output
        assert "Exports" in r.output or "export" in r.output
        assert "Imports" in r.output or "import" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_net_import(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            _win(_hourly_interchange(24, direction="from", partner="X", value="1000")),
            _win(_hourly_interchange(24, direction="to", partner="Y", value="5000")),
        ]
        r = tool.execute(mode="interchange", region="PJM")
        assert r.success, r.output
        assert "Net: import" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_net_export(self, mock_fetch, tool):
        mock_fetch.side_effect = [
            _win(_hourly_interchange(24, direction="from", partner="X", value="5000")),
            _win(_hourly_interchange(24, direction="to", partner="Y", value="1000")),
        ]
        r = tool.execute(mode="interchange", region="PJM")
        assert r.success, r.output
        assert "Net: export" in r.output

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_interchange_empty_is_red(self, mock_fetch, tool):
        mock_fetch.side_effect = [_win([], total=0), _win([], total=0)]
        r = tool.execute(mode="interchange", region="PJM")
        assert not r.success
        assert "No interchange data" in r.output
        tool._cache.put.assert_not_called()

    @patch("agent.tools.electricity_monitor._fetch_eia")
    def test_interchange_cache_hit(self, mock_fetch, tool):
        tool._cache.get.return_value = {
            "text": "cached interchange",
            "records": _hourly_interchange(24),
        }
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

    # -- _clean_period --
    def test_clean_period_absent(self):
        assert _clean_period(None, bound="start") is None
        assert _clean_period("  ", bound="end") is None

    def test_clean_period_hourly(self):
        assert _clean_period("2026-09-20T05", bound="start") == "2026-09-20T05"

    def test_clean_period_date_widens_to_whole_day(self):
        assert _clean_period("2026-09-20", bound="start") == "2026-09-20T00"
        assert _clean_period("2026-09-20", bound="end") == "2026-09-20T23"

    def test_clean_period_garbage_raises(self):
        """Dropping a bad bound silently widens the window to "most recent N"
        and still reports success — a backfill refilling the wrong days."""
        with pytest.raises(PeriodFormatError):
            _clean_period("last tuesday", bound="start")
        with pytest.raises(PeriodFormatError):
            _clean_period(20260920, bound="end")

    # -- _expected_hours --
    def test_expected_hours_from_days(self):
        assert _expected_hours(None, None, 3) == 72

    def test_expected_hours_from_window(self):
        assert _expected_hours("2026-09-20T00", "2026-09-20T23", 1) == 24

    def test_expected_hours_half_bounded_unknowable(self):
        assert _expected_hours("2026-09-20T00", None, 1) is None

    def test_expected_hours_reversed_window(self):
        assert _expected_hours("2026-09-21T00", "2026-09-20T00", 1) is None


# ══════════════════════════════════════════════════════════════════════════
# 6. Fetch EIA function
# ══════════════════════════════════════════════════════════════════════════


class TestFetchEia:
    @patch("agent.tools.electricity_monitor.httpx.get")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response(200, json_data={"response": {"total": 1, "data": [{"value": 42}]}})
        result = _fetch_eia("electricity/rto/region-data", None, "key")
        assert result is not None
        assert len(result) == 1
        assert result.total == 1
        assert result.complete

    @patch("agent.tools.electricity_monitor.httpx.get")
    def test_total_is_read_not_discarded(self, mock_get):
        """The whole F-16 fix: a short read must be distinguishable."""
        mock_get.return_value = _mock_response(
            200, json_data={"response": {"total": "270532", "data": [{"value": 1}, {"value": 2}]}}
        )
        result = _fetch_eia("electricity/rto/region-data", None, "key", length=100)
        assert result.total == 270532
        assert not result.complete

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
        mock_get.return_value = _mock_response(200, json_data={"response": {"data": []}})
        result = _fetch_eia("test", {"respondent": ["PJM"]}, "key")
        assert result is not None
        sent = mock_get.call_args.kwargs["params"]
        assert ("facets[respondent][]", "PJM") in sent

    def test_pages_with_offset_until_total_is_reached(self):
        """EIA caps a page; the rest is behind `offset`. Without paging a
        window wider than one page is silently truncated."""
        rows = [_rec(_eia_period(h), type_code="D", value=str(h)) for h in range(25)]
        fake = FakeEIA(rows)
        with (
            patch("agent.tools.electricity_monitor.httpx.get", fake),
            patch("agent.tools.electricity_monitor._EIA_MAX_PAGE", 10),
        ):
            result = _fetch_eia("electricity/rto/region-data", None, "key", length=25)
        assert result is not None
        assert result.total == 25
        assert len(result) == 25
        assert result.complete
        offsets = [dict(c)["offset"] for c in fake.calls]
        assert offsets == [0, 10, 20]

    def test_failure_on_a_later_page_fails_the_whole_fetch(self):
        """Returning the pages that happened to succeed is F-16 itself: a
        partial window that nothing retries."""
        rows = [_rec(_eia_period(h), type_code="D", value=str(h)) for h in range(25)]
        fake = FakeEIA(rows)
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                return _mock_response(503)
            return fake(*a, **kw)

        with (
            patch("agent.tools.electricity_monitor.httpx.get", flaky),
            patch("agent.tools.electricity_monitor._EIA_MAX_PAGE", 10),
        ):
            result = _fetch_eia("electricity/rto/region-data", None, "key", length=25)
        assert result is None


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
        assert len(names) == 61, f"Expected 61, got {len(names)}: {names}"

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


# ══════════════════════════════════════════════════════════════════════════
# 10. Regression: green checkmark, zero rows (2026-09-23)
# ══════════════════════════════════════════════════════════════════════════
#
# This node had written 0 rows in its entire life, via three stacked defects:
#
#   1. the DAG registered it without the required "region" param, so
#      ToolOperator raised and the node failed red;
#   2. _fetch_eia never sent EIA v2's mandatory "data[0]=value" selector, so
#      EIA answered HTTP 200 with the right periods and *no* value field and
#      every reading parsed as 0 MW;
#   3. persistence wrote exactly one metadata-only row per invocation —
#      {"mode","region","region_name"} — with no MW, no period, no stats.
#
# The second-pass audit found a fourth, worse than all of them: the row
# budget (`length = days * 24 * 2`) was sized for one series while
# region-data serves four, so a day's collection recovered 14 of 24 hours
# and hours 14-23 UTC were never collected by any daily run. Nothing in the
# suite could see it, because the fake handed the collector three records
# and the test asserted three rows.
#
# FakeEIA (top of file) is the fix for that: it filters facets server-side,
# sorts then slices by length/offset, and reports a `total` bigger than the
# page — so the tests below measure hours recovered against hours published.


def _count_rows(db_path, obs_type="grid_demand"):
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT observed_at, value_json FROM entity_observations "
        "WHERE source_tool = 'electricity_monitor' AND observation_type = ? "
        "ORDER BY observed_at",
        (obs_type,),
    ).fetchall()
    conn.close()
    return rows


class TestSilentFailureRegression:
    def test_demand_persists_one_row_per_hourly_reading(self, tmp_path):
        """success=True is not the bar — one stored row per published hour is.

        The fake serves 48 hours x {D, NG, TI} plus day-ahead DF rows dated
        past the newest actual hour, exactly as region-data does. Remove
        `facets[type][]=D` from the collector and those DF rows take the top
        of the sort-desc page, the demand budget starves, and the hour-count
        assertion below fails.
        """
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"
        fake = FakeEIA(_region_data_rows(demand_hours=48))

        with patch("agent.tools.electricity_monitor.httpx.get", fake):
            result = tool.execute(mode="demand", region="PJM")

        assert result.success, result.output

        # The request itself: value selector present, demand faceted server-side.
        sent = fake.params_of(0)
        assert ("data[0]", "value") in sent
        assert ("facets[type][]", "D") in sent
        assert dict(sent)["length"] == 24 + 24

        rows = _count_rows(tmp_path / "probe.db")
        # 48 hourly D readings are reachable inside the budget; every one of
        # them must land. Under the old `days*24*2` budget this was 12.
        assert len(rows) == 48, f"expected 48 hourly rows, got {len(rows)}"

        values = [json.loads(v) for _, v in rows]
        # Defect 2: without data[0]=value every reading parses as 0 MW.
        assert [v["mw"] for v in values] == [float(70000 + h) for h in range(48)]
        assert all(v["region"] == "PJM" for v in values)

        # observed_at is the reading's own period, not time.time().
        assert [ts for ts, _ in rows] == [T(h * 3600) for h in range(48)]

        # Net generation / total interchange / day-ahead forecast never became
        # demand history.
        assert store.observation_write_stats()["rejected"] == {}

        # Re-running collapses onto the same rows instead of double-counting.
        with patch("agent.tools.electricity_monitor.httpx.get", fake):
            tool.execute(mode="demand", region="PJM")
        assert len(_count_rows(tmp_path / "probe.db")) == 48

    def test_missing_type_facet_would_starve_the_window(self, tmp_path):
        """The counter-test: this is what the collector did for 26 days.

        Faceting only by respondent — the pre-fix query — leaves the DF
        forecast rows in the payload, and a sort-desc page of 48 is spent
        before it reaches a full day of actual demand. Kept as an executable
        record of why `facets[type][]=D` is load-bearing.
        """
        fake = FakeEIA(_region_data_rows(demand_hours=48))
        with patch("agent.tools.electricity_monitor.httpx.get", fake):
            window = _fetch_eia(
                "electricity/rto/region-data",
                {"respondent": ["PJM"]},  # the old facets
                "test-eia-key",
                length=48,  # the old `days * 24 * 2`
            )
        d_hours = {r["period"] for r in window.records if r.get("type") == "D"}
        assert len(d_hours) < 24, "fake no longer reproduces the starved window"

        # And with the fix, the same budget is all demand.
        fake2 = FakeEIA(_region_data_rows(demand_hours=48))
        with patch("agent.tools.electricity_monitor.httpx.get", fake2):
            fixed = _fetch_eia(
                "electricity/rto/region-data",
                {"respondent": ["PJM"], "type": ["D"]},
                "test-eia-key",
                length=48,
            )
        assert len({r["period"] for r in fixed.records}) == 48

    def test_store_rejection_fails_the_run(self, tmp_path):
        """90% of a window lost used to print "Stored 1 observation(s)." and
        a green checkmark, because the only guard was written == 0."""
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        real_write = store.store_entity_observation
        calls = {"n": 0}

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("disk full")
            return real_write(**kwargs)

        store.store_entity_observation = flaky  # type: ignore[method-assign]
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"

        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(_region_data_rows(24))):
            result = tool.execute(mode="demand", region="PJM")

        assert not result.success, result.output
        assert "PARTIAL PERSISTENCE" in result.output
        assert len(_count_rows(tmp_path / "probe.db")) == 1

    def test_total_persistence_failure_fails_the_run(self, tmp_path):
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        store.store_entity_observation = MagicMock(side_effect=RuntimeError("disk full"))
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"

        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(_region_data_rows(24))):
            result = tool.execute(mode="demand", region="PJM")

        assert not result.success
        assert "PERSISTENCE FAILURE" in result.output

    def test_generation_future_row_is_filtered_not_rejected(self, tmp_path):
        """The `now + 12h` guard has no type filter standing behind it in
        generation mode, so this is where it is actually load-bearing.

        Delete the guard and the future-dated row reaches PipelineStore,
        which refuses observed_at > now + 1 day — that becomes a counted
        rejection and the run goes red.
        """
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"

        rows = _hourly_generation(24, fuels=("NG", "SUN"))
        future = datetime.fromtimestamp(time.time() + 3 * 86400, tz=UTC).strftime("%Y-%m-%dT%H")
        rows.append(_rec(future, value="9999", fueltype="SUN", respondent="CISO"))

        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(rows)):
            result = tool.execute(mode="generation", region="CISO")

        assert result.success, result.output
        assert store.observation_write_stats()["rejected"] == {}
        stored = _count_rows(tmp_path / "probe.db", "grid_generation")
        assert len(stored) == 48, f"expected 24h x 2 fuels, got {len(stored)}"
        assert all(ts < time.time() for ts, _ in stored)

    def test_interchange_persists_both_directions(self, tmp_path):
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"

        rows = _hourly_interchange(24, direction="from", partner="NYIS", value="1200")
        rows += _hourly_interchange(24, direction="to", partner="CPLE", value="900")

        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(rows)):
            result = tool.execute(mode="interchange", region="PJM")

        assert result.success, result.output
        stored = _count_rows(tmp_path / "probe.db", "grid_interchange")
        assert len(stored) == 48

    def test_cache_hit_still_writes_rows(self, tmp_path):
        """Production builds this tool with a real 6h-TTL DataCache; every
        pre-audit test built it with cache=None, so the cache-hit path — a
        green/zero-rows path — had no coverage at all."""
        cache = DataCache(cache_dir=str(tmp_path / "cache"), ttl_seconds=3600)
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = ElectricityMonitorTool(cache=cache, pipeline_store=store)
        tool._api_key = "test-eia-key"
        fake = FakeEIA(_region_data_rows(demand_hours=24))

        with patch("agent.tools.electricity_monitor.httpx.get", fake):
            first = tool.execute(mode="demand", region="PJM")
        assert first.success, first.output
        assert len(_count_rows(tmp_path / "probe.db")) == 24
        http_calls = len(fake.calls)

        # Second run is served from cache — and must still persist.
        store2 = PipelineStore(db_path=str(tmp_path / "probe2.db"))
        tool2 = ElectricityMonitorTool(cache=cache, pipeline_store=store2)
        tool2._api_key = "test-eia-key"
        with patch("agent.tools.electricity_monitor.httpx.get", fake):
            second = tool2.execute(mode="demand", region="PJM")

        assert second.success, second.output
        assert len(fake.calls) == http_calls, "cache hit should not re-fetch"
        assert len(_count_rows(tmp_path / "probe2.db")) == 24, "cache hit wrote nothing"

    def test_empty_response_is_not_cached(self, tmp_path):
        """Caching a zero replays it for the whole 6h TTL without asking EIA."""
        cache = DataCache(cache_dir=str(tmp_path / "cache"), ttl_seconds=3600)
        tool = ElectricityMonitorTool(cache=cache)
        tool._api_key = "test-eia-key"

        empty = FakeEIA([])
        with patch("agent.tools.electricity_monitor.httpx.get", empty):
            first = tool.execute(mode="demand", region="PJM")
        assert not first.success

        full = FakeEIA(_region_data_rows(demand_hours=24))
        with patch("agent.tools.electricity_monitor.httpx.get", full):
            second = tool.execute(mode="demand", region="PJM")
        assert second.success, second.output
        assert full.calls, "emptiness was cached and replayed"

    def test_incomplete_result_is_not_cached(self, tmp_path):
        """A shortfall must not be laundered into a clean run by the cache.

        A cache hit carries no completeness complaint of its own, so caching
        a 10-of-24-hour window would replay it as success for the whole TTL.
        """
        cache = DataCache(cache_dir=str(tmp_path / "cache"), ttl_seconds=3600)
        tool = ElectricityMonitorTool(cache=cache)
        tool._api_key = "test-eia-key"

        short = [_rec(_eia_period(h), type_code="D", type_name="Demand", value=str(70000 + h)) for h in range(10)]
        fake = FakeEIA(short)
        with patch("agent.tools.electricity_monitor.httpx.get", fake):
            first = tool.execute(mode="demand", region="PJM")
        assert not first.success
        assert "only 10 of 24 requested hour(s)" in first.output

        again = FakeEIA(short)
        with patch("agent.tools.electricity_monitor.httpx.get", again):
            second = tool.execute(mode="demand", region="PJM")
        assert again.calls, "the incomplete window was cached and replayed"
        assert not second.success

    # -- start/end backfill window --

    def test_start_end_bounds_are_sent_and_measured(self, tmp_path):
        """A bounded window makes `total` the exact expected count, which is
        the only place completeness can be checked without guessing."""
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"
        fake = FakeEIA(_region_data_rows(demand_hours=48))

        start, end = _eia_period(10), _eia_period(19)
        with patch("agent.tools.electricity_monitor.httpx.get", fake):
            result = tool.execute(mode="demand", region="PJM", start=start, end=end)

        sent = dict(fake.params_of(0))
        assert sent["start"] == start
        assert sent["end"] == end
        assert result.success, result.output
        rows = _count_rows(tmp_path / "probe.db")
        assert len(rows) == 10, f"expected the 10 requested hours, got {len(rows)}"
        assert [ts for ts, _ in rows] == [T(h * 3600) for h in range(10, 20)]

    def test_date_only_bounds_cover_the_whole_day(self, tmp_path):
        tool = ElectricityMonitorTool(cache=None)
        tool._api_key = "test-eia-key"
        day = _eia_period(0)[:10]
        fake = FakeEIA(_region_data_rows(demand_hours=48))
        with patch("agent.tools.electricity_monitor.httpx.get", fake):
            result = tool.execute(mode="demand", region="PJM", start=day, end=day)
        sent = dict(fake.params_of(0))
        assert sent["start"] == f"{day}T00"
        assert sent["end"] == f"{day}T23"
        assert result.success, result.output

    def test_unparseable_bound_is_red(self, tool):
        r = tool.execute(mode="demand", region="PJM", start="last tuesday")
        assert not r.success
        assert "Invalid window bound" in r.output

    def test_bounded_window_shortfall_is_red(self, tmp_path):
        """Ask for 24 hours, get 10: the backfill did not refill the gap."""
        tool = ElectricityMonitorTool(cache=None)
        tool._api_key = "test-eia-key"
        # Only hours 10-19 exist upstream; hours 0-23 were requested.
        rows = [_rec(_eia_period(h), type_code="D", type_name="Demand", value=str(70000 + h)) for h in range(10, 20)]
        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(rows)):
            result = tool.execute(mode="demand", region="PJM", start=_eia_period(0), end=_eia_period(23))
        assert not result.success
        assert "only 10 of 24 requested hour(s)" in result.output

    def test_dag_node_supplies_required_region(self):
        """Defect 1: the node omitted 'region', which the tool requires."""
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag

        node = build_daily_collection_dag().nodes["fetch_electricity_monitor"]
        required = ElectricityMonitorTool().parameters["required"]
        missing = [p for p in required if p not in node.params]
        assert not missing, f"DAG node missing required tool params: {missing}"
        assert node.params["region"] in KNOWN_REGIONS


# ══════════════════════════════════════════════════════════════════════════
# 11. Second-pass audit: what the first repair's own tests could not see
# ══════════════════════════════════════════════════════════════════════════
#
# Each test below was written against a surviving mutant — a change to the
# collector that broke a stated guarantee while the whole suite stayed green.
# The mutation is named in the docstring so the next reader can re-run it.


class TestSecondPassRegression:
    def test_interior_gap_in_the_requested_day_is_red(self, tmp_path):
        """A hole inside the requested day must not be papered over by the
        buffer hours on either side of it.

        The demand budget deliberately over-fetches (hours + 24) to absorb
        EIA's publication lag. Counting *every* distinct hour that budget
        returns means older hours can pay for missing recent ones: with hours
        30-35 absent, the collector stored 18 of the last 24 hours and still
        printed "Coverage: 42 distinct hour(s) of 24 requested" with
        success=True. That is the 14-of-24 defect moved from the query into
        the gate.
        """
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"

        present = list(range(0, 30)) + list(range(36, 48))  # 30-35 missing
        rows = [_rec(_eia_period(h), type_code="D", type_name="Demand", value=str(70000 + h)) for h in present]

        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(rows)):
            result = tool.execute(mode="demand", region="PJM")

        assert not result.success, result.output
        assert "only 18 of 24 requested hour(s)" in result.output
        assert "6 hour(s) missing" in result.output
        # The rows we did get are still kept — throwing them away helps nobody.
        assert len(_count_rows(tmp_path / "probe.db")) == 42
        # And the human-readable line agrees with the gate rather than
        # flattering it.
        assert "Coverage: 18 of 24 requested hour(s)" in result.output

    def test_lagging_edge_is_not_a_gap(self, tmp_path):
        """The counter-test: EIA is hours behind in real time, always.

        Hours 38-47 simply not being published yet is publication lag, not a
        collection gap, and must stay green — otherwise the gate above reds
        every run and gets muted.
        """
        tool = ElectricityMonitorTool(cache=None)
        tool._api_key = "test-eia-key"
        rows = [_rec(_eia_period(h), type_code="D", type_name="Demand", value=str(70000 + h)) for h in range(38)]
        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(rows)):
            result = tool.execute(mode="demand", region="PJM")
        assert result.success, result.output

    def test_valueless_rows_are_not_stored_as_zero_and_fail_the_run(self, tmp_path):
        """Mutation: `_strict_float` -> `_safe_float` in
        `_observation_from_record` left all 87 tests green.

        A record with no `value` is the data[0]=value defect itself. Parsed
        leniently it becomes a fabricated 0 MW reading; dropped quietly it
        loses the hour while the surviving rows land under a green tick and a
        confident count. Neither is acceptable: nothing is stored for it, and
        the run reports the loss.
        """
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"

        rows = [_rec(_eia_period(h), type_code="D", type_name="Demand", value=str(70000 + h)) for h in range(12)]
        # Hours 12-23: EIA returned the period but no value at all.
        rows += [_rec(_eia_period(h), type_code="D", type_name="Demand") for h in range(12, 24)]

        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(rows)):
            result = tool.execute(mode="demand", region="PJM")

        assert not result.success, result.output
        assert "MALFORMED ROWS: 12 of 24" in result.output

        stored = [json.loads(v) for _, v in _count_rows(tmp_path / "probe.db")]
        assert len(stored) == 12
        # Not one fabricated zero.
        assert all(v["mw"] > 0 for v in stored), stored

    def test_future_forecast_row_is_filtered_not_counted_malformed(self, tmp_path):
        """The counter-test to the one above: a deliberately filtered row is
        not a defect and must not turn the run red."""
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"

        rows = _region_data_rows(demand_hours=48)
        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(rows)):
            result = tool.execute(mode="demand", region="PJM")

        assert result.success, result.output
        assert "MALFORMED" not in result.output

    def test_one_sided_interchange_gap_is_red(self, tmp_path):
        """Coverage was measured on both directions merged, so a direction
        missing half its hours was masked by the other direction having them.

        Exports cover 24 hours, imports only 12; the merged set still has 24
        distinct hours, so the pre-audit gate saw a complete window and the
        printed net position was computed from half an import series.
        """
        store = PipelineStore(db_path=str(tmp_path / "probe.db"))
        tool = ElectricityMonitorTool(cache=None, pipeline_store=store)
        tool._api_key = "test-eia-key"

        rows = _hourly_interchange(24, direction="from", partner="NYIS", value="1200")
        rows += _hourly_interchange(12, direction="to", partner="CPLE", value="900")

        with patch("agent.tools.electricity_monitor.httpx.get", FakeEIA(rows)):
            result = tool.execute(mode="interchange", region="PJM")

        assert not result.success, result.output
        assert "interchange/imports: only 12 of 24" in result.output
        assert "interchange/exports" not in result.output.split("INCOMPLETE COLLECTION")[1]
        # The half we have is still persisted, just not reported as complete.
        assert len(_count_rows(tmp_path / "probe.db", "grid_interchange")) == 36

    def test_store_cache_refuses_an_empty_record_list(self, tmp_path):
        """Mutation: dropping `not records` from `_store_cache` left the suite
        green, because every call site happens to guard emptiness first.

        That makes the guard dead code today and load-bearing the moment a
        new mode forgets its own guard — a cached empty entry replays a
        zero-row success for the whole 6h TTL. Pin it directly.
        """
        cache = DataCache(cache_dir=str(tmp_path / "cache"), ttl_seconds=3600)
        tool = ElectricityMonitorTool(cache=cache)
        tool._api_key = "test-eia-key"

        tool._store_cache("electricity_demand", "PJM_1__", "text", [], [])
        assert cache.get("electricity_demand", "PJM_1__") is None

        tool._store_cache("electricity_demand", "PJM_1__", "text", _hourly_demand(2), [])
        assert cache.get("electricity_demand", "PJM_1__") is not None
