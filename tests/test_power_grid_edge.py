"""
Edge case tests for PowerGridTool (NYISO).

Covers: mode validation, zone validation, date validation, CSV parsing,
archive fallback, demand/fuel_mix/pricing/forecast modes, zone filtering,
zone aliases, signal computation, error handling, cache interaction,
tool metadata, output formatting, live network tests.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.power_grid import NYISO_ZONES, PowerGridTool, _safe_float

# ── Fixtures ──────────────────────────────────────────────────


def _make_demand_csv(rows: list[dict] | None = None) -> str:
    """Build a fake NYISO pal (demand) CSV."""
    header = "Time Stamp,Time Zone,Name,PTID,Load\n"
    if rows is None:
        rows = [
            {
                "ts": "03/25/2026 00:00:00",
                "tz": "EDT",
                "name": "N.Y.C.",
                "ptid": "61761",
                "load": "4500.0",
            },
            {
                "ts": "03/25/2026 00:05:00",
                "tz": "EDT",
                "name": "N.Y.C.",
                "ptid": "61761",
                "load": "4520.0",
            },
            {
                "ts": "03/25/2026 00:10:00",
                "tz": "EDT",
                "name": "N.Y.C.",
                "ptid": "61761",
                "load": "4480.0",
            },
            {
                "ts": "03/25/2026 00:00:00",
                "tz": "EDT",
                "name": "CAPITL",
                "ptid": "61757",
                "load": "1100.0",
            },
            {
                "ts": "03/25/2026 00:05:00",
                "tz": "EDT",
                "name": "CAPITL",
                "ptid": "61757",
                "load": "1120.0",
            },
            {
                "ts": "03/25/2026 00:10:00",
                "tz": "EDT",
                "name": "CAPITL",
                "ptid": "61757",
                "load": "1080.0",
            },
        ]
    lines = header
    for r in rows:
        lines += f"{r['ts']},{r['tz']},{r['name']},{r['ptid']},{r['load']}\n"
    return lines


def _make_fuel_mix_csv(rows: list[dict] | None = None) -> str:
    """Build a fake NYISO rtfuelmix CSV."""
    header = "Time Stamp,Time Zone,Fuel Category,Gen MWh\n"
    if rows is None:
        rows = [
            {
                "ts": "03/25/2026 04:00:00",
                "tz": "EDT",
                "fuel": "Natural Gas",
                "gen": "3500.0",
            },
            {
                "ts": "03/25/2026 04:00:00",
                "tz": "EDT",
                "fuel": "Nuclear",
                "gen": "2000.0",
            },
            {
                "ts": "03/25/2026 04:00:00",
                "tz": "EDT",
                "fuel": "Hydro",
                "gen": "1500.0",
            },
            {
                "ts": "03/25/2026 04:05:00",
                "tz": "EDT",
                "fuel": "Natural Gas",
                "gen": "3600.0",
            },
            {
                "ts": "03/25/2026 04:05:00",
                "tz": "EDT",
                "fuel": "Nuclear",
                "gen": "1990.0",
            },
            {
                "ts": "03/25/2026 04:05:00",
                "tz": "EDT",
                "fuel": "Hydro",
                "gen": "1510.0",
            },
        ]
    lines = header
    for r in rows:
        lines += f"{r['ts']},{r['tz']},{r['fuel']},{r['gen']}\n"
    return lines


def _make_da_lbmp_csv(rows: list[dict] | None = None) -> str:
    """Build a fake NYISO damlbmp_zone CSV."""
    header = "Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)\n"
    if rows is None:
        rows = [
            {
                "ts": "03/25/2026 00:00",
                "name": "N.Y.C.",
                "ptid": "61761",
                "lbmp": "42.50",
                "loss": "1.20",
                "cong": "0.30",
            },
            {
                "ts": "03/25/2026 01:00",
                "name": "N.Y.C.",
                "ptid": "61761",
                "lbmp": "40.00",
                "loss": "1.10",
                "cong": "0.25",
            },
            {
                "ts": "03/25/2026 00:00",
                "name": "CAPITL",
                "ptid": "61757",
                "lbmp": "38.00",
                "loss": "0.90",
                "cong": "0.00",
            },
            {
                "ts": "03/25/2026 01:00",
                "name": "CAPITL",
                "ptid": "61757",
                "lbmp": "36.50",
                "loss": "0.85",
                "cong": "0.00",
            },
        ]
    lines = header
    for r in rows:
        lines += f"{r['ts']},{r['name']},{r['ptid']},{r['lbmp']},{r['loss']},{r['cong']}\n"
    return lines


def _make_rt_lbmp_csv(rows: list[dict] | None = None) -> str:
    """Build a fake NYISO realtime_zone CSV. Note: quoted fields like real data."""
    header = '"Time Stamp","Name","PTID","LBMP ($/MWHr)","Marginal Cost Losses ($/MWHr)","Marginal Cost Congestion ($/MWHr)"\n'
    if rows is None:
        rows = [
            {
                "ts": "03/25/2026 00:00:00",
                "name": "N.Y.C.",
                "ptid": "61761",
                "lbmp": "45.00",
                "loss": "1.50",
                "cong": "0.50",
            },
            {
                "ts": "03/25/2026 00:05:00",
                "name": "N.Y.C.",
                "ptid": "61761",
                "lbmp": "44.50",
                "loss": "1.45",
                "cong": "0.48",
            },
            {
                "ts": "03/25/2026 00:00:00",
                "name": "CAPITL",
                "ptid": "61757",
                "lbmp": "35.00",
                "loss": "0.80",
                "cong": "0.00",
            },
            {
                "ts": "03/25/2026 00:05:00",
                "name": "CAPITL",
                "ptid": "61757",
                "lbmp": "34.80",
                "loss": "0.78",
                "cong": "0.00",
            },
        ]
    lines = header
    for r in rows:
        lines += f'"{r["ts"]}","{r["name"]}","{r["ptid"]}","{r["lbmp"]}","{r["loss"]}","{r["cong"]}"\n'
    return lines


def _make_forecast_csv() -> str:
    """Build a fake NYISO isolf CSV (columnar format)."""
    header = '"Time Stamp","Capitl","Centrl","Dunwod","Genese","Hud Vl","Longil","Mhk Vl","Millwd","N.Y.C.","North","West","NYISO"\n'
    rows = [
        '"03/25/2026 00:00",1100,1500,550,960,980,1900,800,270,4600,600,1550,14810\n',
        '"03/25/2026 01:00",1080,1480,530,940,960,1850,790,265,4500,590,1530,14525\n',
        '"03/25/2026 02:00",1060,1460,510,920,940,1800,780,260,4400,580,1510,14220\n',
    ]
    return header + "".join(rows)


def _make_tool(cache=None) -> PowerGridTool:
    return PowerGridTool(cache=cache)


def _mock_response(text: str, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx Response."""
    resp = httpx.Response(status_code=status_code, text=text, request=httpx.Request("GET", "http://test"))
    return resp


# ── 1. Input Validation ──────────────────────────────────────


class TestInputValidation:
    """Mode, zone, and date input validation."""

    def test_invalid_mode(self):
        t = _make_tool()
        r = t.execute(mode="garbage")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_missing_mode(self):
        t = _make_tool()
        r = t.execute()
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        t = _make_tool()
        r = t.execute(mode="")
        assert not r.success

    def test_invalid_zone(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", return_value=[]):
            r = t.execute(mode="demand", zone="FAKE_ZONE")
        assert not r.success
        assert "Unknown zone" in r.output
        assert "CAPITL" in r.output  # lists valid zones

    def test_future_date(self):
        t = _make_tool()
        future = (date.today() + timedelta(days=5)).strftime("%Y-%m-%d")
        r = t.execute(mode="demand", date=future)
        assert not r.success
        assert "future" in r.output.lower()

    def test_invalid_date_format(self):
        t = _make_tool()
        r = t.execute(mode="demand", date="25-03-2026")
        assert not r.success
        assert "Invalid date format" in r.output

    def test_date_bad_string(self):
        t = _make_tool()
        r = t.execute(mode="demand", date="not-a-date")
        assert not r.success

    def test_valid_modes(self):
        t = _make_tool()
        for mode in ("demand", "fuel_mix", "pricing", "forecast"):
            with patch.object(t, "_fetch_csv", return_value=None):
                r = t.execute(mode=mode)
            # Should fail due to no data, not validation
            assert "Invalid mode" not in r.output


# ── 2. Tool Metadata ─────────────────────────────────────────


class TestToolMetadata:
    """Tool name, description, parameters, OpenAI schema."""

    def test_name(self):
        assert _make_tool().name == "power_grid"

    def test_description_nonempty(self):
        assert len(_make_tool().description) > 20

    def test_parameters_schema(self):
        params = _make_tool().parameters
        assert params["type"] == "object"
        assert "mode" in params["properties"]
        assert "zone" in params["properties"]
        assert "date" in params["properties"]
        assert params["required"] == ["mode"]

    def test_mode_enum(self):
        modes = _make_tool().parameters["properties"]["mode"]["enum"]
        assert set(modes) == {"demand", "fuel_mix", "pricing", "forecast"}

    def test_openai_tool_schema(self):
        schema = _make_tool().to_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "power_grid"


# ── 3. Zone Normalization ────────────────────────────────────


class TestZoneNormalization:
    """Zone alias mapping and case insensitivity."""

    def test_nyc_alias(self):
        assert PowerGridTool._normalize_zone("NYC") == "N.Y.C."

    def test_capital_alias(self):
        assert PowerGridTool._normalize_zone("CAPITAL") == "CAPITL"

    def test_hudson_valley_alias(self):
        assert PowerGridTool._normalize_zone("HUDSON VALLEY") == "HUD VL"

    def test_long_island_alias(self):
        assert PowerGridTool._normalize_zone("LONG ISLAND") == "LONGIL"

    def test_mohawk_alias(self):
        assert PowerGridTool._normalize_zone("MOHAWK") == "MHK VL"

    def test_no_mapping_passthrough(self):
        assert PowerGridTool._normalize_zone("CAPITL") == "CAPITL"

    def test_zone_case_insensitive(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", return_value=None):
            r = t.execute(mode="demand", zone="n.y.c.")
        # Should not fail on zone validation — N.Y.C. is valid
        assert "Unknown zone" not in r.output

    def test_zone_whitespace_stripped(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", return_value=None):
            r = t.execute(mode="demand", zone="  CAPITL  ")
        assert "Unknown zone" not in r.output


# ── 4. CSV Parsing ───────────────────────────────────────────


class TestCsvParsing:
    """CSV parse edge cases."""

    def test_normal_csv(self):
        t = _make_tool()
        rows = t._parse_csv(_make_demand_csv())
        assert len(rows) == 6
        assert rows[0]["Name"] == "N.Y.C."

    def test_empty_csv(self):
        t = _make_tool()
        rows = t._parse_csv("")
        assert rows == []

    def test_header_only_csv(self):
        t = _make_tool()
        rows = t._parse_csv("Time Stamp,Name,PTID,Load\n")
        assert rows == []

    def test_whitespace_csv(self):
        t = _make_tool()
        rows = t._parse_csv("   \n  \n")
        assert rows == []

    def test_quoted_csv(self):
        t = _make_tool()
        rows = t._parse_csv(_make_rt_lbmp_csv())
        assert len(rows) == 4
        # Quoted fields should be unquoted by csv module
        assert "N.Y.C." in rows[0].get("Name", "")

    def test_extra_columns_ignored(self):
        t = _make_tool()
        csv_text = "Time Stamp,Name,PTID,Load,ExtraCol\n03/25/2026 00:00:00,N.Y.C.,61761,4500.0,foo\n"
        rows = t._parse_csv(csv_text)
        assert len(rows) == 1
        assert "ExtraCol" in rows[0]


# ── 5. Demand Mode ───────────────────────────────────────────


class TestDemandMode:
    """Demand mode with mocked CSV data."""

    def test_demand_all_zones(self):
        t = _make_tool()
        csv_text = _make_demand_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="demand", date="2026-03-25")
        assert r.success
        assert "N.Y.C." in r.output
        assert "CAPITL" in r.output
        assert r.data["zones"]
        assert len(r.data["zones"]) == 2

    def test_demand_zone_filter(self):
        t = _make_tool()
        csv_text = _make_demand_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="demand", date="2026-03-25", zone="N.Y.C.")
        assert r.success
        assert len(r.data["zones"]) == 1
        assert r.data["zones"][0]["zone"] == "N.Y.C."

    def test_demand_zone_not_found(self):
        t = _make_tool()
        csv_text = _make_demand_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="demand", date="2026-03-25", zone="WEST")
        assert not r.success
        assert "No data for zone" in r.output

    def test_demand_no_data(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", return_value=None):
            r = t.execute(mode="demand", date="2026-03-25")
        assert not r.success
        assert "No demand data" in r.output

    def test_demand_empty_data(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", return_value=[]):
            r = t.execute(mode="demand", date="2026-03-25")
        assert not r.success
        assert "Empty demand data" in r.output

    def test_demand_peak_trough_avg(self):
        t = _make_tool()
        csv_text = _make_demand_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="demand", date="2026-03-25", zone="N.Y.C.")
        zone = r.data["zones"][0]
        assert zone["peak_mw"] == 4520.0
        assert zone["trough_mw"] == 4480.0
        assert zone["avg_mw"] == 4500.0
        assert zone["readings"] == 3

    def test_demand_total_peak(self):
        t = _make_tool()
        csv_text = _make_demand_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="demand", date="2026-03-25")
        # Total peak = peak of NYC + peak of CAPITL
        assert r.data["total_peak_mw"] == 4520.0 + 1120.0

    def test_demand_unparseable_load(self):
        """Rows with non-numeric load values should be skipped."""
        t = _make_tool()
        rows = [
            {
                "ts": "03/25/2026 00:00:00",
                "tz": "EDT",
                "name": "N.Y.C.",
                "ptid": "61761",
                "load": "not_a_number",
            },
            {
                "ts": "03/25/2026 00:05:00",
                "tz": "EDT",
                "name": "N.Y.C.",
                "ptid": "61761",
                "load": "4500.0",
            },
        ]
        csv_text = _make_demand_csv(rows)
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="demand", date="2026-03-25")
        assert r.success
        assert r.data["zones"][0]["readings"] == 1


# ── 6. Fuel Mix Mode ─────────────────────────────────────────


class TestFuelMixMode:
    """Fuel mix mode with mocked CSV data."""

    def test_fuel_mix_normal(self):
        t = _make_tool()
        csv_text = _make_fuel_mix_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="fuel_mix", date="2026-03-25")
        assert r.success
        assert r.data["fuels"]
        # Latest timestamp should be 04:05
        assert "04:05" in r.data["timestamp"]

    def test_fuel_mix_uses_latest_snapshot(self):
        t = _make_tool()
        csv_text = _make_fuel_mix_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="fuel_mix", date="2026-03-25")
        # Should use 04:05 data, not 04:00
        fuels = {f["fuel_type"]: f for f in r.data["fuels"]}
        assert fuels["Natural Gas"]["mw"] == 3600.0
        assert fuels["Nuclear"]["mw"] == 1990.0
        assert fuels["Hydro"]["mw"] == 1510.0

    def test_fuel_mix_proportions(self):
        t = _make_tool()
        csv_text = _make_fuel_mix_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="fuel_mix", date="2026-03-25")
        total = sum(f["mw"] for f in r.data["fuels"])
        for f in r.data["fuels"]:
            expected_pct = round((f["mw"] / total) * 100, 1)
            assert f["pct"] == expected_pct

    def test_fuel_mix_sorted_descending(self):
        t = _make_tool()
        csv_text = _make_fuel_mix_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="fuel_mix", date="2026-03-25")
        mws = [f["mw"] for f in r.data["fuels"]]
        assert mws == sorted(mws, reverse=True)

    def test_fuel_mix_no_data(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", return_value=None):
            r = t.execute(mode="fuel_mix", date="2026-03-25")
        assert not r.success

    def test_fuel_mix_empty_data(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", return_value=[]):
            r = t.execute(mode="fuel_mix", date="2026-03-25")
        assert not r.success

    def test_fuel_mix_zero_generation(self):
        """All generation values are 0."""
        t = _make_tool()
        rows = [
            {
                "ts": "03/25/2026 04:00:00",
                "tz": "EDT",
                "fuel": "Natural Gas",
                "gen": "0",
            },
            {"ts": "03/25/2026 04:00:00", "tz": "EDT", "fuel": "Nuclear", "gen": "0"},
        ]
        csv_text = _make_fuel_mix_csv(rows)
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="fuel_mix", date="2026-03-25")
        assert not r.success
        assert "Zero total generation" in r.output


# ── 7. Pricing Mode ──────────────────────────────────────────


class TestPricingMode:
    """Pricing mode with mocked CSV data."""

    def _setup_pricing(self, da_csv=None, rt_csv=None):
        t = _make_tool()
        da_text = da_csv or _make_da_lbmp_csv()
        rt_text = rt_csv or _make_rt_lbmp_csv()
        da_rows = t._parse_csv(da_text)
        rt_rows = t._parse_csv(rt_text)

        call_count = [0]

        def mock_fetch(dataset, date_str, directory=None):
            call_count[0] += 1
            if "damlbmp" in dataset:
                return da_rows
            if "realtime" in dataset:
                return rt_rows
            return None

        return t, mock_fetch

    def test_pricing_normal(self):
        t, mock = self._setup_pricing()
        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="pricing", date="2026-03-25")
        assert r.success
        assert r.data["zones"]
        zone_names = [z["zone"] for z in r.data["zones"]]
        assert "N.Y.C." in zone_names
        assert "CAPITL" in zone_names

    def test_pricing_spread_computation(self):
        t, mock = self._setup_pricing()
        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="pricing", date="2026-03-25")
        zones = {z["zone"]: z for z in r.data["zones"]}
        # NYC: DA latest=40.00 (01:00), RT latest=44.50 (00:05)
        # Actually RT latest is 00:05 > 00:00 so RT=44.50
        nyc = zones["N.Y.C."]
        assert nyc["da_lbmp"] == 40.0  # latest hour 01:00
        assert nyc["rt_lbmp"] == 44.5  # latest 5-min 00:05
        assert nyc["spread"] == 4.5  # RT - DA

    def test_pricing_zone_filter(self):
        t, mock = self._setup_pricing()
        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="pricing", date="2026-03-25", zone="CAPITL")
        assert r.success
        assert len(r.data["zones"]) == 1
        assert r.data["zones"][0]["zone"] == "CAPITL"

    def test_pricing_stressed_zone(self):
        """Zone with |spread| > $5 should be flagged."""
        da_rows = [
            {
                "ts": "03/25/2026 00:00",
                "name": "N.Y.C.",
                "ptid": "61761",
                "lbmp": "30.00",
                "loss": "1.0",
                "cong": "0.0",
            },
        ]
        rt_rows = [
            {
                "ts": "03/25/2026 00:00:00",
                "name": "N.Y.C.",
                "ptid": "61761",
                "lbmp": "50.00",
                "loss": "2.0",
                "cong": "5.0",
            },
        ]
        t = _make_tool()
        da_csv = _make_da_lbmp_csv(da_rows)
        rt_csv = _make_rt_lbmp_csv(rt_rows)

        def mock(dataset, date_str, directory=None):
            if "damlbmp" in dataset:
                return t._parse_csv(da_csv)
            return t._parse_csv(rt_csv)

        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="pricing", date="2026-03-25")
        assert r.success
        assert "N.Y.C." in r.data["stressed_zones"]

    def test_pricing_no_data(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", return_value=None):
            r = t.execute(mode="pricing", date="2026-03-25")
        assert not r.success

    def test_pricing_da_only(self):
        """Only DA available, no RT."""
        t = _make_tool()
        da_text = _make_da_lbmp_csv()

        def mock(dataset, date_str, directory=None):
            if "damlbmp" in dataset:
                return t._parse_csv(da_text)
            return None

        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="pricing", date="2026-03-25")
        assert r.success
        for z in r.data["zones"]:
            assert z["rt_lbmp"] is None
            assert z["spread"] is None

    def test_pricing_rt_only(self):
        """Only RT available, no DA."""
        t = _make_tool()
        rt_text = _make_rt_lbmp_csv()

        def mock(dataset, date_str, directory=None):
            if "realtime" in dataset:
                return t._parse_csv(rt_text)
            return None

        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="pricing", date="2026-03-25")
        assert r.success
        for z in r.data["zones"]:
            assert z["da_lbmp"] is None


# ── 8. Forecast Mode ─────────────────────────────────────────


class TestForecastMode:
    """Forecast mode with mocked CSV data."""

    def _setup_forecast(self):
        t = _make_tool()
        fc_text = _make_forecast_csv()
        demand_text = _make_demand_csv()
        fc_rows = t._parse_csv(fc_text)
        demand_rows = t._parse_csv(demand_text)

        def mock(dataset, date_str, directory=None):
            if dataset == "isolf":
                return fc_rows
            if dataset == "pal":
                return demand_rows
            return None

        return t, mock

    def test_forecast_normal(self):
        t, mock = self._setup_forecast()
        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="forecast", date="2026-03-25")
        assert r.success
        assert r.data["zones"]

    def test_forecast_deviation_computation(self):
        t, mock = self._setup_forecast()
        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="forecast", date="2026-03-25")
        # NYC forecast = 4600 at 00:00, actual avg at 00:00 hour = 4500
        # deviation = (4500 - 4600) / 4600 * 100 = -2.17%
        nyc = [z for z in r.data["zones"] if z["zone"] == "N.Y.C."]
        assert len(nyc) == 1
        assert nyc[0]["avg_deviation_pct"] is not None

    def test_forecast_no_data(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", return_value=None):
            r = t.execute(mode="forecast", date="2026-03-25")
        assert not r.success

    def test_forecast_no_actuals(self):
        """Forecast available but no actual data."""
        t = _make_tool()
        fc_text = _make_forecast_csv()
        fc_rows = t._parse_csv(fc_text)

        def mock(dataset, date_str, directory=None):
            if dataset == "isolf":
                return fc_rows
            return None  # no actual data

        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="forecast", date="2026-03-25")
        assert r.success
        # Should show forecast-only or no-actuals message
        assert "no actuals" in r.output.lower() or "forecast" in r.output.lower()

    def test_forecast_zone_filter(self):
        t, mock = self._setup_forecast()
        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="forecast", date="2026-03-25", zone="CAPITL")
        assert r.success
        zone_names = [z["zone"] for z in r.data["zones"]]
        assert "CAPITL" in zone_names
        # Other zones should be excluded
        assert "N.Y.C." not in zone_names

    def test_forecast_persistent_deviation(self):
        """Zone with avg deviation > 3% should be flagged."""
        t = _make_tool()
        # Forecast: 1000, Actual: 1100 → +10% deviation
        fc_csv = '"Time Stamp","Capitl","NYISO"\n"03/25/2026 00:00",1000,1000\n'
        demand_csv = "Time Stamp,Time Zone,Name,PTID,Load\n03/25/2026 00:00:00,EDT,CAPITL,61757,1100.0\n"
        fc_rows = t._parse_csv(fc_csv)
        demand_rows = t._parse_csv(demand_csv)

        def mock(dataset, date_str, directory=None):
            if dataset == "isolf":
                return fc_rows
            if dataset == "pal":
                return demand_rows
            return None

        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="forecast", date="2026-03-25")
        assert r.success
        assert "CAPITL" in r.data.get("persistent_deviation_zones", [])
        assert "OVER-consuming" in r.output


# ── 9. HTTP / Fetch ──────────────────────────────────────────


class TestFetchCsv:
    """HTTP fetch and fallback logic."""

    def test_fetch_daily_csv(self):
        t = _make_tool()
        csv_text = _make_demand_csv()
        mock_resp = _mock_response(csv_text, 200)
        with patch("agent.tools.power_grid.httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = mock_resp
            rows = t._fetch_csv("pal", "2026-03-25")
        assert rows is not None
        assert len(rows) == 6

    def test_fetch_daily_404_falls_back(self):
        """Daily CSV returns 404, should try archive."""
        t = _make_tool()
        csv_text = _make_demand_csv()

        # Build a ZIP containing the target CSV
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("20260325pal.csv", csv_text)
        zip_bytes = zip_buf.getvalue()

        with patch("agent.tools.power_grid.httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value

            def get_side_effect(url, **kwargs):
                if "20260325pal.csv" in url:
                    return _mock_response("", 404)
                else:
                    # archive ZIP URL
                    resp = httpx.Response(200, content=zip_bytes, request=httpx.Request("GET", url))
                    return resp

            instance.get.side_effect = get_side_effect
            rows = t._fetch_csv("pal", "2026-03-25")
        assert rows is not None
        assert len(rows) == 6

    def test_fetch_both_404(self):
        """Both daily CSV and archive return 404."""
        t = _make_tool()
        with patch("agent.tools.power_grid.httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = _mock_response("", 404)
            rows = t._fetch_csv("pal", "2026-03-25")
        assert rows is None

    def test_timeout_error(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", side_effect=httpx.TimeoutException("timeout")):
            r = t.execute(mode="demand", date="2026-03-25")
        assert not r.success
        assert "timed out" in r.output.lower()

    def test_http_error(self):
        t = _make_tool()
        with patch.object(t, "_fetch_csv", side_effect=httpx.HTTPError("500 Server Error")):
            r = t.execute(mode="demand", date="2026-03-25")
        assert not r.success
        assert "HTTP error" in r.output


# ── 10. Archive ZIP Parsing ──────────────────────────────────


class TestArchiveParsing:
    """Monthly archive ZIP handling."""

    def test_archive_extract_correct_day(self):
        t = _make_tool()
        csv_day15 = "Time Stamp,Time Zone,Name,PTID,Load\n03/15/2026 00:00:00,EDT,N.Y.C.,61761,5000.0\n"
        csv_day16 = "Time Stamp,Time Zone,Name,PTID,Load\n03/16/2026 00:00:00,EDT,N.Y.C.,61761,5100.0\n"

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("20260315pal.csv", csv_day15)
            zf.writestr("20260316pal.csv", csv_day16)

        with patch("agent.tools.power_grid.httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            resp = httpx.Response(
                200,
                content=zip_buf.getvalue(),
                request=httpx.Request("GET", "http://test"),
            )
            instance.get.return_value = resp
            rows = t._fetch_from_archive("pal", "2026-03-15")
        assert rows is not None
        assert len(rows) == 1
        assert "5000" in rows[0].get("Load", "")

    def test_archive_bad_zip(self):
        t = _make_tool()
        with patch("agent.tools.power_grid.httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            resp = httpx.Response(
                200,
                content=b"not a zip file",
                request=httpx.Request("GET", "http://test"),
            )
            instance.get.return_value = resp
            rows = t._fetch_from_archive("pal", "2026-03-15")
        assert rows is None

    def test_archive_date_not_in_zip(self):
        t = _make_tool()
        csv_day15 = "Time Stamp,Name,PTID,Load\n03/15/2026 00:00:00,N.Y.C.,61761,5000.0\n"
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("20260315pal.csv", csv_day15)

        with patch("agent.tools.power_grid.httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            resp = httpx.Response(
                200,
                content=zip_buf.getvalue(),
                request=httpx.Request("GET", "http://test"),
            )
            instance.get.return_value = resp
            rows = t._fetch_from_archive("pal", "2026-03-20")
        assert rows is None


# ── 11. Cache Interaction ────────────────────────────────────


class TestCacheInteraction:
    """Cache hit/miss behavior."""

    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = [{"Name": "N.Y.C.", "Load": "4500.0"}]
        t = _make_tool(cache=cache)
        rows = t._fetch_csv("pal", "2026-03-25")
        assert rows is not None
        cache.get.assert_called_once()
        # Should not make HTTP call

    def test_cache_miss_then_put(self):
        cache = MagicMock()
        cache.get.return_value = None
        t = _make_tool(cache=cache)
        csv_text = _make_demand_csv()
        with patch("agent.tools.power_grid.httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = _mock_response(csv_text, 200)
            rows = t._fetch_csv("pal", "2026-03-25")
        cache.put.assert_called_once()
        assert rows is not None

    def test_no_cache(self):
        """Tool works without cache."""
        t = _make_tool(cache=None)
        csv_text = _make_demand_csv()
        with patch("agent.tools.power_grid.httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            instance.get.return_value = _mock_response(csv_text, 200)
            rows = t._fetch_csv("pal", "2026-03-25")
        assert rows is not None

    def test_archive_cache(self):
        """Archive data is cached."""
        cache = MagicMock()
        cache.get.return_value = None
        t = _make_tool(cache=cache)

        csv_text = _make_demand_csv()
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("20260315pal.csv", csv_text)

        with patch("agent.tools.power_grid.httpx.Client") as MockClient:
            instance = MockClient.return_value.__enter__.return_value
            resp = httpx.Response(
                200,
                content=zip_buf.getvalue(),
                request=httpx.Request("GET", "http://test"),
            )
            instance.get.return_value = resp
            t._fetch_from_archive("pal", "2026-03-15")
        # Should cache with nyiso_archive source
        cache.put.assert_called_once()
        call_args = cache.put.call_args
        assert call_args[0][0] == "nyiso_archive"


# ── 12. _safe_float ──────────────────────────────────────────


class TestSafeFloat:
    """Float parsing utility."""

    def test_normal_float(self):
        assert _safe_float("42.5") == 42.5

    def test_integer(self):
        assert _safe_float("100") == 100.0

    def test_comma_separated(self):
        assert _safe_float("1,234.5") == 1234.5

    def test_negative(self):
        assert _safe_float("-3.14") == -3.14

    def test_none(self):
        assert _safe_float(None) is None

    def test_empty(self):
        assert _safe_float("") is None

    def test_dash(self):
        assert _safe_float("-") is None

    def test_na(self):
        assert _safe_float("N/A") is None
        assert _safe_float("n/a") is None

    def test_whitespace(self):
        assert _safe_float("  42.5  ") == 42.5

    def test_non_numeric(self):
        assert _safe_float("abc") is None

    def test_zero(self):
        assert _safe_float("0") == 0.0

    def test_numeric_type(self):
        assert _safe_float(42) == 42.0
        assert _safe_float(42.5) == 42.5


# ── 13. Hour Truncation ─────────────────────────────────────


class TestHourTruncation:
    """Timestamp truncation for forecast matching."""

    def test_with_seconds(self):
        assert PowerGridTool._truncate_to_hour("03/25/2026 04:35:00") == "03/25/2026 04:00"

    def test_without_seconds(self):
        assert PowerGridTool._truncate_to_hour("03/25/2026 04:35") == "03/25/2026 04:00"

    def test_on_hour(self):
        assert PowerGridTool._truncate_to_hour("03/25/2026 04:00:00") == "03/25/2026 04:00"

    def test_invalid_format(self):
        assert PowerGridTool._truncate_to_hour("2026-03-25T04:00:00") is None

    def test_empty(self):
        assert PowerGridTool._truncate_to_hour("") is None

    def test_whitespace(self):
        assert PowerGridTool._truncate_to_hour("  03/25/2026 04:35:00  ") == "03/25/2026 04:00"


# ── 14. Output Formatting ───────────────────────────────────


class TestOutputFormatting:
    """Output text formatting checks."""

    def test_demand_output_has_header(self):
        t = _make_tool()
        csv_text = _make_demand_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="demand", date="2026-03-25")
        assert "NYISO Demand" in r.output
        assert "2026-03-25" in r.output

    def test_fuel_mix_output_has_percentages(self):
        t = _make_tool()
        csv_text = _make_fuel_mix_csv()
        with patch.object(t, "_fetch_csv", return_value=t._parse_csv(csv_text)):
            r = t.execute(mode="fuel_mix", date="2026-03-25")
        assert "%" in r.output
        assert "MW" in r.output

    def test_pricing_output_has_table(self):
        t, mock = TestPricingMode()._setup_pricing()
        with patch.object(t, "_fetch_csv", side_effect=mock):
            r = t.execute(mode="pricing", date="2026-03-25")
        assert "DA $/MWh" in r.output
        assert "RT $/MWh" in r.output
        assert "Spread" in r.output


# ── 15. NYISO Zones Constant ─────────────────────────────────


class TestNyisoZones:
    """Zone list integrity."""

    def test_zone_count(self):
        assert len(NYISO_ZONES) == 11

    def test_zones_are_uppercase(self):
        for z in NYISO_ZONES:
            assert z == z.upper() or "." in z  # N.Y.C. has dots

    def test_known_zones_present(self):
        assert "N.Y.C." in NYISO_ZONES
        assert "CAPITL" in NYISO_ZONES
        assert "LONGIL" in NYISO_ZONES
        assert "WEST" in NYISO_ZONES


# ── 16. Bandit Arm Integration ───────────────────────────────


class TestBanditIntegration:
    """Verify energy_demand arm is correctly configured."""

    def test_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = [a.name for a in DEFAULT_ARMS]
        assert "energy_demand" in names

    def test_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = [a for a in DEFAULT_ARMS if a.name == "energy_demand"][0]
        assert "power_grid" in arm.tools

    def test_arm_has_examples(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = [a for a in DEFAULT_ARMS if a.name == "energy_demand"][0]
        assert len(arm.examples) >= 2


# ── 17. Live Network Tests ──────────────────────────────────


@(
    pytest.mark.skipif(
        not _can_reach_nyiso(),  # noqa: F821  # defined below; if False short-circuits call
        reason="NYISO MIS not reachable",
    )
    if False
    else lambda f: f
)  # Always attempt, skip decorator handled below
class TestLiveNetwork:
    """Live tests against NYISO MIS. Auto-skip if network unavailable."""

    @pytest.fixture(autouse=True)
    def check_network(self):
        try:
            r = httpx.get("http://mis.nyiso.com/public/csv/pal/", timeout=5)
        except Exception:
            pytest.skip("NYISO not reachable")

    def test_live_demand(self):
        t = _make_tool()
        r = t.execute(mode="demand")
        assert r.success
        assert r.data["zones"]
        assert r.data["total_peak_mw"] > 0

    def test_live_fuel_mix(self):
        t = _make_tool()
        r = t.execute(mode="fuel_mix")
        assert r.success
        assert r.data["fuels"]
        assert r.data["total_mw"] > 0

    def test_live_pricing(self):
        t = _make_tool()
        # Use yesterday — today's data might not have full pricing yet
        from datetime import date, timedelta

        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        r = t.execute(mode="pricing", date=yesterday)
        assert r.success
        assert r.data["zones"]

    def test_live_forecast(self):
        t = _make_tool()
        r = t.execute(mode="forecast")
        assert r.success


def _can_reach_nyiso() -> bool:
    try:
        r = httpx.get("http://mis.nyiso.com/public/csv/pal/", timeout=5)
        return r.status_code < 500
    except Exception:
        return False
