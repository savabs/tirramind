"""Edge-case tests for agent.tools.central_bank_balance (7b-Z).

Tests cover:
- All 4 modes (balance_sheets, liquidity_index, policy_divergence, rate_monitor)
- Invalid inputs (mode, period, banks)
- Missing FRED API key
- HTTP errors (4xx, 5xx, timeouts)
- Empty / partial data responses
- FX conversion edge cases (missing rate, zero rate, inversion logic)
- Discontinued FRED series (graceful fallback)
- ECB SDMX parsing (valid + malformed)
- Rate change detection (no changes, recent change, multi-change)
- Growth rate computation (short series, identical values, missing dates)
- Cache integration
- Registration (cli.py + bandit.py)
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

UTC = UTC
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.central_bank_balance import (
    CB_REGISTRY,
    VALID_MODES,
    CentralBankBalanceTool,
    _find_value_n_days_back,
    _fmt_pct,
    _parse_ecb_observations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(api_key: str = "test-key", cache: MagicMock | None = None) -> CentralBankBalanceTool:
    return CentralBankBalanceTool(fred_api_key=api_key, cache=cache)


def _fred_response(series_id: str, observations: list[dict]) -> dict:
    """Build a FRED-like JSON response."""
    return {"observations": observations}


def _fred_obs(date: str, value: str) -> dict:
    return {"date": date, "value": value}


def _make_series(start_date: str, n: int, start_val: float, step: float = 0.0) -> list[dict]:
    """Generate a simple time series of {date, value} dicts."""
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    obs = []
    for i in range(n):
        d = dt + timedelta(days=i * 7)  # weekly
        obs.append({"date": d.strftime("%Y-%m-%d"), "value": str(start_val + i * step)})
    return obs


def _ecb_json_response(values: list[tuple[str, float]]) -> dict:
    """Build ECB SDMX-like JSON response."""
    time_values = [{"id": date, "name": date} for date, _ in values]
    observations = {str(i): [val, 0, 0, None, None] for i, (_, val) in enumerate(values)}
    return {
        "header": {
            "id": "test",
            "test": False,
            "prepared": "2026-01-01T00:00:00Z",
            "sender": {"id": "ECB"},
        },
        "dataSets": [
            {
                "action": "Replace",
                "series": {
                    "0:0:0:0:0:0": {
                        "attributes": [],
                        "observations": observations,
                    }
                },
            }
        ],
        "structure": {
            "dimensions": {
                "observation": [
                    {
                        "id": "TIME_PERIOD",
                        "values": time_values,
                    }
                ]
            }
        },
    }


# ---------------------------------------------------------------------------
# I. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation(unittest.TestCase):
    """Test boundary conditions on inputs."""

    def test_invalid_mode(self):
        tool = _make_tool()
        r = tool.execute(mode="bogus")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        tool = _make_tool()
        r = tool.execute(mode="")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_missing_api_key(self):
        tool = _make_tool(api_key="")
        r = tool.execute(mode="balance_sheets")
        assert not r.success
        assert "FRED API key" in r.output

    def test_invalid_banks_filter(self):
        tool = _make_tool()
        r = tool.execute(mode="balance_sheets", banks="xyz,abc")
        assert not r.success
        assert "No valid banks" in r.output

    def test_valid_banks_filter_mixed(self):
        """Should keep valid banks and ignore invalid ones."""
        tool = _make_tool()
        with patch.object(
            tool,
            "_mode_balance_sheets",
            return_value=MagicMock(success=True, output="ok"),
        ) as m:
            tool.execute(mode="balance_sheets", banks="fed,bogus,ecb")
            m.assert_called_once()
            _, bank_list = m.call_args[0]
            assert bank_list == ["fed", "ecb"]

    def test_unknown_period_defaults_to_1y(self):
        tool = _make_tool()
        with patch.object(
            tool,
            "_mode_balance_sheets",
            return_value=MagicMock(success=True, output="ok"),
        ) as m:
            tool.execute(mode="balance_sheets", period="99y")
            period_arg = m.call_args[0][0]
            assert period_arg == "1y"

    def test_valid_periods(self):
        tool = _make_tool()
        for p in ("1m", "3m", "6m", "1y", "2y", "5y"):
            with patch.object(
                tool,
                "_mode_balance_sheets",
                return_value=MagicMock(success=True, output="ok"),
            ) as m:
                tool.execute(mode="balance_sheets", period=p)
                assert m.call_args[0][0] == p

    def test_all_modes_dispatch(self):
        """Ensure every mode dispatches to its handler."""
        tool = _make_tool()
        for mode in VALID_MODES:
            method = f"_mode_{mode}"
            with patch.object(tool, method, return_value=MagicMock(success=True, output="ok")) as m:
                tool.execute(mode=mode)
                assert m.called, f"{method} not called for mode={mode}"


# ---------------------------------------------------------------------------
# II. FRED fetch + caching
# ---------------------------------------------------------------------------


class TestFredFetch(unittest.TestCase):
    """Test FRED observation fetching."""

    def test_fred_success(self):
        tool = _make_tool()
        obs = [_fred_obs("2026-01-01", "100"), _fred_obs("2026-01-08", "101")]
        resp_mock = MagicMock()
        resp_mock.json.return_value = _fred_response("WALCL", obs)
        resp_mock.raise_for_status = MagicMock()

        with patch("agent.tools.central_bank_balance.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = resp_mock

            result = tool._fetch_fred_observations("WALCL", "2026-01-01", "2026-01-31", "bs_fed", 3600)
            assert len(result) == 2
            assert result[0]["date"] == "2026-01-01"
            assert result[0]["value"] == "100"

    def test_fred_filters_missing_values(self):
        tool = _make_tool()
        obs = [
            _fred_obs("2026-01-01", "100"),
            _fred_obs("2026-01-08", "."),
            _fred_obs("2026-01-15", ""),
            _fred_obs("2026-01-22", "102"),
        ]
        resp_mock = MagicMock()
        resp_mock.json.return_value = _fred_response("WALCL", obs)
        resp_mock.raise_for_status = MagicMock()

        with patch("agent.tools.central_bank_balance.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = resp_mock

            result = tool._fetch_fred_observations("WALCL", "2026-01-01", "2026-01-31", "test", 3600)
            assert len(result) == 2  # Only non-missing

    def test_fred_http_error_returns_empty(self):
        tool = _make_tool()
        with patch("agent.tools.central_bank_balance.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock(status_code=500)
            )
            result = tool._fetch_fred_observations("WALCL", "2026-01-01", "2026-01-31", "test", 3600)
            assert result == []

    def test_fred_timeout_returns_empty(self):
        tool = _make_tool()
        with patch("agent.tools.central_bank_balance.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.ConnectTimeout("timeout")
            result = tool._fetch_fred_observations("WALCL", "2026-01-01", "2026-01-31", "test", 3600)
            assert result == []

    def test_fred_cache_hit(self):
        cache = MagicMock()
        cached_data = [{"date": "2026-01-01", "value": "100"}]
        cache.get.return_value = cached_data

        tool = _make_tool(cache=cache)
        result = tool._fetch_fred_observations("WALCL", "2026-01-01", "2026-01-31", "test", 3600)
        assert result == cached_data
        cache.get.assert_called_once()

    def test_fred_cache_miss_stores(self):
        cache = MagicMock()
        cache.get.return_value = None

        tool = _make_tool(cache=cache)
        obs = [_fred_obs("2026-01-01", "100")]
        resp_mock = MagicMock()
        resp_mock.json.return_value = _fred_response("WALCL", obs)
        resp_mock.raise_for_status = MagicMock()

        with patch("agent.tools.central_bank_balance.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = resp_mock

            result = tool._fetch_fred_observations("WALCL", "2026-01-01", "2026-01-31", "test", 3600)
            assert len(result) == 1
            cache.put.assert_called_once()


# ---------------------------------------------------------------------------
# III. FX conversion
# ---------------------------------------------------------------------------


class TestFxConversion(unittest.TestCase):
    """Test USD normalization logic."""

    def test_usd_no_conversion(self):
        tool = _make_tool()
        result = tool._to_usd(1e12, "fed", {})
        assert result == 1e12

    def test_eur_usd_per_foreign(self):
        """DEXUSEU gives USD per EUR → multiply."""
        tool = _make_tool()
        fx = {"DEXUSEU": 1.10}
        result = tool._to_usd(6e12, "ecb", fx)
        assert result == pytest.approx(6.6e12)

    def test_jpy_foreign_per_usd(self):
        """DEXJPUS gives JPY per USD → divide."""
        tool = _make_tool()
        fx = {"DEXJPUS": 150.0}
        result = tool._to_usd(750e12, "boj", fx)
        assert result == pytest.approx(5e12)

    def test_missing_fx_returns_none(self):
        tool = _make_tool()
        result = tool._to_usd(1e12, "ecb", {})
        assert result is None

    def test_zero_fx_rate_returns_none(self):
        tool = _make_tool()
        fx = {"DEXUSEU": 0.0}
        result = tool._to_usd(1e12, "ecb", fx)
        assert result is None

    def test_gbp_usd_per_foreign(self):
        """DEXUSUK gives USD per GBP → multiply."""
        tool = _make_tool()
        fx = {"DEXUSUK": 1.27}
        # BOE is skip_bs but conversion logic still works
        result = tool._to_usd(1e12, "boe", fx)
        assert result == pytest.approx(1.27e12)

    def test_chf_foreign_per_usd(self):
        """DEXSZUS gives CHF per USD → divide."""
        tool = _make_tool()
        fx = {"DEXSZUS": 0.88}
        result = tool._to_usd(880e9, "snb", fx)
        assert result == pytest.approx(1e12)

    def test_aud_usd_per_foreign(self):
        """DEXUSAL gives USD per AUD → multiply."""
        tool = _make_tool()
        fx = {"DEXUSAL": 0.65}
        result = tool._to_usd(1e12, "rba", fx)
        assert result == pytest.approx(0.65e12)


# ---------------------------------------------------------------------------
# IV. ECB SDMX parsing
# ---------------------------------------------------------------------------


class TestEcbParsing(unittest.TestCase):
    """Test ECB SDW JSON response parsing."""

    def test_valid_response(self):
        data = _ecb_json_response(
            [
                ("2026-01-07", 6200000),
                ("2026-01-14", 6180000),
                ("2026-01-21", 6150000),
            ]
        )
        result = _parse_ecb_observations(data)
        assert len(result) == 3
        assert result[0]["date"] == "2026-01-07"
        assert result[0]["value"] == 6200000

    def test_empty_datasets(self):
        data = {"dataSets": [], "structure": {"dimensions": {"observation": []}}}
        assert _parse_ecb_observations(data) == []

    def test_missing_structure(self):
        data = {"dataSets": [{"series": {}}]}
        assert _parse_ecb_observations(data) == []

    def test_none_observation_value(self):
        data = _ecb_json_response([("2026-01-07", 100)])
        # Make one obs have None value
        data["dataSets"][0]["series"]["0:0:0:0:0:0"]["observations"]["1"] = [None, 0]
        # Add extra time value
        data["structure"]["dimensions"]["observation"][0]["values"].append({"id": "2026-01-14", "name": "2026-01-14"})
        result = _parse_ecb_observations(data)
        assert len(result) == 1  # Only the non-None

    def test_malformed_json(self):
        result = _parse_ecb_observations({"garbage": True})
        assert result == []

    def test_empty_series(self):
        data = {
            "dataSets": [{"series": {}}],
            "structure": {"dimensions": {"observation": [{"values": []}]}},
        }
        assert _parse_ecb_observations(data) == []


# ---------------------------------------------------------------------------
# V. Rate change detection
# ---------------------------------------------------------------------------


class TestRateChangeDetection(unittest.TestCase):
    def test_no_changes(self):
        tool = _make_tool()
        series = [
            {"date": "2026-01-01", "value": "5.33"},
            {"date": "2026-01-02", "value": "5.33"},
        ]
        result = tool._detect_rate_change(series)
        assert result == {}

    def test_single_hike(self):
        tool = _make_tool()
        series = [
            {"date": "2026-01-01", "value": "5.25"},
            {"date": "2026-01-15", "value": "5.25"},
            {"date": "2026-01-29", "value": "5.50"},
        ]
        result = tool._detect_rate_change(series)
        assert result["direction"] == "hike"
        assert result["bps"] == pytest.approx(25.0, abs=0.5)
        assert result["date"] == "2026-01-29"

    def test_single_cut(self):
        tool = _make_tool()
        series = [
            {"date": "2026-01-01", "value": "5.50"},
            {"date": "2026-02-01", "value": "5.25"},
        ]
        result = tool._detect_rate_change(series)
        assert result["direction"] == "cut"
        assert result["bps"] == pytest.approx(-25.0, abs=0.5)

    def test_multiple_changes_finds_most_recent(self):
        tool = _make_tool()
        series = [
            {"date": "2025-01-01", "value": "4.50"},
            {"date": "2025-06-01", "value": "4.75"},
            {"date": "2025-12-01", "value": "5.00"},
            {"date": "2026-01-01", "value": "5.00"},
        ]
        result = tool._detect_rate_change(series)
        assert result["date"] == "2025-12-01"
        assert result["direction"] == "hike"

    def test_empty_series(self):
        tool = _make_tool()
        assert tool._detect_rate_change([]) == {}

    def test_single_observation(self):
        tool = _make_tool()
        assert tool._detect_rate_change([{"date": "2026-01-01", "value": "5.0"}]) == {}

    def test_tiny_change_below_threshold(self):
        """Changes < 0.001 should be treated as no change."""
        tool = _make_tool()
        series = [
            {"date": "2026-01-01", "value": "5.33000"},
            {"date": "2026-01-02", "value": "5.33001"},  # < 0.001 diff
        ]
        result = tool._detect_rate_change(series)
        assert result == {}


# ---------------------------------------------------------------------------
# VI. Growth rate helpers
# ---------------------------------------------------------------------------


class TestGrowthHelpers(unittest.TestCase):
    def test_compute_changes_basic(self):
        now = datetime.now(UTC)
        series = []
        for days_ago in [400, 60, 14, 7, 0]:
            d = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            series.append({"date": d, "value": str(100 + (400 - days_ago))})

        result = CentralBankBalanceTool._compute_changes(series, 1.0)
        assert "wow" in result
        assert "mom" in result
        assert "yoy" in result

    def test_compute_changes_empty(self):
        assert CentralBankBalanceTool._compute_changes([], 1.0) == {}

    def test_compute_growth_rate_expanding(self):
        now = datetime.now(UTC)
        series = [
            {"date": (now - timedelta(days=100)).strftime("%Y-%m-%d"), "value": "100"},
            {"date": now.strftime("%Y-%m-%d"), "value": "110"},
        ]
        result = CentralBankBalanceTool._compute_growth_rate(series, 90, 1.0)
        assert result is not None
        assert result > 0  # Expanding

    def test_compute_growth_rate_contracting(self):
        now = datetime.now(UTC)
        series = [
            {"date": (now - timedelta(days=100)).strftime("%Y-%m-%d"), "value": "110"},
            {"date": now.strftime("%Y-%m-%d"), "value": "100"},
        ]
        result = CentralBankBalanceTool._compute_growth_rate(series, 90, 1.0)
        assert result is not None
        assert result < 0  # Contracting

    def test_compute_growth_rate_empty(self):
        assert CentralBankBalanceTool._compute_growth_rate([], 90, 1.0) is None

    def test_compute_growth_rate_single_obs(self):
        series = [{"date": "2026-01-01", "value": "100"}]
        assert CentralBankBalanceTool._compute_growth_rate(series, 90, 1.0) is None

    def test_find_value_n_days_back_exact(self):
        series = _make_series("2025-01-01", 52, 100.0, 1.0)  # ~1 year of weekly data
        val = _find_value_n_days_back(series, 7, 1.0)
        assert val is not None

    def test_find_value_n_days_back_too_far(self):
        """If target is outside tolerance, return None."""
        series = [
            {"date": "2025-01-01", "value": "100"},
            {"date": "2026-03-01", "value": "110"},
        ]
        # Looking 365 days back from March 1 2026 → target = March 1 2025
        # Jan 1 2025 is ~59 days before target. tolerance = max(365*0.2, 30) = 73
        # 59 < 73 so it would match. Use a very short series that can't match.
        short_series = [
            {"date": "2026-02-28", "value": "100"},
            {"date": "2026-03-01", "value": "110"},
        ]
        # Looking 365 days back — only data is 1 day ago, tolerance = max(73, 30) = 73
        # 364 days off → > 73 → None
        val = _find_value_n_days_back(short_series, 365, 1.0)
        assert val is None

    def test_find_value_invalid_date(self):
        series = [
            {"date": "not-a-date", "value": "100"},
            {"date": "2026-01-01", "value": "110"},
        ]
        # With only 2 obs, one invalid date, the function still returns
        # the valid obs if it's within tolerance. The last valid date is
        # Jan 1. Looking 7 days back targets Dec 25. Jan 1 is 7 days off
        # from target, within tolerance. But the last obs date is Jan 1
        # and the function uses that as the reference. So target = Dec 25.
        # Only the invalid-date obs is skipped. The remaining obs IS the
        # last one, which is 7 days from target → matches.
        val = _find_value_n_days_back(series, 7, 1.0)
        # Only one valid date can't look back from itself
        assert val == 110.0  # The valid obs is closest


# ---------------------------------------------------------------------------
# VII. Formatting helpers
# ---------------------------------------------------------------------------


class TestFormatHelpers(unittest.TestCase):
    def test_fmt_pct_positive(self):
        assert _fmt_pct(5.25) == "+5.25%"

    def test_fmt_pct_negative(self):
        assert _fmt_pct(-3.10) == "-3.10%"

    def test_fmt_pct_zero(self):
        assert _fmt_pct(0.0) == "0.00%"

    def test_fmt_pct_none(self):
        assert _fmt_pct(None) == "N/A"


# ---------------------------------------------------------------------------
# VIII. Mode: balance_sheets
# ---------------------------------------------------------------------------


class TestModeBalanceSheets(unittest.TestCase):
    def _mock_fred_fetch(self, tool, responses: dict[str, list[dict]]):
        """Patch _fetch_fred_observations to return canned data per series."""
        original = tool._fetch_fred_observations

        def fake_fetch(series_id, start, end, cache_key, ttl):
            return responses.get(series_id, [])

        return patch.object(tool, "_fetch_fred_observations", side_effect=fake_fetch)

    def test_basic_snapshot(self):
        tool = _make_tool()
        obs_walcl = [{"date": "2026-01-01", "value": "7000000"}]  # 7T USD (millions)
        obs_ecb = [{"date": "2026-01-01", "value": "6200000"}]  # 6.2T EUR (millions)
        obs_boj = [{"date": "2026-01-01", "value": "7500000"}]  # 750T JPY (100M yen)

        fx_rates = {"DEXUSEU": 1.10, "DEXJPUS": 150.0}

        with (
            self._mock_fred_fetch(
                tool,
                {
                    "WALCL": obs_walcl,
                    "ECBASSETSW": obs_ecb,
                    "JPNASSETS": obs_boj,
                    "SNBASSETM": [],
                    "BCBASSETM": [],
                    "RBASSETSM": [],
                },
            ),
            patch.object(tool, "_fetch_fx_rates", return_value=fx_rates),
        ):
            r = tool.execute(mode="balance_sheets", banks="fed,ecb,boj")

        assert r.success
        assert "Federal Reserve" in r.output
        assert "European Central Bank" in r.output
        assert r.data["banks"]
        assert len(r.data["banks"]) == 3

    def test_partial_data(self):
        """Some CBs return no data — should still succeed with partial results."""
        tool = _make_tool()
        with (
            self._mock_fred_fetch(tool, {"WALCL": [{"date": "2026-01-01", "value": "7000000"}]}),
            patch.object(tool, "_fetch_fx_rates", return_value={}),
        ):
            r = tool.execute(mode="balance_sheets", banks="fed,ecb,boj")
        assert r.success
        assert len(r.data["banks"]) == 1  # Only Fed succeeded
        assert len(r.data["errors"]) >= 1

    def test_all_data_missing(self):
        """No CB returns data → failure."""
        tool = _make_tool()
        with self._mock_fred_fetch(tool, {}), patch.object(tool, "_fetch_fx_rates", return_value={}):
            r = tool.execute(mode="balance_sheets", banks="fed,ecb")
        assert not r.success

    def test_boe_skipped(self):
        """BOE has _skip_bs flag → should be skipped with a note."""
        tool = _make_tool()
        with (
            self._mock_fred_fetch(tool, {"WALCL": [{"date": "2026-01-01", "value": "7000000"}]}),
            patch.object(tool, "_fetch_fx_rates", return_value={}),
        ):
            r = tool.execute(mode="balance_sheets", banks="fed,boe")
        assert r.success
        assert any("BOE" in e or "Bank of England" in e for e in r.data["errors"])

    def test_fx_conversion_failure(self):
        """ECB data present but no FX rate → USD value is None."""
        tool = _make_tool()
        obs_ecb = [{"date": "2026-01-01", "value": "6200000"}]
        with self._mock_fred_fetch(tool, {"ECBASSETSW": obs_ecb}):
            with patch.object(tool, "_fetch_fx_rates", return_value={}):
                r = tool.execute(mode="balance_sheets", banks="ecb")
        assert r.success
        assert r.data["banks"][0]["usd_trillions"] is None


# ---------------------------------------------------------------------------
# IX. Mode: liquidity_index
# ---------------------------------------------------------------------------


class TestModeLiquidityIndex(unittest.TestCase):
    def _mock_fred_fetch(self, tool, responses: dict[str, list[dict]]):
        original = tool._fetch_fred_observations

        def fake_fetch(series_id, start, end, cache_key, ttl):
            return responses.get(series_id, [])

        return patch.object(tool, "_fetch_fred_observations", side_effect=fake_fetch)

    def test_basic_liquidity_index(self):
        tool = _make_tool()
        responses = {
            "WALCL": [{"date": "2026-01-01", "value": "7000000"}],  # 7T
            "ECBASSETSW": [{"date": "2026-01-01", "value": "6200000"}],  # 6.2T EUR
            "JPNASSETS": [{"date": "2026-01-01", "value": "7500000"}],  # 750T JPY
            "RRPONTSYD": [{"date": "2026-01-01", "value": "500"}],  # 500B = 0.5T
            "WDTGAL": [{"date": "2026-01-01", "value": "750000"}],  # 750B USD = 0.75T
        }
        fx_rates = {"DEXUSEU": 1.10, "DEXJPUS": 150.0}

        with self._mock_fred_fetch(tool, responses):
            with patch.object(tool, "_fetch_fx_rates", return_value=fx_rates):
                r = tool.execute(mode="liquidity_index", banks="fed,ecb,boj")

        assert r.success
        assert "Net Liquidity" in r.output
        # Gross = 7T + 6.2T*1.1 + 750T/150 = 7 + 6.82 + 5 = 18.82T
        # Net = 18.82 - 0.5 - 0.75 = 17.57T
        assert r.data["gross_usd"] > 0
        assert r.data["net_usd"] < r.data["gross_usd"]
        assert r.data["rrp_usd"] == pytest.approx(500e9)  # 500B
        assert r.data["tga_usd"] == pytest.approx(750e9)  # 750B

    def test_no_drain_data(self):
        """RRP and TGA missing → drains = 0."""
        tool = _make_tool()
        responses = {"WALCL": [{"date": "2026-01-01", "value": "7000000"}]}

        with self._mock_fred_fetch(tool, responses), patch.object(tool, "_fetch_fx_rates", return_value={}):
            r = tool.execute(mode="liquidity_index", banks="fed")

        assert r.success
        assert r.data["rrp_usd"] == 0
        assert r.data["tga_usd"] == 0
        assert r.data["net_usd"] == r.data["gross_usd"]

    def test_all_data_missing(self):
        tool = _make_tool()
        with self._mock_fred_fetch(tool, {}), patch.object(tool, "_fetch_fx_rates", return_value={}):
            r = tool.execute(mode="liquidity_index", banks="fed")
        assert not r.success


# ---------------------------------------------------------------------------
# X. Mode: policy_divergence
# ---------------------------------------------------------------------------


class TestModePolicyDivergence(unittest.TestCase):
    def _mock_fred_fetch(self, tool, responses: dict[str, list[dict]]):
        def fake_fetch(series_id, start, end, cache_key, ttl):
            return responses.get(series_id, [])

        return patch.object(tool, "_fetch_fred_observations", side_effect=fake_fetch)

    def _expanding_series(self) -> list[dict]:
        """Series that grows ~10% over 12 months."""
        now = datetime.now(UTC)
        return [
            {"date": (now - timedelta(days=365)).strftime("%Y-%m-%d"), "value": "1000"},
            {"date": (now - timedelta(days=180)).strftime("%Y-%m-%d"), "value": "1050"},
            {"date": (now - timedelta(days=90)).strftime("%Y-%m-%d"), "value": "1080"},
            {"date": now.strftime("%Y-%m-%d"), "value": "1100"},
        ]

    def _contracting_series(self) -> list[dict]:
        """Series that shrinks ~10% over 12 months."""
        now = datetime.now(UTC)
        return [
            {"date": (now - timedelta(days=365)).strftime("%Y-%m-%d"), "value": "1100"},
            {"date": (now - timedelta(days=180)).strftime("%Y-%m-%d"), "value": "1050"},
            {"date": (now - timedelta(days=90)).strftime("%Y-%m-%d"), "value": "1020"},
            {"date": now.strftime("%Y-%m-%d"), "value": "1000"},
        ]

    def test_divergence_detected(self):
        tool = _make_tool()
        responses = {
            "WALCL": self._expanding_series(),
            "ECBASSETSW": self._contracting_series(),
            "DFF": [{"date": "2026-01-01", "value": "5.33"}],
        }
        with (
            self._mock_fred_fetch(tool, responses),
            patch.object(tool, "_fetch_ecb_rate", return_value={"current_rate": 3.75}),
        ):
            r = tool.execute(mode="policy_divergence", banks="fed,ecb")

        assert r.success
        assert len(r.data["divergences"]) > 0
        assert "EXPANDING" in r.data["divergences"][0]
        assert "CONTRACTING" in r.data["divergences"][0]

    def test_synchronized_expanding(self):
        tool = _make_tool()
        exp = self._expanding_series()
        responses = {"WALCL": exp, "ECBASSETSW": exp, "DFF": []}
        with self._mock_fred_fetch(tool, responses), patch.object(tool, "_fetch_ecb_rate", return_value=None):
            r = tool.execute(mode="policy_divergence", banks="fed,ecb")

        assert r.success
        assert "SYNCHRONIZED" in r.data["synchronized"]

    def test_rate_differentials(self):
        tool = _make_tool()
        exp = self._expanding_series()
        responses = {
            "WALCL": exp,
            "ECBASSETSW": exp,
            "DFF": [{"date": "2026-01-01", "value": "5.33"}],
        }
        with (
            self._mock_fred_fetch(tool, responses),
            patch.object(tool, "_fetch_ecb_rate", return_value={"current_rate": 3.75}),
        ):
            r = tool.execute(mode="policy_divergence", banks="fed,ecb")

        assert r.success
        assert "ecb" in r.data["rates"]
        assert "fed" in r.data["rates"]
        assert "Rate Differentials" in r.output

    def test_insufficient_data(self):
        tool = _make_tool()
        responses = {"WALCL": [{"date": "2026-01-01", "value": "100"}]}  # Only 1 obs
        with self._mock_fred_fetch(tool, responses):
            r = tool.execute(mode="policy_divergence", banks="fed")
        # Should still succeed even with errors
        assert len(r.data["errors"]) > 0


# ---------------------------------------------------------------------------
# XI. Mode: rate_monitor
# ---------------------------------------------------------------------------


class TestModeRateMonitor(unittest.TestCase):
    def _mock_fred_fetch(self, tool, responses: dict[str, list[dict]]):
        def fake_fetch(series_id, start, end, cache_key, ttl):
            return responses.get(series_id, [])

        return patch.object(tool, "_fetch_fred_observations", side_effect=fake_fetch)

    def test_basic_rate_monitor(self):
        tool = _make_tool()
        now = datetime.now(UTC)
        responses = {
            "DFF": [
                {
                    "date": (now - timedelta(days=60)).strftime("%Y-%m-%d"),
                    "value": "5.25",
                },
                {
                    "date": (now - timedelta(days=30)).strftime("%Y-%m-%d"),
                    "value": "5.25",
                },
                {
                    "date": (now - timedelta(days=10)).strftime("%Y-%m-%d"),
                    "value": "5.50",
                },
                {"date": now.strftime("%Y-%m-%d"), "value": "5.50"},
            ]
        }
        with (
            self._mock_fred_fetch(tool, responses),
            patch.object(
                tool,
                "_fetch_ecb_rate",
                return_value={
                    "current_rate": 3.75,
                    "rate_date": "2026-01-01",
                    "last_change_date": "2025-09-01",
                    "last_change_direction": "cut",
                    "last_change_bps": -25,
                    "days_since_change": 180,
                },
            ),
        ):
            r = tool.execute(mode="rate_monitor", banks="fed,ecb")

        assert r.success
        assert len(r.data["rates"]) == 2
        # Fed should have recent change flag
        fed_rate = next(x for x in r.data["rates"] if x["code"] == "fed")
        assert fed_rate["current_rate"] == 5.50
        assert fed_rate["last_change_direction"] == "hike"

    def test_recent_change_flag(self):
        tool = _make_tool()
        now = datetime.now(UTC)
        responses = {
            "DFF": [
                {
                    "date": (now - timedelta(days=5)).strftime("%Y-%m-%d"),
                    "value": "5.25",
                },
                {"date": now.strftime("%Y-%m-%d"), "value": "5.50"},
            ]
        }
        with self._mock_fred_fetch(tool, responses):
            r = tool.execute(mode="rate_monitor", banks="fed")

        assert r.success
        assert "[RECENT CHANGE]" in r.output

    def test_no_rate_series(self):
        """CBs without rate_series should be skipped (not error)."""
        tool = _make_tool()
        with self._mock_fred_fetch(tool, {}):
            r = tool.execute(mode="rate_monitor", banks="boj")
        # BOJ has no rate_series → no results, not a failure
        assert len(r.data["rates"]) == 0

    def test_ecb_rate_fetch_failure(self):
        tool = _make_tool()
        with self._mock_fred_fetch(tool, {}), patch.object(tool, "_fetch_ecb_rate", return_value=None):
            r = tool.execute(mode="rate_monitor", banks="ecb")
        assert len(r.data["errors"]) > 0


# ---------------------------------------------------------------------------
# XII. ECB rate fetching
# ---------------------------------------------------------------------------


class TestEcbRateFetch(unittest.TestCase):
    def test_ecb_rate_success(self):
        tool = _make_tool()
        ecb_data = _ecb_json_response(
            [
                ("2025-06-01", 3.50),
                ("2025-09-01", 3.25),
                ("2026-01-01", 3.25),
            ]
        )
        resp_mock = MagicMock()
        resp_mock.json.return_value = ecb_data
        resp_mock.raise_for_status = MagicMock()

        with patch("agent.tools.central_bank_balance.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = resp_mock

            result = tool._fetch_ecb_rate("2025-01-01", "2026-03-01")

        assert result is not None
        assert result["current_rate"] == 3.25
        assert result["last_change_direction"] == "cut"

    def test_ecb_rate_http_error(self):
        tool = _make_tool()
        with patch("agent.tools.central_bank_balance.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock(status_code=500)
            )
            result = tool._fetch_ecb_rate("2025-01-01", "2026-03-01")
        assert result is None

    def test_ecb_rate_cache_hit(self):
        cache = MagicMock()
        cached = {"current_rate": 3.75, "rate_date": "2026-01-01"}
        cache.get.return_value = cached
        tool = _make_tool(cache=cache)
        result = tool._fetch_ecb_rate("2025-01-01", "2026-03-01")
        assert result == cached


# ---------------------------------------------------------------------------
# XIII. CB registry integrity
# ---------------------------------------------------------------------------


class TestCBRegistry(unittest.TestCase):
    def test_all_cbs_have_required_fields(self):
        required = {
            "name",
            "bs_series",
            "currency",
            "fx_series",
            "rate_series",
            "unit_scale",
            "frequency",
        }
        for code, cb in CB_REGISTRY.items():
            for field in required:
                assert field in cb, f"CB '{code}' missing field '{field}'"

    def test_unit_scale_is_numeric(self):
        for code, cb in CB_REGISTRY.items():
            val = float(cb["unit_scale"])
            assert val > 0, f"CB '{code}' has invalid unit_scale"

    def test_frequency_is_valid(self):
        valid_freqs = {"weekly", "monthly", "daily"}
        for code, cb in CB_REGISTRY.items():
            assert cb["frequency"] in valid_freqs, f"CB '{code}' has invalid frequency '{cb['frequency']}'"

    def test_core_cbs_present(self):
        for code in ("fed", "ecb", "boj"):
            assert code in CB_REGISTRY

    def test_usd_does_not_need_fx(self):
        assert CB_REGISTRY["fed"]["fx_series"] == ""
        assert CB_REGISTRY["fed"]["currency"] == "USD"


# ---------------------------------------------------------------------------
# XIV. Registration tests
# ---------------------------------------------------------------------------


class TestRegistration(unittest.TestCase):
    def test_tool_in_registry(self):
        try:
            from agent.cli import build_tool_registry
            from agent.config.settings import AgentConfig
        except (ImportError, ModuleNotFoundError):
            pytest.skip("optional deps")

        config = AgentConfig()
        registry = build_tool_registry(config)
        names = registry.list_names()
        assert "central_bank_balance" in names
        assert len(names) == 61, f"Expected 61 tools, got {len(names)}: {sorted(names)}"

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "global_liquidity" in arm_names
        assert len(DEFAULT_ARMS) == 48, f"Expected 48 arms, got {len(DEFAULT_ARMS)}"

    def test_bandit_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "global_liquidity")
        assert "central_bank_balance" in arm.tools


# ---------------------------------------------------------------------------
# XV. Tool interface conformance
# ---------------------------------------------------------------------------


class TestToolInterface(unittest.TestCase):
    def test_has_required_properties(self):
        tool = _make_tool()
        assert tool.name == "central_bank_balance"
        assert isinstance(tool.description, str)
        assert len(tool.description) > 10
        assert isinstance(tool.parameters, dict)
        assert "mode" in tool.parameters["properties"]

    def test_parameters_schema_valid(self):
        tool = _make_tool()
        props = tool.parameters["properties"]
        assert props["mode"]["type"] == "string"
        assert set(props["mode"]["enum"]) == VALID_MODES
        assert props["period"]["type"] == "string"
        assert props["banks"]["type"] == "string"

    def test_required_fields(self):
        tool = _make_tool()
        assert "mode" in tool.parameters.get("required", [])

    def test_execute_with_extra_kwargs(self):
        """Unknown kwargs should be ignored."""
        tool = _make_tool()
        r = tool.execute(mode="bogus", foo="bar", baz=123)
        assert not r.success  # Invalid mode, but shouldn't crash


# ---------------------------------------------------------------------------
# XVI. _fetch_fx_rates
# ---------------------------------------------------------------------------


class TestFetchFxRates(unittest.TestCase):
    def test_fetches_all_unique_fx_series(self):
        tool = _make_tool()
        fx_series = set()
        for cb in CB_REGISTRY.values():
            if cb.get("fx_series"):
                fx_series.add(cb["fx_series"])

        called_series = []

        def track_fetch(series_id, start, end, cache_key, ttl):
            called_series.append(series_id)
            return [{"date": "2026-01-01", "value": "1.0"}]

        with patch.object(tool, "_fetch_fred_observations", side_effect=track_fetch):
            rates = tool._fetch_fx_rates()

        assert set(called_series) == fx_series
        assert len(rates) == len(fx_series)

    def test_partial_fx_failure(self):
        """If some FX series fail, others still return."""
        tool = _make_tool()
        call_count = [0]

        def intermittent_fetch(series_id, start, end, cache_key, ttl):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return []
            return [{"date": "2026-01-01", "value": "1.5"}]

        with patch.object(tool, "_fetch_fred_observations", side_effect=intermittent_fetch):
            rates = tool._fetch_fx_rates()

        assert len(rates) > 0  # At least some succeeded
        assert len(rates) < len([cb for cb in CB_REGISTRY.values() if cb.get("fx_series")])


# ---------------------------------------------------------------------------
# XVII. _fetch_policy_rate
# ---------------------------------------------------------------------------


class TestFetchPolicyRate(unittest.TestCase):
    def test_fed_rate(self):
        tool = _make_tool()

        def fake_fetch(series_id, start, end, cache_key, ttl):
            if series_id == "DFF":
                return [{"date": "2026-01-01", "value": "5.33"}]
            return []

        with patch.object(tool, "_fetch_fred_observations", side_effect=fake_fetch):
            rate = tool._fetch_policy_rate("fed", "2025-01-01", "2026-03-01")
        assert rate == pytest.approx(5.33)

    def test_ecb_rate_delegation(self):
        tool = _make_tool()
        with patch.object(tool, "_fetch_ecb_rate", return_value={"current_rate": 3.75}):
            rate = tool._fetch_policy_rate("ecb", "2025-01-01", "2026-03-01")
        assert rate == pytest.approx(3.75)

    def test_no_rate_series(self):
        tool = _make_tool()
        rate = tool._fetch_policy_rate("boj", "2025-01-01", "2026-03-01")
        assert rate is None


# ---------------------------------------------------------------------------
# Phase 27: L2 country entity persistence tests
# ---------------------------------------------------------------------------


def _make_store_mock() -> MagicMock:
    """Build a mock PipelineStore for L2 persistence testing."""
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    store.link_entities = MagicMock(return_value=1)
    return store


class TestL2PersistenceBalanceSheets:
    """Phase 27 — cb_balance_sheet observations on country nodes."""

    def test_balance_sheet_mode_persists_obs(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "banks": [
                {
                    "code": "fed",
                    "native_trillions": 7.5,
                    "usd_trillions": 7.5,
                    "wow_pct": 0.1,
                    "mom_pct": -0.5,
                    "yoy_pct": 2.0,
                },
                {
                    "code": "ecb",
                    "native_trillions": 6.0,
                    "usd_trillions": 6.5,
                    "wow_pct": 0.0,
                    "mom_pct": -1.0,
                    "yoy_pct": -3.0,
                },
            ],
            "errors": [],
        }
        counts = tool._persist_entities(data, "balance_sheets", ["fed", "ecb"])
        assert counts["balance_sheet_obs"] == 2
        assert counts["rate_obs"] == 0

    def test_balance_sheet_obs_type_is_correct(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "banks": [
                {
                    "code": "fed",
                    "native_trillions": 7.5,
                    "usd_trillions": 7.5,
                    "wow_pct": 0.1,
                    "mom_pct": -0.5,
                    "yoy_pct": 2.0,
                }
            ],
            "errors": [],
        }
        tool._persist_entities(data, "balance_sheets", ["fed"])
        obs_calls = store.store_entity_observation.call_args_list
        assert len(obs_calls) == 1
        assert obs_calls[0].kwargs["observation_type"] == "cb_balance_sheet"
        assert obs_calls[0].kwargs["source_tool"] == "central_bank_balance"
        assert obs_calls[0].kwargs["depth_level"] == 2

    def test_balance_sheet_targets_correct_country_entity(self):
        from agent.pipeline.entity import entity_id_from_key

        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "banks": [
                {
                    "code": "boj",
                    "native_trillions": 5.0,
                    "usd_trillions": 4.0,
                    "wow_pct": None,
                    "mom_pct": 1.0,
                    "yoy_pct": 5.0,
                }
            ],
            "errors": [],
        }
        tool._persist_entities(data, "balance_sheets", ["boj"])

        jp_eid = entity_id_from_key("country", "JP")
        obs_call = store.store_entity_observation.call_args_list[0]
        assert obs_call.kwargs["entity_id"] == jp_eid

    def test_balance_sheet_registers_country_entity(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "banks": [
                {
                    "code": "snb",
                    "native_trillions": 0.8,
                    "usd_trillions": 0.9,
                    "wow_pct": None,
                    "mom_pct": 0.0,
                    "yoy_pct": -1.0,
                }
            ],
            "errors": [],
        }
        tool._persist_entities(data, "balance_sheets", ["snb"])
        country_regs = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "country"]
        assert len(country_regs) == 1
        assert country_regs[0].kwargs["canonical_name"] == "CH"

    def test_balance_sheet_unknown_cb_code_skipped(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "banks": [
                {
                    "code": "unknown_bank",
                    "native_trillions": 1.0,
                    "usd_trillions": 1.0,
                    "wow_pct": None,
                    "mom_pct": None,
                    "yoy_pct": None,
                }
            ],
            "errors": [],
        }
        counts = tool._persist_entities(data, "balance_sheets", [])
        assert counts["balance_sheet_obs"] == 0

    def test_balance_sheet_empty_banks_list(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {"banks": [], "errors": []}
        counts = tool._persist_entities(data, "balance_sheets", [])
        assert counts["balance_sheet_obs"] == 0
        assert counts["rate_obs"] == 0

    def test_balance_sheet_obs_value_fields(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "banks": [
                {
                    "code": "fed",
                    "native_trillions": 7.5,
                    "usd_trillions": 7.5,
                    "wow_pct": 0.1,
                    "mom_pct": -0.5,
                    "yoy_pct": 2.0,
                }
            ],
            "errors": [],
        }
        tool._persist_entities(data, "balance_sheets", ["fed"])
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        assert val["cb_code"] == "fed"
        assert val["native_trillions"] == 7.5
        assert val["usd_trillions"] == 7.5
        assert val["wow_pct"] == 0.1
        assert val["mom_pct"] == -0.5
        assert val["yoy_pct"] == 2.0


class TestL2PersistencePolicyDivergence:
    """Phase 27 — policy_divergence mode persists both obs types."""

    def test_divergence_persists_balance_and_rate(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "assessments": [
                {"code": "fed", "growth_3m_ann": 5.0, "growth_12m": 3.0, "stance": "expanding"},
            ],
            "rates": {"fed": 5.33},
            "divergences": [],
            "synchronized": "",
            "errors": [],
        }
        counts = tool._persist_entities(data, "policy_divergence", ["fed"])
        assert counts["balance_sheet_obs"] == 1
        assert counts["rate_obs"] == 1

    def test_divergence_rate_value_is_scalar(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "assessments": [],
            "rates": {"ecb": 3.75},
            "divergences": [],
            "synchronized": "",
            "errors": [],
        }
        tool._persist_entities(data, "policy_divergence", ["ecb"])
        rate_calls = [
            c for c in store.store_entity_observation.call_args_list if c.kwargs["observation_type"] == "cb_policy_rate"
        ]
        assert len(rate_calls) == 1
        assert rate_calls[0].kwargs["value"]["current_rate"] == 3.75


class TestL2PersistenceRateMonitor:
    """Phase 27 — rate_monitor mode persists cb_policy_rate."""

    def test_rate_monitor_persists_rate_obs(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "rates": [
                {
                    "code": "fed",
                    "current_rate": 5.33,
                    "rate_date": "2026-04-10",
                    "last_change_date": "2025-12-18",
                    "last_change_direction": "cut",
                    "last_change_bps": -25.0,
                    "days_since_change": 114,
                },
            ],
            "errors": [],
        }
        counts = tool._persist_entities(data, "rate_monitor", ["fed"])
        assert counts["rate_obs"] == 1
        assert counts["balance_sheet_obs"] == 0

    def test_rate_monitor_obs_type_correct(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "rates": [
                {
                    "code": "fed",
                    "current_rate": 5.33,
                    "rate_date": "2026-04-10",
                    "last_change_date": None,
                    "last_change_direction": None,
                    "last_change_bps": None,
                    "days_since_change": None,
                },
            ],
            "errors": [],
        }
        tool._persist_entities(data, "rate_monitor", ["fed"])
        obs_call = store.store_entity_observation.call_args_list[0]
        assert obs_call.kwargs["observation_type"] == "cb_policy_rate"
        assert obs_call.kwargs["depth_level"] == 2

    def test_rate_monitor_targets_correct_country(self):
        from agent.pipeline.entity import entity_id_from_key

        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "rates": [
                {
                    "code": "ecb",
                    "current_rate": 3.75,
                    "rate_date": "2026-04-10",
                    "last_change_date": None,
                    "last_change_direction": None,
                    "last_change_bps": None,
                    "days_since_change": None,
                },
            ],
            "errors": [],
        }
        tool._persist_entities(data, "rate_monitor", ["ecb"])
        obs_call = store.store_entity_observation.call_args_list[0]
        eu_eid = entity_id_from_key("country", "EU")
        assert obs_call.kwargs["entity_id"] == eu_eid

    def test_rate_monitor_multiple_banks(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "rates": [
                {
                    "code": "fed",
                    "current_rate": 5.33,
                    "rate_date": "2026-04-10",
                    "last_change_date": None,
                    "last_change_direction": None,
                    "last_change_bps": None,
                    "days_since_change": None,
                },
                {
                    "code": "ecb",
                    "current_rate": 3.75,
                    "rate_date": "2026-04-10",
                    "last_change_date": None,
                    "last_change_direction": None,
                    "last_change_bps": None,
                    "days_since_change": None,
                },
            ],
            "errors": [],
        }
        counts = tool._persist_entities(data, "rate_monitor", ["fed", "ecb"])
        assert counts["rate_obs"] == 2

    def test_rate_monitor_obs_value_fields(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "rates": [
                {
                    "code": "fed",
                    "current_rate": 5.33,
                    "rate_date": "2026-04-10",
                    "last_change_date": "2025-12-18",
                    "last_change_direction": "cut",
                    "last_change_bps": -25.0,
                    "days_since_change": 114,
                },
            ],
            "errors": [],
        }
        tool._persist_entities(data, "rate_monitor", ["fed"])
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        assert val["cb_code"] == "fed"
        assert val["current_rate"] == 5.33
        assert val["last_change_direction"] == "cut"
        assert val["last_change_bps"] == -25.0
        assert val["days_since_change"] == 114


class TestL2PersistenceEdgeCases:
    """Phase 27 — edge cases for L2 persistence."""

    def test_no_store_returns_zeros(self):
        tool = _make_tool(cache=MagicMock())
        tool._store = None
        data = {"banks": [{"code": "fed"}], "errors": []}
        counts = tool._persist_entities(data, "balance_sheets", ["fed"])
        assert counts == {"balance_sheet_obs": 0, "rate_obs": 0}

    def test_no_entity_id_from_key_returns_zeros(self):
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store
        import agent.tools.central_bank_balance as cb_mod

        original = cb_mod._entity_id_from_key
        try:
            cb_mod._entity_id_from_key = None  # type: ignore
            counts = tool._persist_entities(
                {"banks": [{"code": "fed"}], "errors": []},
                "balance_sheets",
                ["fed"],
            )
            assert counts == {"balance_sheet_obs": 0, "rate_obs": 0}
        finally:
            cb_mod._entity_id_from_key = original

    def test_inner_exception_caught(self):
        store = _make_store_mock()
        store.register_entity.side_effect = RuntimeError("DB down")
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "banks": [
                {
                    "code": "fed",
                    "native_trillions": 7.5,
                    "usd_trillions": 7.5,
                    "wow_pct": None,
                    "mom_pct": None,
                    "yoy_pct": None,
                }
            ],
            "errors": [],
        }
        counts = tool._persist_entities(data, "balance_sheets", ["fed"])
        assert counts == {"balance_sheet_obs": 0, "rate_obs": 0}

    def test_liquidity_index_mode_no_persistence(self):
        """liquidity_index mode produces aggregate data — no L2 obs expected."""
        store = _make_store_mock()
        tool = _make_tool(cache=MagicMock())
        tool._store = store

        data = {
            "gross_usd": 20e12,
            "rrp_usd": 0.5e12,
            "tga_usd": 0.8e12,
            "net_usd": 18.7e12,
            "components": [],
            "errors": [],
        }
        counts = tool._persist_entities(data, "liquidity_index", ["fed"])
        assert counts["balance_sheet_obs"] == 0
        assert counts["rate_obs"] == 0

    def test_cb_to_country_mapping_complete(self):
        """Every CB in the registry has a country mapping."""
        from agent.tools.central_bank_balance import CB_REGISTRY, CB_TO_COUNTRY

        for cb_code in CB_REGISTRY:
            assert cb_code in CB_TO_COUNTRY, f"CB {cb_code} missing from CB_TO_COUNTRY mapping"

    def test_cb_to_country_values_are_iso(self):
        from agent.tools.central_bank_balance import CB_TO_COUNTRY

        for cb_code, country in CB_TO_COUNTRY.items():
            assert country == country.upper(), f"CB_TO_COUNTRY[{cb_code}]={country!r} not uppercase"
            assert 2 <= len(country) <= 6

    def test_execute_calls_persist_on_success(self):
        """execute() should call _persist_entities when mode returns success."""
        tool = _make_tool()
        tool._store = _make_store_mock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"banks": [{"code": "fed"}], "errors": []}

        with patch.object(tool, "_mode_balance_sheets", return_value=mock_result):
            with patch.object(tool, "_persist_entities") as mock_persist:
                tool.execute(mode="balance_sheets", banks="fed")
                mock_persist.assert_called_once()

    def test_execute_skips_persist_on_failure(self):
        """execute() should NOT call _persist_entities when mode fails."""
        tool = _make_tool()
        tool._store = _make_store_mock()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.data = None

        with patch.object(tool, "_mode_balance_sheets", return_value=mock_result):
            with patch.object(tool, "_persist_entities") as mock_persist:
                tool.execute(mode="balance_sheets", banks="fed")
                mock_persist.assert_not_called()


class TestGraphBuilderPhase27:
    """Phase 27 — graph builder registration for CB obs types."""

    def test_cb_balance_sheet_in_obs_types(self):
        from agent.models.gnn.graph_builder import OBSERVATION_TYPES

        assert "cb_balance_sheet" in OBSERVATION_TYPES

    def test_cb_policy_rate_in_obs_types(self):
        from agent.models.gnn.graph_builder import OBSERVATION_TYPES

        assert "cb_policy_rate" in OBSERVATION_TYPES

    def test_enrichment_dim_updated(self):
        from agent.models.gnn.graph_builder import ENRICHMENT_DIM, OBSERVATION_TYPES

        expected = 9 + len(OBSERVATION_TYPES)
        assert expected == ENRICHMENT_DIM, f"ENRICHMENT_DIM={ENRICHMENT_DIM} != expected {expected}"

    def test_obs_types_alphabetically_sorted(self):
        from agent.models.gnn.graph_builder import OBSERVATION_TYPES

        assert sorted(OBSERVATION_TYPES) == OBSERVATION_TYPES, "OBSERVATION_TYPES must be alphabetically sorted"


if __name__ == "__main__":
    unittest.main()
