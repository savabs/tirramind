"""
Edge-case test suite for FinraShortVolumeTool (Phase 6e).

Tests cover both data modes: short_volume and short_interest.
No API key required — all FINRA endpoints are free/public.

Coverage:
  - Mode validation (invalid, missing)
  - Parameter validation (date format, days_back bounds, limit bounds, min_vol)
  - Short volume ticker: normal, empty, multi-day trend, single day, anomaly detection
  - Short volume scan: normal, empty (weekend/holiday), pagination, min_vol filter
  - Short interest: normal, no ticker error, no data found, squeeze risk, building, covering
  - Aggregation: multi-facility merge, zero volume, fractional par quantities
  - Signal computation: trend (rising, falling, flat), z-score, anomaly flag
  - Date helpers: trading_dates (weekday skip), si_settlement_dates generation
  - Cache interaction (get/put with correct API)
  - API error handling: timeout, 204, 400, 429, 500, non-JSON, non-list
  - Output formatting
  - CLI registration (21 tools)
  - Bandit arm (institutional_flow)
  - _safe_float edge cases
  - _si_record_to_dict normalization
"""

import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/home/becmachlean/2024/projects/tirramind_v1")

from agent.tools.finra_short_volume import FinraShortVolumeTool, _safe_float

# ──────────────────────────────────────────────────────────────────
# Mock helpers
# ──────────────────────────────────────────────────────────────────


def _reg_sho_record(ticker: str, total: float, short: float, exempt: float = 0, facility: str = "NQTRF") -> dict:
    return {
        "securitiesInformationProcessorSymbolIdentifier": ticker,
        "totalParQuantity": total,
        "shortParQuantity": short,
        "shortExemptParQuantity": exempt,
        "reportingFacilityCode": facility,
    }


def _si_record(
    ticker: str = "AAPL",
    date: str = "2026-01-15",
    current_si: int = 100_000_000,
    previous_si: int = 95_000_000,
    change_pct: float = 5.3,
    dtc: float = 2.5,
    adv: int = 40_000_000,
    market_class: str = "Q",
    issue_name: str = "APPLE INC",
) -> dict:
    return {
        "symbolCode": ticker,
        "settlementDate": date,
        "currentShortPositionQuantity": current_si,
        "previousShortPositionQuantity": previous_si,
        "changePercent": change_pct,
        "daysToCoverQuantity": dtc,
        "averageDailyVolumeQuantity": adv,
        "marketClassCode": market_class,
        "issueName": issue_name,
    }


def _make_tool(cache=None):
    return FinraShortVolumeTool(cache=cache)


def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data if json_data is not None else []
    return resp


# ──────────────────────────────────────────────────────────────────
# 1. Mode & Parameter Validation
# ──────────────────────────────────────────────────────────────────


class TestModeValidation(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    def test_invalid_mode(self):
        result = self.tool.execute(mode="invalid_mode")
        self.assertFalse(result.success)
        self.assertIn("Invalid mode", result.output)

    def test_empty_mode(self):
        result = self.tool.execute(mode="")
        self.assertFalse(result.success)
        self.assertIn("Invalid mode", result.output)

    def test_missing_mode(self):
        result = self.tool.execute()
        self.assertFalse(result.success)
        self.assertIn("Invalid mode", result.output)

    def test_invalid_date_format(self):
        result = self.tool.execute(mode="short_volume", ticker="AAPL", date="03-25-2026")
        self.assertFalse(result.success)
        self.assertIn("Invalid date format", result.output)

    def test_invalid_date_gibberish(self):
        result = self.tool.execute(mode="short_volume", date="not-a-date")
        self.assertFalse(result.success)
        self.assertIn("Invalid date format", result.output)

    def test_days_back_clamped_min(self):
        """days_back < 1 should be clamped to 1, not error."""
        with patch.object(self.tool, "_fetch_reg_sho", return_value=[]):
            result = self.tool.execute(mode="short_volume", ticker="AAPL", days_back=0)
            self.assertTrue(result.success)  # 0 clamped to 1, returns empty but no error

    def test_days_back_clamped_max(self):
        """days_back > 20 should be clamped to 20."""
        with patch.object(self.tool, "_fetch_reg_sho", return_value=[]):
            result = self.tool.execute(mode="short_volume", ticker="AAPL", days_back=100)
            self.assertTrue(result.success)

    def test_limit_clamped_min(self):
        """limit < 1 should be clamped to 1."""
        with patch.object(self.tool, "_fetch_reg_sho", return_value=[]):
            result = self.tool.execute(mode="short_volume", limit=0)
            self.assertTrue(result.success)

    def test_limit_clamped_max(self):
        """limit > 100 should be clamped to 100."""
        with patch.object(self.tool, "_fetch_reg_sho", return_value=[]):
            result = self.tool.execute(mode="short_volume", limit=999)
            self.assertTrue(result.success)

    def test_min_total_volume_zero(self):
        """min_total_volume=0 is valid (show everything)."""
        with patch.object(self.tool, "_fetch_reg_sho", return_value=[]):
            result = self.tool.execute(mode="short_volume", min_total_volume=0)
            self.assertTrue(result.success)

    def test_min_total_volume_negative_clamped(self):
        """Negative min_total_volume clamped to 0."""
        with patch.object(self.tool, "_fetch_reg_sho", return_value=[]):
            result = self.tool.execute(mode="short_volume", min_total_volume=-50)
            self.assertTrue(result.success)


# ──────────────────────────────────────────────────────────────────
# 2. Tool Metadata
# ──────────────────────────────────────────────────────────────────


class TestToolMetadata(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    def test_name(self):
        self.assertEqual(self.tool.name, "finra_short_volume")

    def test_description_nonempty(self):
        self.assertTrue(len(self.tool.description) > 20)

    def test_parameters_has_mode(self):
        params = self.tool.parameters
        self.assertIn("mode", params["properties"])
        self.assertEqual(params["required"], ["mode"])

    def test_mode_enum(self):
        mode_prop = self.tool.parameters["properties"]["mode"]
        self.assertEqual(set(mode_prop["enum"]), {"short_volume", "short_interest"})


# ──────────────────────────────────────────────────────────────────
# 3. Short Volume — Ticker Mode
# ──────────────────────────────────────────────────────────────────


class TestShortVolumeTicker(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    def test_single_day_success(self):
        records = [
            _reg_sho_record("AAPL", 5_000_000, 2_000_000, 10_000, "NQTRF"),
            _reg_sho_record("AAPL", 3_000_000, 1_500_000, 5_000, "NYTRF"),
        ]
        with patch.object(self.tool, "_fetch_reg_sho", return_value=records):
            result = self.tool.execute(mode="short_volume", ticker="AAPL", days_back=1)
            self.assertTrue(result.success)
            self.assertIn("AAPL", result.output)
            self.assertEqual(result.data["ticker"], "AAPL")
            self.assertEqual(len(result.data["records"]), 1)
            rec = result.data["records"][0]
            self.assertAlmostEqual(rec["total_volume"], 8_000_000)
            self.assertAlmostEqual(rec["short_volume"], 3_500_000)
            self.assertAlmostEqual(rec["short_ratio"], 3_500_000 / 8_000_000, places=4)

    def test_no_data_found(self):
        with patch.object(self.tool, "_fetch_reg_sho", return_value=[]):
            result = self.tool.execute(mode="short_volume", ticker="ZZZZ", days_back=3)
            self.assertTrue(result.success)  # empty is still success
            self.assertIn("No Reg SHO data found", result.output)
            self.assertEqual(result.data["records"], [])

    def test_multi_day_trend(self):
        """5 days of data should compute trend and z-score."""
        call_count = [0]

        def mock_fetch(date_str, ticker=None, offset=0):
            call_count[0] += 1
            ratios = [0.40, 0.42, 0.38, 0.45, 0.43]
            idx = min(call_count[0] - 1, len(ratios) - 1)
            short = ratios[idx] * 10_000_000
            return [_reg_sho_record("AAPL", 10_000_000, short)]

        with patch.object(self.tool, "_fetch_reg_sho", side_effect=mock_fetch):
            result = self.tool.execute(mode="short_volume", ticker="AAPL", days_back=5)
            self.assertTrue(result.success)
            self.assertEqual(len(result.data["records"]), 5)
            signals = result.data["signals"]
            self.assertIn("trend", signals)
            self.assertIn(signals["trend"], ("rising", "falling", "flat"))
            self.assertIsNotNone(signals.get("zscore"))
            self.assertIsNotNone(signals.get("avg_ratio"))

    def test_ticker_case_normalization(self):
        """Lowercase ticker should be uppercased."""
        with patch.object(
            self.tool,
            "_fetch_reg_sho",
            return_value=[
                _reg_sho_record("AAPL", 1_000_000, 400_000),
            ],
        ):
            result = self.tool.execute(mode="short_volume", ticker="aapl", days_back=1)
            self.assertTrue(result.success)
            self.assertEqual(result.data["ticker"], "AAPL")

    def test_ticker_whitespace_stripped(self):
        """Ticker with spaces should be stripped."""
        with patch.object(
            self.tool,
            "_fetch_reg_sho",
            return_value=[
                _reg_sho_record("NVDA", 1_000_000, 500_000),
            ],
        ):
            result = self.tool.execute(mode="short_volume", ticker="  nvda  ", days_back=1)
            self.assertTrue(result.success)
            self.assertEqual(result.data["ticker"], "NVDA")


# ──────────────────────────────────────────────────────────────────
# 4. Short Volume — Scan Mode
# ──────────────────────────────────────────────────────────────────


class TestShortVolumeScan(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    def test_scan_basic(self):
        records = [
            _reg_sho_record("AAPL", 10_000_000, 4_000_000),
            _reg_sho_record("TSLA", 8_000_000, 5_000_000),
            _reg_sho_record("AMZN", 6_000_000, 1_500_000),
        ]
        with patch.object(self.tool, "_fetch_reg_sho", return_value=records):
            result = self.tool.execute(mode="short_volume", date="2026-03-24", limit=10)
            self.assertTrue(result.success)
            self.assertIn("Scan", result.output)
            # Results sorted by short ratio desc: TSLA (62.5%) > AAPL (40%) > AMZN (25%)
            results = result.data["results"]
            self.assertEqual(results[0]["ticker"], "TSLA")
            self.assertEqual(results[-1]["ticker"], "AMZN")

    def test_scan_empty_weekend(self):
        with patch.object(self.tool, "_fetch_reg_sho", return_value=[]):
            result = self.tool.execute(mode="short_volume", date="2026-03-22")  # Saturday
            self.assertTrue(result.success)
            self.assertIn("No Reg SHO data", result.output)

    def test_scan_min_vol_filter(self):
        """Tickers below min_total_volume should be excluded."""
        records = [
            _reg_sho_record("AAPL", 10_000_000, 4_000_000),
            _reg_sho_record("TINY", 50_000, 40_000),  # below 100k default
        ]
        with patch.object(self.tool, "_fetch_reg_sho", return_value=records):
            result = self.tool.execute(mode="short_volume", date="2026-03-24", min_total_volume=100_000)
            self.assertTrue(result.success)
            tickers = [r["ticker"] for r in result.data["results"]]
            self.assertIn("AAPL", tickers)
            self.assertNotIn("TINY", tickers)

    def test_scan_limit_truncation(self):
        """Only `limit` results should be returned."""
        records = [_reg_sho_record(f"SYM{i}", 1_000_000, 500_000) for i in range(50)]
        with patch.object(self.tool, "_fetch_reg_sho", return_value=records):
            result = self.tool.execute(mode="short_volume", date="2026-03-24", limit=5, min_total_volume=0)
            self.assertTrue(result.success)
            self.assertEqual(len(result.data["results"]), 5)

    def test_scan_pagination(self):
        """Multiple pages should be fetched and aggregated."""
        page1 = [_reg_sho_record(f"SYM{i}", 1_000_000, 600_000) for i in range(5000)]
        page2 = [_reg_sho_record(f"SYM{5000 + i}", 1_000_000, 400_000) for i in range(100)]

        call_count = [0]

        def mock_fetch(date_str, ticker=None, offset=0):
            call_count[0] += 1
            if offset == 0:
                return page1
            elif offset == 5000:
                return page2
            return []

        with patch.object(self.tool, "_fetch_reg_sho", side_effect=mock_fetch):
            result = self.tool.execute(mode="short_volume", date="2026-03-24", limit=10, min_total_volume=0)
            self.assertTrue(result.success)
            # Should have paginated
            self.assertGreaterEqual(call_count[0], 2)
            self.assertGreater(result.data["total_tickers"], 5000)


# ──────────────────────────────────────────────────────────────────
# 5. Short Interest Mode
# ──────────────────────────────────────────────────────────────────


class TestShortInterest(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    def test_no_ticker_error(self):
        result = self.tool.execute(mode="short_interest")
        self.assertFalse(result.success)
        self.assertIn("requires a ticker", result.output)

    def test_basic_success(self):
        records = [_si_record()]
        with patch.object(self.tool, "_fetch_short_interest_recent", return_value=records):
            result = self.tool.execute(mode="short_interest", ticker="AAPL")
            self.assertTrue(result.success)
            self.assertIn("AAPL", result.output)
            self.assertEqual(result.data["ticker"], "AAPL")
            self.assertEqual(len(result.data["records"]), 1)
            signals = result.data["signals"]
            self.assertEqual(signals["current_short_position"], 100_000_000)
            self.assertEqual(signals["days_to_cover"], 2.5)

    def test_no_data_found(self):
        with patch.object(self.tool, "_fetch_short_interest_recent", return_value=[]):
            result = self.tool.execute(mode="short_interest", ticker="ZZZZ")
            self.assertTrue(result.success)
            self.assertIn("No short interest data", result.output)
            self.assertEqual(result.data["records"], [])

    def test_squeeze_risk_flag(self):
        """DTC > 5.0 should flag squeeze risk."""
        records = [_si_record(dtc=7.5)]
        with patch.object(self.tool, "_fetch_short_interest_recent", return_value=records):
            result = self.tool.execute(mode="short_interest", ticker="GME")
            self.assertTrue(result.success)
            self.assertTrue(result.data["signals"]["squeeze_risk"])
            self.assertIn("SQUEEZE RISK", result.output)

    def test_building_short_flag(self):
        """change_pct > 15 should flag building."""
        records = [_si_record(change_pct=25.0)]
        with patch.object(self.tool, "_fetch_short_interest_recent", return_value=records):
            result = self.tool.execute(mode="short_interest", ticker="TSLA")
            self.assertTrue(result.success)
            self.assertTrue(result.data["signals"]["building_short"])
            self.assertIn("BUILDING", result.output)

    def test_covering_flag(self):
        """change_pct < -15 should flag covering."""
        records = [_si_record(change_pct=-20.0)]
        with patch.object(self.tool, "_fetch_short_interest_recent", return_value=records):
            result = self.tool.execute(mode="short_interest", ticker="SPY")
            self.assertTrue(result.success)
            self.assertTrue(result.data["signals"]["covering"])
            self.assertIn("COVERING", result.output)

    def test_no_flags_moderate_change(self):
        """Moderate change should not trigger any flags."""
        records = [_si_record(dtc=2.0, change_pct=5.0)]
        with patch.object(self.tool, "_fetch_short_interest_recent", return_value=records):
            result = self.tool.execute(mode="short_interest", ticker="AAPL")
            self.assertTrue(result.success)
            self.assertFalse(result.data["signals"]["squeeze_risk"])
            self.assertFalse(result.data["signals"]["building_short"])
            self.assertFalse(result.data["signals"]["covering"])

    def test_multiple_periods(self):
        """Multiple settlement dates returned."""
        records = [
            _si_record(date="2026-01-31", current_si=120_000_000, change_pct=10.0),
            _si_record(date="2026-01-15", current_si=110_000_000, change_pct=5.0),
        ]
        with patch.object(self.tool, "_fetch_short_interest_recent", return_value=records):
            result = self.tool.execute(mode="short_interest", ticker="AAPL")
            self.assertTrue(result.success)
            self.assertEqual(len(result.data["records"]), 2)
            # Signals based on latest (first) record
            self.assertEqual(result.data["signals"]["current_short_position"], 120_000_000)

    def test_si_record_normalization(self):
        """_si_record_to_dict should normalize all fields."""
        raw = _si_record(ticker="TSLA", date="2026-02-28", current_si=50_000, dtc=3.2)
        normalized = self.tool._si_record_to_dict(raw)
        self.assertEqual(normalized["symbol"], "TSLA")
        self.assertEqual(normalized["settlement_date"], "2026-02-28")
        self.assertEqual(normalized["current_short_position"], 50_000)
        self.assertEqual(normalized["days_to_cover"], 3.2)

    def test_si_record_missing_fields(self):
        """Missing fields should default gracefully."""
        normalized = self.tool._si_record_to_dict({})
        self.assertEqual(normalized["symbol"], "")
        self.assertEqual(normalized["settlement_date"], "")
        self.assertEqual(normalized["current_short_position"], 0)


# ──────────────────────────────────────────────────────────────────
# 6. Aggregation Logic
# ──────────────────────────────────────────────────────────────────


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    def test_multi_facility_merge(self):
        """Three facilities for same ticker should be summed."""
        records = [
            _reg_sho_record("AAPL", 5_000_000, 2_000_000, 1_000, "NQTRF"),
            _reg_sho_record("AAPL", 3_000_000, 1_500_000, 500, "NYTRF"),
            _reg_sho_record("AAPL", 2_000_000, 800_000, 200, "NCTRF"),
        ]
        agg = self.tool._aggregate_facilities(records)
        self.assertIn("AAPL", agg)
        self.assertAlmostEqual(agg["AAPL"]["total_volume"], 10_000_000)
        self.assertAlmostEqual(agg["AAPL"]["short_volume"], 4_300_000)
        self.assertAlmostEqual(agg["AAPL"]["exempt_volume"], 1_700)
        self.assertEqual(agg["AAPL"]["facility_count"], 3)
        self.assertAlmostEqual(agg["AAPL"]["short_ratio"], 0.43, places=2)

    def test_zero_total_volume(self):
        """Zero total volume should produce 0 ratio, not division error."""
        records = [_reg_sho_record("ZERO", 0, 0, 0)]
        agg = self.tool._aggregate_facilities(records)
        self.assertEqual(agg["ZERO"]["short_ratio"], 0.0)

    def test_empty_records(self):
        """Empty input should return empty dict."""
        agg = self.tool._aggregate_facilities([])
        self.assertEqual(agg, {})

    def test_missing_ticker_field(self):
        """Records without ticker field should be skipped."""
        records = [{"totalParQuantity": 1000, "shortParQuantity": 500}]
        agg = self.tool._aggregate_facilities(records)
        self.assertEqual(agg, {})

    def test_fractional_par_quantities(self):
        """FINRA sometimes returns floats like 109953.7638."""
        records = [_reg_sho_record("BOND", 109953.7638, 50000.1234, 100.5678)]
        agg = self.tool._aggregate_facilities(records)
        self.assertAlmostEqual(agg["BOND"]["total_volume"], 109953.7638, places=2)
        self.assertAlmostEqual(agg["BOND"]["short_volume"], 50000.1234, places=2)

    def test_multiple_tickers(self):
        """Multiple tickers should be aggregated independently."""
        records = [
            _reg_sho_record("AAPL", 10_000_000, 4_000_000),
            _reg_sho_record("TSLA", 8_000_000, 5_000_000),
            _reg_sho_record("AAPL", 5_000_000, 2_000_000, 0, "NYTRF"),
        ]
        agg = self.tool._aggregate_facilities(records)
        self.assertEqual(len(agg), 2)
        self.assertAlmostEqual(agg["AAPL"]["total_volume"], 15_000_000)
        self.assertAlmostEqual(agg["TSLA"]["total_volume"], 8_000_000)


# ──────────────────────────────────────────────────────────────────
# 7. Signal Computation
# ──────────────────────────────────────────────────────────────────


class TestSignalComputation(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    def test_empty_data(self):
        signals = self.tool._compute_volume_signals([])
        self.assertEqual(signals, {})

    def test_single_day_insufficient_data(self):
        data = [{"short_ratio": 0.45, "date": "2026-03-24"}]
        signals = self.tool._compute_volume_signals(data)
        self.assertAlmostEqual(signals["latest_ratio"], 0.45)
        self.assertIsNone(signals["avg_ratio"])
        self.assertIsNone(signals["zscore"])
        self.assertEqual(signals["trend"], "insufficient_data")

    def test_two_days_no_zscore(self):
        data = [
            {"short_ratio": 0.48, "date": "2026-03-24"},
            {"short_ratio": 0.42, "date": "2026-03-23"},
        ]
        signals = self.tool._compute_volume_signals(data)
        self.assertAlmostEqual(signals["avg_ratio"], 0.45, places=2)
        self.assertIsNone(signals["zscore"])  # Need >=3 for stdev

    def test_rising_trend(self):
        """Later days have higher ratios than earlier days."""
        data = [
            {"short_ratio": 0.50, "date": "2026-03-24"},  # most recent
            {"short_ratio": 0.48, "date": "2026-03-23"},
            {"short_ratio": 0.40, "date": "2026-03-22"},
            {"short_ratio": 0.38, "date": "2026-03-21"},
        ]
        signals = self.tool._compute_volume_signals(data)
        self.assertEqual(signals["trend"], "rising")

    def test_falling_trend(self):
        data = [
            {"short_ratio": 0.35, "date": "2026-03-24"},
            {"short_ratio": 0.38, "date": "2026-03-23"},
            {"short_ratio": 0.45, "date": "2026-03-22"},
            {"short_ratio": 0.50, "date": "2026-03-21"},
        ]
        signals = self.tool._compute_volume_signals(data)
        self.assertEqual(signals["trend"], "falling")

    def test_flat_trend(self):
        data = [
            {"short_ratio": 0.45, "date": "2026-03-24"},
            {"short_ratio": 0.44, "date": "2026-03-23"},
            {"short_ratio": 0.45, "date": "2026-03-22"},
            {"short_ratio": 0.44, "date": "2026-03-21"},
        ]
        signals = self.tool._compute_volume_signals(data)
        self.assertEqual(signals["trend"], "flat")

    def test_anomaly_high_zscore(self):
        """Large spike should trigger anomaly."""
        data = [
            {"short_ratio": 0.80, "date": "2026-03-24"},  # spike
            {"short_ratio": 0.45, "date": "2026-03-23"},
            {"short_ratio": 0.44, "date": "2026-03-22"},
            {"short_ratio": 0.43, "date": "2026-03-21"},
            {"short_ratio": 0.44, "date": "2026-03-20"},
        ]
        signals = self.tool._compute_volume_signals(data)
        self.assertTrue(signals.get("is_anomaly"))
        self.assertGreater(signals["zscore"], 1.5)

    def test_no_anomaly_normal_variation(self):
        data = [
            {"short_ratio": 0.45, "date": "2026-03-24"},
            {"short_ratio": 0.44, "date": "2026-03-23"},
            {"short_ratio": 0.46, "date": "2026-03-22"},
            {"short_ratio": 0.43, "date": "2026-03-21"},
            {"short_ratio": 0.45, "date": "2026-03-20"},
        ]
        signals = self.tool._compute_volume_signals(data)
        self.assertFalse(signals.get("is_anomaly", False))

    def test_zero_stdev(self):
        """All identical ratios → zero stdev → no anomaly, zscore 0."""
        data = [
            {"short_ratio": 0.50, "date": "2026-03-24"},
            {"short_ratio": 0.50, "date": "2026-03-23"},
            {"short_ratio": 0.50, "date": "2026-03-22"},
        ]
        signals = self.tool._compute_volume_signals(data)
        self.assertAlmostEqual(signals["zscore"], 0.0)
        self.assertFalse(signals["is_anomaly"])


# ──────────────────────────────────────────────────────────────────
# 8. Date Helpers
# ──────────────────────────────────────────────────────────────────


class TestDateHelpers(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    def test_trading_dates_skips_weekends(self):
        # Monday 2026-03-23
        monday = datetime(2026, 3, 23)
        dates = self.tool._trading_dates(monday, 5)
        self.assertEqual(len(dates), 5)
        for d in dates:
            self.assertLess(d.weekday(), 5, f"{d} is a weekend")

    def test_trading_dates_from_saturday(self):
        sat = datetime(2026, 3, 21)  # Saturday
        dates = self.tool._trading_dates(sat, 3)
        self.assertEqual(len(dates), 3)
        for d in dates:
            self.assertLess(d.weekday(), 5)
        # Saturday itself should not be in the list
        self.assertNotIn(sat, dates)

    def test_trading_dates_from_sunday(self):
        sun = datetime(2026, 3, 22)  # Sunday
        dates = self.tool._trading_dates(sun, 1)
        self.assertEqual(len(dates), 1)
        self.assertLess(dates[0].weekday(), 5)

    def test_trading_dates_order(self):
        """Most recent dates should come first."""
        fri = datetime(2026, 3, 20)  # Friday
        dates = self.tool._trading_dates(fri, 3)
        self.assertEqual(dates[0], fri)
        self.assertTrue(dates[0] > dates[1] > dates[2])

    def test_si_settlement_dates_generation(self):
        target = datetime(2026, 3, 25)
        candidates = self.tool._si_settlement_dates(target, lookback_days=60)
        self.assertTrue(len(candidates) > 0)
        # All should be YYYY-MM-DD strings
        for c in candidates:
            datetime.strptime(c, "%Y-%m-%d")
        # Should include 15th and end-of-month dates
        has_15 = any("15" in c.split("-")[2] for c in candidates)
        has_eom = any(c.split("-")[2] in ("28", "29", "30", "31") for c in candidates)
        self.assertTrue(has_15)
        self.assertTrue(has_eom)

    def test_si_settlement_dates_sorted_desc(self):
        target = datetime(2026, 3, 25)
        candidates = self.tool._si_settlement_dates(target, lookback_days=90)
        self.assertEqual(candidates, sorted(candidates, reverse=True))

    def test_si_settlement_dates_no_future(self):
        """No date should be after target date."""
        target = datetime(2026, 3, 25)
        candidates = self.tool._si_settlement_dates(target, lookback_days=90)
        for c in candidates:
            dt = datetime.strptime(c, "%Y-%m-%d")
            self.assertLessEqual(dt, target)

    def test_parse_date_valid(self):
        result = self.tool._parse_date("2026-03-24")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2026)
        self.assertEqual(result.month, 3)
        self.assertEqual(result.day, 24)

    def test_parse_date_empty_returns_today(self):
        result = self.tool._parse_date("")
        self.assertIsNotNone(result)
        self.assertEqual(result.date(), datetime.now().date())

    def test_parse_date_invalid(self):
        self.assertIsNone(self.tool._parse_date("not-a-date"))
        self.assertIsNone(self.tool._parse_date("2026/03/24"))
        self.assertIsNone(self.tool._parse_date("03-24-2026"))


# ──────────────────────────────────────────────────────────────────
# 9. _safe_float
# ──────────────────────────────────────────────────────────────────


class TestSafeFloat(unittest.TestCase):
    def test_integer(self):
        self.assertAlmostEqual(_safe_float(1000), 1000.0)

    def test_float(self):
        self.assertAlmostEqual(_safe_float(109953.7638), 109953.7638)

    def test_string_number(self):
        self.assertAlmostEqual(_safe_float("5000.50"), 5000.50)

    def test_none(self):
        self.assertAlmostEqual(_safe_float(None), 0.0)

    def test_empty_string(self):
        self.assertAlmostEqual(_safe_float(""), 0.0)

    def test_non_numeric_string(self):
        self.assertAlmostEqual(_safe_float("abc"), 0.0)

    def test_zero(self):
        self.assertAlmostEqual(_safe_float(0), 0.0)

    def test_negative(self):
        self.assertAlmostEqual(_safe_float(-100.5), -100.5)

    def test_very_large(self):
        self.assertAlmostEqual(_safe_float(999_999_999_999), 999_999_999_999.0)


# ──────────────────────────────────────────────────────────────────
# 10. API Error Handling
# ──────────────────────────────────────────────────────────────────


class TestApiErrorHandling(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_204_no_content(self, mock_post):
        mock_post.return_value = _mock_response(204)
        result = self.tool._api_post("https://fake.url", {})
        self.assertEqual(result, [])

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_400_bad_request(self, mock_post):
        mock_post.return_value = _mock_response(400, text="Bad request")
        result = self.tool._api_post("https://fake.url", {})
        self.assertIsNone(result)

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_429_rate_limit(self, mock_post):
        mock_post.return_value = _mock_response(429)
        result = self.tool._api_post("https://fake.url", {})
        self.assertIsNone(result)

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_500_server_error(self, mock_post):
        mock_post.return_value = _mock_response(500, text="Internal error")
        result = self.tool._api_post("https://fake.url", {})
        self.assertIsNone(result)

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_timeout_exception(self, mock_post):
        import httpx as httpx_mod

        mock_post.side_effect = httpx_mod.TimeoutException("timeout")
        with self.assertRaises(httpx_mod.TimeoutException):
            self.tool._api_post("https://fake.url", {})

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_timeout_caught_in_execute(self, mock_post):
        """Timeout during execute should return graceful error."""
        import httpx as httpx_mod

        mock_post.side_effect = httpx_mod.TimeoutException("timeout")
        result = self.tool.execute(mode="short_volume", ticker="AAPL", days_back=1)
        self.assertFalse(result.success)
        self.assertIn("timed out", result.output)

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_http_error(self, mock_post):
        import httpx as httpx_mod

        mock_post.side_effect = httpx_mod.HTTPError("connection failed")
        result = self.tool._api_post("https://fake.url", {})
        self.assertIsNone(result)

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_non_json_response(self, mock_post):
        resp = _mock_response(200)
        resp.json.side_effect = ValueError("not json")
        mock_post.return_value = resp
        result = self.tool._api_post("https://fake.url", {})
        self.assertIsNone(result)

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_non_list_json_response(self, mock_post):
        """API returning dict instead of list should return None."""
        mock_post.return_value = _mock_response(200, json_data={"error": "bad"})
        result = self.tool._api_post("https://fake.url", {})
        self.assertIsNone(result)

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_unexpected_status_code(self, mock_post):
        mock_post.return_value = _mock_response(403)
        result = self.tool._api_post("https://fake.url", {})
        self.assertIsNone(result)


# ──────────────────────────────────────────────────────────────────
# 11. Cache Interaction
# ──────────────────────────────────────────────────────────────────


class TestCacheInteraction(unittest.TestCase):
    def test_cache_hit_reg_sho(self):
        """Cached data should be returned without API call."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = [_reg_sho_record("AAPL", 10_000_000, 4_000_000)]
        tool = _make_tool(cache=mock_cache)

        with patch("agent.tools.finra_short_volume.httpx.post") as mock_post:
            records = tool._fetch_reg_sho("2026-03-24", ticker="AAPL")
            mock_post.assert_not_called()
            mock_cache.get.assert_called_once()
            self.assertEqual(len(records), 1)

    def test_cache_miss_fetches_and_stores(self):
        """Cache miss should fetch from API and store."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        tool = _make_tool(cache=mock_cache)

        with patch("agent.tools.finra_short_volume.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(
                200,
                json_data=[
                    _reg_sho_record("AAPL", 10_000_000, 4_000_000),
                ],
            )
            records = tool._fetch_reg_sho("2026-03-24", ticker="AAPL")
            mock_cache.put.assert_called_once()
            self.assertEqual(len(records), 1)

    def test_cache_hit_short_interest(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = [_si_record()]
        tool = _make_tool(cache=mock_cache)

        with patch("agent.tools.finra_short_volume.httpx.post") as mock_post:
            records = tool._fetch_short_interest("AAPL", "2026-01-15")
            mock_post.assert_not_called()
            self.assertEqual(len(records), 1)

    def test_no_cache_works(self):
        """Tool should work without a cache (cache=None)."""
        tool = _make_tool(cache=None)
        with patch("agent.tools.finra_short_volume.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(
                200,
                json_data=[
                    _reg_sho_record("AAPL", 10_000_000, 4_000_000),
                ],
            )
            records = tool._fetch_reg_sho("2026-03-24", ticker="AAPL")
            self.assertEqual(len(records), 1)

    def test_cache_not_stored_on_empty_result(self):
        """API returning 204/empty should NOT store in cache."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        tool = _make_tool(cache=mock_cache)

        with patch("agent.tools.finra_short_volume.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(204)
            records = tool._fetch_reg_sho("2026-03-22")
            self.assertEqual(records, [])
            mock_cache.put.assert_not_called()

    def test_cache_api_signature(self):
        """Cache should be called with (source_str, params_dict) not (single_key)."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        tool = _make_tool(cache=mock_cache)

        with patch("agent.tools.finra_short_volume.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(
                200,
                json_data=[
                    _reg_sho_record("AAPL", 1_000_000, 400_000),
                ],
            )
            tool._fetch_reg_sho("2026-03-24", ticker="AAPL")
            # Verify get was called with source string and params dict
            get_args = mock_cache.get.call_args
            self.assertEqual(get_args[0][0], "finra_regsho")
            self.assertIsInstance(get_args[0][1], dict)
            # Verify put was called with source string, params dict, and data
            put_args = mock_cache.put.call_args
            self.assertEqual(put_args[0][0], "finra_regsho")
            self.assertIsInstance(put_args[0][1], dict)
            self.assertIsInstance(put_args[0][2], list)


# ──────────────────────────────────────────────────────────────────
# 12. Output Formatting
# ──────────────────────────────────────────────────────────────────


class TestOutputFormatting(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    def test_short_volume_output_has_header(self):
        with patch.object(
            self.tool,
            "_fetch_reg_sho",
            return_value=[
                _reg_sho_record("NVDA", 60_000_000, 26_000_000, 5_000),
            ],
        ):
            result = self.tool.execute(mode="short_volume", ticker="NVDA", days_back=1)
            self.assertIn("## FINRA Short Volume: NVDA", result.output)

    def test_short_interest_output_has_header(self):
        with patch.object(self.tool, "_fetch_short_interest_recent", return_value=[_si_record()]):
            result = self.tool.execute(mode="short_interest", ticker="AAPL")
            self.assertIn("## FINRA Short Interest: AAPL", result.output)

    def test_scan_output_has_stats(self):
        records = [_reg_sho_record("SPY", 20_000_000, 10_000_000)]
        with patch.object(self.tool, "_fetch_reg_sho", return_value=records):
            result = self.tool.execute(mode="short_volume", date="2026-03-24", limit=5, min_total_volume=0)
            self.assertIn("Total tickers", result.output)
            self.assertIn("Total records fetched", result.output)

    def test_anomaly_flag_in_output(self):
        """Anomaly should appear in formatted output."""
        data = [
            {
                "short_ratio": 0.80,
                "date": "2026-03-24",
                "total_volume": 10e6,
                "short_volume": 8e6,
                "exempt_volume": 0,
            },
            {
                "short_ratio": 0.45,
                "date": "2026-03-23",
                "total_volume": 10e6,
                "short_volume": 4.5e6,
                "exempt_volume": 0,
            },
            {
                "short_ratio": 0.44,
                "date": "2026-03-22",
                "total_volume": 10e6,
                "short_volume": 4.4e6,
                "exempt_volume": 0,
            },
            {
                "short_ratio": 0.43,
                "date": "2026-03-21",
                "total_volume": 10e6,
                "short_volume": 4.3e6,
                "exempt_volume": 0,
            },
            {
                "short_ratio": 0.44,
                "date": "2026-03-20",
                "total_volume": 10e6,
                "short_volume": 4.4e6,
                "exempt_volume": 0,
            },
        ]
        # Manually call _compute_volume_signals to verify anomaly detection
        signals = self.tool._compute_volume_signals(data)
        self.assertTrue(signals["is_anomaly"])


# ──────────────────────────────────────────────────────────────────
# 13. CLI Registration
# ──────────────────────────────────────────────────────────────────


class TestCLIRegistration(unittest.TestCase):
    def test_tool_registered(self):
        try:
            from agent.cli import build_tool_registry
        except (ImportError, ModuleNotFoundError, AttributeError):
            self.skipTest("CLI dependencies not available (hmmlearn/pandas)")
        from agent.config.settings import AgentConfig

        config = AgentConfig()
        registry = build_tool_registry(config)
        tool_names = [t.name for t in registry._tools.values()] if hasattr(registry, "_tools") else []
        if not tool_names:
            # Try alternative attribute name
            tool_names = list(registry._tools.keys()) if hasattr(registry, "_tools") else []
        self.assertIn("finra_short_volume", tool_names)

    def test_total_tool_count(self):
        """Should now have 60 registered tools."""
        try:
            from agent.cli import build_tool_registry
        except (ImportError, ModuleNotFoundError, AttributeError):
            self.skipTest("CLI dependencies not available (hmmlearn/pandas)")
        from agent.config.settings import AgentConfig

        config = AgentConfig()
        registry = build_tool_registry(config)
        count = len(registry._tools) if hasattr(registry, "_tools") else 0
        self.assertEqual(count, 60)


# ──────────────────────────────────────────────────────────────────
# 14. Bandit Arm
# ──────────────────────────────────────────────────────────────────


class TestBanditArm(unittest.TestCase):
    def test_institutional_flow_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = [arm.name for arm in DEFAULT_ARMS]
        self.assertIn("institutional_flow", names)

    def test_institutional_flow_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "institutional_flow")
        self.assertIn("finra_short_volume", arm.tools)
        self.assertIn("market_data", arm.tools)
        self.assertIn("cftc", arm.tools)

    def test_arm_count(self):
        """Should now have 48 bandit arms."""
        from agent.learning.bandit import DEFAULT_ARMS

        self.assertEqual(len(DEFAULT_ARMS), 48)


# ──────────────────────────────────────────────────────────────────
# 15. Fetch with API failures → graceful ToolResult
# ──────────────────────────────────────────────────────────────────


class TestGracefulFailures(unittest.TestCase):
    def setUp(self):
        self.tool = _make_tool()

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_api_returns_none_for_ticker(self, mock_post):
        """_api_post returning None should produce empty results, not crash."""
        mock_post.return_value = _mock_response(500)
        result = self.tool.execute(mode="short_volume", ticker="AAPL", days_back=1)
        self.assertTrue(result.success)
        self.assertIn("No Reg SHO data found", result.output)

    @patch("agent.tools.finra_short_volume.httpx.post")
    def test_api_returns_none_for_scan(self, mock_post):
        mock_post.return_value = _mock_response(500)
        result = self.tool.execute(mode="short_volume", date="2026-03-24")
        self.assertTrue(result.success)
        self.assertIn("No Reg SHO data found", result.output)

    def test_unexpected_exception_caught(self):
        """Any unexpected exception during execute should be caught."""
        with patch.object(self.tool, "_fetch_reg_sho", side_effect=RuntimeError("boom")):
            result = self.tool.execute(mode="short_volume", ticker="AAPL", days_back=1)
            self.assertFalse(result.success)
            self.assertIn("Error", result.output)


if __name__ == "__main__":
    unittest.main()
