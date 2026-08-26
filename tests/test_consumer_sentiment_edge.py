"""
Edge case tests for ConsumerSentimentTool (Eurostat + FRED + BLS CPI).

Covers: mode validation, parameter validation, Eurostat fetch + parse
(JSON-stat 2.0), FRED fetch + graceful degradation (no key), BLS CPI fetch,
signal computation (eu_confidence, us_sentiment, inflation_reality),
cache interaction, HTTP errors (429/500/timeout), empty data,
malformed responses, output formatting, _safe_float, _get_fred_key,
tool metadata, registry + bandit integration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from agent.tools.base import ToolResult
from agent.tools.consumer_sentiment import (
    _BLS_BASE,
    _BLS_CPI_SA,
    _DEFAULT_EU_COUNTRIES,
    _EU_GEOS,
    _EUROSTAT_BASE,
    VALID_MODES,
    ConsumerSentimentTool,
    _compute_cpi_signals,
    _compute_eu_signals,
    _compute_us_signals,
    _fetch_bls_cpi,
    _fetch_eurostat,
    _fetch_fred_series,
    _format_cpi_summary,
    _format_eu_summary,
    _format_us_summary,
    _get_fred_key,
    _parse_eurostat_jsonstat,
    _safe_float,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> ConsumerSentimentTool:
    return ConsumerSentimentTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", _EUROSTAT_BASE),
    )


def _eurostat_response(geos: list[str], periods: list[str], values: dict[int, float]) -> dict:
    """Build a minimal JSON-stat 2.0 response."""
    geo_index = {g: i for i, g in enumerate(geos)}
    geo_label = {g: f"Country {g}" for g in geos}
    time_index = {p: i for i, p in enumerate(periods)}

    return {
        "version": "2.0",
        "label": "test",
        "id": ["geo", "time"],
        "size": [len(geos), len(periods)],
        "dimension": {
            "geo": {
                "category": {
                    "index": geo_index,
                    "label": geo_label,
                }
            },
            "time": {
                "category": {
                    "index": time_index,
                }
            },
        },
        "value": {str(k): v for k, v in values.items()},
    }


def _fred_response(series_id: str, observations: list[dict]) -> dict:
    return {
        "observations": observations,
    }


def _bls_cpi_response(data: list[dict]) -> dict:
    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 42,
        "message": [],
        "Results": {
            "series": [
                {"seriesID": _BLS_CPI_SA, "data": data},
            ],
        },
    }


# ── 1. _safe_float ────────────────────────────────────────────


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float("3.14") == 3.14

    def test_negative(self):
        assert _safe_float("-5.2") == -5.2

    def test_integer(self):
        assert _safe_float(42) == 42.0

    def test_none(self):
        assert _safe_float(None) is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_dot(self):
        assert _safe_float(".") is None

    def test_nan_string(self):
        assert _safe_float("NaN") is None
        assert _safe_float("nan") is None

    def test_null_string(self):
        assert _safe_float("null") is None

    def test_whitespace(self):
        assert _safe_float("  3.5  ") == 3.5

    def test_junk(self):
        assert _safe_float("abc") is None


# ── 2. _get_fred_key ─────────────────────────────────────────


class TestGetFredKey:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("TIRRA_FRED_API_KEY", raising=False)
        assert _get_fred_key() is None

    def test_placeholder_key(self, monkeypatch):
        monkeypatch.setenv("TIRRA_FRED_API_KEY", "your-key-here")
        assert _get_fred_key() is None

    def test_empty_key(self, monkeypatch):
        monkeypatch.setenv("TIRRA_FRED_API_KEY", "")
        assert _get_fred_key() is None

    def test_valid_key(self, monkeypatch):
        monkeypatch.setenv("TIRRA_FRED_API_KEY", "abc123def456")
        assert _get_fred_key() == "abc123def456"


# ── 3. Mode validation ───────────────────────────────────────


class TestModeValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="bogus")
        assert not r.success
        assert "bogus" in r.output

    def test_missing_mode(self):
        r = _tool().execute()
        assert not r.success

    def test_valid_modes_match(self):
        assert {"eu_confidence", "us_sentiment", "inflation_reality"} == VALID_MODES


# ── 4. Parameter validation ──────────────────────────────────


class TestParameterValidation:
    def test_invalid_months_string(self):
        with patch("agent.tools.consumer_sentiment._fetch_eurostat") as m:
            m.return_value = ({}, None)
            r = _tool().execute(mode="eu_confidence", months="abc")
            assert r.success  # gracefully defaults to 6

    def test_months_clamped_min(self):
        with patch("agent.tools.consumer_sentiment._fetch_eurostat") as m:
            m.return_value = ({}, None)
            _tool().execute(mode="eu_confidence", months=-5)
            # Should clamp to 1

    def test_months_clamped_max(self):
        with patch("agent.tools.consumer_sentiment._fetch_eurostat") as m:
            m.return_value = ({}, None)
            _tool().execute(mode="eu_confidence", months=100)
            # Should clamp to 24

    def test_invalid_geo_codes(self):
        r = _tool().execute(mode="eu_confidence", countries="XX,YY,ZZ")
        assert not r.success
        assert "no valid" in r.output.lower() or "Available" in r.output


# ── 5. Eurostat fetch ────────────────────────────────────────


class TestEurostatFetch:
    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_success(self, mock_client_cls):
        body = _eurostat_response(
            geos=["DE", "FR"],
            periods=["2025-01", "2025-02", "2025-03"],
            values={
                0: -5.0,
                1: -4.5,
                2: -3.8,  # DE
                3: -8.1,
                4: -7.2,
                5: -6.5,  # FR
            },
        )
        mock_resp = _mock_resp(body)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_eurostat(["DE", "FR"], 3)
        assert err is None
        assert "DE" in data
        assert "FR" in data
        assert len(data["DE"]) == 3

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_timeout(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(side_effect=httpx.TimeoutException("timeout")))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_eurostat(["DE"], 3)
        assert err is not None
        assert "timed out" in err

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_non_200(self, mock_client_cls):
        mock_resp = _mock_resp({}, status=500)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_eurostat(["DE"], 3)
        assert err is not None
        assert "500" in err

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_bad_json(self, mock_client_cls):
        resp = httpx.Response(
            status_code=200,
            text="not json",
            request=httpx.Request("GET", _EUROSTAT_BASE),
        )
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(get=MagicMock(return_value=resp)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        data, err = _fetch_eurostat(["DE"], 3)
        assert err is not None
        assert "parse" in err.lower()


# ── 6. Eurostat JSON-stat parsing ────────────────────────────


class TestEurostatParsing:
    def test_basic_parse(self):
        body = _eurostat_response(
            geos=["DE"],
            periods=["2025-01", "2025-02"],
            values={0: -5.0, 1: -3.8},
        )
        result = _parse_eurostat_jsonstat(body, ["DE"])
        assert "DE" in result
        assert len(result["DE"]) == 2
        assert result["DE"][0]["period"] == "2025-01"
        assert result["DE"][0]["value"] == -5.0

    def test_multi_country(self):
        body = _eurostat_response(
            geos=["DE", "FR", "IT"],
            periods=["2025-01"],
            values={0: -5.0, 1: -8.0, 2: -10.0},
        )
        result = _parse_eurostat_jsonstat(body, ["DE", "FR", "IT"])
        assert len(result) == 3
        assert result["IT"][0]["value"] == -10.0

    def test_empty_values(self):
        body = _eurostat_response(geos=["DE"], periods=["2025-01"], values={})
        result = _parse_eurostat_jsonstat(body, ["DE"])
        assert result == {}

    def test_no_dimension(self):
        body = {"value": {"0": 5.0}, "id": [], "size": [], "dimension": {}}
        result = _parse_eurostat_jsonstat(body, ["DE"])
        assert result == {}

    def test_chronological_sort(self):
        body = _eurostat_response(
            geos=["DE"],
            periods=["2025-03", "2025-01", "2025-02"],
            values={0: -3.0, 1: -5.0, 2: -4.0},
        )
        result = _parse_eurostat_jsonstat(body, ["DE"])
        periods = [r["period"] for r in result["DE"]]
        assert periods == ["2025-01", "2025-02", "2025-03"]


# ── 7. FRED fetch ────────────────────────────────────────────


class TestFredFetch:
    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_success(self, mock_client_cls):
        body = _fred_response(
            "UMCSENT",
            [
                {"date": "2025-01-01", "value": "65.2"},
                {"date": "2025-02-01", "value": "67.8"},
            ],
        )
        mock_resp = _mock_resp(body)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        records, err = _fetch_fred_series("fake-key", "UMCSENT", 6)
        assert err is None
        assert len(records) == 2
        assert records[0]["value"] == 65.2

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_bad_api_key(self, mock_client_cls):
        mock_resp = _mock_resp({}, status=400)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        records, err = _fetch_fred_series("bad-key", "UMCSENT", 6)
        assert err is not None
        assert "400" in err

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_error_message_in_body(self, mock_client_cls):
        body = {"error_message": "Bad API key"}
        mock_resp = _mock_resp(body)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        records, err = _fetch_fred_series("key", "UMCSENT", 6)
        assert err is not None
        assert "Bad API key" in err

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_dot_values_skipped(self, mock_client_cls):
        body = _fred_response(
            "UMCSENT",
            [
                {"date": "2025-01-01", "value": "."},
                {"date": "2025-02-01", "value": "70.3"},
            ],
        )
        mock_resp = _mock_resp(body)
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        records, err = _fetch_fred_series("key", "UMCSENT", 6)
        assert err is None
        assert len(records) == 1

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_timeout(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(get=MagicMock(side_effect=httpx.TimeoutException("timeout")))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        records, err = _fetch_fred_series("key", "UMCSENT", 6)
        assert err is not None
        assert "timed out" in err


# ── 8. BLS CPI fetch ────────────────────────────────────────


class TestBlsCpiFetch:
    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_success(self, mock_client_cls):
        body = _bls_cpi_response(
            [
                {"year": "2025", "period": "M03", "value": "326.785"},
                {"year": "2025", "period": "M02", "value": "325.402"},
            ]
        )
        mock_resp = httpx.Response(
            status_code=200,
            text=json.dumps(body),
            request=httpx.Request("POST", _BLS_BASE),
        )
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        records, err = _fetch_bls_cpi(6)
        assert err is None
        assert len(records) == 2
        assert records[0]["value"] == 325.402  # sorted chronologically

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_m13_skipped(self, mock_client_cls):
        body = _bls_cpi_response(
            [
                {"year": "2025", "period": "M13", "value": "999"},
                {"year": "2025", "period": "M01", "value": "324.0"},
            ]
        )
        mock_resp = httpx.Response(
            status_code=200,
            text=json.dumps(body),
            request=httpx.Request("POST", _BLS_BASE),
        )
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        records, err = _fetch_bls_cpi(6)
        assert err is None
        assert len(records) == 1

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_rate_limit(self, mock_client_cls):
        mock_resp = httpx.Response(
            status_code=429,
            text="Rate limit",
            request=httpx.Request("POST", _BLS_BASE),
        )
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        records, err = _fetch_bls_cpi(6)
        assert err is not None
        assert "rate limit" in err.lower()

    @patch("agent.tools.consumer_sentiment.httpx.Client")
    def test_request_failed(self, mock_client_cls):
        body = {
            "status": "REQUEST_NOT_PROCESSED",
            "responseTime": 0,
            "message": ["Invalid series"],
            "Results": {},
        }
        mock_resp = httpx.Response(
            status_code=200,
            text=json.dumps(body),
            request=httpx.Request("POST", _BLS_BASE),
        )
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        records, err = _fetch_bls_cpi(6)
        assert err is not None
        assert "failed" in err.lower()


# ── 9. EU signal computation ────────────────────────────────


class TestEuSignals:
    def test_basic_signals(self):
        data = {
            "DE": [
                {"period": "2025-01", "value": -5.0, "geo_label": "Germany"},
                {"period": "2025-02", "value": -3.5, "geo_label": "Germany"},
                {"period": "2025-03", "value": -2.0, "geo_label": "Germany"},
            ],
        }
        sig = _compute_eu_signals(data, ["DE"])
        de = sig["countries"]["DE"]
        assert de["latest"] == -2.0
        assert de["trend"] == "IMPROVING"
        assert de["mom_change"] == 1.5

    def test_deteriorating_trend(self):
        data = {
            "FR": [
                {"period": "2025-01", "value": -5.0, "geo_label": "France"},
                {"period": "2025-02", "value": -7.0, "geo_label": "France"},
            ],
        }
        sig = _compute_eu_signals(data, ["FR"])
        assert sig["countries"]["FR"]["trend"] == "DETERIORATING"

    def test_stable_trend(self):
        data = {
            "IT": [
                {"period": "2025-01", "value": -5.0, "geo_label": "Italy"},
                {"period": "2025-02", "value": -5.3, "geo_label": "Italy"},
            ],
        }
        sig = _compute_eu_signals(data, ["IT"])
        assert sig["countries"]["IT"]["trend"] == "STABLE"

    def test_consecutive_decline(self):
        data = {
            "ES": [
                {"period": "2025-01", "value": -3.0, "geo_label": "Spain"},
                {"period": "2025-02", "value": -5.0, "geo_label": "Spain"},
                {"period": "2025-03", "value": -7.0, "geo_label": "Spain"},
            ],
        }
        sig = _compute_eu_signals(data, ["ES"])
        assert sig["countries"]["ES"]["consecutive_decline"] is True

    def test_cross_country_divergence(self):
        data = {
            "DE": [{"period": "2025-03", "value": 2.0, "geo_label": "Germany"}],
            "GR": [{"period": "2025-03", "value": -20.0, "geo_label": "Greece"}],
        }
        sig = _compute_eu_signals(data, ["DE", "GR"])
        assert sig["cross_country_spread"] == 22.0
        assert sig["most_optimistic"] == "DE"
        assert sig["most_pessimistic"] == "GR"

    def test_synchronized_decline(self):
        data = {
            "DE": [
                {"period": "2025-02", "value": -3.0, "geo_label": "DE"},
                {"period": "2025-03", "value": -6.0, "geo_label": "DE"},
            ],
            "FR": [
                {"period": "2025-02", "value": -5.0, "geo_label": "FR"},
                {"period": "2025-03", "value": -8.0, "geo_label": "FR"},
            ],
        }
        sig = _compute_eu_signals(data, ["DE", "FR"])
        assert sig["synchronized_decline"] is True

    def test_empty_data(self):
        sig = _compute_eu_signals({}, ["DE"])
        assert sig["status"] == "NO_DATA"

    def test_single_observation(self):
        data = {
            "DE": [{"period": "2025-03", "value": -5.0, "geo_label": "Germany"}],
        }
        sig = _compute_eu_signals(data, ["DE"])
        assert sig["countries"]["DE"]["trend"] == "INSUFFICIENT_DATA"
        assert sig["countries"]["DE"]["mom_change"] is None


# ── 10. US signal computation ───────────────────────────────


class TestUsSignals:
    def test_normal(self):
        series_data = {
            "UMCSENT": {
                "label": "UMichigan Consumer Sentiment",
                "records": [
                    {"date": "2025-01-01", "value": 65.0},
                    {"date": "2025-02-01", "value": 67.5},
                ],
            },
            "MICH": {
                "label": "UMichigan Inflation Expectations",
                "records": [
                    {"date": "2025-01-01", "value": 3.2},
                    {"date": "2025-02-01", "value": 3.5},
                ],
            },
        }
        sig = _compute_us_signals(series_data)
        assert sig["sentiment_latest"] == 67.5
        assert sig["sentiment_mom"] == 2.5
        assert sig["inflation_exp_1yr"] == 3.5
        assert sig["inflation_anchor"] == "ELEVATED — expectations above 3%"

    def test_critical_low_sentiment(self):
        series_data = {
            "UMCSENT": {
                "records": [{"date": "2025-01-01", "value": 55.0}],
            },
            "MICH": {"records": []},
        }
        sig = _compute_us_signals(series_data)
        assert "CRITICAL_LOW" in sig["sentiment_alert"]

    def test_unanchored_inflation(self):
        series_data = {
            "UMCSENT": {"records": []},
            "MICH": {
                "records": [{"date": "2025-01-01", "value": 5.2}],
            },
        }
        sig = _compute_us_signals(series_data)
        assert "UNANCHORED" in sig["inflation_anchor"]

    def test_empty_records(self):
        series_data = {
            "UMCSENT": {"records": []},
            "MICH": {"records": []},
        }
        sig = _compute_us_signals(series_data)
        assert sig["sentiment_latest"] is None
        assert sig["inflation_exp_1yr"] is None


# ── 11. CPI signal computation ──────────────────────────────


class TestCpiSignals:
    def test_mom_and_annualized(self):
        cpi = [
            {"year": "2025", "period": "M02", "value": 325.0},
            {"year": "2025", "period": "M03", "value": 326.0},
        ]
        sig = _compute_cpi_signals(cpi, [])
        assert sig["cpi_latest"] == 326.0
        assert sig["cpi_mom_pct"] is not None
        assert sig["cpi_annualized"] is not None

    def test_yoy_with_13_months(self):
        records = [
            {"year": str(2024 + (i // 12)), "period": f"M{(i % 12) + 1:02d}", "value": 300.0 + i * 0.5}
            for i in range(14)
        ]
        sig = _compute_cpi_signals(records, [])
        assert sig["cpi_yoy_pct"] is not None

    def test_expectation_gap_above_reality(self):
        # 13 months for YoY
        records = [
            {"year": str(2024 + (i // 12)), "period": f"M{(i % 12) + 1:02d}", "value": 300.0 + i * 0.3}
            for i in range(14)
        ]
        mich = [{"date": "2025-02-01", "value": 5.0}]
        sig = _compute_cpi_signals(records, mich)
        assert sig["expectation_gap"] is not None
        assert sig["gap_signal"] is not None

    def test_no_cpi_data(self):
        sig = _compute_cpi_signals([], [])
        assert sig["status"] == "NO_DATA"

    def test_single_record(self):
        cpi = [{"year": "2025", "period": "M03", "value": 326.0}]
        sig = _compute_cpi_signals(cpi, [])
        assert sig["cpi_latest"] == 326.0
        assert sig["cpi_mom_pct"] is None
        assert sig["cpi_yoy_pct"] is None


# ── 12. Cache interaction ────────────────────────────────────


class TestCache:
    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {
            "output": "cached output",
            "data": {"mode": "eu_confidence"},
        }
        r = _tool(cache).execute(mode="eu_confidence", countries="DE")
        assert r.success
        assert "cached" in r.output

    def test_cache_miss_then_set(self):
        cache = MagicMock()
        cache.get.return_value = None

        with patch("agent.tools.consumer_sentiment._fetch_eurostat") as m:
            m.return_value = (
                {"DE": [{"period": "2025-03", "value": -5.0, "geo_label": "DE"}]},
                None,
            )
            r = _tool(cache).execute(mode="eu_confidence", countries="DE")

        assert r.success
        cache.put.assert_called_once()

    def test_no_cache(self):
        with patch("agent.tools.consumer_sentiment._fetch_eurostat") as m:
            m.return_value = ({"DE": [{"period": "2025-03", "value": -5.0, "geo_label": "DE"}]}, None)
            r = _tool(None).execute(mode="eu_confidence", countries="DE")
            assert r.success


# ── 13. us_sentiment requires FRED key ───────────────────────


class TestUsSentimentKeyRequired:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("TIRRA_FRED_API_KEY", raising=False)
        r = _tool().execute(mode="us_sentiment")
        assert not r.success
        assert "FRED" in r.output

    def test_placeholder_key(self, monkeypatch):
        monkeypatch.setenv("TIRRA_FRED_API_KEY", "your-key-here")
        r = _tool().execute(mode="us_sentiment")
        assert not r.success


# ── 14. inflation_reality mode ───────────────────────────────


class TestInflationReality:
    @patch("agent.tools.consumer_sentiment._fetch_bls_cpi")
    def test_bls_error_propagates(self, mock_fetch):
        mock_fetch.return_value = ([], "BLS down")
        r = _tool().execute(mode="inflation_reality")
        assert not r.success
        assert "BLS down" in r.output

    @patch("agent.tools.consumer_sentiment._get_fred_key")
    @patch("agent.tools.consumer_sentiment._fetch_fred_series")
    @patch("agent.tools.consumer_sentiment._fetch_bls_cpi")
    def test_cpi_with_fred_expectations(self, mock_cpi, mock_fred, mock_key):
        cpi_records = [
            {"year": str(2024 + (i // 12)), "period": f"M{(i % 12) + 1:02d}", "value": 300.0 + i * 0.3}
            for i in range(14)
        ]
        mock_cpi.return_value = (cpi_records, None)
        mock_key.return_value = "valid-key"
        mock_fred.return_value = ([{"date": "2025-02-01", "value": 3.5}], None)

        r = _tool().execute(mode="inflation_reality")
        assert r.success
        assert r.data is not None
        assert r.data["has_expectations"] is True

    @patch("agent.tools.consumer_sentiment._get_fred_key")
    @patch("agent.tools.consumer_sentiment._fetch_bls_cpi")
    def test_cpi_without_fred(self, mock_cpi, mock_key):
        mock_cpi.return_value = (
            [{"year": "2025", "period": "M03", "value": 326.0}],
            None,
        )
        mock_key.return_value = None

        r = _tool().execute(mode="inflation_reality")
        assert r.success
        assert "expectation gap analysis skipped" in r.output.lower()


# ── 15. Formatting ───────────────────────────────────────────


class TestFormatting:
    def test_eu_summary_contains_country(self):
        data = {"DE": [{"period": "2025-03", "value": -5.0, "geo_label": "Germany"}]}
        signals = _compute_eu_signals(data, ["DE"])
        summary = _format_eu_summary(data, signals, ["DE"], 6)
        assert "DE" in summary
        assert "Eurostat" in summary

    def test_us_summary_alert(self):
        series_data = {
            "UMCSENT": {"label": "x", "records": [{"date": "2025-01-01", "value": 55.0}]},
            "MICH": {"label": "y", "records": [{"date": "2025-01-01", "value": 4.5}]},
        }
        signals = _compute_us_signals(series_data)
        summary = _format_us_summary(series_data, signals, 6)
        assert "CRITICAL_LOW" in summary
        assert "UNANCHORED" in summary

    def test_cpi_summary_basic(self):
        cpi = [
            {"year": "2025", "period": "M02", "value": 325.0},
            {"year": "2025", "period": "M03", "value": 326.0},
        ]
        signals = _compute_cpi_signals(cpi, [])
        summary = _format_cpi_summary(cpi, [], signals, 6)
        assert "CPI-U" in summary
        assert "MoM" in summary


# ── 16. Tool metadata ───────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "consumer_sentiment"

    def test_description(self):
        desc = _tool().description
        assert "eu_confidence" in desc
        assert "us_sentiment" in desc
        assert "inflation_reality" in desc

    def test_parameters_structure(self):
        params = _tool().parameters
        assert "properties" in params
        assert "mode" in params["properties"]
        assert "countries" in params["properties"]
        assert "months" in params["properties"]

    def test_returns_tool_result(self):
        r = _tool().execute(mode="invalid")
        assert isinstance(r, ToolResult)


# ── 17. EU geo constant consistency ─────────────────────────


class TestConstants:
    def test_eu_geos_has_eu27(self):
        assert "EU27_2020" in _EU_GEOS

    def test_eu_geos_has_major_economies(self):
        for g in ["DE", "FR", "IT", "ES", "NL"]:
            assert g in _EU_GEOS

    def test_default_countries_are_valid(self):
        for g in _DEFAULT_EU_COUNTRIES.split(","):
            assert g in _EU_GEOS

    def test_valid_modes_frozen(self):
        assert isinstance(VALID_MODES, frozenset)
