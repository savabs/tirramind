"""
Edge case tests for WeatherAlertsTool (NOAA NWS + NASA FIRMS).

Covers: mode validation, severity validation, state validation, parameter
clamping, alert formatting, fire CSV parsing, infrastructure zone matching,
severity ranking, cache interaction, HTTP error handling, timeout handling,
market-relevant filtering, summary mode aggregation, output formatting,
tool metadata, empty data, malformed data, registry + bandit integration.
"""

from __future__ import annotations

import csv
import io
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.weather_alerts import (
    WeatherAlertsTool,
    INFRA_ZONES,
    _MARKET_EVENTS,
    _SEVERITIES,
    _US_STATES,
    _format_alert,
    _parse_fires_csv,
    _point_in_zone,
    _severity_rank,
)
from agent.tools.base import ToolResult


# ── Fixtures ──────────────────────────────────────────────────


def _make_nws_feature(
    event: str = "Tornado Warning",
    severity: str = "Extreme",
    urgency: str = "Immediate",
    certainty: str = "Observed",
    area: str = "Oklahoma County, OK",
    headline: str = "Tornado Warning for Oklahoma County",
    onset: str = "2025-07-01T18:00:00Z",
    expires: str = "2025-07-01T19:00:00Z",
    sender: str = "NWS Norman OK",
    category: str = "Met",
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {
            "event": event,
            "severity": severity,
            "urgency": urgency,
            "certainty": certainty,
            "areaDesc": area,
            "headline": headline,
            "onset": onset,
            "expires": expires,
            "senderName": sender,
            "category": category,
        },
    }


def _make_nws_response(features: list[dict] | None = None) -> dict[str, Any]:
    if features is None:
        features = [
            _make_nws_feature(),
            _make_nws_feature(
                event="Flood Warning",
                severity="Severe",
                area="Harris County, TX",
                headline="Flood Warning for Houston area",
            ),
        ]
    return {"type": "FeatureCollection", "features": features}


def _make_fires_csv(rows: list[dict] | None = None) -> str:
    if rows is None:
        rows = [
            {
                "latitude": "31.9",
                "longitude": "-102.1",
                "brightness": "350.0",
                "confidence": "85",
                "frp": "45.2",
                "acq_date": "2025-07-01",
                "acq_time": "1430",
                "daynight": "D",
                "satellite": "Terra",
                "scan": "1.0",
                "track": "1.0",
                "bright_t31": "310.0",
                "version": "6.1NRT",
            },
            {
                "latitude": "29.8",
                "longitude": "-93.9",
                "brightness": "320.0",
                "confidence": "90",
                "frp": "30.1",
                "acq_date": "2025-07-01",
                "acq_time": "1500",
                "daynight": "D",
                "satellite": "Aqua",
                "scan": "1.0",
                "track": "1.0",
                "bright_t31": "305.0",
                "version": "6.1NRT",
            },
            {
                "latitude": "0.0",
                "longitude": "0.0",
                "brightness": "300.0",
                "confidence": "40",
                "frp": "10.0",
                "acq_date": "2025-07-01",
                "acq_time": "1200",
                "daynight": "D",
                "satellite": "Terra",
                "scan": "1.0",
                "track": "1.0",
                "bright_t31": "290.0",
                "version": "6.1NRT",
            },
        ]
    header = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,confidence,version,bright_t31,frp,daynight\n"
    lines = header
    for r in rows:
        lines += (
            f"{r['latitude']},{r['longitude']},{r['brightness']},{r.get('scan','1.0')},"
            f"{r.get('track','1.0')},{r['acq_date']},{r['acq_time']},{r['satellite']},"
            f"{r['confidence']},{r.get('version','6.1NRT')},{r.get('bright_t31','300.0')},"
            f"{r['frp']},{r['daynight']}\n"
        )
    return lines


def _tool(cache=None) -> WeatherAlertsTool:
    return WeatherAlertsTool(cache=cache)


def _mock_resp(body: Any, status: int = 200, is_json: bool = True) -> httpx.Response:
    if is_json:
        import json

        text = json.dumps(body)
    else:
        text = body
    return httpx.Response(
        status_code=status, text=text, request=httpx.Request("GET", "http://test")
    )


# ── 1. Tool Metadata ─────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "weather_alerts"

    def test_description_nonempty(self):
        assert len(_tool().description) > 50

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        assert "mode" in params["properties"]
        assert "severity" in params["properties"]
        assert "state" in params["properties"]
        assert "market_only" in params["properties"]
        assert "min_confidence" in params["properties"]
        assert "limit" in params["properties"]

    def test_mode_enum(self):
        modes = _tool().parameters["properties"]["mode"]["enum"]
        assert set(modes) == {"alerts", "fires", "summary"}


# ── 2. Input Validation ──────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_invalid_state(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            r = _tool().execute(mode="alerts", state="ZZ")
        assert not r.success
        assert "Unknown state" in r.output

    def test_valid_state_uppercase(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="alerts", state="tx")
        assert r.success  # "tx" → "TX"

    def test_limit_clamped_low(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = ([], None)
            # Should not crash with limit=0
            r = _tool().execute(mode="alerts", limit=0)
        assert r.success

    def test_limit_clamped_high(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = ([_make_nws_feature()] * 300, None)
            r = _tool().execute(mode="alerts", limit=999)
        # Should be capped to 200
        assert r.success
        assert len(r.data["alerts"]) <= 200

    def test_min_confidence_clamped(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_firms") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="fires", min_confidence=-10)
        assert r.success

    def test_min_confidence_clamped_high(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_firms") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="fires", min_confidence=200)
        assert r.success

    def test_invalid_severity_defaults(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="alerts", severity="SuperBad")
        assert r.success  # invalid severity → defaults to "Severe"

    def test_extra_kwargs_ignored(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="alerts", bogus="thing")
        assert r.success


# ── 3. Helper Functions ──────────────────────────────────────


class TestSeverityRank:
    def test_extreme(self):
        assert _severity_rank("Extreme") == 0

    def test_severe(self):
        assert _severity_rank("Severe") == 1

    def test_moderate(self):
        assert _severity_rank("Moderate") == 2

    def test_minor(self):
        assert _severity_rank("Minor") == 3

    def test_unknown(self):
        assert _severity_rank("Unknown") == 4

    def test_garbage(self):
        assert _severity_rank("banana") == 4


class TestFormatAlert:
    def test_basic_format(self):
        props = {
            "event": "Tornado Warning",
            "severity": "Extreme",
            "urgency": "Immediate",
            "certainty": "Observed",
            "areaDesc": "Oklahoma County",
            "headline": "Big tornado coming",
            "onset": "2025-07-01T18:00:00Z",
            "expires": "2025-07-01T19:00:00Z",
            "senderName": "NWS Norman",
            "category": "Met",
        }
        result = _format_alert(props)
        assert result["event"] == "Tornado Warning"
        assert result["severity"] == "Extreme"
        assert result["market_relevant"] is True

    def test_non_market_event(self):
        props = {"event": "Wind Advisory", "severity": "Minor"}
        result = _format_alert(props)
        assert result["market_relevant"] is False

    def test_missing_fields(self):
        result = _format_alert({})
        assert result["event"] == ""
        assert result["severity"] == "Unknown"
        assert result["headline"] == ""
        assert result["market_relevant"] is False

    def test_headline_truncation(self):
        props = {"headline": "x" * 500}
        result = _format_alert(props)
        assert len(result["headline"]) == 200

    def test_area_truncation(self):
        props = {"areaDesc": "y" * 500}
        result = _format_alert(props)
        assert len(result["area"]) == 200

    def test_none_headline(self):
        props = {"headline": None}
        result = _format_alert(props)
        assert result["headline"] == ""

    def test_none_area(self):
        props = {"areaDesc": None}
        result = _format_alert(props)
        assert result["area"] == ""


class TestPointInZone:
    def test_inside(self):
        zone = {"lat": 31.9, "lon": -102.1, "radius": 2.0}
        assert _point_in_zone(31.9, -102.1, zone) is True  # exact center

    def test_edge(self):
        zone = {"lat": 31.9, "lon": -102.1, "radius": 2.0}
        assert _point_in_zone(33.9, -102.1, zone) is True  # exactly at edge

    def test_outside(self):
        zone = {"lat": 31.9, "lon": -102.1, "radius": 2.0}
        assert _point_in_zone(40.0, -102.1, zone) is False

    def test_negative_coords(self):
        zone = {"lat": -24.0, "lon": -69.0, "radius": 1.0}
        assert _point_in_zone(-24.5, -69.5, zone) is True

    def test_all_infra_zones_valid(self):
        for zone in INFRA_ZONES:
            assert "lat" in zone
            assert "lon" in zone
            assert "radius" in zone
            assert "name" in zone
            assert "sector" in zone
            assert zone["radius"] > 0


class TestParseFiresCsv:
    def test_basic_parse(self):
        csv_text = _make_fires_csv()
        fires = _parse_fires_csv(csv_text)
        assert len(fires) == 3
        assert fires[0]["lat"] == 31.9
        assert fires[0]["lon"] == -102.1
        assert fires[0]["brightness"] == 350.0
        assert fires[0]["confidence"] == 85

    def test_empty_csv(self):
        csv_text = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,confidence,version,bright_t31,frp,daynight\n"
        fires = _parse_fires_csv(csv_text)
        assert fires == []

    def test_malformed_rows_skipped(self):
        csv_text = (
            "latitude,longitude,brightness,confidence,frp,acq_date,acq_time,daynight,satellite\n"
            "abc,def,ghi,jkl,mno,2025-01-01,1200,D,Terra\n"
            "31.9,-102.1,350.0,85,45.2,2025-01-01,1430,D,Terra\n"
        )
        fires = _parse_fires_csv(csv_text)
        assert len(fires) == 1

    def test_missing_fields_defaults(self):
        csv_text = "latitude,longitude\n" "31.9,-102.1\n"
        fires = _parse_fires_csv(csv_text)
        # Missing brightness/confidence/frp fields default to 0 via float("")/int("") → caught
        # Actually _parse_fires_csv tries float(row.get("brightness", "0")) which defaults "0"
        # so the row IS parsed successfully with brightness=0, confidence=0, frp=0
        assert len(fires) == 1
        assert fires[0]["brightness"] == 0.0
        assert fires[0]["confidence"] == 0

    def test_completely_empty_string(self):
        fires = _parse_fires_csv("")
        assert fires == []


# ── 4. Alerts Mode ────────────────────────────────────────────


class TestAlertsMode:
    def test_basic_alerts(self):
        features = [
            _make_nws_feature(),
            _make_nws_feature(event="Flood Warning", severity="Severe"),
        ]
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = (features, None)
            r = _tool().execute(mode="alerts")
        assert r.success
        assert r.data["count"] == 2

    def test_market_only_filter(self):
        features = [
            _make_nws_feature(event="Tornado Warning"),
            _make_nws_feature(
                event="Wind Advisory", severity="Minor"
            ),  # not market relevant
        ]
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = (features, None)
            r = _tool().execute(mode="alerts", market_only=True)
        assert r.success
        assert r.data["count"] == 1

    def test_empty_alerts(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="alerts")
        assert r.success
        assert r.data["count"] == 0
        assert "No active alerts" in r.output

    def test_alerts_sorted_by_severity(self):
        features = [
            _make_nws_feature(event="Minor thing", severity="Minor"),
            _make_nws_feature(event="Tornado Warning", severity="Extreme"),
        ]
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = (features, None)
            r = _tool().execute(mode="alerts")
        assert r.success
        assert r.data["alerts"][0]["severity"] == "Extreme"

    def test_alerts_fetch_error(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = ([], "NWS API: Rate limited.")
            r = _tool().execute(mode="alerts")
        assert not r.success
        assert "Rate limited" in r.output

    def test_state_filter_shown_in_output(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="alerts", state="TX")
        assert "TX" in r.output

    def test_limit_respected(self):
        features = [_make_nws_feature()] * 50
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_nws") as m:
            m.return_value = (features, None)
            r = _tool().execute(mode="alerts", limit=5)
        assert len(r.data["alerts"]) == 5


# ── 5. Fires Mode ────────────────────────────────────────────


class TestFiresMode:
    def test_fires_near_infra(self):
        fires = [
            {
                "lat": 31.9,
                "lon": -102.1,
                "brightness": 350.0,
                "confidence": 85,
                "frp": 45.2,
                "acq_date": "2025-07-01",
                "acq_time": "1430",
                "daynight": "D",
                "satellite": "Terra",
            },
        ]
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_firms") as m:
            m.return_value = (fires, None)
            r = _tool().execute(mode="fires", min_confidence=70)
        assert r.success
        assert r.data["count"] >= 1
        assert "Permian Basin" in r.output

    def test_fires_low_confidence_filtered(self):
        fires = [
            {
                "lat": 31.9,
                "lon": -102.1,
                "brightness": 300.0,
                "confidence": 30,
                "frp": 10.0,
                "acq_date": "2025-07-01",
                "acq_time": "1200",
                "daynight": "D",
                "satellite": "Terra",
            },
        ]
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_firms") as m:
            m.return_value = (fires, None)
            r = _tool().execute(mode="fires", min_confidence=70)
        assert r.success
        assert r.data["count"] == 0

    def test_fires_not_near_infra(self):
        fires = [
            {
                "lat": 0.0,
                "lon": 0.0,
                "brightness": 400.0,
                "confidence": 95,
                "frp": 100.0,
                "acq_date": "2025-07-01",
                "acq_time": "1200",
                "daynight": "D",
                "satellite": "Terra",
            },
        ]
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_firms") as m:
            m.return_value = (fires, None)
            r = _tool().execute(mode="fires", min_confidence=70)
        assert r.success
        assert r.data["count"] == 0
        assert "No high-confidence fires" in r.output

    def test_fires_empty(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_firms") as m:
            m.return_value = ([], None)
            r = _tool().execute(mode="fires")
        assert r.success
        assert r.data["count"] == 0

    def test_fires_fetch_error(self):
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_firms") as m:
            m.return_value = ([], "NASA FIRMS: Request timed out.")
            r = _tool().execute(mode="fires")
        assert not r.success

    def test_fires_limit(self):
        fires = [
            {
                "lat": 31.9 + i * 0.01,
                "lon": -102.1,
                "brightness": 350.0 - i,
                "confidence": 90,
                "frp": 40.0,
                "acq_date": "2025-07-01",
                "acq_time": "1200",
                "daynight": "D",
                "satellite": "Terra",
            }
            for i in range(20)
        ]
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_firms") as m:
            m.return_value = (fires, None)
            r = _tool().execute(mode="fires", limit=3)
        assert r.success
        assert r.data["count"] <= 3

    def test_fires_sorted_by_brightness(self):
        fires = [
            {
                "lat": 31.9,
                "lon": -102.1,
                "brightness": 300.0,
                "confidence": 90,
                "frp": 20.0,
                "acq_date": "2025-07-01",
                "acq_time": "1200",
                "daynight": "D",
                "satellite": "Terra",
            },
            {
                "lat": 31.95,
                "lon": -102.05,
                "brightness": 400.0,
                "confidence": 95,
                "frp": 60.0,
                "acq_date": "2025-07-01",
                "acq_time": "1300",
                "daynight": "D",
                "satellite": "Aqua",
            },
        ]
        with patch("agent.tools.weather_alerts.WeatherAlertsTool._fetch_firms") as m:
            m.return_value = (fires, None)
            r = _tool().execute(mode="fires")
        assert r.success
        if r.data["fires"]:
            assert r.data["fires"][0]["brightness"] >= r.data["fires"][-1]["brightness"]


# ── 6. Summary Mode ──────────────────────────────────────────


class TestSummaryMode:
    def test_summary_basic(self):
        features = [_make_nws_feature()]
        fires = [
            {
                "lat": 31.9,
                "lon": -102.1,
                "brightness": 350.0,
                "confidence": 85,
                "frp": 45.0,
                "acq_date": "2025-07-01",
                "acq_time": "1430",
                "daynight": "D",
                "satellite": "Terra",
            },
        ]
        with patch.object(
            WeatherAlertsTool, "_fetch_nws", return_value=(features, None)
        ):
            with patch.object(
                WeatherAlertsTool, "_fetch_firms", return_value=(fires, None)
            ):
                r = _tool().execute(mode="summary")
        assert r.success
        assert r.data["alert_count"] == 1
        assert r.data["fire_count_global"] == 1

    def test_summary_nws_error(self):
        fires = []
        with patch.object(
            WeatherAlertsTool, "_fetch_nws", return_value=([], "NWS API error")
        ):
            with patch.object(
                WeatherAlertsTool, "_fetch_firms", return_value=(fires, None)
            ):
                r = _tool().execute(mode="summary")
        assert r.success
        assert "ERROR" in r.output
        assert r.data["alert_count"] == 0

    def test_summary_firms_error(self):
        features = [_make_nws_feature()]
        with patch.object(
            WeatherAlertsTool, "_fetch_nws", return_value=(features, None)
        ):
            with patch.object(
                WeatherAlertsTool, "_fetch_firms", return_value=([], "FIRMS error")
            ):
                r = _tool().execute(mode="summary")
        assert r.success
        assert "ERROR" in r.output

    def test_summary_both_error(self):
        with patch.object(
            WeatherAlertsTool, "_fetch_nws", return_value=([], "NWS fail")
        ):
            with patch.object(
                WeatherAlertsTool, "_fetch_firms", return_value=([], "FIRMS fail")
            ):
                r = _tool().execute(mode="summary")
        assert r.success

    def test_summary_market_only(self):
        features = [
            _make_nws_feature(event="Tornado Warning"),
            _make_nws_feature(event="Wind Advisory"),
        ]
        with patch.object(
            WeatherAlertsTool, "_fetch_nws", return_value=(features, None)
        ):
            with patch.object(
                WeatherAlertsTool, "_fetch_firms", return_value=([], None)
            ):
                r = _tool().execute(mode="summary", market_only=True)
        assert r.success


# ── 7. NWS Fetch ─────────────────────────────────────────────


class TestNWSFetch:
    def test_successful_fetch(self):
        resp_data = _make_nws_response()
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp(resp_data)
            mock_client.return_value = mc

            features, err = _tool()._fetch_nws(severity="Severe", state="")
        assert err is None
        assert len(features) == 2

    def test_rate_limited(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp({}, status=429)
            mock_client.return_value = mc

            features, err = _tool()._fetch_nws(severity="Severe", state="")
        assert features == []
        assert "Rate limited" in err

    def test_http_error(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp({}, status=500)
            mock_client.return_value = mc

            features, err = _tool()._fetch_nws(severity="Severe", state="")
        assert "500" in err

    def test_timeout(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.side_effect = httpx.TimeoutException("timed out")
            mock_client.return_value = mc

            features, err = _tool()._fetch_nws(severity="Severe", state="")
        assert features == []
        assert "timed out" in err.lower()

    def test_exception(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.side_effect = ConnectionError("DNS failed")
            mock_client.return_value = mc

            features, err = _tool()._fetch_nws(severity="Severe", state="")
        assert features == []
        assert "error" in err.lower()

    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {"features": [_make_nws_feature()]}
        features, err = WeatherAlertsTool(cache=cache)._fetch_nws(
            severity="Severe", state=""
        )
        assert err is None
        assert len(features) == 1
        cache.get.assert_called_once()

    def test_cache_miss_then_put(self):
        cache = MagicMock()
        cache.get.return_value = None
        resp_data = _make_nws_response()
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp(resp_data)
            mock_client.return_value = mc

            features, err = WeatherAlertsTool(cache=cache)._fetch_nws(
                severity="Severe", state=""
            )
        assert err is None
        cache.put.assert_called_once()

    def test_severity_filter_includes_higher(self):
        """Requesting 'Moderate' should include Extreme and Severe too."""
        cache = MagicMock()
        cache.get.return_value = None
        resp_data = _make_nws_response([])
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp(resp_data)
            mock_client.return_value = mc

            _tool()._fetch_nws(severity="Moderate", state="")
        # Verify severity param in the URL includes Extreme,Severe,Moderate
        call_url = mc.get.call_args[0][0]
        assert "Extreme" in call_url
        assert "Severe" in call_url
        assert "Moderate" in call_url


# ── 8. FIRMS Fetch ────────────────────────────────────────────


class TestFIRMSFetch:
    def test_successful_fetch(self):
        csv_data = _make_fires_csv()
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp(csv_data, is_json=False)
            mock_client.return_value = mc

            fires, err = _tool()._fetch_firms()
        assert err is None
        assert len(fires) == 3

    def test_timeout(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.side_effect = httpx.TimeoutException("timeout")
            mock_client.return_value = mc

            fires, err = _tool()._fetch_firms()
        assert fires == []
        assert "timed out" in err.lower()

    def test_http_error(self):
        with patch("httpx.Client") as mock_client:
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.get.return_value = _mock_resp("", status=500, is_json=False)
            mock_client.return_value = mc

            fires, err = _tool()._fetch_firms()
        assert fires == []
        assert "500" in err

    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = [{"lat": 1, "lon": 2}]
        fires, err = WeatherAlertsTool(cache=cache)._fetch_firms()
        assert err is None
        assert len(fires) == 1


# ── 9. Constants / Data Quality ───────────────────────────────


class TestConstants:
    def test_market_events_nonempty(self):
        assert len(_MARKET_EVENTS) >= 15

    def test_us_states_count(self):
        assert len(_US_STATES) >= 50

    def test_infra_zones_valid(self):
        assert len(INFRA_ZONES) >= 10
        for z in INFRA_ZONES:
            assert -90 <= z["lat"] <= 90
            assert -180 <= z["lon"] <= 180
            assert z["radius"] > 0

    def test_severities(self):
        assert "Extreme" in _SEVERITIES
        assert "Severe" in _SEVERITIES
        assert "Moderate" in _SEVERITIES
        assert "Minor" in _SEVERITIES

    def test_all_market_events_are_strings(self):
        for e in _MARKET_EVENTS:
            assert isinstance(e, str)
            assert len(e) > 3


# ── 10. Registry & Bandit Integration ────────────────────────


class TestRegistryIntegration:
    def test_tool_count(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert len(registry.list_names()) == 60

    def test_weather_alerts_registered(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert "weather_alerts" in registry.list_names()

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = [a.name for a in DEFAULT_ARMS]
        assert "weather_disruption" in names

    def test_bandit_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "weather_disruption")
        assert "weather_alerts" in arm.tools

    def test_bandit_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48
