"""
Edge case tests for InternetOutagesTool (OONI + RIPE Atlas).

Covers: mode validation, country validation, date validation, test_name
validation, OONI measurements fetch, OONI aggregation fetch, RIPE Atlas
probes fetch, response parsing, signal computation (anomaly rate, disconnect
rate, censorship alerts, network health), cache interaction, HTTP errors
(429/500/timeout), empty data, malformed responses, output formatting,
tool metadata, registry + bandit integration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from agent.tools.internet_outages import (
    _CACHE_TTL,
    _OONI_BASE,
    _RIPE_BASE,
    VALID_MODES,
    VALID_TESTS,
    InternetOutagesTool,
    _aggregation_signals,
    _censorship_signals,
    _extract_aggregation,
    _format_censorship,
    _format_network_health,
    _network_health_signals,
    _parse_ooni_measurements,
    _parse_ripe_probes,
    _resolve_dates,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> InternetOutagesTool:
    return InternetOutagesTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


def _ooni_measurement(
    anomaly: bool = False,
    confirmed: bool = False,
    failure: bool = False,
    probe_cc: str = "IR",
    probe_asn: str = "AS12880",
    test_name: str = "web_connectivity",
    input_url: str = "https://example.com",
) -> dict:
    return {
        "measurement_uid": "20240101T000000Z_abc",
        "probe_cc": probe_cc,
        "probe_asn": probe_asn,
        "test_name": test_name,
        "input": input_url,
        "anomaly": anomaly,
        "confirmed": confirmed,
        "failure": failure,
        "measurement_start_time": "2024-01-01T00:00:00Z",
        "scores": {"blocking_general": 1.0 if confirmed else 0},
    }


def _ooni_agg(
    anomaly: int = 10,
    confirmed: int = 2,
    failure: int = 5,
    ok: int = 100,
) -> dict:
    return {
        "result": {
            "anomaly_count": anomaly,
            "confirmed_count": confirmed,
            "failure_count": failure,
            "ok_count": ok,
            "measurement_count": anomaly + confirmed + failure + ok,
        }
    }


def _ripe_probe(
    probe_id: int = 1,
    status: str = "Connected",
    asn: int = 12345,
    country: str = "RU",
) -> dict:
    return {
        "id": probe_id,
        "asn_v4": asn,
        "country_code": country,
        "status": {"name": status, "since": "2024-01-01T00:00:00Z"},
        "address_v4": "1.2.3.4",
        "is_anchor": False,
        "tags": [{"name": "Native IPv6"}, {"name": "system-v4"}],
    }


def _ripe_response(probes: list, total: int = 0) -> dict:
    return {
        "count": total or len(probes),
        "results": probes,
    }


# ── TestToolMetadata ──────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "internet_outages"

    def test_description_mentions_ooni_and_ripe(self):
        desc = _tool().description
        assert "OONI" in desc
        assert "RIPE" in desc

    def test_parameters_mode_required(self):
        assert "mode" in _tool().parameters["required"]

    def test_parameters_mode_enum(self):
        enum = _tool().parameters["properties"]["mode"]["enum"]
        assert set(enum) == VALID_MODES


# ── TestInputValidation ───────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        r = _tool().execute(mode="")
        assert not r.success

    def test_invalid_country_length(self):
        r = _tool().execute(mode="censorship", country="USA")
        assert not r.success
        assert "2-letter" in r.output

    def test_single_char_country(self):
        r = _tool().execute(mode="censorship", country="X")
        assert not r.success

    def test_country_uppercased(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({"results": []})
            r = _tool().execute(mode="censorship", country="ir")
        assert r.success

    def test_invalid_test_name_censorship(self):
        r = _tool().execute(
            mode="censorship",
            country="IR",
            test_name="invalid_test",
        )
        assert not r.success
        assert "Invalid test_name" in r.output

    def test_invalid_test_name_outage(self):
        r = _tool().execute(
            mode="outage_detection",
            test_name="bad_test",
        )
        assert not r.success

    def test_invalid_since_date(self):
        r = _tool().execute(
            mode="censorship",
            country="IR",
            since="not-a-date",
        )
        assert not r.success
        assert "YYYY-MM-DD" in r.output

    def test_invalid_until_date(self):
        r = _tool().execute(
            mode="censorship",
            country="IR",
            until="2024/01/01",
        )
        assert not r.success

    def test_network_health_requires_country(self):
        r = _tool().execute(mode="network_health")
        assert not r.success
        assert "country" in r.output.lower()

    def test_limit_clamped(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({"results": []})
            r = _tool().execute(mode="censorship", country="IR", limit=500)
        assert r.success
        call_params = mc.return_value.get.call_args[1]["params"]
        assert call_params["limit"] == "200"


# ── TestDateResolution ────────────────────────────────────────


class TestDateResolution:
    def test_default_dates(self):
        since, until, err = _resolve_dates({})
        assert err is None
        assert len(since) == 10
        assert len(until) == 10

    def test_custom_dates(self):
        since, until, err = _resolve_dates(
            {
                "since": "2024-01-01",
                "until": "2024-01-31",
            }
        )
        assert err is None
        assert since == "2024-01-01"
        assert until == "2024-01-31"

    def test_bad_since(self):
        _, _, err = _resolve_dates({"since": "bad"})
        assert err is not None
        assert "since" in err

    def test_bad_until(self):
        _, _, err = _resolve_dates({"until": "bad"})
        assert err is not None
        assert "until" in err


# ── TestOONIMeasurementParsing ────────────────────────────────


class TestOONIMeasurementParsing:
    def test_parse_basic(self):
        records, counts = _parse_ooni_measurements(
            [
                _ooni_measurement(anomaly=True),
                _ooni_measurement(confirmed=True, anomaly=True),
                _ooni_measurement(),
            ]
        )
        assert len(records) == 3
        assert counts["total"] == 3
        assert counts["anomaly"] == 2
        assert counts["confirmed"] == 1

    def test_parse_empty(self):
        records, counts = _parse_ooni_measurements([])
        assert records == []
        assert counts["total"] == 0

    def test_parse_all_failures(self):
        records, counts = _parse_ooni_measurements(
            [
                _ooni_measurement(failure=True),
                _ooni_measurement(failure=True),
            ]
        )
        assert counts["failure"] == 2

    def test_record_fields(self):
        records, _ = _parse_ooni_measurements([_ooni_measurement(probe_cc="CN")])
        r = records[0]
        assert r["probe_cc"] == "CN"
        assert r["test_name"] == "web_connectivity"
        assert "scores" in r


# ── TestCensorshipSignals ─────────────────────────────────────


class TestCensorshipSignals:
    def test_empty(self):
        signals = _censorship_signals(
            {"total": 0, "anomaly": 0, "confirmed": 0, "failure": 0},
            "IR",
        )
        assert signals["anomaly_rate_pct"] == 0
        assert "alert" not in signals

    def test_critical_alert(self):
        signals = _censorship_signals(
            {"total": 100, "anomaly": 60, "confirmed": 10, "failure": 5},
            "IR",
        )
        assert "CRITICAL" in signals["alert"]
        assert signals["anomaly_rate_pct"] == 60.0

    def test_warning_alert(self):
        signals = _censorship_signals(
            {"total": 100, "anomaly": 25, "confirmed": 0, "failure": 5},
            "IR",
        )
        assert "WARNING" in signals["alert"]

    def test_notice_confirmed(self):
        signals = _censorship_signals(
            {"total": 100, "anomaly": 5, "confirmed": 3, "failure": 2},
            "CN",
        )
        assert "NOTICE" in signals["alert"]
        assert "3" in signals["alert"]

    def test_no_alert_clean(self):
        signals = _censorship_signals(
            {"total": 100, "anomaly": 5, "confirmed": 0, "failure": 2},
            "US",
        )
        assert "alert" not in signals


# ── TestAggregationExtraction ─────────────────────────────────


class TestAggregationExtraction:
    def test_standard_response(self):
        agg = _extract_aggregation(_ooni_agg())
        assert agg["anomaly"] == 10
        assert agg["ok"] == 100
        assert agg["total"] == 117

    def test_list_result(self):
        body = {
            "result": [
                {
                    "anomaly_count": 5,
                    "confirmed_count": 1,
                    "failure_count": 2,
                    "ok_count": 50,
                    "measurement_count": 58,
                }
            ]
        }
        agg = _extract_aggregation(body)
        assert agg["anomaly"] == 5
        assert agg["total"] == 58

    def test_empty_list_result(self):
        body = {"result": []}
        agg = _extract_aggregation(body)
        assert agg["total"] == 0

    def test_flat_body_no_result_key(self):
        body = {
            "anomaly_count": 3,
            "confirmed_count": 0,
            "failure_count": 1,
            "ok_count": 20,
            "measurement_count": 24,
        }
        agg = _extract_aggregation(body)
        assert agg["anomaly"] == 3
        assert agg["total"] == 24


# ── TestAggregationSignals ────────────────────────────────────


class TestAggregationSignals:
    def test_critical(self):
        agg = {"anomaly": 60, "confirmed": 5, "failure": 10, "ok": 25, "total": 100}
        signals = _aggregation_signals(agg)
        assert "CRITICAL" in signals["alert"]

    def test_warning_anomaly(self):
        agg = {"anomaly": 25, "confirmed": 0, "failure": 5, "ok": 70, "total": 100}
        signals = _aggregation_signals(agg)
        assert "WARNING" in signals["alert"]
        assert "anomaly" in signals["alert"].lower()

    def test_warning_failure(self):
        agg = {"anomaly": 5, "confirmed": 0, "failure": 35, "ok": 60, "total": 100}
        signals = _aggregation_signals(agg)
        assert "failure" in signals["alert"].lower()

    def test_no_alert(self):
        agg = {"anomaly": 2, "confirmed": 0, "failure": 3, "ok": 95, "total": 100}
        signals = _aggregation_signals(agg)
        assert "alert" not in signals

    def test_zero_total(self):
        agg = {"anomaly": 0, "confirmed": 0, "failure": 0, "ok": 0, "total": 0}
        signals = _aggregation_signals(agg)
        assert signals["anomaly_rate_pct"] == 0


# ── TestRIPEProbeParsing ──────────────────────────────────────


class TestRIPEProbeParsing:
    def test_parse_basic(self):
        records, status, asns = _parse_ripe_probes(
            [_ripe_probe(), _ripe_probe(probe_id=2, status="Disconnected")],
            "RU",
        )
        assert len(records) == 2
        assert status["Connected"] == 1
        assert status["Disconnected"] == 1

    def test_parse_empty(self):
        records, status, asns = _parse_ripe_probes([], "US")
        assert records == []
        assert all(v == 0 for v in status.values())

    def test_parse_asn_counting(self):
        records, _, asns = _parse_ripe_probes(
            [
                _ripe_probe(probe_id=1, asn=100),
                _ripe_probe(probe_id=2, asn=100),
                _ripe_probe(probe_id=3, asn=200),
            ],
            "US",
        )
        assert asns[100] == 2
        assert asns[200] == 1

    def test_parse_status_as_string(self):
        """Handle status as string instead of dict."""
        probe = _ripe_probe()
        probe["status"] = "Connected"
        records, status, _ = _parse_ripe_probes([probe], "US")
        # Should not crash; status_name comes from str()
        assert len(records) == 1

    def test_parse_tags(self):
        records, _, _ = _parse_ripe_probes([_ripe_probe()], "RU")
        assert "Native IPv6" in records[0]["tags"]

    def test_parse_no_tags(self):
        probe = _ripe_probe()
        probe["tags"] = None
        records, _, _ = _parse_ripe_probes([probe], "RU")
        assert records[0]["tags"] == []


# ── TestNetworkHealthSignals ──────────────────────────────────


class TestNetworkHealthSignals:
    def test_critical_disconnect(self):
        status = {
            "Connected": 20,
            "Disconnected": 30,
            "Abandoned": 0,
            "Never Connected": 0,
        }
        signals = _network_health_signals(status, {}, 50, "RU")
        assert "CRITICAL" in signals["alert"]
        assert signals["disconnect_rate_pct"] == 60.0

    def test_warning_disconnect(self):
        status = {
            "Connected": 70,
            "Disconnected": 25,
            "Abandoned": 5,
            "Never Connected": 0,
        }
        signals = _network_health_signals(status, {}, 100, "IR")
        assert "WARNING" in signals["alert"]

    def test_notice_disconnect(self):
        status = {
            "Connected": 80,
            "Disconnected": 15,
            "Abandoned": 5,
            "Never Connected": 0,
        }
        signals = _network_health_signals(status, {}, 100, "CN")
        assert "NOTICE" in signals["alert"]

    def test_healthy(self):
        status = {
            "Connected": 95,
            "Disconnected": 3,
            "Abandoned": 2,
            "Never Connected": 0,
        }
        signals = _network_health_signals(status, {}, 100, "US")
        assert "alert" not in signals

    def test_zero_active(self):
        status = {
            "Connected": 0,
            "Disconnected": 0,
            "Abandoned": 5,
            "Never Connected": 0,
        }
        signals = _network_health_signals(status, {}, 5, "XX")
        assert signals["disconnect_rate_pct"] == 0

    def test_top_asns(self):
        asns = {100: 10, 200: 5, 300: 3}
        signals = _network_health_signals(
            {"Connected": 18, "Disconnected": 0, "Abandoned": 0, "Never Connected": 0},
            asns,
            18,
            "US",
        )
        assert len(signals["top_asns"]) == 3
        assert signals["top_asns"][0]["asn"] == 100


# ── TestOutputFormatting ──────────────────────────────────────


class TestOutputFormatting:
    def test_censorship_format(self):
        records = [
            {
                "confirmed": True,
                "anomaly": True,
                "input": "https://blocked.com",
                "probe_asn": "AS1234",
                "measurement_start_time": "2024-01-01",
            },
        ]
        signals = {
            "total_measurements": 1,
            "anomaly_count": 1,
            "confirmed_count": 1,
            "failure_count": 0,
            "anomaly_rate_pct": 100.0,
            "alert": "CRITICAL: test",
        }
        out = _format_censorship(records, signals, "IR")
        assert "IR" in out
        assert "CRITICAL" in out
        assert "blocked.com" in out

    def test_censorship_empty(self):
        out = _format_censorship(
            [],
            {
                "total_measurements": 0,
                "anomaly_count": 0,
                "confirmed_count": 0,
                "failure_count": 0,
                "anomaly_rate_pct": 0,
            },
            "US",
        )
        assert "0" in out

    def test_network_health_format(self):
        records = [{"probe_id": 1}]
        signals = {
            "total_probes": 50,
            "total_in_country": 100,
            "connected": 40,
            "disconnected": 10,
            "abandoned": 0,
            "disconnect_rate_pct": 20.0,
            "unique_asns": 5,
            "alert": "WARNING: test",
            "top_asns": [{"asn": 100, "probe_count": 20}],
        }
        out = _format_network_health(records, signals, 100, "RU", {100: 20})
        assert "RU" in out
        assert "WARNING" in out
        assert "AS100" in out


# ── TestCensorshipMode ────────────────────────────────────────


class TestCensorshipMode:
    def test_success(self):
        body = {"results": [_ooni_measurement(anomaly=True)]}
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="censorship", country="IR")
        assert r.success
        assert r.data["country"] == "IR"

    def test_no_country(self):
        """Censorship without country should query all."""
        body = {"results": [_ooni_measurement()]}
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="censorship")
        assert r.success
        call_params = mc.return_value.get.call_args[1]["params"]
        assert "probe_cc" not in call_params

    def test_custom_test_name(self):
        body = {"results": []}
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(
                mode="censorship",
                country="CN",
                test_name="whatsapp",
            )
        assert r.success
        call_params = mc.return_value.get.call_args[1]["params"]
        assert call_params["test_name"] == "whatsapp"


# ── TestNetworkHealthMode ─────────────────────────────────────


class TestNetworkHealthMode:
    def test_success(self):
        body = _ripe_response([_ripe_probe()])
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="network_health", country="RU")
        assert r.success
        assert r.data["country"] == "RU"

    def test_mixed_status(self):
        body = _ripe_response(
            [
                _ripe_probe(probe_id=1, status="Connected"),
                _ripe_probe(probe_id=2, status="Disconnected"),
                _ripe_probe(probe_id=3, status="Abandoned"),
            ]
        )
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(body)
            r = _tool().execute(mode="network_health", country="IR")
        assert r.success
        assert r.data["signals"]["connected"] == 1
        assert r.data["signals"]["disconnected"] == 1


# ── TestOutageDetectionMode ───────────────────────────────────


class TestOutageDetectionMode:
    def test_success(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_ooni_agg())
            r = _tool().execute(mode="outage_detection", country="CN")
        assert r.success
        assert r.data["signals"]["anomaly_count"] == 10

    def test_no_country(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_ooni_agg())
            r = _tool().execute(mode="outage_detection")
        assert r.success
        assert r.data["country"] == "ALL"


# ── TestHTTPErrors ────────────────────────────────────────────


class TestHTTPErrors:
    def test_ooni_timeout(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.side_effect = httpx.TimeoutException("t")
            r = _tool().execute(mode="censorship", country="IR")
        assert not r.success
        assert "timed out" in r.output

    def test_ooni_http_error(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.side_effect = httpx.HTTPError("e")
            r = _tool().execute(mode="censorship", country="IR")
        assert not r.success

    def test_ooni_429(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({}, 429)
            r = _tool().execute(mode="censorship", country="IR")
        assert not r.success
        assert "rate limit" in r.output.lower()

    def test_ooni_500(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({}, 500)
            r = _tool().execute(mode="censorship", country="IR")
        assert not r.success
        assert "500" in r.output

    def test_ripe_timeout(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.side_effect = httpx.TimeoutException("t")
            r = _tool().execute(mode="network_health", country="US")
        assert not r.success
        assert "timed out" in r.output

    def test_ripe_429(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({}, 429)
            r = _tool().execute(mode="network_health", country="US")
        assert not r.success

    def test_ripe_500(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({}, 500)
            r = _tool().execute(mode="network_health", country="US")
        assert not r.success

    def test_aggregation_timeout(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.side_effect = httpx.TimeoutException("t")
            r = _tool().execute(mode="outage_detection", country="CN")
        assert not r.success

    def test_malformed_json_ooni(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            resp = httpx.Response(
                status_code=200,
                text="not json",
                request=httpx.Request("GET", "http://test"),
            )
            mc.return_value.get.return_value = resp
            r = _tool().execute(mode="censorship", country="IR")
        assert not r.success
        assert "parse" in r.output.lower()

    def test_malformed_json_ripe(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            resp = httpx.Response(
                status_code=200,
                text="<html>",
                request=httpx.Request("GET", "http://test"),
            )
            mc.return_value.get.return_value = resp
            r = _tool().execute(mode="network_health", country="US")
        assert not r.success

    def test_malformed_json_aggregation(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            resp = httpx.Response(
                status_code=200,
                text="{{bad",
                request=httpx.Request("GET", "http://test"),
            )
            mc.return_value.get.return_value = resp
            r = _tool().execute(mode="outage_detection")
        assert not r.success


# ── TestCache ─────────────────────────────────────────────────


class TestCache:
    def test_censorship_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {"output": "cached", "data": {"cached": True}}
        r = _tool(cache=cache).execute(mode="censorship", country="IR")
        assert r.success
        assert r.output == "cached"

    def test_network_health_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {"output": "cached", "data": {"cached": True}}
        r = _tool(cache=cache).execute(mode="network_health", country="US")
        assert r.success

    def test_outage_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {"output": "cached", "data": {"cached": True}}
        r = _tool(cache=cache).execute(mode="outage_detection", country="CN")
        assert r.success

    def test_cache_miss_stores(self):
        cache = MagicMock()
        cache.get.return_value = None
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({"results": []})
            r = _tool(cache=cache).execute(mode="censorship", country="IR")
        assert r.success
        cache.set.assert_called_once()
        assert cache.set.call_args[1]["ttl"] == _CACHE_TTL

    def test_no_cache_works(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({"results": []})
            r = _tool(cache=None).execute(mode="censorship", country="IR")
        assert r.success


# ── TestConstants ─────────────────────────────────────────────


class TestConstants:
    def test_valid_modes(self):
        assert {"censorship", "network_health", "outage_detection"} == VALID_MODES

    def test_valid_tests_include_key_protocols(self):
        assert "web_connectivity" in VALID_TESTS
        assert "whatsapp" in VALID_TESTS
        assert "telegram" in VALID_TESTS
        assert "signal" in VALID_TESTS
        assert "tor" in VALID_TESTS

    def test_ooni_base_url(self):
        assert "ooni.io" in _OONI_BASE

    def test_ripe_base_url(self):
        assert "atlas.ripe.net" in _RIPE_BASE


# ── TestRegistryAndBandit ─────────────────────────────────────


class TestRegistryAndBandit:
    def test_tool_in_cli_registry(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        assert "internet_outages" in registry.list_names()

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "internet_infrastructure_monitor" in arm_names

    def test_bandit_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "internet_infrastructure_monitor")
        assert "internet_outages" in arm.tools


# ── TestEdgeCombinations ──────────────────────────────────────


class TestEdgeCombinations:
    def test_all_valid_tests_accepted(self):
        for test in VALID_TESTS:
            with patch("httpx.Client") as mc:
                mc.return_value.__enter__ = lambda s: s
                mc.return_value.__exit__ = MagicMock(return_value=False)
                mc.return_value.get.return_value = _mock_resp({"results": []})
                r = _tool().execute(
                    mode="censorship",
                    country="US",
                    test_name=test,
                )
            assert r.success, f"test_name '{test}' should be valid"

    def test_empty_ooni_results(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({"results": []})
            r = _tool().execute(mode="censorship", country="US")
        assert r.success
        assert r.data["signals"]["total_measurements"] == 0

    def test_empty_ripe_results(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp(_ripe_response([], total=0))
            r = _tool().execute(mode="network_health", country="XX")
        assert r.success
        assert r.data["signals"]["total_probes"] == 0

    def test_high_anomaly_count(self):
        measurements = [_ooni_measurement(anomaly=True) for _ in range(80)] + [_ooni_measurement() for _ in range(20)]
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__ = lambda s: s
            mc.return_value.__exit__ = MagicMock(return_value=False)
            mc.return_value.get.return_value = _mock_resp({"results": measurements})
            r = _tool().execute(mode="censorship", country="IR")
        assert r.success
        assert "CRITICAL" in r.data["signals"]["alert"]
