"""
Edge case tests for DiseaseSurveillanceTool (CDC NWSS + WHO DON + ECDC + NCBI).

Covers: mode validation, pathogen validation, state validation, parameter
clamping, WHO title parsing, pathogen alias resolution, CDC Socrata fetch,
WHO OData fetch, ECDC fetch, NCBI E-utilities fetch, cache interaction,
HTTP error handling (429/500/timeout), empty data, malformed data, aggregate
mode, multi-state wave detection, genomic velocity signals, ECDC dataset
routing, output formatting, tool metadata, registry + bandit integration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.disease_surveillance import (
    DiseaseSurveillanceTool,
    _CDC_DATASETS,
    _CDC_AGGREGATE_ID,
    _ECDC_DATASETS,
    _PATHOGEN_ALIASES,
    _US_STATES,
    _parse_who_title,
    _resolve_pathogen,
    _safe_float,
    _safe_int,
)
from agent.tools.base import ToolResult


# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> DiseaseSurveillanceTool:
    return DiseaseSurveillanceTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


def _make_cdc_record(
    state: str = "CA",
    detect: str = "yes",
    conc: float = 1234.5,
    pop: int = 500000,
    date: str = "2026-03-20",
    pcr_target: str = "sars-cov-2",
) -> dict[str, Any]:
    return {
        "record_id": f"rec_{state}_{date}",
        "state_territory": state,
        "pcr_target_detect": detect,
        "pcr_target_avg_conc": str(conc),
        "population_served": str(pop),
        "sample_collect_date": date,
        "pcr_target": pcr_target,
        "county_fips": "06001",
        "counties_served": f"{state} County",
    }


def _make_cdc_aggregate(
    jurisdiction: str = "California",
    ptc: float = 50.0,
    detect_prop: float = 0.6,
    percentile: float = 75.0,
    date_end: str = "2026-03-25",
) -> dict[str, Any]:
    return {
        "wwtp_jurisdiction": jurisdiction,
        "wwtp_id": "WWTP001",
        "county_fips": "06001",
        "population_served": "1000000",
        "date_start": "2026-03-10",
        "date_end": date_end,
        "ptc_15d": str(ptc),
        "detect_prop_15d": str(detect_prop),
        "percentile": str(percentile),
    }


def _make_who_entry(
    title: str = "Mpox - Democratic Republic of the Congo",
    date: str = "2026-03-15T00:00:00Z",
    don_id: str = "DON-123",
) -> dict[str, Any]:
    return {
        "Title": title,
        "PublicationDate": date,
        "DonId": don_id,
        "UrlName": f"/item/{don_id}",
        "Summary": "Test summary",
    }


def _make_ecdc_case(
    country: str = "Germany",
    country_code: str = "DE",
    year_week: str = "2026-W12",
    indicator: str = "cases",
    count: int = 1500,
) -> dict[str, Any]:
    return {
        "country": country,
        "country_code": country_code,
        "year_week": year_week,
        "indicator": indicator,
        "weekly_count": count,
    }


def _make_ecdc_variant(
    variant: str = "BA.2.86",
    country_code: str = "DE",
    year_week: str = "2026-W12",
    pct: float = 35.0,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "country_code": country_code,
        "year_week": year_week,
        "percent_variant": pct,
    }


def _make_ecdc_hospital(
    country: str = "France",
    year_week: str = "2026-W12",
    indicator: str = "Weekly new ICU admissions per 100k",
    value: float = 2.5,
) -> dict[str, Any]:
    return {
        "country": country,
        "year_week": year_week,
        "indicator": indicator,
        "value": value,
    }


def _make_ncbi_response(count: int = 12897) -> dict[str, Any]:
    return {
        "header": {"type": "esearch", "version": "0.3"},
        "esearchresult": {
            "count": str(count),
            "retmax": "0",
            "retstart": "0",
            "idlist": [],
        },
    }


# ── 1. Tool Metadata ─────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "disease_surveillance"

    def test_description_nonempty(self):
        assert len(_tool().description) > 50

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        props = params["properties"]
        assert "mode" in props
        assert "pathogen" in props
        assert "state" in props
        assert "disease" in props
        assert "dataset" in props
        assert "country" in props
        assert "organism" in props
        assert "days_back" in props
        assert "limit" in props

    def test_mode_enum(self):
        modes = _tool().parameters["properties"]["mode"]["enum"]
        assert set(modes) == {"wastewater", "outbreaks", "eu_surveillance", "genomics"}

    def test_dataset_enum(self):
        datasets = _tool().parameters["properties"]["dataset"]["enum"]
        assert set(datasets) == {"cases", "variants", "hospital"}


# ── 2. Input Validation ──────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="bad")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_invalid_mode_case(self):
        """Mode is lowered internally."""
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_wastewater",
            return_value=ToolResult(success=True, output="ok"),
        ):
            r = _tool().execute(mode="WASTEWATER")
            assert r.success

    def test_days_back_clamped_low(self):
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_wastewater",
            return_value=ToolResult(success=True, output="ok"),
        ) as m:
            _tool().execute(mode="wastewater", days_back=-5)
            _, kwargs = m.call_args
            assert kwargs["days_back"] >= 1

    def test_days_back_clamped_high(self):
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_wastewater",
            return_value=ToolResult(success=True, output="ok"),
        ) as m:
            _tool().execute(mode="wastewater", days_back=9999)
            _, kwargs = m.call_args
            assert kwargs["days_back"] <= 180

    def test_limit_clamped_low(self):
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_wastewater",
            return_value=ToolResult(success=True, output="ok"),
        ) as m:
            _tool().execute(mode="wastewater", limit=0)
            _, kwargs = m.call_args
            assert kwargs["limit"] >= 1

    def test_limit_clamped_high(self):
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_wastewater",
            return_value=ToolResult(success=True, output="ok"),
        ) as m:
            _tool().execute(mode="wastewater", limit=99999)
            _, kwargs = m.call_args
            assert kwargs["limit"] <= 1000

    def test_extra_kwargs_ignored(self):
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_wastewater",
            return_value=ToolResult(success=True, output="ok"),
        ):
            r = _tool().execute(mode="wastewater", bogus="thing")
            assert r.success

    def test_invalid_pathogen(self):
        r = _tool().execute(mode="wastewater", pathogen="nonexistent_virus")
        assert not r.success
        assert "Unknown pathogen" in r.output

    def test_invalid_state(self):
        r = _tool().execute(mode="wastewater", pathogen="covid", state="ZZ")
        assert not r.success
        assert "Unknown US state" in r.output

    def test_invalid_ecdc_dataset(self):
        r = _tool().execute(mode="eu_surveillance", dataset="bogus")
        assert not r.success
        assert "Unknown ECDC dataset" in r.output


# ── 3. Helper Functions ──────────────────────────────────────


class TestResolvePathogen:
    def test_canonical_names(self):
        for key in _CDC_DATASETS:
            assert _resolve_pathogen(key) == key

    def test_alias_covid(self):
        assert _resolve_pathogen("covid") == "sars-cov-2"
        assert _resolve_pathogen("covid-19") == "sars-cov-2"
        assert _resolve_pathogen("COVID19") == "sars-cov-2"

    def test_alias_flu(self):
        assert _resolve_pathogen("flu") == "influenza_a"
        assert _resolve_pathogen("influenza") == "influenza_a"

    def test_alias_h5n1(self):
        assert _resolve_pathogen("h5n1") == "avian_h5"
        assert _resolve_pathogen("bird_flu") == "avian_h5"
        assert _resolve_pathogen("H5") == "avian_h5"

    def test_alias_monkeypox(self):
        assert _resolve_pathogen("monkeypox") == "mpox"

    def test_unknown_returns_none(self):
        assert _resolve_pathogen("ebola") is None
        assert _resolve_pathogen("") is None

    def test_whitespace_handling(self):
        assert _resolve_pathogen("  covid  ") == "sars-cov-2"

    def test_hyphen_underscore_equivalence(self):
        assert _resolve_pathogen("sars-cov-2") == "sars-cov-2"
        assert _resolve_pathogen("sars_cov_2") == "sars-cov-2"


class TestParseWhoTitle:
    def test_standard_format(self):
        r = _parse_who_title("Nipah virus infection - Bangladesh")
        assert r["disease"] == "Nipah virus infection"
        assert r["country"] == "Bangladesh"

    def test_en_dash(self):
        r = _parse_who_title("Avian Influenza A(H5N1) – United States of America")
        assert r["disease"] == "Avian Influenza A(H5N1)"
        assert r["country"] == "United States of America"

    def test_colon_format(self):
        r = _parse_who_title("Mpox: Global situation update")
        assert r["disease"] == "Mpox"
        assert r["country"] == "Global situation update"

    def test_no_separator(self):
        r = _parse_who_title("Disease X outbreak")
        assert r["disease"] == "Disease X outbreak"
        assert r["country"] == ""

    def test_empty_title(self):
        r = _parse_who_title("")
        assert r["disease"] == ""
        assert r["country"] == ""

    def test_multiple_dashes(self):
        r = _parse_who_title("Cholera - Democratic Republic of the Congo - update")
        assert "Cholera" in r["disease"]

    def test_em_dash(self):
        r = _parse_who_title("Ebola virus disease — Uganda")
        assert r["disease"] == "Ebola virus disease"
        assert r["country"] == "Uganda"


class TestSafeFloat:
    def test_normal(self):
        assert _safe_float("123.45") == 123.45

    def test_int_string(self):
        assert _safe_float("100") == 100.0

    def test_none(self):
        assert _safe_float(None) is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_invalid(self):
        assert _safe_float("abc") is None

    def test_numeric(self):
        assert _safe_float(42) == 42.0


class TestSafeInt:
    def test_normal(self):
        assert _safe_int("100") == 100

    def test_float_string(self):
        assert _safe_int("100.7") == 100

    def test_none(self):
        assert _safe_int(None) is None

    def test_invalid(self):
        assert _safe_int("abc") is None


# ── 4. Wastewater Mode — CDC Pathogen ─────────────────────────


class TestWastewaterPathogen:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_basic_fetch(self, mock_client_cls):
        records = [
            _make_cdc_record("CA"),
            _make_cdc_record("CA", detect="no", conc=0),
            _make_cdc_record("NY"),
        ]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert r.success
        assert "SARS-COV-2" in r.output.upper() or "sars-cov-2" in r.output.lower()
        assert r.data["total_samples"] == 3
        assert r.data["states_count"] == 2

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_state_filter(self, mock_client_cls):
        records = [_make_cdc_record("TX")]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="flu", state="TX")
        assert r.success

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_empty_data(self, mock_client_cls):
        mock_resp = _mock_resp([])
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="measles")
        assert r.success
        assert "No" in r.output
        assert r.data["count"] == 0

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_multi_state_wave_alert(self, mock_client_cls):
        """6+ states with >50% detection should trigger wave alert."""
        states = ["CA", "NY", "TX", "FL", "IL", "WA", "PA"]
        records = []
        for st in states:
            records.append(_make_cdc_record(st, detect="yes"))
            records.append(_make_cdc_record(st, detect="yes"))
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="rsv")
        assert r.success
        assert "MULTI-STATE WAVE" in r.output
        assert r.data["hot_states"] >= 5

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_detection_rate_sorting(self, mock_client_cls):
        records = [
            _make_cdc_record("CA", detect="yes"),
            _make_cdc_record("CA", detect="no"),
            _make_cdc_record("NY", detect="yes"),
            _make_cdc_record("NY", detect="yes"),
        ]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert r.success
        sums = r.data["summaries"]
        # NY (100%) should be before CA (50%)
        assert sums[0]["state"] == "NY"
        assert sums[0]["detection_rate"] == 1.0
        assert sums[1]["state"] == "CA"
        assert sums[1]["detection_rate"] == 0.5

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_zero_concentration_excluded(self, mock_client_cls):
        """Records with conc=0 should not count in mean/max."""
        records = [
            _make_cdc_record("CA", conc=0),
            _make_cdc_record("CA", conc=100.0),
        ]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert r.success
        ca = [s for s in r.data["summaries"] if s["state"] == "CA"][0]
        assert ca["mean_concentration"] == 100.0

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_missing_concentration_field(self, mock_client_cls):
        """Record without concentration field should be handled."""
        rec = _make_cdc_record("CA")
        del rec["pcr_target_avg_conc"]
        mock_resp = _mock_resp([rec])
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert r.success
        ca = r.data["summaries"][0]
        assert ca["mean_concentration"] is None


# ── 5. Wastewater Mode — CDC Aggregate ────────────────────────


class TestWastewaterAggregate:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_basic_aggregate(self, mock_client_cls):
        records = [
            _make_cdc_aggregate("California", ptc=50.0),
            _make_cdc_aggregate("New York", ptc=120.0),
        ]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater")  # no pathogen = aggregate
        assert r.success
        assert "Aggregate" in r.output
        assert r.data["total_records"] == 2

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_surge_detection(self, mock_client_cls):
        records = [
            _make_cdc_aggregate("California", ptc=150.0),
            _make_cdc_aggregate("Texas", ptc=200.0),
            _make_cdc_aggregate("New York", ptc=30.0),
        ]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater")
        assert r.success
        assert r.data["surge_count"] == 2
        assert "SURGE" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_empty_aggregate(self, mock_client_cls):
        mock_resp = _mock_resp([])
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater")
        assert r.success
        assert "No aggregate" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_aggregate_sorted_by_ptc(self, mock_client_cls):
        records = [
            _make_cdc_aggregate("Low", ptc=10.0),
            _make_cdc_aggregate("High", ptc=300.0),
            _make_cdc_aggregate("Mid", ptc=80.0),
        ]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater")
        assert r.success
        sums = r.data["summaries"]
        assert sums[0]["jurisdiction"] == "High"
        assert sums[2]["jurisdiction"] == "Low"

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_null_ptc_values(self, mock_client_cls):
        """Records with missing ptc_15d should not crash."""
        rec = _make_cdc_aggregate("Test")
        del rec["ptc_15d"]
        mock_resp = _mock_resp([rec])
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater")
        assert r.success
        assert r.data["summaries"][0]["avg_ptc_15d"] is None


# ── 6. CDC Socrata Errors ─────────────────────────────────────


class TestCDCSocrataErrors:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_rate_limited_429(self, mock_client_cls):
        mock_resp = _mock_resp({"error": "rate limited"}, status=429)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert not r.success
        assert "Rate limited" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_server_error_500(self, mock_client_cls):
        mock_resp = _mock_resp({"error": "server error"}, status=500)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert not r.success
        assert "500" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_timeout(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=httpx.TimeoutException("timeout"),
        )

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert not r.success
        assert "timed out" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_connection_error(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=httpx.ConnectError("dns fail"),
        )

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert not r.success
        assert "fetch error" in r.output.lower()

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_non_list_response(self, mock_client_cls):
        """Socrata sometimes returns error objects instead of arrays."""
        mock_resp = _mock_resp({"error": True, "message": "bad query"})
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert not r.success
        assert "Unexpected" in r.output


# ── 7. Outbreaks Mode — WHO DON ──────────────────────────────


class TestOutbreaksMode:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_basic_fetch(self, mock_client_cls):
        entries = [
            _make_who_entry("Mpox - Democratic Republic of the Congo"),
            _make_who_entry("Cholera - Haiti", don_id="DON-124"),
            _make_who_entry(
                "Avian Influenza A(H5N1) - United States of America", don_id="DON-125"
            ),
        ]
        mock_resp = _mock_resp({"value": entries})
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="outbreaks")
        assert r.success
        assert r.data["count"] == 3
        assert "disease_frequency" in r.data

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_disease_filter(self, mock_client_cls):
        entries = [_make_who_entry("Mpox - Global")]
        mock_resp = _mock_resp({"value": entries})
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="outbreaks", disease="mpox")
        assert r.success

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_empty_entries(self, mock_client_cls):
        mock_resp = _mock_resp({"value": []})
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="outbreaks")
        assert r.success
        assert "No outbreak entries" in r.output
        assert r.data["count"] == 0

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_disease_frequency(self, mock_client_cls):
        entries = [
            _make_who_entry("Cholera - Haiti", don_id="1"),
            _make_who_entry("Cholera - Yemen", don_id="2"),
            _make_who_entry("Mpox - DRC", don_id="3"),
        ]
        mock_resp = _mock_resp({"value": entries})
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="outbreaks")
        assert r.success
        freq = r.data["disease_frequency"]
        assert freq.get("cholera", 0) == 2
        assert freq.get("mpox", 0) == 1


class TestWHOErrors:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_who_429(self, mock_client_cls):
        mock_resp = _mock_resp({}, status=429)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="outbreaks")
        assert not r.success
        assert "Rate limited" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_who_500(self, mock_client_cls):
        mock_resp = _mock_resp({}, status=500)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="outbreaks")
        assert not r.success
        assert "500" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_who_timeout(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=httpx.TimeoutException("timeout"),
        )

        r = _tool().execute(mode="outbreaks")
        assert not r.success
        assert "timed out" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_who_unexpected_format(self, mock_client_cls):
        """WHO returns something without 'value' key."""
        mock_resp = _mock_resp({"weird": "data"})
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="outbreaks")
        # Should still succeed with empty results since value defaults to []
        assert r.success
        assert r.data["count"] == 0


# ── 8. EU Surveillance Mode — ECDC ───────────────────────────


class TestEUSurveillanceMode:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_cases_mode(self, mock_client_cls):
        records = [
            _make_ecdc_case("Germany", "DE", "2026-W12"),
            _make_ecdc_case("France", "FR", "2026-W12"),
            _make_ecdc_case("Germany", "DE", "2026-W11"),
        ]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="eu_surveillance", dataset="cases")
        assert r.success
        assert "countries" in r.data

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_variants_mode(self, mock_client_cls):
        records = [
            _make_ecdc_variant("BA.2.86", "DE"),
            _make_ecdc_variant("XBB.1.5", "DE"),
            _make_ecdc_variant("BA.2.86", "FR"),
        ]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="eu_surveillance", dataset="variants")
        assert r.success
        assert "variants" in r.data

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_hospital_mode(self, mock_client_cls):
        records = [_make_ecdc_hospital()]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="eu_surveillance", dataset="hospital")
        assert r.success
        assert "Hospital" in r.output or "ICU" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_country_filter(self, mock_client_cls):
        records = [
            _make_ecdc_case("Germany", "DE"),
            _make_ecdc_case("France", "FR"),
        ]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="eu_surveillance", dataset="cases", country="DE")
        assert r.success
        # Should only contain DE results
        for rec in r.data.get("records", []):
            cc = rec.get("country_code", rec.get("country", ""))
            assert cc in ("DE", "Germany")

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_empty_ecdc(self, mock_client_cls):
        mock_resp = _mock_resp([])
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="eu_surveillance")
        assert r.success
        assert "No data" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_country_no_match(self, mock_client_cls):
        records = [_make_ecdc_case("Germany", "DE")]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="eu_surveillance", dataset="cases", country="XX")
        assert r.success
        assert "No data" in r.output


class TestECDCErrors:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_ecdc_429(self, mock_client_cls):
        mock_resp = _mock_resp({}, status=429)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="eu_surveillance")
        assert not r.success
        assert "Rate limited" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_ecdc_timeout(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=httpx.TimeoutException("timeout"),
        )

        r = _tool().execute(mode="eu_surveillance")
        assert not r.success
        assert "timed out" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_ecdc_non_list_response(self, mock_client_cls):
        mock_resp = _mock_resp({"error": "not found"})
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="eu_surveillance")
        assert not r.success
        assert "Unexpected" in r.output


# ── 9. Genomics Mode — NCBI ──────────────────────────────────


class TestGenomicsMode:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_basic_fetch(self, mock_client_cls):
        """Two NCBI calls: current year + prior year."""
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=[
                _mock_resp(_make_ncbi_response(12000)),  # current year
                _mock_resp(_make_ncbi_response(50000)),  # prior year
            ]
        )

        r = _tool().execute(mode="genomics")
        assert r.success
        assert r.data["current_count"] == 12000
        assert r.data["prior_count"] == 50000
        assert "SARS-CoV-2" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_custom_organism(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=[
                _mock_resp(_make_ncbi_response(500)),
                _mock_resp(_make_ncbi_response(200)),
            ]
        )

        r = _tool().execute(mode="genomics", organism="H5N1")
        assert r.success
        assert "H5N1" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_accelerating_signal(self, mock_client_cls):
        """High current / low prior → ACCELERATING."""
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=[
                _mock_resp(_make_ncbi_response(100000)),  # current partial year
                _mock_resp(_make_ncbi_response(10000)),  # prior full year
            ]
        )

        r = _tool().execute(mode="genomics")
        assert r.success
        assert r.data["signal"] == "ACCELERATING"
        assert "ACCELERATING" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_declining_signal(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=[
                _mock_resp(_make_ncbi_response(10)),  # current very low
                _mock_resp(_make_ncbi_response(100000)),  # prior high
            ]
        )

        r = _tool().execute(mode="genomics")
        assert r.success
        assert r.data["signal"] == "DECLINING"

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_no_baseline(self, mock_client_cls):
        """Prior count = 0 → NO_BASELINE."""
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=[
                _mock_resp(_make_ncbi_response(100)),
                _mock_resp(_make_ncbi_response(0)),
            ]
        )

        r = _tool().execute(mode="genomics")
        assert r.success
        assert r.data["signal"] == "NO_BASELINE"

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_default_organism(self, mock_client_cls):
        """Empty organism defaults to SARS-CoV-2."""
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=[
                _mock_resp(_make_ncbi_response(1000)),
                _mock_resp(_make_ncbi_response(1000)),
            ]
        )

        r = _tool().execute(mode="genomics", organism="")
        assert r.success
        assert r.data["organism"] == "SARS-CoV-2"


class TestNCBIErrors:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_ncbi_429(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            return_value=_mock_resp({}, status=429),
        )

        r = _tool().execute(mode="genomics")
        assert not r.success
        assert "Rate limited" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_ncbi_timeout(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=httpx.TimeoutException("timeout"),
        )

        r = _tool().execute(mode="genomics")
        assert not r.success
        assert "timed out" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_ncbi_malformed_count(self, mock_client_cls):
        """Count field is not parseable as int."""
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            return_value=_mock_resp(
                {
                    "esearchresult": {"count": "not_a_number"},
                }
            ),
        )

        r = _tool().execute(mode="genomics")
        assert not r.success
        assert "Could not parse" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_ncbi_second_call_fails(self, mock_client_cls):
        """First NCBI call succeeds, second fails."""
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=[
                _mock_resp(_make_ncbi_response(1000)),
                _mock_resp({}, status=500),
            ]
        )

        r = _tool().execute(mode="genomics")
        assert not r.success
        assert "500" in r.output


# ── 10. Cache Interaction ─────────────────────────────────────


class TestCacheInteraction:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_cache_miss_then_put(self, mock_client_cls):
        cache = MagicMock()
        cache.get.return_value = None

        records = [_make_cdc_record("CA")]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool(cache=cache).execute(mode="wastewater", pathogen="covid")
        assert r.success
        cache.get.assert_called_once()
        cache.put.assert_called_once()

    def test_cache_hit_skips_fetch(self):
        cache = MagicMock()
        cache.get.return_value = [_make_cdc_record("CA")]

        r = _tool(cache=cache).execute(mode="wastewater", pathogen="covid")
        assert r.success
        # No HTTP call should be made
        cache.get.assert_called()

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_empty_data_not_cached(self, mock_client_cls):
        cache = MagicMock()
        cache.get.return_value = None

        mock_resp = _mock_resp([])
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool(cache=cache).execute(mode="wastewater", pathogen="covid")
        assert r.success
        cache.put.assert_not_called()

    def test_who_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = [
            _make_who_entry("Cholera - Haiti"),
        ]

        r = _tool(cache=cache).execute(mode="outbreaks")
        assert r.success
        assert r.data["count"] == 1

    def test_ecdc_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = [_make_ecdc_case()]

        r = _tool(cache=cache).execute(mode="eu_surveillance")
        assert r.success

    def test_ncbi_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = {
            "organism": "SARS-CoV-2",
            "current_year": 2026,
            "current_count": 10000,
            "prior_year": 2025,
            "prior_count": 50000,
            "signal": "DECLINING",
        }

        r = _tool(cache=cache).execute(mode="genomics")
        assert r.success
        assert "DECLINING" in r.output


# ── 11. Output Formatting ────────────────────────────────────


class TestOutputFormatting:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_wastewater_output_has_header(self, mock_client_cls):
        records = [_make_cdc_record("CA")]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="covid")
        assert "CDC NWSS Wastewater" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_outbreaks_output_has_header(self, mock_client_cls):
        entries = [_make_who_entry()]
        mock_resp = _mock_resp({"value": entries})
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="outbreaks")
        assert "WHO Disease Outbreak News" in r.output

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_genomics_output_has_header(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(
            side_effect=[
                _mock_resp(_make_ncbi_response(100)),
                _mock_resp(_make_ncbi_response(100)),
            ]
        )

        r = _tool().execute(mode="genomics")
        assert "NCBI GenBank" in r.output

    def test_toolresult_data_always_dict(self):
        """ToolResult.data should always be a dict when present."""
        cache = MagicMock()
        cache.get.return_value = [_make_cdc_record("CA")]

        r = _tool(cache=cache).execute(mode="wastewater", pathogen="covid")
        assert isinstance(r.data, dict)


# ── 12. Data Constants ────────────────────────────────────────


class TestDataConstants:
    def test_six_pathogen_datasets(self):
        assert len(_CDC_DATASETS) == 6

    def test_dataset_ids_are_strings(self):
        for name, did in _CDC_DATASETS.items():
            assert isinstance(did, str)
            assert len(did) > 5

    def test_aggregate_id(self):
        assert isinstance(_CDC_AGGREGATE_ID, str)
        assert len(_CDC_AGGREGATE_ID) > 5

    def test_ecdc_datasets(self):
        assert len(_ECDC_DATASETS) == 3
        assert set(_ECDC_DATASETS.keys()) == {"cases", "variants", "hospital"}

    def test_us_states_count(self):
        assert len(_US_STATES) == 51  # 50 states + DC

    def test_pathogen_aliases_resolve(self):
        for alias, canonical in _PATHOGEN_ALIASES.items():
            assert canonical in _CDC_DATASETS


# ── 13. Registry + Bandit Integration ─────────────────────────


class TestRegistryIntegration:
    def test_cli_registration(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        names = registry.list_names()
        assert "disease_surveillance" in names

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm_names = [a.name for a in DEFAULT_ARMS]
        assert "pandemic_surveillance" in arm_names

    def test_bandit_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "pandemic_surveillance")
        assert "disease_surveillance" in arm.tools
        assert "weather_alerts" in arm.tools

    def test_bandit_arm_examples(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "pandemic_surveillance")
        assert len(arm.examples) >= 3


# ── 14. Mode Routing ─────────────────────────────────────────


class TestModeRouting:
    def test_wastewater_routes_correctly(self):
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_wastewater",
            return_value=ToolResult(success=True, output="ww"),
        ) as m:
            r = _tool().execute(mode="wastewater")
            m.assert_called_once()
            assert r.output == "ww"

    def test_outbreaks_routes_correctly(self):
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_outbreaks",
            return_value=ToolResult(success=True, output="ob"),
        ) as m:
            r = _tool().execute(mode="outbreaks")
            m.assert_called_once()
            assert r.output == "ob"

    def test_eu_surveillance_routes_correctly(self):
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_eu_surveillance",
            return_value=ToolResult(success=True, output="eu"),
        ) as m:
            r = _tool().execute(mode="eu_surveillance")
            m.assert_called_once()
            assert r.output == "eu"

    def test_genomics_routes_correctly(self):
        with patch.object(
            DiseaseSurveillanceTool,
            "_execute_genomics",
            return_value=ToolResult(success=True, output="gen"),
        ) as m:
            r = _tool().execute(mode="genomics")
            m.assert_called_once()
            assert r.output == "gen"


# ── 15. Edge Case Combinations ───────────────────────────────


class TestEdgeCombinations:
    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_all_pathogen_keys_fetchable(self, mock_client_cls):
        """Each pathogen key should produce a valid Socrata URL."""
        records = [_make_cdc_record("CA")]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        for key in _CDC_DATASETS:
            r = _tool().execute(mode="wastewater", pathogen=key)
            assert r.success, f"Failed for pathogen={key}"

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_all_ecdc_datasets_fetchable(self, mock_client_cls):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        for ds_key in _ECDC_DATASETS:
            if ds_key == "cases":
                mock_resp = _mock_resp([_make_ecdc_case()])
            elif ds_key == "variants":
                mock_resp = _mock_resp([_make_ecdc_variant()])
            else:
                mock_resp = _mock_resp([_make_ecdc_hospital()])

            mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)
            r = _tool().execute(mode="eu_surveillance", dataset=ds_key)
            assert r.success, f"Failed for dataset={ds_key}"

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_wastewater_with_state_and_pathogen(self, mock_client_cls):
        records = [_make_cdc_record("TX")]
        mock_resp = _mock_resp(records)
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="wastewater", pathogen="h5n1", state="TX")
        assert r.success

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_who_entries_with_no_title(self, mock_client_cls):
        entries = [{"PublicationDate": "2026-01-01", "DonId": "1"}]
        mock_resp = _mock_resp({"value": entries})
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="outbreaks")
        assert r.success  # Should not crash on missing Title

    @patch("agent.tools.disease_surveillance.httpx.Client")
    def test_ecdc_variant_no_percent(self, mock_client_cls):
        rec = _make_ecdc_variant()
        del rec["percent_variant"]
        mock_resp = _mock_resp([rec])
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.get = MagicMock(return_value=mock_resp)

        r = _tool().execute(mode="eu_surveillance", dataset="variants")
        assert r.success  # Should handle missing percent gracefully
