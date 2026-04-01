"""
Edge case tests for MacroDataTool — multi-source expansion (FRED + ECB + World Bank).

Covers: source validation, ECB alias resolution, ECB SDMX JSON parsing,
World Bank indicator fetching, country filtering, error handling, timeout,
empty/malformed responses, caching, output formatting, backward compatibility.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.macro_data import (
    MacroDataTool,
    VALID_SOURCES,
    _ECB_ALIASES,
    _ECB_BASE,
    _WB_BASE,
)
from agent.tools.base import ToolResult


# ── Fixtures ──────────────────────────────────────────────────


def _tool(api_key: str = "test_key", cache=None) -> MacroDataTool:
    return MacroDataTool(fred_api_key=api_key, cache=cache)


def _tool_nokey(cache=None) -> MacroDataTool:
    return MacroDataTool(fred_api_key="", cache=cache)


SAMPLE_FRED_RESPONSE = {
    "observations": [
        {"date": "2024-01-01", "value": "100.5"},
        {"date": "2024-02-01", "value": "101.2"},
        {"date": "2024-03-01", "value": "."},
        {"date": "2024-04-01", "value": "102.0"},
    ]
}

SAMPLE_ECB_SDMX_RESPONSE = {
    "dataSets": [
        {
            "series": {
                "0:0:0:0:0": {
                    "observations": {
                        "0": [1.0845],
                        "1": [1.0950],
                        "2": [1.1020],
                    }
                }
            }
        }
    ],
    "structure": {
        "dimensions": {
            "observation": [
                {
                    "values": [
                        {"id": "2024-01-02", "name": "2024-01-02"},
                        {"id": "2024-01-03", "name": "2024-01-03"},
                        {"id": "2024-01-04", "name": "2024-01-04"},
                    ]
                }
            ]
        }
    },
}

SAMPLE_ECB_MULTI_SERIES = {
    "dataSets": [
        {
            "series": {
                "0:0:0:0:0": {
                    "observations": {
                        "0": [1.0845],
                        "1": [1.0950],
                    }
                },
                "0:1:0:0:0": {
                    "observations": {
                        "0": [0.8530],
                        "1": [0.8610],
                    }
                },
            }
        }
    ],
    "structure": {
        "dimensions": {
            "observation": [
                {
                    "values": [
                        {"id": "2024-03-01", "name": "2024-03-01"},
                        {"id": "2024-03-02", "name": "2024-03-02"},
                    ]
                }
            ]
        }
    },
}

SAMPLE_WB_RESPONSE = [
    {
        "page": 1,
        "pages": 1,
        "per_page": 300,
        "total": 3,
    },
    [
        {
            "country": {"id": "US", "value": "United States"},
            "countryiso3code": "USA",
            "date": "2023",
            "value": 2.54e13,
            "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
        },
        {
            "country": {"id": "GB", "value": "United Kingdom"},
            "countryiso3code": "GBR",
            "date": "2023",
            "value": 3.09e12,
            "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
        },
        {
            "country": {"id": "DE", "value": "Germany"},
            "countryiso3code": "DEU",
            "date": "2023",
            "value": 4.46e12,
            "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
        },
    ],
]

SAMPLE_WB_SINGLE_COUNTRY = [
    {"page": 1, "pages": 1, "per_page": 300, "total": 2},
    [
        {
            "country": {"id": "US", "value": "United States"},
            "countryiso3code": "USA",
            "date": "2022",
            "value": 2.51e13,
            "indicator": {"id": "NY.GDP.MKTP.CD"},
        },
        {
            "country": {"id": "US", "value": "United States"},
            "countryiso3code": "USA",
            "date": "2023",
            "value": 2.54e13,
            "indicator": {"id": "NY.GDP.MKTP.CD"},
        },
    ],
]


# ══════════════════════════════════════════════════════════════
# 1. SOURCE PARAMETER
# ══════════════════════════════════════════════════════════════


class TestSourceParameter:
    def test_valid_sources_constant(self):
        assert VALID_SOURCES == {"fred", "ecb", "world_bank"}

    def test_invalid_source_rejected(self):
        r = _tool().execute(series_id="GDP", source="bloomberg")
        assert not r.success
        assert "Invalid source" in r.output

    def test_empty_source_defaults_to_fred(self):
        """Empty source string → FRED (backward compatible)."""
        r = _tool_nokey().execute(series_id="GDP", source="")
        assert not r.success
        assert "FRED API key" in r.output  # Confirms it tried FRED path

    def test_none_source_defaults_to_fred(self):
        r = _tool_nokey().execute(series_id="GDP")
        assert not r.success
        assert "FRED API key" in r.output

    def test_source_case_insensitive(self):
        with patch.object(MacroDataTool, "_fetch_ecb", return_value=[]):
            r = _tool().execute(series_id="EURUSD", source="ECB")
            # Should not fail with "invalid source", may fail with "No data"
            assert "Invalid source" not in r.output

    def test_source_whitespace_stripped(self):
        with patch.object(MacroDataTool, "_fetch_ecb", return_value=[]):
            r = _tool().execute(series_id="EURUSD", source="  ecb  ")
            assert "Invalid source" not in r.output

    def test_source_in_parameters_schema(self):
        params = _tool().parameters
        assert "source" in params["properties"]
        src = params["properties"]["source"]
        assert src["type"] == "string"
        assert set(src["enum"]) == {"ecb", "fred", "world_bank"}

    def test_country_in_parameters_schema(self):
        params = _tool().parameters
        assert "country" in params["properties"]


# ══════════════════════════════════════════════════════════════
# 2. FRED BACKWARD COMPATIBILITY
# ══════════════════════════════════════════════════════════════


class TestFREDBackwardCompat:
    def test_no_api_key_error(self):
        r = _tool_nokey().execute(series_id="GDP")
        assert not r.success
        assert "FRED API key" in r.output

    def test_basic_fred_fetch(self):
        with patch.object(
            MacroDataTool,
            "_fetch_series",
            return_value=SAMPLE_FRED_RESPONSE["observations"],
        ):
            r = _tool().execute(series_id="GDP")
            assert r.success
            assert "GDP" in r.data

    def test_fred_filters_missing_values(self):
        with patch.object(
            MacroDataTool,
            "_fetch_series",
            return_value=SAMPLE_FRED_RESPONSE["observations"],
        ):
            r = _tool().execute(series_id="GDP")
            # "." values should be filtered out
            for obs in r.data["GDP"]:
                assert obs["value"] != "."

    def test_fred_empty_series_id(self):
        r = _tool().execute(series_id="")
        assert not r.success

    def test_fred_comma_separated(self):
        with patch.object(
            MacroDataTool,
            "_fetch_series",
            return_value=SAMPLE_FRED_RESPONSE["observations"],
        ):
            r = _tool().execute(series_id="GDP, UNRATE")
            assert r.success
            assert "GDP" in r.data
            assert "UNRATE" in r.data

    def test_fred_explicit_source(self):
        with patch.object(
            MacroDataTool,
            "_fetch_series",
            return_value=SAMPLE_FRED_RESPONSE["observations"],
        ):
            r = _tool().execute(series_id="GDP", source="fred")
            assert r.success

    def test_fred_exception_handling(self):
        with patch.object(
            MacroDataTool,
            "_fetch_series",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = _tool().execute(series_id="GDP")
            # Should not crash — error is caught per-series
            assert "Error" in r.output


# ══════════════════════════════════════════════════════════════
# 3. ECB ALIAS RESOLUTION
# ══════════════════════════════════════════════════════════════


class TestECBAliases:
    def test_known_aliases_exist(self):
        assert "EURUSD" in _ECB_ALIASES
        assert "ECB_RATE" in _ECB_ALIASES
        assert "ECB_BALANCE_SHEET" in _ECB_ALIASES
        assert "HICP" in _ECB_ALIASES

    def test_alias_resolves_to_sdmx_path(self):
        assert "/" in _ECB_ALIASES["EURUSD"]
        assert _ECB_ALIASES["EURUSD"] == "EXR/D.USD.EUR.SP00.A"

    def test_unknown_alias_without_slash_rejected(self):
        r = _tool().execute(series_id="BOGUS_ALIAS", source="ecb")
        assert not r.success or "Unknown ECB alias" in r.output

    def test_direct_sdmx_path_accepted(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            return_value=[{"date": "2024-01-01", "value": "1.0"}],
        ):
            r = _tool().execute(series_id="EXR/D.USD.EUR.SP00.A", source="ecb")
            assert r.success

    def test_alias_case_insensitive(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            return_value=[{"date": "2024-01-01", "value": "1.0"}],
        ):
            r = _tool().execute(series_id="eurusd", source="ecb")
            assert r.success

    def test_comma_separated_ecb(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            return_value=[{"date": "2024-01-01", "value": "1.0"}],
        ):
            r = _tool().execute(series_id="EURUSD, ECB_RATE", source="ecb")
            assert r.success
            assert "EURUSD" in r.data
            assert "ECB_RATE" in r.data

    def test_mixed_alias_and_path(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            return_value=[{"date": "2024-01-01", "value": "1.0"}],
        ):
            r = _tool().execute(
                series_id="EURUSD, FM/B.U2.EUR.4F.KR.MFI.NWT",
                source="ecb",
            )
            assert r.success


# ══════════════════════════════════════════════════════════════
# 4. ECB SDMX JSON PARSING
# ══════════════════════════════════════════════════════════════


class TestECBParsing:
    def test_parse_basic_response(self):
        result = MacroDataTool._parse_ecb_sdmx_json(SAMPLE_ECB_SDMX_RESPONSE)
        assert len(result) == 3
        assert result[0]["date"] == "2024-01-02"
        assert result[0]["value"] == "1.0845"

    def test_parse_multi_series(self):
        result = MacroDataTool._parse_ecb_sdmx_json(SAMPLE_ECB_MULTI_SERIES)
        assert len(result) == 4  # 2 series × 2 obs each

    def test_parse_empty_datasets(self):
        data = {"dataSets": [], "structure": {"dimensions": {"observation": []}}}
        result = MacroDataTool._parse_ecb_sdmx_json(data)
        assert result == []

    def test_parse_no_datasets_key(self):
        result = MacroDataTool._parse_ecb_sdmx_json({})
        assert result == []

    def test_parse_no_observation_dims(self):
        data = {"dataSets": [{"series": {}}], "structure": {"dimensions": {}}}
        result = MacroDataTool._parse_ecb_sdmx_json(data)
        assert result == []

    def test_parse_empty_series(self):
        data = {
            "dataSets": [{"series": {}}],
            "structure": {
                "dimensions": {"observation": [{"values": [{"id": "2024-01-01"}]}]}
            },
        }
        result = MacroDataTool._parse_ecb_sdmx_json(data)
        assert result == []

    def test_parse_none_value_skipped(self):
        data = {
            "dataSets": [
                {
                    "series": {
                        "0:0": {
                            "observations": {
                                "0": [None],
                                "1": [1.05],
                            }
                        }
                    }
                }
            ],
            "structure": {
                "dimensions": {
                    "observation": [
                        {
                            "values": [
                                {"id": "2024-01-01"},
                                {"id": "2024-01-02"},
                            ]
                        }
                    ]
                }
            },
        }
        result = MacroDataTool._parse_ecb_sdmx_json(data)
        assert len(result) == 1
        assert result[0]["value"] == "1.05"

    def test_parse_empty_observation_list_skipped(self):
        data = {
            "dataSets": [
                {
                    "series": {
                        "0:0": {
                            "observations": {
                                "0": [],
                            }
                        }
                    }
                }
            ],
            "structure": {
                "dimensions": {"observation": [{"values": [{"id": "2024-01-01"}]}]}
            },
        }
        result = MacroDataTool._parse_ecb_sdmx_json(data)
        assert result == []

    def test_parse_sorted_by_date(self):
        result = MacroDataTool._parse_ecb_sdmx_json(SAMPLE_ECB_SDMX_RESPONSE)
        dates = [r["date"] for r in result]
        assert dates == sorted(dates)

    def test_values_are_strings(self):
        result = MacroDataTool._parse_ecb_sdmx_json(SAMPLE_ECB_SDMX_RESPONSE)
        for obs in result:
            assert isinstance(obs["value"], str)
            assert isinstance(obs["date"], str)


# ══════════════════════════════════════════════════════════════
# 5. ECB EXECUTE END-TO-END
# ══════════════════════════════════════════════════════════════


class TestECBExecute:
    def test_basic_ecb_fetch(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            return_value=[
                {"date": "2024-01-02", "value": "1.0845"},
                {"date": "2024-01-03", "value": "1.0950"},
            ],
        ):
            r = _tool().execute(series_id="EURUSD", source="ecb")
            assert r.success
            assert "EURUSD" in r.data
            assert len(r.data["EURUSD"]) == 2

    def test_ecb_no_data(self):
        with patch.object(MacroDataTool, "_fetch_ecb", return_value=[]):
            r = _tool().execute(series_id="EURUSD", source="ecb")
            assert not r.success or "No data" in r.output

    def test_ecb_empty_series_id(self):
        r = _tool().execute(series_id="", source="ecb")
        assert not r.success

    def test_ecb_output_format(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            return_value=[
                {"date": "2024-01-01", "value": "2.65"},
                {"date": "2024-02-01", "value": "2.40"},
            ],
        ):
            r = _tool().execute(series_id="ECB_RATE", source="ecb")
            assert "2 observations" in r.output
            assert "First:" in r.output
            assert "Last:" in r.output

    def test_ecb_does_not_require_api_key(self):
        """ECB source should work even without FRED API key."""
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            return_value=[{"date": "2024-01-01", "value": "1.0"}],
        ):
            r = _tool_nokey().execute(series_id="EURUSD", source="ecb")
            assert r.success

    def test_ecb_dates_passed_to_fetch(self):
        with patch.object(MacroDataTool, "_fetch_ecb", return_value=[]) as mock:
            _tool().execute(
                series_id="EURUSD",
                source="ecb",
                start_date="2024-01-01",
                end_date="2024-12-31",
            )
            mock.assert_called_once_with(
                "EXR/D.USD.EUR.SP00.A", "2024-01-01", "2024-12-31"
            )


# ══════════════════════════════════════════════════════════════
# 6. ECB ERROR HANDLING
# ══════════════════════════════════════════════════════════════


class TestECBErrors:
    def test_timeout(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = _tool().execute(series_id="EURUSD", source="ecb")
            assert "Error" in r.output

    def test_http_error(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            side_effect=httpx.HTTPStatusError(
                "404",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(404),
            ),
        ):
            r = _tool().execute(series_id="EURUSD", source="ecb")
            assert "Error" in r.output

    def test_connection_error(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            side_effect=httpx.ConnectError("fail"),
        ):
            r = _tool().execute(series_id="EURUSD", source="ecb")
            assert "Error" in r.output

    def test_generic_exception(self):
        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            side_effect=RuntimeError("boom"),
        ):
            r = _tool().execute(series_id="EURUSD", source="ecb")
            assert "Error" in r.output

    def test_partial_failure_multi_series(self):
        """One series fails, another succeeds — partial result."""
        call_count = 0

        def side_effect(sdmx_key, start, end):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("timeout")
            return [{"date": "2024-01-01", "value": "1.0"}]

        with patch.object(MacroDataTool, "_fetch_ecb", side_effect=side_effect):
            r = _tool().execute(series_id="EURUSD, ECB_RATE", source="ecb")
            # Should have partial success
            assert r.success  # at least one series worked
            assert "Error" in r.output
            assert "ECB_RATE" in r.data


# ══════════════════════════════════════════════════════════════
# 7. ECB CACHING
# ══════════════════════════════════════════════════════════════


class TestECBCaching:
    def test_cache_hit(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = {
            "summary": "2 obs",
            "data": [{"date": "2024-01-01", "value": "1.0"}],
        }

        r = _tool(cache=mock_cache).execute(series_id="EURUSD", source="ecb")
        assert r.success
        assert "cached" in r.output.lower()
        mock_cache.get.assert_called_once()

    def test_cache_miss_then_put(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch.object(
            MacroDataTool,
            "_fetch_ecb",
            return_value=[{"date": "2024-01-01", "value": "1.0"}],
        ):
            r = _tool(cache=mock_cache).execute(series_id="EURUSD", source="ecb")
            assert r.success
            mock_cache.put.assert_called_once()


# ══════════════════════════════════════════════════════════════
# 8. WORLD BANK BASIC
# ══════════════════════════════════════════════════════════════


class TestWorldBankBasic:
    def test_basic_wb_fetch(self):
        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            return_value=[
                {
                    "country_code": "USA",
                    "country": "United States",
                    "date": "2023",
                    "value": "25400000000000",
                },
                {
                    "country_code": "GBR",
                    "country": "United Kingdom",
                    "date": "2023",
                    "value": "3090000000000",
                },
            ],
        ):
            r = _tool().execute(series_id="NY.GDP.MKTP.CD", source="world_bank")
            assert r.success
            assert "USA" in r.data
            assert "GBR" in r.data

    def test_wb_does_not_require_api_key(self):
        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            return_value=[
                {
                    "country_code": "USA",
                    "country": "United States",
                    "date": "2023",
                    "value": "100",
                },
            ],
        ):
            r = _tool_nokey().execute(series_id="NY.GDP.MKTP.CD", source="world_bank")
            assert r.success

    def test_wb_empty_indicator(self):
        r = _tool().execute(series_id="", source="world_bank")
        assert not r.success

    def test_wb_whitespace_indicator(self):
        r = _tool().execute(series_id="   ", source="world_bank")
        assert not r.success

    def test_wb_no_data(self):
        with patch.object(MacroDataTool, "_fetch_world_bank", return_value=[]):
            r = _tool().execute(series_id="BOGUS.IND", source="world_bank")
            assert not r.success
            assert "No data" in r.output

    def test_wb_country_filter(self):
        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            return_value=[
                {
                    "country_code": "USA",
                    "country": "United States",
                    "date": "2023",
                    "value": "100",
                },
            ],
        ) as mock:
            r = _tool().execute(
                series_id="NY.GDP.MKTP.CD",
                source="world_bank",
                country="US",
            )
            assert r.success
            # Verify country was passed through
            mock.assert_called_once()
            args = mock.call_args[0]
            assert args[1] == "US"

    def test_wb_output_format(self):
        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            return_value=[
                {
                    "country_code": "USA",
                    "country": "United States",
                    "date": "2023",
                    "value": "100",
                },
                {
                    "country_code": "GBR",
                    "country": "United Kingdom",
                    "date": "2023",
                    "value": "50",
                },
            ],
        ):
            r = _tool().execute(series_id="NY.GDP.MKTP.CD", source="world_bank")
            assert "2 observations" in r.output
            assert "2 countries" in r.output

    def test_wb_data_grouped_by_country(self):
        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            return_value=[
                {"country_code": "USA", "country": "US", "date": "2022", "value": "90"},
                {
                    "country_code": "USA",
                    "country": "US",
                    "date": "2023",
                    "value": "100",
                },
                {"country_code": "GBR", "country": "UK", "date": "2023", "value": "50"},
            ],
        ):
            r = _tool().execute(series_id="NY.GDP.MKTP.CD", source="world_bank")
            assert len(r.data["USA"]) == 2
            assert len(r.data["GBR"]) == 1


# ══════════════════════════════════════════════════════════════
# 9. WORLD BANK RESPONSE PARSING
# ══════════════════════════════════════════════════════════════


class TestWorldBankParsing:
    def test_parse_full_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_WB_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = _tool()._fetch_world_bank("NY.GDP.MKTP.CD", "all", "", "")
            assert len(result) == 3
            assert result[0]["country_code"] in {"USA", "GBR", "DEU"}

    def test_parse_null_values_skipped(self):
        wb_data = [
            {"page": 1, "pages": 1, "total": 2},
            [
                {
                    "country": {"id": "US", "value": "US"},
                    "countryiso3code": "USA",
                    "date": "2023",
                    "value": 100,
                },
                {
                    "country": {"id": "US", "value": "US"},
                    "countryiso3code": "USA",
                    "date": "2022",
                    "value": None,
                },
            ],
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = wb_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = _tool()._fetch_world_bank("NY.GDP.MKTP.CD", "all", "", "")
            assert len(result) == 1

    def test_parse_invalid_format(self):
        """World Bank returns non-list (error XML etc)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "invalid"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = _tool()._fetch_world_bank("BOGUS", "all", "", "")
            assert result == []

    def test_parse_empty_records(self):
        wb_data = [{"page": 1, "pages": 0, "total": 0}, None]
        mock_resp = MagicMock()
        mock_resp.json.return_value = wb_data
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = _tool()._fetch_world_bank("NY.GDP.MKTP.CD", "all", "", "")
            assert result == []

    def test_results_sorted_by_country_and_date(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_WB_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = _tool()._fetch_world_bank("NY.GDP.MKTP.CD", "all", "", "")
            keys = [(r["country_code"], r["date"]) for r in result]
            assert keys == sorted(keys)


# ══════════════════════════════════════════════════════════════
# 10. WORLD BANK ERROR HANDLING
# ══════════════════════════════════════════════════════════════


class TestWorldBankErrors:
    def test_timeout(self):
        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = _tool().execute(series_id="NY.GDP.MKTP.CD", source="world_bank")
            assert not r.success
            assert "error" in r.output.lower()

    def test_http_error(self):
        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            side_effect=httpx.HTTPStatusError(
                "404",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(404),
            ),
        ):
            r = _tool().execute(series_id="NY.GDP.MKTP.CD", source="world_bank")
            assert not r.success

    def test_connection_error(self):
        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            side_effect=httpx.ConnectError("fail"),
        ):
            r = _tool().execute(series_id="NY.GDP.MKTP.CD", source="world_bank")
            assert not r.success

    def test_generic_exception(self):
        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            side_effect=RuntimeError("exploded"),
        ):
            r = _tool().execute(series_id="NY.GDP.MKTP.CD", source="world_bank")
            assert not r.success
            assert "error" in r.output.lower()


# ══════════════════════════════════════════════════════════════
# 11. WORLD BANK CACHING
# ══════════════════════════════════════════════════════════════


class TestWorldBankCaching:
    def test_cache_hit(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = {
            "summary": "3 obs",
            "data": {"USA": [{"date": "2023", "value": "100"}]},
        }

        r = _tool(cache=mock_cache).execute(
            series_id="NY.GDP.MKTP.CD",
            source="world_bank",
        )
        assert r.success
        assert "cached" in r.output.lower()

    def test_cache_miss_then_put(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch.object(
            MacroDataTool,
            "_fetch_world_bank",
            return_value=[
                {
                    "country_code": "USA",
                    "country": "US",
                    "date": "2023",
                    "value": "100",
                },
            ],
        ):
            r = _tool(cache=mock_cache).execute(
                series_id="NY.GDP.MKTP.CD",
                source="world_bank",
            )
            assert r.success
            mock_cache.put.assert_called_once()


# ══════════════════════════════════════════════════════════════
# 12. WORLD BANK OUTPUT TRUNCATION
# ══════════════════════════════════════════════════════════════


class TestWorldBankOutputTruncation:
    def test_many_countries_truncated(self):
        """Output summary truncates after 20 countries."""
        obs = [
            {
                "country_code": f"C{i:02d}",
                "country": f"Country {i}",
                "date": "2023",
                "value": str(i * 100),
            }
            for i in range(30)
        ]
        with patch.object(MacroDataTool, "_fetch_world_bank", return_value=obs):
            r = _tool().execute(series_id="SP.POP.TOTL", source="world_bank")
            assert r.success
            assert "more countries" in r.output


# ══════════════════════════════════════════════════════════════
# 13. TOOL METADATA
# ══════════════════════════════════════════════════════════════


class TestMacroDataMetadata:
    def test_name(self):
        assert _tool().name == "macro_data"

    def test_description_mentions_all_sources(self):
        desc = _tool().description
        assert "fred" in desc.lower()
        assert "ecb" in desc.lower()
        assert "world_bank" in desc.lower() or "world bank" in desc.lower()

    def test_parameters_required(self):
        assert "series_id" in _tool().parameters["required"]

    def test_series_id_mentions_ecb(self):
        desc = _tool().parameters["properties"]["series_id"]["description"]
        assert "ECB" in desc or "ecb" in desc.lower()

    def test_series_id_mentions_world_bank(self):
        desc = _tool().parameters["properties"]["series_id"]["description"]
        assert (
            "World Bank" in desc
            or "world_bank" in desc.lower()
            or "indicator" in desc.lower()
        )
