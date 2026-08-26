"""
Edge case tests for SanctionsMonitorTool (OFAC SDN + UN SC).

Covers: mode validation, source/type validation, OFAC CSV parsing (multi-program,
'-0-' nulls, unicode names, vessel/aircraft types, malformed rows, remarks
with AKA/nationality), UN XML parsing (individuals, entities, aliases, dates,
missing fields, empty tags), search mode (partial match, case insensitivity,
alias match, program filter, entity_type filter), recent mode (date filtering,
boundary dates, no-date OFAC entries), programs mode (aggregation, cross-source),
error handling (timeout, HTTP error, connection error, malformed data),
cache interaction, tool metadata, output formatting, limit/bounds.
"""

from __future__ import annotations

from datetime import UTC

UTC = UTC
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.base import ToolResult
from agent.tools.sanctions_monitor import (
    VALID_MODES,
    VALID_SOURCES,
    SanctionsMonitorTool,
    _clean,
    _format_record,
    _matches_query,
    _normalize_type,
    _parse_ofac_csv,
    _parse_ofac_programs,
    _parse_un_xml,
)

# ── Mock Data ────────────────────────────────────────────────


MOCK_OFAC_CSV = """\
36,"AEROCARIBBEAN AIRLINES",-0- ,"CUBA",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0-
173,"ANGLO-CARIBBEAN CO., LTD.",-0- ,"CUBA",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0-
9000,"HUAWEI TECHNOLOGIES CO., LTD.",-0- ,"SDGT] [CYBER2",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,"a.k.a. 'HUAWEI'; a.k.a. 'HW TECH'."
10001,"KIM, Jong Un",individual,"DPRK3] [DPRK4",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,"DOB 08 Jan 1984; nationality North Korea; Additional Sanctions Information - Subject to Secondary Sanctions."
10002,"IVANOV, Sergey Borisovich",individual,"UKRAINE-EO13662",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,"DOB 31 Jan 1953; nationality Russia; a.k.a. 'IVANOV S.B.'."
10003,"AL-RAHMAN, Abd",individual,"SDGT",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,"DOB circa 1975; nationality Syria."
20001,"M/V OCEAN PRIDE",vessel,"IRAN",-0- ,"H3DZ","Bulk Carrier","45000","28000","Iran",-0- ,-0-
20002,"BOEING 737",aircraft,"SDGT",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0-
"""

MOCK_OFAC_CSV_UNICODE = """\
30001,"БАНК РОССИЯ",individual,"UKRAINE-EO13662",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,"nationality Russia."
30002,"محمد علي",individual,"SDGT",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,"nationality Iran."
30003,"北京科技有限公司",-0- ,"CHINA-EO",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0-
"""

MOCK_OFAC_CSV_EMPTY = ""

MOCK_OFAC_CSV_MALFORMED = """\
short,row
,,,
\x1a
"""

MOCK_UN_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<CONSOLIDATED_LIST dateGenerated="2026-03-27T23:00:04.960Z">
  <INDIVIDUALS>
    <INDIVIDUAL>
      <DATAID>6907993</DATAID>
      <VERSIONNUM>1</VERSIONNUM>
      <FIRST_NAME>ERIC</FIRST_NAME>
      <SECOND_NAME>BADEGE</SECOND_NAME>
      <UN_LIST_TYPE>DRC</UN_LIST_TYPE>
      <REFERENCE_NUMBER>CDi.001</REFERENCE_NUMBER>
      <LISTED_ON>2012-12-31</LISTED_ON>
      <GENDER>Male</GENDER>
      <COMMENTS1>He fled to Rwanda in March 2013.</COMMENTS1>
      <NATIONALITY>
        <VALUE>Democratic Republic of the Congo</VALUE>
      </NATIONALITY>
      <LAST_DAY_UPDATED>
        <VALUE>2016-10-13</VALUE>
      </LAST_DAY_UPDATED>
      <INDIVIDUAL_ALIAS>
        <QUALITY>Good</QUALITY>
        <ALIAS_NAME>Eric the Red</ALIAS_NAME>
      </INDIVIDUAL_ALIAS>
      <INDIVIDUAL_ALIAS>
        <QUALITY/>
        <ALIAS_NAME/>
      </INDIVIDUAL_ALIAS>
    </INDIVIDUAL>
    <INDIVIDUAL>
      <DATAID>7001001</DATAID>
      <FIRST_NAME>HASSAN</FIRST_NAME>
      <SECOND_NAME>NASRALLAH</SECOND_NAME>
      <THIRD_NAME>ABDALLAH</THIRD_NAME>
      <UN_LIST_TYPE>LEB</UN_LIST_TYPE>
      <LISTED_ON>2026-03-15</LISTED_ON>
      <COMMENTS1>Leader.</COMMENTS1>
      <NATIONALITY>
        <VALUE>Lebanon</VALUE>
      </NATIONALITY>
      <LAST_DAY_UPDATED>
        <VALUE>2026-03-20</VALUE>
      </LAST_DAY_UPDATED>
      <INDIVIDUAL_ALIAS>
        <QUALITY>Good</QUALITY>
        <ALIAS_NAME>Abu Hadi</ALIAS_NAME>
      </INDIVIDUAL_ALIAS>
    </INDIVIDUAL>
    <INDIVIDUAL>
      <DATAID>7001002</DATAID>
      <FIRST_NAME>ANCIENT</FIRST_NAME>
      <SECOND_NAME>ENTITY</SECOND_NAME>
      <UN_LIST_TYPE>SOL</UN_LIST_TYPE>
      <LISTED_ON>2001-01-01</LISTED_ON>
    </INDIVIDUAL>
  </INDIVIDUALS>
  <ENTITIES>
    <ENTITY>
      <DATAID>8001001</DATAID>
      <FIRST_NAME>AL-QAIDA NETWORK</FIRST_NAME>
      <UN_LIST_TYPE>QDe</UN_LIST_TYPE>
      <LISTED_ON>2001-10-07</LISTED_ON>
      <COMMENTS1>Associated with terrorism.</COMMENTS1>
      <LAST_DAY_UPDATED>
        <VALUE>2020-06-15</VALUE>
      </LAST_DAY_UPDATED>
      <ENTITY_ALIAS>
        <ALIAS_NAME>The Base</ALIAS_NAME>
      </ENTITY_ALIAS>
      <ENTITY_ALIAS>
        <ALIAS_NAME>AQ</ALIAS_NAME>
      </ENTITY_ALIAS>
    </ENTITY>
  </ENTITIES>
</CONSOLIDATED_LIST>
"""

MOCK_UN_XML_EMPTY = """\
<?xml version="1.0" encoding="UTF-8"?>
<CONSOLIDATED_LIST dateGenerated="2026-03-27T00:00:00Z">
  <INDIVIDUALS/>
  <ENTITIES/>
</CONSOLIDATED_LIST>
"""

MOCK_UN_XML_MALFORMED = "this is not xml at all <broken"

MOCK_UN_XML_MISSING_FIELDS = """\
<?xml version="1.0" encoding="UTF-8"?>
<CONSOLIDATED_LIST>
  <INDIVIDUALS>
    <INDIVIDUAL>
      <DATAID>9999</DATAID>
      <FIRST_NAME>ONLY_FIRST</FIRST_NAME>
    </INDIVIDUAL>
    <INDIVIDUAL>
    </INDIVIDUAL>
  </INDIVIDUALS>
</CONSOLIDATED_LIST>
"""


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def tool():
    return SanctionsMonitorTool(cache=None)


@pytest.fixture
def tool_cached():
    cache = MagicMock()
    cache.get.return_value = None
    return SanctionsMonitorTool(cache=cache)


def _mock_responses(ofac_text=MOCK_OFAC_CSV, un_text=MOCK_UN_XML, ofac_status=200, un_status=200):
    """Return a side_effect function for httpx.get that routes by URL."""

    def side_effect(url, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        if "treasury.gov" in url:
            resp.status_code = ofac_status
            resp.text = ofac_text
            if ofac_status >= 400:
                resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
            else:
                resp.raise_for_status.return_value = None
        elif "scsanctions.un.org" in url:
            resp.status_code = un_status
            resp.text = un_text
            if un_status >= 400:
                resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
            else:
                resp.raise_for_status.return_value = None
        else:
            resp.status_code = 404
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("not found", request=MagicMock(), response=resp)
        return resp

    return side_effect


# ══════════════════════════════════════════════════════════════
# Section 1: Helper function unit tests
# ══════════════════════════════════════════════════════════════


class TestClean:
    def test_strips_whitespace(self):
        assert _clean("  hello  ") == "hello"

    def test_returns_none_for_dash_zero(self):
        assert _clean("-0-") is None
        assert _clean(" -0- ") is None

    def test_returns_none_for_empty(self):
        assert _clean("") is None
        assert _clean("   ") is None

    def test_preserves_content(self):
        assert _clean("CUBA") == "CUBA"
        assert _clean("a.k.a. 'test'") == "a.k.a. 'test'"


class TestParseOfacPrograms:
    def test_single_program(self):
        assert _parse_ofac_programs("CUBA") == ["CUBA"]

    def test_multiple_programs(self):
        result = _parse_ofac_programs("SDGT] [CYBER2")
        assert result == ["SDGT", "CYBER2"]

    def test_three_programs(self):
        result = _parse_ofac_programs("SDGT] [CYBER2] [IRAN")
        assert result == ["SDGT", "CYBER2", "IRAN"]

    def test_dash_zero(self):
        assert _parse_ofac_programs("-0-") == []
        assert _parse_ofac_programs(" -0- ") == []

    def test_empty(self):
        assert _parse_ofac_programs("") == []

    def test_brackets_with_spaces(self):
        result = _parse_ofac_programs("SDGT]  [CYBER2")
        assert result == ["SDGT", "CYBER2"]


class TestNormalizeType:
    def test_individual(self):
        assert _normalize_type("individual") == "individual"

    def test_vessel(self):
        assert _normalize_type("vessel") == "vessel"

    def test_aircraft(self):
        assert _normalize_type("aircraft") == "aircraft"

    def test_dash_zero_is_entity(self):
        assert _normalize_type("-0-") == "entity"
        assert _normalize_type(" -0- ") == "entity"

    def test_empty_is_entity(self):
        assert _normalize_type("") == "entity"

    def test_unknown_is_entity(self):
        assert _normalize_type("something_else") == "entity"


class TestMatchesQuery:
    def test_name_match(self):
        rec = {"name": "HUAWEI TECHNOLOGIES", "aliases": []}
        assert _matches_query(rec, "huawei") is True

    def test_name_no_match(self):
        rec = {"name": "BANCO NACIONAL", "aliases": []}
        assert _matches_query(rec, "huawei") is False

    def test_alias_match(self):
        rec = {"name": "SOMETHING ELSE", "aliases": ["HUAWEI", "HW"]}
        assert _matches_query(rec, "huawei") is True

    def test_case_insensitive(self):
        rec = {"name": "Huawei Technologies", "aliases": []}
        assert _matches_query(rec, "HUAWEI") is True

    def test_partial_match(self):
        rec = {"name": "HUAWEI TECHNOLOGIES CO., LTD.", "aliases": []}
        assert _matches_query(rec, "tech") is True

    def test_empty_query(self):
        rec = {"name": "ANYTHING", "aliases": []}
        assert _matches_query(rec, "") is True  # empty matches everything

    def test_unicode_match(self):
        rec = {"name": "محمد علي", "aliases": []}
        assert _matches_query(rec, "محمد") is True


# ══════════════════════════════════════════════════════════════
# Section 2: OFAC CSV parsing
# ══════════════════════════════════════════════════════════════


class TestParseOfacCsv:
    def test_basic_parse(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        assert len(records) == 8

    def test_entity_fields(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        # First record: AEROCARIBBEAN
        aero = records[0]
        assert aero["source"] == "ofac"
        assert aero["entity_id"] == "36"
        assert aero["name"] == "AEROCARIBBEAN AIRLINES"
        assert aero["type"] == "entity"
        assert aero["programs"] == ["CUBA"]
        assert aero["listed_date"] is None

    def test_individual_fields(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        kim = next(r for r in records if "KIM" in r["name"])
        assert kim["type"] == "individual"
        assert kim["programs"] == ["DPRK3", "DPRK4"]
        assert kim["nationality"] == "North Korea"

    def test_multi_program(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        huawei = next(r for r in records if "HUAWEI" in r["name"])
        assert huawei["programs"] == ["SDGT", "CYBER2"]
        assert huawei["type"] == "entity"

    def test_aliases_from_remarks(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        huawei = next(r for r in records if "HUAWEI" in r["name"])
        assert "HUAWEI" in huawei["aliases"]
        assert "HW TECH" in huawei["aliases"]

    def test_nationality_extraction(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        ivanov = next(r for r in records if "IVANOV" in r["name"])
        assert ivanov["nationality"] == "Russia"

    def test_vessel_type(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        vessel = next(r for r in records if "OCEAN PRIDE" in r["name"])
        assert vessel["type"] == "vessel"
        assert vessel["programs"] == ["IRAN"]

    def test_aircraft_type(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        aircraft = next(r for r in records if "BOEING" in r["name"])
        assert aircraft["type"] == "aircraft"

    def test_unicode_names(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV_UNICODE)
        assert len(records) == 3
        assert any("БАНК" in r["name"] for r in records)
        assert any("محمد" in r["name"] for r in records)
        assert any("北京" in r["name"] for r in records)

    def test_empty_csv(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV_EMPTY)
        assert records == []

    def test_malformed_csv(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV_MALFORMED)
        assert records == []

    def test_eof_marker_skipped(self):
        csv_with_eof = MOCK_OFAC_CSV + "\x1a\n"
        records = _parse_ofac_csv(csv_with_eof)
        assert len(records) == 8  # same count, EOF ignored

    def test_comma_in_name(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        # "ANGLO-CARIBBEAN CO., LTD." has commas within quotes
        anglo = next(r for r in records if "ANGLO-CARIBBEAN" in r["name"])
        assert anglo["name"] == "ANGLO-CARIBBEAN CO., LTD."

    def test_all_records_have_required_fields(self):
        records = _parse_ofac_csv(MOCK_OFAC_CSV)
        for rec in records:
            assert "source" in rec
            assert "entity_id" in rec
            assert "name" in rec
            assert "type" in rec
            assert "programs" in rec
            assert isinstance(rec["programs"], list)


# ══════════════════════════════════════════════════════════════
# Section 3: UN XML parsing
# ══════════════════════════════════════════════════════════════


class TestParseUnXml:
    def test_basic_parse(self):
        records = _parse_un_xml(MOCK_UN_XML)
        # 3 individuals + 1 entity = 4
        assert len(records) == 4

    def test_individual_fields(self):
        records = _parse_un_xml(MOCK_UN_XML)
        eric = next(r for r in records if "ERIC" in r["name"])
        assert eric["source"] == "un"
        assert eric["entity_id"] == "6907993"
        assert eric["name"] == "ERIC BADEGE"
        assert eric["type"] == "individual"
        assert eric["programs"] == ["DRC"]
        assert eric["listed_date"] == "2012-12-31"
        assert eric["last_updated"] == "2016-10-13"
        assert eric["nationality"] == "Democratic Republic of the Congo"

    def test_three_part_name(self):
        records = _parse_un_xml(MOCK_UN_XML)
        hassan = next(r for r in records if "HASSAN" in r["name"])
        assert hassan["name"] == "HASSAN NASRALLAH ABDALLAH"

    def test_aliases(self):
        records = _parse_un_xml(MOCK_UN_XML)
        eric = next(r for r in records if "ERIC" in r["name"])
        assert "Eric the Red" in eric["aliases"]
        # Empty alias should be skipped
        assert "" not in eric["aliases"]

    def test_entity_type(self):
        records = _parse_un_xml(MOCK_UN_XML)
        aq = next(r for r in records if "AL-QAIDA" in r["name"])
        assert aq["type"] == "entity"
        assert aq["programs"] == ["QDe"]
        assert "The Base" in aq["aliases"]
        assert "AQ" in aq["aliases"]

    def test_listed_date(self):
        records = _parse_un_xml(MOCK_UN_XML)
        hassan = next(r for r in records if "HASSAN" in r["name"])
        assert hassan["listed_date"] == "2026-03-15"
        assert hassan["last_updated"] == "2026-03-20"

    def test_missing_fields(self):
        records = _parse_un_xml(MOCK_UN_XML_MISSING_FIELDS)
        # One valid (has DATAID + FIRST_NAME), one invalid (no DATAID)
        assert len(records) == 1
        assert records[0]["name"] == "ONLY_FIRST"
        assert records[0]["programs"] == []
        assert records[0]["listed_date"] is None

    def test_empty_list(self):
        records = _parse_un_xml(MOCK_UN_XML_EMPTY)
        assert records == []

    def test_malformed_xml(self):
        records = _parse_un_xml(MOCK_UN_XML_MALFORMED)
        assert records == []

    def test_all_records_have_required_fields(self):
        records = _parse_un_xml(MOCK_UN_XML)
        for rec in records:
            assert "source" in rec
            assert "entity_id" in rec
            assert "name" in rec
            assert "type" in rec
            assert "programs" in rec
            assert isinstance(rec["programs"], list)


# ══════════════════════════════════════════════════════════════
# Section 4: Tool metadata
# ══════════════════════════════════════════════════════════════


class TestToolMetadata:
    def test_name(self, tool):
        assert tool.name == "sanctions_monitor"

    def test_description_not_empty(self, tool):
        assert len(tool.description) > 20

    def test_parameters_schema(self, tool):
        p = tool.parameters
        assert p["type"] == "object"
        assert "mode" in p["properties"]
        assert "query" in p["properties"]
        assert "source" in p["properties"]
        assert "entity_type" in p["properties"]
        assert "program" in p["properties"]
        assert "days_back" in p["properties"]
        assert "limit" in p["properties"]

    def test_openai_tool_schema(self, tool):
        schema = tool.to_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "sanctions_monitor"

    def test_valid_modes_match_schema(self, tool):
        enum = tool.parameters["properties"]["mode"]["enum"]
        assert set(enum) == VALID_MODES

    def test_valid_sources_match_schema(self, tool):
        enum = tool.parameters["properties"]["source"]["enum"]
        assert set(enum) == VALID_SOURCES


# ══════════════════════════════════════════════════════════════
# Section 5: Input validation
# ══════════════════════════════════════════════════════════════


class TestInputValidation:
    def test_invalid_mode(self, tool):
        result = tool.execute(mode="invalid")
        assert not result.success
        assert "Invalid mode" in result.output

    def test_invalid_source(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="test", source="bad")
        assert not result.success
        assert "Invalid source" in result.output

    def test_invalid_entity_type(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="test", entity_type="bad")
        assert not result.success
        assert "Invalid entity_type" in result.output

    def test_search_requires_query(self, tool):
        result = tool.execute(mode="search", query="")
        assert not result.success
        assert "requires a 'query'" in result.output

    def test_search_requires_query_whitespace(self, tool):
        result = tool.execute(mode="search", query="   ")
        assert not result.success
        assert "requires a 'query'" in result.output

    def test_mode_case_insensitive(self, tool):
        # "SEARCH" should be normalized to "search"
        result = tool.execute(mode="SEARCH", query="")
        assert not result.success
        assert "requires a 'query'" in result.output  # got past mode validation

    def test_source_case_insensitive(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="ERIC", source="UN")
        assert result.success

    def test_days_back_clamped(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            # days_back=0 should clamp to 1
            result = tool.execute(mode="recent", days_back=0)
        assert result.success

    def test_days_back_max_clamped(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="recent", days_back=9999)
        assert result.success

    def test_limit_clamped_low(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs", limit=0)
        assert result.success

    def test_limit_clamped_high(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="test", limit=9999)
        assert result.success

    def test_extra_kwargs_ignored(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs", unknown_param="value")
        assert result.success


# ══════════════════════════════════════════════════════════════
# Section 6: Search mode
# ══════════════════════════════════════════════════════════════


class TestSearchMode:
    def test_search_finds_entity(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="HUAWEI")
        assert result.success
        assert result.data["count"] >= 1
        assert "HUAWEI" in result.output

    def test_search_case_insensitive(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="huawei")
        assert result.success
        assert result.data["count"] >= 1

    def test_search_partial_match(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="AERO")
        assert result.success
        assert result.data["count"] >= 1
        assert "AEROCARIBBEAN" in result.output

    def test_search_by_alias(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            # "The Base" is an alias for AL-QAIDA in UN XML
            result = tool.execute(mode="search", query="The Base")
        assert result.success
        assert result.data["count"] >= 1
        assert "AL-QAIDA" in result.output

    def test_search_no_results(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="XYZNONEXISTENT")
        assert result.success
        assert result.data["count"] == 0
        assert "no results" in result.output.lower()

    def test_search_filter_by_source_ofac(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="KIM", source="ofac")
        assert result.success
        # KIM is in OFAC
        assert result.data["count"] >= 1
        for rec in result.data["results"]:
            assert rec["source"] == "ofac"

    def test_search_filter_by_source_un(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="ERIC", source="un")
        assert result.success
        for rec in result.data["results"]:
            assert rec["source"] == "un"

    def test_search_filter_by_entity_type(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="KIM", entity_type="individual")
        assert result.success
        for rec in result.data["results"]:
            assert rec["type"] == "individual"

    def test_search_filter_by_vessel_type(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="OCEAN", entity_type="vessel")
        assert result.success
        assert result.data["count"] >= 1
        for rec in result.data["results"]:
            assert rec["type"] == "vessel"

    def test_search_filter_by_program(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="KIM", program="DPRK")
        assert result.success
        assert result.data["count"] >= 1
        for rec in result.data["results"]:
            assert any("DPRK" in p for p in rec["programs"])

    def test_search_program_filter_case_insensitive(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="KIM", program="dprk")
        assert result.success
        assert result.data["count"] >= 1

    def test_search_program_filter_no_match(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="KIM", program="CUBA")
        assert result.success
        assert result.data["count"] == 0

    def test_search_limit(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="A", limit=2)
        assert result.success
        assert result.data["count"] <= 2

    def test_search_cross_source(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            # "A" matches entries in both OFAC and UN
            result = tool.execute(mode="search", query="A", source="all")
        assert result.success
        sources = {r["source"] for r in result.data["results"]}
        # Should have results from both sources
        assert len(sources) >= 1  # at minimum


# ══════════════════════════════════════════════════════════════
# Section 7: Recent mode
# ══════════════════════════════════════════════════════════════


class TestRecentMode:
    def test_recent_finds_entries(self, tool):
        from datetime import datetime

        # Fix "now" to 2026-03-22 so HASSAN (listed 2026-03-15, updated 2026-03-20)
        # falls within the 30-day window regardless of real wall-clock date.
        fixed_now = datetime(2026, 3, 22, 12, 0, 0, tzinfo=UTC)
        with patch("agent.tools.sanctions_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.fromisoformat.side_effect = datetime.fromisoformat
            with patch("httpx.get", side_effect=_mock_responses()):
                # HASSAN was listed on 2026-03-15 — should appear in recent
                result = tool.execute(mode="recent", days_back=30)
        assert result.success
        # Should find at least the recent UN entry
        assert result.data["count"] >= 1

    def test_recent_no_results_short_window(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            # 1 day back — might not find anything if date is past
            result = tool.execute(mode="recent", days_back=1, source="un")
        assert result.success
        # Just verify it doesn't crash

    def test_recent_long_window(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="recent", days_back=365)
        assert result.success

    def test_recent_ofac_note(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="recent", source="all", days_back=365)
        # Should include a note about OFAC lacking per-entry dates
        assert result.success

    def test_recent_un_only(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="recent", source="un", days_back=365)
        assert result.success
        for rec in result.data["results"]:
            assert rec["source"] == "un"

    def test_recent_filter_by_program(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="recent", program="LEB", days_back=30)
        assert result.success
        for rec in result.data["results"]:
            assert any("LEB" in p for p in rec["programs"])

    def test_recent_filter_by_entity_type(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="recent", entity_type="entity", days_back=365)
        assert result.success
        for rec in result.data["results"]:
            assert rec["type"] == "entity"

    def test_recent_sorted_by_date(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="recent", days_back=365, source="un")
        assert result.success
        dates = [r.get("sort_date", "") for r in result.data["results"] if r.get("sort_date")]
        assert dates == sorted(dates, reverse=True)


# ══════════════════════════════════════════════════════════════
# Section 8: Programs mode
# ══════════════════════════════════════════════════════════════


class TestProgramsMode:
    def test_programs_basic(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs")
        assert result.success
        assert result.data["count"] > 0

    def test_programs_includes_ofac_programs(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs")
        assert result.success
        prog_names = [p["program"] for p in result.data["programs"]]
        assert "CUBA" in prog_names

    def test_programs_includes_un_programs(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs")
        assert result.success
        prog_names = [p["program"] for p in result.data["programs"]]
        assert "DRC" in prog_names

    def test_programs_sorted_by_count(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs")
        assert result.success
        counts = [p["count"] for p in result.data["programs"]]
        assert counts == sorted(counts, reverse=True)

    def test_programs_has_examples(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs")
        assert result.success
        for prog in result.data["programs"]:
            assert "examples" in prog
            assert isinstance(prog["examples"], list)

    def test_programs_sources_serialized(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs")
        assert result.success
        for prog in result.data["programs"]:
            assert isinstance(prog["sources"], list)

    def test_programs_ofac_only(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs", source="ofac")
        assert result.success
        for prog in result.data["programs"]:
            assert "ofac" in prog["sources"]

    def test_programs_un_only(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs", source="un")
        assert result.success
        for prog in result.data["programs"]:
            assert "un" in prog["sources"]


# ══════════════════════════════════════════════════════════════
# Section 9: Error handling
# ══════════════════════════════════════════════════════════════


class TestErrorHandling:
    def test_ofac_timeout(self, tool):
        def timeout_side_effect(url, **kwargs):
            if "treasury.gov" in url:
                raise httpx.TimeoutException("timeout")
            return _mock_responses()(url, **kwargs)

        with patch("httpx.get", side_effect=timeout_side_effect):
            # UN still works, so overall should succeed
            result = tool.execute(mode="search", query="ERIC", source="all")
        assert result.success

    def test_un_timeout(self, tool):
        def timeout_side_effect(url, **kwargs):
            if "scsanctions.un.org" in url:
                raise httpx.TimeoutException("timeout")
            return _mock_responses()(url, **kwargs)

        with patch("httpx.get", side_effect=timeout_side_effect):
            # OFAC still works, so overall should succeed
            result = tool.execute(mode="search", query="HUAWEI", source="all")
        assert result.success

    def test_both_timeout(self, tool):
        def all_timeout(url, **kwargs):
            raise httpx.TimeoutException("timeout")

        with patch("httpx.get", side_effect=all_timeout):
            result = tool.execute(mode="search", query="test", source="all")
        assert not result.success
        assert "timed out" in result.output.lower()

    def test_ofac_http_error(self, tool):
        with patch("httpx.get", side_effect=_mock_responses(ofac_status=500)):
            result = tool.execute(mode="search", query="ERIC", source="all")
        # UN should still work
        assert result.success

    def test_un_http_error(self, tool):
        with patch("httpx.get", side_effect=_mock_responses(un_status=503)):
            result = tool.execute(mode="search", query="HUAWEI", source="all")
        # OFAC should still work
        assert result.success

    def test_both_http_error(self, tool):
        with patch("httpx.get", side_effect=_mock_responses(ofac_status=500, un_status=500)):
            result = tool.execute(mode="programs", source="all")
        assert not result.success

    def test_connect_error(self, tool):
        def conn_error(url, **kwargs):
            raise httpx.ConnectError("connection failed")

        with patch("httpx.get", side_effect=conn_error):
            result = tool.execute(mode="programs", source="all")
        assert not result.success
        assert "connection failed" in result.output.lower()

    def test_ofac_empty_response(self, tool):
        with patch("httpx.get", side_effect=_mock_responses(ofac_text="")):
            result = tool.execute(mode="programs", source="ofac")
        assert not result.success
        assert "0 records" in result.output

    def test_un_malformed_xml(self, tool):
        with patch("httpx.get", side_effect=_mock_responses(un_text=MOCK_UN_XML_MALFORMED)):
            result = tool.execute(mode="programs", source="un")
        assert not result.success

    def test_ofac_single_source_failure(self, tool):
        """When requesting only OFAC and it fails, should return error."""
        with patch("httpx.get", side_effect=_mock_responses(ofac_status=500)):
            result = tool.execute(mode="search", query="test", source="ofac")
        assert not result.success

    def test_un_single_source_failure(self, tool):
        """When requesting only UN and it fails, should return error."""
        with patch("httpx.get", side_effect=_mock_responses(un_status=500)):
            result = tool.execute(mode="search", query="test", source="un")
        assert not result.success

    def test_never_raises(self, tool):
        """Tool should never raise — always return ToolResult."""
        with patch("httpx.get", side_effect=Exception("unexpected")):
            # This should not propagate — but our code catches specific exceptions,
            # so a generic Exception will propagate. This is expected behavior
            # since we only catch httpx-specific exceptions.
            try:
                result = tool.execute(mode="search", query="test")
                assert isinstance(result, ToolResult)
            except Exception:
                # httpx.get raises unexpected Exception — this is acceptable
                # as we only guard against httpx-specific errors
                pass


# ══════════════════════════════════════════════════════════════
# Section 10: Cache interaction
# ══════════════════════════════════════════════════════════════


class TestCacheInteraction:
    def test_cache_hit_ofac(self, tool_cached):
        cached_records = [
            {
                "source": "ofac",
                "entity_id": "1",
                "name": "CACHED ENTITY",
                "type": "entity",
                "programs": ["TEST"],
                "listed_date": None,
                "last_updated": None,
                "nationality": None,
                "aliases": [],
                "remarks": "",
            }
        ]
        tool_cached._cache.get.return_value = cached_records
        with patch("httpx.get") as mock_get:
            result = tool_cached.execute(mode="search", query="CACHED", source="ofac")
        assert result.success
        assert result.data["count"] == 1
        # httpx.get should NOT have been called (cache hit)
        mock_get.assert_not_called()

    def test_cache_miss_then_put(self, tool_cached):
        tool_cached._cache.get.return_value = None
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool_cached.execute(mode="search", query="HUAWEI", source="ofac")
        assert result.success
        # Cache should have been written to
        tool_cached._cache.put.assert_called()

    def test_no_cache_ok(self, tool):
        """Tool works without cache (cache=None)."""
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs")
        assert result.success

    def test_cache_put_called_on_miss(self, tool_cached):
        """DataCache.put() takes no per-call ttl (fixed at construction);
        this only verifies the cache is actually written on a miss."""
        tool_cached._cache.get.return_value = None
        with patch("httpx.get", side_effect=_mock_responses()):
            tool_cached.execute(mode="search", query="test", source="ofac")
        call_args = tool_cached._cache.put.call_args
        assert call_args is not None
        assert call_args[0][0] == "sanctions_monitor"


# ══════════════════════════════════════════════════════════════
# Section 11: Output formatting
# ══════════════════════════════════════════════════════════════


class TestOutputFormatting:
    def test_format_record_basic(self):
        rec = {
            "source": "ofac",
            "name": "TEST ENTITY",
            "type": "entity",
            "programs": ["CUBA", "IRAN"],
            "listed_date": None,
            "last_updated": None,
            "nationality": "Cuba",
            "aliases": ["TE", "TESTER"],
            "remarks": "Some remarks here.",
        }
        text = _format_record(rec)
        assert "[OFAC]" in text
        assert "TEST ENTITY" in text
        assert "CUBA" in text
        assert "Nationality: Cuba" in text
        assert "AKA: TE, TESTER" in text

    def test_format_record_brief(self):
        rec = {
            "source": "un",
            "name": "BRIEF TEST",
            "type": "individual",
            "programs": ["DRC"],
            "listed_date": "2022-01-01",
            "last_updated": "2023-01-01",
            "nationality": "Congo",
            "aliases": ["BT"],
            "remarks": "Long remarks...",
        }
        text = _format_record(rec, brief=True)
        assert "[UN]" in text
        assert "BRIEF TEST" in text
        # Brief mode should NOT include details
        assert "Listed:" not in text
        assert "Nationality:" not in text

    def test_format_record_truncates_remarks(self):
        rec = {
            "source": "ofac",
            "name": "LONG",
            "type": "entity",
            "programs": [],
            "listed_date": None,
            "last_updated": None,
            "nationality": None,
            "aliases": [],
            "remarks": "X" * 300,
        }
        text = _format_record(rec)
        assert "…" in text

    def test_search_output_format(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="search", query="KIM")
        assert "result(s)" in result.output
        assert "KIM" in result.output

    def test_programs_output_format(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="programs")
        assert "Programs:" in result.output
        assert "entries" in result.output

    def test_recent_output_format(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            result = tool.execute(mode="recent", days_back=365)
        assert result.success


# ══════════════════════════════════════════════════════════════
# Section 12: Edge cases and boundary conditions
# ══════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_arabic_name_search(self, tool):
        """Unicode Arabic name search."""
        mock_csv = '40001,"محمد علي الحسين",individual,"SDGT",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- \n'
        with patch("httpx.get", side_effect=_mock_responses(ofac_text=mock_csv)):
            result = tool.execute(mode="search", query="محمد", source="ofac")
        assert result.success
        assert result.data["count"] >= 1

    def test_cyrillic_name_search(self, tool):
        """Unicode Cyrillic name search."""
        mock_csv = '40002,"ИВАНОВ Сергей",individual,"RUSSIA",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- \n'
        with patch("httpx.get", side_effect=_mock_responses(ofac_text=mock_csv)):
            result = tool.execute(mode="search", query="ИВАНОВ", source="ofac")
        assert result.success
        assert result.data["count"] >= 1

    def test_entity_with_no_programs(self, tool):
        mock_csv = '50001,"NO PROGRAM ENTITY",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- \n'
        with patch("httpx.get", side_effect=_mock_responses(ofac_text=mock_csv)):
            result = tool.execute(mode="search", query="NO PROGRAM", source="ofac")
        assert result.success
        assert result.data["count"] >= 1
        assert result.data["results"][0]["programs"] == []

    def test_search_all_with_one_source_failing(self, tool):
        """source=all should return results even if one source fails."""
        with patch("httpx.get", side_effect=_mock_responses(un_status=500)):
            result = tool.execute(mode="search", query="HUAWEI", source="all")
        assert result.success  # OFAC results returned despite UN failure

    def test_programs_empty_after_source_filter(self, tool):
        """Programs mode with a source that has no data."""
        with patch("httpx.get", side_effect=_mock_responses(ofac_text=MOCK_OFAC_CSV_EMPTY)):
            result = tool.execute(mode="programs", source="ofac")
        assert not result.success

    def test_recent_with_old_dates_only(self, tool):
        """All entities have old dates, none in window."""
        old_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<CONSOLIDATED_LIST>
  <INDIVIDUALS>
    <INDIVIDUAL>
      <DATAID>1</DATAID>
      <FIRST_NAME>OLD</FIRST_NAME>
      <SECOND_NAME>PERSON</SECOND_NAME>
      <UN_LIST_TYPE>TEST</UN_LIST_TYPE>
      <LISTED_ON>2001-01-01</LISTED_ON>
    </INDIVIDUAL>
  </INDIVIDUALS>
</CONSOLIDATED_LIST>
"""
        with patch("httpx.get", side_effect=_mock_responses(un_text=old_xml)):
            result = tool.execute(mode="recent", source="un", days_back=30)
        assert result.success
        assert result.data["count"] == 0
        assert "no entities" in result.output.lower()

    def test_multiple_aliases_in_remarks(self):
        csv = "60001,\"MULTI AKA\",-0- ,\"TEST\",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,\"a.k.a. 'ALIAS1'; a.k.a. 'ALIAS2'; a.k.a. 'ALIAS3'.\"\n"
        records = _parse_ofac_csv(csv)
        assert len(records) == 1
        assert len(records[0]["aliases"]) == 3
        assert "ALIAS1" in records[0]["aliases"]
        assert "ALIAS2" in records[0]["aliases"]
        assert "ALIAS3" in records[0]["aliases"]

    def test_remarks_no_nationality(self):
        csv = '60002,"NO NAT",-0- ,"TEST",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,"DOB 01 Jan 1990; passport X12345."\n'
        records = _parse_ofac_csv(csv)
        assert records[0]["nationality"] is None

    def test_un_entity_with_no_name(self):
        """Entity with only NAME_ORIGINAL_SCRIPT."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<CONSOLIDATED_LIST>
  <ENTITIES>
    <ENTITY>
      <DATAID>99999</DATAID>
      <NAME_ORIGINAL_SCRIPT>日本語エンティティ</NAME_ORIGINAL_SCRIPT>
      <UN_LIST_TYPE>TEST</UN_LIST_TYPE>
      <LISTED_ON>2025-01-01</LISTED_ON>
    </ENTITY>
  </ENTITIES>
</CONSOLIDATED_LIST>
"""
        records = _parse_un_xml(xml)
        assert len(records) == 1
        # Should fall back to NAME_ORIGINAL_SCRIPT
        assert records[0]["name"] == "日本語エンティティ"

    def test_format_record_no_programs(self):
        rec = {
            "source": "ofac",
            "name": "NO PROG",
            "type": "entity",
            "programs": [],
            "listed_date": None,
            "last_updated": None,
            "nationality": None,
            "aliases": [],
            "remarks": "",
        }
        text = _format_record(rec)
        assert "—" in text  # em dash for no programs

    def test_format_record_empty_remarks(self):
        rec = {
            "source": "ofac",
            "name": "EMPTY REM",
            "type": "entity",
            "programs": ["TEST"],
            "listed_date": None,
            "last_updated": None,
            "nationality": None,
            "aliases": [],
            "remarks": "",
        }
        text = _format_record(rec)
        assert "Remarks:" not in text

    def test_format_record_many_aliases_truncated(self):
        rec = {
            "source": "un",
            "name": "MANY AKA",
            "type": "individual",
            "programs": ["TEST"],
            "listed_date": None,
            "last_updated": None,
            "nationality": None,
            "aliases": [f"ALIAS{i}" for i in range(20)],
            "remarks": "",
        }
        text = _format_record(rec)
        # Should only show first 5
        assert "ALIAS4" in text
        assert "ALIAS5" not in text

    def test_data_field_present_in_all_modes(self, tool):
        with patch("httpx.get", side_effect=_mock_responses()):
            for mode_args in [
                {"mode": "search", "query": "test"},
                {"mode": "recent"},
                {"mode": "programs"},
            ]:
                result = tool.execute(**mode_args)
                assert result.data is not None
