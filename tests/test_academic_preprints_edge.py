"""
Edge case tests for AcademicPreprintsTool (arXiv + ClinicalTrials.gov).

Covers: mode validation, required params, limit clamping, papers mode (arXiv),
trending mode (arXiv), trials mode (ClinicalTrials.gov), XML parsing, JSON parsing,
HTTP errors, timeout, empty responses, malformed data, cache, output formatting,
constants, registry + bandit.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.academic_preprints import (
    _ARXIV_URL,
    _CT_URL,
    _MARKET_CATEGORIES,
    _NS,
    VALID_MODES,
    AcademicPreprintsTool,
)

# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> AcademicPreprintsTool:
    return AcademicPreprintsTool(cache=cache)


SAMPLE_ARXIV_XML = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>42</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Quantum Error Correction with Machine Learning</title>
    <summary>We present a novel approach to quantum error correction using deep reinforcement learning agents.</summary>
    <published>2024-01-15T12:00:00Z</published>
    <updated>2024-01-16T09:00:00Z</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.AI"/>
    <category term="quant-ph"/>
    <link href="https://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf" title="pdf"/>
    <link href="https://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Transformer Architectures for Financial Time Series</title>
    <summary>A comprehensive study of transformer-based models for predicting financial markets.</summary>
    <published>2024-01-14T10:00:00Z</published>
    <updated>2024-01-14T10:00:00Z</updated>
    <author><name>Charlie Wang</name></author>
    <category term="q-fin.ST"/>
    <category term="cs.LG"/>
    <link href="https://arxiv.org/pdf/2401.00002v1" rel="related" type="application/pdf" title="pdf"/>
  </entry>
</feed>"""

SAMPLE_ARXIV_EMPTY = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>0</opensearch:totalResults>
</feed>"""

SAMPLE_CT_RESPONSE = {
    "totalCount": 150,
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Phase III Trial of Drug X for Lung Cancer",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2024-03"},
                    "completionDateStruct": {"date": "2026-12"},
                },
                "conditionsModule": {
                    "conditions": ["Lung Cancer", "Non-Small Cell Lung Cancer"],
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {"name": "Drug X 100mg"},
                        {"name": "Placebo"},
                    ],
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {
                        "name": "Pfizer",
                        "class": "INDUSTRY",
                    },
                },
            }
        },
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT87654321",
                    "briefTitle": "Diabetes Prevention Study",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "startDateStruct": {"date": "2020-01"},
                    "completionDateStruct": {"date": "2023-06"},
                },
                "conditionsModule": {
                    "conditions": ["Type 2 Diabetes"],
                },
                "armsInterventionsModule": {
                    "interventions": [{"name": "Metformin Extended Release"}],
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {
                        "name": "NIH",
                        "class": "NIH",
                    },
                },
            }
        },
    ],
}


# ── 1. Tool Metadata ─────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "academic_preprints"

    def test_description_nonempty(self):
        assert len(_tool().description) > 50

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        props = params["properties"]
        assert "mode" in props
        assert "query" in props
        assert "category" in props
        assert "sponsor" in props
        assert "status" in props
        assert "limit" in props

    def test_mode_enum(self):
        modes = _tool().parameters["properties"]["mode"]["enum"]
        assert set(modes) == {"papers", "trials", "trending"}

    def test_required_fields(self):
        assert _tool().parameters["required"] == ["mode"]


# ── 2. Input Validation ──────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        r = _tool().execute(mode="")
        assert not r.success

    def test_no_mode(self):
        r = _tool().execute()
        assert not r.success

    def test_mode_case_sensitive(self):
        r = _tool().execute(mode="PAPERS")
        assert not r.success

    def test_papers_requires_query(self):
        r = _tool().execute(mode="papers")
        assert not r.success
        assert "query" in r.output.lower()

    def test_papers_empty_query(self):
        r = _tool().execute(mode="papers", query="")
        assert not r.success

    def test_trials_requires_query_or_sponsor(self):
        r = _tool().execute(mode="trials")
        assert not r.success
        assert "query" in r.output.lower() or "sponsor" in r.output.lower()

    def test_trials_empty_both(self):
        r = _tool().execute(mode="trials", query="", sponsor="")
        assert not r.success

    def test_extra_kwargs_ignored(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="papers", query="quantum", bogus="thing")
            assert r.success


# ── 3. Papers Mode (arXiv) ───────────────────────────────────


class TestPapersMode:
    def test_basic_papers(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="papers", query="quantum")
            assert r.success
            assert "papers" in r.data
            assert r.data["count"] == 2
            assert r.data["source"] == "arxiv"

    def test_papers_fields(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="papers", query="quantum")
            p = r.data["papers"][0]
            assert p["id"] == "http://arxiv.org/abs/2401.00001v1"
            assert "Quantum" in p["title"]
            assert "Alice Smith" in p["authors"]
            assert "cs.AI" in p["categories"]
            assert p["pdf_url"] == "https://arxiv.org/pdf/2401.00001v1"
            assert p["published"] == "2024-01-15T12:00:00Z"

    def test_papers_total(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="papers", query="quantum")
            assert r.data["total"] == 42

    def test_papers_with_category(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML) as mock:
            r = _tool().execute(mode="papers", query="quantum", category="cs.AI")
            assert r.success
            # Check search_query includes category
            call_params = mock.call_args[0][1]
            assert "cat:cs.AI" in call_params["search_query"]

    def test_papers_without_category(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML) as mock:
            r = _tool().execute(mode="papers", query="quantum")
            call_params = mock.call_args[0][1]
            assert "all:quantum" in call_params["search_query"]
            assert "cat:" not in call_params["search_query"]

    def test_papers_limit(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML) as mock:
            r = _tool().execute(mode="papers", query="test", limit=5)
            call_params = mock.call_args[0][1]
            assert call_params["max_results"] == "5"

    def test_papers_fetch_failure(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=None):
            r = _tool().execute(mode="papers", query="test")
            assert not r.success
            assert "Failed" in r.output

    def test_papers_summary_truncated(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="papers", query="quantum")
            for p in r.data["papers"]:
                if p["summary"]:
                    assert len(p["summary"]) <= 300

    def test_papers_authors_capped(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="papers", query="quantum")
            for p in r.data["papers"]:
                assert len(p["authors"]) <= 5


# ── 4. Trending Mode ────────────────────────────────────────


class TestTrendingMode:
    def test_basic_trending(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="trending")
            assert r.success
            assert r.data["source"] == "arxiv"

    def test_trending_with_category(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML) as mock:
            r = _tool().execute(mode="trending", category="cs.AI")
            call_params = mock.call_args[0][1]
            assert "cat:cs.AI" in call_params["search_query"]

    def test_trending_no_category_uses_market(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML) as mock:
            r = _tool().execute(mode="trending")
            call_params = mock.call_args[0][1]
            # Should include market categories
            assert "q-fin" in call_params["search_query"]
            assert "cs.AI" in call_params["search_query"]

    def test_trending_fetch_failure(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=None):
            r = _tool().execute(mode="trending")
            assert not r.success


# ── 5. Trials Mode (ClinicalTrials.gov) ──────────────────────


class TestTrialsMode:
    def test_basic_trials_by_condition(self):
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=SAMPLE_CT_RESPONSE):
            r = _tool().execute(mode="trials", query="cancer")
            assert r.success
            assert "trials" in r.data
            assert r.data["count"] == 2
            assert r.data["source"] == "clinicaltrials"

    def test_trials_fields(self):
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=SAMPLE_CT_RESPONSE):
            r = _tool().execute(mode="trials", query="cancer")
            t = r.data["trials"][0]
            assert t["nct_id"] == "NCT12345678"
            assert "Lung Cancer" in t["title"]
            assert t["status"] == "RECRUITING"
            assert t["sponsor"] == "Pfizer"
            assert t["sponsor_class"] == "INDUSTRY"
            assert "Lung Cancer" in t["conditions"]
            assert "Drug X 100mg" in t["interventions"]

    def test_trials_by_sponsor(self):
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=SAMPLE_CT_RESPONSE) as mock:
            r = _tool().execute(mode="trials", sponsor="Pfizer")
            assert r.success
            call_params = mock.call_args[0][1]
            assert call_params["query.spons"] == "Pfizer"

    def test_trials_by_status(self):
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=SAMPLE_CT_RESPONSE) as mock:
            r = _tool().execute(mode="trials", query="cancer", status="COMPLETED")
            call_params = mock.call_args[0][1]
            assert call_params["filter.overallStatus"] == "COMPLETED"

    def test_trials_total_count(self):
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=SAMPLE_CT_RESPONSE):
            r = _tool().execute(mode="trials", query="cancer")
            assert r.data["total"] == 150

    def test_trials_fetch_failure(self):
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=None):
            r = _tool().execute(mode="trials", query="cancer")
            assert not r.success

    def test_trials_empty(self):
        with patch.object(
            AcademicPreprintsTool,
            "_fetch_json",
            return_value={"studies": [], "totalCount": 0},
        ):
            r = _tool().execute(mode="trials", query="xyznonexistent")
            assert r.success
            assert r.data["count"] == 0

    def test_trials_missing_modules(self):
        data = {
            "totalCount": 1,
            "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT000"}}}],
        }
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=data):
            r = _tool().execute(mode="trials", query="test")
            assert r.success
            t = r.data["trials"][0]
            assert t["nct_id"] == "NCT000"
            assert t["status"] is None
            assert t["conditions"] == []
            assert t["interventions"] == []

    def test_trials_interventions_capped(self):
        many_interventions = {"interventions": [{"name": f"Drug{i}"} for i in range(20)]}
        data = {
            "totalCount": 1,
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT000"},
                        "armsInterventionsModule": many_interventions,
                    }
                }
            ],
        }
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=data):
            r = _tool().execute(mode="trials", query="test")
            assert len(r.data["trials"][0]["interventions"]) <= 5


# ── 6. XML Parsing Edge Cases ────────────────────────────────


class TestXMLParsing:
    def test_parse_valid_xml(self):
        t = _tool()
        papers = t._parse_arxiv_xml(SAMPLE_ARXIV_XML)
        assert len(papers) == 2

    def test_parse_empty_xml(self):
        t = _tool()
        papers = t._parse_arxiv_xml(SAMPLE_ARXIV_EMPTY)
        assert len(papers) == 0

    def test_parse_invalid_xml(self):
        t = _tool()
        papers = t._parse_arxiv_xml("not xml at all")
        assert papers == []

    def test_parse_total_results(self):
        t = _tool()
        total = t._parse_arxiv_total(SAMPLE_ARXIV_XML)
        assert total == 42

    def test_parse_total_empty(self):
        t = _tool()
        total = t._parse_arxiv_total(SAMPLE_ARXIV_EMPTY)
        assert total == 0

    def test_parse_total_invalid_xml(self):
        t = _tool()
        total = t._parse_arxiv_total("not xml")
        assert total == 0

    def test_entry_missing_title(self):
        xml = """<?xml version='1.0' encoding='UTF-8'?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
          <opensearch:totalResults>1</opensearch:totalResults>
          <entry>
            <id>http://arxiv.org/abs/2401.00001v1</id>
          </entry>
        </feed>"""
        t = _tool()
        papers = t._parse_arxiv_xml(xml)
        assert len(papers) == 1
        assert papers[0]["title"] is None

    def test_title_whitespace_normalized(self):
        t = _tool()
        papers = t._parse_arxiv_xml(SAMPLE_ARXIV_XML)
        for p in papers:
            if p["title"]:
                assert "\n" not in p["title"]
                assert "  " not in p["title"]


# ── 7. HTTP Error Handling ────────────────────────────────────


class TestHTTPErrors:
    def test_timeout_arxiv(self):
        with patch.object(
            AcademicPreprintsTool,
            "_fetch_text",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = _tool().execute(mode="papers", query="test")
            assert not r.success
            assert "timed out" in r.output

    def test_timeout_ct(self):
        with patch.object(
            AcademicPreprintsTool,
            "_fetch_json",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            r = _tool().execute(mode="trials", query="test")
            assert not r.success

    def test_http_error(self):
        with patch.object(
            AcademicPreprintsTool,
            "_fetch_text",
            side_effect=httpx.HTTPStatusError(
                "503",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(503),
            ),
        ):
            r = _tool().execute(mode="papers", query="test")
            assert not r.success

    def test_connection_error(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", side_effect=httpx.ConnectError("fail")):
            r = _tool().execute(mode="papers", query="test")
            assert not r.success

    def test_generic_exception(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", side_effect=RuntimeError("boom")):
            r = _tool().execute(mode="papers", query="test")
            assert not r.success
            assert "Unexpected" in r.output


# ── 8. Limit Clamping ────────────────────────────────────────


class TestLimitClamping:
    def test_limit_low(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML) as mock:
            r = _tool().execute(mode="papers", query="test", limit=0)
            call_params = mock.call_args[0][1]
            assert call_params["max_results"] == "1"

    def test_limit_high(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML) as mock:
            r = _tool().execute(mode="papers", query="test", limit=999)
            call_params = mock.call_args[0][1]
            assert call_params["max_results"] == "50"

    def test_limit_string_coerced(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="papers", query="test", limit="10")
            assert r.success


# ── 9. Output Formatting ─────────────────────────────────────


class TestOutputFormatting:
    def test_papers_output(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="papers", query="quantum")
            assert "quantum" in r.output
            assert "arXiv" in r.output

    def test_trending_output(self):
        with patch.object(AcademicPreprintsTool, "_fetch_text", return_value=SAMPLE_ARXIV_XML):
            r = _tool().execute(mode="trending")
            assert "Trending" in r.output or "trending" in r.output.lower()

    def test_trials_output(self):
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=SAMPLE_CT_RESPONSE):
            r = _tool().execute(mode="trials", query="cancer")
            assert "clinical trial" in r.output.lower()

    def test_trials_output_mentions_sponsor(self):
        with patch.object(AcademicPreprintsTool, "_fetch_json", return_value=SAMPLE_CT_RESPONSE):
            r = _tool().execute(mode="trials", sponsor="Pfizer")
            assert "Pfizer" in r.output


# ── 10. Constants ─────────────────────────────────────────────


class TestConstants:
    def test_valid_modes(self):
        assert {"papers", "trials", "trending"} == VALID_MODES

    def test_market_categories(self):
        assert "q-fin" in _MARKET_CATEGORIES
        assert "cs.AI" in _MARKET_CATEGORIES

    def test_urls(self):
        assert "arxiv.org" in _ARXIV_URL
        assert "clinicaltrials.gov" in _CT_URL

    def test_xml_namespaces(self):
        assert "atom" in _NS
        assert "opensearch" in _NS


# ── 11. Registry + Bandit Integration ────────────────────────


class TestRegistryAndBandit:
    def test_tool_count(self):
        try:
            from agent.cli import build_tool_registry
        except (ImportError, ModuleNotFoundError):
            pytest.skip("optional dep not installed")
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.tool_timeout = 30
        mock_config.fred_api_key = ""
        registry = build_tool_registry(mock_config)
        assert len(registry._tools) == 61

    def test_academic_preprints_registered(self):
        try:
            from agent.cli import build_tool_registry
        except (ImportError, ModuleNotFoundError):
            pytest.skip("optional dep not installed")
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.tool_timeout = 30
        mock_config.fred_api_key = ""
        registry = build_tool_registry(mock_config)
        assert "academic_preprints" in registry._tools

    def test_bandit_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48

    def test_research_pipeline_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = {a.name for a in DEFAULT_ARMS}
        assert "research_pipeline" in names

    def test_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "research_pipeline")
        assert "academic_preprints" in arm.tools


# ── L2 Entity Persistence Tests ──────────────────────────────────────────────

import unittest


def _make_store_mock():
    """Build a mock PipelineStore for L2 persistence testing."""
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    return store


class TestL2PersistenceNoStore(unittest.TestCase):
    """Persistence is a no-op when store is absent."""

    def test_no_store_returns_zeros(self):
        tool = AcademicPreprintsTool()
        tool._store = None
        counts = tool._persist_entities({"papers": [{"categories": ["cs.AI"], "title": "test"}]}, "papers")
        assert counts == {"research_velocity_obs": 0}

    def test_no_entity_id_fn_returns_zeros(self):
        import agent.tools.academic_preprints as ap_mod

        tool = AcademicPreprintsTool()
        tool._store = _make_store_mock()
        original = ap_mod._entity_id_from_key
        try:
            ap_mod._entity_id_from_key = None
            counts = tool._persist_entities({"papers": [{"categories": ["cs.AI"]}]}, "papers")
            assert counts == {"research_velocity_obs": 0}
        finally:
            ap_mod._entity_id_from_key = original


class TestL2PersistenceTrials(unittest.TestCase):
    """trials mode persists research_velocity obs on company entity nodes."""

    def test_persists_one_trial(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "trials": [
                {
                    "nct_id": "NCT00001234",
                    "title": "Phase III Study of Drug X",
                    "status": "RECRUITING",
                    "sponsor": "Pfizer Inc",
                    "conditions": ["Cancer"],
                }
            ]
        }
        counts = tool._persist_entities(data, "trials")
        assert counts["research_velocity_obs"] == 1
        store.register_entity.assert_called_once()
        call_kw = store.register_entity.call_args.kwargs
        assert call_kw["entity_type"] == "company"
        assert call_kw["canonical_name"] == "Pfizer Inc"

    def test_obs_type_and_depth(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "trials": [
                {
                    "nct_id": "NCT99999999",
                    "title": "Study A",
                    "status": "COMPLETED",
                    "sponsor": "Moderna",
                    "conditions": ["COVID-19"],
                }
            ]
        }
        tool._persist_entities(data, "trials")
        obs_kw = store.store_entity_observation.call_args.kwargs
        assert obs_kw["observation_type"] == "research_velocity"
        assert obs_kw["depth_level"] == 2
        assert obs_kw["source_tool"] == "academic_preprints"
        assert obs_kw["value"]["source"] == "clinicaltrials"
        assert obs_kw["value"]["nct_id"] == "NCT99999999"

    def test_multiple_trials(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "trials": [
                {"sponsor": "Pfizer", "nct_id": "A"},
                {"sponsor": "Moderna", "nct_id": "B"},
                {"sponsor": "Novartis", "nct_id": "C"},
            ]
        }
        counts = tool._persist_entities(data, "trials")
        assert counts["research_velocity_obs"] == 3
        assert store.register_entity.call_count == 3

    def test_empty_sponsor_skipped(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "trials": [
                {"sponsor": "", "nct_id": "A"},
                {"sponsor": None, "nct_id": "B"},
                {"sponsor": "   ", "nct_id": "C"},
            ]
        }
        counts = tool._persist_entities(data, "trials")
        assert counts["research_velocity_obs"] == 0
        store.register_entity.assert_not_called()

    def test_missing_sponsor_key_skipped(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {"trials": [{"nct_id": "A", "title": "No sponsor field"}]}
        counts = tool._persist_entities(data, "trials")
        assert counts["research_velocity_obs"] == 0

    def test_trial_value_contains_conditions(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "trials": [
                {
                    "sponsor": "BioNTech",
                    "nct_id": "NCT123",
                    "title": "mRNA Study",
                    "status": "ACTIVE_NOT_RECRUITING",
                    "conditions": ["Influenza", "RSV"],
                }
            ]
        }
        tool._persist_entities(data, "trials")
        obs_val = store.store_entity_observation.call_args.kwargs["value"]
        assert obs_val["conditions"] == ["Influenza", "RSV"]
        assert obs_val["title"] == "mRNA Study"
        assert obs_val["status"] == "ACTIVE_NOT_RECRUITING"


class TestL2PersistencePapers(unittest.TestCase):
    """papers mode persists research_velocity obs on topic entity nodes."""

    def test_persists_one_paper(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "papers": [
                {
                    "id": "2401.12345",
                    "title": "Quantum Error Correction",
                    "categories": ["quant-ph", "cs.IT"],
                    "published": "2026-04-01",
                }
            ]
        }
        counts = tool._persist_entities(data, "papers")
        assert counts["research_velocity_obs"] == 1
        call_kw = store.register_entity.call_args.kwargs
        assert call_kw["entity_type"] == "topic"
        assert call_kw["canonical_name"] == "quant-ph"

    def test_uses_first_category(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "papers": [
                {
                    "id": "2401.99999",
                    "title": "ML for Finance",
                    "categories": ["q-fin.CP", "cs.LG", "stat.ML"],
                    "published": "2026-04-15",
                }
            ]
        }
        tool._persist_entities(data, "papers")
        call_kw = store.register_entity.call_args.kwargs
        assert call_kw["canonical_name"] == "q-fin.CP"

    def test_paper_obs_value_fields(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "papers": [
                {
                    "id": "2401.00001",
                    "title": "Deep RL Trading",
                    "categories": ["cs.AI"],
                    "published": "2026-01-01",
                }
            ]
        }
        tool._persist_entities(data, "papers")
        obs_val = store.store_entity_observation.call_args.kwargs["value"]
        assert obs_val["source"] == "arxiv"
        assert obs_val["paper_id"] == "2401.00001"
        assert obs_val["title"] == "Deep RL Trading"
        assert obs_val["published"] == "2026-01-01"

    def test_empty_categories_skipped(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "papers": [
                {"id": "A", "title": "No Cat", "categories": []},
                {"id": "B", "title": "None Cat", "categories": None},
                {"id": "C", "title": "Missing Cat"},
            ]
        }
        counts = tool._persist_entities(data, "papers")
        assert counts["research_velocity_obs"] == 0
        store.register_entity.assert_not_called()

    def test_multiple_papers(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "papers": [
                {"categories": ["cs.AI"], "id": "1"},
                {"categories": ["q-fin"], "id": "2"},
                {"categories": ["cs.LG"], "id": "3"},
            ]
        }
        counts = tool._persist_entities(data, "papers")
        assert counts["research_velocity_obs"] == 3


class TestL2PersistenceTrending(unittest.TestCase):
    """trending mode uses same code path as papers."""

    def test_trending_persists_topics(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "papers": [
                {"categories": ["cs.CR"], "id": "trend-1", "title": "Zero-Day"},
                {"categories": ["econ"], "id": "trend-2", "title": "Macro Model"},
            ]
        }
        counts = tool._persist_entities(data, "trending")
        assert counts["research_velocity_obs"] == 2
        assert store.register_entity.call_count == 2


class TestL2PersistenceExceptionHandling(unittest.TestCase):
    """Exceptions in persistence are non-fatal."""

    def test_exception_in_inner_returns_zeros(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        store.register_entity.side_effect = RuntimeError("DB failure")
        tool._store = store
        counts = tool._persist_entities({"trials": [{"sponsor": "TestCo", "nct_id": "X"}]}, "trials")
        assert counts == {"research_velocity_obs": 0}

    def test_exception_does_not_propagate(self):
        tool = AcademicPreprintsTool()
        store = _make_store_mock()
        store.store_entity_observation.side_effect = ValueError("bad value")
        tool._store = store
        # Should not raise
        counts = tool._persist_entities({"papers": [{"categories": ["cs.AI"], "id": "err"}]}, "papers")
        assert counts == {"research_velocity_obs": 0}


class TestL2PersistenceEmptyData(unittest.TestCase):
    """Empty data dicts produce zero counts."""

    def test_empty_trials(self):
        tool = AcademicPreprintsTool()
        tool._store = _make_store_mock()
        counts = tool._persist_entities({"trials": []}, "trials")
        assert counts["research_velocity_obs"] == 0

    def test_empty_papers(self):
        tool = AcademicPreprintsTool()
        tool._store = _make_store_mock()
        counts = tool._persist_entities({"papers": []}, "papers")
        assert counts["research_velocity_obs"] == 0

    def test_missing_key(self):
        tool = AcademicPreprintsTool()
        tool._store = _make_store_mock()
        counts = tool._persist_entities({}, "trials")
        assert counts["research_velocity_obs"] == 0

    def test_unknown_mode_returns_zeros(self):
        tool = AcademicPreprintsTool()
        tool._store = _make_store_mock()
        counts = tool._persist_entities({"papers": [{"categories": ["cs.AI"]}]}, "unknown_mode")
        assert counts["research_velocity_obs"] == 0


class TestL2PersistenceIdempotent(unittest.TestCase):
    """Same data persisted twice produces correct cumulative counts."""

    def test_double_persist_doubles_count(self):
        tool = AcademicPreprintsTool()
        tool._store = _make_store_mock()
        data = {"trials": [{"sponsor": "Pfizer", "nct_id": "NCT1"}]}
        c1 = tool._persist_entities(data, "trials")
        c2 = tool._persist_entities(data, "trials")
        assert c1["research_velocity_obs"] == 1
        assert c2["research_velocity_obs"] == 1
