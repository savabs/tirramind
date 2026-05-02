"""Tests for Phase 13c-d: L2 entity persistence for corporate intel, energy, and DeFi tools.

Covers lobbying, patent_filings, defi_flows, and interconnection_queue L2 upgrades:
    - Optional pipeline_store in constructor
    - Entity registration (company/protocol types)
    - Observation storage (lobbying_spend/patent_filing/tvl_change/project_status)
    - Error isolation (persistence failure doesn't crash tool)
    - Empty results and missing fields handled gracefully
    - Company name normalization and deduplication
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.pipeline.entity import entity_id_from_key, normalize_company_name
from agent.pipeline.store import PipelineStore
from agent.tools.defi_flows import DefiFlowsTool
from agent.tools.interconnection_queue import InterconnectionQueueTool
from agent.tools.lobbying import LobbyingTool
from agent.tools.patent_filings import PatentFilingsTool

# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture()
def store() -> PipelineStore:
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


# ── LobbyingTool L2 Tests ─────────────────────────────────────


class TestLobbyingL2:
    """L2 entity persistence for LobbyingTool."""

    def test_constructor_accepts_pipeline_store(self, store):
        tool = LobbyingTool(pipeline_store=store)
        assert tool._store is store

    def test_constructor_without_pipeline_store(self):
        tool = LobbyingTool()
        assert tool._store is None

    def test_persist_entities_registers_company(self, store):
        tool = LobbyingTool(pipeline_store=store)
        filings = [
            {
                "registrant_name": "Google LLC",
                "registrant_id": 12345,
                "client_name": "Self",
                "amount": 5000000.0,
                "filing_year": 2025,
                "filing_period": "first_quarter",
                "dt_posted": "2025-04-15T00:00:00Z",
                "issue_codes": ["CPT", "TRD"],
            }
        ]
        tool._persist_entities(filings)

        entities = store.query_all_entities()
        assert len(entities) == 1
        assert entities[0]["entity_type"] == "company"

    def test_persist_entities_stores_lobbying_observation(self, store):
        tool = LobbyingTool(pipeline_store=store)
        filings = [
            {
                "registrant_name": "Amazon.com Inc.",
                "registrant_id": 99999,
                "client_name": "Self",
                "amount": 8000000.0,
                "filing_year": 2025,
                "filing_period": "second_quarter",
                "dt_posted": "2025-07-15T00:00:00Z",
                "issue_codes": ["CPT"],
            }
        ]
        tool._persist_entities(filings)

        canon = normalize_company_name("Amazon.com Inc.")
        company_eid = entity_id_from_key("company", canon)
        obs = store.query_entity_observations(company_eid, source_tool="lobbying")
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "lobbying_spend"
        assert obs[0]["depth_level"] == 2
        val = obs[0]["value"]
        assert val["amount"] == 8000000.0
        assert val["filing_year"] == 2025
        assert val["issue_codes"] == ["CPT"]

    def test_persist_entities_dedup_registrants(self, store):
        """Same registrant in multiple filings should register once."""
        tool = LobbyingTool(pipeline_store=store)
        filings = [
            {
                "registrant_name": "Google LLC",
                "amount": 1000000,
                "dt_posted": "2025-01-01",
            },
            {
                "registrant_name": "Google LLC",
                "amount": 2000000,
                "dt_posted": "2025-04-01",
            },
        ]
        tool._persist_entities(filings)

        entities = store.query_all_entities()
        assert len(entities) == 1

    def test_persist_entities_empty_filings(self, store):
        tool = LobbyingTool(pipeline_store=store)
        tool._persist_entities([])
        assert len(store.query_all_entities()) == 0

    def test_persist_entities_no_store(self):
        tool = LobbyingTool()
        tool._persist_entities([{"registrant_name": "Test Corp"}])

    def test_persist_entities_error_isolation(self, store):
        tool = LobbyingTool(pipeline_store=store)
        with patch.object(store, "register_entity", side_effect=RuntimeError("DB error")):
            tool._persist_entities([{"registrant_name": "Test", "dt_posted": "2025-01-01"}])

    def test_persist_entities_missing_registrant(self, store):
        """Filing without registrant_name should be skipped."""
        tool = LobbyingTool(pipeline_store=store)
        tool._persist_entities([{"amount": 1000, "dt_posted": "2025-01-01"}])
        assert len(store.query_all_entities()) == 0

    def test_persist_entities_bad_date(self, store):
        tool = LobbyingTool(pipeline_store=store)
        filings = [{"registrant_name": "Test Inc", "dt_posted": "invalid-date"}]
        tool._persist_entities(filings)
        canon = normalize_company_name("Test Inc")
        eid = entity_id_from_key("company", canon)
        obs = store.query_entity_observations(eid, source_tool="lobbying")
        assert len(obs) == 1
        assert obs[0]["observed_at"] > 0


# ── PatentFilingsTool L2 Tests ─────────────────────────────────


class TestPatentFilingsL2:
    """L2 entity persistence for PatentFilingsTool."""

    def test_constructor_accepts_pipeline_store(self, store):
        tool = PatentFilingsTool(pipeline_store=store)
        assert tool._store is store

    def test_constructor_without_pipeline_store(self):
        tool = PatentFilingsTool()
        assert tool._store is None

    def test_persist_entities_registers_company(self, store):
        tool = PatentFilingsTool(pipeline_store=store)
        patents = [
            {
                "patent_number": "US12345678",
                "patent_title": "Neural Network Accelerator",
                "patent_date": "2025-03-15",
                "assignee_organization": "NVIDIA Corporation",
                "cpc_subgroup_id": "G06F17/50",
            }
        ]
        tool._persist_entities(patents)

        entities = store.query_all_entities()
        company_entities = [e for e in entities if e["entity_type"] == "company"]
        assert len(company_entities) == 1
        assert company_entities[0]["entity_type"] == "company"

    def test_persist_entities_stores_patent_observation(self, store):
        tool = PatentFilingsTool(pipeline_store=store)
        patents = [
            {
                "patent_number": "US99999999",
                "patent_title": "Quantum Computing Method",
                "patent_date": "2025-06-01",
                "assignee_organization": "IBM",
                "cpc_subgroup_id": "H03K19/195",
            }
        ]
        tool._persist_entities(patents)

        canon = normalize_company_name("IBM")
        eid = entity_id_from_key("company", canon)
        obs = store.query_entity_observations(eid, source_tool="patent_filings")
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "patent_filing"
        assert obs[0]["depth_level"] == 2
        val = obs[0]["value"]
        assert val["patent_number"] == "US99999999"
        assert val["cpc_subgroup_id"] == "H03K19/195"

    def test_persist_entities_assignee_as_list(self, store):
        """assignee_organization can be a list — should use first element."""
        tool = PatentFilingsTool(pipeline_store=store)
        patents = [
            {
                "assignee_organization": ["Apple Inc.", "Beats Electronics"],
                "patent_number": "US11111",
                "patent_date": "2025-01-01",
                "cpc_subgroup_id": ["H04R1/10"],
            }
        ]
        tool._persist_entities(patents)

        entities = store.query_all_entities()
        company_entities = [e for e in entities if e["entity_type"] == "company"]
        assert len(company_entities) == 1

    def test_persist_entities_dedup_assignees(self, store):
        tool = PatentFilingsTool(pipeline_store=store)
        patents = [
            {
                "assignee_organization": "Tesla Inc",
                "patent_number": "P1",
                "patent_date": "2025-01-01",
            },
            {
                "assignee_organization": "Tesla Inc",
                "patent_number": "P2",
                "patent_date": "2025-02-01",
            },
        ]
        tool._persist_entities(patents)
        company_entities = [e for e in store.query_all_entities() if e["entity_type"] == "company"]
        assert len(company_entities) == 1

    def test_persist_entities_empty(self, store):
        tool = PatentFilingsTool(pipeline_store=store)
        tool._persist_entities([])
        assert len(store.query_all_entities()) == 0

    def test_persist_entities_no_store(self):
        tool = PatentFilingsTool()
        tool._persist_entities([{"assignee_organization": "Test"}])

    def test_persist_entities_error_isolation(self, store):
        tool = PatentFilingsTool(pipeline_store=store)
        with patch.object(store, "register_entity", side_effect=RuntimeError):
            tool._persist_entities([{"assignee_organization": "X", "patent_date": "2025-01-01"}])

    def test_persist_entities_missing_assignee(self, store):
        tool = PatentFilingsTool(pipeline_store=store)
        tool._persist_entities([{"patent_number": "P1", "patent_date": "2025-01-01"}])
        assert len(store.query_all_entities()) == 0

    def test_persist_entities_bad_date(self, store):
        tool = PatentFilingsTool(pipeline_store=store)
        patents = [{"assignee_organization": "Test Corp", "patent_date": "bad"}]
        tool._persist_entities(patents)
        canon = normalize_company_name("Test Corp")
        eid = entity_id_from_key("company", canon)
        obs = store.query_entity_observations(eid, source_tool="patent_filings")
        assert len(obs) == 1
        assert obs[0]["observed_at"] > 0


# ── DefiFlowsTool L2 Tests ────────────────────────────────────


class TestDefiFlowsL2:
    """L2 entity persistence for DefiFlowsTool."""

    def test_constructor_accepts_pipeline_store(self, store):
        tool = DefiFlowsTool(pipeline_store=store)
        assert tool._store is store

    def test_constructor_without_pipeline_store(self):
        tool = DefiFlowsTool()
        assert tool._store is None

    def test_persist_entities_registers_protocol(self, store):
        tool = DefiFlowsTool(pipeline_store=store)
        protocols = [
            {
                "name": "Aave",
                "tvl_usd": 10000000000.0,
                "chain": "Ethereum",
                "chains": ["Ethereum", "Polygon", "Avalanche"],
                "category": "Lending",
                "change_1d_pct": 2.5,
                "change_7d_pct": -1.2,
            }
        ]
        tool._persist_entities(protocols)

        entities = store.query_all_entities()
        assert len(entities) == 1
        assert entities[0]["entity_type"] == "protocol"
        assert entities[0]["canonical_name"] == "Aave"

    def test_persist_entities_stores_tvl_observation(self, store):
        tool = DefiFlowsTool(pipeline_store=store)
        protocols = [
            {
                "name": "Uniswap",
                "tvl_usd": 5000000000.0,
                "chain": "Ethereum",
                "chains": ["Ethereum", "Polygon"],
                "category": "DEX",
                "change_1d_pct": -3.0,
            }
        ]
        tool._persist_entities(protocols)

        eid = entity_id_from_key("protocol", "uniswap")
        obs = store.query_entity_observations(eid, source_tool="defi_flows")
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "tvl_change"
        assert obs[0]["depth_level"] == 2
        val = obs[0]["value"]
        assert val["tvl_usd"] == 5000000000.0
        assert val["chain"] == "Ethereum"
        assert val["category"] == "DEX"
        assert val["change_1d_pct"] == -3.0

    def test_persist_entities_dedup_protocols(self, store):
        """Same protocol name (case-insensitive) should register once."""
        tool = DefiFlowsTool(pipeline_store=store)
        protocols = [
            {"name": "Aave", "tvl_usd": 10e9},
            {"name": "aave", "tvl_usd": 10e9},
        ]
        tool._persist_entities(protocols)
        assert len(store.query_all_entities()) == 1

    def test_persist_entities_multiple_protocols(self, store):
        tool = DefiFlowsTool(pipeline_store=store)
        protocols = [
            {"name": "Aave", "tvl_usd": 10e9},
            {"name": "Uniswap", "tvl_usd": 5e9},
            {"name": "Curve", "tvl_usd": 3e9},
        ]
        tool._persist_entities(protocols)
        assert len(store.query_all_entities()) == 3

    def test_persist_entities_empty(self, store):
        tool = DefiFlowsTool(pipeline_store=store)
        tool._persist_entities([])
        assert len(store.query_all_entities()) == 0

    def test_persist_entities_no_store(self):
        tool = DefiFlowsTool()
        tool._persist_entities([{"name": "Test"}])

    def test_persist_entities_error_isolation(self, store):
        tool = DefiFlowsTool(pipeline_store=store)
        with patch.object(store, "register_entity", side_effect=RuntimeError):
            tool._persist_entities([{"name": "Test"}])

    def test_persist_entities_missing_name(self, store):
        tool = DefiFlowsTool(pipeline_store=store)
        tool._persist_entities([{"tvl_usd": 1e9}])
        assert len(store.query_all_entities()) == 0

    def test_protocol_alias_stored(self, store):
        tool = DefiFlowsTool(pipeline_store=store)
        tool._persist_entities([{"name": "MakerDAO"}])
        eid = entity_id_from_key("protocol", "makerdao")
        aliases = store.query_entity_aliases(eid)
        assert any(a["source"] == "protocol_name" and a["external_id"] == "MakerDAO" for a in aliases)


# ── InterconnectionQueueTool L2 Tests ──────────────────────────


class TestInterconnectionQueueL2:
    """L2 entity persistence for InterconnectionQueueTool."""

    def test_constructor_accepts_pipeline_store(self, store):
        tool = InterconnectionQueueTool(pipeline_store=store)
        assert tool._store is store

    def test_constructor_without_pipeline_store(self):
        tool = InterconnectionQueueTool()
        assert tool._store is None

    def test_persist_entities_registers_company(self, store):
        tool = InterconnectionQueueTool(pipeline_store=store)
        records = [
            {
                "entityName": "NextEra Energy Inc",
                "plantName": "Solar Farm Alpha",
                "nameplate-capacity-mw": 200.0,
                "energy-source-code": "SUN",
                "stateid": "FL",
                "status": "PL",
                "technology": "Photovoltaic",
            }
        ]
        tool._persist_entities(records)

        entities = store.query_all_entities()
        company_entities = [e for e in entities if e["entity_type"] == "company"]
        assert len(company_entities) == 1
        assert company_entities[0]["entity_type"] == "company"

    def test_persist_entities_stores_project_observation(self, store):
        tool = InterconnectionQueueTool(pipeline_store=store)
        records = [
            {
                "entityName": "Amazon Data Services",
                "plantName": "AWS-East Data Center Power",
                "nameplate-capacity-mw": 500.0,
                "energy-source-code": "NG",
                "stateid": "VA",
                "status": "U",
                "technology": "Natural Gas Turbine",
            }
        ]
        tool._persist_entities(records)

        canon = normalize_company_name("Amazon Data Services")
        eid = entity_id_from_key("company", canon)
        obs = store.query_entity_observations(eid, source_tool="interconnection_queue")
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "project_status"
        assert obs[0]["depth_level"] == 2
        val = obs[0]["value"]
        assert val["plant_name"] == "AWS-East Data Center Power"
        assert val["nameplate_capacity_mw"] == 500.0
        assert val["energy_source_code"] == "NG"
        assert val["state"] == "VA"

    def test_persist_entities_snake_case_keys(self, store):
        """Tool should handle both kebab-case and snake_case keys."""
        tool = InterconnectionQueueTool(pipeline_store=store)
        records = [
            {
                "entity_name": "Google LLC",
                "plant_name": "Solar Project Beta",
                "nameplate_capacity_mw": 300.0,
                "energy_source_code": "SUN",
                "state": "TX",
                "status": "PL",
            }
        ]
        tool._persist_entities(records)

        company_entities = [e for e in store.query_all_entities() if e["entity_type"] == "company"]
        assert len(company_entities) == 1

    def test_persist_entities_dedup_companies(self, store):
        tool = InterconnectionQueueTool(pipeline_store=store)
        records = [
            {"entityName": "NextEra Energy", "plantName": "Plant A", "status": "PL"},
            {"entityName": "NextEra Energy", "plantName": "Plant B", "status": "U"},
        ]
        tool._persist_entities(records)
        company_entities = [e for e in store.query_all_entities() if e["entity_type"] == "company"]
        assert len(company_entities) == 1

    def test_persist_entities_empty(self, store):
        tool = InterconnectionQueueTool(pipeline_store=store)
        tool._persist_entities([])
        assert len(store.query_all_entities()) == 0

    def test_persist_entities_none_records(self, store):
        """None records should be handled gracefully."""
        tool = InterconnectionQueueTool(pipeline_store=store)
        tool._persist_entities(None)
        assert len(store.query_all_entities()) == 0

    def test_persist_entities_no_store(self):
        tool = InterconnectionQueueTool()
        tool._persist_entities([{"entityName": "Test"}])

    def test_persist_entities_error_isolation(self, store):
        tool = InterconnectionQueueTool(pipeline_store=store)
        with patch.object(store, "register_entity", side_effect=RuntimeError):
            tool._persist_entities([{"entityName": "Test"}])

    def test_persist_entities_missing_entity_name(self, store):
        tool = InterconnectionQueueTool(pipeline_store=store)
        tool._persist_entities([{"plantName": "Orphan Plant", "status": "PL"}])
        assert len(store.query_all_entities()) == 0


# ── Cross-Tool Entity Dedup ───────────────────────────────────


class TestCrossToolCompanyDedup:
    """Verify company entities from different tools are deduplicated via normalize_company_name."""

    def test_same_company_from_lobbying_and_patents(self, store):
        """Google from lobbying and Google from patents → same entity."""
        lobby = LobbyingTool(pipeline_store=store)
        patents = PatentFilingsTool(pipeline_store=store)

        lobby._persist_entities(
            [
                {
                    "registrant_name": "Google LLC",
                    "amount": 5e6,
                    "dt_posted": "2025-01-01",
                },
            ]
        )
        patents._persist_entities(
            [
                {
                    "assignee_organization": "Google LLC",
                    "patent_number": "P1",
                    "patent_date": "2025-06-01",
                },
            ]
        )

        company_entities = [e for e in store.query_all_entities() if e["entity_type"] == "company"]
        assert len(company_entities) == 1

        canon = normalize_company_name("Google LLC")
        eid = entity_id_from_key("company", canon)
        obs = store.query_entity_observations(eid)
        assert len(obs) == 2
        tools = {o["source_tool"] for o in obs}
        assert tools == {"lobbying", "patent_filings"}

    def test_same_company_lobbying_and_interconnection(self, store):
        lobby = LobbyingTool(pipeline_store=store)
        iq = InterconnectionQueueTool(pipeline_store=store)

        lobby._persist_entities(
            [
                {
                    "registrant_name": "Amazon.com Inc.",
                    "amount": 8e6,
                    "dt_posted": "2025-01-01",
                },
            ]
        )
        iq._persist_entities(
            [
                {
                    "entityName": "Amazon.com Inc.",
                    "plantName": "Data Center Power",
                    "status": "PL",
                },
            ]
        )

        company_entities = [e for e in store.query_all_entities() if e["entity_type"] == "company"]
        assert len(company_entities) == 1
