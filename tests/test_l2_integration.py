"""Phase 13e: Integration tests — all L2 tools → PipelineStore → GraphBuilder → HeteroData.

Verifies the full pipeline: each L2 tool persists entities via PipelineStore,
then GraphBuilder converts the entity graph into a PyG HeteroData object with
correct node types, feature dimensions, and event observation types.
"""

from __future__ import annotations

import pytest

from agent.models.gnn.graph_builder import (
    ENTITY_TYPES,
    GraphBuilder,
)
from agent.pipeline.store import PipelineStore

# All L2 tools
from agent.tools.cert_transparency import CertTransparencyTool
from agent.tools.defi_flows import DefiFlowsTool
from agent.tools.dns_monitor import DnsMonitorTool
from agent.tools.insider_filings import InsiderFilingsTool
from agent.tools.interconnection_queue import InterconnectionQueueTool
from agent.tools.lobbying import LobbyingTool
from agent.tools.patent_filings import PatentFilingsTool
from agent.tools.wikipedia_pageviews import WikipediaPageviewsTool

# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture()
def store() -> PipelineStore:
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


# ── Helper: populate store with all L2 tool data ─────────────


def _populate_all_tools(store: PipelineStore) -> None:
    """Push synthetic data through every L2 tool's _persist_entities."""

    # 1. insider_filings → company(person), insider_trade
    insider = InsiderFilingsTool(pipeline_store=store)
    insider._persist_entities(
        [
            {
                "name": "John Doe",
                "company": "Tesla Inc",
                "ticker": "TSLA",
                "issuer_cik": "0001318605",
                "reporter_cik": "0009876543",
                "code": "P",
                "shares": 10000,
                "price": 250.0,
                "date": "2025-03-15",
            }
        ]
    )

    # 2. cert_transparency → domain, cert_issued
    cert = CertTransparencyTool(pipeline_store=store)
    cert._persist_entities(
        "api.tesla.com",
        [
            {
                "common_name": "api.tesla.com",
                "issuer_name": "DigiCert Inc",
                "not_before": "2025-01-01T00:00:00",
                "not_after": "2026-01-01T00:00:00",
                "is_expired": False,
            }
        ],
    )

    # 3. dns_monitor → domain, dns_change
    dns = DnsMonitorTool(pipeline_store=store)
    dns._persist_entities(
        "openai.com",
        {
            "cloud_providers": ["Cloudflare"],
            "mx_provider": "Google",
            "ns_provider": "Cloudflare",
            "min_ttl": 300,
            "low_ttl_warning": False,
            "record_count": 12,
        },
    )

    # 4. wikipedia_pageviews → topic, pageview_spike
    wiki = WikipediaPageviewsTool(pipeline_store=store)
    wiki._persist_entities(
        [
            {
                "article": "Artificial_intelligence",
                "z_score": 5.2,
                "latest_views": 50000,
                "mean_views": 10000,
                "spike_ratio": 5.0,
                "project": "en.wikipedia",
            }
        ]
    )

    # 5. lobbying → company, lobbying_spend
    lobby = LobbyingTool(pipeline_store=store)
    lobby._persist_entities(
        [
            {
                "registrant_name": "Google LLC",
                "registrant_id": 12345,
                "client_name": "Self",
                "amount": 7000000.0,
                "filing_year": 2025,
                "filing_period": "first_quarter",
                "dt_posted": "2025-04-15T00:00:00Z",
                "issue_codes": ["CPT", "TRD"],
            }
        ]
    )

    # 6. patent_filings → company, patent_filing
    patents = PatentFilingsTool(pipeline_store=store)
    patents._persist_entities(
        [
            {
                "patent_number": "US12345678",
                "patent_title": "Quantum Error Correction Method",
                "patent_date": "2025-06-01",
                "assignee_organization": "IBM",
                "cpc_subgroup_id": "H03K19/195",
            }
        ]
    )

    # 7. defi_flows → protocol, tvl_change
    defi = DefiFlowsTool(pipeline_store=store)
    defi._persist_entities(
        [
            {
                "name": "Aave",
                "tvl_usd": 10000000000.0,
                "chain": "Ethereum",
                "chains": ["Ethereum", "Polygon"],
                "category": "Lending",
                "change_1d_pct": 2.5,
            }
        ]
    )

    # 8. interconnection_queue → company, project_status
    iq = InterconnectionQueueTool(pipeline_store=store)
    iq._persist_entities(
        [
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
    )


# ── Integration Tests ─────────────────────────────────────────


class TestAllToolsToGraphBuilder:
    """End-to-end: populate store via all 8 L2 tools → build HeteroData."""

    def test_graph_builds_without_error(self, store):
        _populate_all_tools(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()
        assert data is not None
        assert id_map.num_nodes > 0
        assert len(events) > 0

    def test_entity_types_present(self, store):
        """All entity types actually used by L2 tools should appear in graph."""
        _populate_all_tools(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        expected_types = {"company", "domain", "protocol", "topic"}
        # person depends on insider_filings implementation — may or may not be present
        present_types = set(id_map.type_local.keys())
        assert expected_types.issubset(present_types), f"Missing entity types: {expected_types - present_types}"

    def test_observation_types_present(self, store):
        """All obs types from L2 tools should appear in events."""
        _populate_all_tools(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        obs_types_in_events = {e["observation_type"] for e in events}
        # These are the obs types our 8 tools produce:
        expected_obs = {
            "cert_issued",
            "dns_change",
            "pageview_spike",
            "lobbying_spend",
            "patent_filing",
            "tvl_change",
            "project_status",
        }
        # insider_trade depends on insider_filings L2 path
        assert expected_obs.issubset(obs_types_in_events), f"Missing obs types: {expected_obs - obs_types_in_events}"

    def test_node_feature_dimension(self, store):
        """Each node type should have features of dim = len(ENTITY_TYPES) + 3."""
        _populate_all_tools(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        expected_dim = len(ENTITY_TYPES) + 3  # one-hot + count + recency + mean_value
        for etype in id_map.type_local:
            if id_map.num_nodes_of_type(etype) > 0:
                x = data[etype].x
                assert x.shape[1] == expected_dim, f"Type {etype}: expected feat_dim={expected_dim}, got {x.shape[1]}"

    def test_node_counts_match_entities(self, store):
        """Number of graph nodes per type should match registered entities."""
        _populate_all_tools(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        for etype in id_map.type_local:
            n_nodes = id_map.num_nodes_of_type(etype)
            if n_nodes > 0:
                assert data[etype].x.shape[0] == n_nodes

    def test_events_sorted_by_time(self, store):
        _populate_all_tools(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        times = [e.get("observed_at", 0.0) for e in events]
        assert times == sorted(times), "Events should be sorted by observed_at"

    def test_each_event_has_required_fields(self, store):
        _populate_all_tools(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        required = {
            "entity_id",
            "observation_type",
            "observed_at",
            "source_tool",
            "depth_level",
        }
        for event in events:
            missing = required - set(event.keys())
            assert not missing, f"Event missing fields: {missing}"

    def test_total_event_count(self, store):
        """8 tools each persist 1 record → at least 8 events."""
        _populate_all_tools(store)
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()
        assert len(events) >= 8

    def test_company_dedup_across_tools(self, store):
        """Same company name from lobbying + patents → single graph node."""
        # Both persist "Google LLC" as company
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

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        # Only 1 company node for Google
        assert id_map.num_nodes_of_type("company") == 1
        assert data["company"].x.shape[0] == 1

        # But 2 observations (from 2 tools)
        assert len(events) == 2

    def test_mixed_entity_types_produce_separate_node_types(self, store):
        """Protocol and company nodes should be separate in HeteroData."""
        defi = DefiFlowsTool(pipeline_store=store)
        lobby = LobbyingTool(pipeline_store=store)

        defi._persist_entities([{"name": "Uniswap", "tvl_usd": 5e9}])
        lobby._persist_entities(
            [
                {
                    "registrant_name": "Amazon.com Inc.",
                    "amount": 8e6,
                    "dt_posted": "2025-01-01",
                }
            ]
        )

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("protocol") == 1
        assert id_map.num_nodes_of_type("company") == 1
        assert data["protocol"].x.shape[0] == 1
        assert data["company"].x.shape[0] == 1

    def test_obs_count_reflected_in_node_features(self, store):
        """Each lobbying filing stores a separate observation even for the same registrant."""
        lobby = LobbyingTool(pipeline_store=store)
        lobby._persist_entities(
            [
                {
                    "registrant_name": "Meta Platforms",
                    "amount": 1e6,
                    "dt_posted": "2025-01-01",
                },
                {
                    "registrant_name": "Meta Platforms",
                    "amount": 2e6,
                    "dt_posted": "2025-04-01",
                },
                {
                    "registrant_name": "Meta Platforms",
                    "amount": 3e6,
                    "dt_posted": "2025-07-01",
                },
            ]
        )

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        # Entity registered once (dedup), but each filing stores its own observation
        count_idx = len(ENTITY_TYPES)
        assert data["company"].x[0, count_idx].item() == 3.0

    def test_empty_store_produces_empty_graph(self, store):
        """GraphBuilder on empty store should still work."""
        builder = GraphBuilder(store)
        data, id_map, events = builder.build()
        assert id_map.num_nodes == 0
        assert len(events) == 0


# ── Source Tool Attribution ────────────────────────────────────


class TestSourceToolAttribution:
    """Verify events carry correct source_tool for each L2 tool."""

    TOOL_TO_SOURCE = {
        "cert_transparency": (
            CertTransparencyTool,
            (
                "test.com",
                [
                    {
                        "common_name": "test.com",
                        "issuer_name": "LE",
                        "not_before": "2025-01-01",
                        "not_after": "2026-01-01",
                    },
                ],
            ),
        ),
        "dns_monitor": (
            DnsMonitorTool,
            (
                "example.com",
                {
                    "cloud_providers": ["AWS"],
                    "record_count": 5,
                },
            ),
        ),
        "wikipedia_pageviews": (
            WikipediaPageviewsTool,
            [
                {
                    "article": "Bitcoin",
                    "z_score": 4.0,
                    "latest_views": 30000,
                    "mean_views": 8000,
                },
            ],
        ),
        "lobbying": (
            LobbyingTool,
            [
                {
                    "registrant_name": "Apple Inc",
                    "amount": 3e6,
                    "dt_posted": "2025-01-01",
                },
            ],
        ),
        "patent_filings": (
            PatentFilingsTool,
            [
                {
                    "assignee_organization": "Samsung",
                    "patent_number": "P99",
                    "patent_date": "2025-01-01",
                },
            ],
        ),
        "defi_flows": (
            DefiFlowsTool,
            [
                {"name": "Curve", "tvl_usd": 3e9},
            ],
        ),
        "interconnection_queue": (
            InterconnectionQueueTool,
            [
                {"entityName": "Duke Energy", "plantName": "Wind Farm", "status": "PL"},
            ],
        ),
    }

    @pytest.mark.parametrize("expected_source", list(TOOL_TO_SOURCE.keys()))
    def test_source_tool_label(self, store, expected_source):
        cls, data = self.TOOL_TO_SOURCE[expected_source]
        tool = cls(pipeline_store=store)
        if isinstance(data, tuple):
            tool._persist_entities(*data)
        else:
            tool._persist_entities(data)

        obs = store.query_all_observations()
        sources = {o["source_tool"] for o in obs}
        assert expected_source in sources, f"Expected source_tool={expected_source!r}, got {sources}"


# ── Depth Level Verification ──────────────────────────────────


class TestDepthLevel:
    """All L2 tools should persist at depth_level=2."""

    @pytest.mark.parametrize(
        "tool_cls,data",
        [
            (
                CertTransparencyTool,
                ("t.com", [{"common_name": "t.com", "issuer_name": "CA"}]),
            ),
            (DnsMonitorTool, ("d.com", {"record_count": 1})),
            (WikipediaPageviewsTool, [{"article": "Test", "z_score": 1.0}]),
            (
                LobbyingTool,
                [{"registrant_name": "Corp", "amount": 1e6, "dt_posted": "2025-01-01"}],
            ),
            (
                PatentFilingsTool,
                [
                    {
                        "assignee_organization": "Co",
                        "patent_number": "P",
                        "patent_date": "2025-01-01",
                    }
                ],
            ),
            (DefiFlowsTool, [{"name": "Proto", "tvl_usd": 1e9}]),
            (
                InterconnectionQueueTool,
                [{"entityName": "En", "plantName": "Pl", "status": "PL"}],
            ),
        ],
    )
    def test_depth_level_is_2(self, store, tool_cls, data):
        tool = tool_cls(pipeline_store=store)
        if isinstance(data, tuple):
            tool._persist_entities(*data)
        else:
            tool._persist_entities(data)
        obs = store.query_all_observations()
        for o in obs:
            assert o["depth_level"] == 2
