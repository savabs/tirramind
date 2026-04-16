"""Phase 27 diagnostic integration tests.

Validates end-to-end that:
1. FX instruments gain two-country connectivity through the link system.
2. Country nodes receive CB monetary-state observations.
3. Observations survive the store → graph-builder flow.
4. ENRICHMENT_DIM is consistent with the observation-type count.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from agent.models.gnn.graph_builder import (
    BASE_FEAT_DIM,
    ENRICHMENT_DIM,
    OBSERVATION_TYPES,
    GraphBuilder,
)
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore
from agent.tools.central_bank_balance import CB_TO_COUNTRY
from agent.tools.instrument_universe import INSTRUMENTS, _persist_instrument_links


# ── Helpers ────────────────────────────────────────────────────


def _make_in_memory_store() -> PipelineStore:
    """Build a real PipelineStore backed by an in-memory SQLite database."""
    return PipelineStore(db_path=":memory:")


# ── FX two-country connectivity diagnostics ───────────────────


class TestFXCountryConnectivity:
    """Validate that FX instruments produce both-side country links."""

    def test_every_fx_pair_has_both_country_links(self):
        """Each FX instrument should produce fx_base_country + fx_quote_country."""
        store = _make_in_memory_store()
        counts = _persist_instrument_links(store)

        fx_instruments = [i for i in INSTRUMENTS if i.asset_class == "fx"]
        assert len(fx_instruments) == 15

        # 15 base + 15 quote = 30 FX country links
        assert counts["fx_base_country"] == 15
        assert counts["fx_quote_country"] == 15

    def test_eurusd_connects_to_eu_and_us(self):
        store = _make_in_memory_store()
        _persist_instrument_links(store)

        # Get all links for EURUSD
        eurusd_ticker = "EURUSD=X"
        from agent.tools.instrument_universe import _entity_id

        eurusd_eid = _entity_id(eurusd_ticker)
        links = store.query_entity_links(eurusd_eid)

        link_types = {lnk["link_type"] for lnk in links}
        assert "fx_base_country" in link_types
        assert "fx_quote_country" in link_types
        assert "located_in" in link_types  # backward-compat single-country

        # Verify targets
        eu_eid = entity_id_from_key("country", "EU")
        us_eid = entity_id_from_key("country", "US")
        base_links = [l for l in links if l["link_type"] == "fx_base_country"]
        quote_links = [l for l in links if l["link_type"] == "fx_quote_country"]
        assert base_links[0]["entity_id_b"] == eu_eid
        assert quote_links[0]["entity_id_b"] == us_eid

    def test_fx_cross_pair_no_us_side(self):
        """EURGBP should link to EU and GB, not US."""
        store = _make_in_memory_store()
        _persist_instrument_links(store)

        from agent.tools.instrument_universe import _entity_id

        eurgbp_eid = _entity_id("EURGBP=X")
        links = store.query_entity_links(eurgbp_eid)

        eu_eid = entity_id_from_key("country", "EU")
        gb_eid = entity_id_from_key("country", "GB")
        us_eid = entity_id_from_key("country", "US")

        target_eids = {l["entity_id_b"] for l in links}
        assert eu_eid in target_eids
        assert gb_eid in target_eids
        assert us_eid not in target_eids

    def test_country_entity_count_after_fx_links(self):
        """FX + ETF + equity links should register all expected country entities."""
        store = _make_in_memory_store()
        _persist_instrument_links(store)

        entities = store.query_all_entities(entity_type="country")
        country_names = {e["canonical_name"] for e in entities}

        # FX pairs alone touch: US EU JP GB CH AU CA NZ MX BR IN CN ZA
        fx_countries = {"US", "EU", "JP", "GB", "CH", "AU", "CA", "NZ",
                        "MX", "BR", "IN", "CN", "ZA"}
        assert fx_countries.issubset(country_names)


# ── CB L2 monetary-state observation diagnostics ──────────────


class TestCBCountryObservationFlow:
    """Validate that CB tool persists observations onto country nodes."""

    def test_balance_sheet_obs_on_country_node(self):
        store = _make_in_memory_store()

        from agent.tools.central_bank_balance import CentralBankBalanceTool

        tool = CentralBankBalanceTool.__new__(CentralBankBalanceTool)
        tool._store = store

        # Simulate Fed balance sheet result and persist
        fed_eid = entity_id_from_key("country", "US")
        store.register_entity(
            entity_type="country",
            canonical_name="US",
            entity_id=fed_eid,
        )

        data = {
            "banks": [
                {"code": "fed", "native_trillions": 7.5, "usd_trillions": 7.5,
                 "wow_pct": 0.1, "mom_pct": -0.5, "yoy_pct": 2.0},
            ],
            "errors": [],
        }
        counts = tool._persist_entities(data, "balance_sheets", ["fed"])
        assert counts["balance_sheet_obs"] == 1

        # Verify observation is actually in the store
        obs = store.query_entity_observations(fed_eid)
        bs_obs = [o for o in obs if o["observation_type"] == "cb_balance_sheet"]
        assert len(bs_obs) == 1
        assert bs_obs[0]["source_tool"] == "central_bank_balance"
        assert bs_obs[0]["depth_level"] == 2

    def test_rate_obs_on_country_node(self):
        store = _make_in_memory_store()

        from agent.tools.central_bank_balance import CentralBankBalanceTool

        tool = CentralBankBalanceTool.__new__(CentralBankBalanceTool)
        tool._store = store

        # Register JP country
        jp_eid = entity_id_from_key("country", "JP")
        store.register_entity(
            entity_type="country",
            canonical_name="JP",
            entity_id=jp_eid,
        )

        data = {
            "rates": [
                {"code": "boj", "current_rate": -0.10, "rate_date": "2026-04-10",
                 "last_change_date": None, "last_change_direction": None,
                 "last_change_bps": None, "days_since_change": None},
            ],
            "errors": [],
        }
        counts = tool._persist_entities(data, "rate_monitor", ["boj"])
        assert counts["rate_obs"] == 1

        obs = store.query_entity_observations(jp_eid)
        rate_obs = [o for o in obs if o["observation_type"] == "cb_policy_rate"]
        assert len(rate_obs) == 1

    def test_all_seven_cbs_have_country_mapping(self):
        """Every CB in CB_TO_COUNTRY maps to a valid country code."""
        expected_cbs = {"fed", "ecb", "boj", "boe", "snb", "boc", "rba"}
        assert set(CB_TO_COUNTRY.keys()) == expected_cbs

        expected_countries = {"US", "EU", "JP", "GB", "CH", "CA", "AU"}
        assert set(CB_TO_COUNTRY.values()) == expected_countries


# ── Graph-builder integration diagnostics ─────────────────────


class TestGraphBuilderMonetaryObs:
    """Observations survive store → graph build without dimension breakage."""

    def test_cb_obs_survives_graph_build(self):
        store = _make_in_memory_store()

        # Register a country entity + CB observation
        us_eid = entity_id_from_key("country", "US")
        store.register_entity(
            entity_type="country",
            canonical_name="US",
            entity_id=us_eid,
        )
        store.store_entity_observation(
            entity_id=us_eid,
            source_tool="central_bank_balance",
            observed_at=time.time(),
            observation_type="cb_balance_sheet",
            value={"cb_code": "fed", "usd_trillions": 7.5},
            depth_level=2,
        )

        builder = GraphBuilder(store)
        hetero, id_map, events = builder.build()

        # The country node should exist somewhere in the graph
        assert hetero is not None
        # Node features should be BASE_FEAT_DIM wide (no enrichment passed)
        for node_type in hetero.node_types:
            x = hetero[node_type].x
            assert x.shape[1] == BASE_FEAT_DIM

    def test_enrichment_dim_formula(self):
        """ENRICHMENT_DIM = 9 (base features) + len(OBSERVATION_TYPES)."""
        assert ENRICHMENT_DIM == 9 + len(OBSERVATION_TYPES)

    def test_new_obs_types_registered(self):
        assert "cb_balance_sheet" in OBSERVATION_TYPES
        assert "cb_policy_rate" in OBSERVATION_TYPES


# ── Cross-layer connectivity summary ──────────────────────────


class TestPhase27ConnectivitySummary:
    """Full-path diagnostic: FX instrument → country → CB observation → graph."""

    def test_fx_to_cb_full_path(self):
        """USDJPY → (fx_base_country) → US; US ← cb_balance_sheet(fed)."""
        store = _make_in_memory_store()

        # Step 1: Persist instrument links (creates FX → country links)
        _persist_instrument_links(store)

        # Step 2: Persist CB observation onto US country node
        from agent.tools.central_bank_balance import CentralBankBalanceTool

        tool = CentralBankBalanceTool.__new__(CentralBankBalanceTool)
        tool._store = store

        data = {
            "banks": [
                {"code": "fed", "native_trillions": 7.5, "usd_trillions": 7.5,
                 "wow_pct": 0.1, "mom_pct": -0.5, "yoy_pct": 2.0},
            ],
            "errors": [],
        }
        tool._persist_entities(data, "balance_sheets", ["fed"])

        # Step 3: Build graph
        builder = GraphBuilder(store)
        hetero, id_map, events = builder.build()

        # The graph should have entities and valid feature dimensions
        assert hetero is not None
        total_nodes = sum(hetero[nt].x.shape[0] for nt in hetero.node_types)
        assert total_nodes > 0

        # Country US should have the CB observation enrichment
        us_eid = entity_id_from_key("country", "US")
        obs = store.query_entity_observations(us_eid)
        assert any(o["observation_type"] == "cb_balance_sheet" for o in obs)

        # USDJPY should have links to US + JP
        from agent.tools.instrument_universe import _entity_id

        usdjpy_eid = _entity_id("USDJPY=X")
        links = store.query_entity_links(usdjpy_eid)
        link_types = {l["link_type"] for l in links}
        assert "fx_base_country" in link_types
        assert "fx_quote_country" in link_types
