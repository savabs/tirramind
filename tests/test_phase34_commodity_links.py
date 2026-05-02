"""Tests for Phase 34 — Commodity Country Links + Graph Diagnostics.

Covers:
- InstrumentDef.primary_exchange_country field
- _persist_instrument_links: exchange_country link creation
- graph_diagnostics.diagnose_graph
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.models.gnn.graph_builder import ENRICHMENT_DIM, OBSERVATION_TYPES
from agent.models.gnn.graph_diagnostics import diagnose_graph
from agent.tools.instrument_universe import (
    INSTRUMENTS,
    InstrumentDef,
    _persist_instrument_links,
    instruments_by_class,
)

# ── Helpers ───────────────────────────────────────────────────


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.link_entities = MagicMock(return_value=1)
    return store


def _make_diag_store(
    entities: list[dict] | None = None,
    observations: list[dict] | None = None,
    links: list[dict] | None = None,
) -> MagicMock:
    store = MagicMock()
    store.query_all_entities.return_value = entities or []
    store.query_all_observations.return_value = observations or []
    store.query_all_entity_links.return_value = links or []
    return store


# ═══════════════════════════════════════════════════════════════
# InstrumentDef.primary_exchange_country field
# ═══════════════════════════════════════════════════════════════


class TestInstrumentDefExchangeCountry:
    """Phase 34: primary_exchange_country field on InstrumentDef."""

    def test_field_defaults_to_none(self):
        inst = InstrumentDef("TEST", "Test Inst", "vol", "US")
        assert inst.primary_exchange_country is None

    def test_field_set_explicitly(self):
        inst = InstrumentDef(
            "TEST",
            "Test Inst",
            "commodity_future",
            "US",
            primary_exchange_country="US",
        )
        assert inst.primary_exchange_country == "US"

    def test_all_commodity_futures_have_exchange_country(self):
        commodities = instruments_by_class("commodity_future")
        assert len(commodities) == 20
        for inst in commodities:
            assert inst.primary_exchange_country == "US", f"{inst.ticker} missing primary_exchange_country"

    def test_non_commodity_instruments_default_none(self):
        """Non-commodity instruments should NOT have primary_exchange_country set."""
        non_commodities = [i for i in INSTRUMENTS if i.asset_class != "commodity_future"]
        for inst in non_commodities:
            assert inst.primary_exchange_country is None, (
                f"{inst.ticker} has unexpected primary_exchange_country={inst.primary_exchange_country}"
            )

    def test_frozen_dataclass(self):
        inst = InstrumentDef(
            "TEST",
            "Test",
            "commodity_future",
            "US",
            primary_exchange_country="US",
        )
        with pytest.raises(AttributeError):
            inst.primary_exchange_country = "UK"  # type: ignore[misc]

    def test_commodity_country_remains_none(self):
        """Commodity futures should still have country=None."""
        commodities = instruments_by_class("commodity_future")
        for inst in commodities:
            assert inst.country is None, f"{inst.ticker} has country={inst.country}, expected None"


# ═══════════════════════════════════════════════════════════════
# _persist_instrument_links: exchange_country
# ═══════════════════════════════════════════════════════════════


class TestPersistExchangeCountryLinks:
    """Phase 34: exchange_country link creation."""

    def test_returns_exchange_country_count(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        assert "exchange_country" in result

    def test_creates_exchange_country_links_for_commodities(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        # All 20 commodity futures should create exchange_country links
        assert result["exchange_country"] == 20

    def test_exchange_country_link_type_correct(self):
        store = _make_store()
        _persist_instrument_links(store)
        exc_links = [c for c in store.link_entities.call_args_list if c.kwargs.get("link_type") == "exchange_country"]
        assert len(exc_links) == 20

    def test_exchange_country_link_confidence(self):
        store = _make_store()
        _persist_instrument_links(store)
        exc_links = [c for c in store.link_entities.call_args_list if c.kwargs.get("link_type") == "exchange_country"]
        for call in exc_links:
            assert call.kwargs["confidence"] == 1.0

    def test_exchange_country_link_source(self):
        store = _make_store()
        _persist_instrument_links(store)
        exc_links = [c for c in store.link_entities.call_args_list if c.kwargs.get("link_type") == "exchange_country"]
        for call in exc_links:
            assert call.kwargs["source"] == "instrument_universe"

    def test_exchange_country_link_metadata_has_ticker(self):
        store = _make_store()
        _persist_instrument_links(store)
        exc_links = [c for c in store.link_entities.call_args_list if c.kwargs.get("link_type") == "exchange_country"]
        commodity_tickers = {i.ticker for i in instruments_by_class("commodity_future")}
        link_tickers = {c.kwargs["metadata"]["ticker"] for c in exc_links}
        assert link_tickers == commodity_tickers

    def test_no_exchange_country_for_non_commodity(self):
        """Instruments without primary_exchange_country should not get exchange_country links."""
        store = _make_store()
        _persist_instrument_links(store)
        exc_links = [c for c in store.link_entities.call_args_list if c.kwargs.get("link_type") == "exchange_country"]
        # Should only be 20 (commodity futures)
        assert len(exc_links) == 20

    def test_registers_country_entity_for_exchange(self):
        store = _make_store()
        _persist_instrument_links(store)
        # At least one country registration should be for "US" via exchange_country
        country_regs = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "country" and c.kwargs.get("canonical_name") == "US"
        ]
        assert len(country_regs) > 0

    def test_total_link_counts_include_exchange_country(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        total = sum(result.values())
        assert total > result["exchange_country"]  # other link types also created

    def test_link_entities_fail_returns_zero_count(self):
        """When store.link_entities returns None for exchange_country, count stays 0."""
        store = _make_store()
        original_side_effect = store.link_entities.side_effect

        def selective_fail(**kw):
            if kw.get("link_type") == "exchange_country":
                return None
            return 1

        store.link_entities = MagicMock(side_effect=selective_fail)
        result = _persist_instrument_links(store)
        assert result["exchange_country"] == 0

    def test_existing_link_types_unaffected(self):
        """Adding exchange_country should not change counts for other link types."""
        store = _make_store()
        result = _persist_instrument_links(store)
        # These should still have the same counts as before Phase 34
        assert result["tracks_issuer"] > 0
        assert result["inst_country"] > 0
        assert result["issuer_country"] > 0
        assert result["fx_base_country"] > 0
        assert result["fx_quote_country"] > 0
        assert result["tracks_protocol"] > 0


# ═══════════════════════════════════════════════════════════════
# Graph Builder: unchanged (dynamic edge types)
# ═══════════════════════════════════════════════════════════════


class TestGraphBuilderStability:
    """Verify graph_builder constants are stable after Phase 34."""

    def test_observation_types_count(self):
        assert len(OBSERVATION_TYPES) == 46

    def test_enrichment_dim(self):
        assert ENRICHMENT_DIM == 55

    def test_observation_types_sorted(self):
        assert sorted(OBSERVATION_TYPES) == OBSERVATION_TYPES


# ═══════════════════════════════════════════════════════════════
# Graph Diagnostics
# ═══════════════════════════════════════════════════════════════


class TestDiagnoseGraphEmpty:
    """diagnose_graph on an empty store."""

    def test_empty_store_returns_zeros(self):
        store = _make_diag_store()
        result = diagnose_graph(store)
        assert result["total_entities"] == 0
        assert result["total_observations"] == 0
        assert result["total_links"] == 0

    def test_empty_store_no_orphans(self):
        store = _make_diag_store()
        result = diagnose_graph(store)
        assert result["orphan_entities"] == []

    def test_empty_store_obs_types_without_instances(self):
        store = _make_diag_store()
        result = diagnose_graph(store)
        # All OBSERVATION_TYPES should be listed as having zero instances
        assert set(result["obs_types_without_instances"]) == set(OBSERVATION_TYPES)

    def test_empty_store_entity_types_without_obs(self):
        store = _make_diag_store()
        result = diagnose_graph(store)
        assert result["entity_types_without_obs"] == []  # no entity types present


class TestDiagnoseGraphPopulated:
    """diagnose_graph with populated store."""

    def test_entity_counts(self):
        entities = [
            {"entity_id": "e1", "entity_type": "company", "canonical_name": "Acme"},
            {"entity_id": "e2", "entity_type": "company", "canonical_name": "Beta"},
            {"entity_id": "e3", "entity_type": "country", "canonical_name": "US"},
        ]
        store = _make_diag_store(entities=entities)
        result = diagnose_graph(store)
        assert result["entity_counts"] == {"company": 2, "country": 1}
        assert result["total_entities"] == 3

    def test_observation_counts(self):
        observations = [
            {
                "entity_id": "e1",
                "observation_type": "insider_trade",
                "observed_at": 1.0,
            },
            {
                "entity_id": "e1",
                "observation_type": "insider_trade",
                "observed_at": 2.0,
            },
            {"entity_id": "e2", "observation_type": "whale_trade", "observed_at": 3.0},
        ]
        store = _make_diag_store(observations=observations)
        result = diagnose_graph(store)
        assert result["observation_counts"]["insider_trade"] == 2
        assert result["observation_counts"]["whale_trade"] == 1
        assert result["total_observations"] == 3

    def test_link_counts(self):
        links = [
            {"entity_id_a": "e1", "entity_id_b": "e2", "link_type": "located_in"},
            {"entity_id_a": "e1", "entity_id_b": "e3", "link_type": "located_in"},
            {"entity_id_a": "e2", "entity_id_b": "e3", "link_type": "tracks_issuer"},
        ]
        store = _make_diag_store(links=links)
        result = diagnose_graph(store)
        assert result["link_counts"] == {"located_in": 2, "tracks_issuer": 1}
        assert result["total_links"] == 3

    def test_orphan_detection(self):
        entities = [
            {"entity_id": "e1", "entity_type": "company", "canonical_name": "Acme"},
            {"entity_id": "e2", "entity_type": "company", "canonical_name": "Beta"},
            {"entity_id": "e3", "entity_type": "country", "canonical_name": "US"},
        ]
        links = [
            {"entity_id_a": "e1", "entity_id_b": "e3", "link_type": "located_in"},
        ]
        store = _make_diag_store(entities=entities, links=links)
        result = diagnose_graph(store)
        # e2 is not in any link
        orphan_ids = {o["entity_id"] for o in result["orphan_entities"]}
        assert orphan_ids == {"e2"}

    def test_no_orphans_when_all_linked(self):
        entities = [
            {"entity_id": "e1", "entity_type": "company", "canonical_name": "Acme"},
            {"entity_id": "e2", "entity_type": "country", "canonical_name": "US"},
        ]
        links = [
            {"entity_id_a": "e1", "entity_id_b": "e2", "link_type": "located_in"},
        ]
        store = _make_diag_store(entities=entities, links=links)
        result = diagnose_graph(store)
        assert result["orphan_entities"] == []

    def test_entity_as_link_target_not_orphan(self):
        """Entity appearing only as entity_id_b should not be an orphan."""
        entities = [
            {"entity_id": "e1", "entity_type": "company", "canonical_name": "Acme"},
            {"entity_id": "e2", "entity_type": "country", "canonical_name": "US"},
        ]
        links = [
            {"entity_id_a": "e1", "entity_id_b": "e2", "link_type": "located_in"},
        ]
        store = _make_diag_store(entities=entities, links=links)
        result = diagnose_graph(store)
        assert result["orphan_entities"] == []

    def test_entity_types_without_obs(self):
        entities = [
            {"entity_id": "e1", "entity_type": "company", "canonical_name": "Acme"},
            {"entity_id": "e2", "entity_type": "country", "canonical_name": "US"},
        ]
        observations = [
            {
                "entity_id": "e1",
                "observation_type": "insider_trade",
                "observed_at": 1.0,
            },
        ]
        store = _make_diag_store(entities=entities, observations=observations)
        result = diagnose_graph(store)
        # country type has e2 but no observations for e2
        assert "country" in result["entity_types_without_obs"]
        assert "company" not in result["entity_types_without_obs"]

    def test_obs_types_without_instances(self):
        observations = [
            {
                "entity_id": "e1",
                "observation_type": "insider_trade",
                "observed_at": 1.0,
            },
        ]
        store = _make_diag_store(observations=observations)
        result = diagnose_graph(store)
        assert "insider_trade" not in result["obs_types_without_instances"]
        assert "whale_trade" in result["obs_types_without_instances"]

    def test_result_keys_complete(self):
        store = _make_diag_store()
        result = diagnose_graph(store)
        expected_keys = {
            "entity_counts",
            "observation_counts",
            "link_counts",
            "orphan_entities",
            "entity_types_without_obs",
            "obs_types_without_instances",
            "total_entities",
            "total_observations",
            "total_links",
        }
        assert set(result.keys()) == expected_keys

    def test_multiple_orphans_different_types(self):
        entities = [
            {"entity_id": "e1", "entity_type": "company", "canonical_name": "Acme"},
            {"entity_id": "e2", "entity_type": "country", "canonical_name": "US"},
            {"entity_id": "e3", "entity_type": "instrument", "canonical_name": "SPY"},
        ]
        store = _make_diag_store(entities=entities)
        result = diagnose_graph(store)
        assert len(result["orphan_entities"]) == 3
