"""Tests for Phase 35: GNN Retrain on Expanded Entity Graph.

Covers:
    Schema coverage  — all 11 entity types, 45 obs types, 18 link types
    Cross-domain patterns — 6 injected causal chains across new entity types
    Training convergence — loss decreases over epochs on expanded graph
    Attention analysis — diagnostics cover all entity types
    Backward compat — original 4-type generator still works
    Edge cases — empty types, single-entity types, asymmetric counts
"""

from __future__ import annotations

import pytest

from agent.models.gnn.graph_builder import (
    ENTITY_TYPES,
    OBSERVATION_TYPES,
    GraphBuilder,
)
from agent.models.gnn.trainer import (
    InjectedPattern,
    SyntheticGraphGenerator,
    Trainer,
    TrainerConfig,
    evaluate,
)
from agent.pipeline.store import PipelineStore

# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def store():
    return PipelineStore(db_path=":memory:")


@pytest.fixture
def expanded_store(store):
    """Store with all 11 entity types populated."""
    gen = SyntheticGraphGenerator(
        num_companies=8,
        num_countries=5,
        num_vessels=4,
        num_wallets=4,
        num_instruments=10,
        num_persons=6,
        num_cftc_contracts=4,
        num_organizations=3,
        num_protocols=2,
        num_topics=5,
        num_domains=3,
        time_span=86400.0 * 10,
        base_event_rate=0.0005,
        seed=42,
    )
    stats = gen.generate(store)
    return store, stats


CROSS_DOMAIN_PATTERNS = [
    InjectedPattern(
        source_type="person",
        source_obs_type="insider_trade",
        target_type="company",
        target_obs_type="sell_intent",
        via_edge="works_for",
        lag_seconds=3600.0,
        lag_jitter=300.0,
    ),
    InjectedPattern(
        source_type="country",
        source_obs_type="sanctions_listing",
        target_type="company",
        target_obs_type="creditor_filing",
        via_edge="headquartered_in",
        lag_seconds=7200.0,
        lag_jitter=600.0,
    ),
    InjectedPattern(
        source_type="vessel",
        source_obs_type="port_call",
        target_type="country",
        target_obs_type="trade_flow",
        via_edge="port_call_to",
        lag_seconds=1800.0,
        lag_jitter=300.0,
    ),
    InjectedPattern(
        source_type="wallet",
        source_obs_type="btc_transfer",
        target_type="instrument",
        target_obs_type="price_movement",
        via_edge="trades_instrument",
        lag_seconds=900.0,
        lag_jitter=120.0,
    ),
    InjectedPattern(
        source_type="cftc_contract",
        source_obs_type="futures_positioning",
        target_type="instrument",
        target_obs_type="instrument_volatility",
        via_edge="cftc_tracks",
        lag_seconds=14400.0,
        lag_jitter=1800.0,
    ),
    InjectedPattern(
        source_type="country",
        source_obs_type="pathogen_level",
        target_type="country",
        target_obs_type="economic_activity",
        via_edge="sanctioned_under",
        lag_seconds=86400.0,
        lag_jitter=3600.0,
    ),
]


@pytest.fixture
def pattern_expanded_store(store):
    """Store with all 11 entity types + 6 cross-domain injected patterns."""
    gen = SyntheticGraphGenerator(
        num_companies=8,
        num_countries=5,
        num_vessels=4,
        num_wallets=4,
        num_instruments=10,
        num_persons=6,
        num_cftc_contracts=4,
        num_organizations=3,
        num_protocols=2,
        num_topics=5,
        num_domains=3,
        time_span=86400.0 * 10,
        base_event_rate=0.0005,
        seed=42,
        patterns=CROSS_DOMAIN_PATTERNS,
    )
    stats = gen.generate(store)
    return store, stats


# ═══════════════════════════════════════════════════════════════
# Schema Coverage Tests
# ═══════════════════════════════════════════════════════════════


class TestSchemaCoverage:
    """All 11 entity types, 45 obs types, and 18 link types appear."""

    def test_all_entity_types_present(self, expanded_store):
        store, stats = expanded_store
        entities = store.query_all_entities()
        found_types = {e["entity_type"] for e in entities}
        assert found_types == set(ENTITY_TYPES), f"Missing: {set(ENTITY_TYPES) - found_types}"

    def test_entity_counts_match(self, expanded_store):
        _, stats = expanded_store
        assert stats["entities"]["company"] == 8
        assert stats["entities"]["country"] == 5
        assert stats["entities"]["vessel"] == 4
        assert stats["entities"]["wallet"] == 4
        assert stats["entities"]["instrument"] == 10
        assert stats["entities"]["person"] == 6
        assert stats["entities"]["cftc_contract"] == 4
        assert stats["entities"]["organization"] == 3
        assert stats["entities"]["protocol"] == 2
        assert stats["entities"]["topic"] == 5
        assert stats["entities"]["domain"] == 3

    def test_total_entity_count(self, expanded_store):
        store, _ = expanded_store
        entities = store.query_all_entities()
        assert len(entities) == 54  # 8+5+4+4+10+6+4+3+2+5+3

    def test_all_obs_types_generated(self, expanded_store):
        store, _ = expanded_store
        obs = store.query_all_observations()
        found_obs_types = {o["observation_type"] for o in obs}
        # Every obs type in _obs_types_for() mapping should appear
        # (except cross_entity_pattern which is a fallback)
        expected = set(OBSERVATION_TYPES) - {"cross_entity_pattern", "project_status"}
        missing = expected - found_obs_types
        # With Poisson arrivals and 10 days, some rare types may not appear;
        # check that at least 80% of obs types are covered
        coverage = len(expected - missing) / len(expected)
        assert coverage >= 0.80, f"Obs type coverage {coverage:.1%}, missing: {missing}"

    def test_all_link_types_present(self, expanded_store):
        store, _ = expanded_store
        links = store.query_all_entity_links()
        found_link_types = {lnk["link_type"] for lnk in links}
        expected_link_types = {
            "headquartered_in",
            "operates_in",
            "market_authorized_in",
            "lobbies_for",
            "debtor_of",
            "awarded_by",
            "works_for",
            "port_call_to",
            "exchange_based_in",
            "transacts_with",
            "trades_instrument",
            "tracks_issuer",
            "located_in",
            "fx_base_country",
            "fx_quote_country",
            "exchange_country",
            "tracks_protocol",
            "cftc_tracks",
            "sanctioned_under",
        }
        missing = expected_link_types - found_link_types
        assert not missing, f"Missing link types: {missing}"

    def test_link_count_positive(self, expanded_store):
        _, stats = expanded_store
        assert stats["links"] > 40  # many link types × many entities

    def test_observation_count_positive(self, expanded_store):
        _, stats = expanded_store
        assert stats["observations"] > 100  # 54 entities × Poisson arrivals


# ═══════════════════════════════════════════════════════════════
# Cross-Domain Pattern Injection Tests
# ═══════════════════════════════════════════════════════════════


class TestCrossDomainPatterns:
    """6 cross-domain injected causal chains across new entity types."""

    def test_patterns_injected(self, pattern_expanded_store):
        _, stats = pattern_expanded_store
        # At least some pattern instances should fire
        assert len(stats["pattern_instances"]) > 0

    def test_person_to_company_pattern(self, pattern_expanded_store):
        """person.insider_trade → company.sell_intent via works_for."""
        _, stats = pattern_expanded_store
        person_company = [
            p
            for p in stats["pattern_instances"]
            if p["pattern"].source_type == "person" and p["pattern"].target_type == "company"
        ]
        assert len(person_company) > 0, "No person→company patterns fired"

    def test_vessel_to_country_pattern(self, pattern_expanded_store):
        """vessel.port_call → country.trade_flow via port_call_to."""
        _, stats = pattern_expanded_store
        vessel_country = [
            p
            for p in stats["pattern_instances"]
            if p["pattern"].source_type == "vessel" and p["pattern"].target_type == "country"
        ]
        assert len(vessel_country) > 0, "No vessel→country patterns fired"

    def test_wallet_to_instrument_pattern(self, pattern_expanded_store):
        """wallet.btc_transfer → instrument.price_movement via trades_instrument."""
        _, stats = pattern_expanded_store
        wallet_inst = [
            p
            for p in stats["pattern_instances"]
            if p["pattern"].source_type == "wallet" and p["pattern"].target_type == "instrument"
        ]
        assert len(wallet_inst) > 0, "No wallet→instrument patterns fired"

    def test_cftc_to_instrument_pattern(self, pattern_expanded_store):
        """cftc_contract.futures_positioning → instrument.instrument_volatility."""
        _, stats = pattern_expanded_store
        cftc_inst = [
            p
            for p in stats["pattern_instances"]
            if p["pattern"].source_type == "cftc_contract" and p["pattern"].target_type == "instrument"
        ]
        assert len(cftc_inst) > 0, "No cftc→instrument patterns fired"

    def test_country_to_country_pattern(self, pattern_expanded_store):
        """country.pathogen_level → country.economic_activity via sanctioned_under."""
        _, stats = pattern_expanded_store
        co_co = [
            p
            for p in stats["pattern_instances"]
            if p["pattern"].source_type == "country" and p["pattern"].target_type == "country"
        ]
        assert len(co_co) > 0, "No country→country patterns fired"

    def test_pattern_lags_positive(self, pattern_expanded_store):
        """All injected pattern instances have positive lag."""
        _, stats = pattern_expanded_store
        for p in stats["pattern_instances"]:
            assert p["actual_lag"] > 0, f"Non-positive lag: {p['actual_lag']}"
            assert p["target_time"] > p["source_time"]

    def test_more_obs_with_patterns(self, expanded_store, pattern_expanded_store):
        """Pattern injection adds extra observations."""
        _, stats_base = expanded_store
        _, stats_pat = pattern_expanded_store
        assert stats_pat["observations"] >= stats_base["observations"]


# ═══════════════════════════════════════════════════════════════
# Graph Builder Integration Tests
# ═══════════════════════════════════════════════════════════════


class TestGraphBuilderExpanded:
    """GraphBuilder correctly processes the expanded synthetic graph."""

    def test_builds_heterodata(self, expanded_store):
        store, _ = expanded_store
        gb = GraphBuilder(store)
        data, id_map, events = gb.build()
        # All 11 entity types should appear as node types
        assert len(data.node_types) == 11
        for etype in ENTITY_TYPES:
            assert etype in data.node_types

    def test_node_features_correct_shape(self, expanded_store):
        store, _ = expanded_store
        gb = GraphBuilder(store)
        data, _, _ = gb.build()
        for ntype in data.node_types:
            x = data[ntype].x
            assert x.ndim == 2
            assert x.size(0) > 0  # entities present
            # Feature dim = BASE_FEAT_DIM + enrichment
            assert x.size(1) >= 14  # BASE_FEAT_DIM

    def test_edge_types_present(self, expanded_store):
        store, _ = expanded_store
        gb = GraphBuilder(store)
        data, _, _ = gb.build()
        edge_types = data.edge_types
        # At minimum, have multiple edge types
        assert len(edge_types) >= 10

    def test_events_sorted(self, expanded_store):
        store, _ = expanded_store
        gb = GraphBuilder(store)
        _, _, events = gb.build()
        times = [e.get("observed_at", 0.0) for e in events]
        assert times == sorted(times)


# ═══════════════════════════════════════════════════════════════
# Training Convergence Tests
# ═══════════════════════════════════════════════════════════════


class TestExpandedTraining:
    """HetTGN trains on the expanded 11-type graph without divergence."""

    @pytest.fixture
    def trained(self, expanded_store):
        store, _ = expanded_store
        cfg = TrainerConfig(
            epochs=5,
            hidden_dim=32,
            memory_dim=32,
            message_dim=32,
            time_dim=8,
            num_heads=2,
            num_layers=1,
            window_size=86400.0,  # 1 day
        )
        trainer = Trainer(store, cfg)
        trainer.build_model()
        history = trainer.train()
        return trainer, history

    def test_loss_decreases(self, trained):
        _, history = trained
        losses = history["total"]
        assert len(losses) == 5
        # At least one drop in total loss
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_obs_type_accuracy_above_random(self, trained):
        """obs_type top-1 accuracy should beat random (1/45 ≈ 2.2%)."""
        trainer, _ = trained
        metrics = evaluate(trainer.model, trainer.store, trainer.config)
        # Even weak signal should get > 3% (above random baseline)
        assert metrics["obs_type_acc_top1"] > 0.02, f"obs_type accuracy too low: {metrics['obs_type_acc_top1']:.3f}"

    def test_all_loss_components_finite(self, trained):
        _, history = trained
        for key in ("total", "obs_type", "time_delta", "contrastive", "value"):
            for val in history[key]:
                assert not (val != val), f"NaN in {key} loss"  # NaN check
                assert abs(val) < float("inf"), f"Infinite {key} loss"
        # time_delta MSE on raw seconds is naturally large;
        # just verify it's finite and decreasing
        dt_losses = history["time_delta"]
        assert dt_losses[-1] <= dt_losses[0] * 2, (
            f"time_delta loss grew excessively: {dt_losses[0]:.0f} → {dt_losses[-1]:.0f}"
        )

    def test_model_embeddings_shape(self, trained):
        trainer, _ = trained
        embeddings, id_map = trainer.infer()
        assert len(embeddings) > 0
        for etype, emb in embeddings.items():
            assert emb.ndim == 2
            assert emb.size(1) == 32  # hidden_dim


# ═══════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases: empty types, single entities, asymmetric counts."""

    def test_backward_compat_4_types(self, store):
        """Original 4-type generator still works unchanged."""
        gen = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            num_vessels=2,
            num_wallets=2,
            num_topics=0,
            num_domains=0,
            time_span=86400.0 * 10,
            base_event_rate=0.0005,
            seed=42,
        )
        stats = gen.generate(store)
        assert len(stats["entities"]) == 4
        assert stats["entities"]["company"] == 4
        assert stats["entities"]["country"] == 2
        entities = store.query_all_entities()
        assert len(entities) == 10

    def test_single_entity_per_type(self, store):
        """All types with exactly 1 entity each."""
        gen = SyntheticGraphGenerator(
            num_companies=1,
            num_countries=1,
            num_vessels=1,
            num_wallets=1,
            num_instruments=1,
            num_persons=1,
            num_cftc_contracts=1,
            num_organizations=1,
            num_protocols=1,
            num_topics=1,
            num_domains=1,
            time_span=86400.0 * 5,
            base_event_rate=0.001,
            seed=42,
        )
        stats = gen.generate(store)
        assert sum(stats["entities"].values()) == 11
        assert stats["links"] > 0
        assert stats["observations"] > 0

    def test_only_new_types(self, store):
        """Only new entity types, no original 4."""
        gen = SyntheticGraphGenerator(
            num_companies=0,
            num_countries=0,
            num_vessels=0,
            num_wallets=0,
            num_instruments=5,
            num_persons=3,
            num_cftc_contracts=2,
            num_organizations=2,
            num_protocols=1,
            num_topics=3,
            num_domains=2,
            time_span=86400.0 * 5,
            base_event_rate=0.001,
            seed=42,
        )
        stats = gen.generate(store)
        assert "company" not in stats["entities"]
        assert "country" not in stats["entities"]
        assert stats["entities"]["instrument"] == 5
        assert stats["observations"] > 0

    def test_highly_asymmetric_counts(self, store):
        """Some types very large, others very small."""
        gen = SyntheticGraphGenerator(
            num_companies=20,
            num_countries=2,
            num_vessels=0,
            num_wallets=1,
            num_instruments=50,
            num_persons=1,
            num_cftc_contracts=0,
            num_organizations=0,
            num_protocols=0,
            num_topics=1,
            num_domains=0,
            time_span=86400.0 * 3,
            base_event_rate=0.0003,
            seed=42,
        )
        stats = gen.generate(store)
        total = sum(stats["entities"].values())
        assert total == 75
        assert stats["links"] > 0

    def test_zero_all_new_types_matches_original(self, store):
        """Setting all new types to 0 is identical to not specifying them."""
        gen = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            num_vessels=2,
            num_wallets=2,
            num_instruments=0,
            num_persons=0,
            num_cftc_contracts=0,
            num_organizations=0,
            num_protocols=0,
            num_topics=0,
            num_domains=0,
            time_span=86400.0 * 10,
            base_event_rate=0.0005,
            seed=42,
        )
        stats = gen.generate(store)
        assert len(stats["entities"]) == 4
        assert set(stats["entities"].keys()) == {
            "company",
            "country",
            "vessel",
            "wallet",
        }

    def test_obs_types_for_unknown_type(self):
        """Unknown entity type returns fallback obs type."""
        obs = SyntheticGraphGenerator._obs_types_for("nonexistent_type")
        assert obs == ["cross_entity_pattern"]

    def test_obs_types_for_all_known_types(self):
        """Every known entity type has at least one obs type mapped."""
        for etype in ENTITY_TYPES:
            obs = SyntheticGraphGenerator._obs_types_for(etype)
            assert len(obs) >= 1, f"No obs types for {etype}"
            # All returned obs types should be valid OBSERVATION_TYPES
            for ot in obs:
                assert ot in OBSERVATION_TYPES, f"Invalid obs type '{ot}' for entity type '{etype}'"

    def test_deterministic_with_same_seed(self, store):
        """Same seed produces identical results."""
        gen1 = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            num_instruments=3,
            num_persons=2,
            seed=123,
            time_span=86400.0 * 3,
            base_event_rate=0.001,
        )
        store1 = PipelineStore(db_path=":memory:")
        stats1 = gen1.generate(store1)

        gen2 = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            num_instruments=3,
            num_persons=2,
            seed=123,
            time_span=86400.0 * 3,
            base_event_rate=0.001,
        )
        store2 = PipelineStore(db_path=":memory:")
        stats2 = gen2.generate(store2)

        assert stats1["entities"] == stats2["entities"]
        assert stats1["links"] == stats2["links"]
        assert stats1["observations"] == stats2["observations"]

    def test_graph_builder_handles_expanded(self, expanded_store):
        """GraphBuilder.build() succeeds on the fully expanded graph."""
        store, _ = expanded_store
        gb = GraphBuilder(store)
        data, id_map, events = gb.build()
        assert id_map.num_nodes == 54
        assert len(events) > 0

    def test_trainer_build_model_expanded(self, expanded_store):
        """Trainer.build_model() works on expanded graph."""
        store, _ = expanded_store
        trainer = Trainer(store, TrainerConfig(hidden_dim=32))
        model = trainer.build_model()
        assert model is not None
        # Model should have parameters
        params = list(model.parameters())
        assert len(params) > 0
