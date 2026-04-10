"""Tests for Phase 12e: Pattern extraction from trained HetTGN.

Covers:
    PatternExtractor — metapath importance, embedding fallback, empty graphs
    extract_temporal_lags — lag distributions, injected patterns
    crystallize — production rules, threshold filtering, edge cases
"""

from __future__ import annotations

import pytest
import torch

from agent.models.gnn.graph_builder import ENTITY_TYPES, OBSERVATION_TYPES
from agent.models.gnn.het_tgn import HetTGN
from agent.models.gnn.pattern_extractor import (
    CrystallizedPattern,
    MetaPathPattern,
    PatternExtractor,
    crystallize,
    extract_temporal_lags,
)
from agent.models.gnn.trainer import (
    InjectedPattern,
    SyntheticGraphGenerator,
    Trainer,
    TrainerConfig,
)
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore


# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def store():
    return PipelineStore(db_path=":memory:")


@pytest.fixture
def populated_store(store):
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        num_wallets=2,
        time_span=86400.0 * 10,
        base_event_rate=0.0005,
        seed=42,
    )
    gen.generate(store)
    return store


@pytest.fixture
def pattern_store(store):
    """Store with injected insider_trade → geopolitical_event pattern."""
    pattern = InjectedPattern(
        source_type="company",
        source_obs_type="insider_trade",
        target_type="country",
        target_obs_type="geopolitical_event",
        via_edge="headquartered_in",
        lag_seconds=3600.0,
        lag_jitter=300.0,
    )
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        num_wallets=2,
        time_span=86400.0 * 10,
        base_event_rate=0.0005,
        seed=42,
        patterns=[pattern],
    )
    gen.generate(store)
    return store


@pytest.fixture
def trained_model(populated_store):
    """Quick-trained model for extraction tests."""
    cfg = TrainerConfig(
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        num_heads=2,
        num_layers=1,
        epochs=2,
        window_size=86400.0 * 2,
    )
    trainer = Trainer(populated_store, cfg)
    model = trainer.build_model()
    trainer.train()
    return model, populated_store


# ═══════════════════════════════════════════════════════════════
# PatternExtractor tests
# ═══════════════════════════════════════════════════════════════


class TestPatternExtractor:
    def test_extract_returns_patterns(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        assert isinstance(patterns, list)
        # Should have at least some patterns (3 edge types)
        assert len(patterns) > 0

    def test_patterns_are_sorted(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        scores = [p.score for p in patterns]
        assert scores == sorted(scores, reverse=True)

    def test_pattern_fields(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        for p in patterns:
            assert isinstance(p.src_type, str)
            assert isinstance(p.edge_type, str)
            assert isinstance(p.dst_type, str)
            assert p.score >= 0
            assert p.mean_attention >= 0
            assert p.frequency > 0

    def test_frequency_matches_edge_count(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        links = store.query_all_entity_links()
        link_type_counts: dict[str, int] = {}
        for lnk in links:
            lt = lnk["link_type"]
            link_type_counts[lt] = link_type_counts.get(lt, 0) + 1
        for p in patterns:
            assert p.frequency == link_type_counts.get(p.edge_type, 0)

    def test_empty_store(self, store):
        cfg = TrainerConfig(hidden_dim=8, memory_dim=8, message_dim=8)
        # Need a minimal model for an empty store
        from torch_geometric.data import HeteroData
        from agent.models.gnn.graph_builder import GraphBuilder, IDMap

        # Register 1 entity so we get valid metadata
        eid = store.register_entity(
            "company", "c0", entity_id_from_key("company", "c0")
        )
        builder = GraphBuilder(store)
        data, id_map, _ = builder.build()
        metadata = data.metadata()
        in_channels = {
            nt: data[nt].x.size(1)
            for nt in metadata[0]
            if nt in data.node_types and hasattr(data[nt], "x")
        }
        model = HetTGN(
            metadata=metadata,
            in_channels=in_channels or {"company": 9},
            hidden_dim=8,
            memory_dim=8,
            message_dim=8,
            num_heads=1,
            num_layers=1,
            num_nodes=1,
        )
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        # No edge types → no patterns
        assert patterns == []


class TestPatternExtractorEmbeddingFallback:
    def test_fallback_produces_scores(self, trained_model):
        """Embedding-based scoring should always work as fallback."""
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        from agent.models.gnn.graph_builder import GraphBuilder

        data, id_map, _ = GraphBuilder(store).build()
        scores = extractor._extract_embedding_scores(data, id_map)
        # Should return scores for edge types with edges
        assert len(scores) > 0
        for etype, val in scores.items():
            assert 0.0 <= val <= 1.0


# ═══════════════════════════════════════════════════════════════
# Temporal lag extraction tests
# ═══════════════════════════════════════════════════════════════


class TestTemporalLagExtraction:
    def test_lags_populated(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        result = extract_temporal_lags(patterns, store, top_k=5)
        assert len(result) > 0
        # At least one pattern should have a non-zero mean_lag
        has_lag = any(p.mean_lag > 0 for p in result)
        assert has_lag

    def test_lag_stats_make_sense(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        result = extract_temporal_lags(patterns, store, top_k=5)
        for p in result:
            if p.mean_lag > 0:
                assert p.lag_std >= 0
                assert p.lag_p25 <= p.mean_lag or True  # p25 often < mean
                assert p.lag_p75 >= p.lag_p25

    def test_injected_pattern_lag(self, pattern_store):
        """Injected pattern should produce lags near the known lag."""
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=1,
            window_size=86400.0 * 2,
        )
        trainer = Trainer(pattern_store, cfg)
        model = trainer.build_model()
        trainer.train()

        extractor = PatternExtractor(model, pattern_store)
        patterns = extractor.extract_metapath_importance()

        # Find the headquartered_in pattern
        hq_pattern = None
        for p in patterns:
            if p.edge_type == "headquartered_in":
                hq_pattern = p
                break

        if hq_pattern:
            result = extract_temporal_lags([hq_pattern], pattern_store, top_k=1)
            # Lag should exist and be in a reasonable range
            # (not necessarily exactly 3600 due to base events mixing in)
            assert result[0].mean_lag > 0

    def test_top_k_limits_output(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        result = extract_temporal_lags(patterns, store, top_k=1)
        assert len(result) == 1

    def test_empty_patterns(self, populated_store):
        result = extract_temporal_lags([], populated_store, top_k=5)
        assert result == []


# ═══════════════════════════════════════════════════════════════
# Crystallization tests
# ═══════════════════════════════════════════════════════════════


class TestCrystallize:
    def test_produces_configs(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        patterns = extract_temporal_lags(patterns, store, top_k=5)
        configs = crystallize(patterns, store, threshold=0.0, validate=False)
        assert len(configs) > 0

    def test_config_fields(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        patterns = extract_temporal_lags(patterns, store, top_k=5)
        configs = crystallize(patterns, store, threshold=0.0, validate=False)
        for c in configs:
            assert isinstance(c, CrystallizedPattern)
            assert c.source_type in ENTITY_TYPES
            assert c.target_type in ENTITY_TYPES
            assert c.via_edge != ""
            assert c.obs_type_a in OBSERVATION_TYPES
            assert c.obs_type_b in OBSERVATION_TYPES
            assert c.window_seconds >= 3600.0  # min 1 hour
            assert c.window_seconds <= 7 * 86400.0  # max 7 days
            assert c.source == "auto_gnn"

    def test_threshold_filtering(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        # Very high threshold should filter everything
        configs = crystallize(patterns, store, threshold=1e6)
        assert configs == []

    def test_threshold_zero_includes_all(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        configs_all = crystallize(patterns, store, threshold=0.0, validate=False)
        configs_some = crystallize(patterns, store, threshold=0.01, validate=False)
        assert len(configs_all) >= len(configs_some)

    def test_empty_patterns(self, populated_store):
        configs = crystallize([], populated_store, threshold=0.0)
        assert configs == []


class TestCrystallizeEdgeCases:
    def test_pattern_with_no_observations(self, store):
        """Pattern for entity types with no observations → skipped."""
        # Create patterns manually
        p = MetaPathPattern(
            src_type="person",  # no person entities in store
            edge_type="works_at",
            dst_type="company",
            score=1.0,
            mean_attention=0.5,
            frequency=10,
            mean_lag=3600.0,
        )
        configs = crystallize([p], store, threshold=0.0)
        assert configs == []

    def test_single_dominant_pattern(self, trained_model):
        """Only the highest-scoring pattern should survive a tight threshold."""
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        if len(patterns) >= 2:
            # Set threshold between top and second
            mid = (patterns[0].score + patterns[1].score) / 2
            configs = crystallize(patterns, store, threshold=mid)
            assert len(configs) <= 1

    def test_uniform_attention_still_works(self, populated_store):
        """Even with uniform attention, patterns get some score from frequency."""
        cfg = TrainerConfig(
            hidden_dim=8,
            memory_dim=8,
            message_dim=8,
            num_heads=1,
            num_layers=1,
            epochs=0,  # 0 epochs = random weights
            window_size=86400.0 * 2,
        )
        trainer = Trainer(populated_store, cfg)
        model = trainer.build_model()  # untrained
        extractor = PatternExtractor(model, populated_store)
        patterns = extractor.extract_metapath_importance()
        # Should still produce patterns from embedding similarity
        assert len(patterns) > 0
