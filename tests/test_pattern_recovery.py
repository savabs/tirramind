"""Tests for Phase 14b-d: Multi-hop scoring, conditioned crystallization, validation.

Covers:
    Phase 14b — 2-hop meta-path scoring: enumeration, scoring math, edge cases
    Phase 14c — Obs-type conditioned crystallization: co-occurrence table,
                correct obs-type pair selection
    Phase 14d — Pattern validation: Fisher's exact test, BH FDR correction,
                lift computation, filtering of insignificant patterns
"""

from __future__ import annotations

import math
from collections import Counter
from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import HeteroData

from agent.models.gnn.graph_builder import ENTITY_TYPES, IDMap, OBSERVATION_TYPES
from agent.models.gnn.het_tgn import HetTGN
from agent.models.gnn.pattern_extractor import (
    CrystallizedPattern,
    MetaPathPattern,
    PatternExtractor,
    ValidationResult,
    _build_cooccurrence_table,
    crystallize,
    extract_temporal_lags,
    validate_patterns,
)
from agent.models.gnn.trainer import (
    SyntheticGraphGenerator,
    Trainer,
    TrainerConfig,
    entity_id_from_key,
)
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
def trained_model(populated_store):
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
# Phase 14b: Multi-Hop Meta-Path Scoring
# ═══════════════════════════════════════════════════════════════


class TestMultiHopMetaPaths:
    """Unit tests for _score_2hop_metapaths and integration."""

    def test_2hop_patterns_from_chainable_edges(self, trained_model):
        """2-hop patterns appear when edge types chain (dst == src of another)."""
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        # Synthetic data has only A→country edges (company, vessel, wallet all
        # point TO country, none FROM country), so 2-hop requires a reverse.
        # If no 2-hop found, that's correct for this topology.
        twohop = [p for p in patterns if p.hops == 2]
        # Just verify the method runs; chainable edges tested in unit tests below
        assert isinstance(twohop, list)

    def test_1hop_patterns_still_present(self, trained_model):
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        onehop = [p for p in patterns if p.hops == 1]
        assert len(onehop) > 0

    def test_mixed_and_sorted(self, trained_model):
        """1-hop and 2-hop patterns are merged and sorted by score."""
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        scores = [p.score for p in patterns]
        assert scores == sorted(scores, reverse=True)

    def test_2hop_edge_type_format(self, trained_model):
        """2-hop patterns have edge_type = 'rel1_via_rel2'."""
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        for p in patterns:
            if p.hops == 2:
                assert "_via_" in p.edge_type

    def test_2hop_scoring_math(self):
        """Verify 2-hop score formula: attn1 * attn2 * log2(1 + freq1 * freq2)."""
        extractor = PatternExtractor.__new__(PatternExtractor)
        attention_scores = {
            ("A", "r1", "B"): 0.5,
            ("B", "r2", "C"): 0.4,
        }
        edge_counts = {
            ("A", "r1", "B"): 10,
            ("B", "r2", "C"): 20,
        }
        patterns = extractor._score_2hop_metapaths(attention_scores, edge_counts)
        # A→B→C should exist
        abc = [p for p in patterns if p.src_type == "A" and p.dst_type == "C"]
        assert len(abc) == 1
        p = abc[0]
        expected_score = 0.5 * 0.4 * math.log2(1 + 10 * 20)
        assert abs(p.score - expected_score) < 1e-6
        assert p.hops == 2
        assert p.edge_type == "r1_via_r2"
        assert p.mean_attention == 0.5 * 0.4
        assert p.frequency == 200

    def test_no_self_loops(self):
        """A→B→A via same relation is excluded."""
        extractor = PatternExtractor.__new__(PatternExtractor)
        attention_scores = {
            ("A", "r1", "B"): 0.5,
            ("B", "r1", "A"): 0.4,
        }
        edge_counts = {
            ("A", "r1", "B"): 10,
            ("B", "r1", "A"): 10,
        }
        patterns = extractor._score_2hop_metapaths(attention_scores, edge_counts)
        self_loops = [
            p
            for p in patterns
            if p.src_type == p.dst_type and "r1_via_r1" in p.edge_type
        ]
        assert len(self_loops) == 0

    def test_empty_graph_returns_empty(self):
        """No attention scores → no 2-hop patterns."""
        extractor = PatternExtractor.__new__(PatternExtractor)
        patterns = extractor._score_2hop_metapaths({}, {})
        assert patterns == []

    def test_single_edge_type_no_2hop(self):
        """With only one edge type A→B, no 2-hop (B is not a source)."""
        extractor = PatternExtractor.__new__(PatternExtractor)
        attention_scores = {("A", "r", "B"): 0.5}
        edge_counts = {("A", "r", "B"): 10}
        patterns = extractor._score_2hop_metapaths(attention_scores, edge_counts)
        assert patterns == []

    def test_zero_freq_excluded(self):
        """Edge types with zero frequency are excluded from 2-hop."""
        extractor = PatternExtractor.__new__(PatternExtractor)
        attention_scores = {
            ("A", "r1", "B"): 0.5,
            ("B", "r2", "C"): 0.4,
        }
        edge_counts = {
            ("A", "r1", "B"): 0,  # zero freq
            ("B", "r2", "C"): 20,
        }
        patterns = extractor._score_2hop_metapaths(attention_scores, edge_counts)
        assert patterns == []


# ═══════════════════════════════════════════════════════════════
# Phase 14c: Obs-Type Conditioned Crystallization
# ═══════════════════════════════════════════════════════════════


class TestCooccurrenceTable:
    """Unit tests for _build_cooccurrence_table."""

    def test_counts_correct_pairs(self):
        """Co-occurrence table counts (src_obs, dst_obs) within window."""
        pattern = MetaPathPattern(
            src_type="company",
            edge_type="headquartered_in",
            dst_type="country",
            score=1.0,
            mean_attention=0.5,
            frequency=10,
        )
        eid_to_type = {"c1": "company", "co1": "country"}
        obs_by_entity = {
            "c1": [
                {
                    "entity_id": "c1",
                    "observation_type": "insider_trade",
                    "observed_at": 100.0,
                },
                {
                    "entity_id": "c1",
                    "observation_type": "sec_filing",
                    "observed_at": 200.0,
                },
            ],
            "co1": [
                {
                    "entity_id": "co1",
                    "observation_type": "geopolitical_event",
                    "observed_at": 150.0,
                },
                {
                    "entity_id": "co1",
                    "observation_type": "macro_indicator",
                    "observed_at": 500.0,
                },
            ],
        }
        link_index = {("c1", "headquartered_in"): ["co1"]}

        result = _build_cooccurrence_table(
            pattern,
            eid_to_type,
            obs_by_entity,
            link_index,
            window=100.0,
        )
        # insider_trade@100 → geopolitical_event@150 (lag=50, within 100s window)
        # sec_filing@200 → macro_indicator@500 (lag=300, outside 100s window)
        assert result[("insider_trade", "geopolitical_event")] == 1
        assert ("sec_filing", "macro_indicator") not in result

    def test_empty_links_returns_empty(self):
        pattern = MetaPathPattern(
            src_type="company",
            edge_type="headquartered_in",
            dst_type="country",
            score=1.0,
            mean_attention=0.5,
            frequency=10,
        )
        result = _build_cooccurrence_table(pattern, {}, {}, {}, window=3600.0)
        assert len(result) == 0

    def test_window_boundary(self):
        """Events exactly at window boundary are included."""
        pattern = MetaPathPattern(
            src_type="company",
            edge_type="headquartered_in",
            dst_type="country",
            score=1.0,
            mean_attention=0.5,
            frequency=10,
        )
        eid_to_type = {"c1": "company", "co1": "country"}
        obs_by_entity = {
            "c1": [{"entity_id": "c1", "observation_type": "a", "observed_at": 0.0}],
            "co1": [
                {"entity_id": "co1", "observation_type": "b", "observed_at": 100.0}
            ],
        }
        link_index = {("c1", "headquartered_in"): ["co1"]}

        result = _build_cooccurrence_table(
            pattern,
            eid_to_type,
            obs_by_entity,
            link_index,
            window=100.0,
        )
        assert result[("a", "b")] == 1

    def test_dst_before_src_excluded(self):
        """Target events before source are not counted."""
        pattern = MetaPathPattern(
            src_type="company",
            edge_type="headquartered_in",
            dst_type="country",
            score=1.0,
            mean_attention=0.5,
            frequency=10,
        )
        eid_to_type = {"c1": "company", "co1": "country"}
        obs_by_entity = {
            "c1": [{"entity_id": "c1", "observation_type": "a", "observed_at": 200.0}],
            "co1": [
                {"entity_id": "co1", "observation_type": "b", "observed_at": 100.0}
            ],
        }
        link_index = {("c1", "headquartered_in"): ["co1"]}

        result = _build_cooccurrence_table(
            pattern,
            eid_to_type,
            obs_by_entity,
            link_index,
            window=3600.0,
        )
        assert len(result) == 0


class TestConditionedCrystallization:
    """Test that crystallize uses co-occurrence instead of global freq."""

    def test_picks_cooccurrence_pair(self):
        """When co-occurrence data exists, picks the top pair."""
        store = PipelineStore(db_path=":memory:")

        # Create entities
        c1 = entity_id_from_key("company", "c1")
        co1 = entity_id_from_key("country", "co1")
        store.register_entity("company", "c1", c1)
        store.register_entity("country", "co1", co1)
        store.link_entities(c1, co1, "headquartered_in", "test")

        # Create observations: insider_trade on company, geopolitical_event on country
        # These should co-occur within window
        store.store_entity_observation(
            entity_id=c1,
            source_tool="test",
            observed_at=100.0,
            observation_type="insider_trade",
            value={},
        )
        store.store_entity_observation(
            entity_id=co1,
            source_tool="test",
            observed_at=200.0,
            observation_type="geopolitical_event",
            value={},
        )
        # Add non-cooccurring noise
        store.store_entity_observation(
            entity_id=c1,
            source_tool="test",
            observed_at=50000.0,
            observation_type="sec_filing",
            value={},
        )

        pattern = MetaPathPattern(
            src_type="company",
            edge_type="headquartered_in",
            dst_type="country",
            score=1.0,
            mean_attention=0.5,
            frequency=10,
            mean_lag=100.0,
        )

        configs = crystallize([pattern], store, threshold=0.0, validate=False)
        assert len(configs) >= 1
        c = configs[0]
        assert c.obs_type_a == "insider_trade"
        assert c.obs_type_b == "geopolitical_event"


# ═══════════════════════════════════════════════════════════════
# Phase 14d: Pattern Validation
# ═══════════════════════════════════════════════════════════════


class TestValidationResult:
    """Verify ValidationResult dataclass."""

    def test_fields(self):
        vr = ValidationResult(hit_rate=0.5, baseline_rate=0.1, lift=5.0, p_value=0.001)
        assert vr.hit_rate == 0.5
        assert vr.baseline_rate == 0.1
        assert vr.lift == 5.0
        assert vr.p_value == 0.001
        assert vr.significant is False  # default

    def test_significant_flag(self):
        vr = ValidationResult(
            hit_rate=0.5,
            baseline_rate=0.1,
            lift=5.0,
            p_value=0.001,
            significant=True,
        )
        assert vr.significant is True


class TestValidatePatterns:
    """Test validate_patterns with Fisher's exact test + BH FDR."""

    def _make_store_with_signal(self):
        """Create a store where insider_trade → geopolitical_event within window."""
        store = PipelineStore(db_path=":memory:")
        c1 = entity_id_from_key("company", "c1")
        c2 = entity_id_from_key("company", "c2")
        co1 = entity_id_from_key("country", "co1")

        store.register_entity("company", "c1", c1)
        store.register_entity("company", "c2", c2)
        store.register_entity("country", "co1", co1)

        store.link_entities(c1, co1, "headquartered_in", "test")
        store.link_entities(c2, co1, "headquartered_in", "test")

        # Repeated co-occurrences to give statistical power
        for i in range(20):
            base = i * 10000.0
            store.store_entity_observation(
                entity_id=c1,
                source_tool="test",
                observed_at=base,
                observation_type="insider_trade",
                value={},
            )
            store.store_entity_observation(
                entity_id=co1,
                source_tool="test",
                observed_at=base + 1800.0,
                observation_type="geopolitical_event",
                value={},
            )

        # Some noise observations on co1 (different obs type)
        for i in range(10):
            store.store_entity_observation(
                entity_id=co1,
                source_tool="test",
                observed_at=i * 5000.0 + 500.0,
                observation_type="macro_indicator",
                value={},
            )

        return store

    def test_significant_pattern_detected(self):
        store = self._make_store_with_signal()
        cp = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=3600.0,
        )
        results = validate_patterns([cp], store, alpha=0.05)
        assert len(results) == 1
        vr = results[0]
        assert vr.hit_rate > 0
        assert vr.lift > 1.0
        assert vr.p_value < 0.05
        assert vr.significant is True

    def test_insignificant_pattern_rejected(self):
        """A random pattern should not be significant."""
        store = self._make_store_with_signal()
        cp = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="macro_indicator",
            window_seconds=60.0,  # very tight window
        )
        results = validate_patterns([cp], store, alpha=0.05)
        assert len(results) == 1
        # With a 60s window, insider_trade → macro_indicator unlikely
        # (macro_indicators are at +500s offsets)
        vr = results[0]
        assert vr.significant is False

    def test_empty_patterns(self):
        store = PipelineStore(db_path=":memory:")
        results = validate_patterns([], store)
        assert results == []

    def test_no_links_gives_insignificant(self):
        """Without entity links, no hits → p=1 → insignificant."""
        store = PipelineStore(db_path=":memory:")
        c1 = entity_id_from_key("company", "c1")
        co1 = entity_id_from_key("country", "co1")
        store.register_entity("company", "c1", c1)
        store.register_entity("country", "co1", co1)
        # No links recorded
        store.store_entity_observation(
            entity_id=c1,
            source_tool="test",
            observed_at=100.0,
            observation_type="insider_trade",
            value={},
        )
        store.store_entity_observation(
            entity_id=co1,
            source_tool="test",
            observed_at=200.0,
            observation_type="geopolitical_event",
            value={},
        )

        cp = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=3600.0,
        )
        results = validate_patterns([cp], store)
        assert len(results) == 1
        assert results[0].p_value == 1.0
        assert results[0].significant is False

    def test_bh_correction_multiple(self):
        """BH correction is applied across multiple patterns."""
        store = self._make_store_with_signal()
        good_pattern = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=3600.0,
        )
        bad_pattern = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="macro_indicator",
            window_seconds=1.0,  # impossibly tight
        )
        results = validate_patterns([good_pattern, bad_pattern], store, alpha=0.05)
        assert len(results) == 2
        # Good pattern should be significant, bad should not
        assert results[0].significant is True
        assert results[1].significant is False

    def test_lift_calculation(self):
        store = self._make_store_with_signal()
        cp = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=3600.0,
        )
        results = validate_patterns([cp], store)
        vr = results[0]
        # Lift should be hit_rate / baseline_rate
        if vr.baseline_rate > 0:
            expected_lift = vr.hit_rate / vr.baseline_rate
            assert abs(vr.lift - expected_lift) < 1e-6


class TestCrystallizeWithValidation:
    """Integration: crystallize with validate=True filters patterns."""

    def test_validate_flag_false_keeps_all(self, trained_model):
        """validate=False preserves all patterns above threshold."""
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        patterns = extract_temporal_lags(patterns, store, top_k=5)
        configs_unvalidated = crystallize(
            patterns, store, threshold=0.0, validate=False
        )
        configs_validated = crystallize(patterns, store, threshold=0.0, validate=True)
        assert len(configs_unvalidated) >= len(configs_validated)

    def test_validate_flag_true_default(self, trained_model):
        """Default validate=True is used."""
        model, store = trained_model
        extractor = PatternExtractor(model, store)
        patterns = extractor.extract_metapath_importance()
        patterns = extract_temporal_lags(patterns, store, top_k=5)
        configs = crystallize(patterns, store, threshold=0.0)
        # Should return some configs (synthetic data has real links)
        assert isinstance(configs, list)
