"""Tests for StateAssembler — Phase 21b.1

Mathematical proofs:
    1. Fixed dimensionality:  ∀ input sizes, output.shape == (state_dim,)
    2. Top-K ordering:        entities sorted by composite_surprise descending
    3. Tradeable filtering:   non-tradeable entities are excluded
    4. Belief matching:       belief → entity aligned correctly
    5. Zero-padding:          empty slots are exactly 0
    6. Normalised count:      last element = n_active / max_entities ∈ [0, 1]
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from agent.fusion.alert import EntityAlert
from agent.learning.policy.state_assembler import StateAssembler
from agent.models.belief import BeliefState


# ── Helpers ───────────────────────────────────────────────────


def _make_alert(
    entity_id: str,
    composite: float = 1.0,
    obs: float = 0.1,
    temporal: float = 0.2,
    value: float = 0.3,
    neighborhood: float = 0.4,
    drift: float = 0.5,
) -> EntityAlert:
    return EntityAlert(
        entity_id=entity_id,
        entity_type="company",
        entity_name=f"Name-{entity_id}",
        alert_time=time.time(),
        obs_type_surprise=obs,
        temporal_surprise=temporal,
        value_surprise=value,
        neighborhood_surprise=neighborhood,
        memory_drift=drift,
        cusum_statistic=0.0,
        hawkes_intensity=0.0,
        event_study_score=0.0,
        composite_surprise=composite,
        observation_count=1,
        evidence_sources=("test",),
    )


def _make_belief(
    entity_id: str,
    mean: float = 0.5,
    variance: float = 0.1,
    confidence: float = 0.8,
    stale: bool = False,
) -> BeliefState:
    now = time.time()
    return BeliefState(
        variable_name="regime.test",
        version=1,
        effective_at=now,
        computed_at=now,
        dist_type="gaussian",
        mean=mean,
        variance=variance,
        confidence=confidence,
        stale=stale,
        entity_id=entity_id,
    )


# ── Tests ─────────────────────────────────────────────────────


class TestStateDimProperty:
    """Proof 1: state_dim is deterministic from config."""

    def test_default_config(self):
        sa = StateAssembler()
        # 50*5 + 50*4 + 8 + 1 + 4 = 250 + 200 + 8 + 1 + 4 = 463
        assert sa.state_dim == 463

    def test_custom_config(self):
        sa = StateAssembler(max_entities=10, surprise_dim=3, belief_dim=2, market_dim=4)
        # 10*3 + 10*2 + 4 + 1 + 4 = 30 + 20 + 4 + 1 + 4 = 59
        assert sa.state_dim == 59


class TestFixedDimensionality:
    """Proof 1: ∀ input sizes, output.shape == (state_dim,)."""

    def test_empty_inputs(self):
        sa = StateAssembler(max_entities=5)
        state, meta = sa.assemble([], [], {}, {})
        assert state.shape == (sa.state_dim,)
        assert state.dtype == torch.float32

    def test_fewer_than_max(self):
        sa = StateAssembler(max_entities=5)
        alerts = [_make_alert("e1", composite=2.0)]
        asset_map = {"e1": "AAPL"}
        state, meta = sa.assemble(alerts, [], {}, asset_map)
        assert state.shape == (sa.state_dim,)

    def test_exactly_max(self):
        sa = StateAssembler(max_entities=3)
        alerts = [_make_alert(f"e{i}", composite=float(i)) for i in range(3)]
        asset_map = {f"e{i}": f"T{i}" for i in range(3)}
        state, meta = sa.assemble(alerts, [], {}, asset_map)
        assert state.shape == (sa.state_dim,)

    def test_more_than_max(self):
        sa = StateAssembler(max_entities=3)
        alerts = [_make_alert(f"e{i}", composite=float(i)) for i in range(10)]
        asset_map = {f"e{i}": f"T{i}" for i in range(10)}
        state, meta = sa.assemble(alerts, [], {}, asset_map)
        assert state.shape == (sa.state_dim,)


class TestTopKOrdering:
    """Proof 2: entities are sorted by composite_surprise descending."""

    def test_descending_order(self):
        sa = StateAssembler(max_entities=3)
        alerts = [
            _make_alert("low", composite=1.0),
            _make_alert("high", composite=5.0),
            _make_alert("mid", composite=3.0),
        ]
        asset_map = {"low": "T1", "high": "T2", "mid": "T3"}
        _, meta = sa.assemble(alerts, [], {}, asset_map)
        assert meta["entity_order"] == ["high", "mid", "low"]

    def test_truncation_keeps_largest(self):
        sa = StateAssembler(max_entities=2)
        alerts = [
            _make_alert("a", composite=1.0),
            _make_alert("b", composite=5.0),
            _make_alert("c", composite=3.0),
        ]
        asset_map = {"a": "T1", "b": "T2", "c": "T3"}
        _, meta = sa.assemble(alerts, [], {}, asset_map)
        assert meta["entity_order"] == ["b", "c"]
        assert len(meta["entity_order"]) == 2


class TestTradeableFiltering:
    """Proof 3: only entities in asset_map appear in output."""

    def test_non_tradeable_excluded(self):
        sa = StateAssembler(max_entities=5)
        alerts = [
            _make_alert("tradeable", composite=3.0),
            _make_alert("not_tradeable", composite=5.0),
        ]
        asset_map = {"tradeable": "AAPL"}
        _, meta = sa.assemble(alerts, [], {}, asset_map)
        assert meta["entity_order"] == ["tradeable"]
        assert "not_tradeable" not in meta["entity_order"]

    def test_empty_asset_map(self):
        sa = StateAssembler(max_entities=5)
        alerts = [_make_alert("e1", composite=3.0)]
        _, meta = sa.assemble(alerts, [], {}, {})
        assert meta["entity_order"] == []
        assert meta["n_active"] == 0


class TestBeliefMatching:
    """Proof 4: belief features align with the correct entity."""

    def test_belief_correctly_placed(self):
        sa = StateAssembler(max_entities=2, surprise_dim=5, belief_dim=4, market_dim=0)
        alerts = [
            _make_alert("e1", composite=2.0),
            _make_alert("e2", composite=5.0),
        ]
        beliefs = [
            _make_belief("e2", mean=0.7, variance=0.3, confidence=0.9, stale=False),
            _make_belief("e1", mean=0.1, variance=0.2, confidence=0.5, stale=True),
        ]
        asset_map = {"e1": "T1", "e2": "T2"}
        state, meta = sa.assemble(alerts, beliefs, {}, asset_map)

        # e2 is first (higher composite), e1 second
        assert meta["entity_order"] == ["e2", "e1"]

        # Belief block starts after surprise block: 2*5 = 10
        belief_start = 2 * 5
        # e2 belief at index 0
        np.testing.assert_allclose(
            state[belief_start : belief_start + 4].numpy(),
            [0.7, 0.3, 0.9, 0.0],
            rtol=1e-5,
        )
        # e1 belief at index 1
        np.testing.assert_allclose(
            state[belief_start + 4 : belief_start + 8].numpy(),
            [0.1, 0.2, 0.5, 1.0],  # stale=True → 1.0
            rtol=1e-5,
        )

    def test_missing_belief_zero_fills(self):
        sa = StateAssembler(max_entities=2, surprise_dim=5, belief_dim=4, market_dim=0)
        alerts = [_make_alert("e1", composite=2.0)]
        asset_map = {"e1": "T1"}
        state, _ = sa.assemble(alerts, [], {}, asset_map)

        belief_start = 2 * 5
        # All belief values should be 0 for e1 (no belief provided)
        np.testing.assert_array_equal(
            state[belief_start : belief_start + 4].numpy(), [0, 0, 0, 0]
        )


class TestZeroPadding:
    """Proof 5: unused entity slots are exactly zero."""

    def test_padding_is_zero(self):
        sa = StateAssembler(max_entities=5, surprise_dim=5, belief_dim=4, market_dim=0)
        alerts = [_make_alert("e1", composite=1.0, obs=9.9)]
        asset_map = {"e1": "T1"}
        state, _ = sa.assemble(alerts, [], {}, asset_map)

        # Slots 1-4 of surprise block should be zero
        surprise_block = state[5:25].numpy()  # 4 unused entities × 5 dims
        np.testing.assert_array_equal(surprise_block, 0.0)

        # Slots 1-4 of belief block should be zero
        belief_block = state[29:45].numpy()  # skip e1 belief (25:29), rest zeros
        np.testing.assert_array_equal(belief_block, 0.0)


class TestNormalisedCount:
    """Proof 6: entity_count = n_active / max_entities (at index -5, before adversarial block)."""

    def test_zero_entities(self):
        sa = StateAssembler(max_entities=5, market_dim=0)
        state, _ = sa.assemble([], [], {}, {})
        # Entity count is 5th from last (4 adversarial features follow)
        assert state[-5].item() == pytest.approx(0.0)

    def test_partial_fill(self):
        sa = StateAssembler(max_entities=4, market_dim=0)
        alerts = [_make_alert(f"e{i}", composite=float(i)) for i in range(2)]
        asset_map = {f"e{i}": f"T{i}" for i in range(2)}
        state, _ = sa.assemble(alerts, [], {}, asset_map)
        assert state[-5].item() == pytest.approx(0.5)

    def test_full_fill(self):
        sa = StateAssembler(max_entities=3, market_dim=0)
        alerts = [_make_alert(f"e{i}", composite=float(i)) for i in range(3)]
        asset_map = {f"e{i}": f"T{i}" for i in range(3)}
        state, _ = sa.assemble(alerts, [], {}, asset_map)
        assert state[-5].item() == pytest.approx(1.0)


class TestMarketFeatures:
    """Market feature block is correctly placed and sorted."""

    def test_market_features_sorted(self):
        sa = StateAssembler(max_entities=2, surprise_dim=5, belief_dim=4, market_dim=3)
        mf = {"c_vol": 1.0, "a_return": 2.0, "b_regime": 3.0}
        state, _ = sa.assemble([], [], mf, {})

        # Market block starts after surprise(2*5=10) + belief(2*4=8) = 18
        market_start = 18
        np.testing.assert_allclose(
            state[market_start : market_start + 3].numpy(),
            [2.0, 3.0, 1.0],  # sorted: a_return, b_regime, c_vol
            rtol=1e-5,
        )

    def test_excess_market_features_truncated(self):
        sa = StateAssembler(max_entities=1, surprise_dim=1, belief_dim=1, market_dim=2)
        mf = {"a": 1.0, "b": 2.0, "c": 3.0}
        state, _ = sa.assemble([], [], mf, {})
        # Only first 2 sorted keys used: a, b
        market_start = 1 + 1
        np.testing.assert_allclose(
            state[market_start : market_start + 2].numpy(), [1.0, 2.0], rtol=1e-5
        )


class TestSurpriseValues:
    """Surprise signal values are correctly placed in the state tensor."""

    def test_surprise_block_values(self):
        sa = StateAssembler(max_entities=2, surprise_dim=5, belief_dim=4, market_dim=0)
        alerts = [
            _make_alert(
                "e1",
                composite=3.0,
                obs=0.1,
                temporal=0.2,
                value=0.3,
                neighborhood=0.4,
                drift=0.5,
            )
        ]
        asset_map = {"e1": "T1"}
        state, _ = sa.assemble(alerts, [], {}, asset_map)
        np.testing.assert_allclose(
            state[:5].numpy(), [0.1, 0.2, 0.3, 0.4, 0.5], rtol=1e-5
        )


class TestMetadata:
    """Metadata contains correct entity and ticker ordering."""

    def test_ticker_order_matches_entity_order(self):
        sa = StateAssembler(max_entities=3)
        alerts = [
            _make_alert("e1", composite=1.0),
            _make_alert("e2", composite=3.0),
            _make_alert("e3", composite=2.0),
        ]
        asset_map = {"e1": "AAPL", "e2": "GOOG", "e3": "MSFT"}
        _, meta = sa.assemble(alerts, [], {}, asset_map)
        assert meta["entity_order"] == ["e2", "e3", "e1"]
        assert meta["ticker_order"] == ["GOOG", "MSFT", "AAPL"]
