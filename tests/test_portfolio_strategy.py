"""Tests for Portfolio Strategy Adapters — Phase 21b.5

Mathematical proofs:
    1. Strategy ABC compliance: both classes implement required interface
    2. WeightedSurprise: threshold → binary position logic
    3. WeightedSurprise: learned weights applied correctly
    4. SACPortfolio: deterministic action → conviction weights
    5. Empty/missing inputs → zero weights (safe fallback)
    6. Weight array shape: always (test_length,)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.fusion.alert import EntityAlert
from agent.learning.policy.asset_mapper import AssetMapper
from agent.learning.policy.config import SACConfig
from agent.learning.policy.portfolio_strategy import (
    SACPortfolioStrategy,
    WeightedSurpriseStrategy,
)
from agent.learning.policy.sac import SACTrainer
from agent.learning.policy.state_assembler import StateAssembler
from agent.models.belief import BeliefState
from agent.quant.backtest import Strategy


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


def _make_belief(entity_id: str) -> BeliefState:
    now = time.time()
    return BeliefState(
        variable_name="regime.test",
        version=1,
        effective_at=now,
        computed_at=now,
        dist_type="gaussian",
        mean=0.5,
        variance=0.1,
        confidence=0.8,
        entity_id=entity_id,
    )


def _mock_asset_mapper(mappings: dict[str, str]) -> AssetMapper:
    """Create a mock AssetMapper with fixed mappings."""
    mapper = MagicMock(spec=AssetMapper)
    mapper.resolve.side_effect = lambda eid: mappings.get(eid)
    mapper.tradeable_entities.return_value = dict(mappings)
    mapper.resolve_batch.side_effect = lambda ids: {
        eid: mappings[eid] for eid in ids if eid in mappings
    }
    return mapper


# ── 1. Strategy ABC Compliance ────────────────────────────────


class TestStrategyABC:
    """Proof 1: both classes are proper Strategy subclasses."""

    def test_weighted_surprise_is_strategy(self):
        mapper = _mock_asset_mapper({"e1": "AAPL"})
        s = WeightedSurpriseStrategy((0.3, 0.15, 0.25, 0.2, 0.1), mapper)
        assert isinstance(s, Strategy)
        assert s.name == "weighted_surprise"

    def test_sac_portfolio_is_strategy(self):
        cfg = SACConfig(hidden_dim=16, num_hidden=1)
        trainer = SACTrainer(20, 3, cfg)
        assembler = StateAssembler(max_entities=5)
        mapper = _mock_asset_mapper({})
        s = SACPortfolioStrategy(trainer, assembler, mapper)
        assert isinstance(s, Strategy)
        assert s.name == "sac_rl_policy"


# ── 2. WeightedSurprise Threshold Logic ──────────────────────


class TestWeightedSurpriseThreshold:
    """Proof 2: threshold → binary long position."""

    def test_above_threshold_triggers(self):
        mapper = _mock_asset_mapper({"e1": "AAPL"})
        weights = (1.0, 1.0, 1.0, 1.0, 1.0)  # all equal → composite = sum of 5 signals
        s = WeightedSurpriseStrategy(weights, mapper, threshold=1.0)

        # Alert with signals that sum to 1.5 > threshold 1.0
        alert = _make_alert(
            "e1", obs=0.3, temporal=0.3, value=0.3, neighborhood=0.3, drift=0.3
        )
        result = s.generate_weights(
            np.zeros(10), 3, test_extra={"alerts": [[alert], [], [alert]]}
        )
        assert result[0] == 1.0  # triggered
        assert result[1] == 0.0  # no alerts
        assert result[2] == 1.0  # triggered

    def test_below_threshold_no_position(self):
        mapper = _mock_asset_mapper({"e1": "AAPL"})
        weights = (0.1, 0.1, 0.1, 0.1, 0.1)
        s = WeightedSurpriseStrategy(weights, mapper, threshold=10.0)

        alert = _make_alert("e1")
        result = s.generate_weights(
            np.zeros(10), 3, test_extra={"alerts": [[alert], [alert], [alert]]}
        )
        np.testing.assert_array_equal(result, 0.0)


# ── 3. WeightedSurprise Weights Application ──────────────────


class TestWeightedSurpriseWeights:
    """Proof 3: learned weights correctly applied as dot product."""

    def test_dot_product_computation(self):
        mapper = _mock_asset_mapper({"e1": "AAPL"})
        weights = (2.0, 0.0, 0.0, 0.0, 0.0)  # only obs_type matters
        s = WeightedSurpriseStrategy(weights, mapper, threshold=0.5)

        # obs=0.3 → composite = 2.0 * 0.3 = 0.6 > 0.5 → triggered
        alert_above = _make_alert("e1", obs=0.3)
        # obs=0.2 → composite = 2.0 * 0.2 = 0.4 < 0.5 → not triggered
        alert_below = _make_alert("e1", obs=0.2)

        result = s.generate_weights(
            np.zeros(10), 2, test_extra={"alerts": [[alert_above], [alert_below]]}
        )
        assert result[0] == 1.0
        assert result[1] == 0.0

    def test_wrong_weight_count_raises(self):
        mapper = _mock_asset_mapper({})
        with pytest.raises(ValueError, match="5 surprise weights"):
            WeightedSurpriseStrategy((0.5, 0.5), mapper)

    def test_non_tradeable_entity_skipped(self):
        mapper = _mock_asset_mapper({"e1": "AAPL"})  # e2 not in map
        weights = (1.0, 1.0, 1.0, 1.0, 1.0)
        s = WeightedSurpriseStrategy(weights, mapper, threshold=0.5)

        # e2 is not tradeable → even if surprise is huge, not counted
        alert = _make_alert("e2", obs=100.0)
        result = s.generate_weights(np.zeros(10), 1, test_extra={"alerts": [[alert]]})
        assert result[0] == 0.0


# ── 4. SACPortfolio Deterministic Weights ─────────────────────


class TestSACPortfolioWeights:
    """Proof 4: SAC policy → conviction weights."""

    def test_produces_weights_from_alerts(self):
        state_dim = StateAssembler(max_entities=5).state_dim
        cfg = SACConfig(
            hidden_dim=16, num_hidden=1, max_position=0.5, leverage_limit=1.0
        )
        trainer = SACTrainer(state_dim, 5, cfg)
        assembler = StateAssembler(max_entities=5)
        mapper = _mock_asset_mapper({"e1": "AAPL", "e2": "GOOG"})

        alerts = [
            [_make_alert("e1", composite=3.0), _make_alert("e2", composite=2.0)]
            for _ in range(3)
        ]
        beliefs = [[_make_belief("e1"), _make_belief("e2")] for _ in range(3)]
        market = [{"vol": 0.02, "return": 0.01} for _ in range(3)]

        s = SACPortfolioStrategy(trainer, assembler, mapper)
        result = s.generate_weights(
            np.zeros(10),
            3,
            test_extra={
                "alerts": alerts,
                "beliefs": beliefs,
                "market_features": market,
            },
        )
        assert result.shape == (3,)
        # Weights should be non-negative (mean absolute position)
        assert (result >= 0).all()

    def test_deterministic_is_repeatable(self):
        state_dim = StateAssembler(max_entities=5).state_dim
        cfg = SACConfig(hidden_dim=16, num_hidden=1)
        trainer = SACTrainer(state_dim, 5, cfg)
        assembler = StateAssembler(max_entities=5)
        mapper = _mock_asset_mapper({"e1": "AAPL"})

        alerts = [[_make_alert("e1", composite=3.0)]]
        test_extra = {
            "alerts": alerts,
            "beliefs": [[_make_belief("e1")]],
            "market_features": [{"vol": 0.02}],
        }

        s = SACPortfolioStrategy(trainer, assembler, mapper)
        r1 = s.generate_weights(np.zeros(10), 1, test_extra=test_extra)
        r2 = s.generate_weights(np.zeros(10), 1, test_extra=test_extra)
        np.testing.assert_array_equal(r1, r2)


# ── 5. Safe Fallbacks ────────────────────────────────────────


class TestSafeFallbacks:
    """Proof 5: missing/empty inputs → zero weights."""

    def test_weighted_no_test_extra(self):
        mapper = _mock_asset_mapper({"e1": "AAPL"})
        s = WeightedSurpriseStrategy((0.3, 0.15, 0.25, 0.2, 0.1), mapper)
        result = s.generate_weights(np.zeros(10), 5, test_extra=None)
        np.testing.assert_array_equal(result, 0.0)
        assert result.shape == (5,)

    def test_weighted_no_alerts_key(self):
        mapper = _mock_asset_mapper({"e1": "AAPL"})
        s = WeightedSurpriseStrategy((0.3, 0.15, 0.25, 0.2, 0.1), mapper)
        result = s.generate_weights(np.zeros(10), 5, test_extra={"foo": "bar"})
        np.testing.assert_array_equal(result, 0.0)

    def test_sac_no_test_extra(self):
        state_dim = StateAssembler(max_entities=5).state_dim
        cfg = SACConfig(hidden_dim=16, num_hidden=1)
        trainer = SACTrainer(state_dim, 5, cfg)
        assembler = StateAssembler(max_entities=5)
        mapper = _mock_asset_mapper({})
        s = SACPortfolioStrategy(trainer, assembler, mapper)
        result = s.generate_weights(np.zeros(10), 5, test_extra=None)
        np.testing.assert_array_equal(result, 0.0)

    def test_sac_empty_alerts(self):
        state_dim = StateAssembler(max_entities=5).state_dim
        cfg = SACConfig(hidden_dim=16, num_hidden=1)
        trainer = SACTrainer(state_dim, 5, cfg)
        assembler = StateAssembler(max_entities=5)
        mapper = _mock_asset_mapper({})
        s = SACPortfolioStrategy(trainer, assembler, mapper)
        result = s.generate_weights(
            np.zeros(10),
            3,
            test_extra={
                "alerts": [[], [], []],
                "beliefs": [[], [], []],
                "market_features": [{}, {}, {}],
            },
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_sac_no_active_entities(self):
        """All alerts are non-tradeable → zero weights."""
        state_dim = StateAssembler(max_entities=5).state_dim
        cfg = SACConfig(hidden_dim=16, num_hidden=1)
        trainer = SACTrainer(state_dim, 5, cfg)
        assembler = StateAssembler(max_entities=5)
        mapper = _mock_asset_mapper({})  # nothing tradeable
        s = SACPortfolioStrategy(trainer, assembler, mapper)
        alerts = [[_make_alert("e1", composite=5.0)]]
        result = s.generate_weights(
            np.zeros(10),
            1,
            test_extra={"alerts": alerts, "beliefs": [[]], "market_features": [{}]},
        )
        assert result[0] == 0.0


# ── 6. Shape Correctness ─────────────────────────────────────


class TestShapeCorrectness:
    """Proof 6: result is always (test_length,)."""

    @pytest.mark.parametrize("length", [0, 1, 5, 100])
    def test_weighted_shapes(self, length):
        mapper = _mock_asset_mapper({"e1": "AAPL"})
        s = WeightedSurpriseStrategy((0.3, 0.15, 0.25, 0.2, 0.1), mapper)
        result = s.generate_weights(np.zeros(10), length, test_extra=None)
        assert result.shape == (length,)

    @pytest.mark.parametrize("length", [0, 1, 5])
    def test_sac_shapes(self, length):
        state_dim = StateAssembler(max_entities=5).state_dim
        cfg = SACConfig(hidden_dim=16, num_hidden=1)
        trainer = SACTrainer(state_dim, 5, cfg)
        assembler = StateAssembler(max_entities=5)
        mapper = _mock_asset_mapper({})
        s = SACPortfolioStrategy(trainer, assembler, mapper)
        result = s.generate_weights(np.zeros(10), length, test_extra=None)
        assert result.shape == (length,)
