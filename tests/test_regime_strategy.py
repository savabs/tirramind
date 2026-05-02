"""Tests for RegimeStrategy — world model belief → position weights.

Covers:
    - Basic weight generation from regime beliefs
    - All-expansion / all-crisis / all-stable / mixed scenarios
    - Missing beliefs → neutral
    - Boundary: P(expansion)=1 → weight=1, P(crisis)=1 → weight=-1
    - Custom regime variable / state labels
    - Walk-Forward integration (expanding window)
    - Edge cases: empty beliefs, wrong variable, no probabilities key, etc.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from agent.models.belief import BeliefState
from agent.quant.backtest import WalkForward
from agent.quant.regime_strategy import RegimeStrategy

# ── Helpers ────────────────────────────────────────────────────


def _belief(
    probs: dict[str, float],
    variable: str = "regime.macro",
    dist_type: str = "categorical",
) -> BeliefState:
    """Shorthand for creating a test BeliefState."""
    now = time.time()
    return BeliefState(
        variable_name=variable,
        version=1,
        effective_at=now - 60,
        computed_at=now,
        dist_type=dist_type,
        probabilities=probs,
    )


def _belief_set(probs: dict[str, float], variable: str = "regime.macro") -> list[BeliefState]:
    """One-element belief set (common case)."""
    return [_belief(probs, variable=variable)]


# ── Basic Behavior ─────────────────────────────────────────────


class TestBasicWeightGeneration:
    """Core weight generation logic."""

    def test_pure_expansion(self):
        """P(expansion)=1 → weight = +1."""
        s = RegimeStrategy()
        beliefs = [_belief_set({"expansion": 1.0, "stable": 0.0, "crisis": 0.0})] * 5
        w = s.generate_weights(np.zeros(10), 5, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, 1.0)

    def test_pure_crisis(self):
        """P(crisis)=1 → weight = -1."""
        s = RegimeStrategy()
        beliefs = [_belief_set({"expansion": 0.0, "stable": 0.0, "crisis": 1.0})] * 5
        w = s.generate_weights(np.zeros(10), 5, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, -1.0)

    def test_equal_expansion_crisis(self):
        """Equal P(expansion) and P(crisis) → weight = 0."""
        s = RegimeStrategy()
        beliefs = [_belief_set({"expansion": 0.4, "stable": 0.2, "crisis": 0.4})] * 3
        w = s.generate_weights(np.zeros(10), 3, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, 0.0)

    def test_mild_bullish(self):
        """P(expansion)=0.6, P(crisis)=0.1 → weight = 0.5."""
        s = RegimeStrategy()
        beliefs = [_belief_set({"expansion": 0.6, "stable": 0.3, "crisis": 0.1})] * 4
        w = s.generate_weights(np.zeros(10), 4, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, 0.5)

    def test_output_shape(self):
        """Weights have shape (test_length,)."""
        s = RegimeStrategy()
        beliefs = [_belief_set({"expansion": 0.5, "crisis": 0.5})] * 7
        w = s.generate_weights(np.zeros(20), 7, test_extra={"beliefs": beliefs})
        assert w.shape == (7,)
        assert w.dtype == np.float64

    def test_strategy_name(self):
        s = RegimeStrategy()
        assert s.name == "RegimeStrategy"


# ── Missing / Absent Beliefs ──────────────────────────────────


class TestMissingBeliefs:
    """Graceful handling of absent or partial beliefs."""

    def test_no_test_extra(self):
        """No test_extra at all → default (neutral)."""
        s = RegimeStrategy()
        w = s.generate_weights(np.zeros(10), 5)
        np.testing.assert_array_almost_equal(w, 0.0)
        assert len(w) == 5

    def test_no_beliefs_key(self):
        """test_extra without 'beliefs' key → default."""
        s = RegimeStrategy()
        w = s.generate_weights(np.zeros(10), 5, test_extra={"other": [1, 2, 3]})
        np.testing.assert_array_almost_equal(w, 0.0)

    def test_none_in_beliefs_list(self):
        """None entries in belief list → default for that period."""
        s = RegimeStrategy()
        beliefs = [
            _belief_set({"expansion": 0.8, "crisis": 0.1}),
            None,
            _belief_set({"expansion": 0.2, "crisis": 0.7}),
        ]
        w = s.generate_weights(np.zeros(10), 3, test_extra={"beliefs": beliefs})
        assert w[0] == pytest.approx(0.7)
        assert w[1] == pytest.approx(0.0)  # None → default
        assert w[2] == pytest.approx(-0.5)

    def test_empty_belief_set(self):
        """Empty list of beliefs for a period → default."""
        s = RegimeStrategy()
        beliefs = [[], _belief_set({"expansion": 1.0, "crisis": 0.0})]
        w = s.generate_weights(np.zeros(10), 2, test_extra={"beliefs": beliefs})
        assert w[0] == pytest.approx(0.0)
        assert w[1] == pytest.approx(1.0)

    def test_wrong_variable_name(self):
        """Belief for different variable → default."""
        s = RegimeStrategy()
        beliefs = [_belief_set({"expansion": 1.0, "crisis": 0.0}, variable="latent.stress_level")] * 3
        w = s.generate_weights(np.zeros(10), 3, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, 0.0)

    def test_belief_without_probabilities(self):
        """Gaussian belief (no probabilities dict) → default."""
        s = RegimeStrategy()
        now = time.time()
        b = BeliefState(
            variable_name="regime.macro",
            version=1,
            effective_at=now - 60,
            computed_at=now,
            dist_type="gaussian",
            mean=0.5,
            variance=0.1,
        )
        beliefs = [[b]] * 3
        w = s.generate_weights(np.zeros(10), 3, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, 0.0)

    def test_shorter_beliefs_than_test_length(self):
        """Fewer belief sets than test_length → trailing defaults."""
        s = RegimeStrategy()
        beliefs = [_belief_set({"expansion": 0.9, "crisis": 0.0})] * 2
        w = s.generate_weights(np.zeros(10), 5, test_extra={"beliefs": beliefs})
        assert w[0] == pytest.approx(0.9)
        assert w[1] == pytest.approx(0.9)
        assert w[2] == pytest.approx(0.0)  # default
        assert w[3] == pytest.approx(0.0)
        assert w[4] == pytest.approx(0.0)

    def test_test_extra_none(self):
        """Explicit test_extra=None → default."""
        s = RegimeStrategy()
        w = s.generate_weights(np.zeros(10), 3, test_extra=None)
        np.testing.assert_array_almost_equal(w, 0.0)


# ── Custom Configuration ──────────────────────────────────────


class TestCustomConfig:
    """Non-default regime variable / state labels."""

    def test_custom_variable_name(self):
        s = RegimeStrategy(regime_variable="regime.sector")
        beliefs = [_belief_set({"expansion": 0.7, "crisis": 0.1}, variable="regime.sector")] * 3
        w = s.generate_weights(np.zeros(10), 3, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, 0.6)

    def test_custom_state_labels(self):
        s = RegimeStrategy(expansion_state="bullish", crisis_state="bearish")
        beliefs = [_belief_set({"bullish": 0.8, "neutral": 0.1, "bearish": 0.1})] * 4
        w = s.generate_weights(np.zeros(10), 4, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, 0.7)

    def test_custom_default_weight(self):
        s = RegimeStrategy(default_weight=0.5)
        w = s.generate_weights(np.zeros(10), 3)
        np.testing.assert_array_almost_equal(w, 0.5)

    def test_missing_expansion_label(self):
        """Probabilities dict lacks the expansion state label → P(expansion)=0."""
        s = RegimeStrategy()
        beliefs = [_belief_set({"stable": 0.6, "crisis": 0.4})] * 3
        w = s.generate_weights(np.zeros(10), 3, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, -0.4)

    def test_missing_crisis_label(self):
        """Probabilities dict lacks the crisis state label → P(crisis)=0."""
        s = RegimeStrategy()
        beliefs = [_belief_set({"expansion": 0.7, "stable": 0.3})] * 3
        w = s.generate_weights(np.zeros(10), 3, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, 0.7)


# ── Boundary & Clipping ──────────────────────────────────────


class TestBoundary:
    """Weight clamping and extreme inputs."""

    def test_weight_clamped_at_plus_one(self):
        """Even if P(expansion)-P(crisis) > 1 (shouldn't happen), clamp."""
        s = RegimeStrategy()
        # Degenerate probabilities that don't sum to 1
        beliefs = [_belief_set({"expansion": 1.5, "crisis": 0.0})] * 2
        w = s.generate_weights(np.zeros(10), 2, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, 1.0)

    def test_weight_clamped_at_minus_one(self):
        s = RegimeStrategy()
        beliefs = [_belief_set({"expansion": 0.0, "crisis": 1.5})] * 2
        w = s.generate_weights(np.zeros(10), 2, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, -1.0)

    def test_all_weights_in_range(self):
        """Randomized: all weights in [-1, 1]."""
        rng = np.random.default_rng(42)
        s = RegimeStrategy()
        beliefs = []
        for _ in range(50):
            p = rng.dirichlet([1, 1, 1])
            beliefs.append(
                _belief_set(
                    {
                        "expansion": float(p[0]),
                        "stable": float(p[1]),
                        "crisis": float(p[2]),
                    }
                )
            )
        w = s.generate_weights(np.zeros(100), 50, test_extra={"beliefs": beliefs})
        assert np.all(w >= -1.0)
        assert np.all(w <= 1.0)


# ── Mixed Time-Varying Beliefs ────────────────────────────────


class TestTimeVarying:
    """Beliefs that change over time."""

    def test_transition_expansion_to_crisis(self):
        s = RegimeStrategy()
        beliefs = [
            _belief_set({"expansion": 0.8, "stable": 0.1, "crisis": 0.1}),
            _belief_set({"expansion": 0.4, "stable": 0.3, "crisis": 0.3}),
            _belief_set({"expansion": 0.1, "stable": 0.1, "crisis": 0.8}),
        ]
        w = s.generate_weights(np.zeros(10), 3, test_extra={"beliefs": beliefs})
        assert w[0] > w[1] > w[2]
        assert w[0] == pytest.approx(0.7)
        assert w[1] == pytest.approx(0.1)
        assert w[2] == pytest.approx(-0.7)

    def test_multiple_beliefs_in_set(self):
        """Multiple BeliefStates per period — uses the one with matching variable."""
        s = RegimeStrategy()
        other = _belief({"high": 0.9, "low": 0.1}, variable="latent.stress_level")
        regime = _belief({"expansion": 0.3, "stable": 0.3, "crisis": 0.4})
        beliefs = [[other, regime]] * 3
        w = s.generate_weights(np.zeros(10), 3, test_extra={"beliefs": beliefs})
        np.testing.assert_array_almost_equal(w, -0.1)


# ── Walk-Forward Integration ──────────────────────────────────


class TestWalkForwardIntegration:
    """RegimeStrategy with the WalkForward engine."""

    def test_single_fold(self):
        """Minimal walk-forward: one fold with expansion beliefs."""
        rng = np.random.default_rng(123)
        n = 30  # min_train=20, test=10
        returns = rng.normal(0.001, 0.02, n)

        # Build beliefs aligned with the full returns array.
        # WalkForward will slice extra["beliefs"][split:split+test_size].
        all_beliefs = []
        for _ in range(n):
            all_beliefs.append(_belief_set({"expansion": 0.7, "stable": 0.2, "crisis": 0.1}))

        wf = WalkForward(min_train=20, test_size=10, periods_per_year=52)
        result = wf.run(
            RegimeStrategy(),
            returns,
            extra={"beliefs": all_beliefs},
        )
        assert result.strategy_name == "RegimeStrategy"
        assert len(result.folds) == 1
        assert result.folds[0].test_size == 10
        # All weights should be 0.6 (= 0.7 - 0.1)
        np.testing.assert_array_almost_equal(result.folds[0].weights, 0.6)

    def test_multiple_folds(self):
        """Multiple expanding folds."""
        rng = np.random.default_rng(456)
        n = 50
        returns = rng.normal(0.001, 0.02, n)

        all_beliefs = []
        for i in range(n):
            # shift from expansion to crisis over time
            t = i / n
            p_exp = max(0.0, 0.8 - t)
            p_cri = min(0.8, t)
            p_sta = 1.0 - p_exp - p_cri
            all_beliefs.append(
                _belief_set(
                    {
                        "expansion": p_exp,
                        "stable": p_sta,
                        "crisis": p_cri,
                    }
                )
            )

        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(
            RegimeStrategy(),
            returns,
            extra={"beliefs": all_beliefs},
        )
        assert len(result.folds) == 3  # 50 total: folds at 20,30,40 start
        assert len(result.all_weights) == 30
        assert result.equity_curve.shape == (30,)

        # Early fold → bullish weights, later fold → bearish weights
        assert np.mean(result.folds[0].weights) > np.mean(result.folds[2].weights)

    def test_backtest_result_has_metrics(self):
        """BacktestResult includes standard scoring metrics."""
        rng = np.random.default_rng(789)
        n = 30
        returns = rng.normal(0.002, 0.02, n)
        beliefs = [_belief_set({"expansion": 0.6, "crisis": 0.1})] * n

        wf = WalkForward(min_train=20, test_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra={"beliefs": beliefs})

        assert "sharpe" in result.aggregate_metrics
        assert "sortino" in result.aggregate_metrics
        assert "max_drawdown" in result.aggregate_metrics

    def test_crisis_beliefs_reduce_exposure(self):
        """Crisis regime → negative weights → inverse correlation with returns."""
        rng = np.random.default_rng(101)
        n = 40
        # Positive returns market
        returns = np.abs(rng.normal(0.005, 0.01, n))
        # All crisis beliefs
        beliefs = [_belief_set({"expansion": 0.0, "stable": 0.0, "crisis": 1.0})] * n

        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra={"beliefs": beliefs})

        # Crisis weights are -1.0 → weighted returns are negative
        assert np.all(result.all_weights == pytest.approx(-1.0))
        assert np.all(result.all_test_returns <= 0)
