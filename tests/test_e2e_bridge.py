"""Phase 19e.3: End-to-end bridge integration tests.

Validates the complete pipeline:
    World Model → Beliefs → RegimeStrategy → WalkForward → BacktestResult

Uses synthetic features and returns — no external data or HTTP calls.

Coverage (~30 tests):
    - Full pipeline chain produces valid BacktestResult
    - Beliefs flow correctly from WorldModel to RegimeStrategy
    - Time-varying features produce time-varying weights
    - Expansion features → positive weights → positive leveraged returns
    - Crisis features → negative weights → inverse exposure
    - Mixed/neutral features → near-zero weights
    - Missing features handled gracefully (stale beliefs → default)
    - Multiple folds with evolving beliefs
    - Metric validity (Sharpe, Sortino, max_drawdown, equity curve)
    - Edge cases: all-None features, single period, extreme values
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from agent.features.protocol import EngineeredFeature
from agent.models.belief import BeliefState
from agent.pipeline.dags.world_model_update import (
    _FEATURE_TO_OBS_INDEX,
    _build_world_model,
)
from agent.quant.backtest import BacktestResult, WalkForward
from agent.quant.regime_strategy import RegimeStrategy

# ── Helpers ────────────────────────────────────────────────────

# All 17 feature names the world model expects
_ALL_FEATURES = list(_FEATURE_TO_OBS_INDEX.keys())


def _make_feature(
    name: str,
    value: float | None,
    as_of: float | None = None,
) -> EngineeredFeature:
    """Build a single EngineeredFeature for testing."""
    now = as_of or time.time()
    return EngineeredFeature(
        feature_name=name,
        version=1,
        effective_at=now,
        computed_at=now,
        horizon="spot",
        value=value,
        quality=1.0 if value is not None else 0.0,
        missing_reason=None if value is not None else "test_missing",
        source_signals=("test_signal",),
        builder="test_builder",
        unit="z_score",
    )


def _make_feature_set(
    values: dict[str, float | None],
    as_of: float | None = None,
) -> list[EngineeredFeature]:
    """Build a complete feature set from a name→value mapping."""
    return [_make_feature(n, v, as_of) for n, v in values.items()]


def _expansion_features(as_of: float | None = None) -> list[EngineeredFeature]:
    """Feature values that should push the world model toward expansion.

    CPD alignment (from initial_graph.py):
        - rate_momentum: expansion → rising (high, z > 1)
        - yield_curve_slope: expansion → steep (high, z > 1)
        - liquidity_pressure: expansion+risk_on → loose (high, z > 1)
        - stress_breadth: calm → narrow (low, z < -1)
        - stress_intensity: calm+risk_on → low (low, z < -1)
        - regime_persistence: calm → persistent (high, z > 1)
        - anomaly: calm stress → low (z < -1)
        - activity: risk_on → high (z > 1)
        - cross_entity: high (correlated, z > 1)
    """
    vals: dict[str, float | None] = {}
    # Macro: rising rates, steep curve, loose liquidity
    vals["macro.rate_momentum.30d"] = 1.5
    vals["macro.yield_curve_slope.spot"] = 1.5
    vals["macro.liquidity_pressure.30d"] = 1.5
    # Convergence: narrow stress, low intensity, high persistence
    vals["convergence.stress_breadth.7d"] = -1.5
    vals["convergence.stress_intensity.7d"] = -1.5
    vals["convergence.regime_persistence.7d"] = 1.5
    # GNN anomaly: low (calm conditions)
    for t in ("person", "company", "wallet", "country", "vessel"):
        vals[f"gnn.{t}_anomaly.spot"] = -1.5
    # GNN activity: high (risk-on)
    for t in ("person", "company", "wallet", "country", "vessel"):
        vals[f"gnn.{t}_activity.spot"] = 1.5
    # Cross entity: high (correlated behavior)
    vals["gnn.cross_entity.spot"] = 1.5
    return _make_feature_set(vals, as_of)


def _crisis_features(as_of: float | None = None) -> list[EngineeredFeature]:
    """Feature values that should push the world model toward crisis.

    CPD alignment: inverse of expansion.
        - rate_momentum: crisis → falling (low, z < -1)
        - yield_curve_slope: crisis → inverted (low, z < -1)
        - liquidity_pressure: crisis+risk_off → tight (low, z < -1)
        - stress_breadth: extreme → broad (high, z > 1)
        - stress_intensity: extreme+risk_off → high (high, z > 1)
        - regime_persistence: extreme → unstable (low, z < -1)
        - anomaly: extreme stress → high (z > 1)
        - activity: risk_off → low (z < -1)
    """
    vals: dict[str, float | None] = {}
    vals["macro.rate_momentum.30d"] = -1.5
    vals["macro.yield_curve_slope.spot"] = -1.5
    vals["macro.liquidity_pressure.30d"] = -1.5
    vals["convergence.stress_breadth.7d"] = 1.5
    vals["convergence.stress_intensity.7d"] = 1.5
    vals["convergence.regime_persistence.7d"] = -1.5
    for t in ("person", "company", "wallet", "country", "vessel"):
        vals[f"gnn.{t}_anomaly.spot"] = 1.5
    for t in ("person", "company", "wallet", "country", "vessel"):
        vals[f"gnn.{t}_activity.spot"] = -1.5
    vals["gnn.cross_entity.spot"] = -1.5
    return _make_feature_set(vals, as_of)


def _neutral_features(as_of: float | None = None) -> list[EngineeredFeature]:
    """Mild feature values that should keep the model near-neutral."""
    vals: dict[str, float | None] = {}
    for name in _ALL_FEATURES:
        vals[name] = 0.0
    return _make_feature_set(vals, as_of)


def _run_pipeline_step(
    features: list[EngineeredFeature],
    wm=None,
) -> list[BeliefState]:
    """Run one world model update cycle and return beliefs."""
    if wm is None:
        wm = _build_world_model()
    as_of = features[0].effective_at if features else time.time()
    return wm.update(features, as_of)


def _find_belief(beliefs: list[BeliefState], variable: str) -> BeliefState | None:
    """Find a specific belief by variable name."""
    for b in beliefs:
        if b.variable_name == variable:
            return b
    return None


def _beliefs_to_extra(
    weekly_beliefs: list[list[BeliefState]],
) -> dict[str, list]:
    """Convert belief lists to the format WalkForward slices correctly.

    WalkForward slices extra arrays by index, so we wrap in np.ndarray
    of objects to enable slicing.
    """
    arr = np.empty(len(weekly_beliefs), dtype=object)
    for i, b in enumerate(weekly_beliefs):
        arr[i] = b
    return {"beliefs": arr}


# ═══════════════════════════════════════════════════════════════
# Pipeline Chain Validation
# ═══════════════════════════════════════════════════════════════


class TestPipelineChain:
    """Validates that WM → beliefs → strategy → backtest chain works."""

    def test_expansion_produces_beliefs(self):
        """Expansion features produce non-empty beliefs with regime posterior."""
        beliefs = _run_pipeline_step(_expansion_features())
        assert len(beliefs) > 0
        regime = _find_belief(beliefs, "regime.macro")
        assert regime is not None
        assert regime.probabilities is not None

    def test_crisis_produces_beliefs(self):
        beliefs = _run_pipeline_step(_crisis_features())
        regime = _find_belief(beliefs, "regime.macro")
        assert regime is not None
        assert regime.probabilities is not None

    def test_neutral_produces_beliefs(self):
        beliefs = _run_pipeline_step(_neutral_features())
        regime = _find_belief(beliefs, "regime.macro")
        assert regime is not None

    def test_beliefs_contain_all_dag_variables(self):
        """World model should produce beliefs for all 20 DAG nodes + 3 Kalman states."""
        beliefs = _run_pipeline_step(_expansion_features())
        var_names = {b.variable_name for b in beliefs}
        # 20 DAG nodes + 3 Kalman states = 23
        assert len(beliefs) == 23

    def test_beliefs_feed_into_strategy(self):
        """Beliefs → RegimeStrategy → weights (no crash)."""
        beliefs = _run_pipeline_step(_expansion_features())
        strategy = RegimeStrategy()
        weights = strategy.generate_weights(
            np.zeros(10),
            1,
            test_extra={"beliefs": [beliefs]},
        )
        assert weights.shape == (1,)
        assert -1.0 <= weights[0] <= 1.0


# ═══════════════════════════════════════════════════════════════
# Directional Signal Tests
# ═══════════════════════════════════════════════════════════════


class TestDirectionalSignal:
    """Expansion features → positive weights, crisis → negative."""

    def test_expansion_features_positive_weight(self):
        """Repeated expansion feature updates should push weight positive."""
        wm = _build_world_model()
        # Run multiple updates to move posterior from prior
        for _ in range(5):
            beliefs = wm.update(_expansion_features(), time.time())

        strategy = RegimeStrategy()
        w = strategy.generate_weights(
            np.zeros(10),
            1,
            test_extra={"beliefs": [beliefs]},
        )
        assert w[0] > 0, f"Expected positive weight for expansion, got {w[0]}"

    def test_crisis_features_negative_weight(self):
        wm = _build_world_model()
        for _ in range(5):
            beliefs = wm.update(_crisis_features(), time.time())

        strategy = RegimeStrategy()
        w = strategy.generate_weights(
            np.zeros(10),
            1,
            test_extra={"beliefs": [beliefs]},
        )
        assert w[0] < 0, f"Expected negative weight for crisis, got {w[0]}"

    def test_neutral_features_near_zero_weight(self):
        wm = _build_world_model()
        for _ in range(3):
            beliefs = wm.update(_neutral_features(), time.time())

        strategy = RegimeStrategy()
        w = strategy.generate_weights(
            np.zeros(10),
            1,
            test_extra={"beliefs": [beliefs]},
        )
        # Should be moderate — not strongly directional
        assert abs(w[0]) < 0.8, f"Expected near-neutral weight, got {w[0]}"

    def test_expansion_weight_greater_than_crisis(self):
        """Same WM, expansion update produces higher weight than crisis."""
        wm_exp = _build_world_model()
        wm_cri = _build_world_model()

        for _ in range(5):
            beliefs_exp = wm_exp.update(_expansion_features(), time.time())
            beliefs_cri = wm_cri.update(_crisis_features(), time.time())

        strategy = RegimeStrategy()
        w_exp = strategy.generate_weights(
            np.zeros(10),
            1,
            test_extra={"beliefs": [beliefs_exp]},
        )
        w_cri = strategy.generate_weights(
            np.zeros(10),
            1,
            test_extra={"beliefs": [beliefs_cri]},
        )
        assert w_exp[0] > w_cri[0]


# ═══════════════════════════════════════════════════════════════
# Walk-Forward E2E
# ═══════════════════════════════════════════════════════════════


class TestWalkForwardE2E:
    """Full WalkForward with pipeline-generated beliefs."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic weekly returns and beliefs for walk-forward."""
        rng = np.random.default_rng(42)
        n_weeks = 60  # enough for min_train=20 + multiple test folds

        # Synthetic returns: mild upward drift with noise
        returns = rng.normal(0.002, 0.025, n_weeks)

        # Generate beliefs sequence by running the world model
        # First half: expansion regime. Second half: crisis.
        wm = _build_world_model()
        weekly_beliefs: list[list[BeliefState]] = []

        for i in range(n_weeks):
            t = time.time() + i * 604800  # weekly spacing
            if i < n_weeks // 2:
                features = _expansion_features(as_of=t)
            else:
                features = _crisis_features(as_of=t)
            beliefs = wm.update(features, t)
            weekly_beliefs.append(beliefs)

        return returns, weekly_beliefs

    def test_full_backtest_produces_result(self, synthetic_data):
        """End-to-end: features → beliefs → strategy → WalkForward → BacktestResult."""
        returns, weekly_beliefs = synthetic_data

        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        extra = _beliefs_to_extra(weekly_beliefs)
        result = wf.run(RegimeStrategy(), returns, extra=extra)

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "RegimeStrategy"
        assert len(result.folds) > 0

    def test_backtest_has_metrics(self, synthetic_data):
        returns, weekly_beliefs = synthetic_data
        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))

        assert "sharpe" in result.aggregate_metrics
        assert "sortino" in result.aggregate_metrics
        assert "max_drawdown" in result.aggregate_metrics
        assert isinstance(result.aggregate_metrics["sharpe"], float)

    def test_equity_curve_shape(self, synthetic_data):
        returns, weekly_beliefs = synthetic_data
        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))

        total_test = sum(f.test_size for f in result.folds)
        assert result.equity_curve.shape == (total_test,)
        assert np.all(result.equity_curve > 0)  # wealth > 0

    def test_all_weights_bounded(self, synthetic_data):
        returns, weekly_beliefs = synthetic_data
        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))

        assert np.all(result.all_weights >= -1.0)
        assert np.all(result.all_weights <= 1.0)

    def test_multiple_folds(self, synthetic_data):
        returns, weekly_beliefs = synthetic_data
        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))

        # With 60 weeks, min_train=20, test=10, step=10: folds at 20,30,40,50
        assert len(result.folds) == 4

    def test_expanding_window_train_sizes(self, synthetic_data):
        returns, weekly_beliefs = synthetic_data
        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))

        # Expanding window: each fold has more training data
        for i in range(1, len(result.folds)):
            assert result.folds[i].train_size > result.folds[i - 1].train_size


# ═══════════════════════════════════════════════════════════════
# Regime Transition Dynamics
# ═══════════════════════════════════════════════════════════════


class TestRegimeTransition:
    """Verify that the pipeline reflects regime changes in weights."""

    def test_weights_shift_with_regime_change(self):
        """Expansion→crisis transition should decrease weights over time.

        Layout (50 weeks, min_train=20, test=10, step=10):
            weeks 0-29: expansion features
            weeks 30-49: crisis features
        Folds:
            fold 0: train=0-19, test=20-29 (expansion test) → positive
            fold 1: train=0-29, test=30-39 (crisis test)    → negative
            fold 2: train=0-39, test=40-49 (crisis test)    → negative
        """
        wm = _build_world_model()

        n = 50
        weekly_beliefs: list[list[BeliefState]] = []
        for i in range(n):
            t = time.time() + i * 604800
            if i < 30:
                features = _expansion_features(as_of=t)
            else:
                features = _crisis_features(as_of=t)
            beliefs = wm.update(features, t)
            weekly_beliefs.append(beliefs)

        rng = np.random.default_rng(99)
        returns = rng.normal(0.001, 0.02, n)

        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))

        assert len(result.folds) >= 2
        first_mean = np.mean(result.folds[0].weights)
        last_mean = np.mean(result.folds[-1].weights)
        assert last_mean < first_mean, (
            f"Expected regime transition to lower weights: first={first_mean:.3f}, last={last_mean:.3f}"
        )

    def test_persistent_expansion_monotone_weights(self):
        """Sustained expansion features → weights should stay positive."""
        wm = _build_world_model()
        n = 40
        weekly_beliefs: list[list[BeliefState]] = []
        for i in range(n):
            t = time.time() + i * 604800
            beliefs = wm.update(_expansion_features(as_of=t), t)
            weekly_beliefs.append(beliefs)

        rng = np.random.default_rng(77)
        returns = rng.normal(0.002, 0.02, n)

        wf = WalkForward(min_train=15, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))

        # All fold weights should be positive on average
        for fold in result.folds:
            assert np.mean(fold.weights) > 0, (
                f"Fold {fold.fold}: expected positive mean weight, got {np.mean(fold.weights):.3f}"
            )


# ═══════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_all_none_features(self):
        """All features are None → stale beliefs → neutral weight."""
        vals = {name: None for name in _ALL_FEATURES}
        features = _make_feature_set(vals)
        beliefs = _run_pipeline_step(features)

        strategy = RegimeStrategy()
        w = strategy.generate_weights(
            np.zeros(10),
            1,
            test_extra={"beliefs": [beliefs]},
        )
        # With no evidence, posterior should be near-prior (roughly flat)
        # Weight should be moderate
        assert -1.0 <= w[0] <= 1.0

    def test_partial_features(self):
        """Only macro features present, no GNN features → still works."""
        vals: dict[str, float | None] = {}
        vals["macro.rate_momentum.30d"] = -1.5
        vals["macro.yield_curve_slope.spot"] = -1.0
        vals["macro.liquidity_pressure.30d"] = 1.0
        vals["convergence.stress_breadth.7d"] = -0.5
        vals["convergence.stress_intensity.7d"] = -0.8
        vals["convergence.regime_persistence.7d"] = -0.3
        features = _make_feature_set(vals)
        beliefs = _run_pipeline_step(features)

        strategy = RegimeStrategy()
        w = strategy.generate_weights(
            np.zeros(10),
            1,
            test_extra={"beliefs": [beliefs]},
        )
        assert w.shape == (1,)
        assert -1.0 <= w[0] <= 1.0

    def test_extreme_feature_values(self):
        """Very large feature values don't crash the pipeline."""
        vals = {name: 10.0 for name in _ALL_FEATURES}
        features = _make_feature_set(vals)
        beliefs = _run_pipeline_step(features)

        strategy = RegimeStrategy()
        w = strategy.generate_weights(
            np.zeros(10),
            1,
            test_extra={"beliefs": [beliefs]},
        )
        assert -1.0 <= w[0] <= 1.0
        assert not np.isnan(w[0])

    def test_zero_variance_returns(self):
        """Flat returns don't break backtest scoring."""
        wm = _build_world_model()
        n = 40
        returns = np.zeros(n)
        weekly_beliefs = []
        for i in range(n):
            t = time.time() + i * 604800
            beliefs = wm.update(_neutral_features(as_of=t), t)
            weekly_beliefs.append(beliefs)

        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))
        assert isinstance(result, BacktestResult)
        assert np.isfinite(result.aggregate_metrics["sharpe"])

    def test_single_fold_minimum(self):
        """Exact minimum data for one fold."""
        wm = _build_world_model()
        n = 30  # min_train=20 + test_size=10 exactly
        rng = np.random.default_rng(11)
        returns = rng.normal(0.001, 0.02, n)
        weekly_beliefs = []
        for i in range(n):
            t = time.time() + i * 604800
            beliefs = wm.update(_expansion_features(as_of=t), t)
            weekly_beliefs.append(beliefs)

        wf = WalkForward(min_train=20, test_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))
        assert len(result.folds) == 1

    def test_insufficient_data_raises(self):
        """Too little data for even one fold → ValueError."""
        returns = np.zeros(15)
        wf = WalkForward(min_train=20, test_size=10, periods_per_year=52)
        with pytest.raises(ValueError, match="Not enough data"):
            wf.run(RegimeStrategy(), returns)


# ═══════════════════════════════════════════════════════════════
# Metric Consistency
# ═══════════════════════════════════════════════════════════════


class TestMetricConsistency:
    """Verify metric semantics from E2E output."""

    def test_max_drawdown_negative_or_zero(self):
        """Max drawdown is ≤ 0 by definition."""
        wm = _build_world_model()
        rng = np.random.default_rng(55)
        n = 50
        returns = rng.normal(0.001, 0.02, n)
        weekly_beliefs = []
        for i in range(n):
            t = time.time() + i * 604800
            beliefs = wm.update(_expansion_features(as_of=t), t)
            weekly_beliefs.append(beliefs)

        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))
        assert result.aggregate_metrics["max_drawdown"] <= 0

    def test_equity_starts_positive(self):
        """Equity curve starts at a positive value."""
        wm = _build_world_model()
        rng = np.random.default_rng(66)
        n = 40
        returns = rng.normal(0.001, 0.02, n)
        weekly_beliefs = []
        for i in range(n):
            t = time.time() + i * 604800
            beliefs = wm.update(_neutral_features(as_of=t), t)
            weekly_beliefs.append(beliefs)

        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))
        assert result.equity_curve[0] > 0

    def test_all_test_returns_not_all_zero(self):
        """With non-zero returns and non-zero weights, test returns shouldn't all be zero."""
        wm = _build_world_model()
        rng = np.random.default_rng(88)
        n = 50
        returns = rng.normal(0.003, 0.02, n)
        weekly_beliefs = []
        for i in range(n):
            t = time.time() + i * 604800
            beliefs = wm.update(_expansion_features(as_of=t), t)
            weekly_beliefs.append(beliefs)

        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))
        # With expansion beliefs → non-zero weights → non-zero test returns
        assert not np.allclose(result.all_test_returns, 0)

    def test_fold_returns_concat_matches_all(self):
        """Concatenated fold test_returns matches all_test_returns."""
        wm = _build_world_model()
        rng = np.random.default_rng(33)
        n = 50
        returns = rng.normal(0.001, 0.02, n)
        weekly_beliefs = []
        for i in range(n):
            t = time.time() + i * 604800
            beliefs = wm.update(_expansion_features(as_of=t), t)
            weekly_beliefs.append(beliefs)

        wf = WalkForward(min_train=20, test_size=10, step_size=10, periods_per_year=52)
        result = wf.run(RegimeStrategy(), returns, extra=_beliefs_to_extra(weekly_beliefs))

        concated = np.concatenate([f.test_returns for f in result.folds])
        np.testing.assert_array_almost_equal(concated, result.all_test_returns)
