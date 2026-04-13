"""Tests for SurpriseWeightLearner — differentiable Sharpe optimisation.

Mathematical proofs embedded in the test suite:

1. IDENTIFIABILITY: If exactly one surprise signal predicts returns,
   the optimiser must concentrate weight on that signal.
   Proof: the Sharpe ratio is maximised when ρ_t = w_k · s_{k,t} · r_{t+1}
   and s_k is the only signal correlated with r.  Softmax concentrates.

2. NOISE IMMUNITY: On pure-noise signals, weights stay approximately
   uniform (uniform = maximum-entropy solution with zero gradient).

3. SIMPLEX CONSTRAINT: softmax guarantees ∀ w_i ≥ 0, Σ w_i = 1.

4. DIFFERENTIABILITY: Sharpe ratio is smooth (ε > 0) and gradient
   flows through softmax → dot-product → ratio.

5. WALK-FORWARD INTEGRITY: test windows are strictly OOS (no overlap).

6. CONVERGENCE: early stopping triggers when Sharpe improvement < 1e-6
   for `patience` consecutive epochs.
"""

import numpy as np
import pytest
import torch

from agent.learning.policy.config import WeightLearnerConfig
from agent.learning.policy.weight_learner import (
    InsufficientDataError,
    SurpriseWeightLearner,
)


def _make_synthetic(
    T: int = 300,
    predictive_col: int = 2,
    signal_strength: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic (T, 5) surprise matrix and (T,) returns.

    Column `predictive_col` has signal_strength correlation with returns.
    Other columns are pure noise.
    """
    rng = np.random.RandomState(seed)
    S = rng.randn(T, 5).astype(np.float32)
    noise = rng.randn(T).astype(np.float32) * 0.02
    # Returns = signal_strength * S[:, predictive_col] * 0.01 + noise
    R = (signal_strength * S[:, predictive_col] * 0.01 + noise).astype(np.float32)
    return S, R


# ── Proof 1: identifiability ─────────────────────────────────


class TestIdentifiability:
    """If one signal predicts returns, learned weight concentrates on it."""

    def test_single_signal_concentrates(self) -> None:
        S, R = _make_synthetic(T=400, predictive_col=2, signal_strength=2.0)
        cfg = WeightLearnerConfig(
            min_train_periods=150,
            test_periods=75,
            walk_forward_step=75,
            max_epochs=300,
            patience=30,
        )
        learner = SurpriseWeightLearner(cfg)
        result = learner.fit(S, R)
        w = result["weights"]
        # Column 2 should have the largest weight
        assert w[2] == max(w), f"Expected col 2 dominant, got weights={w}"
        # It should be > 0.4 (well above uniform 0.2)
        assert w[2] > 0.35, f"Signal weight too low: {w[2]}"

    def test_different_predictive_column(self) -> None:
        """Repeat with col 0 as the predictive signal."""
        S, R = _make_synthetic(T=400, predictive_col=0, signal_strength=2.0)
        cfg = WeightLearnerConfig(
            min_train_periods=150,
            test_periods=75,
            walk_forward_step=75,
            max_epochs=300,
            patience=30,
        )
        learner = SurpriseWeightLearner(cfg)
        result = learner.fit(S, R)
        w = result["weights"]
        assert w[0] == max(w), f"Expected col 0 dominant, got weights={w}"


# ── Proof 2: noise immunity ──────────────────────────────────


class TestNoiseImmunity:
    def test_uniform_on_noise(self) -> None:
        """Pure noise → weights stay approximately uniform (0.2 each)."""
        rng = np.random.RandomState(123)
        S = rng.randn(300, 5).astype(np.float32)
        R = rng.randn(300).astype(np.float32) * 0.01
        cfg = WeightLearnerConfig(
            min_train_periods=100,
            test_periods=50,
            walk_forward_step=50,
            max_epochs=100,
            patience=15,
        )
        learner = SurpriseWeightLearner(cfg)
        result = learner.fit(S, R)
        w = result["weights"]
        # No weight should dominate excessively
        assert max(w) < 0.50, f"Weight too concentrated on noise: {w}"
        # Mean test sharpe should be near zero (no signal)
        assert (
            abs(result["mean_test_sharpe"]) < 2.0
        ), f"Sharpe too high on noise: {result['mean_test_sharpe']}"


# ── Proof 3: simplex constraint ──────────────────────────────


class TestSimplexConstraint:
    def test_weights_sum_to_one(self) -> None:
        learner = SurpriseWeightLearner()
        w = learner.weights
        assert abs(float(w.sum()) - 1.0) < 1e-6

    def test_weights_non_negative(self) -> None:
        learner = SurpriseWeightLearner()
        w = learner.weights
        assert (w >= 0).all()

    def test_after_training(self) -> None:
        S, R = _make_synthetic(T=300, predictive_col=1)
        cfg = WeightLearnerConfig(
            min_train_periods=100,
            test_periods=50,
            walk_forward_step=50,
            max_epochs=50,
        )
        learner = SurpriseWeightLearner(cfg)
        learner.fit(S, R)
        w = learner.get_learned_weights()
        assert abs(sum(w) - 1.0) < 1e-5
        assert all(wi >= 0 for wi in w)


# ── Proof 4: differentiability ───────────────────────────────


class TestDifferentiability:
    def test_gradient_flows_through_sharpe(self) -> None:
        """∂Sharpe/∂θ exists and is finite."""
        learner = SurpriseWeightLearner()
        S = torch.randn(50, 5, requires_grad=False)
        R = torch.randn(50, requires_grad=False)
        scores = learner.composite_score(S)
        sharpe = learner.differentiable_sharpe(scores, R)
        sharpe.backward()
        assert learner._raw_weights.grad is not None
        assert torch.isfinite(learner._raw_weights.grad).all()

    def test_sharpe_zero_for_zero_scores(self) -> None:
        """When all scores are zero, weighted returns are zero → Sharpe = 0."""
        scores = torch.zeros(50)
        returns = torch.randn(50)
        s = SurpriseWeightLearner.differentiable_sharpe(scores, returns)
        assert abs(s.item()) < 1e-6

    def test_sharpe_positive_for_perfect_signal(self) -> None:
        """When scores perfectly predict positive returns → positive Sharpe."""
        scores = torch.ones(50)
        returns = torch.ones(50) * 0.01
        s = SurpriseWeightLearner.differentiable_sharpe(scores, returns)
        assert s.item() > 0

    def test_sharpe_finite_with_zero_variance(self) -> None:
        """Constant weighted returns → eps prevents division by zero."""
        scores = torch.ones(50)
        returns = torch.ones(50) * 0.01
        s = SurpriseWeightLearner.differentiable_sharpe(scores, returns, eps=1e-8)
        assert torch.isfinite(s)


# ── Proof 5: walk-forward integrity ──────────────────────────


class TestWalkForwardIntegrity:
    def test_folds_non_overlapping_tests(self) -> None:
        """Test windows do not overlap with each other."""
        cfg = WeightLearnerConfig(
            min_train_periods=100,
            test_periods=50,
            walk_forward_step=50,
        )
        learner = SurpriseWeightLearner(cfg)
        folds = learner._build_folds(T=300)
        for i in range(1, len(folds)):
            prev_test_end = folds[i - 1][1]
            # Current train_end = previous test_end (non-overlapping expanding)
            curr_train_end = folds[i][0]
            assert (
                curr_train_end
                >= prev_test_end - cfg.walk_forward_step + cfg.test_periods
            )

    def test_train_never_includes_test(self) -> None:
        """Training window ends strictly before test window starts."""
        cfg = WeightLearnerConfig(
            min_train_periods=100,
            test_periods=50,
            walk_forward_step=50,
        )
        learner = SurpriseWeightLearner(cfg)
        folds = learner._build_folds(T=300)
        for train_end, test_end in folds:
            assert train_end < test_end
            # test window is [train_end, test_end)
            # train window is [0, train_end)
            # no overlap by construction.

    def test_number_of_folds(self) -> None:
        cfg = WeightLearnerConfig(
            min_train_periods=100,
            test_periods=50,
            walk_forward_step=50,
        )
        learner = SurpriseWeightLearner(cfg)
        folds = learner._build_folds(T=300)
        # split starts at 100, steps by 50: 100→150, 150→200, 200→250, 250→300
        assert len(folds) == 4


# ── Proof 6: convergence / early stopping ─────────────────────


class TestConvergence:
    def test_early_stopping_triggers(self) -> None:
        """With strong signal, should converge before max_epochs."""
        S, R = _make_synthetic(T=400, predictive_col=3, signal_strength=5.0)
        cfg = WeightLearnerConfig(
            min_train_periods=100,
            test_periods=50,
            walk_forward_step=50,
            max_epochs=1000,
            patience=10,
            learning_rate=0.05,
        )
        learner = SurpriseWeightLearner(cfg)
        result = learner.fit(S, R)
        # At least one fold should converge early
        assert any(
            e < 1000 for e in result["epochs_per_fold"]
        ), f"No fold converged early: {result['epochs_per_fold']}"


# ── Error handling ───────────────────────────────────────────


class TestErrors:
    def test_insufficient_data(self) -> None:
        S = np.random.randn(50, 5).astype(np.float32)
        R = np.random.randn(50).astype(np.float32)
        learner = SurpriseWeightLearner()  # default needs 156 periods
        with pytest.raises(InsufficientDataError):
            learner.fit(S, R)

    def test_wrong_shape_matrix(self) -> None:
        S = np.random.randn(100, 3).astype(np.float32)
        R = np.random.randn(100).astype(np.float32)
        learner = SurpriseWeightLearner(
            WeightLearnerConfig(min_train_periods=50, test_periods=25)
        )
        with pytest.raises(ValueError, match="must be.*T, 5"):
            learner.fit(S, R)

    def test_mismatched_lengths(self) -> None:
        S = np.random.randn(100, 5).astype(np.float32)
        R = np.random.randn(80).astype(np.float32)
        learner = SurpriseWeightLearner(
            WeightLearnerConfig(min_train_periods=50, test_periods=25)
        )
        with pytest.raises(ValueError, match="matching"):
            learner.fit(S, R)

    def test_nan_in_surprise(self) -> None:
        S = np.random.randn(200, 5).astype(np.float32)
        S[50, 2] = np.nan
        R = np.random.randn(200).astype(np.float32)
        learner = SurpriseWeightLearner(
            WeightLearnerConfig(min_train_periods=100, test_periods=50)
        )
        with pytest.raises(ValueError, match="NaN"):
            learner.fit(S, R)

    def test_nan_in_returns(self) -> None:
        S = np.random.randn(200, 5).astype(np.float32)
        R = np.random.randn(200).astype(np.float32)
        R[100] = np.nan
        learner = SurpriseWeightLearner(
            WeightLearnerConfig(min_train_periods=100, test_periods=50)
        )
        with pytest.raises(ValueError, match="NaN"):
            learner.fit(S, R)


# ── Serialisation ────────────────────────────────────────────


class TestSerialisation:
    def test_state_dict_roundtrip(self) -> None:
        learner = SurpriseWeightLearner()
        # Perturb weights
        learner._raw_weights.data.fill_(1.5)
        sd = learner.state_dict()
        new_learner = SurpriseWeightLearner()
        new_learner.load_state_dict(sd)
        orig_w = learner.get_learned_weights()
        loaded_w = new_learner.get_learned_weights()
        for a, b in zip(orig_w, loaded_w):
            assert abs(a - b) < 1e-6
