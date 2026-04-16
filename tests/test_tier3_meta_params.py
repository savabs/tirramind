"""
Edge-case tests for Tier 3 — Learn Meta-Parameters.

Covers:
1. BayesianParamOptimizer (GP math, EI, boundaries, persistence)
2. RewardWeightOptimizer (weight suggestion, recording, best selection)
3. ThresholdOptimizer (multi-detector, Hawkes sub-criticality, unknown detector)
4. Novel arm discovery (promotion, dedup, persistence, history tracking)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from agent.learning.param_optimizer import (
    BayesianParamOptimizer,
    ParamSpace,
    Trial,
    _expected_improvement,
    _gp_posterior,
    _rbf_kernel,
)
from agent.learning.reward import (
    DEFAULT_WEIGHTS,
    RewardWeightOptimizer,
    RewardWeights,
)
from agent.learning.threshold_optimizer import (
    DETECTOR_SPACES,
    ThresholdOptimizer,
)
from agent.learning.bandit import (
    DEFAULT_ARMS,
    GoalArm,
    StrategyBandit,
    _NOVEL_PROMOTE_MIN_SUCCESSES,
    _NOVEL_PROMOTE_REWARD_THRESHOLD,
)


# =====================================================================
# 1. ParamSpace
# =====================================================================


class TestParamSpace:
    def test_valid_construction(self):
        ps = ParamSpace(names=["a", "b"], bounds=[(0, 1), (0, 2)])
        assert ps.ndim == 2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            ParamSpace(names=["a"], bounds=[(0, 1), (2, 3)])

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError, match="lo=.*>= hi"):
            ParamSpace(names=["a"], bounds=[(5.0, 2.0)])

    def test_equal_bounds_raises(self):
        with pytest.raises(ValueError, match="lo=.*>= hi"):
            ParamSpace(names=["a"], bounds=[(1.0, 1.0)])

    def test_clip(self):
        ps = ParamSpace(names=["a", "b"], bounds=[(0, 1), (0, 2)])
        x = np.array([-0.5, 3.0])
        clipped = ps.clip(x)
        np.testing.assert_array_equal(clipped, [0.0, 2.0])

    def test_sample_uniform_within_bounds(self):
        ps = ParamSpace(names=["a", "b"], bounds=[(0, 1), (10, 20)])
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = ps.sample_uniform(rng)
            assert 0 <= x[0] <= 1
            assert 10 <= x[1] <= 20

    def test_to_dict_and_from_dict_roundtrip(self):
        ps = ParamSpace(names=["x", "y"], bounds=[(0, 1), (0, 1)])
        x = np.array([0.3, 0.7])
        d = ps.to_dict(x)
        assert d == {"x": 0.3, "y": 0.7}
        np.testing.assert_allclose(ps.from_dict(d), x)


# =====================================================================
# 2. GP Math
# =====================================================================


class TestGPMath:
    def test_rbf_kernel_self_similarity(self):
        X = np.array([[0.0], [1.0], [2.0]])
        K = _rbf_kernel(X, X, length_scale=1.0)
        # Diagonal should be 1.0 (signal_var=1)
        np.testing.assert_allclose(np.diag(K), 1.0)
        # Symmetric
        np.testing.assert_allclose(K, K.T)

    def test_rbf_kernel_decay_with_distance(self):
        X1 = np.array([[0.0]])
        X_near = np.array([[0.1]])
        X_far = np.array([[5.0]])
        k_near = _rbf_kernel(X1, X_near, length_scale=1.0)[0, 0]
        k_far = _rbf_kernel(X1, X_far, length_scale=1.0)[0, 0]
        assert k_near > k_far

    def test_gp_posterior_interpolates_training_points(self):
        """GP posterior mean should be close to training y at training X."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([1.0, 0.5, 0.0])
        mu, var = _gp_posterior(X, y, X, length_scale=0.3, noise_var=1e-6)
        np.testing.assert_allclose(mu, y, atol=0.05)
        # Variance at training points should be near zero
        assert np.all(var < 0.1)

    def test_gp_posterior_high_variance_far_from_data(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        X_far = np.array([[10.0]])
        _, var = _gp_posterior(X, y, X_far, length_scale=0.3, noise_var=1e-4)
        # Far from training data → variance should be high (close to prior=1)
        assert var[0] > 0.5

    def test_expected_improvement_zero_at_best(self):
        """EI should be near zero at a point equal to f_best."""
        mu = np.array([1.0])
        var = np.array([0.01])
        ei = _expected_improvement(mu, var, f_best=1.0, xi=0.0)
        assert ei[0] < 0.1

    def test_expected_improvement_high_above_best(self):
        """EI should be high when mu is well above f_best."""
        mu = np.array([2.0])
        var = np.array([0.1])
        ei = _expected_improvement(mu, var, f_best=0.5, xi=0.01)
        assert ei[0] > 1.0

    def test_expected_improvement_zero_variance(self):
        """EI should be 0 when variance is 0 (no uncertainty)."""
        mu = np.array([0.5])
        var = np.array([0.0])
        ei = _expected_improvement(mu, var, f_best=0.4)
        assert ei[0] == 0.0


# =====================================================================
# 3. BayesianParamOptimizer
# =====================================================================


class TestBayesianParamOptimizer:
    @pytest.fixture
    def simple_space(self):
        return ParamSpace(names=["x", "y"], bounds=[(0, 1), (0, 1)])

    def test_suggest_returns_within_bounds(self, simple_space):
        opt = BayesianParamOptimizer(simple_space, seed=42)
        for _ in range(10):
            params = opt.suggest()
            opt.record(params, np.random.uniform())
            assert 0 <= params["x"] <= 1
            assert 0 <= params["y"] <= 1

    def test_first_suggestions_are_random(self, simple_space):
        opt = BayesianParamOptimizer(simple_space, n_random=3, seed=42)
        p1 = opt.suggest()
        opt.record(p1, 0.5)
        p2 = opt.suggest()
        opt.record(p2, 0.3)
        p3 = opt.suggest()
        opt.record(p3, 0.7)
        # After 3 random, next should use EI (just check it doesn't crash)
        p4 = opt.suggest()
        assert "x" in p4 and "y" in p4

    def test_best_returns_highest_objective(self, simple_space):
        opt = BayesianParamOptimizer(simple_space, seed=42)
        opt.record({"x": 0.1, "y": 0.2}, 0.3)
        opt.record({"x": 0.5, "y": 0.6}, 0.9)
        opt.record({"x": 0.8, "y": 0.1}, 0.1)
        best = opt.best()
        assert best.objective == 0.9
        assert best.params == {"x": 0.5, "y": 0.6}

    def test_best_empty_returns_none(self, simple_space):
        opt = BayesianParamOptimizer(simple_space)
        assert opt.best() is None
        assert opt.best_params() is None

    def test_n_trials(self, simple_space):
        opt = BayesianParamOptimizer(simple_space, seed=42)
        assert opt.n_trials == 0
        opt.record({"x": 0.5, "y": 0.5}, 0.5)
        assert opt.n_trials == 1

    def test_persistence_roundtrip(self, simple_space):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "opt.json"
            opt1 = BayesianParamOptimizer(simple_space, persist_path=path, seed=42)
            opt1.record({"x": 0.1, "y": 0.2}, 0.3)
            opt1.record({"x": 0.5, "y": 0.6}, 0.9)

            # Load from disk
            opt2 = BayesianParamOptimizer(simple_space, persist_path=path)
            assert opt2.n_trials == 2
            assert opt2.best().objective == 0.9

    def test_persistence_file_format(self, simple_space):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "opt.json"
            opt = BayesianParamOptimizer(simple_space, persist_path=path, seed=42)
            opt.record({"x": 0.1, "y": 0.2}, 0.5, metadata={"run": 1})

            data = json.loads(path.read_text())
            assert "space" in data
            assert "trials" in data
            assert len(data["trials"]) == 1
            assert data["trials"][0]["objective"] == 0.5
            assert data["trials"][0]["metadata"]["run"] == 1

    def test_record_with_metadata(self, simple_space):
        opt = BayesianParamOptimizer(simple_space, seed=42)
        opt.record({"x": 0.5, "y": 0.5}, 0.7, metadata={"epoch": 10})
        assert opt.trials[0].metadata == {"epoch": 10}

    def test_many_trials_converge_toward_optimum(self):
        """After many trials, suggestions should cluster near the optimum."""
        space = ParamSpace(names=["x"], bounds=[(0, 1)])
        opt = BayesianParamOptimizer(space, n_random=3, seed=42)

        # Objective: maximize -(x - 0.7)^2 + 1 (optimum at x=0.7)
        for _ in range(15):
            params = opt.suggest()
            obj = -((params["x"] - 0.7) ** 2) + 1.0
            opt.record(params, obj)

        best = opt.best()
        assert abs(best.params["x"] - 0.7) < 0.25  # within 0.25 of optimum

    def test_high_dimensional_space(self):
        """5D space should work without crashing."""
        space = ParamSpace(
            names=["a", "b", "c", "d", "e"],
            bounds=[(0, 1)] * 5,
        )
        opt = BayesianParamOptimizer(space, n_random=3, seed=42)
        for _ in range(8):
            p = opt.suggest()
            opt.record(p, np.random.uniform())
        assert opt.n_trials == 8


# =====================================================================
# 4. RewardWeightOptimizer
# =====================================================================


class TestRewardWeightOptimizer:
    def test_suggest_returns_reward_weights(self):
        opt = RewardWeightOptimizer(seed=42)
        w = opt.suggest_weights()
        assert isinstance(w, RewardWeights)
        assert 0.01 <= w.eval_weight <= 1.0
        assert 0.01 <= w.sharpe_weight <= 1.0

    def test_record_and_best(self):
        opt = RewardWeightOptimizer(seed=42)
        w1 = RewardWeights(
            eval_weight=0.3,
            sharpe_weight=0.4,
            facts_weight=0.2,
            novelty_bonus=0.05,
            dead_end_penalty=0.25,
        )
        w2 = RewardWeights(
            eval_weight=0.5,
            sharpe_weight=0.2,
            facts_weight=0.1,
            novelty_bonus=0.15,
            dead_end_penalty=0.35,
        )
        opt.record_trial(w1, 0.8)
        opt.record_trial(w2, 1.2)
        best = opt.current_best()
        assert best is not None
        assert best.eval_weight == w2.eval_weight

    def test_empty_best_returns_none(self):
        opt = RewardWeightOptimizer(seed=42)
        assert opt.current_best() is None

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rw.json"
            opt1 = RewardWeightOptimizer(persist_path=path, seed=42)
            w = opt1.suggest_weights()
            opt1.record_trial(w, 0.5)

            opt2 = RewardWeightOptimizer(persist_path=path)
            assert opt2.n_trials == 1

    def test_n_trials(self):
        opt = RewardWeightOptimizer(seed=42)
        assert opt.n_trials == 0
        opt.record_trial(DEFAULT_WEIGHTS, 0.5)
        assert opt.n_trials == 1


# =====================================================================
# 5. ThresholdOptimizer
# =====================================================================


class TestThresholdOptimizer:
    def test_detector_names(self):
        opt = ThresholdOptimizer(seed=42)
        names = opt.detector_names
        assert "cusum" in names
        assert "hawkes" in names
        assert "convergence" in names

    def test_suggest_cusum_within_bounds(self):
        opt = ThresholdOptimizer(seed=42)
        for _ in range(5):
            params = opt.suggest("cusum")
            assert 0.1 <= params["k"] <= 2.0
            assert 2.0 <= params["h"] <= 10.0
            opt.record("cusum", params, np.random.uniform())

    def test_suggest_hawkes_subcritical(self):
        """Hawkes alpha must be < beta (sub-criticality enforced)."""
        opt = ThresholdOptimizer(seed=42)
        for _ in range(10):
            params = opt.suggest("hawkes")
            assert (
                params["alpha"] < params["beta"]
            ), f"alpha={params['alpha']} >= beta={params['beta']}"
            opt.record("hawkes", params, np.random.uniform())

    def test_suggest_convergence_within_bounds(self):
        opt = ThresholdOptimizer(seed=42)
        params = opt.suggest("convergence")
        assert 1.0 <= params["z_threshold"] <= 4.0
        assert 0.001 <= params["p_threshold"] <= 0.20
        assert 0.01 <= params["fdr_q"] <= 0.20

    def test_unknown_detector_raises(self):
        opt = ThresholdOptimizer(seed=42)
        with pytest.raises(ValueError, match="Unknown detector"):
            opt.suggest("nonexistent")

    def test_record_and_best(self):
        opt = ThresholdOptimizer(seed=42)
        opt.record("cusum", {"k": 0.5, "h": 5.0}, 0.7)
        opt.record("cusum", {"k": 0.8, "h": 4.0}, 0.9)
        best = opt.current_best("cusum")
        assert best == {"k": 0.8, "h": 4.0}

    def test_empty_best_returns_none(self):
        opt = ThresholdOptimizer(seed=42)
        assert opt.current_best("hawkes") is None

    def test_persistence_per_detector(self):
        with tempfile.TemporaryDirectory() as td:
            persist_dir = Path(td)
            opt1 = ThresholdOptimizer(persist_dir=persist_dir, seed=42)
            opt1.record("cusum", {"k": 0.5, "h": 5.0}, 0.7)
            opt1.record("hawkes", {"mu": 0.1, "alpha": 0.3, "beta": 1.0}, 0.8)

            # Verify separate files
            assert (persist_dir / "cusum_bo.json").exists()
            assert (persist_dir / "hawkes_bo.json").exists()

            # Reload and verify
            opt2 = ThresholdOptimizer(persist_dir=persist_dir)
            assert opt2.n_trials("cusum") == 1
            assert opt2.n_trials("hawkes") == 1
            assert opt2.n_trials("convergence") == 0

    def test_n_trials_independent_per_detector(self):
        opt = ThresholdOptimizer(seed=42)
        opt.record("cusum", {"k": 0.5, "h": 5.0}, 0.5)
        opt.record("cusum", {"k": 0.6, "h": 4.0}, 0.6)
        opt.record("hawkes", {"mu": 0.1, "alpha": 0.3, "beta": 1.0}, 0.7)
        assert opt.n_trials("cusum") == 2
        assert opt.n_trials("hawkes") == 1
        assert opt.n_trials("convergence") == 0


# =====================================================================
# 6. Novel Arm Discovery (Change 8)
# =====================================================================


class TestNovelArmDiscovery:
    @pytest.fixture
    def bandit(self):
        """Bandit with just 3 arms + novel for faster tests."""
        arms = [
            GoalArm(name="arm_a", description="A", tools=["t1"]),
            GoalArm(name="arm_b", description="B", tools=["t2"]),
            GoalArm(
                name="novel_exploration", description="Open-ended exploration", tools=[]
            ),
        ]
        return StrategyBandit(arms=arms, seed=42)

    def test_novel_arm_in_default_arms(self):
        """novel_exploration should be in DEFAULT_ARMS."""
        names = [arm.name for arm in DEFAULT_ARMS]
        assert "novel_exploration" in names

    def test_novel_arm_tools_empty(self):
        """novel_exploration should have empty tools (all tools allowed)."""
        arm = next(a for a in DEFAULT_ARMS if a.name == "novel_exploration")
        assert arm.tools == []

    def test_record_novel_pull_low_reward_no_promote(self, bandit):
        result = bandit.record_novel_pull(
            tools_used=["t1", "t3"],
            reward=0.3,
            description="Low reward exploration",
        )
        assert result is None
        assert len(bandit.novel_history) == 1

    def test_record_novel_pull_high_reward_insufficient_count(self, bandit):
        """One success isn't enough for promotion."""
        result = bandit.record_novel_pull(
            tools_used=["t1", "t3"],
            reward=0.8,
            description="Good but first",
        )
        assert result is None

    def test_promotion_after_enough_successes(self, bandit):
        """After N successes with same tools, a new arm should be created."""
        promoted = None
        for i in range(_NOVEL_PROMOTE_MIN_SUCCESSES):
            promoted = bandit.record_novel_pull(
                tools_used=["t3", "t4"],
                reward=0.8,
                description=f"Success {i+1}",
            )
        assert promoted is not None
        assert promoted.name == "t3_t4"
        assert "t3" in promoted.tools
        assert "t4" in promoted.tools
        # The arm should be in the bandit
        assert bandit.get_arm("t3_t4") is not None

    def test_no_duplicate_promotion(self, bandit):
        """Same tool signature shouldn't be promoted twice."""
        for i in range(_NOVEL_PROMOTE_MIN_SUCCESSES):
            bandit.record_novel_pull(
                tools_used=["t5", "t6"],
                reward=0.8,
                description=f"S{i+1}",
            )
        # Now try again — should not create another arm
        result = bandit.record_novel_pull(
            tools_used=["t5", "t6"],
            reward=0.9,
            description="Another success",
        )
        assert result is None  # already exists

    def test_promotion_uses_informative_prior(self, bandit):
        """Promoted arm should start with informative prior, not uniform."""
        for i in range(_NOVEL_PROMOTE_MIN_SUCCESSES):
            bandit.record_novel_pull(
                tools_used=["tnew"],
                reward=0.8,
                description=f"S{i+1}",
            )
        stats = {s.name: s for s in bandit.stats()}
        promoted = stats.get("tnew")
        assert promoted is not None
        # Informative prior: α = 1 + 0.8 = 1.8, β = 1 + 0.2 = 1.2
        assert promoted.alpha > 1.0
        assert promoted.mean_reward > 0.5  # should be skewed toward success

    def test_different_tool_signatures_tracked_separately(self, bandit):
        """Different tool combos should not interfere."""
        for _ in range(2):
            bandit.record_novel_pull(["t1"], 0.9, "sig1")
        for _ in range(2):
            bandit.record_novel_pull(["t2"], 0.9, "sig2")
        # Neither should be promoted (need 3 each)
        assert bandit.get_arm("t1") is None  # "t1" would be the promoted name
        assert bandit.get_arm("t2") is None

    def test_add_arm_with_prior(self, bandit):
        arm = GoalArm(name="new_arm", description="Test", tools=["t1"])
        bandit.add_arm(arm, prior_reward=0.7)
        stats = {s.name: s for s in bandit.stats()}
        s = stats["new_arm"]
        assert s.alpha == pytest.approx(1.7)
        assert s.beta == pytest.approx(1.3)
        assert s.pulls == 0

    def test_add_arm_without_prior(self, bandit):
        arm = GoalArm(name="new_arm", description="Test", tools=["t1"])
        bandit.add_arm(arm)
        stats = {s.name: s for s in bandit.stats()}
        s = stats["new_arm"]
        assert s.alpha == 1.0
        assert s.beta == 1.0

    def test_add_duplicate_arm_ignored(self, bandit):
        arm = GoalArm(name="arm_a", description="Dup", tools=["t1"])
        bandit.add_arm(arm)
        # Original should still be there, not replaced
        assert bandit.get_arm("arm_a").description == "A"

    def test_novel_history_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bandit.json"
            arms = [
                GoalArm(name="a", description="A", tools=["t1"]),
                GoalArm(name="novel_exploration", description="N", tools=[]),
            ]
            b1 = StrategyBandit(arms=arms, persist_path=path, seed=42)
            b1.record_novel_pull(["t1", "t2"], 0.7, "test pull")

            b2 = StrategyBandit(arms=arms, persist_path=path)
            assert len(b2.novel_history) == 1
            assert b2.novel_history[0]["reward"] == 0.7

    def test_promoted_arm_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bandit.json"
            arms = [
                GoalArm(name="a", description="A", tools=["t1"]),
                GoalArm(name="novel_exploration", description="N", tools=[]),
            ]
            b1 = StrategyBandit(arms=arms, persist_path=path, seed=42)
            for i in range(_NOVEL_PROMOTE_MIN_SUCCESSES):
                b1.record_novel_pull(["tx", "ty"], 0.85, f"S{i+1}")

            # The promoted arm is in the JSON
            data = json.loads(path.read_text())
            assert "tx_ty" in data["arms"]

    def test_novel_history_property_is_copy(self, bandit):
        """novel_history property should return a copy, not internal state."""
        bandit.record_novel_pull(["t1"], 0.5, "test")
        history = bandit.novel_history
        history.clear()  # mutate the copy
        assert len(bandit.novel_history) == 1  # original untouched

    def test_choose_can_select_novel_arm(self):
        """With high enough uncertainty, novel arm should sometimes be chosen."""
        arms = [
            GoalArm(name="saturated", description="S", tools=["t1"]),
            GoalArm(name="novel_exploration", description="N", tools=[]),
        ]
        bandit = StrategyBandit(arms=arms, seed=42)
        # Pull saturated many times to reduce its uncertainty
        for _ in range(50):
            bandit.update("saturated", 0.5)

        # Now novel_exploration has uniform Beta(1,1) — high uncertainty
        # Over many draws, it should be chosen at least once
        chosen = set()
        for _ in range(100):
            chosen.add(bandit.choose().name)
        assert "novel_exploration" in chosen


# =====================================================================
# 7. Integration: GP-BO end-to-end with detector spaces
# =====================================================================


class TestDetectorSpaces:
    """Verify the pre-defined detector spaces are well-formed."""

    def test_all_spaces_valid(self):
        for name, space in DETECTOR_SPACES.items():
            assert space.ndim >= 2, f"{name} should have ≥2 dims"
            for pname, (lo, hi) in zip(space.names, space.bounds):
                assert lo < hi, f"{name}.{pname}: lo={lo} >= hi={hi}"

    def test_hawkes_bounds_allow_subcritical(self):
        """Alpha upper bound should be < beta upper bound."""
        space = DETECTOR_SPACES["hawkes"]
        a_bounds = dict(zip(space.names, space.bounds))
        # alpha max (0.95) < beta max (3.0) — sub-criticality is possible
        assert a_bounds["alpha"][1] < a_bounds["beta"][1]

    def test_convergence_p_below_z(self):
        """p_threshold upper bound should be reasonable (< 0.5)."""
        space = DETECTOR_SPACES["convergence"]
        p_bounds = dict(zip(space.names, space.bounds))
        assert p_bounds["p_threshold"][1] <= 0.5
