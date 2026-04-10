"""Tests for depth evaluation module (MI, KL divergence, integration loop)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from agent.pipeline.depth_eval import (
    compute_conditional_mi,
    compute_kl_divergence,
    measure_belief_shift,
    run_depth_evaluation,
)
from agent.pipeline.store import PipelineStore


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def store():
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


# ── compute_conditional_mi ─────────────────────────────────────


class TestConditionalMI:
    def test_independent_variables(self):
        """MI between independent random variables should be near 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        mi = compute_conditional_mi(x, None, y)
        # Should be close to 0 (with noise, but <0.1)
        assert mi < 0.1

    def test_correlated_variables(self):
        """MI between correlated variables should be positive."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = x + rng.normal(0, 0.3, 200)  # y ≈ x + noise
        mi = compute_conditional_mi(x, None, y)
        assert mi > 0.1

    def test_conditional_mi_positive_gain(self):
        """New observations correlated with target beyond existing data."""
        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(0, 1, n)  # hidden cause
        x_existing = z + rng.normal(0, 1, n)  # noisy z
        x_new = z + rng.normal(0, 0.3, n)  # less noisy z
        target = z + rng.normal(0, 0.1, n)  # very clean z

        cmi = compute_conditional_mi(x_new, x_existing, target)
        # x_new adds info about target beyond what x_existing provides
        assert cmi > 0.0

    def test_conditional_mi_redundant(self):
        """New observations add nothing beyond existing data."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        target = rng.normal(0, 1, n)  # independent

        cmi = compute_conditional_mi(x, x, target)
        # Duplicate feature should add ~0 MI
        assert cmi < 0.1

    def test_insufficient_samples(self):
        """Should return NaN with fewer than 30 samples."""
        x = np.ones(10)
        y = np.ones(10)
        mi = compute_conditional_mi(x, None, y)
        assert math.isnan(mi)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="samples"):
            compute_conditional_mi(np.ones(50), None, np.ones(60))

    def test_mismatched_existing_lengths_raises(self):
        with pytest.raises(ValueError, match="samples"):
            compute_conditional_mi(
                np.ones(50),
                np.ones(40),
                np.ones(50),
            )

    def test_non_finite_filtered(self):
        """NaN/Inf values should be filtered, not crash."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        x[0] = np.nan
        x[1] = np.inf
        y[2] = -np.inf
        mi = compute_conditional_mi(x, None, y)
        # Should work without crash, result may be NaN if too few remain
        assert isinstance(mi, float)

    def test_all_nan_returns_nan(self):
        """All NaN data should return NaN."""
        x = np.full(50, np.nan)
        y = np.full(50, np.nan)
        mi = compute_conditional_mi(x, None, y)
        assert math.isnan(mi)

    def test_discrete_target(self):
        """Test with discrete target classification."""
        rng = np.random.default_rng(42)
        n = 200
        # Create features that predict the class
        target = rng.choice([0, 1, 2], size=n)
        x = target + rng.normal(0, 0.5, n)  # feature correlated with class

        mi = compute_conditional_mi(x, None, target, discrete_target=True)
        assert mi > 0.1

    def test_2d_observations(self):
        """Multi-feature observations should work."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, (n, 3))
        y = x[:, 0] + x[:, 1] + rng.normal(0, 0.1, n)

        mi = compute_conditional_mi(x, None, y)
        assert mi > 0.1


# ── compute_kl_divergence ──────────────────────────────────────


class TestKLDivergence:
    def test_identical_distributions(self):
        p = {"a": 0.5, "b": 0.3, "c": 0.2}
        kl = compute_kl_divergence(p, p)
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_shifted_distribution(self):
        prior = {"a": 0.5, "b": 0.3, "c": 0.2}
        posterior = {"a": 0.8, "b": 0.1, "c": 0.1}
        kl = compute_kl_divergence(prior, posterior)
        assert kl > 0.0

    def test_symmetry_broken(self):
        """KL divergence is not symmetric."""
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.9, "b": 0.1}
        kl_pq = compute_kl_divergence(p, q)
        kl_qp = compute_kl_divergence(q, p)
        assert kl_pq != pytest.approx(kl_qp)

    def test_different_keys_raises(self):
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.5, "c": 0.5}
        with pytest.raises(ValueError, match="keys differ"):
            compute_kl_divergence(p, q)

    def test_non_normalized_prior_raises(self):
        p = {"a": 0.5, "b": 0.3}  # sums to 0.8
        q = {"a": 0.5, "b": 0.5}
        with pytest.raises(ValueError, match="sum to"):
            compute_kl_divergence(p, q)

    def test_non_normalized_posterior_raises(self):
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.9, "b": 0.3}  # sums to 1.2
        with pytest.raises(ValueError, match="sum to"):
            compute_kl_divergence(p, q)

    def test_zero_probability_handled(self):
        """0 * log(0/q) = 0 by convention (scipy handles this)."""
        prior = {"a": 0.5, "b": 0.5}
        posterior = {"a": 1.0, "b": 0.0}
        kl = compute_kl_divergence(prior, posterior)
        assert kl >= 0.0
        assert math.isfinite(kl)

    def test_known_value(self):
        """Test against manually computed KL divergence."""
        prior = {"a": 0.5, "b": 0.5}
        posterior = {"a": 0.75, "b": 0.25}
        kl = compute_kl_divergence(prior, posterior)
        # Manual: 0.75*ln(0.75/0.5) + 0.25*ln(0.25/0.5)
        expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
        assert kl == pytest.approx(expected, rel=1e-6)


# ── measure_belief_shift ───────────────────────────────────────


class TestMeasureBeliefShift:
    def _store_belief(self, store, var_name, version, probs):
        """Helper to store a belief with probabilities."""
        from agent.models.belief import BeliefState

        belief = BeliefState(
            variable_name=var_name,
            version=version,
            effective_at=1700000000.0 + version,
            computed_at=1700000000.0 + version,
            dist_type="categorical",
            mean=None,
            variance=None,
            probabilities=probs,
            evidence_count=10,
            model_graph_hash="a" * 64,
            confidence=0.9,
            stale=False,
        )
        store.store_belief(belief)

    def test_basic_shift(self, store: PipelineStore):
        self._store_belief(store, "state.regime", 1, {"bull": 0.5, "bear": 0.5})
        self._store_belief(store, "state.regime", 2, {"bull": 0.8, "bear": 0.2})

        kl = measure_belief_shift(store, "state.regime", 1, 2)
        assert kl is not None
        assert kl > 0.0

    def test_no_shift(self, store: PipelineStore):
        self._store_belief(store, "state.regime", 1, {"bull": 0.5, "bear": 0.5})
        self._store_belief(store, "state.regime", 2, {"bull": 0.5, "bear": 0.5})

        kl = measure_belief_shift(store, "state.regime", 1, 2)
        assert kl is not None
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_missing_before(self, store: PipelineStore):
        self._store_belief(store, "state.regime", 2, {"bull": 0.5, "bear": 0.5})
        result = measure_belief_shift(store, "state.regime", 1, 2)
        assert result is None

    def test_missing_after(self, store: PipelineStore):
        self._store_belief(store, "state.regime", 1, {"bull": 0.5, "bear": 0.5})
        result = measure_belief_shift(store, "state.regime", 1, 2)
        assert result is None

    def test_non_discrete_belief(self, store: PipelineStore):
        """Beliefs without probabilities should return None."""
        from agent.models.belief import BeliefState

        for v in (1, 2):
            belief = BeliefState(
                variable_name="state.continuous",
                version=v,
                effective_at=1700000000.0 + v,
                computed_at=1700000000.0 + v,
                dist_type="gaussian",
                mean=0.5,
                variance=0.1,
                probabilities=None,
                evidence_count=10,
                model_graph_hash="a" * 64,
                confidence=0.9,
                stale=False,
            )
            store.store_belief(belief)

        result = measure_belief_shift(store, "state.continuous", 1, 2)
        assert result is None


# ── run_depth_evaluation (integration) ─────────────────────────


class TestRunDepthEvaluation:
    def test_full_loop(self, store: PipelineStore):
        """End-to-end: entity → observations → MI → depth_evaluation record."""
        from agent.pipeline.entity import entity_id_from_key

        # 1. Register entity
        eid = entity_id_from_key("company", "320193")
        store.register_entity("company", "apple", eid)
        store.add_entity_alias(eid, "sec_cik", "320193")

        # 2. Simulate L1 and L2 observations
        rng = np.random.default_rng(42)
        n = 200
        # Hidden signal (e.g., insider conviction)
        signal = rng.normal(0, 1, n)
        # L1: aggregate filing count (noisy)
        l1_obs = signal + rng.normal(0, 2, n)
        # L2: per-insider filing details (cleaner)
        l2_obs = signal + rng.normal(0, 0.5, n)
        # Target: next-day return
        target = signal + rng.normal(0, 0.3, n)

        # Store observations
        for i in range(n):
            store.store_entity_observation(
                eid,
                "insider_filings",
                1700000000.0 + i * 3600,
                "filing",
                {"value": float(l1_obs[i])},
                depth_level=1,
            )
            store.store_entity_observation(
                eid,
                "insider_filings",
                1700000000.0 + i * 3600,
                "filing_detail",
                {"value": float(l2_obs[i])},
                depth_level=2,
            )

        # 3. Run depth evaluation
        result = run_depth_evaluation(
            store=store,
            tool_name="insider_filings",
            depth_level=2,
            target_variable="equity_return",
            observations_new=l2_obs,
            targets=target,
            observations_existing=l1_obs,
        )

        # 4. Verify result
        assert result["sample_size"] == 200
        assert result["mi_gain"] is not None
        assert result["mi_gain"] >= 0.0
        assert result["row_id"] is not None

        # 5. Verify stored in DB
        evals = store.query_depth_evaluations("insider_filings")
        assert len(evals) == 1
        assert evals[0]["depth_level"] == 2
        assert evals[0]["target_variable"] == "equity_return"

    def test_with_belief_shift(self, store: PipelineStore):
        """Integration test including KL divergence measurement."""
        from agent.models.belief import BeliefState

        # Store before/after beliefs
        for v, probs in [
            (1, {"bull": 0.5, "bear": 0.5}),
            (2, {"bull": 0.7, "bear": 0.3}),
        ]:
            belief = BeliefState(
                variable_name="state.regime",
                version=v,
                effective_at=1700000000.0 + v,
                computed_at=1700000000.0 + v,
                dist_type="categorical",
                mean=None,
                variance=None,
                probabilities=probs,
                evidence_count=10,
                model_graph_hash="a" * 64,
                confidence=0.9,
                stale=False,
            )
            store.store_belief(belief)

        rng = np.random.default_rng(42)
        n = 100
        obs = rng.normal(0, 1, n)
        target = rng.normal(0, 1, n)

        result = run_depth_evaluation(
            store=store,
            tool_name="insider_filings",
            depth_level=2,
            target_variable="state.regime",
            observations_new=obs,
            targets=target,
            belief_before_version=1,
            belief_after_version=2,
        )

        assert result["kl_divergence"] is not None
        assert result["kl_divergence"] > 0.0

    def test_insufficient_samples_stores_null_mi(self, store: PipelineStore):
        """With <30 samples, MI should be stored as None."""
        obs = np.ones(10)
        target = np.ones(10)

        result = run_depth_evaluation(
            store=store,
            tool_name="gdelt",
            depth_level=2,
            target_variable="regime",
            observations_new=obs,
            targets=target,
        )

        # MI is NaN → stored as None
        assert math.isnan(result["mi_gain"])

        evals = store.query_depth_evaluations("gdelt")
        assert evals[0]["mi_gain"] is None
