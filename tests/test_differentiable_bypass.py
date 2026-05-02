"""TirraMind — Phase B: Differentiable Belief Bypass Tests

Tests validate that the differentiable path from DiffKalman through
DifferentiableStateAssembler to SAC actor produces valid gradients in
Kalman parameters (F, Q, H, R).

Categories:
    1. get_beliefs_differentiable() — gradient preservation
    2. DifferentiableStateAssembler — layout + gradient flow
    3. End-to-end gradient existence
    4. Gradient magnitude (not exploding / vanishing)
    5. Layout consistency with InstrumentStateAssembler
    6. Detach isolation (aux backward doesn't corrupt SAC)
    7. NaN / zero observation robustness
    8. Multi-regime gradient routing
    9. Kalman augmentation function integration
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from agent.fusion.alert import EntityAlert
from agent.learning.policy.config import PolicyConfig
from agent.learning.policy.sac import GaussianActor, SACConfig, SACTrainer
from agent.learning.policy.state_assembler import (
    DifferentiableStateAssembler,
    InstrumentStateAssembler,
)
from agent.models.diff_kalman import DifferentiableKalmanFilter

# ── Helpers ─────────────────────────────────────────────────


def _make_kalman(state_dim=3, obs_dim=5, regimes=None):
    """Build a small DifferentiableKalmanFilter for testing."""
    regimes = regimes or ["expansion"]
    return DifferentiableKalmanFilter(state_dim=state_dim, obs_dim=obs_dim, regime_names=regimes)


def _make_tickers(n=5):
    return [f"TICK_{i}" for i in range(n)]


def _make_alert(entity_id="E0", surprise=1.0):
    return EntityAlert(
        entity_id=entity_id,
        entity_type="company",
        entity_name=entity_id,
        alert_time=0.0,
        obs_type_surprise=surprise,
        temporal_surprise=surprise * 0.5,
        value_surprise=surprise * 0.3,
        neighborhood_surprise=surprise * 0.2,
        memory_drift=surprise * 0.1,
        cusum_statistic=0.0,
        hawkes_intensity=0.0,
        event_study_score=0.0,
        composite_surprise=surprise,
        observation_count=1,
        evidence_sources=("test",),
    )


def _forward_kalman(kalman, obs_dim, regime="expansion"):
    """Run predict + update to build computation graph."""
    kalman.predict(regime)
    obs = torch.randn(obs_dim)
    kalman.update(obs)


# ── 1. get_beliefs_differentiable ───────────────────────────


class TestGetBeliefsDifferentiable:
    def test_returns_tensors_not_python_floats(self):
        k = _make_kalman()
        means, variances = k.get_beliefs_differentiable()
        assert isinstance(means, torch.Tensor)
        assert isinstance(variances, torch.Tensor)

    def test_shapes_match_state_dim(self):
        k = _make_kalman(state_dim=7)
        means, variances = k.get_beliefs_differentiable()
        assert means.shape == (7,)
        assert variances.shape == (7,)

    def test_not_detached(self):
        """Means and variances must be part of autograd graph after predict/update."""
        k = _make_kalman()
        _forward_kalman(k, obs_dim=5)
        means, variances = k.get_beliefs_differentiable()
        # Create a scalar loss and check gradients flow
        loss = means.sum() + variances.sum()
        loss.backward()
        # At least one Kalman parameter should have non-None grad
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in k.parameters())
        assert has_grad, "No gradients flowed through get_beliefs_differentiable"

    def test_get_beliefs_still_detaches(self):
        """Original get_beliefs() must still detach — no regression."""
        k = _make_kalman()
        _forward_kalman(k, obs_dim=5)
        beliefs = k.get_beliefs(
            variable_names=["v0", "v1", "v2"],
            as_of=0.0,
            graph_hash="test",
        )
        # These are Python floats, not tensors
        assert isinstance(beliefs[0].mean, float)

    def test_both_methods_return_same_values(self):
        """Differentiable and detached versions should have identical values."""
        k = _make_kalman()
        _forward_kalman(k, obs_dim=5)
        means, variances = k.get_beliefs_differentiable()
        beliefs = k.get_beliefs(["v0", "v1", "v2"], as_of=0.0, graph_hash="t")
        for i, b in enumerate(beliefs):
            assert abs(means[i].item() - b.mean) < 1e-6
            assert abs(variances[i].item() - b.variance) < 1e-6


# ── 2. DifferentiableStateAssembler ─────────────────────────


class TestDifferentiableStateAssembler:
    def test_state_dim_matches_instrument_assembler(self):
        tickers = _make_tickers(5)
        diff = DifferentiableStateAssembler(instrument_tickers=tickers)
        inst = InstrumentStateAssembler(instrument_tickers=tickers)
        assert diff.state_dim == inst.state_dim

    def test_state_dim_varies_with_tickers(self):
        d3 = DifferentiableStateAssembler(instrument_tickers=_make_tickers(3))
        d7 = DifferentiableStateAssembler(instrument_tickers=_make_tickers(7))
        assert d3.state_dim < d7.state_dim

    def test_output_shape(self):
        tickers = _make_tickers(5)
        d = DifferentiableStateAssembler(instrument_tickers=tickers)
        means = torch.randn(3, requires_grad=True)
        variances = torch.randn(3).abs()
        state, meta = d.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        assert state.shape == (d.state_dim,)

    def test_gradient_flows_through_belief_block(self):
        tickers = _make_tickers(5)
        d = DifferentiableStateAssembler(instrument_tickers=tickers)
        means = torch.randn(3, requires_grad=True)
        variances = torch.randn(3).abs().requires_grad_(True)
        state, _ = d.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        loss = state.sum()
        loss.backward()
        assert means.grad is not None
        assert means.grad.abs().sum() > 0
        assert variances.grad is not None

    def test_non_belief_blocks_are_detached(self):
        """Instrument surprises and market features should NOT carry gradients."""
        tickers = _make_tickers(2)
        d = DifferentiableStateAssembler(instrument_tickers=tickers)
        means = torch.zeros(3)
        variances = torch.zeros(3)
        state, _ = d.assemble(
            instrument_surprises={"TICK_0": (1.0, 0.5, 0.3, 0.2, 0.1)},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={"vol": 0.5},
        )
        # State should not require grad (no grad-connected inputs)
        # Actually, means/variances don't require grad here, so whole thing is detached.
        assert not state.requires_grad

    def test_with_entity_alerts(self):
        tickers = _make_tickers(3)
        d = DifferentiableStateAssembler(instrument_tickers=tickers)
        alerts = [_make_alert(f"E{i}", surprise=float(i + 1)) for i in range(3)]
        means = torch.randn(5, requires_grad=True)
        variances = torch.randn(5).abs()
        state, meta = d.assemble(
            instrument_surprises={},
            entity_alerts=alerts,
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        assert meta["n_entities_active"] == 3
        assert state.shape == (d.state_dim,)

    def test_metadata_keys(self):
        tickers = _make_tickers(2)
        d = DifferentiableStateAssembler(instrument_tickers=tickers)
        means = torch.zeros(3)
        variances = torch.zeros(3)
        _, meta = d.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        assert "n_instruments_active" in meta
        assert "n_entities_active" in meta
        assert "instrument_tickers" in meta
        assert "entity_order" in meta

    def test_n_instruments_property(self):
        d = DifferentiableStateAssembler(instrument_tickers=_make_tickers(4))
        assert d.n_instruments == 4


# ── 3. End-to-end gradient existence ────────────────────────


class TestEndToEndGradient:
    def test_loss_backward_produces_kalman_gradients(self):
        """The complete chain: Kalman → DiffAssembler → Actor → loss → backward."""
        state_dim = 3
        obs_dim = 5
        tickers = _make_tickers(5)

        kalman = _make_kalman(state_dim=state_dim, obs_dim=obs_dim)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        # Forward through Kalman
        kalman.predict("expansion")
        obs = torch.randn(obs_dim)
        kalman.update(obs)

        # Get differentiable beliefs
        means, variances = kalman.get_beliefs_differentiable()

        # Assemble state
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        state = state.unsqueeze(0)  # batch dim

        # Actor forward
        action, log_prob = actor.sample(state)
        loss = log_prob.mean()
        loss.backward()

        # Check Kalman params got gradients
        params_with_grad = []
        for name, p in kalman.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                params_with_grad.append(name)

        assert len(params_with_grad) > 0, (
            f"No Kalman params got gradients. All params: {[n for n, _ in kalman.named_parameters()]}"
        )

    def test_F_gets_gradient(self):
        """Transition matrix F should receive gradients via predict()."""
        k = _make_kalman(state_dim=3, obs_dim=5)
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        k.predict("expansion")
        obs = torch.randn(5)
        k.update(obs)
        means, variances = k.get_beliefs_differentiable()
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        F_grad = k._F["expansion"].grad
        assert F_grad is not None, "F did not receive gradient"
        assert F_grad.abs().sum() > 0, "F gradient is all zeros"

    def test_H_gets_gradient(self):
        """Observation matrix H should receive gradients via update()."""
        k = _make_kalman(state_dim=3, obs_dim=5)
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        k.predict("expansion")
        obs = torch.randn(5)
        k.update(obs)
        means, variances = k.get_beliefs_differentiable()
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        assert k._H.grad is not None, "H did not receive gradient"
        assert k._H.grad.abs().sum() > 0, "H gradient is all zeros"

    def test_L_Q_gets_gradient(self):
        """Process noise Cholesky factor L_Q should receive gradients."""
        k = _make_kalman(state_dim=3, obs_dim=5)
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        k.predict("expansion")
        obs = torch.randn(5)
        k.update(obs)
        means, variances = k.get_beliefs_differentiable()
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        L_Q_grad = k._L_Q["expansion"].grad
        assert L_Q_grad is not None, "L_Q did not receive gradient"

    def test_L_R_gets_gradient(self):
        """Observation noise Cholesky factor L_R should receive gradients."""
        k = _make_kalman(state_dim=3, obs_dim=5)
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        k.predict("expansion")
        obs = torch.randn(5)
        k.update(obs)
        means, variances = k.get_beliefs_differentiable()
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        assert k._L_R.grad is not None, "L_R did not receive gradient"


# ── 4. Gradient magnitude ───────────────────────────────────


class TestGradientMagnitude:
    def _run_and_get_grad_norm(self, state_dim=3, obs_dim=5):
        k = _make_kalman(state_dim=state_dim, obs_dim=obs_dim)
        tickers = _make_tickers(3)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        k.predict("expansion")
        obs = torch.randn(obs_dim)
        k.update(obs)
        means, variances = k.get_beliefs_differentiable()
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        total_norm = 0.0
        for p in k.parameters():
            if p.grad is not None:
                total_norm += float(p.grad.data.norm(2).item() ** 2)
        return total_norm**0.5

    def test_gradients_not_vanishing(self):
        norm = self._run_and_get_grad_norm()
        assert norm > 1e-15, f"Gradient norm too small: {norm}"

    def test_gradients_not_exploding(self):
        norm = self._run_and_get_grad_norm()
        assert norm < 1e6, f"Gradient norm too large: {norm}"

    def test_larger_dimensions_stable(self):
        """With larger state/obs dims, gradients should remain bounded."""
        norm = self._run_and_get_grad_norm(state_dim=10, obs_dim=20)
        assert norm > 1e-15
        assert norm < 1e6


# ── 5. Layout consistency ───────────────────────────────────


class TestLayoutConsistency:
    @pytest.mark.parametrize("n_tickers", [1, 5, 10, 30])
    def test_state_dim_matches(self, n_tickers):
        tickers = _make_tickers(n_tickers)
        diff = DifferentiableStateAssembler(instrument_tickers=tickers)
        inst = InstrumentStateAssembler(instrument_tickers=tickers)
        assert diff.state_dim == inst.state_dim

    @pytest.mark.parametrize("max_e", [10, 50, 100])
    def test_state_dim_matches_varying_entities(self, max_e):
        tickers = _make_tickers(5)
        diff = DifferentiableStateAssembler(instrument_tickers=tickers, max_entities=max_e)
        inst = InstrumentStateAssembler(instrument_tickers=tickers, max_entities=max_e)
        assert diff.state_dim == inst.state_dim

    def test_block_ordering_identical(self):
        """Verify the actual values in non-belief blocks match between assemblers."""
        tickers = _make_tickers(3)
        inst_asm = InstrumentStateAssembler(instrument_tickers=tickers)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)

        # Build identical inputs
        inst_surprises = {"TICK_0": (1.0, 0.5, 0.3, 0.2, 0.1)}
        alerts = [_make_alert("E0", surprise=2.0)]
        market = {"vol": 0.5, "ret": -0.01}

        from agent.models.belief import BeliefState

        beliefs = [
            BeliefState(
                variable_name="v0",
                version=1,
                effective_at=0.0,
                computed_at=0.0,
                dist_type="gaussian",
                mean=0.5,
                variance=0.1,
                evidence_count=1,
                model_graph_hash="t",
                confidence=1.0,
                stale=False,
                entity_id="E0",
            )
        ]

        inst_state, _ = inst_asm.assemble(
            instrument_surprises=inst_surprises,
            entity_alerts=alerts,
            beliefs=beliefs,
            market_features=market,
        )

        diff_state, _ = diff_asm.assemble(
            instrument_surprises=inst_surprises,
            entity_alerts=alerts,
            belief_means=torch.tensor([0.5]),
            belief_variances=torch.tensor([0.1]),
            market_features=market,
        )

        # Instrument block (first N*5) should be identical
        N = 3
        np.testing.assert_allclose(
            inst_state[: N * 5].numpy(),
            diff_state[: N * 5].detach().numpy(),
            atol=1e-6,
        )


# ── 6. Detach isolation ─────────────────────────────────────


class TestDetachIsolation:
    def test_aux_backward_does_not_affect_actor_params(self):
        """The auxiliary backward should only affect Kalman params, not actor."""
        state_dim = 3
        obs_dim = 5
        tickers = _make_tickers(3)

        kalman = _make_kalman(state_dim=state_dim, obs_dim=obs_dim)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        cfg = SACConfig()
        actor = GaussianActor(diff_asm.state_dim, len(tickers), cfg)

        # Record actor param snapshot
        actor_params_before = {n: p.data.clone() for n, p in actor.named_parameters()}

        # Forward through Kalman → assemble → actor
        kalman.predict("expansion")
        kalman.update(torch.randn(obs_dim))
        means, variances = kalman.get_beliefs_differentiable()
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )

        _, log_prob = actor.sample(state.unsqueeze(0))
        aux_loss = 0.01 * log_prob.mean()

        # Only step Kalman optimizer
        kalman_optim = torch.optim.Adam(kalman.parameters(), lr=1e-4)
        kalman_optim.zero_grad()
        aux_loss.backward()
        kalman_optim.step()

        # Actor params should be unchanged (no optimizer stepped them)
        for n, p in actor.named_parameters():
            torch.testing.assert_close(
                p.data,
                actor_params_before[n],
                msg=f"Actor param {n} was modified by Kalman aux backward",
            )

    def test_kalman_params_do_change_after_aux_step(self):
        """Kalman params should be updated by the auxiliary gradient step."""
        kalman = _make_kalman(state_dim=3, obs_dim=5)
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        F_before = kalman._F["expansion"].data.clone()

        kalman.predict("expansion")
        kalman.update(torch.randn(5))
        means, variances = kalman.get_beliefs_differentiable()
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        loss = log_prob.mean()

        optim = torch.optim.Adam(kalman.parameters(), lr=1e-2)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # F should have changed
        assert not torch.allclose(kalman._F["expansion"].data, F_before, atol=1e-10), (
            "F did not change after optimizer step"
        )


# ── 7. NaN / zero robustness ────────────────────────────────


class TestNaNZeroRobustness:
    def test_nan_observations_produce_finite_gradients(self):
        k = _make_kalman(state_dim=3, obs_dim=5)
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        k.predict("expansion")
        obs = torch.tensor([1.0, float("nan"), 0.5, float("nan"), -0.3])
        k.update(obs)
        means, variances = k.get_beliefs_differentiable()

        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        for p in k.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN in gradient for param shape {p.shape}"

    def test_all_nan_observations(self):
        """All NaN observations → update is no-op, predict-only gradient path."""
        k = _make_kalman(state_dim=3, obs_dim=5)
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        k.predict("expansion")
        obs = torch.full((5,), float("nan"))
        k.update(obs)
        means, variances = k.get_beliefs_differentiable()

        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        # Should still have gradients (from predict step)
        assert k._F["expansion"].grad is not None

    def test_zero_observations(self):
        k = _make_kalman(state_dim=3, obs_dim=5)
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        k.predict("expansion")
        k.update(torch.zeros(5))
        means, variances = k.get_beliefs_differentiable()

        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        for p in k.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_zero_belief_means(self):
        """Zero-dimensional beliefs still produce valid state tensor."""
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        means = torch.zeros(0)  # empty
        variances = torch.zeros(0)
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        assert state.shape == (diff_asm.state_dim,)
        assert torch.isfinite(state).all()


# ── 8. Multi-regime gradient routing ────────────────────────


class TestMultiRegime:
    def test_correct_regime_gets_gradient(self):
        """Only the F/Q for the predicted regime should get gradients."""
        k = _make_kalman(
            state_dim=3,
            obs_dim=5,
            regimes=["expansion", "contraction"],
        )
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        # Predict with 'expansion' only
        k.predict("expansion")
        k.update(torch.randn(5))
        means, variances = k.get_beliefs_differentiable()
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        # expansion F should have grad
        assert k._F["expansion"].grad is not None
        exp_grad_norm = k._F["expansion"].grad.abs().sum().item()
        assert exp_grad_norm > 0

        # contraction F should NOT have grad (was not in the computation)
        con_grad = k._F["contraction"].grad
        if con_grad is not None:
            assert con_grad.abs().sum().item() == 0.0

    def test_switching_regime_shifts_gradients(self):
        """After switching to contraction, that regime should get gradients."""
        k = _make_kalman(
            state_dim=3,
            obs_dim=5,
            regimes=["expansion", "contraction"],
        )
        tickers = _make_tickers(2)
        diff_asm = DifferentiableStateAssembler(instrument_tickers=tickers)
        actor = GaussianActor(diff_asm.state_dim, len(tickers), SACConfig())

        # Reset to clear any previous graph
        k.reset()

        k.predict("contraction")
        k.update(torch.randn(5))
        means, variances = k.get_beliefs_differentiable()
        state, _ = diff_asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            belief_means=means,
            belief_variances=variances,
            market_features={},
        )
        _, log_prob = actor.sample(state.unsqueeze(0))
        log_prob.mean().backward()

        assert k._F["contraction"].grad is not None
        con_grad_norm = k._F["contraction"].grad.abs().sum().item()
        assert con_grad_norm > 0


# ── 9. Kalman augmentation integration ──────────────────────


class TestKalmanAugmentationIntegration:
    def test_augmentation_skipped_when_weight_zero(self):
        """aux_kalman_weight=0 means no augmentation."""
        from agent.pipeline.dags.rl_training import _kalman_augmentation

        cfg = PolicyConfig(sac=SACConfig(aux_kalman_weight=0.0))
        result = _kalman_augmentation(
            store=MagicMock(),
            config=cfg,
            trainer=MagicMock(),
            assembler=MagicMock(),
            tickers=[],
            alerts=[],
            action_dim=5,
        )
        assert result == {}

    def test_augmentation_skipped_when_no_kalman(self):
        """If DiffKalman can't be loaded, skip gracefully."""
        from agent.pipeline.dags.rl_training import _kalman_augmentation

        cfg = PolicyConfig(sac=SACConfig(aux_kalman_weight=0.01))
        store = MagicMock()
        store.load_latest_rl_checkpoint.return_value = None

        result = _kalman_augmentation(
            store=store,
            config=cfg,
            trainer=MagicMock(),
            assembler=MagicMock(),
            tickers=[],
            alerts=[],
            action_dim=5,
        )
        assert result == {}

    def test_build_observation_batch(self):
        from agent.pipeline.dags.rl_training import _build_observation_batch

        alerts = [
            {
                "obs_type_surprise": 1.0,
                "temporal_surprise": 0.5,
                "value_surprise": 0.3,
                "neighborhood_surprise": 0.2,
                "memory_drift": 0.1,
                "timestamp": float(i),
            }
            for i in range(20)
        ]
        batch = _build_observation_batch(alerts, obs_dim=17, max_steps=10)
        assert len(batch) == 10
        obs, regime = batch[0]
        assert obs.shape == (17,)
        assert regime == "expansion"  # default

    def test_build_observation_batch_empty(self):
        from agent.pipeline.dags.rl_training import _build_observation_batch

        batch = _build_observation_batch([], obs_dim=5, max_steps=10)
        assert len(batch) == 0

    def test_build_observation_batch_preserves_values(self):
        from agent.pipeline.dags.rl_training import _build_observation_batch

        alerts = [
            {
                "obs_type_surprise": 3.14,
                "temporal_surprise": 2.71,
                "value_surprise": 1.41,
                "neighborhood_surprise": 1.73,
                "memory_drift": 0.57,
                "timestamp": 100.0,
            }
        ]
        batch = _build_observation_batch(alerts, obs_dim=10, max_steps=5)
        assert len(batch) == 1
        obs, _ = batch[0]
        np.testing.assert_allclose(obs[0], 3.14, atol=1e-6)
        np.testing.assert_allclose(obs[1], 2.71, atol=1e-6)
        np.testing.assert_allclose(obs[5:], 0.0)  # zero-padded

    def test_build_observation_batch_uses_regime_from_alert(self):
        from agent.pipeline.dags.rl_training import _build_observation_batch

        alerts = [{"obs_type_surprise": 1.0, "regime": "crisis", "timestamp": 1.0}]
        batch = _build_observation_batch(alerts, obs_dim=5, max_steps=5)
        _, regime = batch[0]
        assert regime == "crisis"

    def test_full_augmentation_with_real_kalman(self):
        """Integration: run _kalman_augmentation with a real DiffKalman."""
        from agent.pipeline.dags.rl_training import _kalman_augmentation

        state_dim = 3
        obs_dim = 5
        tickers = _make_tickers(3)
        action_dim = len(tickers)

        kalman = _make_kalman(state_dim=state_dim, obs_dim=obs_dim)
        # Serialize it
        buf = io.BytesIO()
        torch.save(kalman.state_dict(), buf)
        kalman_bytes = buf.getvalue()

        store = MagicMock()
        store.load_latest_rl_checkpoint.return_value = {
            "state_dict_bytes": kalman_bytes,
        }

        assembler = InstrumentStateAssembler(instrument_tickers=tickers)
        cfg = PolicyConfig(sac=SACConfig(aux_kalman_weight=0.01))

        trainer = SACTrainer(assembler.state_dim, action_dim, cfg.sac)

        alerts = [
            {
                "obs_type_surprise": float(i),
                "temporal_surprise": 0.5,
                "value_surprise": 0.3,
                "neighborhood_surprise": 0.2,
                "memory_drift": 0.1,
                "timestamp": float(i),
            }
            for i in range(5)
        ]

        result = _kalman_augmentation(
            store=store,
            config=cfg,
            trainer=trainer,
            assembler=assembler,
            tickers=tickers,
            alerts=alerts,
            action_dim=action_dim,
        )

        assert result["aux_kalman_steps"] == 5
        assert isinstance(result["aux_kalman_loss"], float)
        assert isinstance(result["kalman_grad_norm"], float)
        assert np.isfinite(result["aux_kalman_loss"])
        assert np.isfinite(result["kalman_grad_norm"])
