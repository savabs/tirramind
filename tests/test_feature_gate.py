"""TirraMind — Edge Case Tests for FeatureGate (Change 11)

Covers: shape correctness, gradient flow, entropy loss, gate floor
enforcement, NaN/Inf robustness, regime conditioning, save/load
round-trip, diagnostic output, integration with LearnedStateEncoder.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from agent.learning.policy.feature_gate import FeatureGate, FeatureGateConfig
from agent.learning.policy.state_encoder import LearnedStateEncoder


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def default_gate() -> FeatureGate:
    return FeatureGate(FeatureGateConfig())


@pytest.fixture
def custom_gate() -> FeatureGate:
    """Gate with 3 groups and custom dims."""
    return FeatureGate(
        FeatureGateConfig(
            n_feature_groups=3,
            regime_dim=2,
            gate_hidden_dim=8,
            gate_floor=0.1,
            entropy_weight=0.05,
            group_dims=(10, 5, 3),
        )
    )


# ── Shape Tests ───────────────────────────────────────────────


class TestFeatureGateShape:
    """Forward pass shape correctness."""

    def test_single_sample(self, default_gate: FeatureGate) -> None:
        state = torch.randn(463)
        regime = torch.randn(4)
        out = default_gate(state, regime)
        assert out.shape == (463,)

    def test_batch(self, default_gate: FeatureGate) -> None:
        state = torch.randn(8, 463)
        regime = torch.randn(8, 4)
        out = default_gate(state, regime)
        assert out.shape == (8, 463)

    def test_batch_size_1(self, default_gate: FeatureGate) -> None:
        state = torch.randn(1, 463)
        regime = torch.randn(1, 4)
        out = default_gate(state, regime)
        assert out.shape == (1, 463)

    def test_custom_dims(self, custom_gate: FeatureGate) -> None:
        state = torch.randn(4, 18)  # 10+5+3=18
        regime = torch.randn(4, 2)
        out = custom_gate(state, regime)
        assert out.shape == (4, 18)

    def test_gate_values_shape_single(self, default_gate: FeatureGate) -> None:
        regime = torch.randn(4)
        g = default_gate.gate_values(regime)
        assert g.shape == (5,)

    def test_gate_values_shape_batch(self, default_gate: FeatureGate) -> None:
        regime = torch.randn(3, 4)
        g = default_gate.gate_values(regime)
        assert g.shape == (3, 5)


# ── Gate Value Range ──────────────────────────────────────────


class TestGateValueRange:
    """Gate values must be in [floor, 1.0]."""

    def test_default_floor(self, default_gate: FeatureGate) -> None:
        regime = torch.randn(100, 4)
        g = default_gate.gate_values(regime)
        assert g.min().item() >= 0.05 - 1e-6
        assert g.max().item() <= 1.0 + 1e-6

    def test_zero_floor(self) -> None:
        gate = FeatureGate(FeatureGateConfig(gate_floor=0.0))
        regime = torch.randn(100, 4)
        g = gate.gate_values(regime)
        assert g.min().item() >= 0.0 - 1e-6
        assert g.max().item() <= 1.0 + 1e-6

    def test_high_floor(self) -> None:
        gate = FeatureGate(FeatureGateConfig(gate_floor=0.5))
        regime = torch.randn(100, 4)
        g = gate.gate_values(regime)
        assert g.min().item() >= 0.5 - 1e-6

    def test_extreme_regime_values(self, default_gate: FeatureGate) -> None:
        """Very large/small regime → clamped → no NaN."""
        regime = torch.tensor([1e6, -1e6, float("inf"), float("-inf")])
        g = default_gate.gate_values(regime)
        assert not torch.isnan(g).any()
        assert not torch.isinf(g).any()


# ── Gradient Flow ─────────────────────────────────────────────


class TestGradientFlow:
    """Gradient flow through the gate."""

    def test_gradient_to_state(self, default_gate: FeatureGate) -> None:
        state = torch.randn(1, 463, requires_grad=True)
        regime = torch.randn(1, 4)
        out = default_gate(state, regime)
        out.sum().backward()
        assert state.grad is not None
        assert state.grad.shape == (1, 463)

    def test_gradient_to_gate_params(self, default_gate: FeatureGate) -> None:
        state = torch.randn(2, 463)
        regime = torch.randn(2, 4)
        out = default_gate(state, regime)
        loss = out.sum()
        loss.backward()
        for p in default_gate.parameters():
            assert p.grad is not None

    def test_gradient_through_encoder(self) -> None:
        """Gradients flow from encoder output back through gate."""
        gate = FeatureGate(FeatureGateConfig())
        encoder = LearnedStateEncoder()
        encoder.set_feature_gate(gate)

        state = torch.randn(2, 463, requires_grad=True)
        regime = torch.randn(2, 4)
        out = encoder(state, regime)
        out.sum().backward()

        assert state.grad is not None
        # Gate params should have gradients
        gate_params = list(gate.parameters())
        assert len(gate_params) > 0
        assert all(p.grad is not None for p in gate_params)


# ── Entropy Loss ──────────────────────────────────────────────


class TestEntropyLoss:
    """Entropy regularization loss."""

    def test_returns_scalar(self, default_gate: FeatureGate) -> None:
        state = torch.randn(2, 463)
        regime = torch.randn(2, 4)
        default_gate(state, regime)
        loss = default_gate.entropy_loss()
        assert loss.dim() == 0  # scalar

    def test_no_forward_returns_zero(self, default_gate: FeatureGate) -> None:
        """Before any forward pass, entropy loss = 0."""
        loss = default_gate.entropy_loss()
        assert loss.item() == 0.0

    def test_entropy_is_negative(self, default_gate: FeatureGate) -> None:
        """Entropy loss should be negative (encourages entropy maximization)."""
        state = torch.randn(4, 463)
        regime = torch.randn(4, 4)
        default_gate(state, regime)
        loss = default_gate.entropy_loss()
        assert loss.item() < 0  # negative because -λ * entropy

    def test_entropy_magnitude_scales_with_weight(self) -> None:
        """Higher entropy_weight → larger magnitude loss."""
        state = torch.randn(4, 463)
        regime = torch.randn(4, 4)

        gate_small = FeatureGate(FeatureGateConfig(entropy_weight=0.001))
        gate_large = FeatureGate(FeatureGateConfig(entropy_weight=0.1))
        # Use same params for fair comparison
        gate_large.load_state_dict(gate_small.state_dict())

        gate_small(state, regime)
        gate_large(state, regime)
        assert abs(gate_large.entropy_loss().item()) > abs(
            gate_small.entropy_loss().item()
        )


# ── Regime Conditioning ──────────────────────────────────────


class TestRegimeConditioning:
    """Different regime contexts produce different gate values."""

    def test_different_regimes_different_gates(self, default_gate: FeatureGate) -> None:
        r1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        r2 = torch.tensor([0.0, 0.0, 0.0, 1.0])
        g1 = default_gate.gate_values(r1)
        g2 = default_gate.gate_values(r2)
        assert not torch.allclose(g1, g2), "Different regimes should give different gates"

    def test_same_regime_same_gates(self, default_gate: FeatureGate) -> None:
        r = torch.tensor([0.5, 0.3, 0.1, 0.1])
        g1 = default_gate.gate_values(r)
        g2 = default_gate.gate_values(r)
        assert torch.allclose(g1, g2)

    def test_zero_regime_stable(self, default_gate: FeatureGate) -> None:
        """Zero regime context → stable default gates (sigmoid of bias)."""
        r = torch.zeros(4)
        g = default_gate.gate_values(r)
        assert not torch.isnan(g).any()
        assert g.shape == (5,)


# ── Configuration Validation ─────────────────────────────────


class TestConfigValidation:
    """Config mismatch errors."""

    def test_mismatched_group_dims(self) -> None:
        with pytest.raises(ValueError, match="n_feature_groups"):
            FeatureGate(
                FeatureGateConfig(n_feature_groups=3, group_dims=(10, 5))
            )

    def test_total_dim_property(self, default_gate: FeatureGate) -> None:
        assert default_gate.total_dim == 463

    def test_config_accessible(self, default_gate: FeatureGate) -> None:
        cfg = default_gate.config
        assert cfg.n_feature_groups == 5
        assert cfg.regime_dim == 4


# ── Diagnostics ───────────────────────────────────────────────


class TestDiagnostics:
    """Diagnostic output format."""

    def test_diagnostics_keys(self, default_gate: FeatureGate) -> None:
        regime = torch.randn(4)
        diag = default_gate.gate_diagnostics(regime)
        assert "group_names" in diag
        assert "gate_values" in diag
        assert "entropy" in diag

    def test_diagnostics_group_names(self, default_gate: FeatureGate) -> None:
        regime = torch.randn(4)
        diag = default_gate.gate_diagnostics(regime)
        assert diag["group_names"] == [
            "surprise",
            "belief",
            "market",
            "entity_count",
            "adversarial",
        ]

    def test_diagnostics_values_count(self, default_gate: FeatureGate) -> None:
        regime = torch.randn(4)
        diag = default_gate.gate_diagnostics(regime)
        assert len(diag["gate_values"]) == 5

    def test_diagnostics_entropy_finite(self, default_gate: FeatureGate) -> None:
        regime = torch.randn(4)
        diag = default_gate.gate_diagnostics(regime)
        assert math.isfinite(diag["entropy"])


# ── Save/Load Round-Trip ─────────────────────────────────────


class TestSaveLoad:
    """State dict round-trip."""

    def test_state_dict_round_trip(self, default_gate: FeatureGate) -> None:
        sd = default_gate.state_dict()
        gate2 = FeatureGate(FeatureGateConfig())
        gate2.load_state_dict(sd)

        regime = torch.randn(4)
        g1 = default_gate.gate_values(regime)
        g2 = gate2.gate_values(regime)
        assert torch.allclose(g1, g2)

    def test_save_load_with_encoder(self) -> None:
        torch.manual_seed(99)
        gate = FeatureGate(FeatureGateConfig())
        encoder = LearnedStateEncoder()
        encoder.set_feature_gate(gate)
        encoder.eval()

        # Save
        sd = encoder.state_dict()

        # Load into fresh encoder+gate
        torch.manual_seed(99)
        gate2 = FeatureGate(FeatureGateConfig())
        encoder2 = LearnedStateEncoder()
        encoder2.set_feature_gate(gate2)
        encoder2.load_state_dict(sd)
        encoder2.eval()

        state = torch.randn(2, 463)
        regime = torch.randn(2, 4)
        out1 = encoder(state, regime)
        out2 = encoder2(state, regime)
        assert torch.allclose(out1, out2, atol=1e-6)


# ── Integration with LearnedStateEncoder ──────────────────────


class TestEncoderIntegration:
    """Feature gate works correctly inside LearnedStateEncoder."""

    def test_set_feature_gate_type_check(self) -> None:
        encoder = LearnedStateEncoder()
        with pytest.raises(TypeError, match="Expected FeatureGate"):
            encoder.set_feature_gate("not a gate")  # type: ignore

    def test_encoder_with_gate_output_shape(self) -> None:
        gate = FeatureGate(FeatureGateConfig())
        encoder = LearnedStateEncoder()
        encoder.set_feature_gate(gate)

        state = torch.randn(4, 463)
        regime = torch.randn(4, 4)
        out = encoder(state, regime)
        assert out.shape == (4, encoder.output_dim)

    def test_encoder_without_regime_skips_gate(self) -> None:
        """If no regime_context, gate is not applied even if attached."""
        torch.manual_seed(77)
        gate = FeatureGate(FeatureGateConfig())
        encoder = LearnedStateEncoder()
        encoder.set_feature_gate(gate)
        encoder.eval()

        state = torch.randn(2, 463)
        out_no_regime = encoder(state)

        # Build fresh encoder with identical non-gate weights
        torch.manual_seed(77)
        encoder2 = LearnedStateEncoder()
        # Copy only the non-gate weights from the original
        sd_orig = encoder.state_dict()
        sd_no_gate = {
            k: v for k, v in sd_orig.items() if "feat_gate" not in k
        }
        encoder2.load_state_dict(sd_no_gate)
        encoder2.eval()
        out_no_gate = encoder2(state)
        assert torch.allclose(out_no_regime, out_no_gate, atol=1e-5)

    def test_gated_output_differs_from_ungated(self) -> None:
        """With non-trivial gates, output should differ from ungated."""
        torch.manual_seed(42)
        gate = FeatureGate(FeatureGateConfig())
        encoder = LearnedStateEncoder()
        encoder.set_feature_gate(gate)

        state = torch.randn(2, 463)
        regime = torch.randn(2, 4)

        out_gated = encoder(state, regime)
        out_ungated = encoder(state)  # no regime → no gating
        # Very unlikely to be identical unless all gates ≈ 1.0
        assert not torch.allclose(out_gated, out_ungated, atol=1e-4)

    def test_feature_gate_property(self) -> None:
        encoder = LearnedStateEncoder()
        assert encoder.feature_gate is None

        gate = FeatureGate(FeatureGateConfig())
        encoder.set_feature_gate(gate)
        assert encoder.feature_gate is gate

    def test_gate_params_in_encoder_params(self) -> None:
        """Gate parameters are discoverable via encoder.parameters()."""
        gate = FeatureGate(FeatureGateConfig())
        encoder = LearnedStateEncoder()
        n_before = sum(1 for _ in encoder.parameters())
        encoder.set_feature_gate(gate)
        n_after = sum(1 for _ in encoder.parameters())
        assert n_after > n_before

    def test_single_sample_with_gate(self) -> None:
        """1D input (no batch) works with gate."""
        gate = FeatureGate(FeatureGateConfig())
        encoder = LearnedStateEncoder()
        encoder.set_feature_gate(gate)

        state = torch.randn(463)
        regime = torch.randn(4)
        out = encoder(state, regime)
        assert out.shape == (encoder.output_dim,)
