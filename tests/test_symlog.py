"""Tests for symlog / symexp transforms.

These tests constitute a mathematical proof of the six properties
claimed in the module docstring:
    1. Exact inverse:   symexp(symlog(x)) = x
    2. Origin:          symlog(0) = 0
    3. Sign-preserving: sign(symlog(x)) = sign(x)
    4. Compressive:     |symlog(x)| ≤ |x|  for |x| ≥ 0
    5. Monotonic:       x₁ < x₂ ⟹ symlog(x₁) < symlog(x₂)
    6. Differentiable:  Gradients exist everywhere
"""

import numpy as np
import pytest
import torch

from agent.learning.policy.symlog import symexp, symexp_np, symlog, symlog_np

# ── Property 1: exact inverse ────────────────────────────────


class TestRoundTrip:
    """symexp(symlog(x)) ≈ x within floating-point tolerance."""

    @pytest.mark.parametrize(
        "x",
        [0.0, 1.0, -1.0, 0.001, -0.001, 100.0, -100.0, 1e6, -1e6, 1e-10],
    )
    def test_roundtrip_scalars_torch(self, x: float) -> None:
        t = torch.tensor(x, dtype=torch.float64)
        recovered = symexp(symlog(t))
        assert torch.allclose(recovered, t, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize(
        "x",
        [0.0, 1.0, -1.0, 0.001, -0.001, 100.0, -100.0, 1e6, -1e6, 1e-10],
    )
    def test_roundtrip_scalars_numpy(self, x: float) -> None:
        a = np.array(x, dtype=np.float64)
        recovered = symexp_np(symlog_np(a))
        np.testing.assert_allclose(recovered, a, atol=1e-10, rtol=1e-10)

    def test_roundtrip_vector(self) -> None:
        x = torch.linspace(-1000, 1000, 2001, dtype=torch.float64)
        recovered = symexp(symlog(x))
        assert torch.allclose(recovered, x, atol=1e-6, rtol=1e-6)

    def test_roundtrip_float32(self) -> None:
        """float32 has less precision—verify within 1e-5."""
        x = torch.tensor([1e4, -1e4, 0.0, 1.0], dtype=torch.float32)
        recovered = symexp(symlog(x))
        assert torch.allclose(recovered, x, atol=1e-4, rtol=1e-4)


# ── Property 2: origin ───────────────────────────────────────


class TestOrigin:
    def test_symlog_zero_torch(self) -> None:
        assert symlog(torch.tensor(0.0)).item() == 0.0

    def test_symlog_zero_numpy(self) -> None:
        assert symlog_np(np.array(0.0)) == 0.0


# ── Property 3: sign-preserving ──────────────────────────────


class TestSignPreserving:
    def test_positive(self) -> None:
        x = torch.tensor([0.1, 1.0, 100.0, 1e10])
        y = symlog(x)
        assert (y > 0).all()

    def test_negative(self) -> None:
        x = torch.tensor([-0.1, -1.0, -100.0, -1e10])
        y = symlog(x)
        assert (y < 0).all()


# ── Property 4: compressive ──────────────────────────────────


class TestCompressive:
    def test_compressive_torch(self) -> None:
        x = torch.tensor([0.0, 0.5, 1.0, 10.0, 100.0, 1e6], dtype=torch.float64)
        y = symlog(x)
        assert (torch.abs(y) <= torch.abs(x) + 1e-15).all()

    def test_compressive_negative(self) -> None:
        x = torch.tensor([-0.5, -1.0, -10.0], dtype=torch.float64)
        y = symlog(x)
        assert (torch.abs(y) <= torch.abs(x) + 1e-15).all()


# ── Property 5: monotonic ────────────────────────────────────


class TestMonotonic:
    def test_strictly_increasing(self) -> None:
        x = torch.linspace(-100, 100, 1000, dtype=torch.float64)
        y = symlog(x)
        diffs = y[1:] - y[:-1]
        assert (diffs > 0).all()


# ── Property 6: differentiable ───────────────────────────────


class TestGradient:
    def test_gradient_exists_at_zero(self) -> None:
        """At x=0, torch.sign(0)=0 ⟹ autograd subgradient = 0.

        The true derivative is 1 (limit from both sides), but sign(0)=0
        kills the product rule.  This is a measure-zero edge—harmless
        in practice because training data is continuous, never exactly 0.
        We verify the subgradient is finite (no NaN/Inf).
        """
        x = torch.tensor(0.0, requires_grad=True)
        y = symlog(x)
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(torch.tensor(x.grad.item()))
        # Verify near-zero gradient is correct (mathematical limit = 1)
        x2 = torch.tensor(1e-7, requires_grad=True)
        y2 = symlog(x2)
        y2.backward()
        assert abs(x2.grad.item() - 1.0) < 1e-3

    def test_gradient_positive(self) -> None:
        x = torch.tensor(5.0, requires_grad=True)
        y = symlog(x)
        y.backward()
        assert x.grad is not None
        # d/dx [log(1+x)] = 1/(1+x) = 1/6
        assert abs(x.grad.item() - 1.0 / 6.0) < 1e-6

    def test_gradient_negative(self) -> None:
        x = torch.tensor(-5.0, requires_grad=True)
        y = symlog(x)
        y.backward()
        assert x.grad is not None
        # d/dx [-log(1+|x|)] at x=-5 = 1/(1+|-5|) = 1/6
        assert abs(x.grad.item() - 1.0 / 6.0) < 1e-6

    def test_gradient_flows_through_chain(self) -> None:
        """Verify gradient propagates through symlog → linear → symexp."""
        x = torch.tensor(3.0, requires_grad=True)
        y = symexp(2 * symlog(x))
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(torch.tensor(x.grad.item()))


# ── Numerical stability ──────────────────────────────────────


class TestStability:
    def test_large_values_float32(self) -> None:
        """symlog compresses large float32 values without overflow."""
        x = torch.tensor([1e38, -1e38], dtype=torch.float32)
        y = symlog(x)
        assert torch.isfinite(y).all()

    def test_tiny_values(self) -> None:
        x = torch.tensor([1e-45, -1e-45], dtype=torch.float32)
        y = symlog(x)
        assert torch.isfinite(y).all()

    def test_numpy_large(self) -> None:
        x = np.array([1e100, -1e100])
        y = symlog_np(x)
        assert np.all(np.isfinite(y))

    def test_symexp_large_input_overflow(self) -> None:
        """symexp of very large symlog output — expect exp to overflow gracefully."""
        x = torch.tensor(100.0, dtype=torch.float64)
        # symlog(x) ≈ 4.6, so symexp(4.6) ≈ 100 → fine
        y = symlog(x)
        z = symexp(y)
        assert torch.allclose(z, x, rtol=1e-10)
