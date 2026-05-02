"""Tests for agent.pipeline.regime_gate — Phase 49b additions.

Covers:
    - world_model_prior_decay: stable regime → 1.0, regime changed → 0.8
    - feature_trust_scale: stability >= 3d → 1.0, stability < 3d → 0.7
"""

from __future__ import annotations

import time

import pytest

from agent.pipeline.regime_gate import (
    RegimeContext,
    feature_trust_scale,
    is_high_changepoint,
    sac_entropy_scale,
    world_model_prior_decay,
)

# ── Helpers ────────────────────────────────────────────────────


def _ctx(
    regime_label: str = "expansion",
    changepoint_posterior: float = 0.1,
    stability_duration_days: float = 10.0,
    regime_changed: bool = False,
) -> RegimeContext:
    return RegimeContext(
        regime_label=regime_label,
        changepoint_posterior=changepoint_posterior,
        stability_duration_days=stability_duration_days,
        regime_changed=regime_changed,
        as_of=time.time(),
    )


# ── world_model_prior_decay ────────────────────────────────────


class TestWorldModelPriorDecay:
    def test_stable_regime_returns_one(self):
        """Stable regime (regime_changed=False) → decay = 1.0."""
        ctx = _ctx(regime_changed=False)
        assert world_model_prior_decay(ctx) == 1.0

    def test_changed_regime_returns_0_8(self):
        """Regime just changed → decay = 0.8."""
        ctx = _ctx(regime_changed=True)
        assert world_model_prior_decay(ctx) == pytest.approx(0.8)

    def test_high_changepoint_stable_label_unchanged_is_still_1(self):
        """High changepoint posterior but regime_changed=False → decay = 1.0.

        regime_changed is the authoritative flag; changepoint_posterior alone
        does not trigger decay.
        """
        ctx = _ctx(changepoint_posterior=0.95, regime_changed=False)
        assert world_model_prior_decay(ctx) == 1.0


# ── feature_trust_scale ────────────────────────────────────────


class TestFeatureTrustScale:
    def test_long_stability_returns_one(self):
        """Stability >= 3 days → trust = 1.0."""
        ctx = _ctx(stability_duration_days=7.0)
        assert feature_trust_scale(ctx) == 1.0

    def test_exact_three_days_boundary_returns_one(self):
        """Exactly 3.0 days is the boundary — should return 1.0 (stable side)."""
        ctx = _ctx(stability_duration_days=3.0)
        assert feature_trust_scale(ctx) == 1.0

    def test_short_stability_returns_0_7(self):
        """Stability < 3 days → trust = 0.7."""
        ctx = _ctx(stability_duration_days=1.5)
        assert feature_trust_scale(ctx) == pytest.approx(0.7)

    def test_zero_stability_returns_0_7(self):
        """Zero stability (brand-new regime) → trust = 0.7."""
        ctx = _ctx(stability_duration_days=0.0)
        assert feature_trust_scale(ctx) == pytest.approx(0.7)


# ── is_high_changepoint ────────────────────────────────────────


class TestIsHighChangepoint:
    def test_above_threshold_is_true(self):
        ctx = _ctx(changepoint_posterior=0.95)
        assert is_high_changepoint(store=None, threshold=0.9, ctx=ctx) is True

    def test_below_threshold_is_false(self):
        ctx = _ctx(changepoint_posterior=0.3)
        assert is_high_changepoint(store=None, threshold=0.9, ctx=ctx) is False

    def test_exactly_at_threshold_is_true(self):
        """Posterior exactly equal to threshold → meets threshold → True."""
        ctx = _ctx(changepoint_posterior=0.9)
        assert is_high_changepoint(store=None, threshold=0.9, ctx=ctx) is True


# ── sac_entropy_scale ──────────────────────────────────────────


class TestSACEntropyScale:
    def test_normal_regime_returns_neg_0_5(self):
        ctx = _ctx(changepoint_posterior=0.1)
        assert sac_entropy_scale(ctx) == pytest.approx(-0.5)

    def test_high_changepoint_returns_neg_0_3(self):
        ctx = _ctx(changepoint_posterior=0.95)
        assert sac_entropy_scale(ctx) == pytest.approx(-0.3)
