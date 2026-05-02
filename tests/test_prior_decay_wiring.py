"""Tests for _apply_prior_decay in world_model_update — Phase 49b.

Covers:
    - decay=1.0 → no mutation (stable regime path)
    - decay<1.0 → Kalman P inflated and CPDs softened toward uniform
    - decay<1.0 → CPDs still sum to 1 per column (valid probability)
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.models.initial_graph import ALL_NODES
from agent.models.world_model import WorldModel
from agent.pipeline.dags.world_model_update import (
    _apply_prior_decay,
    _build_world_model,
)

# ── Helpers ────────────────────────────────────────────────────


def _make_world_model() -> WorldModel:
    """Build a minimal valid WorldModel for testing."""
    return _build_world_model(learned_edges=None, use_differentiable_filter=False)


# ══════════════════════════════════════════════════════════════
# _apply_prior_decay — stable regime (decay=1.0)
# ══════════════════════════════════════════════════════════════


class TestApplyPriorDecayNoop:
    def test_decay_one_does_not_change_P(self):
        """decay=1.0 (stable regime) → Kalman P unchanged."""
        wm = _make_world_model()
        P_before = wm._filter._P.copy()
        _apply_prior_decay(wm, 1.0)
        np.testing.assert_array_almost_equal(wm._filter._P, P_before)

    def test_decay_one_does_not_change_cpds(self):
        """decay=1.0 → CPDs are not mutated."""
        wm = _make_world_model()
        # Snapshot CPD values for all observed nodes that have CPDs
        cpd_snapshots = {}
        for spec in ALL_NODES:
            cpd = wm._graph.get_cpd(spec.name)
            if cpd is not None:
                cpd_snapshots[spec.name] = cpd.get_values().copy()

        _apply_prior_decay(wm, 1.0)

        for spec_name, before_values in cpd_snapshots.items():
            cpd = wm._graph.get_cpd(spec_name)
            if cpd is not None:
                np.testing.assert_array_almost_equal(
                    cpd.get_values(),
                    before_values,
                    err_msg=f"CPD for '{spec_name}' changed with decay=1.0",
                )


# ══════════════════════════════════════════════════════════════
# _apply_prior_decay — regime changed (decay=0.8)
# ══════════════════════════════════════════════════════════════


class TestApplyPriorDecayActive:
    def test_decay_0_8_inflates_kalman_P(self):
        """decay=0.8 → P scaled by 1/0.8 = 1.25."""
        wm = _make_world_model()
        P_before = wm._filter._P.copy()
        _apply_prior_decay(wm, 0.8)
        expected = P_before * (1.0 / 0.8)
        np.testing.assert_array_almost_equal(wm._filter._P, expected, decimal=10)

    def test_decay_softens_observed_cpds(self):
        """decay=0.8 → observed-node CPDs blended toward uniform.

        Each column of the CPD should be strictly closer to uniform than before.
        We check that the maximum absolute deviation from uniform decreases.
        """
        wm = _make_world_model()
        decay = 0.8

        # Find an observed node that has a non-uniform CPD set
        for spec in ALL_NODES:
            if spec.node_type != "observed":
                continue
            cpd = wm._graph.get_cpd(spec.name)
            if cpd is None:
                continue
            if spec.cardinality is None or spec.cardinality < 2:
                continue

            values_before = cpd.get_values().copy()
            k = spec.cardinality
            uniform = 1.0 / k

            # Max deviation from uniform before decay
            dev_before = np.abs(values_before - uniform).max()
            break
        else:
            pytest.skip("No suitable observed node with CPD found in initial graph.")

        _apply_prior_decay(wm, decay)

        cpd_after = wm._graph.get_cpd(spec.name)
        values_after = cpd_after.get_values()
        dev_after = np.abs(values_after - uniform).max()

        # After blending toward uniform, deviation must be strictly smaller
        assert dev_after < dev_before or np.isclose(dev_before, 0.0), (
            f"CPD deviation did not decrease for '{spec.name}': before={dev_before:.4f} after={dev_after:.4f}"
        )

    def test_decay_cpd_columns_sum_to_one(self):
        """After applying prior decay, every CPD column still sums to 1."""
        wm = _make_world_model()
        _apply_prior_decay(wm, 0.8)

        for spec in ALL_NODES:
            if spec.node_type != "observed":
                continue
            cpd = wm._graph.get_cpd(spec.name)
            if cpd is None:
                continue
            values = cpd.get_values()
            col_sums = values.sum(axis=0)
            np.testing.assert_array_almost_equal(
                col_sums,
                np.ones_like(col_sums),
                decimal=12,
                err_msg=f"CPD columns for '{spec.name}' do not sum to 1 after decay.",
            )

    def test_decay_p_is_symmetric(self):
        """After inflation, P remains symmetric (Kalman covariance requirement)."""
        wm = _make_world_model()
        _apply_prior_decay(wm, 0.8)
        P = wm._filter._P
        np.testing.assert_array_almost_equal(P, P.T, decimal=14)
