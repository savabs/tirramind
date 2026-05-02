"""Tests for agent.models.gnn.alignment — Phase 49 GNN Downstream Alignment.

Covers:
    - compute_belief_log_likelihood_delta: categorical KL, gaussian entropy reduction
    - store_entity_alignment: signals written to store
    - load_alignment_weights: loads and inverts signals; None when empty
"""

from __future__ import annotations

import time

import pytest

from agent.models.gnn.alignment import (
    _ALIGNMENT_SOURCE,
    _GNN_ENTITY_TYPES,
    compute_belief_log_likelihood_delta,
    load_alignment_weights,
    store_entity_alignment,
)
from agent.pipeline.store import PipelineStore

# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def store():
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


def _cat_belief(name: str, probs: dict) -> dict:
    return {"variable_name": name, "dist_type": "categorical", "probabilities": probs}


def _gauss_belief(name: str, mean: float, variance: float) -> dict:
    return {
        "variable_name": name,
        "dist_type": "gaussian",
        "mean": mean,
        "variance": variance,
    }


# ══════════════════════════════════════════════════════════════
# compute_belief_log_likelihood_delta
# ══════════════════════════════════════════════════════════════


class TestComputeBeliefLogLikelihoodDelta:
    def test_categorical_sharpened_belief_positive_delta(self):
        """After update, if belief shifted toward a high-prob state, KL > 0."""
        before = [_cat_belief("regime.macro", {"expansion": 0.33, "contraction": 0.33, "crisis": 0.34})]
        after = [_cat_belief("regime.macro", {"expansion": 0.90, "contraction": 0.05, "crisis": 0.05})]

        deltas = compute_belief_log_likelihood_delta(before, after)

        assert "regime.macro" in deltas
        assert deltas["regime.macro"] > 0.0

    def test_categorical_identical_beliefs_zero_delta(self):
        """KL(p || p) = 0 for identical distributions."""
        probs = {"expansion": 0.4, "contraction": 0.4, "crisis": 0.2}
        before = [_cat_belief("regime.macro", probs)]
        after = [_cat_belief("regime.macro", probs)]

        deltas = compute_belief_log_likelihood_delta(before, after)

        assert "regime.macro" in deltas
        assert abs(deltas["regime.macro"]) < 1e-9

    def test_gaussian_variance_reduced_positive_delta(self):
        """Variance reduced after update → entropy reduction > 0."""
        before = [_gauss_belief("state.macro_momentum", mean=0.0, variance=1.0)]
        after = [_gauss_belief("state.macro_momentum", mean=0.1, variance=0.5)]

        deltas = compute_belief_log_likelihood_delta(before, after)

        assert "state.macro_momentum" in deltas
        assert deltas["state.macro_momentum"] > 0.0

    def test_gaussian_variance_unchanged_zero_delta(self):
        """Same variance before and after → entropy reduction = 0."""
        before = [_gauss_belief("state.macro_momentum", mean=0.0, variance=0.8)]
        after = [_gauss_belief("state.macro_momentum", mean=0.2, variance=0.8)]

        deltas = compute_belief_log_likelihood_delta(before, after)

        assert "state.macro_momentum" in deltas
        assert abs(deltas["state.macro_momentum"]) < 1e-9

    def test_missing_before_variable_excluded(self):
        """Variable present in after but not in before is silently excluded."""
        before = []
        after = [_cat_belief("regime.macro", {"expansion": 0.9, "contraction": 0.05, "crisis": 0.05})]

        deltas = compute_belief_log_likelihood_delta(before, after)

        assert "regime.macro" not in deltas

    def test_empty_probabilities_returns_zero_delta(self):
        """Missing probabilities dict → delta = 0.0."""
        before = [
            {
                "variable_name": "regime.macro",
                "dist_type": "categorical",
                "probabilities": None,
            }
        ]
        after = [
            {
                "variable_name": "regime.macro",
                "dist_type": "categorical",
                "probabilities": None,
            }
        ]

        deltas = compute_belief_log_likelihood_delta(before, after)

        # Either not present or 0.0
        delta = deltas.get("regime.macro", 0.0)
        assert abs(delta) < 1e-9


# ══════════════════════════════════════════════════════════════
# store_entity_alignment
# ══════════════════════════════════════════════════════════════


class TestStoreEntityAlignment:
    def test_stores_variable_signals(self, store):
        """Variable deltas are written to store as signals."""
        variable_deltas = {"state.macro_momentum": 0.5, "regime.macro": 0.2}
        as_of = time.time()
        store_entity_alignment(store, variable_deltas, as_of=as_of)

        rows = store.query_signals(f"{_ALIGNMENT_SOURCE}.state.macro_momentum", limit=1)
        assert len(rows) == 1
        assert abs(rows[0]["value"] - 0.5) < 1e-6

    def test_stores_entity_type_aggregate_for_kalman_variables(self, store):
        """Kalman state variables (state.* prefix) → entity-type aggregate signals."""
        variable_deltas = {"state.macro_momentum": 0.4, "state.liquidity_state": 0.6}
        store_entity_alignment(store, variable_deltas, as_of=time.time())

        expected_mean = (0.4 + 0.6) / 2
        for entity_type in _GNN_ENTITY_TYPES:
            rows = store.query_signals(f"{_ALIGNMENT_SOURCE}.entity.{entity_type}", limit=1)
            assert len(rows) == 1, f"Missing signal for entity type '{entity_type}'"
            assert abs(rows[0]["value"] - expected_mean) < 1e-6

    def test_empty_variable_deltas_no_signals(self, store):
        """Empty delta dict → no signals written."""
        store_entity_alignment(store, {}, as_of=time.time())
        rows = store.query_signals(_ALIGNMENT_SOURCE, limit=100)
        assert len(rows) == 0


# ══════════════════════════════════════════════════════════════
# load_alignment_weights
# ══════════════════════════════════════════════════════════════


class TestLoadAlignmentWeights:
    def test_returns_none_when_no_signals(self, store):
        """No alignment signals in store → returns None (uniform weights)."""
        result = load_alignment_weights(store, ["person", "company"])
        assert result is None

    def test_high_delta_gives_low_weight(self, store):
        """High alignment delta → weight < 1.0 (entity already well aligned)."""
        # Store a high delta for 'person'
        store.store_signal(f"{_ALIGNMENT_SOURCE}.entity.person", 2.0, metadata={"as_of": time.time()})
        weights = load_alignment_weights(store, ["person"])
        assert weights is not None
        # weight = 1 / (1 + 2.0) = 0.333...
        assert abs(weights["person"] - 1.0 / 3.0) < 1e-3

    def test_zero_delta_gives_weight_one(self, store):
        """Zero delta → weight = 1.0 (no alignment history, neutral)."""
        store.store_signal(f"{_ALIGNMENT_SOURCE}.entity.company", 0.0, metadata={"as_of": time.time()})
        weights = load_alignment_weights(store, ["company"])
        assert weights is not None
        assert abs(weights["company"] - 1.0) < 1e-6

    def test_missing_entity_type_filled_with_default(self, store):
        """Entity type with no signal gets default weight = 1.0."""
        store.store_signal(f"{_ALIGNMENT_SOURCE}.entity.person", 0.5, metadata={"as_of": time.time()})
        weights = load_alignment_weights(store, ["person", "vessel"])
        assert weights is not None
        assert "vessel" in weights
        assert weights["vessel"] == 1.0

    def test_negative_delta_clamped_to_weight_one(self, store):
        """Negative delta (diffused belief) → max(delta, 0) = 0 → weight = 1.0."""
        store.store_signal(f"{_ALIGNMENT_SOURCE}.entity.wallet", -0.3, metadata={"as_of": time.time()})
        weights = load_alignment_weights(store, ["wallet"])
        assert weights is not None
        assert abs(weights["wallet"] - 1.0) < 1e-6

    def test_expired_signals_not_loaded(self, store):
        """Signals older than lookback_days are ignored → returns None."""
        # Write a signal with old timestamp (manually through raw signal insertion)
        # We simulate by writing via the store but with a computed_at in the past

        conn = store._get_conn()
        old_time = time.time() - 30 * 86400  # 30 days ago
        conn.execute(
            "INSERT INTO signals (signal_name, computed_at, value, metadata_json) VALUES (?, ?, ?, ?)",
            (f"{_ALIGNMENT_SOURCE}.entity.country", old_time, 1.0, None),
        )
        conn.commit()

        weights = load_alignment_weights(store, ["country"], lookback_days=7.0)
        assert weights is None
