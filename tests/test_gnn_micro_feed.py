"""GNN micro feature vector from instrument_daily (Layer 3 feed)."""

from __future__ import annotations

import pytest

from agent.quant.microstructure_signals import (
    GNN_MICRO_DIM,
    compute_gnn_micro_features,
)


def _daily_obs(entity_id: str, n: int = 45) -> list[dict]:
    obs = []
    for i in range(n):
        obs.append(
            {
                "entity_id": entity_id,
                "observation_type": "instrument_daily",
                "observed_at": float(i * 86400),
                "value": {
                    "close": 100.0 + i * 0.05,
                    "log_return": 0.0005 if i % 2 == 0 else -0.0003,
                    "volume": 5000.0 + i * 50,
                },
            }
        )
    return obs


def test_gnn_micro_vector_shape_and_nonzero():
    eid = "inst:test"
    obs = _daily_obs(eid)
    vec = compute_gnn_micro_features(eid, obs, current_time=1e9)
    assert len(vec) == GNN_MICRO_DIM
    assert sum(abs(x) for x in vec) > 0


def test_gnn_micro_insufficient_data_returns_zeros():
    eid = "inst:test"
    obs = _daily_obs(eid, n=5)
    vec = compute_gnn_micro_features(eid, obs, current_time=1e9, min_days=30)
    assert vec == [0.0] * GNN_MICRO_DIM


def test_gnn_micro_no_forward_bias():
    eid = "inst:test"
    obs = _daily_obs(eid, n=50)
    t_cut = 30 * 86400.0
    vec_early = compute_gnn_micro_features(eid, obs, current_time=t_cut)
    vec_late = compute_gnn_micro_features(eid, obs, current_time=50 * 86400.0)
    assert vec_early != vec_late
