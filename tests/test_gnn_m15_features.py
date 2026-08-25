"""M14.1 GNN feed from M15 observations."""

from __future__ import annotations

import time

import pytest
import torch

from agent.models.gnn.graph_builder import (
    GraphBuilder,
    M15_QUANT_DIM,
    MICROSTRUCTURE_DIM,
    PRICE_FEAT_DIM,
    BASE_FEAT_DIM,
)
from agent.pipeline.store import PipelineStore
from agent.quant.gnn_quant_features import (
    M15_QUANT_DIM as M15_LEN,
    OPTIONS_QUANT_DIM,
    compute_gnn_m15_features,
    compute_options_quant_features,
    m15_options_masked,
)


def _daily(entity_id: str, day: int, close: float) -> dict:
    return {
        "entity_id": entity_id,
        "observation_type": "instrument_daily",
        "observed_at": float(day * 86400),
        "value": {"close": close, "log_return": 0.001, "volume": 1e6},
    }


def test_m15_options_masked_by_default(monkeypatch):
    monkeypatch.delenv("TIRRA_MASK_M15_OPTIONS", raising=False)
    assert m15_options_masked() is True
    obs = [
        {
            "entity_id": "inst:SPY",
            "observation_type": "options_chain_eod",
            "observed_at": 1e9,
            "value": {"atm_call_iv": 0.25, "atm_put_iv": 0.22, "spot": 500.0},
        }
    ]
    assert compute_options_quant_features("inst:SPY", obs, 2e9) == [0.0] * OPTIONS_QUANT_DIM


def test_m15_vector_shape_with_options_and_curve(monkeypatch):
    monkeypatch.setenv("TIRRA_MASK_M15_OPTIONS", "0")
    eid = "inst:SPY"
    t = 50 * 86400.0
    obs = [
        _daily(eid, 40, 500.0),
        {
            "entity_id": eid,
            "observation_type": "options_chain_eod",
            "observed_at": t - 86400,
            "value": {
                "atm_call_iv": 0.18,
                "atm_put_iv": 0.20,
                "put_call_oi_ratio": 1.1,
                "total_open_interest": 10000,
                "spot": 500.0,
            },
        },
        {
            "entity_id": eid,
            "observation_type": "dividend",
            "observed_at": t - 30 * 86400,
            "value": {"amount": 1.5, "ex_date": "2026-01-01"},
        },
    ]
    from agent.pipeline.entity import entity_id_from_key

    us_eid = entity_id_from_key("country", "US")
    obs.append(
        {
            "entity_id": us_eid,
            "observation_type": "sovereign_yield",
            "observed_at": t - 86400,
            "value": {
                "source": "us_treasury",
                "yields": {"2y": 4.0, "10y": 4.5},
                "curve_2s10s": 0.5,
                "curve_3m10y": 0.4,
            },
        }
    )
    vec = compute_gnn_m15_features(
        eid, obs, t, spot=500.0, us_country_eid=us_eid
    )
    assert len(vec) == M15_LEN
    assert vec[0] == 1.0  # options mask
    assert vec[7] == 1.0  # rate mask
    assert vec[8] == pytest.approx(4.0)
    assert vec[12] == 1.0  # dividend mask


def test_graph_builder_instrument_dim_49_with_m15(tmp_path):
    store = PipelineStore(str(tmp_path / "t.db"))
    eid = "abc123"
    store.register_entity("instrument", "SPY", eid, {"ticker": "SPY"})
    for day in range(35):
        store.store_entity_observation(
            entity_id=eid,
            source_tool="test",
            observed_at=float(day * 86400),
            observation_type="instrument_daily",
            value={"close": 100.0 + day, "log_return": 0.0, "volume": 5000},
        )
    store.store_entity_observation(
        entity_id=eid,
        source_tool="options_chain",
        observed_at=40 * 86400.0,
        observation_type="options_chain_eod",
        value={
            "atm_call_iv": 0.2,
            "atm_put_iv": 0.21,
            "put_call_oi_ratio": 1.0,
            "total_open_interest": 500,
            "spot": 140.0,
        },
    )
    from agent.pipeline.entity import entity_id_from_key

    us_eid = entity_id_from_key("country", "US")
    store.register_entity("country", "US", us_eid)
    store.store_entity_observation(
        entity_id=us_eid,
        source_tool="sovereign_debt",
        observed_at=40 * 86400.0,
        observation_type="sovereign_yield",
        value={
            "source": "us_treasury",
            "yields": {"10y": 4.2, "2y": 3.9},
            "curve_2s10s": 0.3,
        },
    )

    data, _, _ = GraphBuilder(store).build()
    expected = BASE_FEAT_DIM + PRICE_FEAT_DIM + MICROSTRUCTURE_DIM + M15_QUANT_DIM
    assert expected == 49
    assert data["instrument"].x.shape[1] == expected
    m15_off = BASE_FEAT_DIM + PRICE_FEAT_DIM + MICROSTRUCTURE_DIM
    block = data["instrument"].x[0, m15_off : m15_off + M15_QUANT_DIM]
    assert block[0] == 0.0  # options masked (TIRRA_MASK_M15_OPTIONS default)
    assert block[7] == 1.0  # has rates
    assert float(block.abs().sum()) > 0
    store.close()
