"""Tests for standalone daily microstructure signals."""

import math

from agent.quant.microstructure_signals import (
    MicroThresholds,
    compute_micro_snapshot,
    evaluate_micro_alerts,
    extract_instrument_daily,
)


def _fake_daily(entity_id: str, n: int = 60):
    obs = []
    close = 100.0
    for i in range(n):
        ret = 0.01 * math.sin(i / 5.0)
        close *= math.exp(ret)
        obs.append(
            {
                "entity_id": entity_id,
                "observation_type": "instrument_daily",
                "observed_at": float(1_700_000_000 + i * 86400),
                "value": {
                    "close": close,
                    "log_return": ret,
                    "volume": 1_000_000.0 + i * 1000,
                },
            }
        )
    return obs


def test_extract_instrument_daily():
    obs = _fake_daily("instrument:CL=F")
    rows = extract_instrument_daily(obs, "instrument:CL=F")
    assert len(rows) == 60
    assert rows[-1]["close"] > rows[0]["close"] * 0.5


def test_compute_micro_snapshot():
    obs = _fake_daily("instrument:CL=F", n=80)
    snap = compute_micro_snapshot("instrument:CL=F", obs)
    assert snap is not None
    assert snap.n_days == 80
    assert snap.vol_20d >= 0
    assert isinstance(snap.signed_flow_z, float)


def test_insufficient_data():
    obs = _fake_daily("instrument:NG=F", n=10)
    assert compute_micro_snapshot("instrument:NG=F", obs, min_days=30) is None


def test_evaluate_micro_alerts_flow_strong():
    obs = _fake_daily("instrument:CL=F", n=80)
    snap = compute_micro_snapshot("instrument:CL=F", obs)
    assert snap is not None
    # Force strong flow threshold low for test
    alerts = evaluate_micro_alerts(
        snap, MicroThresholds(flow_z_watch=0.1, flow_z_strong=0.2)
    )
    codes = {a.code for a in alerts}
    assert "FLOW_IMBALANCE" in codes
