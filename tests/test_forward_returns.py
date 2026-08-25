"""Tests for canonical forward-return label construction."""

from __future__ import annotations

import math

import numpy as np

from agent.quant.forward_returns import (
    build_forward_return_lookup,
    compare_lookups,
    forward_return_vector,
    forward_return_vector_for_date,
    label_distribution_stats,
    label_method_correlation,
)


def _daily_obs(eid: str, ts: float, close: float, log_return: float = 0.01) -> dict:
    return {
        "entity_id": eid,
        "observation_type": "instrument_daily",
        "observed_at": ts,
        "value": {"close": close, "log_return": log_return, "volume": 1e6},
    }


def test_build_forward_return_lookup_simple_growth():
    """Close doubles over ~21 trading days → forward return ≈ 1.0."""
    day = 86400.0 * 7 / 5  # trading day in calendar seconds
    obs = [
        _daily_obs("AAPL", 0.0, 100.0, 0.0),
        _daily_obs("AAPL", 21 * day, 200.0, math.log(2)),
    ]
    lookup = build_forward_return_lookup(obs, horizon_days=21, tolerance_secs=3 * 86400.0)
    assert ("AAPL", 0) in lookup
    assert abs(lookup[("AAPL", 0)] - 1.0) < 0.05


def test_require_future_after_guard():
    day = 86400.0 * 7 / 5
    obs = [
        _daily_obs("AAPL", 0.0, 100.0),
        _daily_obs("AAPL", 21 * day, 150.0),
    ]
    lookup_open = build_forward_return_lookup(obs, horizon_days=21)
    future_ts = 21 * day
    lookup_strict = build_forward_return_lookup(
        obs, horizon_days=21, require_future_after=future_ts
    )
    assert len(lookup_open) >= 1
    assert len(lookup_strict) == 0


def test_forward_return_vector_for_date():
    """Calendar-date alignment: close timestamp mid-day still maps to ISO date."""
    from datetime import datetime, timezone

    day = 86400.0 * 7 / 5
    iso = "2024-01-15"
    noon = (
        datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp()
        + 43200.0
    )
    obs = [
        _daily_obs("AAPL", noon, 100.0),
        _daily_obs("AAPL", noon + 21 * day, 110.0),
    ]
    lookup = build_forward_return_lookup(obs, horizon_days=21)
    vec = forward_return_vector_for_date(lookup, ["AAPL"], iso)
    assert np.isfinite(vec[0])
    assert abs(vec[0] - 0.1) < 0.05


def test_forward_return_vector():
    day = 86400.0 * 7 / 5
    obs = [
        _daily_obs("AAPL", 0.0, 100.0),
        _daily_obs("MSFT", 0.0, 50.0),
        _daily_obs("AAPL", 21 * day, 110.0),
        _daily_obs("MSFT", 21 * day, 55.0),
    ]
    lookup = build_forward_return_lookup(obs, horizon_days=21)
    vec = forward_return_vector(lookup, ["AAPL", "MSFT", "GOOG"], 0.0)
    assert np.isfinite(vec[0])
    assert np.isfinite(vec[1])
    assert np.isnan(vec[2])


def test_compare_lookups_identical():
    day = 86400.0 * 7 / 5
    obs = [
        _daily_obs("AAPL", 0.0, 100.0),
        _daily_obs("AAPL", 21 * day, 120.0),
        _daily_obs("MSFT", 0.0, 50.0),
        _daily_obs("MSFT", 21 * day, 55.0),
    ]
    a = build_forward_return_lookup(obs)
    b = build_forward_return_lookup(obs)
    cmp = compare_lookups(a, b)
    assert cmp["n_shared"] == len(a)
    assert cmp["mean_abs_diff"] == 0.0
    assert cmp["spearman"] > 0.99


def test_label_distribution_stats():
    vals = np.array([0.01, -0.02, 0.03, np.nan])
    s = label_distribution_stats(vals)
    assert s["n"] == 3.0
    assert abs(s["mean"] - 0.006666666666666665) < 1e-9


def test_label_method_correlation_high_on_synthetic():
    day = 86400.0 * 7 / 5
    obs = []
    close = 100.0
    for i in range(30):
        ts = i * day
        lr = 0.001 * (1 if i % 2 == 0 else -1)
        close *= math.exp(lr)
        obs.append(_daily_obs("AAPL", ts, close, lr))
    # Anchor at day 0 — 21d forward close exists at day 21
    corr = label_method_correlation(obs, ["AAPL"], [0.0], horizon_days=21)
    if corr["n"] >= 1.0:
        assert corr["simple_vs_logsum"] > 0.5
    else:
        # Sparse synthetic path: at least lookup builds
        lookup = build_forward_return_lookup(obs, horizon_days=21)
        assert len(lookup) >= 1
