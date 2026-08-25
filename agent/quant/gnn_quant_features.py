"""
M14.1 — Quant features from M15 pipeline observations for GNN instrument nodes.

Reads options_chain_eod, dividend, and US sovereign_yield (via country curve cache)
without requiring full option chains at feature time — uses stored aggregates only.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Layout documented for graph_builder.M15_QUANT_DIM
OPTIONS_QUANT_DIM = 7  # mask + 6 scalars
RATE_QUANT_DIM = 5  # mask + 2y, 10y, 2s10s, 3m10y
DIVIDEND_QUANT_DIM = 3  # mask + amount + annualized_yield_proxy
M15_QUANT_DIM = OPTIONS_QUANT_DIM + RATE_QUANT_DIM + DIVIDEND_QUANT_DIM

# Project default (2026-06): defer paid/free options history; keep dim layout stable.
# Unmask when options EOD history is funded: TIRRA_MASK_M15_OPTIONS=0
_MASK_M15_OPTIONS_ENV = "TIRRA_MASK_M15_OPTIONS"


def m15_options_masked() -> bool:
    """True → options 7-dim block is zeroed (mask bit 0) for GNN training."""
    return os.getenv(_MASK_M15_OPTIONS_ENV, "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _clamp(x: float, lo: float = -10.0, hi: float = 10.0) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(lo, min(hi, x))


def _latest_obs(
    observations: list[dict[str, Any]],
    entity_id: str,
    observation_type: str,
    current_time: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_ts = -1.0
    for o in observations:
        if o.get("entity_id") != entity_id:
            continue
        if o.get("observation_type") != observation_type:
            continue
        ts = float(o.get("observed_at", 0.0))
        if ts > current_time or ts <= 0:
            continue
        if ts >= best_ts:
            best_ts = ts
            best = o
    return best


def _us_curve_at_time(
    observations: list[dict[str, Any]],
    current_time: float,
    country_eid: str | None = None,
) -> dict[str, float] | None:
    """Latest US Treasury curve snapshot with observed_at <= current_time."""
    best: dict[str, Any] | None = None
    best_ts = -1.0
    for o in observations:
        if o.get("observation_type") != "sovereign_yield":
            continue
        if country_eid is not None and o.get("entity_id") != country_eid:
            continue
        v = o.get("value", {})
        if not isinstance(v, dict):
            continue
        if v.get("source") != "us_treasury":
            continue
        ts = float(o.get("observed_at", 0.0))
        if ts > current_time or ts <= 0:
            continue
        if ts >= best_ts:
            best_ts = ts
            best = v
    if best is None:
        return None
    yields = best.get("yields") or {}
    return {
        "yield_2y": float(yields.get("2y") or 0.0),
        "yield_10y": float(yields.get("10y") or 0.0),
        "curve_2s10s": float(best.get("curve_2s10s") or 0.0),
        "curve_3m10y": float(best.get("curve_3m10y") or 0.0),
    }


def compute_options_quant_features(
    entity_id: str,
    observations: list[dict[str, Any]],
    current_time: float,
) -> list[float]:
    """7-dim block from latest options_chain_eod on this instrument."""
    if m15_options_masked():
        return [0.0] * OPTIONS_QUANT_DIM
    o = _latest_obs(observations, entity_id, "options_chain_eod", current_time)
    if o is None:
        return [0.0] * OPTIONS_QUANT_DIM
    v = o.get("value", {})
    if not isinstance(v, dict):
        return [0.0] * OPTIONS_QUANT_DIM

    call_iv = v.get("atm_call_iv")
    put_iv = v.get("atm_put_iv")
    call_f = _clamp(float(call_iv), 0.0, 3.0) if call_iv is not None else 0.0
    put_f = _clamp(float(put_iv), 0.0, 3.0) if put_iv is not None else 0.0
    pc = v.get("put_call_oi_ratio")
    pc_f = _clamp(float(pc), 0.0, 5.0) if pc is not None else 0.0
    oi = float(v.get("total_open_interest") or 0.0)
    log_oi = math.log1p(max(oi, 0.0))
    spot = _clamp(float(v.get("spot") or 0.0), 0.0, 1e6)
    iv_spread = put_f - call_f if (call_f > 0 or put_f > 0) else 0.0

    return [
        1.0,
        call_f,
        put_f,
        pc_f,
        _clamp(log_oi, 0.0, 20.0),
        spot,
        _clamp(iv_spread, -2.0, 2.0),
    ]


def compute_rate_quant_features(
    observations: list[dict[str, Any]],
    current_time: float,
    *,
    country_eid: str | None = None,
) -> list[float]:
    """5-dim US (or country) curve block — shared across instruments."""
    curve = _us_curve_at_time(observations, current_time, country_eid=country_eid)
    if curve is None:
        return [0.0] * RATE_QUANT_DIM
    return [
        1.0,
        _clamp(curve["yield_2y"], 0.0, 20.0),
        _clamp(curve["yield_10y"], 0.0, 20.0),
        _clamp(curve["curve_2s10s"], -5.0, 5.0),
        _clamp(curve["curve_3m10y"], -5.0, 5.0),
    ]


def compute_dividend_quant_features(
    entity_id: str,
    observations: list[dict[str, Any]],
    current_time: float,
    *,
    spot: float | None = None,
) -> list[float]:
    """3-dim block from latest dividend observation."""
    o = _latest_obs(observations, entity_id, "dividend", current_time)
    if o is None:
        return [0.0] * DIVIDEND_QUANT_DIM
    v = o.get("value", {})
    if not isinstance(v, dict):
        return [0.0] * DIVIDEND_QUANT_DIM
    amount = _clamp(float(v.get("amount") or 0.0), 0.0, 100.0)
    yield_proxy = 0.0
    if spot and spot > 1e-6 and amount > 0:
        yield_proxy = _clamp((amount * 4.0) / spot, 0.0, 0.5)
    return [1.0, amount, yield_proxy]


def _latest_close(
    entity_id: str,
    observations: list[dict[str, Any]],
    current_time: float,
) -> float | None:
    o = _latest_obs(observations, entity_id, "instrument_daily", current_time)
    if o is None:
        return None
    v = o.get("value", {})
    if not isinstance(v, dict):
        return None
    try:
        close = float(v.get("close"))
        return close if close > 0 and math.isfinite(close) else None
    except (TypeError, ValueError):
        return None


def compute_gnn_m15_features(
    entity_id: str,
    observations: list[dict[str, Any]],
    current_time: float,
    *,
    spot: float | None = None,
    us_country_eid: str | None = None,
) -> list[float]:
    """Full M15 quant vector for one instrument (length M15_QUANT_DIM)."""
    opts = compute_options_quant_features(entity_id, observations, current_time)
    rates = compute_rate_quant_features(
        observations, current_time, country_eid=us_country_eid
    )
    divs = compute_dividend_quant_features(
        entity_id, observations, current_time, spot=spot
    )
    vec = opts + rates + divs
    if len(vec) != M15_QUANT_DIM:
        return [0.0] * M15_QUANT_DIM
    return vec
