"""
Standalone microstructure signals from pipeline observations (no GNN).

Uses instrument_daily (close, log_return, volume). When high/low are absent,
spread estimators use close-derived range proxies (documented in output).

Hero readouts for N1+N4 playground (see [[n1_n4_playground_spec]]):
  - liquidity: spread_roll, spread_cs_proxy
  - flow pressure: signed_flow_z (daily proxy for OFI)
  - impact: kyle_lambda_daily
  - stress: vol_of_vol, range_pct

Tick-level OFI/VPIN require trade data — not computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch

from agent.quant.microstructure import (
    KyleLambdaEstimator,
    MicrostructureFeatureExtractor,
    SpreadEstimator,
)

# M9 vector for graph_builder — must match MICROSTRUCTURE_DIM in graph_builder.py
GNN_MICRO_DIM = 11


@dataclass(frozen=True)
class MicroSnapshot:
    """Latest microstructure readout for one instrument at one as-of time."""

    entity_id: str
    as_of_ts: float
    n_days: int

    spread_roll: float
    spread_cs_proxy: float  # Corwin-Schultz on proxy H/L from rolling vol
    signed_flow: float  # last-day sign(ret) * volume
    signed_flow_z: float  # z-score vs trailing 60d
    kyle_lambda: float  # rolling OLS on daily changes (last window)
    vol_20d: float
    vol_of_vol: float  # std of rolling 20d vol
    range_pct: float  # proxy intraday range / close

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_instrument_daily(
    observations: list[dict[str, Any]],
    entity_id: str,
    *,
    until_ts: float | None = None,
) -> list[dict[str, float]]:
    """Pull instrument_daily bars for one entity, sorted by time."""
    rows: list[dict[str, float]] = []
    for o in observations:
        if o.get("entity_id") != entity_id:
            continue
        if o.get("observation_type") != "instrument_daily":
            continue
        ts = float(o.get("observed_at", 0.0))
        if until_ts is not None and ts > until_ts:
            continue
        v = o.get("value", {})
        if not isinstance(v, dict):
            continue
        close = v.get("close")
        if close is None:
            continue
        try:
            rows.append(
                {
                    "ts": ts,
                    "close": float(close),
                    "log_return": float(v.get("log_return") or 0.0),
                    "volume": float(v.get("volume") or 0.0),
                }
            )
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda x: x["ts"])
    return rows


def _proxy_high_low(
    closes: np.ndarray,
    log_returns: np.ndarray,
    window: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Proxy daily high/low from close and rolling vol (no tick data)."""
    n = len(closes)
    high = np.empty(n)
    low = np.empty(n)
    for i in range(n):
        w = log_returns[max(0, i - window + 1) : i + 1]
        sigma = float(np.std(w)) if len(w) >= 2 else 0.01
        sigma = max(sigma, 1e-4)
        high[i] = closes[i] * math.exp(sigma)
        low[i] = closes[i] * math.exp(-sigma)
    return high, low


def compute_micro_snapshot(
    entity_id: str,
    observations: list[dict[str, Any]],
    *,
    until_ts: float | None = None,
    min_days: int = 30,
) -> MicroSnapshot | None:
    """
    Compute latest microstructure snapshot for one instrument.

    Returns None if fewer than min_days of instrument_daily bars.
    """
    daily = extract_instrument_daily(observations, entity_id, until_ts=until_ts)
    if len(daily) < min_days:
        return None

    closes = np.array([d["close"] for d in daily], dtype=np.float64)
    log_rets = np.array([d["log_return"] for d in daily], dtype=np.float64)
    volumes = np.array([d["volume"] for d in daily], dtype=np.float64)
    ts_last = daily[-1]["ts"]

    spread_est = SpreadEstimator()
    # Match GNN path: Roll on log(close[t]/close[t-1]), not DB log_return field
    if len(closes) >= 2:
        roll_rets = np.log(closes[1:] / closes[:-1])
        spread_roll = float(
            spread_est.roll_measure(torch.tensor(roll_rets, dtype=torch.float32)).item()
        )
    else:
        spread_roll = 0.0

    high, low = _proxy_high_low(closes, log_rets)
    cs = spread_est.corwin_schultz(
        torch.tensor(high, dtype=torch.float32),
        torch.tensor(low, dtype=torch.float32),
    )
    spread_cs_proxy = float(cs[-1].item()) if len(cs) > 0 else 0.0

    signed_flows = np.sign(log_rets) * volumes
    signed_flow = float(signed_flows[-1])
    trail = signed_flows[-60:]
    sf_mean = float(trail.mean())
    sf_std = float(trail.std())
    signed_flow_z = (
        (signed_flow - sf_mean) / sf_std if sf_std > 1e-8 else 0.0
    )

    w = min(60, len(closes) - 1)
    if w >= 10:
        lam_arr = KyleLambdaEstimator().estimate_lambda(
            closes, signed_flows, window_hours=w
        )
        kyle_lambda = float(lam_arr[-1]) if len(lam_arr) > 0 else 0.0
    else:
        kyle_lambda = 0.0

    vol_20d = float(np.std(log_rets[-20:]) * math.sqrt(252)) if len(log_rets) >= 20 else 0.0

    rolling_vols: list[float] = []
    for i in range(20, len(log_rets) + 1):
        rolling_vols.append(float(np.std(log_rets[i - 20 : i]) * math.sqrt(252)))
    vol_of_vol = float(np.std(rolling_vols)) if len(rolling_vols) >= 5 else 0.0

    range_pct = float((high[-1] - low[-1]) / closes[-1]) if closes[-1] > 0 else 0.0

    return MicroSnapshot(
        entity_id=entity_id,
        as_of_ts=ts_last,
        n_days=len(daily),
        spread_roll=spread_roll,
        spread_cs_proxy=spread_cs_proxy,
        signed_flow=signed_flow,
        signed_flow_z=signed_flow_z,
        kyle_lambda=kyle_lambda,
        vol_20d=vol_20d,
        vol_of_vol=vol_of_vol,
        range_pct=range_pct,
    )


def _clamp_micro_scalar(x: float, lo: float = -10.0, hi: float = 10.0) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(lo, min(hi, x))


def compute_gnn_micro_features(
    entity_id: str,
    observations: list[dict[str, Any]],
    current_time: float,
    *,
    min_days: int = 30,
) -> list[float]:
    """
    11-dim M9 feature vector for HetTGN instrument nodes (no forward bias).

    Layout (matches graph_builder.MICROSTRUCTURE_DIM):
      spread_cs, spread_roll, ofi_zscore, vpin,
      vpin_regime (3 one-hot), kyle_lambda, lambda_regime (3 one-hot)

    Daily bars only: synthetic per-day trades feed VPIN/OFI; signed_flow_z
  is used when tick-style OFI z-scores are unavailable (short history).
    """
    daily = extract_instrument_daily(
        observations, entity_id, until_ts=current_time
    )
    if len(daily) < min_days:
        return [0.0] * GNN_MICRO_DIM

    closes = np.array([d["close"] for d in daily], dtype=np.float64)
    log_rets = np.array([d["log_return"] for d in daily], dtype=np.float64)
    volumes = np.array([d["volume"] for d in daily], dtype=np.float64)
    n = len(closes)

    high, low = _proxy_high_low(closes, log_rets)
    close_t = torch.tensor(closes, dtype=torch.float32)
    high_t = torch.tensor(high, dtype=torch.float32)
    low_t = torch.tensor(low, dtype=torch.float32)

    signed_flows = np.sign(log_rets) * volumes
    trail = signed_flows[-60:]
    sf_std = float(trail.std())
    sf_mean = float(trail.mean())
    signed_flow_z = (
        (float(signed_flows[-1]) - sf_mean) / sf_std if sf_std > 1e-8 else 0.0
    )

    w = min(60, n - 1)
    lam_ts = torch.zeros(n, dtype=torch.float32)
    if w >= 10:
        lam_arr = KyleLambdaEstimator().estimate_lambda(
            closes, signed_flows, window_hours=w
        )
        if len(lam_arr) > 0:
            lam_ts[-len(lam_arr) :] = torch.tensor(lam_arr, dtype=torch.float32)

    trades = [
        (float(closes[i]), float(max(volumes[i], 1.0)), i) for i in range(n)
    ]
    med_vol = float(np.median(volumes[volumes > 0])) if np.any(volumes > 0) else 1.0

    extractor = MicrostructureFeatureExtractor()
    extractor.vpin_calculator.bucket_volume = max(med_vol, 1.0)
    extractor.vpin_calculator.n_buckets = min(50, max(10, n // 5))

    ohlcv = {"high": high_t, "low": low_t, "close": close_t}
    feats = extractor(
        ohlcv,
        trades=trades,
        precomputed_lambda=lam_ts,
    )

    def _scalar(key: str, default: float = 0.0) -> float:
        if key not in feats:
            return default
        t = feats[key]
        if t.dim() == 0:
            return float(t.item())
        return float(t[-1].item())

    def _onehot3(key: str) -> list[float]:
        if key not in feats or feats[key].numel() == 0:
            return [1.0, 0.0, 0.0]
        row = feats[key][-1]
        return [float(row[i].item()) for i in range(min(3, row.numel()))]

    spread_cs = _scalar("spread_cs", 0.0)
    spread_roll = _scalar("spread_roll", 0.0)
    ofi_z = _scalar("ofi_zscore", 0.0)
    if abs(ofi_z) < 1e-8:
        ofi_z = signed_flow_z
    vpin = _scalar("vpin", 0.0)
    vpin_reg = _onehot3("vpin_regime")
    kyle_l = _scalar("kyle_lambda", float(lam_ts[-1].item()))
    lam_reg = _onehot3("lambda_regime")

    vec = (
        [_clamp_micro_scalar(spread_cs)]
        + [_clamp_micro_scalar(spread_roll)]
        + [_clamp_micro_scalar(ofi_z)]
        + [_clamp_micro_scalar(vpin, 0.0, 1.0)]
        + vpin_reg
        + [_clamp_micro_scalar(kyle_l)]
        + lam_reg
    )
    if len(vec) != GNN_MICRO_DIM:
        return [0.0] * GNN_MICRO_DIM
    return vec


def rank_snapshots(
    snapshots: list[MicroSnapshot],
    key: str = "signed_flow_z",
) -> list[MicroSnapshot]:
    """Sort snapshots by a field (descending absolute z for flow)."""
    if key == "signed_flow_z":
        return sorted(snapshots, key=lambda s: abs(s.signed_flow_z), reverse=True)
    return sorted(snapshots, key=lambda s: getattr(s, key), reverse=True)


# ── Alert thresholds (N1 micro probes — daily proxies) ─────────────────


@dataclass(frozen=True)
class MicroThresholds:
    """Configurable cutoffs for standalone micro alerts."""

    flow_z_watch: float = 1.5
    flow_z_strong: float = 2.0
    spread_roll_watch: float = 0.01
    spread_roll_strong: float = 0.02
    vol_of_vol_watch: float = 0.15
    vol_of_vol_strong: float = 0.25
    vol_20d_watch: float = 0.50
    vol_20d_strong: float = 0.80


@dataclass(frozen=True)
class MicroAlert:
    """One fired alert on an instrument."""

    code: str
    severity: str  # WATCH | STRONG
    metric: str
    value: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InstrumentMicroPanel:
    """Fused N1 readout: daily micro + optional CFTC positioning."""

    ticker: str
    entity_id: str
    snapshot: MicroSnapshot
    alerts: tuple[MicroAlert, ...]
    cftc_mm_pct_52w_rank: float | None = None
    cftc_positioning_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ticker": self.ticker,
            "entity_id": self.entity_id,
            "as_of_ts": self.snapshot.as_of_ts,
            "micro": self.snapshot.to_dict(),
            "alerts": [a.to_dict() for a in self.alerts],
        }
        if self.cftc_mm_pct_52w_rank is not None:
            d["cftc_mm_pct_52w_rank"] = self.cftc_mm_pct_52w_rank
            d["cftc_mm_pct_52w_rank_pct"] = round(
                normalize_cftc_rank(self.cftc_mm_pct_52w_rank) * 100.0, 1
            )
        if self.cftc_positioning_label is not None:
            d["cftc_positioning_label"] = self.cftc_positioning_label
        return d


def evaluate_micro_alerts(
    snap: MicroSnapshot,
    thresholds: MicroThresholds | None = None,
) -> list[MicroAlert]:
    """Classify daily micro proxies into WATCH/STRONG alerts."""
    th = thresholds or MicroThresholds()
    alerts: list[MicroAlert] = []

    az = abs(snap.signed_flow_z)
    if az >= th.flow_z_strong:
        alerts.append(
            MicroAlert(
                code="FLOW_IMBALANCE",
                severity="STRONG",
                metric="signed_flow_z",
                value=snap.signed_flow_z,
                message=f"Daily signed-flow z={snap.signed_flow_z:.2f} (proxy OFI)",
            )
        )
    elif az >= th.flow_z_watch:
        alerts.append(
            MicroAlert(
                code="FLOW_IMBALANCE",
                severity="WATCH",
                metric="signed_flow_z",
                value=snap.signed_flow_z,
                message=f"Elevated signed-flow z={snap.signed_flow_z:.2f}",
            )
        )

    if snap.spread_roll >= th.spread_roll_strong:
        alerts.append(
            MicroAlert(
                code="WIDE_SPREAD",
                severity="STRONG",
                metric="spread_roll",
                value=snap.spread_roll,
                message=f"Roll spread estimate={snap.spread_roll:.4f}",
            )
        )
    elif snap.spread_roll >= th.spread_roll_watch:
        alerts.append(
            MicroAlert(
                code="WIDE_SPREAD",
                severity="WATCH",
                metric="spread_roll",
                value=snap.spread_roll,
                message=f"Elevated Roll spread={snap.spread_roll:.4f}",
            )
        )

    if snap.vol_of_vol >= th.vol_of_vol_strong:
        alerts.append(
            MicroAlert(
                code="VOL_OF_VOL",
                severity="STRONG",
                metric="vol_of_vol",
                value=snap.vol_of_vol,
                message=f"Vol-of-vol={snap.vol_of_vol:.3f}",
            )
        )
    elif snap.vol_of_vol >= th.vol_of_vol_watch:
        alerts.append(
            MicroAlert(
                code="VOL_OF_VOL",
                severity="WATCH",
                metric="vol_of_vol",
                value=snap.vol_of_vol,
                message=f"Elevated vol-of-vol={snap.vol_of_vol:.3f}",
            )
        )

    if snap.vol_20d >= th.vol_20d_strong:
        alerts.append(
            MicroAlert(
                code="HIGH_REALIZED_VOL",
                severity="STRONG",
                metric="vol_20d",
                value=snap.vol_20d,
                message=f"20d ann. vol={snap.vol_20d:.2f}",
            )
        )
    elif snap.vol_20d >= th.vol_20d_watch:
        alerts.append(
            MicroAlert(
                code="HIGH_REALIZED_VOL",
                severity="WATCH",
                metric="vol_20d",
                value=snap.vol_20d,
                message=f"Elevated 20d vol={snap.vol_20d:.2f}",
            )
        )

    return alerts


def normalize_cftc_rank(rank: float) -> float:
    """Return 52w percentile rank on [0, 1] (DB stores [0, 1] per CFTC derived features)."""
    if rank > 1.0:
        return rank / 100.0
    return rank


def classify_cftc_positioning(rank: float) -> str:
    """Positioning extreme labels on 52w percentile rank (0–1 scale)."""
    r = normalize_cftc_rank(rank)
    if r >= 0.80:
        return "CROWDED_LONG"
    if r <= 0.20:
        return "CROWDED_SHORT"
    if r >= 0.65:
        return "APPROACHING_LONG"
    if r <= 0.35:
        return "APPROACHING_SHORT"
    return "NEUTRAL"


def build_instrument_panel(
    ticker: str,
    entity_id: str,
    observations: list[dict[str, Any]],
    *,
    cftc_rank: float | None = None,
    min_days: int = 30,
    thresholds: MicroThresholds | None = None,
) -> InstrumentMicroPanel | None:
    """Micro snapshot + alerts + optional CFTC rank for one instrument."""
    snap = compute_micro_snapshot(entity_id, observations, min_days=min_days)
    if snap is None:
        return None
    alerts = evaluate_micro_alerts(snap, thresholds)
    label = classify_cftc_positioning(cftc_rank) if cftc_rank is not None else None
    return InstrumentMicroPanel(
        ticker=ticker,
        entity_id=entity_id,
        snapshot=snap,
        alerts=tuple(alerts),
        cftc_mm_pct_52w_rank=cftc_rank,
        cftc_positioning_label=label,
    )


def list_instruments_by_asset_class(
    entities: list[dict[str, Any]],
    asset_class: str,
) -> list[tuple[str, str]]:
    """Return (entity_id, ticker) for instruments with given asset_class metadata."""
    out: list[tuple[str, str]] = []
    for e in entities:
        if e.get("entity_type") != "instrument":
            continue
        meta = e.get("metadata") or {}
        if meta.get("asset_class") != asset_class:
            continue
        ticker = str(meta.get("ticker") or e.get("canonical_name", e["entity_id"]))
        out.append((e["entity_id"], ticker))
    return out


def load_cftc_ranks_by_ticker(
    observations: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Latest mm_pct_52w_rank per instrument ticker via cftc_contract metadata.

    Uses futures_positioning_derived on cftc_contract entities; maps cftc_code → ticker.
    """
    import json

    from agent.tools.instrument_universe import cftc_code_to_ticker

    code_to_ticker = cftc_code_to_ticker()
    code_by_contract_eid: dict[str, str] = {}
    for e in entities:
        if e.get("entity_type") != "cftc_contract":
            continue
        meta = e.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        code = meta.get("cftc_code")
        if code:
            code_by_contract_eid[e["entity_id"]] = str(code)

    latest: dict[str, tuple[float, float]] = {}  # contract_eid -> (ts, rank)
    for o in observations:
        if o.get("observation_type") != "futures_positioning_derived":
            continue
        eid = o.get("entity_id", "")
        if eid not in code_by_contract_eid:
            continue
        ts = float(o.get("observed_at", 0.0))
        v = o.get("value", {})
        if not isinstance(v, dict):
            continue
        rank = v.get("cftc_mm_pct_52w_rank")
        if rank is None:
            continue
        try:
            rank_f = float(rank)
        except (TypeError, ValueError):
            continue
        prev = latest.get(eid)
        if prev is None or ts >= prev[0]:
            latest[eid] = (ts, rank_f)

    out: dict[str, float] = {}
    for contract_eid, (_, rank_f) in latest.items():
        code = code_by_contract_eid.get(contract_eid)
        if not code:
            continue
        ticker = code_to_ticker.get(code)
        if ticker:
            out[ticker] = rank_f
    return out


def panels_with_alerts(
    panels: list[InstrumentMicroPanel],
    *,
    min_severity: str | None = None,
) -> list[InstrumentMicroPanel]:
    """Filter to instruments that have at least one alert."""
    severity_order = {"WATCH": 1, "STRONG": 2}
    min_level = severity_order.get(min_severity or "WATCH", 1)

    def _has_alert(p: InstrumentMicroPanel) -> bool:
        return any(severity_order.get(a.severity, 0) >= min_level for a in p.alerts)

    return [p for p in panels if _has_alert(p)]
