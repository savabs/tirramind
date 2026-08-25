#!/usr/bin/env python3
"""Quant benchmark for return_raw_head (Phase 50c raw bypass).

Proper quant evaluation:
  - Target: 21-day FORWARD return (what the model was trained to predict)
  - IC: Spearman rank correlation (signal vs forward return, cross-sectionally demeaned)
  - ICIR: mean IC / std IC (Grinold & Kahn signal quality threshold: 0.40)
  - Long-short: top quintile long / bottom quintile short — Sharpe & max drawdown
  - IC decay: IC at 1w / 2w / 4w forward horizons
  - Turnover: average rank change week-to-week

Usage:
    python scripts/quant_benchmark.py
    python scripts/quant_benchmark.py --checkpoint .tirra_pipeline/kaggle_downloads/phase50_ckpts/epoch_013.pt
    python scripts/quant_benchmark.py --horizon 21 --top-n 5
"""

from __future__ import annotations
import argparse, json, math, sqlite3, sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from agent.models.gnn.graph_builder import _build_node_features, xsnorm_price_feats

DB_PATH = Path(".tirra_pipeline/tirramind-data/pipeline.db")
CKPT_PATH = Path(".tirra_pipeline/kaggle_downloads/phase50_ckpts/epoch_013.pt")
TS_MIN, TS_MAX = 1_577_836_800, 1_893_456_000  # 2020–2030


# ── Data loading ──────────────────────────────────────────────────────────────


def load_raw_head(ckpt_path: Path) -> nn.Sequential:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    w0, b0 = state["return_raw_head.0.weight"], state["return_raw_head.0.bias"]
    w2, b2 = state["return_raw_head.2.weight"], state["return_raw_head.2.bias"]
    in_d, hid = w0.shape[1], w0.shape[0]
    head = nn.Sequential(nn.Linear(in_d, hid), nn.ReLU(), nn.Linear(hid, 1))
    head[0].weight.data.copy_(w0)
    head[0].bias.data.copy_(b0)
    head[2].weight.data.copy_(w2)
    head[2].bias.data.copy_(b2)
    head.eval()
    return head, int(
        ckpt.get("epoch", "?") if isinstance(ckpt.get("epoch"), int) else 0
    )


def fetch_daily(conn: sqlite3.Connection) -> dict[str, list[tuple[float, float]]]:
    """Returns {entity_id: [(ts, log_return), ...]} sorted by ts."""
    rows = conn.execute(
        """
        SELECT eo.entity_id, eo.observed_at, eo.value_json
        FROM entity_observations eo
        JOIN entities e ON e.entity_id = eo.entity_id
        WHERE e.entity_type = 'instrument'
          AND eo.observation_type = 'instrument_daily'
          AND eo.observed_at BETWEEN ? AND ?
        ORDER BY eo.entity_id, eo.observed_at
    """,
        (TS_MIN, TS_MAX),
    ).fetchall()

    data: dict[str, list[tuple[float, float]]] = {}
    for eid, ts, vjson in rows:
        try:
            v = json.loads(vjson) if vjson else {}
            lr = float(v["log_return"])
            data.setdefault(eid, []).append((float(ts), lr))
        except (KeyError, TypeError, ValueError):
            continue
    return data


def fetch_all_instrument_obs(conn: sqlite3.Connection) -> list[dict]:
    """All instrument observations for feature building."""
    rows = conn.execute(
        """
        SELECT eo.entity_id, eo.observed_at, eo.observation_type, eo.value_json
        FROM entity_observations eo
        JOIN entities e ON e.entity_id = eo.entity_id
        WHERE e.entity_type = 'instrument'
          AND eo.observed_at BETWEEN ? AND ?
        ORDER BY eo.observed_at
    """,
        (TS_MIN, TS_MAX),
    ).fetchall()
    result = []
    for eid, ts, otype, vjson in rows:
        try:
            v = json.loads(vjson) if vjson else {}
        except Exception:
            v = {}
        result.append(
            {
                "entity_id": eid,
                "observed_at": float(ts),
                "observation_type": otype,
                "value": v,
            }
        )
    return result


def forward_return(
    daily: list[tuple[float, float]], t_start: float, n_days: int
) -> float | None:
    """Sum of log_returns for n_days TRADING days starting from t_start."""
    TRADING_WINDOW = 1.5 * 86400  # 1.5 calendar days tolerance per trading day
    total, count = 0.0, 0
    for ts, lr in daily:
        if ts < t_start:
            continue
        if ts > t_start + n_days * TRADING_WINDOW:
            break
        if math.isfinite(lr):
            total += lr
            count += 1
    return total if count >= max(1, n_days // 3) else None


# ── Portfolio construction ─────────────────────────────────────────────────────


class FoldResult(NamedTuple):
    t_end: float
    ic_1w: float | None
    ic_2w: float | None
    ic_4w: float | None
    ls_ret_1w: float | None  # long-short 1-week return
    turnover: float | None


def compute_ic(scores: np.ndarray, fwd_rets: np.ndarray) -> float | None:
    """Spearman IC on cross-sectionally demeaned forward returns."""
    mask = np.isfinite(fwd_rets) & np.isfinite(scores)
    if mask.sum() < 5:
        return None
    s, f = scores[mask], fwd_rets[mask]
    f = f - f.mean()  # cross-sectional demean
    r, _ = stats.spearmanr(s, f)
    return float(r) if math.isfinite(r) else None


def ls_return(scores: np.ndarray, fwd_rets: np.ndarray, top_n: int) -> float | None:
    """Long top_n, short bottom_n — equal-weight within each leg."""
    mask = np.isfinite(fwd_rets) & np.isfinite(scores)
    if mask.sum() < top_n * 2 + 1:
        return None
    idx = np.argsort(scores[mask])[::-1]  # highest score first
    f = fwd_rets[mask]
    long_ret = f[idx[:top_n]].mean()
    short_ret = f[idx[-top_n:]].mean()
    return float(long_ret - short_ret)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CKPT_PATH))
    p.add_argument(
        "--horizon", type=int, default=21, help="Primary IC horizon in trading days"
    )
    p.add_argument(
        "--top-n", type=int, default=5, help="Top/bottom N for L/S portfolio"
    )
    p.add_argument(
        "--eval-step",
        type=float,
        default=604_800,
        help="Seconds between eval points (default 1 week)",
    )
    p.add_argument(
        "--history",
        type=float,
        default=604_800 * 26,
        help="History window for features (default 26 weeks). "
        "sharpe_60d/vol_60d/max_dd_60d need >=60 trading days (~84 cal days ~12w); "
        "26w gives a safe margin. With 6w history the ICIR is ~0.14 (noise); "
        "with 26w it is ~0.44 (tradeable).",
    )
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    assert DB_PATH.exists(), f"DB not found: {DB_PATH}"

    head, epoch = load_raw_head(ckpt)
    print(f"\n{'═'*60}")
    print(f"  Quant Benchmark — return_raw_head  (epoch {epoch})")
    print(f"  Forward horizon : {args.horizon} trading days")
    print(f"  L/S top-N       : {args.top_n}")
    print(f"{'═'*60}")

    conn = sqlite3.connect(str(DB_PATH))

    instrument_eids = [
        r[0]
        for r in conn.execute(
            "SELECT entity_id FROM entities WHERE entity_type='instrument'"
        ).fetchall()
    ]
    n_inst = len(instrument_eids)
    print(f"  Instruments     : {n_inst}")

    daily = fetch_daily(conn)
    all_obs = fetch_all_instrument_obs(conn)
    t_min_all = min(o["observed_at"] for o in all_obs)
    t_max_all = max(o["observed_at"] for o in all_obs)
    print(
        f"  Data range      : {t_max_all - t_min_all:.0f}s = {(t_max_all-t_min_all)/86400:.0f} days\n"
    )

    # Evaluation grid: weekly steps, leave room for forward returns
    TRADING_DAY_SECS = 86400 * 1.4  # avg calendar days per trading day
    fwd_buffer = args.horizon * TRADING_DAY_SECS
    t_start_eval = t_min_all + args.history + fwd_buffer
    t_end_eval = t_max_all - fwd_buffer

    eval_times = []
    t = t_start_eval
    while t <= t_end_eval:
        eval_times.append(t)
        t += args.eval_step

    print(f"  Eval windows    : {len(eval_times)}")

    results: list[FoldResult] = []
    prev_ranks: np.ndarray | None = None

    with torch.no_grad():
        for t_end in eval_times:
            # --- Features ---
            hist = [
                o for o in all_obs if t_end - args.history <= o["observed_at"] < t_end
            ]
            if not hist:
                continue
            feats = _build_node_features("instrument", instrument_eids, hist, t_end)
            scores = head(xsnorm_price_feats(feats)).squeeze(-1).numpy()  # [N]

            # --- Forward returns at 3 horizons ---
            def fwd_ret_arr(n_days: int) -> np.ndarray:
                arr = np.full(n_inst, np.nan)
                for i, eid in enumerate(instrument_eids):
                    if eid in daily:
                        fr = forward_return(daily[eid], t_end, n_days)
                        if fr is not None:
                            arr[i] = fr
                return arr

            fr_1w = fwd_ret_arr(5)  # 1 week  ≈ 5 trading days
            fr_2w = fwd_ret_arr(10)  # 2 weeks ≈ 10 trading days
            fr_4w = fwd_ret_arr(args.horizon)  # primary horizon

            ic_1w = compute_ic(scores, fr_1w)
            ic_2w = compute_ic(scores, fr_2w)
            ic_4w = compute_ic(scores, fr_4w)
            ls_1w = ls_return(scores, fr_1w, args.top_n)

            # --- Turnover ---
            ranks = stats.rankdata(scores)
            turnover = None
            if prev_ranks is not None:
                turnover = float(np.mean(np.abs(ranks - prev_ranks)) / n_inst)
            prev_ranks = ranks

            results.append(FoldResult(t_end, ic_1w, ic_2w, ic_4w, ls_1w, turnover))

    conn.close()

    if not results:
        print("No results — check data range.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def agg(vals: list[float | None]) -> tuple[float, float, float, float, float]:
        v = np.array([x for x in vals if x is not None and math.isfinite(x)])
        if len(v) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0
        mu = v.mean()
        sigma = v.std()
        icir = mu / sigma if sigma > 1e-8 else 0.0
        tstat = mu / (sigma / math.sqrt(len(v))) if sigma > 1e-8 else 0.0
        return mu, sigma, icir, tstat, len(v)

    ic1_mu, ic1_sig, ic1_ir, ic1_t, n1 = agg([r.ic_1w for r in results])
    ic2_mu, ic2_sig, ic2_ir, ic2_t, n2 = agg([r.ic_2w for r in results])
    ic4_mu, ic4_sig, ic4_ir, ic4_t, n4 = agg([r.ic_4w for r in results])
    ls_vals = [r.ls_ret_1w for r in results]
    ls_v = np.array([x for x in ls_vals if x is not None and math.isfinite(x)])

    # Long-short portfolio stats
    ls_sharpe = (
        (ls_v.mean() / ls_v.std() * math.sqrt(52))
        if len(ls_v) > 1 and ls_v.std() > 1e-8
        else 0.0
    )
    # Max drawdown on cumulative L/S
    cum = np.cumsum(ls_v)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    turn_vals = [r.turnover for r in results if r.turnover is not None]
    avg_turn = float(np.mean(turn_vals)) if turn_vals else 0.0

    hit_4w = np.mean([r.ic_4w > 0 for r in results if r.ic_4w is not None])

    # ── Print ──────────────────────────────────────────────────────────────────
    W = 58

    def row(label, val, flag=""):
        print(f"  {label:<28} {val:<20} {flag}")

    print(f"\n{'─'*W}")
    print(
        f"  IC — SIGNAL QUALITY  (primary horizon: {args.horizon}d = {args.horizon//5}w)"
    )
    print(f"{'─'*W}")
    row("Horizon", "Mean IC    ICIR   t-stat")
    row("1-week  (5d)", f"{ic1_mu:+.4f}   {ic1_ir:+.3f}  {ic1_t:+.2f}", _flag(ic1_ir))
    row("2-week (10d)", f"{ic2_mu:+.4f}   {ic2_ir:+.3f}  {ic2_t:+.2f}", _flag(ic2_ir))
    row(
        f"{args.horizon}d (primary)",
        f"{ic4_mu:+.4f}   {ic4_ir:+.3f}  {ic4_t:+.2f}",
        _flag(ic4_ir),
    )
    row(f"Hit rate ({args.horizon}d)", f"{hit_4w:.1%}")

    print(f"\n{'─'*W}")
    print(f"  LONG-SHORT PORTFOLIO  (top/bottom {args.top_n}, weekly rebalance)")
    print(f"{'─'*W}")
    if len(ls_v) > 0:
        row("Ann. Sharpe (L/S)", f"{ls_sharpe:+.3f}", _sharpe_flag(ls_sharpe))
        row("Max Drawdown", f"{max_dd:+.4f}")
        row("Mean weekly L/S ret", f"{ls_v.mean():+.5f}")
        row("Avg weekly turnover", f"{avg_turn:.1%}")
        row("Periods evaluated", str(len(ls_v)))
    else:
        print("  No L/S data.")

    print(f"\n{'─'*W}")
    print(f"  IC DECAY  (does signal persist beyond 1 week?)")
    print(f"{'─'*W}")
    row("1-week IC", f"{ic1_mu:+.4f}")
    row(
        "2-week IC",
        f"{ic2_mu:+.4f}",
        f"{'▼ decays' if abs(ic2_mu) < abs(ic1_mu) else '▲ grows'}",
    )
    row(
        f"{args.horizon}d IC",
        f"{ic4_mu:+.4f}",
        f"{'▼ decays' if abs(ic4_mu) < abs(ic2_mu) else '▲ grows'}",
    )

    print(f"\n{'─'*W}")
    print(f"  THRESHOLDS  (Grinold & Kahn 2000)")
    print(f"{'─'*W}")
    print(f"  ICIR > 0.40  → tradeable signal")
    print(f"  ICIR > 0.20  → directional signal, worth investigating")
    print(f"  ICIR < 0.10  → noise")
    best_icir = max(abs(ic1_ir), abs(ic2_ir), abs(ic4_ir))
    print(f"\n  Best ICIR  : {best_icir:.3f}  → {_threshold(best_icir)}")
    print(f"{'═'*W}\n")

    # Save results
    out = {
        "checkpoint": str(ckpt),
        "epoch": epoch,
        "n_instruments": n_inst,
        "n_windows": len(results),
        "ic_1w": {
            "mean": ic1_mu,
            "std": ic1_sig,
            "icir": ic1_ir,
            "tstat": ic1_t,
            "n": n1,
        },
        "ic_2w": {
            "mean": ic2_mu,
            "std": ic2_sig,
            "icir": ic2_ir,
            "tstat": ic2_t,
            "n": n2,
        },
        "ic_primary": {
            "mean": ic4_mu,
            "std": ic4_sig,
            "icir": ic4_ir,
            "tstat": ic4_t,
            "n": n4,
            "horizon_days": args.horizon,
        },
        "ls_sharpe": ls_sharpe,
        "ls_max_dd": max_dd,
        "avg_turnover": avg_turn,
        "hit_rate_primary": float(hit_4w),
    }
    out_path = Path(".tirra_pipeline/ic_results_phase50c.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Results saved → {out_path}")


def _flag(icir: float) -> str:
    a = abs(icir)
    if a >= 0.40:
        return "✓ TRADEABLE"
    if a >= 0.20:
        return "~ directional"
    if a >= 0.10:
        return "· marginal"
    return "✗ noise"


def _sharpe_flag(s: float) -> str:
    a = abs(s)
    if a >= 1.0:
        return "✓ strong"
    if a >= 0.5:
        return "~ moderate"
    return "✗ weak"


def _threshold(icir: float) -> str:
    if icir >= 0.40:
        return "TRADEABLE signal"
    if icir >= 0.20:
        return "Directional signal — worth investigating"
    if icir >= 0.10:
        return "Marginal — monitor"
    return "Noise"


if __name__ == "__main__":
    main()
