#!/usr/bin/env python3
"""Quick IC diagnostic — Spearman IC from return_raw_head, no full Trainer needed.

Directly imports graph_builder feature logic and queries the local pipeline.db.
Bypasses the Trainer/PipelineStore setup entirely.

Usage:
    python scripts/ic_check.py
    python scripts/ic_check.py --checkpoint .tirra_pipeline/kaggle_downloads/phase50_ckpts/epoch_013.pt
    python scripts/ic_check.py --windows 40 --window-size 604800
"""

from __future__ import annotations
import argparse, json, math, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from agent.models.gnn.graph_builder import (
    _build_node_features,
    _compute_price_features,
    ENTITY_TYPES,
    BASE_FEAT_DIM,
    PRICE_FEAT_DIM,
    xsnorm_price_feats,
)

DB_PATH = Path(".tirra_pipeline/tirramind-data/pipeline.db")
CKPT_PATH = Path(".tirra_pipeline/kaggle_downloads/phase50_ckpts/epoch_013.pt")
WINDOW = 604_800  # 1 week in seconds
TS_MIN, TS_MAX = 1_577_836_800, 1_893_456_000  # 2020-2030 valid range


def load_raw_head(ckpt_path: Path) -> nn.Sequential:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    w0 = state["return_raw_head.0.weight"]
    b0 = state["return_raw_head.0.bias"]
    w2 = state["return_raw_head.2.weight"]
    b2 = state["return_raw_head.2.bias"]
    in_dim = w0.shape[1]
    hidden_dim = w0.shape[0]
    head = nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
    )
    head[0].weight.data.copy_(w0)
    head[0].bias.data.copy_(b0)
    head[2].weight.data.copy_(w2)
    head[2].bias.data.copy_(b2)
    head.eval()
    print(f"  raw_head: {in_dim}→{hidden_dim}→1  (output bias={b2.item():.4f})")
    return head


def fetch_instrument_obs(
    conn: sqlite3.Connection, t_start: float, t_end: float
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT eo.entity_id, eo.observed_at, eo.observation_type, eo.value_json
        FROM entity_observations eo
        JOIN entities e ON e.entity_id = eo.entity_id
        WHERE e.entity_type = 'instrument'
          AND eo.observed_at >= ? AND eo.observed_at < ?
          AND eo.observed_at BETWEEN ? AND ?
    """,
        (t_start, t_end, TS_MIN, TS_MAX),
    ).fetchall()
    obs = []
    for eid, ts, otype, vjson in rows:
        try:
            v = json.loads(vjson) if vjson else {}
        except Exception:
            v = {}
        obs.append(
            {"entity_id": eid, "observed_at": ts, "observation_type": otype, "value": v}
        )
    return obs


def get_instrument_entities(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT entity_id FROM entities WHERE entity_type='instrument'"
    ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CKPT_PATH))
    p.add_argument(
        "--windows",
        type=int,
        default=40,
        help="Number of recent weekly windows to evaluate (default 40)",
    )
    p.add_argument("--window-size", type=float, default=WINDOW)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    assert DB_PATH.exists(), f"DB not found: {DB_PATH}"

    print(f"\nIC Diagnostic — epoch 13 return_raw_head")
    print(f"  checkpoint : {ckpt}")
    print(f"  database   : {DB_PATH}")
    head = load_raw_head(ckpt)

    conn = sqlite3.connect(str(DB_PATH))
    instrument_eids = get_instrument_entities(conn)
    print(f"  instruments: {len(instrument_eids)}")

    # Find time range (filtered to valid timestamps)
    t_min, t_max = conn.execute(
        """
        SELECT MIN(observed_at), MAX(observed_at) FROM entity_observations
        WHERE observed_at BETWEEN ? AND ?
          AND observation_type = 'instrument_daily'
    """,
        (TS_MIN, TS_MAX),
    ).fetchone()
    print(f"  data range : {t_min:.0f} → {t_max:.0f}  ({(t_max-t_min)/86400:.0f} days)")

    win_size = args.window_size
    n_windows = args.windows
    # Define windows: each eval window uses obs in [t_end - win_size, t_end]
    # Target return: instrument_return obs in the NEXT window [t_end, t_end + win_size]
    windows_end = [t_max - win_size * i for i in range(n_windows, 0, -1)]

    ics: list[float] = []
    skipped = 0

    with torch.no_grad():
        for t_end in windows_end:
            t_start = t_end - win_size
            t_next_end = t_end + win_size

            # Features from history window — needs 26 weeks so that sharpe_60d /
            # vol_60d / max_dd_60d (all require 60 trading days ≈ 84 cal days)
            # are computed with sufficient data.  With only 5 weeks the IC drops
            # from ICIR≈0.44 to ICIR≈0.14 (noise level).
            hist_obs = fetch_instrument_obs(conn, t_start - win_size * 26, t_end)
            if not hist_obs:
                skipped += 1
                continue

            # Build feature matrix for all instruments
            feats = _build_node_features(
                entity_type="instrument",
                entity_ids=instrument_eids,
                observations=hist_obs,
                current_time=t_end,
            )  # shape [N, feat_dim]

            # Forward through raw head (price features cross-sectionally z-scored)
            scores = head(xsnorm_price_feats(feats)).squeeze(-1)  # [N]

            # Collect next-window returns (instrument_daily has log_return for all history)
            next_obs = fetch_instrument_obs(conn, t_end, t_next_end)
            return_map: dict[str, float] = {}
            for o in next_obs:
                if o["observation_type"] not in (
                    "instrument_return",
                    "instrument_daily",
                ):
                    continue
                v = o["value"]
                if isinstance(v, dict) and "log_return" in v:
                    try:
                        return_map[o["entity_id"]] = float(v["log_return"])
                    except (TypeError, ValueError):
                        pass

            # Align scores with returns
            preds_w, targets_w = [], []
            for i, eid in enumerate(instrument_eids):
                if eid in return_map:
                    preds_w.append(scores[i].item())
                    targets_w.append(return_map[eid])

            if len(preds_w) < 3:
                skipped += 1
                continue

            ic = spearmanr(preds_w, targets_w).statistic
            if math.isfinite(ic):
                ics.append(ic)

    conn.close()

    if not ics:
        print(f"\nNo IC values (skipped {skipped} windows) — check DB return data.")
        return

    ics_arr = np.array(ics)
    mean_ic = ics_arr.mean()
    ic_std = ics_arr.std()
    icir = mean_ic / ic_std if ic_std > 0 else 0.0
    hit_rate = (ics_arr > 0).mean()

    print(f"\n{'─'*50}")
    print(f"  Windows evaluated : {len(ics)}  (skipped {skipped})")
    print(f"  Mean IC           : {mean_ic:+.4f}")
    print(f"  IC Std            : {ic_std:.4f}")
    print(f"  ICIR              : {icir:+.4f}")
    print(f"  Hit rate (IC > 0) : {hit_rate:.1%}")
    print(f"{'─'*50}")
    if mean_ic > 0.05:
        print("  ✓ Strong positive IC — model is ranking instruments correctly")
    elif mean_ic > 0.01:
        print("  ~ Weak positive IC — signal is present but small")
    elif mean_ic > 0:
        print("  ~ Near-zero positive IC — marginal signal")
    else:
        print("  ✗ Negative/zero IC — predictions are not useful")
    print()


if __name__ == "__main__":
    main()
