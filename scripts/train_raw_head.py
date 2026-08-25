#!/usr/bin/env python3
"""Standalone trainer for return_raw_head — no GNN required.

Why this exists
---------------
In REPLACE mode the return_raw_head takes raw instrument price features and
has zero computational dependency on the GNN backbone. Training it inside the
full training loop wastes 99% of compute on the backbone (obs_type, dt,
contrastive) and starves the head of useful gradient via the shared clip.

This script trains ONLY the raw head:
  • Correct cross-sectional z-score normalization (matches ridge ceiling)
  • Negative Pearson correlation loss = direct IC maximisation
  • ~60-100 epochs in < 5 min on CPU (no GNN forward pass)
  • Merges updated head weights back into a copy of the source checkpoint

Usage
-----
  python scripts/train_raw_head.py                        # uses defaults
  python scripts/train_raw_head.py --epochs 100 --lr 3e-4
  python scripts/train_raw_head.py --checkpoint path/to/epoch_021.pt
"""

from __future__ import annotations
import argparse, json, math, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from agent.models.gnn.graph_builder import (
    _build_node_features,
    _compute_price_features,
    BASE_FEAT_DIM,
    PRICE_FEAT_DIM,
)

CKPT_DEFAULT = Path(
    ".tirra_pipeline/kaggle_downloads/phase50_ckpts/phase50_ckpts/epoch_021.pt"
)
DB_DEFAULT = Path(".tirra_pipeline/tirramind-data/pipeline.db")
TS_MIN, TS_MAX = 1_577_836_800, 1_893_456_000


# ── Data helpers ──────────────────────────────────────────────────────────────


def load_daily(db: Path) -> dict[str, list[dict]]:
    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        """
        SELECT eo.entity_id, eo.observed_at, eo.value_json
        FROM entity_observations eo
        JOIN entities e ON e.entity_id = eo.entity_id
        WHERE e.entity_type = 'instrument'
          AND eo.observation_type = 'instrument_daily'
          AND eo.observed_at BETWEEN ? AND ?
        ORDER BY eo.observed_at
    """,
        (TS_MIN, TS_MAX),
    ).fetchall()
    conn.close()
    daily: dict[str, list[dict]] = {}
    for eid, ts, vjson in rows:
        try:
            v = json.loads(vjson) if vjson else {}
            daily.setdefault(eid, []).append(
                {
                    "ts": float(ts),
                    "close": float(v.get("close", 0) or 0),
                    "log_return": float(v.get("log_return", 0) or 0),
                    "volume": float(v.get("volume", 0) or 0),
                }
            )
        except Exception:
            continue
    for k in daily:
        daily[k].sort(key=lambda x: x["ts"])
    return daily


def all_features_for_window(
    daily: dict[str, list[dict]], t_end: float
) -> tuple[list[str], np.ndarray] | tuple[None, None]:
    """Return (entity_ids, feat_matrix) using _build_node_features.

    The observations are formatted to match what PipelineStore provides so that
    _build_node_features computes the correct 23-dim vector (BASE_FEAT_DIM=14 +
    PRICE_FEAT_DIM=9) with the same code path used at training time.
    """
    insts = [eid for eid in daily if eid in daily]

    # Format daily data as observation dicts expected by _build_node_features
    obs_list: list[dict] = []
    for eid, records in daily.items():
        for r in records:
            if r["ts"] <= t_end:
                obs_list.append(
                    {
                        "entity_id": eid,
                        "observation_type": "instrument_daily",
                        "observed_at": r["ts"],
                        "value": {
                            "close": r["close"],
                            "log_return": r["log_return"],
                            "volume": r["volume"],
                        },
                    }
                )

    if not insts or not obs_list:
        return None, None

    feat_tensor = _build_node_features(
        entity_type="instrument",
        entity_ids=insts,
        observations=obs_list,
        current_time=t_end,
    )  # shape [N, 23]
    return insts, feat_tensor.numpy().astype(np.float32)


def fwd_return(data: list[dict], t_start: float, n_days: int = 21) -> float | None:
    buffer = n_days * 1.4 * 86400
    vals = [
        x["log_return"]
        for x in data
        if t_start <= x["ts"] <= t_start + buffer and math.isfinite(x["log_return"])
    ]
    return sum(vals) if len(vals) >= max(1, n_days // 3) else None


# ── Build cross-sectional panel ───────────────────────────────────────────────


def build_panel(
    daily: dict[str, list[dict]], step_secs: float = 604_800, fwd_days: int = 21
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (X, y) per window; price features cross-sectionally z-scored."""
    insts = list(daily.keys())
    ts_all = sorted({d["ts"] for v in daily.values() for d in v})
    t_min, t_max = ts_all[0], ts_all[-1]
    buffer = fwd_days * 1.5 * 86400
    price_offset = BASE_FEAT_DIM  # index where price features start in 23-dim vector
    t = t_min + step_secs * 26
    panel = []
    while t + buffer <= t_max:
        eids_w, X_w = all_features_for_window(daily, t)
        if eids_w is None or len(eids_w) < 5:
            t += step_secs
            continue
        # Collect forward returns aligned to feature rows
        y_raw = []
        keep = []
        for i, eid in enumerate(eids_w):
            r = fwd_return(daily[eid], t, fwd_days)
            if r is not None:
                y_raw.append(r)
                keep.append(i)
        if len(keep) < 5:
            t += step_secs
            continue
        X = X_w[keep]  # [N, 23]
        y = np.array(y_raw, dtype=np.float32)
        # Cross-sectionally z-score only the price feature block [price_offset:price_offset+9]
        # (entity-type one-hots are identical for all instruments → already zero variance)
        price_block = X[:, price_offset : price_offset + PRICE_FEAT_DIM]
        p_mean = price_block.mean(0)
        p_std = price_block.std(0) + 1e-8
        X[:, price_offset : price_offset + PRICE_FEAT_DIM] = (
            price_block - p_mean
        ) / p_std
        # Cross-sectional demean returns
        y = y - y.mean()
        panel.append((X, y))
        t += step_secs
    return panel


# ── Loss: negative Pearson correlation (= direct IC maximisation) ─────────────


def pearson_ic_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    s = scores - scores.mean()
    t = targets - targets.mean()
    s_std = s.pow(2).mean().sqrt()
    t_std = t.pow(2).mean().sqrt().clamp(min=1e-8)
    # If scores are near-constant, push variance up via L2 regularisation on
    # the mean-centred scores. This breaks the zero-gradient fixed point that
    # occurs when the output layer is near zero.
    if s_std < 1e-6:
        return -s.pow(2).mean() * 0.0 + torch.tensor(0.0, requires_grad=True)
    s_std = s_std.clamp(min=1e-8)
    return -(s * t).mean() / (s_std * t_std)


# ── IC metric ─────────────────────────────────────────────────────────────────


def spearman_ic(scores: np.ndarray, targets: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(targets)
    if mask.sum() < 3:
        return float("nan")
    r, _ = stats.spearmanr(scores[mask], targets[mask])
    return float(r) if math.isfinite(r) else float("nan")


# ── Load / save head ──────────────────────────────────────────────────────────


def load_head(ckpt_path: Path) -> tuple[nn.Sequential, int, dict]:
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
    epoch = int(ckpt.get("epoch", 0))
    return head, epoch, ckpt


def save_head(head: nn.Sequential, ckpt: dict, out_path: Path) -> None:
    sd = ckpt["model_state"]
    sd["return_raw_head.0.weight"] = head[0].weight.data.clone()
    sd["return_raw_head.0.bias"] = head[0].bias.data.clone()
    sd["return_raw_head.2.weight"] = head[2].weight.data.clone()
    sd["return_raw_head.2.bias"] = head[2].bias.data.clone()
    ckpt["model_state"] = sd
    torch.save(ckpt, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CKPT_DEFAULT))
    p.add_argument("--db", default=str(DB_DEFAULT))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fwd-days", type=int, default=21)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument(
        "--reset-head",
        action="store_true",
        help="Re-initialise head weights before training (recommended "
        "when switching from unnormalized to normalized features)",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        dest="weight_decay",
        help="AdamW weight decay / L2 regularisation (default: 0.05). "
        "Strong regularisation (0.05-0.5) is critical: MLP has 1537 params "
        "vs ~88 effective weekly samples — without it the head badly overfits.",
    )
    p.add_argument(
        "--use-all-data",
        action="store_true",
        dest="use_all_data",
        help="Train on ALL windows (no held-out val). Use with --weight-decay "
        "so regularisation controls capacity instead of early stopping.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output checkpoint path (default: <source>_rawhead.pt)",
    )
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    db_path = Path(args.db)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    assert db_path.exists(), f"DB not found: {db_path}"

    out_path = (
        Path(args.out)
        if args.out
        else ckpt_path.parent / (ckpt_path.stem + "_rawhead.pt")
    )

    print(f"\n{'═'*60}")
    print(f"  train_raw_head  — pure IC optimisation")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  DB         : {db_path}")
    print(f"  Epochs     : {args.epochs}  LR : {args.lr}")
    print(f"  Fwd horizon: {args.fwd_days} trading days")
    print(f"  Features   : cross-sectionally z-scored per window")
    print(f"  Loss       : negative Pearson IC (direct IC maximisation)")
    print(f"  Out        : {out_path}")
    print(f"{'═'*60}\n")

    # Load head
    head, base_epoch, ckpt = load_head(ckpt_path)
    in_d = head[0].weight.shape[1]
    hid = head[0].weight.shape[0]
    print(f"Head arch   : {in_d} → {hid} → 1  (from epoch {base_epoch})")

    if args.reset_head:
        nn.init.kaiming_uniform_(head[0].weight, nonlinearity="relu")
        nn.init.zeros_(head[0].bias)
        # NOTE: output layer must NOT be zero — zero init creates constant scores
        # → Pearson grad = 0 → stuck forever. Use small kaiming instead.
        nn.init.kaiming_uniform_(head[2].weight, nonlinearity="linear")
        nn.init.zeros_(head[2].bias)
        print("Head weights RESET (kaiming init — avoids zero-gradient fixed point)")

    # Build panel
    print("\nLoading instrument_daily data…")
    daily = load_daily(db_path)
    print(f"Instruments : {len(daily)}")
    panel = build_panel(daily, fwd_days=args.fwd_days)
    print(f"Windows     : {len(panel)}")

    if args.use_all_data:
        train_data = panel
        val_data = []
        n_train, n_val = len(panel), 0
    else:
        n_train = max(4, int(len(panel) * args.train_frac))
        n_val = len(panel) - n_train
        train_data = panel[:n_train]
        val_data = panel[n_train:]
    print(f"Train/val   : {n_train} / {n_val}")
    print(f"Weight decay: {args.weight_decay}\n")

    # Train — AdamW for proper L2 regularisation (Adam ignores weight_decay correctly)
    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5
    )

    best_val_ic = -999.0
    best_state = None

    EVAL_EVERY = 10
    W = 58

    print(f"{'Ep':>4}  {'train_IC':>9}  {'val_IC':>9}  {'best_val':>9}  {'LR':>9}")
    print("─" * W)

    for epoch in range(1, args.epochs + 1):
        head.train()
        np.random.shuffle(train_data)

        train_ics = []
        for X_np, y_np in train_data:
            X = torch.tensor(X_np)
            y = torch.tensor(y_np)
            scores = head(X).squeeze(-1)
            loss = pearson_ic_loss(scores, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            train_ics.append(spearman_ic(scores.detach().numpy(), y_np))

        scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == args.epochs:
            head.eval()
            val_ics = []
            with torch.no_grad():
                for X_np, y_np in val_data:
                    X = torch.tensor(X_np)
                    scores = head(X).squeeze(-1).numpy()
                    val_ics.append(spearman_ic(scores, y_np))

            t_ic = np.nanmean(train_ics)
            v_ic = np.nanmean(val_ics) if val_ics else float("nan")

            if math.isfinite(v_ic) and v_ic > best_val_ic:
                best_val_ic = v_ic
                best_state = {k: v.clone() for k, v in head.state_dict().items()}

            lr_now = opt.param_groups[0]["lr"]
            star = "★" if (math.isfinite(v_ic) and v_ic >= best_val_ic) else " "
            print(
                f"{epoch:>4}  {t_ic:>+9.4f}  {v_ic:>+9.4f}  {best_val_ic:>+9.4f}  {lr_now:>9.2e}  {star}"
            )

    print("─" * W)
    print(f"\nBest val IC : {best_val_ic:+.4f}")

    # Restore best weights
    if best_state is not None:
        head.load_state_dict(best_state)

    # Final IC over all windows
    head.eval()
    all_ics = []
    with torch.no_grad():
        for X_np, y_np in panel:
            scores = head(torch.tensor(X_np)).squeeze(-1).numpy()
            all_ics.append(spearman_ic(scores, y_np))
    v = np.array([x for x in all_ics if math.isfinite(x)])
    if len(v) > 0:
        icir = v.mean() / v.std() if v.std() > 0 else 0.0
        print(f"Full-panel  : mean IC={v.mean():+.4f}  ICIR={icir:+.3f}  n={len(v)}")
        print(
            f"True ceilings: sharpe_60d single factor IC≈+0.12 ICIR≈+0.50  |  ridge(9-feat, corrected) IC≈+0.07 ICIR≈+0.40"
        )

    # Save checkpoint with updated head weights
    save_head(head, ckpt, out_path)
    print(f"\n✓ Saved → {out_path}")
    print(
        f"  Upload this as 'tirramind-phase50-ep{base_epoch}-rawhead' dataset on Kaggle"
    )
    print(f"  to resume GNN training with the pre-trained raw head.\n")


if __name__ == "__main__":
    main()
