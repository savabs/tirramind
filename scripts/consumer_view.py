"""
consumer_view.py — What would TirraMind show a subscriber right now?

Loads the trained GNN, runs it on the most recent data window, and prints
a ranked prediction sheet: which instruments the model expects to outperform
vs underperform over the next ~21 trading days.

Usage:
    python3 scripts/consumer_view.py
    python3 scripts/consumer_view.py --top 20          # show top/bottom N
    python3 scripts/consumer_view.py --asset-class commodity
"""

import argparse
import bisect
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH    = ROOT / ".tirra_pipeline" / "pipeline.db"
MODEL_PATH = ROOT / ".tirra_pipeline" / "gnn_model.pt"
GNN_LOOKBACK_DAYS = 90   # same as backtest


def _load_returns(db_path: str, entity_ids: list[str]):
    """Return (dates, returns_matrix) using the same fast SQL path as the backtest."""
    import json
    import sqlite3
    from collections import defaultdict

    con = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(entity_ids))
    rows = con.execute(
        f"SELECT entity_id, observed_at, value_json "
        f"FROM entity_observations "
        f"WHERE observation_type='instrument_daily' "
        f"AND entity_id IN ({placeholders}) "
        f"ORDER BY observed_at",
        entity_ids,
    ).fetchall()
    con.close()

    by_date: dict[str, dict[str, float]] = defaultdict(dict)
    for eid, ts, val_json in rows:
        val = json.loads(val_json) if val_json else {}
        lr = val.get("log_return")
        if lr is None:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day = dt.strftime("%Y-%m-%d")
        by_date[day][eid] = float(lr)

    dates = sorted(by_date)
    idx = {e: i for i, e in enumerate(entity_ids)}
    mat = np.zeros((len(dates), len(entity_ids)), dtype=np.float32)
    for d, date in enumerate(dates):
        for eid, val in by_date[date].items():
            if eid in idx:
                mat[d, idx[eid]] = val

    return dates, mat


def main():
    parser = argparse.ArgumentParser(description="TirraMind consumer prediction view")
    parser.add_argument("--top", type=int, default=15,
                        help="Show top and bottom N instruments (default: 15)")
    parser.add_argument("--asset-class", default=None,
                        help="Filter by asset class (e.g. commodity, equity)")
    parser.add_argument("--model", default=str(MODEL_PATH),
                        help="Path to model .pt file")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not DB_PATH.exists():
        print(f"ERROR: pipeline.db not found at {DB_PATH}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    import torch
    from agent.models.gnn.trainer import Trainer
    from agent.pipeline.store import PipelineStore

    store = PipelineStore(str(DB_PATH))
    trainer = Trainer.load_model(model_path, store)
    trainer._model.eval()

    # ── Universe ──────────────────────────────────────────────────────────────
    entities = store.query_all_entities()
    entity_ids, asset_classes, tickers = [], {}, {}
    for e in entities:
        if e["entity_type"] != "instrument":
            continue
        meta = e.get("metadata") or {}
        ac = meta.get("asset_class", "unknown")
        if args.asset_class and ac != args.asset_class:
            continue
        entity_ids.append(e["entity_id"])
        asset_classes[e["entity_id"]] = ac
        tickers[e["entity_id"]] = meta.get("ticker") or meta.get("symbol") or e["entity_id"]

    print(f"Universe: {len(entity_ids)} instruments", end="")
    if args.asset_class:
        print(f"  (filtered: {args.asset_class})", end="")
    print()

    # ── Most recent date window ────────────────────────────────────────────────
    dates, _ = _load_returns(str(DB_PATH), entity_ids)
    as_of = dates[-1]
    as_of_ts = datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc).timestamp()
    since_ts = as_of_ts - GNN_LOOKBACK_DAYS * 86400

    print(f"As of:    {as_of}  (lookback: {GNN_LOOKBACK_DAYS}d)")
    print(f"Horizon:  ~21 trading days forward")
    print()

    # ── Build graph and run model ─────────────────────────────────────────────
    id_map, _, links = trainer._graph_builder.prepare_static()
    all_obs = trainer._graph_builder.prefetch_observations()
    obs_ts  = [o["observed_at"] for o in all_obs]

    end_idx   = bisect.bisect_left(obs_ts, as_of_ts)
    start_idx = bisect.bisect_left(obs_ts, since_ts)
    obs_window = all_obs[start_idx:end_idx]

    if not obs_window:
        print("ERROR: no observations in lookback window.")
        sys.exit(1)

    data, local_id_map, _ = trainer._graph_builder.build_from_cached(
        id_map, links, observations=obs_window
    )

    with torch.no_grad():
        embeddings = trainer._model(data, local_id_map)
        inst_emb = embeddings.get("instrument")

    if inst_emb is None or inst_emb.shape[0] == 0:
        print("ERROR: model returned no instrument embeddings.")
        sys.exit(1)

    ret_scores = trainer._model.return_pred_head(inst_emb).squeeze(-1)  # (n_inst,)
    val_scores = trainer._model.value_pred_head(inst_emb).squeeze(-1)   # (n_inst,)

    # ── Collect scores per entity ─────────────────────────────────────────────
    rows = []
    for eid in entity_ids:
        local_idx = local_id_map.local_id("instrument", eid)
        if local_idx is None:
            continue
        rs = float(ret_scores[local_idx].item())
        vs = float(val_scores[local_idx].item())
        rows.append({
            "id":    eid,
            "ticker": tickers[eid],
            "ac":    asset_classes[eid],
            "ret_score": rs,
            "val_score": vs,
        })

    if not rows:
        print("ERROR: no scored instruments (none in graph window).")
        sys.exit(1)

    # Rank by return-head score (ICIR 0.221 — best head)
    rows.sort(key=lambda r: r["ret_score"], reverse=True)

    # Normalise scores to a 0–100 percentile rank for readability
    n = len(rows)
    for rank, r in enumerate(rows):
        r["pct_rank"] = round(100 * (1 - rank / n))
        direction = "▲ LONG" if r["pct_rank"] >= 60 else ("▼ SHORT" if r["pct_rank"] <= 40 else "  HOLD")
        r["direction"] = direction

    top_n = min(args.top, n // 2)

    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  TirraMind — Predicted Rankings   ({as_of})")
    print(f"  Scored by GNN ReturnHead  (ICIR=0.221, CFTC instruments ICIR=0.403)")
    print("=" * 70)
    print(f"  {'Rank':<5} {'Ticker':<20} {'Asset Class':<14} {'Signal':<8} {'Pct'}")
    print("  " + "-" * 60)

    print(f"\n  TOP {top_n}  — predicted outperformers")
    for rank, r in enumerate(rows[:top_n], 1):
        print(f"  {rank:<5} {r['ticker']:<20} {r['ac']:<14} {r['direction']:<10} {r['pct_rank']:>3}%")

    print(f"\n  BOTTOM {top_n}  — predicted underperformers")
    for rank, r in enumerate(rows[-top_n:], 1):
        actual_rank = n - top_n + rank
        print(f"  {actual_rank:<5} {r['ticker']:<20} {r['ac']:<14} {r['direction']:<10} {r['pct_rank']:>3}%")

    print()
    print(f"  Total instruments scored: {n}")
    print(f"  LONG signals (top 40%):   {sum(1 for r in rows if r['pct_rank'] >= 60)}")
    print(f"  SHORT signals (bot 40%):  {sum(1 for r in rows if r['pct_rank'] <= 40)}")
    print()

    # CFTC commodity breakdown
    cftc_rows = [r for r in rows if r["ac"] == "commodity"]
    if cftc_rows:
        print(f"  CFTC commodities ({len(cftc_rows)}) — where H-G signal is strongest (ICIR=0.403):")
        print(f"  {'Rank':<5} {'Ticker':<20} {'Signal':<8} {'Pct'}")
        print("  " + "-" * 45)
        for r in cftc_rows[:10]:
            rank = rows.index(r) + 1
            print(f"  {rank:<5} {r['ticker']:<20} {r['direction']:<10} {r['pct_rank']:>3}%")
        print()

    print("  Note: Signal quality is directional but sub-threshold (t=1.40, n=40 folds).")
    print("  Use for hypothesis testing; do not size positions on this alone.")


if __name__ == "__main__":
    main()
