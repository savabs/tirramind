#!/usr/bin/env python3
"""Phase 41b — GNN Information Propagation Diagnostic.

North Star test: does the embedding of an upstream entity type at time T-N
Granger-cause the embedding of instrument nodes at time T?

If YES → the GNN perceptual layer is encoding real pre-emergence causal information
that travels from entity space (insiders, CFTC, vessels) into instrument space.

If NO → the GNN is encoding local structure only; return signal cannot propagate
across the graph regardless of the return loss function.

Method
------
For each entity type E and each lag N (in weekly steps):

    X_t = mean L2 norm of all E-type entity embeddings at window T
    Y_t = mean L2 norm of all instrument embeddings at window T

    Granger causality test (maxlag=N):
        H0: past X does NOT help predict Y (given past Y alone)
        Reject H0 at p < 0.05 → X Granger-causes Y at lag N

The L2 norm is used as the aggregate signal because it captures overall
information activity in the entity neighbourhood without requiring
per-entity alignment across time steps (entity IDs change as the graph
is built from different observation windows).

Output
------
Console table:
    Entity type | Lag (weeks) | F-stat | p-value | Granger-causes?

References
----------
    Granger (1969) "Investigating causal relations by econometric models"
    statsmodels.tsa.stattools.grangercausalitytests (v0.14)
    Krstev & Rigoni et al. (2026, ICLR withdrawn) — temporal link prediction
    framing of lead-lag relationships on dynamic entity graphs.

Usage
-----
    /home/becmachlean/anaconda3/bin/python scripts/phase41b_propagation_diagnostic.py
    /home/becmachlean/anaconda3/bin/python scripts/phase41b_propagation_diagnostic.py \\
        --db .tirra_pipeline/pipeline.db \\
        --model .tirra_pipeline/gnn_model.pt \\
        --window-days 7 \\
        --lags 1 2 3 4 \\
        --lookback-days 365
"""

from __future__ import annotations

import argparse
import bisect
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase41b_diag")

# ── Defaults ──────────────────────────────────────────────────────────────────

DB_PATH = Path(".tirra_pipeline/pipeline.db")
MODEL_PATH = Path(".tirra_pipeline/gnn_model.pt")
WINDOW_DAYS = 7  # embedding snapshot interval in days
LAGS = [1, 2, 3, 4]  # lags in units of WINDOW_DAYS (1wk, 2wk, 3wk, 4wk)
LOOKBACK_DAYS = 365  # how many days of history to scan
SIG_LEVEL = 0.05  # Granger causality significance threshold


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mean_embedding_norm(embeddings: dict[str, Any], entity_type: str) -> float | None:
    """Mean L2 norm of all embeddings of entity_type.  None if missing."""
    import torch

    emb = embeddings.get(entity_type)
    if emb is None or emb.shape[0] == 0:
        return None
    norms = emb.norm(dim=-1)  # shape (N_type,)
    return float(norms.mean().item())


def _granger_test(x: np.ndarray, y: np.ndarray, max_lag: int) -> dict:
    """Run Granger causality test: does x Granger-cause y at max_lag?

    Returns dict with keys: f_stat, p_value, significant.
    Returns None values on failure (e.g. insufficient data, constant series).
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return {"f_stat": None, "p_value": None, "significant": False}

    data = np.column_stack([y, x])  # statsmodels expects [y, x] ordering

    min_obs = max_lag * 3 + 10  # rough minimum for a valid test
    if len(data) < min_obs:
        return {"f_stat": None, "p_value": None, "significant": False}

    # Check for degenerate series (constant or near-constant)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return {"f_stat": None, "p_value": None, "significant": False}

    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        # Extract F-test result at the requested lag
        lag_result = results[max_lag][0]
        f_stat = float(lag_result["ssr_ftest"][0])
        p_val = float(lag_result["ssr_ftest"][1])
        return {
            "f_stat": f_stat,
            "p_value": p_val,
            "significant": p_val < SIG_LEVEL,
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("Granger test failed for lag=%d: %s", max_lag, exc)
        return {"f_stat": None, "p_value": None, "significant": False}


def _build_norm_series(
    trainer: Any,
    obs_all: list[dict],
    obs_ts: list[float],
    full_id_map: Any,
    full_links: list[dict],
    start_ts: float,
    end_ts: float,
    window_sec: float,
    entity_types: list[str],
) -> dict[str, list[float | None]]:
    """Build time series of mean embedding norms per entity type.

    Steps through [start_ts, end_ts] in window_sec steps, builds the GNN
    graph from observations in each window, and records mean L2 norms.

    Returns:
        Dict[entity_type, list of float|None]  (None = no embeddings that window)
    """
    import torch

    model = trainer._model
    model.eval()
    graph_builder = trainer._graph_builder

    # Time windows: [t, t + window_sec) stepping by window_sec
    t = start_ts
    series: dict[str, list[float | None]] = {et: [] for et in entity_types}

    n_windows = 0
    while t + window_sec <= end_ts:
        t_end = t + window_sec
        # Slice observations within this window
        lo = bisect.bisect_left(obs_ts, t)
        hi = bisect.bisect_left(obs_ts, t_end)
        obs_window = obs_all[lo:hi]

        if obs_window:
            try:
                data, id_map, _ = graph_builder.build_from_cached(
                    full_id_map, full_links, observations=obs_window
                )
                with torch.no_grad():
                    embeddings = model(data, id_map)
            except Exception as exc:  # noqa: BLE001
                log.debug("Graph build failed at t=%.0f: %s", t, exc)
                embeddings = {}
        else:
            embeddings = {}

        for et in entity_types:
            series[et].append(_mean_embedding_norm(embeddings, et))

        t = t_end
        n_windows += 1

    log.warning(
        "Built %d embedding snapshots over %.0f days",
        n_windows,
        (end_ts - start_ts) / 86400,
    )
    return series


def _fill_na(series: list[float | None]) -> np.ndarray:
    """Forward-fill None values; replace remaining NaNs with 0."""
    arr = np.array([v if v is not None else np.nan for v in series], dtype=np.float64)
    # Forward fill
    mask = np.isnan(arr)
    for i in range(1, len(arr)):
        if mask[i] and not mask[i - 1]:
            arr[i] = arr[i - 1]
            mask[i] = False
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def _print_table(rows: list[dict]) -> None:
    """Print ASCII results table."""
    header = (
        f"{'Entity Type':<22} {'Lag':>6} {'F-stat':>9} {'p-value':>9} {'Granger?':>10}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        lag_wks = f"{r['lag_weeks']}wk"
        f_stat = f"{r['f_stat']:.3f}" if r["f_stat"] is not None else "   N/A"
        p_val = f"{r['p_value']:.4f}" if r["p_value"] is not None else "   N/A"
        sig = "YES ✓" if r["significant"] else "no"
        print(f"{r['entity_type']:<22} {lag_wks:>6} {f_stat:>9} {p_val:>9} {sig:>10}")
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 41b — GNN Propagation Diagnostic"
    )
    parser.add_argument("--db", default=str(DB_PATH), help="Path to pipeline DB")
    parser.add_argument(
        "--model", default=str(MODEL_PATH), help="Path to trained GNN model"
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=WINDOW_DAYS,
        help=f"Embedding snapshot interval in days (default: {WINDOW_DAYS})",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=LAGS,
        help=f"Lags to test in units of window-days (default: {LAGS})",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS,
        help=f"Days of history to scan (default: {LOOKBACK_DAYS})",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    model_path = Path(args.model)

    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    # ── Imports ──────────────────────────────────────────────────────────────
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch not installed.")
        sys.exit(1)

    try:
        from statsmodels.tsa.stattools import grangercausalitytests  # noqa: F401
    except ImportError:
        print("ERROR: statsmodels not installed.  Run: pip install statsmodels")
        sys.exit(1)

    from agent.models.gnn.trainer import Trainer
    from agent.pipeline.store import PipelineStore

    print(f"\nPhase 41b — GNN Information Propagation Diagnostic")
    print(f"DB:    {db_path}")
    print(f"Model: {model_path}")
    print(
        f"Window: {args.window_days}d | Lags: {args.lags} windows | Lookback: {args.lookback_days}d\n"
    )

    # ── Load model ────────────────────────────────────────────────────────────
    store = PipelineStore(db_path=str(db_path))
    print("Loading GNN model…")
    trainer = Trainer.load_model(model_path, store)
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  Model loaded: {n_params:,} parameters")

    # ── Pre-fetch graph structure ─────────────────────────────────────────────
    print("Pre-fetching graph structure…")
    full_id_map, _, full_links = trainer._graph_builder.prepare_static()
    obs_all = trainer._graph_builder.prefetch_observations()
    obs_all.sort(key=lambda o: o.get("observed_at", 0.0))
    obs_ts = [o.get("observed_at", 0.0) for o in obs_all]
    print(
        f"  {full_id_map.num_nodes:,} entities | {len(full_links):,} links | {len(obs_all):,} observations"
    )

    if not obs_all:
        print("ERROR: No observations in DB. Cannot run diagnostic.")
        sys.exit(1)

    # ── Time range ────────────────────────────────────────────────────────────
    end_ts = obs_ts[-1]
    start_ts = end_ts - args.lookback_days * 86400
    window_sec = args.window_days * 86400

    print(
        f"\nBuilding embedding norm time series ({args.lookback_days}d range, {args.window_days}d windows)…"
    )
    entity_types = [
        "instrument",
        "company",
        "country",
        "cftc_contract",
        "person",
        "vessel",
        "organization",
        "topic",
        "domain",
        "wallet",
        "protocol",
    ]

    # Build norm series for all entity types simultaneously
    norm_series = _build_norm_series(
        trainer=trainer,
        obs_all=obs_all,
        obs_ts=obs_ts,
        full_id_map=full_id_map,
        full_links=full_links,
        start_ts=start_ts,
        end_ts=end_ts,
        window_sec=window_sec,
        entity_types=entity_types,
    )

    # Instrument is the target (Y); all others are potential upstream predictors (X)
    y_raw = norm_series.get("instrument", [])
    y = _fill_na(y_raw)

    if y.std() < 1e-8:
        print(
            "\nWARNING: Instrument embedding norms are constant → no signal to detect."
        )
        print("This usually means no instrument entities in the graph yet.")
        sys.exit(0)

    upstream_types = [et for et in entity_types if et != "instrument"]

    print(
        f"\nRunning Granger causality tests ({len(upstream_types)} entity types × {len(args.lags)} lags)…\n"
    )

    rows: list[dict] = []
    significant_count = 0

    for et in upstream_types:
        x_raw = norm_series.get(et, [])
        x = _fill_na(x_raw)

        if x.std() < 1e-8:
            # Entity type has no variation → skip silently
            continue

        for lag in args.lags:
            result = _granger_test(x, y, max_lag=lag)
            row = {
                "entity_type": et,
                "lag_weeks": lag,
                "f_stat": result["f_stat"],
                "p_value": result["p_value"],
                "significant": result["significant"],
            }
            rows.append(row)
            if result["significant"]:
                significant_count += 1

    _print_table(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"\nSummary: {significant_count}/{len(rows)} entity-type × lag pairs Granger-cause instrument embeddings (p < {SIG_LEVEL})"
    )

    sig_types = sorted({r["entity_type"] for r in rows if r["significant"]})
    if sig_types:
        print(f"\nEntity types with confirmed Granger-causation:")
        for et in sig_types:
            lags_sig = [
                r["lag_weeks"]
                for r in rows
                if r["entity_type"] == et and r["significant"]
            ]
            print(f"  {et:<22} at lags {lags_sig} (weeks)")
        print("\n✓ GNN perceptual layer IS encoding pre-emergence causal structure.")
        print("  If IC is still low, the return head training signal is insufficient.")
        print(
            "  Action: retrain with --auto-tune --listnet; ensure CFTC/insider data is dense."
        )
    else:
        print(
            "\n✗ No Granger-causation detected from any entity type → instrument embeddings."
        )
        print(
            "  The GNN is NOT propagating information across the heterogeneous graph."
        )
        print("  Likely causes:")
        print(
            "  1. Instrument entities have insufficient cross-type links in the graph"
        )
        print(
            "  2. CFTC/insider data too sparse (300 CFTC obs, 15/entity) — accumulate more"
        )
        print("  3. GNN depth too shallow (2 HGT layers) for multi-hop propagation")

    print()


if __name__ == "__main__":
    main()
