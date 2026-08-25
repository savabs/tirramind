#!/usr/bin/env python3
"""Data / label pipeline audit (DATA_FIX path).

Diagnostics:
  1. Pre vs post-backfill forward-return distribution (H1)
  2. Label-definition mismatch (trainer simple return vs IC eval mean-daily vs log-sum)
  3. Temporal leakage: future close inside feature window (H6)
  4. Shuffled-label IC test (leakage detector)

Usage:
    python scripts/data_label_audit.py
    python scripts/data_label_audit.py --backup-db .tirra_pipeline/pipeline.db.bak_20260512
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from agent.models.gnn.graph_builder import (  # noqa: E402
    BASE_FEAT_DIM,
    GraphBuilder,
    PRICE_FEAT_DIM,
)
from agent.pipeline.store import PipelineStore  # noqa: E402
from agent.quant.forward_returns import (  # noqa: E402
    DEFAULT_HORIZON_DAYS,
    audit_window_label_leakage,
    build_forward_return_lookup,
    compare_lookups,
    label_distribution_stats,
    label_method_correlation,
)
from honest_baseline_audit import _instrument_price_feature_matrix  # noqa: E402


def _spearman_ic(scores: np.ndarray, targets: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(targets)
    if mask.sum() < 3:
        return float("nan")
    r, _ = stats.spearmanr(scores[mask], targets[mask])
    return float(r) if math.isfinite(r) else float("nan")

log = logging.getLogger("data_label_audit")

DB_DEFAULT = Path(".tirra_pipeline/pipeline.db")
BACKUP_DEFAULT = Path(".tirra_pipeline/pipeline.db.bak_20260512")
OUT_DEFAULT = Path(".tirra_pipeline/data_label_audit.json")


def _load_prefetched(db_path: Path) -> list[dict]:
    store = PipelineStore(str(db_path))
    gb = GraphBuilder(store)
    return gb.prefetch_observations()


def _date_to_ts(iso_date: str) -> float:
    return (
        datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc).timestamp()
    )


def shuffled_label_ic_test(
    graph_builder: GraphBuilder,
    dates: list[str],
    returns: np.ndarray,
    prefetched: list[dict],
    id_map: object,
    links: list[dict],
    entity_ids: list[str],
    *,
    min_train: int,
    n_shuffles: int = 5,
) -> dict[str, float]:
    """Ridge on raw price features with shuffled targets — IC should ≈ 0 if honest."""
    from phase40_gnn_backtest import STEP_SIZE, TEST_SIZE

    lookup = build_forward_return_lookup(prefetched)
    obs_ts = [o["observed_at"] for o in prefetched]
    ics: list[float] = []

    split = min_train
    while split + TEST_SIZE <= len(dates):
        fold_date = dates[split]
        feats = _instrument_price_feature_matrix(
            graph_builder,
            fold_date,
            entity_ids,
            id_map,
            links,
            prefetched,
            obs_ts,
        )
        from agent.quant.forward_returns import forward_return_vector_for_date

        y_true = forward_return_vector_for_date(lookup, entity_ids, fold_date)
        valid = np.isfinite(feats).all(axis=1) & np.isfinite(y_true)
        if valid.sum() < 5:
            split += STEP_SIZE
            continue

        X = feats[valid]
        y = y_true[valid]
        n = len(y)
        if n < 10:
            split += STEP_SIZE
            continue
        # Out-of-sample: train on first 70%, score held-out 30% (avoids in-sample
        # Ridge overfit that falsely flags leakage when labels are shuffled).
        n_train = max(5, int(n * 0.7))
        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]
        if len(y_te) < 3:
            split += STEP_SIZE
            continue
        rng = np.random.default_rng(42)
        for _ in range(n_shuffles):
            y_shuf_tr = y_tr.copy()
            rng.shuffle(y_shuf_tr)
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr_s, y_shuf_tr)
            scores = ridge.predict(X_te_s)
            ic = _spearman_ic(scores, y_te)
            if np.isfinite(ic):
                ics.append(float(ic))
        split += STEP_SIZE

    arr = np.array(ics, dtype=np.float64)
    return {
        "n_tests": float(len(arr)),
        "mean_ic": float(arr.mean()) if len(arr) else float("nan"),
        "std_ic": float(arr.std()) if len(arr) > 1 else float("nan"),
        "max_abs_ic": float(np.max(np.abs(arr))) if len(arr) else float("nan"),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="Data / label pipeline audit")
    ap.add_argument("--db-path", type=Path, default=DB_DEFAULT)
    ap.add_argument("--backup-db", type=Path, default=BACKUP_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--smoke", action="store_true", help="Fewer shuffles / shorter window sample")
    args = ap.parse_args()

    if not args.db_path.exists():
        print(f"ERROR: DB not found: {args.db_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DATA / LABEL AUDIT — DATA_FIX diagnostics")
    print("=" * 60)

    prefetched = _load_prefetched(args.db_path)
    lookup = build_forward_return_lookup(prefetched)
    vals = np.array(list(lookup.values()), dtype=np.float64)
    current_stats = label_distribution_stats(vals)

    print("\n── 1. Current DB forward-return distribution ──")
    print(
        f"  n={int(current_stats['n'])}  mean={current_stats['mean']:.4f}  "
        f"std={current_stats['std']:.4f}  "
        f"range=[{current_stats['min']:.4f}, {current_stats['max']:.4f}]"
    )

    backfill_compare: dict[str, float] | None = None
    if args.backup_db.exists():
        print(f"\n── 2. Pre/post backfill comparison ──")
        print(f"  current : {args.db_path}")
        print(f"  backup  : {args.backup_db}")
        backup_prefetched = _load_prefetched(args.backup_db)
        backup_lookup = build_forward_return_lookup(backup_prefetched)
        backfill_compare = compare_lookups(lookup, backup_lookup)
        print(
            f"  shared keys : {int(backfill_compare['n_shared'])}  "
            f"Spearman={backfill_compare['spearman']:.4f}  "
            f"mean|diff|={backfill_compare['mean_abs_diff']:.4f}"
        )
        print(
            f"  current mean/std : {backfill_compare['mean_a']:.4f} / {backfill_compare['std_a']:.4f}"
        )
        print(
            f"  backup  mean/std : {backfill_compare['mean_b']:.4f} / {backfill_compare['std_b']:.4f}"
        )
    else:
        print(f"\n── 2. Pre/post backfill: SKIP (no backup at {args.backup_db}) ──")

    print("\n── 3. Label definition mismatch (trainer vs eval methods) ──")
    store = PipelineStore(str(args.db_path))
    gb = GraphBuilder(store)
    entities = store.query_all_entities()
    entity_ids = [e["entity_id"] for e in entities if e["entity_type"] == "instrument"]

    from phase40_gnn_backtest import _load_instrument_returns_fast

    dates, returns = _load_instrument_returns_fast(str(args.db_path), entity_ids)
    sample_dates = dates[:: max(1, len(dates) // (5 if args.smoke else 20))][: 20 if not args.smoke else 5]
    anchor_ts = [_date_to_ts(d) for d in sample_dates]
    label_corr = label_method_correlation(
        prefetched, entity_ids, anchor_ts, horizon_days=DEFAULT_HORIZON_DAYS
    )
    print(f"  samples compared : {int(label_corr['n'])}")
    print(
        f"  Spearman(simple vs log-sum)    : {label_corr['simple_vs_logsum']:.4f}"
    )
    print(
        f"  Spearman(simple vs mean-daily) : {label_corr['simple_vs_meandaily']:.4f}"
    )
    print(
        f"  mean |simple - log_sum|        : {label_corr['mean_abs_simple_minus_logsum']:.4f}"
    )
    if label_corr.get("simple_vs_meandaily", 1.0) < 0.85:
        print(
            "  ⚠ MISMATCH: phase40 IC uses mean-daily log_return; "
            "trainer uses simple forward return from closes."
        )

    print("\n── 4. Temporal leakage audit (H6) ──")
    # Sample windows from graph builder timeline
    id_map, _, links = gb.prepare_static()
    obs_ts_sorted = sorted(o["observed_at"] for o in prefetched)
    ws = 168 * 3600.0  # 168h windows (Kaggle default)
    windows: list[tuple[float, float, list]] = []
    if len(obs_ts_sorted) > 10:
        t_min, t_max = obs_ts_sorted[0], obs_ts_sorted[-1]
        t = t_min + ws * 10
        while t + ws < t_max and len(windows) < (5 if args.smoke else 30):
            t_start = t
            t_end = t + ws
            start_idx = bisect_left(obs_ts_sorted, t_start)
            end_idx = bisect_right(obs_ts_sorted, t_end)
            windows.append((t_start, t_end, prefetched[start_idx:end_idx]))
            t += ws

    leak_stats = (
        audit_window_label_leakage(windows, lookup)
        if windows
        else {"n_labels_checked": 0.0, "n_leaked": 0.0, "pct_leaked": 0.0}
    )
    print(
        f"  labels checked : {int(leak_stats['n_labels_checked'])}  "
        f"leaked (future close ≤ t_end) : {int(leak_stats['n_leaked'])}  "
        f"({leak_stats['pct_leaked']*100:.2f}%)"
    )

    print("\n── 5. Shuffled-label IC test (leakage detector) ──")
    n_shuf = 2 if args.smoke else 5
    shuf = shuffled_label_ic_test(
        gb,
        dates[-400:] if args.smoke and len(dates) > 400 else dates,
        returns[-400:] if args.smoke and len(returns) > 400 else returns,
        prefetched,
        id_map,
        links,
        entity_ids,
        min_train=126 if args.smoke else 252,
        n_shuffles=n_shuf,
    )
    print(
        f"  shuffled IC mean={shuf['mean_ic']:+.4f}  "
        f"std={shuf['std_ic']:.4f}  max|IC|={shuf['max_abs_ic']:.4f}"
    )
    if shuf["max_abs_ic"] > 0.15:
        print("  ⚠ HIGH shuffled IC — possible feature-label leakage or overfit.")
    else:
        print("  ✓ Shuffled IC near zero — no obvious leakage in raw-feature path.")

    # Verdict
    issues: list[str] = []
    if backfill_compare and backfill_compare.get("spearman", 1.0) < 0.9:
        issues.append("BACKFILL_SHIFT")
    if label_corr.get("simple_vs_meandaily", 1.0) < 0.85:
        issues.append("LABEL_DEFINITION_MISMATCH")
    if leak_stats.get("pct_leaked", 0.0) > 0.01:
        issues.append("TEMPORAL_LEAKAGE")
    if shuf.get("max_abs_ic", 0.0) > 0.15:
        issues.append("SHUFFLED_IC_HIGH")

    if issues:
        verdict = "FIX_REQUIRED: " + ", ".join(issues)
    else:
        verdict = "LABELS_HONEST_LOW_SIGNAL"

    print(f"\n  VERDICT: {verdict}")

    out = {
        "audit": "data_label_audit",
        "db_path": str(args.db_path),
        "backup_db": str(args.backup_db) if args.backup_db.exists() else None,
        "current_forward_return": current_stats,
        "backfill_compare": backfill_compare,
        "label_method_correlation": label_corr,
        "temporal_leakage": leak_stats,
        "shuffled_label_ic": shuf,
        "issues": issues,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n  Results → {args.out}\n")


if __name__ == "__main__":
    main()
