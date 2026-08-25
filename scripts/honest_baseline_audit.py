#!/usr/bin/env python3
"""Honest baseline audit — Phase A of GNN architecture overhaul.

Answers whether post-backfill labels are learnable before building new architecture.

Baselines (purged walk-forward, same folds as phase40):
  1. Label distribution audit (H1 — backfill label shift)
  2. RawPrice-PurgedRanker — Ridge on 9 price features, no GNN
  3. Momentum-Rank — single-factor 1m momentum rank
  4. GNN-PurgedRanker — Ridge on frozen GNN embeddings (--checkpoint, optional)

Usage:
    python scripts/honest_baseline_audit.py --smoke
    python scripts/honest_baseline_audit.py --checkpoint .tirra_pipeline/gnn_model_phase50.pt
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
    build_forward_return_lookup,
    forward_return_vector_for_date,
)
from phase40_gnn_backtest import (  # noqa: E402
    GNNFoldPurgedRankerStrategy,
    GNN_LOOKBACK_DAYS,
    IC_EXIT_MEAN,
    IC_EXIT_TSTAT,
    MIN_TRAIN,
    STEP_SIZE,
    TEMPERATURE,
    TEST_SIZE,
    _align_graph_features_to_model,
    _load_instrument_returns_fast,
    _print_ic_report,
    _softmax,
)

log = logging.getLogger("honest_baseline")

DB_DEFAULT = Path(".tirra_pipeline/pipeline.db")
OUT_DEFAULT = Path(".tirra_pipeline/honest_baseline_audit.json")
CKPT_DEFAULT = Path(".tirra_pipeline/gnn_model_phase50.pt")


def _date_to_ts(iso_date: str) -> float:
    return (
        datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc).timestamp()
    )


# ── Label distribution audit ─────────────────────────────────────────────────


def audit_label_distribution(
    prefetched: list[dict], *, horizon_days: int = 21
) -> dict[str, float]:
    """Log forward-return and daily-return distributions (H1 diagnostic)."""
    fwd = build_forward_return_lookup(prefetched, horizon_days=horizon_days)
    fwd_vals = np.array(list(fwd.values()), dtype=np.float64)
    fwd_vals = fwd_vals[np.isfinite(fwd_vals)]

    daily_rets: list[float] = []
    for o in prefetched:
        if o.get("observation_type") != "instrument_daily":
            continue
        v = o.get("value", {})
        if not isinstance(v, dict):
            continue
        lr = v.get("log_return")
        if lr is None:
            continue
        try:
            r = float(lr)
            if np.isfinite(r):
                daily_rets.append(r)
        except (TypeError, ValueError):
            continue

    daily_arr = np.array(daily_rets, dtype=np.float64) if daily_rets else np.array([])

    stats: dict[str, float] = {
        "fwd_n": float(len(fwd_vals)),
        "fwd_mean": float(fwd_vals.mean()) if len(fwd_vals) else float("nan"),
        "fwd_std": float(fwd_vals.std()) if len(fwd_vals) else float("nan"),
        "fwd_min": float(fwd_vals.min()) if len(fwd_vals) else float("nan"),
        "fwd_max": float(fwd_vals.max()) if len(fwd_vals) else float("nan"),
        "fwd_pct_zero": float((np.abs(fwd_vals) < 1e-8).mean()) if len(fwd_vals) else float("nan"),
        "daily_n": float(len(daily_arr)),
        "daily_std": float(daily_arr.std()) if len(daily_arr) else float("nan"),
        "fwd_to_daily_std_ratio": (
            float(fwd_vals.std() / daily_arr.std())
            if len(fwd_vals) and len(daily_arr) and daily_arr.std() > 1e-12
            else float("nan")
        ),
    }
    log.info(
        "FWD_RETURN_DIST: n=%d mean=%.4f std=%.4f min=%.4f max=%.4f pct_zero=%.2f",
        int(stats["fwd_n"]),
        stats["fwd_mean"],
        stats["fwd_std"],
        stats["fwd_min"],
        stats["fwd_max"],
        stats["fwd_pct_zero"] * 100,
    )
    log.info(
        "DAILY_LOG_RETURN: n=%d std=%.6f  fwd/daily_std=%.2f",
        int(stats["daily_n"]),
        stats["daily_std"],
        stats["fwd_to_daily_std_ratio"],
    )
    return stats


# ── Feature matrix at fold cutoff ────────────────────────────────────────────

_FEAT_CACHE: dict[str, np.ndarray] = {}


def _instrument_price_feature_matrix(
    graph_builder: GraphBuilder,
    fold_date: str,
    instrument_names: list[str],
    id_map: Any,
    links: list[dict],
    prefetched: list[dict],
    obs_ts: list[float],
    *,
    model: Any | None = None,
) -> np.ndarray:
    """Price feature block (N, PRICE_FEAT_DIM) at fold cutoff; NaN if missing."""
    cache_key = f"{fold_date}:{len(instrument_names)}"
    if cache_key in _FEAT_CACHE:
        return _FEAT_CACHE[cache_key]

    N = len(instrument_names)
    fold_ts = (
        datetime.fromisoformat(fold_date).replace(tzinfo=timezone.utc).timestamp()
    )
    since_ts = fold_ts - GNN_LOOKBACK_DAYS * 86400
    end_idx = bisect.bisect_left(obs_ts, fold_ts)
    start_idx = bisect.bisect_left(obs_ts, since_ts)
    obs_window = prefetched[start_idx:end_idx]
    if not obs_window:
        return np.full((N, PRICE_FEAT_DIM), np.nan)

    data, local_map, _ = graph_builder.build_from_cached(
        id_map, links, observations=obs_window
    )
    if model is not None:
        _align_graph_features_to_model(data, model)

    if "instrument" not in data.node_types or not hasattr(data["instrument"], "x"):
        return np.full((N, PRICE_FEAT_DIM), np.nan)

    x = data["instrument"].x.detach().cpu().numpy()
    price_start = BASE_FEAT_DIM
    price_end = price_start + PRICE_FEAT_DIM
    out = np.full((N, PRICE_FEAT_DIM), np.nan, dtype=np.float64)
    for i, eid in enumerate(instrument_names):
        local_idx = local_map.local_id("instrument", eid)
        if local_idx is None or local_idx >= x.shape[0]:
            continue
        row = x[local_idx]
        if row.shape[0] >= price_end:
            out[i] = row[price_start:price_end]
    _FEAT_CACHE[cache_key] = out
    return out


def _warm_ic_caches(
    strategies: list[Any],
    dates: list[str],
    returns: np.ndarray,
    instrument_names: list[str],
    *,
    min_train: int,
) -> int:
    """Warm per-fold weight caches for IC (skip full portfolio backtest)."""
    n_folds = 0
    split = min_train
    while split + TEST_SIZE <= len(dates):
        train_returns = returns[:split]
        for strat in strategies:
            strat.generate_weights(
                train_returns, TEST_SIZE, instrument_names
            )
        n_folds += 1
        if n_folds % 5 == 0:
            log.info("  IC cache warm: %d folds (%s)", n_folds, dates[split])
        split += STEP_SIZE
    log.info("  IC cache warm: done — %d folds", n_folds)
    return n_folds


# ── RawPrice-PurgedRanker ──────────────────────────────────────────────────────


class RawPricePurgedRankerStrategy:
    """Ridge on price features only — honest ceiling without GNN."""

    def __init__(
        self,
        graph_builder: GraphBuilder,
        dates: list[str],
        returns: np.ndarray,
        prefetched: list[dict],
        id_map: Any,
        links: list[dict],
        *,
        temperature: float = TEMPERATURE,
        ridge_alpha: float = 1.0,
        min_train: int = MIN_TRAIN,
        fwd_lookup: dict[tuple[str, int], float] | None = None,
    ) -> None:
        self._gb = graph_builder
        self._dates = dates
        self._returns = returns
        self._fwd_lookup = fwd_lookup or {}
        self._obs = prefetched
        self._obs_ts = [o["observed_at"] for o in prefetched]
        self._id_map = id_map
        self._links = links
        self._temperature = temperature
        self._ridge_alpha = ridge_alpha
        self._min_train = min_train
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "RawPrice-PurgedRanker"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        instrument_names: list[str],
        *,
        train_extra: dict | None = None,
        test_extra: dict | None = None,
    ) -> np.ndarray:
        train_len = len(train_returns)
        fold_date = self._dates[train_len]
        if fold_date not in self._cache:
            self._cache[fold_date] = self._compute_weights(
                train_len, fold_date, instrument_names
            )
        return np.tile(self._cache[fold_date], (test_length, 1))

    def _compute_weights(
        self, train_len: int, fold_date: str, instrument_names: list[str]
    ) -> np.ndarray:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        N = len(instrument_names)
        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        train_start = self._min_train
        train_end = train_len
        for split in range(train_start, train_end - TEST_SIZE, STEP_SIZE):
            feats = _instrument_price_feature_matrix(
                self._gb,
                self._dates[split],
                instrument_names,
                self._id_map,
                self._links,
                self._obs,
                self._obs_ts,
            )
            y = forward_return_vector_for_date(
                self._fwd_lookup, instrument_names, self._dates[split]
            )
            for i in range(N):
                if np.all(np.isfinite(feats[i])) and np.isfinite(y[i]):
                    X_rows.append(feats[i])
                    y_rows.append(float(y[i]))
        if len(X_rows) < N * 2:
            log.warning(
                "RawPrice fold %s: only %d train samples — equal weights",
                fold_date,
                len(X_rows),
            )
            return np.ones(N) / N

        X = np.stack(X_rows)
        y = np.array(y_rows, dtype=np.float64)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        ridge = Ridge(alpha=self._ridge_alpha, fit_intercept=True)
        ridge.fit(Xs, y)

        test_feats = _instrument_price_feature_matrix(
            self._gb,
            fold_date,
            instrument_names,
            self._id_map,
            self._links,
            self._obs,
            self._obs_ts,
        )
        scores = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if np.all(np.isfinite(test_feats[i])):
                scores[i] = float(
                    ridge.predict(scaler.transform(test_feats[i].reshape(1, -1)))[0]
                )
        if not np.any(np.isfinite(scores)):
            return np.ones(N) / N
        s = scores.std()
        z = (scores - scores.mean()) / s if s > 1e-8 else np.zeros(N)
        return _softmax(z, self._temperature)

    def compute_fold_ics(
        self,
        dates: list[str],
        returns: np.ndarray,
        instrument_names: list[str] | None = None,
    ) -> np.ndarray:
        from scipy.stats import spearmanr

        fold_ics: list[float] = []
        split = self._min_train
        while split + TEST_SIZE <= len(dates):
            fold_date = dates[split]
            if fold_date not in self._cache:
                split += STEP_SIZE
                continue
            w = self._cache[fold_date]
            if instrument_names and self._fwd_lookup:
                fwd_ret = forward_return_vector_for_date(
                    self._fwd_lookup, instrument_names, fold_date
                )
            else:
                fwd_ret = returns[split : split + TEST_SIZE].mean(axis=0)
            valid = np.isfinite(w) & np.isfinite(fwd_ret)
            if valid.sum() >= 5:
                ic, _ = spearmanr(w[valid], fwd_ret[valid])
                if np.isfinite(ic):
                    fold_ics.append(float(ic))
            split += STEP_SIZE
        return np.array(fold_ics, dtype=np.float64)


# ── Momentum-Rank (single factor) ──────────────────────────────────────────────


class MomentumRankStrategy:
    """Rank by 1-month momentum (price feature index 0)."""

    MOMENTUM_IDX = 0

    def __init__(
        self,
        graph_builder: GraphBuilder,
        dates: list[str],
        prefetched: list[dict],
        id_map: Any,
        links: list[dict],
        *,
        temperature: float = TEMPERATURE,
        min_train: int = MIN_TRAIN,
        fwd_lookup: dict[tuple[str, int], float] | None = None,
    ) -> None:
        self._gb = graph_builder
        self._dates = dates
        self._fwd_lookup = fwd_lookup or {}
        self._obs = prefetched
        self._obs_ts = [o["observed_at"] for o in prefetched]
        self._id_map = id_map
        self._links = links
        self._temperature = temperature
        self._min_train = min_train
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "Momentum-Rank"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        instrument_names: list[str],
        *,
        train_extra: dict | None = None,
        test_extra: dict | None = None,
    ) -> np.ndarray:
        fold_date = self._dates[len(train_returns)]
        if fold_date not in self._cache:
            self._cache[fold_date] = self._compute_weights(fold_date, instrument_names)
        return np.tile(self._cache[fold_date], (test_length, 1))

    def _compute_weights(
        self, fold_date: str, instrument_names: list[str]
    ) -> np.ndarray:
        N = len(instrument_names)
        feats = _instrument_price_feature_matrix(
            self._gb,
            fold_date,
            instrument_names,
            self._id_map,
            self._links,
            self._obs,
            self._obs_ts,
        )
        scores = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if np.isfinite(feats[i, self.MOMENTUM_IDX]):
                scores[i] = feats[i, self.MOMENTUM_IDX]
        if not np.any(np.isfinite(scores)):
            return np.ones(N) / N
        s = scores.std()
        z = (scores - np.nanmean(scores)) / s if s > 1e-8 else np.zeros(N)
        return _softmax(z, self._temperature)

    def compute_fold_ics(
        self,
        dates: list[str],
        returns: np.ndarray,
        instrument_names: list[str] | None = None,
    ) -> np.ndarray:
        from scipy.stats import spearmanr

        fold_ics: list[float] = []
        split = self._min_train
        while split + TEST_SIZE <= len(dates):
            fold_date = dates[split]
            if fold_date not in self._cache:
                split += STEP_SIZE
                continue
            w = self._cache[fold_date]
            if instrument_names and self._fwd_lookup:
                fwd_ret = forward_return_vector_for_date(
                    self._fwd_lookup, instrument_names, fold_date
                )
            else:
                fwd_ret = returns[split : split + TEST_SIZE].mean(axis=0)
            valid = np.isfinite(w) & np.isfinite(fwd_ret)
            if valid.sum() >= 5:
                ic, _ = spearmanr(w[valid], fwd_ret[valid])
                if np.isfinite(ic):
                    fold_ics.append(float(ic))
            split += STEP_SIZE
        return np.array(fold_ics, dtype=np.float64)


def _compute_ic_diagnostic_aligned(
    strategies: list[Any],
    dates: list[str],
    returns: np.ndarray,
    instrument_names: list[str],
) -> dict[str, dict]:
    """IC diagnostic using canonical forward-return labels."""
    results: dict[str, dict] = {}
    for strat in strategies:
        if not hasattr(strat, "compute_fold_ics"):
            continue
        try:
            ics = strat.compute_fold_ics(dates, returns, instrument_names)
        except TypeError:
            ics = strat.compute_fold_ics(dates, returns)
        n = len(ics)
        mean_ic = float(ics.mean()) if n > 0 else 0.0
        std_ic = float(ics.std(ddof=1)) if n > 1 else 0.0
        icir = mean_ic / (std_ic + 1e-8)
        t_stat = (
            (mean_ic / (std_ic / np.sqrt(n))) if (n > 0 and std_ic > 1e-10) else 0.0
        )
        results[strat.name] = {
            "fold_ics": ics.tolist(),
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "t_stat": t_stat,
            "n_folds": n,
            "label": "canonical_forward_return",
        }
    return results


# ── Gate + recommendation ──────────────────────────────────────────────────────


def _gate_passed(mean_ic: float, t_stat: float) -> bool:
    return mean_ic > IC_EXIT_MEAN and t_stat > IC_EXIT_TSTAT


def _recommendation(ic_results: dict[str, dict]) -> str:
    raw = ic_results.get("RawPrice-PurgedRanker", {})
    gnn = ic_results.get("GNN-PurgedRanker", {})
    mom = ic_results.get("Momentum-Rank", {})

    raw_pass = _gate_passed(raw.get("mean_ic", 0.0), raw.get("t_stat", 0.0))
    gnn_pass = _gate_passed(gnn.get("mean_ic", 0.0), gnn.get("t_stat", 0.0))
    mom_pass = _gate_passed(mom.get("mean_ic", 0.0), mom.get("t_stat", 0.0))

    if not raw_pass and not gnn_pass and not mom_pass:
        return "DATA_FIX"
    if raw_pass and not gnn_pass:
        return "TWO_STAGE"
    if raw_pass or gnn_pass:
        return "TWO_STAGE"
    return "INVESTIGATE"


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="Honest baseline audit (Phase A)")
    ap.add_argument("--db-path", type=Path, default=DB_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional GNN checkpoint for GNN-PurgedRanker comparison",
    )
    ap.add_argument(
        "--weights-from-epoch",
        type=Path,
        default=None,
        help="Per-epoch weights (e.g. epoch_090.pt) with --checkpoint metadata shell",
    )
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Fast mode: last 400d, min_train=126 (~7 folds)",
    )
    ap.add_argument(
        "--ic-only",
        action="store_true",
        default=True,
        help="IC diagnostic only — skip full portfolio backtest (default: on)",
    )
    ap.add_argument(
        "--full-backtest",
        action="store_true",
        help="Run full MultiAssetWalkForward backtest (slow; off by default)",
    )
    args = ap.parse_args()
    if args.full_backtest:
        args.ic_only = False

    if not args.db_path.exists():
        print(f"ERROR: DB not found: {args.db_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("HONEST BASELINE AUDIT — Phase A (architecture overhaul)")
    print("=" * 60)

    store = PipelineStore(str(args.db_path))
    gb = GraphBuilder(store)

    # ── A.1 Label audit ──────────────────────────────────────────────────────
    print("\n── A.1 Label distribution audit (H1) ──")
    prefetched = gb.prefetch_observations()
    label_stats = audit_label_distribution(prefetched)

    # ── Load returns matrix ──────────────────────────────────────────────────
    entities = store.query_all_entities()
    entity_ids = [e["entity_id"] for e in entities if e["entity_type"] == "instrument"]
    log.info("Instrument universe: %d", len(entity_ids))

    dates, returns = _load_instrument_returns_fast(str(args.db_path), entity_ids)
    T = len(dates)
    min_train = MIN_TRAIN
    if args.smoke:
        min_train = 126
        if T > 400:
            dates = dates[-400:]
            returns = returns[-400:]
            T = len(dates)
        print(f"  SMOKE MODE: last {T}d, min_train={min_train}")

    if T < min_train + TEST_SIZE:
        print(f"ERROR: Not enough data ({T} rows)")
        sys.exit(1)

    id_map, _, links = gb.prepare_static()
    fwd_lookup = build_forward_return_lookup(prefetched)
    log.info("Forward-return lookup: %d entries (canonical trainer labels)", len(fwd_lookup))

    # ── Build strategies (pre-warm caches via walk-forward) ──────────────────
    from agent.quant.backtest import MultiAssetWalkForward

    strategies: list[Any] = [
        RawPricePurgedRankerStrategy(
            gb,
            dates,
            returns,
            prefetched,
            id_map,
            links,
            ridge_alpha=args.ridge_alpha,
            min_train=min_train,
            fwd_lookup=fwd_lookup,
        ),
        MomentumRankStrategy(
            gb,
            dates,
            prefetched,
            id_map,
            links,
            min_train=min_train,
            fwd_lookup=fwd_lookup,
        ),
    ]

    trainer = None
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        from agent.models.gnn.trainer import Trainer

        if args.weights_from_epoch is not None:
            log.info(
                "Loading GNN: metadata %s, weights %s",
                args.checkpoint,
                args.weights_from_epoch,
            )
            trainer = Trainer.load_model_with_epoch_weights(
                args.checkpoint, args.weights_from_epoch, store
            )
        else:
            log.info("Loading GNN checkpoint: %s", args.checkpoint)
            trainer = Trainer.load_model(args.checkpoint, store)
        strategies.append(
            GNNFoldPurgedRankerStrategy(
                trainer,
                dates,
                returns,
                prefetched,
                id_map,
                links,
                ridge_alpha=args.ridge_alpha,
            )
        )

    print("\n── A.2–A.4 Walk-forward baselines ──")
    _FEAT_CACHE.clear()
    if args.ic_only:
        print("  IC-ONLY mode (skipping portfolio backtest; feature cache enabled)")
        log.info("Warming IC caches for %d strategies…", len(strategies))
        _warm_ic_caches(
            strategies, dates, returns, entity_ids, min_train=min_train
        )
    else:
        runner = MultiAssetWalkForward(
            min_train=min_train,
            test_size=TEST_SIZE,
            step_size=STEP_SIZE,
            instrument_names=entity_ids,
            instrument_classes={},
            periods_per_year=252,
        )
        for strat in strategies:
            log.info("Running full backtest: %s", strat.name)
            runner.run(strat, returns)

    ic_results = _compute_ic_diagnostic_aligned(
        strategies, dates, returns, entity_ids
    )
    _print_ic_report(ic_results, primary_strategy="RawPrice-PurgedRanker")
    print("  (IC targets: canonical 21d forward simple return — matches trainer)")

    recommendation = _recommendation(ic_results)
    print(f"\n  RECOMMENDATION: {recommendation}")
    if recommendation == "DATA_FIX":
        print("  → Labels/features may not support IC>0.03. Fix data pipeline before new architecture.")
    elif recommendation == "TWO_STAGE":
        print("  → Signal exists. Proceed to decoupled two-stage pipeline (Phase B).")
    else:
        print("  → Mixed results. Review per-strategy IC before Phase B.")

    # Gate summary
    print("\n  Gate results (IC > 0.03 AND t > 2.0):")
    for name, r in ic_results.items():
        passed = _gate_passed(r["mean_ic"], r["t_stat"])
        status = "PASS" if passed else "FAIL"
        print(f"    {name:<26} {status}  IC={r['mean_ic']:+.4f}  t={r['t_stat']:.2f}")

    out = {
        "audit": "honest_baseline_phase_a",
        "db_path": str(args.db_path),
        "smoke": args.smoke,
        "label_distribution": label_stats,
        "ic_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "fold_ics"}
            for k, v in ic_results.items()
        },
        "fold_ics": {k: v["fold_ics"] for k, v in ic_results.items()},
        "recommendation": recommendation,
        "gate_threshold": {"mean_ic": IC_EXIT_MEAN, "t_stat": IC_EXIT_TSTAT},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n  Results written → {args.out}\n")


if __name__ == "__main__":
    main()
