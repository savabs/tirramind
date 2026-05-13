#!/usr/bin/env python3
"""Phase 40 — GNN-Signal Walk-Forward Backtest.

Loads the trained HetTGN model and runs a multi-asset walk-forward backtest
comparing three strategies:

  1. EqualWeight     — 1/N across all instruments (baseline, target Sharpe ~0.995)
  2. GNN-EmbNorm     — cross-sectional signal from instrument embedding L2 norms
  3. GNN-ValueHead   — value_pred_head output for instrument nodes, trained to
                       predict return quantile buckets

Signal construction (EmbNorm):
    s_i = ||h_i||_2  (L2 norm of 128-dim GNN embedding for instrument i)
    z_i = (s_i − mean_s) / std_s   (cross-sectional z-score)
    w_i = softmax(z_i × temperature)   (long-only, sums to 1)

Signal construction (ValueHead):
    v_i = value_pred_head(h_i)   (scalar ∈ [0, 1] — predicted return quantile)
    w_i = softmax(v_i × temperature)

Temporal gating:
    For fold with training rows [:split], the fold cutoff timestamp is
    dates[split] (first day of the test window). The GNN graph is built
    with `until=fold_ts` so only observations before the test window
    are visible — no forward-looking leakage in node features.

    Note: the model's TGN memory was trained on all data (in-sample).
    Phase 41 will address this via per-fold GNN retraining.

Usage:
    /home/becmachlean/anaconda3/bin/python scripts/phase40_gnn_backtest.py
"""

from __future__ import annotations

import bisect
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase40")

# ── Constants ────────────────────────────────────────────────────────────────

DB_PATH = Path(".tirra_pipeline/pipeline.db")
MODEL_PATH = Path(".tirra_pipeline/gnn_model.pt")

TEMPERATURE = 1.0  # softmax temperature for signal→weight conversion
MIN_TRAIN = 252  # calendar days in training window
TEST_SIZE = 21  # monthly test window
STEP_SIZE = 21  # roll forward one month per fold
GNN_LOOKBACK_DAYS = 90  # max observation history fed to GNN node features per fold
# 90d covers ~3 months of geopolitical event recency
# _build_node_features is recency-weighted → old obs negligible


# ── GNN-aware MultiAssetStrategy ─────────────────────────────────────────────


class GNNEmbeddingNormStrategy:
    """Cross-sectional signal: instrument GNN embedding L2 norm.

    High embedding norm ↔ high information flow through this node in the
    heterogeneous event graph → proxy for activity / momentum signal.

    Parameters
    ----------
    trainer     : loaded Trainer (Trainer.load_model result)
    dates       : sorted list of ISO date strings, length T
    prefetched  : all observations pre-sorted by observed_at (from prefetch_observations)
    id_map      : full IDMap covering all entities (from prepare_static)
    links       : all entity links (from prepare_static)
    temperature : softmax temperature
    """

    def __init__(
        self,
        trainer: Any,
        dates: list[str],
        prefetched: list[dict],
        id_map: Any,
        links: list[dict],
        temperature: float = TEMPERATURE,
    ) -> None:
        self._trainer = trainer
        self._dates = dates
        self._obs = prefetched  # sorted by observed_at
        self._obs_ts = [o["observed_at"] for o in prefetched]  # for bisect
        self._id_map = id_map
        self._links = links
        self._temperature = temperature
        self._cache: dict[str, np.ndarray] = {}  # date_str → weight vector

    @property
    def name(self) -> str:
        return "GNN-EmbNorm"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        instrument_names: list[str],
        *,
        train_extra: dict | None = None,
        test_extra: dict | None = None,
    ) -> np.ndarray:
        fold_date = self._dates[len(train_returns)]  # first day of test window
        if fold_date not in self._cache:
            self._cache[fold_date] = self._compute_weights(fold_date, instrument_names)
        w = self._cache[fold_date]
        return np.tile(w, (test_length, 1))

    def _compute_weights(
        self, fold_date: str, instrument_names: list[str]
    ) -> np.ndarray:
        import torch

        N = len(instrument_names)

        fold_ts = (
            datetime.fromisoformat(fold_date).replace(tzinfo=timezone.utc).timestamp()
        )

        # Slice observations: [fold_ts - lookback, fold_ts)
        # Node features are recency-weighted so old obs add noise not signal.
        since_ts = fold_ts - GNN_LOOKBACK_DAYS * 86400
        end_idx = bisect.bisect_left(self._obs_ts, fold_ts)
        start_idx = bisect.bisect_left(self._obs_ts, since_ts)
        obs_window = self._obs[start_idx:end_idx]

        if not obs_window:
            log.warning("No observations before %s — using equal weights", fold_date)
            return np.ones(N) / N

        data, id_map, _ = self._trainer._graph_builder.build_from_cached(
            self._id_map, self._links, observations=obs_window
        )

        model = self._trainer._model
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)  # {node_type: (N_type, hidden_dim)}

        inst_emb = embeddings.get("instrument")
        if inst_emb is None or inst_emb.shape[0] == 0:
            log.warning(
                "No instrument embeddings at %s — using equal weights", fold_date
            )
            return np.ones(N) / N

        # Map entity_ids → embedding norms
        norms = np.zeros(N, dtype=np.float64)
        found = 0
        for i, eid in enumerate(instrument_names):
            local_idx = id_map.local_id("instrument", eid)
            if local_idx is not None:
                norms[i] = float(inst_emb[local_idx].norm().item())
                found += 1

        if found == 0:
            return np.ones(N) / N

        # Cross-sectional z-score → softmax
        s = norms.std()
        z = (norms - norms.mean()) / s if s > 1e-8 else np.zeros(N)
        return _softmax(z, self._temperature)

    def compute_fold_ics(self, dates: list[str], returns: np.ndarray) -> np.ndarray:
        """Spearman IC per fold using cached weights.

        rank(softmax(z)) == rank(z) == rank(norm) — monotone chain — so IC
        computed on the weight vector is identical to IC on raw embedding norms.
        No additional GNN forward passes needed (reads self._cache populated by
        walk-forward run).
        """
        from scipy.stats import spearmanr

        fold_ics: list[float] = []
        split = MIN_TRAIN
        while split + TEST_SIZE <= len(dates):
            fold_date = dates[split]
            if fold_date not in self._cache:
                split += STEP_SIZE
                continue
            w = self._cache[fold_date]  # shape (N,)
            fwd_ret = returns[split : split + TEST_SIZE].mean(axis=0)
            valid = np.isfinite(w) & np.isfinite(fwd_ret)
            if valid.sum() >= 5:
                ic, _ = spearmanr(w[valid], fwd_ret[valid])
                if np.isfinite(ic):
                    fold_ics.append(float(ic))
            split += STEP_SIZE
        return np.array(fold_ics, dtype=np.float64)


class GNNReturnHeadStrategy:
    """Cross-sectional signal: return_pred_head output for instrument nodes.

    The return head (Phase 41) was directly supervised on log_return targets
    from instrument_daily observations.  This is the head most likely to carry
    forward-return IC — it is the only head trained on an explicitly return-
    labelled objective.

    Parameters: same as GNNEmbeddingNormStrategy.
    """

    def __init__(
        self,
        trainer: Any,
        dates: list[str],
        prefetched: list[dict],
        id_map: Any,
        links: list[dict],
        temperature: float = TEMPERATURE,
    ) -> None:
        self._trainer = trainer
        self._dates = dates
        self._obs = prefetched
        self._obs_ts = [o["observed_at"] for o in prefetched]
        self._id_map = id_map
        self._links = links
        self._temperature = temperature
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "GNN-ReturnHead"

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
        w = self._cache[fold_date]
        return np.tile(w, (test_length, 1))

    def _compute_weights(
        self, fold_date: str, instrument_names: list[str]
    ) -> np.ndarray:
        import torch

        N = len(instrument_names)

        fold_ts = (
            datetime.fromisoformat(fold_date).replace(tzinfo=timezone.utc).timestamp()
        )

        since_ts = fold_ts - GNN_LOOKBACK_DAYS * 86400
        end_idx = bisect.bisect_left(self._obs_ts, fold_ts)
        start_idx = bisect.bisect_left(self._obs_ts, since_ts)
        obs_window = self._obs[start_idx:end_idx]

        if not obs_window:
            return np.ones(N) / N

        data, id_map, _ = self._trainer._graph_builder.build_from_cached(
            self._id_map, self._links, observations=obs_window
        )

        model = self._trainer._model
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)
            inst_emb = embeddings.get("instrument")

        if inst_emb is None or inst_emb.shape[0] == 0:
            return np.ones(N) / N

        ret_preds = model.return_pred_head(inst_emb).squeeze(-1)  # (n_inst,)

        scores = np.zeros(N, dtype=np.float64)
        found = 0
        for i, eid in enumerate(instrument_names):
            local_idx = id_map.local_id("instrument", eid)
            if local_idx is not None:
                scores[i] = float(ret_preds[local_idx].item())
                found += 1

        if found == 0:
            return np.ones(N) / N

        return _softmax(scores, self._temperature)

    def compute_fold_ics(self, dates: list[str], returns: np.ndarray) -> np.ndarray:
        """Spearman IC per fold using cached return-head scores."""
        from scipy.stats import spearmanr

        fold_ics: list[float] = []
        split = MIN_TRAIN
        while split + TEST_SIZE <= len(dates):
            fold_date = dates[split]
            if fold_date not in self._cache:
                split += STEP_SIZE
                continue
            w = self._cache[fold_date]
            fwd_ret = returns[split : split + TEST_SIZE].mean(axis=0)
            valid = np.isfinite(w) & np.isfinite(fwd_ret)
            if valid.sum() >= 5:
                ic, _ = spearmanr(w[valid], fwd_ret[valid])
                if np.isfinite(ic):
                    fold_ics.append(float(ic))
            split += STEP_SIZE
        return np.array(fold_ics, dtype=np.float64)


class GNNValueHeadStrategy:
    """Cross-sectional signal: value_pred_head output for instrument nodes.

    The value head was trained to predict the quantile bucket of the next
    observation's value (≈ next log-return quantile, normalised to [0,1]).
    Higher predicted value → instrument expected to have better next return.

    Parameters: same as GNNEmbeddingNormStrategy.
    """

    def __init__(
        self,
        trainer: Any,
        dates: list[str],
        prefetched: list[dict],
        id_map: Any,
        links: list[dict],
        temperature: float = TEMPERATURE,
    ) -> None:
        self._trainer = trainer
        self._dates = dates
        self._obs = prefetched
        self._obs_ts = [o["observed_at"] for o in prefetched]
        self._id_map = id_map
        self._links = links
        self._temperature = temperature
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "GNN-ValueHead"

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
        w = self._cache[fold_date]
        return np.tile(w, (test_length, 1))

    def _compute_weights(
        self, fold_date: str, instrument_names: list[str]
    ) -> np.ndarray:
        import torch

        N = len(instrument_names)

        fold_ts = (
            datetime.fromisoformat(fold_date).replace(tzinfo=timezone.utc).timestamp()
        )

        since_ts = fold_ts - GNN_LOOKBACK_DAYS * 86400
        end_idx = bisect.bisect_left(self._obs_ts, fold_ts)
        start_idx = bisect.bisect_left(self._obs_ts, since_ts)
        obs_window = self._obs[start_idx:end_idx]

        if not obs_window:
            return np.ones(N) / N

        data, id_map, _ = self._trainer._graph_builder.build_from_cached(
            self._id_map, self._links, observations=obs_window
        )

        model = self._trainer._model
        model.eval()
        with torch.no_grad():
            embeddings = model(data, id_map)
            value_preds = model.predict_value(embeddings)  # {ntype: (N, 1)}

        inst_vals = value_preds.get("instrument")
        if inst_vals is None or inst_vals.shape[0] == 0:
            return np.ones(N) / N

        # Map entity_ids → predicted values
        vals = np.zeros(N, dtype=np.float64)
        found = 0
        for i, eid in enumerate(instrument_names):
            local_idx = id_map.local_id("instrument", eid)
            if local_idx is not None:
                vals[i] = float(inst_vals[local_idx].item())
                found += 1

        if found == 0:
            return np.ones(N) / N

        return _softmax(vals, self._temperature)

    def compute_fold_ics(self, dates: list[str], returns: np.ndarray) -> np.ndarray:
        """Spearman IC per fold using cached value-head weights.

        rank(softmax(val)) == rank(val) — monotone — so IC on weight vector
        equals IC on raw value-head predictions.  No extra GNN passes.
        """
        from scipy.stats import spearmanr

        fold_ics: list[float] = []
        split = MIN_TRAIN
        while split + TEST_SIZE <= len(dates):
            fold_date = dates[split]
            if fold_date not in self._cache:
                split += STEP_SIZE
                continue
            w = self._cache[fold_date]  # shape (N,)
            fwd_ret = returns[split : split + TEST_SIZE].mean(axis=0)
            valid = np.isfinite(w) & np.isfinite(fwd_ret)
            if valid.sum() >= 5:
                ic, _ = spearmanr(w[valid], fwd_ret[valid])
                if np.isfinite(ic):
                    fold_ics.append(float(ic))
            split += STEP_SIZE
        return np.array(fold_ics, dtype=np.float64)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    scaled = x * temperature
    scaled = scaled - scaled.max()  # subtract max for numerical stability
    exp_x = np.exp(scaled)
    return exp_x / exp_x.sum()


def _load_instrument_returns_fast(
    db_path: str,
    entity_ids: list[str],
) -> tuple[list[str], np.ndarray]:
    """Load instrument daily log-returns via direct SQL (skips 977K GDELT rows).

    Uses a typed WHERE clause on observation_type='instrument_daily' so only
    the 68K price rows are fetched, not the full 977K observation set.

    Returns
    -------
    (dates, returns) — same contract as walkforward_runner.load_instrument_returns
    """
    import json
    import sqlite3
    from collections import defaultdict
    from datetime import datetime, timezone

    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in entity_ids)
    rows = conn.execute(
        f"SELECT entity_id, observed_at, value_json "  # noqa: S608
        f"FROM entity_observations "
        f"WHERE observation_type='instrument_daily' "
        f"AND entity_id IN ({placeholders}) "
        f"ORDER BY observed_at",
        entity_ids,
    ).fetchall()
    conn.close()

    data: dict[str, dict[str, float]] = defaultdict(dict)
    for eid, ts, val_json in rows:
        val = json.loads(val_json) if val_json else {}
        lr = val.get("log_return")
        if lr is None:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day = dt.strftime("%Y-%m-%d")
        data[day][eid] = float(lr)

    if not data:
        raise ValueError("No instrument_daily observations found in DB.")

    dates = sorted(data.keys())
    idx = {t: i for i, t in enumerate(entity_ids)}
    N = len(entity_ids)
    T = len(dates)
    returns = np.zeros((T, N), dtype=np.float64)
    for t, day in enumerate(dates):
        for eid, lr in data[day].items():
            if eid in idx:
                returns[t, idx[eid]] = lr

    log.info(
        "Fast price load: %d dates × %d instruments (%s → %s)",
        T,
        N,
        dates[0],
        dates[-1],
    )
    return dates, returns


def _compute_ic_diagnostic(
    strategies: list,
    dates: list[str],
    returns: np.ndarray,
) -> dict[str, dict]:
    """Compute fold-level Spearman IC for each GNN strategy.

    IC_t = Spearman( score_i_t , fwd_return_i_t )

    fwd_return_i_t = mean daily log-return over TEST_SIZE days after fold cutoff.
    Score is the per-instrument portfolio weight (rank-preserving proxy for raw
    GNN signal — softmax is monotone so Spearman is unchanged).

    Also computes ICIR = mean_IC / std_IC (Grinold & Kahn 2000) — the
    standard quant measure of signal consistency. Unlike t-stat, ICIR
    does not depend on fold count and is directly comparable across runs.

    Returns
    -------
    dict  strategy_name → {fold_ics, mean_ic, std_ic, t_stat, n_folds}
    """
    gnn_strats = [s for s in strategies if hasattr(s, "compute_fold_ics")]
    results: dict[str, dict] = {}
    for strat in gnn_strats:
        ics = strat.compute_fold_ics(dates, returns)
        n = len(ics)
        mean_ic = float(ics.mean()) if n > 0 else 0.0
        std_ic = float(ics.std(ddof=1)) if n > 1 else 0.0
        # ICIR = mean_IC / std_IC — signal consistency metric (Grinold & Kahn 2000).
        # ICIR > 0.40 = has real signal; ICIR > 0.0 = directionally consistent.
        # Unlike t-stat, ICIR is fold-count-independent and comparable across runs.
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
        }
    return results


def _print_ic_report(ic_results: dict) -> None:
    """Print IC diagnostic with quantile distribution and verdict."""
    print("\n" + "=" * 60)
    print("IC DIAGNOSTIC  — Spearman(score_i, 21d_fwd_return_i) per fold")
    print("=" * 60)
    hdr = f"  {'Strategy':<22} {'Mean IC':>9} {'Std IC':>8} {'ICIR':>7} {'t-stat':>8} {'p25':>7} {'p75':>7} {'Folds':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, r in ic_results.items():
        ics = np.array(r["fold_ics"])
        p25 = float(np.percentile(ics, 25)) if len(ics) > 0 else 0.0
        p75 = float(np.percentile(ics, 75)) if len(ics) > 0 else 0.0
        print(
            f"  {name:<22} {r['mean_ic']:>9.4f} {r['std_ic']:>8.4f}"
            f" {r.get('icir', 0.0):>7.3f} {r['t_stat']:>8.2f}"
            f" {p25:>7.4f} {p75:>7.4f} {r['n_folds']:>6d}"
        )
    print()
    print("  Thresholds (ICIR is the primary metric — fold-count independent):")
    print("    ICIR > 0.40   →  real signal (Grinold & Kahn 2000)")
    print("    ICIR > 0.20   →  directional signal, worth investigating")
    print("    ICIR < 0.10   →  noise")
    print("    (legacy) |Mean IC| > 0.05 AND |t| > 2.0  →  statistically significant")
    for name, r in ic_results.items():
        mic, t = r["mean_ic"], r["t_stat"]
        if abs(mic) < 0.02 or abs(t) < 1.0:
            verdict = "NO SIGNAL — embedding does not encode return info"
        elif abs(mic) < 0.05 or abs(t) < 2.0:
            verdict = "MARGINAL — present but not statistically reliable"
        else:
            verdict = "REAL SIGNAL — statistically significant"
        direction = "positive" if mic > 0 else "negative"
        print(f"\n  {name}: {verdict}")
        print(f"    IC={mic:+.4f} ({direction}), t={t:+.2f}, n={r['n_folds']} folds")


def _print_result(name: str, m: dict, n_folds: int) -> None:
    print(f"\n{name}")
    print("-" * 50)
    print(f"  Folds:         {n_folds}")
    total_ret = m.get("total_return", 0)
    print(f"  Total Return:  {total_ret:.4f}  ({total_ret*100:.2f}%)")
    print(f"  Sharpe Ratio:  {m.get('sharpe', float('nan')):.3f}")
    max_dd = m.get("max_drawdown", 0)
    print(f"  Max Drawdown:  {max_dd:.4f}  ({max_dd*100:.2f}%)")
    print(f"  Win Rate:      {m.get('win_rate', 0):.3f}")
    print(f"  Volatility:    {m.get('volatility', 0):.4f}")
    print(f"  Max Weight:    {m.get('max_weight', 0):.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    import json as _json_bt
    import torch
    from agent.models.gnn.trainer import Trainer
    from agent.pipeline.store import PipelineStore
    from agent.quant.backtest import EqualWeightStrategy, MultiAssetWalkForward

    ap = argparse.ArgumentParser(description="Phase 40 GNN walk-forward backtest")
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH,
                    help="Path to GNN checkpoint .pt file")
    ap.add_argument("--db-path", type=Path, default=DB_PATH,
                    help="Path to pipeline.db")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write IC summary JSON so auto_improve.py can read it")
    args = ap.parse_args()
    _model_path: Path = args.model_path
    _db_path: Path = args.db_path
    _out_path: Path | None = args.out

    # ── 1. Validate paths ────────────────────────────────────────────────────
    if not _db_path.exists():
        print(f"ERROR: DB not found at {_db_path}. Run daily_collection first.")
        sys.exit(1)
    if not _model_path.exists():
        print(f"ERROR: Model not found at {_model_path}. Run retrain_gnn.py first.")
        sys.exit(1)

    # ── 2. Load store + instrument universe ──────────────────────────────────
    log.info("Loading PipelineStore from %s", _db_path)
    store = PipelineStore(str(_db_path))

    entities = store.query_all_entities()
    entity_ids: list[str] = []
    instrument_classes: dict[str, str] = {}
    for e in entities:
        if e["entity_type"] != "instrument":
            continue
        meta = e.get("metadata") or {}
        entity_ids.append(e["entity_id"])
        instrument_classes[e["entity_id"]] = meta.get("asset_class", "unknown")

    log.info("Instrument universe: %d instruments", len(entity_ids))

    # ── 3. Load price returns (fast: typed SQL, skips 977K GDELT rows) ────────
    log.info("Loading instrument returns (fast SQL path)…")
    dates, returns = _load_instrument_returns_fast(str(_db_path), entity_ids)
    T, N = returns.shape
    log.info(
        "Returns matrix: %d dates × %d instruments  (%s → %s)",
        T,
        N,
        dates[0],
        dates[-1],
    )

    if T < MIN_TRAIN + TEST_SIZE:
        print(f"ERROR: Not enough data ({T} rows). Need ≥ {MIN_TRAIN + TEST_SIZE}.")
        sys.exit(1)

    # ── 4. Load trained GNN model ─────────────────────────────────────────────
    log.info("Loading GNN model from %s", _model_path)
    trainer = Trainer.load_model(_model_path, store)
    log.info(
        "Model loaded: %d params", sum(p.numel() for p in trainer.model.parameters())
    )

    # ── 5. Pre-fetch graph structure once (fast per-fold builds) ──────────────
    log.info("Pre-fetching graph structure (entities + links + all observations)…")
    full_id_map, _, full_links = trainer._graph_builder.prepare_static()
    prefetched_obs = trainer._graph_builder.prefetch_observations()
    log.info(
        "Pre-fetched: %d entities, %d links, %d observations",
        full_id_map.num_nodes,
        len(full_links),
        len(prefetched_obs),
    )

    # ── 6. Build strategies ──────────────────────────────────────────────────
    strategies = [
        EqualWeightStrategy(),
        GNNEmbeddingNormStrategy(
            trainer, dates, prefetched_obs, full_id_map, full_links
        ),
        GNNValueHeadStrategy(trainer, dates, prefetched_obs, full_id_map, full_links),
        GNNReturnHeadStrategy(trainer, dates, prefetched_obs, full_id_map, full_links),
    ]

    # ── 7. Walk-forward runner ────────────────────────────────────────────────
    runner = MultiAssetWalkForward(
        min_train=MIN_TRAIN,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        instrument_names=entity_ids,
        instrument_classes=instrument_classes,
        periods_per_year=252,
    )

    n_folds = (T - MIN_TRAIN - TEST_SIZE) // STEP_SIZE + 1
    log.info(
        "Walk-forward config: min_train=%d, test=%d, step=%d → ~%d folds",
        MIN_TRAIN,
        TEST_SIZE,
        STEP_SIZE,
        n_folds,
    )

    results = {}
    for strat in strategies:
        log.info("Running strategy: %s", strat.name)
        result = runner.run(strat, returns)
        results[strat.name] = result

    # ── 8. Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 40 — GNN WALK-FORWARD BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Universe:    {N} instruments")
    print(f"  Period:      {dates[0]} → {dates[-1]}")
    print(f"  Folds:       {len(results[strategies[0].name].folds)}")
    print(f"  Window:      {MIN_TRAIN}d train / {TEST_SIZE}d test / {STEP_SIZE}d step")
    print(
        f"  GNN:         {_model_path.name}  ({sum(p.numel() for p in trainer.model.parameters()):,} params)"
    )

    for strat in strategies:
        r = results[strat.name]
        _print_result(strat.name, r.aggregate_metrics, len(r.folds))

    # ── 9. Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON  (GNN vs EqualWeight baseline)")
    print("=" * 60)
    baseline = results["equal_weight"].aggregate_metrics
    for strat in strategies[1:]:
        m = results[strat.name].aggregate_metrics
        delta_sharpe = m.get("sharpe", 0) - baseline.get("sharpe", 0)
        delta_ret = m.get("total_return", 0) - baseline.get("total_return", 0)
        delta_dd = m.get("max_drawdown", 0) - baseline.get("max_drawdown", 0)
        print(f"\n  {strat.name} vs EqualWeight:")
        print(f"    ΔSharpe:      {delta_sharpe:+.3f}")
        print(f"    ΔTotal Ret:   {delta_ret:+.4f}  ({delta_ret*100:+.2f}%)")
        print(f"    ΔMax Drawdown:{delta_dd:+.4f}  ({delta_dd*100:+.2f}%)")

    print()

    # ── 10. IC Diagnostic (with ICIR) ─────────────────────────────────────────
    # Uses fold weights already cached during step 7 — no extra GNN passes.
    # rank(softmax(score)) == rank(score), so Spearman IC is exact.
    # ICIR = mean_IC / std_IC added as primary metric (Grinold & Kahn 2000).
    log.info("Computing IC diagnostic (signal quality — no extra GNN passes)…")
    ic_results = _compute_ic_diagnostic(strategies, dates, returns)
    _print_ic_report(ic_results)

    # ── 10b. Write IC summary JSON (for auto_improve.py) ────
    _ic_summary: dict = {"model_path": str(_model_path), "strategies": ic_results}
    _best_strat = max(ic_results, key=lambda k: ic_results[k]["mean_ic"], default=None)
    if _best_strat:
        _ic_summary["best"] = {"strategy": _best_strat, **ic_results[_best_strat]}
    _default_ic_out = _model_path.parent / "ic_results.json"
    for _dest in dict.fromkeys(d for d in [_default_ic_out, _out_path] if d is not None):
        try:
            _dest.parent.mkdir(parents=True, exist_ok=True)
            _dest.write_text(_json_bt.dumps(_ic_summary, indent=2))
            print(f"  IC results → {_dest}")
        except OSError as _e:
            log.warning("Could not write IC results to %s: %s", _dest, _e)

    # ── 11. Stratified IC — source attribution ────────────────────────────────
    # Partitions instrument universe by data source coverage (has_cftc / no_cftc,
    # has_polymarket / no_polymarket, has_geo_link / no_geo_link).
    # Computes IC per partition so we know EXACTLY which sources contribute signal.
    # This answers: "does CFTC data make the GNN more predictive for those instruments?"
    # No retraining needed — uses fold weight cache from step 7.
    from agent.quant.experiment_tracker import (
        ExperimentTracker,
        compute_stratified_ic,
        print_stratified_ic_report,
    )

    log.info("Computing stratified IC (source attribution)…")
    stratified = compute_stratified_ic(
        strategies=strategies,
        dates=dates,
        returns=returns,
        instrument_names=entity_ids,
        db_path=str(_db_path),
        min_train=MIN_TRAIN,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
    )
    print_stratified_ic_report(stratified)

    # ── 12. Experiment manifest — auto-saved every run ────────────────────────
    # Writes JSON to .tirra_pipeline/experiments/exp_{timestamp}.json
    # Contains: data snapshot, IC results (with ICIR), stratified IC, model state.
    # Run 'python scripts/compare_experiments.py' to diff two runs.
    tracker = ExperimentTracker(_db_path, _model_path)
    manifest = tracker.build_manifest(
        ic_results=ic_results,
        stratified_ic=stratified,
        extra={
            "n_instruments": N,
            "n_dates": T,
            "date_range": [dates[0], dates[-1]],
            "min_train": MIN_TRAIN,
            "test_size": TEST_SIZE,
            "step_size": STEP_SIZE,
        },
    )
    manifest_path = tracker.save(manifest)
    print(f"\n  Manifest saved → {manifest_path}")
    print("  Run 'python scripts/compare_experiments.py' to compare runs.")


if __name__ == "__main__":
    main()
