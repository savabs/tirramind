#!/usr/bin/env python3
"""Source Ablation Test — Inference-Time Data Attribution

Answers the question: "How much does each data source contribute to GNN signal?"

Method: For each major source_tool, removes that source's observations from the
90-day lookback window before each GNN forward pass, then recomputes IC.
The delta (IC_full - IC_masked) is the inference-time contribution of that source.

This does NOT require retraining. The model is fixed; we vary what it sees.
Key insight: if removing CFTC data drops IC by 0.02, then CFTC positioning
is adding 0.02 IC at inference time — even with the current model.

After CFTC backfill, this should show a larger delta because the model now
has richer CFTC signal in its lookback window.

Usage:
    python scripts/source_ablation.py [--db-path PATH] [--model-path PATH]
    python scripts/source_ablation.py --sources cftc,gdelt,polymarket
"""

from __future__ import annotations

import argparse
import bisect
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("source_ablation")

DB_PATH = Path(".tirra_pipeline/pipeline.db")
MODEL_PATH = Path(".tirra_pipeline/gnn_model.pt")

MIN_TRAIN = 252
TEST_SIZE = 21
STEP_SIZE = 21
GNN_LOOKBACK_DAYS = 90

# Sources to ablate — in priority order (highest expected contribution first)
DEFAULT_SOURCES = [
    "cftc",           # CFTC managed money positioning → commodity futures
    "gdelt",          # Geopolitical events → country nodes → instruments
    "polymarket",     # Prediction market odds → topic nodes → instruments
    "whale_alert",    # Crypto large transfers → wallet nodes
    "sovereign_debt", # Country credit spreads → country nodes
    "global_pmi",     # PMI → country nodes
    "capital_flows",  # Cross-border flows → country nodes
]


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = x * temperature
    scaled = scaled - scaled.max()
    exp_x = np.exp(scaled)
    return exp_x / exp_x.sum()


def _compute_ic_for_weights(
    weight_cache: dict[str, np.ndarray],
    dates: list[str],
    returns: np.ndarray,
) -> dict:
    from scipy.stats import spearmanr

    fold_ics: list[float] = []
    split = MIN_TRAIN
    while split + TEST_SIZE <= len(dates):
        fold_date = dates[split]
        if fold_date not in weight_cache:
            split += STEP_SIZE
            continue
        w = weight_cache[fold_date]
        fwd_ret = returns[split : split + TEST_SIZE].mean(axis=0)
        valid = np.isfinite(w) & np.isfinite(fwd_ret)
        if valid.sum() >= 5:
            ic, _ = spearmanr(w[valid], fwd_ret[valid])
            if np.isfinite(ic):
                fold_ics.append(float(ic))
        split += STEP_SIZE

    ics = np.array(fold_ics)
    n = len(ics)
    mean_ic = float(ics.mean()) if n > 0 else 0.0
    std_ic = float(ics.std(ddof=1)) if n > 1 else 0.0
    icir = mean_ic / (std_ic + 1e-8)
    t_stat = (mean_ic / (std_ic / np.sqrt(n))) if (n > 0 and std_ic > 1e-10) else 0.0
    return {"mean_ic": mean_ic, "std_ic": std_ic, "icir": icir, "t_stat": t_stat, "n_folds": n}


def _build_weights_with_masked_source(
    trainer,
    dates: list[str],
    all_obs: list[dict],
    all_obs_ts: list[float],
    id_map,
    links: list[dict],
    instrument_names: list[str],
    masked_source: str | None = None,
) -> dict[str, np.ndarray]:
    """Build per-fold weight vectors using the return head, with optional source masking.

    If masked_source is None, uses full observation set (baseline).
    If masked_source is set, strips that source's observations from every fold window.
    """
    import torch

    N = len(instrument_names)
    weight_cache: dict[str, np.ndarray] = {}

    split = MIN_TRAIN
    while split + TEST_SIZE <= len(dates):
        fold_date = dates[split]
        fold_ts = datetime.fromisoformat(fold_date).replace(tzinfo=timezone.utc).timestamp()
        since_ts = fold_ts - GNN_LOOKBACK_DAYS * 86400

        end_idx = bisect.bisect_left(all_obs_ts, fold_ts)
        start_idx = bisect.bisect_left(all_obs_ts, since_ts)
        obs_window = all_obs[start_idx:end_idx]

        # Apply source mask
        if masked_source is not None:
            obs_window = [o for o in obs_window if o.get("source_tool") != masked_source]

        if not obs_window:
            weight_cache[fold_date] = np.ones(N) / N
            split += STEP_SIZE
            continue

        try:
            data, fold_id_map, _ = trainer._graph_builder.build_from_cached(
                id_map, links, observations=obs_window
            )
            model = trainer._model
            model.eval()
            with torch.no_grad():
                embeddings = model(data, fold_id_map)
                inst_emb = embeddings.get("instrument")

            if inst_emb is None or inst_emb.shape[0] == 0:
                weight_cache[fold_date] = np.ones(N) / N
                split += STEP_SIZE
                continue

            ret_preds = model.return_pred_head(inst_emb).squeeze(-1)
            scores = np.zeros(N, dtype=np.float64)
            for i, eid in enumerate(instrument_names):
                local_idx = fold_id_map.local_id("instrument", eid)
                if local_idx is not None:
                    scores[i] = float(ret_preds[local_idx].item())

            weight_cache[fold_date] = _softmax(scores)
        except Exception as exc:
            log.warning("Fold %s failed (masked=%s): %s", fold_date, masked_source, exc)
            weight_cache[fold_date] = np.ones(N) / N

        split += STEP_SIZE

    return weight_cache


def run_ablation(
    db_path: Path,
    model_path: Path,
    sources: list[str],
) -> dict[str, dict]:
    """Run source ablation for GNN-ReturnHead strategy.

    Returns {source_name: {baseline, masked, delta_ic, delta_icir}}
    """
    import torch

    from agent.models.gnn.trainer import Trainer
    from agent.pipeline.store import PipelineStore

    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    log.info("Loading PipelineStore and model…")
    store = PipelineStore(str(db_path))
    trainer = Trainer.load_model(model_path, store)

    # Get instrument universe
    entities = store.query_all_entities()
    entity_ids = [e["entity_id"] for e in entities if e["entity_type"] == "instrument"]
    log.info("Instrument universe: %d instruments", len(entity_ids))

    # Load returns
    import json
    import sqlite3
    from collections import defaultdict

    conn = sqlite3.connect(str(db_path))
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

    data_by_day: dict[str, dict[str, float]] = defaultdict(dict)
    for eid, ts, val_json in rows:
        val = json.loads(val_json) if val_json else {}
        lr = val.get("log_return")
        if lr is None:
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        data_by_day[day][eid] = float(lr)

    dates = sorted(data_by_day.keys())
    idx = {t: i for i, t in enumerate(entity_ids)}
    T, N = len(dates), len(entity_ids)
    returns = np.zeros((T, N), dtype=np.float64)
    for t, day in enumerate(dates):
        for eid, lr in data_by_day[day].items():
            if eid in idx:
                returns[t, idx[eid]] = lr

    if T < MIN_TRAIN + TEST_SIZE:
        print(f"Not enough data ({T} rows). Need ≥ {MIN_TRAIN + TEST_SIZE}.")
        sys.exit(1)

    log.info("Pre-fetching graph structure…")
    full_id_map, _, full_links = trainer._graph_builder.prepare_static()
    all_obs = trainer._graph_builder.prefetch_observations()
    all_obs_ts = [o["observed_at"] for o in all_obs]

    log.info("Computing baseline (full data)…")
    baseline_weights = _build_weights_with_masked_source(
        trainer, dates, all_obs, all_obs_ts, full_id_map, full_links, entity_ids,
        masked_source=None,
    )
    baseline_ic = _compute_ic_for_weights(baseline_weights, dates, returns)

    results: dict[str, dict] = {}
    for source in sources:
        log.info("Ablating source: %s…", source)
        masked_weights = _build_weights_with_masked_source(
            trainer, dates, all_obs, all_obs_ts, full_id_map, full_links, entity_ids,
            masked_source=source,
        )
        masked_ic = _compute_ic_for_weights(masked_weights, dates, returns)

        delta_ic = baseline_ic["mean_ic"] - masked_ic["mean_ic"]
        delta_icir = baseline_ic["icir"] - masked_ic["icir"]

        results[source] = {
            "baseline": baseline_ic,
            "masked": masked_ic,
            "delta_mean_ic": round(delta_ic, 6),
            "delta_icir": round(delta_icir, 4),
            "interpretation": _interpret_delta(source, delta_ic, delta_icir),
        }

    return results


def _interpret_delta(source: str, delta_ic: float, delta_icir: float) -> str:
    """Interpret a source's IC contribution delta."""
    if delta_ic > 0.01:
        return f"CONTRIBUTING: removing {source} drops IC by {delta_ic:+.4f} — source adds real signal"
    elif delta_ic < -0.01:
        return f"HURTING: removing {source} IMPROVES IC by {abs(delta_ic):.4f} — source adds noise"
    else:
        return f"NEUTRAL: removing {source} has minimal IC impact ({delta_ic:+.4f})"


def print_ablation_report(results: dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print("SOURCE ABLATION TEST — Inference-time data attribution")
    print("=" * 70)
    print("  Baseline = full observation set (all sources)")
    print("  ΔIC = baseline_IC - masked_IC")
    print("  Positive ΔIC = removing source HURTS signal → source is contributing")
    print("  Negative ΔIC = removing source HELPS signal → source is adding noise")
    print()
    print(f"  {'Source':<20} {'Base IC':>9} {'Masked IC':>10} {'ΔIC':>8} {'ΔICIR':>8}  Interpretation")
    print("  " + "-" * 85)

    # Sort by |delta_ic| descending
    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]["delta_mean_ic"]), reverse=True)
    for source, r in sorted_results:
        base = r["baseline"]["mean_ic"]
        masked = r["masked"]["mean_ic"]
        delta_ic = r["delta_mean_ic"]
        delta_icir = r["delta_icir"]
        interp_short = r["interpretation"].split(":")[0]
        print(f"  {source:<20} {base:>9.4f} {masked:>10.4f} {delta_ic:>+8.4f} {delta_icir:>+8.3f}  {interp_short}")

    print()
    print("  Full interpretations:")
    for source, r in sorted_results:
        print(f"  → {r['interpretation']}")

    print()
    print("  Next action: sources with ΔIC > 0.01 are already contributing.")
    print("  Sources with ΔIC ≈ 0 need more data (backfill / run pipeline longer).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Source ablation test for GNN signal attribution")
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated list of source_tool names to ablate",
    )
    parser.add_argument("--save", action="store_true", help="Save results to experiment manifest")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    log.info("Running ablation for sources: %s", sources)

    results = run_ablation(Path(args.db_path), Path(args.model_path), sources)
    print_ablation_report(results)

    if args.save:
        import json
        from agent.quant.experiment_tracker import ExperimentTracker, _json_safe
        tracker = ExperimentTracker(args.db_path, args.model_path)
        manifest = tracker.build_manifest(
            ic_results={},
            extra={"type": "source_ablation", "ablation_results": results},
        )
        path = tracker.save(manifest)
        print(f"\n  Ablation results saved → {path}")


if __name__ == "__main__":
    main()
