"""TirraMind — Experiment Tracker

Writes a self-contained JSON manifest every time a GNN backtest runs.
Enables scientific attribution: what data was in the DB, which sources
contributed to IC, and how attention weights changed over time.

Usage:
    tracker = ExperimentTracker(db_path, model_path)
    manifest = tracker.build_manifest(ic_results, stratified_ic, attention)
    tracker.save(manifest)

    # Compare two experiments:
    diff = ExperimentTracker.diff(exp_a, exp_b)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

# datetime.UTC is Python 3.11+; Kaggle / many dev boxes are still on 3.10.
UTC = timezone.utc
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_EXPERIMENTS_DIR = Path(".tirra_pipeline/experiments")


class ExperimentTracker:
    """Build and persist experiment manifests for every GNN backtest run."""

    def __init__(
        self,
        db_path: str | Path,
        model_path: str | Path | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._model_path = Path(model_path) if model_path else None

    # ── Snapshot helpers ──────────────────────────────────────────────────────

    def snapshot_data(self) -> dict[str, Any]:
        """Return entity/observation counts from the live DB."""
        con = sqlite3.connect(self._db_path)
        try:
            entity_rows = con.execute("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type").fetchall()
            obs_rows = con.execute(
                "SELECT source_tool, COUNT(*) FROM entity_observations GROUP BY source_tool ORDER BY COUNT(*) DESC"
            ).fetchall()
            total_obs = con.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0]
            total_ent = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            link_rows = con.execute(
                "SELECT link_type, COUNT(*) FROM entity_links GROUP BY link_type ORDER BY COUNT(*) DESC"
            ).fetchall()
        finally:
            con.close()

        return {
            "total_entities": total_ent,
            "total_observations": total_obs,
            "entity_counts": dict(entity_rows),
            "obs_by_source": dict(obs_rows),
            "link_counts": dict(link_rows),
        }

    def snapshot_model(self) -> dict[str, Any]:
        """Return metadata from the saved model checkpoint."""
        if self._model_path is None or not self._model_path.exists():
            return {"available": False}
        try:
            import torch

            ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                epoch = ckpt.get("epoch", -1)
                cfg = ckpt.get("config", {})
                log_vars = {k: float(v.item()) for k, v in ckpt.get("model_state_dict", {}).items() if "log_var" in k}
                return {
                    "available": True,
                    "path": str(self._model_path),
                    "epoch": epoch,
                    "config": cfg if isinstance(cfg, dict) else str(cfg),
                    "log_vars": log_vars,
                }
            return {"available": True, "path": str(self._model_path)}
        except Exception as exc:
            return {"available": True, "path": str(self._model_path), "error": str(exc)}

    # ── Manifest builder ──────────────────────────────────────────────────────

    def build_manifest(
        self,
        ic_results: dict[str, dict],
        stratified_ic: dict[str, dict] | None = None,
        attention_weights: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a full experiment manifest.

        Parameters
        ----------
        ic_results:
            Output of _compute_ic_diagnostic — strategy_name → {mean_ic, icir, ...}
        stratified_ic:
            Output of compute_stratified_ic — strategy_name → {source → {mean_ic, ...}}
        attention_weights:
            edge_type → mean HGT attention weight
        extra:
            Any additional metadata to include (fold config, etc.)
        """
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        data_snap = self.snapshot_data()
        model_snap = self.snapshot_model()

        manifest: dict[str, Any] = {
            "timestamp": ts,
            "run_id": f"exp_{ts}",
            "data_snapshot": data_snap,
            "model_snapshot": model_snap,
            "ic_results": ic_results,
        }
        if stratified_ic is not None:
            manifest["stratified_ic"] = stratified_ic
        if attention_weights is not None:
            manifest["attention_weights"] = attention_weights
        if extra is not None:
            manifest["extra"] = extra
        return manifest

    def save(self, manifest: dict[str, Any]) -> Path:
        """Write manifest JSON to the experiments directory. Returns file path."""
        _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        path = _EXPERIMENTS_DIR / f"{manifest['run_id']}.json"
        path.write_text(json.dumps(manifest, indent=2, default=_json_safe))
        log.info("Experiment manifest saved → %s", path)
        return path

    # ── Comparison ────────────────────────────────────────────────────────────

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text())

    @staticmethod
    def list_experiments() -> list[Path]:
        """Return experiment manifests sorted newest first."""
        if not _EXPERIMENTS_DIR.exists():
            return []
        return sorted(_EXPERIMENTS_DIR.glob("exp_*.json"), reverse=True)

    @staticmethod
    def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        """Compare two experiment manifests. Returns delta dict (b - a)."""
        result: dict[str, Any] = {
            "run_a": a.get("run_id"),
            "run_b": b.get("run_id"),
            "timestamp_a": a.get("timestamp"),
            "timestamp_b": b.get("timestamp"),
        }

        # Data changes
        da, db = a.get("data_snapshot", {}), b.get("data_snapshot", {})
        result["data_delta"] = {
            "total_observations": db.get("total_observations", 0) - da.get("total_observations", 0),
            "total_entities": db.get("total_entities", 0) - da.get("total_entities", 0),
        }
        # Per-source obs delta
        sa, sb = da.get("obs_by_source", {}), db.get("obs_by_source", {})
        all_sources = set(sa) | set(sb)
        src_delta = {s: sb.get(s, 0) - sa.get(s, 0) for s in sorted(all_sources) if sb.get(s, 0) - sa.get(s, 0) != 0}
        result["data_delta"]["obs_by_source_delta"] = src_delta

        # IC changes
        ia, ib = a.get("ic_results", {}), b.get("ic_results", {})
        ic_delta: dict[str, dict] = {}
        for strategy in set(ia) | set(ib):
            ra = ia.get(strategy, {})
            rb = ib.get(strategy, {})
            ic_delta[strategy] = {
                "mean_ic_delta": rb.get("mean_ic", 0) - ra.get("mean_ic", 0),
                "icir_delta": rb.get("icir", 0) - ra.get("icir", 0),
                "t_stat_delta": rb.get("t_stat", 0) - ra.get("t_stat", 0),
                "mean_ic_a": ra.get("mean_ic"),
                "mean_ic_b": rb.get("mean_ic"),
                "icir_a": ra.get("icir"),
                "icir_b": rb.get("icir"),
            }
        result["ic_delta"] = ic_delta

        # Attention changes
        aw_a = a.get("attention_weights", {})
        aw_b = b.get("attention_weights", {})
        all_edges = set(aw_a) | set(aw_b)
        attn_delta = {
            e: aw_b.get(e, 0.0) - aw_a.get(e, 0.0)
            for e in sorted(all_edges)
            if abs(aw_b.get(e, 0.0) - aw_a.get(e, 0.0)) > 1e-5
        }
        result["attention_delta"] = attn_delta

        return result


# ── Stratified IC computation ─────────────────────────────────────────────────


def compute_stratified_ic(
    strategies: list,
    dates: list[str],
    returns: np.ndarray,
    instrument_names: list[str],
    db_path: str | Path,
    min_train: int = 252,
    test_size: int = 21,
    step_size: int = 21,
    min_group_size: int = 5,
) -> dict[str, dict]:
    """Compute IC stratified by which data source covers each instrument.

    For each major source_tool, partitions the instrument universe into:
    - "has_<source>": instruments that have at least one observation from that source
    - "no_<source>": the rest

    Computes IC separately for each partition per fold, then aggregates.
    This reveals whether source X is actually predictive AT INFERENCE TIME.

    Returns
    -------
    {strategy_name: {source_key: {mean_ic, icir, t_stat, n_folds, n_instruments}}}
    """
    from scipy.stats import spearmanr

    # Find which instruments have obs from each major source
    con = sqlite3.connect(db_path)
    try:
        # Sources that could affect instrument inference (have obs linked to instruments,
        # directly or via entity_links to cftc_contract/topic that connects to instruments)
        direct_sources = con.execute(
            """SELECT DISTINCT source_tool FROM entity_observations o
               JOIN entities e ON o.entity_id = e.entity_id
               WHERE e.entity_type = 'instrument'"""
        ).fetchall()
        direct_sources = {r[0] for r in direct_sources}

        # For CFTC: instruments linked via cftc_tracks
        cftc_instruments = set(
            r[0]
            for r in con.execute("""SELECT DISTINCT el.entity_id_b
                   FROM entity_links el
                   JOIN entities ea ON el.entity_id_a = ea.entity_id
                   WHERE ea.entity_type = 'cftc_contract'
                   AND el.link_type = 'cftc_tracks'""").fetchall()
        )
        # Reverse: instrument is entity_id_a in some links?
        cftc_instruments |= set(
            r[0]
            for r in con.execute("""SELECT DISTINCT el.entity_id_a
                   FROM entity_links el
                   JOIN entities eb ON el.entity_id_b = eb.entity_id
                   WHERE eb.entity_type = 'cftc_contract'
                   AND el.link_type = 'cftc_tracks'""").fetchall()
        )

        # For polymarket: instruments linked via topic_relates_to_inst
        poly_instruments = set(
            r[0]
            for r in con.execute("""SELECT DISTINCT el.entity_id_b
                   FROM entity_links el
                   JOIN entities ea ON el.entity_id_a = ea.entity_id
                   WHERE ea.entity_type = 'topic'
                   AND el.link_type = 'topic_relates_to_inst'""").fetchall()
        )

        # For country links: instruments with any located_in / produced_in link
        geo_instruments = set(
            r[0]
            for r in con.execute(
                """SELECT DISTINCT entity_id_a FROM entity_links
                   WHERE link_type IN ('located_in','produced_in','exchange_country')"""
            ).fetchall()
        )
    finally:
        con.close()

    # Build coverage groups
    inst_set = set(instrument_names)
    idx = {t: i for i, t in enumerate(instrument_names)}

    coverage_groups: dict[str, list[int]] = {
        "has_cftc": [idx[i] for i in cftc_instruments if i in inst_set],
        "no_cftc": [idx[i] for i in inst_set if i not in cftc_instruments],
        "has_polymarket": [idx[i] for i in poly_instruments if i in inst_set],
        "no_polymarket": [idx[i] for i in inst_set if i not in poly_instruments],
        "has_geo_link": [idx[i] for i in geo_instruments if i in inst_set],
        "no_geo_link": [idx[i] for i in inst_set if i not in geo_instruments],
    }

    gnn_strats = [s for s in strategies if hasattr(s, "compute_fold_ics")]
    result: dict[str, dict] = {}

    for strat in gnn_strats:
        strat_result: dict[str, dict] = {}
        for group_name, group_idx in coverage_groups.items():
            if len(group_idx) < min_group_size:
                strat_result[group_name] = {
                    "n_instruments": len(group_idx),
                    "skipped": "too_few_instruments",
                }
                continue

            # Recompute IC for this instrument subset
            fold_ics: list[float] = []
            split = min_train
            while split + test_size <= len(dates):
                fold_date = dates[split]
                cache = getattr(strat, "_cache", {})
                if fold_date not in cache:
                    split += step_size
                    continue

                w_full = cache[fold_date]  # shape (N_all,)
                w_sub = w_full[group_idx]
                fwd_ret = returns[split : split + test_size].mean(axis=0)[group_idx]
                valid = np.isfinite(w_sub) & np.isfinite(fwd_ret)
                if valid.sum() >= min_group_size:
                    ic, _ = spearmanr(w_sub[valid], fwd_ret[valid])
                    if np.isfinite(ic):
                        fold_ics.append(float(ic))
                split += step_size

            ics = np.array(fold_ics)
            n = len(ics)
            mean_ic = float(ics.mean()) if n > 0 else 0.0
            std_ic = float(ics.std(ddof=1)) if n > 1 else 0.0
            icir = mean_ic / (std_ic + 1e-8)
            t_stat = (mean_ic / (std_ic / np.sqrt(n))) if (n > 0 and std_ic > 1e-10) else 0.0

            strat_result[group_name] = {
                "n_instruments": len(group_idx),
                "n_folds": n,
                "mean_ic": round(mean_ic, 6),
                "std_ic": round(std_ic, 6),
                "icir": round(icir, 4),
                "t_stat": round(t_stat, 3),
            }
        result[strat.name] = strat_result

    return result


def print_stratified_ic_report(stratified: dict[str, dict]) -> None:
    """Print stratified IC attribution report."""
    print("\n" + "=" * 70)
    print("STRATIFIED IC — Signal attribution by data source coverage")
    print("=" * 70)
    print("  Interpretation: if IC_with_X >> IC_without_X, source X is carrying signal")
    print()

    for strat_name, groups in stratified.items():
        print(f"  Strategy: {strat_name}")
        # Print paired comparisons
        pairs = [
            ("has_cftc", "no_cftc"),
            ("has_polymarket", "no_polymarket"),
            ("has_geo_link", "no_geo_link"),
        ]
        for has_key, no_key in pairs:
            g_has = groups.get(has_key, {})
            g_no = groups.get(no_key, {})
            if "skipped" in g_has or "skipped" in g_no:
                continue
            source_name = has_key.replace("has_", "")
            mic_has = g_has.get("mean_ic", float("nan"))
            mic_no = g_no.get("mean_ic", float("nan"))
            icir_has = g_has.get("icir", float("nan"))
            icir_no = g_no.get("icir", float("nan"))
            n_has = g_has.get("n_instruments", 0)
            n_no = g_no.get("n_instruments", 0)
            delta = mic_has - mic_no if (isinstance(mic_has, float) and isinstance(mic_no, float)) else float("nan")

            verdict = ""
            if isinstance(delta, float):
                if delta > 0.01:
                    verdict = "✓ SOURCE CONTRIBUTING"
                elif delta < -0.01:
                    verdict = "✗ SOURCE HURTING"
                else:
                    verdict = "~ NEUTRAL"

            print(
                f"    {source_name:<14} "
                f"WITH({n_has:>2} inst): IC={mic_has:>+.4f} ICIR={icir_has:>+.3f}  "
                f"WITHOUT({n_no:>2} inst): IC={mic_no:>+.4f} ICIR={icir_no:>+.3f}  "
                f"Δ={delta:>+.4f}  {verdict}"
            )
        print()


# ── Utilities ─────────────────────────────────────────────────────────────────


def _json_safe(obj: Any) -> Any:
    """JSON-safe serialiser for numpy types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
