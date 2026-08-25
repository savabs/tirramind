#!/usr/bin/env python3
"""Post-train eval diagnostics — artifact audit, embedding health, head probe, recommendations.

Designed for Kaggle post-train cell and tirramind-phase50-eval notebook.
Avoids the brittle inline scipy IC cell; phase40_gnn_backtest.py handles walk-forward IC.

Usage:
    python scripts/gnn_eval_diagnostics.py \\
        --model-path .tirra_pipeline/gnn_model_phase50.pt \\
        --weights-from-epoch .tirra_pipeline/checkpoints/phase50/epoch_002.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_run_config(checkpoint_dir: Path) -> dict:
    p = checkpoint_dir / "run_config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _head_weight_norm(state: dict, prefix: str) -> float | None:
    keys = [k for k in state if k.startswith(prefix) and k.endswith(".weight")]
    if not keys:
        return None
    total = 0.0
    for k in keys:
        # tensors stored as lists in some ckpts — handle via torch if needed
        import torch

        v = state[k]
        if not isinstance(v, torch.Tensor):
            continue
        total += float(v.norm().item())
    return total if total > 0 else None


def audit_artifacts(
    model_path: Path,
    weights_path: Path | None,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size if model_path.exists() else 0,
    }
    if model_path.stat().st_size < 1_000_000:
        out["model_warning"] = "Model file < 1 MB — likely stub, not trained weights"

    if weights_path:
        out["weights_path"] = str(weights_path)
        out["weights_bytes"] = weights_path.stat().st_size if weights_path.exists() else 0
        if weights_path.exists() and weights_path.stat().st_size < 1_000_000:
            out["weights_warning"] = "Epoch checkpoint < 1 MB — likely stub"

    if weights_path and weights_path.exists():
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        out["checkpoint_epoch"] = ckpt.get("epoch")
        out["heads"] = {
            "return_pred_head": _head_weight_norm(state, "return_pred_head"),
            "return_raw_head": _head_weight_norm(state, "return_raw_head"),
            "return_concat_head": _head_weight_norm(state, "return_concat_head"),
        }
        out["n_state_keys"] = len(state) if isinstance(state, dict) else 0

    if checkpoint_dir:
        rc = _load_run_config(checkpoint_dir)
        if rc:
            out["run_config"] = rc

    return out


def embedding_health(trainer: Any, dates: list, prefetched_obs: list, id_map: Any, links: list) -> dict[str, Any]:
    """Single-snapshot instrument embedding collapse metrics."""
    import bisect
    import torch
    from datetime import datetime, timezone

    from scripts.phase40_gnn_backtest import GNN_LOOKBACK_DAYS, _align_graph_features_to_model

    fold_date = dates[-1]
    obs_ts = [o["observed_at"] for o in prefetched_obs]
    fold_ts = (
        datetime.fromisoformat(fold_date).replace(tzinfo=timezone.utc).timestamp()
    )
    since_ts = fold_ts - GNN_LOOKBACK_DAYS * 86400
    end_idx = bisect.bisect_left(obs_ts, fold_ts)
    start_idx = bisect.bisect_left(obs_ts, since_ts)
    obs_window = prefetched_obs[start_idx:end_idx]
    if not obs_window:
        return {"error": "no observations in lookback window"}

    data, local_map, _ = trainer._graph_builder.build_from_cached(
        id_map, links, observations=obs_window
    )

    _align_graph_features_to_model(data, trainer.model)
    trainer.model.eval()
    with torch.no_grad():
        embeddings = trainer.model(data, local_map)
    inst = embeddings.get("instrument")
    if inst is None or inst.size(0) < 2:
        return {"error": "no instrument embeddings"}

    norms = inst.norm(dim=1).cpu().numpy()
    # Pairwise cosine diversity: fraction of pairs with cos > 0.95
    em = inst / (inst.norm(dim=1, keepdim=True) + 1e-8)
    n = em.size(0)
    import torch

    if n > 200:
        idx = torch.randperm(n)[:200]
        em = em[idx]
        n = em.size(0)
    cos = em @ em.T
    triu = cos.triu(diagonal=1).cpu().numpy().ravel()
    collapse_frac = float((triu > 0.95).mean()) if triu.size else 0.0

    return {
        "n_instruments": int(inst.size(0)),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_cv": float(norms.std() / (norms.mean() + 1e-8)),
        "collapse_frac_cos_gt_0.95": collapse_frac,
        "fold_cutoff": fold_ts,
    }


def print_audit(audit: dict) -> None:
    print("\n" + "=" * 60)
    print("ARTIFACT AUDIT")
    print("=" * 60)
    print(f"  Model:     {audit.get('model_path')}  ({audit.get('model_bytes', 0) / 1e6:.1f} MB)")
    if audit.get("model_warning"):
        print(f"  ⚠ {audit['model_warning']}")
    if "weights_path" in audit:
        print(
            f"  Weights:   {audit['weights_path']}  "
            f"({audit.get('weights_bytes', 0) / 1e6:.1f} MB)  "
            f"epoch={audit.get('checkpoint_epoch')}"
        )
    if audit.get("weights_warning"):
        print(f"  ⚠ {audit['weights_warning']}")
    heads = audit.get("heads") or {}
    print("  Return heads (weight L2 norm — 0/None = absent or zero-init):")
    for name, norm in heads.items():
        print(f"    {name:<22} {norm if norm is not None else '—'}")
    rc = audit.get("run_config") or {}
    cfg = rc.get("config") or {}
    if cfg:
        print("  Training run_config:")
        for k in (
            "epochs",
            "n1_doctrine",
            "embedding_only_return",
            "zero_price_feats",
            "use_concat_head",
            "gdelt_frac",
            "defi_frac",
            "max_windows",
            "return_weight",
        ):
            if k in cfg:
                print(f"    {k:<24} {cfg[k]}")


def print_embedding_health(h: dict) -> None:
    print("\n" + "=" * 60)
    print("EMBEDDING HEALTH (latest snapshot)")
    print("=" * 60)
    if "error" in h:
        print(f"  ✗ {h['error']}")
        return
    print(f"  Instruments:     {h['n_instruments']}")
    print(f"  Norm mean/std:     {h['norm_mean']:.4f} / {h['norm_std']:.4f}")
    print(f"  Norm CV:           {h['norm_cv']:.4f}  (low → collapse risk)")
    print(f"  Collapse frac:     {h['collapse_frac_cos_gt_0.95']:.2%}  (pairs cos>0.95)")
    if h["collapse_frac_cos_gt_0.95"] > 0.30:
        print("  ⚠ HIGH COLLAPSE — embeddings cluster; return_pred_head unlikely to discriminate")
    elif h["norm_cv"] < 0.05:
        print("  ⚠ LOW DIVERSITY — norm CV < 0.05")


def print_recommendations(
    audit: dict,
    emb: dict,
    notebook_cfg: dict | None,
    smoke: bool,
) -> None:
    print("\n" + "=" * 60)
    print("NEXT-STEP RECOMMENDATIONS")
    print("=" * 60)
    cfg = (audit.get("run_config") or {}).get("config") or notebook_cfg or {}
    embedding_only = cfg.get("embedding_only_return") or cfg.get("n1_doctrine")
    concat = cfg.get("use_concat_head", False)

    if audit.get("weights_warning") or audit.get("model_warning"):
        print("  1. FIX ARTIFACTS — checkpoint/model too small; re-download from Kaggle Output")
    if emb.get("collapse_frac_cos_gt_0.95", 0) > 0.25:
        print("  2. EMBEDDING COLLAPSE — try: lower obs_type_weight, enable concat head ablation, or more raw sensor data")
    if embedding_only and not concat:
        print("  3. N1 embedding-only path — if PurgedRanker IC < 0.03: run V56b ablation with --use-concat-head")
    heads = audit.get("heads") or {}
    if heads.get("return_raw_head") and embedding_only:
        print("  4. return_raw_head exists but embedding_only_return=True — raw head not used for training loss")
    if smoke:
        print("  5. SMOKE MODE — eval cell OK; proceed to full 90ep / 200-window train if no errors above")
    else:
        print("  5. Compare PurgedRanker IC to V52 (+0.047) before promoting")


def main() -> None:
    ap = argparse.ArgumentParser(description="GNN post-train eval diagnostics")
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--weights-from-epoch", type=Path, default=None)
    ap.add_argument("--db-path", type=Path, default=Path(".tirra_pipeline/pipeline.db"))
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--notebook-config", type=str, default=None, help="JSON blob from _NOTEBOOK_CONFIG")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None and args.weights_from_epoch:
        ckpt_dir = args.weights_from_epoch.parent

    nb_cfg = json.loads(args.notebook_config) if args.notebook_config else None

    audit = audit_artifacts(args.model_path, args.weights_from_epoch, ckpt_dir)
    print_audit(audit)

    from agent.models.gnn.trainer import Trainer
    from agent.pipeline.store import PipelineStore
    from scripts.phase40_gnn_backtest import _load_instrument_returns_fast

    store = PipelineStore(str(args.db_path))
    if args.weights_from_epoch:
        trainer = Trainer.load_model_with_epoch_weights(
            args.model_path, args.weights_from_epoch, store
        )
    else:
        trainer = Trainer.load_model(args.model_path, store)

    entities = store.query_all_entities()
    entity_ids = [e["entity_id"] for e in entities if e["entity_type"] == "instrument"]
    dates, _ = _load_instrument_returns_fast(str(args.db_path), entity_ids)
    id_map, _, links = trainer._graph_builder.prepare_static()
    prefetched = trainer._graph_builder.prefetch_observations()

    emb = embedding_health(trainer, dates, prefetched, id_map, links)
    print_embedding_health(emb)
    print_recommendations(audit, emb, nb_cfg, args.smoke)

    summary = {"audit": audit, "embedding_health": emb, "smoke": args.smoke}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\n  Diagnostics JSON → {args.out}")


if __name__ == "__main__":
    main()
