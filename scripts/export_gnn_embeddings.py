#!/usr/bin/env python3
"""Export frozen GNN instrument embeddings at walk-forward fold cutoffs (Phase B.2).

Usage:
    python scripts/export_gnn_embeddings.py --checkpoint .tirra_pipeline/gnn_model_phase50.pt \\
        --weights-from-epoch .tirra_pipeline/checkpoints/phase50/epoch_090.pt --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from agent.models.gnn.graph_builder import GraphBuilder  # noqa: E402
from agent.models.gnn.trainer import Trainer  # noqa: E402
from agent.pipeline.store import PipelineStore  # noqa: E402
from agent.quant.forward_returns import build_forward_return_lookup  # noqa: E402
from phase40_gnn_backtest import (  # noqa: E402
    MIN_TRAIN,
    STEP_SIZE,
    TEST_SIZE,
    _instrument_embedding_matrix,
    _load_instrument_returns_fast,
)

log = logging.getLogger("export_gnn_embeddings")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Export GNN embeddings per walk-forward fold")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--weights-from-epoch", type=Path, default=None)
    ap.add_argument("--db-path", type=Path, default=Path(".tirra_pipeline/pipeline.db"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".tirra_pipeline/stage1_embeddings"),
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not args.db_path.exists():
        print(f"ERROR: db not found: {args.db_path}")
        sys.exit(1)

    min_train = MIN_TRAIN
    if args.smoke:
        min_train = 126

    store = PipelineStore(str(args.db_path))
    gb = GraphBuilder(store)
    prefetched = gb.prefetch_observations()
    obs_ts = [o["observed_at"] for o in prefetched]
    id_map, _, links = gb.prepare_static()

    entities = store.query_all_entities()
    entity_ids = [e["entity_id"] for e in entities if e["entity_type"] == "instrument"]
    dates, returns = _load_instrument_returns_fast(str(args.db_path), entity_ids)
    if args.smoke and len(dates) > 400:
        dates = dates[-400:]
        returns = returns[-400:]

    if args.weights_from_epoch is not None:
        trainer = Trainer.load_model_with_epoch_weights(
            args.checkpoint, args.weights_from_epoch, store
        )
    else:
        trainer = Trainer.load_model(args.checkpoint, store)

    hidden_dim = int(getattr(trainer.model, "hidden_dim", 128))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    folds: list[dict[str, str | int]] = []
    split = min_train
    while split + TEST_SIZE <= len(dates):
        fold_date = dates[split]
        emb = _instrument_embedding_matrix(
            trainer,
            fold_date,
            entity_ids,
            id_map,
            links,
            prefetched,
            obs_ts,
        )
        out_path = args.out_dir / f"{fold_date}.npy"
        np.save(out_path, emb)
        n_valid = int(np.isfinite(emb).all(axis=1).sum())
        folds.append(
            {
                "date": fold_date,
                "file": out_path.name,
                "n_valid": n_valid,
            }
        )
        log.info("Exported %s — %d/%d instruments finite", fold_date, n_valid, len(entity_ids))
        split += STEP_SIZE

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "weights_from_epoch": str(args.weights_from_epoch) if args.weights_from_epoch else None,
        "entity_ids": entity_ids,
        "hidden_dim": hidden_dim,
        "min_train": min_train,
        "test_size": TEST_SIZE,
        "step_size": STEP_SIZE,
        "smoke": args.smoke,
        "n_folds": len(folds),
        "folds": folds,
        "fwd_lookup_n": len(build_forward_return_lookup(prefetched)),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Exported {len(folds)} folds → {args.out_dir}")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
