#!/usr/bin/env python3
"""Report training-time observation mix after modal subsampling.

Usage:
    python scripts/training_mix_report.py
    python scripts/training_mix_report.py --gdelt-frac 0.05 --defi-frac 0.05
    python scripts/training_mix_report.py --n1-doctrine
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.models.gnn.obs_subsample import (
    apply_training_obs_subsample,
    training_mix_summary,
)
from agent.pipeline.store import PipelineStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Training obs mix report")
    parser.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    parser.add_argument("--gdelt-frac", type=float, default=0.05)
    parser.add_argument("--defi-frac", type=float, default=1.0)
    parser.add_argument(
        "--n1-doctrine",
        action="store_true",
        help="Use N1 POC subsample: gdelt=0.05 defi=0.05",
    )
    args = parser.parse_args()

    if args.n1_doctrine:
        args.gdelt_frac = 0.05
        args.defi_frac = 0.05

    db = Path(args.db_path)
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        return 2

    store = PipelineStore(str(db))
    try:
        obs = store.query_all_observations()
        obs.sort(key=lambda o: float(o.get("observed_at", 0.0)))
        n_raw = len(obs)
        filtered, stats = apply_training_obs_subsample(
            obs,
            gdelt_subsample_frac=args.gdelt_frac,
            defi_subsample_frac=args.defi_frac,
        )
    finally:
        store.close()

    print("=" * 72)
    print(f"Training mix report — {db}")
    print("=" * 72)
    print(f"Raw DB observations:     {n_raw:,}")
    print(f"After subsample:         {len(filtered):,}")
    print(f"gdelt_frac={args.gdelt_frac}  defi_frac={args.defi_frac}")
    if stats:
        print(f"Stats: {stats}")
    print()
    print("Top observation types (post-subsample):")
    for name, count, pct in training_mix_summary(filtered)[:12]:
        print(f"  {name:32s} {count:8,}  {pct:5.1f}%")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
