#!/usr/bin/env python3
"""Compare Experiments — diff two GNN backtest runs.

Lists all saved experiment manifests and diffs any two,
showing what changed in data, IC, ICIR, and stratified attribution.

Usage:
    python scripts/compare_experiments.py              # list all experiments
    python scripts/compare_experiments.py --diff a b   # diff two run_ids
    python scripts/compare_experiments.py --latest 2   # diff last 2 runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.quant.experiment_tracker import ExperimentTracker


def _short_id(run_id: str) -> str:
    """Display last 15 chars of run_id."""
    return run_id[-15:] if len(run_id) > 15 else run_id


def cmd_list() -> None:
    exps = ExperimentTracker.list_experiments()
    if not exps:
        print(
            "No experiments found. Run 'python scripts/phase40_gnn_backtest.py' first."
        )
        return
    print(
        f"\n{'#':<4} {'Run ID':<26} {'Model Epoch':>12} {'Total Obs':>12} {'Best Mean IC':>14}"
    )
    print("-" * 72)
    for i, path in enumerate(exps):
        try:
            m = ExperimentTracker.load(path)
            epoch = m.get("model_snapshot", {}).get("epoch", "?")
            total_obs = m.get("data_snapshot", {}).get("total_observations", "?")
            # Find best IC across strategies
            ic_results = m.get("ic_results", {})
            best_ic = max(
                (r.get("mean_ic", 0.0) for r in ic_results.values()), default=0.0
            )
            best_strategy = max(
                ic_results.items(),
                key=lambda x: x[1].get("mean_ic", 0.0),
                default=("none", {}),
            )[0]
            print(
                f"{i:<4} {m.get('run_id','?'):<26} {str(epoch):>12} "
                f"{str(total_obs):>12} {best_ic:>+14.4f} ({best_strategy})"
            )
        except Exception as exc:
            print(f"{i:<4} {path.name:<26} [error: {exc}]")


def cmd_diff(run_id_a: str, run_id_b: str) -> None:
    exps = ExperimentTracker.list_experiments()
    exp_map = {m.stem: m for m in exps}

    # Support partial match
    def find(run_id: str) -> Path | None:
        if run_id in exp_map:
            return exp_map[run_id]
        matches = [k for k in exp_map if run_id in k]
        return exp_map[matches[0]] if len(matches) == 1 else None

    path_a = find(run_id_a)
    path_b = find(run_id_b)
    if not path_a or not path_b:
        print(f"Could not find experiments for '{run_id_a}' and/or '{run_id_b}'")
        print("Run with no args to list available experiments.")
        return

    a = ExperimentTracker.load(path_a)
    b = ExperimentTracker.load(path_b)
    diff = ExperimentTracker.diff(a, b)

    print("\n" + "=" * 70)
    print("EXPERIMENT DIFF")
    print("=" * 70)
    print(f"  A: {diff['run_a']}  (timestamp: {diff['timestamp_a']})")
    print(f"  B: {diff['run_b']}  (timestamp: {diff['timestamp_b']})")

    # Data changes
    dd = diff.get("data_delta", {})
    print(f"\n  Data Changes:")
    print(f"    Total observations: {dd.get('total_observations', 0):+,}")
    print(f"    Total entities:     {dd.get('total_entities', 0):+,}")
    src_delta = dd.get("obs_by_source_delta", {})
    if src_delta:
        print(f"    Per-source delta:")
        for src, delta in sorted(
            src_delta.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            print(f"      {src:<24} {delta:>+10,}")

    # IC changes
    ic_delta = diff.get("ic_delta", {})
    print(f"\n  IC Changes (A → B):")
    print(
        f"    {'Strategy':<22} {'Mean IC (A)':>12} {'Mean IC (B)':>12} {'ΔMean IC':>10} {'ICIR (A)':>10} {'ICIR (B)':>10} {'ΔICIR':>8}"
    )
    print("    " + "-" * 88)
    for strat, d in ic_delta.items():
        print(
            f"    {strat:<22} {str(d.get('mean_ic_a','?')):>12} {str(d.get('mean_ic_b','?')):>12}"
            f" {d.get('mean_ic_delta',0):>+10.4f}"
            f" {str(d.get('icir_a','?')):>10} {str(d.get('icir_b','?')):>10}"
            f" {d.get('icir_delta',0):>+8.3f}"
        )

    # Attention changes
    attn_delta = diff.get("attention_delta", {})
    if attn_delta:
        print(f"\n  HGT Attention Changes (A → B):")
        for edge, delta in sorted(
            attn_delta.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            print(f"    {edge:<40} {delta:>+.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff", nargs=2, metavar=("A", "B"), help="Diff two run IDs")
    parser.add_argument(
        "--latest", type=int, metavar="N", help="Diff the last N runs (default 2)"
    )
    args = parser.parse_args()

    if args.diff:
        cmd_diff(args.diff[0], args.diff[1])
    elif args.latest:
        exps = ExperimentTracker.list_experiments()
        if len(exps) < args.latest:
            print(f"Only {len(exps)} experiments available.")
            cmd_list()
            return
        n = args.latest
        # Diff consecutive pairs: exp[n-1] vs exp[0] (oldest vs newest in the last N)
        a = ExperimentTracker.load(exps[n - 1])
        b = ExperimentTracker.load(exps[0])
        diff = ExperimentTracker.diff(a, b)
        ExperimentTracker  # just for the import
        # reuse cmd_diff logic
        cmd_diff(a["run_id"], b["run_id"])
    else:
        cmd_list()


if __name__ == "__main__":
    main()
