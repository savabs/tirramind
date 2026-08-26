"""TirraMind — run the full DAG chain in dependency order.

The 11 registered DAGs each declare a cron schedule that together form a nightly
chain (18:00 collect → 19:45 inference). Those schedules only fire if a
long-running `PipelineScheduler.start()` process is alive, and nothing in
production ever started one: both real entry points hardcode
`trigger("daily_collection")`. Result — 8 of 11 DAGs had never executed even
once, so `signals`, `beliefs`, `entity_alerts`, `convergence_clusters`,
`rl_transitions`, `portfolio_weights` and `paper_trade_pnl` were all empty while
collection happily filled `entity_observations` with 365k rows.

This runs the chain in **dependency order** rather than by wall-clock. Cron
cannot express "after upstream actually succeeded", which matters because the
chain has genuine cold-start dependencies:

    rl_training     needs alerts + beliefs that don't exist on a fresh DB
    inference       needs a SAC checkpoint that only rl_training produces
    rl_transitions  only materialises on the SECOND consecutive inference run
                    (T+1 reward close-out) — 0 rows after run 1 is correct

Usage::

    .venv/bin/python scripts/run_chain.py                  # full chain
    .venv/bin/python scripts/run_chain.py --skip-collection  # downstream only
    .venv/bin/python scripts/run_chain.py --only world_model_update,inference
    .venv/bin/python scripts/run_chain.py --dry-run

Exit code is non-zero if any DAG failed, so a systemd unit or cron job surfaces
the failure instead of reporting success like the inference DAG used to.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Dependency order. Not wall-clock order — see module docstring.
CHAIN: list[str] = [
    "daily_collection",
    "convergence_detection",
    "gnn_inference",
    "entity_scoring",
    "feature_generation",
    "adversarial_scan",
    "world_model_update",
    "rl_training",
    "inference",
]

# DAGs that load trained weights. Schema drift makes these produce garbage
# rather than fail cleanly, so validate before spending time on them.
_MODEL_DEPENDENT = {"gnn_inference", "entity_scoring", "inference"}

# Tables worth reporting deltas for — the ones that were empty.
_WATCHED_TABLES = [
    "signals",
    "features",
    "beliefs",
    "entity_alerts",
    "convergence_clusters",
    "rl_transitions",
    "portfolio_weights",
    "paper_trade_pnl",
    "entity_observations",
    "entity_links",
]


def _counts(db_path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error:
        return out
    for t in _WATCHED_TABLES:
        try:
            out[t] = list(conn.execute(f"SELECT COUNT(*) FROM {t}"))[0][0]
        except sqlite3.Error:
            pass
    conn.close()
    return out


def _print_deltas(before: dict[str, int], after: dict[str, int], indent: str = "      ") -> None:
    deltas = {t: after[t] - before[t] for t in after if t in before and after[t] != before[t]}
    if deltas:
        for t, d in sorted(deltas.items(), key=lambda kv: -abs(kv[1])):
            print(f"{indent}+{d} {t}" if d > 0 else f"{indent}{d} {t}")
    else:
        print(f"{indent}(no rows written)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-collection", action="store_true", help="skip daily_collection (slow); run downstream only")
    ap.add_argument("--only", default=None, help="comma-separated subset of DAGs to run, in chain order")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without running anything")
    ap.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="keep going after a DAG fails (default: keep going, "
        "since a failed upstream may still leave useful work)",
    )
    args = ap.parse_args()

    plan = list(CHAIN)
    if args.skip_collection:
        plan = [d for d in plan if d != "daily_collection"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - set(CHAIN)
        if unknown:
            print(f"[chain] unknown DAG(s): {sorted(unknown)}", file=sys.stderr)
            print(f"[chain] known: {CHAIN}", file=sys.stderr)
            return 2
        plan = [d for d in plan if d in wanted]

    print(f"[chain] plan ({len(plan)} DAGs, dependency order):")
    for i, d in enumerate(plan, 1):
        tag = "  [model-dependent]" if d in _MODEL_DEPENDENT else ""
        print(f"[chain]   {i}. {d}{tag}")
    if args.dry_run:
        print("[chain] dry run — nothing executed.")
        return 0

    from agent.cli import build_tool_registry
    from agent.config.settings import AgentConfig
    from agent.models.gnn.graph_builder import validate_schema_against_store
    from agent.pipeline.executor import DAGExecutor
    from agent.pipeline.registry import DAGRegistry
    from agent.pipeline.scheduler import PipelineScheduler
    from agent.pipeline.store import PipelineStore

    cfg = AgentConfig.from_env()
    store = PipelineStore(db_path=args.db_path)
    tools = build_tool_registry(cfg)
    executor = DAGExecutor(tool_registry=tools, store=store, max_workers=args.workers)
    registry = DAGRegistry()
    registry.load_defaults(tools)
    scheduler = PipelineScheduler(executor=executor, registry=registry)

    results: list[tuple[str, str, float]] = []
    schema_ok = True

    for name in plan:
        # Before burning time on a model DAG, check the store doesn't contain
        # entity/observation types the model cannot encode. This is the guard
        # that was missing while instrument features drifted 23 -> 49.
        if name in _MODEL_DEPENDENT and schema_ok:
            try:
                validate_schema_against_store(store)
            except Exception as exc:
                schema_ok = False
                print(f"\n[chain] !! schema drift detected before {name}:\n{exc}\n")

        print(f"\n[chain] ── {name} ──")
        before = _counts(args.db_path)
        t0 = time.time()
        try:
            run = scheduler.trigger(name)
            status = run.status
            n_ok = sum(1 for r in run.node_results.values() if getattr(r, "status", "") in ("success", "completed"))
            n_total = len(run.node_results)
            print(f"[chain]    status={status}  nodes_ok={n_ok}/{n_total}")
            for nid, nr in run.node_results.items():
                st = getattr(nr, "status", "?")
                if st not in ("success", "completed"):
                    err = str(getattr(nr, "error", "") or "")[:200]
                    print(f"[chain]      [{st}] {nid}: {err}")
        except Exception as exc:
            status = "failed"
            print(f"[chain]    status=failed (trigger raised): {str(exc)[:300]}")

        elapsed = time.time() - t0
        _print_deltas(before, _counts(args.db_path))
        print(f"[chain]    {elapsed:.1f}s")
        results.append((name, status, elapsed))

        if status == "failed" and not args.continue_on_failure:
            # Default is still to continue: a failed DAG mid-chain does not make
            # the later independent ones pointless, and stopping would hide them.
            pass

    print(f"\n[chain] {'=' * 60}\n[chain] summary")
    failed = [n for n, s, _ in results if s == "failed"]
    for name, status, elapsed in results:
        mark = "OK  " if status not in ("failed",) else "FAIL"
        print(f"[chain]   {mark} {name:24s} {status:12s} {elapsed:6.1f}s")

    print(f"\n[chain] final table state ({args.db_path}):")
    for t, n in sorted(_counts(args.db_path).items()):
        print(f"[chain]   {t:24s} {n}")

    # ── Honest exit status ────────────────────────────────────────────────
    # This previously printed "all DAGs completed" and exited 0 after detecting
    # schema drift, running the model DAGs anyway, and writing zero rows. That
    # is the same green-but-broken pattern the chain exists to expose, and a
    # systemd timer would have reported success indefinitely.
    #
    # Drift is a FAILURE of the run, not a footnote: every model-dependent DAG
    # downstream of it is producing garbage or nothing.
    problems: list[str] = []

    if not schema_ok:
        problems.append(
            "schema drift — model-dependent DAGs cannot produce valid output " "until a retrain on the current schema"
        )
        print("\n[chain] FAIL: schema drift was detected before the model DAGs ran.")
        print("[chain]       Their output (if any) is not trustworthy.")

    if failed:
        problems.append(f"{len(failed)} DAG(s) failed: {failed}")
        print(f"\n[chain] FAIL: {len(failed)} DAG(s) failed: {failed}")

    if problems:
        print(f"\n[chain] run NOT clean — {len(problems)} problem(s). Exit 1.")
        return 1

    print("\n[chain] all DAGs completed, no schema drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
