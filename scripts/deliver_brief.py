"""Deliver the Intelligence Brief — run now or on a schedule.

Builds the fused brief (contract opportunities + live anomalies), persists it
via BriefDeliverer (JSON + Markdown + delivery log), and optionally runs on a
periodic schedule using APScheduler.

Usage:
    # Build + write + render once
    .venv/bin/python scripts/deliver_brief.py --once

    # Report last delivery + status
    .venv/bin/python scripts/deliver_brief.py --status

    # Run every 60 minutes until Ctrl+C
    .venv/bin/python scripts/deliver_brief.py --interval-min 60

Environment:
    TIRRA_DELIVERY_DIR    output dir (default .tirra_delivery)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.delivery.brief_deliverer import BriefDeliverer  # noqa: E402
from scripts.intelligence_brief import build_brief, render_markdown  # noqa: E402


def _deliver_once(
    *,
    contracts: int,
    anomalies: int,
    max_rows: int,
    learner: str,
    db: str,
    out_dir: str,
) -> dict:
    """Build + deliver one brief; returns the delivery record dict."""
    deliverer = BriefDeliverer(out_dir=out_dir, render_md=render_markdown)
    brief = build_brief(
        contracts_limit=contracts,
        anomalies_limit=anomalies,
        learner_path=learner,
        db_path=db,
        max_contract_rows=max_rows,
    )
    record = deliverer.deliver(brief)
    print(json.dumps(record.as_dict(), indent=2, ensure_ascii=False))
    return {"brief": brief, "record": record.as_dict()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Deliver the Intelligence Brief (once or scheduled)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="build + deliver once and exit")
    mode.add_argument("--status", action="store_true", help="show last delivery + count")
    mode.add_argument("--interval-min", type=int, default=0, help="run every N minutes until Ctrl+C")
    parser.add_argument("--contracts", type=int, default=10, help="awards to fetch")
    parser.add_argument("--anomalies", type=int, default=8, help="anomalies to include")
    parser.add_argument("--max-contract-rows", type=int, default=5, help="contract rows to keep")
    parser.add_argument("--learner", type=str, default=".tirra_opportunities/win_learner.jsonl")
    parser.add_argument("--db", type=str, default=".tirra_pipeline/pipeline.db")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out or ".tirra_delivery"

    deliverer = BriefDeliverer(out_dir=out_dir, render_md=render_markdown)

    if args.status:
        print(json.dumps(deliverer.status(), indent=2, ensure_ascii=False))
        return 0

    if args.interval_min > 0:
        from apscheduler.schedulers.blocking import BlockingScheduler
        sched = BlockingScheduler(timezone="UTC")

        def _job() -> None:
            try:
                _deliver_once(
                    contracts=args.contracts, anomalies=args.anomalies,
                    max_rows=args.max_contract_rows, learner=args.learner,
                    db=args.db, out_dir=out_dir,
                )
            except Exception as exc:
                print(f"[deliver] job failed: {exc}", file=sys.stderr)

        sched.add_job(_job, "interval", minutes=args.interval_min, next_run_time=None)
        print(f"[deliver] scheduled every {args.interval_min} min → {out_dir} (Ctrl+C to stop)")
        # run immediately first, then on schedule
        _job()
        try:
            sched.start()
        except (KeyboardInterrupt, SystemExit):
            print("[deliver] stopped")
        return 0

    # default: --once
    _deliver_once(
        contracts=args.contracts, anomalies=args.anomalies,
        max_rows=args.max_contract_rows, learner=args.learner,
        db=args.db, out_dir=out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
