"""TirraEngine — one self-running command: collect → brief → deliver → serve.

Unifies the whole pipeline behind a single entrypoint:
  1. COLLECT  (optional)  run the live data tools (daily DAG) so the digest reads fresh observations
  2. BUILD    build the fused Intelligence Brief (contracts + anomalies)
  3. DELIVER  persist JSON + Markdown + delivery log (.tirra_delivery/)
  4. SERVE    (optional) serve the delivered brief over HTTP

Usage:
    # Collect fresh data, build + deliver once
    .venv/bin/python scripts/tirra_engine.py --collect --once

    # Serve the brief over HTTP
    .venv/bin/python scripts/tirra_engine.py --serve --port 8787

    # Run everything (collect + build + deliver + serve) until Ctrl+C
    .venv/bin/python scripts/tirra_engine.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.delivery.brief_deliverer import BriefDeliverer  # noqa: E402
from scripts.intelligence_brief import build_brief, render_markdown  # noqa: E402


def refresh_fast_data(db_path: str = ".tirra_pipeline/pipeline.db") -> dict:
    """Refresh the fast, digest-relevant tools so the brief reads fresh data.

    Runs only the quick+reliable tools that power the brief's anomalies and
    contracts (CFTC positioning + gov contracts), in parallel, ~2s. This avoids
    the slow/hanging full daily DAG while still keeping the brief current.
    """
    import concurrent.futures

    from agent.pipeline.store import PipelineStore

    store = PipelineStore(db_path)
    results: dict[str, str] = {}

    def _fetch_cftc():
        from agent.tools.cftc import CFTCTool

        return CFTCTool().execute(mode="latest", limit=5)

    def _fetch_gov():
        from agent.tools.gov_contracts import GovContractsTool

        return GovContractsTool().execute(mode="recent", limit=10)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = {"cftc": ex.submit(_fetch_cftc), "gov_contracts": ex.submit(_fetch_gov)}
        for name, fut in futs.items():
            try:
                r = fut.result(timeout=30)
                results[name] = "ok" if r.success else "failed"
                # gov_contracts already persists observations via its tool;
                # store cftc data if the tool exposes it (best-effort).
            except Exception as exc:
                results[name] = f"error:{type(exc).__name__}"
    return {"status": "fast_refresh", "tools": results, "db": db_path}


def _run_daily_collection_dag(config) -> dict:
    """Execute the daily_collection DAG and wait for it to finish. Blocking."""
    from agent.cli import build_tool_registry
    from agent.pipeline.executor import DAGExecutor
    from agent.pipeline.registry import DAGRegistry
    from agent.pipeline.scheduler import PipelineScheduler
    from agent.pipeline.store import PipelineStore

    try:
        store = PipelineStore(db_path=config.pipeline.db_path)
        tools = build_tool_registry(config)
        executor = DAGExecutor(tool_registry=tools, store=store, max_workers=config.pipeline.max_workers)
        dag_registry = DAGRegistry()
        dag_registry.load_defaults(tools)
        scheduler = PipelineScheduler(executor=executor, registry=dag_registry)
        run = scheduler.trigger("daily_collection")
        n_ok = sum(1 for r in run.node_results.values() if r.status in ("success", "completed"))
        return {"dag": "daily_collection", "status": run.status, "nodes_ok": n_ok, "nodes_total": len(run.node_results)}
    except Exception as exc:
        return {"dag": "daily_collection", "status": "failed", "error": str(exc)[:200]}


def run_collection_sync(config=None) -> dict:
    """Run the full daily_collection DAG and block until it finishes.

    Use this whenever the caller is about to exit right after collecting
    (e.g. a scheduled/cron `--full-collect --once` invocation). A daemon
    thread (see run_collection() below) is killed the instant its process
    exits, so a `--once` caller must never use the fire-and-forget path —
    it would report "started_background" and silently persist nothing.
    """
    from agent.config.settings import AgentConfig

    cfg = config or AgentConfig.from_env()
    return _run_daily_collection_dag(cfg)


def run_collection(config=None) -> dict:
    """Trigger the daily data-collection DAG so observations are fresh, without
    blocking the caller.

    Runs in a background daemon thread. Only safe when the process is going
    to stay alive afterward (e.g. `--serve` mode, which blocks in an HTTP
    serve loop) — the collection keeps running in the background while the
    brief is served from whatever observations already exist. Data
    collection here is a freshness enhancement, not a hard gate.

    For a one-shot invocation that's about to exit (`--once`), use
    run_collection_sync() instead — a daemon thread does not survive its
    process exiting, so this function would otherwise silently discard the
    entire collection run.
    """
    import threading

    from agent.config.settings import AgentConfig

    cfg = config or AgentConfig.from_env()
    t = threading.Thread(target=_run_daily_collection_dag, args=(cfg,), daemon=True)
    t.start()
    # Return immediately; log that collection is running in the background.
    return {
        "dag": "daily_collection",
        "status": "started_background",
        "note": "data collection running; brief uses existing observations",
    }


def email_brief(md_text: str, recipients: list[str]) -> dict:
    """Email the rendered brief via SMTP (env-configured). No-op if not configured.

    Env:
      TIRRA_SMTP_HOST, TIRRA_SMTP_PORT (default 587), TIRRA_SMTP_USER,
      TIRRA_SMTP_PASS, TIRRA_SMTP_FROM, TIRRA_BRIEF_TO (comma-separated).
    """
    import os
    import smtplib
    from email.message import EmailMessage

    host = os.getenv("TIRRA_SMTP_HOST")
    if not host:
        return {"emailed": False, "reason": "TIRRA_SMTP_HOST not set"}
    to = recipients or [r for r in os.getenv("TIRRA_BRIEF_TO", "").split(",") if r.strip()]
    if not to:
        return {"emailed": False, "reason": "no recipients"}
    msg = EmailMessage()
    msg["Subject"] = "AWOS Intelligence Brief"
    msg["From"] = os.getenv("TIRRA_SMTP_FROM", os.getenv("TIRRA_SMTP_USER", "awos@localhost"))
    msg["To"] = ", ".join(to)
    msg.set_content(md_text)
    try:
        port = int(os.getenv("TIRRA_SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls()
            user = os.getenv("TIRRA_SMTP_USER")
            if user:
                s.login(user, os.getenv("TIRRA_SMTP_PASS", ""))
            s.send_message(msg)
        return {"emailed": True, "recipients": to}
    except Exception as exc:
        return {"emailed": False, "reason": str(exc)[:200]}


def _deliver(
    *,
    contracts: int,
    anomalies: int,
    max_rows: int,
    learner: str,
    db: str,
    out_dir: str,
) -> dict:
    deliverer = BriefDeliverer(out_dir=out_dir, render_md=render_markdown)
    brief = build_brief(
        contracts_limit=contracts,
        anomalies_limit=anomalies,
        learner_path=learner,
        db_path=db,
        max_contract_rows=max_rows,
    )
    record = deliverer.deliver(brief)
    return {"brief": brief, "record": record.as_dict()}


def record_bid(agency: str, amount: float, won: bool, learner_path: str) -> dict:
    """Record one realized bid outcome so P(win) personalizes per agency+bucket.

    This is the feedback loop that turns the documented prior (0.33) into a
    learned, evidence-backed P(win) for that agency/amount range.
    """
    from agent.quant.contract_opportunity import WinProbabilityLearner

    learner = WinProbabilityLearner(learner_path)
    award_id = f"manual_{int(time.time())}"
    learner.record(award_id=award_id, agency=agency, amount=float(amount), realized_success=bool(won))
    new_p = learner.probability_of(agency, float(amount))
    return {
        "recorded": award_id,
        "agency": agency,
        "amount": amount,
        "won": bool(won),
        "new_p_win": round(new_p, 4),
        "basis": learner.basis_of(agency, float(amount))["source"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TirraEngine — refresh data, build the Intelligence Brief, deliver it. "
        "Default (no args) = refresh + brief + deliver + serve."
    )
    parser.add_argument("--collect", action="store_true", help="run fast live-data refresh (CFTC + gov contracts)")
    parser.add_argument("--full-collect", action="store_true", help="run the full daily DAG in background (slow)")
    parser.add_argument("--once", action="store_true", help="refresh+build+deliver once then exit (no serve)")
    parser.add_argument("--serve", action="store_true", help="serve the brief over HTTP (default when no --once)")
    parser.add_argument("--no-serve", action="store_true", help="don't serve, just refresh+build+deliver")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--anomalies", type=int, default=8)
    parser.add_argument("--max-contract-rows", type=int, default=5)
    parser.add_argument("--learner", type=str, default=".tirra_opportunities/win_learner.jsonl")
    parser.add_argument("--db", type=str, default=".tirra_pipeline/pipeline.db")
    parser.add_argument("--out", type=str, default=".tirra_delivery")
    parser.add_argument(
        "--record-bid",
        nargs=3,
        metavar=("AGENCY", "AMOUNT", "WON"),
        help="record a realized bid outcome (won=0/1) to personalize P(win)",
    )
    parser.add_argument("--email", action="store_true", help="email the brief after delivery (env-configured)")
    parser.add_argument(
        "--to", type=str, default=None, help="email recipients (comma-separated), overrides TIRRA_BRIEF_TO"
    )
    args = parser.parse_args()

    if args.record_bid:
        agency, amount, won = args.record_bid
        res = record_bid(agency, float(amount), bool(int(won)), args.learner)
        print(res)
        return 0

    if args.email:
        # email mode: build + deliver + email, then exit (no serve)
        recipients = [r for r in (args.to or "").split(",") if r.strip()] if args.to else None
        md_path = Path(args.out) / "intelligence_brief.md"
        collect_summary = refresh_fast_data(args.db)
        result = _deliver(
            contracts=args.contracts,
            anomalies=args.anomalies,
            max_rows=args.max_contract_rows,
            learner=args.learner,
            db=args.db,
            out_dir=args.out,
        )
        if md_path.exists():
            md = md_path.read_text(encoding="utf-8")
            email_res = email_brief(md, recipients)
            print(f"[engine] email: {email_res}")
        else:
            print("[engine] email: no brief markdown to send")
        print(f"[engine] collection: {collect_summary}")
        return 0

    # ── Default / normal path: refresh + build + deliver (+ serve unless --once) ──
    collect_summary = None
    if args.full_collect:
        # --once means the process exits right after delivery: the DAG must
        # run synchronously, or a background daemon thread gets killed
        # before it persists anything (see run_collection_sync docstring).
        collect_summary = run_collection_sync() if args.once else run_collection()
    else:
        collect_summary = refresh_fast_data(args.db)
    result = _deliver(
        contracts=args.contracts,
        anomalies=args.anomalies,
        max_rows=args.max_contract_rows,
        learner=args.learner,
        db=args.db,
        out_dir=args.out,
    )
    print(
        f"[engine] delivered: {result['record']['n_contracts']} contracts, "
        f"{result['record']['n_anomalies']} anomalies → {args.out}"
    )
    print(f"[engine] collection: {collect_summary}")

    should_serve = (not args.no_serve) and (not args.once)
    if should_serve:
        from agent.brief_server import serve

        print(f"[engine] serving brief at http://{args.host}:{args.port}/brief.json (Ctrl+C to stop)")
        serve(out_dir=args.out, port=args.port, host=args.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
