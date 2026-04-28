"""AWOS command-line entrypoint.

Subcommands:
  daemon        — run the blocking scheduler loop
  run-once      — single pass: scan + dispatch (useful in cron/tests)
  classify      — classify text from stdin or file; print JSON
  events        — list recent events
  proposals     — list pending proposals
  accept        — accept a proposal (applies its diff to AWOS)
  reject        — reject a proposal (moves it to a rejected dir)
  scan          — run all enabled watchers once
  install-hooks — install git hooks
  uninstall-hooks — remove git hooks
  status        — print runtime status summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from agent.awos.classifiers.composite import CompositeClassifier
from agent.awos.config import AWOSConfig
from agent.awos.events.bus import EventBus
from agent.awos.events.schema import Event, EventStatus, TriggerCategory
from agent.awos.hooks.install import install as install_hooks
from agent.awos.hooks.install import uninstall as uninstall_hooks
from agent.awos.orchestrator.daemon import Daemon, run_once
from agent.awos.orchestrator.dispatcher import Dispatcher
from agent.awos.policies.engine import PolicyEngine


def _build_config(args: argparse.Namespace) -> AWOSConfig:
    yaml_path = Path(args.config) if args.config else None
    cfg = AWOSConfig.from_env(yaml_path=yaml_path)
    cfg.ensure_dirs()
    return cfg


# ---------------------------------------------------------------- daemon
def cmd_daemon(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    d = Daemon(cfg)
    d.start()
    return 0


def cmd_run_once(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    scanned, dispatched = run_once(cfg)
    print(json.dumps({"scanned": scanned, "dispatched": dispatched}))
    return 0


# -------------------------------------------------------------- classify
def cmd_classify(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text()
    else:
        text = sys.stdin.read()
    cls = CompositeClassifier.from_config(cfg)
    res = cls.classify(text)
    print(res.model_dump_json(indent=2))
    return 0


# ---------------------------------------------------------------- events
def cmd_events(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    bus = EventBus(cfg.db_path, dedup_window_s=cfg.dedup_window_s)  # type: ignore[arg-type]
    status = EventStatus(args.status) if args.status else None
    events = bus.fetch(limit=args.limit, status=status)
    for e in events:
        print(
            f"{e.ts.isoformat()}  {e.status.value:10s}  "
            f"{e.category.value:20s}  conf={e.confidence:.2f}  "
            f"{e.source:18s}  {e.id[:8]}  "
            f"{(e.rationale or '')[:80]}"
        )
    return 0


# ------------------------------------------------------------- proposals
def cmd_proposals(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    pdir = cfg.proposals_dir
    if pdir is None or not pdir.exists():
        print("(no proposals dir)")
        return 0
    files = sorted(pdir.glob("*.md"))
    for p in files:
        print(p.name)
    if not files:
        print("(no pending proposals)")
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    pdir = cfg.proposals_dir
    assert pdir is not None
    path = pdir / args.name if not Path(args.name).is_absolute() else Path(args.name)
    if not path.exists():
        print(f"proposal not found: {args.name}", file=sys.stderr)
        return 2
    accepted_dir = pdir.parent / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    target = accepted_dir / path.name
    path.replace(target)
    print(f"moved proposal to {target}")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    pdir = cfg.proposals_dir
    assert pdir is not None
    path = pdir / args.name if not Path(args.name).is_absolute() else Path(args.name)
    if not path.exists():
        print(f"proposal not found: {args.name}", file=sys.stderr)
        return 2
    rejected_dir = pdir.parent / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    target = rejected_dir / path.name
    path.replace(target)
    print(f"moved proposal to {target}")
    return 0


# ------------------------------------------------------------------ scan
def cmd_scan(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    scanned, dispatched = run_once(cfg)
    print(
        json.dumps(
            {
                "source": args.source or "cli",
                "scanned": scanned,
                "dispatched": dispatched,
            }
        )
    )
    return 0


# -------------------------------------------------------------- hooks
def cmd_install_hooks(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    paths = install_hooks(cfg.repo_root)
    print(json.dumps({"installed": [str(p) for p in paths]}, indent=2))
    return 0


def cmd_uninstall_hooks(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    paths = uninstall_hooks(cfg.repo_root)
    print(json.dumps({"removed": [str(p) for p in paths]}, indent=2))
    return 0


# -------------------------------------------------------------- status
def cmd_status(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    bus = EventBus(cfg.db_path, dedup_window_s=cfg.dedup_window_s)  # type: ignore[arg-type]
    new = bus.count(EventStatus.NEW)
    processed = bus.count(EventStatus.PROCESSED)
    errored = bus.count(EventStatus.ERRORED)
    proposals = len(list((cfg.proposals_dir or Path("/")).glob("*.md"))) if cfg.proposals_dir else 0
    awos = cfg.awos_file
    summary: dict[str, Any] = {
        "repo_root": str(cfg.repo_root),
        "awos_file": str(awos),
        "awos_exists": awos.exists(),
        "classifier_mode": cfg.classifier_mode,
        "anthropic_configured": bool(cfg.anthropic_api_key),
        "events": {
            "new": new,
            "processed": processed,
            "errored": errored,
        },
        "proposals_pending": proposals,
    }
    print(json.dumps(summary, indent=2))
    return 0


# -------------------------------------------------------------- publish (test-only)
def cmd_publish(args: argparse.Namespace) -> int:
    """Internal helper: publish a manual event."""
    cfg = _build_config(args)
    bus = EventBus(cfg.db_path, dedup_window_s=cfg.dedup_window_s)  # type: ignore[arg-type]
    event = Event(
        source=args.source or "cli",
        category=TriggerCategory(args.category),
        confidence=float(args.confidence),
        rationale=args.rationale or "",
        payload={"extracted_principle": args.principle} if args.principle else {},
    )
    stored = bus.publish(event)
    # optionally dispatch immediately
    if args.dispatch:
        policies = PolicyEngine.load(cfg.policies_file)
        dispatcher = Dispatcher(cfg, bus, policies)
        report = dispatcher.dispatch(stored)
        print(
            json.dumps(
                {
                    "event_id": stored.id,
                    "planned": report.planned,
                    "executed": report.executed,
                    "failed": report.failed,
                },
                indent=2,
            )
        )
    else:
        print(json.dumps({"event_id": stored.id}, indent=2))
    return 0


# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tirra-awos", description=__doc__)
    p.add_argument("--config", help="path to awos yaml config")
    p.add_argument("--log-level", default="WARNING")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("daemon", help="run the blocking scheduler")
    sp.set_defaults(func=cmd_daemon)

    sp = sub.add_parser("run-once", help="single-pass scan + dispatch")
    sp.set_defaults(func=cmd_run_once)

    sp = sub.add_parser("classify", help="classify text")
    sp.add_argument("--text")
    sp.add_argument("--file")
    sp.set_defaults(func=cmd_classify)

    sp = sub.add_parser("events", help="list recent events")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--status", choices=[s.value for s in EventStatus])
    sp.set_defaults(func=cmd_events)

    sp = sub.add_parser("proposals", help="list pending proposals")
    sp.set_defaults(func=cmd_proposals)

    sp = sub.add_parser("accept", help="accept a proposal file")
    sp.add_argument("name")
    sp.set_defaults(func=cmd_accept)

    sp = sub.add_parser("reject", help="reject a proposal file")
    sp.add_argument("name")
    sp.set_defaults(func=cmd_reject)

    sp = sub.add_parser("scan", help="run watchers once")
    sp.add_argument("--source", default=None)
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("install-hooks", help="install git hooks")
    sp.set_defaults(func=cmd_install_hooks)

    sp = sub.add_parser("uninstall-hooks", help="remove git hooks")
    sp.set_defaults(func=cmd_uninstall_hooks)

    sp = sub.add_parser("status", help="print status summary")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("publish", help="publish a manual event (for tests)")
    sp.add_argument("--source", default="cli")
    sp.add_argument(
        "--category",
        required=True,
        choices=[c.value for c in TriggerCategory],
    )
    sp.add_argument("--confidence", type=float, default=0.9)
    sp.add_argument("--rationale", default="")
    sp.add_argument("--principle", default=None)
    sp.add_argument("--dispatch", action="store_true")
    sp.set_defaults(func=cmd_publish)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
