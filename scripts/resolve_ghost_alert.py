#!/usr/bin/env python3
"""Resolve ghost pattern alerts using forward readout returns.

MVP uses 5 trading sessions (not 21 calendar days) so feedback loops stay fast.

Usage:
    python scripts/resolve_ghost_alert.py ghost_archive/alerts/2026-06-09_MP-1_EIA_REGIME_CFTC_001.json
    python scripts/resolve_ghost_alert.py --all
    python scripts/resolve_ghost_alert.py --all --sessions 5
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.quant.ghost_brief import update_brief_outcome
from agent.tools.instrument_universe import _entity_id

DB_PATH = Path(".tirra_pipeline/pipeline.db")
ARCHIVE_DIR = Path("ghost_archive/alerts")
DEFAULT_SESSIONS = 5


def _parse_ts(ts: str | float) -> datetime:
    s = str(ts).strip()
    if s.replace(".", "", 1).isdigit():
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _load_closes(con: sqlite3.Connection, ticker: str) -> dict[str, float]:
    """Load trading-day closes; prefer ``instrument_daily`` (bar date) over ingest snapshots."""
    eid = _entity_id(ticker)
    rows = con.execute(
        """
        SELECT observed_at, observation_type, value_json FROM entity_observations
        WHERE entity_id=? AND observation_type IN ('instrument_daily', 'instrument_return')
        ORDER BY observed_at
        """,
        (eid,),
    ).fetchall()
    out: dict[str, float] = {}
    for ts, obs_type, vj in rows:
        val = json.loads(vj)
        close = val.get("close")
        if close is None:
            continue
        day = _parse_ts(ts).strftime("%Y-%m-%d")
        # instrument_daily rows use the bar date as observed_at — preferred source.
        if obs_type == "instrument_daily" or day not in out:
            out[day] = float(close)
    return out


def _forward_return(
    closes: dict[str, float], anchor_date: str, n_sessions: int
) -> tuple[str, str, float] | None:
    days = sorted(closes.keys())
    if not days:
        return None
    # First trading day on or after anchor
    start_candidates = [d for d in days if d >= anchor_date[:10]]
    if not start_candidates:
        return None
    start = start_candidates[0]
    i = days.index(start)
    if i + n_sessions >= len(days):
        return None
    end = days[i + n_sessions]
    ret = (closes[end] / closes[start] - 1.0) * 100.0
    return start, end, ret


def resolve_alert(
    alert_path: Path,
    con: sqlite3.Connection,
    n_sessions: int = DEFAULT_SESSIONS,
    *,
    dry_run: bool = False,
) -> dict | None:
    alert = json.loads(alert_path.read_text(encoding="utf-8"))
    if alert.get("outcome") is not None:
        print(f"SKIP {alert['alert_id']}: already resolved")
        return alert

    ticker = alert.get("readout_instrument", "CL=F")
    closes = _load_closes(con, ticker)
    # Anchor at latest chain node — when the physical/readout signal completes
    anchor = max(n["observed_at"] for n in alert["nodes"])
    hit = _forward_return(closes, anchor, n_sessions)
    if hit is None and n_sessions > 2:
        hit = _forward_return(closes, anchor, 2)
        if hit:
            n_sessions = 2
    if hit is None:
        print(f"PENDING {alert['alert_id']}: need price sessions after {anchor[:10]}")
        return None

    start, end, ret = hit
    direction = "up" if ret > 0.5 else ("down" if ret < -0.5 else "flat")
    outcome = {
        "resolved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "direction": direction,
        "return_pct": round(ret, 2),
        "notes": f"{n_sessions}-session return {ticker} {start} -> {end}",
    }
    alert["evaluation_window_days"] = n_sessions
    alert["outcome"] = outcome

    if not dry_run:
        alert_path.write_text(json.dumps(alert, indent=2) + "\n", encoding="utf-8")
        for brief in update_brief_outcome(alert):
            print(f"  → updated brief {brief}")
    print(
        f"RESOLVED {alert['alert_id']}: {direction} {ret:+.2f}% "
        f"({start} -> {end}, {n_sessions} sessions)"
    )
    return alert


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve ghost alerts (fast MVP window)")
    parser.add_argument("alert", nargs="?", help="Path to alert JSON")
    parser.add_argument("--all", action="store_true", help="Resolve all pending in archive")
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db-path", default=str(DB_PATH))
    args = parser.parse_args()

    con = sqlite3.connect(args.db_path)
    paths: list[Path] = []
    if args.all:
        paths = sorted(ARCHIVE_DIR.glob("*.json"))
    elif args.alert:
        paths = [Path(args.alert)]
    else:
        parser.error("Provide alert path or --all")

    resolved = 0
    for p in paths:
        if resolve_alert(p, con, args.sessions, dry_run=args.dry_run):
            resolved += 1
    con.close()
    print(f"\n{resolved} alert(s) processed")


if __name__ == "__main__":
    main()
