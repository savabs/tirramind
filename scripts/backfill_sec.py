#!/usr/bin/env python3
"""Backfill SEC EDGAR history for form144 / insider_filings.

Why this exists
---------------
Both collectors hardcoded ``end_dt = date.today()``. EDGAR full-text search
returns newest-first, so asking for a longer ``days_back`` re-read the same
recent page instead of reaching further back. Measured 2026-09-23:

    form144        4,350 observations over  950 entities  =  4 obs/entity
    insider_filings  989 observations over  178 entities  =  5 obs/entity

A chain node needs a z-score, and a z-score needs roughly 20+ points in one
series. At 4-5 observations per entity neither source can be used for anything.
EDGAR holds years of both — the depth was always there, we just could not ask
for it.

How it works
------------
Walks the window backwards from newest to oldest in small slices. Newest-first
is deliberate: if a long run is interrupted, what you have is contiguous with
the present rather than stranded in the past.

Window sizing matters. Measured on live EDGAR:

    1 day  of Form 144  ~   87-111 filings
    3 days              ~   360 filings
    30 days             ~ 3,282 filings

``fetch_edgar_hits`` stops at ``EDGAR_PAGE_CAP`` (500) per window, so a window
wide enough to exceed that silently loses the remainder. Keep slices small;
the script refuses to run with a window it believes will overflow.

Safety
------
Default is ``--dry-run``: it reports what it WOULD fetch and writes nothing.
``--apply`` requires ``--backup`` to have been taken, because this writes to
the live pipeline DB — the project's only irreplaceable asset.

Resumable: re-running skips windows that already have observations, so an
interrupted run continues rather than restarting.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline.store import PipelineStore  # noqa: E402
from agent.tools._sec_window import EDGAR_PAGE_CAP, EdgarTruncated, walk_back  # noqa: E402

# Filings per day, measured against live EDGAR on 2026-09-23. Used only to
# refuse an obviously-overflowing window before wasting hours on it.
FILINGS_PER_DAY = {"form144": 110, "insider_filings": 700}

TOOLS = {
    "form144": ("agent.tools.form144", "Form144Tool", "144"),
    "insider_filings": ("agent.tools.insider_filings", "InsiderFilingsTool", "4"),
}


def _load_tool(name: str, store: PipelineStore):
    module_path, cls_name, _ = TOOLS[name]
    module = __import__(module_path, fromlist=[cls_name])
    return getattr(module, cls_name)(pipeline_store=store)


def _covered_days(db_path: str, source_tool: str) -> set[date]:
    """Days that already have at least one observation from this source."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT date(observed_at, 'unixepoch') FROM entity_observations WHERE source_tool = ?",
            (source_tool,),
        ).fetchall()
    finally:
        conn.close()
    out: set[date] = set()
    for (d,) in rows:
        if d:
            try:
                out.add(datetime.strptime(d, "%Y-%m-%d").date())
            except ValueError:
                continue
    return out


def _row_count(db_path: str, source_tool: str) -> int:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM entity_observations WHERE source_tool = ?", (source_tool,)
        ).fetchone()[0]
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", required=True, choices=sorted(TOOLS))
    ap.add_argument("--oldest", required=True, help="Earliest date to reach, YYYY-MM-DD")
    ap.add_argument("--newest", default="", help="Latest date (default: today)")
    ap.add_argument("--window-days", type=int, default=3)
    ap.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    ap.add_argument("--apply", action="store_true", help="Actually write. Requires --backup.")
    ap.add_argument("--backup", default="", help="Path to the backup you already took.")
    ap.add_argument("--skip-covered", action="store_true", default=True)
    args = ap.parse_args()

    oldest = datetime.strptime(args.oldest, "%Y-%m-%d").date()
    newest = datetime.strptime(args.newest, "%Y-%m-%d").date() if args.newest else date.today()

    expected = FILINGS_PER_DAY.get(args.tool, 200) * args.window_days
    if expected > EDGAR_PAGE_CAP:
        print(
            f"REFUSING: a {args.window_days}-day window of {args.tool} is ~{expected} filings, "
            f"over the {EDGAR_PAGE_CAP} per-window cap — it would silently lose the remainder.\n"
            f"Use --window-days {max(1, EDGAR_PAGE_CAP // FILINGS_PER_DAY.get(args.tool, 200))} or less."
        )
        return 2

    windows = walk_back(oldest=oldest, newest=newest, window_days=args.window_days)
    covered = _covered_days(args.db_path, args.tool) if args.skip_covered else set()

    todo = []
    for start, end in windows:
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        if covered and all(d in covered for d in days):
            continue
        todo.append((start, end))

    print(f"tool            : {args.tool}")
    print(f"range           : {oldest} .. {newest}")
    print(f"windows total   : {len(windows)}  ({args.window_days}d each)")
    print(f"already covered : {len(windows) - len(todo)}")
    print(f"to fetch        : {len(todo)}")
    print(f"rows now        : {_row_count(args.db_path, args.tool):,}")
    print(
        f"est. runtime    : ~{len(todo) * expected * 0.5 / 60:.0f} min (~{expected} filings/window, one XML fetch each)"
    )

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply --backup <path> to execute.")
        return 0

    if not args.backup or not Path(args.backup).exists():
        print("\nREFUSING: --apply requires --backup pointing at an existing backup file.")
        return 2

    store = PipelineStore(db_path=args.db_path)
    tool = _load_tool(args.tool, store)
    before = _row_count(args.db_path, args.tool)
    truncated: list[str] = []
    failed: list[str] = []

    for i, (start, end) in enumerate(todo, 1):
        span = (end - start).days + 1
        t0 = time.time()
        try:
            result = tool.execute(days_back=span - 1, end_date=str(end), _backfill=True)
            ok = bool(getattr(result, "success", False))
            n = (getattr(result, "data", None) or {}).get("total_filings", 0)
        except EdgarTruncated as exc:
            truncated.append(f"{start}..{end}: {exc}")
            print(f"[{i}/{len(todo)}] {start}..{end}  TRUNCATED — {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{start}..{end}: {type(exc).__name__}: {exc}")
            print(f"[{i}/{len(todo)}] {start}..{end}  FAILED — {type(exc).__name__}: {exc}")
            continue
        total = _row_count(args.db_path, args.tool)
        print(
            f"[{i}/{len(todo)}] {start}..{end}  ok={ok} filings={n} "
            f"rows={total:,} (+{total - before:,} total)  {time.time() - t0:.0f}s"
        )

    after = _row_count(args.db_path, args.tool)
    print(f"\nrows before : {before:,}")
    print(f"rows after  : {after:,}   (+{after - before:,})")
    if truncated:
        print(f"\n{len(truncated)} TRUNCATED windows — narrow --window-days and re-run:")
        for t in truncated[:10]:
            print(f"   {t}")
    if failed:
        print(f"\n{len(failed)} FAILED windows:")
        for f in failed[:10]:
            print(f"   {f}")
    # A run that truncated or failed anywhere is not a success, even if rows grew.
    return 1 if (truncated or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
