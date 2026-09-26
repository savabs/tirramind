#!/usr/bin/env python3
"""Give every entity_link the date it actually became true (audit P2.3).

The problem
-----------
``entity_links`` carries exactly one timestamp: ``created_at``, which is when
we INSERTED the row. Every insert happened between 2026-04-18 and 2026-09-24.
It says nothing about when the relation became true.

That single fact forces a choice with no good branch:

  * treat relations as always-true and a 1997 backtest sees who works where in
    2026 — F-04, silent and flattering; or
  * gate on ``created_at`` and the graph only exists for five months, so 83.5%
    of observations train against no graph at all.

Measured 2026-09-24 after gating: 416,631 observations total, 68,633 (16.5%)
inside the window where links exist. 142 trading days. Not enough to conclude
anything.

The recovery
------------
The dates were always there — nobody wrote them onto the links. A ``works_for``
link exists because someone filed with the SEC, and that filing has a date. A
``transacts_with`` link exists because of a blockchain transaction, which has a
timestamp. Checked against the live DB, 94-100% of links in every event
relation have dated evidence from the same source tool.

The rule
--------
    effective_from = MAX( MIN(observed_at of A from S), MIN(observed_at of B from S) )

A link between A and B, asserted by source S, cannot have been knowable before
S had seen BOTH endpoints. Taking the later of the two first-sightings is the
conservative direction: it can only ever move a link FORWARD in time relative
to a naive MIN, and moving a link later can hide a real edge but can never
invent one. Given the choice, lose an edge rather than leak one.

Where an endpoint has no observations from S (a company entity registered by a
filing but never observed itself), fall back to the endpoint that does. Where
neither does, leave NULL — ``_link_effective_time`` then falls back to
``created_at`` exactly as before, so nothing regresses.

This is an APPROXIMATION and should be read as one: it recovers when the
source first had evidence of the pair, not the true start of the underlying
relationship. An employment that began in 2015 and first appears in a 2026
filing will be dated 2026. That is the safe direction.

Safety
------
``--dry-run`` is the default and writes nothing. ``--apply`` requires
``--backup`` pointing at an existing backup file. The column is added nullable,
so the schema change alone changes no behaviour.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EPOCH_MIN = 631152000.0  # 1990-01-01, matching the store's write guard
EPOCH_MAX_SLACK = 86400.0


def _iso(ts: float | None) -> str:
    if ts is None:
        return "—"
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d")


def add_column(conn: sqlite3.Connection) -> bool:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(entity_links)")}
    if "effective_from" in cols:
        return False
    conn.execute("ALTER TABLE entity_links ADD COLUMN effective_from REAL")
    conn.commit()
    return True


def compute(conn: sqlite3.Connection) -> list[tuple[float, int]]:
    """Return (effective_from, link_id) pairs to write."""
    # First sighting of each (entity, source) pair — one scan, not one query
    # per link. 30k links x 2 endpoints would otherwise be 60k queries.
    first_seen: dict[tuple[str, str], float] = {}
    for eid, src, ts in conn.execute(
        "SELECT entity_id, source_tool, MIN(observed_at) FROM entity_observations "
        "WHERE observed_at >= ? GROUP BY entity_id, source_tool",
        (EPOCH_MIN,),
    ):
        if eid and src and ts is not None:
            first_seen[(eid, src)] = float(ts)

    updates: list[tuple[float, int]] = []
    for link_id, a, b, src, created in conn.execute(
        "SELECT link_id, entity_id_a, entity_id_b, source, created_at FROM entity_links"
    ):
        ta = first_seen.get((a, src))
        tb = first_seen.get((b, src))
        # Both endpoints seen -> the later first-sighting. One seen -> that one.
        # Neither -> leave NULL and let created_at fallback stand.
        if ta is not None and tb is not None:
            eff = max(ta, tb)
        elif ta is not None:
            eff = ta
        elif tb is not None:
            eff = tb
        else:
            continue
        # Never date a link after we inserted it: created_at is a hard ceiling
        # on when we could possibly have known, so a later value is a bug.
        if created is not None and eff > float(created):
            eff = float(created)
        updates.append((eff, link_id))
    return updates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", default="")
    args = ap.parse_args()

    ro = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
    total = ro.execute("SELECT COUNT(*) FROM entity_links").fetchone()[0]
    updates = compute(ro)

    print(f"links total          : {total:,}")
    print(f"datable from evidence: {len(updates):,}  ({len(updates) / max(total, 1) * 100:.1f}%)")
    print(f"left NULL            : {total - len(updates):,}  (fall back to created_at, as today)")

    if updates:
        vals = sorted(e for e, _ in updates)
        print(f"effective_from range : {_iso(vals[0])} .. {_iso(vals[-1])}")
        print()
        print("distribution by year:")
        by_year: dict[int, int] = {}
        for e in vals:
            y = dt.datetime.fromtimestamp(e, dt.UTC).year
            by_year[y] = by_year.get(y, 0) + 1
        for y in sorted(by_year):
            bar = "#" * max(1, round(by_year[y] / max(by_year.values()) * 40))
            print(f"   {y}  {by_year[y]:>6,}  {bar}")
    ro.close()

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply --backup <path>.")
        return 0
    if not args.backup or not Path(args.backup).exists():
        print("\nREFUSING: --apply needs --backup pointing at an existing backup.")
        return 2

    rw = sqlite3.connect(args.db_path)
    added = add_column(rw)
    print(f"\ncolumn effective_from: {'added' if added else 'already present'}")
    rw.executemany("UPDATE entity_links SET effective_from = ? WHERE link_id = ?", updates)
    rw.commit()
    n = rw.execute("SELECT COUNT(*) FROM entity_links WHERE effective_from IS NOT NULL").fetchone()[0]
    after = rw.execute("SELECT COUNT(*) FROM entity_links").fetchone()[0]
    rw.close()

    # A migration that changed the row count did something other than what it said.
    if after != total:
        print(f"FAILED: link count changed {total:,} -> {after:,}")
        return 1
    print(f"written              : {n:,} of {after:,} links now dated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
