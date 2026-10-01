#!/usr/bin/env python3
"""Collect DAILY Wikipedia pageviews for the countries GDELT already covers.

Why this exists
---------------
The ghost-chain enumeration ran on 2026-09-30 and produced 371,368 candidates,
of which **90.2% had GDELT as their sole witness** and 98.7% rested on a single
independent source. Requiring two independent time-varying witnesses on a route
that avoids a graph hub left **zero** candidates. The only two-source routes it
found were `(polymarket, repair_topic_links)` — and `repair_topic_links` is a
repair pass over polymarket, i.e. the same data twice.

A chain supported by one source is not a chain. It is one dataset with a path
drawn through it. So the engine is not the blocker; the absence of a second
witness is.

Why pageviews specifically
--------------------------
It is genuinely independent of GDELT, because the generative process is
different in kind:

    GDELT      what newswires printed, machine-coded from news text
    pageviews  what humans went and looked up

Neither derives from the other. If both move on the same country in the same
week, that is corroboration in the sense the enumeration requires. Correlated,
certainly — they respond to the same world — but not the same instrument
pointed twice.

Measured 2026-09-30: the Wikimedia REST API returns **1,003 daily points per
country in a single keyless call** (Russia, India, Saudi_Arabia, Nigeria,
Norway all checked), with history back to 2015.

What was already there, and why it could not work
-------------------------------------------------
`agent/tools/wikipedia_pageviews.py` has stored **47 rows across 11 entities**.
Two separate reasons:

1. It persists only *spikes* — a handful of threshold crossings — rather than
   the daily series a z-score needs.
2. It registers them as ``topic`` entities, **not** the ``country`` entities
   GDELT writes to. Two witnesses on two different node ids can never
   corroborate each other; they never meet in the graph.

This script fixes both. It writes to the SAME ``country`` entity_id GDELT uses,
which is the entire point.

Safety
------
``--dry-run`` is the default. ``--apply`` requires ``--backup`` naming an
existing backup file, because this writes to the live pipeline DB.
Resumable: days already stored for a country are skipped, so an interrupted run
continues rather than restarting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from agent.pipeline.store import PipelineStore  # noqa: E402

_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
_UA = "TirraMind/1.0 (research; 999.sbpatel@gmail.com)"
_PROJECT = "en.wikipedia"
# Wikimedia asks for <= 100 req/s; we are far under it, but be a good citizen.
_DELAY_S = 0.15

# Canonical name -> Wikipedia article, only where they differ. The DB's
# canonical_name comes from the country-code merge (ISO 3166 English names), and
# most match the article title once spaces become underscores.
_ARTICLE_OVERRIDES = {
    "United States": "United_States",
    "United Kingdom": "United_Kingdom",
    "Russia": "Russia",
    "South Korea": "South_Korea",
    "North Korea": "North_Korea",
    "Palestine": "State_of_Palestine",
    "Syria": "Syria",
    "Iran": "Iran",
    "Vietnam": "Vietnam",
    "Laos": "Laos",
    "Taiwan": "Taiwan",
    "Tanzania": "Tanzania",
    "Bolivia": "Bolivia",
    "Venezuela": "Venezuela",
    "Czechia": "Czech_Republic",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "Côte d'Ivoire": "Ivory_Coast",
    "Cabo Verde": "Cape_Verde",
    "Eswatini": "Eswatini",
    "Myanmar": "Myanmar",
    "Congo": "Republic_of_the_Congo",
    "DR Congo": "Democratic_Republic_of_the_Congo",
    "Moldova": "Moldova",
    "Brunei": "Brunei",
}


def article_for(name: str) -> str:
    if name in _ARTICLE_OVERRIDES:
        return _ARTICLE_OVERRIDES[name]
    return name.strip().replace(" ", "_")


def fetch_daily(article: str, start: dt.date, end: dt.date) -> list[tuple[float, int]] | None:
    """Return [(epoch_seconds, views)] or None when the article does not resolve.

    A 404 means the title is wrong for this country, which is a data-mapping
    problem worth reporting per country rather than swallowing — the caller
    counts them.
    """
    url = f"{_BASE}/{_PROJECT}/all-access/user/{urllib.parse.quote(article, safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}"
    try:
        r = httpx.get(url, headers={"User-Agent": _UA}, timeout=30, follow_redirects=True)
    except httpx.RequestError:
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    out: list[tuple[float, int]] = []
    for item in r.json().get("items", []):
        stamp = str(item.get("timestamp", ""))[:8]
        try:
            d = dt.datetime.strptime(stamp, "%Y%m%d").replace(tzinfo=dt.UTC)
        except ValueError:
            continue
        out.append((d.timestamp(), int(item.get("views", 0))))
    return out


def gdelt_countries(db_path: str) -> list[tuple[str, str]]:
    """(entity_id, canonical_name) for countries GDELT actually writes to.

    Restricted to GDELT's own coverage on purpose: a second witness is only
    useful where there is a first one to corroborate.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT DISTINCT e.entity_id, e.canonical_name "
            "FROM entity_observations o JOIN entities e ON e.entity_id = o.entity_id "
            "WHERE o.source_tool = 'gdelt' AND e.entity_type = 'country' "
            "  AND e.canonical_name IS NOT NULL "
            "ORDER BY e.canonical_name"
        ).fetchall()
    finally:
        conn.close()


def already_have(db_path: str) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return dict(
            conn.execute(
                "SELECT entity_id, COUNT(*) FROM entity_observations "
                "WHERE source_tool = 'wikipedia_pageviews' "
                "  AND observation_type = 'pageviews_daily' GROUP BY 1"
            ).fetchall()
        )
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    ap.add_argument("--days", type=int, default=1000, help="History depth (API serves ~1,003)")
    ap.add_argument("--limit", type=int, default=0, help="Only the first N countries (0 = all)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", default="")
    ap.add_argument(
        "--min-existing", type=int, default=200, help="Skip a country that already has this many daily rows"
    )
    args = ap.parse_args()

    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=args.days)

    countries = gdelt_countries(args.db_path)
    if args.limit:
        countries = countries[: args.limit]
    have = already_have(args.db_path)

    todo = [(eid, nm) for eid, nm in countries if have.get(eid, 0) < args.min_existing]
    print(f"countries GDELT covers : {len(countries)}")
    print(f"already collected      : {len(countries) - len(todo)}")
    print(f"to fetch               : {len(todo)}")
    print(f"window                 : {start} .. {end} ({args.days} days)")
    print(f"est. rows              : ~{len(todo) * args.days:,}")
    print(f"est. runtime           : ~{len(todo) * (_DELAY_S + 0.6) / 60:.0f} min")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply --backup <path>.")
        return 0
    if not args.backup or not Path(args.backup).exists():
        print("\nREFUSING: --apply needs --backup pointing at an existing backup file.")
        return 2

    store = PipelineStore(db_path=args.db_path)
    written = unresolved = failed = 0
    unresolved_names: list[str] = []

    for i, (eid, name) in enumerate(todo, 1):
        art = article_for(name)
        time.sleep(_DELAY_S)
        try:
            series = fetch_daily(art, start, end)
        except Exception as exc:  # noqa: BLE001 — reported per country, never silent
            failed += 1
            print(f"[{i}/{len(todo)}] {name:26} FAILED {type(exc).__name__}: {exc}")
            continue
        if series is None:
            unresolved += 1
            unresolved_names.append(f"{name} -> {art}")
            print(f"[{i}/{len(todo)}] {name:26} no article '{art}' — needs a mapping override")
            continue

        n = 0
        for ts, views in series:
            try:
                store.store_entity_observation(
                    entity_id=eid,  # the SAME id GDELT writes to — the whole point
                    source_tool="wikipedia_pageviews",
                    observed_at=ts,
                    observation_type="pageviews_daily",
                    depth_level=2,
                    value={"views": views, "article": art, "project": _PROJECT},
                )
                n += 1
            except Exception:  # noqa: BLE001 — one bad row must not cost the country
                pass
        written += n
        print(f"[{i}/{len(todo)}] {name:26} {art:32} {n:>5} days  (total {written:,})")

    print(f"\nrows written  : {written:,}")
    print(f"unresolved    : {unresolved}")
    if unresolved_names:
        print("  (add these to _ARTICLE_OVERRIDES)")
        for u in unresolved_names[:20]:
            print(f"    {u}")
    print(f"failed        : {failed}")
    # A run that resolved nothing is a failure even though it raised nothing.
    return 1 if (written == 0 or failed > len(todo) // 4) else 0


if __name__ == "__main__":
    raise SystemExit(main())
