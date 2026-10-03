#!/usr/bin/env python3
"""Merge duplicate country entities onto canonical ISO 3166-1 alpha-2 keys.

Research: docs/research/graph_connectivity_failure.md
Spec:     docs/specs/graph_connectivity_repair_spec.md  (step 3)
Task:     tasks/active/graph_connectivity_repair.md

GDELT keyed country entities on CAMEO alpha-3 (``country:USA``); the instrument
universe keyed them on ISO alpha-2 (``country:US``). ``entity_id`` is
``sha256(f"{type}:{key}")``, so the same country became two unrelated entities
and 0 of 215 GDELT countries could reach any instrument.

This repoints every reference from the alpha-3 record onto the alpha-2 record.

Safety
------
- ``--dry-run`` is the DEFAULT. ``--apply`` additionally requires ``--backup``.
- Every row is accounted for. Links that collide after the merge, and links that
  become self-links, are COUNTED and reported -- never silently dropped
  (spec INV-2). A migration that quietly loses rows is the failure mode this
  repo has hit repeatedly; see docs/publications/thirteen_ways_a_pipeline_lies.md
- CAMEO regional blocs (EUR, MEA, SEA ...) are NOT merged and NOT retyped.
  Retyping them would introduce an entity type the model cannot encode, which is
  the F-12 schema-drift class. They are reported for a separate decision.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.pipeline.country_codes import (  # noqa: E402
    country_name,
    is_region_code,
    resolve_country_key,
)

DEFAULT_DB = Path(".tirra_pipeline/pipeline.db")


def entity_id_for(key: str) -> str:
    return hashlib.sha256(f"country:{key}".encode()).hexdigest()[:16]


def build_plan(con: sqlite3.Connection) -> dict:
    """Decide, read-only, what the merge would do."""
    countries = {
        eid: name
        for eid, name in con.execute("SELECT entity_id, canonical_name FROM entities WHERE entity_type='country'")
    }
    alias_code = {
        eid: code
        for eid, code in con.execute(
            "SELECT a.entity_id, a.external_id FROM entity_aliases a "
            "JOIN entities e ON e.entity_id = a.entity_id "
            "WHERE e.entity_type='country' AND a.source='fips'"
        )
    }

    remap: dict[str, str] = {}
    regional: list[tuple[str, str]] = []
    unresolved: list[tuple[str, str]] = []

    for eid, name in countries.items():
        code = alias_code.get(eid) or name
        if is_region_code(code):
            regional.append((eid, code))
            continue
        key = resolve_country_key(code)
        if key is None:
            unresolved.append((eid, code))
            continue
        target = entity_id_for(key)
        if target != eid:
            remap[eid] = target
        # record the canonical name even when the id is already correct
        countries[eid] = countries[eid]

    # canonical names for every surviving country entity
    rename: dict[str, str] = {}
    for eid, name in countries.items():
        code = alias_code.get(eid) or name
        if is_region_code(code):
            continue
        key = resolve_country_key(code)
        if key is None:
            continue
        proper = country_name(key)
        target = remap.get(eid, eid)
        if proper:
            rename[target] = proper

    targets_missing = {t for t in set(remap.values()) if t not in countries}

    return {
        "remap": remap,
        "rename": rename,
        "regional": regional,
        "unresolved": unresolved,
        "targets_missing": targets_missing,
        "alias_code": alias_code,
        "n_countries": len(countries),
    }


def analyse_links(con: sqlite3.Connection, remap: dict[str, str]) -> dict:
    """Classify every link under the merge. Nothing is dropped unaccounted."""
    rows = list(
        con.execute(
            "SELECT link_id, entity_id_a, entity_id_b, link_type, confidence, "
            "source, created_at, metadata_json FROM entity_links"
        )
    )
    keep: dict[tuple[str, str, str], tuple] = {}
    self_links: list[int] = []
    collisions: list[int] = []

    for row in rows:
        link_id, a, b, ltype = row[0], row[1], row[2], row[3]
        na, nb = remap.get(a, a), remap.get(b, b)
        if na == nb:
            self_links.append(link_id)
            continue
        key = (na, nb, ltype)
        if key in keep:
            collisions.append(link_id)
            # keep the higher-confidence edge
            if (row[4] or 0) > (keep[key][4] or 0):
                keep[key] = (link_id, na, nb, ltype, *row[4:])
            continue
        keep[key] = (link_id, na, nb, ltype, *row[4:])

    return {
        "total": len(rows),
        "keep": keep,
        "self_links": self_links,
        "collisions": collisions,
    }


def counts(con: sqlite3.Connection) -> dict[str, int]:
    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    return {
        "entities": q("SELECT COUNT(*) FROM entities"),
        "countries": q("SELECT COUNT(*) FROM entities WHERE entity_type='country'"),
        "observations": q("SELECT COUNT(*) FROM entity_observations"),
        "links": q("SELECT COUNT(*) FROM entity_links"),
        "aliases": q("SELECT COUNT(*) FROM entity_aliases"),
        "alerts": q("SELECT COUNT(*) FROM entity_alerts"),
    }


def connectivity(con: sqlite3.Connection) -> tuple[int, int, int]:
    """(gdelt countries, reach instrument 1-hop, within 2 hops)."""
    types = dict(con.execute("SELECT entity_id, entity_type FROM entities"))
    instruments = {e for e, t in types.items() if t == "instrument"}
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in con.execute("SELECT entity_id_a, entity_id_b FROM entity_links"):
        adj[a].add(b)
        adj[b].add(a)
    srcs = set()
    for a, b in con.execute("SELECT entity_id_a, entity_id_b FROM entity_links WHERE link_type='event_involves'"):
        srcs.add(a)
        srcs.add(b)
    h1 = sum(1 for s in srcs if adj[s] & instruments)
    h2 = sum(1 for s in srcs if (adj[s] & instruments) or any(adj[n] & instruments for n in adj[s]))
    return len(srcs), h1, h2


def apply_merge(con: sqlite3.Connection, plan: dict, links: dict) -> None:
    remap, rename = plan["remap"], plan["rename"]
    cur = con.cursor()
    cur.execute("BEGIN")

    # 1. Create any target country entity that does not exist yet.
    for target in plan["targets_missing"]:
        src = next(s for s, t in remap.items() if t == target)
        created = con.execute("SELECT created_at FROM entities WHERE entity_id=?", (src,)).fetchone()
        cur.execute(
            "INSERT OR IGNORE INTO entities (entity_id, entity_type, canonical_name, created_at) "
            "VALUES (?, 'country', ?, ?)",
            (target, rename.get(target, target), created[0] if created else 0.0),
        )

    # 2. Repoint simple references.
    for src, target in remap.items():
        cur.execute("UPDATE entity_observations SET entity_id=? WHERE entity_id=?", (target, src))
        cur.execute("UPDATE entity_alerts SET entity_id=? WHERE entity_id=?", (target, src))
        cur.execute(
            "UPDATE OR IGNORE entity_aliases SET entity_id=? WHERE entity_id=?",
            (target, src),
        )

    # 3. Rebuild links from the classified set (dedup + self-link removal
    #    already decided and counted in analyse_links).
    cur.execute("DELETE FROM entity_links")
    cur.executemany(
        "INSERT INTO entity_links (entity_id_a, entity_id_b, link_type, confidence, "
        "source, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(v[1], v[2], v[3], v[4], v[5], v[6], v[7]) for v in links["keep"].values()],
    )

    # 4. Drop the now-empty source entities, then fix display names.
    for src in remap:
        cur.execute("DELETE FROM entities WHERE entity_id=?", (src,))
    for eid, proper in rename.items():
        cur.execute(
            "UPDATE entities SET canonical_name=? WHERE entity_id=? AND entity_type='country'",
            (proper, eid),
        )

    con.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--apply", action="store_true", help="actually write (requires --backup)")
    ap.add_argument("--backup", type=Path, help="path to copy the DB to before writing")
    args = ap.parse_args()

    if args.apply and not args.backup:
        ap.error("--apply requires --backup")
    if not args.db.exists():
        ap.error(f"no such database: {args.db}")

    con = sqlite3.connect(args.db)
    before = counts(con)
    plan = build_plan(con)
    links = analyse_links(con, plan["remap"])
    g0, h1_0, h2_0 = connectivity(con)

    print("=" * 66)
    print("COUNTRY ENTITY MERGE — " + ("APPLY" if args.apply else "DRY RUN"))
    print("=" * 66)
    print(f"country entities            : {plan['n_countries']}")
    print(f"  merged onto ISO alpha-2   : {len(plan['remap'])}")
    print(f"  regional blocs (untouched): {len(plan['regional'])}")
    print(f"  unresolved (untouched)    : {len(plan['unresolved'])}")
    if plan["unresolved"]:
        print(f"    {[c for _, c in plan['unresolved']]}")
    print(f"  new target entities       : {len(plan['targets_missing'])}")
    print(f"  display names corrected   : {len(plan['rename'])}")
    print()
    print(f"links total                 : {links['total']}")
    print(f"  kept                      : {len(links['keep'])}")
    print(f"  dropped as self-links     : {len(links['self_links'])}  (a==b after merge)")
    print(f"  dropped as duplicates     : {len(links['collisions'])}  (same a,b,type)")
    assert len(links["keep"]) + len(links["self_links"]) + len(links["collisions"]) == links["total"], (
        "link accounting does not balance — refusing to proceed"
    )
    print("  accounting balances       : yes")
    print()
    print(f"connectivity before         : {g0} GDELT countries, 1-hop {h1_0}, within 2 hops {h2_0}")

    if not args.apply:
        print()
        print("DRY RUN — nothing written. Re-run with --apply --backup <path>.")
        return 0

    print()
    print(f"backing up {args.db} -> {args.backup} ...")
    shutil.copy2(args.db, args.backup)
    print(f"backup size: {args.backup.stat().st_size:,} bytes")

    apply_merge(con, plan, links)

    after = counts(con)
    g1, h1_1, h2_1 = connectivity(con)
    print()
    print("--- verification " + "-" * 49)
    for k in before:
        delta = after[k] - before[k]
        print(f"{k:<14} {before[k]:>8} -> {after[k]:>8}   ({delta:+d})")
    obs_lost = before["observations"] - after["observations"]
    print()
    if obs_lost:
        print(f"!! {obs_lost} observations lost — THIS IS A BUG, restore the backup")
        return 1
    print("observations preserved exactly (INV-2): yes")
    print(
        f"connectivity after          : {g1} GDELT countries, "
        f"1-hop {h1_1}, within 2 hops {h2_1}"
        f"  ({100 * h2_1 / max(g1, 1):.0f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
