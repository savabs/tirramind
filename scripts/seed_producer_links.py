#!/usr/bin/env python3
"""Seed Producer-Country Links for Commodity Futures

Creates `produced_in` entity_links from commodity instrument entities to
their top-3 producer country entities. This enables GDELT geopolitical events
(country-level) to propagate through the GNN to commodity prices.

Without these links:
  CL=F (WTI crude) → exchange: US only (NYMEX)
  ZC=F (corn) → exchange: US only (CBOT)
  ZS=F (soybeans) → exchange: US only (CBOT)

With these links (what we want):
  CL=F → produced_in → [US, SA, RU, IQ, AE, ...]
  ZC=F → produced_in → [US, BR, AR]
  ZS=F → produced_in → [US, BR, AR]
  ZW=F → produced_in → [US, RU, UA, AU, CA]
  GC=F → produced_in → [US, AU, CN, ZA, CA, RU]
  HG=F → produced_in → [CL, PE, CN, US]

So when Ukraine invasion GDELT events fire on UA/RU country nodes,
the GNN signal propagates to ZW=F via the produced_in edge.

Sources verified against:
  - USDA World Agricultural Supply and Demand Estimates (wheat, corn, soybeans)
  - IEA Oil Market Report (crude oil)
  - World Gold Council (gold)
  - US Geological Survey Minerals Yearbook (copper, silver, platinum, palladium)
  - ICO (coffee), ICCO (cocoa), ISO (sugar), Cotton Council International
  - US EIA (natural gas, crude oil)

Usage:
    python scripts/seed_producer_links.py [--db-path PATH]
    python scripts/seed_producer_links.py --dry-run  # show what would be created
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = Path(".tirra_pipeline/pipeline.db")

# Producer country mappings: ticker → [ISO2, ...]
# Listed in rough production-share order (most important first).
# Each ISO2 maps to a country entity that the GNN uses.
# Confidence 0.95 for top producers, 0.80 for significant but secondary.
PRODUCER_MAP: dict[str, list[tuple[str, float]]] = {
    # Energy
    "CL=F": [  # WTI Crude Oil — NYMEX
        ("US", 0.95), ("SA", 0.95), ("RU", 0.95),
        ("IQ", 0.90), ("AE", 0.90), ("CA", 0.85),
        ("KW", 0.85), ("IR", 0.80), ("NG", 0.80),
        ("KZ", 0.80), ("LY", 0.80), ("VE", 0.80),
    ],
    "BZ=F": [  # Brent Crude — ICE
        ("SA", 0.95), ("RU", 0.95), ("IQ", 0.90),
        ("AE", 0.90), ("KW", 0.85), ("NG", 0.85),
        ("LY", 0.80), ("NO", 0.80),
    ],
    "NG=F": [  # Natural Gas — NYMEX
        ("US", 0.95), ("RU", 0.95), ("IR", 0.90),
        ("QA", 0.90), ("AU", 0.85), ("CA", 0.85),
        ("DZ", 0.80),
    ],
    "RB=F": [  # RBOB Gasoline — NYMEX (refined product, refinery-country matters)
        ("US", 0.95), ("SA", 0.90), ("RU", 0.85),
    ],
    # Metals
    "GC=F": [  # Gold — COMEX
        ("US", 0.90), ("AU", 0.95), ("CN", 0.95),
        ("ZA", 0.90), ("CA", 0.85), ("RU", 0.85),
        ("PE", 0.80), ("ID", 0.80),
    ],
    "SI=F": [  # Silver — COMEX
        ("MX", 0.95), ("PE", 0.95), ("CN", 0.90),
        ("RU", 0.85), ("CL", 0.85), ("AU", 0.80),
        ("US", 0.80), ("PL", 0.80),
    ],
    "HG=F": [  # Copper — COMEX
        ("CL", 0.95), ("PE", 0.95), ("CN", 0.90),
        ("US", 0.80), ("AU", 0.80), ("ZM", 0.80),
    ],
    "PL=F": [  # Platinum — NYMEX
        ("ZA", 0.95), ("RU", 0.90), ("ZW", 0.80),
        ("CA", 0.75), ("US", 0.75),
    ],
    "PA=F": [  # Palladium — NYMEX
        ("RU", 0.95), ("ZA", 0.95),
        ("US", 0.80), ("CA", 0.75),
    ],
    # Agriculture — Grains
    "ZC=F": [  # Corn — CBOT
        ("US", 0.95), ("BR", 0.95), ("AR", 0.90),
        ("UA", 0.90), ("CN", 0.85), ("MX", 0.75),
    ],
    "ZS=F": [  # Soybeans — CBOT
        ("US", 0.95), ("BR", 0.95), ("AR", 0.95),
        ("CN", 0.80), ("PY", 0.75), ("CA", 0.70),
    ],
    "ZW=F": [  # Wheat SRW — CBOT
        ("US", 0.95), ("RU", 0.95), ("UA", 0.95),
        ("AU", 0.90), ("CA", 0.90), ("FR", 0.85),
        ("AR", 0.85), ("KZ", 0.80),
    ],
    # Agriculture — Softs
    "KC=F": [  # Coffee C — ICE
        ("BR", 0.95), ("VN", 0.95), ("CO", 0.90),
        ("ID", 0.85), ("ET", 0.80), ("HN", 0.80),
    ],
    "CC=F": [  # Cocoa — ICE
        ("CI", 0.95), ("GH", 0.95), ("ID", 0.85),
        ("CM", 0.80), ("NG", 0.75), ("EC", 0.75),
    ],
    "SB=F": [  # Sugar #11 — ICE
        ("BR", 0.95), ("IN", 0.90), ("EU", 0.85),
        ("TH", 0.80), ("AU", 0.75), ("CN", 0.75),
    ],
    "CT=F": [  # Cotton #2 — ICE
        ("CN", 0.95), ("IN", 0.95), ("US", 0.90),
        ("BR", 0.85), ("PK", 0.80), ("AU", 0.80),
        ("UZ", 0.75),
    ],
    "OJ=F": [  # Frozen OJ — ICE
        ("BR", 0.95), ("US", 0.85), ("MX", 0.75),
    ],
    # Livestock
    "LE=F": [  # Live Cattle — CME
        ("US", 0.95), ("BR", 0.90), ("AU", 0.85),
        ("AR", 0.85), ("MX", 0.75),
    ],
    "HE=F": [  # Lean Hogs — CME
        ("US", 0.95), ("CN", 0.95), ("EU", 0.85),
        ("BR", 0.80), ("CA", 0.75),
    ],
}


class LinkSpec(NamedTuple):
    instrument_ticker: str
    country_iso: str
    confidence: float


def get_existing_links(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Return set of (entity_id_a, entity_id_b) for existing produced_in links."""
    rows = conn.execute(
        "SELECT entity_id_a, entity_id_b FROM entity_links WHERE link_type='produced_in'"
    ).fetchall()
    return {(a, b) for a, b in rows}


def seed_links(db_path: Path, dry_run: bool = False) -> dict:
    from agent.pipeline.entity import entity_id_from_key
    from agent.tools.instrument_universe import tradeable_instruments, _entity_id

    conn = sqlite3.connect(str(db_path))

    # Build ticker → entity_id via instrument_universe name mapping
    from agent.tools.instrument_universe import tradeable_instruments
    name_to_ticker = {inst.name: inst.ticker for inst in tradeable_instruments()}

    inst_rows = conn.execute(
        "SELECT entity_id, canonical_name FROM entities WHERE entity_type='instrument'"
    ).fetchall()
    # Map: ticker → entity_id (via canonical_name=full name)
    ticker_to_eid = {name_to_ticker[name]: eid for eid, name in inst_rows if name in name_to_ticker}

    # Get all existing country entity_ids
    country_rows = conn.execute(
        "SELECT entity_id, canonical_name FROM entities WHERE entity_type='country'"
    ).fetchall()
    iso_to_eid = {name: eid for eid, name in country_rows}

    existing_links = get_existing_links(conn)

    counts = {
        "links_created": 0,
        "links_skipped_existing": 0,
        "links_skipped_no_instrument": 0,
        "links_skipped_no_country": 0,
        "countries_registered": 0,
    }

    link_rows: list[tuple] = []

    import time
    now = time.time()

    for ticker, producers in PRODUCER_MAP.items():
        inst_eid = ticker_to_eid.get(ticker)
        if inst_eid is None:
            print(f"  SKIP {ticker}: not in DB instrument entities")
            counts["links_skipped_no_instrument"] += len(producers)
            continue

        for iso, confidence in producers:
            country_eid = iso_to_eid.get(iso)
            if country_eid is None:
                # Register the country entity — deterministic entity_id
                country_eid = entity_id_from_key("country", iso)
                if not dry_run:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO entities (entity_id, entity_type, canonical_name, created_at) "
                            "VALUES (?, 'country', ?, ?)",
                            (country_eid, iso, now),
                        )
                        conn.commit()
                    except Exception as exc:
                        print(f"  WARN: Could not register country {iso}: {exc}")
                        counts["links_skipped_no_country"] += 1
                        continue
                iso_to_eid[iso] = country_eid
                counts["countries_registered"] += 1
                print(f"  + Registered country: {iso} ({country_eid})")

            link_key = (inst_eid, country_eid)
            if link_key in existing_links:
                counts["links_skipped_existing"] += 1
                continue

            link_rows.append((inst_eid, country_eid, "produced_in", confidence, "seed_producer_links", now,
                              f'{{"ticker": "{ticker}", "iso": "{iso}"}}'))
            existing_links.add(link_key)
            counts["links_created"] += 1

    if not dry_run and link_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO entity_links "
            "(entity_id_a, entity_id_b, link_type, confidence, source, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            link_rows,
        )
        conn.commit()

    conn.close()
    return counts


def verify_links(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT inst.canonical_name, c.canonical_name, el.confidence
        FROM entity_links el
        JOIN entities inst ON inst.entity_id = el.entity_id_a
        JOIN entities c ON c.entity_id = el.entity_id_b
        WHERE el.link_type = 'produced_in'
        ORDER BY inst.canonical_name, el.confidence DESC
    """).fetchall()
    conn.close()

    from collections import defaultdict
    by_ticker = defaultdict(list)
    for ticker, iso, conf in rows:
        by_ticker[ticker].append(f"{iso}({conf:.0%})")

    print(f"\n  produced_in links in DB: {len(rows)}")
    for ticker in sorted(by_ticker):
        print(f"  {ticker:<12} → {', '.join(by_ticker[ticker])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed commodity producer-country links")
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created, no writes")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    print("Seed Commodity Producer-Country Links")
    print("=" * 50)
    print(f"  DB: {db_path}")
    if args.dry_run:
        print("  [DRY RUN — no writes]")
    print()

    counts = seed_links(db_path, dry_run=args.dry_run)

    print()
    print("  Results:")
    for k, v in counts.items():
        print(f"    {k}: {v}")

    if not args.dry_run:
        verify_links(db_path)
        print()
        links_total = counts["links_created"]
        if links_total >= 30:
            print(f"  ✓ SUCCESS: {links_total} produced_in links created")
            print("  GDELT events on commodity-producing countries will now propagate")
            print("  through the GNN to the corresponding futures instruments.")
        else:
            print(f"  ⚠ Only {links_total} links created. Check PRODUCER_MAP coverage.")

    print()
    print("  Next: run 'python scripts/phase40_gnn_backtest.py' to measure IC impact")
    print("        and 'python scripts/source_ablation.py --sources gdelt' to check attribution")


if __name__ == "__main__":
    main()
