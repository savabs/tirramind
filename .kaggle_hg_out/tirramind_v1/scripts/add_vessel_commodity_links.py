"""Add carries_commodity links from vessel entities to instrument entities.

Reads vessel entities from the pipeline DB and maps each vessel's ``ship_type``
field (stored in entity metadata by the AIS tool) to commodity instrument tickers
using a hand-coded AIS-ship-type → commodity mapping.

Mapping rationale (L0 physical causality — ships cannot fake cargo):
  tanker      → crude oil (CL=F, BZ=F), refined products (RB=F, HO=F)
  cargo/bulker → grains (ZC=F, ZW=F, ZS=F), fertilisers (UAN), coal
  passenger   → (no commodity link — passenger travel only)
  fishing     → (not tracked in instrument universe)
  tug         → (no commodity link — service vessel)

The Baltic region AIS data skews toward tankers and bulk cargo because
the Gulf of Finland / Danish Straits are the primary route for Russian
crude oil and grain exports.

Idempotent: uses INSERT OR IGNORE semantics via PipelineStore.link_entities.

Usage::

    python scripts/add_vessel_commodity_links.py [--db-path .tirra_pipeline/pipeline.db]
    python scripts/add_vessel_commodity_links.py --dry-run   # preview without writing

"""

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ── Ship type → commodity tickers mapping ─────────────────────────────────────
# Source: AIS ship type taxonomy (ITU-R M.1371 Table 50) cross-referenced with
# Baltic/Northern European trade route commodity flows.
# Confidence: 0.6 for tankers (crude is dominant but not exclusive cargo);
#             0.5 for cargo/bulk (diverse cargo, grain is most common Baltic bulk).
SHIP_TYPE_TO_COMMODITIES: dict[str, list[tuple[str, float]]] = {
    # AIS type code 80-89 (tankers): primarily crude and refined petroleum
    "tanker": [
        ("CL=F", 0.7),  # WTI Crude — most common Baltic tanker cargo
        ("BZ=F", 0.7),  # Brent Crude — European benchmark
        ("RB=F", 0.5),  # RBOB Gasoline (refined product tankers)
        ("HO=F", 0.5),  # Heating Oil (refined product tankers)
    ],
    # AIS type code 70-79 (cargo): bulk dry commodities on Baltic routes
    "cargo": [
        ("ZC=F", 0.55),  # Corn — major Baltic bulk export
        ("ZW=F", 0.55),  # Wheat — major Baltic bulk export (Russia/Ukraine)
        ("ZS=F", 0.50),  # Soybeans — Baltic cargo flow
    ],
}


def add_vessel_commodity_links(db_path: Path, dry_run: bool = False) -> dict[str, int]:
    """Read vessel entities, create carries_commodity links.

    Returns count statistics.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # ── Load vessel entities ──────────────────────────────────────────────
    vessel_rows = conn.execute(
        "SELECT entity_id, canonical_name, metadata_json FROM entities "
        "WHERE entity_type='vessel'"
    ).fetchall()

    if not vessel_rows:
        log.warning("No vessel entities found in DB.")
        return {"vessels_processed": 0, "links_created": 0, "links_skipped": 0}

    log.info("Found %d vessel entities.", len(vessel_rows))

    # ── Load instrument entity IDs (ticker → entity_id) ───────────────────
    inst_rows = conn.execute(
        "SELECT entity_id, canonical_name, metadata_json FROM entities "
        "WHERE entity_type='tradeable_instrument'"
    ).fetchall()

    # Build ticker → entity_id map from entity aliases or canonical names
    # The AIS instrument universe uses _entity_id(ticker) = sha256-based ID,
    # so we find them via the aliases table.
    ticker_to_eid: dict[str, str] = {}
    alias_rows = conn.execute(
        "SELECT entity_id, alias_type, alias_value FROM entity_aliases "
        "WHERE alias_type='ticker'"
    ).fetchall()
    for ar in alias_rows:
        ticker_to_eid[ar["alias_value"]] = ar["entity_id"]

    # Fallback: if no ticker aliases, try to match via canonical_name
    if not ticker_to_eid:
        for ir in inst_rows:
            meta = {}
            try:
                meta = json.loads(ir["metadata_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            ticker = meta.get("ticker") or ir["canonical_name"]
            if ticker:
                ticker_to_eid[ticker] = ir["entity_id"]

    log.info("Resolved %d ticker → entity_id mappings.", len(ticker_to_eid))

    # ── Process vessels ───────────────────────────────────────────────────
    links_created = 0
    links_skipped = 0
    vessels_processed = 0

    for vr in vessel_rows:
        meta: dict = {}
        try:
            meta = json.loads(vr["metadata_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

        ship_type = meta.get("ship_type", "").lower()
        commodity_targets = SHIP_TYPE_TO_COMMODITIES.get(ship_type)
        if not commodity_targets:
            continue  # passenger, tug, fishing — no commodity mapping

        vessels_processed += 1
        vessel_eid = vr["entity_id"]

        for ticker, confidence in commodity_targets:
            inst_eid = ticker_to_eid.get(ticker)
            if not inst_eid:
                log.debug("Ticker %s not in DB — skipping link.", ticker)
                links_skipped += 1
                continue

            if dry_run:
                log.info(
                    "  [DRY RUN] carries_commodity: vessel %s → %s (conf=%.2f)",
                    vr["canonical_name"],
                    ticker,
                    confidence,
                )
                links_created += 1
                continue

            # Idempotent insert via INSERT OR IGNORE
            result = conn.execute(
                "INSERT OR IGNORE INTO entity_links "
                "(entity_id_a, entity_id_b, link_type, source, confidence, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, strftime('%s', 'now'))",
                (
                    vessel_eid,
                    inst_eid,
                    "carries_commodity",
                    "add_vessel_commodity_links.py",
                    confidence,
                    json.dumps(
                        {
                            "ship_type": ship_type,
                            "ticker": ticker,
                            "basis": "ais_ship_type_to_commodity_mapping",
                        }
                    ),
                ),
            )
            if result.rowcount > 0:
                links_created += 1
            else:
                links_skipped += 1  # already existed

    if not dry_run:
        conn.commit()
    conn.close()

    return {
        "vessels_processed": vessels_processed,
        "links_created": links_created,
        "links_skipped": links_skipped,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Create carries_commodity links from vessel entities to instrument entities."
    )
    parser.add_argument(
        "--db-path",
        default=".tirra_pipeline/pipeline.db",
        help="Path to PipelineStore SQLite DB (default: .tirra_pipeline/pipeline.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned links without writing to DB.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        sys.exit(1)

    log.info("Scanning vessel entities in %s ...", db_path)
    stats = add_vessel_commodity_links(db_path, dry_run=args.dry_run)

    action = "Would create" if args.dry_run else "Created"
    log.info(
        "%s %d carries_commodity links (%d vessels processed, %d skipped/existing).",
        action,
        stats["links_created"],
        stats["vessels_processed"],
        stats["links_skipped"],
    )
    if args.dry_run:
        log.info("Dry run complete — no changes made.")


if __name__ == "__main__":
    main()
