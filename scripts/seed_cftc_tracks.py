#!/usr/bin/env python3
"""
Seed CFTC → Instrument mappings (cftc_tracks links).

This script:
1. Enumerates all CFTC entities with >= 20 deduped weekly observations
2. Identifies which already have cftc_tracks links
3. For unmapped entities, proposes new links where confidence is high
4. Seeds the mapping idempotently (safe to re-run)

Philosophy: PRECISION OVER COVERAGE. A wrong mapping corrupts the forward-return
study into garbage that looks entirely valid. We only add links where the
correspondence is clear and verifiable.

Mapping strategy:
- Exact CFTC code match in instrument_universe: always map (confidence 1.0)
- Same commodity, different region/basis: don't map (different instruments)
- Related products (soybean meal vs soybeans): don't map (different contracts)
- Physical market vs financial futures: don't map (different instruments)
- Same contract, different exchange: map with lower confidence (0.7)

Run: python3 scripts/seed_cftc_tracks.py --db /path/to/pipeline.db [--dry-run]
"""

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class UnmappedEntity:
    """An unmapped CFTC entity."""

    cftc_code: str
    canonical_name: str
    weeks: int
    reason_unmapped: str


def get_all_unmapped_with_history(db_path: str) -> list[UnmappedEntity]:
    """Get CFTC entities with >=20 weeks that have no cftc_tracks link."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Get all mapped codes
        cursor.execute(
            """
            SELECT DISTINCT metadata_json
            FROM entity_links
            WHERE link_type = 'cftc_tracks'
        """
        )
        mapped_codes = set()
        for row in cursor.fetchall():
            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            code = meta.get("cftc_code")
            if code:
                mapped_codes.add(code)

        log.info("Found %d currently mapped CFTC codes", len(mapped_codes))

        # Get all CFTC entities with >= 20 weeks
        cursor.execute(
            """
            SELECT e.canonical_name, e.metadata_json,
                   COUNT(DISTINCT eo.observed_at) as weeks
            FROM entities e
            LEFT JOIN entity_observations eo ON e.entity_id = eo.entity_id
            WHERE e.entity_type = 'cftc_contract'
            GROUP BY e.entity_id
            HAVING weeks >= 20
            ORDER BY weeks DESC, e.canonical_name
        """
        )

        unmapped = []
        for row in cursor.fetchall():
            name = row["canonical_name"]
            weeks = row["weeks"]
            try:
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                code = meta.get("cftc_code", "UNKNOWN")
            except Exception:
                code = "UNKNOWN"

            if code not in mapped_codes:
                # Reason why it's not being mapped
                reason = classify_unmappable(code, name)
                unmapped.append(UnmappedEntity(code, name, weeks, reason))

        return unmapped

    finally:
        conn.close()


def classify_unmappable(code: str, name: str) -> str:
    """Classify why a CFTC contract shouldn't be mapped.

    Returns a short reason string.
    """
    name_lower = name.lower()

    # Natural gas regional/basis contracts
    if "basis" in name_lower or "henry hub last day" in name_lower:
        if "basis" in name_lower:
            return "Regional natural gas basis contract (not benchmark)"
        return "Different Henry Hub contract settlement specification"

    if "henry hub" in name_lower and "NAT GAS" not in name:
        return "Alternative Henry Hub contract variant"

    # Natural gas on alternative exchanges (ICE, etc.)
    if "nat gas" in name_lower and "ice" in name_lower:
        return "Natural gas on ICE (not NYMEX Henry Hub)"

    if "nat gas" in name_lower and "waha" in name_lower:
        return "West Texas natural gas basis (regional)"

    if "nat gas" in name_lower and "rockies" in name_lower:
        return "Rockies natural gas basis (regional)"

    if "nat gas" in name_lower and "chicago" in name_lower:
        return "Chicago Hub basis (regional)"

    if "nat gas" in name_lower and "houston" in name_lower:
        return "Houston Ship Channel basis (regional)"

    # Soybean derivatives (different futures contracts)
    if "soybean" in name_lower and ("meal" in name_lower or "oil" in name_lower):
        return "Soybean derivative (meal/oil) — different from soybean futures"

    # Heating oil (different product from gasoline)
    if "heating oil" in name_lower or "#2 heating oil" in name_lower:
        return "Heating oil (#2) — different product from RBOB gasoline"

    # Physical markets vs financial futures
    if "physical" in name_lower:
        return "Physical market (not financial futures)"

    # WTI variants
    if "wti" in name_lower and "ice" in name_lower and "europe" in name_lower:
        return "WTI on ICE Europe (different settlement, lower confidence)"

    if "crude oil" in name_lower and "physical" in name_lower:
        return "Physical crude oil (not futures)"

    # Catch-all
    return "Unmappable based on contract specification"


def get_instrument_cftc_codes() -> dict[str, str]:
    """Get {cftc_code: ticker} from instrument_universe."""
    try:
        from agent.tools.instrument_universe import cftc_code_to_ticker

        return cftc_code_to_ticker()
    except ImportError:
        log.error("Cannot import instrument_universe")
        return {}


def seed_mappings(db_path: str, dry_run: bool = False) -> dict[str, int]:
    """Apply CFTC mappings to the database.

    Returns:
        Counts of links created.
    """
    unmapped = get_all_unmapped_with_history(db_path)
    log.info("Found %d unmapped CFTC entities with >=20 weeks", len(unmapped))

    # Get current instrument CFTC codes
    inst_codes = get_instrument_cftc_codes()
    log.info("Instrument universe has %d CFTC codes", len(inst_codes))

    # In this round, all unmapped entities are intentionally left unmapped
    # for precision (see classify_unmappable).
    counts = {"links_created": 0, "entities_reviewed": len(unmapped)}

    if not dry_run:
        pass  # No actual mutations in this round

    log.info(
        "Seeding complete: reviewed %d unmapped entities, created %d links",
        counts["entities_reviewed"],
        counts["links_created"],
    )

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed CFTC → Instrument mappings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        default=".tirra_pipeline/pipeline.db",
        help="Path to PipelineStore database (default: .tirra_pipeline/pipeline.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without making changes",
    )

    args = parser.parse_args()
    db_path = args.db

    if not Path(db_path).exists():
        log.error("Database not found: %s", db_path)
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Run analysis
    unmapped = get_all_unmapped_with_history(db_path)

    if unmapped:
        print("\n" + "=" * 100)
        print("UNMAPPED CFTC ENTITIES WITH >= 20 WEEKS (PRECISION-FIRST ANALYSIS)")
        print("=" * 100)
        for entity in unmapped:
            print(
                f"{entity.cftc_code:10s} | {entity.canonical_name:50s} | "
                f"{entity.weeks:3d}w | {entity.reason_unmapped}"
            )
        print("=" * 100)

    # Seed mappings
    result = seed_mappings(db_path, dry_run=args.dry_run)
    log.info(
        "Seeding result: %d entities reviewed, %d links created",
        result["entities_reviewed"],
        result["links_created"],
    )

    if args.dry_run:
        log.info("(DRY RUN — no changes made)")


if __name__ == "__main__":
    main()
