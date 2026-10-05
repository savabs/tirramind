"""Thin CLI over ``agent.quant.cftc_features`` — derive CFTC positioning features.

The math no longer lives here.  It lives in ``agent/quant/cftc_features.py``
(Layer 2) and runs on a schedule as the ``derive_cftc_features`` node of the
``daily_collection`` DAG.  This file remains only as a manual entry point for
a one-off run or a dry-run preview.

Why it moved, and why this file no longer touches SQLite directly:

* ``scripts/`` is not an importable package, so no DAG could ever call the
  derivation that lived here.  It ran once, by hand, on 2026-05-03 — then
  ``cftc_derived`` sat frozen at ``observed_at`` 2026-04-14 for 162 days while
  the raw ``cftc`` source kept collecting (audit P6.4).
* The old code opened the live database *writable* and issued a raw
  ``INSERT INTO entity_observations``, bypassing
  ``PipelineStore.store_entity_observation`` — the single write boundary — and
  with it the ``observed_at`` range guard and the idempotency collapse.  It
  also counted the live table's 442 byte-identical duplicate raw rows as
  separate weeks, which skews every rolling window they sit in.  Inserts now
  go through the store, and the derivation enforces one derived row per
  ``(entity_id, observed_at)`` — superseding a stale value rather than
  appending a second, contradictory row beside it.

Usage::

    python scripts/add_cftc_derived_features.py [--db-path .tirra_pipeline/pipeline.db]
    python scripts/add_cftc_derived_features.py --dry-run   # preview, writes nothing

Prefer letting the DAG run it.  Running this by hand against the live database
while the orchestrator holds the writer adds a second writer to one SQLite
file; the scheduled node derives exactly the same rows.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline.store import PipelineStore  # noqa: E402
from agent.quant.cftc_features import (  # noqa: E402
    coverage_stats,
    derive_and_store,
    derive_series,
    load_raw_positioning,
)

log = logging.getLogger(__name__)


def preview(store: PipelineStore) -> dict[str, Any]:
    """Compute what a real run would write, without writing anything."""
    raw = load_raw_positioning(store)
    rows = sum(len(derive_series(obs)) for obs in raw.by_entity.values())
    return {
        "rows_derived": rows,
        "entities": len(raw.by_entity),
        "raw_duplicates_dropped": raw.duplicates_dropped,
        "raw_conflicts_collapsed": raw.conflicts_collapsed,
        "coverage": coverage_stats(raw.by_entity),
    }


def _log_coverage(coverage: dict[str, Any]) -> None:
    """Say how much of the recent COT history the *input* actually holds.

    Row counts alone read as success even when the derivation only had three
    of the last twenty-three weekly reports to work with — the derivation
    cannot create a week ``fetch_cftc`` never fetched.
    """
    log.info(
        "Input coverage: %d of the last %d weekly COT report dates present (%.0f%%).",
        coverage["report_dates_present"],
        coverage["report_dates_expected"],
        coverage["ratio"] * 100,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Compute and store derived CFTC positioning features.")
    parser.add_argument(
        "--db-path",
        default=".tirra_pipeline/pipeline.db",
        help="Path to PipelineStore SQLite DB (default: .tirra_pipeline/pipeline.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print statistics without writing to DB.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        sys.exit(1)

    log.info("Reading futures_positioning observations from %s ...", db_path)
    store = PipelineStore(str(db_path))
    try:
        if args.dry_run:
            stats = preview(store)
            log.info(
                "Would derive %d futures_positioning_derived observations across %d entities "
                "(%d duplicate raw rows dropped, %d conflicting raw rows collapsed).",
                stats["rows_derived"],
                stats["entities"],
                stats["raw_duplicates_dropped"],
                stats["raw_conflicts_collapsed"],
            )
            _log_coverage(stats["coverage"])
            log.info("Dry run complete — no changes made.")
            return

        result = derive_and_store(store)
        log.info(
            "Inserted %d, superseded %d, left %d unchanged (%d derived) across %d entities. max observed_at=%s",
            result["rows_written"],
            result["rows_superseded"],
            result["rows_unchanged"],
            result["rows_derived"],
            result["entities"],
            result["max_observed_at"],
        )
        _log_coverage(result["coverage"])
    finally:
        store.close()


if __name__ == "__main__":
    main()
