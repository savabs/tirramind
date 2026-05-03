"""Add derived CFTC positioning features to the pipeline DB.

Reads existing ``futures_positioning`` observations (stored by the CFTC tool)
and writes new ``futures_positioning_derived`` observations for each entity,
containing normalised / transformed versions of the raw positioning data.

Derived features (all per-entity time-ordered):
  cftc_mm_pct_52w_rank : float [0, 1]
      Rolling 52-week percentile rank of ``mm_net_pct_oi``.
      Tells the GNN whether current speculative positioning is historically
      crowded (near 1.0) or bare (near 0.0) — the raw value alone is not
      interpretable without this context.

  cftc_mm_direction_change : float {-1, 0, +1}
      Sign of (current mm_net_pct_oi − previous week's mm_net_pct_oi).
      +1 = speculators added net longs, -1 = reduced, 0 = unchanged.
      Provides momentum-of-positioning signal.

  cftc_oi_vs_52w_avg : float
      Current open_interest / rolling 52-week mean of open_interest.
      > 1.0 means unusually high volume; < 1.0 means thinning liquidity.

All three features are written as a single dict under the new obs type
so the GNN encoder receives them together in one snapshot.

Usage::

    python scripts/add_cftc_derived_features.py [--db-path .tirra_pipeline/pipeline.db]
    python scripts/add_cftc_derived_features.py --dry-run   # preview without writing

"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


def _rolling_percentile_rank(series: list[float], window: int) -> list[float]:
    """For each element i, compute rank among the last `window` values (inclusive).

    Returns a list the same length as `series`.  Values for which fewer than
    2 observations exist in the window are set to 0.5 (neutral).
    """
    ranks: list[float] = []
    for i, val in enumerate(series):
        start = max(0, i - window + 1)
        window_vals = [v for v in series[start : i + 1] if v == v]  # drop NaN
        if len(window_vals) < 2:
            ranks.append(0.5)
            continue
        below = sum(1 for v in window_vals if v < val)
        rank = below / (len(window_vals) - 1)
        ranks.append(rank)
    return ranks


def compute_derived_features(
    db_path: Path, dry_run: bool = False
) -> dict[str, int]:
    """Read futures_positioning obs and write derived obs.

    Returns dict of count statistics.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # ── 1. Load all futures_positioning observations ──────────────────────
    rows = conn.execute(
        "SELECT entity_id, observed_at, value_json FROM entity_observations "
        "WHERE observation_type='futures_positioning' "
        "ORDER BY entity_id, observed_at"
    ).fetchall()

    if not rows:
        log.warning("No futures_positioning observations found in DB — nothing to derive.")
        return {"derived_written": 0, "entities": 0}

    # ── 2. Group by entity, build time-ordered series ─────────────────────
    by_entity: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for r in rows:
        try:
            val = json.loads(r["value_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        by_entity[r["entity_id"]].append((float(r["observed_at"]), val))

    WEEKS_52 = 52  # rolling window in observations (weekly CFTC reports)

    # ── 3. Compute and write derived obs ─────────────────────────────────
    written = 0
    now = time.time()

    # Prepare insert if not dry_run
    if not dry_run:
        # Avoid re-inserting observations we already derived
        existing_derived: set[tuple[str, float]] = set(
            (r[0], float(r[1]))
            for r in conn.execute(
                "SELECT entity_id, observed_at FROM entity_observations "
                "WHERE observation_type='futures_positioning_derived'"
            ).fetchall()
        )
    else:
        existing_derived = set()

    for entity_id, ts_vals in by_entity.items():
        # Extract aligned time series
        timestamps = [ts for ts, _ in ts_vals]
        mm_pct_series: list[float] = []
        oi_series: list[float] = []

        for _, val in ts_vals:
            mm_pct = val.get("mm_net_pct_oi")
            oi = val.get("open_interest")
            mm_pct_series.append(float(mm_pct) if mm_pct is not None else float("nan"))
            oi_series.append(float(oi) if oi is not None else float("nan"))

        # Compute rolling 52-week percentile rank of mm_net_pct_oi
        valid_mm = [v if v == v else 0.0 for v in mm_pct_series]  # replace NaN with 0
        pct_ranks = _rolling_percentile_rank(valid_mm, WEEKS_52)

        for i, (ts, val) in enumerate(ts_vals):
            if (entity_id, ts) in existing_derived:
                continue  # already written in a previous run

            mm_pct = mm_pct_series[i]
            oi = oi_series[i]
            prev_mm = mm_pct_series[i - 1] if i > 0 else mm_pct

            # cftc_mm_direction_change: sign of week-over-week change
            if mm_pct == mm_pct and prev_mm == prev_mm:  # both finite
                delta = mm_pct - prev_mm
                direction = 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)
            else:
                direction = 0.0

            # cftc_oi_vs_52w_avg: current OI / rolling mean OI
            start = max(0, i - WEEKS_52 + 1)
            window_oi = [v for v in oi_series[start : i + 1] if v == v]
            if window_oi and oi == oi:
                oi_vs_avg = oi / (sum(window_oi) / len(window_oi))
            else:
                oi_vs_avg = 1.0  # neutral default

            derived_value = {
                "cftc_mm_pct_52w_rank": round(pct_ranks[i], 4),
                "cftc_mm_direction_change": direction,
                "cftc_oi_vs_52w_avg": round(oi_vs_avg, 4),
                # Pass through the raw mm_net_pct_oi for reference
                "mm_net_pct_oi_raw": mm_pct if mm_pct == mm_pct else None,
            }

            if not dry_run:
                conn.execute(
                    "INSERT INTO entity_observations "
                    "(entity_id, source_tool, observed_at, ingested_at, "
                    " observation_type, depth_level, value_json, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entity_id,
                        "cftc_derived",
                        ts,
                        now,
                        "futures_positioning_derived",
                        2,
                        json.dumps(derived_value),
                        json.dumps({"source": "add_cftc_derived_features.py"}),
                    ),
                )
            written += 1

        if not dry_run:
            conn.commit()

    conn.close()
    return {"derived_written": written, "entities": len(by_entity)}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Compute and store derived CFTC positioning features."
    )
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
    stats = compute_derived_features(db_path, dry_run=args.dry_run)

    action = "Would write" if args.dry_run else "Wrote"
    log.info(
        "%s %d futures_positioning_derived observations across %d entities.",
        action,
        stats["derived_written"],
        stats["entities"],
    )
    if args.dry_run:
        log.info("Dry run complete — no changes made.")


if __name__ == "__main__":
    main()
