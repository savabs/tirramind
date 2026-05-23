#!/usr/bin/env python3
"""Phase 47 Density Audit — mandatory exit gate before Phase 40.

Queries entity_observations and reports coverage by entity_type and source_tool.
Flags sparse types (< min_obs observations OR < min_days temporal span OR
< min_entities distinct entities).

Exit code 0 = all types pass (Phase 40 ready).
Exit code 1 = one or more sparse types (see output for recommendations).

Usage:
    python scripts/density_audit.py [--db-path PATH]
    python scripts/density_audit.py --min-obs 500 --min-days 365
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_DB = ".tirra_pipeline/pipeline.db"
_DEFAULT_MIN_OBS = 100
_DEFAULT_MIN_DAYS = 180
_DEFAULT_MIN_ENTITIES = 5


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------


def _query_by_entity_type(conn: sqlite3.Connection) -> list[dict]:
    """Return per entity_type aggregates joined from entity_observations + entities."""
    sql = """
        SELECT
            e.entity_type,
            COUNT(DISTINCT e.entity_id)         AS entity_count,
            COUNT(eo.id)                        AS obs_count,
            MIN(eo.observed_at)                 AS earliest_ts,
            MAX(eo.observed_at)                 AS latest_ts
        FROM entity_observations eo
        JOIN entities e ON eo.entity_id = e.entity_id
        GROUP BY e.entity_type
        ORDER BY obs_count DESC
    """
    rows = conn.execute(sql).fetchall()
    results = []
    for row in rows:
        entity_type, entity_count, obs_count, earliest_ts, latest_ts = row
        span_days = 0.0
        if (
            earliest_ts is not None
            and latest_ts is not None
            and latest_ts > earliest_ts
        ):
            span_days = (latest_ts - earliest_ts) / 86400.0
        obs_per_entity = round(obs_count / entity_count, 1) if entity_count else 0.0
        results.append(
            {
                "entity_type": entity_type,
                "entity_count": entity_count,
                "obs_count": obs_count,
                "earliest_ts": earliest_ts,
                "latest_ts": latest_ts,
                "span_days": span_days,
                "obs_per_entity": obs_per_entity,
            }
        )
    return results


def _query_by_source_tool(conn: sqlite3.Connection) -> list[dict]:
    """Return per source_tool observation counts."""
    sql = """
        SELECT source_tool, COUNT(*) AS obs_count
        FROM entity_observations
        GROUP BY source_tool
        ORDER BY obs_count DESC
    """
    rows = conn.execute(sql).fetchall()
    return [{"source_tool": r[0], "obs_count": r[1]} for r in rows]


def _query_total(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (total_obs, total_entities)."""
    total_obs = conn.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0]
    total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    return total_obs, total_entities


# ---------------------------------------------------------------------------
# Sparse classification
# ---------------------------------------------------------------------------


def _is_sparse(
    row: dict,
    min_obs: int,
    min_days: int,
    min_entities: int,
) -> tuple[bool, list[str]]:
    """Return (is_sparse, list_of_reasons)."""
    reasons: list[str] = []
    if row["obs_count"] < min_obs:
        reasons.append(f"obs={row['obs_count']} < {min_obs}")
    if row["span_days"] < min_days:
        reasons.append(f"span={row['span_days']:.0f}d < {min_days}d")
    if row["entity_count"] < min_entities:
        reasons.append(f"entities={row['entity_count']} < {min_entities}")
    return bool(reasons), reasons


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


# ---------------------------------------------------------------------------
# Recommendation heuristics
# ---------------------------------------------------------------------------


def _recommend(row: dict, reasons: list[str]) -> str:
    et = row["entity_type"]
    live_only_types = {"domain", "dns", "cert", "ip_address", "ip", "hostname"}
    if any(tok in et.lower() for tok in live_only_types):
        return "Live-only source. Accept with justification or skip in BACKFILL_PLAN."
    if row["obs_count"] < 100 and row["span_days"] < 7:
        return "No backfill reached this type. Run: python scripts/backfill.py --tool <label>"
    if row["span_days"] < _DEFAULT_MIN_DAYS:
        return (
            f"Temporal span too short ({row['span_days']:.0f}d). "
            "Extend days_back or loop backfill with more history."
        )
    return f"Add more observations for {et} via targeted backfill runs."


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _fmt_date(ts: float | None) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _run_audit(
    db_path: str,
    min_obs: int = _DEFAULT_MIN_OBS,
    min_days: int = _DEFAULT_MIN_DAYS,
    min_entities: int = _DEFAULT_MIN_ENTITIES,
) -> int:
    """Run the full density audit. Returns exit code (0=pass, 1=fail)."""
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.OperationalError as exc:
        print(f"ERROR: Cannot open database: {exc}", file=sys.stderr)
        return 1

    try:
        total_obs, total_entities = _query_total(conn)
        by_type = _query_by_entity_type(conn)
        by_tool = _query_by_source_tool(conn)
    except sqlite3.OperationalError as exc:
        conn.close()
        if "no such table" in str(exc):
            print(
                f"ERROR: Database has no entity_observations table: {exc}",
                file=sys.stderr,
            )
            print(
                "Run the pipeline at least once or run: python scripts/backfill.py",
                file=sys.stderr,
            )
            return 1
        raise
    finally:
        conn.close()

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    width = 80

    print("=" * width)
    print(f"Phase 47 Density Audit — {db_path} — {today_str}")
    print("=" * width)
    print(f"Total observations: {total_obs:,}   Total entities: {total_entities:,}")
    print()

    # ── By entity_type ──────────────────────────────────────────────────────
    sparse_rows: list[tuple[dict, list[str]]] = []

    if not by_type:
        print("No entity_observations found.")
        print()
    else:
        hdr = f"  {'TYPE':<22} {'ENTITIES':>10} {'OBS':>10} {'OBS/ENT':>8}  {'SPAN(d)':>8}  STATUS"
        print("By entity_type:")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for row in by_type:
            is_sp, reasons = _is_sparse(row, min_obs, min_days, min_entities)
            status = "SPARSE" if is_sp else "OK"
            if is_sp:
                sparse_rows.append((row, reasons))
            print(
                f"  {row['entity_type']:<22} "
                f"{row['entity_count']:>10,} "
                f"{row['obs_count']:>10,} "
                f"{row['obs_per_entity']:>8.1f}  "
                f"{row['span_days']:>8.0f}  "
                f"{status}"
            )
        print()

    # ── Shannon entropy across entity types ─────────────────────────────────
    if by_type:
        ent = _entropy([r["obs_count"] for r in by_type])
        print(f"Entity-type Shannon entropy: {ent:.2f} nats")
        print()

    # ── By source_tool ───────────────────────────────────────────────────────
    print("By source_tool:")
    if not by_tool:
        print("  (none)")
    else:
        for row in by_tool:
            warn = "  WARNING: zero observations" if row["obs_count"] == 0 else ""
            print(f"  {row['source_tool']:<40} {row['obs_count']:>12,}{warn}")
    print()

    # ── Sparse summary ──────────────────────────────────────────────────────
    if sparse_rows:
        print("=" * width)
        print(f"SPARSE TYPES ({len(sparse_rows)}):")
        print()
        for row, reasons in sparse_rows:
            print(f"  {row['entity_type']}")
            print(f"    Reasons: {', '.join(reasons)}")
            print(
                f"    Earliest: {_fmt_date(row['earliest_ts'])}  "
                f"Latest: {_fmt_date(row['latest_ts'])}"
            )
            print(f"    Recommendation: {_recommend(row, reasons)}")
            print()

        print("=" * width)
        print(f"VERDICT: FAIL ({len(sparse_rows)} sparse type(s))")
        print(
            "Document override in tasks/active/quant_training_ground.md "
            "before starting Phase 40."
        )
        print("=" * width)
        return 1
    else:
        print("=" * width)
        print("VERDICT: PASS — all entity types meet density thresholds")
        print("Phase 40 (GNN retrain on real data) may proceed.")
        print("=" * width)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 47 Density Audit — exit gate for Phase 40"
    )
    parser.add_argument("--db-path", default=_DEFAULT_DB)
    parser.add_argument(
        "--min-obs",
        type=int,
        default=_DEFAULT_MIN_OBS,
        help=f"Min observations per entity_type (default: {_DEFAULT_MIN_OBS})",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=_DEFAULT_MIN_DAYS,
        help=f"Min temporal span in days (default: {_DEFAULT_MIN_DAYS})",
    )
    parser.add_argument(
        "--min-entities",
        type=int,
        default=_DEFAULT_MIN_ENTITIES,
        help=f"Min distinct entities per type (default: {_DEFAULT_MIN_ENTITIES})",
    )
    args = parser.parse_args(argv)

    return _run_audit(
        db_path=args.db_path,
        min_obs=args.min_obs,
        min_days=args.min_days,
        min_entities=args.min_entities,
    )


if __name__ == "__main__":
    sys.exit(main())
