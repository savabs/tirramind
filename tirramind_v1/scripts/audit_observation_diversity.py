#!/usr/bin/env python3
"""Phase 42 — Observation diversity audit.

Reports per-source_tool and per-entity_type observation counts, computes the
Shannon entropy of the entity-type distribution, and prints a pass/fail
verdict against the Phase 42 acceptance criteria:

  - entropy >= 1.0 nats (~56% of max for 6 types)
  - every entity type has >= 100 observations

Usage:
    python scripts/audit_observation_diversity.py [--db-path PATH]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline.store import PipelineStore  # noqa: E402


MIN_ENTROPY_NATS = 1.0
MIN_OBS_PER_TYPE = 100


def _fetch_counts(
    store: PipelineStore,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return (by_source_tool, by_entity_type, by_link_type)."""
    by_source = dict(
        store._conn.execute(
            "SELECT source_tool, COUNT(*) FROM entity_observations "
            "GROUP BY source_tool"
        ).fetchall()
    )
    by_entity_type = dict(
        store._conn.execute(
            "SELECT e.entity_type, COUNT(*) "
            "FROM entity_observations o "
            "JOIN entities e ON o.entity_id = e.entity_id "
            "GROUP BY e.entity_type"
        ).fetchall()
    )
    by_link_type = dict(
        store._conn.execute(
            "SELECT link_type, COUNT(*) FROM entity_links GROUP BY link_type"
        ).fetchall()
    )
    return by_source, by_entity_type, by_link_type


def _shannon_entropy_nats(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        h -= p * math.log(p)
    return h


def _fmt_row(k: str, n: int, total: int) -> str:
    pct = (n / total * 100.0) if total else 0.0
    return f"  {k:35s} {n:>8,}  ({pct:5.2f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=str,
        default=".tirra_pipeline/pipeline.db",
        help="PipelineStore database path.",
    )
    args = parser.parse_args()

    store = PipelineStore(db_path=args.db_path)
    try:
        by_src, by_type, by_link = _fetch_counts(store)
    finally:
        store.close()

    total_obs = sum(by_src.values())

    print("=" * 72)
    print(f"Observation diversity audit — {args.db_path}")
    print("=" * 72)

    print(f"\nTotal observations: {total_obs:,}")

    print("\nBy source_tool:")
    for k, n in sorted(by_src.items(), key=lambda kv: -kv[1]):
        print(_fmt_row(str(k) if k is not None else "(null)", n, total_obs))

    print("\nBy entity_type:")
    total_by_type = sum(by_type.values())
    for k, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(_fmt_row(str(k), n, total_by_type))

    entropy = _shannon_entropy_nats(by_type)
    k_types = len(by_type)
    h_max = math.log(k_types) if k_types > 1 else 1.0
    pct_of_max = (entropy / h_max * 100.0) if h_max > 0 else 0.0

    print(
        f"\nEntity-type Shannon entropy: {entropy:.3f} nats "
        f"(max for {k_types} types = {h_max:.3f}, {pct_of_max:.1f}% of max)"
    )
    effective_k = math.exp(entropy)
    print(f"Effective type count (exp(H)): {effective_k:.2f}")

    print("\nBy link_type:")
    total_links = sum(by_link.values())
    for k, n in sorted(by_link.items(), key=lambda kv: -kv[1]):
        print(_fmt_row(str(k) if k is not None else "(null)", n, total_links))
    print(f"Total links: {total_links:,}")

    # Acceptance verdict
    print("\n" + "=" * 72)
    print("Phase 42 acceptance check")
    print("=" * 72)

    entropy_ok = entropy >= MIN_ENTROPY_NATS
    all_types_ok = all(n >= MIN_OBS_PER_TYPE for n in by_type.values()) and bool(
        by_type
    )

    under = [k for k, n in by_type.items() if n < MIN_OBS_PER_TYPE]

    def _mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(
        f"  Entropy \u2265 {MIN_ENTROPY_NATS:.2f} nats: {_mark(entropy_ok)} "
        f"(got {entropy:.3f})"
    )
    print(
        f"  Every type \u2265 {MIN_OBS_PER_TYPE} obs:  {_mark(all_types_ok)}"
        + (f"  under-observed: {under}" if under else "")
    )

    ok = entropy_ok and all_types_ok
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
