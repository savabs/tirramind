"""Live Intelligence Digest — real signals computed from stored live data.

The "considerable output" proof for the live path. Reads real observations that
the 63 data tools collected (sovereign yields, CFTC positioning, volatility),
computes genuine anomaly signals with the existing math stack (z-scores + BOCPD
changepoint detection), and emits a ranked JSON digest.

This is not a toy: it reads the actual `.tirra_pipeline/pipeline.db` and computes
from real stored numbers. No GNN/LLM required — deterministic math on live data.

Usage:
    .venv/bin/python scripts/live_intelligence_digest.py [--limit N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.awos.learning.learning_core import LearningCore  # noqa: E402
from agent.pipeline.store import PipelineStore  # noqa: E402
from agent.quant.changepoint import BOCPD  # noqa: E402
from agent.quant.signal_outcome_store import SignalOutcomeStore  # noqa: E402

# Observation types we know carry numeric signals we can score.
# FIELD NAMES MUST MATCH THE STORED PAYLOAD KEYS EXACTLY. Three of these five
# did not, and _series_for silently skips any field it cannot find — so the
# digest scanned 175,275 observations per run and built ZERO series from them,
# which is why every edition was 100% CFTC:
#
#   instrument_volatility  declared ("volatility", "value")   -> 0 of   4,294
#   tvl_change             declared ("tvl_change", "value")   -> 0 of 162,251
#   market_probability     declared ("probability", "value")  -> 0 of   8,730
#
# Verified against entity_observations.value_json on 2026-08-27. If you add a
# source here, query one real payload first — a typo costs an entire source
# with no error anywhere.
_SCORABLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "sovereign_debt": ("sovereign_yield", ("yield_pct",)),
    "cftc": ("futures_positioning", ("mm_net", "open_interest")),
    "instrument_universe": ("instrument_volatility", ("realized_vol_20d", "intraday_range")),
    "defi_flows": ("tvl_change", ("tvl_usd",)),
    # yes_price IS the market-implied probability; volume_24h catches
    # liquidity shocks that move ahead of the price.
    "polymarket": ("market_probability", ("yes_price", "volume_24h")),
}

# Per (source, obs_type) we keep series in memory keyed by (entity, field).
SeriesKey = tuple[str, str]  # type: ignore[name-defined]


def _extract_series(
    store: PipelineStore, source: str, obs_type: str, fields: tuple[str, ...]
) -> dict[SeriesKey, tuple[list[float], float]]:
    """Pull numeric series from real observations, keyed by (entity, field).

    Returns {key: (values, latest_observed_at)} so surfaced findings can carry
    the flagged timestamp (needed for honest forward-data realization).
    """
    cur = store._conn.cursor()
    rows = cur.execute(
        "select entity_id, observed_at, value_json from entity_observations "
        "where source_tool=? and observation_type=? order by observed_at asc",
        (source, obs_type),
    ).fetchall()
    series: dict[SeriesKey, tuple[list[float], float]] = {}
    for entity_id, _ts, value_json in rows:
        try:
            value = json.loads(value_json)
            if not isinstance(value, dict):
                continue
            for field in fields:
                fv = value.get(field)
                if isinstance(fv, int | float) and not isinstance(fv, bool):
                    entry = series.setdefault((entity_id, field), ([], _ts or 0.0))
                    entry[0].append(float(fv))
                    series[(entity_id, field)] = (entry[0], _ts or entry[1])
        except (json.JSONDecodeError, TypeError):
            continue
    return series


def _zscore_anomaly(values: list[float]) -> float | None:
    """Latest value vs. prior distribution → z-score (None if too short)."""
    if len(values) < 20:
        return None
    x = np.array(values, dtype=float)
    hist = x[:-1]
    # std ~ 0 means a flat series — no anomaly signal; distinct-value check is
    # handled by std alone (a genuine spike still has spread in the history).
    if float(np.std(hist)) < 1e-12:
        return None
    z = (x[-1] - np.mean(hist)) / (np.std(hist) + 1e-12)
    return round(float(z), 2)


def _changepoint_flag(values: list[float]) -> bool:
    """True if BOCPD detects a changepoint in the most recent window."""
    if len(values) < 30:
        return False
    x = np.array(values, dtype=float)
    # Work on the latest contiguous window (BOCPD is online).
    b = BOCPD(hazard_lambda=200.0)
    res = b.fit(x)
    cps = res.changepoints()
    if not cps:
        return False
    return cps[-1] >= len(x) // 3  # changepoint in the recent third of the series


def build_digest(store, top_n=10):
    """Compute a real anomaly digest over the scorable signal surface."""
    findings = []
    for source, (obs_type, fields) in _SCORABLE.items():
        series = _extract_series(store, source, obs_type, fields)
        for (entity_id, field), (values, latest_ts) in series.items():
            z = _zscore_anomaly(values)
            if z is None or abs(z) < 2.0:
                continue
            cp = _changepoint_flag(values)
            findings.append(
                {
                    "source": source,
                    "observation_type": obs_type,
                    "entity_id": entity_id,
                    "field": field,
                    "zscore": z,
                    "changepoint": cp,
                    "flagged_ts": latest_ts,
                    "n_points": len(values),
                    "latest_value": round(values[-1], 4),
                }
            )

    # Rank by |z| (anomaly magnitude), changepoints first.
    findings.sort(key=lambda f: (f["changepoint"], abs(f["zscore"])), reverse=True)

    total_series = 0
    for source, (obs_type, fields) in _SCORABLE.items():
        total_series += len(_extract_series(store, source, obs_type, fields))

    return {
        "surface_scored": len(_SCORABLE),
        "series_found": total_series,
        "anomalies_flagged": len(findings),
        "top_confidence": findings[0]["zscore"] if findings else 0.0,
        "digest": findings[:top_n],
    }


def surface_findings(
    findings: list[dict[str, Any]],
    store_path: str = ".tirra_opportunities/signal_outcomes.jsonl",
) -> int:
    """Phase 1 — record surfaced anomalies as PENDING (no reward assigned).

    Honest learning: surfacing an anomaly is NOT a success. It only becomes a
    reward after Phase 2 (realize) confirms the forward move. This prevents the
    fake-reward disease (rewarding every surface) that earlier broke the loop.
    """
    ledger = SignalOutcomeStore(store_path)
    surfaced = 0
    for f in findings:
        ledger.surface(
            source=f["source"],
            observation_type=f["observation_type"],
            entity_id=f["entity_id"],
            field=f["field"],
            direction=1.0 if f["zscore"] >= 0 else -1.0,
            flagged_ts=f.get("flagged_ts", 0.0),
            ref_value=f["latest_value"],
            zscore=f["zscore"],
        )
        surfaced += 1
    return surfaced


def realize_pending(
    db_path: str,
    store_path: str = ".tirra_opportunities/signal_outcomes.jsonl",
    state_dir: str = ".awos",
    min_forward_points: int = 3,
) -> dict[str, Any]:
    """Phase 2 — check forward data; record ONLY honest outcomes into the loop.

    For each pending signal: query the same (source, obs_type, entity, field)
    AFTER the flagged time. If enough forward points exist, compute the actual
    post-flag move and compare vs direction. success = moved in flagged
    direction. Signals without forward data stay pending (no reward — no guess).
    """
    ledger = SignalOutcomeStore(store_path)
    core = LearningCore(state_dir=state_dir)
    store = PipelineStore(db_path)
    cur = store._conn.cursor()

    realized = 0
    still_pending = 0
    for sig in ledger.pending():
        rows = cur.execute(
            "select observed_at, value_json from entity_observations "
            "where source_tool=? and observation_type=? and entity_id=? "
            "and observed_at > ? order by observed_at asc",
            (sig.source, sig.observation_type, sig.entity_id, sig.flagged_ts),
        ).fetchall()
        forward: list[float] = []
        for _ts, value_json in rows:
            try:
                v = json.loads(value_json)
                fv = v.get(sig.field)
                if isinstance(fv, int | float) and not isinstance(fv, bool):
                    forward.append(float(fv))
            except (json.JSONDecodeError, TypeError):
                continue
            if len(forward) >= min_forward_points:
                break

        if len(forward) < min_forward_points:
            still_pending += 1
            continue

        # Post-flag mean vs ref value, compared against the flagged direction.
        post_mean = sum(forward) / len(forward)
        move = post_mean - sig.ref_value
        success = bool((move * sig.direction) > 0.0)

        # Record the HONEST outcome into the learning loop.
        core.record_outcome(
            task_id=f"{sig.source}_{sig.entity_id[:8]}_{sig.field}_{sig.signal_id[:6]}",
            operation=f"{sig.source} {sig.observation_type} {sig.field} anomaly",
            action_id=3,  # statistical method tier
            success=success,
            cost_usd=0.0,
            signal_name=sig.source,
            source_tool=sig.source,
        )
        ledger.realize(sig.signal_id, success)
        realized += 1

    return {
        "realized": realized,
        "still_pending": still_pending,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Intelligence Digest (real signals from stored data)")
    parser.add_argument("--limit", type=int, default=10, help="max findings to show")
    parser.add_argument("--out", type=str, default=None, help="output JSON path")
    parser.add_argument("--db", type=str, default=".tirra_pipeline/pipeline.db", help="pipeline DB path")
    parser.add_argument("--surface", action="store_true", help="Phase 1: record findings as pending (no reward)")
    parser.add_argument("--realize", action="store_true", help="Phase 2: check forward data, record honest outcomes")
    parser.add_argument(
        "--signal-store", type=str, default=".tirra_opportunities/signal_outcomes.jsonl", help="signal ledger path"
    )
    args = parser.parse_args()

    store = PipelineStore(args.db)

    if args.realize:
        result = realize_pending(args.db, store_path=args.signal_store)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    report = build_digest(store, top_n=args.limit)

    if args.surface:
        surfaced = surface_findings(report["digest"], store_path=args.signal_store)
        report["surfaced_pending"] = surfaced
        report["note"] = "Surfaced as PENDING — no reward yet. Run --realize after forward data accrues."

    text = json.dumps(report, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {len(report['digest'])} findings -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
