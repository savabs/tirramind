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
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

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
#
# VALUE SHAPE (2026-08-29, $29 tier pass): each source maps to a TUPLE of
# (observation_type, fields) pairs, not a single pair. A source can carry the
# same field under more than one observation_type — instrument_universe
# stores "close"/"realized_vol_20d"/"volume" under the superseded
# instrument_daily type (752-1,096 points/entity back to 2023-04-18) AND under
# the newer instrument_return/instrument_volatility/instrument_volume types
# (37-50 points/entity, Apr-Jun 2026 only). Scoring only the newer type's ~50
# points made ordinary seasonal variation look anomalous — see
# _extract_series_multi. List OLDEST/base type first: when both types have a
# row for the same (entity, field, observed_at) in the overlap window, the
# LATER entry in the tuple wins (it's the superseding collector).
#
# polymarket REMOVED (2026-08-29): market_probability tops out at 15
# points/entity across all 1,493 entities — a hard structural ceiling, not a
# quiet period. _zscore_anomaly requires >=20 points. This source has scanned
# 1,493 entities every run and produced exactly 0 findings, ever, and
# mathematically never can until far more history accrues. Scanning it nightly
# and counting it in surface_scored/sources_ok reports a working source that
# cannot work. Re-add once any entity actually clears the 20-point floor.
#
# sovereign_debt KEPT despite 0 findings so far: only 5 of 13 entities clear
# the 20-point floor, and their current |z| tops out at 1.78 — genuinely
# quiet, not structurally incapable (unlike polymarket, there is no hard
# ceiling here; it computes real variance and can fire on a real yield move).
# This is the same "quiet source" status any of the other sources can have on
# a given night, not dead config.
_SCORABLE: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "sovereign_debt": (("sovereign_yield", ("yield_pct",)),),
    "cftc": (
        (
            "futures_positioning",
            (
                "mm_net",
                "open_interest",
                # CFTC COT payload carries nine numeric fields; only two were
                # ever scored. These seven are commercial hedger/producer
                # positioning and concentration — not derivable from the
                # public CFTC website's headline number the way mm_net is.
                "swap_net",  # swap-dealer net position (commercial hedger)
                "pm_net",  # producer/merchant net position
                "conc_top4_long",  # top-4 traders' share of long open interest
                "conc_top4_short",  # top-4 traders' share of short open interest
                "mm_net_pct_oi",  # managed-money net as % of open interest
                "mm_weekly_flow",  # week-over-week change in managed-money net
                "oi_change",  # week-over-week change in open interest
            ),
        ),
    ),
    "instrument_universe": (
        # Base/superseded type: long history, listed first so the newer
        # types below win on overlapping (entity, field, observed_at).
        #
        # "close" is deliberately NOT scored here. Measured on the real DB
        # (2026-08-29): once "close" gets the correct ~1,096-point baseline
        # instead of the ~50-point one, 28 of its 33 findings have a z-sign
        # that matches the direction of the last 20 log-returns — i.e. they
        # are "this asset has been trending for weeks," not a distinct
        # anomaly, on a source we sell as anomaly detection, not trend
        # detection. realized_vol_20d and volume do not have this problem
        # (they're each derived from short-window stats, not raw level) and
        # are kept.
        ("instrument_daily", ("realized_vol_20d", "volume")),
        ("instrument_volatility", ("realized_vol_20d", "intraday_range")),
        ("instrument_volume", ("volume", "avg_volume_20d")),
    ),
    "defi_flows": (("tvl_change", ("tvl_usd",)),),
}

# Per (source, obs_type) we keep series in memory keyed by (entity, field).
SeriesKey = tuple[str, str]  # type: ignore[name-defined]


def _extract_series(
    store: PipelineStore, source: str, obs_type: str, fields: tuple[str, ...]
) -> dict[SeriesKey, tuple[list[float], float, list[float]]]:
    """Pull numeric series from real observations, keyed by (entity, field).

    Returns {key: (values, latest_observed_at, timestamps)}. `timestamps` is
    aligned 1:1 with `values` (both ascending by observed_at) so callers can
    convert a BOCPD changepoint index into real elapsed time (e.g. "6 weeks
    ago") instead of a raw array position.
    """
    cur = store._conn.cursor()
    # DEDUPE ON (entity_id, observed_at). Collection re-ingests the same report
    # on every run without an upsert, so entity_observations holds 442
    # duplicate futures_positioning rows — 2026-08-18 alone has 272 against a
    # typical ~35. Left raw, the last 8 "weekly" cotton points all carry the
    # SAME date, BOCPD flags the discontinuity where the duplicates begin, and
    # elapsed time from that index is 0.0 weeks — which rendered to customers
    # as "structural break this week". That claim was an artifact of duplicate
    # ingestion, not market structure.
    #
    # max(rowid) keeps the most recently written value for a timestamp. This is
    # a read-side guard; the ingestion path still needs an upsert (see
    # tasks/active/nineteen_dollar_tier.md).
    rows = cur.execute(
        "select entity_id, observed_at, value_json from entity_observations "
        "where source_tool=? and observation_type=? "
        "and rowid in ("
        "  select max(rowid) from entity_observations"
        "  where source_tool=? and observation_type=?"
        "  group by entity_id, observed_at"
        ") order by observed_at asc",
        (source, obs_type, source, obs_type),
    ).fetchall()
    series: dict[SeriesKey, tuple[list[float], float, list[float]]] = {}
    for entity_id, _ts, value_json in rows:
        try:
            value = json.loads(value_json)
            if not isinstance(value, dict):
                continue
            for field in fields:
                fv = value.get(field)
                if isinstance(fv, int | float) and not isinstance(fv, bool):
                    entry = series.setdefault((entity_id, field), ([], _ts or 0.0, []))
                    entry[0].append(float(fv))
                    entry[2].append(float(_ts or 0.0))
                    series[(entity_id, field)] = (entry[0], _ts or entry[1], entry[2])
        except (json.JSONDecodeError, TypeError):
            continue
    return series


def _extract_series_multi(
    store: PipelineStore, source: str, type_specs: tuple[tuple[str, tuple[str, ...]], ...]
) -> dict[SeriesKey, tuple[list[float], float, list[float], str]]:
    """Union multiple observation types onto one series per (entity, field).

    Each (obs_type, fields) pair is pulled through `_extract_series` (already
    deduped on (entity_id, observed_at) within that type). Series are then
    merged per (entity, field) keyed by observed_at — this is a SECOND dedupe
    pass, across types, needed because instrument_daily and
    instrument_return/instrument_volatility/instrument_volume overlap Apr-Jun
    2026 (576 rows) for the same 89 entities. `type_specs` order matters: for
    a timestamp present in more than one type, whichever type appears LATER
    in the tuple wins (the newer/superseding collector).

    Returns {(entity, field): (values, latest_ts, timestamps, obs_type)}
    where obs_type is whichever type actually produced the latest point —
    this keeps `realize_pending`'s forward-looking query pointed at the type
    that is still being actively collected, not a superseded one.
    """
    merged_values: dict[SeriesKey, dict[float, float]] = {}
    merged_type: dict[SeriesKey, dict[float, str]] = {}
    for obs_type, fields in type_specs:
        s = _extract_series(store, source, obs_type, fields)
        for key, (values, _latest, timestamps) in s.items():
            vd = merged_values.setdefault(key, {})
            td = merged_type.setdefault(key, {})
            for v, t in zip(values, timestamps, strict=True):
                vd[t] = v
                td[t] = obs_type

    result: dict[SeriesKey, tuple[list[float], float, list[float], str]] = {}
    for key, vd in merged_values.items():
        ts_sorted = sorted(vd)
        values = [vd[t] for t in ts_sorted]
        latest_ts = ts_sorted[-1]
        latest_type = merged_type[key][latest_ts]
        result[key] = (values, latest_ts, ts_sorted, latest_type)
    return result


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


def _changepoint_indices(values: list[float]) -> list[int]:
    """Raw BOCPD changepoint indices (positions in `values`), or [] if too short."""
    if len(values) < 30:
        return []
    x = np.array(values, dtype=float)
    # Work on the latest contiguous window (BOCPD is online).
    b = BOCPD(hazard_lambda=200.0)
    res = b.fit(x)
    return res.changepoints()


def _changepoint_flag(values: list[float]) -> bool:
    """True if BOCPD detects a changepoint in the most recent window."""
    cps = _changepoint_indices(values)
    if not cps:
        return False
    return cps[-1] >= len(values) // 3  # changepoint in the recent third of the series


_SECONDS_PER_WEEK = 604800.0

# entity_id -> canonical_name, memoized per build_digest() call (entities are
# static reference data; avoids one store.get_entity() round trip per finding).
_EntityNameCache = dict[str, str]


def _entity_name(store, entity_id: str, cache: _EntityNameCache) -> str:
    """Resolve entity_id -> human-readable canonical_name (falls back to id)."""
    if entity_id in cache:
        return cache[entity_id]
    name = entity_id
    try:
        row = store.get_entity(entity_id)
        if row and row.get("canonical_name"):
            name = row["canonical_name"]
    except Exception:
        pass
    cache[entity_id] = name
    return name


def build_digest(store, top_n=10):
    """Compute a real anomaly digest over the scorable signal surface.

    Robustness (2026-08-28, $19 tier launch pass): a single source's query
    failing (locked DB, missing/renamed table, malformed row) used to abort
    `_extract_series` for that source with an uncaught exception, which
    propagated all the way up through `fetch_anomalies` -> `build_brief` ->
    `BriefDeliverer.deliver()` and killed the ENTIRE weekly job — one bad
    source took four healthy ones down with it, and the subscriber got no
    edition at all that week (a `oneshot` systemd unit does not retry until
    the next weekly trigger).

    Each source is now queried in isolation: a failure is logged and that
    source is skipped, the rest still contribute. But an empty `digest` must
    always mean "genuinely checked every source, nothing crossed the
    threshold" — never "failed to check anything and quietly shipped an
    empty list that looks identical to a quiet week." So if EVERY source
    fails, this raises instead of returning a fake-quiet report; the caller
    (tirra_engine.py) then exits non-zero and no brief is delivered, which is
    visibly a failure (systemd/journalctl) rather than a silent lie.
    """
    findings = []
    name_cache: _EntityNameCache = {}
    series_by_source: dict[str, dict] = {}
    sources_failed: list[str] = []

    for source, type_specs in _SCORABLE.items():
        try:
            series_by_source[source] = _extract_series_multi(store, source, type_specs)
        except Exception as exc:  # sqlite3.OperationalError (locked/missing table), malformed rows, etc.
            logger.warning("[digest] source=%s query failed, skipping (other sources still scored): %s", source, exc)
            sources_failed.append(source)

    if sources_failed and len(sources_failed) == len(_SCORABLE):
        raise RuntimeError(
            f"all {len(_SCORABLE)} scorable sources failed to query "
            f"({', '.join(sorted(sources_failed))}) — refusing to return an "
            "empty digest that would be indistinguishable from a genuine "
            "no-anomalies edition"
        )

    for source, series in series_by_source.items():
        for (entity_id, field), (values, latest_ts, timestamps, obs_type) in series.items():
            z = _zscore_anomaly(values)
            if z is None or abs(z) < 2.0:
                continue
            cps = _changepoint_indices(values)
            cp = bool(cps) and cps[-1] >= len(values) // 3
            weeks_ago = None
            if cp and timestamps and len(timestamps) == len(values):
                delta = timestamps[-1] - timestamps[cps[-1]]
                weeks_ago = round(delta / _SECONDS_PER_WEEK, 1)
            findings.append(
                {
                    "source": source,
                    "observation_type": obs_type,
                    "entity_id": entity_id,
                    "entity_name": _entity_name(store, entity_id, name_cache),
                    "field": field,
                    "zscore": z,
                    "direction": "up" if z >= 0 else "down",
                    "changepoint": cp,
                    "changepoint_weeks_ago": weeks_ago,
                    "flagged_ts": latest_ts,
                    "n_points": len(values),
                    "latest_value": round(values[-1], 4),
                }
            )

    # Rank by |z| (anomaly magnitude), changepoints first.
    findings.sort(key=lambda f: (f["changepoint"], abs(f["zscore"])), reverse=True)

    total_series = sum(len(s) for s in series_by_source.values())

    return {
        "surface_scored": len(_SCORABLE),
        "sources_ok": sorted(series_by_source),
        "sources_failed": sources_failed,
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
