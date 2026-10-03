#!/usr/bin/env python3
"""Automatic ghost-chain discovery, end to end, with its own denominator.

    load graph -> enumerate -> screen -> test -> correct -> explain -> render

WHAT THIS IS
    The first run of chain discovery in this repo that a human did not author the
    pattern for. Every previously published chain brief came from a hand-written
    YAML template in `templates/ghost_chains/mp1/`; the old automatic scorer
    ranked meta-paths by `mean_attention * log(frequency)` taken from an HGT that
    was never trained, so the attention was uniform 1/k fractions and the ranking
    was frequency in a costume. It never produced a candidate.

WHAT A CHAIN IS, OPERATIONALLY
    A candidate is (event stream, threshold, direction, target stream, lag):

        country/gdelt.goldstein  z <= -2.5  ->  instrument.close  @ 10 steps

    The statistical test is run by `agent.verify.study.verify`, which grades the
    event against the target through a DIRECT graph link, with a causal z-score
    (prior points only), a population baseline, a circular block bootstrap CI and
    a cluster-resampled p. The MECHANISM — the multi-hop path and how many
    genuinely independent time-varying sources witness it — comes from
    `agent.mechanism.routes.top_routes` + `independence`, computed for chains that
    survive. The two halves answer different questions and neither is derived
    from the other.

WHY THE DENOMINATOR IS THE PRODUCT
    This grid generates tens of thousands of candidates. At m tested, 0.05 * m
    are expected to clear an uncorrected 0.05 by chance. Every headline this
    project has produced died when measured properly (the CFTC study: 0 of 51
    survived BH, 9 of 51 would have "worked" uncorrected), so this script is
    built to make the dying visible: every filter reports its count, every
    candidate that leaves the funnel lands in a named bucket, and
    `agent.ghost.publish.Scoreboard` refuses to exist unless
    enumerated == untestable + tested.

WHAT IT NEVER DOES
    Writes to the database (opened `mode=ro` by every callee), retrains anything,
    or emits advice. It prints markdown.

MODULE OWNERSHIP
    Enumeration and scoring are being written concurrently as
    `agent.ghost.enumerate` / `agent.ghost.score`. This script imports them when
    they exist and otherwise uses the self-contained fallback below, which is
    marked as such and reports which path it took on the page. The fallback is
    deliberately dumb: a cartesian product plus monotone screens.

Usage:
    .venv/bin/python scripts/ghost_discovery_run.py --out /tmp/board.md
    .venv/bin/python scripts/ghost_discovery_run.py --smoke        # tiny grid
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import random
import sqlite3
import statistics
import sys
import time
from dataclasses import asdict as dc_asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.ghost.publish import (  # noqa: E402
    DEFAULT_MIN_CLUSTERS,
    DEFAULT_MIN_EVENTS,
    ChainResult,
    FilterStep,
    Scoreboard,
    SeriesCheck,
    UntestableReason,
    render_scoreboard,
    series_check,
    summarise,
)
from agent.verify.stats import benjamini_hochberg  # noqa: E402
from agent.verify.study import Hypothesis, verify  # noqa: E402

DEFAULT_DB = ".tirra_pipeline/pipeline.db"

# --- the grid ---------------------------------------------------------------
# Printed verbatim on the page so a reader can multiply it out and check the
# enumerated count by hand.
# `abs` is deliberately absent: `agent.verify.study.Hypothesis` documents that it
# "cancels a signed effect by construction", so an abs candidate is a weaker test
# of the same chain as its up/down pair, not an extra one. Every remaining axis is
# printed on the page and the product is the denominator.
Z_GRID = (2.0, 3.0)
DIRECTION_GRID = ("up", "down")
HORIZON_GRID = (1, 5, 21)

SMOKE_Z = (2.0,)
SMOKE_DIRECTION = ("abs",)
SMOKE_HORIZON = (5,)

# An event stream needs enough rows for a causal z-score to exist at all. Below
# this nothing downstream can fire, and enumerating it would pad the denominator
# with candidates that were never arguable.
MIN_STREAM_OBSERVATIONS = 50
VALUE_JSON_SAMPLE = 400
MIN_FIELD_HITS = 25


class TargetStream(NamedTuple):
    """A movable quantity a chain can point at.

    `steps_label` exists because `Hypothesis.horizon_days` counts SERIES STEPS of
    the target, not calendar days. On a weekly series "10" means 10 weeks and the
    field name lies; this label is what gets printed.
    """

    entity_type: str
    obs_type: str
    source: str
    value_field: str
    market: str
    steps_label: str


# Verified present in the live DB by `_audit_target_streams` before use; the run
# aborts rather than silently enumerating a target with no series.
TARGET_STREAMS = (
    TargetStream(
        "instrument", "instrument_daily", "instrument_universe", "close", "instrument daily close", "trading days"
    ),
    TargetStream(
        "country", "sovereign_yield", "sovereign_debt", "yield_pct", "sovereign bond yield", "observation steps"
    ),
    TargetStream("protocol", "tvl_change", "defi_flows", "tvl_usd", "protocol total value locked", "observation steps"),
    TargetStream(
        "topic", "market_probability", "polymarket", "yes_price", "prediction-market probability", "observation steps"
    ),
)


class EventStream(NamedTuple):
    """A numeric series that can fire an event, discovered from the live DB."""

    entity_type: str
    source: str
    obs_type: str
    field: str
    n_obs: int
    n_entities: int

    @property
    def label(self) -> str:
        return f"{self.entity_type}/{self.source}.{self.field}"


@dataclass
class Candidate:
    """One enumerated chain, and wherever it ended up."""

    event: EventStream
    target: TargetStream
    z: float
    direction: str
    horizon: int
    status: str = "enumerated"
    reason: str = ""

    @property
    def chain_id(self) -> str:
        return (
            f"{self.event.entity_type}.{self.event.source}.{self.event.obs_type}."
            f"{self.event.field}|z{self.z:g}{self.direction}|"
            f"{self.target.entity_type}.{self.target.value_field}|h{self.horizon}"
        )

    @property
    def name(self) -> str:
        arrow = {"up": "z >= +", "down": "z <= -", "abs": "|z| >= "}[self.direction]
        return (
            f"{self.event.entity_type}:{self.event.source}.{self.event.field} "
            f"{arrow}{self.z:g} -> {self.target.entity_type}.{self.target.value_field} "
            f"@ {self.horizon}"
        )

    def hypothesis(self) -> Hypothesis:
        return Hypothesis(
            event_source=self.event.source,
            event_obs_type=self.event.obs_type,
            event_field=self.event.field,
            z_threshold=self.z,
            direction=self.direction,  # type: ignore[arg-type]
            target_entity_type=self.target.entity_type,
            target_obs_type=self.target.obs_type,
            target_source=self.target.source,
            target_value_field=self.target.value_field,
            horizon_days=self.horizon,
            label=self.name,
        )


# ---------------------------------------------------------------------------
# read-only DB introspection
# ---------------------------------------------------------------------------


def connect_ro(db_path: str) -> sqlite3.Connection:
    """Open the live pipeline DB read-only. The only connection this script makes.

    `mode=ro` is not a convention here: a discovery run that mutated the graph it
    was measuring would invalidate every number it printed.
    """
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"no database at {db_path}")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def numeric_fields(con: sqlite3.Connection, source: str, obs_type: str) -> list[str]:
    """Numeric keys inside `value_json`, from a bounded sample of rows.

    Dotted paths one level deep are included (`yields.10y`), matching what
    `Hypothesis.event_field` can resolve. Booleans are excluded: `True` is not 1
    and z-scoring a flag produces a number that means nothing.

    Where it misleads: this is a SAMPLE of `VALUE_JSON_SAMPLE` rows. A field that
    appears only in later rows can be missed, which shrinks the enumerated space
    silently — the sample size and the hit floor are printed in the provenance
    block so the omission is at least bounded and visible.
    """
    counts: collections.Counter[str] = collections.Counter()
    rows = con.execute(
        "SELECT value_json FROM entity_observations WHERE source_tool=? AND observation_type=? LIMIT ?",
        (source, obs_type, VALUE_JSON_SAMPLE),
    ).fetchall()
    for (raw,) in rows:
        try:
            doc = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        for key, val in doc.items():
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                counts[key] += 1
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, bool):
                        continue
                    if isinstance(v2, (int, float)) and math.isfinite(float(v2)):
                        counts[f"{key}.{k2}"] += 1
    floor = min(MIN_FIELD_HITS, max(1, len(rows)))
    return sorted(k for k, n in counts.items() if n >= floor)


def discover_event_streams(con: sqlite3.Connection) -> tuple[list[EventStream], list[str]]:
    """Every (entity type, source, observation type, numeric field) in the live DB.

    Returns the streams and a human log of what was skipped and why. Nothing is
    dropped without a line in that log.
    """
    log: list[str] = []
    q = """
        SELECT e.entity_type, o.source_tool, o.observation_type,
               COUNT(*) AS c, COUNT(DISTINCT o.entity_id)
        FROM entity_observations o JOIN entities e ON e.entity_id = o.entity_id
        GROUP BY 1, 2, 3 ORDER BY 1, 4 DESC
    """
    streams: list[EventStream] = []
    for etype, source, obs_type, n_obs, n_ent in con.execute(q).fetchall():
        if n_obs < MIN_STREAM_OBSERVATIONS:
            log.append(f"skipped {etype}/{source}/{obs_type}: {n_obs} observations < {MIN_STREAM_OBSERVATIONS}")
            continue
        fields = numeric_fields(con, source, obs_type)
        if not fields:
            log.append(
                f"skipped {etype}/{source}/{obs_type}: no numeric value_json field in a {VALUE_JSON_SAMPLE}-row sample"
            )
            continue
        for f in fields:
            streams.append(EventStream(etype, source, obs_type, f, int(n_obs), int(n_ent)))
    return streams, log


def type_link_pairs(con: sqlite3.Connection) -> set[tuple[str, str]]:
    """Unordered entity-type pairs joined by at least one direct link.

    `agent.verify.study.verify` resolves targets through DIRECT links in either
    column, so a candidate whose event type has no direct link to its target type
    cannot produce a single pair. This is the cheapest true filter available and
    it is applied before any series is read.

    Where it misleads: a type pair existing does not mean the two SPECIFIC
    entities with observations are linked. That is what the probe stage measures.
    """
    q = """
        SELECT DISTINCT ea.entity_type, eb.entity_type
        FROM entity_links l
        JOIN entities ea ON ea.entity_id = l.entity_id_a
        JOIN entities eb ON eb.entity_id = l.entity_id_b
    """
    pairs: set[tuple[str, str]] = set()
    for a, b in con.execute(q).fetchall():
        pairs.add((a, b))
        pairs.add((b, a))
    return pairs


def audit_target_streams(con: sqlite3.Connection) -> list[str]:
    """Confirm every declared target stream has rows on entities of its type.

    Raises rather than enumerating against a target that cannot move: a target
    with no series makes every one of its candidates untestable for a reason that
    is a configuration bug, and burying thousands of those in the funnel would be
    exactly the kind of padded denominator this page is meant to expose.
    """
    notes: list[str] = []
    for t in TARGET_STREAMS:
        n = con.execute(
            "SELECT COUNT(*) FROM entity_observations o JOIN entities e "
            "ON e.entity_id = o.entity_id WHERE e.entity_type=? AND o.source_tool=? "
            "AND o.observation_type=?",
            (t.entity_type, t.source, t.obs_type),
        ).fetchone()[0]
        if n == 0:
            raise SystemExit(
                f"target stream {t.entity_type}/{t.source}/{t.obs_type} has no rows in this DB; "
                "fix the declaration rather than enumerating against an empty target."
            )
        notes.append(f"{t.entity_type}.{t.value_field}: {n:,} target observations")
    return notes


# ---------------------------------------------------------------------------
# enumeration (fallback; `agent.ghost.enumerate` takes over when it lands)
# ---------------------------------------------------------------------------


def enumerate_candidates(
    streams: list[EventStream],
    targets: tuple[TargetStream, ...],
    z_grid: tuple[float, ...],
    dir_grid: tuple[str, ...],
    horizon_grid: tuple[int, ...],
) -> list[Candidate]:
    """The full cartesian product. This is the denominator, and nothing else is.

    No filtering happens here on purpose. Every narrowing is a named, counted
    stage afterwards, so `len(enumerate_candidates(...))` is a number a reader
    can reproduce from the printed grid by multiplication.
    """
    out: list[Candidate] = []
    for ev in streams:
        for tg in targets:
            for z in z_grid:
                for d in dir_grid:
                    for h in horizon_grid:
                        out.append(Candidate(ev, tg, z, d, h))
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def _drop_reason_summary(result: Any) -> str:
    """The dominant pair-level reason a probe found nothing, in words."""
    reasons = getattr(result, "drop_reasons", {}) or {}
    pair_reasons = {k: v for k, v in reasons.items() if k.startswith("pair_")}
    pool = pair_reasons or reasons
    if not pool:
        return "no gradeable events and no drop reason recorded"
    top = max(pool.items(), key=lambda kv: kv[1])
    # The bucket label carries the REASON only. Folding the per-base occurrence
    # count into it would split one reason into a dozen near-identical rows and
    # make the untestable table unreadable; the counts live in `--probe-log`.
    return str(top[0])


def _falsifier(cand: Candidate, res: Any) -> str:
    """The observation that would kill this specific chain.

    Written per chain rather than boilerplate, because a falsifier that does not
    name a number is not a falsifier.
    """
    sign = "positive" if res.edge > 0 else "negative"
    return (
        f"the next {max(10, cand.horizon * 2)} firings of "
        f"{cand.event.source}.{cand.event.field} at this threshold, on dates not in this "
        f"sample, failing to show a {sign} mean {cand.horizon}-step effect against the same "
        f"population baseline; or the distinct-date count staying at "
        f"{res.n_event_clusters} while events accumulate, which would mean the effect lives "
        "on a handful of dates rather than in the mechanism."
    )


def _route_for(cand: Candidate, res: Any, graph_cache: dict[str, Any]) -> dict[str, Any]:
    """Mechanism route + independent time-varying source count for one chain.

    Uses the first USED (event entity, target entity) pair from the study's own
    attrition list, so the route explains a pair that actually contributed
    returns rather than an arbitrary pair of the right types.

    Where it misleads: one pair's route is not the route for every pair the study
    graded. The hop count and source count here describe a representative, and
    that is stated on the page. Failures are returned as `error`, never as a
    plausible zero.
    """
    out: dict[str, Any] = {
        "route": (),
        "hops": 0,
        "independent": 0,
        "sources": (),
        "scaffold_only": False,
        "error": None,
    }
    used = [p for p in getattr(res, "pair_attrition", []) if getattr(p, "used", False)]
    if not used:
        out["error"] = "no used pair to explain"
        return out
    pair = used[0]
    try:
        if "g" not in graph_cache:
            from agent.mechanism.routes import load_route_graph

            graph_cache["g"] = load_route_graph(graph_cache["db"], as_of=graph_cache["as_of"])
        from agent.mechanism.routes import independence, top_routes

        routes = top_routes(graph_cache["g"], pair.event_entity_id, pair.target_entity_id, max_hops=4, k=25)
        ind = independence(routes)
        if routes:
            # The top-mass route is often pure static geography, which routes a
            # signal without witnessing anything. Display the highest-mass route
            # that carries at least one time-varying hop, and record that the
            # very top one did not.
            evidenced = [r for r in routes if r.has_evidence]
            best = evidenced[0] if evidenced else routes[0]
            out["scaffold_only"] = bool(evidenced) and not routes[0].has_evidence
            labels: list[str] = [best.node_names[0]]
            for i, hop in enumerate(best.hops_detail):
                tag = {"evidence": "time-varying", "scaffold": "static", "unclassified": "?"}[hop.kind]
                labels.append(f"[{hop.link_type}/{hop.source}/{tag}]")
                labels.append(best.node_names[i + 1])
            out["route"] = tuple(labels)
            out["hops"] = best.hops
        out["independent"] = int(ind.get("independent_sources", 0))
        out["sources"] = tuple(sorted(ind.get("evidence_sources", []) or []))
        if not routes:
            out["error"] = str(ind.get("status", "no_route"))
    except Exception as exc:  # noqa: BLE001 - a route failure must not silently zero
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def run(args: argparse.Namespace) -> int:
    t_start = time.time()
    db = args.db
    as_of = None if args.as_of is None else _as_of_epoch(args.as_of)
    as_of_label = args.as_of or "none (the live graph, everything to today)"

    con = connect_ro(db)
    try:
        target_notes = audit_target_streams(con)
        streams, stream_log = discover_event_streams(con)
        linked = type_link_pairs(con)
        n_entities = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        n_links = con.execute("SELECT COUNT(*) FROM entity_links").fetchone()[0]
        n_obs = con.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0]
    finally:
        con.close()

    z_grid = SMOKE_Z if args.smoke else Z_GRID
    dir_grid = SMOKE_DIRECTION if args.smoke else DIRECTION_GRID
    h_grid = SMOKE_HORIZON if args.smoke else HORIZON_GRID
    targets = TARGET_STREAMS[: args.max_targets] if args.max_targets else TARGET_STREAMS
    if args.max_streams:
        streams = streams[: args.max_streams]

    print(f"[graph] {n_entities:,} entities, {n_links:,} links, {n_obs:,} observations")
    for line in target_notes:
        print(f"[target] {line}")
    print(f"[streams] {len(streams)} numeric event streams discovered")
    for line in stream_log:
        print(f"[streams] {line}")

    candidates = enumerate_candidates(streams, targets, z_grid, dir_grid, h_grid)
    enumerated = len(candidates)
    print(
        f"[enumerate] {enumerated:,} candidates = {len(streams)} streams x {len(targets)} "
        f"targets x {len(z_grid)} thresholds x {len(dir_grid)} directions x {len(h_grid)} lags"
    )

    filters: list[FilterStep] = []
    untestable: collections.Counter[str] = collections.Counter()

    # --- stage 1: the event type must link directly to the target type -------
    stage1: list[Candidate] = []
    for c in candidates:
        if (c.event.entity_type, c.target.entity_type) in linked:
            stage1.append(c)
        else:
            untestable[
                f"no direct graph link between the event's entity type and the target's "
                f"({c.event.entity_type} -> {c.target.entity_type})"
            ] += 1
    filters.append(
        FilterStep(
            "direct link between entity types",
            examined=enumerated,
            dropped=enumerated - len(stage1),
            reason="the event study resolves targets through direct entity_links only, so a "
            "type pair with no link can never produce a pair",
        )
    )
    print(f"[stage 1] {len(stage1):,} candidates survive the type-link filter")

    # --- stage 2: the event stream must not be the target stream -------------
    stage2: list[Candidate] = []
    for c in stage1:
        same = (
            c.event.source == c.target.source
            and c.event.obs_type == c.target.obs_type
            and c.event.field == c.target.value_field
        )
        if same:
            untestable["the event stream and the target stream are the same series"] += 1
        else:
            stage2.append(c)
    filters.append(
        FilterStep(
            "event stream is not the target stream",
            examined=len(stage1),
            dropped=len(stage1) - len(stage2),
            reason="z-scoring a series against its own forward return is autocorrelation, not a chain",
        )
    )
    print(f"[stage 2] {len(stage2):,} candidates survive the self-series filter")

    # --- stage 3: one probe per (event stream, target stream) ----------------
    # Monotonicity: raising the threshold cannot raise the gradeable event count,
    # and lengthening the horizon cannot either (a longer horizon needs a later
    # exit point). A probe at the loosest threshold and the shortest horizon is
    # therefore an upper bound for the whole base, so a base that cannot clear
    # the floors at the probe cannot clear them anywhere, and its whole block is
    # untestable without running it.
    bases: dict[tuple[str, ...], list[Candidate]] = collections.defaultdict(list)
    for c in stage2:
        bases[
            (
                c.event.entity_type,
                c.event.source,
                c.event.obs_type,
                c.event.field,
                c.target.entity_type,
                c.target.value_field,
            )
        ].append(c)
    print(f"[stage 3] probing {len(bases)} (event stream, target) bases")

    stage3: list[Candidate] = []
    probe_rows: list[dict[str, Any]] = []
    min_h = min(h_grid)
    # Probe order is a seeded shuffle, not alphabetical: if the budget runs out, the
    # candidates left unreached must not be a whole contiguous block of related
    # streams. The order depends on nothing this run measures, so a budget cut is
    # not a selection on outcome.
    base_items = sorted(bases.items())
    random.Random(args.seed).shuffle(base_items)
    t_probe = time.time()
    probe_budget_hit = False
    for i, (key, block) in enumerate(base_items, start=1):
        if time.time() - t_probe > args.budget_probe_s:
            probe_budget_hit = True
            untestable[
                f"not screened: the {args.budget_probe_s:.0f}s probe budget expired before this "
                "candidate's event stream was reached"
            ] += len(block)
            probe_rows.append({"base": key, "error": "probe budget expired"})
            continue
        proto = block[0]
        probe = Candidate(proto.event, proto.target, 0.0, "abs", min_h)
        t0 = time.time()
        try:
            res = verify(db, probe.hypothesis(), as_of=as_of, max_audit_samples=0)
        except Exception as exc:  # noqa: BLE001 - a probe failure is a counted outcome
            untestable[f"probe raised {type(exc).__name__}: {exc}"] += len(block)
            probe_rows.append({"base": key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        dt_s = time.time() - t0
        ceiling_n, ceiling_c = res.n_events, res.n_event_clusters
        probe_rows.append(
            {
                "base": key,
                "pairs": f"{res.n_pairs_used}/{res.n_pairs_entered}",
                "n": ceiling_n,
                "clusters": ceiling_c,
                "secs": round(dt_s, 2),
            }
        )
        if res.n_pairs_used == 0:
            untestable[
                f"no (event entity, target entity) pair survived data availability: {_drop_reason_summary(res)}"
            ] += len(block)
        elif ceiling_n < args.min_events:
            untestable[
                f"the event stream's ceiling against this target is below the "
                f"{args.min_events}-gradeable-event floor even at threshold 0"
            ] += len(block)
        elif ceiling_c < args.min_clusters:
            untestable[
                f"the event stream's ceiling against this target is below the "
                f"{args.min_clusters}-distinct-date floor even at threshold 0"
            ] += len(block)
        else:
            stage3.extend(block)
        if i % max(1, len(base_items) // 25) == 0 or i == len(base_items):
            print(
                f"[stage 3] {i}/{len(base_items)} bases probed, {len(stage3):,} candidates "
                f"alive, {time.time() - t_start:.0f}s elapsed"
            )
    if probe_budget_hit:
        print(f"[stage 3] PROBE BUDGET EXPIRED after {args.budget_probe_s:.0f}s")
    filters.append(
        FilterStep(
            "event-stream ceiling clears the sample floors",
            examined=len(stage2),
            dropped=len(stage2) - len(stage3),
            reason=f"one probe per (event stream, target) at threshold 0 and the shortest lag "
            f"bounds the gradeable events from above; a base whose ceiling is under "
            f"{args.min_events} events or {args.min_clusters} distinct dates cannot clear them "
            f"at any threshold",
        )
    )
    print(f"[stage 3] {len(stage3):,} candidates survive the ceiling filter")

    # --- stage 4: run the study on every survivor ---------------------------
    tested: list[ChainResult] = []
    pending: list[tuple[Candidate, Any]] = []
    below_floor_n = below_floor_c = 0
    errors = 0
    # Same seeded shuffle, same reason: a budget cut must not be correlated with
    # anything about the candidate.
    study_order = list(stage3)
    random.Random(args.seed + 1).shuffle(study_order)
    t_study = time.time()
    study_budget_hit = 0
    for i, c in enumerate(study_order, start=1):
        if time.time() - t_study > args.budget_study_s:
            study_budget_hit += 1
            untestable[
                f"not tested: the {args.budget_study_s:.0f}s study budget expired before this candidate was reached"
            ] += 1
            continue
        try:
            res = verify(db, c.hypothesis(), as_of=as_of, max_audit_samples=0)
        except Exception as exc:  # noqa: BLE001
            untestable[f"study raised {type(exc).__name__}"] += 1
            errors += 1
            continue
        if not res.reconciled:
            untestable["the study could not reconcile its own observation counts"] += 1
            continue
        if res.n_events < args.min_events:
            untestable[f"fewer than {args.min_events} gradeable events at this threshold and lag"] += 1
            below_floor_n += 1
            continue
        if res.n_event_clusters < args.min_clusters:
            untestable[f"fewer than {args.min_clusters} distinct event dates at this threshold and lag"] += 1
            below_floor_c += 1
            continue
        p = res.p_value_clustered if math.isfinite(res.p_value_clustered) else res.p_value
        if not math.isfinite(p):
            untestable["the bootstrap returned no p-value"] += 1
            continue
        pending.append((c, res))
        if i % max(1, len(study_order) // 25) == 0 or i == len(study_order):
            print(
                f"[stage 4] {i:,}/{len(study_order):,} studied, {len(pending):,} testable, "
                f"{time.time() - t_start:.0f}s elapsed"
            )
    if study_budget_hit:
        print(f"[stage 4] STUDY BUDGET EXPIRED: {study_budget_hit:,} candidates were never run")
    filters.append(
        FilterStep(
            "the study produced a valid p-value above the floors",
            examined=len(stage3),
            dropped=len(stage3) - len(pending),
            reason=f"per-candidate floors on n_events ({args.min_events}) and distinct event "
            f"dates ({args.min_clusters}), plus the study's own reconciliation; "
            f"{below_floor_n} failed on events, {below_floor_c} on dates, {errors} raised, "
            f"{study_budget_hit} never reached within the time budget",
        )
    )
    print(f"[stage 4] {len(pending):,} chains tested; running the correction")

    # --- stage 5: one Benjamini-Hochberg over the whole tested family -------
    p_values = [(r.p_value_clustered if math.isfinite(r.p_value_clustered) else r.p_value) for _, r in pending]
    if p_values:
        bh = benjamini_hochberg(p_values, alpha=args.alpha)
        adjusted, rejected = list(bh.p_adjusted), list(bh.rejected)
    else:
        adjusted, rejected = [], []

    # F-18: print the full series and confirm it varies before trusting any of it.
    checks = [
        series_check("p (uncorrected)", p_values),
        series_check("p (BH-adjusted)", adjusted),
        series_check("power", [r.power for _, r in pending]),
        series_check("effect (edge)", [r.edge for _, r in pending]),
        series_check("n_events", [float(r.n_events) for _, r in pending]),
        series_check("n_clusters", [float(r.n_event_clusters) for _, r in pending]),
    ]
    print("[series] full-series variation check (LESSONS F-18):")
    for chk in checks:
        print(f"[series]   {chk.line()}")
    if any(chk.degenerate for chk in checks):
        print("[series] WARNING: a series above does not vary. Do not trust figures from it.")

    graph_cache: dict[str, Any] = {"db": db, "as_of": as_of}
    n_explained = 0
    for idx, ((cand, res), padj, rej) in enumerate(zip(pending, adjusted, rejected, strict=True)):
        want_route = bool(rej) or idx < args.explain_top
        route = (
            _route_for(cand, res, graph_cache)
            if want_route
            else {"route": (), "hops": 0, "independent": 0, "sources": (), "error": "not requested"}
        )
        if want_route:
            n_explained += 1
        returns = [r for r in res.event_returns if math.isfinite(r)]
        sigma = statistics.pstdev(returns) if len(returns) >= 2 else float("nan")
        p_unc = res.p_value_clustered if math.isfinite(res.p_value_clustered) else res.p_value
        notes = [
            "p is the cluster-resampled bootstrap p"
            if math.isfinite(res.p_value_clustered)
            else "p is the i.i.d. bootstrap p; the clustered one was not computable",
        ]
        if not res.bootstrap_time_ordered:
            notes.append("the bootstrap sample was not time-ordered, so the block CI is weaker")
        if res.n_links_undatable:
            notes.append(
                f"{res.n_links_undatable} routing links have a NULL effective_from and could not be date-gated"
            )
        if p_unc == 0.0:
            notes.append(
                "the bootstrap p is exactly 0: no resample reached the observed statistic, "
                "so the true p is below 1/n_resamples rather than zero"
            )
        if res.n_event_clusters and res.n_events / res.n_event_clusters > 5:
            notes.append(
                f"{res.n_events / res.n_event_clusters:.0f} gradeable events share each distinct "
                f"event date ({res.max_events_per_cluster} on the busiest), so the nominal n is "
                "not a count of independent observations"
            )
        if route.get("error"):
            notes.append(f"mechanism route: {route['error']}")
        tested.append(
            ChainResult(
                chain_id=cand.chain_id,
                name=cand.name,
                route=tuple(route["route"]),
                hops=int(route["hops"]),
                independent_sources=int(route["independent"]),
                evidence_sources=tuple(route["sources"]),
                lag_steps=cand.horizon,
                lag_label=cand.target.steps_label,
                market=cand.target.market,
                mechanism=(
                    f"{cand.event.entity_type}/{cand.event.source}.{cand.event.obs_type}"
                    f" -> {cand.target.entity_type}.{cand.target.value_field}"
                ),
                n_events=res.n_events,
                n_clusters=res.n_event_clusters,
                sigma=sigma,
                power=res.power if math.isfinite(res.power) else 0.0,
                p_uncorrected=float(p_unc),
                p_adjusted=float(padj),
                rejected=bool(rej),
                effect=res.edge,
                event_mean=res.mean_event_return,
                baseline=res.baseline_mean_return,
                ci_low=res.ci_low,
                ci_high=res.ci_high,
                falsifier=_falsifier(cand, res),
                route_is_scaffold_only=bool(route.get("scaffold_only")),
                notes=tuple(notes),
            )
        )

    board = Scoreboard(
        title="Ghost chain discovery — automatic run",
        generated_at=dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        db_path=db,
        as_of_label=as_of_label,
        alpha=args.alpha,
        min_events=args.min_events,
        min_clusters=args.min_clusters,
        enumerated=enumerated,
        filters=tuple(filters),
        untestable=tuple(UntestableReason(reason=r, count=n) for r, n in sorted(untestable.items())),
        tested=tuple(tested),
        grid={
            "streams": f"{len(streams)} numeric event streams discovered from the live DB",
            "targets": ", ".join(f"{t.entity_type}.{t.value_field}" for t in targets),
            "z": ", ".join(f"{z:g}" for z in z_grid),
            "direction": ", ".join(dir_grid),
            "lag": ", ".join(str(h) for h in h_grid),
            "product": f"{len(streams)} x {len(targets)} x {len(z_grid)} x {len(dir_grid)} x "
            f"{len(h_grid)} = {enumerated:,}",
        },
        caveats=(
            "Event studies here grade a DIRECT graph link; the multi-hop route is the "
            "explanation, not the test, and a route is computed for survivors and the "
            f"{args.explain_top} nearest misses only "
            f"({n_explained} routes computed this run).",
            "The value_json field discovery samples "
            f"{VALUE_JSON_SAMPLE} rows per stream, so a field appearing only in later rows "
            "would have been left out of the enumeration entirely.",
            f"Time budgets: {args.budget_probe_s:.0f}s for screening and "
            f"{args.budget_study_s:.0f}s for testing. Candidates not reached are counted as "
            "untestable under a named reason, never dropped; the processing order is a "
            f"seeded shuffle (seed {args.seed}) so an expiry cannot correlate with any "
            "property of the candidates.",
            "Probe-based screening assumes the gradeable-event count is monotone in the "
            "threshold and the horizon. It is monotone in the threshold by construction; for "
            "the horizon it holds because a longer horizon needs a later exit observation.",
        ),
        provenance={
            "graph": f"{n_entities:,} entities, {n_links:,} links, {n_obs:,} observations",
            "engine": "agent.verify.study.verify + agent.verify.stats.benjamini_hochberg",
            "mechanism": "agent.mechanism.routes.top_routes + independence",
            "enumerator": "scripts/ghost_discovery_run.py fallback (cartesian product)",
            "field_sample": f"{VALUE_JSON_SAMPLE} rows per stream, >= {MIN_FIELD_HITS} hits",
            "runtime_s": f"{time.time() - t_start:.0f}",
        },
        checks=tuple(checks),
    )

    if args.probe_log:
        Path(args.probe_log).write_text(json.dumps(probe_rows, indent=2), encoding="utf-8")
        print(f"[out] probe log -> {args.probe_log}")
    _emit(board, args)
    return 0


def _emit(board: Scoreboard, args: argparse.Namespace) -> None:
    """Print the summary, render the page, and write whatever was asked for."""
    print()
    print(summarise(board))
    if args.board_json:
        Path(args.board_json).write_text(board_to_json(board), encoding="utf-8")
        print(f"[out] board JSON -> {args.board_json}")
    markdown = render_scoreboard(board)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"\n[out] {len(markdown):,} bytes of markdown -> {args.out}")
    else:
        print()
        print(markdown)


def board_to_json(board: Scoreboard) -> str:
    """Serialise a Scoreboard so a run's page can be re-rendered without recomputing.

    A full run is roughly twenty minutes of event studies. Re-rendering the same
    numbers after a wording change must not require re-running them, because a
    re-run invites quietly accepting whichever wording produced the nicer figures.
    The JSON is the run's result, frozen; `--render-from` reads it back.
    """
    return json.dumps(
        {
            "scalars": {
                k: getattr(board, k)
                for k in (
                    "title",
                    "generated_at",
                    "db_path",
                    "as_of_label",
                    "alpha",
                    "min_events",
                    "min_clusters",
                    "enumerated",
                )
            },
            "filters": [dc_asdict(f) for f in board.filters],
            "untestable": [dc_asdict(u) for u in board.untestable],
            "tested": [dc_asdict(c) for c in board.tested],
            "grid": dict(board.grid),
            "caveats": list(board.caveats),
            "provenance": dict(board.provenance),
            "checks": [dc_asdict(c) for c in board.checks],
        },
        indent=1,
    )


def board_from_json(text: str) -> Scoreboard:
    """Rebuild a Scoreboard from `board_to_json`, re-running every invariant.

    The reconciliation and the floor checks run again on the way back in, so a
    hand-edited JSON cannot be rendered into a flattering page.
    """
    doc = json.loads(text)
    return Scoreboard(
        **doc["scalars"],
        filters=tuple(FilterStep(**f) for f in doc["filters"]),
        untestable=tuple(UntestableReason(**u) for u in doc["untestable"]),
        tested=tuple(
            ChainResult(
                **{
                    **c,
                    "route": tuple(c["route"]),
                    "evidence_sources": tuple(c["evidence_sources"]),
                    "notes": tuple(c["notes"]),
                }
            )
            for c in doc["tested"]
        ),
        grid=doc["grid"],
        caveats=tuple(doc["caveats"]),
        provenance=doc["provenance"],
        checks=tuple(SeriesCheck(**c) for c in doc["checks"]),
    )


def _as_of_epoch(value: str) -> float:
    from agent.mechanism.graph import as_of_from_iso

    return as_of_from_iso(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=DEFAULT_DB, help="pipeline.db path (opened read-only)")
    p.add_argument("--as-of", default=None, help="ISO date bound, e.g. 2025-06-30; default is the live graph")
    p.add_argument("--alpha", type=float, default=0.05, help="Benjamini-Hochberg FDR level")
    p.add_argument("--min-events", type=int, default=DEFAULT_MIN_EVENTS)
    p.add_argument("--min-clusters", type=int, default=DEFAULT_MIN_CLUSTERS)
    p.add_argument("--out", default=None, help="write the markdown here instead of stdout")
    p.add_argument("--probe-log", default=None, help="write the per-base probe log as JSON")
    p.add_argument("--explain-top", type=int, default=10, help="nearest misses to compute a route for")
    p.add_argument("--max-streams", type=int, default=0, help="truncate the stream list (debug)")
    p.add_argument("--max-targets", type=int, default=0, help="truncate the target list (debug)")
    p.add_argument("--budget-probe-s", type=float, default=1200.0, help="wall-clock budget for screening")
    p.add_argument("--budget-study-s", type=float, default=3600.0, help="wall-clock budget for testing")
    p.add_argument("--seed", type=int, default=0, help="seed for the outcome-independent processing order")
    p.add_argument("--board-json", default=None, help="freeze the run's numbers as JSON")
    p.add_argument(
        "--render-from",
        default=None,
        help="re-render a page from a --board-json file, running no studies at all",
    )
    p.add_argument("--smoke", action="store_true", help="one threshold, one direction, one lag")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.render_from:
        board = board_from_json(Path(args.render_from).read_text(encoding="utf-8"))
        _emit(board, args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
