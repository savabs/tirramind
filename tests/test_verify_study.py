"""Tests for agent/verify/study.py.

Each test names the specific wrong behaviour it exists to catch, because four tests in
this repo have historically asserted the bug they were meant to prevent. The fixtures
plant values whose CORRECT answer differs in SIGN from the answer a broken
implementation gives, so a test cannot pass by accident.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from agent.verify.study import (
    EVENT_DROP_REASONS,
    Hypothesis,
    _bootstrap_p_clustered,
    _bootstrap_p_iid,
    _dig,
    _load_series,
    _numeric,
    _reconciles,
    benjamini_hochberg,
    causal_zscore,
    power_two_sided,
    verify,
)

REPO = Path(__file__).resolve().parents[1]
LIVE_DB = REPO / ".tirra_pipeline" / "pipeline.db"
DAY = 86400.0
T0 = dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp()

SCHEMA = """
create table entities (
    entity_id text primary key, entity_type text not null, canonical_name text not null,
    created_at real not null, metadata_json text);
create table entity_observations (
    id integer primary key autoincrement, entity_id text not null, source_tool text not null,
    observed_at real not null, ingested_at real not null, observation_type text not null,
    depth_level integer not null default 1, value_json text not null, metadata_json text);
create table entity_links (
    link_id integer primary key autoincrement, entity_id_a text not null, entity_id_b text not null,
    link_type text not null, confidence real not null default 1.0, source text not null,
    created_at real not null, metadata_json text, effective_from real);
"""


def build_db(path: Path, entities, links, observations) -> str:
    """Write a minimal pipeline.db. Returns the path as a string."""
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany(
        "insert into entities (entity_id, entity_type, canonical_name, created_at, metadata_json) values (?,?,?,0,?)",
        entities,
    )
    con.executemany(
        "insert into entity_links (entity_id_a, entity_id_b, link_type, source, created_at, effective_from) "
        "values (?,?,?,'test',0,?)",
        links,
    )
    con.executemany(
        "insert into entity_observations (entity_id, source_tool, observed_at, ingested_at, "
        "observation_type, value_json) values (?,?,?,?,?,?)",
        [(eid, src, ts, ts, otype, json.dumps(val)) for eid, src, otype, ts, val in observations],
    )
    con.commit()
    con.close()
    return str(path)


# --- the canonical fixture -------------------------------------------------------------
# One country, one instrument, linked instrument --produced_in--> country.
# Event series: 30 daily points alternating 0/1, with a single spike of 5.0 at index 28.
#   -> exactly one point crosses |z| >= 2, at event_ts = T0 + 28d.
# Publication lag is 2 days, so publication_ts = T0 + 30d.
# Target closes are planted so the CORRECT entry (the close at T0+30d) and the WRONG
# entry (the as-of close at T0+28d) give forward returns of OPPOSITE SIGN.
SPIKE_IDX = 28
EVENT_TS = T0 + SPIKE_IDX * DAY
PUB_LAG = 2 * DAY
PUB_TS = EVENT_TS + PUB_LAG


def event_values() -> list[float]:
    vals = [float(i % 2) for i in range(30)]
    vals[SPIKE_IDX] = 5.0
    return vals


def canonical_db(tmp_path: Path, *, n_closes: int = 61, close_override: dict[int, float] | None = None) -> str:
    closes = {i: 100.0 for i in range(n_closes)}
    closes.update({28: 100.0, 29: 90.0, 30: 50.0, 31: 80.0})  # see comment above
    if close_override:
        closes.update(close_override)
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": c}) for i, c in sorted(closes.items())]
    return build_db(
        tmp_path / "p.db",
        [("C1", "country", "Coldland", None), ("I1", "instrument", "Widget", None)],
        [("I1", "C1", "produced_in", None)],
        obs,
    )


def canonical_hypothesis(**over) -> Hypothesis:
    kw = dict(
        event_source="src",
        event_obs_type="evt",
        event_field="f",
        z_threshold=2.0,
        direction="abs",
        target_entity_type="instrument",
        target_obs_type="instrument_daily",
        horizon_days=1,
        publication_lag_s=PUB_LAG,
        min_history=20,
        min_target_points=40,
    )
    kw.update(over)
    return Hypothesis(**kw)


# =====================================================================================
# 1. the z-score: causal, and identical to the reference
# =====================================================================================


def _reference_module():
    spec = importlib.util.spec_from_file_location("_ref_cftc", REPO / "scripts" / "cftc_event_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_zscore_matches_reference_implementation_exactly():
    """Catches any drift from scripts/cftc_event_study.py's zscore_causal (ddof, +1e-12, x[:-1])."""
    ref = _reference_module()
    rng = np.random.default_rng(0)
    checked = 0
    for n in (19, 20, 21, 40, 137):
        for scale in (1e-6, 1.0, 1e6):
            series = list(rng.normal(3.0, scale, size=n))
            mine = causal_zscore(series, min_history=20)
            theirs = ref.zscore_causal(series)
            assert (mine is None) == (theirs is None), (n, scale)
            if mine is not None:
                assert mine == pytest.approx(theirs, rel=1e-12, abs=1e-12)
                checked += 1
    assert checked >= 9, "the comparison never actually ran on a computable z"


def test_zscore_excludes_the_current_point_from_its_own_history():
    """Catches hist = x (leakage). The two answers differ by more than a factor of two here."""
    series = [0.0, 1.0] * 10 + [10.0]  # history mean 0.5, population std 0.5
    z = causal_zscore(series, min_history=20)
    assert z == pytest.approx((10.0 - 0.5) / 0.5, rel=1e-9)  # 19.0
    contaminated = np.asarray(series)
    leaky = (contaminated[-1] - contaminated.mean()) / contaminated.std()
    assert abs(leaky) < 0.6 * abs(z), "fixture no longer separates causal from leaky"


def test_zscore_returns_none_never_zero_when_not_computable():
    """A silent 0.0 turns 'we cannot tell' into 'nothing happened' — F-17's signature."""
    assert causal_zscore([1.0] * 19, min_history=20) is None  # too short
    assert causal_zscore([2.0] * 25, min_history=20) is None  # degenerate history std
    assert causal_zscore([1.0] * 24 + [float("nan")], min_history=20) is None
    assert causal_zscore([1.0] * 24 + [float("inf")], min_history=20) is None


def test_zscore_min_history_is_a_floor_not_a_window():
    """20 points must be ENOUGH, and the window must keep expanding past it."""
    assert causal_zscore([0.0, 1.0] * 10, min_history=20) is not None
    long_series = [0.0, 1.0] * 50 + [4.0]
    short_series = [0.0, 1.0] * 10 + [4.0]
    # same mean/std by construction, so an expanding window gives the same z; a rolling
    # 20-window would too. What must NOT happen is a None for the long series.
    assert causal_zscore(long_series, 20) == pytest.approx(causal_zscore(short_series, 20))


# =====================================================================================
# 2. multiplicity and power
# =====================================================================================


def test_benjamini_hochberg_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(1)
    for m in (1, 5, 51, 200):
        p = rng.random(m) ** 3
        rej, adj = benjamini_hochberg(p, alpha=0.05)
        ref_rej, ref_adj, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
        assert np.allclose(adj, ref_adj), m
        assert np.array_equal(rej, ref_rej), m


def test_benjamini_hochberg_reproduces_the_published_cot_arithmetic():
    """p=0.002, rank 1 of m=51 -> 0.102. Straight from docs/publications/cot_null_result.md."""
    rej, adj = benjamini_hochberg([0.002] + [0.5] * 50, alpha=0.05)
    assert adj[0] == pytest.approx(0.102, abs=5e-4)
    assert not rej.any()


def test_benjamini_hochberg_does_not_let_nonfinite_pvalues_inflate_the_family():
    """Coercing NaN to 1.0 would make m=3 here and shrink the adjusted p."""
    rej, adj = benjamini_hochberg([0.01, float("nan"), float("nan")])
    assert np.isnan(adj[1]) and np.isnan(adj[2])
    assert adj[0] == pytest.approx(0.01)  # m = 1, not 3
    assert bool(rej[0]) and not rej[1] and not rej[2]
    assert benjamini_hochberg([float("nan")])[0].sum() == 0


def test_power_reproduces_the_published_power_table():
    """Section 7.1: n=123 -> 9.6%, n=160 -> 11.0%, n=116 -> ~9.3%."""
    assert power_two_sided(123) == pytest.approx(0.096, abs=0.001)
    assert power_two_sided(160) == pytest.approx(0.110, abs=0.001)
    assert power_two_sided(116) == pytest.approx(0.093, abs=0.001)
    assert power_two_sided(1451) == pytest.approx(0.576, abs=0.01)  # t = reference_t = 2.14
    assert np.isnan(power_two_sided(0))
    # monotone in n, and the sensitivity knob moves it
    ns = [10, 50, 123, 500, 2000]
    powers = [power_two_sided(n) for n in ns]
    assert powers == sorted(powers)
    assert len(set(np.round(powers, 4))) == len(ns), f"power does not vary with n: {powers}"
    assert power_two_sided(123, reference_t=3.0) > power_two_sided(123, reference_t=1.5)


# =====================================================================================
# 3. the hypothesis object refuses nonsense
# =====================================================================================


@pytest.mark.parametrize(
    "over",
    [
        {"direction": "sideways"},
        {"event_aggregate": "median"},
        {"target_aggregate": "median"},
        {"z_threshold": -1.0},
        {"horizon_days": 0},
        {"publication_lag_s": -1.0},
        {"min_history": 1},
        {"entry_tolerance_s": -1.0},
    ],
)
def test_hypothesis_rejects_invalid_specifications(over):
    with pytest.raises(ValueError):
        canonical_hypothesis(**over)


def test_hypothesis_describe_states_the_whole_specification():
    text = canonical_hypothesis(direction="down", z_threshold=2.5).describe()
    for token in ("src", "evt", "f", "2.5", "instrument", "instrument_daily", "close", "1 steps"):
        assert token in text, text


# =====================================================================================
# 4. publication lag — the leakage guard
# =====================================================================================


def test_entry_is_the_first_close_after_publication_not_the_as_of_close(tmp_path):
    """Catches entry at the event timestamp. The two give forward returns of opposite sign."""
    r = verify(canonical_db(tmp_path), canonical_hypothesis())
    assert r.n_events == 1, r.summary()
    s = r.leakage_audit[0]
    assert s.event_ts == EVENT_TS
    assert s.publication_ts == PUB_TS
    assert s.entry_ts == PUB_TS, "entry did not land on the first close at/after publication"
    assert s.entry_price == 50.0, "entry used a close other than the first post-publication one"
    assert s.exit_ts == PUB_TS + DAY
    assert r.mean_event_return == pytest.approx(np.log(80.0 / 50.0), rel=1e-12)
    assert r.mean_event_return > 0
    # the wrong answer an as-of entry would have produced:
    assert np.log(90.0 / 100.0) < 0


def test_every_leakage_audit_triple_is_ordered(tmp_path):
    r = verify(canonical_db(tmp_path), canonical_hypothesis(z_threshold=0.0))
    assert r.leakage_audit
    for s in r.leakage_audit:
        assert s.event_ts <= s.publication_ts <= s.entry_ts < s.exit_ts, s.as_row()
        assert s.ordered
    assert "event=" in r.leakage_audit[0].as_row()


def test_a_bad_triple_would_be_detected():
    """The audit's own check must be capable of failing; otherwise it asserts nothing."""
    from agent.verify.study import LeakageSample

    bad = LeakageSample(
        "a",
        "b",
        event_ts=100.0,
        publication_ts=50.0,
        entry_ts=60.0,
        exit_ts=70.0,
        z=3.0,
        entry_price=1.0,
        log_return=0.0,
    )
    assert not bad.ordered


def test_event_is_dropped_not_shifted_when_no_close_lands_within_tolerance(tmp_path):
    """Catches 'snap to the next available close', which silently buys weeks later."""
    closes = {i: 100.0 for i in range(28)}
    closes.update({i: 100.0 for i in range(45, 80)})  # 14-day hole straddling publication
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": c}) for i, c in sorted(closes.items())]
    db = build_db(
        tmp_path / "hole.db",
        [("C1", "country", "Coldland", None), ("I1", "instrument", "Widget", None)],
        [("I1", "C1", "produced_in", None)],
        obs,
    )
    r = verify(db, canonical_hypothesis(entry_tolerance_s=7 * DAY))
    assert r.n_fired == 1
    assert r.n_events == 0, "an event with no close inside tolerance was graded anyway"
    assert r.drop_reasons["event_no_entry_close_within_tolerance"] >= 1
    assert any("event_no_entry_close_within_tolerance" in e for e in r.excluded)
    assert r.verdict == "not_testable"
    # widening the tolerance grades it — proving the drop was the tolerance, not a bug
    r2 = verify(db, canonical_hypothesis(entry_tolerance_s=30 * DAY))
    assert r2.n_events == 1
    assert r2.leakage_audit[0].entry_ts > PUB_TS


def test_zero_publication_lag_is_honoured_when_asserted(tmp_path):
    """A lag of 0 must mean the as-of close, not a silently defaulted lag."""
    r = verify(canonical_db(tmp_path), canonical_hypothesis(publication_lag_s=0.0))
    s = r.leakage_audit[0]
    assert s.publication_ts == EVENT_TS
    assert s.entry_ts == EVENT_TS
    assert s.entry_price == 100.0
    assert r.mean_event_return == pytest.approx(np.log(90.0 / 100.0))  # opposite sign to the lagged run
    assert r.mean_event_return < 0


# =====================================================================================
# 5. the baseline is unconditional, not zero
# =====================================================================================


def test_baseline_is_the_unconditional_return_distribution(tmp_path):
    """Catches a zero null, which in a trending market manufactures an edge from drift."""
    n = 80
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0 * (1.01**i)}) for i in range(n)]
    db = build_db(
        tmp_path / "drift.db",
        [("C1", "country", "C", None), ("I1", "instrument", "I", None)],
        [("I1", "C1", "produced_in", None)],
        obs,
    )
    r = verify(db, canonical_hypothesis())
    assert r.n_baseline == r.n_z_computable, "not every z-computable entry entered the baseline"
    assert r.n_baseline > r.n_events, "the baseline is not unconditional; it equals the event set"
    assert r.baseline_mean_return == pytest.approx(np.log(1.01), rel=1e-9)
    assert r.baseline_mean_return != 0.0
    # the event return here IS the drift, so the honest edge is zero even though the
    # raw event return is a healthy +1%
    assert r.mean_event_return == pytest.approx(np.log(1.01), rel=1e-9)
    assert r.edge == pytest.approx(0.0, abs=1e-12)
    assert r.edge == pytest.approx(r.mean_event_return - r.baseline_mean_return, abs=1e-15)
    assert r.baseline_returns and len(r.baseline_returns) == r.n_baseline
    assert r.hit_rate_baseline == 1.0


def test_events_below_threshold_are_baseline_not_drops(tmp_path):
    r = verify(canonical_db(tmp_path), canonical_hypothesis())
    assert r.n_below_threshold > 0
    assert r.n_below_threshold + r.n_fired == r.n_z_computable
    assert "event_below_threshold" not in r.drop_reasons
    assert not any("below_threshold" in e for e in r.excluded)


# =====================================================================================
# 6. attrition, reconciliation and completeness
# =====================================================================================


def test_attrition_reconciles_and_every_bucket_is_named(tmp_path):
    r = verify(canonical_db(tmp_path), canonical_hypothesis())
    assert r.reconciled, r.summary()
    entered = sum(p.n_event_points for p in r.pair_attrition if p.used)
    assert entered == r.drop_reasons.get("event_z_not_computable", 0) + r.n_z_computable
    assert set(r.drop_reasons) <= set(EVENT_DROP_REASONS) | {
        "pair_no_event_points",
        "pair_event_history_below_min",
        "pair_target_points_below_min",
        "pair_no_calendar_overlap",
    }
    assert r.n_pairs_entered == len(r.pair_attrition) == 1
    assert r.n_pairs_used == 1
    assert r.pair_attrition[0].status == "OK" and r.pair_attrition[0].used


def test_pairs_are_dropped_on_data_availability_with_a_stated_reason(tmp_path):
    """Four distinct pair-level failures must each surface a distinct named status."""
    entities = [
        ("C1", "country", "C", None),
        ("SHORT", "instrument", "TooFewCloses", None),
        ("EARLY", "instrument", "EndsTooEarly", None),
        ("LATE", "instrument", "StartsTooLate", None),
        ("EMPTY", "instrument", "NoClosesAtAll", None),
    ]
    links = [(t, "C1", "produced_in", None) for t, *_ in [("SHORT",), ("EARLY",), ("LATE",), ("EMPTY",)]]
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    obs += [("SHORT", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(10)]
    obs += [("EARLY", "prices", "instrument_daily", T0 - (200 - i) * DAY, {"close": 100.0}) for i in range(60)]
    obs += [("LATE", "prices", "instrument_daily", T0 + (400 + i) * DAY, {"close": 100.0}) for i in range(60)]
    db = build_db(tmp_path / "pairs.db", entities, links, obs)
    r = verify(db, canonical_hypothesis())
    statuses = {p.target_entity_id: p.status for p in r.pair_attrition}
    assert statuses["SHORT"].startswith("pair_target_points_below_min")
    assert statuses["EMPTY"].startswith("pair_target_points_below_min")
    assert statuses["EARLY"].startswith("pair_no_calendar_overlap")
    assert statuses["LATE"].startswith("pair_no_calendar_overlap")
    assert r.n_pairs_used == 0 and r.n_pairs_entered == 4
    assert r.drop_reasons["pair_no_calendar_overlap"] == 2
    assert r.drop_reasons["pair_target_points_below_min"] == 2
    assert r.n_events == 0 and r.verdict == "not_testable"
    assert r.reconciled, "zero usable pairs must still reconcile"
    for p in r.pair_attrition:
        assert not p.used and p.n_target_points >= 0


def test_pair_dropped_when_event_history_cannot_support_a_zscore(tmp_path):
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": float(i)}) for i in range(5)]
    obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
    db = build_db(
        tmp_path / "short.db",
        [("C1", "country", "C", None), ("I1", "instrument", "I", None)],
        [("I1", "C1", "produced_in", None)],
        obs,
    )
    r = verify(db, canonical_hypothesis())
    assert r.pair_attrition[0].status.startswith("pair_event_history_below_min")
    assert r.drop_reasons["pair_event_history_below_min"] == 1
    assert any("pair_event_history_below_min" in e for e in r.excluded)


def test_complete_is_false_whenever_anything_was_excluded(tmp_path):
    r = verify(canonical_db(tmp_path), canonical_hypothesis())
    assert not r.complete
    assert r.excluded
    # the 19 warm-up points are structural (min_history=20, one pair) and are NOT an
    # exclusion; what makes this run incomplete is the undatable link and the fact that
    # a single event is too few to test.
    assert r.drop_reasons["event_z_not_computable"] == 19
    assert r.n_warmup_excluded == 19 and r.n_degenerate_excluded == 0
    assert any("NULL effective_from" in e for e in r.excluded)
    assert any("not tested" in e for e in r.excluded)


def test_complete_is_true_only_when_no_gradable_evidence_was_lost(tmp_path):
    """A dense calendar and a healthy variance leave nothing behind but the structural
    warm-up, so `complete` must be True. If it can never be True it is a dead flag."""
    n = 60
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": float(i % 3)}) for i in range(1, n)]
    obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0 + i}) for i in range(0, n + 40)]
    db = build_db(
        tmp_path / "clean.db",
        [("C1", "country", "C", None), ("I1", "instrument", "I", None)],
        [("I1", "C1", "produced_in", 0.0)],
        obs,
    )
    r = verify(db, canonical_hypothesis(min_history=3, z_threshold=0.0, publication_lag_s=0.0, min_target_points=40))
    assert r.drop_reasons == {"event_z_not_computable": 2}, r.summary()
    assert r.n_warmup_excluded == 2 and r.n_degenerate_excluded == 0
    assert r.excluded == [], r.excluded
    assert r.complete and r.reconciled
    assert r.n_events == r.n_z_computable == r.n_baseline

    # and a series with a flat stretch loses REAL evidence, which must flip `complete`
    flat = [("C1", "src", "evt", T0 + i * DAY, {"f": 1.0}) for i in range(0, 25)]
    flat += [("C1", "src", "evt", T0 + (25 + i) * DAY, {"f": float(i % 3)}) for i in range(35)]
    flat += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0 + i}) for i in range(0, 110)]
    db2 = build_db(
        tmp_path / "flat.db",
        [("C1", "country", "C", None), ("I1", "instrument", "I", None)],
        [("I1", "C1", "produced_in", 0.0)],
        flat,
    )
    r2 = verify(db2, canonical_hypothesis(min_history=3, z_threshold=0.0, publication_lag_s=0.0))
    assert r2.n_degenerate_excluded > 0
    assert not r2.complete
    assert any("degenerate history variance" in e for e in r2.excluded)


def test_as_of_is_itself_recorded_as_an_exclusion(tmp_path):
    db = canonical_db(tmp_path)
    r = verify(db, canonical_hypothesis(), as_of=T0 + 40 * DAY)
    assert not r.complete
    assert any("as_of" in e for e in r.excluded)
    assert r.as_of == T0 + 40 * DAY


# =====================================================================================
# 7. repeated timestamps must not vanish quietly
# =====================================================================================


def test_repeated_timestamps_are_collapsed_loudly_and_the_aggregate_is_honoured(tmp_path):
    """gdelt files dozens of rows per country-day. 'last' throws them away; say so."""
    obs = [
        ("C1", "src", "evt", T0, {"f": 1.0}),
        ("C1", "src", "evt", T0, {"f": 3.0}),
        ("C1", "src", "evt", T0, {"f": 5.0}),
    ]
    obs += [("C1", "src", "evt", T0 + DAY, {"f": 7.0})]
    db = build_db(tmp_path / "dupe.db", [("C1", "country", "C", None)], [], obs)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        expect = {
            "last": [5.0, 7.0],
            "mean": [3.0, 7.0],
            "sum": [9.0, 7.0],
            "count": [3.0, 1.0],
            "max": [5.0, 7.0],
            "min": [1.0, 7.0],
        }
        for agg, want in expect.items():
            s = _load_series(con, "C1", "evt", "src", "f", agg, None)
            assert list(s.val) == want, agg
            assert s.n_raw == 4
            assert len(s) == 2
            assert s.n_collapsed == (2 if agg in ("last", "max", "min") else 0), agg
    finally:
        con.close()


def test_lossy_aggregation_is_reported_in_excluded(tmp_path):
    base = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    dupes = [("C1", "src", "evt", T0 + 3 * DAY, {"f": 99.0})]  # a second row on one day
    closes = [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
    db = build_db(
        tmp_path / "lossy.db",
        [("C1", "country", "C", None), ("I1", "instrument", "I", None)],
        [("I1", "C1", "produced_in", None)],
        base + dupes + closes,
    )
    r_last = verify(db, canonical_hypothesis(event_aggregate="last"))
    assert r_last.n_collapsed_observations == 1
    assert any("discarded by lossy aggregation" in e for e in r_last.excluded)
    r_mean = verify(db, canonical_hypothesis(event_aggregate="mean"))
    assert r_mean.n_collapsed_observations == 0
    assert not any("lossy aggregation" in e for e in r_mean.excluded)


def test_unparsable_and_nonnumeric_observations_are_counted_not_swallowed(tmp_path):
    path = tmp_path / "junk.db"
    build_db(path, [("C1", "country", "C", None)], [], [])
    con = sqlite3.connect(path)
    con.executemany(
        "insert into entity_observations (entity_id, source_tool, observed_at, ingested_at, observation_type, value_json) values (?,?,?,?,?,?)",
        [
            ("C1", "src", T0, T0, "evt", "{not json"),
            ("C1", "src", T0 + DAY, T0, "evt", "[1,2,3]"),
            ("C1", "src", T0 + 2 * DAY, T0, "evt", json.dumps({"f": "9"})),
            ("C1", "src", T0 + 3 * DAY, T0, "evt", json.dumps({"f": True})),
            ("C1", "src", T0 + 4 * DAY, T0, "evt", json.dumps({"other": 1.0})),
            ("C1", "src", T0 + 5 * DAY, T0, "evt", json.dumps({"f": 2.0})),
        ],
    )
    con.commit()
    con.close()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        s = _load_series(con, "C1", "evt", "src", "f", "last", None)
    finally:
        con.close()
    assert list(s.val) == [2.0], "a string, a bool or a missing key was coerced into a number"
    assert s.n_unparsable == 2
    assert s.n_missing_field == 3
    assert s.n_raw == 6


def test_numeric_and_dig_helpers_reject_what_they_should():
    assert _numeric(True) is None and _numeric(False) is None
    assert _numeric("1.0") is None and _numeric(None) is None
    assert _numeric(float("nan")) is None and _numeric(float("inf")) is None
    assert _numeric(3) == 3.0 and _numeric(-0.5) == -0.5
    assert _dig({"a": {"b": 2}}, "a.b") == 2
    assert _dig({"a": {"b": 2}}, "a.c") is None
    assert _dig({"a": 1}, "a.b") is None
    assert _dig([1, 2], "a") is None


def test_nested_field_path_is_supported(tmp_path):
    obs = [("C1", "src", "evt", T0 + i * DAY, {"outer": {"inner": v}}) for i, v in enumerate(event_values())]
    obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"px": {"close": 100.0 + i}}) for i in range(60)]
    db = build_db(
        tmp_path / "nested.db",
        [("C1", "country", "C", None), ("I1", "instrument", "I", None)],
        [("I1", "C1", "produced_in", None)],
        obs,
    )
    r = verify(db, canonical_hypothesis(event_field="outer.inner", target_value_field="px.close"))
    assert r.n_events == 1 and r.n_baseline > 1


# =====================================================================================
# 8. direction
# =====================================================================================


def test_direction_selects_different_event_sets(tmp_path):
    """Catches a direction that is ignored, or 'down' implemented as abs()."""
    up = event_values()
    down = [float(i % 2) for i in range(30)]
    down[SPIKE_IDX] = -5.0
    for vals, fires_up, fires_down in ((up, True, False), (down, False, True)):
        obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(vals)]
        obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
        db = build_db(
            tmp_path / f"dir{fires_up}.db",
            [("C1", "country", "C", None), ("I1", "instrument", "I", None)],
            [("I1", "C1", "produced_in", None)],
            obs,
        )
        assert (verify(db, canonical_hypothesis(direction="up")).n_fired == 1) is fires_up
        assert (verify(db, canonical_hypothesis(direction="down")).n_fired == 1) is fires_down
        assert verify(db, canonical_hypothesis(direction="abs")).n_fired == 1
        # the baseline must not change with the direction — it is unconditional
        pops = {d: verify(db, canonical_hypothesis(direction=d)).n_baseline for d in ("up", "down", "abs")}
        assert len(set(pops.values())) == 1, pops


# =====================================================================================
# 9. as_of — no lookahead in prices or in the graph
# =====================================================================================


def test_as_of_bounds_the_target_series_so_no_future_price_is_used(tmp_path):
    db = canonical_db(tmp_path)
    full = verify(db, canonical_hypothesis(horizon_days=5, min_target_points=20))
    assert full.n_events == 1
    cut = verify(db, canonical_hypothesis(horizon_days=5, min_target_points=20), as_of=PUB_TS + 2 * DAY)
    assert cut.n_events == 0, "a forward price after as_of was used"
    assert cut.drop_reasons["event_no_close_at_horizon"] >= 1
    assert cut.reconciled


def test_as_of_excludes_links_created_later(tmp_path):
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
    db = build_db(
        tmp_path / "dated.db",
        [("C1", "country", "C", None), ("I1", "instrument", "I", None)],
        [("I1", "C1", "produced_in", T0 + 100 * DAY)],
        obs,
    )
    assert verify(db, canonical_hypothesis()).n_pairs_entered == 1
    late = verify(db, canonical_hypothesis(), as_of=T0 + 50 * DAY)
    assert late.n_pairs_entered == 0, "a link that did not exist at as_of was used to route evidence"
    assert late.n_event_entities == 1
    assert late.n_events == 0


def test_events_predating_their_own_routing_link_are_counted_and_optionally_dropped(tmp_path):
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0 + i}) for i in range(60)]
    db = build_db(
        tmp_path / "late_link.db",
        [("C1", "country", "C", None), ("I1", "instrument", "I", None)],
        [("I1", "C1", "produced_in", T0 + 40 * DAY)],
        obs,
    )
    kept = verify(db, canonical_hypothesis())
    assert kept.n_events_before_link_known == 1
    assert kept.n_links_undatable == 0
    assert any("predate the effective_from" in e for e in kept.excluded)
    dropped = verify(db, canonical_hypothesis(require_link_predates_event=True))
    assert dropped.n_events == 0
    assert dropped.drop_reasons["event_link_not_yet_known"] == 1
    assert dropped.reconciled


def test_undatable_links_are_kept_but_counted(tmp_path):
    r = verify(canonical_db(tmp_path), canonical_hypothesis())
    assert r.n_links_undatable == 1
    assert r.n_pairs_used == 1, "an undatable geography link must still route"
    assert any("NULL effective_from" in e for e in r.excluded)


# =====================================================================================
# 10. link resolution
# =====================================================================================


def test_links_are_resolved_in_both_directions(tmp_path):
    """produced_in runs instrument->country; cftc_tracks runs contract->instrument.
    Reading only entity_id_a finds zero commodity pairs."""
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    obs += [("Ia", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
    obs += [("Ib", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
    db = build_db(
        tmp_path / "both.db",
        [("C1", "country", "C", None), ("Ia", "instrument", "A", None), ("Ib", "instrument", "B", None)],
        [("Ia", "C1", "produced_in", None), ("C1", "Ib", "affects", None)],
        obs,
    )
    r = verify(db, canonical_hypothesis())
    assert {p.target_entity_id for p in r.pair_attrition} == {"Ia", "Ib"}
    assert r.n_target_entities == 2


def test_link_types_filter_and_target_type_filter_are_applied(tmp_path):
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    obs += [("Ia", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
    obs += [("Ib", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
    obs += [("K", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
    db = build_db(
        tmp_path / "filters.db",
        [
            ("C1", "country", "C", None),
            ("Ia", "instrument", "A", None),
            ("Ib", "instrument", "B", None),
            ("K", "company", "NotAnInstrument", None),
        ],
        [("Ia", "C1", "produced_in", None), ("Ib", "C1", "rumoured_about", None), ("K", "C1", "produced_in", None)],
        obs,
    )
    assert {p.target_entity_id for p in verify(db, canonical_hypothesis()).pair_attrition} == {"Ia", "Ib"}
    only = verify(db, canonical_hypothesis(link_types=("produced_in",)))
    assert {p.target_entity_id for p in only.pair_attrition} == {"Ia"}
    assert only.pair_attrition[0].link_type == "produced_in"
    assert verify(db, canonical_hypothesis(target_entity_type="company")).n_pairs_entered == 1


def test_self_links_do_not_create_a_pair(tmp_path):
    obs = [("I1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    obs += [("I1", "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0}) for i in range(60)]
    db = build_db(
        tmp_path / "self.db",
        [("I1", "instrument", "I", None)],
        [("I1", "I1", "is_itself", None)],
        obs,
    )
    assert verify(db, canonical_hypothesis()).n_pairs_entered == 0


# =====================================================================================
# 11. inference
# =====================================================================================


def test_clustered_bootstrap_is_weaker_than_iid_when_events_cluster():
    """Catches an i.i.d.-only p-value sold as the honest one."""
    rng = np.random.default_rng(3)
    cluster_means = rng.normal(0.01, 0.02, size=6)
    events, clusters = [], []
    for c, m in enumerate(cluster_means):
        for _ in range(25):  # 25 near-identical targets firing on one date
            events.append(m + rng.normal(0, 1e-4))
            clusters.append(float(c))
    ev = np.asarray(events)
    cl = np.asarray(clusters)
    p_iid = _bootstrap_p_iid(ev, 0.0)
    p_cl = _bootstrap_p_clustered(ev, cl, 0.0)
    assert p_iid < 0.05, p_iid
    assert p_cl > p_iid * 3, (p_iid, p_cl)
    assert np.isnan(_bootstrap_p_clustered(ev, np.zeros_like(cl), 0.0)), "one cluster is not resampleable"


def test_pvalue_is_two_sided_and_bounded():
    rng = np.random.default_rng(4)
    same = rng.normal(0.0, 0.01, size=200)
    assert _bootstrap_p_iid(same, 0.0) > 0.3
    assert _bootstrap_p_iid(rng.normal(0.5, 0.01, size=200), 0.0) <= 0.01
    assert _bootstrap_p_iid(rng.normal(-0.5, 0.01, size=200), 0.0) <= 0.01  # negative edge, also significant
    for p in (_bootstrap_p_iid(same, 0.0), _bootstrap_p_iid(same, 0.5)):
        assert 0.0 <= p <= 1.0


def test_verdict_never_says_supported_without_the_uncorrected_caveat(tmp_path):
    r = verify(canonical_db(tmp_path), canonical_hypothesis())
    assert r.verdict in ("not_testable", "not_supported", "supported_uncorrected", "invalid_bookkeeping")
    assert r.verdict != "supported"
    assert r.n_events < 3 and r.verdict == "not_testable"
    assert np.isnan(r.p_value)


def test_too_few_events_is_not_tested_at_all(tmp_path):
    r = verify(canonical_db(tmp_path), canonical_hypothesis())
    assert r.n_events == 1
    assert np.isnan(r.p_value) and np.isnan(r.ci_low) and np.isnan(r.ci_high)
    assert any("not tested" in e for e in r.excluded)
    assert not np.isnan(r.mean_event_return), "the point estimate is still reported"


def test_no_events_at_all_is_a_result_not_a_crash(tmp_path):
    db = canonical_db(tmp_path)
    r = verify(db, canonical_hypothesis(z_threshold=100.0))
    assert r.n_fired == 0 and r.n_events == 0
    assert r.verdict == "not_testable"
    assert np.isnan(r.mean_event_return) and np.isnan(r.edge)
    assert r.n_baseline > 0, "the baseline is independent of whether anything fired"
    assert r.effective_sample_size == 0 and np.isnan(r.power)
    assert r.reconciled
    assert isinstance(r.summary(), str)


def test_unknown_source_yields_an_empty_but_valid_result(tmp_path):
    r = verify(canonical_db(tmp_path), canonical_hypothesis(event_source="nope"))
    assert r.n_event_entities == 0 and r.n_pairs_entered == 0 and r.reconciled
    assert r.verdict == "not_testable"


def test_clusters_measure_pseudo_replication(tmp_path):
    """Two instruments from one country firing on one date is one observation, not two."""
    obs = [("C1", "src", "evt", T0 + i * DAY, {"f": v}) for i, v in enumerate(event_values())]
    for t in ("Ia", "Ib", "Ic"):
        obs += [(t, "prices", "instrument_daily", T0 + i * DAY, {"close": 100.0 + i}) for i in range(60)]
    db = build_db(
        tmp_path / "clust.db",
        [("C1", "country", "C", None)] + [(t, "instrument", t, None) for t in ("Ia", "Ib", "Ic")],
        [(t, "C1", "produced_in", None) for t in ("Ia", "Ib", "Ic")],
        obs,
    )
    r = verify(db, canonical_hypothesis())
    assert r.n_events == 3, r.summary()
    assert r.n_event_clusters == 1, "three correlated targets on one date were counted as three dates"
    assert r.max_events_per_cluster == 3
    assert r.effective_sample_size == 1
    assert r.power < r.power_n_events, "power was computed from events rather than clusters"
    # one cluster cannot be resampled, so the clustered p must be NaN rather than a
    # quiet fallback to the i.i.d. p-value that the clustering was meant to correct
    assert np.isfinite(r.p_value)
    assert np.isnan(r.p_value_clustered), r.p_value_clustered


# =====================================================================================
# 12. read-only
# =====================================================================================


def test_verify_does_not_mutate_the_database(tmp_path):
    db = Path(canonical_db(tmp_path))
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    verify(str(db), canonical_hypothesis())
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_the_connection_really_is_read_only(tmp_path):
    from agent.verify.study import _connect_ro

    db = canonical_db(tmp_path)
    con = _connect_ro(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("insert into entities values ('X','y','z',0,null)")
    finally:
        con.close()


# =====================================================================================
# 13. live data
# =====================================================================================

pytestmark_live = pytest.mark.skipif(not LIVE_DB.exists(), reason="live pipeline.db not present")


def live_gdelt_hypothesis(**over) -> Hypothesis:
    kw = dict(
        event_source="gdelt",
        event_obs_type="geopolitical_event",
        event_field="goldstein",
        z_threshold=2.0,
        direction="abs",
        target_entity_type="instrument",
        target_obs_type="instrument_daily",
        horizon_days=5,
        publication_lag_s=DAY,
        link_types=("produced_in",),
        event_aggregate="mean",
    )
    kw.update(over)
    return Hypothesis(**kw)


@pytestmark_live
def test_live_gdelt_goldstein_study_is_internally_consistent():
    r = verify(str(LIVE_DB), live_gdelt_hypothesis())
    assert r.reconciled, r.summary()
    assert not r.complete and r.excluded
    assert r.n_event_entities >= 100
    assert r.n_pairs_used > 0 and r.n_pairs_used <= r.n_pairs_entered
    assert r.n_events >= 3 and r.n_events <= r.n_fired
    assert r.n_baseline > r.n_events
    assert 0.0 < r.hit_rate_baseline < 1.0
    assert r.ci_low < r.mean_event_return < r.ci_high
    assert r.n_event_clusters < r.n_events, "every event on its own date would be a surprise here"
    assert r.effective_sample_size == r.n_event_clusters
    assert r.power < 0.5, "a sample this size cannot be well powered"
    assert r.verdict in ("not_supported", "supported_uncorrected")
    for s in r.leakage_audit:
        assert s.ordered, s.as_row()
    assert len(r.event_returns) == r.n_events
    ev = np.asarray(r.event_returns)
    assert ev.std() > 1e-4, f"event returns do not vary: {ev[:20]}"
    # attrition must be dominated by the head gap, and must be stated
    assert r.drop_reasons["event_no_entry_close_within_tolerance"] > r.n_events
    assert any("event_no_entry_close_within_tolerance" in e for e in r.excluded)


@pytestmark_live
def test_live_gdelt_clustered_p_is_weaker_than_iid_p():
    r = verify(str(LIVE_DB), live_gdelt_hypothesis())
    assert r.p_value_clustered > r.p_value, (r.p_value, r.p_value_clustered)


@pytestmark_live
def test_live_gdelt_downside_hypothesis_is_empty_because_goldstein_is_floored():
    """Not a bug: daily-mean Goldstein sits on its -10 floor, so z can never fall 2sd below
    its own mean. A verification product must return 'untestable', not a plausible zero."""
    r = verify(str(LIVE_DB), live_gdelt_hypothesis(direction="down"))
    assert r.n_z_computable > 10_000
    assert r.n_fired == 0
    assert r.verdict == "not_testable"
    assert r.reconciled
    up = verify(str(LIVE_DB), live_gdelt_hypothesis(direction="up"))
    assert up.n_fired > 0, "the same data with the sign flipped must produce events"


@pytestmark_live
def test_live_as_of_run_sees_strictly_less_than_the_live_run():
    full = verify(str(LIVE_DB), live_gdelt_hypothesis())
    cut = verify(str(LIVE_DB), live_gdelt_hypothesis(), as_of=dt.datetime(2025, 1, 1, tzinfo=dt.UTC).timestamp())
    assert cut.n_events < full.n_events
    assert cut.n_baseline < full.n_baseline
    assert cut.reconciled
    for s in cut.leakage_audit:
        assert s.exit_ts <= dt.datetime(2025, 1, 1, tzinfo=dt.UTC).timestamp()


@pytestmark_live
def test_live_family_of_hypotheses_gets_one_bh_correction():
    rs = [verify(str(LIVE_DB), live_gdelt_hypothesis(horizon_days=h)) for h in (1, 5, 20)]
    rej, adj = benjamini_hochberg([r.p_value for r in rs])
    assert len(adj) == 3 and np.all(np.isfinite(adj))
    assert np.all(adj >= [r.p_value for r in rs]), "BH made a p-value smaller"
    assert not rej.any(), f"a gdelt/goldstein cell survived BH: {adj}"


# =====================================================================================
# 14. the seam with the report renderer
# =====================================================================================


def test_study_result_exposes_every_quantity_the_report_renderer_looks_for(tmp_path):
    """agent/verify/report.py resolves quantities through an alias table instead of
    importing StudyResult. A quantity it cannot find is silently rendered "not computed",
    so this test reads THEIR table and fails if a rename on either side opens a hole."""
    report = pytest.importorskip("agent.verify.report")
    aliases = getattr(report, "_ALIASES", None)
    if not isinstance(aliases, dict):
        pytest.skip("report.py no longer exposes an alias table")
    r = verify(canonical_db(tmp_path), canonical_hypothesis())
    # quantities this module deliberately does NOT compute: multiplicity correction is
    # the caller's job across a family, and these have no single-hypothesis meaning.
    not_ours = {"p_adj", "alpha", "significant", "n_tests", "correction", "n_for_80_power", "effect_units", "ci"}
    missing = [
        quantity
        for quantity, names in aliases.items()
        if quantity not in not_ours and not any(hasattr(r, n) for n in names)
    ]
    assert not missing, f"report.py cannot find: {missing}"
    assert r.mean_event_ret == r.mean_event_return
    assert r.mean_baseline_ret == r.baseline_mean_return
    assert r.n_clusters == r.n_event_clusters
    assert r.z_threshold == r.hypothesis.z_threshold
    assert r.field == "f" and r.horizon_days == 1
    assert r.label and isinstance(r.label, str)


# =====================================================================================
# 15. the reconciliation predicate must be able to fail
# =====================================================================================

CONSISTENT = dict(
    entered=100,
    n_not_computable=19,
    n_z_computable=81,
    n_graded=70,
    n_downstream_drops=11,
    n_fired=5,
    n_below_threshold=76,
    n_events=4,
)


def test_reconciliation_holds_on_consistent_counts():
    assert _reconciles(**CONSISTENT)


@pytest.mark.parametrize(
    "broken",
    [
        {"entered": 101},  # a point that entered vanished
        {"n_not_computable": 18},  # a warm-up point went unrecorded
        {"n_graded": 69},  # a graded return went missing
        {"n_downstream_drops": 10},  # a drop was swallowed (F-16's shape)
        {"n_below_threshold": 75},  # a non-firing point vanished
        {"n_fired": 4},  # fewer fired than were graded as events
        {"n_events": 71},  # more events than graded points
    ],
)
def test_reconciliation_fails_on_any_broken_identity(broken):
    """A flag that cannot be False is not a check. Each clause must be load-bearing."""
    assert not _reconciles(**{**CONSISTENT, **broken})


def test_verdict_refuses_to_answer_when_bookkeeping_does_not_reconcile(tmp_path, monkeypatch):
    """Wiring test: if the arithmetic fails, no statistical verdict may be issued."""
    import agent.verify.study as study_mod

    db = canonical_db(tmp_path)
    honest = verify(db, canonical_hypothesis(z_threshold=0.0))
    assert honest.reconciled and honest.verdict in ("not_supported", "supported_uncorrected")
    monkeypatch.setattr(study_mod, "_reconciles", lambda **kw: False)
    broken = verify(db, canonical_hypothesis(z_threshold=0.0))
    assert not broken.reconciled
    assert broken.verdict == "invalid_bookkeeping"
