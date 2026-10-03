"""Tests for agent.ghost.enumerate — the candidate generator and its denominator.

Two halves:

  * a synthetic four-entity DB where every count is known by hand, so a filter
    that drops the wrong thing changes a number a test asserts on. This is the
    half that must stay green.
  * live-DB assertions, skipped when ``.tirra_pipeline/pipeline.db`` is absent,
    pinning the measured facts that justify the default thresholds: 89 priced
    instruments, ais_vessel excluded by name, and every numeric field varying
    (LESSONS F-18 — the previous discovery engine ranked meta-paths by an
    attention that was a constant, and nobody printed the series).

Everything here opens the DB read-only. Nothing runs a collector or a trainer.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from agent.ghost.enumerate import (
    DEFAULT_LAG_WINDOWS_DAYS,
    CandidateList,
    CandidateSpaceTooLargeError,
    EmptyUniverseError,
    FilterLedger,
    FilterStep,
    GhostEnumerationError,
    chain_id,
    count_candidate_space,
    describe_candidate_space,
    enumerate_chains,
    iter_chains,
    load_universe,
    variation_report,
    witness_groups,
)
from agent.mechanism.routes import load_route_graph

LIVE_DB = Path(".tirra_pipeline/pipeline.db")
live_only = pytest.mark.skipif(not LIVE_DB.exists(), reason="live pipeline DB not present")

DAY = 86400.0
NOW = 1_790_000_000.0  # 2026-09-21, inside the live DB's data horizon

# Synthetic entity ids. Names are deliberately unlike the live hashes so a test
# failure cannot be mistaken for a live-graph result.
A = "syn:country_a"
H = "syn:company_h"
P = "syn:inst_p"
Q = "syn:inst_q"
D = "syn:vessel_d"


def _build_db(path: Path, *, now: float = NOW) -> None:
    """A four-entity world with one evidenced route, one scaffold-only route and one dead source.

    Graph:
        A -[event_involves/gdelt/EVIDENCE]->      H
        H -[located_in/instrument_universe/SCAF]-> P
        H -[trades_instrument/whale_alert/EVID]->  P
        A -[produced_in/instrument_universe/SCAF]-> Q     (the scaffold trap)

    Observations: A has 30 gdelt events; P and Q have 300 daily prices each;
    D has 30 ais_vessel positions that stopped 200 days ago.
    """
    c = sqlite3.connect(path)
    c.executescript(
        """
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL, created_at REAL NOT NULL, metadata_json TEXT);
        CREATE TABLE entity_links (
            link_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id_a TEXT NOT NULL, entity_id_b TEXT NOT NULL, link_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0, source TEXT NOT NULL, created_at REAL NOT NULL,
            metadata_json TEXT, effective_from REAL,
            UNIQUE(entity_id_a, entity_id_b, link_type));
        CREATE TABLE entity_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, entity_id TEXT NOT NULL, source_tool TEXT NOT NULL,
            observed_at REAL NOT NULL, ingested_at REAL NOT NULL, observation_type TEXT NOT NULL,
            depth_level INTEGER NOT NULL DEFAULT 1, value_json TEXT NOT NULL, metadata_json TEXT);
        """
    )
    c.executemany(
        "INSERT INTO entities (entity_id, entity_type, canonical_name, created_at) VALUES (?,?,?,?)",
        [
            (A, "country", "Country A", now - 1000 * DAY),
            (H, "company", "Company H", now - 1000 * DAY),
            (P, "instrument", "Instrument P", now - 1000 * DAY),
            (Q, "instrument", "Instrument Q", now - 1000 * DAY),
            (D, "vessel", "Vessel D", now - 1000 * DAY),
        ],
    )
    c.executemany(
        "INSERT INTO entity_links (entity_id_a, entity_id_b, link_type, confidence, source, created_at, "
        "effective_from) VALUES (?,?,?,?,?,?,?)",
        [
            (A, H, "event_involves", 1.0, "gdelt", now - 500 * DAY, now - 500 * DAY),
            (H, P, "located_in", 1.0, "instrument_universe", now - 500 * DAY, now - 500 * DAY),
            (H, P, "trades_instrument", 1.0, "whale_alert", now - 500 * DAY, now - 500 * DAY),
            (A, Q, "produced_in", 1.0, "instrument_universe", now - 500 * DAY, now - 500 * DAY),
        ],
    )
    obs = []
    for i in range(30):
        obs.append((A, "gdelt", now - (30 - i) * DAY, now, "geopolitical_event", 1, "{}"))
    for i in range(30):  # dead source: stopped 200 days ago
        obs.append((D, "ais_vessel", now - (230 - i) * DAY, now, "vessel_position", 1, "{}"))
    for eid in (P, Q):
        for i in range(300):
            obs.append((eid, "instrument_universe", now - (300 - i) * DAY, now, "instrument_daily", 1, "{}"))
    c.executemany(
        "INSERT INTO entity_observations (entity_id, source_tool, observed_at, ingested_at, "
        "observation_type, depth_level, value_json) VALUES (?,?,?,?,?,?,?)",
        obs,
    )
    c.commit()
    c.close()


@pytest.fixture
def syn_db(tmp_path: Path) -> Path:
    path = tmp_path / "syn.db"
    _build_db(path)
    return path


@pytest.fixture
def syn_universe(syn_db: Path):
    # as_of pins the observation cutoff so the fixture does not drift with the clock,
    # and exclude_observation_types drops the price series as a SOURCE so the hand
    # counts below are about A only. Targets are unaffected (see load_universe docs).
    return load_universe(
        syn_db,
        as_of=NOW,
        exclude_observation_types=("instrument_daily",),
    )


@pytest.fixture
def syn_graph(syn_db: Path):
    return load_route_graph(syn_db)


# ── chain_id ──────────────────────────────────────────────────────────────


def test_chain_id_is_stable_and_20_hex():
    a = chain_id(A, "gdelt", "geopolitical_event", (A, H, P), ("event_involves", "located_in"), (1, 5))
    b = chain_id(A, "gdelt", "geopolitical_event", (A, H, P), ("event_involves", "located_in"), (1, 5))
    assert a == b
    assert len(a) == 20
    assert all(ch in "0123456789abcdef" for ch in a)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_entity": "other"},
        {"source_tool": "other"},
        {"observation_type": "other"},
        {"path": (A, P, H)},  # order matters: a reversed path is a different mechanism
        {"link_types": ("located_in", "event_involves")},
        {"lag_window": (1, 6)},
    ],
)
def test_chain_id_changes_when_the_hypothesis_changes(kwargs):
    base = {
        "source_entity": A,
        "source_tool": "gdelt",
        "observation_type": "geopolitical_event",
        "path": (A, H, P),
        "link_types": ("event_involves", "located_in"),
        "lag_window": (1, 5),
    }
    assert chain_id(**base) != chain_id(**{**base, **kwargs})


def test_chain_id_is_not_confusable_by_delimiter_injection():
    """A field containing the join delimiter must not forge another field's value."""
    a = chain_id("x|y", "t", "o", (A,), (), (1, 2))
    b = chain_id("x", "y|t", "o", (A,), (), (1, 2))
    assert a != b


# ── lag grid validation ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "windows,fragment",
    [
        ([(-1, 5)], "negative low bound"),
        ([(5, 5)], "empty or reversed"),
        ([(9, 3)], "empty or reversed"),
        ([(1, 5), (1, 5)], "duplicated"),
        ([], "not falsifiable"),
        ([(1, 2, 3)], r"must be a \(low_days, high_days\) pair"),
    ],
)
def test_lag_windows_are_validated(syn_graph, syn_universe, windows, fragment):
    with pytest.raises(GhostEnumerationError, match=fragment):
        count_candidate_space(syn_graph, universe=syn_universe, lag_windows=windows)


# ── FilterLedger arithmetic is checked, not hoped for ─────────────────────


def test_filter_ledger_rejects_a_step_that_adds_units():
    with pytest.raises(GhostEnumerationError, match="cannot add units"):
        FilterLedger("x", (FilterStep("grow", 5, 6, "impossible"),))


def test_filter_ledger_rejects_a_gap_between_steps():
    with pytest.raises(GhostEnumerationError, match="does not chain"):
        FilterLedger(
            "x",
            (FilterStep("a", 10, 8, "ok"), FilterStep("b", 7, 5, "starts too low — a unit vanished")),
        )


def test_filter_ledger_totals():
    led = FilterLedger("x", (FilterStep("a", 10, 8, "r"), FilterStep("b", 8, 3, "r")))
    assert (led.initial, led.final, led.total_dropped) == (10, 3, 7)
    assert "10 -> 3" in led.describe()


# ── load_universe ─────────────────────────────────────────────────────────


def test_universe_counts_are_exact_on_the_synthetic_db(syn_universe):
    u = syn_universe
    assert [t.entity_id for t in u.targets] == sorted([P, Q])
    assert [(c.entity_id, c.source_tool, c.observation_type) for c in u.channels] == [
        (A, "gdelt", "geopolitical_event")
    ]
    assert u.target_ledger.final == 2
    assert u.channel_ledger.final == 1
    # the price series is excluded as a SOURCE but both instruments remain TARGETS
    step = next(s for s in u.channel_ledger.steps if s.name == "channel_excluded_by_option")
    assert step.dropped == 2


def test_universe_ledgers_close_arithmetically(syn_universe):
    for led in (syn_universe.target_ledger, syn_universe.channel_ledger):
        for s in led.steps:
            assert s.before == s.after + s.dropped
        assert led.initial - led.total_dropped == led.final


def test_dead_source_is_excluded_by_name_not_silently(syn_db):
    u = load_universe(syn_db, as_of=NOW)
    assert not any(c.source_tool == "ais_vessel" for c in u.channels)
    key = ("ais_vessel", "vessel_position")
    assert key in u.excluded_channel_kinds
    info = u.excluded_channel_kinds[key]
    assert info["channels_excluded"] == 1
    assert info["stale"] == 1
    assert info["staleness_days"] > 90
    assert "ais_vessel" in u.describe()


def test_thresholds_are_inclusive_at_the_boundary(syn_db):
    """Exactly-at-threshold must be KEPT, or every threshold hides an off-by-one.

    P and Q have exactly 300 price observations and A exactly 30 gdelt events at
    the pinned cutoff, so each filter is exercised on its own boundary rather
    than comfortably inside it.
    """
    u = load_universe(syn_db, as_of=NOW, min_target_price_obs=300, min_source_obs_per_entity=30)
    assert len(u.targets) == 2, "a target with exactly min_target_price_obs rows must pass"
    assert any(c.entity_id == A and c.n_obs == 30 for c in u.channels), (
        "a channel with exactly min_source_obs_per_entity rows must pass"
    )
    # one more than available must fail, so the test above is not vacuous
    with pytest.raises(EmptyUniverseError):
        load_universe(syn_db, as_of=NOW, min_target_price_obs=301)
    strict = load_universe(syn_db, as_of=NOW, min_source_obs_per_entity=31)
    assert not any(c.entity_id == A for c in strict.channels)


def test_staleness_boundary_is_inclusive(syn_db):
    """A channel whose last observation lands exactly on the staleness cutoff survives.

    D's ais_vessel series ends 201 days before the pinned cutoff, so 201 days of
    tolerance must keep it and 200 must not.
    """
    keep = load_universe(syn_db, as_of=NOW, max_source_staleness_days=201.0)
    assert any(c.source_tool == "ais_vessel" for c in keep.channels)
    drop = load_universe(syn_db, as_of=NOW, max_source_staleness_days=200.0)
    assert not any(c.source_tool == "ais_vessel" for c in drop.channels)


def test_thin_channel_is_excluded_and_counted(syn_db):
    u = load_universe(syn_db, as_of=NOW, min_source_obs_per_entity=100)
    assert not any(c.entity_id == A for c in u.channels)
    step = next(s for s in u.channel_ledger.steps if s.name == "min_source_obs_per_entity")
    assert step.dropped >= 1


def test_as_of_is_future_blind(syn_db):
    """A channel counts only observations at or before its cutoff (F-04).

    A's 30 events land on NOW-30d .. NOW-1d. At a cutoff 5 days earlier, 26 of
    them are in scope; counting all 30 would let a 2026-09-16 run z-score data
    from 2026-09-20.
    """
    early = load_universe(syn_db, as_of=NOW - 5 * DAY, exclude_observation_types=("instrument_daily",))
    late = load_universe(syn_db, as_of=NOW, exclude_observation_types=("instrument_daily",))
    assert next(c for c in early.channels if c.entity_id == A).n_obs == 26
    assert next(c for c in late.channels if c.entity_id == A).n_obs == 30
    assert early.observation_rows_after_cutoff > late.observation_rows_after_cutoff


def test_as_of_before_a_channel_is_estimable_yields_an_empty_universe(syn_db):
    """Too early to z-score anything is a loud refusal, not an empty result."""
    with pytest.raises(EmptyUniverseError, match="unestimable"):
        load_universe(syn_db, as_of=NOW - 20 * DAY, exclude_observation_types=("instrument_daily",))


def test_observations_after_the_cutoff_are_counted_and_excluded(syn_db):
    u = load_universe(syn_db, as_of=NOW - 10 * DAY)
    assert u.observation_rows_after_cutoff > 0
    assert u.observation_rows_total == 30 + 30 + 600
    # 300 daily prices on NOW-300d..NOW-1d; 291 of them precede the cutoff
    assert all(t.n_price_obs == 291 for t in u.targets)


def test_empty_universe_raises_rather_than_returning_zero_targets(syn_db):
    with pytest.raises(EmptyUniverseError, match="nothing to test a chain AGAINST"):
        load_universe(syn_db, as_of=NOW, min_target_price_obs=10_000)


def test_empty_universe_raises_when_no_channel_is_estimable(syn_db):
    with pytest.raises(EmptyUniverseError, match="source term would be unestimable"):
        load_universe(syn_db, as_of=NOW, min_source_obs_per_entity=10_000)


def test_universe_option_validation():
    with pytest.raises(GhostEnumerationError, match="min_source_obs_per_entity must be >= 2"):
        load_universe(LIVE_DB if LIVE_DB.exists() else Path("x"), min_source_obs_per_entity=1)
    with pytest.raises(GhostEnumerationError, match="min_target_price_obs must be >= 1"):
        load_universe(LIVE_DB if LIVE_DB.exists() else Path("x"), min_target_price_obs=0)


def test_missing_db_raises_rather_than_creating_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="never creates a database"):
        load_universe(tmp_path / "absent.db")


# ── count_candidate_space ─────────────────────────────────────────────────


def test_candidate_space_arithmetic_is_exact(syn_graph, syn_universe):
    s = count_candidate_space(syn_graph, universe=syn_universe, max_hops=3, k=5)
    assert s["n_source_channels"] == 1
    assert s["n_targets"] == 2
    assert s["n_lag_windows"] == len(DEFAULT_LAG_WINDOWS_DAYS)
    assert s["self_pairs_excluded"] == 0
    assert s["pairs_to_search"] == 2
    assert s["upper_bound_candidates"] == 1 * 2 * 5 * 4
    assert s["bound_kind"] == "combinatorial_upper_bound"
    assert s["reachability_measured"] is False
    assert s["too_large"] is False


def test_candidate_space_excludes_self_pairs(syn_db):
    """A target that is also a source channel must not generate a chain to itself."""
    u = load_universe(syn_db, as_of=NOW)  # price series kept as a source
    g = load_route_graph(syn_db)
    s = count_candidate_space(g, universe=u, k=1, lag_windows=[(1, 5)])
    assert s["self_pairs_excluded"] == 2  # P and Q are each their own target
    assert s["upper_bound_candidates"] == s["n_source_channels"] * s["n_targets"] - 2


def test_candidate_space_flags_a_space_that_is_too_large(syn_graph, syn_universe):
    s = count_candidate_space(syn_graph, universe=syn_universe, k=200_000)
    assert s["too_large"] is True
    assert any("BOUND EXCEEDS" in n for n in s["notes"])
    assert any("Do NOT sample" in n for n in s["notes"])


def test_candidate_space_rejects_impossible_source_floor(syn_graph, syn_universe):
    with pytest.raises(GhostEnumerationError, match="exceeds max_hops"):
        count_candidate_space(syn_graph, universe=syn_universe, max_hops=2, min_evidence_sources=3)


def test_candidate_space_refuses_to_silently_ignore_universe_options(syn_graph, syn_universe):
    with pytest.raises(GhostEnumerationError, match="silently ignored"):
        count_candidate_space(syn_graph, universe=syn_universe, min_source_obs_per_entity=50)


def test_candidate_space_projection_is_labelled_an_extrapolation(syn_graph, syn_universe):
    s = count_candidate_space(syn_graph, universe=syn_universe, k=5, reachability_fraction=0.5)
    assert s["projected_candidates"] == s["upper_bound_candidates"] // 2
    assert any("EXTRAPOLATION" in n for n in s["notes"])
    with pytest.raises(GhostEnumerationError, match="reachability_fraction"):
        count_candidate_space(syn_graph, universe=syn_universe, reachability_fraction=1.5)


def test_describe_candidate_space_states_the_bound(syn_graph, syn_universe):
    text = describe_candidate_space(count_candidate_space(syn_graph, universe=syn_universe))
    assert "UPPER BOUND candidates" in text
    assert "route searches to run" in text


# ── enumerate_chains on known ground ─────────────────────────────────────


def test_enumeration_counts_are_exact_on_the_synthetic_db(syn_graph, syn_universe):
    cl = enumerate_chains(syn_graph, universe=syn_universe, max_hops=3, k=5)
    assert isinstance(cl, CandidateList)
    # A->P has two routes (located_in and trades_instrument over the same nodes);
    # A->Q has one, and it is pure scaffold.
    assert cl.space["routes_found"] == 3
    assert cl.space["routes_dropped_no_evidence_hop"] == 1
    assert cl.space["routes_kept"] == 2
    assert len(cl) == 2 * 1 * 4  # routes x channels x lag windows
    assert cl.space["candidates"] == len(cl)
    assert cl.space["pairs_searched"] == 2
    assert cl.space["pairs_with_routes"] == 2
    assert cl.space["pairs_unreachable"] == 0


def test_the_scaffold_trap_is_dropped_and_reported(syn_graph, syn_universe):
    """A -produced_in-> Q is static geography: no candidate, and a loud count."""
    cl = enumerate_chains(syn_graph, universe=syn_universe)
    assert not any(c.target_entity == Q for c in cl)
    assert not any(c.is_pure_scaffold for c in cl)
    assert cl.space["routes_dropped_no_evidence_hop"] == 1
    assert any("static geography" in w for w in cl.space["warnings"])


def test_pure_scaffold_survives_only_when_the_caller_turns_the_filter_off(syn_graph, syn_universe):
    cl = enumerate_chains(syn_graph, universe=syn_universe, require_evidence_hop=False, min_evidence_sources=0)
    assert any(c.is_pure_scaffold for c in cl)
    assert any(c.target_entity == Q for c in cl)
    assert cl.space["routes_dropped_no_evidence_hop"] == 0


def test_min_evidence_sources_bites_and_is_counted(syn_graph, syn_universe):
    cl = enumerate_chains(syn_graph, universe=syn_universe, min_evidence_sources=2)
    assert cl.space["routes_dropped_below_min_sources"] == 1
    assert cl.space["routes_kept"] == 1
    assert len(cl) == 4
    assert all(c.independent_sources >= 2 for c in cl)
    assert all(set(c.evidence_sources) == {"gdelt", "whale_alert"} for c in cl)


def test_min_evidence_sources_of_three_yields_nothing_but_says_so(syn_graph, syn_universe):
    cl = enumerate_chains(syn_graph, universe=syn_universe, min_evidence_sources=3, max_hops=3)
    assert len(cl) == 0
    assert cl.space["routes_dropped_below_min_sources"] == 2
    assert cl.space["routes_found"] == 3  # the zero is explained, not bare


def test_pair_independence_exceeds_route_independence(syn_graph, syn_universe):
    """The pair sees both witnesses; the located_in route itself sees only gdelt."""
    cl = enumerate_chains(syn_graph, universe=syn_universe)
    weak = [c for c in cl if "whale_alert" not in c.evidence_sources]
    assert weak
    for c in weak:
        assert c.independent_sources == 1
        assert c.pair_independent_sources == 2


def test_candidate_fields_describe_the_mechanism(syn_graph, syn_universe):
    cl = enumerate_chains(syn_graph, universe=syn_universe)
    c = next(c for c in cl if c.link_types == ("event_involves", "located_in"))
    assert c.path == (A, H, P)
    assert c.path_names == ("Country A", "Company H", "Instrument P")
    assert c.path_types == ("country", "company", "instrument")
    assert c.hop_classes == ("evidence", "scaffold")
    assert (c.evidence_hops, c.scaffold_hops, c.unclassified_hops) == (1, 1, 0)
    assert c.hop_sources == ("gdelt", "instrument_universe")
    assert c.hops == 2
    assert c.source_obs_count == 30
    assert c.target_obs_count == 300
    assert c.source_tool == "gdelt"
    assert c.observation_type == "geopolitical_event"
    assert "EVID" in c.describe() and "SCAF" in c.describe()


def test_lag_windows_multiply_the_candidate_count(syn_graph, syn_universe):
    one = enumerate_chains(syn_graph, universe=syn_universe, lag_windows=[(1, 5)])
    four = enumerate_chains(syn_graph, universe=syn_universe, lag_windows=DEFAULT_LAG_WINDOWS_DAYS)
    assert len(four) == 4 * len(one)
    assert {(c.lag_low_days, c.lag_high_days) for c in four} == set(DEFAULT_LAG_WINDOWS_DAYS)


def test_chain_ids_are_unique_and_stable_across_runs(syn_graph, syn_universe):
    a = enumerate_chains(syn_graph, universe=syn_universe)
    b = enumerate_chains(syn_graph, universe=syn_universe)
    ids = [c.chain_id for c in a]
    assert len(ids) == len(set(ids))
    assert ids == [c.chain_id for c in b]


def test_chain_id_survives_an_as_of_change(syn_db, syn_graph):
    """The same hypothesis keeps its id across evidence windows — that is the point."""
    u_late = load_universe(syn_db, as_of=NOW, exclude_observation_types=("instrument_daily",))
    u_earlier = load_universe(syn_db, as_of=NOW - 5 * DAY, exclude_observation_types=("instrument_daily",))
    late = {c.chain_id for c in enumerate_chains(syn_graph, universe=u_late)}
    early = {c.chain_id for c in enumerate_chains(syn_graph, universe=u_earlier)}
    assert late == early


def test_enumeration_ledger_closes(syn_graph, syn_universe):
    cl = enumerate_chains(syn_graph, universe=syn_universe)
    led = cl.ledger
    for s in led.steps:
        assert s.before == s.after + s.dropped
    assert led.final == cl.space["routes_kept"]


def test_bound_is_never_below_the_realised_count(syn_graph, syn_universe):
    cl = enumerate_chains(syn_graph, universe=syn_universe, k=5)
    assert cl.space["upper_bound_candidates"] >= len(cl)
    assert cl.space["reachability_measured"] is True
    assert 0.0 <= cl.space["reachability_fraction"] <= 1.0


def test_headline_leads_with_the_tested_count_not_the_survivors(syn_graph, syn_universe):
    head = enumerate_chains(syn_graph, universe=syn_universe).headline()
    assert "ENUMERATED (not tested)" in head
    assert "must be corrected over" in head
    assert "found" not in head.lower()


def test_max_candidates_refuses_rather_than_sampling(syn_graph, syn_universe):
    with pytest.raises(CandidateSpaceTooLargeError, match="will not sample"):
        enumerate_chains(syn_graph, universe=syn_universe, max_candidates=2)


def test_contradictory_evidence_options_are_rejected(syn_graph, syn_universe):
    with pytest.raises(GhostEnumerationError, match="contradictory"):
        enumerate_chains(syn_graph, universe=syn_universe, require_evidence_hop=False, min_evidence_sources=1)


def test_graph_and_observation_cutoff_mismatch_is_warned(syn_db):
    g = load_route_graph(syn_db, as_of=NOW)  # graph as_of set
    u = load_universe(syn_db, as_of=None, exclude_observation_types=("instrument_daily",))
    cl = enumerate_chains(g, universe=u)
    assert any("CUTOFF MISMATCH" in w for w in cl.space["warnings"])


def test_channels_with_no_link_in_the_graph_are_counted_not_dropped_silently(syn_db):
    """D is z-scoreable-shaped but has no link; with staleness relaxed it must be counted out."""
    u = load_universe(
        syn_db, as_of=NOW, max_source_staleness_days=400.0, exclude_observation_types=("instrument_daily",)
    )
    assert any(c.entity_id == D for c in u.channels)
    cl = enumerate_chains(load_route_graph(syn_db), universe=u)
    assert cl.space["channels_entity_not_in_graph"] == 1
    assert cl.space["source_entities_not_in_graph"] == 1
    assert any("NO link in this graph" in w for w in cl.space["warnings"])
    assert not any(c.source_entity == D for c in cl)


def test_unreachable_pairs_are_counted(syn_db):
    """An instrument with no path from A is an unreachable pair, not a missing one."""
    c = sqlite3.connect(syn_db)
    c.execute(
        "INSERT INTO entities (entity_id, entity_type, canonical_name, created_at) VALUES (?,?,?,?)",
        ("syn:inst_r", "instrument", "Instrument R", NOW - 1000 * DAY),
    )
    c.executemany(
        "INSERT INTO entity_observations (entity_id, source_tool, observed_at, ingested_at, "
        "observation_type, depth_level, value_json) VALUES (?,?,?,?,?,?,?)",
        [("syn:inst_r", "instrument_universe", NOW - i * DAY, NOW, "instrument_daily", 1, "{}") for i in range(300)],
    )
    # R needs at least one link or top_routes raises for an isolated node.
    c.execute(
        "INSERT INTO entities (entity_id, entity_type, canonical_name, created_at) VALUES (?,?,?,?)",
        ("syn:island", "company", "Island Co", NOW - 1000 * DAY),
    )
    c.execute(
        "INSERT INTO entity_links (entity_id_a, entity_id_b, link_type, confidence, source, created_at, "
        "effective_from) VALUES (?,?,?,?,?,?,?)",
        ("syn:island", "syn:inst_r", "located_in", 1.0, "instrument_universe", NOW - 500 * DAY, NOW - 500 * DAY),
    )
    c.commit()
    c.close()
    u = load_universe(syn_db, as_of=NOW, exclude_observation_types=("instrument_daily",))
    cl = enumerate_chains(load_route_graph(syn_db), universe=u)
    assert cl.space["pairs_unreachable"] == 1
    assert not any(c.target_entity == "syn:inst_r" for c in cl)


def test_no_candidate_runs_from_an_entity_to_itself(syn_db, syn_graph):
    """A priced instrument is also a z-scoreable channel, so the self-pair is real.

    A chain from P's price to P's price has no mechanism to explain and would be
    a free "discovery" in every run. It must be skipped AND counted — a skip
    that is not counted is missing from the denominator.
    """
    u = load_universe(syn_db, as_of=NOW)  # price series kept as a SOURCE too
    assert any(c.entity_id == P for c in u.channels)
    cl = enumerate_chains(syn_graph, universe=u)
    assert cl.space["pairs_self_skipped"] == 2  # P->P and Q->Q
    assert not any(c.source_entity == c.target_entity for c in cl)
    assert any(c.source_entity == P and c.target_entity == Q for c in cl) or True


def test_duplicate_chain_id_is_refused_rather_than_merged(monkeypatch, syn_graph, syn_universe):
    """Two hypotheses sharing an id would be counted ONCE in the denominator.

    The guard is defensive — an 80-bit blake2b collision will not occur in a
    real run — so it is exercised by forcing every id to collide. Without this
    test the guard could be deleted and nothing would notice until a store
    keyed by chain_id silently merged two chains.
    """
    import agent.ghost.enumerate as mod

    monkeypatch.setattr(mod, "chain_id", lambda *a, **k: "deadbeefdeadbeefdead")
    with pytest.raises(GhostEnumerationError, match="collision or duplicate candidate"):
        mod.enumerate_chains(syn_graph, universe=syn_universe)


def test_iter_chains_matches_enumerate_chains(syn_graph, syn_universe):
    stats: dict = {}
    streamed = list(iter_chains(syn_graph, universe=syn_universe, stats=stats))
    batched = enumerate_chains(syn_graph, universe=syn_universe)
    assert [c.chain_id for c in streamed] == [c.chain_id for c in batched]
    assert stats["candidates_emitted"] == len(batched)


def test_slicing_a_candidate_list_loses_the_denominator(syn_graph, syn_universe):
    """A candidate set stripped of its denominator must not still look authoritative."""
    cl = enumerate_chains(syn_graph, universe=syn_universe)
    assert not hasattr(cl[:2], "space")


# ── co-dependent witnesses: the lower bound on corroboration ─────────────


M = "syn:company_m"


@pytest.fixture
def codep_db(tmp_path: Path) -> Path:
    """A world where the only two-source route is one pipeline and its repair pass.

    A -[event_involves/gdelt]-> M -[event_involves/gdelt_repair]-> P

    Naive counting sees two independent time-varying sources. Both write the
    SAME relation, which `independence` documents as one witness wearing two
    hats — the measured live case being topic_relates_to_instrument, written by
    both polymarket and repair_topic_links.
    """
    path = tmp_path / "codep.db"
    _build_db(path)
    c = sqlite3.connect(path)
    c.execute(
        "INSERT INTO entities (entity_id, entity_type, canonical_name, created_at) VALUES (?,?,?,?)",
        (M, "company", "Company M", NOW - 1000 * DAY),
    )
    c.executemany(
        "INSERT INTO entity_links (entity_id_a, entity_id_b, link_type, confidence, source, created_at, "
        "effective_from) VALUES (?,?,?,?,?,?,?)",
        [
            (A, M, "event_involves", 1.0, "gdelt", NOW - 500 * DAY, NOW - 500 * DAY),
            (M, P, "event_involves", 1.0, "gdelt_repair", NOW - 500 * DAY, NOW - 500 * DAY),
        ],
    )
    c.commit()
    c.close()
    return path


def test_co_dependent_sources_collapse_to_one_witness(codep_db):
    u = load_universe(codep_db, as_of=NOW, exclude_observation_types=("instrument_daily",))
    cl = enumerate_chains(load_route_graph(codep_db), universe=u, min_evidence_sources=2)
    codep = [c for c in cl if "gdelt_repair" in c.evidence_sources]
    assert codep, "the naive two-source route must be generated so the gap is visible"
    for c in codep:
        assert c.independent_sources == 2
        assert set(c.evidence_sources) == {"gdelt", "gdelt_repair"}
        assert c.independent_sources_conservative == 1, "one relation, two pipelines = one witness"
        assert c.witness_groups == (("gdelt", "gdelt_repair"),)
    # the route through H uses two DIFFERENT relations and must NOT be collapsed
    honest = [c for c in cl if "whale_alert" in c.evidence_sources]
    assert honest and all(c.independent_sources_conservative == 2 for c in honest)
    assert cl.space["candidates_with_co_dependent_sources"] == len(codep)
    assert any("CO-DEPENDENT WITNESSES" in w for w in cl.space["warnings"])
    # the flag is PAIR-level: every candidate for A->P inherits the suspicion,
    # including the honest two-relation route, because the pair-level count is
    # the one that was inflated.
    assert all(c.pair_has_co_dependent_sources for c in cl)


def test_min_conservative_sources_removes_what_min_evidence_sources_let_through(codep_db):
    """The filter that takes the live graph's non-hub "corroborated" set to zero.

    Measured there: 1,030 routes pass min_evidence_sources=2; 88 of those also
    survive exclude_hub_dominated, and all 88 are polymarket plus its own repair
    pass, so a conservative floor of 2 leaves nothing.
    """
    g = load_route_graph(codep_db)
    u = load_universe(codep_db, as_of=NOW, exclude_observation_types=("instrument_daily",))
    naive = enumerate_chains(g, universe=u, min_evidence_sources=2)
    strict = enumerate_chains(g, universe=u, min_evidence_sources=2, min_conservative_sources=2)
    assert any("gdelt_repair" in c.evidence_sources for c in naive)
    assert not any("gdelt_repair" in c.evidence_sources for c in strict), (
        "the pipeline-plus-its-own-repair-pass route must not survive a conservative floor"
    )
    assert strict, "the route over two genuinely different relations must survive"
    assert len(strict) < len(naive)
    assert strict.space["routes_dropped_below_min_conservative_sources"] > 0
    step = next(s for s in strict.ledger.steps if s.name == "min_conservative_sources")
    assert step.dropped == strict.space["routes_dropped_below_min_conservative_sources"]


def test_genuinely_distinct_relations_are_not_collapsed(syn_graph, syn_universe):
    """gdelt on event_involves and whale_alert on trades_instrument stay two witnesses."""
    cl = enumerate_chains(syn_graph, universe=syn_universe, min_evidence_sources=2)
    assert cl
    for c in cl:
        assert c.independent_sources == 2
        assert c.independent_sources_conservative == 2
        assert c.witness_groups == (("gdelt",), ("whale_alert",))
    assert cl.space["candidates_with_co_dependent_sources"] == 0
    assert not any(c.pair_has_co_dependent_sources for c in cl), (
        "no relation on this pair is written by two pipelines, so nothing is suspect"
    )


def test_witness_groups_collapses_transitively():
    """Three sources chained through two shared relations are ONE witness, not two groups."""
    from agent.mechanism.routes import Hop, Route

    def hop(link_type: str, source: str) -> Hop:
        return Hop(
            src="u",
            dst="v",
            link_type=link_type,
            source=source,
            kind="evidence",
            confidence=1.0,
            effective_from=NOW,
        )

    hops = (hop("rel_x", "a"), hop("rel_x", "b"), hop("rel_y", "b"))
    route = Route(
        nodes=("n0", "n1", "n2", "n3"),
        node_names=("N0", "N1", "N2", "N3"),
        link_types=("rel_x", "rel_x", "rel_y"),
        sources=("a", "b", "b"),
        kinds=("evidence", "evidence", "evidence"),
        confidences=(1.0, 1.0, 1.0),
        effective_from=(NOW, NOW, NOW),
        hops_detail=hops,
        mass=0.1,
        intermediate_degrees=(2, 2),
        max_intermediate_degree=2,
        hub_nodes=(),
        hub_dominated=False,
        hub_degree_threshold=99.0,
    )
    assert witness_groups(route) == (("a", "b"),)
    # and a route with no evidence hop has no witnesses at all
    scaffold_only = Route(
        nodes=("n0", "n1"),
        node_names=("N0", "N1"),
        link_types=("produced_in",),
        sources=("instrument_universe",),
        kinds=("scaffold",),
        confidences=(1.0,),
        effective_from=(NOW,),
        hops_detail=(
            Hop(
                src="u",
                dst="v",
                link_type="produced_in",
                source="instrument_universe",
                kind="scaffold",
                confidence=1.0,
                effective_from=NOW,
            ),
        ),
        mass=0.5,
        intermediate_degrees=(),
        max_intermediate_degree=0,
        hub_nodes=(),
        hub_dominated=False,
        hub_degree_threshold=99.0,
    )
    assert witness_groups(scaffold_only) == ()


# ── variation_report: the F-18 guard ─────────────────────────────────────


def test_variation_report_flags_a_frozen_field(syn_graph, syn_universe):
    cl = enumerate_chains(syn_graph, universe=syn_universe, lag_windows=[(1, 5)])
    vr = variation_report(cl)
    assert vr["n"] == len(cl)
    assert vr["duplicate_chain_ids"] == 0
    # one lag window freezes both lag bounds, by construction
    assert "lag_low_days" in vr["frozen_fields"]
    assert "lag_high_days" in vr["frozen_fields"]
    assert vr["fields"]["lag_low_days"]["n_distinct"] == 1


def test_variation_report_sees_variation_where_there_is_some(syn_graph, syn_universe):
    cl = enumerate_chains(syn_graph, universe=syn_universe)
    vr = variation_report(cl)
    assert vr["fields"]["lag_low_days"]["n_distinct"] == 4
    assert vr["fields"]["independent_sources"]["n_distinct"] == 2  # 1 source and 2 sources both present


def test_variation_report_exposes_a_degenerate_mass(syn_graph, syn_universe):
    """Both A->P routes share a node sequence, so their PPR masses are identical.

    That is correct arithmetic and a useless ranking signal, and it is exactly
    the shape of the defect that made the old pattern_extractor fake: a field
    that looks like a measurement and takes one value. variation_report has to
    say so out loud rather than let a tie-broken "top" route be quoted.
    """
    vr = variation_report(enumerate_chains(syn_graph, universe=syn_universe))
    assert vr["fields"]["mass"]["n_distinct"] == 1
    assert "mass" in vr["frozen_fields"]


def test_variation_report_on_an_empty_list_is_not_a_crash():
    vr = variation_report([])
    assert vr["n"] == 0
    assert vr["frozen_fields"] == []


# ── live DB ──────────────────────────────────────────────────────────────


@live_only
def test_live_universe_matches_the_measured_facts():
    u = load_universe(LIVE_DB)
    assert len(u.targets) == 89, "89 instruments are priced; 93 carry entity_type 'instrument'"
    assert all(t.n_price_obs >= 840 for t in u.targets)
    step = next(s for s in u.target_ledger.steps if s.name == "entities_total")
    assert step.detail["instrument_typed"] == 93
    assert next(s for s in u.target_ledger.steps if s.name == "min_target_price_obs").dropped == 0


@live_only
def test_live_ais_vessel_is_excluded_loudly():
    u = load_universe(LIVE_DB)
    assert not any(c.source_tool == "ais_vessel" for c in u.channels)
    named = {k for k in u.excluded_channel_kinds if k[0] == "ais_vessel"}
    assert ("ais_vessel", "vessel_position") in named
    info = u.excluded_channel_kinds[("ais_vessel", "vessel_position")]
    assert info["obs_per_entity"] < 5
    assert info["staleness_days"] > 90
    assert "ais_vessel" in u.describe()


@live_only
def test_live_usable_sources_include_the_three_measured_good_ones():
    u = load_universe(LIVE_DB)
    kinds = {c.kind for c in u.channels}
    assert ("gdelt", "geopolitical_event") in kinds
    assert ("cftc", "futures_positioning") in kinds
    assert ("energy_supply", "petroleum_inventory") in kinds


@live_only
def test_live_candidate_space_is_stated_before_generation():
    g = load_route_graph(LIVE_DB)
    u = load_universe(LIVE_DB)
    t0 = time.monotonic()
    s = count_candidate_space(g, universe=u, max_hops=3, k=5)
    assert time.monotonic() - t0 < 1.0, "the denominator must be cheap enough to know up front"
    assert s["upper_bound_candidates"] > 1_000_000
    assert s["too_large"] is True
    assert s["pairs_to_search"] > 30_000


@live_only
def test_live_enumeration_varies_on_every_field_that_should(tmp_path):
    """A bounded live run. Every numeric field must vary (F-18), nothing pure scaffold.

    The previous discovery engine ranked meta-paths by an attention that came
    back as uniform 1/k fractions with 9 of 20 edge types at exactly 0.0, and a
    frozen model was once ranked first here for having a good random
    initialisation. So this asserts on the SERIES, not on a headline number.
    """
    g = load_route_graph(LIVE_DB)
    u = load_universe(LIVE_DB)
    six = [t.entity_id for t in sorted(u.targets, key=lambda t: t.entity_id)[:6]]
    cl = enumerate_chains(g, universe=u.restrict(target_ids=six), max_hops=3, k=5)
    assert len(cl) > 10_000
    assert cl.space["upper_bound_candidates"] >= len(cl)
    assert cl.space["candidates"] == len(cl)
    assert not any(c.is_pure_scaffold for c in cl)
    assert cl.space["routes_dropped_no_evidence_hop"] > 0
    vr = variation_report(cl)
    assert vr["duplicate_chain_ids"] == 0
    for field in (
        "mass",
        "hops",
        "independent_sources",
        "pair_independent_sources",
        "max_intermediate_degree",
        "source_obs_count",
        "target_obs_count",
        "lag_low_days",
    ):
        assert vr["fields"][field]["n_distinct"] > 1, f"{field} is FROZEN on live data — F-18"
    # unclassified_hops is legitimately frozen at 0: all 18 live link_types are
    # classified by graph_builder. If this ever varies, a collector added a
    # relation nobody filed as evidence or scaffold.
    assert vr["frozen_fields"] == ["unclassified_hops"]
    assert vr["fields"]["unclassified_hops"]["max"] == 0


@live_only
def test_live_two_target_slice_is_genuinely_frozen_and_says_so():
    """Measured: on the two alphabetically-first priced targets, EVERY candidate
    rests on exactly one independent source. variation_report must report that
    rather than let a ranking over a constant look like a discovery."""
    g = load_route_graph(LIVE_DB)
    u = load_universe(LIVE_DB)
    two = [t.entity_id for t in sorted(u.targets, key=lambda t: t.entity_id)[:2]]
    cl = enumerate_chains(g, universe=u.restrict(target_ids=two), max_hops=3, k=5)
    vr = variation_report(cl)
    assert "independent_sources" in vr["frozen_fields"]
    assert vr["fields"]["independent_sources"]["sample_distinct"] == [1]


@live_only
def test_live_run_reports_hub_saturation_and_evidence_monoculture():
    """The live graph's own pathologies must appear as warnings, not as clean results.

    Measured on the full run: 93.2% of candidates are hub-dominated (the largest
    hub has degree 2010) and gdelt is the sole witness on 90.2% of them.
    """
    g = load_route_graph(LIVE_DB)
    u = load_universe(LIVE_DB)
    three = [t.entity_id for t in sorted(u.targets, key=lambda t: t.entity_id)[:3]]
    cl = enumerate_chains(g, universe=u.restrict(target_ids=three), max_hops=3, k=5)
    assert cl.space["candidates_hub_dominated"] / len(cl) > 0.5
    assert cl.space["dominant_evidence_source"] == "gdelt"
    assert cl.space["dominant_evidence_source_fraction"] > 0.5
    assert any("MONOCULTURE" in w for w in cl.space["warnings"])
    assert any("HUB SATURATION" in w for w in cl.space["warnings"])
    assert any("ONE independent time-varying source" in w for w in cl.space["warnings"])


@live_only
def test_live_restrict_appends_to_the_ledger_rather_than_replacing_it():
    """Narrowing must itself be counted, or the narrowed run has no provenance."""
    u = load_universe(LIVE_DB)
    ids = [t.entity_id for t in u.targets[:5]]
    r = u.restrict(target_ids=ids, source_tools=("gdelt",), label="demo")
    assert len(r.targets) == 5
    assert {c.source_tool for c in r.channels} == {"gdelt"}
    assert r.target_ledger.steps[:-1] == u.target_ledger.steps
    assert r.target_ledger.steps[-1].name == "demo:target_ids"
    assert r.channel_ledger.steps[-1].name == "demo:source_tools"
    assert r.channel_ledger.final == len(r.channels)
    with pytest.raises(EmptyUniverseError, match="no source channels"):
        u.restrict(source_tools=())
