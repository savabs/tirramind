"""Tests for `agent.ghost.score` — the multiple-testing accounting layer.

The thing under test is not "does it compute a p-value" (``agent.verify`` does
that, and has 517 tests). It is: **can this module be made to report a
survivor without the family size, or to turn "cannot tell" into "not
supported"?** Every test below is an attempt to do one of those.

Fixture strategy: a real SQLite database built per test with the live schema,
with the event and price series constructed so the expected verdict is known in
advance. The live database is never touched here.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from agent.ghost.score import (
    UNTESTABLE_MIN_CLUSTERS,
    UNTESTABLE_MIN_EVENTS,
    UNTESTABLE_REASONS,
    CandidateError,
    ChainCandidate,
    ScoreBoard,
    _normalise_edge,
    _ReadOnlyLagStore,
    coerce_candidate,
    extract_lags,
    from_enumerated,
    horizon_from_lag_window,
    relook_warnings,
    score_all,
    score_chain,
)
from agent.verify.study import N_BOOTSTRAP

DAY = 86400.0

SCHEMA = """
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, canonical_name TEXT NOT NULL,
    created_at REAL NOT NULL, metadata_json TEXT);
CREATE TABLE entity_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, entity_id TEXT NOT NULL, source_tool TEXT NOT NULL,
    observed_at REAL NOT NULL, ingested_at REAL NOT NULL, observation_type TEXT NOT NULL,
    depth_level INTEGER NOT NULL DEFAULT 1, value_json TEXT NOT NULL, metadata_json TEXT);
CREATE TABLE entity_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT, entity_id_a TEXT NOT NULL, entity_id_b TEXT NOT NULL,
    link_type TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 1.0, source TEXT NOT NULL,
    created_at REAL NOT NULL, metadata_json TEXT, effective_from REAL,
    UNIQUE(entity_id_a, entity_id_b, link_type));
"""


# --------------------------------------------------------------------------- #
# fixture construction
# --------------------------------------------------------------------------- #


class Builder:
    """Minimal writable pipeline DB. Only used on tmp_path — never the live DB."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(SCHEMA)

    def entity(self, eid: str, etype: str, name: str | None = None) -> str:
        self.con.execute(
            "insert or ignore into entities values (?,?,?,?,NULL)",
            (eid, etype, name or eid, 0.0),
        )
        return eid

    def link(self, a: str, b: str, ltype: str, eff: float | None = 0.0, source: str = "test") -> None:
        self.con.execute(
            "insert or ignore into entity_links "
            "(entity_id_a, entity_id_b, link_type, confidence, source, created_at, effective_from) "
            "values (?,?,?,1.0,?,0.0,?)",
            (a, b, ltype, source, eff),
        )

    def obs(self, eid: str, source: str, otype: str, ts: float, payload: dict) -> None:
        self.con.execute(
            "insert into entity_observations (entity_id, source_tool, observed_at, ingested_at, "
            "observation_type, depth_level, value_json) values (?,?,?,?,?,1,?)",
            (eid, source, ts, ts, otype, json.dumps(payload)),
        )

    def series(
        self,
        eid: str,
        source: str,
        otype: str,
        field: str,
        values,
        *,
        start: float = 0.0,
        step: float = DAY,
    ) -> None:
        for i, v in enumerate(values):
            self.obs(eid, source, otype, start + i * step, {field: float(v)})

    def done(self) -> str:
        self.con.commit()
        self.con.close()
        return self.path


def _signal_with_spikes(n: int, spike_at, *, magnitude: float = 6.0, seed: int = 0):
    """A quiet series with large positive spikes, so causal z fires exactly at `spike_at`."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    for i in spike_at:
        x[i] = magnitude
    return x


def _prices_that_rise_after(n: int, rise_at, *, per_step: float = 0.02, window: int = 25, seed: int = 1):
    """A flat random-walk price that trends up for `window` steps after each `rise_at`.

    `window` must be shorter than the gap between spikes, or the "effect"
    windows overlap, the UNCONDITIONAL baseline absorbs the drift, and the edge
    against that baseline collapses to nothing. That is not a fixture quirk —
    it is the reason `verify` tests against the population mean rather than
    against zero, and it is worth stating here because a naive fixture prints a
    null for a planted effect and looks like a bug in the scorer.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.002, n)
    for i in rise_at:
        rets[i + 1 : i + 1 + window] += per_step
    return 100.0 * np.exp(np.cumsum(rets))


def make_db(
    tmp_path: Path,
    *,
    n: int = 520,
    n_pairs: int = 4,
    spike_every: int = 35,
    stagger: int = 3,
    effect: float = 0.02,
    name: str = "fx.db",
) -> str:
    """A DB where an up-spike in `sig` genuinely precedes a rise in `close`.

    Each pair spikes on its OWN dates (offset by `k * stagger`), so the events
    are not calendar-clustered and `effective_clusters` can reach the
    testability floor. `spike_every` controls how many events there are and
    `effect` how big the real move is; the clustered and untestable cases get
    their own builders below rather than being coaxed out of this one.
    """
    b = Builder(tmp_path / name)
    for k in range(n_pairs):
        spikes = list(range(40 + k * stagger, n - 40, spike_every))
        src = b.entity(f"src{k}", "cftc_contract")
        dst = b.entity(f"dst{k}", "instrument")
        b.link(src, dst, "cftc_tracks")
        b.series(src, "cftc", "futures_positioning", "sig", _signal_with_spikes(n, spikes, seed=k))
        b.series(
            dst,
            "instrument_universe",
            "instrument_daily",
            "close",
            _prices_that_rise_after(n, spikes, per_step=effect, window=min(25, spike_every - 5), seed=100 + k),
        )
    return b.done()


def make_null_db(tmp_path: Path, *, n: int = 520, n_pairs: int = 4, name: str = "null.db") -> str:
    """Same shape, but the price is independent of the signal. Nothing should survive."""
    return make_db(tmp_path, n=n, n_pairs=n_pairs, effect=0.0, name=name)


CAND_KW = dict(
    event_source="cftc",
    event_obs_type="futures_positioning",
    event_field="sig",
    z_threshold=2.0,
    direction="up",
    target_entity_type="instrument",
    link_types=("cftc_tracks",),
    event_entity_type="cftc_contract",
)


def cand(**over) -> ChainCandidate:
    kw = dict(CAND_KW)
    kw.update(over)
    return ChainCandidate(**kw)


def family(n: int, **over) -> list[ChainCandidate]:
    """`n` distinct candidates that all run against the same fixture columns."""
    out = []
    for i in range(n):
        out.append(cand(z_threshold=1.0 + 0.05 * i, **over))
    return out


# --------------------------------------------------------------------------- #
# 1. the candidate contract
# --------------------------------------------------------------------------- #


def test_coerce_accepts_mapping_object_and_candidate():
    c = cand()
    assert coerce_candidate(c) is c

    from_map = coerce_candidate({k: v for k, v in CAND_KW.items()})
    assert from_map.chain_id == c.chain_id

    class Duck:
        pass

    d = Duck()
    for k, v in CAND_KW.items():
        setattr(d, k, v)
    assert coerce_candidate(d).chain_id == c.chain_id


def test_coerce_names_every_missing_required_field_at_once():
    with pytest.raises(CandidateError) as exc:
        coerce_candidate({"event_source": "cftc"})
    msg = str(exc.value)
    for f in ("event_obs_type", "event_field", "z_threshold", "direction", "target_entity_type"):
        assert f in msg


def test_coerce_refuses_unknown_key_rather_than_defaulting_it():
    # A generator that writes `field` instead of `event_field` must not get a
    # study of the wrong thing that still prints a p-value.
    kw = dict(CAND_KW)
    kw["field"] = "sig"
    with pytest.raises(CandidateError, match="unknown candidate key"):
        coerce_candidate(kw)


def test_coerce_refuses_bare_string_link_types():
    with pytest.raises(CandidateError, match="character by character"):
        coerce_candidate({**CAND_KW, "link_types": "cftc_tracks"})


def test_chain_id_is_stable_and_specification_sensitive():
    assert cand().chain_id == cand().chain_id
    assert cand(z_threshold=2.5).chain_id != cand().chain_id
    assert cand(direction="down").chain_id != cand().chain_id
    assert cand(link_types=("other",)).chain_id != cand().chain_id


def test_candidate_carries_no_horizon():
    # A candidate that owned its horizon would make the family miscount trivial.
    assert not hasattr(cand(), "horizon_days")
    assert "horizon_days" not in ChainCandidate.__dataclass_fields__


# --------------------------------------------------------------------------- #
# 2. score_chain: one candidate, no correction
# --------------------------------------------------------------------------- #


def test_score_chain_returns_one_cell_per_horizon_with_no_survival_flag(tmp_path):
    db = make_db(tmp_path)
    r = score_chain(cand(), db, horizons=(1, 5, 10))
    assert r.n_tests == 3
    assert [c.horizon_days for c in r.cells] == [1, 5, 10]
    # The load-bearing assertion: a single candidate is not a family.
    assert all(c.p_adjusted is None for c in r.cells)
    assert all(c.survived is None for c in r.cells)
    assert all(c.verdict in {"uncorrected_only", "not_supported", "untestable"} for c in r.cells)
    assert not any(c.verdict.startswith("survived") for c in r.cells)


def test_score_chain_rejects_duplicate_horizons(tmp_path):
    db = make_db(tmp_path)
    with pytest.raises(CandidateError, match="duplicate horizon"):
        score_chain(cand(), db, horizons=(5, 5))


def test_score_chain_rejects_empty_horizons(tmp_path):
    db = make_db(tmp_path)
    with pytest.raises(CandidateError, match="no horizon"):
        score_chain(cand(), db, horizons=())


def clustered_db(tmp_path, *, name="clu.db", n=520, n_pairs=6, dates=18) -> str:
    """Correlated pairs firing on the SAME dates: n is inflated, information is not.

    This is the fixture where the i.i.d. and the cluster-resampled p-values
    genuinely differ, which is what makes `p_basis` testable at all.
    """
    b = Builder(tmp_path / name)
    spikes = list(range(40, 40 + dates * 24, 24))
    for k in range(n_pairs):
        src, dst = b.entity(f"s{k}", "cftc_contract"), b.entity(f"d{k}", "instrument")
        b.link(src, dst, "cftc_tracks")
        b.series(src, "cftc", "futures_positioning", "sig", _signal_with_spikes(n, spikes, seed=k))
        b.series(
            dst,
            "instrument_universe",
            "instrument_daily",
            "close",
            # per_step=0 on purpose: with no planted effect the i.i.d. bootstrap
            # still prints p=0.074 on these correlated events while resampling the
            # 22 event DATES prints 0.452. That gap is the reason the default basis
            # is "clustered", and it is what makes the knob testable.
            _prices_that_rise_after(n, spikes, per_step=0.0, window=18, seed=7),
        )
    return b.done()


def test_score_chain_records_both_p_values_and_stamps_the_basis(tmp_path):
    db = clustered_db(tmp_path)
    c = score_chain(cand(), db, horizons=(5,)).cells[0]
    assert c.testable, "precondition: the cell must be testable for the basis to matter"
    assert c.p_basis == "clustered"
    assert c.p_uncorrected == c.p_clustered
    c2 = score_chain(cand(), db, horizons=(5,), p_basis="iid").cells[0]
    assert c2.p_basis == "iid"
    assert c2.p_uncorrected == c2.p_iid
    # The whole point of the knob: on correlated events the two differ, and the
    # clustered figure is the larger (more honest) one. If they were equal the
    # default would be untestable and a silent switch to iid would be invisible.
    assert c.p_clustered > c2.p_iid
    assert c.p_uncorrected != c2.p_uncorrected
    # Measured on this fixture: 0.074 i.i.d. vs 0.452 clustered on a PLANTED NULL.
    # The i.i.d. figure is the one that would have been published.
    assert c2.p_iid < 0.10 < c.p_clustered


def test_readonly_store_pair_count_is_direction_sensitive(tmp_path):
    """`produced_in` runs instrument -> country; the reverse must count zero.

    The upstream extractor indexes links by `entity_id_a` only, so a meta-path
    stated in the wrong direction finds nothing and returns mean_lag=0.0. This
    count is what lets `extract_lags` say "no_directed_pairs" instead.
    """
    db = lag_db(tmp_path)
    with _ReadOnlyLagStore(db) as store:
        assert store.metapath_pair_count("instrument", "produced_in", "country") == 1
        assert store.metapath_pair_count("country", "produced_in", "instrument") == 0
        assert store.metapath_pair_count("instrument", "no_such", "country") == 0


def test_a_broken_candidate_becomes_an_error_cell_not_a_shrunken_family(tmp_path):
    db = make_db(tmp_path)
    r = score_chain(cand(), str(tmp_path / "does-not-exist.db"), horizons=(1, 5))
    assert r.n_tests == 2, "a crashed candidate must still occupy its slots in the family"
    assert all(c.error is not None for c in r.cells)
    assert all(not c.testable for c in r.cells)
    assert all(c.verdict == "error" for c in r.cells)
    assert all(math.isnan(c.p_uncorrected) for c in r.cells)


def test_power_and_resolution_are_attached_to_every_cell(tmp_path):
    db = make_db(tmp_path)
    for c in score_chain(cand(), db, horizons=(1, 5, 21)).cells:
        assert math.isfinite(c.power)
        assert c.p_resolution == pytest.approx(1.0 / N_BOOTSTRAP)
        assert "stipulated" in c.power_reference
        assert "NOT post-hoc" in c.power_reference


# --------------------------------------------------------------------------- #
# 3. the untestable bucket — "cannot tell" is a verdict
# --------------------------------------------------------------------------- #


def test_few_events_is_untestable_not_insignificant(tmp_path):
    # 3 spikes only: above verify's bootstrap floor, far below our n floor.
    b = Builder(tmp_path / "tiny.db")
    n = 300
    spikes = [100, 150, 200]
    src, dst = b.entity("s", "cftc_contract"), b.entity("d", "instrument")
    b.link(src, dst, "cftc_tracks")
    b.series(src, "cftc", "futures_positioning", "sig", _signal_with_spikes(n, spikes))
    b.series(dst, "instrument_universe", "instrument_daily", "close", _prices_that_rise_after(n, spikes))
    db = b.done()

    c = score_chain(cand(), db, horizons=(5,)).cells[0]
    assert 0 < c.n_events < UNTESTABLE_MIN_EVENTS
    assert not c.testable
    assert c.untestable_reason == "n_events_below_minimum"
    assert c.verdict == "untestable"
    assert c.verdict != "not_supported"


def test_zero_events_is_named_no_events(tmp_path):
    db = make_db(tmp_path)
    c = score_chain(cand(z_threshold=50.0), db, horizons=(5,)).cells[0]
    assert c.n_events == 0
    assert c.untestable_reason == "no_events"


def test_clustered_events_are_untestable_even_when_n_is_large(tmp_path):
    """Many events on few distinct weeks: n passes, independence does not.

    Four correlated contracts spiking on the same dates give 4x the events and
    1x the information. This is the "123 events on 67 weeks" failure, and the
    cell must be bucketed on clusters, not waved through on n.
    """
    b = Builder(tmp_path / "clustered.db")
    n = 400
    spikes = list(range(40, 40 + 10 * 8, 8))  # 10 dates only
    for k in range(6):
        src, dst = b.entity(f"s{k}", "cftc_contract"), b.entity(f"d{k}", "instrument")
        b.link(src, dst, "cftc_tracks")
        b.series(src, "cftc", "futures_positioning", "sig", _signal_with_spikes(n, spikes, seed=k))
        b.series(dst, "instrument_universe", "instrument_daily", "close", _prices_that_rise_after(n, spikes, seed=k))
    db = b.done()

    c = score_chain(cand(), db, horizons=(5,)).cells[0]
    assert c.n_events >= UNTESTABLE_MIN_EVENTS, "precondition: n must pass so the cluster gate is what bites"
    assert c.effective_clusters < UNTESTABLE_MIN_CLUSTERS
    assert not c.testable
    assert c.untestable_reason == "effective_clusters_below_minimum"
    assert c.max_cluster_size > 1
    assert c.pseudo_replication > 1.0


def test_every_untestable_reason_is_a_declared_one(tmp_path):
    db = make_db(tmp_path, n=400, n_pairs=1, spike_every=120)
    board = score_all(family(6), db, horizons=(1, 5, 10, 21))
    for c in board.cells:
        if not c.testable:
            assert c.untestable_reason in UNTESTABLE_REASONS


def test_cluster_window_is_a_reported_knob(tmp_path):
    db = make_db(tmp_path, spike_every=20)
    wide = score_chain(cand(), db, horizons=(5,), cluster_window_s=60 * DAY).cells[0]
    narrow = score_chain(cand(), db, horizons=(5,), cluster_window_s=1 * DAY).cells[0]
    assert wide.effective_clusters < narrow.effective_clusters
    assert wide.cluster_window_s == 60 * DAY and narrow.cluster_window_s == 1 * DAY
    # The origin is pinned to the first event, not to the 1970 epoch grid.
    assert wide.cluster_origin > 0.0


# --------------------------------------------------------------------------- #
# 4. the family and the correction — the product
# --------------------------------------------------------------------------- #


def test_n_tested_is_candidates_times_horizons(tmp_path):
    db = make_db(tmp_path)
    board = score_all(family(5), db, horizons=(1, 5, 10, 21))
    assert board.n_tested == 20
    assert board.n_candidates == 5
    assert board.horizons == (1, 5, 10, 21)
    assert board.n_testable + board.n_untestable == board.n_tested


def test_bh_is_applied_across_candidates_and_horizons_not_per_candidate(tmp_path):
    """The family is the whole search. Correcting per candidate is the lie.

    Same cells, two accountings: BH over all 24 versus BH over each candidate's
    4 horizons separately. The whole-family adjusted p must be no smaller.
    """
    from agent.verify.stats import benjamini_hochberg

    db = make_db(tmp_path, effect=0.004)
    board = score_all(family(6), db, horizons=(1, 5, 10, 21))
    testable = [c for c in board.cells if c.testable]
    assert len(testable) >= 12, "precondition: most cells testable"

    per_cand_rejected = 0
    for c in board.results:
        ps = [x.p_uncorrected for x in c.cells if x.testable]
        if ps:
            per_cand_rejected += benjamini_hochberg(ps, alpha=0.05).n_rejected
    assert board.bh.n_tested == len(testable)
    assert len(board.survivors) <= per_cand_rejected


def test_survivors_cannot_be_read_without_the_family_size(tmp_path):
    db = make_db(tmp_path, effect=0.03)
    board = score_all(family(4), db, horizons=(1, 5, 10, 21))
    text = board.report()
    assert f"TESTED            {board.n_tested}" in text
    assert f"{len(board.survivors)} of {board.n_tested} tested" in text
    d = board.to_dict()
    assert d["n_tested"] == board.n_tested
    assert "n_survivors" in d and "bh_denominator" in d


def test_uncorrected_count_is_reported_next_to_the_corrected_one(tmp_path):
    db = make_null_db(tmp_path)
    board = score_all(family(12), db, horizons=(1, 5, 10, 21))
    assert board.n_would_survive_uncorrected >= len(board.survivors)
    assert "would 'work' uncorrected" in board.report()


def test_a_real_effect_survives_and_is_labelled_underpowered(tmp_path):
    db = make_db(tmp_path, effect=0.05, n_pairs=4)
    board = score_all(family(3), db, horizons=(10, 21))
    assert board.survivors, "a 5%-per-step planted effect should survive BH on this fixture"
    for c in board.survivors:
        assert c.survived is True
        assert c.p_adjusted is not None
        assert c.family_size == board.n_tested
        assert c.bh_denominator == board.bh.n_tested
        # Power on a handful of clusters is single digits, so the verdict says so.
        assert c.verdict in {"survived", "survived_underpowered"}
        if c.power < 0.5:
            assert c.verdict == "survived_underpowered"


def test_pure_noise_family_produces_no_survivors(tmp_path):
    db = make_null_db(tmp_path, n=520, n_pairs=5)
    board = score_all(family(14), db, horizons=(1, 5, 10, 21))
    assert board.n_tested == 56
    assert len(board.survivors) == 0, board.report()
    assert "0 of 56 tested survived correction" in board.report()


def test_untestable_cells_never_carry_a_corrected_p_but_are_counted(tmp_path):
    db = make_db(tmp_path, n=400, n_pairs=1, spike_every=120)  # ~3 events -> untestable
    board = score_all(family(6), db, horizons=(1, 5, 10, 21))
    assert board.n_untestable > 0
    for c in board.cells:
        if not c.testable:
            assert c.p_adjusted is None, "an untested cell must not own a corrected p-value"
            assert c.survived is False
            assert c.verdict == "untestable"
    assert sum(board.untestable_reasons.values()) == board.n_untestable
    # Excluded from the denominator, but never from the family size.
    assert board.bh.n_tested == board.n_testable
    assert board.n_tested == board.n_testable + board.n_untestable


def test_conservative_correction_is_computed_and_disagreement_is_warned(tmp_path):
    db = make_db(tmp_path, effect=0.05)
    board = score_all(family(8), db, horizons=(1, 5, 10, 21))
    assert board.bh_conservative.n_tested == board.n_tested
    assert len(board.survivors_conservative) <= len(board.survivors)
    lost = {(c.chain_id, c.horizon_days) for c in board.survivors} - {
        (c.chain_id, c.horizon_days) for c in board.survivors_conservative
    }
    joined = " ".join(board.warnings())
    if lost:
        assert "conservative correction" in joined


def mixed_board(tmp_path):
    """A family that contains BOTH real survivors and untestable cells.

    Six candidates fire on the planted effect; six are set at z >= 8 and fire
    never. Measured on this fixture: 48 cells, 24 testable, 24 untestable
    (``no_events``), 7 BH survivors, 6 conservative survivors, 10 cells that
    would "work" uncorrected. That disagreement is exactly what the three
    accounting rules below need in order to be testable at all — a family
    where every cell is testable cannot detect a broken conservative
    correction, which is how three mutants originally survived this suite.
    """
    db = make_db(tmp_path, effect=0.05, n_pairs=4, name="mixed.db")
    cands = family(6) + [cand(z_threshold=8.0 + i) for i in range(6)]
    board = score_all(cands, db, horizons=(1, 5, 10, 21))
    if not (board.n_untestable and board.survivors):
        pytest.fail(
            f"fixture precondition lost: untestable={board.n_untestable} survivors={len(board.survivors)}. "
            "This fixture must mix both, or the tests below assert nothing."
        )
    return board


def test_untestable_cells_enter_the_conservative_correction_at_p_one(tmp_path):
    """An untestable cell must never become a CONSERVATIVE survivor.

    ``bh_conservative`` exists to ask "would this survive if the cells we could
    not test were entered at p=1.0 instead of dropped?". Entering them at
    anything else inverts the question: at p=0.0 the 1193 untestable cells of
    the live run would all be "rejected" and the conservative count would read
    as overwhelming support.
    """
    board = mixed_board(tmp_path)
    assert board.bh_conservative.n_tested == board.n_tested
    conservative = {(c.chain_id, c.horizon_days) for c in board.survivors_conservative}
    for c in board.cells:
        if not c.testable:
            assert (c.chain_id, c.horizon_days) not in conservative, (
                f"untestable cell {c.chain_id}@{c.horizon_days} is a conservative survivor; "
                "it entered the correction at something other than p=1.0"
            )
    # Larger denominator, same p-values: the conservative set can only shrink.
    assert len(board.survivors_conservative) <= len(board.survivors)


def test_conservative_disagreement_is_warned_on_a_family_that_disagrees(tmp_path):
    """The unconditional version of the earlier ``if lost:`` test.

    The original test only checked the warning when the fixture happened to
    produce a disagreement, so a mutant that deleted the warning entirely
    survived. This fixture disagrees by construction.
    """
    board = mixed_board(tmp_path)
    lost = {(c.chain_id, c.horizon_days) for c in board.survivors} - {
        (c.chain_id, c.horizon_days) for c in board.survivors_conservative
    }
    assert lost, "fixture precondition: a survivor must be lost under the conservative correction"
    joined = " ".join(board.warnings())
    assert "conservative correction" in joined
    assert "carried by the" in joined
    assert str(board.bh_conservative.n_tested) in joined


def test_uncorrected_count_strictly_exceeds_the_corrected_one(tmp_path):
    """The multiple-testing problem, stated as the gap between two counts.

    In the published CFTC study it was 9 versus 0; on the live graph it is 97
    versus 8. If this property reports the corrected count instead, the gap
    closes to zero and the board stops showing what correction cost.
    """
    board = mixed_board(tmp_path)
    assert board.n_would_survive_uncorrected > len(board.survivors), (
        f"uncorrected={board.n_would_survive_uncorrected} survivors={len(board.survivors)}: "
        "this family must show a gap, or the count is untested"
    )
    # It counts testable cells below alpha, not survivors — recomputed here
    # from the cells so the property cannot quietly become an alias.
    assert board.n_would_survive_uncorrected == sum(
        1 for c in board.cells if c.testable and c.p_uncorrected < board.alpha
    )
    assert f"would 'work' uncorrected: {board.n_would_survive_uncorrected}" in board.report()


def test_scoreboard_rejects_a_duplicated_test(tmp_path):
    db = make_db(tmp_path)
    with pytest.raises(CandidateError, match="duplicate chain_id"):
        score_all([cand(), cand()], db, horizons=(5,))


def test_scoreboard_rejects_an_empty_family(tmp_path):
    db = make_db(tmp_path)
    with pytest.raises(CandidateError, match="no candidates"):
        score_all([], db, horizons=(5,))


def test_scoreboard_rejects_a_bad_alpha(tmp_path):
    db = make_db(tmp_path)
    for bad in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(CandidateError, match="alpha"):
            score_all(family(2), db, horizons=(5,), alpha=bad)


def test_upstream_prefilter_count_is_reported(tmp_path):
    db = make_db(tmp_path)
    board = score_all(family(3), db, horizons=(1, 5), n_generated=4000)
    assert board.n_generated == 4000
    assert "of 4000 generated" in board.report()
    assert any("filtered out before" in w for w in board.warnings())


def test_warnings_always_state_the_one_look_rule(tmp_path):
    db = make_db(tmp_path)
    board = score_all(family(2), db, horizons=(5,))
    assert any("ONE look" in w for w in board.warnings())


def test_bootstrap_resolution_floor_is_warned_when_a_survivor_reports_p_zero(tmp_path):
    db = make_db(tmp_path, effect=0.06)
    board = score_all(family(3), db, horizons=(10, 21))
    zeros = [c for c in board.survivors if c.p_uncorrected <= 0.0]
    if zeros:
        assert any("RESOLUTION FLOOR" in w for w in board.warnings())
    else:  # pragma: no cover - fixture-dependent
        pytest.skip("this fixture produced no p == 0 survivor")


# --------------------------------------------------------------------------- #
# 5. F-18: the numbers must actually vary
# --------------------------------------------------------------------------- #


def test_variation_report_flags_a_degenerate_column(tmp_path):
    """A constant column is a broken engine, not a finding.

    The engine this module replaces ranked meta-paths by an attention score
    that was a uniform 1/k fraction from an untrained net. This asserts the
    detector would have caught it.
    """
    db = make_db(tmp_path)
    board = score_all(family(4), db, horizons=(1, 5, 10, 21))
    txt = board.variation_report()
    assert "p_uncorrected" in txt and "edge" in txt and "power" in txt
    assert "DEGENERATE" not in txt, txt

    frozen = tuple(type(c)(**{**c.__dict__, "p_uncorrected": 0.333, "p_clustered": 0.333}) for c in board.cells)
    broken = ScoreBoard(
        cells=frozen,
        results=board.results,
        alpha=board.alpha,
        p_basis=board.p_basis,
        cluster_window_s=board.cluster_window_s,
        min_events=board.min_events,
        min_clusters=board.min_clusters,
        horizons=board.horizons,
        n_candidates=board.n_candidates,
        n_generated=None,
        db_path=board.db_path,
        db_fingerprint=board.db_fingerprint,
        as_of=board.as_of,
        family_id=board.family_id,
        bh=board.bh,
        bh_conservative=board.bh_conservative,
        bh_excludes_untestable=board.bh_excludes_untestable,
    )
    lines = [ln for ln in broken.variation_report().splitlines() if "p_clustered" in ln]
    assert lines and "DEGENERATE" in lines[0]


def test_variation_report_survives_an_all_nonfinite_column(tmp_path):
    board = score_all(family(2), str(tmp_path / "nope.db"), horizons=(1, 5))
    txt = board.variation_report()
    assert "NO FINITE VALUES" in txt
    assert board.n_errors == 4
    assert board.n_tested == 4


# --------------------------------------------------------------------------- #
# 6. re-looks
# --------------------------------------------------------------------------- #


def test_relook_on_same_family_new_data_is_named_a_second_look(tmp_path):
    db1 = make_db(tmp_path, name="a.db")
    db2 = make_db(tmp_path, n=460, name="b.db")
    b1 = score_all(family(2), db1, horizons=(5,))
    b2 = score_all(family(2), db2, horizons=(5,))
    assert b1.family_id == b2.family_id
    assert b1.db_fingerprint != b2.db_fingerprint
    w = " ".join(relook_warnings(b1, b2))
    assert "SECOND LOOK" in w and "one look" in w


def test_relook_on_a_different_family_says_the_adjusted_ps_are_incomparable(tmp_path):
    db = make_db(tmp_path)
    b1 = score_all(family(2), db, horizons=(5,))
    b2 = score_all(family(5), db, horizons=(5,))
    assert b1.family_id != b2.family_id
    w = " ".join(relook_warnings(b1, b2))
    assert "DIFFERENT family" in w and "not comparable" in w


def test_relook_on_identical_family_and_data_provides_no_new_evidence(tmp_path):
    db = make_db(tmp_path)
    b1 = score_all(family(2), db, horizons=(5,))
    b2 = score_all(family(2), db, horizons=(5,))
    w = " ".join(relook_warnings(b1, b2))
    assert "no new evidence" in w


def test_relook_reports_a_changed_testability_floor(tmp_path):
    db = make_db(tmp_path)
    b1 = score_all(family(2), db, horizons=(5,))
    b2 = score_all(family(5), db, horizons=(5,), min_events=5, min_clusters=2)
    assert any("testability floors changed" in w for w in relook_warnings(b1, b2))


# --------------------------------------------------------------------------- #
# 7. extract_lags — the inherited defect
# --------------------------------------------------------------------------- #


def test_normalise_edge_handles_rev_and_refuses_composites():
    assert _normalise_edge("a", "x", "b")["resolved_link_type"] == "x"
    rev = _normalise_edge("country", "rev_produced_in", "instrument")
    assert rev["resolved_link_type"] == "produced_in"
    assert (rev["src_type"], rev["dst_type"]) == ("instrument", "country")
    comp = _normalise_edge("a", "x_via_y", "b")
    assert comp["status"] == "composite_metapath_unsupported"
    assert comp["components"] == ("x", "y")
    # A refused composite must not report a plausible-looking link name it did
    # not resolve. Naming its first component here would put "produced_in" in a
    # report row whose lag was never computed.
    assert comp["resolved_link_type"] is None


def lag_db(tmp_path) -> str:
    b = Builder(tmp_path / "lags.db")
    inst = b.entity("WTI", "instrument")
    ctry = b.entity("RU", "country")
    b.link(inst, ctry, "produced_in", eff=None)  # stored instrument -> country, like the live graph
    b.series(inst, "instrument_universe", "instrument_daily", "close", [1.0] * 5, start=0.0, step=10 * DAY)
    b.series(ctry, "gdelt", "geopolitical_event", "goldstein", [1.0] * 5, start=2 * DAY, step=10 * DAY)
    return b.done()


def test_reversed_edge_gets_a_real_lag_where_upstream_would_return_zero(tmp_path):
    """The defect, demonstrated and then routed around.

    Upstream matches ``edge_type`` against the raw ``link_type`` and indexes
    links by ``entity_id_a`` only, so ``rev_produced_in`` finds nothing and the
    pattern keeps its ``mean_lag=0.0`` default — a number that means "no data".
    """
    from agent.models.gnn.pattern_extractor import MetaPathPattern, extract_temporal_lags

    db = lag_db(tmp_path)

    with _ReadOnlyLagStore(db) as store:
        naive = MetaPathPattern("country", "rev_produced_in", "instrument", 0.0, 0.0, 1)
        extract_temporal_lags([naive], store, top_k=1)
    assert naive.mean_lag == 0.0, "precondition: upstream returns a zero that means 'no data'"

    out = extract_lags(cand(metapath=("country", "rev_produced_in", "instrument")), db)
    assert out["status"] == "ok"
    (mp,) = out["metapaths"]
    assert mp["normalisation"] == "reversed"
    assert mp["resolved_link_type"] == "produced_in"
    assert mp["mean_lag"] is not None and mp["mean_lag"] > 0.0
    assert mp["mean_lag_days"] == pytest.approx(2.0)


def test_composite_edge_is_refused_with_its_components_named(tmp_path):
    db = lag_db(tmp_path)
    out = extract_lags(cand(metapath=("country", "produced_in_via_topic", "instrument")), db)
    assert out["status"] == "composite_metapath_unsupported"
    (mp,) = out["metapaths"]
    assert mp["mean_lag"] is None, "a composite must not inherit a 0.0"
    assert mp["components"] == ["produced_in", "topic"]
    assert any("mean_lag=0.0" in w for w in out["warnings"])


def test_unknown_link_type_is_named_not_zeroed(tmp_path):
    db = lag_db(tmp_path)
    out = extract_lags(cand(metapath=("country", "no_such_link", "instrument")), db)
    assert out["status"] == "unknown_link_type"
    assert out["metapaths"][0]["mean_lag"] is None


def test_wrong_direction_metapath_is_named_no_directed_pairs(tmp_path):
    db = lag_db(tmp_path)
    out = extract_lags(cand(metapath=("country", "produced_in", "instrument")), db)
    assert out["status"] == "no_directed_pairs"
    assert out["metapaths"][0]["mean_lag"] is None
    assert out["metapaths"][0]["n_pairs"] == 0


def test_pairs_but_no_forward_observation_gap_is_not_zero(tmp_path):
    """Links exist, but no dst observation ever follows a src observation."""
    b = Builder(tmp_path / "nolag.db")
    a = b.entity("A", "cftc_contract")
    z = b.entity("Z", "instrument")
    b.link(a, z, "cftc_tracks")
    b.series(a, "cftc", "futures_positioning", "sig", [1.0, 2.0], start=1000 * DAY)
    b.series(z, "instrument_universe", "instrument_daily", "close", [1.0, 2.0], start=0.0)
    db = b.done()
    out = extract_lags(cand(metapath=("cftc_contract", "cftc_tracks", "instrument")), db)
    assert out["status"] == "no_lag_observations"
    assert out["metapaths"][0]["n_pairs"] == 1
    assert out["metapaths"][0]["mean_lag"] is None


def test_lags_derive_the_metapath_from_the_candidate_when_absent(tmp_path):
    db = make_db(tmp_path)
    out = extract_lags(cand(), db)
    assert out["resolved_from"].startswith("derived")
    assert out["status"] == "ok"
    assert out["metapaths"][0]["resolved_link_type"] == "cftc_tracks"


def test_lags_without_declared_link_types_warns_that_it_searched(tmp_path):
    db = make_db(tmp_path)
    out = extract_lags(cand(link_types=None), db)
    assert any("search over link types" in w for w in out["warnings"])


def test_readonly_store_projection_gives_the_same_lags_as_the_full_rows(tmp_path):
    """The adapter projects columns and entities; that must not change a lag.

    If the upstream extractor ever starts reading a third observation key, this
    fails rather than the projection quietly returning a different number.
    """
    from agent.models.gnn.pattern_extractor import MetaPathPattern, extract_temporal_lags

    db = lag_db(tmp_path)
    got = []
    for types in (None, ["instrument", "country"]):
        with _ReadOnlyLagStore(db, entity_types=types) as store:
            p = MetaPathPattern("instrument", "produced_in", "country", 0.0, 0.0, 1, mean_lag=float("nan"))
            extract_temporal_lags([p], store, top_k=1)
            got.append(p.mean_lag)
    assert got[0] == got[1] and math.isfinite(got[0])


def test_readonly_store_cannot_write(tmp_path):
    db = lag_db(tmp_path)
    with _ReadOnlyLagStore(db) as store:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store._con.execute("delete from entity_links")


# --------------------------------------------------------------------------- #
# 8. nothing is dropped silently
# --------------------------------------------------------------------------- #


def test_every_cell_carries_its_own_attrition(tmp_path):
    db = make_db(tmp_path)
    for c in score_chain(cand(), db, horizons=(5,)).cells:
        assert c.n_pairs_entered >= c.n_pairs_used >= 1
        assert c.n_baseline > c.n_events
        assert isinstance(c.drop_reasons, tuple)
        assert c.reconciled is True


def test_board_reports_error_cells_in_its_warnings(tmp_path):
    board = score_all(family(2), str(tmp_path / "missing.db"), horizons=(1, 5))
    assert board.n_errors == 4
    assert any("raised and were counted as untestable" in w for w in board.warnings())


def test_to_dict_is_json_serialisable(tmp_path):
    db = make_db(tmp_path)
    board = score_all(family(3), db, horizons=(1, 5))
    json.dumps(board.to_dict())


# --------------------------------------------------------------------------- #
# 9. interop with the route-level generator
# --------------------------------------------------------------------------- #


class FakeEnumerated:
    """Mirrors the field names of `agent.ghost.enumerate.ChainCandidate`.

    A stub rather than the real class on purpose: that module is written
    concurrently, and this test asserts the ADAPTER's behaviour, not its
    sibling's current field list.
    """

    chain_id = "gc_abc123"
    source_entity = "RU"
    source_name = "Russia"
    source_entity_type = "country"
    source_tool = "gdelt"
    observation_type = "geopolitical_event"
    target_entity = "WTI"
    target_entity_type = "instrument"
    path = ("RU", "WTI")
    path_types = ("country", "instrument")
    link_types = ("produced_in",)
    hops = 1
    lag_low_days = 2
    lag_high_days = 23


def test_from_enumerated_maps_the_route_and_names_what_it_lost():
    c, warns = from_enumerated(FakeEnumerated(), event_field="goldstein", z_threshold=2.0, direction="down")
    assert c.event_source == "gdelt"
    assert c.event_obs_type == "geopolitical_event"
    assert c.target_entity_type == "instrument"
    assert c.link_types == ("produced_in",)
    assert c.event_entity_type == "country"
    assert c.metapath == ("country", "produced_in", "instrument")
    assert c.publication_lag_s == 2 * DAY
    assert "gc_abc123" in c.provenance
    joined = " ".join(warns)
    assert "entity restriction LOST" in joined
    assert "another member of the BH family" in joined


def test_from_enumerated_warns_that_a_multi_hop_route_is_collapsed():
    class TwoHop(FakeEnumerated):
        hops = 2
        link_types = ("produced_in", "exchange_country")
        path_types = ("country", "instrument", "country")

    _, warns = from_enumerated(TwoHop(), event_field="goldstein", z_threshold=2.0, direction="abs")
    assert any("collapsed to a link_types filter" in w for w in warns)


def test_from_enumerated_rejects_an_object_that_is_not_a_route():
    with pytest.raises(CandidateError, match="not an enumerated chain candidate"):
        from_enumerated(object(), event_field="x", z_threshold=1.0, direction="up")


def test_from_enumerated_warns_when_it_has_to_default_the_publication_lag():
    class NoLag(FakeEnumerated):
        lag_low_days = 0

    _, warns = from_enumerated(NoLag(), event_field="goldstein", z_threshold=2.0, direction="up")
    assert any("free lookahead" in w for w in warns)


def test_horizon_from_lag_window_is_never_zero():
    assert horizon_from_lag_window(FakeEnumerated()) == 21

    class Degenerate(FakeEnumerated):
        lag_low_days = 5
        lag_high_days = 5

    assert horizon_from_lag_window(Degenerate()) == 1


def test_adapted_candidate_is_scoreable(tmp_path):
    db = make_db(tmp_path)

    class CftcRoute(FakeEnumerated):
        source_entity_type = "cftc_contract"
        source_tool = "cftc"
        observation_type = "futures_positioning"
        link_types = ("cftc_tracks",)
        path_types = ("cftc_contract", "instrument")
        lag_low_days = 0

    c, _ = from_enumerated(CftcRoute(), event_field="sig", z_threshold=2.0, direction="up")
    board = score_all([c], db, horizons=(5, 10))
    assert board.n_tested == 2
    assert board.n_testable >= 1


def test_as_of_is_part_of_the_fingerprint_so_a_widened_window_is_a_second_look(tmp_path):
    """Same family, same file, later cutoff: that is a SECOND LOOK, not a replication.

    Without the cutoff in the fingerprint the two boards compare as "identical
    data, no new evidence", which is precisely how the CFTC family's 0-of-51
    became "3 of 54 survivors" on a re-run.
    """
    db = make_db(tmp_path)
    early = score_all(family(2), db, horizons=(5,), as_of=200 * DAY)
    late = score_all(family(2), db, horizons=(5,))
    assert early.family_id == late.family_id
    assert early.as_of == 200 * DAY and late.as_of is None
    assert early.db_fingerprint != late.db_fingerprint
    w = " ".join(relook_warnings(early, late))
    assert "SECOND LOOK" in w
    assert "as_of cutoff moved" in w
    assert "no new evidence" not in w
    assert "as_of" in early.report()
    assert early.to_dict()["as_of"] == 200 * DAY


def test_as_of_actually_narrows_the_sample(tmp_path):
    db = make_db(tmp_path)
    early = score_chain(cand(), db, horizons=(5,), as_of=200 * DAY).cells[0]
    late = score_chain(cand(), db, horizons=(5,)).cells[0]
    assert 0 < early.n_events < late.n_events
