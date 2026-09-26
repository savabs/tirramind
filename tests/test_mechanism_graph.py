"""Tests for agent.mechanism.graph.

Two kinds of test here, deliberately:

*   Synthetic tests on a throwaway SQLite file with the real `entities` /
    `entity_links` schema. These pin the contract: dedup, as_of gating, the
    evidence/scaffold split, and loud failure on degenerate input.
*   Live-DB acceptance tests, skipped when `.tirra_pipeline/pipeline.db` is
    absent. These pin the numbers the mechanism layer was designed against,
    including a personalised PageRank computed here in the test (the algorithm
    lives in another module; what is being tested is that THIS module's
    adjacency is the one those numbers came from).

The live DB is opened read-only, and the loader has no other mode.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from agent.mechanism.graph import (
    DisconnectedEntityError,
    EmptyGraphError,
    MechanismGraphError,
    UnknownEntityError,
    as_of_from_iso,
    load_graph,
)

# ── fixtures ───────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    created_at REAL NOT NULL,
    metadata_json TEXT
);
CREATE TABLE entity_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id_a TEXT NOT NULL,
    entity_id_b TEXT NOT NULL,
    link_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL,
    created_at REAL NOT NULL,
    metadata_json TEXT,
    effective_from REAL
);
"""

T2015 = as_of_from_iso("2015-01-01")
T2019 = as_of_from_iso("2019-01-01")
T2024 = as_of_from_iso("2024-01-01")
T2026 = as_of_from_iso("2026-01-01")


def _make_db(tmp_path: Path, entities, links, name: str = "t.db") -> Path:
    """entities: (id, type, name). links: (a, b, link_type, source, eff_from, created)."""
    path = tmp_path / name
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO entities (entity_id, entity_type, canonical_name, created_at) VALUES (?,?,?,0)",
        entities,
    )
    con.executemany(
        "INSERT INTO entity_links (entity_id_a, entity_id_b, link_type, confidence, "
        "source, created_at, effective_from) VALUES (?,?,?,1.0,?,?,?)",
        [(a, b, lt, src, created, eff) for a, b, lt, src, eff, created in links],
    )
    con.commit()
    con.close()
    return path


@pytest.fixture
def toy_db(tmp_path: Path) -> Path:
    """Russia -> {Canada, Nigeria} -> WTI, plus a hub and an isolated node.

    Mirrors the real shape: the only time-varying edges are event_involves, and
    every route to the instrument runs over a static `produced_in`.
    """
    entities = [
        ("ru", "country", "Russia"),
        ("ca", "country", "Canada"),
        ("ng", "country", "Nigeria"),
        ("us", "country", "United States"),
        ("wti", "instrument", "WTI Crude Oil"),
        ("lonely", "vessel", "MV Nobody"),
    ]
    links = [
        # event edges, dated
        ("ru", "ca", "event_involves", "gdelt", T2015, T2026),
        ("ru", "ng", "event_involves", "gdelt", T2024, T2026),
        ("ru", "us", "event_involves", "gdelt", T2015, T2026),
        # the same pair stored in both directions: one edge, not two
        ("ca", "ru", "event_involves", "gdelt", T2015, T2026),
        # a second relation over an existing pair: a second edge, same adjacency cell
        ("ru", "us", "transacts_with", "whale_alert", T2015, T2026),
        # scaffold, undated (effective_from NULL -> created_at, which is recent)
        ("ca", "wti", "produced_in", "eia", None, T2026),
        ("ng", "wti", "produced_in", "eia", None, T2026),
        ("us", "wti", "produced_in", "eia", None, T2026),
    ]
    return _make_db(tmp_path, entities, links)


# ── contract: as_of is required and real ────────────────────────


def test_as_of_is_a_required_keyword(toy_db: Path) -> None:
    """A defaulted time bound is how F-14 happened; there must be no default."""
    with pytest.raises(TypeError):
        load_graph(toy_db)  # type: ignore[call-arg]


def test_as_of_excludes_future_event_edges(toy_db: Path) -> None:
    now = load_graph(toy_db, as_of=None)
    past = load_graph(toy_db, as_of=T2019)

    assert now.edges_between("ru", "ng") != ()
    assert past.edges_between("ru", "ng") == ()  # the 2024 event had not happened
    assert past.edges_between("ru", "ca") != ()  # the 2015 event had
    assert past.report.rows_dropped_by_as_of == 1
    assert past.report.n_undirected_pairs < now.report.n_undirected_pairs


def test_scaffold_is_timeless_by_default_and_gated_on_request(toy_db: Path) -> None:
    """Geography must route a historical question; the strict audit must not."""
    lax = load_graph(toy_db, as_of=T2019)
    strict = load_graph(toy_db, as_of=T2019, scaffold_policy="gated")

    # produced_in is undated, so its created_at (2026) gates it out under "gated".
    assert lax.edges_between("ca", "wti") != ()
    assert strict.edges_between("ca", "wti") == ()
    assert lax.degree("wti") == 3
    assert strict.degree("wti") == 0
    assert lax.report.scaffold_policy == "timeless"
    assert strict.report.scaffold_policy == "gated"


def test_effective_from_null_falls_back_to_created_at(tmp_path: Path) -> None:
    ents = [("a", "country", "A"), ("b", "country", "B")]
    links = [("a", "b", "event_involves", "gdelt", None, T2024)]
    db = _make_db(tmp_path, ents, links)

    assert load_graph(db, as_of=T2026).report.n_edges == 1
    with pytest.raises(EmptyGraphError):
        load_graph(db, as_of=T2019)  # created_at 2024 > 2019, nothing survives
    assert load_graph(db, as_of=T2026).report.rows_with_effective_from == 0


def test_as_of_from_iso_is_utc_and_start_of_day() -> None:
    import datetime as dt

    assert as_of_from_iso("2019-01-01") == dt.datetime(2019, 1, 1, tzinfo=dt.UTC).timestamp()
    assert as_of_from_iso("2019-01-01T00:00:00+00:00") == as_of_from_iso("2019-01-01")


# ── contract: dedup ────────────────────────────────────────────


def test_parallel_edge_in_both_directions_collapses_to_one(toy_db: Path) -> None:
    g = load_graph(toy_db, as_of=None)
    ru_ca = g.edges_between("ru", "ca")
    assert len(ru_ca) == 1, "the same pair+link_type stored a->b and b->a is ONE edge"
    assert g.report.rows_collapsed_by_dedup == 1
    assert g.degree("ca") == 2  # ru and wti, counted once each


def test_two_link_types_over_one_pair_are_two_edges_but_one_adjacency_cell(toy_db: Path) -> None:
    g = load_graph(toy_db, as_of=None)
    assert len(g.edges_between("ru", "us")) == 2
    assert {e.link_type for e in g.edges_between("ru", "us")} == {"event_involves", "transacts_with"}
    i, j = g.index_of("ru"), g.index_of("us")
    assert g.adjacency[i, j] == 1.0, "adjacency must be binary: multiplicity is not evidence"
    assert g.n_edges > g.report.n_undirected_pairs


def test_adjacency_is_symmetric_binary_and_loop_free(toy_db: Path) -> None:
    g = load_graph(toy_db, as_of=None)
    for m in (g.adjacency, g.evidence_adjacency):
        assert (m != m.T).nnz == 0
        assert set(np.unique(m.data)) <= {1.0}
        assert m.diagonal().sum() == 0


def test_self_loops_are_dropped_and_counted(tmp_path: Path) -> None:
    ents = [("a", "country", "A"), ("b", "country", "B")]
    links = [
        ("a", "a", "event_involves", "gdelt", T2015, T2015),
        ("a", "b", "event_involves", "gdelt", T2015, T2015),
    ]
    g = load_graph(_make_db(tmp_path, ents, links), as_of=None)
    assert g.report.rows_dropped_self_loop == 1
    assert g.report.n_edges == 1


# ── contract: evidence vs scaffold ─────────────────────────────


def test_evidence_adjacency_excludes_scaffold(toy_db: Path) -> None:
    g = load_graph(toy_db, as_of=None)
    i, j = g.index_of("ca"), g.index_of("wti")
    assert g.adjacency[i, j] == 1.0
    assert g.evidence_adjacency[i, j] == 0.0, "produced_in is geography, not evidence"
    assert g.degree("wti") == 3
    assert g.degree("wti", evidence_only=True) == 0


def test_relation_sets_are_not_redefined_here() -> None:
    """The split must come from graph_builder, or the two copies will drift."""
    import agent.mechanism.graph as mg
    from agent.models.gnn.graph_builder import EVENT_RELATIONS, STRUCTURAL_RELATIONS

    assert mg.EVENT_RELATIONS is EVENT_RELATIONS
    assert mg.STRUCTURAL_RELATIONS is STRUCTURAL_RELATIONS


def test_scaffold_source_values_do_not_masquerade_as_independent_support(toy_db: Path) -> None:
    """The measured trap: 5 sources on a Russia->WTI route that is ONE observation."""
    g = load_graph(toy_db, as_of=None)
    route = [*g.edges_between("ru", "ca"), *g.edges_between("ca", "wti")]
    all_sources = {e.source for e in route}
    evidence_sources = {e.source for e in route if e.is_evidence}
    assert len(all_sources) == 2  # gdelt + eia: naive counting says "2 datasets"
    assert evidence_sources == {"gdelt"}  # only one thing actually varied with the date


def test_unknown_link_type_is_reported_not_silently_filed(tmp_path: Path) -> None:
    ents = [("a", "country", "A"), ("b", "country", "B")]
    links = [("a", "b", "invented_relation", "somewhere", T2024, T2024)]
    g = load_graph(_make_db(tmp_path, ents, links), as_of=None)
    assert g.report.unknown_link_types == ("invented_relation",)
    # treated as evidence => gated on its date, the conservative side
    assert g.edges[0].is_evidence is True
    with pytest.raises(EmptyGraphError):
        load_graph(_make_db(tmp_path, ents, links, name="u2.db"), as_of=T2019)


# ── contract: loud failure, never a plausible zero ─────────────


def test_unknown_entity_raises_rather_than_returning_a_sentinel(toy_db: Path) -> None:
    g = load_graph(toy_db, as_of=None)
    for call in (
        lambda: g.index_of("nope"),
        lambda: g.name_of("nope"),
        lambda: g.type_of("nope"),
        lambda: g.degree("nope"),
        lambda: g.neighbours("nope"),
        lambda: g.incident_edges("nope"),
        lambda: g.edges_between("nope", "ru"),
        lambda: g.degree_percentile("nope"),
    ):
        with pytest.raises(UnknownEntityError):
            call()
    assert g.has("nope") is False
    assert g.has("ru") is True


def test_isolated_node_has_degree_zero_but_require_connected_raises(toy_db: Path) -> None:
    g = load_graph(toy_db, as_of=None)
    assert g.degree("lonely") == 0  # a true zero
    assert g.neighbours("lonely") == ()
    assert g.report.n_isolated_entities == 1
    with pytest.raises(DisconnectedEntityError):
        g.require_connected("lonely")
    assert g.require_connected("ru") == g.index_of("ru")
    # "no evidence edges" is its own failure, distinct from "no edges at all"
    with pytest.raises(DisconnectedEntityError):
        g.require_connected("wti", evidence_only=True)


def test_empty_entities_and_no_surviving_edges_both_raise(tmp_path: Path) -> None:
    with pytest.raises(EmptyGraphError):
        load_graph(_make_db(tmp_path, [], [], name="e.db"), as_of=None)
    ents = [("a", "country", "A"), ("b", "country", "B")]
    links = [("a", "b", "event_involves", "gdelt", T2024, T2024)]
    with pytest.raises(EmptyGraphError):
        load_graph(_make_db(tmp_path, ents, links, name="e2.db"), as_of=T2015)


def test_missing_db_raises_instead_of_creating_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_graph(tmp_path / "absent.db", as_of=None)
    assert not (tmp_path / "absent.db").exists()


def test_row_referencing_an_absent_entity_is_dropped_and_counted(tmp_path: Path) -> None:
    ents = [("a", "country", "A"), ("b", "country", "B")]
    links = [
        ("a", "b", "event_involves", "gdelt", T2015, T2015),
        ("a", "ghost", "event_involves", "gdelt", T2015, T2015),
    ]
    g = load_graph(_make_db(tmp_path, ents, links), as_of=None)
    assert g.report.rows_dropped_unknown_entity == 1
    assert g.report.n_edges == 1


def test_bad_scaffold_policy_raises(toy_db: Path) -> None:
    with pytest.raises(ValueError, match="scaffold_policy"):
        load_graph(toy_db, as_of=None, scaffold_policy="whatever")  # type: ignore[arg-type]


def test_connection_is_read_only(toy_db: Path) -> None:
    from agent.mechanism.graph import _connect_readonly

    con = _connect_readonly(toy_db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("DELETE FROM entity_links")
    finally:
        con.close()


# ── contract: accounting, no silent truncation ─────────────────


def test_every_row_read_is_accounted_for(toy_db: Path) -> None:
    for as_of in (None, T2019, T2015):
        g = load_graph(toy_db, as_of=as_of)
        r = g.report
        assert (
            r.n_edges
            + r.rows_collapsed_by_dedup
            + r.rows_dropped_by_as_of
            + r.rows_dropped_self_loop
            + r.rows_dropped_unknown_entity
            == r.rows_read
        )
        assert r.rows_kept == r.n_edges + r.rows_collapsed_by_dedup
        assert r.truncated is False
        assert sum(r.link_type_counts.values()) == r.n_edges
        assert r.n_evidence_pairs + r.n_scaffold_pairs >= r.n_undirected_pairs


def test_accounting_guard_raises_when_rows_go_missing() -> None:
    """The guard itself, fed numbers that do not add up."""
    from agent.mechanism.graph import _reconcile

    ok = dict(rows_read=10, kept=6, collapsed=2, dropped_as_of=1, dropped_self=1, dropped_unknown=0)
    _reconcile(**ok)  # reconciles, returns None
    for field in ("kept", "collapsed", "dropped_as_of", "dropped_self", "dropped_unknown"):
        bad = dict(ok)
        bad[field] -= 1
        with pytest.raises(MechanismGraphError, match="accounting does not reconcile"):
            _reconcile(**bad)


# ── contract: hub-ness is inspectable ──────────────────────────


def test_hubs_are_ranked_and_never_look_complete(toy_db: Path) -> None:
    g = load_graph(toy_db, as_of=None)
    top, n_ranked = g.hubs(2)
    assert n_ranked == g.n_nodes, "a top-N must report the size of the population it came from"
    assert len(top) == 2
    degrees = [h.degree for h in top]
    assert degrees == sorted(degrees, reverse=True)
    assert top[0].entity_id in {"ru", "wti"}  # both degree 3
    assert g.hubs(0)[0] == ()
    with pytest.raises(ValueError):
        g.hubs(-1)


def test_degree_percentile_places_the_hub_above_the_isolated_node(toy_db: Path) -> None:
    g = load_graph(toy_db, as_of=None)
    assert g.degree_percentile("lonely") == 0.0
    assert g.degree_percentile("ru") > g.degree_percentile("ca")


def test_node_order_is_stable_across_loads(toy_db: Path) -> None:
    a = load_graph(toy_db, as_of=None)
    b = load_graph(toy_db, as_of=T2019)
    assert a.entity_ids == b.entity_ids, "index maps must be interchangeable across as_of"
    assert a.entity_ids == tuple(sorted(a.entity_ids))


def test_entities_of_type(toy_db: Path) -> None:
    g = load_graph(toy_db, as_of=None)
    assert g.entities_of_type("instrument") == ("wti",)
    assert g.entities_of_type("nonexistent") == ()


# ── live DB acceptance ─────────────────────────────────────────

LIVE_DB = Path(__file__).resolve().parents[1] / ".tirra_pipeline" / "pipeline.db"
live_only = pytest.mark.skipif(not LIVE_DB.exists(), reason="live pipeline.db not present")

RUSSIA = "c9db6e2a9784c856"
WTI = "da678733bb9c745f"
UNITED_STATES = "f07b20bdd8b5f9bb"


@pytest.fixture(scope="module")
def live_now():
    if not LIVE_DB.exists():
        pytest.skip("live pipeline.db not present")
    return load_graph(LIVE_DB, as_of=None)


def _ppr(adj: sp.csr_matrix, seed_index: int, alpha: float = 0.15, iters: int = 300) -> np.ndarray:
    """Personalised PageRank by power iteration. Reference implementation for the test only.

    Restart probability `alpha`, row-stochastic transition from a symmetric
    adjacency, dangling mass returned to the seed.
    """
    n = adj.shape[0]
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros(n)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    trans = (sp.diags(inv) @ adj).tocsr()
    restart = np.zeros(n)
    restart[seed_index] = 1.0
    p = restart.copy()
    for _ in range(iters):
        q = alpha * restart + (1.0 - alpha) * (trans.T @ p)
        q += (1.0 - q.sum()) * restart
        if np.abs(q - p).sum() < 1e-13:
            return q
        p = q
    return p


@live_only
def test_live_graph_shape(live_now) -> None:
    r = live_now.report
    assert r.n_entities == 20026
    assert r.rows_read == 30131
    assert r.rows_with_effective_from == 28302
    assert r.n_edges == 26087  # distinct (pair, link_type)
    assert r.n_undirected_pairs == 26056  # distinct pairs
    assert r.rows_collapsed_by_dedup == 4044
    assert r.rows_dropped_by_as_of == 0  # as_of=None keeps everything
    assert r.unknown_link_types == ()
    assert len(r.link_type_counts) == 18


@live_only
def test_live_graph_shrinks_at_2019(live_now) -> None:
    past = load_graph(LIVE_DB, as_of=as_of_from_iso("2019-01-01"))
    assert past.report.n_undirected_pairs == 5942
    assert past.report.n_undirected_pairs < live_now.report.n_undirected_pairs / 4
    assert past.report.rows_dropped_by_as_of == 20195
    assert past.entity_ids == live_now.entity_ids


@live_only
def test_live_evidence_is_most_of_the_graph_but_not_all(live_now) -> None:
    r = live_now.report
    assert r.n_evidence_pairs == 25463
    assert r.n_evidence_pairs < r.n_undirected_pairs
    # produced_in routes every country to an instrument and is scaffold
    assert live_now.evidence_adjacency[live_now.index_of(UNITED_STATES), live_now.index_of(WTI)] == 0.0
    assert live_now.adjacency[live_now.index_of(UNITED_STATES), live_now.index_of(WTI)] == 1.0


@live_only
def test_live_hub_degrees_are_the_deduplicated_ones(live_now) -> None:
    assert live_now.degree(UNITED_STATES) == 522
    top, n_ranked = live_now.hubs(3)
    assert n_ranked == 20026
    assert top[0].canonical_name == "Bitcoin"
    assert top[0].degree == 2010


@live_only
def test_live_ppr_from_russia_reproduces_the_reference_numbers(live_now) -> None:
    """The acceptance test for this module's adjacency.

    Reference (alpha=0.15, restart from Russia, binary symmetric adjacency):
    WTI mass 2.648e-03, rank 127 of 20,026, instrument #1 of 93, 28x the median
    instrument. If any of these move, the adjacency changed — check dedup and
    weighting before blaming the walk.
    """
    seed = live_now.require_connected(RUSSIA)
    p = _ppr(live_now.adjacency, seed)
    mass = float(p[live_now.index_of(WTI)])

    assert mass == pytest.approx(2.648e-03, rel=5e-4)
    rank = int((p > mass).sum()) + 1
    assert rank == 127
    assert (live_now.n_nodes - rank) / live_now.n_nodes * 100 == pytest.approx(99.37, abs=0.01)

    instruments = live_now.entities_of_type("instrument")
    assert len(instruments) == 93
    inst_mass = np.array([p[live_now.index_of(i)] for i in instruments])
    assert mass == inst_mass.max()
    assert mass / float(np.median(inst_mass)) == pytest.approx(28.1, abs=0.2)


@live_only
def test_live_russia_to_wti_routes_have_one_independent_source(live_now) -> None:
    """Requirement A on real data: five `source` values, one actual observation."""
    producers = [e for e in live_now.incident_edges(WTI) if e.link_type == "produced_in"]
    assert producers, "produced_in is the scaffold every route uses"
    ru = live_now.index_of(RUSSIA)
    hop_sources: set[str] = set()
    evidence_sources: set[str] = set()
    for edge in producers:
        country = edge.a if edge.b == live_now.index_of(WTI) else edge.b
        via = live_now.edges_between(RUSSIA, live_now.entity_ids[country])
        if not via:
            continue
        hop_sources |= {e.source for e in via} | {edge.source}
        evidence_sources |= {e.source for e in via if e.is_evidence}
    assert len(hop_sources) > 1, "naive source counting inflates this"
    assert evidence_sources == {"gdelt"}, "only the event hop varies with the date"
    assert ru == live_now.index_of(RUSSIA)


@live_only
def test_live_load_does_not_modify_the_database() -> None:
    before = (LIVE_DB.stat().st_size, LIVE_DB.stat().st_mtime_ns)
    load_graph(LIVE_DB, as_of=None)
    assert (LIVE_DB.stat().st_size, LIVE_DB.stat().st_mtime_ns) == before
