"""
Tests for agent/mechanism/spectral.py.

Two halves:

1. Synthetic graphs with hand-computable answers. These are the tests that go
   red when the implementation is wrong: the spectrum is checked against a
   dense numpy SVD of the matrix the module claims to decompose, so a wrong
   symmetrisation, a kept diagonal or a silently reduced rank all fail here.

2. Live-graph tests, read-only, skipped when .tirra_pipeline/pipeline.db is
   absent. These pin BOTH the headline result (effective rank ~50 of 64) and
   the module's two measured limitations (16% coverage; "Russia" similarity is
   a degree ranking). Pinning the limitations is deliberate — if a future change
   improves them, the assertion must be updated, which forces the new number
   into the report instead of letting the old claim stand.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from agent.mechanism.spectral import (
    DegenerateEmbeddingError,
    EmptyGraphError,
    RankTooLargeError,
    SpectralGraph,
    UnknownEntityError,
    clear_cache,
    degree_rank_correlation,
    effective_rank,
    load_graph,
    nearest_structural,
    spectral_embedding,
    spectral_model,
    structural_similarity,
)

LIVE_DB = Path(__file__).resolve().parents[1] / ".tirra_pipeline" / "pipeline.db"
live_only = pytest.mark.skipif(not LIVE_DB.exists(), reason="live pipeline.db not present")


# ── Fixtures ───────────────────────────────────────────────────────


def _graph_from_edges(
    edges: list[tuple[str, str]],
    *,
    extra_nodes: list[str] | None = None,
    types: dict[str, str] | None = None,
) -> SpectralGraph:
    nodes = sorted({n for e in edges for n in e} | set(extra_nodes or []))
    index = {n: i for i, n in enumerate(nodes)}
    rows = [index[a] for a, _ in edges] + [index[b] for _, b in edges]
    cols = [index[b] for _, b in edges] + [index[a] for a, _ in edges]
    matrix = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    matrix = (matrix > 0).astype(np.float64)
    entity_types = [(types or {}).get(n, "widget") for n in nodes]
    return SpectralGraph(matrix, nodes, entity_types, [n.upper() for n in nodes], {"synthetic": True})


@pytest.fixture(autouse=True)
def _no_cross_test_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def two_clusters() -> SpectralGraph:
    """
    Two 4-cliques joined by one bridge edge, plus two structural twins that
    attach to the same three cluster-A nodes but not to each other, plus a
    genuinely isolated node.

    Hand-known facts:
        t1 and t2 have byte-identical adjacency rows -> cosine exactly 1.
        iso has degree 0 -> no position at all.
    """
    a = ["a0", "a1", "a2", "a3"]
    b = ["b0", "b1", "b2", "b3"]
    edges = [(x, y) for i, x in enumerate(a) for y in a[i + 1 :]]
    edges += [(x, y) for i, x in enumerate(b) for y in b[i + 1 :]]
    edges += [("a0", "b0")]
    edges += [("t1", n) for n in ("a1", "a2", "a3")]
    edges += [("t2", n) for n in ("a1", "a2", "a3")]
    return _graph_from_edges(
        edges,
        extra_nodes=["iso"],
        types={"t1": "twin", "t2": "twin", "iso": "widget"},
    )


@pytest.fixture
def lopsided() -> SpectralGraph:
    """A dense 6-clique plus a far-away single edge: at rank 1 only the clique
    is in the subspace, so the pair is connected-but-uncovered."""
    c = [f"c{i}" for i in range(6)]
    edges = [(x, y) for i, x in enumerate(c) for y in c[i + 1 :]]
    edges += [("p0", "p1")]
    return _graph_from_edges(edges)


def _write_temp_db(path: Path, entities: list[tuple[str, str, str]], links: list[tuple]) -> None:
    """Build a throwaway pipeline-shaped DB. Never touches the live database."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL, created_at REAL NOT NULL, metadata_json TEXT);
        CREATE TABLE entity_links (
            link_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id_a TEXT NOT NULL, entity_id_b TEXT NOT NULL, link_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0, source TEXT NOT NULL, created_at REAL NOT NULL,
            metadata_json TEXT, effective_from REAL, UNIQUE(entity_id_a, entity_id_b, link_type));
        """
    )
    conn.executemany("INSERT INTO entities VALUES (?,?,?,0.0,NULL)", entities)
    conn.executemany(
        "INSERT INTO entity_links (entity_id_a, entity_id_b, link_type, confidence, source, "
        "created_at, effective_from) VALUES (?,?,?,?,?,?,?)",
        links,
    )
    conn.commit()
    conn.close()


# ── effective_rank ─────────────────────────────────────────────────


def test_effective_rank_of_flat_spectrum_is_its_length():
    assert effective_rank(np.ones(8)) == pytest.approx(8.0)
    assert effective_rank([3.0] * 5) == pytest.approx(5.0)


def test_effective_rank_of_one_direction_is_one():
    assert effective_rank([7.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_effective_rank_falls_as_mass_concentrates():
    flat = effective_rank([1.0, 1.0, 1.0, 1.0])
    skewed = effective_rank([1.0, 0.1, 0.1, 0.1])
    peaked = effective_rank([1.0, 0.001, 0.001, 0.001])
    assert flat > skewed > peaked > 1.0


def test_effective_rank_is_order_independent():
    s = np.array([5.0, 0.2, 3.0, 0.9])
    assert effective_rank(s) == pytest.approx(effective_rank(s[::-1]))


def test_effective_rank_is_scale_invariant():
    s = np.array([5.0, 0.2, 3.0, 0.9])
    assert effective_rank(s) == pytest.approx(effective_rank(s * 1000.0))


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(np.zeros(4), id="all-zero"),
        pytest.param([], id="empty"),
        pytest.param([1.0, -1.0], id="negative"),
        pytest.param([1.0, np.nan], id="nan"),
        pytest.param([1.0, np.inf], id="inf"),
        pytest.param(np.ones((2, 2)), id="two-dimensional"),
    ],
)
def test_effective_rank_refuses_degenerate_input(bad):
    """An all-zero spectrum must NOT come back as 0.0 or 1.0 — a plausible
    number here would silently pass an acceptance threshold."""
    with pytest.raises(ValueError):
        effective_rank(bad)


# ── spectral_embedding / spectral_model ────────────────────────────


def test_embedding_shapes_match_requested_rank(two_clusters):
    U, S = spectral_embedding(two_clusters, rank=5)
    assert U.shape == (two_clusters.adjacency.shape[0], 5)
    assert S.shape == (5,)


def test_singular_values_are_descending(two_clusters):
    _, S = spectral_embedding(two_clusters, rank=6)
    assert np.all(np.diff(S) <= 1e-12)


def test_spectrum_matches_dense_svd_of_the_stated_matrix(two_clusters):
    """
    The module documents that it decomposes the symmetrised, zero-diagonal,
    binarised adjacency. This checks that claim against a dense reference, so a
    wrong normalisation or a kept diagonal cannot hide behind a plausible
    effective rank.
    """
    dense = two_clusters.adjacency.toarray()
    reference = np.linalg.svd(dense, compute_uv=False)[:6]
    _, S = spectral_embedding(two_clusters, rank=6)
    assert pytest.approx(reference, rel=1e-8, abs=1e-8) == S


def test_asymmetric_input_and_self_loops_are_normalised_away():
    n = 6
    rng = np.random.default_rng(0)
    upper = np.triu((rng.random((n, n)) > 0.4).astype(float), 1)
    asym = upper.copy()
    np.fill_diagonal(asym, 1.0)  # self-loops the module promises to drop
    g = SpectralGraph(sp.csr_matrix(asym), [f"n{i}" for i in range(n)])
    symmetric = ((upper + upper.T) > 0).astype(float)
    reference = np.linalg.svd(symmetric, compute_uv=False)[:3]
    _, S = spectral_embedding(g, rank=3)
    assert pytest.approx(reference, rel=1e-8, abs=1e-8) == S


def test_rank_too_large_raises_instead_of_silently_shrinking(two_clusters):
    n = two_clusters.adjacency.shape[0]
    with pytest.raises(RankTooLargeError) as err:
        spectral_embedding(two_clusters, rank=n)
    assert str(n - 1) in str(err.value)


def test_returned_spectrum_length_always_equals_requested_rank(two_clusters):
    for rank in (1, 3, 7):
        _, S = spectral_embedding(two_clusters, rank=rank)
        assert len(S) == rank, "a shortened spectrum makes effective_rank unfalsifiable"


def test_rank_below_one_raises(two_clusters):
    with pytest.raises(RankTooLargeError):
        spectral_embedding(two_clusters, rank=0)


def test_graph_with_no_edges_raises_rather_than_returning_zeros():
    g = SpectralGraph(sp.csr_matrix((5, 5)), [f"n{i}" for i in range(5)])
    with pytest.raises(EmptyGraphError):
        spectral_embedding(g, rank=2)


def test_graph_with_no_nodes_raises():
    g = SpectralGraph(sp.csr_matrix((0, 0)), [])
    with pytest.raises(EmptyGraphError):
        spectral_embedding(g, rank=1)


def test_embedding_is_deterministic_across_calls(two_clusters):
    clear_cache()
    u1, s1 = spectral_embedding(two_clusters, rank=4)
    clear_cache()
    u2, s2 = spectral_embedding(two_clusters, rank=4)
    assert s1 == pytest.approx(s2)
    assert np.abs(u1) == pytest.approx(np.abs(u2), abs=1e-8)


def test_coverage_report_counts_the_unreachable_rows(lopsided):
    model = spectral_model(lopsided, rank=1)
    report = model.coverage_report()
    assert report["n_nodes"] == 8
    assert report["n_covered"] == 6, "only the 6-clique lies in the top-1 subspace"
    assert report["coverage_fraction"] == pytest.approx(6 / 8)
    assert report["n_components"] == 2
    assert report["n_components_covered"] == 1
    assert report["largest_component_size"] == 6


def test_coverage_report_breaks_down_by_entity_type(two_clusters):
    per_type = spectral_model(two_clusters, rank=4).coverage_report()["covered_by_entity_type"]
    assert per_type["twin"][1] == 2
    assert per_type["widget"][1] == 9  # 8 clique members + iso


def test_raw_sparse_matrix_is_accepted_without_ids():
    matrix = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    U, S = spectral_embedding(matrix, rank=2)
    assert U.shape == (3, 2)
    assert S[0] > S[1]


def test_object_without_an_adjacency_matrix_is_refused_loudly():
    class NotAGraph:
        pass

    with pytest.raises(TypeError) as err:
        spectral_embedding(NotAGraph(), rank=1)
    assert "adjacency" in str(err.value)


# ── structural_similarity ──────────────────────────────────────────


def test_structural_twins_score_one(two_clusters):
    """t1 and t2 have identical neighbour sets and no edge between them —
    the exact case a path search cannot see and this module can."""
    assert structural_similarity(two_clusters, "t1", "t2", rank=5) == pytest.approx(1.0, abs=1e-8)
    assert two_clusters.adjacency[two_clusters.node_ids.index("t1"), two_clusters.node_ids.index("t2")] == 0


def test_within_cluster_beats_across_cluster(two_clusters):
    within = structural_similarity(two_clusters, "a1", "a2", rank=5)
    across = structural_similarity(two_clusters, "a1", "b1", rank=5)
    assert within > across


def test_similarity_is_symmetric_and_bounded(two_clusters):
    ab = structural_similarity(two_clusters, "a1", "b2", rank=5)
    ba = structural_similarity(two_clusters, "b2", "a1", rank=5)
    assert ab == pytest.approx(ba)
    assert -1.0 - 1e-9 <= ab <= 1.0 + 1e-9


def test_self_similarity_is_one(two_clusters):
    assert structural_similarity(two_clusters, "a1", "a1", rank=5) == pytest.approx(1.0, abs=1e-8)


def test_unknown_entity_raises_and_does_not_return_zero(two_clusters):
    with pytest.raises(UnknownEntityError) as err:
        structural_similarity(two_clusters, "a1", "no-such-entity", rank=5)
    assert "no-such-entity" in str(err.value)


def test_isolated_entity_raises_degree_zero_not_similarity_zero(two_clusters):
    with pytest.raises(DegenerateEmbeddingError) as err:
        structural_similarity(two_clusters, "a1", "iso", rank=5)
    assert "degree 0" in str(err.value)


def test_connected_but_untruncated_entity_gets_a_different_message(lopsided):
    """
    The two kinds of "no answer" must be distinguishable: `p0` has degree 1 but
    sits outside a rank-1 subspace. Collapsing this into the degree-0 message
    would tell a caller the graph lacks a link it actually has.
    """
    with pytest.raises(DegenerateEmbeddingError) as err:
        structural_similarity(lopsided, "c0", "p0", rank=1)
    message = str(err.value)
    assert "degree 0" not in message
    assert "outside the top-1 singular subspace" in message
    assert "not zero" in message


def test_similarity_needs_entity_ids(two_clusters):
    matrix = two_clusters.adjacency
    with pytest.raises(UnknownEntityError):
        structural_similarity(matrix, "a1", "a2", rank=3)


# ── nearest_structural ─────────────────────────────────────────────


def test_nearest_structural_puts_the_twin_first(two_clusters):
    hits = nearest_structural(two_clusters, "t1", k=3, same_type=False, rank=5)
    assert hits[0].entity_id == "t2"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-8)
    assert two_clusters.adjacency[two_clusters.node_ids.index("t1"), two_clusters.node_ids.index("t2")] == 0
    assert hits[0].hops == 2, "no edge joins the twins — they are found through their shared neighbours"


def test_nearest_structural_is_sorted_descending(two_clusters):
    sims = [h.similarity for h in nearest_structural(two_clusters, "a1", k=8, same_type=False, rank=5)]
    assert sims == sorted(sims, reverse=True)


def test_nearest_structural_reports_the_truncation_it_performed(two_clusters):
    hits = nearest_structural(two_clusters, "a1", k=2, same_type=False, rank=5)
    assert len(hits) == 2
    assert hits.truncated is True
    assert hits.n_candidates == 9  # 10 covered nodes, minus the query itself
    assert hits.k_requested == 2


def test_nearest_structural_does_not_claim_truncation_when_none_happened(two_clusters):
    hits = nearest_structural(two_clusters, "t1", k=50, same_type=True, rank=5)
    assert hits.truncated is False
    assert len(hits) == hits.n_candidates == 1
    assert hits.excluded_other_type == 8  # 8 covered widgets, excluded by type


def test_nearest_structural_reports_candidates_lost_to_degeneracy(two_clusters):
    hits = nearest_structural(two_clusters, "a1", k=20, same_type=False, rank=5)
    assert hits.excluded_uncovered == 1, "iso is excluded and the caller is told so"
    assert "iso" not in [h.entity_id for h in hits]


def test_nearest_structural_honours_same_type(two_clusters):
    hits = nearest_structural(two_clusters, "a1", k=20, same_type=True, rank=5)
    assert {h.entity_type for h in hits} == {"widget"}
    assert "t1" not in [h.entity_id for h in hits]
    assert hits.excluded_other_type == 2


def test_nearest_structural_reports_hops_and_direct_edges(two_clusters):
    hits = nearest_structural(two_clusters, "a1", k=20, same_type=False, rank=5, max_hops=3)
    by_id = {h.entity_id: h for h in hits}
    assert by_id["a2"].hops == 1, "a1-a2 is a clique edge, not a hidden connection"
    assert by_id["b1"].hops == 3, "a1 -> a0 -> b0 -> b1"


def test_nearest_structural_hops_none_means_beyond_budget_not_unreachable(two_clusters):
    hits = nearest_structural(two_clusters, "a1", k=20, same_type=False, rank=5, max_hops=1)
    by_id = {h.entity_id: h for h in hits}
    assert by_id["a2"].hops == 1
    assert by_id["b1"].hops is None
    assert hits.max_hops == 1, "the budget that produced the Nones is on the result"


def test_nearest_structural_reports_degree_so_hubs_are_visible(two_clusters):
    hits = nearest_structural(two_clusters, "a1", k=20, same_type=False, rank=5)
    by_id = {h.entity_id: h for h in hits}
    assert by_id["a0"].degree == 4  # 3 clique peers + the bridge to b0
    assert by_id["t1"].degree == 3


def test_nearest_structural_rejects_bad_arguments(two_clusters):
    with pytest.raises(ValueError):
        nearest_structural(two_clusters, "a1", k=0, rank=5)
    with pytest.raises(ValueError):
        nearest_structural(two_clusters, "a1", max_hops=0, rank=5)


def test_nearest_structural_refuses_same_type_without_types():
    matrix = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    g = SpectralGraph(matrix, ["x", "y", "z"], entity_types=None)
    with pytest.raises(ValueError) as err:
        nearest_structural(g, "x", k=2, same_type=True, rank=2)
    assert "entity types" in str(err.value)


def test_nearest_structural_raises_for_an_isolated_query(two_clusters):
    with pytest.raises(DegenerateEmbeddingError):
        nearest_structural(two_clusters, "iso", k=5, same_type=False, rank=5)


# ── degree_rank_correlation ────────────────────────────────────────


def test_degree_rank_correlation_saturates_when_similarity_is_pure_degree():
    """
    A star of stars: the hub's similarity to each satellite hub is a strict
    function of that hub's own size, so the similarity ranking and the degree
    ranking agree up to direction. |rho| == 1 is the signature of a measure that
    adds nothing to `ORDER BY degree`, and the sign does not soften that — a
    perfect anti-correlation is just the same list reversed.
    """
    edges: list[tuple[str, str]] = []
    for i, size in enumerate((2, 4, 8, 16, 32)):
        edges.append(("centre", f"h{i}"))
        edges += [(f"h{i}", f"h{i}_leaf{j}") for j in range(size)]
    g = _graph_from_edges(edges, types={f"h{i}": "hub" for i in range(5)})
    rho = degree_rank_correlation(g, "h4", same_type=True, rank=5)
    assert abs(rho) == pytest.approx(1.0, abs=1e-9)


def test_degree_rank_correlation_is_not_saturated_on_a_structured_graph(two_clusters):
    rho = degree_rank_correlation(two_clusters, "a1", same_type=False, rank=5)
    assert -1.0 <= rho <= 1.0
    assert abs(rho) < 0.99


def test_degree_rank_correlation_refuses_too_few_candidates(two_clusters):
    with pytest.raises(ValueError) as err:
        degree_rank_correlation(two_clusters, "t1", same_type=True, rank=5)
    assert "candidates" in str(err.value)


# ── load_graph ─────────────────────────────────────────────────────


def test_load_graph_opens_the_database_read_only(tmp_path, monkeypatch):
    db = tmp_path / "p.db"
    _write_temp_db(
        db,
        [("e1", "country", "A"), ("e2", "instrument", "B")],
        [("e1", "e2", "produced_in", 1.0, "test", 100.0, 100.0)],
    )
    seen: list[str] = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        seen.append(str(target))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    load_graph(db_path=db)
    assert seen and all("mode=ro" in uri for uri in seen), "the live DB is the one irreplaceable asset"


def test_load_graph_reports_link_counts_and_multiplicity_collapse(tmp_path):
    db = tmp_path / "p.db"
    _write_temp_db(
        db,
        [("e1", "country", "A"), ("e2", "instrument", "B"), ("e3", "topic", "C")],
        [
            ("e1", "e2", "produced_in", 1.0, "s", 100.0, 100.0),
            ("e1", "e2", "event_involves", 1.0, "s", 100.0, 100.0),  # same pair, 2nd type
            ("e2", "e3", "topic_relates_to_instrument", 1.0, "s", 100.0, 100.0),
        ],
    )
    g = load_graph(db_path=db)
    p = g.provenance
    assert p["n_entities"] == 3 == len(g.node_ids)
    assert p["n_links_in_db"] == 3
    assert p["n_links_kept"] == 3
    assert p["n_undirected_edges"] == 2
    assert p["edge_multiplicity_collapsed"] == 1, "the collapse must be stated, not hidden"


def test_load_graph_as_of_excludes_later_links(tmp_path):
    db = tmp_path / "p.db"
    _write_temp_db(
        db,
        [("e1", "country", "A"), ("e2", "instrument", "B"), ("e3", "topic", "C")],
        [
            ("e1", "e2", "produced_in", 1.0, "s", 5000.0, 1000.0),
            ("e2", "e3", "event_involves", 1.0, "s", 5000.0, 9000.0),
        ],
    )
    early = load_graph(db_path=db, as_of=2000.0)
    assert early.provenance["n_links_kept"] == 1
    assert early.provenance["dropped_after_as_of"] == 1
    late = load_graph(db_path=db, as_of=10_000.0)
    assert late.provenance["n_links_kept"] == 2
    assert late.provenance["dropped_after_as_of"] == 0


def test_load_graph_counts_created_at_fallbacks(tmp_path):
    """A NULL effective_from is dated by an INGEST timestamp. That is the F-04
    leakage surface, so the count has to reach the caller."""
    db = tmp_path / "p.db"
    _write_temp_db(
        db,
        [("e1", "country", "A"), ("e2", "instrument", "B")],
        [("e1", "e2", "produced_in", 1.0, "s", 7000.0, None)],
    )
    g = load_graph(db_path=db, as_of=8000.0)
    assert g.provenance["links_dated_by_created_at_fallback"] == 1
    assert g.provenance["n_links_kept"] == 1


def test_load_graph_keeps_confidence_as_the_edge_weight(tmp_path):
    """The default `binary=True` throws these away; `binary=False` must have
    something real to use, or the flag is decoration."""
    db = tmp_path / "p.db"
    _write_temp_db(
        db,
        [("e1", "country", "A"), ("e2", "instrument", "B"), ("e3", "topic", "C")],
        [
            ("e1", "e2", "produced_in", 0.6, "s", 100.0, 100.0),
            ("e1", "e2", "event_involves", 0.9, "s", 100.0, 100.0),  # same pair, higher confidence
            ("e2", "e3", "topic_relates_to_instrument", 0.7, "s", 100.0, 100.0),
        ],
    )
    g = load_graph(db_path=db)
    assert g.adjacency[0, 1] == pytest.approx(0.9), "the strongest link_type wins; weights are not summed"
    assert g.adjacency[1, 2] == pytest.approx(0.7)
    assert g.provenance["edge_multiplicity_collapsed"] == 1


def test_load_graph_min_confidence_drops_weak_links(tmp_path):
    db = tmp_path / "p.db"
    _write_temp_db(
        db,
        [("e1", "country", "A"), ("e2", "instrument", "B"), ("e3", "topic", "C")],
        [
            ("e1", "e2", "produced_in", 0.6, "s", 100.0, 100.0),
            ("e2", "e3", "event_involves", 1.0, "s", 100.0, 100.0),
        ],
    )
    g = load_graph(db_path=db, min_confidence=0.8)
    assert g.provenance["n_links_kept"] == 1
    assert g.provenance["dropped_low_confidence"] == 1


def test_load_graph_filters_link_types(tmp_path):
    db = tmp_path / "p.db"
    _write_temp_db(
        db,
        [("e1", "country", "A"), ("e2", "instrument", "B"), ("e3", "topic", "C")],
        [
            ("e1", "e2", "produced_in", 1.0, "s", 100.0, 100.0),
            ("e2", "e3", "event_involves", 1.0, "s", 100.0, 100.0),
        ],
    )
    g = load_graph(db_path=db, link_types={"event_involves"})
    assert g.provenance["n_links_kept"] == 1
    assert g.provenance["dropped_wrong_link_type"] == 1


def test_load_graph_keeps_every_entity_as_a_row(tmp_path):
    """Filtered-out entities stay as isolated rows; dropping them would renumber
    the matrix and quietly change what a row index means."""
    db = tmp_path / "p.db"
    _write_temp_db(
        db,
        [("e1", "country", "A"), ("e2", "instrument", "B"), ("e3", "vessel", "C")],
        [("e1", "e2", "produced_in", 1.0, "s", 100.0, 100.0)],
    )
    g = load_graph(db_path=db)
    assert len(g.node_ids) == 3
    assert g.adjacency.shape == (3, 3)
    assert g.provenance["n_entities"] == 3, "the row count and the reported entity count must agree"
    assert spectral_model(g, rank=1).coverage_report()["n_isolated"] == 1


def test_load_graph_raises_when_no_link_survives(tmp_path):
    db = tmp_path / "p.db"
    _write_temp_db(
        db,
        [("e1", "country", "A"), ("e2", "instrument", "B")],
        [("e1", "e2", "produced_in", 1.0, "s", 100.0, 100.0)],
    )
    with pytest.raises(EmptyGraphError) as err:
        load_graph(db_path=db, as_of=1.0)
    assert "date=1" in str(err.value)


def test_load_graph_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_graph(db_path=tmp_path / "nope.db")


# ── Live graph (read-only) ─────────────────────────────────────────


@pytest.fixture(scope="module")
def live_graph() -> SpectralGraph:
    return load_graph(db_path=LIVE_DB)


@live_only
def test_live_graph_effective_rank_beats_the_gnn(live_graph):
    """
    The acceptance test. The HetTGN reached effective rank 2.6-10.6 of 64 after
    2.5 hours and 811,282 parameters. This must clear 25 of 64 with none.
    """
    started = time.perf_counter()
    model = spectral_model(live_graph, rank=64)
    total = time.perf_counter() - started
    rank = effective_rank(model.S)
    print(f"\nlive effective rank {rank:.3f} of 64 in {model.wall_clock_s:.3f}s svd / {total:.3f}s total")
    assert rank > 25.0, f"effective rank {rank:.2f} of 64 — under 40% of dimensions used"
    assert 48.0 < rank < 53.0, "the measured reference is 50.43; a big move means the graph or matrix changed"
    assert model.wall_clock_s < 60.0


@live_only
def test_live_graph_coverage_is_small_and_stated(live_graph):
    """The module's first documented limitation, pinned. Raising `rank` or
    fixing connectivity must move these numbers and this assertion with them."""
    report = spectral_model(live_graph, rank=64).coverage_report()
    print(f"\ncoverage {report['n_covered']}/{report['n_nodes']} = {report['coverage_fraction']:.1%}")
    assert report["n_nodes"] == 20026
    assert report["coverage_fraction"] < 0.25
    assert report["n_covered"] == report["largest_component_size"]
    assert report["n_components_covered"] == 1, "a rank-64 SVD reaches exactly one of this graph's components"
    covered_person, total_person = report["covered_by_entity_type"]["person"]
    assert covered_person < 0.01 * total_person, "people are effectively absent from this embedding"


@live_only
def test_live_graph_country_similarity_is_mostly_degree(live_graph):
    """
    The honest limitation, as a test. If similarity for a hub country ever
    stops tracking degree, this goes red and the claim in the docstring must be
    re-measured rather than carried forward.
    """
    russia = live_graph.node_ids[list(live_graph.canonical_names).index("Russia")]
    rho = degree_rank_correlation(live_graph, russia, same_type=True, rank=64)
    print(f"\nRussia: spearman(similarity, degree) over covered countries = {rho:+.3f}")
    assert abs(rho) > 0.8, "documented as +0.957; a real improvement must update the docstring"


@live_only
def test_live_graph_instrument_similarity_is_less_degree_driven(live_graph):
    wti = live_graph.node_ids[list(live_graph.canonical_names).index("WTI Crude Oil")]
    rho = degree_rank_correlation(live_graph, wti, same_type=True, rank=64)
    print(f"\nWTI: spearman(similarity, degree) over covered instruments = {rho:+.3f}")
    assert abs(rho) < 0.5


@live_only
def test_live_graph_instrument_neighbours_have_no_direct_edge_but_are_two_hops(live_graph):
    """Instruments never link to each other, so every hit is a 'hidden'
    connection — and every one of them is 2 hops through a hub, which is the
    honest reading of what was found."""
    wti = live_graph.node_ids[list(live_graph.canonical_names).index("WTI Crude Oil")]
    hits = nearest_structural(live_graph, wti, k=5, same_type=True, rank=64)
    print(
        "\nWTI nearest instruments: " + ", ".join(f"{h.canonical_name} ({h.similarity:.3f}, {h.hops}h)" for h in hits)
    )
    assert hits[0].canonical_name == "Brent Crude Oil"
    assert all(h.hops == 2 for h in hits), "no direct instrument-instrument edges; all reached via a shared hub"
    assert hits.excluded_uncovered > 10_000


@live_only
def test_live_graph_uncovered_entity_raises_rather_than_scoring_zero(live_graph):
    types = list(live_graph.entity_types)
    vessel = live_graph.node_ids[types.index("vessel")]
    country = live_graph.node_ids[types.index("country")]
    with pytest.raises(DegenerateEmbeddingError):
        structural_similarity(live_graph, country, vessel, rank=64)


@live_only
def test_live_graph_as_of_shrinks_the_graph_and_reports_it():
    recent = load_graph(db_path=LIVE_DB, as_of=1_546_300_800.0)  # 2019-01-01
    full = load_graph(db_path=LIVE_DB)
    print(
        f"\nas-of 2019-01-01: {recent.provenance['n_links_kept']} of "
        f"{full.provenance['n_links_kept']} links, "
        f"{recent.provenance['links_dated_by_created_at_fallback']} dated by created_at fallback"
    )
    assert recent.provenance["n_links_kept"] < full.provenance["n_links_kept"]
    assert recent.provenance["dropped_after_as_of"] > 0


@live_only
def test_live_graph_binary_and_weighted_spectra_are_both_real_and_differ(live_graph):
    """Reported both ways because the choice is not neutral and the default
    (binary) is the one the acceptance number was measured with."""
    binary = effective_rank(spectral_model(live_graph, rank=64, binary=True).S)
    weighted = effective_rank(spectral_model(live_graph, rank=64, binary=False).S)
    print(f"\neffective rank: binary {binary:.3f} vs confidence-weighted {weighted:.3f}")
    assert binary != weighted, "binary=False must actually use the confidence weights"
    assert weighted > 25.0


@live_only
def test_live_graph_event_only_spectrum_differs_from_the_full_one():
    """Scaffold edges route evidence; they are not evidence. An event-only
    spectrum is a different object and must not be confused with the full one."""
    from agent.mechanism.spectral import EVENT_LINK_TYPES

    events = load_graph(db_path=LIVE_DB, link_types=EVENT_LINK_TYPES)
    full = load_graph(db_path=LIVE_DB)
    assert events.provenance["n_links_kept"] < full.provenance["n_links_kept"]
    e_rank = effective_rank(spectral_model(events, rank=64).S)
    f_rank = effective_rank(spectral_model(full, rank=64).S)
    print(f"\neffective rank: event-edges-only {e_rank:.3f} vs all-edges {f_rank:.3f}")
    assert e_rank > 25.0


# ── Integration with the shared mechanism-layer graph loader ────────


@live_only
def test_accepts_the_shared_mechanism_graph_unchanged():
    """
    The adapter's whole job. `agent.mechanism.graph.MechanismGraph` is written by
    a different module and exposes `adjacency` / `entity_ids` / `entity_types` /
    `canonical_names`; this module must consume it without either side importing
    the other's loader. Skipped, not failed, if that module is not present yet —
    but if it IS present and its field names move, this goes red instead of
    silently falling back to a locally-loaded graph.
    """
    graph_module = pytest.importorskip("agent.mechanism.graph")
    shared = graph_module.load_graph(db_path=LIVE_DB, as_of=None)

    model = spectral_model(shared, rank=64)
    mine = spectral_model(load_graph(db_path=LIVE_DB), rank=64)
    assert effective_rank(model.S) == pytest.approx(effective_rank(mine.S), rel=1e-6)
    assert model.n_covered == mine.n_covered

    wti = shared.entity_ids[list(shared.canonical_names).index("WTI Crude Oil")]
    hits = nearest_structural(shared, wti, k=3, same_type=True, rank=64)
    assert hits[0].canonical_name == "Brent Crude Oil"
    assert hits[0].entity_type == "instrument", "entity types must survive the adapter"
