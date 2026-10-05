"""
Tests for agent/mechanism/connectivity.py.

Two halves:

  1. SYNTHETIC graphs with hand-derivable answers. These are the tests that
     catch a wrong implementation: closed-form PageRank on a star, an
     independent power-iteration cross-check, textbook resistances (single
     edge = 1, triangle = 2/3, parallel conductance = 1/2), and a hub-splitting
     case where naive path counting gets the ordering backwards.

  2. The LIVE-GRAPH acceptance test: Russia -> WTI Crude Oil at alpha=0.15 over
     all 20,026 entities must reproduce the measured reference (mass 2.648e-03,
     global rank 127, instrument #1 of 93, ~28x the instrument median).

The graph loader here is deliberately LOCAL to this test file: agent/mechanism/
graph.py is owned by another module and may not exist yet, and this file must
not pre-empt its shape. It opens the live DB read-only (``mode=ro``) and never
writes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from agent.mechanism.connectivity import (
    DisconnectedEntityError,
    EmptyGraphError,
    IsolatedEntityError,
    MechanismError,
    UnknownEntityError,
    connectivity,
    effective_resistance,
    personalised_pagerank,
)

DB_PATH = Path(__file__).resolve().parents[1] / ".tirra_pipeline" / "pipeline.db"

# Measured reference (2026-09-26) on the full simple graph, alpha=0.15.
RUSSIA = "c9db6e2a9784c856"
WTI = "da678733bb9c745f"
REF_MASS = 2.6476e-03
REF_GLOBAL_RANK = 127
REF_N_ENTITIES = 20026
REF_PERCENTILE = 99.37
REF_INSTRUMENT_RANK = 1
REF_INSTRUMENT_COUNT = 93
REF_INSTRUMENT_MEDIAN = 9.428e-05


# ── Minimal stub graph (synthetic tests) ────────────────────────────────


@dataclass
class StubGraph:
    """Smallest object satisfying the connectivity module's graph contract."""

    A: sp.csr_matrix
    ents: list[str]
    etype: list[str]
    name: list[str] = field(default_factory=list)
    idx: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.idx:
            self.idx = {e: i for i, e in enumerate(self.ents)}
        if not self.name:
            self.name = list(self.ents)


def undirected(n: int, edges: list[tuple[int, int, float]]) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for i, j, w in edges:
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def stub(n: int, edges: list[tuple[int, int, float]], types: list[str] | None = None) -> StubGraph:
    ents = [f"e{i}" for i in range(n)]
    return StubGraph(
        A=undirected(n, edges),
        ents=ents,
        etype=types if types is not None else ["thing"] * n,
    )


def power_iteration_ppr(A: sp.csr_matrix, seed: int, alpha: float, iters: int = 20000) -> np.ndarray:
    """Independent reference implementation — deliberately a different algorithm."""
    n = A.shape[0]
    degree = np.asarray(A.sum(axis=1)).ravel()
    x = np.zeros(n)
    x[seed] = 1.0
    for _ in range(iters):
        contrib = np.where(degree > 0, x / np.maximum(degree, 1e-300), 0.0)
        nxt = (1.0 - alpha) * (A.T @ contrib)
        nxt[seed] += alpha
        if np.abs(nxt - x).max() < 1e-15:
            return nxt
        x = nxt
    return x


# ── Live graph loader (local to this file, read-only) ───────────────────


def load_live_graph(as_of: float | None = None) -> StubGraph:
    """Simple (deduplicated, unweighted, undirected) live graph, read-only.

    A link counts if ``coalesce(effective_from, created_at) <= as_of``; with
    ``as_of=None`` every link counts. Parallel links between the same pair under
    different ``link_type`` collapse to ONE edge — the reference numbers were
    measured that way, and summing them into weights changes every rank.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT entity_id, entity_type, canonical_name FROM entities ORDER BY entity_id").fetchall()
        if as_of is None:
            links = conn.execute("SELECT entity_id_a, entity_id_b FROM entity_links").fetchall()
        else:
            links = conn.execute(
                "SELECT entity_id_a, entity_id_b FROM entity_links WHERE COALESCE(effective_from, created_at) <= ?",
                (float(as_of),),
            ).fetchall()
    finally:
        conn.close()

    ents = [r[0] for r in rows]
    idx = {e: i for i, e in enumerate(ents)}
    pairs = {(min(idx[a], idx[b]), max(idx[a], idx[b])) for a, b in links}
    pairs.discard(None)  # defensive; no self-loops exist in this table
    edges = [(i, j, 1.0) for i, j in pairs if i != j]
    return StubGraph(
        A=undirected(len(ents), edges),
        ents=ents,
        etype=[r[1] for r in rows],
        name=[r[2] for r in rows],
        idx=idx,
    )


live_only = pytest.mark.skipif(not DB_PATH.exists(), reason=f"live pipeline DB not present at {DB_PATH}")


@pytest.fixture(scope="module")
def live_graph() -> StubGraph:
    return load_live_graph()


# ── PageRank: closed form and cross-check ───────────────────────────────


def test_star_pagerank_matches_closed_form() -> None:
    """Star, seed at centre: centre mass is alpha / (1 - (1-alpha)^2), exactly."""
    k = 7
    g = stub(k + 1, [(0, i, 1.0) for i in range(1, k + 1)])
    alpha = 0.15
    x = personalised_pagerank(g, "e0", alpha=alpha)

    centre = alpha / (1.0 - (1.0 - alpha) ** 2)
    leaf = (1.0 - alpha) * centre / k
    assert x[0] == pytest.approx(centre, rel=1e-12)
    for i in range(1, k + 1):
        assert x[i] == pytest.approx(leaf, rel=1e-12)
    assert x.sum() == pytest.approx(1.0, abs=1e-12)


def test_pagerank_matches_independent_power_iteration() -> None:
    rng = np.random.default_rng(11)
    n = 60
    edges = [(int(i), int(j), 1.0) for i, j in rng.integers(0, n, size=(200, 2)) if i != j]
    edges.append((0, 1, 1.0))  # guarantee the seed has a neighbour
    g = stub(n, edges)
    for alpha in (0.05, 0.15, 0.30):
        mine = personalised_pagerank(g, "e0", alpha=alpha)
        theirs = power_iteration_ppr(g.A, 0, alpha)
        assert np.abs(mine - theirs).max() < 1e-10, alpha


def test_pagerank_mass_sums_to_one_and_is_nonnegative() -> None:
    g = stub(6, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (4, 5, 1.0)])
    x = personalised_pagerank(g, "e0", alpha=0.15)
    assert x.min() >= 0.0
    assert x.sum() == pytest.approx(1.0, abs=1e-12)
    # A separate component gets exactly zero — a computed "not connected".
    assert x[4] == 0.0 and x[5] == 0.0


def test_zero_degree_nodes_do_not_divide_by_zero() -> None:
    """Node 3 is isolated; the solve must still be exact and leak nothing."""
    g = stub(4, [(0, 1, 1.0), (1, 2, 1.0)])
    x = personalised_pagerank(g, "e0", alpha=0.15)
    assert np.isfinite(x).all()
    assert x[3] == 0.0
    assert x.sum() == pytest.approx(1.0, abs=1e-12)


def test_alpha_controls_reach() -> None:
    """Smaller alpha = longer walks = more mass out at distance."""
    g = stub(6, [(i, i + 1, 1.0) for i in range(5)])
    far_small = personalised_pagerank(g, "e0", alpha=0.05)[5]
    far_large = personalised_pagerank(g, "e0", alpha=0.60)[5]
    assert far_small > far_large * 5


# ── Hub contamination: the failure this measure exists to avoid ──────────


def test_hub_route_is_discounted_relative_to_a_narrow_route() -> None:
    """Both targets are 2 hops away via exactly one path, so naive path counting
    calls them equal. Degree splitting must not: the hub target's route passes a
    102-degree node, the other a 2-degree node."""
    n_filler = 100
    seed, hub, mid, t_hub, t_mid = 0, 1, 2, 3, 4
    base = 5
    edges = [
        (seed, hub, 1.0),
        (seed, mid, 1.0),
        (hub, t_hub, 1.0),
        (mid, t_mid, 1.0),
    ]
    edges += [(hub, base + i, 1.0) for i in range(n_filler)]
    g = stub(base + n_filler, edges)

    x = personalised_pagerank(g, "e0", alpha=0.15)
    assert x[t_mid] > 20 * x[t_hub], (x[t_mid], x[t_hub])


@pytest.mark.parametrize("filler", [0, 10, 100, 190])
def test_hub_discount_is_exactly_one_over_hub_degree(filler: int) -> None:
    """seed -> hub -> target is ONE path however fat the hub is, so naive path
    counting scores it identically at every hub size. The mass must instead fall
    exactly as 1/degree(hub) — that is what makes "United States" (193 links)
    stop manufacturing confirmations."""
    edges = [(0, 1, 1.0), (1, 2, 1.0)] + [(1, 3 + i, 1.0) for i in range(filler)]
    g = stub(3 + filler, edges)
    alpha = 0.15
    x = personalised_pagerank(g, "e0", alpha=alpha)
    hub_degree = 2 + filler
    # Closed form for this shape: (1-a)^2 * a/(1-(1-a)^2) / degree(hub).
    expected = (1.0 - alpha) ** 2 * alpha / (1.0 - (1.0 - alpha) ** 2)
    assert x[2] * hub_degree == pytest.approx(expected, rel=1e-9), (filler, x[2])


# ── connectivity(): interpretability contract ───────────────────────────


def test_connectivity_reports_same_type_rank_and_ratio() -> None:
    types = ["country", "hub", "instrument", "instrument", "instrument"]
    # seed -> hub -> each instrument, with instrument 2 also directly linked.
    g = StubGraph(
        A=undirected(5, [(0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (0, 2, 1.0)]),
        ents=["ru", "hub", "wti", "brent", "gold"],
        etype=types,
        name=["Russia", "Hub", "WTI", "Brent", "Gold"],
    )
    out = connectivity(g, "ru", "wti", alpha=0.15)
    assert out["dst_type"] == "instrument"
    assert out["type_count"] == 3
    assert out["type_rank"] == 1
    assert out["ratio_to_type_median"] > 1.0
    assert out["connected"] is True
    assert out["global_rank"] == 1 + int((personalised_pagerank(g, "ru") > out["mass"]).sum())
    assert out["percentile"] == pytest.approx(100.0 * (1 - out["global_rank"] / 5))
    assert out["alpha"] == 0.15
    assert out["notes"], "notes must always carry the caveats that apply"


def test_connectivity_zero_mass_is_a_computed_no_not_an_error() -> None:
    g = StubGraph(
        A=undirected(4, [(0, 1, 1.0), (2, 3, 1.0)]),
        ents=["a", "b", "c", "d"],
        etype=["country", "country", "instrument", "instrument"],
    )
    out = connectivity(g, "a", "c")
    assert out["mass"] == 0.0
    assert out["connected"] is False
    assert out["ratio_to_type_median"] is None  # median is 0 -> undefined, not inf
    assert any("NOT reachable" in n for n in out["notes"])
    assert any("undefined rather than infinite" in n for n in out["notes"])


def test_connectivity_surfaces_isolated_and_tie_counts() -> None:
    g = StubGraph(
        A=undirected(5, [(0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)]),
        ents=["a", "b", "c", "d", "lonely"],
        etype=["country", "hub", "instrument", "instrument", "instrument"],
    )
    out = connectivity(g, "a", "c")
    assert out["isolated_count"] == 1
    assert out["reachable_count"] == 4
    assert out["global_ties"] == 1  # c and d are symmetric
    assert any("degree 0" in n for n in out["notes"])


def test_connectivity_flags_multi_edge_weights() -> None:
    weighted = StubGraph(
        A=undirected(3, [(0, 1, 2.0), (1, 2, 1.0)]),
        ents=["a", "b", "c"],
        etype=["country", "hub", "instrument"],
    )
    out = connectivity(weighted, "a", "c")
    assert out["multi_edge_weights_present"] is True
    assert any("simple unweighted graph" in n for n in out["notes"])

    simple = StubGraph(
        A=undirected(3, [(0, 1, 1.0), (1, 2, 1.0)]),
        ents=["a", "b", "c"],
        etype=["country", "hub", "instrument"],
    )
    assert connectivity(simple, "a", "c")["multi_edge_weights_present"] is False


def test_connectivity_reports_directed_mass_leak() -> None:
    """A sink row leaks mass; sub-stochastic results must say so, not hide it."""
    A = sp.csr_matrix(([1.0, 1.0], ([0, 1], [1, 2])), shape=(3, 3))  # 0->1->2, 2 is a sink
    g = StubGraph(A=A, ents=["a", "b", "c"], etype=["country", "hub", "instrument"])
    out = connectivity(g, "a", "c")
    assert out["teleport_mass_lost"] > 1e-9
    assert any("dangling rows" in n for n in out["notes"])


# ── Graph-contract adapters (the shipped MechanismGraph spells these differently) ──


@dataclass
class ShippedStyleGraph:
    """Attribute names as agent/mechanism/graph.py's MechanismGraph exposes them."""

    adjacency: sp.csr_matrix
    evidence_adjacency: sp.csr_matrix
    entity_ids: tuple[str, ...]
    entity_types: tuple[str, ...]
    canonical_names: tuple[str, ...]


def _shipped_pair() -> ShippedStyleGraph:
    # a -- b -- c, but only the a--b hop is time-varying (evidence).
    return ShippedStyleGraph(
        adjacency=undirected(3, [(0, 1, 1.0), (1, 2, 1.0)]),
        evidence_adjacency=undirected(3, [(0, 1, 1.0)]),
        entity_ids=("a", "b", "c"),
        entity_types=("country", "country", "instrument"),
        canonical_names=("Russia", "Kazakhstan", "WTI Crude Oil"),
    )


def test_accepts_the_shipped_attribute_spelling() -> None:
    g = _shipped_pair()
    out = connectivity(g, "a", "c", alpha=0.15)
    assert out["src_name"] == "Russia"
    assert out["dst_name"] == "WTI Crude Oil"
    assert out["dst_type"] == "instrument"
    assert out["n_entities"] == 3
    assert out["connected"] is True
    assert effective_resistance(g, "a", "c") == pytest.approx(2.0, rel=1e-12)


def test_evidence_only_walk_refuses_a_scaffold_route() -> None:
    """The a--c route exists only through a non-evidence edge. On the evidence
    graph the answer must be a computed zero with its rank marked degenerate —
    never "instrument #1 of 1"."""
    g = _shipped_pair()
    assert connectivity(g, "a", "c", edges="all")["mass"] > 0.0

    out = connectivity(g, "a", "c", edges="evidence")
    assert out["mass"] == 0.0
    assert out["connected"] is False
    assert out["type_ties"] == 0  # only one instrument, and it is unreachable
    assert any("must NOT be quoted" in n for n in out["notes"])
    # "c" has no evidence edge at all, which is a different failure from
    # "both have edges but sit in different components" — and must not be
    # reported as the same thing.
    with pytest.raises(IsolatedEntityError):
        effective_resistance(g, "a", "c", edges="evidence")

    two_islands = ShippedStyleGraph(
        adjacency=undirected(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]),
        evidence_adjacency=undirected(4, [(0, 1, 1.0), (2, 3, 1.0)]),
        entity_ids=("a", "b", "c", "d"),
        entity_types=("country", "country", "instrument", "instrument"),
        canonical_names=("Russia", "Kazakhstan", "WTI Crude Oil", "Brent"),
    )
    with pytest.raises(DisconnectedEntityError):
        effective_resistance(two_islands, "a", "c", edges="evidence")
    assert effective_resistance(two_islands, "a", "c", edges="all") == pytest.approx(2.0)


def test_evidence_selector_requires_the_graph_to_have_one() -> None:
    g = stub(3, [(0, 1, 1.0), (1, 2, 1.0)])  # no .evidence_adjacency
    with pytest.raises(MechanismError, match="evidence"):
        connectivity(g, "e0", "e2", edges="evidence")
    with pytest.raises(MechanismError, match="edges must be one of"):
        personalised_pagerank(g, "e0", edges="nonsense")


def test_zero_mass_type_rank_is_marked_degenerate_with_its_tie_count() -> None:
    g = StubGraph(
        A=undirected(4, [(0, 1, 1.0), (2, 3, 1.0)]),
        ents=["ru", "kz", "wti", "brent"],
        etype=["country", "country", "instrument", "instrument"],
    )
    out = connectivity(g, "ru", "wti")
    assert out["mass"] == 0.0
    assert out["type_rank"] == 1  # tied at zero, hence meaningless on its own
    assert out["type_ties"] == 1
    assert any("must NOT be quoted" in n for n in out["notes"])


# ── Degenerate inputs fail loudly ───────────────────────────────────────


def test_unknown_entity_raises() -> None:
    g = stub(3, [(0, 1, 1.0), (1, 2, 1.0)])
    with pytest.raises(UnknownEntityError):
        personalised_pagerank(g, "nope")
    with pytest.raises(UnknownEntityError):
        connectivity(g, "e0", "nope")
    with pytest.raises(UnknownEntityError):
        effective_resistance(g, "nope", "e0")


def test_isolated_seed_raises_instead_of_returning_e_seed() -> None:
    g = stub(3, [(0, 1, 1.0)])
    with pytest.raises(IsolatedEntityError):
        personalised_pagerank(g, "e2")
    with pytest.raises(IsolatedEntityError):
        connectivity(g, "e2", "e0")
    with pytest.raises(IsolatedEntityError):
        effective_resistance(g, "e0", "e2")


def test_empty_and_edgeless_graphs_raise() -> None:
    empty = StubGraph(A=sp.csr_matrix((0, 0)), ents=[], etype=[])
    with pytest.raises(EmptyGraphError):
        personalised_pagerank(empty, "e0")
    edgeless = StubGraph(A=sp.csr_matrix((3, 3)), ents=["a", "b", "c"], etype=["x", "y", "z"])
    with pytest.raises(EmptyGraphError):
        connectivity(edgeless, "a", "b")


def test_bad_alpha_and_self_pair_and_contract_violations_raise() -> None:
    g = stub(3, [(0, 1, 1.0), (1, 2, 1.0)])
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(MechanismError):
            personalised_pagerank(g, "e0", alpha=bad)
    with pytest.raises(MechanismError):
        connectivity(g, "e0", "e0")
    with pytest.raises(MechanismError):
        effective_resistance(g, "e0", "e0")

    no_types = StubGraph(A=g.A, ents=g.ents, etype=[])
    no_types.etype = None  # type: ignore[assignment]
    with pytest.raises(MechanismError, match="etype"):
        connectivity(no_types, "e0", "e2")

    negative = StubGraph(A=undirected(3, [(0, 1, -1.0), (1, 2, 1.0)]), ents=g.ents, etype=["a", "b", "c"])
    with pytest.raises(MechanismError, match="negative"):
        personalised_pagerank(negative, "e0")

    not_sparse = StubGraph(A=np.eye(3), ents=g.ents, etype=["a", "b", "c"])  # type: ignore[arg-type]
    with pytest.raises(MechanismError, match="scipy.sparse"):
        personalised_pagerank(not_sparse, "e0")


def test_an_unconverged_solve_raises_instead_of_returning_a_fake_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The residual guard is not decorative: a wrong solve must not be returned."""
    import agent.mechanism.connectivity as mod

    g = stub(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
    monkeypatch.setattr(mod.spla, "spsolve", lambda *a, **k: np.full(4, 0.25))
    with pytest.raises(MechanismError, match="residual"):
        personalised_pagerank(g, "e0")

    monkeypatch.setattr(mod.spla, "spsolve", lambda *a, **k: np.full(3, np.nan))
    with pytest.raises(MechanismError, match="residual|tolerance"):
        effective_resistance(g, "e0", "e3")


def test_a_corrupted_solve_is_rejected_not_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shifted (hence partly negative, non-solution) vector must not come back
    clipped into something that looks like a probability distribution. Whichever
    guard fires first is fine; returning it is not."""
    import agent.mechanism.connectivity as mod

    g = stub(3, [(0, 1, 1.0), (1, 2, 1.0)])
    real = mod.spla.spsolve
    monkeypatch.setattr(mod.spla, "spsolve", lambda M, b, **k: real(M, b) - 0.5)
    with pytest.raises(MechanismError):
        personalised_pagerank(g, "e0")


# ── Effective resistance: textbook values ───────────────────────────────


def test_resistance_single_edge_is_one_over_conductance() -> None:
    assert effective_resistance(stub(2, [(0, 1, 1.0)]), "e0", "e1") == pytest.approx(1.0, rel=1e-12)
    assert effective_resistance(stub(2, [(0, 1, 2.0)]), "e0", "e1") == pytest.approx(0.5, rel=1e-12)


def test_resistance_triangle_is_two_thirds() -> None:
    g = stub(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
    assert effective_resistance(g, "e0", "e2") == pytest.approx(2.0 / 3.0, rel=1e-12)


def test_resistance_is_symmetric_and_falls_with_redundancy() -> None:
    path = stub(3, [(0, 1, 1.0), (1, 2, 1.0)])
    assert effective_resistance(path, "e0", "e2") == pytest.approx(2.0, rel=1e-12)
    assert effective_resistance(path, "e2", "e0") == pytest.approx(2.0, rel=1e-12)
    # Same shortest-path length, one extra parallel chain -> lower resistance.
    two_chains = stub(4, [(0, 1, 1.0), (1, 3, 1.0), (0, 2, 1.0), (2, 3, 1.0)])
    assert effective_resistance(two_chains, "e0", "e3") == pytest.approx(1.0, rel=1e-12)


def test_resistance_ignores_a_far_away_component() -> None:
    """Grounding must happen inside the component, not on an unrelated node."""
    g = stub(5, [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)])
    assert effective_resistance(g, "e0", "e2") == pytest.approx(2.0, rel=1e-12)


def test_resistance_across_components_raises_not_returns_a_number() -> None:
    g = stub(4, [(0, 1, 1.0), (2, 3, 1.0)])
    with pytest.raises(DisconnectedEntityError):
        effective_resistance(g, "e0", "e3")


# ── LIVE ACCEPTANCE TEST ────────────────────────────────────────────────


@live_only
def test_live_graph_shape_is_the_one_the_reference_was_measured_on(
    live_graph: StubGraph,
) -> None:
    assert len(live_graph.ents) == REF_N_ENTITIES
    assert live_graph.A.nnz == 2 * 26056  # 30,131 links -> 26,056 unique pairs
    assert live_graph.A.max() == 1.0  # simple graph, no summed multi-edges
    assert sum(1 for t in live_graph.etype if t == "instrument") == REF_INSTRUMENT_COUNT


@live_only
def test_acceptance_russia_to_wti_reproduces_the_measured_reference(
    live_graph: StubGraph,
) -> None:
    out = connectivity(live_graph, RUSSIA, WTI, alpha=0.15)

    assert out["src_name"] == "Russia"
    assert out["dst_name"] == "WTI Crude Oil"
    assert out["dst_type"] == "instrument"
    assert out["n_entities"] == REF_N_ENTITIES

    assert out["mass"] == pytest.approx(REF_MASS, rel=1e-3), out["mass"]
    assert out["global_rank"] == REF_GLOBAL_RANK, out["global_rank"]
    assert out["percentile"] == pytest.approx(REF_PERCENTILE, abs=0.01), out["percentile"]
    assert out["type_rank"] == REF_INSTRUMENT_RANK
    assert out["type_count"] == REF_INSTRUMENT_COUNT
    assert out["type_median"] == pytest.approx(REF_INSTRUMENT_MEDIAN, rel=1e-3), out["type_median"]
    assert out["ratio_to_type_median"] == pytest.approx(28.1, abs=0.3)
    assert out["teleport_mass_lost"] == pytest.approx(0.0, abs=1e-12)
    assert out["multi_edge_weights_present"] is False


@live_only
@pytest.mark.parametrize(
    ("alpha", "mass", "global_rank", "type_rank", "type_median"),
    [
        (0.05, 2.9782e-03, 117, 6, 1.508e-04),
        (0.15, 2.6476e-03, 127, 1, 9.428e-05),
        (0.30, 2.5189e-03, 130, 1, 5.536e-05),
    ],
)
def test_alpha_sensitivity_is_stable_and_reported(
    live_graph: StubGraph,
    alpha: float,
    mass: float,
    global_rank: int,
    type_rank: int,
    type_median: float,
) -> None:
    """alpha is the one tunable in this layer, so its effect is pinned, not assumed."""
    out = connectivity(live_graph, RUSSIA, WTI, alpha=alpha)
    assert out["alpha"] == alpha
    assert out["mass"] == pytest.approx(mass, rel=1e-3), out["mass"]
    assert out["global_rank"] == global_rank, out["global_rank"]
    assert out["type_rank"] == type_rank, out["type_rank"]
    assert out["type_median"] == pytest.approx(type_median, rel=1e-3), out["type_median"]
    assert any(f"alpha={alpha:g}" in n for n in out["notes"])


@live_only
def test_live_percentile_is_not_quoted_without_its_denominators(
    live_graph: StubGraph,
) -> None:
    """1,724 live entities have degree 0; the result must say so."""
    out = connectivity(live_graph, RUSSIA, WTI)
    assert out["isolated_count"] == 1724
    assert out["reachable_count"] == 3203  # Russia's component
    assert out["reachable_count"] < out["n_entities"] / 4
    assert any("can never rank" in n for n in out["notes"])


@live_only
def test_live_effective_resistance_russia_wti(live_graph: StubGraph) -> None:
    r = effective_resistance(live_graph, RUSSIA, WTI)
    assert r == pytest.approx(0.029928, rel=1e-3), r
    assert effective_resistance(live_graph, WTI, RUSSIA) == pytest.approx(r, rel=1e-9)


@live_only
def test_as_of_2019_graph_is_smaller_and_still_answerable() -> None:
    """Time-honesty: asking the 2019 graph must use fewer links than the 2026 one."""
    as_of_2019 = 1546300800.0  # 2019-01-01T00:00:00Z
    past = load_live_graph(as_of=as_of_2019)
    full = load_live_graph()
    assert past.A.nnz < full.A.nnz
    out_full = connectivity(full, RUSSIA, WTI, alpha=0.15)
    try:
        out_past = connectivity(past, RUSSIA, WTI, alpha=0.15)
    except (IsolatedEntityError, UnknownEntityError):
        pytest.skip("Russia has no 2019-dated links in the current backfill")
    # Different graph must give a different verdict; identical numbers would mean
    # the as_of filter silently did nothing.
    assert out_past["mass"] != out_full["mass"]
