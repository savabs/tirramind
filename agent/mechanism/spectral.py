"""
TirraMind — Mechanism Layer: spectral structure (Layer 3, world model).

WHAT THIS COMPUTES
------------------
A rank-`r` truncated SVD of the *symmetrised entity-link adjacency matrix* of
the knowledge graph, and three measures derived from it:

    spectral_embedding(g, rank=64) -> (U, S)   the embedding itself
    effective_rank(S)                          how many dimensions are actually used
    structural_similarity(g, a, b)             cosine between two entities' positions
    nearest_structural(g, e, k=10)             "what sits where this thing sits"

`nearest_structural` is the interesting one: two entities can score high without
any edge between them, because they are reached by the same *kinds* of things.
That is the "connection with no direct path" the HetTGN was supposed to find.

WHY LINEAR ALGEBRA AND NOT THE GNN
----------------------------------
Measured on this graph (see the report in the task, and the acceptance test in
tests/test_mechanism_spectral.py::test_live_graph_effective_rank_beats_gnn):
a rank-64 SVD reaches effective rank ~50 of 64 in ~1.5 s with zero fitted
parameters, and cannot collapse, because singular vectors are orthogonal by
construction. The HetTGN reached effective rank 2.6-10.6 of 64 after 2.5 h and
811,282 parameters. There is nothing to train here, so there is nothing to
collapse.

WHERE THIS MISLEADS — read before quoting a similarity to anyone
----------------------------------------------------------------
1. **COVERAGE. A rank-64 SVD of a graph with 4,447 connected components
   describes almost none of it.** Measured on the live graph at rank 64:
   3,203 of 20,026 entities (16.0%) get a non-degenerate embedding, and they
   all sit in ONE component (the largest, 3,203 nodes). Every other component
   is numerically invisible at this rank: person 6/12,075 covered,
   vessel 0/502, protocol 2/54, domain 2/20. This is not a bug, it is what
   truncation means; raising `rank` adds components one at a time, slowly.
   So an absent or degenerate answer here is usually "outside the top-64
   subspace", NOT "structurally unrelated". Functions in this module therefore
   RAISE `DegenerateEmbeddingError` for such an entity rather than return 0.0.
   `SpectralModel.coverage_report()` states the numbers for the graph you gave.

2. **SIMILARITY IS PARTLY JUST DEGREE, and how much depends on entity type.**
   Measured median Spearman rho between similarity rank and plain degree rank,
   over same-type candidates (100 sampled queries per type). What matters is
   |rho|, not its sign: a perfect anti-correlation is the same degree ranking
   read backwards, so -0.913 is as damning as +0.913.
       company     -0.913   |rho| 0.91  -> effectively a degree ranking
       topic       +0.577   |rho| 0.58
       country     +0.540   |rho| 0.54  -> and for "Russia" specifically
                                           +0.957, i.e. `ORDER BY degree`
                                           wearing a costume
       instrument  -0.402   |rho| 0.40  -> carries real extra information
       wallet      -0.300   |rho| 0.30  -> carries real extra information
   So it is NOT uniformly a degree ranking, but for companies and for the big
   hub countries — the type a commodity customer will actually ask about — it
   adds little over `ORDER BY degree`. `degree_rank_correlation` computes this
   for any entity, so a caller can check before trusting a list.

3. **"NO DIRECT PATH" IS NOT "FAR AWAY".** Measured: no two of the 93
   instruments share a direct edge, yet every covered instrument pair is
   2 hops apart via a shared country or topic hub. A high instrument-to-
   instrument similarity is a statement about shared hub neighbourhoods, not
   about a long hidden chain. `nearest_structural` reports `hops` so the
   caller can see this instead of inferring depth that is not there.

4. **COMPANY EMBEDDINGS ARE NEARLY DEGENERATE.** 67% of covered
   company-company pairs score cosine > 0.9 (mean 0.728). Ranking companies
   by this is close to meaningless.

5. **NO TIME INSIDE.** The adjacency matrix is whatever `g` contains. If `g`
   was built from today's links, every answer is a 2026 answer, and using it
   to explain a 2019 pattern is F-04 leakage. `load_graph(as_of=...)` exists
   for this; it reports how many links it dropped and how many had to fall
   back from a NULL `effective_from` to `created_at`.

6. **SCAFFOLD AND EVIDENCE ARE NOT SEPARATED HERE.** A `produced_in` edge
   (static geography) and an `event_involves` edge (a dated event) count the
   same in this matrix. Independence must be counted over event edges only —
   that is the evidence module's job, not this one. Use `link_types=` on
   `load_graph` if you want an event-only spectrum.

CONTRACT
--------
Every function in this module fails loudly on a degenerate input. A returned
0.0 always means "computed, and orthogonal"; "unknown", "isolated" and
"outside the truncated subspace" are three distinct exceptions. Nothing is
capped, sampled or pruned without that fact appearing in the return value.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import weakref
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DB_PATH",
    "DegenerateEmbeddingError",
    "EmptyGraphError",
    "MechanismSpectralError",
    "NeighbourList",
    "RankTooLargeError",
    "SpectralGraph",
    "SpectralModel",
    "StructuralNeighbour",
    "UnknownEntityError",
    "degree_rank_correlation",
    "effective_rank",
    "load_graph",
    "nearest_structural",
    "spectral_embedding",
    "spectral_model",
    "structural_similarity",
]

DEFAULT_DB_PATH = Path(".tirra_pipeline/pipeline.db")

# A row of U*S whose norm is below this fraction of the largest row norm is
# treated as "not described by the truncated subspace" rather than as a
# position at the origin. The live graph's gap is ~1e-16 vs ~1.4, i.e. eight
# orders of margin either side of this threshold.
_COVERAGE_REL_TOL = 1e-9

# Link types that are static geography. Kept in sync by hand with
# agent/models/gnn/graph_builder.py::STRUCTURAL_RELATIONS — this module does not
# import that file, because importing it drags in torch and torch_geometric.
SCAFFOLD_LINK_TYPES: frozenset[str] = frozenset(
    {
        "cftc_tracks",
        "domain_owned_by",
        "exchange_country",
        "fx_base_country",
        "fx_quote_country",
        "located_in",
        "market_authorized_in",
        "operates_in",
        "produced_in",
        "tracks_issuer",
        "tracks_protocol",
    }
)

EVENT_LINK_TYPES: frozenset[str] = frozenset(
    {
        "awarded_by",
        "event_involves",
        "sanctioned_under",
        "topic_relates_to_instrument",
        "trades_instrument",
        "transacts_with",
        "works_for",
    }
)


# ── Errors ─────────────────────────────────────────────────────────
# Four distinct failures that a float return value would flatten into 0.0.


class MechanismSpectralError(Exception):
    """Base class for every failure in this module."""


class UnknownEntityError(MechanismSpectralError, KeyError):
    """The entity_id is not a node of this graph. Not the same as 'unconnected'."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return " ".join(str(a) for a in self.args)


class DegenerateEmbeddingError(MechanismSpectralError):
    """
    The entity exists but has no usable position in the truncated subspace —
    either it is isolated (degree 0) or its component is outside the top-`rank`
    singular directions. Explicitly NOT similarity 0.0.
    """


class EmptyGraphError(MechanismSpectralError):
    """The graph has no nodes, or no edges at all, so no spectrum exists."""


class RankTooLargeError(MechanismSpectralError):
    """
    The requested rank cannot be computed for this matrix. Raised instead of
    silently computing a smaller one, because a silently-shortened spectrum
    makes `effective_rank` unfalsifiable (F-16/F-17 pattern).
    """


# ── The graph container ────────────────────────────────────────────


@dataclass(frozen=True)
class SpectralGraph:
    """
    The minimum a graph must supply for this module: an adjacency matrix and
    the entity_id behind each row.

    This module does not require *this* class. `_adapt` accepts any object that
    exposes an equivalent adjacency matrix and node-id list (see `_adapt` for
    the attribute names it probes), so the shared mechanism-layer graph loader
    can be passed straight in. `SpectralGraph` + `load_graph` exist so this
    module is runnable and testable on its own.

    Attributes:
        adjacency: symmetric scipy sparse matrix, shape (n, n), zero diagonal.
        node_ids: entity_id for each row, in row order.
        entity_types: entity_type per row, or None if unknown.
        canonical_names: display name per row, or None if unknown.
        provenance: facts about how this matrix was built — which links were
            dropped, and why. Never empty for `load_graph` output.
    """

    adjacency: Any
    node_ids: Sequence[str]
    entity_types: Sequence[str] | None = None
    canonical_names: Sequence[str] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.adjacency.shape[0]
        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError(f"adjacency must be square, got {self.adjacency.shape}")
        if len(self.node_ids) != n:
            raise ValueError(f"node_ids has {len(self.node_ids)} entries for a {n}x{n} adjacency")
        for name, seq in (("entity_types", self.entity_types), ("canonical_names", self.canonical_names)):
            if seq is not None and len(seq) != n:
                raise ValueError(f"{name} has {len(seq)} entries for a {n}x{n} adjacency")


class _AdaptedGraph(NamedTuple):
    matrix: Any
    node_ids: Sequence[str] | None
    entity_types: Sequence[str] | None
    canonical_names: Sequence[str] | None
    index: Mapping[str, int] | None


_MATRIX_ATTRS = ("adjacency", "adjacency_matrix", "matrix", "A", "csr", "to_scipy", "to_csr")
_NODE_ID_ATTRS = ("node_ids", "node_id", "entity_ids", "nodes", "index_to_id")
_TYPE_ATTRS = ("entity_types", "node_types", "types")
_NAME_ATTRS = ("canonical_names", "names", "labels")
_INDEX_ATTRS = ("index", "node_index", "id_to_index", "index_of")


def _first_attr(g: Any, names: Iterable[str]) -> Any:
    for name in names:
        if not hasattr(g, name):
            continue
        value = getattr(g, name)
        if callable(value) and not isinstance(value, (list, tuple, dict)):
            try:
                value = value()
            except TypeError:
                continue
        if value is not None:
            return value
    return None


def _as_sequence(value: Any, n: int, node_ids: Sequence[str] | None) -> Sequence[str] | None:
    """
    Accept a per-row sequence, or a {entity_id: value} mapping keyed by id.

    Returns None — meaning "this graph does not carry it" — rather than a
    part-filled list, so a caller relying on entity types gets a loud refusal
    instead of a silently mistyped comparison.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        if node_ids is None or any(eid not in value for eid in node_ids):
            return None
        return [value[eid] for eid in node_ids]
    seq = list(value)
    return seq if len(seq) == n else None


def _adapt(g: Any) -> _AdaptedGraph:
    """
    Coerce a caller's graph object into (matrix, node_ids, types, names, index).

    Raises TypeError naming what it looked for, rather than guessing — a wrong
    guess here would produce a plausible embedding of the wrong matrix, which is
    the worst failure this module could have.
    """
    if sp.issparse(g):
        matrix = g
        node_ids = entity_types = names = None
    elif isinstance(g, SpectralGraph):
        return _AdaptedGraph(
            g.adjacency, list(g.node_ids), _opt_list(g.entity_types), _opt_list(g.canonical_names), None
        )
    else:
        matrix = _first_attr(g, _MATRIX_ATTRS)
        if matrix is None:
            raise TypeError(
                f"{type(g).__name__} exposes no adjacency matrix: looked for attributes "
                f"{_MATRIX_ATTRS}. Pass a SpectralGraph, a scipy sparse matrix, or an object "
                f"with one of those attributes."
            )
        if not sp.issparse(matrix):
            if isinstance(matrix, np.ndarray):
                matrix = sp.csr_matrix(matrix)
            else:
                raise TypeError(
                    f"adjacency on {type(g).__name__} is {type(matrix).__name__}, not a scipy sparse matrix"
                )
        n = matrix.shape[0]
        node_ids = _as_sequence(_first_attr(g, _NODE_ID_ATTRS), n, None)
        if node_ids is None:
            index = _first_attr(g, _INDEX_ATTRS)
            if isinstance(index, Mapping):
                ordered = sorted(index.items(), key=lambda kv: kv[1])
                if [i for _, i in ordered] == list(range(n)):
                    node_ids = [k for k, _ in ordered]
        entity_types = _as_sequence(_first_attr(g, _TYPE_ATTRS), n, node_ids)
        names = _as_sequence(_first_attr(g, _NAME_ATTRS), n, node_ids)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency must be square, got {matrix.shape}")
    return _AdaptedGraph(matrix, node_ids, entity_types, names, None)


def _opt_list(seq: Sequence[str] | None) -> list[str] | None:
    return None if seq is None else list(seq)


# ── Loader (read-only; the shared loader supersedes this) ──────────


def load_graph(
    *,
    db_path: str | Path | None = None,
    as_of: float | None = None,
    link_types: Iterable[str] | None = None,
    min_confidence: float | None = None,
) -> SpectralGraph:
    """
    Build a symmetrised, confidence-weighted entity-link adjacency matrix from
    the pipeline SQLite database, READ-ONLY.

    Args:
        db_path: path to pipeline.db. Opened as `file:...?mode=ro` — this
            function cannot write, by construction.
        as_of: unix seconds. Keep only links whose effective date is <= this.
            A link with NULL `effective_from` falls back to `created_at`, and
            the count of such fallbacks is reported in `provenance`, because a
            `created_at` fallback is an ingest timestamp, not an event date,
            and trusting it is how F-04 leakage gets in.
        link_types: keep only these link_types. Pass `EVENT_LINK_TYPES` for a
            spectrum over time-varying edges only (see module docstring note 6).
        min_confidence: drop links below this confidence.

    Returns:
        SpectralGraph over ALL entities (including ones every link was filtered
        off, which then appear as isolated rows — dropping them silently would
        renumber rows and hide the filtering).

    Where this misleads:
        - Edge multiplicity is discarded: a pair joined by 3 link_types becomes
          ONE edge carrying the highest of the three confidences. On the live
          graph that collapses 30,131 links into 26,056 distinct undirected
          pairs, so "number of links" and "number of edges in this matrix" are
          different numbers and `provenance` reports both.
        - Weights are confidences in [0.6, 1.0] on the live graph, which is a
          narrow band — `spectral_model(binary=True)`, the default, throws them
          away, and measured it barely matters (effective rank 50.433 binary vs
          51.31 weighted). Do not read the weights as edge importance; they are
          a collector's self-assessment.
        - Link direction is discarded.

    Raises:
        FileNotFoundError: no database at db_path.
        EmptyGraphError: no entities, or no links survive the filters.
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if not path.exists():
        raise FileNotFoundError(f"pipeline database not found at {path}")

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT entity_id, entity_type, canonical_name FROM entities ORDER BY entity_id").fetchall()
        if not rows:
            raise EmptyGraphError(f"{path} has no entities")
        node_ids = [r[0] for r in rows]
        entity_types = [r[1] for r in rows]
        names = [r[2] for r in rows]
        index = {eid: i for i, eid in enumerate(node_ids)}

        links = conn.execute(
            "SELECT entity_id_a, entity_id_b, link_type, confidence, effective_from, created_at FROM entity_links"
        ).fetchall()
    finally:
        conn.close()

    wanted = None if link_types is None else frozenset(link_types)
    # Keyed by the unordered pair, so the several link_types that can join one
    # pair collapse to one edge at the strongest confidence rather than summing
    # into a weight nobody intended.
    best: dict[tuple[int, int], float] = {}
    n_links_kept = 0
    dropped_type = dropped_date = dropped_conf = dropped_unknown = self_loops = 0
    null_effective_from = 0

    for a, b, link_type, confidence, effective_from, created_at in links:
        if wanted is not None and link_type not in wanted:
            dropped_type += 1
            continue
        if min_confidence is not None and (confidence if confidence is not None else 1.0) < min_confidence:
            dropped_conf += 1
            continue
        if effective_from is None:
            null_effective_from += 1
        stamp = effective_from if effective_from is not None else created_at
        if as_of is not None and (stamp is None or stamp > as_of):
            dropped_date += 1
            continue
        ia = index.get(a)
        ib = index.get(b)
        if ia is None or ib is None:
            dropped_unknown += 1
            continue
        if ia == ib:
            self_loops += 1
            continue
        n_links_kept += 1
        key = (ia, ib) if ia < ib else (ib, ia)
        weight = float(confidence) if confidence is not None else 1.0
        if weight > best.get(key, 0.0):
            best[key] = weight

    if not best:
        raise EmptyGraphError(
            f"no links survived the filters (as_of={as_of}, link_types={wanted}, "
            f"min_confidence={min_confidence}); dropped: type={dropped_type} date={dropped_date} "
            f"confidence={dropped_conf} unknown_endpoint={dropped_unknown} self_loop={self_loops}"
        )

    n = len(node_ids)
    pairs = np.fromiter((v for key in best for v in key), dtype=np.int64, count=2 * len(best)).reshape(-1, 2)
    weights = np.fromiter(best.values(), dtype=np.float64, count=len(best))
    matrix = sp.coo_matrix(
        (
            np.concatenate([weights, weights]),
            (np.concatenate([pairs[:, 0], pairs[:, 1]]), np.concatenate([pairs[:, 1], pairs[:, 0]])),
        ),
        shape=(n, n),
    ).tocsr()

    provenance = {
        "db_path": str(path),
        "as_of": as_of,
        "link_types": None if wanted is None else sorted(wanted),
        "min_confidence": min_confidence,
        "n_entities": n,
        "n_links_in_db": len(links),
        "n_links_kept": n_links_kept,
        "n_undirected_edges": len(best),
        "edge_weight": "max confidence over the link_types joining the pair",
        "dropped_wrong_link_type": dropped_type,
        "dropped_after_as_of": dropped_date,
        "dropped_low_confidence": dropped_conf,
        "dropped_unknown_endpoint": dropped_unknown,
        "dropped_self_loop": self_loops,
        "links_dated_by_created_at_fallback": null_effective_from,
        "edge_multiplicity_collapsed": n_links_kept - len(best),
    }
    if as_of is not None and null_effective_from:
        log.warning(
            "load_graph(as_of=%s): %d of %d links had NULL effective_from and were dated by created_at "
            "(an ingest timestamp, not an event date) — treat as-of answers over those edges as suspect (F-04).",
            as_of,
            null_effective_from,
            len(links),
        )
    return SpectralGraph(matrix, node_ids, entity_types, names, provenance)


# ── The model ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SpectralModel:
    """
    A computed rank-`rank` SVD of a graph, plus every fact about what it left
    out. `spectral_embedding` returns only `(U, S)` from this; anything that
    needs to know how much of the graph the embedding actually describes must
    use this object.

    Attributes:
        U: (n, rank) left singular vectors, columns ordered by descending S.
        S: (rank,) singular values, descending.
        embedding: U * S, the coordinates used for cosine similarity.
        covered: (n,) bool. False = this row is outside the truncated subspace
            or isolated, so its similarity is undefined, not zero.
        adjacency: the exact symmetrised, de-selfed matrix the SVD was taken
            of — not the caller's object, so hop counts and degrees always
            describe the matrix that produced U and S.
        node_ids / entity_types / canonical_names / index: row metadata.
        degree: (n,) int, number of distinct neighbours in the matrix.
        n_components / largest_component_size / component_labels: connectivity
            of the input. `component_labels[i]` is i's component id.
        wall_clock_s: seconds spent inside `svds`.
        binary: whether edge weights were binarised before the SVD.
        graph_provenance: carried through from the input graph, if any.
    """

    U: np.ndarray
    S: np.ndarray
    embedding: np.ndarray
    adjacency: Any
    covered: np.ndarray
    degree: np.ndarray
    node_ids: Sequence[str] | None
    entity_types: Sequence[str] | None
    canonical_names: Sequence[str] | None
    index: Mapping[str, int] | None
    rank: int
    n_nodes: int
    n_edges: int
    n_components: int
    largest_component_size: int
    component_labels: np.ndarray
    wall_clock_s: float
    binary: bool
    graph_provenance: Mapping[str, Any]

    @property
    def effective_rank(self) -> float:
        """Entropy-based effective rank of this model's spectrum."""
        return effective_rank(self.S)

    @property
    def n_covered(self) -> int:
        return int(self.covered.sum())

    def coverage_report(self) -> dict[str, Any]:
        """
        What fraction of the graph this embedding actually describes, overall
        and per entity_type. Call this before believing any similarity: on the
        live graph at rank 64 it reports 16% overall.
        """
        report: dict[str, Any] = {
            "rank": self.rank,
            "effective_rank": self.effective_rank,
            "n_nodes": self.n_nodes,
            "n_covered": self.n_covered,
            "coverage_fraction": self.n_covered / self.n_nodes if self.n_nodes else 0.0,
            "n_isolated": int((self.degree == 0).sum()),
            "n_components": self.n_components,
            "largest_component_size": self.largest_component_size,
            "n_components_covered": int(len(set(self.component_labels[self.covered].tolist()))),
            "wall_clock_s": self.wall_clock_s,
        }
        if self.entity_types is not None:
            per_type: dict[str, tuple[int, int]] = {}
            types = np.asarray(self.entity_types)
            for t in sorted(set(self.entity_types)):
                mask = types == t
                per_type[t] = (int(self.covered[mask].sum()), int(mask.sum()))
            report["covered_by_entity_type"] = per_type
        return report

    def row(self, entity_id: str) -> int:
        """Row index for an entity_id. Raises UnknownEntityError if absent."""
        if self.index is None:
            raise UnknownEntityError(
                "this graph carried no entity_ids, so entities cannot be addressed by id; "
                "use the row indices of the matrix you passed"
            )
        try:
            return self.index[entity_id]
        except KeyError:
            raise UnknownEntityError(f"{entity_id!r} is not a node of this graph ({self.n_nodes} nodes)") from None

    def require_covered(self, entity_id: str) -> int:
        """Row index, or DegenerateEmbeddingError explaining which kind of nothing it is."""
        i = self.row(entity_id)
        if self.covered[i]:
            return i
        label = self._label(i)
        if self.degree[i] == 0:
            raise DegenerateEmbeddingError(
                f"{label} has degree 0 in this graph — it has no structural position at all, "
                f"which is not the same as similarity 0.0"
            )
        raise DegenerateEmbeddingError(
            f"{label} has degree {int(self.degree[i])} but lies outside the top-{self.rank} singular subspace "
            f"(its component is not among the {self.rank} strongest directions). Its similarity is undefined, "
            f"not zero. Raise rank, or restrict the graph to its component. "
            f"Only {self.n_covered} of {self.n_nodes} nodes are covered at this rank."
        )

    def _label(self, i: int) -> str:
        eid = self.node_ids[i] if self.node_ids is not None else f"row {i}"
        if self.canonical_names is not None:
            return f"{eid!r} ({self.canonical_names[i]!r})"
        return repr(eid)


class StructuralNeighbour(NamedTuple):
    """
    One row of `nearest_structural`.

    Fields:
        entity_id / canonical_name / entity_type: who it is.
        similarity: cosine in embedding space, in [-1, 1].
        degree: its degree, so a caller can see a hub for what it is.
        hops: shortest-path length to the query over the SAME matrix, computed
            only up to `max_hops` — `None` means "further than max_hops", NOT
            "unreachable". 1 means there is a direct edge, i.e. this neighbour
            is not a hidden connection at all.
    """

    entity_id: str
    canonical_name: str | None
    entity_type: str | None
    similarity: float
    degree: int
    hops: int | None


class NeighbourList(list):
    """
    The result of `nearest_structural` — a real `list` of StructuralNeighbour,
    carrying what the ranking left out. Read `.excluded_uncovered` before
    concluding that something is absent from the list because it is dissimilar.

    Attributes:
        k_requested: the k that was asked for.
        n_candidates: candidates actually ranked.
        truncated: True if there were more candidates than k, so the list is a
            prefix of a longer ranking.
        excluded_uncovered: candidates dropped because their embedding is
            degenerate at this rank (see module docstring note 1).
        excluded_other_type: candidates dropped by the same_type filter.
        max_hops: the hop budget used; neighbours further away report hops=None.
        query: the entity_id that was asked about.
    """

    def __init__(
        self,
        items: Iterable[StructuralNeighbour],
        *,
        query: str,
        k_requested: int,
        n_candidates: int,
        excluded_uncovered: int,
        excluded_other_type: int,
        max_hops: int,
    ) -> None:
        super().__init__(items)
        self.query = query
        self.k_requested = k_requested
        self.n_candidates = n_candidates
        self.truncated = n_candidates > len(self)
        self.excluded_uncovered = excluded_uncovered
        self.excluded_other_type = excluded_other_type
        self.max_hops = max_hops

    def __repr__(self) -> str:
        return (
            f"NeighbourList(query={self.query!r}, returned={len(self)}, k_requested={self.k_requested}, "
            f"n_candidates={self.n_candidates}, truncated={self.truncated}, "
            f"excluded_uncovered={self.excluded_uncovered}, excluded_other_type={self.excluded_other_type})"
        )


# ── Caching ────────────────────────────────────────────────────────
# svds on the live graph costs ~1.5 s; nearest_structural must not pay that per
# call. Keyed on the object's identity AND its shape/nnz, and the entry holds a
# weak reference so a recycled id() cannot serve a stale model.

_CACHE: dict[tuple[int, int, int, int, bool], tuple[Any, SpectralModel]] = {}
_CACHE_MAX = 8


def _cached_model(g: Any, rank: int, binary: bool) -> SpectralModel:
    adapted = _adapt(g)
    key = (id(g), adapted.matrix.shape[0], int(adapted.matrix.nnz), rank, binary)
    hit = _CACHE.get(key)
    if hit is not None:
        ref, model = hit
        if ref is None or ref() is g:
            return model
        del _CACHE[key]
    model = spectral_model(g, rank=rank, binary=binary)
    try:
        ref = weakref.ref(g)
    except TypeError:
        ref = None
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (ref, model)
    return model


def clear_cache() -> None:
    """Drop memoised models. Call this in a test that mutates a graph in place."""
    _CACHE.clear()


# ── Public API ─────────────────────────────────────────────────────


def spectral_model(g: Any, *, rank: int = 64, binary: bool = True) -> SpectralModel:
    """
    Compute the rank-`rank` SVD of `g`'s symmetrised adjacency matrix and every
    diagnostic that goes with it.

    This is the function to use. `spectral_embedding` is the two-value form
    fixed by the module's contract and drops the diagnostics.

    Args:
        g: a SpectralGraph, a scipy sparse matrix, or any object exposing an
            adjacency matrix and node ids (see `_adapt`).
        rank: number of singular triplets. Must be < min(n, m) — `svds` cannot
            return the full spectrum, and a silently reduced rank is refused.
        binary: binarise edge weights before the SVD. True reproduces the
            measured reference (effective rank 50.43 on the live graph);
            confidence-weighted gives 51.31 — the choice is not neutral, so it
            is explicit and recorded on the result.

    Returns:
        SpectralModel.

    Where this misleads:
        The matrix is symmetrised with an elementwise maximum, so link direction
        is discarded — `works_for(person, company)` and its converse are the
        same edge here. And see the whole "WHERE THIS MISLEADS" section of the
        module docstring: at rank 64 on the live graph this model describes 16%
        of the entities.

    Raises:
        EmptyGraphError: no nodes, or no edges.
        RankTooLargeError: rank >= min(n, m), or rank < 1.
        MechanismSpectralError: svds failed to converge.
    """
    adapted = _adapt(g)
    matrix = adapted.matrix
    n = matrix.shape[0]
    if n == 0:
        raise EmptyGraphError("graph has no nodes")
    matrix = matrix.tocsr(copy=True)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    matrix = matrix.maximum(matrix.T)
    if matrix.nnz == 0:
        raise EmptyGraphError(f"graph has {n} nodes and no edges — there is no spectrum to compute")
    if binary:
        matrix = (matrix > 0).astype(np.float64)
    else:
        matrix = matrix.astype(np.float64)

    if rank < 1:
        raise RankTooLargeError(f"rank must be >= 1, got {rank}")
    limit = min(matrix.shape) - 1
    if rank > limit:
        raise RankTooLargeError(
            f"rank={rank} is not computable for a {matrix.shape[0]}x{matrix.shape[1]} matrix: "
            f"scipy.sparse.linalg.svds needs rank <= min(shape)-1 = {limit}. Refusing to compute a "
            f"smaller rank silently, because a shortened spectrum makes effective_rank meaningless. "
            f"Pass rank <= {limit}."
        )

    # Deterministic start vector: svds' default is random, and a report whose
    # numbers move between runs cannot be checked by anyone.
    v0 = np.full(matrix.shape[0], 1.0 / np.sqrt(matrix.shape[0]))
    started = time.perf_counter()
    try:
        U, S, _ = svds(matrix, k=rank, solver="arpack", v0=v0)
    except Exception as exc:  # ArpackNoConvergence and friends
        raise MechanismSpectralError(
            f"svds failed at rank={rank} on a {matrix.shape[0]}-node graph with {matrix.nnz} nonzeros: {exc}"
        ) from exc
    wall_clock_s = time.perf_counter() - started

    order = np.argsort(S)[::-1]
    S = np.asarray(S, dtype=np.float64)[order]
    U = np.asarray(U, dtype=np.float64)[:, order]
    if S.shape[0] != rank:
        raise MechanismSpectralError(
            f"svds returned {S.shape[0]} singular values for rank={rank} — refusing to proceed"
        )

    embedding = U * S
    norms = np.linalg.norm(embedding, axis=1)
    peak = float(norms.max())
    if peak <= 0.0:
        raise EmptyGraphError("every embedding row is zero — the matrix has no spectral structure at all")
    covered = norms > _COVERAGE_REL_TOL * peak

    degree = np.asarray((matrix > 0).sum(axis=1)).ravel().astype(np.int64)
    n_components, labels = connected_components(matrix, directed=False)
    largest = int(np.bincount(labels).max()) if n else 0

    node_ids = adapted.node_ids
    index = {eid: i for i, eid in enumerate(node_ids)} if node_ids is not None else None
    if index is not None and len(index) != len(node_ids):
        raise ValueError(f"node_ids contains duplicates ({len(node_ids)} ids, {len(index)} distinct)")

    provenance = dict(getattr(g, "provenance", {}) or {})
    model = SpectralModel(
        U=U,
        S=S,
        embedding=embedding,
        adjacency=matrix,
        covered=covered,
        degree=degree,
        node_ids=node_ids,
        entity_types=adapted.entity_types,
        canonical_names=adapted.canonical_names,
        index=index,
        rank=rank,
        n_nodes=n,
        n_edges=int(matrix.nnz // 2),
        n_components=int(n_components),
        largest_component_size=largest,
        component_labels=labels,
        wall_clock_s=wall_clock_s,
        binary=binary,
        graph_provenance=provenance,
    )
    if model.n_covered < n:
        log.info(
            "spectral_model(rank=%d): %d of %d nodes lie outside the truncated subspace "
            "(%d isolated, %d connected components in the input) — their similarity is undefined, not zero.",
            rank,
            n - model.n_covered,
            n,
            int((degree == 0).sum()),
            n_components,
        )
    return model


def spectral_embedding(g: Any, *, rank: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    Rank-`rank` truncated SVD of `g`'s symmetrised adjacency matrix.

    Returns:
        (U, S) — U is (n, rank) left singular vectors with columns ordered by
        descending singular value; S is (rank,) descending singular values.
        `U * S` gives the coordinates used by `structural_similarity`.

    Where this misleads:
        This two-value form CANNOT tell you how much of the graph it describes.
        On the live graph at rank 64, 16,823 of 20,026 rows of U are numerically
        zero — not "at the origin", but "in a connected component that the top
        64 directions do not reach". Treating those rows as positions produces
        confident nonsense. Use `spectral_model(g, rank=...)` and read
        `.coverage_report()` whenever the answer will be shown to anyone.

        Nothing is truncated silently: an uncomputable `rank` raises
        `RankTooLargeError` rather than quietly returning fewer components, so
        `len(S) == rank` always holds on return.

    Raises:
        EmptyGraphError, RankTooLargeError, MechanismSpectralError — see
        `spectral_model`.
    """
    model = _cached_model(g, rank, True)
    return model.U, model.S


def effective_rank(S: Any) -> float:
    """
    Effective rank of a spectrum: the exponential of the Shannon entropy of the
    singular values normalised to sum to 1 (Roy & Vetterli 2007).

    A flat spectrum of k equal values gives exactly k; a spectrum with all mass
    on one value gives 1.0. It answers "how many dimensions is this embedding
    really using", which is the number the HetTGN failed: 2.6-10.6 of 64.

    Args:
        S: 1-D non-negative singular values, in any order.

    Returns:
        float in [1, len(S)].

    Where this misleads:
        It is a property of the SPECTRUM ONLY. It says nothing about whether the
        embedding covers the graph — this module's rank-64 model on the live
        graph scores ~50/64 while describing 16% of the entities. High effective
        rank plus low coverage is exactly the shape of the live result, and
        quoting only the first number would be dishonest.

        It is also not comparable across different `rank`: exp-entropy is bounded
        by len(S), so a rank-32 model cannot exceed 32.

    Raises:
        ValueError: empty, non-1-D, non-finite, negative, or all-zero input. An
            all-zero spectrum has no defined effective rank and must not be
            reported as 0.0 or 1.0.
    """
    s = np.asarray(S, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"S must be 1-D, got shape {s.shape}")
    if s.size == 0:
        raise ValueError("S is empty — there is no spectrum to measure")
    if not np.all(np.isfinite(s)):
        raise ValueError("S contains non-finite values")
    if np.any(s < 0):
        raise ValueError("S contains negative values — singular values cannot be negative")
    total = s.sum()
    if total <= 0:
        raise ValueError("S sums to zero — effective rank is undefined for an all-zero spectrum, not 0.0 or 1.0")
    p = s / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def structural_similarity(g: Any, a_id: str, b_id: str, *, rank: int = 64) -> float:
    """
    Cosine similarity between two entities' positions in the rank-`rank`
    embedding: high when the two are reached by the same parts of the graph,
    whether or not an edge joins them.

    Returns:
        float in [-1, 1]. 1.0 = identical structural position. 0.0 means
        "computed, and orthogonal" — every other kind of nothing raises.

    Where this misleads:
        - For countries and companies this measure is close to a degree ranking
          (median |Spearman rho| against plain degree: country 0.54 and 0.96 for
          Russia; company 0.91). Call `degree_rank_correlation` before trusting
          it on a hub type.
        - 67% of covered company pairs score > 0.9: the company block of this
          embedding barely discriminates.
        - Cosine ignores magnitude, so a degree-3 node and a degree-300 node can
          score 0.99 by sitting along the same direction. Read the `degree`
          field that `nearest_structural` returns.

    Raises:
        UnknownEntityError: an id is not a node of this graph.
        DegenerateEmbeddingError: an id is isolated, or outside the truncated
            subspace, so its position is undefined — deliberately not 0.0.
    """
    model = _cached_model(g, rank, True)
    ia = model.require_covered(a_id)
    ib = model.require_covered(b_id)
    va = model.embedding[ia]
    vb = model.embedding[ib]
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0.0:  # unreachable: require_covered guarantees non-zero norms
        raise DegenerateEmbeddingError(
            f"zero-norm embedding for {a_id!r} or {b_id!r} despite passing the coverage test"
        )
    return float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))


def nearest_structural(
    g: Any,
    entity_id: str,
    *,
    k: int = 10,
    same_type: bool = True,
    rank: int = 64,
    max_hops: int = 3,
) -> NeighbourList:
    """
    The `k` entities whose structural position is most like `entity_id`'s.

    This is the "hidden long connection" search: a neighbour can rank first with
    no edge to the query, because both are reached by the same neighbourhoods.

    Args:
        g: graph (see `spectral_model`).
        entity_id: the query entity.
        k: how many to return. Fewer come back when fewer candidates exist, and
            the returned `NeighbourList` says so via `.n_candidates`.
        same_type: restrict candidates to the query's entity_type. Requires the
            graph to carry entity types; raises if it does not, rather than
            silently comparing a country to a wallet.
        rank: embedding rank.
        max_hops: hop budget for the `hops` field. Beyond it, `hops` is None,
            meaning "not within max_hops", not "unreachable".

    Returns:
        NeighbourList (a `list`) of StructuralNeighbour, descending by
        similarity, carrying `.truncated`, `.n_candidates`,
        `.excluded_uncovered` and `.excluded_other_type`.

    Where this misleads — the honest summary:
        - **It is partly a degree ranking, and how much depends on the type.**
          Measured median Spearman rho vs plain degree rank, same-type
          candidates: company -0.91, topic +0.58, country +0.54,
          instrument -0.40, wallet -0.30. Judge by |rho| — a perfect
          anti-correlation is the same list reversed. For "Russia" among
          countries it is +0.957, i.e. essentially `ORDER BY degree`.
        - **"No direct edge" does not mean "far".** On the live graph no two
          instruments share an edge, yet every covered instrument pair is 2 hops
          apart through a shared country or topic. Check `hops` before calling
          a result a long-range discovery: hops=2 through a 193-degree hub is a
          co-mention, not a mechanism.
        - **Most of the graph cannot be queried at all.** 16,823 of 20,026 live
          entities are outside the rank-64 subspace and raise instead of
          appearing; `.excluded_uncovered` counts the candidates lost the same
          way.

    Raises:
        UnknownEntityError, DegenerateEmbeddingError: as `structural_similarity`.
        ValueError: k < 1, max_hops < 1, or same_type=True on a graph with no
            entity types.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    model = _cached_model(g, rank, True)
    i = model.require_covered(entity_id)

    mask = model.covered.copy()
    excluded_uncovered = int((~model.covered).sum())
    mask[i] = False
    excluded_other_type = 0
    if same_type:
        if model.entity_types is None:
            raise ValueError(
                "same_type=True but this graph carries no entity types; pass same_type=False "
                "to compare across types explicitly, or supply a graph with entity_types"
            )
        types = np.asarray(model.entity_types)
        same = types == types[i]
        excluded_other_type = int((mask & ~same).sum())
        mask &= same

    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        return NeighbourList(
            [],
            query=entity_id,
            k_requested=k,
            n_candidates=0,
            excluded_uncovered=excluded_uncovered,
            excluded_other_type=excluded_other_type,
            max_hops=max_hops,
        )

    unit = model.embedding / np.linalg.norm(model.embedding, axis=1, keepdims=True).clip(min=np.finfo(float).tiny)
    sims = unit[candidates] @ unit[i]
    top = candidates[np.argsort(-sims)[:k]]
    hops = _hop_distances(model, i, top, max_hops)

    items = [
        StructuralNeighbour(
            entity_id=model.node_ids[j] if model.node_ids is not None else str(j),
            canonical_name=model.canonical_names[j] if model.canonical_names is not None else None,
            entity_type=model.entity_types[j] if model.entity_types is not None else None,
            similarity=float(np.clip(unit[j] @ unit[i], -1.0, 1.0)),
            degree=int(model.degree[j]),
            hops=hops.get(int(j)),
        )
        for j in top
    ]
    return NeighbourList(
        items,
        query=entity_id,
        k_requested=k,
        n_candidates=int(candidates.size),
        excluded_uncovered=excluded_uncovered,
        excluded_other_type=excluded_other_type,
        max_hops=max_hops,
    )


def degree_rank_correlation(g: Any, entity_id: str, *, same_type: bool = True, rank: int = 64) -> float:
    """
    Spearman correlation between the similarity ranking `nearest_structural`
    would produce for `entity_id` and a plain ranking by degree, over the SAME
    candidate set.

    This exists because it is the honest check on this whole module. |rho| near
    1 means the similarity list is a degree list and carries no extra
    information — the SIGN does not rescue it, because a perfect
    anti-correlation is that same list reversed. |rho| near 0 means the ranking
    is genuinely structural. Measured on the live graph: "Russia" among
    countries scores +0.957, and the median company scores -0.913 — do not sell
    either list as an insight.

    Returns:
        float in [-1, 1].

    Where this misleads:
        Computed over the covered candidates only, which on the live graph is
        16% of the graph — it says nothing about the invisible 84%. And a rho
        near 0 means "not degree"; it does not mean "correct".

    Raises:
        UnknownEntityError, DegenerateEmbeddingError: as `structural_similarity`.
        ValueError: fewer than 3 candidates, so a rank correlation is meaningless.
    """
    from scipy.stats import spearmanr

    model = _cached_model(g, rank, True)
    i = model.require_covered(entity_id)
    mask = model.covered.copy()
    mask[i] = False
    if same_type:
        if model.entity_types is None:
            raise ValueError("same_type=True but this graph carries no entity types")
        types = np.asarray(model.entity_types)
        mask &= types == types[i]
    candidates = np.flatnonzero(mask)
    if candidates.size < 3:
        raise ValueError(
            f"only {candidates.size} comparable candidates for {entity_id!r}; a rank correlation over "
            f"fewer than 3 points is not a number worth reporting"
        )
    unit = model.embedding / np.linalg.norm(model.embedding, axis=1, keepdims=True).clip(min=np.finfo(float).tiny)
    sims = unit[candidates] @ unit[i]
    rho = spearmanr(sims, model.degree[candidates]).statistic
    if not np.isfinite(rho):
        raise ValueError(
            f"Spearman rho is undefined for {entity_id!r} — the similarities or the degrees are constant "
            f"over its {candidates.size} candidates"
        )
    return float(rho)


def _hop_distances(model: SpectralModel, source: int, targets: np.ndarray, max_hops: int) -> dict[int, int]:
    """
    BFS from `source` over the model's own adjacency, bounded at `max_hops`.

    Targets not reached inside the budget are simply absent from the returned
    dict, so the caller must render that as "further than max_hops" and never as
    "unreachable" — the budget is the reason, not the graph.
    """
    wanted = {int(t) for t in targets}
    if not wanted:
        return {}
    # Reconstruct connectivity from the embedding's source matrix is not
    # possible, so walk the boolean neighbourhood via repeated sparse products
    # on an indicator vector — cheap for max_hops <= 3.
    adj = model.adjacency
    found: dict[int, int] = {}
    n = model.n_nodes
    visited = np.zeros(n, dtype=bool)
    visited[source] = True
    frontier = np.zeros(n, dtype=bool)
    frontier[source] = True
    for hop in range(1, max_hops + 1):
        nxt = (adj @ frontier.astype(np.float64)) > 0
        nxt &= ~visited
        if not nxt.any():
            break
        for j in np.flatnonzero(nxt):
            jj = int(j)
            if jj in wanted and jj not in found:
                found[jj] = hop
        visited |= nxt
        frontier = nxt
        if len(found) == len(wanted):
            break
    return found
