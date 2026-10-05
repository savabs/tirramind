"""
TirraMind — Mechanism Layer: connectivity measures (Layer 3, world model).

WHAT THIS COMPUTES
    Structural connectivity between two entities of the evidence graph, as
    linear algebra over a sparse adjacency matrix. Three measures:

      * ``personalised_pagerank``  — stationary mass of an alpha-restart random
        walk seeded at one entity. Hub-safe by construction: a walker crossing a
        193-degree node splits its mass 193 ways, so routes through "United
        States" are discounted automatically rather than counted as four
        independent confirmations (which naive path enumeration did).
      * ``connectivity``           — that mass made interpretable: global rank,
        percentile, and — the number a customer can act on — rank among
        entities of the same type as the destination plus a ratio to that
        group's median.
      * ``effective_resistance``   — Laplacian resistance distance, i.e. how
        many *parallel* routes exist, not just how short the shortest is.

    No neural network, no parameters, no training, no collapse mode. A rank-2.6
    embedding cannot happen here because nothing is fitted.

WHERE THIS MISLEADS (read before quoting a number to a customer)
    1. It measures ROUTING, not evidence. Most links in this graph are static
       geography (``produced_in``, ``exchange_country``, ``located_in``): Canada
       produced oil in 1990 and produces it today, so such an edge carries zero
       time-varying information. High PageRank mass through scaffold edges means
       "the map says these could be related", never "something happened".
       Counting independent *sources* is a separate concern and must be done
       over time-varying (event) edges only; this module deliberately does not
       do it and must not be read as having done it.
    2. It is only as time-honest as the graph handed to it. ``load_graph`` takes
       ``as_of``; explaining a 2019 pattern with the 2026 graph is leakage
       (LESSONS F-04). This module cannot detect that mistake for you — it sees
       a matrix, not a date.
    3. Rank and percentile are quoted over EVERY entity in the graph, including
       entities the walker can never reach (the live graph has 1,724 isolated
       entities out of 20,026 and 4,447 connected components). So "rank 127 of
       20,026" is really "127th of the few thousand that are reachable at all,
       and ahead of everything unreachable". ``connectivity`` returns
       ``reachable_count`` and ``isolated_count`` so this is never hidden.
    4. Edge WEIGHTS change every number. On the live graph, 30,131 links reduce
       to 26,056 unique entity pairs; treating the 4,075 duplicates as weight-2
       edges moves Russia -> WTI from global rank 127 to rank 155. This module
       uses the adjacency exactly as given and reports
       ``multi_edge_weights_present`` so the choice is visible at the call site.
       The measured reference numbers in the tests are on the SIMPLE
       (deduplicated, unweighted, undirected) graph.
    5. Mass magnitudes are not comparable across graphs, alphas or seeds. Only
       the rank/ratio fields are. That is why the raw mass never travels alone.

DESIGN NOTES
    ``alpha`` is the single tunable in this layer. It is the restart
    probability: the expected walk length before teleporting home is 1/alpha,
    so alpha *is* the operational definition of "a long connection". It is an
    explicit keyword on every entry point and is echoed back in every result
    dict — it must be shown to the customer, not buried in a default.

GRAPH CONTRACT
    Anything carrying an adjacency matrix, an id order, per-entity names and
    per-entity types works. Two spellings are accepted, because this layer's
    brief and the shipped ``agent/mechanism/graph.py`` disagree on them:

        adjacency   ``.A``      or ``.adjacency``
        id order    ``.ents``   or ``.entity_ids``
        types       ``.etype``  or ``.entity_types``
        names       ``.name``   or ``.canonical_names``
        row lookup  ``.idx``    or derived from the id order

    Names and types may be a Mapping keyed by entity_id or a sequence aligned to
    the matrix rows. Anything else raises — nothing is guessed.

    Every entry point also takes ``edges="all"`` (default) or ``edges="evidence"``,
    the latter selecting ``.evidence_adjacency`` when the graph provides one, so
    "is there a route at all" and "does anything time-varying support it" are
    two explicit questions rather than one ambiguous number.

Layer: 3 (world model / structural beliefs). Stateless. Read-only on the DB —
this module never opens the database at all.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

log = logging.getLogger(__name__)

__all__ = [
    "DisconnectedEntityError",
    "EmptyGraphError",
    "IsolatedEntityError",
    "MechanismError",
    "UnknownEntityError",
    "connectivity",
    "effective_resistance",
    "personalised_pagerank",
]

# Residual tolerance for the sparse solves. Both solves are exact up to
# floating point on this graph size (measured max residual 1.1e-16), so a
# breach means the matrix was not what we assumed, not that precision ran out.
_RESIDUAL_TOL = 1e-8

# Below this, a PageRank mass is numerical dust rather than a real route.
_MASS_EPS = 1e-15


# ── Failure modes are distinct types, never a plausible zero ────────────


class MechanismError(ValueError):
    """Base class: a connectivity question that cannot be answered as asked."""


class UnknownEntityError(MechanismError):
    """An entity id is absent from the graph (wrong id, or wrong ``as_of``)."""


class EmptyGraphError(MechanismError):
    """The graph has no nodes, or no edges at all."""


class IsolatedEntityError(MechanismError):
    """The seed/source entity has degree 0, so nothing can be computed from it.

    Distinct from "computed, and the answer is not connected": a zero-degree
    seed yields a PageRank vector that is exactly ``e_seed``, which would read
    as "everything in the world is unrelated" when the truth is "this entity
    has no edges as of this date".
    """


class DisconnectedEntityError(MechanismError):
    """The two entities lie in different connected components.

    Raised only where the measure is undefined rather than zero: effective
    resistance between two components is infinite. PageRank mass, by contrast,
    is legitimately 0.0 there, and ``connectivity`` returns it with
    ``connected=False``.
    """


# ── Graph-contract helpers ──────────────────────────────────────────────


# The graph object may spell its attributes either way: the short contract from
# this layer's brief (``.A``/``.idx``/``.ents``/``.name``/``.etype``) or the
# longer one that agent/mechanism/graph.py's MechanismGraph actually shipped
# (``.adjacency``/``.entity_ids``/``.canonical_names``/``.entity_types``, with
# ``.evidence_adjacency`` alongside). Both are accepted by name — never guessed
# at positionally — so an unexpected object still fails loudly instead of being
# half-read.
_ADJACENCY_ATTRS: dict[str, tuple[str, ...]] = {
    "all": ("A", "adjacency"),
    "evidence": ("evidence_adjacency",),
}
_IDS_ATTRS = ("ents", "entity_ids")
_TYPE_ATTRS = ("etype", "entity_types")
_NAME_ATTRS = ("name", "canonical_names")


def _first_attr(g: Any, names: tuple[str, ...]) -> tuple[str, Any] | tuple[None, None]:
    for attr in names:
        value = getattr(g, attr, None)
        if value is not None:
            return attr, value
    return None, None


def _ids(g: Any) -> Sequence[str] | None:
    _attr, value = _first_attr(g, _IDS_ATTRS)
    return value


def _adjacency(g: Any, edges: str = "all") -> sp.csr_matrix:
    """Validate and return the adjacency as CSR. Never coerces silently.

    ``edges="evidence"`` selects the graph's evidence-only matrix, i.e. the
    time-varying (event) edges. Static geography routes evidence but is not
    evidence, so a claim about independent support must be computed on that
    matrix. It is not the default because the reference numbers were measured on
    the full graph, and a silent switch between the two would be exactly the
    kind of hidden change this layer exists to catch.
    """
    if edges not in _ADJACENCY_ATTRS:
        raise MechanismError(f"edges must be one of {sorted(_ADJACENCY_ATTRS)}; got {edges!r}")
    attr, A = _first_attr(g, _ADJACENCY_ATTRS[edges])
    if A is None:
        raise MechanismError(
            f"graph has no {' / '.join('.' + a for a in _ADJACENCY_ATTRS[edges])} adjacency matrix for edges={edges!r}"
        )
    if not sp.issparse(A):
        raise MechanismError(f"graph .{attr} must be a scipy.sparse matrix, got {type(A)!r}")
    A = sp.csr_matrix(A)
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        raise MechanismError(f"graph .{attr} must be square, got {A.shape}")
    if n_rows == 0:
        raise EmptyGraphError(f"graph .{attr} is 0x0 — no entities")
    if A.nnz == 0:
        raise EmptyGraphError(
            f"graph .{attr} has {n_rows} entities but no edges; every connectivity "
            f"question over it would answer with a plausible zero"
        )
    if A.data.size and A.data.min() < 0:
        raise MechanismError(
            f"graph .{attr} has negative weights; random-walk and Laplacian measures are undefined for signed graphs"
        )
    ids = _ids(g)
    if ids is not None and len(ids) != n_rows:
        raise MechanismError(f"graph has {len(ids)} entity ids but .{attr} is {A.shape}")
    return A


def _index(g: Any) -> Mapping[str, int]:
    """entity_id -> row index, from ``.idx`` or derived from the id sequence."""
    idx = getattr(g, "idx", None)
    if isinstance(idx, Mapping):
        return idx
    ids = _ids(g)
    if ids is None:
        raise MechanismError(
            "graph exposes neither .idx (entity_id -> row) nor an id sequence "
            "(.ents / .entity_ids), so entity ids cannot be located in the matrix"
        )
    return {str(e): i for i, e in enumerate(ids)}


def _row(g: Any, entity_id: str) -> int:
    """Row index of ``entity_id``, or raise ``UnknownEntityError``."""
    idx = _index(g)
    try:
        return int(idx[entity_id])
    except KeyError:
        raise UnknownEntityError(
            f"entity_id {entity_id!r} is not in this graph "
            f"({len(idx)} entities). Either the id is wrong, or it has no links "
            f"on or before the as_of date this graph was built with."
        ) from None


def _lookup(g: Any, attrs: tuple[str, ...], entity_id: str, row: int) -> str | None:
    """Read one per-entity field, accepting a Mapping or a sequence aligned to rows."""
    attr, table = _first_attr(g, attrs)
    if table is None:
        return None
    if isinstance(table, Mapping):
        value = table.get(entity_id)
    elif isinstance(table, (Sequence, np.ndarray)) and not isinstance(table, (str, bytes)):
        if row >= len(table):
            return None
        value = table[row]
    else:
        raise MechanismError(
            f"graph .{attr} must be a Mapping or a sequence aligned to the matrix rows, got {type(table)!r}"
        )
    return None if value is None else str(value)


def _type_vector(g: Any, n: int) -> np.ndarray:
    """Per-row entity type as a string array, for same-type ranking."""
    attr, table = _first_attr(g, _TYPE_ATTRS)
    if table is None:
        raise MechanismError(
            "graph exposes no entity types (.etype / .entity_types) — the same-type "
            "rank is the interpretable half of this measure and cannot be faked "
            "from the mass alone"
        )
    if isinstance(table, Mapping):
        ids = _ids(g)
        if ids is None or len(ids) != n:
            raise MechanismError(f"graph .{attr} is a Mapping but the id sequence is missing/misaligned")
        return np.asarray([str(table.get(e, "")) for e in ids], dtype=object)
    arr = np.asarray(table, dtype=object)
    if arr.shape[0] != n:
        raise MechanismError(f"graph .{attr} has {arr.shape[0]} entries but the matrix is {n}x{n}")
    return arr


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not (0.0 < a <= 1.0):
        raise MechanismError(
            f"alpha must be in (0, 1]; got {alpha!r}. alpha is the restart "
            f"probability, so the mean walk length is 1/alpha."
        )
    return a


# ── Personalised PageRank ───────────────────────────────────────────────


def _ppr(A: sp.csr_matrix, seed_row: int, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact sparse solve of the PPR linear system. Returns (mass, diagnostics)."""
    n = A.shape[0]
    degree = np.asarray(A.sum(axis=1)).ravel()

    if degree[seed_row] <= 0:
        raise IsolatedEntityError(
            f"seed row {seed_row} has degree 0 — it has no links in this graph, "
            f"so no connectivity can be computed from it. This is not the same "
            f"as 'not connected to the target'."
        )

    # Zero-degree ("dangling") rows: leave the row all-zero instead of dividing
    # by zero. In an undirected graph such a row is unreachable, so it costs
    # nothing; in a directed one it leaks mass, which is reported below and
    # never silently absorbed.
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_degree = np.where(degree > 0, 1.0 / degree, 0.0)
    P = sp.diags(inv_degree) @ A  # row-stochastic transition matrix

    # (I - (1-alpha) P^T) x = alpha e_seed
    M = (sp.eye(n, format="csr") - (1.0 - alpha) * P.T).tocsc()
    b = np.zeros(n, dtype=float)
    b[seed_row] = alpha
    x = spla.spsolve(M, b)
    x = np.asarray(x, dtype=float).ravel()

    residual = float(np.abs(M @ x - b).max()) if n else 0.0
    if not np.all(np.isfinite(x)) or residual > _RESIDUAL_TOL:
        raise MechanismError(
            f"PageRank sparse solve did not converge: max residual {residual:.3e} "
            f"(tolerance {_RESIDUAL_TOL:.0e}). Refusing to return a vector that "
            f"looks like a distribution but is not one."
        )

    # Tiny negatives are round-off around exact zeros; anything larger is a bug.
    most_negative = float(x.min())
    if most_negative < -1e-12:
        raise MechanismError(
            f"PageRank solve produced a mass of {most_negative:.3e}; a probability "
            f"cannot be negative, so the adjacency matrix is not what was assumed."
        )
    x = np.clip(x, 0.0, None)

    total = float(x.sum())
    info = {
        "alpha": alpha,
        "solve_residual": residual,
        # Mass that fell off dangling rows. Exactly 0 for an undirected graph;
        # > 0 means the adjacency was directed and the numbers below are a
        # sub-stochastic measure, not a probability distribution.
        "teleport_mass_lost": max(0.0, 1.0 - total),
        "mass_total": total,
        "isolated_count": int((degree <= 0).sum()),
        # 4,075 of the live graph's 30,131 links are a repeat of an existing
        # entity pair under a different link_type. If they were summed into
        # weights, ranks shift (Russia->WTI: 127 -> 155).
        "multi_edge_weights_present": bool(A.data.size and A.data.max() > 1.0),
    }
    if info["teleport_mass_lost"] > 1e-9:
        log.warning(
            "PPR lost %.4g of its mass on dangling rows (graph is directed or "
            "row-degenerate); masses are sub-stochastic.",
            info["teleport_mass_lost"],
        )
    return x, info


def personalised_pagerank(g: Any, seed_id: str, *, alpha: float = 0.15, edges: str = "all") -> np.ndarray:
    """Personalised PageRank mass over every entity, restarting at ``seed_id``.

    Computes the exact stationary distribution of a random walk that, at each
    step, teleports back to ``seed_id`` with probability ``alpha`` and otherwise
    steps to a uniformly random neighbour. Solved directly as the sparse system
    ``(I - (1-alpha) Pᵀ) x = alpha e_seed`` with ``P`` row-stochastic — no power
    iteration, so there is no iteration cap to silently hit and no convergence
    threshold to quietly accept.

    Why this and not path enumeration: mass through a node is divided by that
    node's degree, so a route via a 193-degree hub such as "United States" is
    discounted by construction. Enumerating paths instead returned the same hub
    route four times and called it four confirmations.

    Args:
        g: any object satisfying the graph contract in the module docstring.
        seed_id: entity to restart from (the hypothesis' cause side).
        alpha: restart probability in (0, 1]. Mean walk length is ``1/alpha``,
            so alpha defines what "a long connection" means. Smaller alpha
            roams further and flattens the ranking; larger alpha stays local.
        edges: ``"all"`` (default) walks every edge, including static geography.
            ``"evidence"`` walks only the graph's time-varying edges — a much
            sparser graph in which many pairs are legitimately unreachable.

    Returns:
        ``np.ndarray`` of length ``n`` indexed by ``g.idx``, non-negative, and
        summing to 1.0 for an undirected graph.

    Raises:
        UnknownEntityError: ``seed_id`` is not in this graph.
        IsolatedEntityError: the seed has degree 0 — nothing to compute.
        EmptyGraphError: the graph has no nodes or no edges.
        MechanismError: the adjacency violates the contract, or the solve did
            not reach tolerance (never returns an unconverged vector).

    Where it misleads:
        A bare mass vector is uninterpretable and NOT comparable across graphs,
        seeds or alphas — only ranks and ratios are, which is what
        ``connectivity`` exists to produce. The vector also cannot distinguish
        a route made of static geography (scaffold) from one made of dated
        events (evidence); high mass is "the map allows this", not "this
        happened". If the graph is directed, mass leaks on sink rows and the
        vector sums to less than 1; that event is logged at WARNING and
        surfaced as ``teleport_mass_lost`` by ``connectivity``.
    """
    alpha = _check_alpha(alpha)
    A = _adjacency(g, edges)
    seed_row = _row(g, seed_id)
    x, _info = _ppr(A, seed_row, alpha)
    return x


# ── Interpretable connectivity verdict ──────────────────────────────────


def connectivity(g: Any, src_id: str, dst_id: str, *, alpha: float = 0.15, edges: str = "all") -> dict[str, Any]:
    """Structural connectivity from ``src_id`` to ``dst_id``, made interpretable.

    Runs one personalised PageRank from ``src_id`` and locates ``dst_id`` in it
    three ways: globally, as a percentile, and — the only framing a customer can
    act on — against the other entities of the SAME TYPE as ``dst_id``
    ("instrument #1 of 93, 28x the median instrument"). A raw mass such as
    2.6e-03 means nothing on its own; the same mass can be extraordinary or
    unremarkable depending on the graph it came from.

    Args:
        g: object satisfying the graph contract in the module docstring.
        src_id: source entity (walk restarts here).
        dst_id: destination entity, of any type; its type sets the peer group.
        alpha: restart probability in (0, 1]; see ``personalised_pagerank``.
        edges: ``"all"`` (default) or ``"evidence"``; see ``personalised_pagerank``.
            The reference figures for this graph were measured with ``"all"``.

    Returns:
        dict with:
          ``src_id``/``src_name``, ``dst_id``/``dst_name``/``dst_type``, ``alpha``
          ``mass``                    PPR mass at ``dst_id``
          ``connected``               False iff the mass is numerical zero
          ``global_rank``/``n_entities``/``percentile``
                                      competition rank (1 + count strictly
                                      greater), over ALL entities including
                                      unreachable ones; the source itself is
                                      included and is normally rank 1
          ``global_ties``             entities sharing exactly this mass
          ``reachable_count``         entities with mass > 0 (the honest
                                      denominator behind the percentile)
          ``isolated_count``          degree-0 entities that could never rank
          ``type_rank``/``type_ties``/``type_count``/``type_median``/
          ``ratio_to_type_median``
                                      the peer-group framing; ``type_ties``
                                      counts peers on exactly this mass, so a
                                      zero-mass "#1 of 93" cannot be quoted as
                                      a finding; ratio is None when the peer
                                      median is 0 (never inf)
          ``teleport_mass_lost``      mass lost on dangling rows (0 if undirected)
          ``multi_edge_weights_present``  True if any edge weight > 1, which
                                      shifts every rank relative to the simple
                                      graph the reference numbers were measured on
          ``solve_residual``          max residual of the sparse solve
          ``notes``                   list of plain-language caveats that apply
                                      to THIS result; never empty-by-omission

    Raises:
        UnknownEntityError, IsolatedEntityError, EmptyGraphError, MechanismError:
            as in ``personalised_pagerank``. Also ``MechanismError`` when
            ``src_id == dst_id`` (self-connectivity is the restart mass, not a
            finding) or when the graph carries no ``.etype``.

    Where it misleads:
        Everything in the module docstring applies, and specifically: this
        function says nothing about WHY the two are connected, how many
        independent dated sources support it, or whether the route is static
        geography. It is the structural half of an explanation only. Rank is
        also asymmetric — ``connectivity(g, a, b)`` and ``connectivity(g, b, a)``
        differ, because each ranks inside a different walk.
    """
    alpha = _check_alpha(alpha)
    A = _adjacency(g, edges)
    src_row = _row(g, src_id)
    dst_row = _row(g, dst_id)
    if src_row == dst_row:
        raise MechanismError(
            f"src_id and dst_id are the same entity ({src_id!r}); its PageRank "
            f"mass is just the restart mass and is not a connectivity finding"
        )

    n = A.shape[0]
    types = _type_vector(g, n)
    x, info = _ppr(A, src_row, alpha)

    mass = float(x[dst_row])
    connected = mass > _MASS_EPS
    notes: list[str] = []

    global_rank = int((x > mass).sum()) + 1
    global_ties = int((x == mass).sum()) - 1
    reachable = int((x > _MASS_EPS).sum())
    percentile = 100.0 * (1.0 - global_rank / n)

    dst_type = _lookup(g, _TYPE_ATTRS, dst_id, dst_row)
    if not dst_type:
        raise MechanismError(
            f"entity {dst_id!r} has no entity_type in this graph; the same-type "
            f"rank is the interpretable half of this result and cannot be omitted"
        )
    peer_rows = np.flatnonzero(types == dst_type)
    if peer_rows.size == 0:  # pragma: no cover — _type_vector guarantees a hit
        raise MechanismError(f"no entities of type {dst_type!r} found in .etype")
    peer_mass = x[peer_rows]
    type_rank = int((peer_mass > mass).sum()) + 1
    type_ties = int((peer_mass == mass).sum()) - 1
    type_median = float(np.median(peer_mass))

    if type_median > 0.0:
        ratio_to_type_median: float | None = mass / type_median
    else:
        ratio_to_type_median = None
        notes.append(
            f"the median {dst_type} has PageRank mass 0.0 (more than half of the "
            f"{peer_rows.size} {dst_type} entities are unreachable from {src_id}), "
            f"so ratio_to_type_median is undefined rather than infinite"
        )

    if not connected:
        notes.append(
            f"mass is {mass:.3e} (numerical zero): {dst_id} is NOT reachable from "
            f"{src_id} in this graph. This is a computed 'no', not a failure. Both "
            f"global_rank and type_rank are therefore degenerate — they only say "
            f"'behind everything reachable', tied with {type_ties} other "
            f"{dst_type} entities, and must NOT be quoted as '#{type_rank} of "
            f"{peer_rows.size}'."
        )
    if global_ties > 0:
        notes.append(
            f"{global_ties} other entities share this exact mass; global_rank is a "
            f"competition rank (ties share the better rank)"
        )
    if info["isolated_count"]:
        notes.append(
            f"{info['isolated_count']} of {n} entities have degree 0 and can never "
            f"rank; percentile {percentile:.2f}% is inflated by them"
        )
    if info["multi_edge_weights_present"]:
        notes.append(
            "adjacency has edge weights > 1 (repeated entity pairs under different "
            "link_types were summed); ranks are NOT comparable to the reference "
            "figures, which were measured on the simple unweighted graph"
        )
    if info["teleport_mass_lost"] > 1e-9:
        notes.append(
            f"{info['teleport_mass_lost']:.3e} of the walk's mass was lost on "
            f"dangling rows (directed adjacency); masses are sub-stochastic"
        )
    notes.append(
        f"alpha={alpha:g}: mean walk length 1/alpha = {1.0 / alpha:.1f} hops — this "
        f"is what 'a long connection' means here, and changing it changes the ranks"
    )
    notes.append(
        "PageRank mass measures ROUTING through the graph, including static "
        "geography edges that carry no time-varying information; it is not "
        "evidence and counts no independent sources"
    )

    return {
        "src_id": src_id,
        "src_name": _lookup(g, _NAME_ATTRS, src_id, src_row),
        "dst_id": dst_id,
        "dst_name": _lookup(g, _NAME_ATTRS, dst_id, dst_row),
        "dst_type": dst_type,
        "alpha": alpha,
        "mass": mass,
        "connected": connected,
        "global_rank": global_rank,
        "n_entities": n,
        "percentile": percentile,
        "global_ties": global_ties,
        "reachable_count": reachable,
        "isolated_count": info["isolated_count"],
        "type_rank": type_rank,
        "type_ties": type_ties,
        "type_count": int(peer_rows.size),
        "type_median": type_median,
        "ratio_to_type_median": ratio_to_type_median,
        "teleport_mass_lost": info["teleport_mass_lost"],
        "multi_edge_weights_present": info["multi_edge_weights_present"],
        "solve_residual": info["solve_residual"],
        "notes": notes,
    }


# ── Effective resistance ────────────────────────────────────────────────


def effective_resistance(g: Any, src_id: str, dst_id: str, *, edges: str = "all") -> float:
    """Resistance distance between two entities: how MANY routes, not how short.

    Treats each edge weight as a conductance and returns
    ``(e_src - e_dst)ᵀ L⁺ (e_src - e_dst)`` for the graph Laplacian ``L``.
    Computed by a sparse grounded-Laplacian solve restricted to the connected
    component containing both entities — never a dense pseudo-inverse, which
    would be a 20,026 x 20,026 matrix.

    Interpretation: adding a parallel route always LOWERS resistance, while
    shortest-path distance ignores it. Two entities joined by one long chain are
    far apart here; two joined by many independent chains are close. That is the
    complement to PageRank rank: rank says "how prominent", resistance says
    "how redundant".

    Args:
        g: object satisfying the graph contract in the module docstring.
        src_id, dst_id: the two entities. Symmetric in them.
        edges: ``"all"`` (default) or ``"evidence"``; see ``personalised_pagerank``.
            Redundancy counted over ``"all"`` can be entirely static geography.

    Returns:
        Non-negative float. 0.0 only for identical entities, which this function
        refuses rather than returns.

    Raises:
        UnknownEntityError: either id is absent from the graph.
        IsolatedEntityError: either entity has degree 0.
        DisconnectedEntityError: the two lie in different components, where the
            resistance is infinite. Raised rather than returned as ``inf`` or,
            worse, as a small number from a mis-grounded solve.
        EmptyGraphError / MechanismError: contract violations, or a solve whose
            residual exceeds tolerance.

    Where it misleads:
        1. It is blind to edge semantics: 12 parallel static-geography edges
           look exactly as redundant as 12 dated event edges, so a low
           resistance can be entirely scaffold. Independence has to be counted
           over time-varying edges elsewhere.
        2. Scale is graph-dependent — compare resistances only within one graph
           (same ``as_of``, same weighting), never across two.
        3. Weights act as conductances, so a ``confidence``-weighted graph and
           an unweighted one give different numbers for the same topology.
        4. It is a whole-component property: adding an unrelated edge far away
           can change it slightly, so small differences are not meaningful.
    """
    A = _adjacency(g, edges)
    src_row = _row(g, src_id)
    dst_row = _row(g, dst_id)
    if src_row == dst_row:
        raise MechanismError(
            f"src_id and dst_id are the same entity ({src_id!r}); its resistance "
            f"to itself is trivially 0.0 and is not a finding"
        )

    degree = np.asarray(A.sum(axis=1)).ravel()
    for entity_id, row in ((src_id, src_row), (dst_id, dst_row)):
        if degree[row] <= 0:
            raise IsolatedEntityError(
                f"entity {entity_id!r} has degree 0 in this graph; resistance to it "
                f"is infinite for want of any edge, not merely large"
            )

    n_components, labels = connected_components(A, directed=False, connection="weak")
    if labels[src_row] != labels[dst_row]:
        raise DisconnectedEntityError(
            f"{src_id!r} and {dst_id!r} are in different connected components "
            f"(of {n_components} in this graph), so their effective resistance is "
            f"infinite. Refusing to return a finite number for an infinite value."
        )

    component = np.flatnonzero(labels == labels[src_row])
    sub = sp.csr_matrix(A[component][:, component])
    sub_degree = np.asarray(sub.sum(axis=1)).ravel()
    laplacian = (sp.diags(sub_degree) - sub).tocsc()

    local = {int(global_row): k for k, global_row in enumerate(component)}
    s_local, d_local = local[src_row], local[dst_row]

    # L is singular (constant null space); ground one node at 0 volts and solve
    # for the potentials induced by injecting +1A at src and -1A at dst.
    size = component.size
    ground = 0 if s_local != 0 else 1
    keep = np.array([k for k in range(size) if k != ground])
    current = np.zeros(size, dtype=float)
    current[s_local] = 1.0
    current[d_local] = -1.0

    reduced = laplacian[keep][:, keep].tocsc()
    solution = spla.spsolve(reduced, current[keep])
    solution = np.asarray(solution, dtype=float).ravel()

    potentials = np.zeros(size, dtype=float)
    potentials[keep] = solution

    residual = float(np.abs(laplacian @ potentials - current).max())
    if not np.all(np.isfinite(potentials)) or residual > _RESIDUAL_TOL:
        raise MechanismError(
            f"Laplacian solve for effective resistance did not reach tolerance: "
            f"max residual {residual:.3e} > {_RESIDUAL_TOL:.0e} over a component of "
            f"{size} entities. Refusing to return an unverified resistance."
        )

    resistance = float(potentials[s_local] - potentials[d_local])
    if resistance < 0.0:
        if resistance > -_RESIDUAL_TOL:
            resistance = 0.0
        else:
            raise MechanismError(
                f"computed a negative effective resistance ({resistance:.3e}); the "
                f"adjacency cannot be a valid undirected weighted graph"
            )
    return resistance
