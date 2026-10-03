"""
TirraMind — Mechanism Layer: route extraction and source independence (Layer 3).

WHAT THIS COMPUTES
    ``top_routes``    the k highest-mass *simple paths* between two entities of
        the evidence graph, each path annotated hop by hop with its link_type,
        its ingest ``source``, whether that hop is EVIDENCE or SCAFFOLD, and the
        degree of the nodes it passes through.
    ``independence``  the number of INDEPENDENT TIME-VARYING sources standing
        behind those paths. This is the commercially load-bearing number: it is
        the difference between "well confirmed by five datasets" and "one
        dataset wearing five hats".

WHY INDEPENDENCE IS COUNTED OVER EVENT EDGES ONLY
    ``agent/models/gnn/graph_builder.py`` splits link types into
    STRUCTURAL_RELATIONS and EVENT_RELATIONS. Structural links are static
    geography: Canada produced oil in 1990 and produces it today, so
    ``produced_in`` carries no time-varying information at all. It *routes*
    evidence; it is not evidence. Measured on the live graph, every short route
    Russia -> WTI Crude Oil has the shape

        Russia --event_involves(gdelt)--> {country} --produced_in(seed_producer_links)--> WTI

    Naive counting of every ``source`` on every hop of the returned routes
    reports several distinct datasets. Only ONE of them varies with the date, and
    every one of those hops is gdelt. So the honest count is 1. Reporting the
    naive number would inflate confidence, which is the exact error this product
    exists to catch, so ``independence`` reports the honest count as
    ``independent_sources`` and the inflated one only as ``naive_source_count``,
    explicitly labelled.

WHY PATHS ARE ENUMERATED BY RANDOM-WALK MASS AND NOT BY BFS
    Naive BFS over this graph does not terminate usefully: it ran for over two
    minutes and returned the same hub route four times, calling it four
    confirmations. ``top_routes`` instead runs a best-first (Dijkstra) search on
    the path's personalised-PageRank contribution

        mass(path) = alpha * prod_hops (1 - alpha) / degree(node left)

    which is monotonically decreasing along a path, so the first k completed
    paths popped are the exact top k by mass — and a walker crossing a
    522-degree node splits its mass 522 ways, so hub routes are discounted by
    construction rather than counted repeatedly. No neural network, no fitted
    parameters, nothing that can collapse.

WHERE THIS MISLEADS (read before quoting any of it to a customer)
    1. ``mass`` is a LOWER BOUND on the personalised-PageRank mass that
       ``agent/mechanism/connectivity.py`` reports for the same pair: non-simple
       walks (those that revisit a node) also carry mass and are deliberately
       not enumerated here. Route masses therefore do not sum to the PPR mass,
       and comparing a route mass across two different graphs or two alphas is
       meaningless. Only the ordering, and the ratio between routes of the same
       query, mean anything.
    2. ``independent_sources`` counts DATASETS, not observations. Two gdelt
       events a decade apart are one source. It answers "how many ways could I
       be wrong at once", not "how much evidence is there".
    3. A source label is an ingest pipeline, not a publisher. If two collectors
       were pointed at the same upstream feed, this function would call them two
       independent sources and would be wrong. ``evidence_channels`` is returned
       so the (link_type, source) pairs can be eyeballed for that.
    4. Time honesty is the caller's job. ``load_route_graph(as_of=...)`` filters
       links by ``effective_from`` (falling back to ``created_at`` when it is
       NULL, which is a *backfilled ingest* date, not necessarily when the
       relation became true). Explaining a 2019 pattern with the 2026 graph is
       leakage (LESSONS F-04); a graph built with no ``as_of`` cannot detect
       that mistake for you, and every result carries the ``as_of`` it used.
    5. Degree — and therefore mass and the hub flag — is computed on the SIMPLE
       (deduplicated, undirected, unweighted) graph, so that these numbers are
       comparable with ``connectivity.py``'s. Link ``confidence`` and duplicate
       rows are reported per hop but do NOT move the mass.
    6. Nothing here is truncated silently. Every cap, prune and collapse is
       counted and returned on ``RouteList.search``; ``independence`` copies the
       relevant ones into its ``warnings``. If you read only
       ``independent_sources`` you may be reading a number computed from a
       partial route set — the dict tells you when.

DEGENERATE INPUT IS AN EXCEPTION, NOT A ZERO
    An unknown entity id, an isolated entity or an empty graph raises. "No route
    exists" returns an empty ``RouteList`` whose ``search.connected`` is False
    and whose ``independence`` status is ``"no_route"`` — never the same value
    as "one source". ``independence`` on a bare ``[]`` (which cannot say which
    of the two happened) raises rather than guess.

GRAPH CONTRACT
    ``top_routes`` needs per-edge link_type and source, which an adjacency
    matrix does not carry. It accepts, in order:
      * a ``RouteGraph`` (built here by ``load_route_graph`` / ``from_links``);
      * ``agent/mechanism/graph.py``'s ``MechanismGraph`` — recognised by
        ``entity_ids`` + row-indexed ``edges``, with its own ``as_of`` and
        filtering counts carried through (its ``is_evidence`` flag is ignored in
        favour of this module's ``link_kind``, so one definition governs);
      * any object exposing an iterable of link records as ``.links``,
        ``.edges``, ``.link_rows`` or ``.edge_records`` (dicts or sequences with
        entity_id_a / entity_id_b / link_type / source [, confidence,
        effective_from, created_at]), optionally with ``.name`` / ``.etype``
        maps — this is how ``agent/mechanism/graph.py``'s ``MechanismGraph``
        plugs in;
      * an ``sqlite3.Connection`` opened read-only, or a path to the DB.
    A graph that exposes only ``.A``/``.idx`` raises
    ``RouteMissingEdgeMetadataError`` rather than inventing edge labels.

    The exception types here are deliberately local (``Route*``) rather than
    shared with ``connectivity.py``: the two modules are written independently
    and a caller must not accidentally catch one and silence the other.

Layer: 3 (world model / structural beliefs). Stateless. Opens the live DB
READ-ONLY (``mode=ro``) and never writes to it.
"""

from __future__ import annotations

import heapq
import logging
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "EVENT_RELATIONS",
    "STRUCTURAL_RELATIONS",
    "Hop",
    "Route",
    "RouteEmptyGraphError",
    "RouteError",
    "RouteGraph",
    "RouteIndeterminateError",
    "RouteIsolatedEntityError",
    "RouteList",
    "RouteMissingEdgeMetadataError",
    "RouteSearch",
    "RouteUnknownEntityError",
    "describe_routes",
    "independence",
    "link_kind",
    "load_route_graph",
    "top_routes",
]

HopKind = Literal["evidence", "scaffold", "unclassified"]

# ── Relation classification ────────────────────────────────────────────────
#
# Copied verbatim from agent/models/gnn/graph_builder.py, which is the source of
# truth. It is copied rather than imported because that module pulls in torch and
# torch_geometric, which this layer must not depend on (the whole point of the
# mechanism layer is that it is linear algebra, not deep learning).
# tests/test_mechanism_routes.py asserts the two copies have not drifted.

STRUCTURAL_RELATIONS: frozenset[str] = frozenset(
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

EVENT_RELATIONS: frozenset[str] = frozenset(
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

DEFAULT_DB_PATH = Path(".tirra_pipeline/pipeline.db")

# Default expansion budget for the best-first search. Measured: Russia -> WTI
# Crude Oil at max_hops=4, k=10 finishes in far fewer pops than this; the budget
# exists so a pathological pair degrades LOUDLY (search.budget_exhausted) rather
# than hanging the way the BFS it replaces did.
_DEFAULT_MAX_EXPANSIONS = 200_000

_INF = float("inf")


# ── Errors ────────────────────────────────────────────────────────────────


class RouteError(ValueError):
    """Base for every failure of this module. Never raised directly."""


class RouteUnknownEntityError(RouteError):
    """An entity id is not in the graph at all (typo, or wrong as_of graph)."""


class RouteEmptyGraphError(RouteError):
    """The graph has no links, so "no route" would be uninformative."""


class RouteIsolatedEntityError(RouteError):
    """The entity exists but has degree 0 — nothing to route through."""


class RouteMissingEdgeMetadataError(RouteError):
    """The graph carries connectivity but no link_type/source per edge."""


class RouteIndeterminateError(RouteError):
    """independence() was handed an empty route set with no search provenance."""


# ── Hop / Route / search provenance ───────────────────────────────────────


def link_kind(link_type: str) -> HopKind:
    """Classify one link_type as time-varying evidence or static scaffold.

    Returns ``"evidence"`` for EVENT_RELATIONS, ``"scaffold"`` for
    STRUCTURAL_RELATIONS, and ``"unclassified"`` for anything else.

    Misleads: ``"unclassified"`` is not a middle ground. It means a link_type
    was added to the collectors and nobody decided whether it witnesses
    anything. ``independence`` therefore excludes it from the evidence count AND
    reports it loudly, so an unclassified relation can never quietly inflate or
    deflate a verdict.
    """
    if link_type in EVENT_RELATIONS:
        return "evidence"
    if link_type in STRUCTURAL_RELATIONS:
        return "scaffold"
    return "unclassified"


@dataclass(frozen=True)
class Hop:
    """One traversed link, in the direction the route walks it.

    ``records`` is how many DB rows collapsed into this hop: the live table
    stores some relations in both directions, so 2 means "the same relation
    twice", not "two observations". ``effective_from`` is the link's effective
    date (``effective_from``, else ``created_at``) as a POSIX timestamp, or None
    when neither was present — a None here means the hop's date is UNKNOWN, not
    that it is old.
    """

    src: str
    dst: str
    link_type: str
    source: str
    kind: HopKind
    confidence: float
    effective_from: float | None
    records: int = 1


@dataclass(frozen=True)
class Route:
    """One simple path, with everything needed to judge whether to believe it.

    ``mass`` is the path's personalised-PageRank contribution (see the module
    docstring): comparable within one query only, and a lower bound on the pair's
    total PPR mass because non-simple walks are not enumerated.

    Misleads: a high-mass route is a *short, low-degree* route. That is not the
    same as a well-evidenced one — a two-hop path of pure geography outranks a
    three-hop path carrying two real events. Read ``kinds`` before ``mass``.
    """

    nodes: tuple[str, ...]
    node_names: tuple[str, ...]
    link_types: tuple[str, ...]
    sources: tuple[str, ...]
    kinds: tuple[HopKind, ...]
    confidences: tuple[float, ...]
    effective_from: tuple[float | None, ...]
    hops_detail: tuple[Hop, ...]
    mass: float
    intermediate_degrees: tuple[int, ...]
    max_intermediate_degree: int
    hub_nodes: tuple[str, ...]
    hub_dominated: bool
    hub_degree_threshold: float

    @property
    def hops(self) -> int:
        """Number of links traversed."""
        return len(self.link_types)

    @property
    def evidence_sources(self) -> frozenset[str]:
        """Sources on time-varying hops only — what may be counted as evidence."""
        return frozenset(h.source for h in self.hops_detail if h.kind == "evidence")

    @property
    def scaffold_sources(self) -> frozenset[str]:
        """Sources on static-geography hops. NOT evidence. Never counted."""
        return frozenset(h.source for h in self.hops_detail if h.kind == "scaffold")

    @property
    def unclassified_sources(self) -> frozenset[str]:
        """Sources on hops whose link_type nobody has classified yet."""
        return frozenset(h.source for h in self.hops_detail if h.kind == "unclassified")

    @property
    def has_evidence(self) -> bool:
        """True when at least one hop varies with the date."""
        return any(h.kind == "evidence" for h in self.hops_detail)

    @property
    def latest_effective_from(self) -> float | None:
        """Newest hop date on the route, or None if no hop is dated.

        Misleads: this is the date at which the whole route first existed only if
        every hop is dated. Check ``effective_from`` for Nones before treating it
        as an as-of date.
        """
        dated = [t for t in self.effective_from if t is not None]
        return max(dated) if dated else None

    def describe(self) -> str:
        """One-line human rendering: names, link types, sources and hop kinds."""
        parts = [self.node_names[0]]
        for i, hop in enumerate(self.hops_detail):
            tag = {"evidence": "EVID", "scaffold": "SCAF", "unclassified": "????"}[hop.kind]
            parts.append(f"-[{hop.link_type}/{hop.source}/{tag}]->")
            parts.append(self.node_names[i + 1])
        flag = f"  HUB(deg {self.max_intermediate_degree})" if self.hub_dominated else ""
        return f"mass={self.mass:.3e}  " + " ".join(parts) + flag


@dataclass
class RouteSearch:
    """Everything the search did, including everything it did NOT do.

    This object exists because a capped result that looks complete is this
    repo's single most repeated bug (F-16, F-17). ``warnings()`` renders the
    fields that mean "the list you are holding is partial".

    Field notes:
      ``more_routes_exist``  exact, not a guess: every path left on the search
          queue is guaranteed completable (see ``_bfs_hops``), so a non-empty
          queue is a proof that a further route exists.
      ``exhaustive``  True only when the queue emptied, i.e. the returned list
          is every route within ``max_hops``.
      ``duplicate_node_sequences_collapsed``  paths identical in BOTH node
          sequence and link_type sequence, emitted once. Two paths over the SAME
          nodes via DIFFERENT link types are kept as separate routes on purpose
          (different relations are different evidence) — ``independence``
          reports how many routes share a node sequence so that cannot be read
          as independent confirmation.
      ``links_excluded_by_as_of`` / ``links_with_undated_fallback``  properties
          of the GRAPH, repeated here so a verdict dict carries its own time
          provenance (F-04).
    """

    src: str
    dst: str
    max_hops: int
    k: int
    alpha: float
    connected: bool
    shortest_hops: int | None
    routes_returned: int
    more_routes_exist: bool
    complete_paths_popped: int
    duplicate_node_sequences_collapsed: int
    expansions: int
    max_expansions: int
    budget_exhausted: bool
    exhaustive: bool
    hub_degree_threshold: float
    hub_degree_percentile: float
    graph_nodes: int
    graph_links: int
    as_of: float | None
    links_excluded_by_as_of: int
    links_with_undated_fallback: int
    reason: str | None = None

    def warnings(self) -> list[str]:
        """Human-readable reasons not to trust this route set as complete."""
        out: list[str] = []
        if self.budget_exhausted:
            out.append(
                f"SEARCH BUDGET EXHAUSTED after {self.expansions} expansions "
                f"(max_expansions={self.max_expansions}): the returned routes are "
                "NOT guaranteed to be the top-k; raise max_expansions or lower "
                "max_hops before quoting them."
            )
        if self.more_routes_exist:
            out.append(
                f"TRUNCATED to k={self.k}: at least one further route exists and "
                "was not returned. Any count derived from these routes is a count "
                "over the top-k, not over the graph."
            )
        if self.links_excluded_by_as_of:
            out.append(
                f"as_of={self.as_of!r} excluded {self.links_excluded_by_as_of} links "
                "from the graph: routes that exist today may be absent here (this is "
                "the point — F-04)."
            )
        if self.links_with_undated_fallback:
            out.append(
                f"{self.links_with_undated_fallback} links had no effective_from and "
                "were dated by created_at (ingest time, not event time), so as_of "
                "filtering on them is approximate."
            )
        if self.reason:
            out.append(self.reason)
        return out


class RouteList(list):
    """A ``list[Route]`` that carries its own search provenance.

    It IS a list, so ``top_routes`` honours its documented return type, but it
    also answers "is this all of them?" via ``.search``. Slicing or copying it
    yields a plain list and LOSES the provenance — that is deliberate: a route
    set stripped of its caveats should not still look authoritative.
    """

    def __init__(self, routes: Iterable[Route], search: RouteSearch) -> None:
        super().__init__(routes)
        self.search = search


# ── Graph ─────────────────────────────────────────────────────────────────


class _Edge(NamedTuple):
    nbr: str
    link_type: str
    source: str
    confidence: float
    effective_from: float | None
    records: int


@dataclass
class RouteGraph:
    """Undirected multigraph with per-edge link_type and source.

    Parallel links of DIFFERENT link_type between the same pair are kept
    separately (they are different evidence). The same (pair, link_type) stored
    twice — the live table holds some relations in both directions — is collapsed
    into one edge with ``records`` > 1 and the LATEST of the two effective dates,
    because the later date is the conservative one for ``as_of`` filtering.

    ``degree`` is the SIMPLE degree (count of distinct neighbours), matching the
    convention in ``connectivity.py`` so masses and hub thresholds are
    comparable. Misleads: a node with 40 parallel links to one neighbour has
    degree 1 here.

    ``known_nodes`` holds every id seen BEFORE the ``as_of`` filter (plus every
    id in ``name``/``etype``), so an entity whose links were all filtered away by
    the cutoff can be reported as isolated-as-of rather than as a typo. Without
    it, a correct id and a wrong one produce the same error, which is how a
    leakage bug gets misread as a bad lookup.
    """

    adj: dict[str, tuple[_Edge, ...]]
    name: dict[str, str] = field(default_factory=dict)
    etype: dict[str, str] = field(default_factory=dict)
    degree: dict[str, int] = field(default_factory=dict)
    known_nodes: set[str] = field(default_factory=set)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        """Nodes with at least one link. Isolated entities are not held here."""
        return len(self.adj)

    @property
    def n_links(self) -> int:
        """Distinct undirected (pair, link_type) edges."""
        return int(self.stats.get("edges", 0))

    def display(self, entity_id: str) -> str:
        """Canonical name if known, else the raw id."""
        return self.name.get(entity_id, entity_id)

    def degree_percentile(self, pct: float) -> float:
        """Degree at the given percentile over nodes with degree > 0.

        Misleads: this distribution is extremely skewed (measured on the live
        graph: median 1, p99 26, max 2010), so a percentile threshold is a
        pragmatic hub cutoff, not a statistically meaningful one.
        """
        if not self.degree:
            raise RouteEmptyGraphError("degree_percentile on a graph with no links")
        degs = np.fromiter((d for d in self.degree.values() if d > 0), dtype=np.int64)
        if degs.size == 0:
            raise RouteEmptyGraphError("degree_percentile on a graph with no links")
        return float(np.percentile(degs, pct))

    @classmethod
    def from_links(
        cls,
        links: Iterable[Mapping[str, Any] | Sequence[Any]],
        *,
        names: Mapping[str, str] | None = None,
        etypes: Mapping[str, str] | None = None,
        as_of: Any = None,
    ) -> RouteGraph:
        """Build from link records (dicts, or 4..7-element sequences).

        Recognised keys/positions: entity_id_a, entity_id_b, link_type, source,
        confidence, effective_from, created_at. A record whose effective date is
        after ``as_of`` is dropped and counted in ``stats``; self-links are
        dropped and counted. Raises on a record missing a required field rather
        than defaulting the source to "unknown", because an unattributed hop
        would silently become an independent dataset.
        """
        cutoff = _normalise_as_of(as_of)
        merged: dict[tuple[str, str, str], _Edge] = {}
        known: set[str] = set(names or ()) | set(etypes or ())
        excluded_as_of = 0
        undated_fallback = 0
        self_links = 0
        n_in = 0
        for raw in links:
            n_in += 1
            rec = _as_link_record(raw)
            a, b = rec["entity_id_a"], rec["entity_id_b"]
            known.add(a)
            known.add(b)
            if a == b:
                self_links += 1
                continue
            eff = rec["effective_from"]
            if eff is None:
                eff = rec["created_at"]
                if eff is not None:
                    undated_fallback += 1
            if cutoff is not None and (eff is None or eff > cutoff):
                excluded_as_of += 1
                continue
            key = (a, b, rec["link_type"]) if a <= b else (b, a, rec["link_type"])
            prev = merged.get(key)
            if prev is None:
                merged[key] = _Edge(
                    nbr="",  # filled when the adjacency is materialised
                    link_type=rec["link_type"],
                    source=rec["source"],
                    confidence=rec["confidence"],
                    effective_from=eff,
                    records=1,
                )
            else:
                keep_eff = prev.effective_from
                if eff is not None and (keep_eff is None or eff > keep_eff):
                    keep_eff = eff
                merged[key] = prev._replace(
                    confidence=max(prev.confidence, rec["confidence"]),
                    effective_from=keep_eff,
                    records=prev.records + 1,
                )

        adj: dict[str, list[_Edge]] = {}
        nbrs: dict[str, set[str]] = {}
        for (u, v, _lt), edge in merged.items():
            adj.setdefault(u, []).append(edge._replace(nbr=v))
            adj.setdefault(v, []).append(edge._replace(nbr=u))
            nbrs.setdefault(u, set()).add(v)
            nbrs.setdefault(v, set()).add(u)

        return cls(
            adj={u: tuple(es) for u, es in adj.items()},
            name=dict(names or {}),
            etype=dict(etypes or {}),
            degree={u: len(s) for u, s in nbrs.items()},
            known_nodes=known,
            stats={
                "records_in": n_in,
                "edges": len(merged),
                "self_links_dropped": self_links,
                "links_excluded_by_as_of": excluded_as_of,
                "links_with_undated_fallback": undated_fallback,
                "as_of": cutoff,
            },
        )


def load_route_graph(db_path: str | Path = DEFAULT_DB_PATH, *, as_of: Any = None) -> RouteGraph:
    """Load the entity graph from the pipeline DB, READ-ONLY, optionally as of a date.

    ``as_of`` accepts a POSIX timestamp, a ``date``/``datetime`` (naive is read
    as UTC) or an ISO ``"YYYY-MM-DD"`` string, and keeps only links whose
    effective date is at or before it.

    The connection is opened with ``file:...?mode=ro`` and ``uri=True``: this
    module must never be able to mutate the only irreplaceable asset in the
    project. Misleads: a link with NULL ``effective_from`` is dated by
    ``created_at``, an INGEST time — on the live graph 1,829 of 30,131 links are
    in that state, so a historical ``as_of`` is approximate for those and the
    count is returned in ``stats``/``search``.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"pipeline DB not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        names: dict[str, str] = {}
        etypes: dict[str, str] = {}
        for row in conn.execute("SELECT entity_id, entity_type, canonical_name FROM entities"):
            names[row["entity_id"]] = row["canonical_name"]
            etypes[row["entity_id"]] = row["entity_type"]
        links = [
            dict(row)
            for row in conn.execute(
                "SELECT entity_id_a, entity_id_b, link_type, source, confidence, "
                "effective_from, created_at FROM entity_links"
            )
        ]
    finally:
        conn.close()
    graph = RouteGraph.from_links(links, names=names, etypes=etypes, as_of=as_of)
    graph.stats["db_path"] = str(path)
    graph.stats["entities_in_db"] = len(names)
    return graph


def _normalise_as_of(value: Any) -> float | None:
    """Coerce an as_of to a POSIX timestamp. Rejects anything ambiguous."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise RouteError(f"as_of must be a date or timestamp, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=UTC)
        return dt.timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp()
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError as exc:
            raise RouteError(f"as_of string must be ISO (YYYY-MM-DD[THH:MM:SS]), got {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    raise RouteError(f"unsupported as_of type {type(value).__name__}")


_REQUIRED_LINK_FIELDS = ("entity_id_a", "entity_id_b", "link_type", "source")
_LINK_POSITIONS = (
    "entity_id_a",
    "entity_id_b",
    "link_type",
    "source",
    "confidence",
    "effective_from",
    "created_at",
)


def _as_link_record(raw: Any) -> dict[str, Any]:
    """Normalise one link record; raise loudly on a missing required field."""
    if isinstance(raw, Mapping):
        rec = dict(raw)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if not 4 <= len(raw) <= len(_LINK_POSITIONS):
            raise RouteError(f"link record sequences must have 4..7 elements {_LINK_POSITIONS}, got {len(raw)}")
        rec = dict(zip(_LINK_POSITIONS, raw))
    else:
        raise RouteError(f"unsupported link record type {type(raw).__name__}")
    missing = [f for f in _REQUIRED_LINK_FIELDS if not rec.get(f)]
    if missing:
        raise RouteError(
            f"link record missing required field(s) {missing}: {raw!r}. "
            "An unattributed hop would be counted as an independent dataset, so "
            "this is not defaulted."
        )
    conf = rec.get("confidence")
    rec["confidence"] = 1.0 if conf is None else float(conf)
    for key in ("effective_from", "created_at"):
        val = rec.get(key)
        rec[key] = None if val is None else float(val)
    return rec


_LINK_ATTRS = ("links", "edges", "link_rows", "edge_records")


def _as_route_graph(g: Any) -> RouteGraph:
    """Adapt whatever the caller passed into a RouteGraph, or raise saying why."""
    if isinstance(g, RouteGraph):
        return g
    if isinstance(g, (str, Path)):
        return load_route_graph(g)
    if isinstance(g, sqlite3.Connection):
        links = [
            dict(zip(_LINK_POSITIONS, row))
            for row in g.execute(
                "SELECT entity_id_a, entity_id_b, link_type, source, confidence, "
                "effective_from, created_at FROM entity_links"
            )
        ]
        names = {row[0]: row[1] for row in g.execute("SELECT entity_id, canonical_name FROM entities")}
        etypes = {row[0]: row[1] for row in g.execute("SELECT entity_id, entity_type FROM entities")}
        return RouteGraph.from_links(links, names=names, etypes=etypes)
    indexed = _indexed_edge_graph(g)
    if indexed is not None:
        return indexed
    for attr in _LINK_ATTRS:
        links = getattr(g, attr, None)
        if links is None:
            continue
        names = _as_id_map(g, "name")
        etypes = _as_id_map(g, "etype")
        return RouteGraph.from_links(links, names=names, etypes=etypes)
    raise RouteMissingEdgeMetadataError(
        f"{type(g).__name__} carries no per-edge link_type/source. Route "
        "extraction cannot be done on an adjacency matrix alone: an edge with no "
        "link_type cannot be classified evidence vs scaffold, and an edge with no "
        "source cannot be counted for independence. Pass a RouteGraph "
        "(load_route_graph(...)), the DB path, or an object exposing one of "
        f"{_LINK_ATTRS}."
    )


def _indexed_edge_graph(g: Any) -> RouteGraph | None:
    """Adapt a graph whose edges carry ROW INDICES, as MechanismGraph's do.

    ``agent/mechanism/graph.py``'s ``MechanismGraph`` exposes ``entity_ids`` plus
    ``edges`` of ``MechanismEdge(a: int, b: int, link_type, source, confidence,
    effective_at, is_evidence)``. Returns None when ``g`` is not that shape, so
    the caller can go on trying other contracts.

    Misleads: that graph has ALREADY applied its own ``as_of`` and its own
    scaffold time policy, so ``effective_at`` is a resolved date, not the raw
    column. Its filtering counts are copied into ``stats`` here rather than
    recomputed — this module re-derives nothing and claims no second opinion on
    the dates. ``is_evidence`` on those edges is likewise NOT trusted: the
    classification is recomputed from ``link_type`` by ``link_kind`` so that one
    definition of "evidence" governs the whole verdict.
    """
    ids = getattr(g, "entity_ids", None)
    edges = getattr(g, "edges", None)
    if ids is None or edges is None or isinstance(edges, Mapping):
        return None
    seq = tuple(edges)
    if seq and not (hasattr(seq[0], "link_type") and hasattr(seq[0], "a")):
        return None
    ids = tuple(ids)
    names = _paired(ids, getattr(g, "canonical_names", None))
    etypes = _paired(ids, getattr(g, "entity_types", None))
    links = [
        {
            "entity_id_a": ids[int(e.a)],
            "entity_id_b": ids[int(e.b)],
            "link_type": e.link_type,
            "source": e.source,
            "confidence": getattr(e, "confidence", 1.0),
            "effective_from": getattr(e, "effective_at", None),
            "created_at": getattr(e, "effective_at", None),
        }
        for e in seq
    ]
    graph = RouteGraph.from_links(links, names=names, etypes=etypes)
    graph.known_nodes |= set(ids)
    report = getattr(g, "report", None)
    graph.stats["as_of"] = getattr(g, "as_of", None)
    graph.stats["adapted_from"] = type(g).__name__
    if report is not None:
        graph.stats["links_excluded_by_as_of"] = int(getattr(report, "rows_dropped_by_as_of", 0))
        rows_read = int(getattr(report, "rows_read", 0))
        dated = int(getattr(report, "rows_with_effective_from", rows_read))
        graph.stats["links_with_undated_fallback"] = max(rows_read - dated, 0)
    return graph


def _paired(ids: tuple[str, ...], values: Any) -> dict[str, str]:
    """Zip an id tuple with an aligned tuple of labels; {} when misaligned."""
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return {str(k): str(v) for k, v in values.items()}
    seq = tuple(values)
    if len(seq) != len(ids):
        return {}
    return {i: str(v) for i, v in zip(ids, seq)}


def _as_id_map(g: Any, attr: str) -> dict[str, str]:
    """Read a name/etype map off a foreign graph object; {} when absent."""
    val = getattr(g, attr, None)
    if val is None:
        return {}
    if isinstance(val, Mapping):
        return {str(k): str(v) for k, v in val.items()}
    ents = getattr(g, "ents", None)
    if isinstance(val, Sequence) and ents is not None and len(val) == len(ents):
        return {str(e): str(v) for e, v in zip(ents, val)}
    return {}


# ── Search ────────────────────────────────────────────────────────────────


def _bfs_hops(graph: RouteGraph, start: str) -> dict[str, int]:
    """Unbounded BFS hop distance from ``start`` over the whole graph.

    Cheap (O(V+E); the live graph is 20k nodes / 26k edges) and exact, which is
    what makes the path search admissible: a partial path is only extended to a
    node that can still reach the destination inside the remaining hops, so
    every queued path is completable and "the queue is non-empty" is proof that
    a further route exists.
    """
    dist = {start: 0}
    frontier = [start]
    while frontier:
        nxt: list[str] = []
        for u in frontier:
            d = dist[u] + 1
            for edge in graph.adj.get(u, ()):
                if edge.nbr not in dist:
                    dist[edge.nbr] = d
                    nxt.append(edge.nbr)
        frontier = nxt
    return dist


def _require_node(graph: RouteGraph, entity_id: str, role: str) -> None:
    if not isinstance(entity_id, str) or not entity_id:
        raise RouteError(f"{role} entity id must be a non-empty string, got {entity_id!r}")
    if entity_id in graph.adj:
        return
    known = entity_id in graph.known_nodes or entity_id in graph.name or entity_id in graph.etype
    if known:
        as_of = graph.stats.get("as_of")
        cutoff_note = (
            f" The graph was built with as_of={as_of!r}, which removed "
            f"{graph.stats.get('links_excluded_by_as_of', 0)} links: this entity may "
            "have had links only after that date."
            if as_of is not None
            else ""
        )
        raise RouteIsolatedEntityError(
            f"{role} {entity_id!r} ({graph.display(entity_id)}) exists but has no "
            "links in this graph (degree 0). There is nothing to route through; "
            "this is not the same as 'no route between two connected entities'." + cutoff_note
        )
    raise RouteUnknownEntityError(
        f"{role} {entity_id!r} is not in this graph. Check the id, and check the "
        "as_of: an entity whose only links post-date the cutoff is absent here."
    )


def top_routes(
    g: Any,
    src_id: str,
    dst_id: str,
    *,
    max_hops: int = 4,
    k: int = 10,
    alpha: float = 0.15,
    hub_degree_percentile: float = 99.0,
    max_expansions: int = _DEFAULT_MAX_EXPANSIONS,
) -> RouteList:
    """The k most important simple paths from ``src_id`` to ``dst_id``.

    Importance is the path's personalised-PageRank contribution
    ``alpha * prod (1-alpha)/degree(node left)``, which decreases monotonically
    along a path, so a best-first search returns the EXACT top k (no heuristic,
    no sampling) whenever ``search.exhaustive`` is True. Hub routes are
    discounted automatically: a path through a 522-degree node divides its mass
    by 522.

    Each returned Route carries its node sequence, link_types, sources, per-hop
    EVIDENCE/SCAFFOLD classification, the degree of every intermediate node and
    a hub flag (any intermediate at or above the ``hub_degree_percentile`` degree
    of the graph).

    Raises rather than returning a plausible zero: ``RouteUnknownEntityError``
    for an id not in the graph, ``RouteIsolatedEntityError`` for degree 0,
    ``RouteEmptyGraphError`` for a graph with no links. Returns an EMPTY
    RouteList — with ``search.connected`` False and ``search.reason`` set — when
    the two entities genuinely have no path, or no path short enough.

    Where it misleads:
      * ``k`` truncates. ``search.more_routes_exist`` is exact (it is proven by
        a non-empty queue of completable paths), and ``search.warnings()`` says
        so in words. Any count taken over the returned routes — including
        ``independence`` — is a count over the top k, not over the graph.
      * ``max_hops`` truncates. A pair connected only by a 5-hop chain returns
        empty at the default 4, with the real shortest distance in
        ``search.shortest_hops`` so the zero is never ambiguous.
      * the top-k is a ranking by structure, not by evidence. A pure-scaffold
        two-hop route outranks an evidenced three-hop one. That is why
        ``independence`` exists and why it is not a function of ``mass``.
      * masses are a lower bound on the pair's PPR mass (simple paths only) and
        are not comparable across queries or graphs.
    """
    graph = _as_route_graph(g)
    if k < 1:
        raise RouteError(f"k must be >= 1, got {k}")
    if max_hops < 1:
        raise RouteError(f"max_hops must be >= 1, got {max_hops}")
    if not 0.0 < alpha < 1.0:
        raise RouteError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 <= hub_degree_percentile <= 100.0:
        raise RouteError(f"hub_degree_percentile must be in [0, 100], got {hub_degree_percentile}")
    if max_expansions < 1:
        raise RouteError(f"max_expansions must be >= 1, got {max_expansions}")
    if not graph.adj:
        raise RouteEmptyGraphError(
            "graph has no links at all — 'no route' would say nothing. Check the DB path and the as_of cutoff."
        )
    _require_node(graph, src_id, "src_id")
    _require_node(graph, dst_id, "dst_id")
    if src_id == dst_id:
        raise RouteError(
            f"src_id and dst_id are the same entity ({src_id!r}); a self-route has no mechanism to explain."
        )

    hub_threshold = graph.degree_percentile(hub_degree_percentile)
    stats = graph.stats

    def _search(**kw: Any) -> RouteSearch:
        base = {
            "src": src_id,
            "dst": dst_id,
            "max_hops": max_hops,
            "k": k,
            "alpha": alpha,
            "hub_degree_threshold": hub_threshold,
            "hub_degree_percentile": hub_degree_percentile,
            "graph_nodes": graph.n_nodes,
            "graph_links": graph.n_links,
            "as_of": stats.get("as_of"),
            "links_excluded_by_as_of": int(stats.get("links_excluded_by_as_of", 0)),
            "links_with_undated_fallback": int(stats.get("links_with_undated_fallback", 0)),
            "max_expansions": max_expansions,
        }
        base.update(kw)
        return RouteSearch(**base)  # type: ignore[arg-type]

    dist = _bfs_hops(graph, dst_id)
    if src_id not in dist:
        return RouteList(
            [],
            _search(
                connected=False,
                shortest_hops=None,
                routes_returned=0,
                more_routes_exist=False,
                complete_paths_popped=0,
                duplicate_node_sequences_collapsed=0,
                expansions=0,
                budget_exhausted=False,
                exhaustive=True,
                reason=(
                    f"NO ROUTE: {graph.display(src_id)} and {graph.display(dst_id)} "
                    "are in different connected components of this graph. This is a "
                    "proven absence of any path, not a failure to find one, and not "
                    "a zero-source verdict."
                ),
            ),
        )
    shortest = dist[src_id]
    if shortest > max_hops:
        return RouteList(
            [],
            _search(
                connected=True,
                shortest_hops=shortest,
                routes_returned=0,
                more_routes_exist=False,
                complete_paths_popped=0,
                duplicate_node_sequences_collapsed=0,
                expansions=0,
                budget_exhausted=False,
                exhaustive=True,
                reason=(
                    f"NO ROUTE WITHIN max_hops={max_hops}: the entities ARE connected, "
                    f"but the shortest path is {shortest} hops. Raise max_hops to see "
                    "it. Do not read this as 'unrelated'."
                ),
            ),
        )

    degree = graph.degree
    results: list[Route] = []
    emitted: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    collapsed = 0
    complete_popped = 0
    expansions = 0
    counter = 0
    # (-mass, tiebreak, nodes, hops) — popping by descending mass makes the first
    # k completed paths the exact top k, because mass only ever decreases.
    heap: list[tuple[float, int, tuple[str, ...], tuple[Hop, ...]]] = [(-alpha, 0, (src_id,), ())]
    while heap and len(results) < k and expansions < max_expansions:
        neg_mass, _, nodes, hops = heapq.heappop(heap)
        u = nodes[-1]
        if u == dst_id:
            complete_popped += 1
            key = (nodes, tuple(h.link_type for h in hops))
            if key in emitted:
                collapsed += 1
                continue
            emitted.add(key)
            results.append(_build_route(graph, nodes, hops, -neg_mass, hub_threshold))
            continue
        remaining = max_hops - (len(nodes) - 1)
        if remaining <= 0:
            continue
        expansions += 1
        step = (1.0 - alpha) / degree[u]
        for edge in graph.adj.get(u, ()):
            v = edge.nbr
            if v in nodes:
                continue
            if v != dst_id and dist.get(v, _INF) > remaining - 1:
                continue
            counter += 1
            hop = Hop(
                src=u,
                dst=v,
                link_type=edge.link_type,
                source=edge.source,
                kind=link_kind(edge.link_type),
                confidence=edge.confidence,
                effective_from=edge.effective_from,
                records=edge.records,
            )
            heapq.heappush(heap, (neg_mass * step, counter, (*nodes, v), (*hops, hop)))

    budget_exhausted = expansions >= max_expansions and bool(heap) and len(results) < k
    return RouteList(
        results,
        _search(
            connected=True,
            shortest_hops=shortest,
            routes_returned=len(results),
            more_routes_exist=bool(heap) and not budget_exhausted,
            complete_paths_popped=complete_popped,
            duplicate_node_sequences_collapsed=collapsed,
            expansions=expansions,
            budget_exhausted=budget_exhausted,
            exhaustive=not heap and not budget_exhausted,
            reason=(
                (
                    f"NO ROUTE RETURNED although a {shortest}-hop path exists — the "
                    "search budget ran out first. Treat this as 'could not compute', "
                    "not as 'not connected'."
                )
                if not results and budget_exhausted
                else None
            ),
        ),
    )


def _build_route(
    graph: RouteGraph,
    nodes: tuple[str, ...],
    hops: tuple[Hop, ...],
    mass: float,
    hub_threshold: float,
) -> Route:
    inter = nodes[1:-1]
    inter_deg = tuple(int(graph.degree.get(n, 0)) for n in inter)
    hub_nodes = tuple(n for n, d in zip(inter, inter_deg) if d >= hub_threshold)
    return Route(
        nodes=nodes,
        node_names=tuple(graph.display(n) for n in nodes),
        link_types=tuple(h.link_type for h in hops),
        sources=tuple(h.source for h in hops),
        kinds=tuple(h.kind for h in hops),
        confidences=tuple(h.confidence for h in hops),
        effective_from=tuple(h.effective_from for h in hops),
        hops_detail=hops,
        mass=mass,
        intermediate_degrees=inter_deg,
        max_intermediate_degree=max(inter_deg) if inter_deg else 0,
        hub_nodes=hub_nodes,
        hub_dominated=bool(hub_nodes),
        hub_degree_threshold=hub_threshold,
    )


# ── Independence ──────────────────────────────────────────────────────────


def independence(routes: Sequence[Route]) -> dict[str, Any]:
    """Count the INDEPENDENT TIME-VARYING sources behind a set of routes.

    ``independent_sources`` counts distinct ``source`` values on EVENT hops only.
    Static-geography hops (``produced_in``, ``exchange_country``, ``located_in``,
    ...) route the signal without witnessing anything, so their sources are
    returned under ``scaffold_sources`` and explicitly excluded from the count.
    Measured on the live graph, Russia -> WTI Crude Oil has several distinct
    sources across its top routes and exactly ONE of them is time-varying
    (gdelt); ``naive_source_count`` is returned alongside precisely so the gap
    between the two is visible rather than papered over.

    Returns a dict whose ``status`` is:
      * ``"ok"``        — routes were supplied and counted;
      * ``"no_route"``  — an empty ``RouteList`` that proved no path exists (or
        none within max_hops). ``independent_sources`` is 0 AND ``status`` says
        why, so "nothing connects these" is never confusable with "one source".
    An empty plain ``list`` raises ``RouteIndeterminateError``: without the
    search provenance it is impossible to say whether that means "not connected"
    or "I could not compute this", and this repo does not return that ambiguity
    as a number.

    Where it misleads:
      * it counts over the routes it is GIVEN. If ``top_routes`` truncated at k,
        or its budget ran out, this is a count over the top-k; those cases are
        copied into ``warnings`` and flagged by ``complete`` being False.
      * a ``source`` is an ingest pipeline. Two pipelines fed by one upstream
        publisher would be counted as two. Inspect ``evidence_channels``.
      * two ``source`` labels on the SAME link_type are probably one witness
        (measured: ``topic_relates_to_instrument`` is written by both
        ``polymarket`` and ``repair_topic_links``, the latter being a repair pass
        over the former). That makes ``independent_sources`` an UPPER bound;
        ``shared_relation_sources`` names every such collision and a warning is
        emitted.
      * the count is over the route set handed in, and the route set depends on
        ``k``. Measured on the live graph, Russia -> WTI Crude Oil returns 1
        independent source at the default k=10 (all event hops are
        ``event_involves``/gdelt) and 3 at k=400, where longer routes through
        prediction-market topics enter. Neither is wrong; the number is only
        meaningful next to the ``search`` block that produced it.
      * ``hub_free_independent_sources`` recounts after dropping hub-dominated
        routes and is the number to quote when a customer asks "and how much of
        that is just everything touching the United States?".
      * ``unclassified_*`` is not folded into either count. A link_type that
        nobody has classified must be classified, not averaged.
    """
    if not isinstance(routes, Sequence):
        raise RouteError(f"independence() expects a sequence of Route, got {type(routes).__name__}")
    search = getattr(routes, "search", None)
    for i, r in enumerate(routes):
        if not isinstance(r, Route):
            raise RouteError(f"independence() element {i} is {type(r).__name__}, not a Route")
    if len(routes) == 0:
        if search is None:
            raise RouteIndeterminateError(
                "independence() was given an empty route list with no search "
                "provenance. 'no path exists' and 'the search failed' are different "
                "facts and must not share the value 0. Pass the RouteList returned "
                "by top_routes()."
            )
        return {
            "status": "no_route",
            "complete": True,
            "routes_considered": 0,
            "independent_sources": 0,
            "evidence_sources": [],
            "evidence_channels": [],
            "sources_per_evidence_link_type": {},
            "evidence_link_types": [],
            "shared_relation_sources": {},
            "evidence_hops": 0,
            "scaffold_sources_non_evidence": [],
            "scaffold_hops": 0,
            "unclassified_sources_non_evidence": [],
            "unclassified_link_types": [],
            "unclassified_hops": 0,
            "naive_source_count": 0,
            "naive_sources": [],
            "source_inflation_factor": None,
            "routes_without_evidence": 0,
            "routes_sharing_node_sequence": 0,
            "hub_dominated_routes": 0,
            "hub_free_independent_sources": 0,
            "evidence_source_route_counts": {},
            "verdict": (
                "NO ROUTE: these entities are not connected in the graph as given, "
                "so there is no mechanism to explain and nothing to count. This is "
                "not a zero-evidence verdict about a real connection."
            ),
            "warnings": list(search.warnings()),
            "search": vars(search).copy(),
        }

    evidence_sources: set[str] = set()
    sources_by_link_type: dict[str, set[str]] = {}
    scaffold_sources: set[str] = set()
    unclassified_sources: set[str] = set()
    unclassified_types: set[str] = set()
    channels: set[tuple[str, str]] = set()
    naive_sources: set[str] = set()
    evidence_hops = scaffold_hops = unclassified_hops = 0
    per_source_routes: dict[str, int] = {}
    hub_routes = 0
    no_evidence_routes = 0
    hub_free_evidence: set[str] = set()
    node_seq_counts: dict[tuple[str, ...], int] = {}

    for route in routes:
        node_seq_counts[route.nodes] = node_seq_counts.get(route.nodes, 0) + 1
        if route.hub_dominated:
            hub_routes += 1
        if not route.has_evidence:
            no_evidence_routes += 1
        for hop in route.hops_detail:
            naive_sources.add(hop.source)
            if hop.kind == "evidence":
                evidence_hops += 1
                evidence_sources.add(hop.source)
                channels.add((hop.link_type, hop.source))
                sources_by_link_type.setdefault(hop.link_type, set()).add(hop.source)
            elif hop.kind == "scaffold":
                scaffold_hops += 1
                scaffold_sources.add(hop.source)
            else:
                unclassified_hops += 1
                unclassified_sources.add(hop.source)
                unclassified_types.add(hop.link_type)
        for src in route.evidence_sources:
            per_source_routes[src] = per_source_routes.get(src, 0) + 1
        if not route.hub_dominated:
            hub_free_evidence |= route.evidence_sources

    n_ind = len(evidence_sources)
    n_naive = len(naive_sources)
    repeated_node_sequences = sum(n - 1 for n in node_seq_counts.values() if n > 1)
    warnings = (
        list(search.warnings())
        if search is not None
        else [
            "no search provenance on this route list: cannot say whether it is the "
            "complete route set, so treat every count below as a lower bound."
        ]
    )
    if n_naive > n_ind:
        warnings.append(
            f"naive counting of every source on every hop would report {n_naive} "
            f"datasets; {n_naive - n_ind} of them are static scaffold "
            f"({sorted(scaffold_sources | unclassified_sources)}) and witness nothing."
        )
    if unclassified_types:
        warnings.append(
            f"UNCLASSIFIED link_type(s) {sorted(unclassified_types)} were traversed "
            "and are counted as neither evidence nor scaffold. Classify them in "
            "agent/models/gnn/graph_builder.py before quoting this verdict."
        )
    shared_relation = {lt: sorted(srcs) for lt, srcs in sources_by_link_type.items() if len(srcs) > 1}
    if shared_relation:
        warnings.append(
            "CO-DEPENDENCE RISK: "
            + "; ".join(f"{lt} is supplied by {srcs}" for lt, srcs in sorted(shared_relation.items()))
            + ". Two pipelines writing the SAME relation (e.g. a repair pass over an "
            "earlier ingest) are one witness, not two — independent_sources counts "
            "them separately and is an UPPER bound here."
        )
    if repeated_node_sequences:
        warnings.append(
            f"{repeated_node_sequences} of these routes retrace a node sequence "
            "another route already used (a different link_type over the same nodes). "
            "They are separate relations, not separate confirmations of the same one."
        )
    if n_ind and not hub_free_evidence:
        warnings.append(
            "every evidenced route passes through a hub node: the time-varying "
            "sources survive only on routes that are weak evidence by construction."
        )

    if n_ind == 0:
        verdict = (
            "NO TIME-VARYING EVIDENCE: every returned route is static geography "
            f"({sorted(scaffold_sources) or 'none'}). The graph says these entities "
            "COULD be related; nothing in it says anything happened. Do not report "
            "a confirmation."
        )
    elif n_ind == 1:
        only = next(iter(evidence_sources))
        verdict = (
            f"ONE independent time-varying source ({only}). Naive counting of the "
            f"{n_naive} source labels on these routes would claim {n_naive}; it "
            "would be one dataset wearing "
            f"{n_naive} hats. A failure of {only} takes the whole finding with it."
        )
    else:
        verdict = (
            f"{n_ind} independent time-varying sources "
            f"({sorted(evidence_sources)}), on {evidence_hops} event hops across "
            f"{len(routes)} routes. Naive counting would claim {n_naive}."
        )

    complete = bool(
        search is not None and search.exhaustive and not search.more_routes_exist and not search.budget_exhausted
    )
    return {
        "status": "ok",
        "complete": complete,
        "routes_considered": len(routes),
        "independent_sources": n_ind,
        "evidence_sources": sorted(evidence_sources),
        "evidence_channels": sorted(f"{lt}/{s}" for lt, s in channels),
        "sources_per_evidence_link_type": {lt: sorted(srcs) for lt, srcs in sorted(sources_by_link_type.items())},
        "evidence_link_types": sorted(sources_by_link_type),
        "shared_relation_sources": shared_relation,
        "evidence_hops": evidence_hops,
        "scaffold_sources_non_evidence": sorted(scaffold_sources),
        "scaffold_hops": scaffold_hops,
        "unclassified_sources_non_evidence": sorted(unclassified_sources),
        "unclassified_link_types": sorted(unclassified_types),
        "unclassified_hops": unclassified_hops,
        "naive_source_count": n_naive,
        "naive_sources": sorted(naive_sources),
        "source_inflation_factor": (n_naive / n_ind) if n_ind else None,
        "routes_without_evidence": no_evidence_routes,
        "routes_sharing_node_sequence": repeated_node_sequences,
        "hub_dominated_routes": hub_routes,
        "hub_free_independent_sources": len(hub_free_evidence),
        "evidence_source_route_counts": dict(sorted(per_source_routes.items())),
        "verdict": verdict,
        "warnings": warnings,
        "search": vars(search).copy() if search is not None else None,
    }


def describe_routes(routes: Sequence[Route]) -> str:
    """Render routes and their independence verdict as text, caveats included.

    Misleads: nothing — this is the one function here allowed to be pretty. It
    prints ``warnings`` last on purpose, because that is what a reader of a
    verdict skips and most needs.
    """
    verdict = independence(routes)
    lines: list[str] = []
    search = getattr(routes, "search", None)
    if search is not None:
        lines.append(
            f"{search.src} -> {search.dst}  "
            f"max_hops={search.max_hops} k={search.k} alpha={search.alpha} "
            f"shortest={search.shortest_hops} exhaustive={search.exhaustive} "
            f"hub_degree>={search.hub_degree_threshold:g}"
        )
    for i, route in enumerate(routes, 1):
        lines.append(f"  {i:2d}. {route.describe()}")
    lines.append(f"  status={verdict['status']}  complete={verdict['complete']}")
    lines.append(f"  independent_sources={verdict['independent_sources']}")
    lines.append(f"  evidence_channels={verdict['evidence_channels']}")
    lines.append(f"  scaffold (NON-evidence)={verdict['scaffold_sources_non_evidence']}")
    lines.append(f"  naive_source_count={verdict['naive_source_count']}")
    lines.append(f"  hub_dominated_routes={verdict['hub_dominated_routes']}")
    lines.append(f"  VERDICT: {verdict['verdict']}")
    for warning in verdict["warnings"]:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)
