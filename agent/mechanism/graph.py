"""Load the TirraMind entity graph as a point-in-time sparse adjacency matrix.

This is the foundation of the mechanism layer. Everything else in
`agent/mechanism/` consumes a `MechanismGraph` and adds no graph loading of its
own, so the two distinctions this module enforces are enforced everywhere:

1.  EVIDENCE vs SCAFFOLD.  `STRUCTURAL_RELATIONS` (produced_in,
    exchange_country, located_in, ...) are static geography. "Canada produces
    oil" was true in 1990 and is true today, so such an edge carries no
    time-varying information: it *routes* evidence, it is not evidence.
    `EVENT_RELATIONS` (event_involves, works_for, transacts_with, ...) happen on
    a date. Independence of support must therefore be counted over evidence
    edges only. Counting scaffold `source` values inflates confidence — the
    measured Russia->WTI routes all have the shape
    ``Russia --event_involves--> {country} --produced_in--> WTI``, show five
    distinct `source` values, and are ONE observation: only the event_involves
    edge varies with the date, and every one of those is gdelt.
    Both relation sets are imported from `agent.models.gnn.graph_builder` and
    are deliberately not redefined here.

2.  POINT-IN-TIME.  `as_of` is a required keyword, not a defaulted one.
    Answering a question about 2019 with the 2026 graph is F-04 leakage, and a
    defaulted time bound is exactly how F-14 happened in this repo. A caller
    that wants the live graph must say `as_of=None` out loud.

WHERE THIS MODULE MISLEADS (read before trusting a number out of it):

*   The adjacency is BINARY. A pair joined by four different link_types has the
    same weight as a pair joined by one. This is intentional: it is what the
    reference personalised-PageRank measurement was computed against, and
    multiplicity weighting is a quiet way to let one noisy collector vote
    repeatedly. Measured on the live graph, seeding Russia at alpha=0.15,
    binary weights put WTI at mass 2.6476e-03 / rank 127 while link_type
    multiplicity puts it at 2.6865e-03 / rank 126 — small, but it moves.
*   The default `scaffold_policy="timeless"` keeps every structural edge in
    scope at every `as_of`. This is correct for geography and necessary for
    routing (all 119 `produced_in` links have a NULL `effective_from`, so a
    strict row-level gate deletes every one of them at any historical date and
    the Russia->WTI route silently ceases to exist). The residual leak is
    *existence*, not geography: an ETF first listed in 2021 still carries its
    `exchange_country` edge in a 2019 graph, so the node is reachable in a year
    it did not trade. Pass `scaffold_policy="gated"` for a strict audit and
    compare; the policy in force is recorded in `GraphLoadReport`.
*   Degree here is the degree of the deduplicated, undirected, point-in-time
    graph. It is not the row count in `entity_links`. "United States" has 724
    rows but 522 distinct neighbours; the deduplicated number is the one that
    governs how a random walker's mass splits.
*   A `link_type` in neither relation set is treated as evidence (i.e. gated on
    its effective time, the conservative side) and named in
    `GraphLoadReport.unknown_link_types`. It is never silently dropped and
    never silently promoted to scaffold.

Nothing here caps, samples or prunes. Every row the loader does not put in the
graph is counted in `GraphLoadReport`, and the loader raises if those counts do
not reconcile against the row count it read — a truncated result that looks
complete is this repo's most repeated bug (F-16, F-17).
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import numpy as np
import scipy.sparse as sp

from agent.models.gnn.graph_builder import EVENT_RELATIONS, STRUCTURAL_RELATIONS

log = logging.getLogger(__name__)

__all__ = [
    "DisconnectedEntityError",
    "EdgeClass",
    "EmptyGraphError",
    "GraphLoadReport",
    "HubEntry",
    "MechanismEdge",
    "MechanismGraph",
    "MechanismGraphError",
    "ScaffoldTimePolicy",
    "UnknownEntityError",
    "as_of_from_iso",
    "load_graph",
]

EdgeClass = Literal["evidence", "scaffold"]
ScaffoldTimePolicy = Literal["timeless", "gated"]


# ── Errors ─────────────────────────────────────────────────────
#
# These exist so that "not connected" and "could not compute" are never the
# same value. A mechanism score of 0.0 must mean "the walker reached this node
# with zero mass", never "I did not recognise the id you gave me".


class MechanismGraphError(Exception):
    """Base class for every loud failure in the mechanism graph layer."""


class UnknownEntityError(KeyError, MechanismGraphError):
    """An entity_id that is not a node of this graph was used as a lookup key."""


class DisconnectedEntityError(MechanismGraphError):
    """A known entity has degree 0 in this point-in-time graph.

    Raised only by `MechanismGraph.require_connected`, which callers that are
    about to compute a walk, path or score should call first. `degree()` itself
    returns 0 for such a node, because 0 is the true answer to "how many
    neighbours".
    """


class EmptyGraphError(MechanismGraphError):
    """The load produced no nodes, or no edges at all at the requested `as_of`."""


# ── Value types ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MechanismEdge:
    """One deduplicated undirected link, with the provenance it came from.

    `a` and `b` are node indices into `MechanismGraph.entity_ids`, always with
    `a < b`, so an edge has exactly one representation. `effective_at` is epoch
    seconds: `entity_links.effective_from` when the column is populated,
    otherwise `created_at` (the ingest stamp, which is only a proxy — it says
    when TirraMind learned the fact, not when the fact became true).

    `is_evidence` is False for static geography. Do not count `source` values
    across scaffold edges and call the result independent support.
    """

    a: int
    b: int
    link_type: str
    source: str
    confidence: float
    effective_at: float
    is_evidence: bool

    @property
    def edge_class(self) -> EdgeClass:
        """ "evidence" for a time-varying event link, "scaffold" for geography."""
        return "evidence" if self.is_evidence else "scaffold"


@dataclass(frozen=True, slots=True)
class HubEntry:
    """One entity's degree, for inspecting hub contamination."""

    entity_id: str
    canonical_name: str
    entity_type: str
    degree: int
    evidence_degree: int


@dataclass(frozen=True, slots=True)
class GraphLoadReport:
    """Full accounting of one `load_graph` call. Nothing is dropped silently.

    Every field is a count the loader actually made, and `load_graph` raises
    `MechanismGraphError` if `rows_read` does not equal the sum of the kept and
    dropped rows. `truncated` is always False: this loader has no cap. It is a
    field rather than an absence so that a caller can assert on it and keep
    asserting if a cap is ever added.
    """

    as_of: float | None
    scaffold_policy: ScaffoldTimePolicy
    db_path: str
    rows_read: int
    rows_with_effective_from: int
    rows_dropped_by_as_of: int
    rows_dropped_self_loop: int
    rows_dropped_unknown_entity: int
    rows_collapsed_by_dedup: int
    n_entities: int
    n_edges: int
    n_undirected_pairs: int
    n_evidence_pairs: int
    n_scaffold_pairs: int
    n_isolated_entities: int
    link_type_counts: dict[str, int]
    unknown_link_types: tuple[str, ...]
    truncated: bool = False

    @property
    def rows_kept(self) -> int:
        """Rows that survived filtering, before parallel-edge deduplication."""
        return self.n_edges + self.rows_collapsed_by_dedup


@dataclass(frozen=True)
class MechanismGraph:
    """A point-in-time, undirected, deduplicated view of the entity graph.

    Node order is fixed (entity_id ascending) so that any two loads of the same
    `as_of` produce identical indices, and a matrix computed from one is safe to
    read with the other's index map.

    `adjacency` is the full graph (evidence + scaffold); `evidence_adjacency` is
    the same node set with scaffold edges removed. Both are symmetric CSR with
    weight 1.0 and no self-loops. Use `adjacency` to ask "is there a route" and
    `evidence_adjacency` to ask "does anything time-varying support it".

    Isolated nodes are kept. A 20,026-node graph with 1,724 nodes of degree 0 is
    the truth about this database; dropping them would renumber every index and
    make percentile claims about rank incomparable across `as_of` values.
    """

    entity_ids: tuple[str, ...]
    entity_types: tuple[str, ...]
    canonical_names: tuple[str, ...]
    adjacency: sp.csr_matrix
    evidence_adjacency: sp.csr_matrix
    edges: tuple[MechanismEdge, ...]
    as_of: float | None
    scaffold_policy: ScaffoldTimePolicy
    report: GraphLoadReport
    _index: dict[str, int] = field(repr=False)
    _incident: dict[int, tuple[int, ...]] = field(repr=False)
    _degree: np.ndarray = field(repr=False)
    _evidence_degree: np.ndarray = field(repr=False)

    # ── size ───────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        """Number of entities, including isolated ones."""
        return len(self.entity_ids)

    @property
    def n_edges(self) -> int:
        """Number of deduplicated undirected (pair, link_type) edge records.

        This is larger than the number of distinct connected pairs when a pair
        is joined by more than one relation. `adjacency.nnz // 2` is the pair
        count; they are not the same number and conflating them overstates how
        much the graph knows.
        """
        return len(self.edges)

    # ── lookups (loud on an unknown id) ────────────────────────

    def index_of(self, entity_id: str) -> int:
        """Row/column index of `entity_id`, raising if it is not a node.

        Never returns -1 or None for an unknown id: a sentinel index would
        silently score the wrong entity.
        """
        try:
            return self._index[entity_id]
        except KeyError:
            raise UnknownEntityError(
                f"entity_id {entity_id!r} is not a node of this graph "
                f"(as_of={self.as_of!r}, {self.n_nodes} nodes). It may exist in "
                f"`entities` but have been filtered out — check spelling before "
                f"assuming the graph is wrong."
            ) from None

    def has(self, entity_id: str) -> bool:
        """Whether `entity_id` is a node here. Use before `index_of` to branch."""
        return entity_id in self._index

    def name_of(self, entity_id: str) -> str:
        """`entities.canonical_name`, raising `UnknownEntityError` if unknown."""
        return self.canonical_names[self.index_of(entity_id)]

    def type_of(self, entity_id: str) -> str:
        """`entities.entity_type`, raising `UnknownEntityError` if unknown."""
        return self.entity_types[self.index_of(entity_id)]

    def entities_of_type(self, entity_type: str) -> tuple[str, ...]:
        """Every entity_id of `entity_type`, in node order.

        Returns an empty tuple for a type that is absent, which is a real answer
        — but check `report.n_entities` if you expected otherwise, because an
        empty result here and a mis-spelled type look identical.
        """
        return tuple(eid for eid, etype in zip(self.entity_ids, self.entity_types, strict=True) if etype == entity_type)

    # ── degree / hub-ness ──────────────────────────────────────

    def degree(self, entity_id: str, *, evidence_only: bool = False) -> int:
        """Number of distinct neighbours of `entity_id` in this graph.

        Counts pairs, not rows and not link_types: a pair joined by three
        relations contributes 1. This is the quantity that governs how a random
        walker splits its mass, which is why hub discounting works at all.

        Returns 0 for a known but isolated entity — that is the true degree.
        Raises `UnknownEntityError` for an id that is not a node. Call
        `require_connected` first if a downstream 0 would be ambiguous.
        """
        i = self.index_of(entity_id)
        arr = self._evidence_degree if evidence_only else self._degree
        return int(arr[i])

    def degrees(self, *, evidence_only: bool = False) -> np.ndarray:
        """Degree of every node, in node order. A copy; mutating it is harmless."""
        arr = self._evidence_degree if evidence_only else self._degree
        return arr.copy()

    def require_connected(self, entity_id: str, *, evidence_only: bool = False) -> int:
        """Return the node index, raising if the entity has no edges here.

        Call this before computing a walk, path or mechanism score. It is the
        guard that keeps "this entity is isolated at this date" from being
        reported as a score of zero indistinguishable from "the walk failed".
        """
        i = self.index_of(entity_id)
        arr = self._evidence_degree if evidence_only else self._degree
        if arr[i] == 0:
            kind = "evidence" if evidence_only else "any"
            raise DisconnectedEntityError(
                f"{entity_id!r} ({self.canonical_names[i]!r}) has no {kind} edges "
                f"at as_of={self.as_of!r}. Any score computed from it would be a "
                f"zero that means 'nothing to compute', not 'computed zero'."
            )
        return i

    def hubs(self, top_n: int = 20, *, evidence_only: bool = False) -> tuple[tuple[HubEntry, ...], int]:
        """The `top_n` highest-degree entities, and the total number of nodes ranked.

        Returns `(entries, n_ranked)` rather than a bare list so a caller cannot
        mistake a truncated top-N for the whole distribution. Ties are broken by
        entity_id for determinism.

        Hub-ness matters because naive path enumeration through a high-degree
        node returns the same route many times and calls them confirmations.
        Personalised PageRank on `adjacency` handles this by construction — a
        walker entering a degree-d node splits its mass d ways — but any code
        that counts paths must inspect these degrees itself.
        """
        if top_n < 0:
            raise ValueError(f"top_n must be >= 0, got {top_n}")
        arr = self._evidence_degree if evidence_only else self._degree
        order = sorted(range(self.n_nodes), key=lambda i: (-int(arr[i]), self.entity_ids[i]))
        entries = tuple(
            HubEntry(
                entity_id=self.entity_ids[i],
                canonical_name=self.canonical_names[i],
                entity_type=self.entity_types[i],
                degree=int(self._degree[i]),
                evidence_degree=int(self._evidence_degree[i]),
            )
            for i in order[:top_n]
        )
        return entries, self.n_nodes

    def degree_percentile(self, entity_id: str, *, evidence_only: bool = False) -> float:
        """Fraction of nodes (0..100) with a strictly lower degree than this one.

        Computed over ALL nodes including the 1,724 isolated ones, so a modest
        degree scores a high percentile. Compare percentiles only against other
        percentiles from the same `as_of`.
        """
        i = self.index_of(entity_id)
        arr = self._evidence_degree if evidence_only else self._degree
        return float((arr < arr[i]).sum()) / self.n_nodes * 100.0

    # ── provenance ─────────────────────────────────────────────

    def edges_between(self, entity_a: str, entity_b: str) -> tuple[MechanismEdge, ...]:
        """Every deduplicated edge joining these two entities, in link_type order.

        Empty means "not adjacent at this `as_of`". Both ids must be nodes;
        an unknown id raises rather than returning empty, so a typo cannot read
        as "no relationship".
        """
        i = self.index_of(entity_a)
        j = self.index_of(entity_b)
        lo, hi = (i, j) if i < j else (j, i)
        if lo == hi:
            return ()
        found = [self.edges[k] for k in self._incident.get(lo, ()) if {self.edges[k].a, self.edges[k].b} == {lo, hi}]
        return tuple(sorted(found, key=lambda e: (e.link_type, e.source)))

    def incident_edges(self, entity_id: str, *, evidence_only: bool = False) -> tuple[MechanismEdge, ...]:
        """Every edge touching `entity_id`. Empty tuple for an isolated node."""
        i = self.index_of(entity_id)
        out = [self.edges[k] for k in self._incident.get(i, ())]
        if evidence_only:
            out = [e for e in out if e.is_evidence]
        return tuple(out)

    def neighbours(self, entity_id: str, *, evidence_only: bool = False) -> tuple[str, ...]:
        """Distinct adjacent entity_ids, sorted. Empty tuple for an isolated node."""
        i = self.index_of(entity_id)
        seen = set()
        for k in self._incident.get(i, ()):
            e = self.edges[k]
            if evidence_only and not e.is_evidence:
                continue
            seen.add(self.entity_ids[e.b if e.a == i else e.a])
        return tuple(sorted(seen))


# ── loading ────────────────────────────────────────────────────


def as_of_from_iso(date_str: str) -> float:
    """Epoch seconds for an ISO date or datetime, interpreted as UTC.

    A bare date means midnight UTC at the START of that day, so
    `as_of=as_of_from_iso("2019-01-01")` excludes everything that happened
    during 1 January 2019. A naive datetime is read as UTC rather than local
    time, because reading it as local would shift every boundary by the
    machine's timezone and make results differ between laptops.
    """
    parsed = dt.datetime.fromisoformat(date_str)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.timestamp()


def _connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open the pipeline DB strictly read-only.

    The live DB is the only irreplaceable asset in this project, so the URI is
    built here from a resolved path and nowhere else. `mode=ro` makes SQLite
    reject every write at the engine level, which is a stronger guarantee than
    "this module contains no INSERT statements".
    """
    path = Path(db_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"pipeline DB not found at {path}. load_graph never creates a "
            f"database — an empty one would answer every mechanism question "
            f"with a confident 'no connection'."
        )
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _classify(link_type: str, unknown: set[str]) -> bool:
    """True if this link_type is evidence (time-varying), False if scaffold.

    An unclassified link_type is treated as evidence — it gets gated on its
    effective time, which is the conservative side — and recorded so the caller
    can see it was never silently filed as geography.
    """
    if link_type in STRUCTURAL_RELATIONS:
        return False
    if link_type in EVENT_RELATIONS:
        return True
    unknown.add(link_type)
    return True


def _reconcile(
    *,
    rows_read: int,
    kept: int,
    collapsed: int,
    dropped_as_of: int,
    dropped_self: int,
    dropped_unknown: int,
) -> None:
    """Raise unless every row read is either kept or counted in a drop bucket.

    This is the guard against the failure this repo keeps repeating: a result
    that is missing rows and looks complete (F-16, F-17). It is a separate
    function so it can be tested against numbers that do not add up, which the
    loader itself will not produce on purpose.
    """
    accounted = kept + collapsed + dropped_as_of + dropped_self + dropped_unknown
    if accounted != rows_read:
        raise MechanismGraphError(
            f"row accounting does not reconcile: read {rows_read} rows from "
            f"entity_links but accounted for {accounted} "
            f"(kept {kept}, deduped {collapsed}, as_of {dropped_as_of}, "
            f"self-loops {dropped_self}, unknown entity {dropped_unknown}). "
            f"Rows went missing uncounted — fix the loader, do not trust the graph."
        )


def load_graph(
    db_path: str | Path,
    *,
    as_of: float | None,
    scaffold_policy: ScaffoldTimePolicy = "timeless",
) -> MechanismGraph:
    """Load the entity graph as it stood at `as_of`, read-only.

    Args:
        db_path: path to `pipeline.db`. Opened `mode=ro`; never written.
        as_of: epoch seconds, or None for "now, everything". REQUIRED keyword,
            with no default, on purpose. Explaining a 2019 pattern with the 2026
            graph is F-04 leakage, and a defaulted time bound is how F-14
            happened in this repo — the only way to get the live graph is to
            type `as_of=None`, which is then visible in the caller's diff and in
            `GraphLoadReport.as_of`. Use `as_of_from_iso("2019-01-01")`.
        scaffold_policy: "timeless" (default) keeps every STRUCTURAL_RELATIONS
            edge in scope at every date, matching `graph_builder` and treating
            geography as the static fact it is. "gated" applies the same
            effective-time filter to scaffold edges too. See the module
            docstring for what each one gets wrong; the choice is recorded in
            the report.

    Returns:
        A `MechanismGraph` whose `report` accounts for every row read.

    Raises:
        FileNotFoundError: no DB at `db_path`.
        EmptyGraphError: no entities, or no edges survived `as_of`. A graph with
            no edges answers every connectivity question with a plausible zero,
            so it fails here instead.
        MechanismGraphError: the row accounting did not reconcile, i.e. this
            loader lost rows without counting them.
        ValueError: unrecognised `scaffold_policy`.

    What it computes: a symmetric binary CSR adjacency over all entities, a
    second CSR restricted to evidence edges, and one `MechanismEdge` per
    distinct (unordered pair, link_type) carrying `source`, `confidence` and
    effective time.

    Where it misleads: it deduplicates, so counts here are smaller than
    `SELECT count(*) FROM entity_links` and are not comparable to numbers
    measured from raw rows. It also says nothing about edge *direction*: the
    graph is symmetrised, because a mechanism route is a route in either
    direction, but that means `works_for(person, company)` and a hypothetical
    reverse are the same edge.
    """
    if scaffold_policy not in ("timeless", "gated"):
        raise ValueError(f"scaffold_policy must be 'timeless' or 'gated', got {scaffold_policy!r}")
    if as_of is not None:
        as_of = float(as_of)

    con = _connect_readonly(db_path)
    try:
        ent_rows = con.execute(
            "SELECT entity_id, entity_type, canonical_name FROM entities ORDER BY entity_id"
        ).fetchall()
        link_rows = con.execute(
            "SELECT entity_id_a, entity_id_b, link_type, source, confidence, "
            "effective_from, created_at FROM entity_links ORDER BY link_id"
        ).fetchall()
    finally:
        con.close()

    if not ent_rows:
        raise EmptyGraphError(f"`entities` is empty in {db_path}; there is no graph to reason over.")

    entity_ids = tuple(r[0] for r in ent_rows)
    entity_types = tuple(r[1] for r in ent_rows)
    canonical_names = tuple(r[2] for r in ent_rows)
    index = {eid: i for i, eid in enumerate(entity_ids)}
    n = len(entity_ids)

    rows_read = len(link_rows)
    rows_with_eff = 0
    dropped_as_of = 0
    dropped_self = 0
    dropped_unknown = 0
    collapsed = 0
    unknown_link_types: set[str] = set()
    link_type_counts: dict[str, int] = defaultdict(int)

    seen: set[tuple[int, int, str]] = set()
    edges: list[MechanismEdge] = []

    for a, b, link_type, source, confidence, effective_from, created_at in link_rows:
        if effective_from is not None:
            rows_with_eff += 1
        effective_at = float(effective_from if effective_from is not None else created_at)

        is_evidence = _classify(link_type, unknown_link_types)
        if as_of is not None and (is_evidence or scaffold_policy == "gated") and effective_at > as_of:
            dropped_as_of += 1
            continue

        ia = index.get(a)
        ib = index.get(b)
        if ia is None or ib is None:
            dropped_unknown += 1
            log.warning(
                "entity_links row references an entity_id absent from `entities` "
                "(%r -> %r, link_type=%r); dropped and counted in the report.",
                a,
                b,
                link_type,
            )
            continue
        if ia == ib:
            dropped_self += 1
            continue

        lo, hi = (ia, ib) if ia < ib else (ib, ia)
        key = (lo, hi, link_type)
        if key in seen:
            # A parallel edge: the same pair and relation stored in both
            # directions, or duplicated by two collectors. Counting it twice is
            # what made naive path enumeration return one hub route four times
            # and call it four confirmations.
            collapsed += 1
            continue
        seen.add(key)
        link_type_counts[link_type] += 1
        edges.append(
            MechanismEdge(
                a=lo,
                b=hi,
                link_type=link_type,
                source=source,
                confidence=float(confidence),
                effective_at=effective_at,
                is_evidence=is_evidence,
            )
        )

    _reconcile(
        rows_read=rows_read,
        kept=len(edges),
        collapsed=collapsed,
        dropped_as_of=dropped_as_of,
        dropped_self=dropped_self,
        dropped_unknown=dropped_unknown,
    )

    if not edges:
        raise EmptyGraphError(
            f"no edges survived as_of={as_of!r} (read {rows_read} rows, dropped "
            f"{dropped_as_of} as future). An edgeless graph reports every "
            f"hypothesis as unsupported, which looks like a finding."
        )

    adjacency = _symmetrise(edges, n, evidence_only=False)
    evidence_adjacency = _symmetrise(edges, n, evidence_only=True)

    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int64)
    evidence_degree = np.asarray(evidence_adjacency.sum(axis=1)).ravel().astype(np.int64)

    incident: dict[int, list[int]] = defaultdict(list)
    for k, e in enumerate(edges):
        incident[e.a].append(k)
        incident[e.b].append(k)

    n_evidence_pairs = evidence_adjacency.nnz // 2
    n_pairs = adjacency.nnz // 2
    scaffold_pairs = len({(e.a, e.b) for e in edges if not e.is_evidence})

    if unknown_link_types:
        log.warning(
            "link_type(s) %s are in neither STRUCTURAL_RELATIONS nor "
            "EVENT_RELATIONS; treated as evidence (time-gated) and reported. "
            "Classify them in agent/models/gnn/graph_builder.py.",
            sorted(unknown_link_types),
        )

    report = GraphLoadReport(
        as_of=as_of,
        scaffold_policy=scaffold_policy,
        db_path=str(db_path),
        rows_read=rows_read,
        rows_with_effective_from=rows_with_eff,
        rows_dropped_by_as_of=dropped_as_of,
        rows_dropped_self_loop=dropped_self,
        rows_dropped_unknown_entity=dropped_unknown,
        rows_collapsed_by_dedup=collapsed,
        n_entities=n,
        n_edges=len(edges),
        n_undirected_pairs=n_pairs,
        n_evidence_pairs=n_evidence_pairs,
        n_scaffold_pairs=scaffold_pairs,
        n_isolated_entities=int((degree == 0).sum()),
        link_type_counts=dict(sorted(link_type_counts.items())),
        unknown_link_types=tuple(sorted(unknown_link_types)),
    )

    return MechanismGraph(
        entity_ids=entity_ids,
        entity_types=entity_types,
        canonical_names=canonical_names,
        adjacency=adjacency,
        evidence_adjacency=evidence_adjacency,
        edges=tuple(edges),
        as_of=as_of,
        scaffold_policy=scaffold_policy,
        report=report,
        _index=index,
        _incident={k: tuple(v) for k, v in incident.items()},
        _degree=degree,
        _evidence_degree=evidence_degree,
    )


def _symmetrise(edges: list[MechanismEdge], n: int, *, evidence_only: bool) -> sp.csr_matrix:
    """Binary symmetric CSR over `n` nodes from deduplicated undirected edges.

    Weight is 1.0 regardless of how many link_types join a pair: `sum_duplicates`
    would otherwise turn "two collectors mention the same pair" into twice the
    transition probability. Deliberately binary — see the module docstring for
    the measured difference this makes to PPR mass.
    """
    pairs = {(e.a, e.b) for e in edges if e.is_evidence or not evidence_only}
    if not pairs:
        return sp.csr_matrix((n, n), dtype=np.float64)
    rows = np.empty(2 * len(pairs), dtype=np.int64)
    cols = np.empty(2 * len(pairs), dtype=np.int64)
    for k, (a, b) in enumerate(pairs):
        rows[2 * k] = a
        cols[2 * k] = b
        rows[2 * k + 1] = b
        cols[2 * k + 1] = a
    data = np.ones(rows.size, dtype=np.float64)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    mat.data[:] = 1.0  # guard: no accumulation even if a pair slipped through twice
    return mat
