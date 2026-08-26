"""Graph analytics over the evidence graph using networkx.

Turns the evidence-links table into a real queryable graph and computes
standard measures a fund/analyst would want:
  - degree / degree-centrality: how connected an entity is
  - top connection hubs: the most-connected entities
  - subgraph for one entity: its immediate neighborhood

These are the graph-level findings that distinguish "a list of links" from
"intelligence" (e.g. which entity is a central hub in the evidence network).
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from agent.evidence.store import EvidenceGraphStore


def build_graph(
    store: EvidenceGraphStore,
    min_confidence: float = 0.0,
    limit: int = 5000,
    weight_key: str = "n_docs",
) -> nx.Graph:
    """Build an undirected weighted graph from evidence links.

    Edge weight defaults to n_docs (recurring pairs dominate) — the strongest,
    most defensible weighting.
    """
    g = nx.Graph()
    for a, b, attrs in store.all_edges(min_confidence=min_confidence, limit=limit):
        w = attrs.get(weight_key, 1.0) or 1.0
        if g.has_edge(a, b):
            g[a][b]["weight"] = max(g[a][b].get("weight", 0.0), float(w))
        else:
            g.add_edge(a, b, weight=float(w), evidence=attrs.get("evidence", ""))
    return g


def degree_centrality(store: EvidenceGraphStore, top_n: int = 10, **kw: Any) -> list[dict[str, Any]]:
    """Entities ranked by degree centrality (fraction of nodes connected to)."""
    g = build_graph(store, **kw)
    if g.number_of_nodes() == 0:
        return []
    dc = nx.degree_centrality(g)
    ranked = sorted(dc.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [{"entity": e, "degree": int(g.degree(e)), "centrality": round(c, 4)} for e, c in ranked]


def neighbors(store: EvidenceGraphStore, entity: str, top_n: int = 20, **kw: Any) -> dict[str, Any]:
    """The immediate neighborhood of an entity (its connected neighbors, weighted)."""
    g = build_graph(store, **kw)
    if entity not in g:
        return {"entity": entity, "found": False}
    nbrs = sorted(g[entity].items(), key=lambda kv: kv[1].get("weight", 0.0), reverse=True)
    return {
        "entity": entity,
        "found": True,
        "degree": int(g.degree(entity)),
        "neighbors": [{"entity": n, "weight": round(d.get("weight", 0.0), 2)} for n, d in nbrs[:top_n]],
    }


__all__ = ["build_graph", "degree_centrality", "neighbors"]
