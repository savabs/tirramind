"""Coincidence graph construction and clique detection.

Builds an undirected graph where:
- **Nodes** = signal IDs (one per active signal stream).
- **Edges** = statistically significant pairwise coincidences,
  weighted by the combined coincidence score.

Cross-category maximal cliques are the convergence events that the
rest of the system cares about.

Design decisions
----------------
- Category mapping is passed in as ``dict[str, str]`` (signal → category)
  so this module has no coupling to the full ``SignalRegistry``.
- Clique enumeration uses ``nx.find_cliques`` (Bron-Kerbosch with pivot).
  A ``max_cliques`` safety cap prevents combinatorial explosion on
  pathological dense graphs.
- Scoring rewards edge strength, cross-category diversity, and clique
  size (log₂ scale), then normalises to [0, 1] via sigmoid.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import networkx as nx

from agent.convergence.coincidence import CoincidenceResult

log = logging.getLogger(__name__)

# Safety cap on how many cliques we enumerate before stopping.
# Bron-Kerbosch worst case is exponential; this prevents runaway.
_MAX_CLIQUES = 10_000


# ── Dataclass ──────────────────────────────────────────────────


@dataclass
class ConvergenceClique:
    """A group of signals exhibiting mutual, cross-category coincidence.

    Attributes
    ----------
    signals : list[str]
        Signal IDs in this clique (sorted for determinism).
    categories : list[str]
        Distinct taxonomy categories represented (sorted).
    edges : list[tuple[str, str, float]]
        (signal_a, signal_b, weight) for every intra-clique edge.
    score : float
        Aggregate convergence score in [0, 1].
    p_values : list[float]
        Per-edge p-values (same order as ``edges``), fed to
        Fisher's combined test downstream.
    """

    signals: list[str]
    categories: list[str]
    edges: list[tuple[str, str, float]]
    score: float
    p_values: list[float] = field(default_factory=list)

    # ── Fingerprint (for persistence tracking) ─────────────────

    def fingerprint(self) -> tuple[str, ...]:
        """Return a hashable, order-invariant identifier for this clique."""
        return tuple(sorted(self.signals))


# ── Graph construction ─────────────────────────────────────────


def build_coincidence_graph(
    scores: dict[tuple[str, str], CoincidenceResult],
    categories: dict[str, str],
    *,
    p_threshold: float = 0.05,
) -> nx.Graph:
    """Build a weighted undirected coincidence graph.

    Parameters
    ----------
    scores : dict[(sig_a, sig_b), CoincidenceResult]
        Pairwise coincidence results.  Key order does not matter;
        only one entry per unordered pair is expected.
    categories : dict[str, str]
        Mapping signal_id → taxonomy category for every signal
        that should appear as a node.
    p_threshold : float
        Maximum p-value for an edge to be included.

    Returns
    -------
    nx.Graph
        Nodes carry a ``"category"`` attribute.
        Edges carry ``"weight"`` (score), ``"p_value"``, and
        ``"direction"`` attributes.
    """
    G = nx.Graph()

    # Add all known nodes with their category attribute.
    for sig_id, cat in categories.items():
        G.add_node(sig_id, category=cat)

    # Add significant edges.
    for (sig_a, sig_b), result in scores.items():
        if result.p_value >= p_threshold:
            continue
        if result.score <= 0.0:
            continue

        # Ensure both endpoints exist as nodes.
        for sig in (sig_a, sig_b):
            if sig not in G:
                G.add_node(sig, category=categories.get(sig, "unknown"))

        G.add_edge(
            sig_a,
            sig_b,
            weight=result.score,
            p_value=result.p_value,
            direction=result.direction,
        )

    return G


# ── Clique detection ───────────────────────────────────────────


def detect_convergence_cliques(
    G: nx.Graph,
    *,
    min_size: int = 3,
    min_categories: int = 2,
    max_cliques: int = _MAX_CLIQUES,
) -> list[ConvergenceClique]:
    """Find cross-category maximal cliques in the coincidence graph.

    Parameters
    ----------
    G : nx.Graph
        Coincidence graph from :func:`build_coincidence_graph`.
    min_size : int
        Minimum number of signals in an accepted clique.
    min_categories : int
        Minimum number of distinct taxonomy categories.
    max_cliques : int
        Safety cap — stop enumeration after this many raw cliques
        to prevent combinatorial explosion.

    Returns
    -------
    list[ConvergenceClique]
        Surviving cliques, ranked by (cross_category_count desc,
        total_edge_weight desc).
    """
    if G.number_of_edges() == 0:
        return []

    results: list[ConvergenceClique] = []
    count = 0

    for raw_clique in nx.find_cliques(G):
        count += 1
        if count > max_cliques:
            log.warning(
                "Clique enumeration capped at %d; graph may be too dense",
                max_cliques,
            )
            break

        if len(raw_clique) < min_size:
            continue

        # Resolve categories.
        node_cats = {n: G.nodes[n].get("category", "unknown") for n in raw_clique}
        distinct_cats = sorted(set(node_cats.values()))

        if len(distinct_cats) < min_categories:
            continue

        # Collect intra-clique edges.
        signals = sorted(raw_clique)
        edges: list[tuple[str, str, float]] = []
        p_values: list[float] = []
        for i, sa in enumerate(signals):
            for sb in signals[i + 1 :]:
                edata = G.edges.get((sa, sb))
                if edata is None:
                    continue
                w = edata.get("weight", 0.0)
                p = edata.get("p_value", 1.0)
                edges.append((sa, sb, w))
                p_values.append(p)

        clique = ConvergenceClique(
            signals=signals,
            categories=distinct_cats,
            edges=edges,
            score=0.0,  # computed below
            p_values=p_values,
        )
        clique.score = score_clique(clique)
        results.append(clique)

    # Rank: most categories first, then highest total weight.
    results.sort(
        key=lambda c: (len(c.categories), sum(w for _, _, w in c.edges)),
        reverse=True,
    )
    return results


# ── Clique scoring ─────────────────────────────────────────────


def score_clique(clique: ConvergenceClique) -> float:
    """Score a convergence clique on [0, 1].

    Formula
    -------
    .. math::

        \\text{raw} = \\frac{1}{|C|} \\sum_{(i,j) \\in C} w_{ij}
            \\times \\frac{\\text{cross\\_cat}}{|C|}
            \\times \\log_2(|C|)

    The raw score is mapped through a sigmoid for [0, 1] normalisation:

    .. math::

        \\text{score} = \\frac{2}{1 + e^{-\\text{raw}}} - 1

    This gives 0.0 for raw = 0, approaches 1.0 for large raw values,
    and has a smooth gradient in between.

    Edge cases
    ----------
    - Empty edges → 0.0
    - Single-signal clique (|C| = 1) → log₂(1) = 0 → 0.0
    """
    n = len(clique.signals)
    if n < 2 or not clique.edges:
        return 0.0

    total_weight = sum(w for _, _, w in clique.edges)
    mean_weight = total_weight / n
    cross_cat = len(clique.categories)
    cat_ratio = cross_cat / n
    size_bonus = math.log2(n)

    raw = mean_weight * cat_ratio * size_bonus

    # Sigmoid normalisation to [0, 1].
    # 2 / (1 + e^{-raw}) - 1  maps  0→0, ∞→1.
    score = 2.0 / (1.0 + math.exp(-raw)) - 1.0

    return round(score, 6)
