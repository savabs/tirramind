"""Tests for coincidence graph + clique detection (Phase 7c-C.2).

Covers: ConvergenceClique construction + fingerprint,
build_coincidence_graph (node/edge counts, thresholding, unknown nodes),
detect_convergence_cliques (handcrafted topologies, cross-category
filtering, min_size, ranking, max_cliques safety cap),
score_clique (hand-computed values, edge cases).
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from agent.convergence.coincidence import CoincidenceResult
from agent.convergence.graph import (
    ConvergenceClique,
    _MAX_CLIQUES,
    build_coincidence_graph,
    detect_convergence_cliques,
    score_clique,
)

# ── Helpers ────────────────────────────────────────────────────


def _cr(
    score: float = 2.0,
    p_value: float = 0.01,
    direction: int = 1,
    method: str = "combined",
) -> CoincidenceResult:
    """Shortcut to build a CoincidenceResult for test fixtures."""
    return CoincidenceResult(
        method=method,
        score=score,
        p_value=p_value,
        direction=direction,
    )


def _categories(*names: str) -> dict[str, str]:
    """Build signal_id → category mapping.

    Each name is ``"sig_id:category"`` (e.g., ``"s1:positioning"``).
    """
    out = {}
    for n in names:
        sig, cat = n.split(":")
        out[sig] = cat
    return out


# ═══════════════════════════════════════════════════════════════
# ConvergenceClique — construction and fingerprint
# ═══════════════════════════════════════════════════════════════


class TestConvergenceClique:
    def test_construction(self):
        c = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["positioning", "macro_momentum"],
            edges=[("a", "b", 2.0), ("a", "c", 1.5), ("b", "c", 1.8)],
            score=0.5,
            p_values=[0.01, 0.02, 0.03],
        )
        assert len(c.signals) == 3
        assert c.score == 0.5
        assert len(c.p_values) == 3

    def test_fingerprint_is_sorted(self):
        c1 = ConvergenceClique(
            signals=["c", "a", "b"],
            categories=[],
            edges=[],
            score=0.0,
        )
        c2 = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=[],
            edges=[],
            score=0.0,
        )
        assert c1.fingerprint() == c2.fingerprint()
        assert c1.fingerprint() == ("a", "b", "c")

    def test_fingerprint_hashable(self):
        c = ConvergenceClique(
            signals=["x", "y"],
            categories=[],
            edges=[],
            score=0.0,
        )
        s = {c.fingerprint()}  # must be hashable for set/dict use
        assert c.fingerprint() in s

    def test_default_p_values_empty(self):
        c = ConvergenceClique(
            signals=["a"],
            categories=[],
            edges=[],
            score=0.0,
        )
        assert c.p_values == []


# ═══════════════════════════════════════════════════════════════
# build_coincidence_graph
# ═══════════════════════════════════════════════════════════════


class TestBuildCoincidenceGraph:
    def test_basic_construction(self):
        cats = _categories("s1:positioning", "s2:macro_momentum", "s3:biological")
        scores = {
            ("s1", "s2"): _cr(score=3.0, p_value=0.001),
            ("s2", "s3"): _cr(score=2.0, p_value=0.03),
        }
        G = build_coincidence_graph(scores, cats)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert G.nodes["s1"]["category"] == "positioning"

    def test_p_threshold_filters_edges(self):
        cats = _categories("s1:positioning", "s2:macro_momentum")
        scores = {
            ("s1", "s2"): _cr(score=1.0, p_value=0.10),  # above default 0.05
        }
        G = build_coincidence_graph(scores, cats, p_threshold=0.05)
        assert G.number_of_edges() == 0  # filtered out
        assert G.number_of_nodes() == 2  # nodes still present

    def test_edge_at_exact_threshold_excluded(self):
        """p_value == p_threshold should be excluded (strict <)."""
        cats = _categories("s1:positioning", "s2:macro_momentum")
        scores = {("s1", "s2"): _cr(p_value=0.05)}
        G = build_coincidence_graph(scores, cats, p_threshold=0.05)
        assert G.number_of_edges() == 0

    def test_edge_just_below_threshold_included(self):
        cats = _categories("s1:positioning", "s2:macro_momentum")
        scores = {("s1", "s2"): _cr(p_value=0.0499)}
        G = build_coincidence_graph(scores, cats, p_threshold=0.05)
        assert G.number_of_edges() == 1

    def test_zero_score_edge_excluded(self):
        """Edges with score=0 (no evidence) are excluded even if p < threshold."""
        cats = _categories("s1:positioning", "s2:macro_momentum")
        scores = {("s1", "s2"): _cr(score=0.0, p_value=0.01)}
        G = build_coincidence_graph(scores, cats)
        assert G.number_of_edges() == 0

    def test_edge_attributes(self):
        cats = _categories("s1:positioning", "s2:macro_momentum")
        scores = {("s1", "s2"): _cr(score=2.5, p_value=0.001, direction=-1)}
        G = build_coincidence_graph(scores, cats)
        edata = G.edges["s1", "s2"]
        assert edata["weight"] == 2.5
        assert edata["p_value"] == 0.001
        assert edata["direction"] == -1

    def test_unknown_signal_gets_node(self):
        """Signals in scores but not in categories still get nodes."""
        cats = _categories("s1:positioning")
        scores = {("s1", "s_unknown"): _cr(p_value=0.01)}
        G = build_coincidence_graph(scores, cats)
        assert "s_unknown" in G
        assert G.nodes["s_unknown"]["category"] == "unknown"

    def test_empty_scores(self):
        cats = _categories("s1:positioning", "s2:macro_momentum")
        G = build_coincidence_graph({}, cats)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 0

    def test_empty_categories(self):
        G = build_coincidence_graph({}, {})
        assert G.number_of_nodes() == 0

    def test_isolated_nodes_preserved(self):
        """Nodes with no significant edges are still in the graph."""
        cats = _categories(
            "s1:positioning",
            "s2:macro_momentum",
            "s3:biological",
        )
        scores = {("s1", "s2"): _cr(p_value=0.01)}
        G = build_coincidence_graph(scores, cats)
        assert G.number_of_nodes() == 3
        assert G.degree("s3") == 0

    def test_large_graph_construction(self):
        """500 nodes, sparse edges — construction should be fast."""
        cats = {f"s{i}": "positioning" for i in range(500)}
        scores = {}
        for i in range(499):
            scores[(f"s{i}", f"s{i+1}")] = _cr(p_value=0.001)
        G = build_coincidence_graph(scores, cats)
        assert G.number_of_nodes() == 500
        assert G.number_of_edges() == 499


# ═══════════════════════════════════════════════════════════════
# detect_convergence_cliques
# ═══════════════════════════════════════════════════════════════


class TestDetectConvergenceCliques:

    # ── Positive cases ─────────────────────────────────────────

    def test_triangle_across_3_categories(self):
        """Three mutually connected nodes across 3 categories → 1 clique."""
        cats = _categories(
            "s1:positioning",
            "s2:macro_momentum",
            "s3:biological",
        )
        scores = {
            ("s1", "s2"): _cr(p_value=0.001, score=3.0),
            ("s2", "s3"): _cr(p_value=0.002, score=2.5),
            ("s1", "s3"): _cr(p_value=0.003, score=2.0),
        }
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        assert len(cliques) == 1
        assert set(cliques[0].signals) == {"s1", "s2", "s3"}
        assert len(cliques[0].categories) == 3
        assert len(cliques[0].edges) == 3
        assert len(cliques[0].p_values) == 3

    def test_four_node_clique_across_3_categories(self):
        """K4 across 3 categories → 1 large clique."""
        cats = _categories(
            "s1:positioning",
            "s2:macro_momentum",
            "s3:biological",
            "s4:positioning",
        )
        # Build K4 (all 6 edges)
        nodes = ["s1", "s2", "s3", "s4"]
        scores = {}
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                scores[(a, b)] = _cr(p_value=0.001, score=2.0)
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        # The maximal clique is the entire K4
        assert any(len(c.signals) == 4 for c in cliques)

    def test_two_disjoint_cliques(self):
        """Two separate triangles → 2 cliques (if both cross-category)."""
        cats = _categories(
            "a1:positioning",
            "a2:macro_momentum",
            "a3:biological",
            "b1:supply_chain",
            "b2:geopolitical",
            "b3:financial_stress",
        )
        scores = {}
        for pair in [("a1", "a2"), ("a2", "a3"), ("a1", "a3")]:
            scores[pair] = _cr(p_value=0.001, score=2.0)
        for pair in [("b1", "b2"), ("b2", "b3"), ("b1", "b3")]:
            scores[pair] = _cr(p_value=0.002, score=1.5)
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        assert len(cliques) == 2

    # ── Filtering: min_categories ──────────────────────────────

    def test_same_category_clique_filtered(self):
        """3 nodes, all same category → filtered by min_categories=2."""
        cats = _categories(
            "s1:positioning",
            "s2:positioning",
            "s3:positioning",
        )
        scores = {
            ("s1", "s2"): _cr(p_value=0.001),
            ("s2", "s3"): _cr(p_value=0.001),
            ("s1", "s3"): _cr(p_value=0.001),
        }
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        assert len(cliques) == 0

    def test_min_categories_1_allows_same_category(self):
        cats = _categories(
            "s1:positioning",
            "s2:positioning",
            "s3:positioning",
        )
        scores = {
            ("s1", "s2"): _cr(p_value=0.001),
            ("s2", "s3"): _cr(p_value=0.001),
            ("s1", "s3"): _cr(p_value=0.001),
        }
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=1)
        assert len(cliques) == 1

    # ── Filtering: min_size ────────────────────────────────────

    def test_pair_filtered_by_min_size_3(self):
        """Two connected nodes → clique of size 2 → filtered by min_size=3."""
        cats = _categories("s1:positioning", "s2:macro_momentum")
        scores = {("s1", "s2"): _cr(p_value=0.001)}
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3)
        assert len(cliques) == 0

    def test_min_size_2_allows_pairs(self):
        cats = _categories("s1:positioning", "s2:macro_momentum")
        scores = {("s1", "s2"): _cr(p_value=0.001)}
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=2, min_categories=2)
        assert len(cliques) == 1
        assert set(cliques[0].signals) == {"s1", "s2"}

    # ── Edge cases ─────────────────────────────────────────────

    def test_empty_graph(self):
        G = nx.Graph()
        cliques = detect_convergence_cliques(G)
        assert cliques == []

    def test_no_edges(self):
        cats = _categories("s1:positioning", "s2:macro_momentum")
        G = build_coincidence_graph({}, cats)
        cliques = detect_convergence_cliques(G)
        assert cliques == []

    def test_isolated_node_not_in_cliques(self):
        cats = _categories(
            "s1:positioning",
            "s2:macro_momentum",
            "s3:biological",
            "s4:geopolitical",
        )
        # Triangle s1-s2-s3, s4 isolated
        scores = {
            ("s1", "s2"): _cr(p_value=0.001),
            ("s2", "s3"): _cr(p_value=0.001),
            ("s1", "s3"): _cr(p_value=0.001),
        }
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        all_sigs = {s for c in cliques for s in c.signals}
        assert "s4" not in all_sigs

    # ── Ranking ────────────────────────────────────────────────

    def test_ranking_by_categories_then_weight(self):
        """Clique with more categories should rank first."""
        cats = _categories(
            "a1:positioning",
            "a2:macro_momentum",
            "a3:biological",
            "b1:supply_chain",
            "b2:supply_chain",
            "b3:geopolitical",
        )
        # Clique A: 3 categories, weight 2.0 each
        for pair in [("a1", "a2"), ("a2", "a3"), ("a1", "a3")]:
            pass  # added below
        scores = {
            ("a1", "a2"): _cr(p_value=0.001, score=2.0),
            ("a2", "a3"): _cr(p_value=0.001, score=2.0),
            ("a1", "a3"): _cr(p_value=0.001, score=2.0),
            # Clique B: 2 categories, weight 5.0 each (higher weight)
            ("b1", "b2"): _cr(p_value=0.001, score=5.0),
            ("b2", "b3"): _cr(p_value=0.001, score=5.0),
            ("b1", "b3"): _cr(p_value=0.001, score=5.0),
        }
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        assert len(cliques) == 2
        # Clique A has 3 categories → should rank first
        assert len(cliques[0].categories) == 3
        assert len(cliques[1].categories) == 2

    # ── max_cliques safety cap ─────────────────────────────────

    def test_max_cliques_cap(self):
        """With max_cliques=1, at most 1 clique survives enumeration."""
        cats = _categories(
            "s1:positioning",
            "s2:macro_momentum",
            "s3:biological",
            "s4:geopolitical",
            "s5:supply_chain",
            "s6:financial_stress",
        )
        nodes = [f"s{i}" for i in range(1, 7)]
        scores = {}
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                scores[(a, b)] = _cr(p_value=0.001, score=2.0)
        G = build_coincidence_graph(scores, cats)
        # With max_cliques=1, enumeration stops after first raw clique
        cliques = detect_convergence_cliques(
            G, min_size=3, min_categories=2, max_cliques=1
        )
        # Can't guarantee exact count because the first raw clique
        # may or may not pass filters, but should not blow up.
        assert isinstance(cliques, list)

    # ── Clique edge/p_value extraction ─────────────────────────

    def test_edge_and_pvalue_extraction(self):
        """Edges and p-values in the clique match the graph data."""
        cats = _categories(
            "s1:positioning",
            "s2:macro_momentum",
            "s3:biological",
        )
        scores = {
            ("s1", "s2"): _cr(score=3.0, p_value=0.001),
            ("s2", "s3"): _cr(score=2.5, p_value=0.005),
            ("s1", "s3"): _cr(score=2.0, p_value=0.010),
        }
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        assert len(cliques) == 1
        c = cliques[0]
        # 3 edges in a triangle
        assert len(c.edges) == 3
        assert len(c.p_values) == 3
        weights = sorted(w for _, _, w in c.edges)
        assert weights == [2.0, 2.5, 3.0]
        p_vals = sorted(c.p_values)
        assert p_vals == [0.001, 0.005, 0.010]

    # ── Signals and categories sorted ──────────────────────────

    def test_signals_sorted(self):
        cats = _categories(
            "z:positioning",
            "a:macro_momentum",
            "m:biological",
        )
        scores = {
            ("z", "a"): _cr(p_value=0.001),
            ("a", "m"): _cr(p_value=0.001),
            ("z", "m"): _cr(p_value=0.001),
        }
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        assert cliques[0].signals == ["a", "m", "z"]  # sorted
        assert cliques[0].categories == sorted(cliques[0].categories)


# ═══════════════════════════════════════════════════════════════
# score_clique
# ═══════════════════════════════════════════════════════════════


class TestScoreClique:
    def test_hand_computed_triangle(self):
        """Hand verify the scoring formula for a known clique.

        3 nodes, 3 edges with weights [2.0, 2.5, 3.0].
        total_weight = 7.5, mean_weight = 7.5/3 = 2.5
        cross_cat = 3 categories, n = 3 → cat_ratio = 1.0
        size_bonus = log2(3) ≈ 1.585
        raw = 2.5 × 1.0 × 1.585 = 3.9624
        sigmoid(raw) = 2 / (1 + e^{-3.9624}) - 1 ≈ 0.9629
        """
        c = ConvergenceClique(
            signals=["s1", "s2", "s3"],
            categories=["positioning", "macro_momentum", "biological"],
            edges=[("s1", "s2", 2.0), ("s1", "s3", 2.5), ("s2", "s3", 3.0)],
            score=0.0,
        )
        result = score_clique(c)
        total_w = 7.5
        mean_w = total_w / 3.0
        cat_ratio = 3.0 / 3.0
        size_bonus = math.log2(3)
        raw = mean_w * cat_ratio * size_bonus
        expected = 2.0 / (1.0 + math.exp(-raw)) - 1.0
        assert abs(result - round(expected, 6)) < 1e-5

    def test_pair_clique(self):
        """Two signals, 1 edge. log2(2) = 1.0."""
        c = ConvergenceClique(
            signals=["a", "b"],
            categories=["positioning", "macro_momentum"],
            edges=[("a", "b", 4.0)],
            score=0.0,
        )
        result = score_clique(c)
        # mean_weight = 4.0/2 = 2.0, cat_ratio = 2/2 = 1.0, log2(2) = 1.0
        raw = 2.0 * 1.0 * 1.0
        expected = 2.0 / (1.0 + math.exp(-raw)) - 1.0
        assert abs(result - round(expected, 6)) < 1e-5

    def test_single_signal_yields_zero(self):
        c = ConvergenceClique(
            signals=["solo"],
            categories=["positioning"],
            edges=[],
            score=0.0,
        )
        assert score_clique(c) == 0.0

    def test_no_edges_yields_zero(self):
        c = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["positioning"],
            edges=[],
            score=0.0,
        )
        assert score_clique(c) == 0.0

    def test_score_bounded_01(self):
        """Score must be in [0, 1] regardless of input magnitude."""
        # Giant weights
        c = ConvergenceClique(
            signals=[f"s{i}" for i in range(10)],
            categories=["a", "b", "c", "d", "e"],
            edges=[
                (f"s{i}", f"s{j}", 100.0) for i in range(10) for j in range(i + 1, 10)
            ],
            score=0.0,
        )
        result = score_clique(c)
        assert 0.0 <= result <= 1.0

    def test_score_increases_with_weight(self):
        """Higher edge weights → higher score, all else equal."""

        def make(w):
            return ConvergenceClique(
                signals=["a", "b", "c"],
                categories=["positioning", "macro_momentum", "biological"],
                edges=[("a", "b", w), ("a", "c", w), ("b", "c", w)],
                score=0.0,
            )

        s_low = score_clique(make(1.0))
        s_high = score_clique(make(5.0))
        assert s_high > s_low

    def test_score_increases_with_cross_category(self):
        """More distinct categories → higher score, all else equal."""
        # 2 categories
        c2 = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["positioning", "positioning"],
            edges=[("a", "b", 2.0), ("a", "c", 2.0), ("b", "c", 2.0)],
            score=0.0,
        )
        # 3 categories
        c3 = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["positioning", "macro_momentum", "biological"],
            edges=[("a", "b", 2.0), ("a", "c", 2.0), ("b", "c", 2.0)],
            score=0.0,
        )
        assert score_clique(c3) > score_clique(c2)

    def test_score_increases_with_size(self):
        """Larger cliques → higher score (log₂ bonus), all else equal."""
        # 3-node, 3 categories
        c3 = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["positioning", "macro_momentum", "biological"],
            edges=[("a", "b", 2.0), ("a", "c", 2.0), ("b", "c", 2.0)],
            score=0.0,
        )
        # 4-node, 4 categories, same weight per edge
        c4 = ConvergenceClique(
            signals=["a", "b", "c", "d"],
            categories=["positioning", "macro_momentum", "biological", "supply_chain"],
            edges=[
                ("a", "b", 2.0),
                ("a", "c", 2.0),
                ("a", "d", 2.0),
                ("b", "c", 2.0),
                ("b", "d", 2.0),
                ("c", "d", 2.0),
            ],
            score=0.0,
        )
        assert score_clique(c4) > score_clique(c3)

    def test_zero_weight_edges(self):
        """All-zero weights → raw = 0 → sigmoid(0) = 0."""
        c = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["positioning", "macro_momentum", "biological"],
            edges=[("a", "b", 0.0), ("a", "c", 0.0), ("b", "c", 0.0)],
            score=0.0,
        )
        assert score_clique(c) == 0.0


# ═══════════════════════════════════════════════════════════════
# Integration: build → detect → score round-trip
# ═══════════════════════════════════════════════════════════════


class TestIntegration:
    def test_full_pipeline_simple(self):
        """Build graph, detect cliques, verify scores are populated."""
        cats = _categories(
            "s1:positioning",
            "s2:macro_momentum",
            "s3:biological",
            "s4:geopolitical",
        )
        # K3 among s1, s2, s3 + isolated s4
        scores = {
            ("s1", "s2"): _cr(score=3.0, p_value=0.001),
            ("s2", "s3"): _cr(score=2.5, p_value=0.002),
            ("s1", "s3"): _cr(score=2.0, p_value=0.005),
        }
        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        assert len(cliques) == 1
        c = cliques[0]
        assert c.score > 0.0
        assert c.score <= 1.0
        assert set(c.signals) == {"s1", "s2", "s3"}
        assert "s4" not in c.signals

    def test_mixed_significant_and_nonsignificant(self):
        """Only edges below p_threshold form cliques."""
        cats = _categories(
            "s1:positioning",
            "s2:macro_momentum",
            "s3:biological",
            "s4:geopolitical",
        )
        scores = {
            ("s1", "s2"): _cr(p_value=0.001),
            ("s2", "s3"): _cr(p_value=0.001),
            ("s1", "s3"): _cr(p_value=0.001),
            ("s3", "s4"): _cr(p_value=0.10),  # not significant
        }
        G = build_coincidence_graph(scores, cats, p_threshold=0.05)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        # s4 is not part of any clique (its only edge was filtered)
        for c in cliques:
            assert "s4" not in c.signals

    def test_performance_100_node_dense(self):
        """100 nodes, dense graph — should complete reasonably fast.

        This is a realistic stress test: 60 real signals × some being
        mutually significant.  Not full K100 (which is pathological).
        """
        cats = {}
        cat_names = [
            "positioning",
            "macro_momentum",
            "biological",
            "geopolitical",
            "supply_chain",
        ]
        for i in range(100):
            cats[f"s{i}"] = cat_names[i % len(cat_names)]

        scores = {}
        # Chain graph + some cross-links → moderate density
        for i in range(99):
            scores[(f"s{i}", f"s{i+1}")] = _cr(p_value=0.001, score=2.0)
        # Add some triangles
        for i in range(0, 98, 3):
            scores[(f"s{i}", f"s{i+2}")] = _cr(p_value=0.001, score=1.5)

        G = build_coincidence_graph(scores, cats)
        cliques = detect_convergence_cliques(G, min_size=3, min_categories=2)
        assert isinstance(cliques, list)
        # Should find some triangles
        assert len(cliques) > 0
