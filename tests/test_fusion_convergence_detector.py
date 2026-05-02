"""Tests for ConvergenceDetector (Phase 20, Step 20.8).

Covers:
    - Two linked surprised entities → cluster
    - One surprised + one normal → no cluster
    - Chain of 3 surprised entities → one cluster
    - Disconnected surprised entities → separate handling
    - Cosine similarity computation
    - Edge cases: empty inputs, single entity, below threshold
"""

from __future__ import annotations

import math

import pytest

from agent.fusion.convergence import ConvergenceDetector, _cosine_similarity
from agent.fusion.surprise import EntitySurprise

# ── Helpers ────────────────────────────────────────────────────


def _make_surprise(
    entity_id: str,
    entity_type: str = "company",
    *,
    obs_type: float = 3.0,
    temporal: float = 2.0,
    value: float = 2.5,
    neighborhood: float = 1.0,
    memory: float = 0.5,
    composite: float | None = None,
) -> EntitySurprise:
    if composite is None:
        composite = obs_type + temporal + value + neighborhood + memory
    return EntitySurprise(
        entity_id=entity_id,
        entity_type=entity_type,
        obs_type_surprise=obs_type,
        temporal_surprise=temporal,
        value_surprise=value,
        neighborhood_surprise=neighborhood,
        memory_drift=memory,
        composite_surprise=composite,
    )


def _make_link(src: str, dst: str) -> dict:
    return {"src_id": src, "dst_id": dst}


# ═══════════════════════════════════════════════════════════════
# Cosine Similarity
# ═══════════════════════════════════════════════════════════════


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = (1.0, 2.0, 3.0, 4.0, 5.0)
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = (1.0, 0.0, 0.0, 0.0, 0.0)
        b = (0.0, 1.0, 0.0, 0.0, 0.0)
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = (1.0, 2.0, 3.0, 0.0, 0.0)
        b = (-1.0, -2.0, -3.0, 0.0, 0.0)
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = (0.0, 0.0, 0.0, 0.0, 0.0)
        b = (1.0, 2.0, 3.0, 4.0, 5.0)
        assert _cosine_similarity(a, b) == 0.0

    def test_both_zero(self):
        a = (0.0, 0.0, 0.0, 0.0, 0.0)
        assert _cosine_similarity(a, a) == 0.0

    def test_known_value(self):
        a = (1.0, 0.0, 0.0, 0.0, 0.0)
        b = (1.0, 1.0, 0.0, 0.0, 0.0)
        expected = 1.0 / math.sqrt(2.0)
        assert _cosine_similarity(a, b) == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════
# Mean Pairwise Cosine
# ═══════════════════════════════════════════════════════════════


class TestMeanPairwiseCosine:
    def test_two_identical_surprises(self):
        s1 = _make_surprise("c0", obs_type=1.0, temporal=1.0, value=1.0, neighborhood=1.0, memory=1.0)
        s2 = _make_surprise("c1", obs_type=1.0, temporal=1.0, value=1.0, neighborhood=1.0, memory=1.0)
        d = ConvergenceDetector()
        score = d._mean_pairwise_cosine([s1, s2])
        assert score == pytest.approx(1.0)

    def test_single_surprise(self):
        s = _make_surprise("c0")
        d = ConvergenceDetector()
        assert d._mean_pairwise_cosine([s]) == 0.0

    def test_three_identical(self):
        surprises = [
            _make_surprise(
                f"c{i}",
                obs_type=2.0,
                temporal=1.0,
                value=1.0,
                neighborhood=0.5,
                memory=0.5,
            )
            for i in range(3)
        ]
        d = ConvergenceDetector()
        assert d._mean_pairwise_cosine(surprises) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════
# ConvergenceDetector.detect()
# ═══════════════════════════════════════════════════════════════


class TestConvergenceDetectorBasic:
    def test_empty_surprises(self):
        d = ConvergenceDetector()
        clusters = d.detect({}, [])
        assert clusters == []

    def test_single_entity_no_cluster(self):
        surprises = {"c0": _make_surprise("c0", composite=10.0)}
        d = ConvergenceDetector()
        clusters = d.detect(surprises, [], surprise_threshold=2.0)
        assert clusters == []

    def test_two_linked_surprised_entities(self):
        """Two connected entities both above threshold → one cluster."""
        surprises = {
            "c0": _make_surprise("c0", composite=5.0),
            "c1": _make_surprise("c1", composite=5.0),
        }
        links = [_make_link("c0", "c1")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 1
        assert len(clusters[0].member_alerts) == 2

    def test_one_surprised_one_normal_no_cluster(self):
        """One above threshold, one below → no cluster."""
        surprises = {
            "c0": _make_surprise("c0", composite=5.0),
            "c1": _make_surprise("c1", composite=0.5),
        }
        links = [_make_link("c0", "c1")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert clusters == []

    def test_chain_of_three(self):
        """c0—c1—c2 all surprised → one cluster with 3 members."""
        surprises = {f"c{i}": _make_surprise(f"c{i}", composite=5.0) for i in range(3)}
        links = [_make_link("c0", "c1"), _make_link("c1", "c2")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 1
        assert len(clusters[0].member_alerts) == 3

    def test_disconnected_entities_no_cluster(self):
        """Two surprised but unlinked entities → no cluster."""
        surprises = {
            "c0": _make_surprise("c0", composite=5.0),
            "c1": _make_surprise("c1", composite=5.0),
        }
        links = []
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert clusters == []

    def test_two_separate_clusters(self):
        """Two disconnected pairs → two clusters."""
        surprises = {
            "c0": _make_surprise("c0", composite=5.0),
            "c1": _make_surprise("c1", composite=5.0),
            "c2": _make_surprise("c2", composite=5.0),
            "c3": _make_surprise("c3", composite=5.0),
        }
        links = [_make_link("c0", "c1"), _make_link("c2", "c3")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 2


class TestConvergenceDetectorScoring:
    def test_identical_surprise_patterns_score_1(self):
        """Same surprise vector → cosine = 1.0."""
        surprises = {
            "c0": _make_surprise(
                "c0",
                obs_type=3.0,
                temporal=2.0,
                value=2.0,
                neighborhood=1.0,
                memory=0.5,
                composite=8.5,
            ),
            "c1": _make_surprise(
                "c1",
                obs_type=3.0,
                temporal=2.0,
                value=2.0,
                neighborhood=1.0,
                memory=0.5,
                composite=8.5,
            ),
        }
        links = [_make_link("c0", "c1")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 1
        assert clusters[0].correlated_surprise_score == pytest.approx(1.0)

    def test_different_patterns_lower_score(self):
        """Different surprise vectors → cosine < 1.0."""
        surprises = {
            "c0": _make_surprise(
                "c0",
                obs_type=10.0,
                temporal=0.0,
                value=0.0,
                neighborhood=0.0,
                memory=0.0,
                composite=10.0,
            ),
            "c1": _make_surprise(
                "c1",
                obs_type=0.0,
                temporal=10.0,
                value=0.0,
                neighborhood=0.0,
                memory=0.0,
                composite=10.0,
            ),
        }
        links = [_make_link("c0", "c1")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 1
        assert clusters[0].correlated_surprise_score == pytest.approx(0.0)


class TestConvergenceDetectorEdgeCases:
    def test_threshold_boundary_excluded(self):
        """Exactly at threshold → excluded (must be >)."""
        surprises = {
            "c0": _make_surprise("c0", composite=2.0),
            "c1": _make_surprise("c1", composite=2.0),
        }
        links = [_make_link("c0", "c1")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert clusters == []

    def test_just_above_threshold(self):
        surprises = {
            "c0": _make_surprise("c0", composite=2.01),
            "c1": _make_surprise("c1", composite=2.01),
        }
        links = [_make_link("c0", "c1")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 1

    def test_mixed_entity_types(self):
        surprises = {
            "c0": _make_surprise("c0", entity_type="company", composite=5.0),
            "w0": _make_surprise("w0", entity_type="wallet", composite=5.0),
        }
        links = [_make_link("c0", "w0")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 1
        assert "company" in clusters[0].contributing_domains
        assert "wallet" in clusters[0].contributing_domains

    def test_unique_cluster_ids(self):
        """Each cluster gets a unique ID."""
        surprises = {
            "c0": _make_surprise("c0", composite=5.0),
            "c1": _make_surprise("c1", composite=5.0),
            "c2": _make_surprise("c2", composite=5.0),
            "c3": _make_surprise("c3", composite=5.0),
        }
        links = [_make_link("c0", "c1"), _make_link("c2", "c3")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        ids = [c.cluster_id for c in clusters]
        assert len(set(ids)) == len(ids)

    def test_alternative_link_key_names(self):
        """Support different key names for src/dst in links."""
        surprises = {
            "c0": _make_surprise("c0", composite=5.0),
            "c1": _make_surprise("c1", composite=5.0),
        }
        links = [{"source_id": "c0", "target_id": "c1"}]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 1

    def test_bidirectional_edges(self):
        """Both directions in links → same cluster (not duplicated)."""
        surprises = {
            "c0": _make_surprise("c0", composite=5.0),
            "c1": _make_surprise("c1", composite=5.0),
        }
        links = [_make_link("c0", "c1"), _make_link("c1", "c0")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 1
        assert len(clusters[0].member_alerts) == 2

    def test_self_loop_ignored(self):
        """Self-loop edge doesn't create cluster from single entity."""
        surprises = {
            "c0": _make_surprise("c0", composite=5.0),
        }
        links = [_make_link("c0", "c0")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert clusters == []

    def test_large_cluster(self):
        """10 fully connected surprised entities → one cluster."""
        n = 10
        surprises = {f"c{i}": _make_surprise(f"c{i}", composite=5.0) for i in range(n)}
        links = [_make_link(f"c{i}", f"c{j}") for i in range(n) for j in range(i + 1, n)]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 1
        assert len(clusters[0].member_alerts) == 10

    def test_cluster_member_alerts_have_correct_ids(self):
        surprises = {
            "apple": _make_surprise("apple", composite=5.0),
            "google": _make_surprise("google", composite=5.0),
        }
        links = [_make_link("apple", "google")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        alert_ids = {a.entity_id for a in clusters[0].member_alerts}
        assert alert_ids == {"apple", "google"}

    def test_cluster_preserves_surprise_values(self):
        s0 = _make_surprise(
            "c0",
            obs_type=5.0,
            temporal=3.0,
            value=2.0,
            neighborhood=1.0,
            memory=0.5,
            composite=11.5,
        )
        surprises = {
            "c0": s0,
            "c1": _make_surprise("c1", composite=5.0),
        }
        links = [_make_link("c0", "c1")]
        d = ConvergenceDetector()
        clusters = d.detect(surprises, links, surprise_threshold=2.0)
        c0_alert = next(a for a in clusters[0].member_alerts if a.entity_id == "c0")
        assert c0_alert.obs_type_surprise == pytest.approx(5.0)
        assert c0_alert.temporal_surprise == pytest.approx(3.0)
        assert c0_alert.value_surprise == pytest.approx(2.0)
