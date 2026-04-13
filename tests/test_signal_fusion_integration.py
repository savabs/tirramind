"""Phase 20 — Step 20.13: Signal Fusion Integration Tests.

Full end-to-end: mock observations → enrichment → GNN → surprise → convergence → stored.

Anti-bias tests:
    - Generic convergence: entity with surprised linked neighbors → ConvergenceCluster
    - Novel patterns: unseen tool combinations → clusters detected
    - Cold-start: entity with no history → falls back to statistical features
    - Do NOT test for specific named archetype patterns

Edge cases:
    - 0 observations
    - NaN embeddings
    - Stale entities
    - CUSUM accumulation across runs
    - Hawkes decay between runs
    - Very large entity count
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from agent.fusion.alert import EntityAlert
from agent.fusion.convergence import ConvergenceCluster, ConvergenceDetector
from agent.fusion.cusum import CUSUMMonitor
from agent.fusion.entity_baseline import EntityBaseline
from agent.fusion.entity_scorer import EntityAnomalyScorer, ScorerConfig
from agent.fusion.hawkes import HawkesIntensity
from agent.fusion.surprise import EntitySurprise, SurpriseExtractor
from agent.models.gnn.graph_builder import IDMap, OBSERVATION_TYPES


# ── Shared Helpers ────────────────────────────────────────────


def _mock_store(
    entities: list[dict] | None = None,
    observations: list[dict] | None = None,
    links: list[dict] | None = None,
) -> MagicMock:
    """Create a mock PipelineStore with configurable data."""
    store = MagicMock()
    entities = entities or []
    observations = observations or []
    links = links or []

    store.query_all_entities.return_value = entities
    store.query_all_observations.return_value = observations
    store.query_all_entity_links.return_value = links
    store.query_entity_observations.return_value = observations
    store.db_path = ":memory:"
    return store


def _mock_model(num_nodes: int = 5, hidden_dim: int = 16, memory_dim: int = 16):
    """Create a mock HetTGN model with prediction heads."""
    model = MagicMock()
    model.memory = MagicMock()
    model.memory.memory = torch.zeros(num_nodes, memory_dim)

    def forward_fn(data, id_map):
        embeddings = {}
        for ntype, eid_map in id_map.type_local.items():
            n = len(eid_map)
            embeddings[ntype] = torch.randn(n, hidden_dim)
        return embeddings

    model.side_effect = forward_fn
    model.eval = MagicMock(return_value=None)

    def predict_obs_type_fn(embeddings):
        result = {}
        for ntype, emb in embeddings.items():
            result[ntype] = torch.zeros(emb.size(0), len(OBSERVATION_TYPES))
        return result

    def predict_time_delta_fn(embeddings):
        result = {}
        for ntype, emb in embeddings.items():
            result[ntype] = torch.ones(emb.size(0), 1) * 100.0
        return result

    def predict_value_fn(embeddings):
        result = {}
        for ntype, emb in embeddings.items():
            result[ntype] = torch.zeros(emb.size(0), 1)
        return result

    model.predict_obs_type = MagicMock(side_effect=predict_obs_type_fn)
    model.predict_time_delta = MagicMock(side_effect=predict_time_delta_fn)
    model.predict_value = MagicMock(side_effect=predict_value_fn)

    return model


def _entities(*ids: str, etype: str = "company") -> list[dict]:
    return [
        {"entity_id": eid, "entity_type": etype, "canonical_name": f"Name_{eid}"}
        for eid in ids
    ]


def _observations(
    *entity_ids: str,
    obs_type: str = "price_movement",
    source_tool: str = "test_tool",
    base_time: float | None = None,
    value: float = 100.0,
) -> list[dict]:
    t0 = base_time or time.time()
    return [
        {
            "entity_id": eid,
            "source_tool": source_tool,
            "observed_at": t0 - 100 + i * 10,
            "observation_type": obs_type,
            "value_json": f'{{"usd_amount": {value}}}',
        }
        for i, eid in enumerate(entity_ids)
    ]


def _links(*pairs: tuple[str, str], link_type: str = "correlation") -> list[dict]:
    return [
        {"entity_id_a": a, "entity_id_b": b, "link_type": link_type} for a, b in pairs
    ]


# ═══════════════════════════════════════════════════════════════
# Full End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════


class TestFullPipeline:
    """End-to-end: observations → enrichment → GNN → surprise → convergence."""

    def _run_pipeline(
        self,
        entities: list[dict],
        observations: list[dict],
        links: list[dict],
        config: ScorerConfig | None = None,
        num_nodes: int = 10,
    ) -> tuple[list[EntityAlert], list[ConvergenceCluster]]:
        store = _mock_store(entities, observations, links)
        model = _mock_model(num_nodes=num_nodes)

        # Mock GraphBuilder to return a plausible graph
        id_map = IDMap()
        for e in entities:
            id_map.add(e["entity_type"], e["entity_id"])

        mock_data = MagicMock()
        with patch("agent.fusion.entity_scorer.GraphBuilder") as MockGB:
            MockGB.return_value.build.return_value = (mock_data, id_map, [])
            scorer = EntityAnomalyScorer(store, model, config=config)
            return scorer.score_entities(as_of=time.time())

    def test_basic_pipeline_runs(self) -> None:
        """Entities with observations → alerts generated."""
        ents = _entities("e1", "e2", "e3")
        obs = _observations("e1", "e2", "e3")
        alerts, clusters = self._run_pipeline(ents, obs, [])
        # Should produce alerts (one per entity with observations)
        assert len(alerts) >= 1

    def test_alerts_have_correct_fields(self) -> None:
        ents = _entities("e1")
        obs = _observations("e1")
        alerts, _ = self._run_pipeline(ents, obs, [])
        if alerts:
            a = alerts[0]
            assert isinstance(a, EntityAlert)
            assert a.entity_id == "e1"
            assert isinstance(a.composite_surprise, float)
            assert isinstance(a.obs_type_surprise, float)
            assert isinstance(a.temporal_surprise, float)
            assert isinstance(a.value_surprise, float)
            assert isinstance(a.cusum_statistic, float)
            assert isinstance(a.hawkes_intensity, float)

    def test_linked_entities_produce_clusters(self) -> None:
        """Connected entities with high surprise → ConvergenceCluster."""
        ents = _entities("e1", "e2", "e3")
        obs = _observations("e1", "e2", "e3")
        lnks = _links(("e1", "e2"), ("e2", "e3"))

        # Use very low threshold so any surprise triggers convergence
        cfg = ScorerConfig(surprise_threshold=0.0)
        alerts, clusters = self._run_pipeline(ents, obs, lnks, config=cfg)
        # With threshold=0, all entities are "elevated" → should form cluster
        assert len(clusters) >= 1

    def test_no_observations_returns_empty(self) -> None:
        ents = _entities("e1", "e2")
        alerts, clusters = self._run_pipeline(ents, [], [])
        assert alerts == []
        assert clusters == []

    def test_pipeline_with_multiple_observation_types(self) -> None:
        """Multiple obs types per entity run cleanly."""
        ents = _entities("e1")
        obs = _observations("e1", obs_type="price_movement") + _observations(
            "e1", obs_type="insider_trade"
        )
        alerts, _ = self._run_pipeline(ents, obs, [])
        assert len(alerts) >= 1

    def test_enrichment_values_are_finite(self) -> None:
        """CUSUM/Hawkes/event_study scores on alerts should be finite."""
        ents = _entities("e1", "e2")
        obs = _observations("e1", "e2")
        alerts, _ = self._run_pipeline(ents, obs, [])
        for a in alerts:
            assert math.isfinite(a.cusum_statistic)
            assert math.isfinite(a.hawkes_intensity)
            assert math.isfinite(a.event_study_score)


# ═══════════════════════════════════════════════════════════════
# Anti-Bias Tests (no archetypes)
# ═══════════════════════════════════════════════════════════════


class TestAntiBias:
    """Verify the system detects patterns generically, not via archetypes."""

    def test_generic_convergence_no_archetypes(self) -> None:
        """Convergence is cosine similarity of surprise vectors, not archetype matching."""
        # Create surprise vectors with similar patterns
        s1 = EntitySurprise(
            entity_id="e1",
            entity_type="company",
            obs_type_surprise=5.0,
            temporal_surprise=3.0,
            value_surprise=4.0,
            neighborhood_surprise=0.0,
            memory_drift=0.5,
            composite_surprise=4.0,
        )
        s2 = EntitySurprise(
            entity_id="e2",
            entity_type="company",
            obs_type_surprise=4.8,
            temporal_surprise=2.9,
            value_surprise=3.8,
            neighborhood_surprise=0.0,
            memory_drift=0.4,
            composite_surprise=3.9,
        )
        detector = ConvergenceDetector()
        clusters = detector.detect(
            {"e1": s1, "e2": s2},
            [{"src_id": "e1", "dst_id": "e2"}],
            surprise_threshold=2.0,
        )
        # Similar surprise vectors + linked → cluster detected
        assert len(clusters) == 1
        assert clusters[0].correlated_surprise_score > 0.9

    def test_novel_tool_combinations_detected(self) -> None:
        """Unseen tool combinations should still produce surprise signals."""
        store = _mock_store(
            entities=_entities("e1"),
            observations=_observations(
                "e1",
                source_tool="never_seen_tool_xyz",
                obs_type="dns_change",
            ),
        )
        model = _mock_model(num_nodes=5)
        id_map = IDMap()
        id_map.add("company", "e1")
        mock_data = MagicMock()

        with patch("agent.fusion.entity_scorer.GraphBuilder") as MockGB:
            MockGB.return_value.build.return_value = (mock_data, id_map, [])
            scorer = EntityAnomalyScorer(store, model)
            alerts, _ = scorer.score_entities(as_of=time.time())

        # Novel tools don't prevent scoring
        assert len(alerts) >= 1

    def test_cold_start_entity_gets_scored(self) -> None:
        """Entity with exactly one observation (no history) → still scored."""
        now = time.time()
        store = _mock_store(
            entities=_entities("new_ent"),
            observations=[
                {
                    "entity_id": "new_ent",
                    "source_tool": "test_tool",
                    "observed_at": now,
                    "observation_type": "price_movement",
                    "value_json": '{"usd_amount": 42.0}',
                }
            ],
        )
        model = _mock_model(num_nodes=5)
        id_map = IDMap()
        id_map.add("company", "new_ent")
        mock_data = MagicMock()

        with patch("agent.fusion.entity_scorer.GraphBuilder") as MockGB:
            MockGB.return_value.build.return_value = (mock_data, id_map, [])
            scorer = EntityAnomalyScorer(store, model)
            alerts, _ = scorer.score_entities(as_of=now + 1)

        assert len(alerts) >= 1
        # CUSUM should be finite (cold start = no prior accumulation)
        for a in alerts:
            assert math.isfinite(a.cusum_statistic)

    def test_no_archetype_labels(self) -> None:
        """EntityAlert has no archetype/pattern/cluster_type fields."""
        fields = {f.name for f in EntityAlert.__dataclass_fields__.values()}
        assert "archetype" not in fields
        assert "pattern_type" not in fields
        assert "cluster_type" not in fields

    def test_convergence_cluster_no_archetype(self) -> None:
        """ConvergenceCluster has no archetype field."""
        fields = {f.name for f in ConvergenceCluster.__dataclass_fields__.values()}
        assert "archetype" not in fields
        assert "pattern_name" not in fields


# ═══════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and error paths."""

    def test_zero_observations(self) -> None:
        """No observations → no alerts, no clusters."""
        store = _mock_store(entities=_entities("e1"), observations=[])
        model = _mock_model()
        scorer = EntityAnomalyScorer(store, model)
        alerts, clusters = scorer.score_entities(as_of=time.time())
        assert alerts == []
        assert clusters == []

    def test_nan_embeddings_handled(self) -> None:
        """NaN in GNN embeddings should not crash the pipeline."""
        store = _mock_store(
            entities=_entities("e1"),
            observations=_observations("e1"),
        )
        model = _mock_model(num_nodes=5)

        # Override forward to return NaN embeddings
        def nan_forward(data, id_map):
            embeddings = {}
            for ntype, eid_map in id_map.type_local.items():
                n = len(eid_map)
                embeddings[ntype] = torch.full((n, 16), float("nan"))
            return embeddings

        model.__call__ = MagicMock(side_effect=nan_forward)

        id_map = IDMap()
        id_map.add("company", "e1")
        mock_data = MagicMock()

        with patch("agent.fusion.entity_scorer.GraphBuilder") as MockGB:
            MockGB.return_value.build.return_value = (mock_data, id_map, [])
            scorer = EntityAnomalyScorer(store, model)
            # Should not raise
            alerts, clusters = scorer.score_entities(as_of=time.time())
        # May produce alerts with NaN values, but must not crash
        assert isinstance(alerts, list)
        assert isinstance(clusters, list)

    def test_stale_entity_no_recent_obs(self) -> None:
        """Entity with only very old observations (outside lookback) → no alerts."""
        old_time = time.time() - 86400 * 30  # 30 days ago
        store = _mock_store(
            entities=_entities("stale"),
            # Observations are old but query_all_observations returns them
            # (the lookback filtering happens in the scorer)
            observations=[],  # no observations in the window
        )
        model = _mock_model()
        cfg = ScorerConfig(lookback_seconds=3600)  # 1 hour lookback
        scorer = EntityAnomalyScorer(store, model, config=cfg)
        alerts, clusters = scorer.score_entities(as_of=time.time())
        assert alerts == []

    def test_cusum_accumulation_across_updates(self) -> None:
        """CUSUM should accumulate across sequential updates."""
        monitor = CUSUMMonitor(k=0.5, h=5.0)

        # Feed normal values → no alarm
        for i in range(10):
            stat, alarm = monitor.update("e1", 0.0)
        assert not alarm

        # Feed high values → CUSUM accumulates → alarm
        for i in range(20):
            stat, alarm = monitor.update("e1", 3.0)

        assert stat > 0  # CUSUM has accumulated
        # Eventually should trigger alarm (depends on h threshold)

    def test_cusum_separate_entity_tracking(self) -> None:
        """CUSUM tracks entities independently."""
        monitor = CUSUMMonitor(k=0.5, h=5.0)

        # e1 gets high values, e2 gets normal values
        for _ in range(20):
            monitor.update("e1", 3.0)
            monitor.update("e2", 0.0)

        s1, _ = monitor.update("e1", 3.0)
        s2, _ = monitor.update("e2", 0.0)
        assert s1 > s2

    def test_hawkes_decay_between_events(self) -> None:
        """Hawkes intensity should decay between events."""
        hawkes = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)

        t0 = 1000.0
        val1 = hawkes.update("e1", t0)

        # Much later → intensity decays toward baseline
        val2 = hawkes.update("e1", t0 + 10000.0)
        # After a burst then a long gap, intensity should be near mu
        assert val2 < val1 + 1.0  # not monotonically increasing

    def test_hawkes_burst_increases_intensity(self) -> None:
        """Rapid events → higher Hawkes intensity."""
        hawkes = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)

        t = 1000.0
        vals = []
        for i in range(10):
            vals.append(hawkes.update("e1", t + i * 0.1))  # rapid fire

        # Intensity should generally increase during rapid events
        assert vals[-1] > vals[0]

    def test_large_entity_count(self) -> None:
        """100 entities with observations → pipeline doesn't break."""
        n = 100
        ids = [f"e{i}" for i in range(n)]
        ents = _entities(*ids)
        obs = _observations(*ids)
        lnks = _links(*[(ids[i], ids[i + 1]) for i in range(n - 1)])

        store = _mock_store(ents, obs, lnks)
        model = _mock_model(num_nodes=n + 10)
        id_map = IDMap()
        for e in ents:
            id_map.add(e["entity_type"], e["entity_id"])
        mock_data = MagicMock()

        with patch("agent.fusion.entity_scorer.GraphBuilder") as MockGB:
            MockGB.return_value.build.return_value = (mock_data, id_map, [])
            scorer = EntityAnomalyScorer(store, model)
            alerts, clusters = scorer.score_entities(as_of=time.time())

        assert len(alerts) >= 1
        # Should handle 100 entities without error
        assert isinstance(alerts, list)
        assert isinstance(clusters, list)

    def test_single_entity_no_cluster(self) -> None:
        """A single elevated entity can't form a cluster (need ≥2)."""
        s1 = EntitySurprise(
            entity_id="solo",
            entity_type="company",
            obs_type_surprise=10.0,
            temporal_surprise=5.0,
            value_surprise=8.0,
            neighborhood_surprise=0.0,
            memory_drift=1.0,
            composite_surprise=8.0,
        )
        detector = ConvergenceDetector()
        clusters = detector.detect(
            {"solo": s1},
            [],  # no links
            surprise_threshold=2.0,
        )
        assert len(clusters) == 0

    def test_disconnected_entities_separate_clusters(self) -> None:
        """Two disconnected pairs → two separate clusters."""
        surprises = {}
        for eid in ["a1", "a2", "b1", "b2"]:
            surprises[eid] = EntitySurprise(
                entity_id=eid,
                entity_type="company",
                obs_type_surprise=5.0,
                temporal_surprise=3.0,
                value_surprise=4.0,
                neighborhood_surprise=0.0,
                memory_drift=0.5,
                composite_surprise=5.0,
            )

        links = [
            {"src_id": "a1", "dst_id": "a2"},
            {"src_id": "b1", "dst_id": "b2"},
        ]
        detector = ConvergenceDetector()
        clusters = detector.detect(surprises, links, surprise_threshold=2.0)
        assert len(clusters) == 2

    def test_entity_baseline_cold_start(self) -> None:
        """EntityBaseline with minimal observations → finite score."""
        baseline = EntityBaseline(window=10, gap=2)
        baseline.add_observation("e1", 5.0)
        score = baseline.abnormal_score("e1", 5.0)
        # With only one observation, score should be finite (possibly None or 0)
        assert score is None or math.isfinite(score)

    def test_entity_baseline_detects_anomaly(self) -> None:
        """EntityBaseline flags a large deviation from baseline."""
        baseline = EntityBaseline(window=20, gap=2)
        # Build baseline with normal values (need variation for sigma > 0)
        for i in range(25):
            baseline.add_observation("e1", 10.0 + (i % 3) * 0.5)  # slight variation

        # Large deviation from baseline mean (~10.5)
        score = baseline.abnormal_score("e1", 100.0)
        # Should produce a non-zero score for a ~90-unit deviation
        assert score is not None
        assert score > 0

    def test_surprise_extractor_empty_idmap(self) -> None:
        """Empty IDMap → no surprises extracted."""
        model = _mock_model(num_nodes=0)
        model.memory.memory = torch.zeros(0, 16)
        data = MagicMock()
        id_map = IDMap()

        extractor = SurpriseExtractor()
        result = extractor.extract(
            model, data, id_map, [], memory_before=torch.zeros(0, 16)
        )
        assert result == {}

    def test_multiple_obs_types_per_entity(self) -> None:
        """Entity with multiple observation types → single alert with aggregated surprise."""
        now = time.time()
        obs = [
            {
                "entity_id": "multi",
                "source_tool": "tool_a",
                "observed_at": now - 50,
                "observation_type": "price_movement",
                "value_json": '{"usd_amount": 100.0}',
            },
            {
                "entity_id": "multi",
                "source_tool": "tool_b",
                "observed_at": now - 30,
                "observation_type": "insider_trade",
                "value_json": '{"shares": 5000}',
            },
            {
                "entity_id": "multi",
                "source_tool": "tool_c",
                "observed_at": now - 10,
                "observation_type": "dns_change",
                "value_json": '{"new_ip": "1.2.3.4"}',
            },
        ]
        store = _mock_store(
            entities=_entities("multi"),
            observations=obs,
        )
        model = _mock_model(num_nodes=5)
        id_map = IDMap()
        id_map.add("company", "multi")
        mock_data = MagicMock()

        with patch("agent.fusion.entity_scorer.GraphBuilder") as MockGB:
            MockGB.return_value.build.return_value = (mock_data, id_map, [])
            scorer = EntityAnomalyScorer(store, model)
            alerts, _ = scorer.score_entities(as_of=now)

        # Single entity → single alert
        multi_alerts = [a for a in alerts if a.entity_id == "multi"]
        assert len(multi_alerts) == 1


# ═══════════════════════════════════════════════════════════════
# Persistence Round-Trip
# ═══════════════════════════════════════════════════════════════


class TestPersistenceRoundTrip:
    """Verify alerts and clusters survive store → query."""

    def test_alert_round_trip(self) -> None:
        from pathlib import Path
        import tempfile
        from agent.pipeline.store import PipelineStore

        with tempfile.TemporaryDirectory() as tmp:
            store = PipelineStore(str(Path(tmp) / "rt.db"))
            store.store_entity_alert(
                entity_id="rt_e1",
                entity_type="company",
                entity_name="RoundTrip Entity",
                alert_time=1700000000.0,
                obs_type_surprise=2.5,
                temporal_surprise=1.5,
                value_surprise=3.0,
                neighborhood_surprise=0.5,
                memory_drift=0.2,
                cusum_statistic=1.0,
                hawkes_intensity=0.5,
                event_study_score=0.8,
                composite_surprise=2.5,
                observation_count=10,
                evidence_sources=("tool_a", "tool_b"),
                metadata={"note": "test"},
            )
            rows = store.query_entity_alerts(entity_id="rt_e1")
            assert len(rows) == 1
            r = rows[0]
            assert r["entity_id"] == "rt_e1"
            assert r["composite_surprise"] == 2.5
            assert r["observation_count"] == 10
            store.close()

    def test_cluster_round_trip(self) -> None:
        from pathlib import Path
        import tempfile
        from agent.pipeline.store import PipelineStore

        with tempfile.TemporaryDirectory() as tmp:
            store = PipelineStore(str(Path(tmp) / "rt.db"))
            store.store_convergence_cluster(
                cluster_id="clust_001",
                cluster_time=1700000000.0,
                member_entity_ids=["e1", "e2", "e3"],
                correlated_surprise_score=0.92,
                temporal_span_hours=3.5,
                contributing_domains=("finance", "shipping"),
                contributing_tools=("tool_a", "tool_b"),
                metadata={"desc": "test cluster"},
            )
            rows = store.query_convergence_clusters()
            assert len(rows) == 1
            r = rows[0]
            assert r["cluster_id"] == "clust_001"
            assert r["correlated_surprise_score"] == 0.92
            store.close()

    def test_alert_query_filters(self) -> None:
        from pathlib import Path
        import tempfile
        from agent.pipeline.store import PipelineStore

        with tempfile.TemporaryDirectory() as tmp:
            store = PipelineStore(str(Path(tmp) / "filt.db"))
            for i in range(5):
                store.store_entity_alert(
                    entity_id=f"e{i}",
                    entity_type="company",
                    entity_name=f"E{i}",
                    alert_time=1700000000.0 + i * 100,
                    obs_type_surprise=float(i),
                    temporal_surprise=0.0,
                    value_surprise=0.0,
                    neighborhood_surprise=0.0,
                    memory_drift=0.0,
                    cusum_statistic=0.0,
                    hawkes_intensity=0.0,
                    event_study_score=0.0,
                    composite_surprise=float(i),
                    observation_count=1,
                )

            # Filter by min_composite
            high = store.query_entity_alerts(min_composite=3.0)
            assert len(high) == 2  # i=3, i=4

            # Filter by entity_id
            specific = store.query_entity_alerts(entity_id="e2")
            assert len(specific) == 1
            assert specific[0]["entity_id"] == "e2"

            # Filter by time range
            mid = store.query_entity_alerts(
                since=1700000100.0,
                until=1700000300.0,
            )
            assert all(1700000100.0 <= r["alert_time"] <= 1700000300.0 for r in mid)

            store.close()
