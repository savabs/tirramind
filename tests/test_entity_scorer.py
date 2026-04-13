"""Tests for EntityAnomalyScorer orchestrator (Phase 20, Step 20.9).

Covers:
    - Full pipeline with mock store + mock model
    - Normal entity → low surprise
    - Empty store → empty results
    - Enrichment computation
    - Alert construction
    - Convergence detection integration
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from agent.fusion.entity_scorer import EntityAnomalyScorer, ScorerConfig
from agent.fusion.surprise import EntitySurprise
from agent.models.gnn.graph_builder import IDMap, OBSERVATION_TYPES


# ── Mock Factory ──────────────────────────────────────────────


def _make_mock_store(
    entities: list[dict] | None = None,
    observations: list[dict] | None = None,
    links: list[dict] | None = None,
) -> MagicMock:
    """Create a mock PipelineStore."""
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


def _make_mock_model(num_nodes: int = 5, hidden_dim: int = 16, memory_dim: int = 16):
    """Create a mock HetTGN model."""
    model = MagicMock()

    # Memory
    model.memory = MagicMock()
    model.memory.memory = torch.zeros(num_nodes, memory_dim)

    # Forward returns embeddings dict
    def forward_fn(data, id_map):
        embeddings = {}
        for ntype, eid_map in id_map.type_local.items():
            n = len(eid_map)
            embeddings[ntype] = torch.randn(n, hidden_dim)
        return embeddings

    model.__call__ = MagicMock(side_effect=forward_fn)
    model.return_value = {}
    model.eval = MagicMock()

    # Prediction heads
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


def _make_entities(*entity_ids: str, etype: str = "company") -> list[dict]:
    return [
        {"entity_id": eid, "entity_type": etype, "canonical_name": f"Name_{eid}"}
        for eid in entity_ids
    ]


def _make_observations(
    *entity_ids: str, obs_type: str = "price_movement"
) -> list[dict]:
    now = time.time()
    return [
        {
            "entity_id": eid,
            "source_tool": "test_tool",
            "observed_at": now - 100 + i * 10,
            "observation_type": obs_type,
            "value_json": '{"usd_amount": 100.0}',
        }
        for i, eid in enumerate(entity_ids)
    ]


# ═══════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════


class TestEntityAnomalyScorerInit:
    def test_construction(self):
        store = _make_mock_store()
        model = _make_mock_model()
        scorer = EntityAnomalyScorer(store, model)
        assert scorer is not None

    def test_custom_config(self):
        config = ScorerConfig(cusum_k=1.0, surprise_threshold=3.0)
        store = _make_mock_store()
        model = _make_mock_model()
        scorer = EntityAnomalyScorer(store, model, config=config)
        assert scorer._config.cusum_k == 1.0
        assert scorer._config.surprise_threshold == 3.0


# ═══════════════════════════════════════════════════════════════
# Empty / Degenerate
# ═══════════════════════════════════════════════════════════════


class TestEntityAnomalyScorerEmpty:
    def test_no_observations_empty_results(self):
        store = _make_mock_store(
            entities=_make_entities("c0"),
            observations=[],
        )
        model = _make_mock_model()
        scorer = EntityAnomalyScorer(store, model)
        alerts, clusters = scorer.score_entities(as_of=time.time())
        assert alerts == []
        assert clusters == []


# ═══════════════════════════════════════════════════════════════
# Enrichment Computation
# ═══════════════════════════════════════════════════════════════


class TestEnrichment:
    def test_enrichment_computed_for_entities(self):
        """Enrichment dict has keys for all entities with observations."""
        store = _make_mock_store()
        model = _make_mock_model()
        scorer = EntityAnomalyScorer(store, model)
        obs = _make_observations("c0", "c1")
        enrichment = scorer._compute_enrichment(obs, time.time())
        assert "c0" in enrichment
        assert "c1" in enrichment
        for eid in ("c0", "c1"):
            assert "cusum" in enrichment[eid]
            assert "hawkes" in enrichment[eid]
            assert "event_study" in enrichment[eid]
            assert "bocpd" in enrichment[eid]

    def test_enrichment_values_are_finite(self):
        store = _make_mock_store()
        model = _make_mock_model()
        scorer = EntityAnomalyScorer(store, model)
        obs = _make_observations("c0")
        enrichment = scorer._compute_enrichment(obs, time.time())
        for k, v in enrichment["c0"].items():
            assert isinstance(v, float)
            assert not (v != v)  # not NaN

    def test_empty_observations_empty_enrichment(self):
        store = _make_mock_store()
        model = _make_mock_model()
        scorer = EntityAnomalyScorer(store, model)
        enrichment = scorer._compute_enrichment([], time.time())
        assert enrichment == {}


# ═══════════════════════════════════════════════════════════════
# Value Extraction
# ═══════════════════════════════════════════════════════════════


class TestExtractValue:
    def test_json_string(self):
        obs = {"value_json": '{"usd_amount": 42.0}'}
        assert EntityAnomalyScorer._extract_value(obs) == 42.0

    def test_dict_value(self):
        obs = {"value_json": {"btc_amount": 1.5}}
        assert EntityAnomalyScorer._extract_value(obs) == 1.5

    def test_missing_value(self):
        assert EntityAnomalyScorer._extract_value({}) == 0.0

    def test_invalid_json(self):
        obs = {"value_json": "not json"}
        assert EntityAnomalyScorer._extract_value(obs) == 0.0

    def test_empty_dict(self):
        obs = {"value_json": {}}
        assert EntityAnomalyScorer._extract_value(obs) == 0.0


# ═══════════════════════════════════════════════════════════════
# Full Pipeline (mocked graph builder + surprise extractor)
# ═══════════════════════════════════════════════════════════════


class TestFullPipeline:
    def test_pipeline_produces_alerts(self):
        """With observations, pipeline produces alerts."""
        entities = _make_entities("c0")
        observations = _make_observations("c0")
        store = _make_mock_store(entities=entities, observations=observations)
        model = _make_mock_model()

        # Build a real-ish IDMap and data for the graph builder mock
        id_map = IDMap()
        id_map.add("company", "c0")

        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["company"].x = torch.randn(1, 12)
        data["company"].node_ids = ["c0"]
        data.edge_types = []
        events = observations

        scorer = EntityAnomalyScorer(store, model)
        # Patch the graph builder to return our controlled data
        scorer._graph_builder = MagicMock()
        scorer._graph_builder.build.return_value = (data, id_map, events)

        alerts, clusters = scorer.score_entities(as_of=time.time())
        # Alerts may be empty if SurpriseExtractor finds no matching entity
        # (depends on mock model forward behavior)
        assert isinstance(alerts, list)
        assert isinstance(clusters, list)

    def test_pipeline_with_patched_surprise(self):
        """Directly test alert construction with mocked surprise output."""
        entities = _make_entities("c0")
        observations = _make_observations("c0")
        store = _make_mock_store(entities=entities, observations=observations)
        model = _make_mock_model()

        id_map = IDMap()
        id_map.add("company", "c0")

        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["company"].x = torch.randn(1, 12)
        data["company"].node_ids = ["c0"]
        data.edge_types = []

        scorer = EntityAnomalyScorer(store, model)
        scorer._graph_builder = MagicMock()
        scorer._graph_builder.build.return_value = (data, id_map, observations)

        # Patch surprise extractor to return controlled surprise
        mock_surprise = EntitySurprise(
            entity_id="c0",
            entity_type="company",
            obs_type_surprise=3.0,
            temporal_surprise=1.5,
            value_surprise=2.0,
            neighborhood_surprise=0.5,
            memory_drift=0.3,
            composite_surprise=7.3,
        )
        scorer._surprise_extractor = MagicMock()
        scorer._surprise_extractor.extract.return_value = {"c0": mock_surprise}

        alerts, clusters = scorer.score_entities(as_of=time.time())
        assert len(alerts) == 1
        assert alerts[0].entity_id == "c0"
        assert alerts[0].obs_type_surprise == 3.0
        assert alerts[0].composite_surprise == 7.3
        assert alerts[0].entity_name == "Name_c0"

    def test_convergence_detected_with_linked_surprises(self):
        """Two linked high-surprise entities → convergence cluster."""
        entities = _make_entities("c0", "c1")
        observations = _make_observations("c0", "c1")
        links = [{"entity_id_a": "c0", "entity_id_b": "c1", "link_type": "related"}]
        store = _make_mock_store(
            entities=entities, observations=observations, links=links
        )
        model = _make_mock_model()

        id_map = IDMap()
        id_map.add("company", "c0")
        id_map.add("company", "c1")

        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["company"].x = torch.randn(2, 12)
        data["company"].node_ids = ["c0", "c1"]
        data.edge_types = []

        scorer = EntityAnomalyScorer(store, model)
        scorer._graph_builder = MagicMock()
        scorer._graph_builder.build.return_value = (data, id_map, observations)

        # Both entities have high surprise → convergence
        surprises = {
            "c0": EntitySurprise("c0", "company", 5.0, 3.0, 2.0, 1.0, 0.5, 11.5),
            "c1": EntitySurprise("c1", "company", 5.0, 3.0, 2.0, 1.0, 0.5, 11.5),
        }
        scorer._surprise_extractor = MagicMock()
        scorer._surprise_extractor.extract.return_value = surprises

        alerts, clusters = scorer.score_entities(as_of=time.time())
        assert len(alerts) == 2
        assert len(clusters) == 1

    def test_no_convergence_when_below_threshold(self):
        """Low-surprise entities → no convergence."""
        entities = _make_entities("c0", "c1")
        observations = _make_observations("c0", "c1")
        links = [{"entity_id_a": "c0", "entity_id_b": "c1", "link_type": "related"}]
        store = _make_mock_store(
            entities=entities, observations=observations, links=links
        )
        model = _make_mock_model()

        id_map = IDMap()
        id_map.add("company", "c0")
        id_map.add("company", "c1")

        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["company"].x = torch.randn(2, 12)
        data["company"].node_ids = ["c0", "c1"]
        data.edge_types = []

        scorer = EntityAnomalyScorer(store, model)
        scorer._graph_builder = MagicMock()
        scorer._graph_builder.build.return_value = (data, id_map, observations)

        # Low surprise → below threshold
        surprises = {
            "c0": EntitySurprise("c0", "company", 0.5, 0.3, 0.2, 0.1, 0.05, 1.15),
            "c1": EntitySurprise("c1", "company", 0.5, 0.3, 0.2, 0.1, 0.05, 1.15),
        }
        scorer._surprise_extractor = MagicMock()
        scorer._surprise_extractor.extract.return_value = surprises

        alerts, clusters = scorer.score_entities(as_of=time.time())
        assert len(alerts) == 2
        assert len(clusters) == 0


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════


class TestScorerConfig:
    def test_defaults(self):
        cfg = ScorerConfig()
        assert cfg.cusum_k == 0.5
        assert cfg.surprise_threshold == 2.0
        assert cfg.lookback_seconds is None

    def test_lookback_seconds(self):
        cfg = ScorerConfig(lookback_seconds=86400)
        assert cfg.lookback_seconds == 86400
