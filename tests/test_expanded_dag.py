"""Phase 19c: Expanded DAG edge case tests.

Validates the 20-node, 19-edge Bayesian DAG and the expanded 17-dim
Kalman filter introduced in Phase 19c.
"""

from __future__ import annotations

import math
import tempfile
import time

import numpy as np
import pytest

from agent.features.protocol import EngineeredFeature
from agent.models.belief import BeliefState
from agent.models.initial_graph import (
    ALL_EDGES,
    ALL_NODES,
    build_initial_graph,
)
from agent.models.graph import WorldModelGraph
from agent.models.propagator import BeliefPropagator
from agent.pipeline.dags.world_model_update import (
    _FEATURE_TO_OBS_INDEX,
    _OBS_DIM,
    _STATE_DIM,
    _build_world_model,
    run_world_model_update,
)
from agent.pipeline.store import PipelineStore


# ── helpers ────────────────────────────────────────────────────


def _make_feature(name: str, value: float | None) -> EngineeredFeature:
    return EngineeredFeature(
        feature_name=name,
        version=1,
        effective_at=time.time(),
        computed_at=time.time(),
        horizon="spot",
        value=value,
        quality=1.0 if value is not None else 0.0,
        source_signals=("test_signal",),
        builder="test_builder",
        unit="z_score",
    )


# ═══════════════════════════════════════════════════════════════
# DAG structure
# ═══════════════════════════════════════════════════════════════


class TestExpandedDAGStructure:
    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return build_initial_graph()

    def test_node_count(self, graph: WorldModelGraph) -> None:
        assert len(graph.node_names) == 20

    def test_edge_count(self, graph: WorldModelGraph) -> None:
        assert len(graph.edges) == 19

    def test_gnn_anomaly_nodes_present(self, graph: WorldModelGraph) -> None:
        names = set(graph.node_names)
        for t in ("person", "company", "wallet", "country", "vessel"):
            assert f"obs.{t}_anomaly" in names

    def test_gnn_activity_nodes_present(self, graph: WorldModelGraph) -> None:
        names = set(graph.node_names)
        for t in ("person", "company", "wallet", "country", "vessel"):
            assert f"obs.{t}_activity" in names

    def test_cross_entity_present(self, graph: WorldModelGraph) -> None:
        assert "obs.cross_entity" in graph.node_names

    def test_anomaly_parent_is_stress(self, graph: WorldModelGraph) -> None:
        for t in ("person", "company", "wallet", "country", "vessel"):
            parents = set(graph.get_parents(f"obs.{t}_anomaly"))
            assert parents == {"regime.stress"}

    def test_person_company_wallet_activity_parent(
        self, graph: WorldModelGraph
    ) -> None:
        for t in ("person", "company", "wallet"):
            parents = set(graph.get_parents(f"obs.{t}_activity"))
            assert parents == {"latent.risk_appetite"}

    def test_country_vessel_activity_are_root(self, graph: WorldModelGraph) -> None:
        """country_activity and vessel_activity have no parent edges."""
        for t in ("country", "vessel"):
            parents = graph.get_parents(f"obs.{t}_activity")
            assert parents == []

    def test_cross_entity_is_root(self, graph: WorldModelGraph) -> None:
        parents = graph.get_parents("obs.cross_entity")
        assert parents == []

    def test_all_gnn_nodes_domain_entity(self, graph: WorldModelGraph) -> None:
        for spec in graph.node_specs.values():
            if spec.name.startswith("obs.") and spec.domain == "entity":
                assert spec.cardinality == 3
                assert spec.states == ("low", "normal", "high")

    def test_gnn_bin_edges_centered(self, graph: WorldModelGraph) -> None:
        """GNN nodes use z-score bins: (-inf, -1, 1, inf)."""
        for spec in graph.node_specs.values():
            if spec.domain == "entity" and spec.bin_edges is not None:
                assert spec.bin_edges == (-math.inf, -1.0, 1.0, math.inf)


# ═══════════════════════════════════════════════════════════════
# CPD validity
# ═══════════════════════════════════════════════════════════════


class TestExpandedCPDs:
    @pytest.fixture
    def graph(self) -> WorldModelGraph:
        return build_initial_graph()

    def test_every_node_has_cpd(self, graph: WorldModelGraph) -> None:
        for name in graph.node_names:
            assert graph.get_cpd(name) is not None, f"{name} missing CPD"

    def test_cpds_sum_to_one(self, graph: WorldModelGraph) -> None:
        for name in graph.node_names:
            cpd = graph.get_cpd(name)
            vals = cpd.get_values()
            col_sums = vals.sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_cpds_non_negative(self, graph: WorldModelGraph) -> None:
        for name in graph.node_names:
            cpd = graph.get_cpd(name)
            assert (cpd.get_values() >= 0).all(), f"{name} has negative CPD values"

    def test_anomaly_extreme_stress_favours_high(self, graph: WorldModelGraph) -> None:
        """Under extreme stress, anomaly should be biased toward 'high'."""
        cpd = graph.get_cpd("obs.person_anomaly")
        vals = cpd.get_values()
        # Column 2 = extreme stress
        assert vals[2, 2] > vals[0, 2]  # P(high|extreme) > P(low|extreme)

    def test_activity_risk_on_favours_high(self, graph: WorldModelGraph) -> None:
        """Under risk_on, activity should be biased toward 'high'."""
        cpd = graph.get_cpd("obs.person_activity")
        vals = cpd.get_values()
        # Column 0 = risk_on
        assert vals[2, 0] > vals[0, 0]  # P(high|risk_on) > P(low|risk_on)

    def test_cross_entity_prior_center_biased(self, graph: WorldModelGraph) -> None:
        cpd = graph.get_cpd("obs.cross_entity")
        vals = cpd.get_values().flatten()
        assert vals[1] > vals[0]  # P(normal) > P(low)
        assert vals[1] > vals[2]  # P(normal) > P(high)

    def test_pgmpy_check_model(self, graph: WorldModelGraph) -> None:
        """pgmpy's own internal validation passes."""
        assert graph._bn.check_model()


# ═══════════════════════════════════════════════════════════════
# Belief propagation with GNN evidence
# ═══════════════════════════════════════════════════════════════


class TestGNNEvidencePropagation:
    def test_anomaly_evidence_affects_stress(self) -> None:
        """Observing high anomaly should shift stress toward extreme."""
        graph = build_initial_graph()
        prop = BeliefPropagator(graph)
        beliefs = prop.propagate(
            evidence={"obs.person_anomaly": "high"},
            as_of=time.time(),
        )
        stress = next(b for b in beliefs if b.variable_name == "regime.stress")
        # P(extreme|high anomaly) should be higher than P(extreme) under prior
        prior_beliefs = prop.propagate_priors(as_of=time.time())
        stress_prior = next(
            b for b in prior_beliefs if b.variable_name == "regime.stress"
        )
        assert stress.probabilities["extreme"] > stress_prior.probabilities["extreme"]

    def test_activity_evidence_affects_risk_appetite(self) -> None:
        graph = build_initial_graph()
        prop = BeliefPropagator(graph)
        beliefs = prop.propagate(
            evidence={"obs.person_activity": "high"},
            as_of=time.time(),
        )
        ra = next(b for b in beliefs if b.variable_name == "latent.risk_appetite")
        prior_beliefs = prop.propagate_priors(as_of=time.time())
        ra_prior = next(
            b for b in prior_beliefs if b.variable_name == "latent.risk_appetite"
        )
        assert ra.probabilities["risk_on"] > ra_prior.probabilities["risk_on"]

    def test_all_anomaly_high_shifts_stress_extreme(self) -> None:
        """All anomaly nodes high → strong shift to extreme stress."""
        graph = build_initial_graph()
        prop = BeliefPropagator(graph)
        evidence = {
            f"obs.{t}_anomaly": "high"
            for t in ("person", "company", "wallet", "country", "vessel")
        }
        beliefs = prop.propagate(evidence=evidence, as_of=time.time())
        stress = next(b for b in beliefs if b.variable_name == "regime.stress")
        assert stress.probabilities["extreme"] > 0.5

    def test_cross_entity_evidence_no_crash(self) -> None:
        """Cross-entity is a root node — evidence should still work."""
        graph = build_initial_graph()
        prop = BeliefPropagator(graph)
        beliefs = prop.propagate(
            evidence={"obs.cross_entity": "high"},
            as_of=time.time(),
        )
        assert len(beliefs) == 20


# ═══════════════════════════════════════════════════════════════
# Kalman filter expansion
# ═══════════════════════════════════════════════════════════════


class TestExpandedKalman:
    def test_obs_dim_is_17(self) -> None:
        assert _OBS_DIM == 17

    def test_state_dim_is_3(self) -> None:
        assert _STATE_DIM == 3

    def test_feature_index_count(self) -> None:
        assert len(_FEATURE_TO_OBS_INDEX) == 17

    def test_gnn_features_mapped(self) -> None:
        for t in ("person", "company", "wallet", "country", "vessel"):
            assert f"gnn.{t}_anomaly.spot" in _FEATURE_TO_OBS_INDEX
            assert f"gnn.{t}_activity.spot" in _FEATURE_TO_OBS_INDEX
        assert "gnn.cross_entity.spot" in _FEATURE_TO_OBS_INDEX

    def test_world_model_builds_with_expanded_filter(self) -> None:
        wm = _build_world_model()
        assert wm._filter.obs_dim == 17

    def test_gnn_features_produce_beliefs(self) -> None:
        wm = _build_world_model()
        features = [
            _make_feature("gnn.person_anomaly.spot", 2.0),
            _make_feature("gnn.company_anomaly.spot", 1.5),
        ]
        beliefs = wm.update(features, time.time())
        assert len(beliefs) == 23  # 20 DAG + 3 Kalman

    def test_all_17_features_accepted(self) -> None:
        wm = _build_world_model()
        features = [_make_feature(name, 0.5) for name in _FEATURE_TO_OBS_INDEX]
        beliefs = wm.update(features, time.time())
        non_stale = [b for b in beliefs if not b.stale]
        assert len(non_stale) > 0

    def test_h_matrix_gnn_anomaly_maps_to_stress(self) -> None:
        wm = _build_world_model()
        H = wm._filter._H
        # GNN anomaly obs indices 6-10 should map to state dim 0 (stress_level)
        for i in range(6, 11):
            assert H[i, 0] == pytest.approx(0.5)

    def test_h_matrix_gnn_activity_maps_to_momentum(self) -> None:
        wm = _build_world_model()
        H = wm._filter._H
        # GNN activity obs indices 11-15 should map to state dim 1 (macro_momentum)
        for i in range(11, 16):
            assert H[i, 1] == pytest.approx(0.3)

    def test_h_matrix_cross_entity_maps_to_liquidity(self) -> None:
        wm = _build_world_model()
        H = wm._filter._H
        assert H[16, 2] == pytest.approx(0.4)

    def test_r_matrix_gnn_higher_noise(self) -> None:
        wm = _build_world_model()
        R = wm._filter._R
        # Original features: noise 0.1
        for i in range(6):
            assert R[i, i] == pytest.approx(0.1)
        # GNN features: noise 0.3
        for i in range(6, 17):
            assert R[i, i] == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════════
# Pipeline integration
# ═══════════════════════════════════════════════════════════════


class TestExpandedPipeline:
    def test_run_with_gnn_feature_in_store(self) -> None:
        """Store a GNN feature into the pipeline store, run world model update."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = PipelineStore(f.name)
            feat = _make_feature("gnn.person_anomaly.spot", 1.5)
            store.store_feature(feat)
            store.close()

            result = run_world_model_update(
                params={"db_path": f.name, "as_of": time.time()},
                upstream={},
            )
            assert result["features_available"] >= 1
            assert result["beliefs_count"] == 23

    def test_run_with_all_17_features(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = PipelineStore(f.name)
            for name in _FEATURE_TO_OBS_INDEX:
                store.store_feature(_make_feature(name, 0.5))
            store.close()

            result = run_world_model_update(
                params={"db_path": f.name, "as_of": time.time()},
                upstream={},
            )
            assert result["features_available"] == 17
            assert result["beliefs_count"] == 23
