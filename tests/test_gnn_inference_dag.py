"""Phase 19d: GNN Inference DAG tests.

Validates the gnn_inference pipeline step: DAG structure, empty/populated
store behavior, model save/load, and pipeline ordering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.pipeline.dag import DAG
from agent.pipeline.dags.gnn_inference import (
    build_gnn_inference_dag,
    run_gnn_inference,
)
from agent.pipeline.store import PipelineStore

# Conditional torch import
torch = pytest.importorskip("torch")

from agent.models.gnn.trainer import (
    InjectedPattern,
    SyntheticGraphGenerator,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def empty_store(tmp_path: Path) -> PipelineStore:
    return PipelineStore(str(tmp_path / "empty.db"))


@pytest.fixture()
def populated_store(tmp_path: Path) -> PipelineStore:
    store = PipelineStore(str(tmp_path / "pop.db"))
    gen = SyntheticGraphGenerator(
        num_companies=6,
        num_countries=3,
        num_vessels=3,
        num_wallets=4,
        time_span=86400.0 * 7,
        base_event_rate=0.005,
        seed=42,
        patterns=[
            InjectedPattern(
                source_type="company",
                source_obs_type="insider_trade",
                target_type="country",
                target_obs_type="geopolitical_event",
                via_edge="headquartered_in",
            ),
        ],
    )
    gen.generate(store)
    return store


@pytest.fixture()
def small_config() -> dict:
    return {
        "hidden_dim": 16,
        "memory_dim": 16,
        "message_dim": 16,
        "time_dim": 8,
        "num_heads": 1,
        "num_layers": 1,
        "epochs": 1,
        "window_size": 86400.0,
    }


# ═══════════════════════════════════════════════════════════════
# DAG structure
# ═══════════════════════════════════════════════════════════════


class TestDAGStructure:
    def test_builds_without_error(self) -> None:
        dag = build_gnn_inference_dag()
        assert isinstance(dag, DAG)

    def test_dag_name(self) -> None:
        dag = build_gnn_inference_dag()
        assert dag.name == "gnn_inference"

    def test_dag_schedule(self) -> None:
        dag = build_gnn_inference_dag()
        assert dag.schedule == "30 18 * * 1-5"

    def test_dag_has_train_node(self) -> None:
        dag = build_gnn_inference_dag()
        assert "train_gnn" in dag.nodes

    def test_dag_validates(self) -> None:
        dag = build_gnn_inference_dag()
        errors = dag.validate()
        assert errors == []


# ═══════════════════════════════════════════════════════════════
# Empty store → skip
# ═══════════════════════════════════════════════════════════════


class TestEmptyStore:
    def test_empty_store_skips(self, empty_store: PipelineStore, tmp_path: Path) -> None:
        result = run_gnn_inference(
            params={
                "db_path": str(empty_store._db_path),
                "model_path": str(tmp_path / "model.pt"),
                "min_entities": 10,
            },
            upstream={},
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_entities"
        assert result["trained"] is False

    def test_empty_store_no_model_saved(self, empty_store: PipelineStore, tmp_path: Path) -> None:
        model_path = tmp_path / "model.pt"
        run_gnn_inference(
            params={
                "db_path": str(empty_store._db_path),
                "model_path": str(model_path),
                "min_entities": 10,
            },
            upstream={},
        )
        assert not model_path.exists()


# ═══════════════════════════════════════════════════════════════
# Populated store → train
# ═══════════════════════════════════════════════════════════════


class TestPopulatedStore:
    def test_trains_and_saves(self, populated_store: PipelineStore, tmp_path: Path, small_config: dict) -> None:
        model_path = tmp_path / "model.pt"
        result = run_gnn_inference(
            params={
                "db_path": str(populated_store._db_path),
                "model_path": str(model_path),
                "min_entities": 2,
                "trainer_config": small_config,
            },
            upstream={},
        )
        assert result["status"] == "completed"
        assert result["trained"] is True
        assert model_path.exists()

    def test_loads_existing_model(self, populated_store: PipelineStore, tmp_path: Path, small_config: dict) -> None:
        model_path = tmp_path / "model.pt"
        # First run trains
        run_gnn_inference(
            params={
                "db_path": str(populated_store._db_path),
                "model_path": str(model_path),
                "min_entities": 2,
                "trainer_config": small_config,
            },
            upstream={},
        )
        # Second run loads
        result = run_gnn_inference(
            params={
                "db_path": str(populated_store._db_path),
                "model_path": str(model_path),
                "min_entities": 2,
                "trainer_config": small_config,
            },
            upstream={},
        )
        assert result["status"] == "completed"
        assert result["trained"] is False

    def test_result_has_entity_count(self, populated_store: PipelineStore, tmp_path: Path, small_config: dict) -> None:
        result = run_gnn_inference(
            params={
                "db_path": str(populated_store._db_path),
                "model_path": str(tmp_path / "model.pt"),
                "min_entities": 2,
                "trainer_config": small_config,
            },
            upstream={},
        )
        assert result["entity_count"] > 0


# ═══════════════════════════════════════════════════════════════
# Pipeline ordering
# ═══════════════════════════════════════════════════════════════


class TestPipelineOrdering:
    def test_gnn_before_features(self) -> None:
        """GNN inference at 18:30 should be before feature_generation at 19:00."""
        gnn_dag = build_gnn_inference_dag()
        assert gnn_dag.schedule == "30 18 * * 1-5"

        from agent.pipeline.dags.feature_generation import build_feature_generation_dag

        feat_dag = build_feature_generation_dag()
        assert feat_dag.schedule == "0 19 * * 1-5"

    def test_features_before_world_model(self) -> None:
        from agent.pipeline.dags.feature_generation import build_feature_generation_dag
        from agent.pipeline.dags.world_model_update import build_world_model_dag

        feat_dag = build_feature_generation_dag()
        wm_dag = build_world_model_dag()
        assert feat_dag.schedule == "0 19 * * 1-5"
        assert wm_dag.schedule == "30 19 * * 1-5"
