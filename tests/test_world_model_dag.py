"""
Tests for agent/pipeline/dags/world_model_update.py — pipeline DAG wiring.

Validates:
    - DAG builds without error
    - DAG has correct name and schedule
    - run_world_model_update works with mock store
    - Empty features produce beliefs marked stale
    - Beliefs are persisted to store
"""

from __future__ import annotations

import tempfile
import time

import pytest

from agent.pipeline.dags.world_model_update import (
    build_world_model_dag,
    run_world_model_update,
)
from agent.pipeline.store import PipelineStore


class TestDAGStructure:
    def test_dag_builds(self) -> None:
        dag = build_world_model_dag()
        assert dag.name == "world_model_update"

    def test_dag_schedule(self) -> None:
        dag = build_world_model_dag()
        assert dag.schedule == "30 19 * * 1-5"

    def test_dag_has_update_node(self) -> None:
        dag = build_world_model_dag()
        assert "update_beliefs" in dag.nodes


class TestRunWorldModelUpdate:
    def test_no_features_produces_stale_beliefs(self) -> None:
        """Empty store → all beliefs should be stale."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            result = run_world_model_update(
                params={"db_path": f.name, "as_of": time.time()},
                upstream={},
            )
            assert result["beliefs_count"] > 0
            assert result["features_available"] == 0
            assert result["stale"] > 0

    def test_beliefs_persisted_to_store(self) -> None:
        """Verify beliefs actually end up in SQLite."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            run_world_model_update(
                params={"db_path": f.name, "as_of": time.time()},
                upstream={},
            )
            store = PipelineStore(f.name)
            try:
                # Query for any regime.macro belief
                beliefs = store.query_beliefs("regime.macro")
                assert len(beliefs) >= 1
            finally:
                store.close()

    def test_result_has_graph_hash(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            result = run_world_model_update(
                params={"db_path": f.name, "as_of": time.time()},
                upstream={},
            )
            assert len(result["graph_hash"]) == 64

    def test_with_synthetic_features(self) -> None:
        """Store synthetic features, then verify world model uses them."""
        from agent.features.protocol import EngineeredFeature

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = PipelineStore(f.name)
            t = time.time()

            # Store a single feature
            feat = EngineeredFeature(
                feature_name="macro.rate_momentum.30d",
                version=1,
                effective_at=t - 100,
                computed_at=t - 50,
                horizon="30d",
                value=0.8,
                quality=1.0,
                source_signals=("test_signal",),
                builder="test_builder",
            )
            store.store_feature(feat)
            store.close()

            result = run_world_model_update(
                params={"db_path": f.name, "as_of": t},
                upstream={},
            )
            assert result["features_available"] >= 1
            assert result["beliefs_count"] == 23  # 20 DAG + 3 Kalman

    def test_idempotent_rerun(self) -> None:
        """Running twice should not fail (upsert on unique index)."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            params = {"db_path": f.name, "as_of": time.time()}
            r1 = run_world_model_update(params, {})
            r2 = run_world_model_update(params, {})
            assert r1["beliefs_count"] == r2["beliefs_count"]
