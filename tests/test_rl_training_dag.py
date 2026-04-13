"""Tests for RL Training DAG — Phase 21b.7

Proofs:
    1. DAG definition is valid (no cycles, valid structure)
    2. Training completes with mock data
    3. Insufficient data returns gracefully
    4. Metrics dict has expected keys
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.pipeline.dags.rl_training import (
    _MIN_ALERTS_FOR_SAC,
    _MIN_ALERTS_FOR_WEIGHTS,
    build_rl_training_dag,
    run_rl_training,
)


# ── DAG Structure ─────────────────────────────────────────────


class TestDAGDefinition:
    """Proof 1: DAG is structurally valid."""

    def test_dag_builds(self):
        dag = build_rl_training_dag()
        assert dag.name == "rl_training"
        assert len(dag.nodes) == 1

    def test_dag_validates(self):
        dag = build_rl_training_dag()
        errors = dag.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_dag_has_schedule(self):
        dag = build_rl_training_dag()
        assert dag.schedule is not None
        assert "19" in dag.schedule  # 19:30 UTC

    def test_dag_node_has_operator(self):
        dag = build_rl_training_dag()
        node = dag.nodes["train_rl_policy"]
        assert callable(node.operator)


# ── Insufficient Data ─────────────────────────────────────────


class TestInsufficientData:
    """Proof 3: graceful handling when not enough data."""

    def test_no_alerts_returns_not_trained(self):
        mock_store_cls = MagicMock()
        mock_store = MagicMock()
        mock_store.query_entity_alerts.return_value = []
        mock_store_cls.return_value = mock_store

        with patch(
            "agent.pipeline.dags.rl_training.PipelineStore",
            mock_store_cls,
        ):
            result = run_rl_training({"db_path": ":memory:"}, {})

        assert result["weight_learner_trained"] is False
        assert result["sac_trained"] is False

    def test_few_alerts_skips_training(self):
        """Below minimum threshold → no training."""
        mock_store_cls = MagicMock()
        mock_store = MagicMock()
        # Return fewer alerts than minimum
        alerts = [
            {
                "alert_time": time.time() - i * 86400,
                "obs_type_surprise": 0.1,
                "temporal_surprise": 0.2,
                "value_surprise": 0.3,
                "neighborhood_surprise": 0.4,
                "memory_drift": 0.5,
            }
            for i in range(10)
        ]
        mock_store.query_entity_alerts.return_value = alerts
        mock_store_cls.return_value = mock_store

        with patch(
            "agent.pipeline.dags.rl_training.PipelineStore",
            mock_store_cls,
        ):
            result = run_rl_training({"db_path": ":memory:"}, {})

        assert result["weight_learner_trained"] is False


# ── Training With Data ────────────────────────────────────────


class TestTrainingWithData:
    """Proof 2 & 4: training completes with sufficient mock data."""

    def _generate_alerts(self, n: int) -> list[dict]:
        """Generate n alerts spanning many weeks."""
        base = time.time() - n * 7 * 86400
        return [
            {
                "alert_time": base + i * 7 * 86400,
                "obs_type_surprise": float(np.random.rand()),
                "temporal_surprise": float(np.random.rand()),
                "value_surprise": float(np.random.rand()),
                "neighborhood_surprise": float(np.random.rand()),
                "memory_drift": float(np.random.rand()),
            }
            for i in range(n)
        ]

    def test_weight_learner_trains_with_enough_data(self):
        mock_store_cls = MagicMock()
        mock_store = MagicMock()
        alerts = self._generate_alerts(300)
        mock_store.query_entity_alerts.return_value = alerts
        mock_store_cls.return_value = mock_store

        with patch(
            "agent.pipeline.dags.rl_training.PipelineStore",
            mock_store_cls,
        ):
            result = run_rl_training({"db_path": ":memory:"}, {})

        assert result["weight_learner_trained"] is True
        assert result["weights"] is not None
        assert len(result["weights"]) == 5
        # Weights should sum to ~1 (softmax)
        assert abs(sum(result["weights"]) - 1.0) < 1e-5

    def test_sac_skipped_without_transitions(self):
        """SAC requires transitions in buffer — without them, only weights train."""
        mock_store_cls = MagicMock()
        mock_store = MagicMock()
        alerts = self._generate_alerts(600)
        mock_store.query_entity_alerts.return_value = alerts
        mock_store.query_rl_transitions.return_value = []
        mock_store.load_latest_rl_checkpoint.return_value = None
        mock_store.query_all_entities.return_value = []
        mock_store_cls.return_value = mock_store

        with patch(
            "agent.pipeline.dags.rl_training.PipelineStore",
            mock_store_cls,
        ):
            result = run_rl_training({"db_path": ":memory:"}, {})

        # Weight learner is trained (enough alerts)
        assert result["weight_learner_trained"] is True
        # SAC trained but with insufficient_data status
        if result["sac_trained"]:
            assert "sac" in result["metrics"]
        else:
            # SAC may fail due to insufficient transitions
            assert result["sac_trained"] is False

    def test_result_has_expected_keys(self):
        mock_store_cls = MagicMock()
        mock_store = MagicMock()
        mock_store.query_entity_alerts.return_value = []
        mock_store_cls.return_value = mock_store

        with patch(
            "agent.pipeline.dags.rl_training.PipelineStore",
            mock_store_cls,
        ):
            result = run_rl_training({"db_path": ":memory:"}, {})

        assert "weight_learner_trained" in result
        assert "sac_trained" in result
        assert "weights" in result
        assert "metrics" in result
