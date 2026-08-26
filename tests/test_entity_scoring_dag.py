"""Phase 20 — Step 20.12: Entity Scoring DAG tests.

Validates the entity_scoring pipeline step: DAG structure, scheduling,
graceful skip when model is unavailable, and end-to-end scoring with
mock components.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline.dag import DAG
from agent.pipeline.dags.entity_scoring import (
    DAG_NAME,
    DEPENDS_ON,
    build_entity_scoring_dag,
    run_entity_scoring,
)
from agent.pipeline.store import PipelineStore

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def empty_store(tmp_path: Path) -> PipelineStore:
    return PipelineStore(str(tmp_path / "empty.db"))


@pytest.fixture()
def populated_store(tmp_path: Path) -> PipelineStore:
    """Store with enough entities to pass the threshold."""
    store = PipelineStore(str(tmp_path / "pop.db"))
    import time

    now = time.time()
    for i in range(8):
        store.register_entity(
            entity_id=f"ent_{i}",
            entity_type="company",
            canonical_name=f"Entity {i}",
        )
        store.store_entity_observation(
            entity_id=f"ent_{i}",
            observation_type="price_movement",
            source_tool="test_tool",
            observed_at=now - 3600 + i * 60,
            value=float(i),
        )
    # Add some links
    for i in range(7):
        store.link_entities(
            entity_id_a=f"ent_{i}",
            entity_id_b=f"ent_{i + 1}",
            link_type="correlation",
            source="test",
        )
    return store


# ═══════════════════════════════════════════════════════════════
# Module-level constants
# ═══════════════════════════════════════════════════════════════


class TestConstants:
    def test_dag_name(self) -> None:
        assert DAG_NAME == "entity_scoring"

    def test_depends_on(self) -> None:
        assert ["gnn_inference"] == DEPENDS_ON


# ═══════════════════════════════════════════════════════════════
# DAG structure
# ═══════════════════════════════════════════════════════════════


class TestDAGStructure:
    def test_builds_without_error(self) -> None:
        dag = build_entity_scoring_dag()
        assert isinstance(dag, DAG)

    def test_dag_name(self) -> None:
        dag = build_entity_scoring_dag()
        assert dag.name == "entity_scoring"

    def test_dag_schedule(self) -> None:
        dag = build_entity_scoring_dag()
        assert dag.schedule == "45 18 * * 1-5"

    def test_dag_has_score_node(self) -> None:
        dag = build_entity_scoring_dag()
        assert "score_entities" in dag.nodes

    def test_dag_validates(self) -> None:
        dag = build_entity_scoring_dag()
        errors = dag.validate()
        assert errors == []

    def test_dag_single_node(self) -> None:
        dag = build_entity_scoring_dag()
        assert len(dag.nodes) == 1

    def test_node_operator_is_callable(self) -> None:
        dag = build_entity_scoring_dag()
        node = dag.nodes["score_entities"]
        assert callable(node.operator)

    def test_custom_db_path(self) -> None:
        dag = build_entity_scoring_dag(db_path="/custom/path.db")
        node = dag.nodes["score_entities"]
        assert node.params["db_path"] == "/custom/path.db"

    def test_custom_model_path(self) -> None:
        dag = build_entity_scoring_dag(model_path="/model/hettgn.pt")
        node = dag.nodes["score_entities"]
        assert node.params["model_path"] == "/model/hettgn.pt"


# ═══════════════════════════════════════════════════════════════
# Pipeline ordering
# ═══════════════════════════════════════════════════════════════


class TestPipelineOrdering:
    def test_after_gnn_before_features(self) -> None:
        """entity_scoring at 18:45 after gnn_inference at 18:30, before features at 19:00."""
        from agent.pipeline.dags.gnn_inference import build_gnn_inference_dag

        gnn_dag = build_gnn_inference_dag()
        entity_dag = build_entity_scoring_dag()
        assert gnn_dag.schedule == "30 18 * * 1-5"
        assert entity_dag.schedule == "45 18 * * 1-5"

    def test_before_feature_generation(self) -> None:
        from agent.pipeline.dags.feature_generation import build_feature_generation_dag

        entity_dag = build_entity_scoring_dag()
        feat_dag = build_feature_generation_dag()
        assert entity_dag.schedule == "45 18 * * 1-5"
        assert feat_dag.schedule == "0 19 * * 1-5"


# ═══════════════════════════════════════════════════════════════
# Empty/insufficient store → skip
# ═══════════════════════════════════════════════════════════════


class TestSkipBehavior:
    def test_empty_store_skips(self, empty_store: PipelineStore) -> None:
        result = run_entity_scoring(
            params={"db_path": str(empty_store._db_path)},
            upstream={},
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_entities"
        assert result["alerts_stored"] == 0
        assert result["clusters_stored"] == 0

    def test_insufficient_entities_skips(self, tmp_path: Path) -> None:
        store = PipelineStore(str(tmp_path / "small.db"))
        for i in range(3):
            store.register_entity(
                entity_id=f"e{i}",
                entity_type="company",
                canonical_name=f"E{i}",
            )
        store.close()
        result = run_entity_scoring(
            params={"db_path": str(tmp_path / "small.db")},
            upstream={},
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_entities"
        assert result["entity_count"] == 3

    def test_no_model_file_skips(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        model_path = tmp_path / "nonexistent_model.pt"
        result = run_entity_scoring(
            params={
                "db_path": str(populated_store._db_path),
                "model_path": str(model_path),
            },
            upstream={},
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "model_not_found"
        assert result["entity_count"] == 8

    def test_torch_unavailable_skips(
        self,
        populated_store: PipelineStore,
    ) -> None:
        """When torch can't be imported, skip gracefully."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mock no torch")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": "/fake/model.pt",
                },
                upstream={},
            )
        assert result["status"] == "skipped"
        assert result["reason"] == "torch_not_available"


# ═══════════════════════════════════════════════════════════════
# Model load failure
# ═══════════════════════════════════════════════════════════════


class TestModelLoadFailure:
    def test_corrupt_model_returns_error(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        model_path = tmp_path / "bad_model.pt"
        model_path.write_bytes(b"not a real model")

        result = run_entity_scoring(
            params={
                "db_path": str(populated_store._db_path),
                "model_path": str(model_path),
            },
            upstream={},
        )
        assert result["status"] == "error"
        assert result["reason"] == "model_load_failed"
        assert result["alerts_stored"] == 0


# ═══════════════════════════════════════════════════════════════
# Successful scoring with mocked GNN
# ═══════════════════════════════════════════════════════════════


class TestSuccessfulScoring:
    """Test end-to-end scoring using mocked EntityAnomalyScorer."""

    def _make_mock_alert(self, entity_id: str, score: float):
        """Create a minimal EntityAlert-like object."""
        from agent.fusion.alert import EntityAlert

        return EntityAlert(
            entity_id=entity_id,
            entity_type="company",
            entity_name=f"Name-{entity_id}",
            alert_time=1700000000.0,
            obs_type_surprise=score,
            temporal_surprise=score * 0.5,
            value_surprise=score * 0.3,
            neighborhood_surprise=score * 0.2,
            memory_drift=score * 0.1,
            composite_surprise=score,
            cusum_statistic=0.5,
            hawkes_intensity=0.3,
            event_study_score=0.2,
            observation_count=5,
            evidence_sources=("test_tool",),
        )

    def _make_mock_cluster(self, alerts):
        """Create a minimal ConvergenceCluster-like object."""
        import hashlib

        from agent.fusion.convergence import ConvergenceCluster

        member_ids = sorted(a.entity_id for a in alerts)
        cluster_id = hashlib.sha256("|".join(member_ids).encode()).hexdigest()[:16]
        return ConvergenceCluster(
            cluster_id=cluster_id,
            cluster_time=1700000000.0,
            member_alerts=tuple(alerts),
            correlated_surprise_score=0.85,
            temporal_span_hours=2.5,
            contributing_domains=("finance",),
            contributing_tools=("test_tool",),
        )

    def test_alerts_stored(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        alerts = [self._make_mock_alert("ent_0", 3.5)]
        clusters = []

        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")  # exists, but mocked load

        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = (alerts, clusters)

            result = run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": 1700000000.0,
                },
                upstream={},
            )

        assert result["status"] == "completed"
        assert result["alerts_stored"] == 1
        assert result["alerts_total"] == 1

        # Verify stored in DB
        stored = populated_store.query_entity_alerts(entity_id="ent_0")
        assert len(stored) == 1
        assert stored[0]["entity_id"] == "ent_0"
        assert stored[0]["composite_surprise"] == 3.5

    def test_clusters_stored(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        a1 = self._make_mock_alert("ent_0", 3.5)
        a2 = self._make_mock_alert("ent_1", 4.0)
        cluster = self._make_mock_cluster([a1, a2])
        alerts = [a1, a2]
        clusters = [cluster]

        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = (alerts, clusters)

            result = run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": 1700000000.0,
                },
                upstream={},
            )

        assert result["clusters_stored"] == 1
        assert result["clusters_total"] == 1

        stored = populated_store.query_convergence_clusters()
        assert len(stored) == 1
        assert stored[0]["correlated_surprise_score"] == 0.85

    def test_empty_results(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        """Scorer returns no alerts/clusters → 0 stored."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = ([], [])

            result = run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": 1700000000.0,
                },
                upstream={},
            )

        assert result["status"] == "completed"
        assert result["alerts_stored"] == 0
        assert result["clusters_stored"] == 0

    def test_as_of_passed_to_scorer(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        """Verify as_of is forwarded correctly."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = ([], [])

            run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": 1700000099.0,
                },
                upstream={},
            )

            mock_scorer_inst.score_entities.assert_called_once_with(1700000099.0)

    def test_scorer_config_forwarded(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        """Verify scorer_config params are passed through."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.ScorerConfig") as MockCfg,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = ([], [])

            run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": 1700000000.0,
                    "scorer_config": {"surprise_threshold": 3.0},
                },
                upstream={},
            )

            MockCfg.assert_called_once_with(surprise_threshold=3.0)

    def test_result_includes_as_of(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = ([], [])

            result = run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": 1700000000.0,
                },
                upstream={},
            )

        assert result["as_of"] == 1700000000.0

    def test_alert_store_failure_continues(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        """If one alert fails to store, others should still be stored."""
        alerts = [
            self._make_mock_alert("ent_0", 3.5),
            self._make_mock_alert("ent_1", 2.0),
        ]

        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        call_count = 0
        original_store_alert = PipelineStore.store_entity_alert

        def flaky_store_alert(self_store, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated DB failure")
            return original_store_alert(self_store, *args, **kwargs)

        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
            patch.object(
                PipelineStore,
                "store_entity_alert",
                flaky_store_alert,
            ),
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = (alerts, [])

            result = run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": 1700000000.0,
                },
                upstream={},
            )

        # One failed, one succeeded
        assert result["alerts_stored"] == 1
        assert result["alerts_total"] == 2
        assert result["status"] == "completed"

    def test_cluster_store_failure_continues(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        """If cluster storage fails, continue gracefully."""
        a1 = self._make_mock_alert("ent_0", 3.0)
        a2 = self._make_mock_alert("ent_1", 4.0)
        cluster = self._make_mock_cluster([a1, a2])

        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
            patch.object(
                PipelineStore,
                "store_convergence_cluster",
                side_effect=RuntimeError("DB fail"),
            ),
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = ([a1, a2], [cluster])

            result = run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": 1700000000.0,
                },
                upstream={},
            )

        assert result["clusters_stored"] == 0
        assert result["clusters_total"] == 1
        assert result["alerts_stored"] == 2  # alerts still stored


# ═══════════════════════════════════════════════════════════════
# Idempotency — rerunning the DAG must not double entity_alerts
# ═══════════════════════════════════════════════════════════════


class TestIdempotentRerun:
    """Running the DAG twice for the same scoring day must not double
    entity_alerts. Before the fix, ``store_entity_alert`` was a plain
    INSERT with no key, so a real chain re-run produced 9,704 rows for
    4,852 distinct entities — exactly 2x. ``run_entity_scoring`` now
    clears the UTC calendar day's existing rows before re-inserting.
    """

    _make_mock_alert = TestSuccessfulScoring._make_mock_alert

    def _run_twice(self, populated_store, model_path, alerts, clusters, as_of):
        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = (alerts, clusters)

            params = {
                "db_path": str(populated_store._db_path),
                "model_path": str(model_path),
                "as_of": as_of,
            }
            first = run_entity_scoring(params=params, upstream={})
            second = run_entity_scoring(params=params, upstream={})
        return first, second

    def test_same_as_of_rerun_does_not_double_row_count(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        alerts = [self._make_mock_alert("ent_0", 3.5), self._make_mock_alert("ent_1", 2.0)]
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        first, second = self._run_twice(populated_store, model_path, alerts, [], as_of=1700000000.0)

        assert first["alerts_stored"] == 2
        assert second["alerts_stored"] == 2

        stored = populated_store.query_entity_alerts(limit=100)
        assert len(stored) == 2, f"expected 2 rows after 2 reruns, got {len(stored)} (doubling regression)"
        assert {r["entity_id"] for r in stored} == {"ent_0", "ent_1"}

    def test_different_as_of_same_day_does_not_double(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        """Real reruns via time.time() get a slightly different as_of each
        call (never bit-identical) — the fix must key on the calendar day,
        not the exact timestamp, or this still doubles."""
        alerts = [self._make_mock_alert("ent_0", 3.5)]
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        base = 1700000000.0  # 2023-11-14 22:13:20 UTC
        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst
            mock_scorer_inst.score_entities.return_value = (alerts, [])

            run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": base,
                },
                upstream={},
            )
            run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": base + 45.0,  # same UTC day, 45s later
                },
                upstream={},
            )

        stored = populated_store.query_entity_alerts(limit=100)
        assert len(stored) == 1

    def test_next_day_rerun_preserves_history(
        self,
        populated_store: PipelineStore,
        tmp_path: Path,
    ) -> None:
        """A scoring run on a later day must NOT delete the prior day's
        alerts — inference reads a 7-day rolling window over alert_time,
        so history across days must survive same-day dedup."""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        day1 = 1700000000.0
        day2 = day1 + 86400.0

        with (
            patch("agent.models.gnn.trainer.Trainer") as MockTrainer,
            patch("agent.fusion.entity_scorer.EntityAnomalyScorer") as MockScorer,
        ):
            mock_trainer_inst = MagicMock()
            MockTrainer.load_model.return_value = mock_trainer_inst
            mock_scorer_inst = MagicMock()
            MockScorer.return_value = mock_scorer_inst

            mock_scorer_inst.score_entities.return_value = ([self._make_mock_alert("ent_0", 3.5)], [])
            run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": day1,
                },
                upstream={},
            )

            mock_scorer_inst.score_entities.return_value = ([self._make_mock_alert("ent_1", 2.0)], [])
            run_entity_scoring(
                params={
                    "db_path": str(populated_store._db_path),
                    "model_path": str(model_path),
                    "as_of": day2,
                },
                upstream={},
            )

        stored = populated_store.query_entity_alerts(limit=100)
        assert len(stored) == 2
        assert {r["entity_id"] for r in stored} == {"ent_0", "ent_1"}


# ═══════════════════════════════════════════════════════════════
# DAG registry integration
# ═══════════════════════════════════════════════════════════════


class TestDAGRegistry:
    def test_entity_scoring_in_default_dags(self) -> None:
        """entity_scoring DAG is included in get_default_dags."""
        from agent.pipeline.dags import get_default_dags

        dags = get_default_dags(tool_registry=None)
        names = [d.name for d in dags]
        assert "entity_scoring" in names

    def test_ordering_in_registry(self) -> None:
        """entity_scoring appears after gnn_inference in the DAG list."""
        from agent.pipeline.dags import get_default_dags

        dags = get_default_dags(tool_registry=None)
        names = [d.name for d in dags]
        gnn_idx = names.index("gnn_inference")
        entity_idx = names.index("entity_scoring")
        assert entity_idx > gnn_idx
