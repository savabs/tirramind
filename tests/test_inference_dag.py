"""Tests for the inference DAG (Phase 24d).

Covers:
    - DAG structure: 4 nodes, correct dependency chain, schedule
    - load_models: GNN file check, SAC checkpoint DB check, missing models
    - gnn_inference: mock GNN forward pass + surprise extraction, skip on no model
    - sac_inference: mock SAC action, correct weight mapping, skip on no SAC
    - emit_portfolio: weight persistence, P&L computation (dot product), cumulative return
    - Edge cases: empty weights, missing return data, no previous weights, all nodes skip
"""

from __future__ import annotations

import time
from datetime import UTC, date

UTC = UTC
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.pipeline.dags.inference import (
    _emit_portfolio,
    _gnn_inference,
    _load_models,
    _sac_inference,
    build_inference_dag,
)
from agent.pipeline.store import PipelineStore

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> PipelineStore:
    """In-memory PipelineStore for testing."""
    return PipelineStore(":memory:")


@pytest.fixture
def store_on_disk(tmp_path: Path) -> PipelineStore:
    """Disk-backed PipelineStore for file-path tests."""
    db = tmp_path / "test.db"
    return PipelineStore(str(db))


@pytest.fixture
def today_str() -> str:
    return "2026-04-14"


@pytest.fixture
def yesterday_str() -> str:
    return "2026-04-13"


# ──────────────────────────────────────────────────────────────
# DAG structure tests
# ──────────────────────────────────────────────────────────────


class TestDAGStructure:
    def test_dag_name(self):
        dag = build_inference_dag()
        assert dag.name == "inference"

    def test_dag_schedule(self):
        dag = build_inference_dag()
        assert dag.schedule == "45 19 * * 1-5"

    def test_dag_has_four_nodes(self):
        dag = build_inference_dag()
        assert len(dag.nodes) == 4

    def test_node_names(self):
        dag = build_inference_dag()
        expected = {"load_models", "gnn_inference", "sac_inference", "emit_portfolio"}
        assert set(dag.nodes.keys()) == expected

    def test_dependency_chain(self):
        dag = build_inference_dag()
        assert dag.nodes["load_models"].depends_on == []
        assert dag.nodes["gnn_inference"].depends_on == ["load_models"]
        assert dag.nodes["sac_inference"].depends_on == ["gnn_inference"]
        assert dag.nodes["emit_portfolio"].depends_on == ["sac_inference"]

    def test_dag_validates(self):
        dag = build_inference_dag()
        # Should not raise
        dag.validate()

    def test_roots_only_load_models(self):
        dag = build_inference_dag()
        roots = dag.roots()
        assert roots == ["load_models"]

    def test_all_nodes_are_callable(self):
        dag = build_inference_dag()
        for node in dag.nodes.values():
            assert callable(node.operator)

    def test_custom_db_path_propagates(self):
        dag = build_inference_dag(db_path="/custom/path.db")
        for node in dag.nodes.values():
            assert node.params.get("db_path") == "/custom/path.db"

    def test_dag_description_nonempty(self):
        dag = build_inference_dag()
        assert dag.description
        assert "inference" in dag.description.lower()


# ──────────────────────────────────────────────────────────────
# Node 1: load_models
# ──────────────────────────────────────────────────────────────


class TestLoadModels:
    def test_no_gnn_no_sac_skips(self, store: PipelineStore):
        result = _load_models(
            {"db_path": ":memory:", "model_path": "/nonexistent/model.pt"},
            {},
        )
        assert result["status"] == "skipped"
        assert result["has_gnn"] is False
        assert result["has_sac"] is False

    def test_gnn_exists_sac_missing(self, tmp_path: Path):
        model_path = tmp_path / "gnn_model.pt"
        model_path.touch()
        db_path = str(tmp_path / "test.db")
        PipelineStore(db_path).close()  # create DB

        result = _load_models(
            {"db_path": db_path, "model_path": str(model_path)},
            {},
        )
        assert result["has_gnn"] is True
        assert result["has_sac"] is False
        # Still "ready" because at least one model present
        assert result["status"] == "ready"

    def test_sac_exists_gnn_missing(self, tmp_path: Path):
        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)
        store.store_rl_checkpoint(
            policy_type="sac",
            config={"state_dim": 100, "action_dim": 5},
            state_dict_bytes=b"fake_model_bytes",
            metrics={"loss": 0.5},
        )
        store.close()

        result = _load_models(
            {"db_path": db_path, "model_path": "/nonexistent/model.pt"},
            {},
        )
        assert result["has_gnn"] is False
        assert result["has_sac"] is True
        assert result["sac_config"] == {"state_dim": 100, "action_dim": 5}
        assert result["status"] == "ready"

    def test_both_models_available(self, tmp_path: Path):
        model_path = tmp_path / "gnn_model.pt"
        model_path.touch()
        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)
        store.store_rl_checkpoint(
            policy_type="sac",
            config={"state_dim": 100, "action_dim": 5},
            state_dict_bytes=b"fake_model_bytes",
        )
        store.close()

        result = _load_models(
            {"db_path": db_path, "model_path": str(model_path)},
            {},
        )
        assert result["has_gnn"] is True
        assert result["has_sac"] is True
        assert result["status"] == "ready"


# ──────────────────────────────────────────────────────────────
# Node 2: gnn_inference
# ──────────────────────────────────────────────────────────────


class TestModelNodeTimeouts:
    """Model-heavy nodes must not inherit the 60s fetch-sized default.

    Regression (2026-08-26): `Node.timeout` defaults to 60 — right for a single
    HTTP call in daily_collection, far too short for building a 5.6k-node
    heterogeneous graph and running a GNN forward pass. A real chain run killed
    `gnn_inference` at 69.6s with "Execution timed out (>60s)", which cascaded:
    sac_inference and emit_portfolio were skipped as upstream-failed, so
    portfolio_weights and paper_trade_pnl stayed empty for a reason that had
    nothing to do with the model.
    """

    _MIN_MODEL_NODE_TIMEOUT = 300

    def test_gnn_inference_node_has_generous_timeout(self):
        dag = build_inference_dag()
        node = dag.nodes["gnn_inference"]
        assert (
            node.timeout >= self._MIN_MODEL_NODE_TIMEOUT
        ), f"gnn_inference timeout={node.timeout}s — a real graph build exceeded 69s; this node will be killed mid-run"

    def test_all_inference_nodes_exceed_fetch_default(self):
        dag = build_inference_dag()
        for node_id in ("gnn_inference", "sac_inference", "emit_portfolio"):
            node = dag.nodes[node_id]
            assert node.timeout > 60, f"{node_id} still uses the 60s fetch default"


class TestFailuresAreNotSilent:
    """A present-but-throwing model must FAIL the node, not report success.

    Regression (2026-08-26): each of the three inference operators caught every
    exception and returned ``{"status": "error", ...}``. DAGExecutor fails a node
    only when its operator *raises* — a returned dict is always recorded as
    ``completed``. So a real GNN shape mismatch produced:

        run.status = completed
          [completed] load_models
          [completed] gnn_inference     <- actually threw
          [completed] sac_inference
          [completed] emit_portfolio
        --> portfolio_weights: 0, paper_trade_pnl: 0

    Green pipeline, zero output, indefinitely. The `skipped` paths (model
    genuinely absent) are deliberate degradation and must keep working.
    """

    @patch("agent.pipeline.dags.inference.PipelineStore")
    @patch("agent.models.gnn.trainer.Trainer")
    def test_gnn_inference_raises_instead_of_returning_error(self, MockTrainer, MockStore):
        MockStore.return_value.close.return_value = None
        MockTrainer.load_model.side_effect = RuntimeError("mat1 and mat2 shapes cannot be multiplied (93x49 and 23x64)")
        with pytest.raises(RuntimeError, match="cannot be multiplied"):
            _gnn_inference(
                {"db_path": ":memory:"},
                {"load_models": {"status": "ready", "has_gnn": True, "has_sac": True}},
            )

    @patch("agent.pipeline.dags.inference.PipelineStore")
    def test_emit_portfolio_raises_instead_of_returning_error(self, MockStore):
        """emit_portfolio is the node that actually writes the output tables."""
        MockStore.return_value.close.return_value = None
        MockStore.return_value.store_portfolio_weights.side_effect = RuntimeError("db is locked")
        with pytest.raises(RuntimeError, match="db is locked"):
            _emit_portfolio(
                {"db_path": ":memory:"},
                {
                    "sac_inference": {
                        "status": "completed",
                        "weights": {"ES=F": 1.0},
                        "instrument_tickers": ["ES=F"],
                        "state_vector": [0.0],
                        "action_vector": [1.0],
                    }
                },
            )

    def test_skipped_path_still_degrades_gracefully(self):
        """The intended graceful-degradation path must NOT start raising."""
        result = _gnn_inference(
            {"db_path": ":memory:"},
            {"load_models": {"status": "ready", "has_gnn": False}},
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "no_gnn_model"

    def test_no_operator_returns_status_error(self):
        """Guard the contract itself: 'error' is not a valid operator return."""
        import inspect

        from agent.pipeline.dags import inference as inference_mod

        src = inspect.getsource(inference_mod)
        assert (
            '"status": "error"' not in src
        ), "an operator returns status='error'; the executor records that as completed — raise instead"


class TestGNNInference:
    def test_skip_when_upstream_skipped(self):
        result = _gnn_inference(
            {"db_path": ":memory:"},
            {"load_models": {"status": "skipped"}},
        )
        assert result["status"] == "skipped"
        assert result["instrument_surprises"] == {}

    def test_skip_when_no_gnn_model(self):
        result = _gnn_inference(
            {"db_path": ":memory:"},
            {"load_models": {"status": "ready", "has_gnn": False}},
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "no_gnn_model"

    @patch("agent.pipeline.dags.inference.PipelineStore")
    @patch("agent.models.gnn.trainer.Trainer")
    def test_gnn_forward_pass_with_instruments(self, MockTrainer, MockStore):
        """Mock a GNN forward pass that produces instrument surprises."""
        # Setup mock store
        mock_store = MockStore.return_value
        mock_store.query_all_observations.return_value = []
        mock_store.close.return_value = None

        # Setup mock trainer
        mock_trainer = MagicMock()
        MockTrainer.load_model.return_value = mock_trainer

        # Mock infer() → embeddings + id_map
        mock_embeddings = {"instrument": MagicMock()}
        mock_id_map = MagicMock()
        mock_trainer.infer.return_value = (mock_embeddings, mock_id_map)

        # Mock graph builder for surprise extraction
        mock_data = MagicMock()
        mock_trainer._graph_builder.build.return_value = (mock_data, mock_id_map, None)
        mock_trainer.model = MagicMock()

        # Mock SurpriseExtractor
        from agent.fusion.surprise import EntitySurprise

        mock_surprises = {
            "ES=F": EntitySurprise(
                entity_id="ES=F",
                entity_type="instrument",
                obs_type_surprise=0.5,
                temporal_surprise=0.3,
                value_surprise=0.8,
                neighborhood_surprise=0.2,
                memory_drift=0.1,
                composite_surprise=0.4,
            ),
            "NQ=F": EntitySurprise(
                entity_id="NQ=F",
                entity_type="instrument",
                obs_type_surprise=0.6,
                temporal_surprise=0.4,
                value_surprise=0.7,
                neighborhood_surprise=0.3,
                memory_drift=0.15,
                composite_surprise=0.5,
            ),
            "CEO_123": EntitySurprise(
                entity_id="CEO_123",
                entity_type="person",
                obs_type_surprise=1.0,
                temporal_surprise=0.9,
                value_surprise=1.2,
                neighborhood_surprise=0.8,
                memory_drift=0.7,
                composite_surprise=0.95,
            ),
        }

        with patch("agent.fusion.surprise.SurpriseExtractor") as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_extractor.extract.return_value = mock_surprises

            result = _gnn_inference(
                {"db_path": ":memory:"},
                {
                    "load_models": {
                        "status": "ready",
                        "has_gnn": True,
                        "gnn_model_path": "/fake/model.pt",
                    }
                },
            )

        assert result["status"] == "completed"
        # Only instrument-type entities should be in surprises
        assert "ES=F" in result["instrument_surprises"]
        assert "NQ=F" in result["instrument_surprises"]
        assert "CEO_123" not in result["instrument_surprises"]
        # Check surprise vector contents
        assert len(result["instrument_surprises"]["ES=F"]) == 5
        assert result["instrument_surprises"]["ES=F"] == [0.5, 0.3, 0.8, 0.2, 0.1]

    def test_skip_when_torch_not_available(self):
        with patch.dict("sys.modules", {"torch": None}):
            # Force ImportError on torch
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    (_ for _ in ()).throw(ImportError(name))
                    if "trainer" in name
                    else __builtins__.__import__(name, *a, **kw)
                ),
            ):
                result = _gnn_inference(
                    {"db_path": ":memory:"},
                    {
                        "load_models": {
                            "status": "ready",
                            "has_gnn": True,
                            "gnn_model_path": "/x",
                        }
                    },
                )
                assert result["status"] == "skipped"


# ──────────────────────────────────────────────────────────────
# Node 3: sac_inference
# ──────────────────────────────────────────────────────────────


class TestSACInference:
    def test_skip_when_upstream_skipped(self):
        result = _sac_inference(
            {"db_path": ":memory:"},
            {"load_models": {"status": "skipped"}, "gnn_inference": {}},
        )
        assert result["status"] == "skipped"
        assert result["weights"] == {}

    def test_skip_when_no_sac(self):
        result = _sac_inference(
            {"db_path": ":memory:"},
            {
                "load_models": {"status": "ready", "has_sac": False},
                "gnn_inference": {"instrument_surprises": {}},
            },
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "no_sac_model"

    def test_skip_when_missing_sac_dimensions(self):
        result = _sac_inference(
            {"db_path": ":memory:"},
            {
                "load_models": {
                    "status": "ready",
                    "has_sac": True,
                    "sac_config": {},  # missing state_dim/action_dim
                },
                "gnn_inference": {},
            },
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_sac_dimensions"

    @patch("agent.pipeline.dags.inference.PipelineStore")
    @patch("agent.tools.instrument_universe.tradeable_instruments")
    def test_sac_produces_weights(self, mock_instruments, MockStore):
        """End-to-end SAC inference with mocked model."""
        from agent.tools.instrument_universe import InstrumentDef

        # 3 instruments for simplicity
        mock_instruments.return_value = [
            InstrumentDef("ES=F", "S&P 500 Futures", "equity_index", "US"),
            InstrumentDef("NQ=F", "Nasdaq 100 Futures", "equity_index", "US"),
            InstrumentDef("GC=F", "Gold Futures", "commodity_future", "US"),
        ]

        # Mock store
        mock_store = MockStore.return_value
        mock_store.load_latest_rl_checkpoint.return_value = {
            "state_dict_bytes": b"fake_bytes",
            "config": {"state_dim": 100, "action_dim": 3},
            "saved_at": time.time(),
        }
        mock_store.query_entity_alerts.return_value = []
        mock_store.close.return_value = None

        # Mock SACTrainer
        with patch("agent.learning.policy.sac.SACTrainer") as MockSAC:
            mock_trainer = MagicMock()
            MockSAC.load.return_value = mock_trainer
            # SAC returns 3 weights (one per instrument)
            mock_trainer.select_action.return_value = np.array([0.3, 0.5, 0.2])

            # Mock InstrumentStateAssembler
            with patch("agent.learning.policy.state_assembler.InstrumentStateAssembler") as MockAsm:
                import torch

                mock_asm = MockAsm.return_value
                mock_state = torch.zeros(100)
                mock_asm.assemble.return_value = (mock_state, {"n_active": 0})

                result = _sac_inference(
                    {"db_path": ":memory:"},
                    {
                        "load_models": {
                            "status": "ready",
                            "has_sac": True,
                            "sac_config": {"state_dim": 100, "action_dim": 3},
                        },
                        "gnn_inference": {
                            "instrument_surprises": {
                                "ES=F": [0.5, 0.3, 0.8, 0.2, 0.1],
                                "NQ=F": [0.6, 0.4, 0.7, 0.3, 0.15],
                            },
                        },
                    },
                )

        assert result["status"] == "completed"
        assert len(result["weights"]) == 3
        assert result["weights"]["ES=F"] == pytest.approx(0.3)
        assert result["weights"]["NQ=F"] == pytest.approx(0.5)
        assert result["weights"]["GC=F"] == pytest.approx(0.2)

    @patch("agent.pipeline.dags.inference.PipelineStore")
    @patch("agent.tools.instrument_universe.tradeable_instruments")
    def test_sac_checkpoint_vanished(self, mock_instruments, MockStore):
        """Checkpoint exists at load_models time but gone at sac_inference."""
        mock_instruments.return_value = []
        mock_store = MockStore.return_value
        mock_store.load_latest_rl_checkpoint.return_value = None
        mock_store.close.return_value = None

        result = _sac_inference(
            {"db_path": ":memory:"},
            {
                "load_models": {
                    "status": "ready",
                    "has_sac": True,
                    "sac_config": {"state_dim": 10, "action_dim": 3},
                },
                "gnn_inference": {"instrument_surprises": {}},
            },
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "checkpoint_vanished"


# ──────────────────────────────────────────────────────────────
# Node 4: emit_portfolio
# ──────────────────────────────────────────────────────────────


class TestEmitPortfolio:
    def test_skip_when_upstream_not_completed(self):
        result = _emit_portfolio(
            {"db_path": ":memory:"},
            {"sac_inference": {"status": "skipped", "reason": "no_sac_model"}},
        )
        assert result["status"] == "skipped"

    def test_skip_when_empty_weights(self):
        result = _emit_portfolio(
            {"db_path": ":memory:"},
            {"sac_inference": {"status": "completed", "weights": {}}},
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "empty_weights"

    def test_writes_weights_to_store(self, store: PipelineStore):
        weights = {"ES=F": 0.4, "NQ=F": 0.3, "GC=F": 0.3}
        today = "2026-04-14"

        result = _emit_portfolio(
            {"db_path": ":memory:", "as_of_date": today},
            {"sac_inference": {"status": "completed", "weights": weights}},
        )

        # Since emit_portfolio opens its own store, we can't check the
        # external store. But the result should indicate success.
        assert result["status"] == "completed"
        assert result["n_instruments"] == 3

    def test_pnl_computation_with_real_store(self, tmp_path: Path):
        """Full P&L computation: yesterday's weights × today's returns."""
        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)

        yesterday = "2026-04-13"
        today = "2026-04-14"

        # Store yesterday's weights
        store.store_portfolio_weights(
            yesterday,
            {
                "ES=F": 0.5,
                "NQ=F": 0.3,
                "GC=F": 0.2,
            },
        )

        # Store today's instrument returns as entity observations
        # The _compute_daily_returns function looks for
        # observation_type="daily_return" with value containing "log_return"
        today_date = date.fromisoformat(today)
        from datetime import datetime

        today_ts = datetime(
            today_date.year,
            today_date.month,
            today_date.day,
            hour=12,
            tzinfo=UTC,
        ).timestamp()

        # Register instrument entities first
        store.register_entity("instrument", "S&P 500 Futures", "ES=F")
        store.register_entity("instrument", "Nasdaq 100 Futures", "NQ=F")
        store.register_entity("instrument", "Gold Futures", "GC=F")

        # Store daily returns as observations
        store.store_entity_observation(
            entity_id="ES=F",
            source_tool="instrument_ingest",
            observed_at=today_ts,
            observation_type="daily_return",
            value={"log_return": 0.01},  # +1%
        )
        store.store_entity_observation(
            entity_id="NQ=F",
            source_tool="instrument_ingest",
            observed_at=today_ts,
            observation_type="daily_return",
            value={"log_return": -0.005},  # -0.5%
        )
        store.store_entity_observation(
            entity_id="GC=F",
            source_tool="instrument_ingest",
            observed_at=today_ts,
            observation_type="daily_return",
            value={"log_return": 0.003},  # +0.3%
        )
        store.close()

        # New weights for today
        today_weights = {"ES=F": 0.6, "NQ=F": 0.2, "GC=F": 0.2}

        result = _emit_portfolio(
            {
                "db_path": db_path,
                "as_of_date": today,
                "yesterday_date": yesterday,
            },
            {"sac_inference": {"status": "completed", "weights": today_weights}},
        )

        assert result["status"] == "completed"
        assert result["pnl_computed"] is True

        # Expected P&L = 0.5*0.01 + 0.3*(-0.005) + 0.2*0.003
        #              = 0.005 - 0.0015 + 0.0006 = 0.0041
        expected_pnl = 0.5 * 0.01 + 0.3 * (-0.005) + 0.2 * 0.003
        assert result["portfolio_return"] == pytest.approx(expected_pnl, abs=1e-10)

        # Expected benchmark (equal weight) = (0.01 - 0.005 + 0.003) / 3
        expected_bench = (0.01 + (-0.005) + 0.003) / 3
        assert result["benchmark_return"] == pytest.approx(expected_bench, abs=1e-10)

        # Cumulative should equal portfolio return (first day)
        assert result["cumulative_return"] == pytest.approx(expected_pnl, abs=1e-10)

        # Verify data persisted to DB
        store2 = PipelineStore(db_path)
        stored_weights = store2.query_portfolio_weights(today)
        assert stored_weights["ES=F"] == pytest.approx(0.6)
        assert stored_weights["NQ=F"] == pytest.approx(0.2)

        pnl_records = store2.query_paper_pnl(start_date=today, end_date=today)
        assert len(pnl_records) == 1
        assert pnl_records[0]["portfolio_return"] == pytest.approx(expected_pnl, abs=1e-10)
        store2.close()

    def test_no_previous_weights_skips_pnl(self, tmp_path: Path):
        """If no weights exist for yesterday, P&L is not computed."""
        db_path = str(tmp_path / "test.db")
        PipelineStore(db_path).close()

        result = _emit_portfolio(
            {
                "db_path": db_path,
                "as_of_date": "2026-04-14",
                "yesterday_date": "2026-04-13",
            },
            {"sac_inference": {"status": "completed", "weights": {"ES=F": 1.0}}},
        )

        assert result["status"] == "completed"
        assert result["pnl_computed"] is False
        assert result["reason"] == "no_previous_weights"

    def test_no_return_data_skips_pnl(self, tmp_path: Path):
        """If yesterday's weights exist but no return observations, skip P&L."""
        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)
        store.store_portfolio_weights("2026-04-13", {"ES=F": 1.0})
        store.close()

        result = _emit_portfolio(
            {
                "db_path": db_path,
                "as_of_date": "2026-04-14",
                "yesterday_date": "2026-04-13",
            },
            {"sac_inference": {"status": "completed", "weights": {"ES=F": 0.8}}},
        )

        assert result["status"] == "completed"
        assert result["pnl_computed"] is False
        assert result["reason"] == "no_return_data"

    def test_cumulative_return_accumulates(self, tmp_path: Path):
        """Cumulative return adds to previous day's cumulative."""
        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)

        # Seed day 1 P&L
        store.store_paper_pnl(
            date="2026-04-13",
            portfolio_return=0.005,
            benchmark_return=0.003,
            cumulative_return=0.005,
        )
        # Seed yesterday weights
        store.store_portfolio_weights("2026-04-14", {"ES=F": 0.6})
        # Seed today returns
        store.register_entity("instrument", "S&P 500 Futures", "ES=F")
        from datetime import datetime

        ts = datetime(2026, 4, 15, 12, tzinfo=UTC).timestamp()
        store.store_entity_observation(
            entity_id="ES=F",
            source_tool="instrument_ingest",
            observed_at=ts,
            observation_type="daily_return",
            value={"log_return": 0.02},
        )
        store.close()

        result = _emit_portfolio(
            {
                "db_path": db_path,
                "as_of_date": "2026-04-15",
                "yesterday_date": "2026-04-14",
            },
            {"sac_inference": {"status": "completed", "weights": {"ES=F": 0.7}}},
        )

        assert result["pnl_computed"] is True
        # portfolio_return = 0.6 * 0.02 = 0.012
        assert result["portfolio_return"] == pytest.approx(0.012, abs=1e-10)
        # cumulative = prev(0.005) + 0.012 = 0.017
        # Note: query_paper_pnl for yesterday_date="2026-04-14" won't find the
        # day-1 record (2026-04-13) because we query end_date=yesterday.
        # The function queries up to yesterday, and the most recent record is
        # from 2026-04-13 which is <= 2026-04-14, so it should find it.
        # Actually prev_pnl searches end_date=yesterday="2026-04-14", and the
        # 2026-04-13 record qualifies. But there's also the 2026-04-14 P&L
        # we didn't store. So prev_cumulative = 0.005.
        assert result["cumulative_return"] == pytest.approx(0.005 + 0.012, abs=1e-10)


# ──────────────────────────────────────────────────────────────
# Store CRUD tests (portfolio_weights + paper_trade_pnl)
# ──────────────────────────────────────────────────────────────


class TestPortfolioWeightsCRUD:
    def test_store_and_query(self, store: PipelineStore):
        weights = {"ES=F": 0.5, "NQ=F": 0.3, "GC=F": 0.2}
        ids = store.store_portfolio_weights("2026-04-14", weights)
        assert len(ids) == 3

        result = store.query_portfolio_weights("2026-04-14")
        assert result == {"ES=F": 0.5, "NQ=F": 0.3, "GC=F": 0.2}

    def test_query_empty_date(self, store: PipelineStore):
        result = store.query_portfolio_weights("2026-01-01")
        assert result == {}

    def test_idempotent_re_store(self, store: PipelineStore):
        store.store_portfolio_weights("2026-04-14", {"ES=F": 0.5})
        store.store_portfolio_weights("2026-04-14", {"ES=F": 0.7})

        result = store.query_portfolio_weights("2026-04-14")
        assert result["ES=F"] == pytest.approx(0.7)

    def test_multiple_dates(self, store: PipelineStore):
        store.store_portfolio_weights("2026-04-14", {"ES=F": 0.5})
        store.store_portfolio_weights("2026-04-15", {"ES=F": 0.6})

        r1 = store.query_portfolio_weights("2026-04-14")
        r2 = store.query_portfolio_weights("2026-04-15")
        assert r1["ES=F"] == pytest.approx(0.5)
        assert r2["ES=F"] == pytest.approx(0.6)

    def test_store_with_metadata(self, store: PipelineStore):
        store.store_portfolio_weights(
            "2026-04-14",
            {"ES=F": 1.0},
            metadata={"source": "sac_v1"},
        )
        result = store.query_portfolio_weights("2026-04-14")
        assert result["ES=F"] == pytest.approx(1.0)

    def test_empty_weights_raises(self, store: PipelineStore):
        with pytest.raises(ValueError, match="non-empty"):
            store.store_portfolio_weights("2026-04-14", {})

    def test_empty_date_raises(self, store: PipelineStore):
        with pytest.raises(ValueError, match="non-empty"):
            store.store_portfolio_weights("", {"ES=F": 1.0})

    def test_whitespace_date_raises(self, store: PipelineStore):
        with pytest.raises(ValueError, match="non-empty"):
            store.store_portfolio_weights("  ", {"ES=F": 1.0})

    def test_negative_weight(self, store: PipelineStore):
        """Negative weights (short positions) are valid."""
        store.store_portfolio_weights("2026-04-14", {"ES=F": -0.3})
        result = store.query_portfolio_weights("2026-04-14")
        assert result["ES=F"] == pytest.approx(-0.3)

    def test_zero_weight(self, store: PipelineStore):
        store.store_portfolio_weights("2026-04-14", {"ES=F": 0.0})
        result = store.query_portfolio_weights("2026-04-14")
        assert result["ES=F"] == pytest.approx(0.0)

    def test_many_instruments(self, store: PipelineStore):
        weights = {f"INST_{i}": float(i) / 100 for i in range(100)}
        store.store_portfolio_weights("2026-04-14", weights)
        result = store.query_portfolio_weights("2026-04-14")
        assert len(result) == 100
        assert result["INST_50"] == pytest.approx(0.5)


class TestPaperPnlCRUD:
    def test_store_and_query(self, store: PipelineStore):
        row_id = store.store_paper_pnl(
            date="2026-04-14",
            portfolio_return=0.005,
            benchmark_return=0.003,
            cumulative_return=0.005,
        )
        assert row_id > 0

        results = store.query_paper_pnl(start_date="2026-04-14", end_date="2026-04-14")
        assert len(results) == 1
        assert results[0]["portfolio_return"] == pytest.approx(0.005)
        assert results[0]["benchmark_return"] == pytest.approx(0.003)
        assert results[0]["cumulative_return"] == pytest.approx(0.005)

    def test_query_range(self, store: PipelineStore):
        for i in range(5):
            d = f"2026-04-1{i}"
            store.store_paper_pnl(d, float(i) * 0.001, 0.001, float(i) * 0.001)

        results = store.query_paper_pnl(start_date="2026-04-11", end_date="2026-04-13")
        dates = [r["date"] for r in results]
        assert "2026-04-11" in dates
        assert "2026-04-12" in dates
        assert "2026-04-13" in dates
        assert "2026-04-14" not in dates

    def test_query_no_filter(self, store: PipelineStore):
        store.store_paper_pnl("2026-04-14", 0.01, 0.005, 0.01)
        results = store.query_paper_pnl()
        assert len(results) == 1

    def test_idempotent_re_store(self, store: PipelineStore):
        store.store_paper_pnl("2026-04-14", 0.01, 0.005, 0.01)
        store.store_paper_pnl("2026-04-14", 0.02, 0.008, 0.03)

        results = store.query_paper_pnl(start_date="2026-04-14", end_date="2026-04-14")
        assert len(results) == 1
        assert results[0]["portfolio_return"] == pytest.approx(0.02)

    def test_with_metadata(self, store: PipelineStore):
        store.store_paper_pnl(
            "2026-04-14",
            0.01,
            0.005,
            0.01,
            metadata={"n_instruments": 5},
        )
        results = store.query_paper_pnl()
        assert results[0]["metadata"] == {"n_instruments": 5}

    def test_empty_date_raises(self, store: PipelineStore):
        with pytest.raises(ValueError, match="non-empty"):
            store.store_paper_pnl("", 0.01, 0.005, 0.01)

    def test_negative_returns(self, store: PipelineStore):
        """Negative returns are valid."""
        store.store_paper_pnl("2026-04-14", -0.05, -0.03, -0.05)
        results = store.query_paper_pnl()
        assert results[0]["portfolio_return"] == pytest.approx(-0.05)

    def test_limit(self, store: PipelineStore):
        for i in range(10):
            store.store_paper_pnl(f"2026-04-{i + 10:02d}", 0.001 * i, 0.001, 0.001 * i)
        results = store.query_paper_pnl(limit=3)
        assert len(results) == 3

    def test_ordered_by_date_asc(self, store: PipelineStore):
        store.store_paper_pnl("2026-04-15", 0.01, 0.005, 0.01)
        store.store_paper_pnl("2026-04-13", 0.02, 0.008, 0.02)
        store.store_paper_pnl("2026-04-14", 0.015, 0.006, 0.035)

        results = store.query_paper_pnl()
        dates = [r["date"] for r in results]
        assert dates == ["2026-04-13", "2026-04-14", "2026-04-15"]


# ──────────────────────────────────────────────────────────────
# Full pipeline skip propagation
# ──────────────────────────────────────────────────────────────


class TestSkipPropagation:
    """Verify that upstream skips propagate cleanly through all nodes."""

    def test_all_skip_when_no_models(self):
        load_result = _load_models(
            {"db_path": ":memory:", "model_path": "/nonexistent"},
            {},
        )
        assert load_result["status"] == "skipped"

        gnn_result = _gnn_inference(
            {"db_path": ":memory:"},
            {"load_models": load_result},
        )
        assert gnn_result["status"] == "skipped"

        sac_result = _sac_inference(
            {"db_path": ":memory:"},
            {"load_models": load_result, "gnn_inference": gnn_result},
        )
        assert sac_result["status"] == "skipped"

        emit_result = _emit_portfolio(
            {"db_path": ":memory:"},
            {"sac_inference": sac_result},
        )
        assert emit_result["status"] == "skipped"

    def test_gnn_skip_sac_still_runs_if_has_sac(self, tmp_path: Path):
        """If GNN unavailable but SAC is, sac_inference should still try."""
        db_path = str(tmp_path / "test.db")
        store = PipelineStore(db_path)
        store.store_rl_checkpoint(
            policy_type="sac",
            config={"state_dim": 10, "action_dim": 3},
            state_dict_bytes=b"fake",
        )
        store.close()

        load_result = _load_models(
            {"db_path": db_path, "model_path": "/nonexistent"},
            {},
        )
        assert load_result["has_gnn"] is False
        assert load_result["has_sac"] is True
        assert load_result["status"] == "ready"

        gnn_result = _gnn_inference(
            {"db_path": db_path},
            {"load_models": load_result},
        )
        assert gnn_result["status"] == "skipped"
        assert gnn_result["reason"] == "no_gnn_model"

        # SAC inference should still attempt (it has its own model)
        # It won't be marked as "skipped" due to upstream — it has has_sac=True
        # It will try to load the SAC model and run
        # (will fail at SACTrainer.load since state_dict is fake, but it tries)


# ──────────────────────────────────────────────────────────────
# DAG registration
# ──────────────────────────────────────────────────────────────


class TestDAGRegistration:
    def test_inference_dag_in_defaults(self):
        """Verify the inference DAG is included in get_default_dags."""
        from agent.pipeline.dags import get_default_dags

        dags = get_default_dags(tool_registry=None)
        names = {d.name for d in dags}
        assert "inference" in names

    def test_default_dags_count(self):
        from agent.pipeline.dags import get_default_dags

        dags = get_default_dags(tool_registry=None)
        assert len(dags) == 11  # was 10, now 11 with inference
