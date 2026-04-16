"""
Tests for Self-Improving Architecture changes (Tier 1 + Tier 2).

Covers:
    Change 1: query_all_latest_beliefs + beliefs wiring in inference DAG
    Change 4: AdaptiveSurpriseWeights — EG on simplex
    Change 9: GNN loss auto-tuning — uncertainty-weighted multi-task loss
    Change 2a: CPD learning via MLE — WorldModel.fit_cpds()
    Change 2b: Kalman EM parameter fitting — ContinuousStateFilter.fit_filter_params()
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════
# Change 1: query_all_latest_beliefs
# ═══════════════════════════════════════════════════════════════


class TestQueryAllLatestBeliefs:
    """PipelineStore.query_all_latest_beliefs returns latest per variable."""

    def _make_store(self, tmp_path: Path):
        from agent.pipeline.store import PipelineStore

        db = tmp_path / "test.db"
        return PipelineStore(str(db))

    # Timestamps must be after 2020-01-01 (1577836800) per BeliefState validation
    _BASE_TS = 1_700_000_000.0

    def _make_belief(self, variable_name: str, effective_at: float, mean: float):
        from agent.models.belief import BeliefState

        return BeliefState(
            variable_name=variable_name,
            version=1,
            effective_at=self._BASE_TS + effective_at,
            computed_at=self._BASE_TS + effective_at + 1.0,
            dist_type="gaussian",
            mean=mean,
            variance=0.1,
            evidence_count=5,
            model_graph_hash="a" * 64,
            confidence=0.9,
            stale=False,
        )

    def test_empty_store_returns_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        result = store.query_all_latest_beliefs()
        assert result == []

    def test_returns_latest_per_variable(self, tmp_path):
        store = self._make_store(tmp_path)
        # Store older and newer beliefs for two variables
        store.store_belief(self._make_belief("latent.stress", 1000.0, 0.5))
        store.store_belief(self._make_belief("latent.stress", 2000.0, 0.8))
        store.store_belief(self._make_belief("regime.macro", 1500.0, 0.3))
        store.store_belief(self._make_belief("regime.macro", 2500.0, 0.6))

        results = store.query_all_latest_beliefs()
        assert len(results) == 2

        by_name = {r["variable_name"]: r for r in results}
        assert by_name["latent.stress"]["mean"] == pytest.approx(0.8)
        assert by_name["latent.stress"]["effective_at"] == pytest.approx(
            self._BASE_TS + 2000.0
        )
        assert by_name["regime.macro"]["mean"] == pytest.approx(0.6)

    def test_single_belief_per_variable(self, tmp_path):
        store = self._make_store(tmp_path)
        store.store_belief(self._make_belief("latent.x", 1000.0, 1.0))
        results = store.query_all_latest_beliefs()
        assert len(results) == 1
        assert results[0]["variable_name"] == "latent.x"

    def test_many_variables(self, tmp_path):
        store = self._make_store(tmp_path)
        for i in range(20):
            store.store_belief(
                self._make_belief(f"obs.feature_{i}", 1000.0 + i, float(i))
            )
        results = store.query_all_latest_beliefs()
        assert len(results) == 20


# ═══════════════════════════════════════════════════════════════
# Change 4: AdaptiveSurpriseWeights
# ═══════════════════════════════════════════════════════════════


class TestAdaptiveSurpriseWeights:
    """Tests for EG-on-simplex adaptive weights."""

    def _cls(self):
        from agent.fusion.surprise import AdaptiveSurpriseWeights

        return AdaptiveSurpriseWeights

    def test_default_weights_sum_to_one(self):
        aw = self._cls()()
        w = aw.weights
        assert sum(w.values()) == pytest.approx(1.0)
        assert len(w) == 5

    def test_custom_weights_normalised(self):
        aw = self._cls()(initial_weights=(1, 1, 1, 1, 1))
        w = aw.weights
        for v in w.values():
            assert v == pytest.approx(0.2)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="5 elements"):
            self._cls()(initial_weights=(1, 2, 3))

    def test_update_moves_weights(self):
        aw = self._cls()(initial_weights=(0.2, 0.2, 0.2, 0.2, 0.2))
        # Gradient says channel 0 was very helpful (negative grad = helpful)
        aw.update((-1.0, 0.0, 0.0, 0.0, 0.0))
        w = aw.weights
        assert w["obs_type"] > 0.2  # should increase
        assert sum(w.values()) == pytest.approx(1.0)
        assert aw.n_updates == 1

    def test_update_wrong_gradient_length_raises(self):
        aw = self._cls()()
        with pytest.raises(ValueError, match="5 elements"):
            aw.update((0.0, 0.0))

    def test_min_weight_prevents_collapse(self):
        aw = self._cls()(
            initial_weights=(0.9, 0.025, 0.025, 0.025, 0.025),
            min_weight=0.05,
        )
        # Push hard against all but the first channel
        for _ in range(50):
            aw.update((-5.0, 5.0, 5.0, 5.0, 5.0))
        w = aw.weights
        for name in ["temporal", "value", "neighborhood", "memory"]:
            assert w[name] >= 0.01  # at least min_weight after normalisation
        assert sum(w.values()) == pytest.approx(1.0)

    def test_zero_gradients_no_change(self):
        aw = self._cls()(initial_weights=(0.3, 0.15, 0.25, 0.2, 0.1))
        w_before = aw.weights_tuple
        aw.update((0.0, 0.0, 0.0, 0.0, 0.0))
        w_after = aw.weights_tuple
        for a, b in zip(w_before, w_after):
            assert a == pytest.approx(b, abs=1e-10)

    def test_compute_gradients_empty_returns_zeros(self):
        aw = self._cls()()
        grad = aw.compute_gradients([], [])
        assert grad == (0.0, 0.0, 0.0, 0.0, 0.0)

    def test_compute_gradients_mismatched_lengths(self):
        aw = self._cls()()
        grad = aw.compute_gradients([(1, 2, 3, 4, 5)], [])
        assert grad == (0.0, 0.0, 0.0, 0.0, 0.0)

    def test_compute_gradients_basic(self):
        aw = self._cls()(initial_weights=(0.2, 0.2, 0.2, 0.2, 0.2))
        sv = [(1.0, 0.0, 0.0, 0.0, 0.0)]
        outcome = [0.5]  # composite = 0.2*1 = 0.2, residual = 0.2 - 0.5 = -0.3
        grad = aw.compute_gradients(sv, outcome)
        assert len(grad) == 5
        # Channel 0 should have negative gradient (helpful)
        assert grad[0] < 0

    def test_serialization_roundtrip(self):
        aw = self._cls()(initial_weights=(0.35, 0.10, 0.25, 0.20, 0.10))
        aw.update((0.1, -0.1, 0.0, 0.0, 0.05))
        d = aw.to_dict()
        aw2 = self._cls().from_dict(d)
        assert aw2.n_updates == aw.n_updates
        for a, b in zip(aw.weights_tuple, aw2.weights_tuple):
            assert a == pytest.approx(b)

    def test_integration_with_extractor(self):
        """AdaptiveSurpriseWeights can be passed to SurpriseExtractor."""
        from agent.fusion.surprise import AdaptiveSurpriseWeights, SurpriseExtractor

        aw = AdaptiveSurpriseWeights(initial_weights=(0.3, 0.15, 0.25, 0.2, 0.1))
        ext = SurpriseExtractor(adaptive_weights=aw)
        assert ext.adaptive is aw
        assert ext._weights == aw.weights

        # Simulating weight change
        aw.update((-0.5, 0.1, 0.1, 0.1, 0.1))
        ext.sync_adaptive_weights()
        assert ext._weights == aw.weights


# ═══════════════════════════════════════════════════════════════
# Change 9: GNN loss auto-tuning
# ═══════════════════════════════════════════════════════════════


class TestGNNLossAutoTuning:
    """Tests for uncertainty-weighted multi-task loss."""

    def test_config_default_auto_tune_off(self):
        from agent.models.gnn.trainer import TrainerConfig

        cfg = TrainerConfig()
        assert cfg.auto_tune_loss_weights is False

    def test_config_auto_tune_on(self):
        from agent.models.gnn.trainer import TrainerConfig

        cfg = TrainerConfig(auto_tune_loss_weights=True)
        assert cfg.auto_tune_loss_weights is True

    def test_log_vars_created_when_auto_tune(self, tmp_path):
        """When auto_tune_loss_weights=True, build_model creates log-var params."""
        import torch
        from agent.models.gnn.trainer import Trainer, TrainerConfig

        store = self._make_mock_store(tmp_path)
        cfg = TrainerConfig(auto_tune_loss_weights=True, epochs=1)
        trainer = Trainer(store, cfg)

        with patch.object(trainer, "_graph_builder") as mock_gb:
            mock_gb.build.return_value = self._make_dummy_graph_data()
            with patch("agent.models.gnn.trainer.HetTGN") as MockHetTGN:
                mock_model = MagicMock()
                mock_model.parameters.return_value = [
                    torch.nn.Parameter(torch.zeros(1))
                ]
                MockHetTGN.return_value = mock_model
                trainer.build_model()

        assert trainer._log_vars is not None
        assert set(trainer._log_vars.keys()) == {
            "obs_type",
            "time_delta",
            "contrastive",
            "value",
        }
        for p in trainer._log_vars.values():
            assert isinstance(p, torch.nn.Parameter)

    def test_log_vars_none_when_auto_tune_off(self, tmp_path):
        """When auto_tune_loss_weights=False, log_vars is None."""
        import torch
        from agent.models.gnn.trainer import Trainer, TrainerConfig

        store = self._make_mock_store(tmp_path)
        cfg = TrainerConfig(auto_tune_loss_weights=False, epochs=1)
        trainer = Trainer(store, cfg)

        with patch.object(trainer, "_graph_builder") as mock_gb:
            mock_gb.build.return_value = self._make_dummy_graph_data()
            with patch("agent.models.gnn.trainer.HetTGN") as MockHetTGN:
                mock_model = MagicMock()
                mock_model.parameters.return_value = [
                    torch.nn.Parameter(torch.zeros(1))
                ]
                MockHetTGN.return_value = mock_model
                trainer.build_model()

        assert trainer._log_vars is None

    def test_effective_loss_weights_fixed(self, tmp_path):
        """When auto_tune off, effective_loss_weights returns config."""
        from agent.models.gnn.trainer import Trainer, TrainerConfig

        store = self._make_mock_store(tmp_path)
        cfg = TrainerConfig(
            obs_type_weight=1.0,
            time_delta_weight=0.1,
            contrastive_weight=0.5,
            value_weight=0.3,
        )
        trainer = Trainer(store, cfg)
        trainer._log_vars = None
        w = trainer.effective_loss_weights()
        assert w["obs_type"] == 1.0
        assert w["time_delta"] == 0.1

    def test_effective_loss_weights_auto_tuned(self, tmp_path):
        """When auto_tune on, effective_loss_weights reflects log-vars."""
        import torch
        from agent.models.gnn.trainer import Trainer, TrainerConfig

        store = self._make_mock_store(tmp_path)
        trainer = Trainer(store, TrainerConfig())
        # Manually set log_vars to known values
        trainer._log_vars = {
            "obs_type": torch.nn.Parameter(torch.tensor(0.0)),  # exp(-0) = 1.0
            "time_delta": torch.nn.Parameter(
                torch.tensor(math.log(2.0))
            ),  # exp(-ln2) ≈ 0.5
            "contrastive": torch.nn.Parameter(
                torch.tensor(-math.log(2.0))
            ),  # exp(ln2) = 2.0
            "value": torch.nn.Parameter(torch.tensor(0.0)),
        }
        w = trainer.effective_loss_weights()
        assert w["obs_type"] == pytest.approx(1.0, abs=1e-4)
        assert w["time_delta"] == pytest.approx(0.5, abs=1e-4)
        assert w["contrastive"] == pytest.approx(2.0, abs=1e-4)

    @staticmethod
    def _make_mock_store(tmp_path):
        from agent.pipeline.store import PipelineStore

        db = tmp_path / "test.db"
        return PipelineStore(str(db))

    @staticmethod
    def _make_dummy_graph_data():
        """Return (data, id_map, events) tuple for build_model."""
        from agent.models.gnn.graph_builder import IDMap

        data = MagicMock()
        data.metadata.return_value = (
            ["company"],
            [("company", "linked_to", "company")],
        )
        data.node_types = ["company"]
        mock_x = MagicMock()
        mock_x.size.return_value = 16
        data.__getitem__ = MagicMock(return_value=MagicMock(x=mock_x))

        id_map = IDMap()
        id_map.add("company", "c0")
        return data, id_map, []


# ═══════════════════════════════════════════════════════════════
# Change 2a: CPD learning via MLE
# ═══════════════════════════════════════════════════════════════


class TestCPDLearning:
    """Tests for WorldModel.fit_cpds() — Bayesian estimation of CPDs."""

    def _make_world_model(self):
        from agent.models.initial_graph import build_initial_graph
        from agent.models.propagator import BeliefPropagator
        from agent.models.state_filter import ContinuousStateFilter, RegimeConfig
        from agent.models.world_model import WorldModel

        graph = build_initial_graph()
        propagator = BeliefPropagator(graph)
        configs = {
            "expansion": RegimeConfig(
                name="expansion", F=np.diag([0.99]), Q=np.diag([0.01])
            ),
        }
        H = np.array([[1.0]])
        R = np.array([[0.1]])
        sf = ContinuousStateFilter(
            state_dim=1,
            obs_dim=1,
            regime_configs=configs,
            H=H,
            R=R,
        )
        return WorldModel(
            graph=graph,
            propagator=propagator,
            state_filter=sf,
        )

    def _make_features(self, n_samples: int = 100):
        """Generate synthetic EngineeredFeature snapshots."""
        from agent.features.protocol import EngineeredFeature

        rng = np.random.default_rng(42)
        feature_names = [
            "macro.rate_momentum.30d",
            "macro.yield_curve_slope.spot",
            "macro.liquidity_pressure.30d",
            "convergence.stress_breadth.7d",
            "convergence.stress_intensity.7d",
            "convergence.regime_persistence.7d",
        ]

        snapshots = []
        for i in range(n_samples):
            snap = []
            for fname in feature_names:
                snap.append(
                    EngineeredFeature(
                        feature_name=fname,
                        version=1,
                        effective_at=1000.0 + i * 86400,
                        computed_at=1000.0 + i * 86400 + 1,
                        horizon="spot",
                        value=float(rng.normal(0, 1)),
                        quality=0.9,
                        missing_reason=None,
                        source_signals=["test"],
                        builder="test",
                        unit="raw",
                    )
                )
            snapshots.append(snap)
        return snapshots

    def test_insufficient_samples_not_fitted(self):
        wm = self._make_world_model()
        result = wm.fit_cpds([], min_samples=10)
        assert result["fitted"] is False
        assert result["n_samples"] == 0

    def test_fit_with_enough_data(self):
        wm = self._make_world_model()
        features = self._make_features(200)
        result = wm.fit_cpds(features, min_samples=50, equivalent_sample_size=5.0)
        assert result["fitted"] is True
        assert result["n_samples"] >= 50
        assert len(result["nodes_fitted"]) > 0

    def test_fit_preserves_graph_structure(self):
        """After fitting, graph still has the same nodes and edges."""
        wm = self._make_world_model()
        nodes_before = set(wm._graph.node_names)
        edges_before = set(wm._graph.edges)
        features = self._make_features(200)
        wm.fit_cpds(features, min_samples=50)
        assert set(wm._graph.node_names) == nodes_before
        assert set(wm._graph.edges) == edges_before

    def test_discretize_basic(self):
        from agent.models.world_model import WorldModel

        result = WorldModel._discretize(
            0.7, (-float("inf"), -0.5, 0.5, float("inf")), ("low", "neutral", "high")
        )
        assert result == "high"

    def test_discretize_boundary(self):
        from agent.models.world_model import WorldModel

        result = WorldModel._discretize(
            -0.5, (-float("inf"), -0.5, 0.5, float("inf")), ("low", "neutral", "high")
        )
        assert result == "neutral"

    def test_discretize_low(self):
        from agent.models.world_model import WorldModel

        result = WorldModel._discretize(
            -2.0, (-float("inf"), -0.5, 0.5, float("inf")), ("low", "neutral", "high")
        )
        assert result == "low"

    def test_discretize_none_states(self):
        from agent.models.world_model import WorldModel

        result = WorldModel._discretize(1.0, None, None)
        assert result is None


# ═══════════════════════════════════════════════════════════════
# Change 2b: Kalman EM parameter fitting
# ═══════════════════════════════════════════════════════════════


class TestKalmanEM:
    """Tests for ContinuousStateFilter.fit_filter_params() — EM algorithm."""

    def _make_filter(self, state_dim=2, obs_dim=2):
        from agent.models.state_filter import ContinuousStateFilter, RegimeConfig

        configs = {
            "stable": RegimeConfig(
                name="stable",
                F=np.eye(state_dim) * 0.95,
                Q=np.eye(state_dim) * 0.01,
            ),
        }
        H = np.eye(obs_dim, state_dim)
        R = np.eye(obs_dim) * 0.1
        return ContinuousStateFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            regime_configs=configs,
            H=H,
            R=R,
        )

    def _generate_synthetic_data(
        self, T=200, state_dim=2, obs_dim=2, F_true=None, H_true=None, seed=42
    ):
        """Generate synthetic linear Gaussian state-space data."""
        rng = np.random.default_rng(seed)
        if F_true is None:
            F_true = np.eye(state_dim) * 0.9
        if H_true is None:
            H_true = np.eye(obs_dim, state_dim)
        Q_true = np.eye(state_dim) * 0.05
        R_true = np.eye(obs_dim) * 0.2

        x = np.zeros(state_dim)
        observations = []
        for _ in range(T):
            x = F_true @ x + rng.multivariate_normal(np.zeros(state_dim), Q_true)
            y = H_true @ x + rng.multivariate_normal(np.zeros(obs_dim), R_true)
            observations.append(y)
        return observations, F_true, Q_true, H_true, R_true

    def test_insufficient_samples(self):
        filt = self._make_filter()
        result = filt.fit_filter_params(
            observations_seq=[np.array([0.0, 0.0])],
            regime_labels=["stable"],
            min_samples=10,
        )
        assert result["fitted"] is False

    def test_mismatched_labels_raises(self):
        filt = self._make_filter()
        with pytest.raises(ValueError, match="regime_labels length"):
            filt.fit_filter_params(
                observations_seq=[np.zeros(2)] * 50,
                regime_labels=["stable"] * 10,
            )

    def test_em_converges_on_synthetic(self):
        """EM should converge (log-likelihood non-decreasing) on synthetic data."""
        filt = self._make_filter()
        obs, _, _, _, _ = self._generate_synthetic_data(T=300)
        result = filt.fit_filter_params(
            observations_seq=obs,
            regime_labels=["stable"] * 300,
            max_iter=15,
            min_samples=30,
        )
        assert result["fitted"] is True
        lls = result["log_likelihoods"]
        assert len(lls) >= 2
        # EM should generally increase log-likelihood (allow small tolerance)
        for i in range(1, len(lls)):
            assert (
                lls[i] >= lls[i - 1] - 1e-3
            ), f"LL decreased at iteration {i}: {lls[i-1]:.2f} → {lls[i]:.2f}"

    def test_em_recovers_approximate_params(self):
        """EM-fitted F should be closer to true F than the initial guess."""
        F_true = np.array([[0.85, 0.05], [0.05, 0.90]])
        filt = self._make_filter()
        obs, _, _, _, _ = self._generate_synthetic_data(T=500, F_true=F_true, seed=123)
        F_init = filt._regime_configs["stable"].F.copy()

        filt.fit_filter_params(
            observations_seq=obs,
            regime_labels=["stable"] * 500,
            max_iter=20,
            min_samples=30,
        )
        F_fitted = filt._regime_configs["stable"].F

        # Fitted F should be closer to true F than initial
        err_init = np.linalg.norm(F_init - F_true)
        err_fitted = np.linalg.norm(F_fitted - F_true)
        assert (
            err_fitted < err_init
        ), f"EM did not improve: init err={err_init:.4f}, fitted err={err_fitted:.4f}"

    def test_em_with_missing_observations(self):
        """EM should handle NaN observations without crashing."""
        filt = self._make_filter()
        obs, _, _, _, _ = self._generate_synthetic_data(T=200)
        # Introduce NaN in 20% of observations
        rng = np.random.default_rng(99)
        for i in range(len(obs)):
            if rng.random() < 0.2:
                obs[i][0] = np.nan

        result = filt.fit_filter_params(
            observations_seq=obs,
            regime_labels=["stable"] * 200,
            max_iter=10,
            min_samples=30,
        )
        assert result["fitted"] is True
        assert result["iterations"] >= 2

    def test_em_multiple_regimes(self):
        """EM should fit distinct params for each regime."""
        from agent.models.state_filter import ContinuousStateFilter, RegimeConfig

        configs = {
            "fast": RegimeConfig(name="fast", F=np.eye(2) * 0.8, Q=np.eye(2) * 0.05),
            "slow": RegimeConfig(name="slow", F=np.eye(2) * 0.99, Q=np.eye(2) * 0.01),
        }
        filt = ContinuousStateFilter(
            state_dim=2,
            obs_dim=2,
            regime_configs=configs,
            H=np.eye(2),
            R=np.eye(2) * 0.1,
        )
        obs, _, _, _, _ = self._generate_synthetic_data(T=300, seed=77)
        # Alternate regimes
        labels = ["fast" if i % 2 == 0 else "slow" for i in range(300)]

        result = filt.fit_filter_params(
            observations_seq=obs,
            regime_labels=labels,
            max_iter=10,
            min_samples=30,
        )
        assert result["fitted"] is True
        # Both regimes should have been fitted
        assert "fast" in filt._regime_configs
        assert "slow" in filt._regime_configs

    def test_psd_guarantee(self):
        """Fitted Q and R should be positive semi-definite."""
        filt = self._make_filter()
        obs, _, _, _, _ = self._generate_synthetic_data(T=300)
        filt.fit_filter_params(
            observations_seq=obs,
            regime_labels=["stable"] * 300,
            max_iter=10,
        )
        Q = filt._regime_configs["stable"].Q
        R = filt._R

        # Check PSD: all eigenvalues >= 0
        assert np.all(np.linalg.eigvalsh(Q) >= -1e-8)
        assert np.all(np.linalg.eigvalsh(R) >= -1e-8)

    def test_1d_filter(self):
        """EM works for 1-dimensional state space."""
        from agent.models.state_filter import ContinuousStateFilter, RegimeConfig

        configs = {
            "single": RegimeConfig(
                name="single",
                F=np.array([[0.9]]),
                Q=np.array([[0.01]]),
            )
        }
        filt = ContinuousStateFilter(
            state_dim=1,
            obs_dim=1,
            regime_configs=configs,
            H=np.array([[1.0]]),
            R=np.array([[0.1]]),
        )
        rng = np.random.default_rng(42)
        obs = [np.array([rng.normal()]) for _ in range(100)]
        result = filt.fit_filter_params(
            observations_seq=obs,
            regime_labels=["single"] * 100,
            max_iter=10,
            min_samples=20,
        )
        assert result["fitted"] is True
