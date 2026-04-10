"""
Tests for agent/models/world_model.py — WorldModel orchestrator.

Validates:
    - End-to-end: features → DAG beliefs + Kalman beliefs
    - Regime switching propagates to Kalman
    - Empty features produce stale beliefs
    - Partial features work (some missing)
    - query() returns cached beliefs
    - DAG-only mode (no Kalman state names)
    - get_graph_hash() is stable
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from agent.features.protocol import EngineeredFeature
from agent.models.graph import NodeSpec, WorldModelGraph
from agent.models.initial_graph import build_initial_graph
from agent.models.propagator import BeliefPropagator
from agent.models.state_filter import ContinuousStateFilter, RegimeConfig
from agent.models.world_model import WorldModel

# ── Helpers ────────────────────────────────────────────────────

AS_OF = 1_700_000_000.0


def _make_feature(
    name: str,
    value: float | None,
    quality: float = 1.0,
) -> EngineeredFeature:
    """Minimal EngineeredFeature for testing."""
    t = time.time()
    return EngineeredFeature(
        feature_name=name,
        version=1,
        effective_at=AS_OF,
        computed_at=t,
        horizon="30d",
        value=value,
        quality=quality,
        missing_reason="source_unavailable" if value is None else None,
    )


def _build_full_world_model() -> WorldModel:
    """Build a WorldModel with the expert DAG + 3D Kalman filter."""
    graph = build_initial_graph()
    propagator = BeliefPropagator(graph)

    # 3D Kalman: stress_level, macro_momentum, liquidity_state
    configs = {
        "expansion": RegimeConfig(
            name="expansion",
            F=np.diag([0.99, 0.98, 0.97]),
            Q=np.diag([0.01, 0.01, 0.01]),
        ),
        "contraction": RegimeConfig(
            name="contraction",
            F=np.diag([0.97, 0.96, 0.95]),
            Q=np.diag([0.02, 0.02, 0.02]),
        ),
        "crisis": RegimeConfig(
            name="crisis",
            F=np.diag([0.90, 0.88, 0.85]),
            Q=np.diag([0.10, 0.10, 0.10]),
        ),
    }
    H = np.zeros((6, 3))
    H[0, 0] = 1.0  # rate_momentum → stress_level
    H[1, 0] = 1.0  # yield_curve_slope → stress_level
    H[2, 1] = 1.0  # liquidity_pressure → macro_momentum
    H[3, 1] = 1.0  # stress_breadth → macro_momentum
    H[4, 2] = 1.0  # stress_intensity → liquidity_state
    H[5, 2] = 1.0  # regime_persistence → liquidity_state
    R = np.diag([0.1] * 6)

    state_filter = ContinuousStateFilter(
        state_dim=3,
        obs_dim=6,
        regime_configs=configs,
        H=H,
        R=R,
    )

    feature_to_obs = {
        "macro.rate_momentum.30d": 0,
        "macro.yield_curve_slope.spot": 1,
        "macro.liquidity_pressure.30d": 2,
        "convergence.stress_breadth.7d": 3,
        "convergence.stress_intensity.7d": 4,
        "convergence.regime_persistence.7d": 5,
    }

    return WorldModel(
        graph=graph,
        propagator=propagator,
        state_filter=state_filter,
        regime_node="regime.macro",
        continuous_state_names=[
            "latent.stress_level",
            "latent.macro_momentum",
            "latent.liquidity_state",
        ],
        feature_to_obs_index=feature_to_obs,
    )


def _expansion_features() -> list[EngineeredFeature]:
    """Features consistent with an expansion regime."""
    return [
        _make_feature("macro.rate_momentum.30d", 0.8),  # rising
        _make_feature("macro.yield_curve_slope.spot", 1.0),  # steep
        _make_feature("macro.liquidity_pressure.30d", 0.5),  # loose
        _make_feature("convergence.stress_breadth.7d", 0.1),  # narrow
        _make_feature("convergence.stress_intensity.7d", 0.1),  # low
        _make_feature("convergence.regime_persistence.7d", 0.8),  # persistent
    ]


def _crisis_features() -> list[EngineeredFeature]:
    """Features consistent with a crisis regime."""
    return [
        _make_feature("macro.rate_momentum.30d", -1.0),  # falling
        _make_feature("macro.yield_curve_slope.spot", -0.5),  # inverted
        _make_feature("macro.liquidity_pressure.30d", -0.5),  # tight
        _make_feature("convergence.stress_breadth.7d", 0.9),  # broad
        _make_feature("convergence.stress_intensity.7d", 0.9),  # high
        _make_feature("convergence.regime_persistence.7d", 0.1),  # unstable
    ]


# ── Tests ──────────────────────────────────────────────────────


class TestEndToEnd:
    def test_update_returns_beliefs(self) -> None:
        wm = _build_full_world_model()
        beliefs = wm.update(_expansion_features(), AS_OF)
        # 9 DAG beliefs + 3 Kalman beliefs = 12
        assert len(beliefs) == 12

    def test_dag_beliefs_present(self) -> None:
        wm = _build_full_world_model()
        beliefs = wm.update(_expansion_features(), AS_OF)
        names = {b.variable_name for b in beliefs}
        assert "regime.macro" in names
        assert "regime.stress" in names
        assert "latent.risk_appetite" in names

    def test_kalman_beliefs_present(self) -> None:
        wm = _build_full_world_model()
        beliefs = wm.update(_expansion_features(), AS_OF)
        names = {b.variable_name for b in beliefs}
        assert "latent.stress_level" in names
        assert "latent.macro_momentum" in names
        assert "latent.liquidity_state" in names

    def test_dag_beliefs_are_categorical(self) -> None:
        wm = _build_full_world_model()
        beliefs = wm.update(_expansion_features(), AS_OF)
        dag_beliefs = [b for b in beliefs if b.dist_type == "categorical"]
        assert len(dag_beliefs) == 9

    def test_kalman_beliefs_are_gaussian(self) -> None:
        wm = _build_full_world_model()
        beliefs = wm.update(_expansion_features(), AS_OF)
        gauss = [b for b in beliefs if b.dist_type == "gaussian"]
        assert len(gauss) == 3
        for b in gauss:
            assert b.mean is not None
            assert b.variance is not None


class TestRegimePropagation:
    def test_expansion_features_favour_expansion(self) -> None:
        wm = _build_full_world_model()
        beliefs = wm.update(_expansion_features(), AS_OF)
        macro = next(b for b in beliefs if b.variable_name == "regime.macro")
        assert macro.probabilities["expansion"] > macro.probabilities["crisis"]

    def test_crisis_features_favour_crisis(self) -> None:
        wm = _build_full_world_model()
        beliefs = wm.update(_crisis_features(), AS_OF)
        macro = next(b for b in beliefs if b.variable_name == "regime.macro")
        assert macro.probabilities["crisis"] > macro.probabilities["expansion"]


class TestMissingFeatures:
    def test_empty_features_produce_stale_dag(self) -> None:
        wm = _build_full_world_model()
        beliefs = wm.update([], AS_OF)
        dag_beliefs = [b for b in beliefs if b.dist_type == "categorical"]
        for b in dag_beliefs:
            assert b.stale is True

    def test_partial_features_work(self) -> None:
        wm = _build_full_world_model()
        partial = [_make_feature("macro.rate_momentum.30d", 0.8)]
        beliefs = wm.update(partial, AS_OF)
        assert len(beliefs) == 12

    def test_none_value_excluded(self) -> None:
        wm = _build_full_world_model()
        features = [_make_feature("macro.rate_momentum.30d", None)]
        beliefs = wm.update(features, AS_OF)
        # Should not crash — None values are excluded from evidence
        assert len(beliefs) == 12


class TestQuery:
    def test_query_returns_cached(self) -> None:
        wm = _build_full_world_model()
        wm.update(_expansion_features(), AS_OF)
        b = wm.query("regime.macro")
        assert b is not None
        assert b.variable_name == "regime.macro"

    def test_query_missing_returns_none(self) -> None:
        wm = _build_full_world_model()
        assert wm.query("nonexistent.var") is None

    def test_query_before_update_returns_none(self) -> None:
        wm = _build_full_world_model()
        assert wm.query("regime.macro") is None


class TestDagOnlyMode:
    def test_no_kalman_produces_dag_only(self) -> None:
        graph = build_initial_graph()
        propagator = BeliefPropagator(graph)
        # Minimal filter with no state names → DAG-only mode
        configs = {
            "expansion": RegimeConfig("expansion", np.eye(1), np.eye(1) * 0.01),
        }
        sf = ContinuousStateFilter(1, 1, configs, np.eye(1), np.eye(1) * 0.1)
        wm = WorldModel(
            graph=graph,
            propagator=propagator,
            state_filter=sf,
            continuous_state_names=[],  # empty → no Kalman beliefs
        )
        beliefs = wm.update(_expansion_features(), AS_OF)
        assert len(beliefs) == 9  # DAG only
        assert all(b.dist_type == "categorical" for b in beliefs)


class TestGraphHash:
    def test_hash_stable(self) -> None:
        wm = _build_full_world_model()
        h1 = wm.get_graph_hash()
        h2 = wm.get_graph_hash()
        assert h1 == h2
        assert len(h1) == 64


class TestMultipleUpdates:
    def test_sequential_updates(self) -> None:
        wm = _build_full_world_model()
        b1 = wm.update(_expansion_features(), AS_OF)
        b2 = wm.update(_crisis_features(), AS_OF + 86400)
        assert len(b1) == 12
        assert len(b2) == 12

        # After crisis features, regime should shift
        macro_b2 = next(b for b in b2 if b.variable_name == "regime.macro")
        assert macro_b2.probabilities["crisis"] > macro_b2.probabilities["expansion"]

    def test_kalman_state_evolves(self) -> None:
        wm = _build_full_world_model()
        wm.update(_expansion_features(), AS_OF)
        b1_stress = wm.query("latent.stress_level")

        wm.update(_expansion_features(), AS_OF + 86400)
        b2_stress = wm.query("latent.stress_level")

        # State should have evolved (not identical)
        assert (
            b1_stress.mean != b2_stress.mean or b1_stress.variance != b2_stress.variance
        )
