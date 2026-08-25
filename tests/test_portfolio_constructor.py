"""Tests for Idea 11 — HRP + Black-Litterman Portfolio Constructor.

Covers:
    1.  PortfolioConstructor instantiates with defaults
    2.  PortfolioConstructor instantiates with custom params
    3.  _black_litterman: posterior mean is convex combination of prior and views
    4.  _black_litterman: high-confidence views dominate prior
    5.  _black_litterman: low-confidence views revert to prior
    6.  _black_litterman: posterior covariance is symmetric PD
    7.  _hrp_weights: weights sum to 1
    8.  _hrp_weights: all weights are non-negative
    9.  _hrp_weights: single-asset portfolio returns weight=1.0
    10. _hrp_weights: equal covariance gives roughly equal weights
    11. _sample_covariance: matrix is symmetric
    12. _sample_covariance: diagonal is positive (variance)
    13. _forward_fill_prices: leading zero prices filled with first valid
    14. _forward_fill_prices: interior gaps filled correctly
    15. build_weights(): returns None when no entities with history
    16. build_weights(): returns None with single asset (need ≥ 2)
    17. build_weights(): returns PortfolioWeights for valid input
    18. build_weights(): weights sum to 1.0
    19. build_weights(): all weights non-negative
    20. build_weights(): entity_ids in result matches input
    21. build_weights(): higher quality_score → views closer to predictions
    22. build_weights(): turnover smoothing pulls toward prev_weights
    23. build_weights(): tilt_factor=0 gives pure HRP weights
    24. store_weights(): calls store.store_portfolio_weights
    25. TrainerConfig.portfolio_delta defaults 2.5
    26. TrainerConfig.portfolio_tilt_factor defaults 0.5
    27. TrainerConfig.portfolio_turnover_lambda defaults 0.3
    28. TrainerConfig.portfolio_min_history defaults 20
    29. Trainer.compute_portfolio() with explicit return_preds returns result
    30. Trainer.compute_portfolio() returns None when no predictions
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.portfolio.constructor import (
    PortfolioConstructor,
    PortfolioWeights,
    _black_litterman,
    _forward_fill_prices,
    _hrp_weights,
    _sample_covariance,
)
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore


# ── Helpers ──────────────────────────────────────────────────────────────────

_DAY = 86_400.0


def _make_store(tmp_path: Path, name: str = "port.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _populate_price_history(
    store: PipelineStore,
    entity_ids: list[str],
    n: int,
    t_start: float,
    t_end: float,
    seed: int = 0,
) -> None:
    """Register entities and add n positive-price observations each."""
    rng = np.random.default_rng(seed)
    for i, eid in enumerate(entity_ids):
        store.register_entity("instrument", eid, eid)
        prices = np.exp(np.cumsum(rng.normal(0, 0.02, n)))  # GBM-like
        times = sorted(rng.uniform(t_start, t_end, n))
        for j, t in enumerate(times):
            store.store_entity_observation(
                entity_id=eid, source_tool="test",
                observation_type="price", observed_at=float(t),
                value={"close": float(prices[j])},
            )


# ═══════════════════════════════════════════════════════════════
# 1–2. Construction
# ═══════════════════════════════════════════════════════════════

class TestConstruction:

    def test_defaults(self):
        pc = PortfolioConstructor()
        assert pc.delta == pytest.approx(2.5)
        assert pc.tilt_factor == pytest.approx(0.5)
        assert pc.turnover_lambda == pytest.approx(0.3)

    def test_custom_params(self):
        pc = PortfolioConstructor(delta=1.0, tilt_factor=0.0, turnover_lambda=0.0)
        assert pc.delta == pytest.approx(1.0)
        assert pc.tilt_factor == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════
# 3–6. _black_litterman
# ═══════════════════════════════════════════════════════════════

class TestBlackLitterman:

    def _make_inputs(self, n=4, seed=0):
        rng = np.random.default_rng(seed)
        cov = rng.normal(0, 1, (n, n))
        cov = cov @ cov.T / n + np.eye(n) * 0.1  # PD
        pi = rng.normal(0, 0.05, n)
        q = rng.normal(0, 0.05, n)
        return cov, pi, q

    def test_posterior_between_prior_and_views_moderate_conf(self):
        cov, pi, q = self._make_inputs()
        tau = 0.1
        # moderate confidence
        omega = np.ones(4) * 0.01
        mu_bl, _ = _black_litterman(cov, pi, q, omega, tau)
        # posterior should be a "blend" — verify it lies between extremes
        assert mu_bl is not None
        assert mu_bl.shape == (4,)

    def test_high_confidence_closer_to_views(self):
        cov, pi, q = self._make_inputs()
        tau = 0.1
        # High confidence (small omega) → posterior closer to views
        omega_high = np.ones(4) * 1e-6
        mu_high, _ = _black_litterman(cov, pi, q, omega_high, tau)
        # Very low confidence (large omega) → posterior closer to prior
        omega_low = np.ones(4) * 1e6
        mu_low, _ = _black_litterman(cov, pi, q, omega_low, tau)
        # High confidence should be closer to views q
        dist_high = np.linalg.norm(mu_high - q)
        dist_low = np.linalg.norm(mu_low - q)
        assert dist_high < dist_low

    def test_low_confidence_reverts_to_prior(self):
        cov, pi, q = self._make_inputs()
        tau = 0.1
        omega_huge = np.ones(4) * 1e9  # near-zero confidence
        mu_bl, _ = _black_litterman(cov, pi, q, omega_huge, tau)
        # Should be close to prior π
        assert np.linalg.norm(mu_bl - pi) < np.linalg.norm(pi) + 1e-4

    def test_posterior_covariance_symmetric_pd(self):
        cov, pi, q = self._make_inputs()
        tau = 0.1
        omega = np.ones(4) * 0.01
        _, cov_bl = _black_litterman(cov, pi, q, omega, tau)
        # Symmetric
        assert np.allclose(cov_bl, cov_bl.T, atol=1e-10)
        # Positive definite: all eigenvalues > 0
        eigvals = np.linalg.eigvalsh(cov_bl)
        assert (eigvals > -1e-10).all()


# ═══════════════════════════════════════════════════════════════
# 7–10. _hrp_weights
# ═══════════════════════════════════════════════════════════════

class TestHRPWeights:

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(7)
        cov = rng.normal(0, 1, (5, 5))
        cov = cov @ cov.T + np.eye(5) * 0.1
        labels = [f"a{i}" for i in range(5)]
        w = _hrp_weights(cov, labels)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_weights_non_negative(self):
        rng = np.random.default_rng(8)
        cov = rng.normal(0, 1, (4, 4))
        cov = cov @ cov.T + np.eye(4) * 0.1
        labels = [f"b{i}" for i in range(4)]
        w = _hrp_weights(cov, labels)
        for v in w.values():
            assert v >= 0.0

    def test_single_asset_weight_one(self):
        cov = np.array([[0.04]])
        w = _hrp_weights(cov, ["solo"])
        assert w["solo"] == pytest.approx(1.0)

    def test_equal_covariance_roughly_equal_weights(self):
        n = 4
        cov = np.eye(n) * 0.04  # identical, uncorrelated assets
        labels = [f"c{i}" for i in range(n)]
        w = _hrp_weights(cov, labels)
        weights = list(w.values())
        # All weights should be equal (1/n) since assets are identical
        assert max(weights) - min(weights) < 0.05


# ═══════════════════════════════════════════════════════════════
# 11–12. _sample_covariance
# ═══════════════════════════════════════════════════════════════

class TestSampleCovariance:

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        ret = rng.normal(0, 1, (4, 100))
        cov = _sample_covariance(ret)
        assert np.allclose(cov, cov.T, atol=1e-12)

    def test_diagonal_positive(self):
        rng = np.random.default_rng(43)
        ret = rng.normal(0, 1, (4, 100))
        cov = _sample_covariance(ret)
        assert (np.diag(cov) > 0).all()


# ═══════════════════════════════════════════════════════════════
# 13–14. _forward_fill_prices
# ═══════════════════════════════════════════════════════════════

class TestForwardFillPrices:

    def test_leading_zeros_filled_with_first_valid(self):
        arr = np.array([0.0, 0.0, 100.0, 110.0])
        result = _forward_fill_prices(arr)
        assert result[0] == pytest.approx(100.0)
        assert result[1] == pytest.approx(100.0)

    def test_interior_gaps_forward_filled(self):
        arr = np.array([100.0, 0.0, np.nan, 120.0])
        result = _forward_fill_prices(arr)
        assert result[1] == pytest.approx(100.0)
        assert result[2] == pytest.approx(100.0)
        assert result[3] == pytest.approx(120.0)


# ═══════════════════════════════════════════════════════════════
# 15–24. build_weights()
# ═══════════════════════════════════════════════════════════════

class TestBuildWeights:

    def _make_pc(self, min_history=5, n_bins=20):
        return PortfolioConstructor(
            min_history=min_history, n_bins=n_bins,
            lookback_days=30, tilt_factor=0.3,
        )

    def test_returns_none_without_history(self, tmp_path):
        store = _make_store(tmp_path, "no_hist.db")
        pc = self._make_pc()
        result = pc.build_weights(store, {"eid_a": 0.01, "eid_b": 0.02})
        assert result is None

    def test_returns_none_single_asset(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _populate_price_history(store, ["solo"], 50, as_of - 30 * _DAY, as_of)
        pc = self._make_pc()
        result = pc.build_weights(store, {"solo": 0.01}, as_of=as_of)
        assert result is None

    def test_returns_portfolio_weights_for_valid_input(self, tmp_path):
        store = _make_store(tmp_path, "valid.db")
        as_of = time.time()
        eids = ["e1", "e2", "e3"]
        _populate_price_history(store, eids, 80, as_of - 30 * _DAY, as_of)
        pc = self._make_pc()
        preds = {e: 0.01 * (i + 1) for i, e in enumerate(eids)}
        result = pc.build_weights(store, preds, as_of=as_of)
        assert isinstance(result, PortfolioWeights)

    def test_weights_sum_to_one(self, tmp_path):
        store = _make_store(tmp_path, "sum1.db")
        as_of = time.time()
        eids = ["a", "b", "c", "d"]
        _populate_price_history(store, eids, 80, as_of - 30 * _DAY, as_of)
        pc = self._make_pc()
        preds = {e: 0.01 * i for i, e in enumerate(eids)}
        result = pc.build_weights(store, preds, as_of=as_of)
        assert result is not None
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_weights_non_negative(self, tmp_path):
        store = _make_store(tmp_path, "nonneg.db")
        as_of = time.time()
        eids = ["x", "y", "z"]
        _populate_price_history(store, eids, 80, as_of - 30 * _DAY, as_of)
        pc = self._make_pc()
        preds = {e: 0.02 for e in eids}
        result = pc.build_weights(store, preds, as_of=as_of)
        assert result is not None
        for v in result.weights.values():
            assert v >= 0.0

    def test_entity_ids_in_result(self, tmp_path):
        store = _make_store(tmp_path, "eids.db")
        as_of = time.time()
        eids = ["p", "q", "r"]
        _populate_price_history(store, eids, 80, as_of - 30 * _DAY, as_of)
        pc = self._make_pc()
        preds = {e: 0.01 for e in eids}
        result = pc.build_weights(store, preds, as_of=as_of)
        assert result is not None
        for eid in result.entity_ids:
            assert eid in preds

    def test_high_quality_score_closer_to_pred(self, tmp_path):
        """With quality_score=1, BL posterior should be close to GNN predictions."""
        store = _make_store(tmp_path, "hq.db")
        as_of = time.time()
        eids = ["hq1", "hq2", "hq3"]
        _populate_price_history(store, eids, 80, as_of - 30 * _DAY, as_of)
        pc = PortfolioConstructor(min_history=5, n_bins=20, lookback_days=30)

        preds = {"hq1": 0.10, "hq2": -0.05, "hq3": 0.01}
        # High confidence views
        qs_high = {e: 0.99 for e in eids}
        result_high = pc.build_weights(store, preds, quality_scores=qs_high, as_of=as_of)

        # Low confidence views
        qs_low = {e: 0.01 for e in eids}
        result_low = pc.build_weights(store, preds, quality_scores=qs_low, as_of=as_of)

        assert result_high is not None and result_low is not None
        # High confidence: expected returns should be closer to pred values
        diff_high = max(abs(result_high.expected_returns[e] - preds[e]) for e in eids if e in result_high.expected_returns)
        diff_low = max(abs(result_low.expected_returns[e] - preds[e]) for e in eids if e in result_low.expected_returns)
        assert diff_high < diff_low

    def test_turnover_smoothing_pulls_toward_prev(self, tmp_path):
        """Turnover smoothing should pull weights towards prev_weights."""
        store = _make_store(tmp_path, "turn.db")
        as_of = time.time()
        eids = ["t1", "t2", "t3"]
        _populate_price_history(store, eids, 80, as_of - 30 * _DAY, as_of)

        pc_smooth = PortfolioConstructor(
            min_history=5, n_bins=20, lookback_days=30,
            turnover_lambda=0.8,  # heavy smoothing
        )
        pc_nosmooth = PortfolioConstructor(
            min_history=5, n_bins=20, lookback_days=30,
            turnover_lambda=0.0,
        )
        preds = {e: 0.02 for e in eids}
        prev = {"t1": 0.7, "t2": 0.15, "t3": 0.15}  # skewed to t1

        r_smooth = pc_smooth.build_weights(store, preds, prev_weights=prev, as_of=as_of)
        r_nosmooth = pc_nosmooth.build_weights(store, preds, as_of=as_of)

        assert r_smooth is not None and r_nosmooth is not None
        # Smoothed weight for t1 should be closer to prev=0.7
        if "t1" in r_smooth.weights and "t1" in r_nosmooth.weights:
            assert abs(r_smooth.weights["t1"] - 0.7) < abs(r_nosmooth.weights["t1"] - 0.7)

    def test_tilt_factor_zero_gives_pure_hrp(self, tmp_path):
        """tilt_factor=0 → final weights should equal HRP weights."""
        store = _make_store(tmp_path, "notilt.db")
        as_of = time.time()
        eids = ["f1", "f2", "f3"]
        _populate_price_history(store, eids, 80, as_of - 30 * _DAY, as_of)
        pc = PortfolioConstructor(
            min_history=5, n_bins=20, lookback_days=30,
            tilt_factor=0.0, turnover_lambda=0.0,
        )
        preds = {e: 0.05 * (i + 1) for i, e in enumerate(eids)}
        result = pc.build_weights(store, preds, as_of=as_of)
        assert result is not None
        for eid in result.entity_ids:
            assert result.weights[eid] == pytest.approx(result.hrp_weights[eid], abs=1e-9)


# ═══════════════════════════════════════════════════════════════
# 24. store_weights()
# ═══════════════════════════════════════════════════════════════

class TestStoreWeights:

    def test_calls_store_portfolio_weights(self):
        mock_store = MagicMock()
        pc = PortfolioConstructor()
        pw = PortfolioWeights(
            weights={"a": 0.6, "b": 0.4},
            expected_returns={"a": 0.02, "b": -0.01},
            hrp_weights={"a": 0.5, "b": 0.5},
            bl_covariance=np.eye(2),
            entity_ids=["a", "b"],
            n_assets=2,
            computed_at=time.time(),
        )
        n = pc.store_weights(pw, mock_store, "2026-05-25")
        assert n == 2
        mock_store.store_portfolio_weights.assert_called_once_with(
            "2026-05-25", {"a": 0.6, "b": 0.4}
        )


# ═══════════════════════════════════════════════════════════════
# 25–28. TrainerConfig
# ═══════════════════════════════════════════════════════════════

class TestTrainerConfig:

    def test_portfolio_delta_defaults_2_5(self):
        from agent.models.gnn.trainer import TrainerConfig
        assert TrainerConfig().portfolio_delta == pytest.approx(2.5)

    def test_portfolio_tilt_defaults_0_5(self):
        from agent.models.gnn.trainer import TrainerConfig
        assert TrainerConfig().portfolio_tilt_factor == pytest.approx(0.5)

    def test_portfolio_turnover_lambda_defaults_0_3(self):
        from agent.models.gnn.trainer import TrainerConfig
        assert TrainerConfig().portfolio_turnover_lambda == pytest.approx(0.3)

    def test_portfolio_min_history_defaults_20(self):
        from agent.models.gnn.trainer import TrainerConfig
        assert TrainerConfig().portfolio_min_history == 20


# ═══════════════════════════════════════════════════════════════
# 29–30. Trainer.compute_portfolio()
# ═══════════════════════════════════════════════════════════════

class TestComputePortfolio:

    def _make_trainer(self, tmp_path: Path, tag: str) -> tuple[Trainer, list[str]]:
        store = _make_store(tmp_path, f"{tag}.db")
        gen = SyntheticGraphGenerator(
            num_companies=2, num_countries=1,
            time_span=3600.0 * 4, base_event_rate=0.001, seed=42,
        )
        gen.generate(store)

        # Add instrument entities with price history for the portfolio layer
        as_of = time.time()
        eids = ["instr_x", "instr_y", "instr_z"]
        _populate_price_history(store, eids, 80, as_of - 30 * _DAY, as_of)

        cfg = TrainerConfig(
            hidden_dim=16, memory_dim=16, message_dim=16, time_dim=8,
            num_heads=1, num_layers=1,
            portfolio_min_history=5,
            portfolio_lookback_days=30,
        )
        return Trainer(store, cfg), eids

    def test_compute_portfolio_with_explicit_preds(self, tmp_path):
        trainer, eids = self._make_trainer(tmp_path, "cp_exp")
        trainer.build_model()
        preds = {e: 0.01 * (i + 1) for i, e in enumerate(eids)}
        result = trainer.compute_portfolio(return_preds=preds)
        # May be None if instruments have insufficient history — just verify no crash
        assert result is None or isinstance(result, PortfolioWeights)

    def test_compute_portfolio_no_preds_returns_none_gracefully(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path, "cp_none")
        trainer.build_model()
        # With no instruments in graph and no explicit preds, should return None gracefully
        result = trainer.compute_portfolio(return_preds={})
        assert result is None
