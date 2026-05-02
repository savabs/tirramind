"""Tests for Phase 24c — Multi-Asset Strategy Refactor.

Covers:
    - MultiAssetStrategy ABC contract
    - MultiAssetWalkForward: fold splits, dot-product P&L, attribution
    - EqualWeightStrategy: 1/N weights
    - BuyAndHoldBenchmarkStrategy: fixed target weights
    - MultiAssetSACStrategy: mocked SAC, state assembly, weight output
    - InstrumentStateAssembler: state_dim, instrument surprise block, entity block
    - Edge cases: single instrument, zero-variance, NaN, empty extra, wrong shapes
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.learning.policy.state_assembler import InstrumentStateAssembler
from agent.quant.backtest import (
    BuyAndHoldBenchmarkStrategy,
    EqualWeightStrategy,
    MultiAssetBacktestResult,
    MultiAssetStrategy,
    MultiAssetWalkForward,
)

# ── Helpers ───────────────────────────────────────────────────


def _synthetic_returns(
    T: int = 500,
    N: int = 5,
    seed: int = 42,
    mu: float = 0.0002,
    sigma: float = 0.015,
) -> np.ndarray:
    """Generate (T, N) synthetic log returns."""
    rng = np.random.RandomState(seed)
    return rng.normal(mu, sigma, (T, N))


def _instrument_names(N: int = 5) -> list[str]:
    tickers = ["SPY", "QQQ", "GLD", "CL=F", "AGG"]
    return tickers[:N]


def _instrument_classes() -> dict[str, str]:
    return {
        "SPY": "equity_etf",
        "QQQ": "equity_etf",
        "GLD": "commodity_future",
        "CL=F": "commodity_future",
        "AGG": "fixed_income",
    }


# ═══════════════════════════════════════════════════════════════
# MultiAssetStrategy ABC
# ═══════════════════════════════════════════════════════════════


class TestMultiAssetStrategyABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            MultiAssetStrategy()

    def test_subclass_requires_generate_weights(self):
        class BadStrat(MultiAssetStrategy):
            @property
            def name(self) -> str:
                return "bad"

        with pytest.raises(TypeError):
            BadStrat()


# ═══════════════════════════════════════════════════════════════
# EqualWeightStrategy
# ═══════════════════════════════════════════════════════════════


class TestEqualWeightStrategy:
    def test_name(self):
        assert EqualWeightStrategy().name == "equal_weight"

    def test_weights_shape(self):
        s = EqualWeightStrategy()
        ret = _synthetic_returns(T=100, N=5)
        w = s.generate_weights(ret, test_length=20, instrument_names=_instrument_names())
        assert w.shape == (20, 5)

    def test_weights_are_1_over_N(self):
        s = EqualWeightStrategy()
        N = 4
        ret = _synthetic_returns(T=100, N=N)
        w = s.generate_weights(ret, test_length=10, instrument_names=["A", "B", "C", "D"])
        np.testing.assert_allclose(w, 1.0 / N)

    def test_single_instrument(self):
        s = EqualWeightStrategy()
        ret = _synthetic_returns(T=50, N=1)
        w = s.generate_weights(ret, test_length=5, instrument_names=["SPY"])
        assert w.shape == (5, 1)
        np.testing.assert_allclose(w, 1.0)

    def test_weights_sum_to_one(self):
        s = EqualWeightStrategy()
        ret = _synthetic_returns(T=100, N=7)
        w = s.generate_weights(ret, test_length=15, instrument_names=[f"A{i}" for i in range(7)])
        np.testing.assert_allclose(w.sum(axis=1), 1.0)


# ═══════════════════════════════════════════════════════════════
# BuyAndHoldBenchmarkStrategy
# ═══════════════════════════════════════════════════════════════


class TestBuyAndHoldBenchmarkStrategy:
    def test_name_format(self):
        bh = BuyAndHoldBenchmarkStrategy({"SPY": 0.6, "AGG": 0.4})
        assert "SPY" in bh.name
        assert "AGG" in bh.name
        assert "buy_hold" in bh.name

    def test_weights_match_target(self):
        targets = {"SPY": 0.6, "AGG": 0.4}
        bh = BuyAndHoldBenchmarkStrategy(targets)
        names = ["SPY", "QQQ", "AGG"]
        ret = _synthetic_returns(T=100, N=3)
        w = bh.generate_weights(ret, test_length=20, instrument_names=names)
        assert w.shape == (20, 3)
        np.testing.assert_allclose(w[:, 0], 0.6)  # SPY
        np.testing.assert_allclose(w[:, 1], 0.0)  # QQQ not in targets
        np.testing.assert_allclose(w[:, 2], 0.4)  # AGG

    def test_constant_over_time(self):
        bh = BuyAndHoldBenchmarkStrategy({"SPY": 1.0})
        names = ["SPY", "QQQ"]
        ret = _synthetic_returns(T=100, N=2)
        w = bh.generate_weights(ret, test_length=50, instrument_names=names)
        # All rows identical
        for t in range(50):
            np.testing.assert_array_equal(w[t], w[0])

    def test_single_asset_spy(self):
        bh = BuyAndHoldBenchmarkStrategy({"SPY": 1.0})
        w = bh.generate_weights(
            _synthetic_returns(T=50, N=1),
            test_length=10,
            instrument_names=["SPY"],
        )
        np.testing.assert_allclose(w, 1.0)

    def test_unknown_instruments_get_zero(self):
        bh = BuyAndHoldBenchmarkStrategy({"AAPL": 0.5})
        names = ["SPY", "QQQ"]
        w = bh.generate_weights(
            _synthetic_returns(T=50, N=2),
            test_length=5,
            instrument_names=names,
        )
        np.testing.assert_allclose(w, 0.0)


# ═══════════════════════════════════════════════════════════════
# MultiAssetWalkForward
# ═══════════════════════════════════════════════════════════════


class TestMultiAssetWalkForward:
    def test_basic_fold_count(self):
        ret = _synthetic_returns(T=500, N=5)
        wf = MultiAssetWalkForward(
            min_train=252,
            test_size=21,
            step_size=21,
            instrument_names=_instrument_names(),
        )
        result = wf.run(EqualWeightStrategy(), ret)
        # (500 - 252) / 21 = 11.8 → 11 full folds
        expected_folds = (500 - 252) // 21
        assert len(result.folds) == expected_folds

    def test_result_type(self):
        ret = _synthetic_returns(T=400, N=3)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B", "C"],
        )
        result = wf.run(EqualWeightStrategy(), ret)
        assert isinstance(result, MultiAssetBacktestResult)
        assert result.strategy_name == "equal_weight"
        assert result.instrument_names == ["A", "B", "C"]

    def test_fold_sizes(self):
        ret = _synthetic_returns(T=400, N=3)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            step_size=50,
            instrument_names=["A", "B", "C"],
        )
        result = wf.run(EqualWeightStrategy(), ret)
        for fold in result.folds:
            assert fold.test_size == 50
            assert fold.weights.shape == (50, 3)
            assert fold.portfolio_returns.shape == (50,)
            assert fold.per_instrument_returns.shape == (50, 3)

    def test_portfolio_return_is_dot_product(self):
        """Portfolio return = sum(weights * returns) per timestep."""
        N = 3
        T = 300
        ret = _synthetic_returns(T=T, N=N, seed=99)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B", "C"],
        )
        result = wf.run(EqualWeightStrategy(), ret)
        fold = result.folds[0]
        test_ret = ret[200:250]
        expected_portfolio = (test_ret * fold.weights).sum(axis=1)
        np.testing.assert_allclose(fold.portfolio_returns, expected_portfolio, atol=1e-12)

    def test_equity_curve_is_cumulative(self):
        ret = _synthetic_returns(T=400, N=3)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B", "C"],
        )
        result = wf.run(EqualWeightStrategy(), ret)
        expected = np.exp(np.cumsum(result.all_portfolio_returns))
        np.testing.assert_allclose(result.equity_curve, expected, atol=1e-12)

    def test_aggregate_metrics_present(self):
        ret = _synthetic_returns(T=400, N=3)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B", "C"],
        )
        result = wf.run(EqualWeightStrategy(), ret)
        m = result.aggregate_metrics
        assert "sharpe" in m
        assert "max_drawdown" in m
        assert "max_weight" in m
        assert "mean_gross_leverage" in m
        assert np.isfinite(m["sharpe"])
        assert m["max_drawdown"] <= 0

    def test_attribution_sums_to_total(self):
        """Sum of per-class attribution should equal total portfolio return."""
        ret = _synthetic_returns(T=400, N=5, seed=77)
        names = _instrument_names(5)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=names,
            instrument_classes=_instrument_classes(),
        )
        result = wf.run(EqualWeightStrategy(), ret)
        attr_total = sum(result.attribution.values())
        portfolio_total = float(result.all_portfolio_returns.sum())
        assert abs(attr_total - portfolio_total) < 1e-10

    def test_attribution_keys_match_classes(self):
        names = _instrument_names(5)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=names,
            instrument_classes=_instrument_classes(),
        )
        result = wf.run(EqualWeightStrategy(), _synthetic_returns(T=400, N=5))
        # Should have equity_etf, commodity_future, fixed_income
        assert "equity_etf" in result.attribution
        assert "commodity_future" in result.attribution
        assert "fixed_income" in result.attribution

    def test_buy_and_hold_60_40(self):
        """60/40 strategy allocates correctly."""
        targets = {"SPY": 0.6, "AGG": 0.4}
        strat = BuyAndHoldBenchmarkStrategy(targets)
        names = ["SPY", "QQQ", "AGG"]
        ret = _synthetic_returns(T=400, N=3)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=names,
        )
        result = wf.run(strat, ret)
        # All weights should match targets
        for fold in result.folds:
            np.testing.assert_allclose(fold.weights[:, 0], 0.6)
            np.testing.assert_allclose(fold.weights[:, 1], 0.0)
            np.testing.assert_allclose(fold.weights[:, 2], 0.4)

    def test_all_weights_shape(self):
        ret = _synthetic_returns(T=500, N=5)
        wf = MultiAssetWalkForward(
            min_train=252,
            test_size=21,
            step_size=21,
            instrument_names=_instrument_names(),
        )
        result = wf.run(EqualWeightStrategy(), ret)
        n_folds = len(result.folds)
        assert result.all_weights.shape == (n_folds * 21, 5)
        assert result.all_portfolio_returns.shape == (n_folds * 21,)

    def test_concentration_metric(self):
        """max_weight should reflect the actual maximum single weight."""
        strat = BuyAndHoldBenchmarkStrategy({"SPY": 0.9, "AGG": 0.1})
        names = ["SPY", "AGG"]
        ret = _synthetic_returns(T=300, N=2)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=names,
        )
        result = wf.run(strat, ret)
        assert result.aggregate_metrics["max_weight"] == pytest.approx(0.9)


# ── Edge cases ───────────────────────────────────────────────


class TestMultiAssetEdgeCases:
    def test_1d_returns_rejected(self):
        ret = _synthetic_returns(T=400, N=5)[:, 0]  # flatten to 1-D
        wf = MultiAssetWalkForward(min_train=200, test_size=50)
        with pytest.raises(ValueError, match="2-D"):
            wf.run(EqualWeightStrategy(), ret)

    def test_insufficient_data(self):
        ret = _synthetic_returns(T=100, N=3)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B", "C"],
        )
        with pytest.raises(ValueError, match="Not enough data"):
            wf.run(EqualWeightStrategy(), ret)

    def test_mismatched_names_and_columns(self):
        ret = _synthetic_returns(T=400, N=3)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B"],  # 2 names, 3 columns
        )
        with pytest.raises(ValueError, match="instrument_names length"):
            wf.run(EqualWeightStrategy(), ret)

    def test_single_instrument_walkforward(self):
        """Single instrument should still work — N=1."""
        ret = _synthetic_returns(T=400, N=1)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["SPY"],
        )
        result = wf.run(EqualWeightStrategy(), ret)
        assert result.all_weights.shape[1] == 1
        np.testing.assert_allclose(result.all_weights, 1.0)

    def test_zero_variance_returns(self):
        """Zero variance (flat) returns should still produce valid metrics."""
        T, N = 400, 3
        ret = np.zeros((T, N))
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B", "C"],
        )
        result = wf.run(EqualWeightStrategy(), ret)
        assert np.isfinite(result.aggregate_metrics["total_return"])
        assert result.aggregate_metrics["total_return"] == pytest.approx(0.0)

    def test_nan_instrument_returns(self):
        """NaN in one instrument — portfolio return propagates NaN for that row."""
        ret = _synthetic_returns(T=400, N=3)
        ret[210, 1] = np.nan  # inject NaN in test window
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B", "C"],
        )
        result = wf.run(EqualWeightStrategy(), ret)
        # First fold includes the NaN row
        assert np.any(np.isnan(result.folds[0].portfolio_returns))

    def test_wrong_weight_shape_raises(self):
        """Strategy returning wrong shape should raise."""

        class BadWeightStrat(MultiAssetStrategy):
            @property
            def name(self) -> str:
                return "bad"

            def generate_weights(
                self,
                train_returns: np.ndarray,
                test_length: int,
                instrument_names: list[str],
                **kwargs: Any,
            ) -> np.ndarray:
                return np.ones((test_length, 999))  # wrong N

        ret = _synthetic_returns(T=400, N=3)
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B", "C"],
        )
        with pytest.raises(ValueError, match="weights shape"):
            wf.run(BadWeightStrat(), ret)

    def test_auto_generated_names(self):
        """If no instrument_names, should auto-generate."""
        ret = _synthetic_returns(T=400, N=3)
        wf = MultiAssetWalkForward(min_train=200, test_size=50)
        result = wf.run(EqualWeightStrategy(), ret)
        assert result.instrument_names == ["asset_0", "asset_1", "asset_2"]

    def test_attribution_unknown_class(self):
        """Instruments without a class mapping → 'unknown'."""
        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=["A", "B", "C"],
            instrument_classes={},  # no mapping
        )
        result = wf.run(EqualWeightStrategy(), _synthetic_returns(T=400, N=3))
        assert "unknown" in result.attribution


# ═══════════════════════════════════════════════════════════════
# InstrumentStateAssembler
# ═══════════════════════════════════════════════════════════════


class TestInstrumentStateAssembler:
    def test_state_dim(self):
        tickers = ["SPY", "QQQ", "GLD", "CL=F", "AGG"]
        asm = InstrumentStateAssembler(tickers)
        # N*5 + E*5 + E*4 + M + 1 + 4
        # 5*5 + 50*5 + 50*4 + 8 + 1 + 4 = 25 + 250 + 200 + 8 + 1 + 4 = 488
        assert asm.state_dim == 488
        assert asm.n_instruments == 5

    def test_state_dim_custom(self):
        asm = InstrumentStateAssembler(
            ["A", "B", "C"],
            max_entities=10,
            market_dim=4,
        )
        # 3*5 + 10*5 + 10*4 + 4 + 1 + 4 = 15 + 50 + 40 + 4 + 1 + 4 = 114
        assert asm.state_dim == 114

    def test_assemble_empty(self):
        asm = InstrumentStateAssembler(["SPY", "QQQ"])
        state, meta = asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            beliefs=[],
            market_features={},
        )
        assert state.shape == (asm.state_dim,)
        assert meta["n_instruments_active"] == 0
        assert meta["n_entities_active"] == 0

    def test_instrument_surprise_block(self):
        tickers = ["SPY", "QQQ"]
        asm = InstrumentStateAssembler(tickers, max_entities=2, market_dim=2)
        surp = {
            "SPY": (1.0, 2.0, 3.0, 4.0, 5.0),
            "QQQ": (0.5, 0.5, 0.5, 0.5, 0.5),
        }
        state, meta = asm.assemble(
            instrument_surprises=surp,
            entity_alerts=[],
            beliefs=[],
            market_features={},
        )
        # First 2*5 = 10 elements are instrument surprises
        inst_block = state[:10].numpy()
        np.testing.assert_allclose(inst_block[:5], [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(inst_block[5:10], [0.5, 0.5, 0.5, 0.5, 0.5])
        assert meta["n_instruments_active"] == 2

    def test_missing_instrument_is_zero(self):
        tickers = ["SPY", "QQQ", "GLD"]
        asm = InstrumentStateAssembler(tickers, max_entities=2, market_dim=2)
        surp = {"SPY": (1.0, 1.0, 1.0, 1.0, 1.0)}  # only SPY
        state, _ = asm.assemble(
            instrument_surprises=surp,
            entity_alerts=[],
            beliefs=[],
            market_features={},
        )
        inst_block = state[:15].numpy().reshape(3, 5)
        np.testing.assert_allclose(inst_block[0], 1.0)  # SPY
        np.testing.assert_allclose(inst_block[1], 0.0)  # QQQ missing
        np.testing.assert_allclose(inst_block[2], 0.0)  # GLD missing

    def test_unknown_instrument_ignored(self):
        asm = InstrumentStateAssembler(["SPY"])
        surp = {"UNKNOWN_TICKER": (9.0, 9.0, 9.0, 9.0, 9.0)}
        state, meta = asm.assemble(
            instrument_surprises=surp,
            entity_alerts=[],
            beliefs=[],
            market_features={},
        )
        # Unknown ticker should not appear anywhere
        assert meta["n_instruments_active"] == 0
        inst_block = state[:5].numpy()
        np.testing.assert_allclose(inst_block, 0.0)

    def test_instrument_tickers_in_metadata(self):
        tickers = ["SPY", "QQQ"]
        asm = InstrumentStateAssembler(tickers)
        _, meta = asm.assemble(
            instrument_surprises={},
            entity_alerts=[],
            beliefs=[],
            market_features={},
        )
        assert meta["instrument_tickers"] == ["SPY", "QQQ"]


# ═══════════════════════════════════════════════════════════════
# MultiAssetSACStrategy (mocked SAC)
# ═══════════════════════════════════════════════════════════════


class TestMultiAssetSACStrategy:
    def _make_mocked_strategy(self, N: int = 5):
        """Create a MultiAssetSACStrategy with a mocked SACTrainer."""
        from agent.learning.policy.portfolio_strategy import MultiAssetSACStrategy

        tickers = _instrument_names(N)
        assembler = InstrumentStateAssembler(tickers, max_entities=10, market_dim=4)

        mock_trainer = MagicMock()
        # SAC returns N-dim action
        mock_trainer.select_action.return_value = np.full(N, 0.1)

        return MultiAssetSACStrategy(mock_trainer, assembler), mock_trainer

    def test_name(self):
        strat, _ = self._make_mocked_strategy()
        assert strat.name == "multi_asset_sac"

    def test_generate_weights_shape(self):
        strat, _ = self._make_mocked_strategy(N=5)
        ret = _synthetic_returns(T=100, N=5)
        w = strat.generate_weights(ret, test_length=10, instrument_names=_instrument_names(5))
        assert w.shape == (10, 5)

    def test_no_extra_returns_zeros(self):
        strat, _ = self._make_mocked_strategy(N=3)
        ret = _synthetic_returns(T=100, N=3)
        w = strat.generate_weights(
            ret,
            test_length=5,
            instrument_names=["A", "B", "C"],
            test_extra=None,
        )
        np.testing.assert_allclose(w, 0.0)

    def test_sac_called_per_timestep(self):
        strat, mock_trainer = self._make_mocked_strategy(N=3)
        test_len = 7
        ret = _synthetic_returns(T=100, N=3)

        test_extra = {
            "instrument_surprises": [{"SPY": (0.1,) * 5}] * test_len,
            "entity_alerts": [[]] * test_len,
            "beliefs": [[]] * test_len,
            "market_features": [{}] * test_len,
        }

        strat.generate_weights(
            ret,
            test_length=test_len,
            instrument_names=["SPY", "QQQ", "GLD"],
            test_extra=test_extra,
        )
        assert mock_trainer.select_action.call_count == test_len

    def test_deterministic_flag(self):
        strat, mock_trainer = self._make_mocked_strategy(N=2)
        test_extra = {
            "instrument_surprises": [{}],
            "entity_alerts": [[]],
            "beliefs": [[]],
            "market_features": [{}],
        }
        strat.generate_weights(
            _synthetic_returns(T=50, N=2),
            test_length=1,
            instrument_names=["A", "B"],
            test_extra=test_extra,
        )
        # Should be called with deterministic=True
        call_args = mock_trainer.select_action.call_args
        assert call_args[1].get("deterministic") is True or call_args[0][1] is True

    def test_weights_match_sac_action(self):
        strat, mock_trainer = self._make_mocked_strategy(N=3)
        expected_action = np.array([0.3, -0.1, 0.5])
        mock_trainer.select_action.return_value = expected_action

        test_extra = {
            "instrument_surprises": [{"SPY": (1.0,) * 5}],
            "entity_alerts": [[]],
            "beliefs": [[]],
            "market_features": [{}],
        }

        w = strat.generate_weights(
            _synthetic_returns(T=50, N=3),
            test_length=1,
            instrument_names=["SPY", "QQQ", "GLD"],
            test_extra=test_extra,
        )
        np.testing.assert_allclose(w[0], expected_action)

    def test_in_walkforward(self):
        """MultiAssetSACStrategy should work inside MultiAssetWalkForward."""
        N = 3
        strat, mock_trainer = self._make_mocked_strategy(N=N)
        mock_trainer.select_action.return_value = np.full(N, 1.0 / N)

        ret = _synthetic_returns(T=400, N=N)
        names = _instrument_names(N)

        # Build per-timestep test_extra in the extra dict
        # The WalkForward slices extra, so we need arrays of length T
        T = 400
        extra = {
            "instrument_surprises": np.array([{"SPY": (0.1,) * 5}] * T, dtype=object),
            "entity_alerts": np.array([[]] * T, dtype=object),
            "beliefs": np.array([[]] * T, dtype=object),
            "market_features": np.array([{}] * T, dtype=object),
        }

        wf = MultiAssetWalkForward(
            min_train=200,
            test_size=50,
            instrument_names=names,
        )
        result = wf.run(strat, ret, extra=extra)
        assert len(result.folds) > 0
        # Each weight should be 1/3
        np.testing.assert_allclose(result.all_weights, 1.0 / N, atol=1e-10)
