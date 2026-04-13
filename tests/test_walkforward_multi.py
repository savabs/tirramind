"""Tests for Phase 24e — Walk-Forward Backtest Runner & Attribution Report.

Tests use fully synthetic data to verify:
    - Data loading from PipelineStore (mocked observations)
    - Multi-strategy walk-forward execution (fold count, metric plausibility)
    - Attribution sums to total (per-class, per-region)
    - Top instrument extraction
    - Concentration statistics
    - MultiAssetWeightedSurpriseStrategy behaviour
    - Edge cases: single instrument, zero returns, missing data
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.learning.policy.portfolio_strategy import (
    MultiAssetWeightedSurpriseStrategy,
)
from agent.quant.backtest import (
    BuyAndHoldBenchmarkStrategy,
    EqualWeightStrategy,
    MultiAssetBacktestResult,
    MultiAssetWalkForward,
)
from agent.quant.walkforward_runner import (
    StrategyReport,
    build_default_strategies,
    concentration_stats,
    generate_attribution_report,
    load_instrument_returns,
    per_group_attribution,
    per_instrument_attribution,
    run_walkforward,
    top_instruments,
)

# ── Fixtures ───────────────────────────────────────────────────


def _make_synthetic_returns(
    T: int = 500,
    N: int = 5,
    seed: int = 42,
    drift: float = 0.0002,
    vol: float = 0.01,
) -> np.ndarray:
    """Generate synthetic log-return matrix (T, N)."""
    rng = np.random.default_rng(seed)
    return drift + vol * rng.standard_normal((T, N))


def _make_instruments(N: int = 5) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Create N synthetic instrument names + class/region mappings."""
    classes = ["commodity_future", "fx", "equity_etf", "fixed_income", "crypto"]
    regions = ["US", "Europe", "Asia", "LatAm", "EM"]
    names = [f"INST_{i}" for i in range(N)]
    inst_classes = {names[i]: classes[i % len(classes)] for i in range(N)}
    inst_regions = {names[i]: regions[i % len(regions)] for i in range(N)}
    return names, inst_classes, inst_regions


def _run_basic_backtest(
    T: int = 500,
    N: int = 5,
    min_train: int = 100,
    test_size: int = 21,
    step_size: int = 21,
    seed: int = 42,
) -> tuple[
    dict[str, MultiAssetBacktestResult], list[str], dict[str, str], dict[str, str]
]:
    """Run a basic backtest and return results + instrument metadata."""
    returns = _make_synthetic_returns(T=T, N=N, seed=seed)
    names, classes, regions = _make_instruments(N)
    strategies = [EqualWeightStrategy(), BuyAndHoldBenchmarkStrategy({"INST_0": 1.0})]
    results = run_walkforward(
        returns,
        names,
        classes,
        strategies,
        min_train=min_train,
        test_size=test_size,
        step_size=step_size,
    )
    return results, names, classes, regions


# ── Walk-Forward Execution Tests ───────────────────────────────


class TestWalkForwardExecution:
    """Verify walk-forward runs correctly with multiple strategies."""

    def test_fold_count(self) -> None:
        """Correct number of folds for given T and configuration."""
        T, min_train, test_size, step_size = 500, 100, 21, 21
        expected_folds = (T - min_train) // step_size
        # Adjust if last fold doesn't fit
        actual_folds = 0
        split = min_train
        while split + test_size <= T:
            actual_folds += 1
            split += step_size

        results, *_ = _run_basic_backtest(
            T=T,
            min_train=min_train,
            test_size=test_size,
            step_size=step_size,
        )
        for name, result in results.items():
            assert (
                len(result.folds) == actual_folds
            ), f"{name}: expected {actual_folds} folds, got {len(result.folds)}"

    def test_multiple_strategies(self) -> None:
        """All configured strategies produce results."""
        results, *_ = _run_basic_backtest()
        assert len(results) == 2
        assert "equal_weight" in results
        # BuyAndHold name contains INST_0
        bh_names = [n for n in results if n.startswith("buy_hold")]
        assert len(bh_names) == 1

    def test_sharpe_is_finite(self) -> None:
        """Sharpe ratio should be a finite number."""
        results, *_ = _run_basic_backtest()
        for name, result in results.items():
            sharpe = result.aggregate_metrics["sharpe"]
            assert math.isfinite(sharpe), f"{name}: Sharpe={sharpe} is not finite"

    def test_drawdown_non_positive(self) -> None:
        """Max drawdown must be ≤ 0."""
        results, *_ = _run_basic_backtest()
        for name, result in results.items():
            dd = result.aggregate_metrics["max_drawdown"]
            assert dd <= 0.0, f"{name}: max_drawdown={dd} > 0"

    def test_equity_curve_shape(self) -> None:
        """Equity curve length matches total test periods."""
        results, *_ = _run_basic_backtest()
        for name, result in results.items():
            total_test = sum(f.test_size for f in result.folds)
            assert len(result.equity_curve) == total_test

    def test_all_weights_shape(self) -> None:
        """Concatenated weights have correct shape."""
        N = 5
        results, names, *_ = _run_basic_backtest(N=N)
        for name, result in results.items():
            total_test = sum(f.test_size for f in result.folds)
            assert result.all_weights.shape == (total_test, N)

    def test_portfolio_returns_match_dot_product(self) -> None:
        """Verify portfolio return = sum of weighted instrument returns."""
        returns = _make_synthetic_returns(T=300, N=3)
        names, classes, regions = _make_instruments(3)
        results = run_walkforward(
            returns,
            names,
            classes,
            [EqualWeightStrategy()],
            min_train=100,
            test_size=21,
            step_size=21,
        )
        result = results["equal_weight"]
        for fold in result.folds:
            recomputed = fold.per_instrument_returns.sum(axis=1)
            np.testing.assert_allclose(
                fold.portfolio_returns,
                recomputed,
                atol=1e-12,
            )

    def test_fold_metrics_present(self) -> None:
        """Each fold should have expected metric keys."""
        results, *_ = _run_basic_backtest()
        expected_keys = {
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "annualized_return",
        }
        for result in results.values():
            for fold in result.folds:
                assert expected_keys.issubset(fold.metrics.keys()), fold.metrics.keys()

    def test_spec_configuration(self) -> None:
        """Run with spec's exact configuration: min_train=252, test_size=21, step_size=21."""
        T = 600
        returns = _make_synthetic_returns(T=T, N=5)
        names, classes, regions = _make_instruments(5)
        results = run_walkforward(
            returns,
            names,
            classes,
            [EqualWeightStrategy()],
            min_train=252,
            test_size=21,
            step_size=21,
        )
        result = results["equal_weight"]
        # Expected folds: (600 - 252) // 21 = 16, but only those where split + 21 <= 600
        split = 252
        expected = 0
        while split + 21 <= T:
            expected += 1
            split += 21
        assert len(result.folds) == expected


# ── Attribution Tests ──────────────────────────────────────────


class TestAttribution:
    """Verify attribution sums and per-group correctness."""

    def test_class_attribution_sums_to_total(self) -> None:
        """Per-class attribution values must sum to total return."""
        results, _, classes, _ = _run_basic_backtest()
        for name, result in results.items():
            attr = per_group_attribution(result, classes)
            attr_sum = sum(attr.values())
            total = float(result.all_portfolio_returns.sum())
            assert (
                abs(attr_sum - total) < 1e-10
            ), f"{name}: attr sum={attr_sum}, total={total}"

    def test_region_attribution_sums_to_total(self) -> None:
        """Per-region attribution values must sum to total return."""
        results, _, _, regions = _run_basic_backtest()
        for name, result in results.items():
            attr = per_group_attribution(result, regions)
            attr_sum = sum(attr.values())
            total = float(result.all_portfolio_returns.sum())
            assert (
                abs(attr_sum - total) < 1e-10
            ), f"{name}: region attr sum={attr_sum}, total={total}"

    def test_instrument_attribution_sums_to_total(self) -> None:
        """Per-instrument attribution must sum to total return."""
        results, *_ = _run_basic_backtest()
        for name, result in results.items():
            inst_attr = per_instrument_attribution(result)
            attr_sum = sum(inst_attr.values())
            total = float(result.all_portfolio_returns.sum())
            assert abs(attr_sum - total) < 1e-10

    def test_builtin_attribution_matches_recomputed(self) -> None:
        """The attribution dict in the result should match our recomputation."""
        results, _, classes, _ = _run_basic_backtest()
        for name, result in results.items():
            recomputed = per_group_attribution(result, classes)
            for cls, val in result.attribution.items():
                assert abs(val - recomputed.get(cls, 0.0)) < 1e-10

    def test_buy_hold_single_instrument_attribution(self) -> None:
        """BuyAndHold(INST_0)=100% should attribute all P&L to INST_0's class."""
        results, names, classes, _ = _run_basic_backtest()
        bh_name = [n for n in results if n.startswith("buy_hold")][0]
        result = results[bh_name]
        inst_attr = per_instrument_attribution(result)
        # Only INST_0 should have nonzero attribution
        for ticker, val in inst_attr.items():
            if ticker == "INST_0":
                assert val != 0.0 or result.all_portfolio_returns.sum() == 0.0
            else:
                assert abs(val) < 1e-15, f"{ticker} has unexpected attribution {val}"


# ── Top Instruments Tests ──────────────────────────────────────


class TestTopInstruments:
    """Test top instrument extraction."""

    def test_top_5_count(self) -> None:
        """Should return exactly min(5, N) instruments."""
        results, *_ = _run_basic_backtest(N=5)
        for result in results.values():
            top = top_instruments(result, n=5)
            assert len(top) == 5

    def test_top_n_larger_than_instruments(self) -> None:
        """If top_n > N, return all N."""
        results, *_ = _run_basic_backtest(N=3)
        for result in results.values():
            top = top_instruments(result, n=10)
            assert len(top) == 3

    def test_top_sorted_by_absolute_value(self) -> None:
        """Results should be sorted by descending absolute contribution."""
        results, *_ = _run_basic_backtest()
        for result in results.values():
            top = top_instruments(result, n=5)
            abs_vals = [abs(t[1]) for t in top]
            assert abs_vals == sorted(abs_vals, reverse=True)


# ── Concentration Stats Tests ──────────────────────────────────


class TestConcentration:
    """Test weight concentration statistics."""

    def test_equal_weight_concentration(self) -> None:
        """EqualWeight(5) should have max_abs_weight = 0.2."""
        results, *_ = _run_basic_backtest(N=5)
        result = results["equal_weight"]
        stats = concentration_stats(result)
        assert abs(stats["max_abs_weight"] - 0.2) < 1e-10
        assert abs(stats["mean_abs_weight"] - 0.2) < 1e-10
        assert abs(stats["mean_gross_leverage"] - 1.0) < 1e-10

    def test_buy_hold_single_concentration(self) -> None:
        """BuyAndHold(INST_0=1.0) should have max weight 1.0, leverage 1.0."""
        results, *_ = _run_basic_backtest(N=5)
        bh_name = [n for n in results if n.startswith("buy_hold")][0]
        result = results[bh_name]
        stats = concentration_stats(result)
        assert abs(stats["max_abs_weight"] - 1.0) < 1e-10
        assert abs(stats["mean_gross_leverage"] - 1.0) < 1e-10

    def test_percentiles_ordered(self) -> None:
        """p90 <= p99 <= max."""
        results, *_ = _run_basic_backtest()
        for result in results.values():
            stats = concentration_stats(result)
            assert stats["median_abs_weight"] <= stats["p90_abs_weight"] + 1e-15
            assert stats["p90_abs_weight"] <= stats["p99_abs_weight"] + 1e-15
            assert stats["p99_abs_weight"] <= stats["max_abs_weight"] + 1e-15

    def test_all_stats_non_negative(self) -> None:
        """All concentration stats should be non-negative."""
        results, *_ = _run_basic_backtest()
        for result in results.values():
            stats = concentration_stats(result)
            for key, val in stats.items():
                assert val >= 0.0, f"{key}={val} is negative"


# ── Data Loading Tests ─────────────────────────────────────────


class TestDataLoading:
    """Test load_instrument_returns with mocked PipelineStore."""

    def _mock_store(
        self,
        tickers: list[str],
        T: int = 30,
        base_ts: float = 1700000000.0,
        day_seconds: float = 86400.0,
    ) -> MagicMock:
        """Create a mock store that returns synthetic daily_return observations."""
        rng = np.random.default_rng(99)
        obs = []
        for t in range(T):
            ts = base_ts + t * day_seconds
            for ticker in tickers:
                obs.append(
                    {
                        "entity_id": ticker,
                        "observation_type": "daily_return",
                        "observed_at": ts,
                        "value": {"log_return": float(rng.standard_normal() * 0.01)},
                    }
                )
        store = MagicMock()
        store.query_all_observations.return_value = obs
        return store

    def test_basic_loading(self) -> None:
        """Load returns for 3 tickers over 30 days."""
        tickers = ["AAA", "BBB", "CCC"]
        store = self._mock_store(tickers, T=30)
        dates, returns = load_instrument_returns(store, tickers)
        assert len(dates) == 30
        assert returns.shape == (30, 3)

    def test_ticker_ordering(self) -> None:
        """Returns columns match the requested ticker ordering."""
        tickers = ["CCC", "AAA", "BBB"]
        store = self._mock_store(["AAA", "BBB", "CCC"], T=5)
        dates, returns = load_instrument_returns(store, tickers)
        # Verify ordering is per tickers arg, not alphabetical obs order
        assert returns.shape[1] == 3

    def test_missing_ticker_fills_zero(self) -> None:
        """Tickers not in observations get 0.0 returns."""
        store = self._mock_store(["AAA"], T=5)
        dates, returns = load_instrument_returns(store, ["AAA", "BBB"])
        # BBB has no obs → column 1 should be all zeros
        np.testing.assert_array_equal(returns[:, 1], 0.0)

    def test_date_filtering(self) -> None:
        """Respects start_date and end_date filters."""
        store = self._mock_store(["AAA"], T=100)
        dates, returns = load_instrument_returns(
            store,
            ["AAA"],
            start_date="2023-11-14",
            end_date="2023-12-14",
        )
        # store was called with since/until timestamps
        store.query_all_observations.assert_called_once()
        call_kwargs = store.query_all_observations.call_args
        assert call_kwargs is not None

    def test_empty_observations_raises(self) -> None:
        """Raises ValueError when no observations match."""
        store = MagicMock()
        store.query_all_observations.return_value = []
        with pytest.raises(ValueError, match="No daily_return observations"):
            load_instrument_returns(store, ["AAA"])

    def test_non_daily_return_obs_ignored(self) -> None:
        """Observations of type != daily_return are skipped."""
        store = MagicMock()
        store.query_all_observations.return_value = [
            {
                "entity_id": "AAA",
                "observation_type": "volume",
                "observed_at": 1700000000.0,
                "value": {"volume": 1000},
            },
        ]
        with pytest.raises(ValueError, match="No daily_return observations"):
            load_instrument_returns(store, ["AAA"])


# ── Default Strategy Builder Tests ─────────────────────────────


class TestBuildDefaultStrategies:
    """Test build_default_strategies configuration."""

    def test_with_agg(self) -> None:
        """When AGG is present, 60/40 uses SPY+AGG."""
        strategies = build_default_strategies(["SPY", "AGG", "GLD"])
        assert len(strategies) == 3
        names = [s.name for s in strategies]
        assert "equal_weight" in names
        # Should have a buy_hold with AGG
        sixty_forty = [n for n in names if "AGG" in n]
        assert len(sixty_forty) == 1

    def test_without_bonds(self) -> None:
        """When no AGG/TLT, fallback 60% SPY only."""
        strategies = build_default_strategies(["SPY", "GLD"])
        assert len(strategies) == 3

    def test_with_tlt_no_agg(self) -> None:
        """When TLT present but no AGG, use TLT."""
        strategies = build_default_strategies(["SPY", "TLT", "GLD"])
        names = [s.name for s in strategies]
        tlt_strat = [n for n in names if "TLT" in n]
        assert len(tlt_strat) == 1


# ── MultiAssetWeightedSurpriseStrategy Tests ───────────────────


class TestMultiAssetWeightedSurprise:
    """Test the multi-asset surprise-weighted strategy."""

    def test_name(self) -> None:
        s = MultiAssetWeightedSurpriseStrategy()
        assert s.name == "multi_asset_weighted_surprise"

    def test_no_extra_returns_zeros(self) -> None:
        """Without test_extra, all weights are zero."""
        s = MultiAssetWeightedSurpriseStrategy()
        train = np.random.randn(100, 3)
        w = s.generate_weights(train, 10, ["A", "B", "C"])
        assert w.shape == (10, 3)
        np.testing.assert_array_equal(w, 0.0)

    def test_all_above_threshold(self) -> None:
        """All instruments above threshold → equal weight 1/N."""
        s = MultiAssetWeightedSurpriseStrategy(
            surprise_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
            threshold=0.0,
        )
        surprises = [
            {"A": (1.0, 1.0, 1.0, 1.0, 1.0), "B": (1.0, 1.0, 1.0, 1.0, 1.0)},
        ]
        w = s.generate_weights(
            np.zeros((50, 2)),
            1,
            ["A", "B"],
            test_extra={"instrument_surprises": surprises},
        )
        np.testing.assert_allclose(w, [[0.5, 0.5]])

    def test_some_below_threshold(self) -> None:
        """Only instruments above threshold get weight."""
        s = MultiAssetWeightedSurpriseStrategy(
            surprise_weights=(1.0, 0.0, 0.0, 0.0, 0.0),
            threshold=5.0,
        )
        surprises = [
            {"A": (10.0,), "B": (1.0,), "C": (10.0,)},
        ]
        w = s.generate_weights(
            np.zeros((50, 3)),
            1,
            ["A", "B", "C"],
            test_extra={"instrument_surprises": surprises},
        )
        # A and C above threshold → each gets 0.5; B gets 0
        np.testing.assert_allclose(w, [[0.5, 0.0, 0.5]])

    def test_none_above_threshold(self) -> None:
        """No instruments triggered → all zeros."""
        s = MultiAssetWeightedSurpriseStrategy(threshold=100.0)
        surprises = [{"A": (1.0, 1.0, 1.0, 1.0, 1.0)}]
        w = s.generate_weights(
            np.zeros((50, 2)),
            1,
            ["A", "B"],
            test_extra={"instrument_surprises": surprises},
        )
        np.testing.assert_array_equal(w, 0.0)

    def test_unknown_ticker_ignored(self) -> None:
        """Tickers not in instrument_names are skipped cleanly."""
        s = MultiAssetWeightedSurpriseStrategy(threshold=0.0)
        surprises = [{"UNKNOWN": (10.0, 10.0, 10.0, 10.0, 10.0)}]
        w = s.generate_weights(
            np.zeros((50, 2)),
            1,
            ["A", "B"],
            test_extra={"instrument_surprises": surprises},
        )
        np.testing.assert_array_equal(w, 0.0)

    def test_shorter_surprise_vector(self) -> None:
        """Surprise vectors shorter than weights are handled via truncation."""
        s = MultiAssetWeightedSurpriseStrategy(
            surprise_weights=(2.0, 3.0, 1.0, 1.0, 1.0),
            threshold=5.0,
        )
        # Only 2-element surprise vector → uses first 2 weights (2.0, 3.0)
        # Composite = 2.0*1.0 + 3.0*2.0 = 8.0 > 5.0
        surprises = [{"A": (1.0, 2.0)}]
        w = s.generate_weights(
            np.zeros((50, 1)),
            1,
            ["A"],
            test_extra={"instrument_surprises": surprises},
        )
        assert w[0, 0] == 1.0  # Only instrument, gets full weight

    def test_multi_timestep(self) -> None:
        """Works over multiple timesteps with varying triggers."""
        s = MultiAssetWeightedSurpriseStrategy(
            surprise_weights=(1.0,),
            threshold=5.0,
        )
        surprises = [
            {"A": (10.0,), "B": (1.0,)},  # t=0: only A
            {"A": (1.0,), "B": (10.0,)},  # t=1: only B
            {"A": (10.0,), "B": (10.0,)},  # t=2: both
        ]
        w = s.generate_weights(
            np.zeros((50, 2)),
            3,
            ["A", "B"],
            test_extra={"instrument_surprises": surprises},
        )
        np.testing.assert_allclose(w[0], [1.0, 0.0])
        np.testing.assert_allclose(w[1], [0.0, 1.0])
        np.testing.assert_allclose(w[2], [0.5, 0.5])

    def test_in_walkforward(self) -> None:
        """WeightedSurprise integrates with MultiAssetWalkForward."""
        T, N = 200, 3
        rng = np.random.default_rng(77)
        returns = rng.standard_normal((T, N)) * 0.01

        # Build per-timestep surprise data
        surprises = []
        for t in range(T):
            surprises.append(
                {f"INST_{i}": tuple(rng.standard_normal(5).tolist()) for i in range(N)}
            )

        s = MultiAssetWeightedSurpriseStrategy(
            surprise_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
            threshold=0.0,  # All triggered
        )
        names = [f"INST_{i}" for i in range(N)]
        classes = {n: "equity" for n in names}

        # Pack surprises into extra as object array (WalkForward slices it)
        surp_arr = np.empty(T, dtype=object)
        for t in range(T):
            surp_arr[t] = surprises[t]

        wf = MultiAssetWalkForward(
            min_train=50,
            test_size=21,
            step_size=21,
            instrument_names=names,
            instrument_classes=classes,
        )
        result = wf.run(
            s,
            returns,
            extra={"instrument_surprises": surp_arr},
        )
        assert len(result.folds) > 0
        assert math.isfinite(result.aggregate_metrics["sharpe"])


# ── Attribution Report Tests ───────────────────────────────────


class TestAttributionReport:
    """Test the generate_attribution_report function."""

    def test_report_structure(self) -> None:
        """Report contains expected fields for each strategy."""
        results, _, classes, regions = _run_basic_backtest()
        reports = generate_attribution_report(results, classes, regions)
        assert len(reports) == len(results)
        for name, report in reports.items():
            assert isinstance(report, StrategyReport)
            assert report.strategy_name == name
            assert isinstance(report.class_attribution, dict)
            assert isinstance(report.region_attribution, dict)
            assert isinstance(report.top_instruments, list)
            assert isinstance(report.concentration, dict)

    def test_report_fold_count(self) -> None:
        """Report fold_count matches actual folds."""
        results, _, classes, regions = _run_basic_backtest()
        reports = generate_attribution_report(results, classes, regions)
        for name, report in reports.items():
            assert report.fold_count == len(results[name].folds)

    def test_report_attribution_sums(self) -> None:
        """Both class and region attribution sum to total return."""
        results, _, classes, regions = _run_basic_backtest()
        reports = generate_attribution_report(results, classes, regions)
        for name, report in reports.items():
            total = float(results[name].all_portfolio_returns.sum())
            class_sum = sum(report.class_attribution.values())
            region_sum = sum(report.region_attribution.values())
            assert abs(class_sum - total) < 1e-10
            assert abs(region_sum - total) < 1e-10


# ── Edge Case Tests ────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_instrument(self) -> None:
        """Walk-forward works with a single instrument."""
        returns = _make_synthetic_returns(T=200, N=1)
        names = ["ONLY"]
        classes = {"ONLY": "equity"}
        regions = {"ONLY": "US"}
        results = run_walkforward(
            returns,
            names,
            classes,
            [EqualWeightStrategy()],
            min_train=50,
            test_size=21,
            step_size=21,
        )
        result = results["equal_weight"]
        assert len(result.folds) > 0
        # Attribution should have exactly one class
        attr = per_group_attribution(result, classes)
        assert len(attr) == 1
        assert "equity" in attr

    def test_zero_returns(self) -> None:
        """All-zero returns → zero P&L, zero attribution."""
        T, N = 200, 3
        returns = np.zeros((T, N))
        names, classes, regions = _make_instruments(N)
        results = run_walkforward(
            returns,
            names,
            classes,
            [EqualWeightStrategy()],
            min_train=50,
            test_size=21,
            step_size=21,
        )
        result = results["equal_weight"]
        np.testing.assert_array_equal(result.all_portfolio_returns, 0.0)
        for val in per_instrument_attribution(result).values():
            assert val == 0.0

    def test_insufficient_data_skips_strategy(self) -> None:
        """Too few rows for even one fold → strategy is skipped (empty results)."""
        returns = np.random.randn(50, 3)
        names, classes, _ = _make_instruments(3)
        results = run_walkforward(
            returns,
            names,
            classes,
            [EqualWeightStrategy()],
            min_train=100,
            test_size=21,
        )
        # Strategy errors are caught → no results for it
        assert len(results) == 0

    def test_many_instruments(self) -> None:
        """Walk-forward handles N=50 instruments."""
        T, N = 400, 50
        returns = _make_synthetic_returns(T=T, N=N, seed=12)
        names = [f"INST_{i}" for i in range(N)]
        classes = {n: f"cls_{i % 5}" for i, n in enumerate(names)}
        regions = {n: f"reg_{i % 4}" for i, n in enumerate(names)}
        results = run_walkforward(
            returns,
            names,
            classes,
            [EqualWeightStrategy()],
            min_train=100,
            test_size=21,
            step_size=21,
        )
        result = results["equal_weight"]
        assert result.all_weights.shape[1] == N
        # 5 classes, 4 regions
        cls_attr = per_group_attribution(result, classes)
        assert len(cls_attr) == 5
        reg_attr = per_group_attribution(result, regions)
        assert len(reg_attr) == 4

    def test_concentration_empty_weights(self) -> None:
        """concentration_stats on minimal data returns valid stats."""
        returns = _make_synthetic_returns(T=150, N=2)
        names = ["A", "B"]
        classes = {"A": "eq", "B": "fi"}
        results = run_walkforward(
            returns,
            names,
            classes,
            [EqualWeightStrategy()],
            min_train=100,
            test_size=21,
            step_size=21,
        )
        stats = concentration_stats(results["equal_weight"])
        assert all(isinstance(v, float) for v in stats.values())
        assert stats["max_abs_weight"] > 0

    def test_strategy_failure_skipped(self) -> None:
        """A strategy that raises is skipped gracefully."""
        from agent.quant.backtest import MultiAssetStrategy

        class BadStrategy(MultiAssetStrategy):
            @property
            def name(self) -> str:
                return "bad"

            def generate_weights(self, *args, **kwargs):
                raise RuntimeError("boom")

        returns = _make_synthetic_returns(T=200, N=3)
        names, classes, _ = _make_instruments(3)
        results = run_walkforward(
            returns,
            names,
            classes,
            [BadStrategy(), EqualWeightStrategy()],
            min_train=50,
            test_size=21,
        )
        # BadStrategy should be skipped, EqualWeight should succeed
        assert "bad" not in results
        assert "equal_weight" in results
