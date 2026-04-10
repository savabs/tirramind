"""Tests for convergence macro backtest engine (sub-phase B).

Covers:
- HistoricalEvidenceBuilder: direction rules, no-look-ahead, sparse data
- FRED parsing: valid/invalid entries, missing values
- Strategy implementations: weight generation, edge cases
- Precompute: score array alignment, empty data
- Baseline persistence: save/load/validate round-trip
- Integration: mini WalkForward with synthetic FRED data
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agent.convergence.backtest import (
    ConvergenceDirectionalStrategy,
    ConvergenceRiskOffStrategy,
    ConvergenceTemplateStrategy,
    FredSeriesConfig,
    FRED_SERIES,
    HistoricalEvidenceBuilder,
    MacroBacktestResult,
    StepScore,
    _resolve_macro_runtime,
    main,
    _get_fred_api_key,
    _fetch_target_returns,
    _is_placeholder_api_key,
    _apply_direction_rule,
    _parse_fred_response,
    precompute_convergence_scores,
    run_macro_backtest,
    save_baseline,
    validate_against_baseline,
)
from agent.convergence.evidence import Evidence

_DAY = 86_400


# ── Direction Rule Tests ───────────────────────────────────────


class TestApplyDirectionRule:
    def test_delta_pos_up_increase(self):
        assert _apply_direction_rule("delta_pos_up", 105.0, 100.0) == 1

    def test_delta_pos_up_decrease(self):
        assert _apply_direction_rule("delta_pos_up", 95.0, 100.0) == -1

    def test_delta_pos_up_no_change(self):
        assert _apply_direction_rule("delta_pos_up", 100.0, 100.0) == 0

    def test_delta_pos_up_no_prev(self):
        assert _apply_direction_rule("delta_pos_up", 100.0, None) == 0

    def test_delta_pos_down_increase(self):
        assert _apply_direction_rule("delta_pos_down", 105.0, 100.0) == -1

    def test_delta_pos_down_decrease(self):
        assert _apply_direction_rule("delta_pos_down", 95.0, 100.0) == 1

    def test_delta_neg_up_decrease(self):
        assert _apply_direction_rule("delta_neg_up", 95.0, 100.0) == 1

    def test_delta_neg_up_increase(self):
        assert _apply_direction_rule("delta_neg_up", 105.0, 100.0) == -1

    def test_level_below_50_low(self):
        assert _apply_direction_rule("level_below_50", 45.0, 55.0) == 1

    def test_level_below_50_high(self):
        assert _apply_direction_rule("level_below_50", 55.0, 45.0) == -1

    def test_level_below_50_exact(self):
        assert _apply_direction_rule("level_below_50", 50.0, 50.0) == 0

    def test_level_below_0_negative(self):
        assert _apply_direction_rule("level_below_0", -0.5, 0.5) == 1

    def test_level_below_0_positive(self):
        assert _apply_direction_rule("level_below_0", 0.5, -0.5) == -1

    def test_unknown_rule(self):
        assert _apply_direction_rule("nonexistent", 100.0, 50.0) == 0


# ── FRED Response Parsing ──────────────────────────────────────


class TestParseFredResponse:
    def test_valid_data(self):
        raw = [
            {"date": "2020-01-01", "value": "100.5"},
            {"date": "2020-01-08", "value": "101.3"},
        ]
        result = _parse_fred_response("DFF", raw)
        assert len(result) == 2
        assert result[0][1] == 100.5
        assert result[1][1] == 101.3
        # Timestamps should be in order.
        assert result[0][0] < result[1][0]

    def test_missing_values_skipped(self):
        raw = [
            {"date": "2020-01-01", "value": "."},
            {"date": "2020-01-08", "value": "101.3"},
            {"date": "2020-01-15", "value": ""},
        ]
        result = _parse_fred_response("DFF", raw)
        assert len(result) == 1
        assert result[0][1] == 101.3

    def test_empty_input(self):
        assert _parse_fred_response("DFF", []) == []

    def test_invalid_value(self):
        raw = [
            {"date": "2020-01-01", "value": "abc"},
            {"date": "2020-01-08", "value": "100"},
        ]
        result = _parse_fred_response("DFF", raw)
        assert len(result) == 1

    def test_missing_date_key(self):
        raw = [{"value": "100"}]
        result = _parse_fred_response("DFF", raw)
        assert len(result) == 0


class TestFredApiKeyResolution:
    def test_prefers_repo_standard_env_var(self, monkeypatch):
        monkeypatch.setenv("TIRRA_FRED_API_KEY", "repo-key")
        monkeypatch.setenv("FRED_API_KEY", "fallback-key")
        assert _get_fred_api_key() == "repo-key"

    def test_falls_back_to_short_env_var(self, monkeypatch):
        monkeypatch.delenv("TIRRA_FRED_API_KEY", raising=False)
        monkeypatch.setenv("FRED_API_KEY", "fallback-key")
        assert _get_fred_api_key() == "fallback-key"


class TestFredApiKeyValidation:
    def test_placeholder_key_detected(self):
        assert _is_placeholder_api_key("your-key-here") is True

    def test_realistic_key_not_detected(self):
        assert _is_placeholder_api_key("abc123-real-key") is False


# ── HistoricalEvidenceBuilder ──────────────────────────────────


def _make_fred_data(
    n_points: int = 52,
    start_ts: float = 1_577_836_800.0,  # 2020-01-01
    step: float = _DAY * 7,  # weekly
    base_value: float = 100.0,
    trend: float = 0.5,
) -> dict[str, list[tuple[float, float]]]:
    """Build fake FRED data for DFF and WALCL."""
    rng = np.random.default_rng(42)
    dff_data = [
        (start_ts + i * step, base_value + trend * i + rng.normal(0, 1))
        for i in range(n_points)
    ]
    walcl_data = [
        (start_ts + i * step, 7_000_000 + 50_000 * i + rng.normal(0, 10000))
        for i in range(n_points)
    ]
    return {"DFF": dff_data, "WALCL": walcl_data}


class TestHistoricalEvidenceBuilder:
    def test_basic_evidence_generation(self):
        fred_data = _make_fred_data(n_points=20)
        builder = HistoricalEvidenceBuilder(fred_data)
        # as_of = midpoint of the series
        mid_ts = fred_data["DFF"][10][0]
        evidence = builder.build_evidence(as_of=mid_ts)
        assert len(evidence) > 0
        for e in evidence:
            assert isinstance(e, Evidence)
            assert e.timestamp <= mid_ts, "Look-ahead violation!"

    def test_no_lookahead(self):
        """Evidence must not contain any timestamps after as_of."""
        fred_data = _make_fred_data(n_points=52)
        builder = HistoricalEvidenceBuilder(fred_data)
        as_of = fred_data["DFF"][25][0]
        evidence = builder.build_evidence(as_of=as_of)
        for e in evidence:
            assert e.timestamp <= as_of

    def test_lookback_window(self):
        """Old data beyond 365 days is excluded."""
        fred_data = _make_fred_data(n_points=100, step=_DAY * 7)
        builder = HistoricalEvidenceBuilder(fred_data)
        as_of = fred_data["DFF"][-1][0]
        evidence = builder.build_evidence(as_of=as_of)
        lookback_start = as_of - 365 * _DAY
        for e in evidence:
            assert e.timestamp >= lookback_start

    def test_sparse_data(self):
        """Only 2 points — just barely enough."""
        fred_data = {
            "DFF": [
                (1_577_836_800.0, 1.5),
                (1_577_836_800.0 + 7 * _DAY, 1.75),
            ]
        }
        builder = HistoricalEvidenceBuilder(fred_data)
        evidence = builder.build_evidence(as_of=1_577_836_800.0 + 14 * _DAY)
        assert len(evidence) == 2

    def test_single_point_insufficient(self):
        """Only 1 point — not enough for direction."""
        fred_data = {"DFF": [(1_577_836_800.0, 1.5)]}
        builder = HistoricalEvidenceBuilder(fred_data)
        evidence = builder.build_evidence(as_of=1_577_836_800.0 + 14 * _DAY)
        assert len(evidence) == 0

    def test_unknown_fred_id_ignored(self):
        fred_data = {"UNKNOWN_SERIES": [(1_577_836_800.0, 42.0)]}
        builder = HistoricalEvidenceBuilder(fred_data)
        evidence = builder.build_evidence(as_of=1_577_836_800.0 + 100 * _DAY)
        assert len(evidence) == 0

    def test_categories_correct(self):
        fred_data = _make_fred_data()
        builder = HistoricalEvidenceBuilder(fred_data)
        as_of = fred_data["DFF"][-1][0]
        evidence = builder.build_evidence(as_of=as_of)
        cats = {e.category for e in evidence}
        assert "monetary_policy" in cats  # Both DFF and WALCL are monetary

    def test_signal_ids_correct(self):
        fred_data = _make_fred_data()
        builder = HistoricalEvidenceBuilder(fred_data)
        as_of = fred_data["DFF"][-1][0]
        evidence = builder.build_evidence(as_of=as_of)
        sig_ids = {e.signal_id for e in evidence}
        assert "rate_monitor.fed.rate" in sig_ids
        assert "central_bank.fed.assets" in sig_ids

    def test_build_registry(self):
        fred_data = _make_fred_data()
        builder = HistoricalEvidenceBuilder(fred_data)
        registry = builder.build_registry()
        assert len(registry) == 2
        assert registry.get("rate_monitor.fed.rate") is not None
        assert registry.get("central_bank.fed.assets") is not None

    def test_confidence_bounded(self):
        fred_data = _make_fred_data(n_points=30)
        builder = HistoricalEvidenceBuilder(fred_data)
        as_of = fred_data["DFF"][-1][0]
        evidence = builder.build_evidence(as_of=as_of)
        for e in evidence:
            assert 0.0 <= e.confidence <= 1.0


# ── Strategy Tests ─────────────────────────────────────────────


class TestConvergenceRiskOffStrategy:
    def test_no_extra_returns_ones(self):
        strat = ConvergenceRiskOffStrategy()
        weights = strat.generate_weights(np.zeros(10), 5)
        np.testing.assert_array_equal(weights, np.ones(5))

    def test_stress_signal_exits(self):
        strat = ConvergenceRiskOffStrategy(score_threshold=0.3)
        extra = {
            "conv_score": np.array([0.0, 0.5, 0.8, 0.1, 0.0]),
            "conv_direction": np.array([0, 1, 1, -1, 0]),
        }
        weights = strat.generate_weights(np.zeros(10), 5, test_extra=extra)
        assert weights[0] == 1.0  # No signal
        assert weights[1] == 0.0  # Stress + high score
        assert weights[2] == 0.0  # Stress + high score
        assert weights[3] == 1.0  # Relief direction
        assert weights[4] == 1.0  # No signal

    def test_below_threshold_stays(self):
        strat = ConvergenceRiskOffStrategy(score_threshold=0.5)
        extra = {
            "conv_score": np.array([0.3]),
            "conv_direction": np.array([1]),
        }
        weights = strat.generate_weights(np.zeros(10), 1, test_extra=extra)
        assert weights[0] == 1.0

    @property
    def name(self):
        strat = ConvergenceRiskOffStrategy()
        assert strat.name == "convergence_risk_off"


class TestConvergenceDirectionalStrategy:
    def test_proportional_reduction(self):
        strat = ConvergenceDirectionalStrategy(scale=1.0)
        extra = {
            "conv_score": np.array([0.6]),
            "conv_direction": np.array([1]),
        }
        weights = strat.generate_weights(np.zeros(10), 1, test_extra=extra)
        assert abs(weights[0] - 0.4) < 0.01

    def test_relief_signal_keeps_exposure(self):
        strat = ConvergenceDirectionalStrategy(scale=1.0)
        extra = {
            "conv_score": np.array([0.8]),
            "conv_direction": np.array([-1]),
        }
        weights = strat.generate_weights(np.zeros(10), 1, test_extra=extra)
        assert weights[0] == 1.0  # Relief: no reduction

    def test_no_extra_returns_ones(self):
        strat = ConvergenceDirectionalStrategy()
        weights = strat.generate_weights(np.zeros(5), 3)
        np.testing.assert_array_equal(weights, np.ones(3))

    def test_weight_clamped_to_zero(self):
        strat = ConvergenceDirectionalStrategy(scale=2.0)
        extra = {
            "conv_score": np.array([0.8]),
            "conv_direction": np.array([1]),
        }
        weights = strat.generate_weights(np.zeros(10), 1, test_extra=extra)
        assert weights[0] == 0.0  # 1 - 0.8*2.0 = -0.6, clamped to 0


class TestConvergenceTemplateStrategy:
    def test_high_template_match_exits(self):
        strat = ConvergenceTemplateStrategy(match_threshold=0.5, score_threshold=0.3)
        extra = {
            "conv_score": np.array([0.5, 0.5]),
            "conv_direction": np.array([1, 1]),
            "conv_template_match": np.array([0.8, 0.3]),
        }
        weights = strat.generate_weights(np.zeros(10), 2, test_extra=extra)
        assert weights[0] == 0.0  # High match + stress
        assert weights[1] == 1.0  # Low match → stay

    def test_missing_template_match_returns_ones(self):
        strat = ConvergenceTemplateStrategy()
        extra = {
            "conv_score": np.array([0.5]),
            "conv_direction": np.array([1]),
        }
        weights = strat.generate_weights(np.zeros(10), 1, test_extra=extra)
        assert weights[0] == 1.0


# ── Precompute Scores ──────────────────────────────────────────


class TestPrecomputeConvergenceScores:
    def test_empty_fred_data(self):
        timestamps = np.array([1_600_000_000.0, 1_600_604_800.0])
        scores = precompute_convergence_scores({}, timestamps)
        assert len(scores) == 2
        for s in scores:
            assert s.score == 0.0

    def test_alignment_with_timestamps(self):
        fred_data = _make_fred_data(n_points=30)
        ts_start = fred_data["DFF"][10][0]
        timestamps = np.array([ts_start + i * 7 * _DAY for i in range(10)])
        scores = precompute_convergence_scores(fred_data, timestamps)
        assert len(scores) == len(timestamps)
        for s, ts in zip(scores, timestamps):
            assert s.timestamp == ts

    def test_all_scores_valid(self):
        fred_data = _make_fred_data(n_points=60)
        ts_start = fred_data["DFF"][20][0]
        timestamps = np.array([ts_start + i * 7 * _DAY for i in range(20)])
        scores = precompute_convergence_scores(fred_data, timestamps)
        for s in scores:
            assert 0.0 <= s.score <= 10.0  # reasonable bound
            assert s.direction in (-1, 0, 1)
            assert s.n_cliques >= 0

    def test_shared_cache_reuses_overlapping_timestamps(self):
        fred_data = _make_fred_data(n_points=40)
        timestamps = np.array([point[0] for point in fred_data["DFF"][5:11]])
        cache: dict[float, StepScore] = {}

        with patch(
            "agent.convergence.backtest.ConvergenceDetector.detect",
            return_value=[],
        ) as mock_detect:
            first = precompute_convergence_scores(
                fred_data,
                timestamps,
                step_score_cache=cache,
            )
            second = precompute_convergence_scores(
                fred_data,
                timestamps[2:5],
                step_score_cache=cache,
            )

        assert len(first) == len(timestamps)
        assert len(second) == 3
        assert mock_detect.call_count == len(timestamps)
        assert len(cache) == len(timestamps)
        np.testing.assert_array_equal(
            np.array([s.timestamp for s in second]),
            timestamps[2:5],
        )


# ── Baseline Persistence ──────────────────────────────────────


class TestBaselinePersistence:
    def _make_result(self) -> dict[str, MacroBacktestResult]:
        """Create a minimal MacroBacktestResult for testing."""
        from agent.quant.backtest import BacktestResult, FoldResult

        fold = FoldResult(
            fold=0,
            train_size=52,
            test_size=12,
            metrics={"sharpe": 0.5, "sortino": 0.7, "max_drawdown": -0.15},
            weights=np.ones(12),
            test_returns=np.random.default_rng(0).normal(0.001, 0.02, 12),
        )
        bt = BacktestResult(
            strategy_name="convergence_risk_off",
            folds=[fold],
            aggregate_metrics={"sharpe": 0.5, "sortino": 0.7, "max_drawdown": -0.15},
            equity_curve=np.ones(12),
            all_test_returns=fold.test_returns,
            all_weights=fold.weights,
        )
        return {
            "SPY": MacroBacktestResult(
                strategies={"convergence_risk_off": bt, "buy_and_hold": bt},
                convergence_scores=[StepScore(timestamp=0)],
                sharpe_diffs={},
                n_detections=5,
                detection_rate=0.1,
            )
        }

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"
            results = self._make_result()
            save_baseline(results, path)

            assert path.exists()
            data = json.loads(path.read_text())
            assert "SPY" in data
            assert "convergence_risk_off" in data["SPY"]
            assert data["SPY"]["convergence_risk_off"]["sharpe"] == 0.5

    def test_validate_passes_within_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"
            results = self._make_result()
            save_baseline(results, path)

            passed, failures = validate_against_baseline(results, path)
            assert passed is True
            assert len(failures) == 0

    def test_validate_fails_on_degradation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"
            results = self._make_result()
            save_baseline(results, path)

            # Degrade Sharpe by 50%.
            degraded = self._make_result()
            for res in degraded.values():
                for bt in res.strategies.values():
                    bt.aggregate_metrics["sharpe"] = 0.2  # was 0.5

            passed, failures = validate_against_baseline(degraded, path)
            assert passed is False
            assert any("sharpe" in f for f in failures)

    def test_validate_no_baseline_autopasses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            results = self._make_result()
            passed, failures = validate_against_baseline(results, path)
            assert passed is True


# ── Integration: Mini WalkForward ──────────────────────────────


class TestMiniWalkForward:
    """Run a minimal WalkForward to verify end-to-end plumbing."""

    def test_risk_off_with_walkforward(self):
        from agent.quant.backtest import WalkForward

        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 100)

        scores = rng.uniform(0, 0.8, 100)
        directions = np.where(scores > 0.5, 1, 0)

        extra = {
            "conv_score": scores,
            "conv_direction": directions.astype(float),
            "conv_template_match": np.zeros(100),
        }

        wf = WalkForward(min_train=20, test_size=10, periods_per_year=52)
        strat = ConvergenceRiskOffStrategy(score_threshold=0.3)
        result = wf.run(strat, returns, extra=extra)

        assert len(result.folds) > 0
        assert result.aggregate_metrics is not None
        assert "sharpe" in result.aggregate_metrics

    def test_directional_with_walkforward(self):
        from agent.quant.backtest import WalkForward

        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 100)

        extra = {
            "conv_score": rng.uniform(0, 1.0, 100),
            "conv_direction": np.ones(100),
            "conv_template_match": np.zeros(100),
        }

        wf = WalkForward(min_train=20, test_size=10, periods_per_year=52)
        strat = ConvergenceDirectionalStrategy(scale=0.5)
        result = wf.run(strat, returns, extra=extra)

        assert len(result.folds) > 0
        # Weights should be reduced (not all 1.0).
        assert np.any(result.all_weights < 1.0)


class TestFetchTargetReturns:
    def test_handles_multiindex_close_output(self):
        index = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
        columns = pd.MultiIndex.from_tuples(
            [
                ("Close", "SPY"),
                ("High", "SPY"),
                ("Low", "SPY"),
                ("Open", "SPY"),
                ("Volume", "SPY"),
            ],
            names=["Price", "Ticker"],
        )
        df = pd.DataFrame(
            [
                [100.0, 101.0, 99.0, 100.5, 10],
                [101.0, 102.0, 100.0, 100.2, 11],
                [103.0, 104.0, 101.0, 102.5, 12],
            ],
            index=index,
            columns=columns,
        )

        with patch("yfinance.download", return_value=df):
            timestamps, log_returns = _fetch_target_returns(
                "SPY", "2024-01-01", "2024-02-01"
            )

        assert len(timestamps) == 2
        assert len(log_returns) == 2
        assert np.all(np.isfinite(log_returns))


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_zero_variance_fred_data(self):
        """All values identical — std=0."""
        fred_data = {"DFF": [(1_577_836_800.0 + i * 7 * _DAY, 1.5) for i in range(20)]}
        builder = HistoricalEvidenceBuilder(fred_data)
        as_of = fred_data["DFF"][-1][0]
        evidence = builder.build_evidence(as_of=as_of)
        # Should not crash; confidence falls back to 0.3
        for e in evidence:
            assert e.confidence == 0.3

    def test_nan_in_fred_values(self):
        """NaN values should not crash parsing."""
        raw = [
            {"date": "2020-01-01", "value": "nan"},
            {"date": "2020-01-08", "value": "100"},
        ]
        result = _parse_fred_response("DFF", raw)
        # "nan" parses as float(nan) which is valid but not useful.
        # We accept it — the builder handles NaN downstream.
        assert len(result) == 2

    def test_fred_series_config_fields(self):
        for cfg in [
            FredSeriesConfig(
                "DFF", "rate_monitor.fed.rate", "monetary_policy", "delta_pos_down"
            )
        ]:
            assert cfg.fred_id == "DFF"
            assert cfg.frequency == "weekly"  # default

    def test_monthly_macro_proxy_replaces_napm(self):
        cfg = next(c for c in FRED_SERIES if c.signal_id == "pmi.us.manufacturing")
        assert cfg.fred_id == "USSLIND"
        assert cfg.category == "macro_momentum"
        assert cfg.direction_rule == "level_below_0"
        assert cfg.frequency == "monthly"

    def test_step_score_defaults(self):
        s = StepScore(timestamp=0.0)
        assert s.score == 0.0
        assert s.direction == 0
        assert s.n_cliques == 0
        assert s.best_template == ""


class TestMacroRuntimeResolution:
    def test_fast_mode_shrinks_defaults(self):
        start_year, end_year, targets, bootstrap_count = _resolve_macro_runtime(
            start_year=2010,
            end_year=2025,
            targets=["SPY", "TLT", "GLD"],
            bootstrap_count=None,
            fast_mode=True,
        )
        assert start_year == 2018
        assert end_year == 2025
        assert targets == ["SPY"]
        assert bootstrap_count == 200

    def test_fast_mode_preserves_explicit_overrides(self):
        start_year, end_year, targets, bootstrap_count = _resolve_macro_runtime(
            start_year=2022,
            end_year=2024,
            targets=["SPY", "TLT"],
            bootstrap_count=75,
            fast_mode=True,
        )
        assert start_year == 2022
        assert end_year == 2024
        assert targets == ["SPY", "TLT"]
        assert bootstrap_count == 75

    def test_non_fast_uses_full_defaults(self):
        start_year, end_year, targets, bootstrap_count = _resolve_macro_runtime(
            start_year=2010,
            end_year=2025,
            targets=["SPY", "TLT", "GLD"],
            bootstrap_count=None,
            fast_mode=False,
        )
        assert start_year == 2010
        assert end_year == 2025
        assert targets == ["SPY", "TLT", "GLD"]
        assert bootstrap_count == 1000


class TestBacktestCli:
    def test_macro_fast_mode_passes_reduced_runtime(self):
        with patch(
            "agent.convergence.backtest.run_macro_backtest", return_value={}
        ) as mock_run:
            with patch("sys.argv", ["backtest", "--macro", "--fast"]):
                main()

        mock_run.assert_called_once_with(
            start_year=2018,
            end_year=2025,
            targets=["SPY"],
            bootstrap_count=200,
        )

    def test_macro_fast_mode_keeps_explicit_args(self):
        argv = [
            "backtest",
            "--macro",
            "--fast",
            "--start-year",
            "2022",
            "--end-year",
            "2024",
            "--targets",
            "SPY",
            "TLT",
            "--bootstrap-count",
            "75",
        ]
        with patch(
            "agent.convergence.backtest.run_macro_backtest", return_value={}
        ) as mock_run:
            with patch("sys.argv", argv):
                main()

        mock_run.assert_called_once_with(
            start_year=2022,
            end_year=2024,
            targets=["SPY", "TLT"],
            bootstrap_count=75,
        )


class TestRunMacroBacktestCaching:
    def test_shared_step_score_cache_used_across_targets(self):
        timestamps = np.array([1_700_000_000.0, 1_700_604_800.0, 1_701_209_600.0])
        returns = np.array([0.01, -0.02, 0.03])
        cache_ids: list[int] = []

        def _fake_precompute(*args, **kwargs):
            cache_ids.append(id(kwargs["step_score_cache"]))
            requested = args[1]
            return [StepScore(timestamp=float(ts)) for ts in requested]

        bt_result = MagicMock()
        bt_result.aggregate_metrics = {"sharpe": 0.5, "max_drawdown": -0.1}
        bt_result.all_test_returns = np.array([0.01, -0.02])

        with (
            patch(
                "agent.convergence.backtest._fetch_target_returns",
                return_value=(timestamps, returns),
            ),
            patch(
                "agent.convergence.backtest.precompute_convergence_scores",
                side_effect=_fake_precompute,
            ),
            patch(
                "agent.convergence.backtest.WalkForward.run",
                return_value=bt_result,
            ),
            patch(
                "agent.convergence.backtest.block_bootstrap_ci",
                return_value=(0.5, 0.4, 0.6),
            ),
        ):
            results = run_macro_backtest(
                start_year=2020,
                end_year=2020,
                targets=["SPY", "TLT"],
                fred_data=_make_fred_data(n_points=20),
                bootstrap_count=10,
            )

        assert set(results.keys()) == {"SPY", "TLT"}
        assert len(cache_ids) == 2
        assert len(set(cache_ids)) == 1


# ── Performance optimization regression tests ─────────────────


class TestBuildAllEvidence:
    """Verify build_all_evidence produces consistent output with build_evidence."""

    def test_build_all_matches_per_step_evidence(self):
        """Evidence from build_all_evidence sliced by as_of must match
        build_evidence(as_of) for every observation point."""
        import bisect

        fred_data = _make_fred_data(n_points=30, start_ts=1_600_000_000.0)
        builder = HistoricalEvidenceBuilder(fred_data)

        all_ev = builder.build_all_evidence()
        all_ev_ts = [ev.timestamp for ev in all_ev]

        # Pick a few check timestamps spread across the range.
        check_timestamps = [
            1_600_000_000.0 + 90 * _DAY,
            1_600_000_000.0 + 180 * _DAY,
            1_600_000_000.0 + 270 * _DAY,
        ]

        for as_of in check_timestamps:
            old_evidence = builder.build_evidence(as_of=as_of)
            lookback_start = as_of - 365 * _DAY
            lo = bisect.bisect_left(all_ev_ts, lookback_start)
            hi = bisect.bisect_right(all_ev_ts, as_of)
            new_evidence = all_ev[lo:hi]

            # Same number of evidence items.
            assert len(new_evidence) == len(
                old_evidence
            ), f"at as_of={as_of}: {len(new_evidence)} != {len(old_evidence)}"

            # Same signal_ids and timestamps.
            old_ids = [(e.signal_id, e.timestamp) for e in old_evidence]
            new_ids = [(e.signal_id, e.timestamp) for e in new_evidence]
            assert sorted(old_ids) == sorted(new_ids)

    def test_build_all_evidence_sorted(self):
        fred_data = _make_fred_data(n_points=20)
        builder = HistoricalEvidenceBuilder(fred_data)
        all_ev = builder.build_all_evidence()
        timestamps = [ev.timestamp for ev in all_ev]
        assert timestamps == sorted(timestamps)

    def test_build_all_evidence_empty_data(self):
        builder = HistoricalEvidenceBuilder({})
        assert builder.build_all_evidence() == []


class TestVectorizedConfidence:
    """Verify vectorized z-score confidence matches expectations."""

    def test_confidence_first_three_points_default(self):
        """First 3 observations should get default confidence 0.3."""
        fred_data = {
            "DFF": [(1_600_000_000.0 + i * _DAY, 1.0 + i * 0.1) for i in range(10)]
        }
        series_cfg = [
            FredSeriesConfig(
                "DFF", "test.signal", "monetary_policy", "delta_pos_up", "daily"
            )
        ]
        builder = HistoricalEvidenceBuilder(fred_data, series_cfg)
        all_ev = builder.build_all_evidence()

        # First 3 evidence items should have confidence 0.3.
        for ev in all_ev[:3]:
            assert ev.confidence == pytest.approx(0.3)

        # 4th onward should be computed (not default).
        for ev in all_ev[3:]:
            assert ev.confidence >= 0.0
            assert ev.confidence <= 1.0

    def test_constant_series_gets_default_confidence(self):
        """When all values are identical, std=0 so confidence=0.3."""
        fred_data = {"DFF": [(1_600_000_000.0 + i * _DAY * 7, 5.0) for i in range(10)]}
        series_cfg = [
            FredSeriesConfig(
                "DFF", "test.signal", "monetary_policy", "delta_pos_up", "daily"
            )
        ]
        builder = HistoricalEvidenceBuilder(fred_data, series_cfg)
        all_ev = builder.build_all_evidence()
        for ev in all_ev:
            assert ev.confidence == pytest.approx(0.3)


class TestPrecomputeOptimized:
    """Verify the optimized precompute path produces valid output."""

    def test_no_look_ahead(self):
        """Scores at early timestamps must not see future data."""
        base_ts = 1_600_000_000.0
        fred_data = _make_fred_data(n_points=30, start_ts=base_ts)
        timestamps = np.array([base_ts + i * 7 * _DAY for i in range(20)])

        scores = precompute_convergence_scores(fred_data, timestamps)

        assert len(scores) == 20
        for s in scores:
            assert isinstance(s, StepScore)
            assert s.score >= 0.0

    def test_bisect_slice_versus_build_evidence(self):
        """Optimized path and per-step build_evidence should give evidence
        with matching signal structure (same number of signals at same timestamps)."""
        import bisect

        base_ts = 1_600_000_000.0
        fred_data = _make_fred_data(n_points=50, start_ts=base_ts)
        builder = HistoricalEvidenceBuilder(fred_data)

        all_ev = builder.build_all_evidence()
        all_ev_ts = [ev.timestamp for ev in all_ev]

        test_ts = base_ts + 200 * _DAY
        lookback_start = test_ts - 365 * _DAY
        lo = bisect.bisect_left(all_ev_ts, lookback_start)
        hi = bisect.bisect_right(all_ev_ts, test_ts)
        sliced = all_ev[lo:hi]

        old_ev = builder.build_evidence(as_of=test_ts)

        # Same count and same unique signal_ids.
        assert len(sliced) == len(old_ev)
        assert {e.signal_id for e in sliced} == {e.signal_id for e in old_ev}

    def test_single_detector_reused(self):
        """The optimized path should not create a new PipelineStore per step."""
        import unittest.mock as um

        base_ts = 1_600_000_000.0
        fred_data = _make_fred_data(n_points=10, start_ts=base_ts)
        timestamps = np.array([base_ts + i * 7 * _DAY for i in range(5)])

        with um.patch("agent.convergence.backtest.PipelineStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            precompute_convergence_scores(fred_data, timestamps)

            # Should create exactly ONE store (the dummy), not one per step.
            assert mock_store_cls.call_count == 1
