"""Walk-forward backtesting engine with strategy protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.quant.scoring import score_returns

# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Abstract base for backtestable strategies.

    Subclasses implement ``generate_weights`` which receives training data
    and must return position weights for the test period.  Weights are
    multiplied element-wise with test-period returns.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return a 1-D array of position weights for the test window.

        Parameters
        ----------
        train_returns : log returns of the asset during the training period.
        test_length : number of periods in the upcoming test window.
        train_extra : optional dict of auxiliary training data (e.g. regime labels).
        test_extra : optional dict of auxiliary test data (e.g. regime labels).

        Returns
        -------
        1-D array of length ``test_length`` with values in [0, 1].
        """
        ...


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""

    fold: int
    train_size: int
    test_size: int
    metrics: dict[str, Any]
    weights: np.ndarray
    test_returns: np.ndarray


@dataclass
class BacktestResult:
    """Aggregate output of a walk-forward backtest."""

    strategy_name: str
    folds: list[FoldResult]
    aggregate_metrics: dict[str, Any]
    equity_curve: np.ndarray  # cumulative wealth from chained test returns
    all_test_returns: np.ndarray  # concatenated test returns across folds
    all_weights: np.ndarray  # concatenated weights across folds


# ---------------------------------------------------------------------------
# Walk-Forward engine
# ---------------------------------------------------------------------------


class WalkForward:
    """Expanding-window walk-forward backtester.

    Parameters
    ----------
    min_train : minimum training periods before first test fold.
    test_size : periods per test fold.
    step_size : how far the split advances each fold (defaults to test_size).
    periods_per_year : for annualization of metrics.
    """

    def __init__(
        self,
        min_train: int = 104,
        test_size: int = 52,
        step_size: int | None = None,
        periods_per_year: int = 52,
    ) -> None:
        self.min_train = min_train
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.periods_per_year = periods_per_year

    def run(
        self,
        strategy: Strategy,
        returns: np.ndarray,
        *,
        extra: dict[str, np.ndarray] | None = None,
    ) -> BacktestResult:
        """Execute walk-forward backtest.

        Parameters
        ----------
        strategy : a Strategy instance.
        returns : 1-D array of log returns for the full period.
        extra : optional dict mapping names to arrays aligned with ``returns``
                (e.g. ``{"regimes": regime_labels}``).  Sliced per fold and
                passed as train_extra / test_extra to the strategy.

        Returns
        -------
        BacktestResult with per-fold and aggregate metrics.
        """
        returns = np.asarray(returns, dtype=np.float64)
        T = len(returns)
        extra = extra or {}

        folds: list[FoldResult] = []
        fold_idx = 0
        split = self.min_train

        while split + self.test_size <= T:
            train_ret = returns[:split]
            test_ret = returns[split : split + self.test_size]

            train_extra = {k: v[:split] for k, v in extra.items()}
            test_extra = {k: v[split : split + self.test_size] for k, v in extra.items()}

            weights = strategy.generate_weights(
                train_ret,
                len(test_ret),
                train_extra=train_extra if train_extra else None,
                test_extra=test_extra if test_extra else None,
            )

            # Weighted returns (weight=0 → cash, weight=1 → full exposure)
            weighted_ret = test_ret * weights

            fold_metrics = score_returns(
                weighted_ret,
                weights=weights,
                periods_per_year=self.periods_per_year,
            )

            folds.append(
                FoldResult(
                    fold=fold_idx,
                    train_size=len(train_ret),
                    test_size=len(test_ret),
                    metrics=fold_metrics,
                    weights=weights,
                    test_returns=weighted_ret,
                )
            )

            fold_idx += 1
            split += self.step_size

        if not folds:
            raise ValueError(
                f"Not enough data for even one fold. T={T}, min_train={self.min_train}, test_size={self.test_size}"
            )

        # Aggregate: chain all test returns
        all_test_returns = np.concatenate([f.test_returns for f in folds])
        all_weights = np.concatenate([f.weights for f in folds])
        equity = np.exp(np.cumsum(all_test_returns))

        agg_metrics = score_returns(
            all_test_returns,
            weights=all_weights,
            periods_per_year=self.periods_per_year,
        )

        return BacktestResult(
            strategy_name=strategy.name,
            folds=folds,
            aggregate_metrics=agg_metrics,
            equity_curve=equity,
            all_test_returns=all_test_returns,
            all_weights=all_weights,
        )


# ---------------------------------------------------------------------------
# Regime-conditional analysis
# ---------------------------------------------------------------------------


def regime_conditional_analysis(
    returns: np.ndarray,
    regimes: np.ndarray,
    periods_per_year: int = 52,
) -> dict[int, dict[str, Any]]:
    """Score returns separately for each regime.

    Parameters
    ----------
    returns : 1-D log return series.
    regimes : 1-D integer regime labels (same length as returns).

    Returns
    -------
    Dict mapping regime label → score_returns() dict.
    """
    returns = np.asarray(returns)
    regimes = np.asarray(regimes)
    unique = np.unique(regimes)
    result: dict[int, dict[str, Any]] = {}
    for reg in unique:
        mask = regimes == reg
        reg_ret = returns[mask]
        if len(reg_ret) < 2:
            result[int(reg)] = {"n_periods": int(mask.sum()), "insufficient_data": True}
        else:
            metrics = score_returns(reg_ret, periods_per_year=periods_per_year)
            metrics["n_periods"] = int(mask.sum())
            metrics["pct_time"] = float(mask.sum() / len(returns))
            result[int(reg)] = metrics
    return result


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class BuyAndHoldStrategy(Strategy):
    """Always fully invested. Baseline reference."""

    @property
    def name(self) -> str:
        return "buy_and_hold"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.ones(test_length)


class RegimeAvoidStrategy(Strategy):
    """Go to cash during specified regime states.

    Requires ``test_extra["regimes"]`` to be provided.
    """

    def __init__(self, avoid_states: set[int]) -> None:
        self._avoid = avoid_states

    @property
    def name(self) -> str:
        states = ",".join(str(s) for s in sorted(self._avoid))
        return f"regime_avoid_{states}"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if test_extra is None or "regimes" not in test_extra:
            return np.ones(test_length)
        regimes = np.asarray(test_extra["regimes"])
        weights = np.ones(test_length)
        for s in self._avoid:
            weights[regimes == s] = 0.0
        return weights


class RegimeOnlyStrategy(Strategy):
    """Only invest during specified regime states; cash otherwise.

    Requires ``test_extra["regimes"]`` to be provided.
    """

    def __init__(self, only_states: set[int]) -> None:
        self._only = only_states

    @property
    def name(self) -> str:
        states = ",".join(str(s) for s in sorted(self._only))
        return f"regime_only_{states}"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if test_extra is None or "regimes" not in test_extra:
            return np.zeros(test_length)
        regimes = np.asarray(test_extra["regimes"])
        weights = np.zeros(test_length)
        for s in self._only:
            weights[regimes == s] = 1.0
        return weights


# ═══════════════════════════════════════════════════════════════
# Multi-asset backtesting (Phase 24c)
# ═══════════════════════════════════════════════════════════════


class MultiAssetStrategy(ABC):
    """Abstract base for multi-instrument backtestable strategies.

    Unlike the 1-D ``Strategy``, this produces a weight vector per
    timestep: shape ``(test_length, N)`` where N = number of instruments.
    Weights are unconstrained by default; strategies may self-normalise
    (e.g. ``leverage_limit`` in SAC) or the backtester can clip.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        instrument_names: list[str],
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return a (test_length, N) array of portfolio weights.

        Parameters
        ----------
        train_returns : (train_len, N) log returns during training window.
        test_length : number of periods in the test window.
        instrument_names : ordered list of instrument tickers (length N).
        train_extra : optional auxiliary training data.
        test_extra : optional auxiliary test data.

        Returns
        -------
        (test_length, N) array. Each row is a weight vector across instruments.
        """
        ...


@dataclass
class MultiAssetFoldResult:
    """Metrics for a single multi-asset walk-forward fold."""

    fold: int
    train_size: int
    test_size: int
    metrics: dict[str, Any]
    weights: np.ndarray  # (test_size, N)
    portfolio_returns: np.ndarray  # (test_size,) — dot-product P&L
    per_instrument_returns: np.ndarray  # (test_size, N) — weighted returns


@dataclass
class MultiAssetBacktestResult:
    """Aggregate output of a multi-asset walk-forward backtest."""

    strategy_name: str
    folds: list[MultiAssetFoldResult]
    aggregate_metrics: dict[str, Any]
    equity_curve: np.ndarray  # cumulative wealth
    all_portfolio_returns: np.ndarray  # (T_total,)
    all_weights: np.ndarray  # (T_total, N)
    instrument_names: list[str]
    attribution: dict[str, float]  # asset_class → cumulative return contribution


class MultiAssetWalkForward:
    """Expanding-window walk-forward backtester for multi-asset strategies.

    Mirrors ``WalkForward`` but operates on (T, N) return matrices and
    computes portfolio-level returns via dot product:
        portfolio_return_t = Σ_i weights_{t,i} × returns_{t,i}

    Parameters
    ----------
    min_train : minimum training periods before first test fold.
    test_size : periods per test fold.
    step_size : advance per fold (defaults to test_size).
    periods_per_year : for annualisation of metrics.
    instrument_names : ordered list of tickers (length N).
    instrument_classes : optional {ticker → asset_class} for attribution.
    """

    def __init__(
        self,
        min_train: int = 252,
        test_size: int = 21,
        step_size: int | None = None,
        periods_per_year: int = 252,
        instrument_names: list[str] | None = None,
        instrument_classes: dict[str, str] | None = None,
    ) -> None:
        self.min_train = min_train
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.periods_per_year = periods_per_year
        self.instrument_names = instrument_names or []
        self.instrument_classes = instrument_classes or {}

    def run(
        self,
        strategy: MultiAssetStrategy,
        returns: np.ndarray,
        *,
        extra: dict[str, np.ndarray] | None = None,
    ) -> MultiAssetBacktestResult:
        """Execute multi-asset walk-forward backtest.

        Parameters
        ----------
        strategy : a MultiAssetStrategy instance.
        returns : (T, N) array of log returns. N must match instrument_names.
        extra : optional dict mapping names to arrays with T rows.

        Returns
        -------
        MultiAssetBacktestResult.
        """
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim != 2:
            raise ValueError(f"returns must be 2-D (T, N), got shape {returns.shape}")
        T, N = returns.shape
        names = self.instrument_names
        if not names:
            names = [f"asset_{i}" for i in range(N)]
        if len(names) != N:
            raise ValueError(f"instrument_names length {len(names)} != returns columns {N}")
        extra = extra or {}

        folds: list[MultiAssetFoldResult] = []
        fold_idx = 0
        split = self.min_train

        while split + self.test_size <= T:
            train_ret = returns[:split]
            test_ret = returns[split : split + self.test_size]

            train_extra = {k: v[:split] for k, v in extra.items()}
            test_extra = {k: v[split : split + self.test_size] for k, v in extra.items()}

            weights = strategy.generate_weights(
                train_ret,
                len(test_ret),
                names,
                train_extra=train_extra if train_extra else None,
                test_extra=test_extra if test_extra else None,
            )

            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (len(test_ret), N):
                raise ValueError(f"Strategy returned weights shape {weights.shape}, expected ({len(test_ret)}, {N})")

            # Per-instrument weighted returns and portfolio P&L
            per_inst_ret = test_ret * weights  # (test_size, N)
            portfolio_ret = per_inst_ret.sum(axis=1)  # (test_size,)

            fold_metrics = score_returns(
                portfolio_ret,
                weights=np.abs(weights).sum(axis=1),  # gross leverage per step
                periods_per_year=self.periods_per_year,
            )
            # Add concentration metric
            fold_metrics["max_weight"] = float(np.abs(weights).max())
            fold_metrics["mean_gross_leverage"] = float(np.abs(weights).sum(axis=1).mean())

            folds.append(
                MultiAssetFoldResult(
                    fold=fold_idx,
                    train_size=split,
                    test_size=len(test_ret),
                    metrics=fold_metrics,
                    weights=weights,
                    portfolio_returns=portfolio_ret,
                    per_instrument_returns=per_inst_ret,
                )
            )

            fold_idx += 1
            split += self.step_size

        if not folds:
            raise ValueError(
                f"Not enough data for even one fold. T={T}, min_train={self.min_train}, test_size={self.test_size}"
            )

        # Aggregate across folds
        all_portfolio_ret = np.concatenate([f.portfolio_returns for f in folds])
        all_weights = np.concatenate([f.weights for f in folds], axis=0)
        all_per_inst = np.concatenate([f.per_instrument_returns for f in folds], axis=0)
        equity = np.exp(np.cumsum(all_portfolio_ret))

        agg_metrics = score_returns(
            all_portfolio_ret,
            weights=np.abs(all_weights).sum(axis=1),
            periods_per_year=self.periods_per_year,
        )
        agg_metrics["max_weight"] = float(np.abs(all_weights).max())
        agg_metrics["mean_gross_leverage"] = float(np.abs(all_weights).sum(axis=1).mean())

        # Per-asset-class attribution: sum of weighted returns per class
        attribution = self._compute_attribution(all_per_inst, names)

        return MultiAssetBacktestResult(
            strategy_name=strategy.name,
            folds=folds,
            aggregate_metrics=agg_metrics,
            equity_curve=equity,
            all_portfolio_returns=all_portfolio_ret,
            all_weights=all_weights,
            instrument_names=names,
            attribution=attribution,
        )

    def _compute_attribution(
        self,
        per_instrument_returns: np.ndarray,
        names: list[str],
    ) -> dict[str, float]:
        """Compute cumulative return attribution by asset class.

        Returns {asset_class → sum of weighted returns}.
        Attribution sums to total portfolio return.
        """
        N = len(names)
        attr: dict[str, float] = {}
        for i in range(N):
            cls = self.instrument_classes.get(names[i], "unknown")
            attr[cls] = attr.get(cls, 0.0) + float(per_instrument_returns[:, i].sum())
        return attr


# ── Multi-asset baseline strategies ──────────────────────────


class EqualWeightStrategy(MultiAssetStrategy):
    """1/N equal weight across all instruments at all timesteps."""

    @property
    def name(self) -> str:
        return "equal_weight"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        instrument_names: list[str],
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        N = train_returns.shape[1] if train_returns.ndim == 2 else 1
        return np.full((test_length, N), 1.0 / N)


class BuyAndHoldBenchmarkStrategy(MultiAssetStrategy):
    """Fixed target weights — for benchmarks like SPY-only or 60/40.

    Instruments not in ``target_weights`` get weight 0.
    Weights are constant across all timesteps.
    """

    def __init__(self, target_weights: dict[str, float]) -> None:
        self._targets = target_weights

    @property
    def name(self) -> str:
        labels = "+".join(f"{t}:{w:.0%}" for t, w in sorted(self._targets.items()))
        return f"buy_hold_{labels}"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        instrument_names: list[str],
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        N = len(instrument_names)
        row = np.zeros(N)
        for i, name in enumerate(instrument_names):
            row[i] = self._targets.get(name, 0.0)
        return np.tile(row, (test_length, 1))
