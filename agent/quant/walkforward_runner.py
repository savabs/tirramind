"""TirraMind — Walk-Forward Backtest Runner & Attribution Report

Orchestrates multi-asset walk-forward backtests from stored data:
    1. Load instrument log-returns from PipelineStore into (T, N) matrix.
    2. Configure baseline + model-based strategies.
    3. Run each strategy through MultiAssetWalkForward.
    4. Generate per-strategy attribution reports.

Designed for Phase 24e: end-to-end integration backtest with SAC,
WeightedSurprise (adapted), EqualWeight, BuyAndHold(SPY), BuyAndHold(60/40).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone; UTC = timezone.utc
from typing import Any

import numpy as np

from agent.quant.backtest import (
    BuyAndHoldBenchmarkStrategy,
    EqualWeightStrategy,
    MultiAssetBacktestResult,
    MultiAssetStrategy,
    MultiAssetWalkForward,
)

log = logging.getLogger(__name__)

# ── Data Loading ───────────────────────────────────────────────


def load_instrument_returns(
    store: Any,
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load daily log-returns from PipelineStore into an aligned (T, N) matrix.

    Parameters
    ----------
    store : PipelineStore instance.
    tickers : ordered list of instrument tickers to include.
    start_date : ISO date string (inclusive), e.g. '2022-01-01'.
    end_date : ISO date string (inclusive).

    Returns
    -------
    (dates, returns) where dates is a sorted list of ISO date strings
    and returns is (T, N) float64 array. Missing values are filled with 0.0.
    """
    since = None
    until = None
    if start_date:
        since = datetime.fromisoformat(start_date).replace(tzinfo=UTC).timestamp()
    if end_date:
        until = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, tzinfo=UTC).timestamp()

    obs = store.query_all_observations(since=since, until=until)

    # Filter to daily return observations for requested tickers.
    # Support both legacy "daily_return" and current "instrument_daily" obs types.
    _RETURN_OBS_TYPES = {"daily_return", "instrument_daily"}
    ticker_set = set(tickers)
    # date_str → {ticker → log_return}
    data: dict[str, dict[str, float]] = defaultdict(dict)
    for o in obs:
        if o["observation_type"] not in _RETURN_OBS_TYPES:
            continue
        eid = o["entity_id"]
        if eid not in ticker_set:
            continue
        val = o.get("value") or {}
        lr = val.get("log_return")
        if lr is None:
            continue
        dt = datetime.fromtimestamp(o["observed_at"], tz=UTC)
        day = dt.strftime("%Y-%m-%d")
        data[day][eid] = float(lr)

    if not data:
        raise ValueError("No daily return observations found for given tickers/dates.")

    dates = sorted(data.keys())
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    N = len(tickers)
    T = len(dates)
    returns = np.zeros((T, N), dtype=np.float64)
    for t, day in enumerate(dates):
        row = data[day]
        for ticker, lr in row.items():
            if ticker in ticker_idx:
                returns[t, ticker_idx[ticker]] = lr

    log.info(
        "Loaded returns: %d dates × %d instruments, range %s to %s",
        T,
        N,
        dates[0],
        dates[-1],
    )
    return dates, returns


# ── Strategy Configuration ─────────────────────────────────────


def build_default_strategies(
    instrument_names: list[str],
) -> list[MultiAssetStrategy]:
    """Build the baseline strategies that need no trained models.

    Returns [EqualWeight, BuyAndHold(SPY), BuyAndHold(60/40)].
    SAC and WeightedSurprise require trained models and are added
    separately via ``add_model_strategies()``.
    """
    strategies: list[MultiAssetStrategy] = [
        EqualWeightStrategy(),
        BuyAndHoldBenchmarkStrategy({"SPY": 1.0}),
    ]

    # 60/40: use SPY (equity) and AGG/TLT (fixed income)
    has_agg = "AGG" in instrument_names
    has_tlt = "TLT" in instrument_names
    bond_ticker = "AGG" if has_agg else ("TLT" if has_tlt else None)
    if bond_ticker:
        strategies.append(BuyAndHoldBenchmarkStrategy({"SPY": 0.6, bond_ticker: 0.4}))
    else:
        # Fallback: just 60% SPY
        strategies.append(BuyAndHoldBenchmarkStrategy({"SPY": 0.6}))

    return strategies


# ── Walk-Forward Execution ─────────────────────────────────────


def run_walkforward(
    returns: np.ndarray,
    instrument_names: list[str],
    instrument_classes: dict[str, str],
    strategies: list[MultiAssetStrategy],
    min_train: int = 252,
    test_size: int = 21,
    step_size: int = 21,
    extra: dict[str, np.ndarray] | None = None,
) -> dict[str, MultiAssetBacktestResult]:
    """Run walk-forward for multiple strategies and return results keyed by name.

    Parameters
    ----------
    returns : (T, N) log-return matrix.
    instrument_names : ordered list of tickers (length N).
    instrument_classes : {ticker → asset_class} for attribution.
    strategies : list of MultiAssetStrategy instances.
    min_train, test_size, step_size : walk-forward configuration.
    extra : optional dict mapping names to (T, ...) arrays for strategies
            that need auxiliary data (SAC, WeightedSurprise).

    Returns
    -------
    {strategy_name → MultiAssetBacktestResult}
    """
    wf = MultiAssetWalkForward(
        min_train=min_train,
        test_size=test_size,
        step_size=step_size,
        periods_per_year=252,
        instrument_names=instrument_names,
        instrument_classes=instrument_classes,
    )

    results: dict[str, MultiAssetBacktestResult] = {}
    for strategy in strategies:
        name = strategy.name
        log.info("Running walk-forward: %s", name)
        try:
            result = wf.run(strategy, returns, extra=extra)
            results[name] = result
            log.info(
                "  %s: %d folds, Sharpe=%.3f, total_return=%.4f",
                name,
                len(result.folds),
                result.aggregate_metrics.get("sharpe", float("nan")),
                result.aggregate_metrics.get("total_return", float("nan")),
            )
        except Exception:
            log.exception("Strategy %s failed", name)

    return results


# ── Attribution & Report ───────────────────────────────────────


def per_instrument_attribution(
    result: MultiAssetBacktestResult,
) -> dict[str, float]:
    """Per-instrument cumulative P&L contribution.

    Returns {ticker → sum of weighted returns across all folds}.
    Values sum to total portfolio return.
    """
    all_per_inst = np.concatenate([f.per_instrument_returns for f in result.folds], axis=0)
    names = result.instrument_names
    return {names[i]: float(all_per_inst[:, i].sum()) for i in range(len(names))}


def per_group_attribution(
    result: MultiAssetBacktestResult,
    grouping: dict[str, str],
) -> dict[str, float]:
    """Attribute P&L contribution by an arbitrary grouping.

    Parameters
    ----------
    result : backtest result with per-instrument returns in folds.
    grouping : {ticker → group_label} (e.g. asset_class or region).

    Returns
    -------
    {group_label → cumulative return contribution}.
    """
    all_per_inst = np.concatenate([f.per_instrument_returns for f in result.folds], axis=0)
    names = result.instrument_names
    attr: dict[str, float] = {}
    for i, name in enumerate(names):
        grp = grouping.get(name, "unknown")
        attr[grp] = attr.get(grp, 0.0) + float(all_per_inst[:, i].sum())
    return attr


def top_instruments(
    result: MultiAssetBacktestResult,
    n: int = 5,
) -> list[tuple[str, float]]:
    """Top N instruments by absolute P&L contribution.

    Returns list of (ticker, contribution) sorted by descending absolute value.
    """
    inst_attr = per_instrument_attribution(result)
    sorted_items = sorted(inst_attr.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_items[:n]


def concentration_stats(result: MultiAssetBacktestResult) -> dict[str, float]:
    """Compute weight concentration statistics across all folds.

    Returns dict with:
        max_abs_weight: maximum absolute weight seen
        mean_abs_weight: mean absolute weight across all (time, instrument) entries
        median_abs_weight: median absolute weight
        p90_abs_weight: 90th percentile of absolute weights
        p99_abs_weight: 99th percentile of absolute weights
        mean_gross_leverage: mean row-sum of absolute weights
        max_gross_leverage: max row-sum of absolute weights
    """
    w = result.all_weights
    abs_w = np.abs(w)
    flat = abs_w.ravel()

    # Gross leverage per timestep
    gross_leverage = abs_w.sum(axis=1)

    return {
        "max_abs_weight": float(flat.max()) if flat.size else 0.0,
        "mean_abs_weight": float(flat.mean()) if flat.size else 0.0,
        "median_abs_weight": float(np.median(flat)) if flat.size else 0.0,
        "p90_abs_weight": float(np.percentile(flat, 90)) if flat.size else 0.0,
        "p99_abs_weight": float(np.percentile(flat, 99)) if flat.size else 0.0,
        "mean_gross_leverage": (float(gross_leverage.mean()) if gross_leverage.size else 0.0),
        "max_gross_leverage": (float(gross_leverage.max()) if gross_leverage.size else 0.0),
    }


@dataclass
class StrategyReport:
    """Attribution report for a single strategy."""

    strategy_name: str
    aggregate_metrics: dict[str, Any]
    fold_count: int
    class_attribution: dict[str, float]
    region_attribution: dict[str, float]
    top_instruments: list[tuple[str, float]]
    concentration: dict[str, float]


def generate_attribution_report(
    results: dict[str, MultiAssetBacktestResult],
    instrument_classes: dict[str, str],
    instrument_regions: dict[str, str],
    top_n: int = 5,
) -> dict[str, StrategyReport]:
    """Generate comprehensive attribution reports for all strategies.

    Parameters
    ----------
    results : {strategy_name → MultiAssetBacktestResult}.
    instrument_classes : {ticker → asset_class}.
    instrument_regions : {ticker → region}.
    top_n : number of top instruments to include.

    Returns
    -------
    {strategy_name → StrategyReport}
    """
    reports: dict[str, StrategyReport] = {}
    for name, result in results.items():
        reports[name] = StrategyReport(
            strategy_name=name,
            aggregate_metrics=result.aggregate_metrics,
            fold_count=len(result.folds),
            class_attribution=per_group_attribution(result, instrument_classes),
            region_attribution=per_group_attribution(result, instrument_regions),
            top_instruments=top_instruments(result, top_n),
            concentration=concentration_stats(result),
        )
    return reports
