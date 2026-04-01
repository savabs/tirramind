"""Performance scoring utilities for return series."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """Annualized Sharpe ratio from periodic log returns."""
    excess = returns - risk_free / periods_per_year
    mu = excess.mean() * periods_per_year
    sigma = excess.std() * np.sqrt(periods_per_year)
    if sigma == 0:
        return 0.0
    return float(mu / sigma)


def sortino_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """Annualized Sortino ratio (penalizes downside volatility only)."""
    excess = returns - risk_free / periods_per_year
    mu = excess.mean() * periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0 if mu == 0 else float("inf")
    downside_std = np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year)
    if downside_std == 0:
        return 0.0
    return float(mu / downside_std)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from a series of log returns.

    Returns a negative number (e.g. -0.25 means -25%).
    """
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 52,
) -> float:
    """Calmar ratio: annualized return / |max drawdown|."""
    ann_ret = returns.mean() * periods_per_year
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(ann_ret / abs(mdd))


def value_at_risk(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Historical Value at Risk at the given confidence level.

    Returns a negative number representing the loss threshold.
    E.g., VaR(0.95) = -0.02 means 5% of periods had losses worse than -2%.
    """
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Conditional Value at Risk (Expected Shortfall) at the given confidence level.

    Mean of returns in the worst (1-confidence) tail.
    """
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return float(var)
    return float(tail.mean())


def drawdown_duration(returns: np.ndarray) -> int:
    """Maximum drawdown duration in periods.

    Counts the longest consecutive stretch where the equity curve
    stays below its running peak.
    """
    if len(returns) == 0:
        return 0
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    underwater = cum < peak

    max_dur = 0
    current = 0
    for uw in underwater:
        if uw:
            current += 1
            if current > max_dur:
                max_dur = current
        else:
            current = 0
    return max_dur


def turnover(weights: np.ndarray) -> float:
    """Mean absolute weight change per period.

    Parameters
    ----------
    weights : 1-D array of position weights over time.
    """
    if len(weights) < 2:
        return 0.0
    return float(np.abs(np.diff(weights)).mean())


def information_ratio(
    returns: np.ndarray,
    benchmark: np.ndarray,
    periods_per_year: int = 52,
) -> float:
    """Annualized information ratio vs a benchmark."""
    active = returns - benchmark
    mu = active.mean() * periods_per_year
    te = active.std() * np.sqrt(periods_per_year)
    if te == 0:
        return 0.0
    return float(mu / te)


def hit_rate(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Fraction of directionally correct predictions."""
    if len(predictions) == 0:
        return 0.0
    correct = np.sign(predictions) == np.sign(actuals)
    return float(correct.mean())


def score_returns(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = 52,
    weights: np.ndarray | None = None,
    benchmark: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute all scoring metrics in one call.

    Returns a dict with keys: sharpe, sortino, calmar, max_drawdown,
    drawdown_duration, var_95, cvar_95, annualized_return, total_return.
    Optionally includes turnover (if weights given) and information_ratio (if benchmark given).
    """
    ann_ret = float(returns.mean() * periods_per_year)
    total_ret = float(np.expm1(returns.sum()))

    result: dict[str, Any] = {
        "annualized_return": ann_ret,
        "total_return": total_ret,
        "sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free, periods_per_year),
        "calmar": calmar_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "drawdown_duration": drawdown_duration(returns),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": cvar(returns, 0.95),
    }

    if weights is not None:
        result["turnover"] = turnover(weights)

    if benchmark is not None:
        result["information_ratio"] = information_ratio(
            returns, benchmark, periods_per_year
        )

    return result


def block_bootstrap_ci(
    returns: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    block_length: int | None = None,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Block bootstrap confidence interval for a metric.

    Uses circular block bootstrap to preserve autocorrelation.

    Parameters
    ----------
    returns : 1-D return series.
    metric_fn : function(returns) -> float (e.g., sharpe_ratio).
    confidence : CI level (default 0.95 → 95% CI).
    n_bootstrap : number of bootstrap resamples.
    block_length : block size; defaults to T^(1/3).
    seed : random seed for reproducibility.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    T = len(returns)

    if block_length is None:
        block_length = max(1, int(round(T ** (1.0 / 3.0))))
    block_length = min(block_length, T // 2) if T > 1 else 1

    point = metric_fn(returns)
    boot_stats = np.empty(n_bootstrap)

    n_blocks = int(np.ceil(T / block_length))

    for b in range(n_bootstrap):
        # Circular block bootstrap: pick random start indices
        starts = rng.integers(0, T, size=n_blocks)
        indices = np.concatenate(
            [np.arange(s, s + block_length) % T for s in starts]
        )[:T]
        boot_stats[b] = metric_fn(returns[indices])

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return point, ci_lower, ci_upper
