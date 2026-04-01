"""Agent tool: run walk-forward backtests on regime-based strategies."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult
from agent.tools.macro_data import MacroDataTool
from agent.tools.market_data import MarketDataTool
from agent.quant.backtest import (
    WalkForward,
    BuyAndHoldStrategy,
    RegimeAvoidStrategy,
    RegimeOnlyStrategy,
    regime_conditional_analysis,
)
from agent.quant.liquidity import LiquidityComposite
from agent.quant.regime import RegimeHMM
from agent.quant.scoring import block_bootstrap_ci, sharpe_ratio

log = logging.getLogger(__name__)


class BacktestTool(Tool):
    """Run a walk-forward backtest of regime-based strategies on SPY."""

    name = "backtest"
    description = (
        "Run a walk-forward backtest using the liquidity regime model. "
        "Supports strategies: buy_and_hold, regime_avoid, regime_only. "
        "Returns aggregate metrics (Sharpe, Sortino, Calmar, max DD, VaR, CVaR), "
        "per-regime breakdown, and optional bootstrap confidence intervals."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": ["buy_and_hold", "regime_avoid", "regime_only"],
                "description": "Strategy type to backtest.",
            },
            "avoid_states": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Regime states to avoid (for regime_avoid). Default: [2].",
                "default": [2],
            },
            "only_states": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Regime states to invest during (for regime_only). Default: [1].",
                "default": [1],
            },
            "min_train_years": {
                "type": "integer",
                "description": "Minimum training window in years. Default: 2.",
                "default": 2,
            },
            "test_years": {
                "type": "integer",
                "description": "Test window size in years. Default: 1.",
                "default": 1,
            },
            "bootstrap": {
                "type": "boolean",
                "description": "Compute bootstrap CIs on Sharpe. Default: false.",
                "default": False,
            },
            "lookback_years": {
                "type": "integer",
                "description": "Total years of data to use. Default: 15.",
                "default": 15,
            },
        },
        "required": ["strategy"],
    }

    def __init__(
        self,
        fred_api_key: str,
        cache: DataCache | None = None,
    ) -> None:
        self._cache = cache or DataCache()
        self._macro = MacroDataTool(fred_api_key=fred_api_key, cache=self._cache)
        self._market = MarketDataTool(cache=self._cache)

    def execute(
        self,
        *,
        strategy: str = "buy_and_hold",
        avoid_states: list[int] | None = None,
        only_states: list[int] | None = None,
        min_train_years: int = 2,
        test_years: int = 1,
        bootstrap: bool = False,
        lookback_years: int = 15,
        **_: Any,
    ) -> ToolResult:
        import pandas as pd

        try:
            # Fetch liquidity composite
            lc = LiquidityComposite(self._macro, self._market)
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            start = (
                pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
            ).strftime("%Y-%m-%d")

            raw = lc.fetch_us(start, end)
            composite = lc.compute(raw)
            values = composite.values

            if len(values) < 104:
                return ToolResult(
                    success=False,
                    output=f"Insufficient data: {len(values)} weeks (need ≥104).",
                )

            # Fit HMM on full series for regime labels
            hmm = RegimeHMM(n_states=3, n_init=10, max_iter=200)
            hmm_result = hmm.fit(values)
            regimes = hmm_result.states

            # Fetch SPY returns aligned to composite dates (use daily, resample to W-WED)
            spy_result = self._market.execute(
                tickers="SPY", period="max", interval="1d"
            )
            spy_bars = spy_result.data.get("SPY", [])
            if not spy_bars:
                return ToolResult(success=False, output="Failed to fetch SPY data.")

            spy_dates = pd.to_datetime(
                [b["Date"] for b in spy_bars], utc=True
            ).tz_localize(None).normalize()
            spy_close = pd.Series(
                [b["Close"] for b in spy_bars], index=spy_dates, name="spy"
            ).sort_index()
            spy_close = spy_close[~spy_close.index.duplicated(keep="last")]

            # Resample to W-WED to match composite grid
            spy_weekly = spy_close.resample("W-WED").last().dropna()
            spy_returns = np.log(spy_weekly / spy_weekly.shift(1)).dropna()

            # Align to composite dates
            common = composite.index.intersection(spy_returns.index)
            if len(common) < 104:
                return ToolResult(
                    success=False,
                    output=f"Insufficient aligned data: {len(common)} weeks.",
                )
            spy_ret = spy_returns.loc[common].values
            regime_aligned = pd.Series(regimes, index=composite.index).loc[common].values

            # Build strategy
            if strategy == "buy_and_hold":
                strat = BuyAndHoldStrategy()
            elif strategy == "regime_avoid":
                states = set(avoid_states) if avoid_states else {2}
                strat = RegimeAvoidStrategy(avoid_states=states)
            elif strategy == "regime_only":
                states = set(only_states) if only_states else {1}
                strat = RegimeOnlyStrategy(only_states=states)
            else:
                return ToolResult(success=False, output=f"Unknown strategy: {strategy}")

            # Run walk-forward
            wf = WalkForward(
                min_train=min_train_years * 52,
                test_size=test_years * 52,
                periods_per_year=52,
            )
            bt_result = wf.run(strat, spy_ret, extra={"regimes": regime_aligned})

            # Regime-conditional analysis
            rca = regime_conditional_analysis(spy_ret, regime_aligned)

            # Format output
            agg = bt_result.aggregate_metrics
            lines = [
                f"Strategy: {bt_result.strategy_name}",
                f"Folds: {len(bt_result.folds)}",
                f"Test periods: {len(bt_result.all_test_returns)} weeks",
                f"Annualized return: {agg['annualized_return']:.2%}",
                f"Total return: {agg['total_return']:.2%}",
                f"Sharpe: {agg['sharpe']:.3f}",
                f"Sortino: {agg['sortino']:.3f}",
                f"Calmar: {agg['calmar']:.3f}",
                f"Max drawdown: {agg['max_drawdown']:.2%}",
                f"Drawdown duration: {agg['drawdown_duration']} weeks",
                f"VaR(95%): {agg['var_95']:.4f}",
                f"CVaR(95%): {agg['cvar_95']:.4f}",
            ]

            if "turnover" in agg:
                lines.append(f"Turnover: {agg['turnover']:.4f}")

            lines.append("")
            lines.append("Per-regime breakdown:")
            for reg, metrics in sorted(rca.items()):
                if "insufficient_data" in metrics:
                    lines.append(f"  Regime {reg}: insufficient data (n={metrics['n_periods']})")
                else:
                    lines.append(
                        f"  Regime {reg}: n={metrics['n_periods']} ({metrics['pct_time']:.0%}), "
                        f"Sharpe={metrics['sharpe']:.3f}, MaxDD={metrics['max_drawdown']:.2%}"
                    )

            # Optional bootstrap
            boot_data = None
            if bootstrap:
                point, lo, hi = block_bootstrap_ci(
                    bt_result.all_test_returns,
                    lambda r: sharpe_ratio(r, periods_per_year=52),
                    confidence=0.95,
                    n_bootstrap=2000,
                )
                lines.append("")
                lines.append(f"Bootstrap 95% CI on Sharpe: [{lo:.3f}, {hi:.3f}]")
                boot_data = {"sharpe_ci_lower": lo, "sharpe_ci_upper": hi}

            result_data = {
                "strategy": bt_result.strategy_name,
                "n_folds": len(bt_result.folds),
                "aggregate_metrics": agg,
                "regime_breakdown": rca,
            }
            if boot_data:
                result_data["bootstrap"] = boot_data

            return ToolResult(
                success=True,
                output="\n".join(lines),
                data=result_data,
            )

        except Exception as e:
            log.exception("BacktestTool failed")
            return ToolResult(success=False, output=f"Error: {e}")
