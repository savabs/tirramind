"""Agent tool: query current liquidity regime."""

from __future__ import annotations

import logging
from typing import Any

from agent.data.cache import DataCache
from agent.quant.changepoint import BOCPD
from agent.quant.liquidity import LiquidityComposite
from agent.quant.regime import RegimeHMM
from agent.tools.base import Tool, ToolResult
from agent.tools.macro_data import MacroDataTool
from agent.tools.market_data import MarketDataTool

log = logging.getLogger(__name__)


class LiquidityRegimeTool(Tool):
    """Detect the current global liquidity regime using HMM + BOCPD."""

    name = "liquidity_regime"
    description = (
        "Analyse the current US (or global) liquidity regime. "
        "Fetches Fed balance-sheet data, computes a net-liquidity composite, "
        "fits a 3-state Hidden Markov Model and BOCPD changepoint detector, "
        "then returns the current regime label, confidence, last changepoint, "
        "and transition probabilities."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "lookback_years": {
                "type": "integer",
                "description": "Years of history to use for model fitting. Default: 5.",
                "default": 5,
            },
            "global_": {
                "type": "boolean",
                "description": ("If true, include ECB and BOJ in the composite. Default: false."),
                "default": False,
            },
        },
        "required": [],
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
        lookback_years: int = 5,
        global_: bool = False,
        **_: Any,
    ) -> ToolResult:
        import pandas as pd

        try:
            lc = LiquidityComposite(self._macro, self._market)

            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            start = (pd.Timestamp.now() - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")

            if global_:
                raw = lc.fetch_global(start, end)
            else:
                raw = lc.fetch_us(start, end)

            composite = lc.compute(raw, global_=global_)
            values = composite.values

            if len(values) < 60:
                return ToolResult(
                    success=False,
                    output=f"Insufficient data: {len(values)} weeks (need ≥60).",
                )

            # Fit HMM
            hmm = RegimeHMM(n_states=3, n_init=10, max_iter=200)
            hmm_result = hmm.fit(values)
            current_state = int(hmm_result.states[-1])
            state_labels = {0: "contraction", 1: "neutral", 2: "expansion"}

            # BOCPD
            bocpd = BOCPD(hazard_lambda=200)
            bocpd_result = bocpd.fit(values)
            cps = bocpd_result.changepoints()
            last_cp_date = composite.index[cps[-1]].strftime("%Y-%m-%d") if cps else None

            # Transition probs from current state
            trans = hmm_result.transition_matrix[current_state]

            summary = (
                f"Current liquidity regime: {state_labels[current_state]} "
                f"(state {current_state})\n"
                f"Composite value: {values[-1]:.2f} (z-score)\n"
                f"Regime means: {', '.join(f's{i}={m:.2f}' for i, m in enumerate(hmm_result.means))}\n"
                f"Last changepoint: {last_cp_date or 'none detected'}\n"
                f"Transition probs from current state: "
                f"{', '.join(f'→s{i}: {p:.1%}' for i, p in enumerate(trans))}\n"
                f"Data: {len(values)} weeks, {composite.index[0].date()} to {composite.index[-1].date()}"
            )

            return ToolResult(
                success=True,
                output=summary,
                data={
                    "current_regime": state_labels[current_state],
                    "current_state": current_state,
                    "composite_zscore": float(values[-1]),
                    "regime_means": hmm_result.means.tolist(),
                    "regime_variances": hmm_result.variances.tolist(),
                    "transition_matrix": hmm_result.transition_matrix.tolist(),
                    "last_changepoint": last_cp_date,
                    "n_changepoints": len(cps),
                    "n_weeks": len(values),
                },
            )
        except Exception as e:
            log.exception("LiquidityRegimeTool failed")
            return ToolResult(success=False, output=f"Error: {e}")
