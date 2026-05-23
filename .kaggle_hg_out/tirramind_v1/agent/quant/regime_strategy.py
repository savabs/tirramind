"""Regime-based trading strategy using world model beliefs.

Converts posterior regime beliefs into position weights for the
WalkForward backtester.

Logic:
    weight_t = P(expansion)_t - P(crisis)_t

    - Pure expansion belief → weight = +1 (full long)
    - Pure crisis belief → weight = -1 (full short)
    - Mixed/uncertain → fractional exposure proportional to belief difference

This is a linear mapping from belief space to weight space. No hand-coded
thresholds — the weight is a continuous function of the posterior.

Spec: docs/specs/world_model_bridge_spec.md (step 19e.1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.models.belief import BeliefState
from agent.quant.backtest import Strategy


class RegimeStrategy(Strategy):
    """Position sizing from world model regime beliefs.

    Parameters
    ----------
    regime_variable : Name of the DAG variable carrying regime posteriors.
    expansion_state : Label for the expansion/bullish state.
    crisis_state : Label for the crisis/bearish state.
    default_weight : Weight when beliefs are unavailable (neutral).
    """

    def __init__(
        self,
        regime_variable: str = "regime.macro",
        expansion_state: str = "expansion",
        crisis_state: str = "crisis",
        default_weight: float = 0.0,
    ) -> None:
        self._regime_var = regime_variable
        self._expansion = expansion_state
        self._crisis = crisis_state
        self._default = default_weight

    @property
    def name(self) -> str:
        return "RegimeStrategy"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Generate position weights from regime beliefs.

        ``test_extra`` must contain:
            ``"beliefs"`` : list[list[BeliefState]]
                One list of BeliefState per test period. Length == test_length.

        Returns 1-D array of weights in [-1, 1].
        """
        if test_extra is None or "beliefs" not in test_extra:
            return np.full(test_length, self._default)

        belief_sets: list[list[BeliefState]] = test_extra["beliefs"]

        weights = np.full(test_length, self._default, dtype=np.float64)

        for t in range(min(test_length, len(belief_sets))):
            beliefs = belief_sets[t]
            if beliefs is None:
                continue

            regime = self._find_regime_belief(beliefs)
            if regime is None:
                continue

            p_exp = regime.probabilities.get(self._expansion, 0.0)
            p_cri = regime.probabilities.get(self._crisis, 0.0)
            w = p_exp - p_cri
            weights[t] = max(-1.0, min(1.0, w))

        return weights

    def _find_regime_belief(self, beliefs: list[BeliefState]) -> BeliefState | None:
        """Find the regime belief in a belief set."""
        for b in beliefs:
            if b.variable_name == self._regime_var and b.probabilities:
                return b
        return None
