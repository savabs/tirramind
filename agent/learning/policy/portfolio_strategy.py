"""TirraMind — Portfolio Strategy Adapters

Two Strategy-ABC implementations that bridge the RL policy (L5) to
the walk-forward backtester (L2):

1. WeightedSurpriseStrategy  (Phase 21a)
   Binary long for entities whose learned-weight composite surprise
   exceeds a z-threshold.  Weights are 1/N equal-weight across
   triggered entities.

2. SACPortfolioStrategy      (Phase 21b)
   Runs the SAC policy on alert/belief state tensors to produce
   continuous portfolio weights per timestep.

Both satisfy the Strategy ABC from agent.quant.backtest and thus
participate in walk-forward evaluation.

Trusted sources:
    - Strategy ABC: agent/quant/backtest.py
    - SAC: Haarnoja 2018 (arXiv:1801.01290)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from agent.learning.policy.asset_mapper import AssetMapper
from agent.learning.policy.sac import SACTrainer
from agent.learning.policy.state_assembler import (
    InstrumentStateAssembler,
    StateAssembler,
)
from agent.quant.backtest import MultiAssetStrategy, Strategy

log = logging.getLogger(__name__)


class WeightedSurpriseStrategy(Strategy):
    """Equal-weight long on entities whose composite surprise > threshold.

    Uses learned surprise weights from Phase 21a to compute composite
    surprises, then triggers a 1/N equal-weight position for entities
    above the z-threshold.

    If no entities pass the threshold on a given timestep, weight = 0
    (flat, no position).
    """

    def __init__(
        self,
        weights: tuple[float, ...],
        asset_mapper: AssetMapper,
        threshold: float = 2.0,
    ) -> None:
        if len(weights) != 5:
            raise ValueError(f"Expected 5 surprise weights, got {len(weights)}")
        self._weights = np.array(weights, dtype=np.float64)
        self._asset_mapper = asset_mapper
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "weighted_surprise"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Generate binary long positions for entities above surprise threshold.

        test_extra must contain:
            'alerts': list[list[EntityAlert]] — one inner list per test timestep
        """
        if test_extra is None or "alerts" not in test_extra:
            return np.zeros(test_length)

        alerts_per_step: list[list] = test_extra["alerts"]
        weights = np.zeros(test_length)

        for t in range(min(test_length, len(alerts_per_step))):
            step_alerts = alerts_per_step[t]
            triggered = 0
            for alert in step_alerts:
                # Check tradeable
                if self._asset_mapper.resolve(alert.entity_id) is None:
                    continue
                # Compute composite from learned weights
                surprises = np.array(
                    [
                        alert.obs_type_surprise,
                        alert.temporal_surprise,
                        alert.value_surprise,
                        alert.neighborhood_surprise,
                        alert.memory_drift,
                    ]
                )
                composite = float(self._weights @ surprises)
                if composite > self._threshold:
                    triggered += 1

            # Equal weight: 1/N across triggered, 0 if none
            if triggered > 0:
                weights[t] = 1.0  # fully invested (1/N is implicit at portfolio level)

        return weights


class SACPortfolioStrategy(Strategy):
    """Wraps a trained SAC policy as a walk-forward Strategy.

    For each test timestep, assembles the state tensor from alerts,
    beliefs, and market features, then queries the SAC policy for
    deterministic portfolio weights.

    The returned weight array contains the mean absolute allocation,
    representing the policy's conviction level at each timestep.
    """

    def __init__(
        self,
        trainer: SACTrainer,
        state_assembler: StateAssembler,
        asset_mapper: AssetMapper,
    ) -> None:
        self._trainer = trainer
        self._assembler = state_assembler
        self._asset_mapper = asset_mapper

    @property
    def name(self) -> str:
        return "sac_rl_policy"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Run SAC policy to generate portfolio weights for each test period.

        test_extra must contain:
            'alerts': list[list[EntityAlert]] — alerts per timestamp
            'beliefs': list[list[BeliefState]] — beliefs per timestamp
            'market_features': list[dict[str, float]] — market features per timestamp
        """
        if test_extra is None:
            return np.zeros(test_length)

        alerts_list = test_extra.get("alerts", [])
        beliefs_list = test_extra.get("beliefs", [])
        market_list = test_extra.get("market_features", [])

        # Build asset map once
        asset_map = self._asset_mapper.tradeable_entities()

        weights = np.zeros(test_length)

        for t in range(test_length):
            alerts = alerts_list[t] if t < len(alerts_list) else []
            beliefs = beliefs_list[t] if t < len(beliefs_list) else []
            market = market_list[t] if t < len(market_list) else {}

            state, meta = self._assembler.assemble(alerts, beliefs, market, asset_map)

            if meta["n_active"] == 0:
                weights[t] = 0.0
                continue

            # Get deterministic action from policy
            action = self._trainer.select_action(state, deterministic=True)

            # Conviction level = mean absolute position
            weights[t] = float(np.abs(action).mean())

        return weights


class MultiAssetSACStrategy(MultiAssetStrategy):
    """Multi-asset SAC policy producing per-instrument weight vectors.

    For each test timestep, assembles the instrument-augmented state tensor
    and queries SAC for an N-dimensional deterministic action, where each
    action dimension corresponds to one instrument.

    test_extra must contain:
        'instrument_surprises': list[dict[str, tuple]] — per-timestep
        'entity_alerts': list[list[EntityAlert]] — per-timestep
        'beliefs': list[list[BeliefState]] — per-timestep
        'market_features': list[dict[str, float]] — per-timestep
    """

    def __init__(
        self,
        trainer: SACTrainer,
        state_assembler: InstrumentStateAssembler,
    ) -> None:
        self._trainer = trainer
        self._assembler = state_assembler

    @property
    def name(self) -> str:
        return "multi_asset_sac"

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
        weights = np.zeros((test_length, N))

        if test_extra is None:
            return weights

        inst_surprises_list = test_extra.get("instrument_surprises", [])
        alerts_list = test_extra.get("entity_alerts", [])
        beliefs_list = test_extra.get("beliefs", [])
        market_list = test_extra.get("market_features", [])

        for t in range(test_length):
            inst_surp = inst_surprises_list[t] if t < len(inst_surprises_list) else {}
            alerts = alerts_list[t] if t < len(alerts_list) else []
            beliefs = beliefs_list[t] if t < len(beliefs_list) else []
            market = market_list[t] if t < len(market_list) else {}

            state, meta = self._assembler.assemble(
                instrument_surprises=inst_surp,
                entity_alerts=alerts,
                beliefs=beliefs,
                market_features=market,
            )

            # SAC deterministic action → N-dim weight vector
            action = self._trainer.select_action(state, deterministic=True)

            # Truncate or pad to match instrument count
            a_len = min(len(action), N)
            weights[t, :a_len] = action[:a_len]

        return weights


class MultiAssetWeightedSurpriseStrategy(MultiAssetStrategy):
    """Multi-asset surprise-weighted strategy (Phase 24e adaptation).

    For each test timestep, computes a composite surprise per instrument
    using learned weights.  Instruments whose composite surprise exceeds
    a z-threshold receive equal-weight long allocation (1/K where K is
    the number of triggered instruments).  Others receive weight 0.

    test_extra must contain:
        'instrument_surprises': list[dict[str, tuple[float, ...]]]
            Per-timestep mapping ticker → surprise vector (e.g. 5 floats).
    """

    def __init__(
        self,
        surprise_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
        threshold: float = 2.0,
    ) -> None:
        self._weights = np.array(surprise_weights, dtype=np.float64)
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "multi_asset_weighted_surprise"

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
        weights = np.zeros((test_length, N))

        if test_extra is None:
            return weights

        inst_surprises_list = test_extra.get("instrument_surprises", [])
        name_to_idx = {n: i for i, n in enumerate(instrument_names)}

        for t in range(test_length):
            if t >= len(inst_surprises_list):
                continue
            surp_map = inst_surprises_list[t]
            triggered: list[int] = []
            for ticker, surp_vec in surp_map.items():
                idx = name_to_idx.get(ticker)
                if idx is None:
                    continue
                sv = np.array(surp_vec, dtype=np.float64)
                # Composite surprise: dot product of learned weights × surprise
                w = self._weights[: len(sv)]
                composite = float(np.dot(w, sv))
                if composite > self._threshold:
                    triggered.append(idx)
            if triggered:
                k = len(triggered)
                for idx in triggered:
                    weights[t, idx] = 1.0 / k

        return weights
