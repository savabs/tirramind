"""TirraMind — State Assembler

Combines heterogeneous inputs (entity alerts, beliefs, market features)
into a fixed-size state tensor for the SAC policy.

State tensor layout (contiguous float32 vector):
    [0 : E*5]               → surprise vectors for top-E entities (by composite_surprise desc)
    [E*5 : E*5 + E*4]       → belief features [mean, var, confidence, stale] per entity
    [E*9 : E*9 + M]         → global market features
    [E*9 + M : E*9 + M + 1] → normalised entity count (n_active / max_entities)
    [E*9 + M + 1 : E*9 + M + 5] → adversarial summary [mean_decay, vpin, max_crowd, n_flags]

where E = max_entities, M = market_dim.

Entities not in the asset_map are excluded.
Padding with zeros for empty slots guarantees a fixed-dim tensor.

Design choice:
    Top-K truncation by composite_surprise is a form of attention-gating:
    the policy sees only the most anomalous tradeable entities.
    Mathematically: argsort descending on composite_surprise → take first E.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from agent.fusion.alert import EntityAlert
from agent.models.belief import BeliefState

if TYPE_CHECKING:
    from agent.adversarial.flags import AdversarialFlag

_ADVERSARIAL_DIM = 4  # [mean_edge_decay, vpin, max_crowd, n_flags_normalised]


class StateAssembler:
    """Assemble heterogeneous signal into a fixed-size SAC state tensor."""

    def __init__(
        self,
        max_entities: int = 50,
        surprise_dim: int = 5,
        belief_dim: int = 4,  # mean, var, confidence, stale
        market_dim: int = 8,
    ) -> None:
        self._max_entities = max_entities
        self._surprise_dim = surprise_dim
        self._belief_dim = belief_dim
        self._market_dim = market_dim

    @property
    def state_dim(self) -> int:
        """Total dimensionality of assembled state."""
        E = self._max_entities
        return (
            E * self._surprise_dim
            + E * self._belief_dim
            + self._market_dim
            + 1
            + _ADVERSARIAL_DIM
        )

    def assemble(
        self,
        alerts: list[EntityAlert],
        beliefs: list[BeliefState],
        market_features: dict[str, float],
        asset_map: dict[str, str],
        adversarial_flags: list[AdversarialFlag] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build a fixed-size state tensor from heterogeneous inputs.

        Parameters
        ----------
        alerts : EntityAlerts from the current timestep.
        beliefs : BeliefStates from the world model.
        market_features : global features (e.g. rolling_return, vol, regime).
        asset_map : {entity_id → ticker} for tradeable entities.

        Returns
        -------
        (state_tensor, metadata) where metadata has entity ordering info.
        """
        E = self._max_entities

        # Filter to tradeable entities only
        tradeable_alerts = [a for a in alerts if a.entity_id in asset_map]

        # Sort by composite_surprise descending → top-K truncation
        tradeable_alerts.sort(key=lambda a: a.composite_surprise, reverse=True)
        selected = tradeable_alerts[:E]
        n_active = len(selected)

        # Build belief lookup
        belief_by_entity: dict[str, BeliefState] = {}
        for b in beliefs:
            if b.entity_id is not None:
                belief_by_entity[b.entity_id] = b

        # Surprise block: (E, 5)
        surprise_block = np.zeros((E, self._surprise_dim), dtype=np.float32)
        for i, alert in enumerate(selected):
            surprise_block[i] = [
                alert.obs_type_surprise,
                alert.temporal_surprise,
                alert.value_surprise,
                alert.neighborhood_surprise,
                alert.memory_drift,
            ]

        # Belief block: (E, 4) — [mean, var, confidence, stale]
        belief_block = np.zeros((E, self._belief_dim), dtype=np.float32)
        for i, alert in enumerate(selected):
            b = belief_by_entity.get(alert.entity_id)
            if b is not None:
                belief_block[i] = [
                    b.mean if b.mean is not None else 0.0,
                    b.variance if b.variance is not None else 0.0,
                    b.confidence,
                    1.0 if b.stale else 0.0,
                ]

        # Market features: (M,)
        market_block = np.zeros(self._market_dim, dtype=np.float32)
        market_keys = sorted(market_features.keys())[: self._market_dim]
        for j, key in enumerate(market_keys):
            market_block[j] = market_features.get(key, 0.0)

        # Entity count (normalised)
        entity_count = np.array([n_active / max(E, 1)], dtype=np.float32)

        # Adversarial summary block: 4 features
        adv_block = self._adversarial_block(adversarial_flags)

        # Concatenate
        state = np.concatenate(
            [
                surprise_block.ravel(),
                belief_block.ravel(),
                market_block,
                entity_count,
                adv_block,
            ]
        )

        metadata = {
            "n_active": n_active,
            "entity_order": [a.entity_id for a in selected],
            "ticker_order": [asset_map[a.entity_id] for a in selected],
        }

        return torch.from_numpy(state), metadata

    @staticmethod
    def _adversarial_block(
        flags: list[AdversarialFlag] | None,
    ) -> np.ndarray:
        """Build the 4-dim adversarial summary.

        [0] mean_edge_decay:  mean severity of edge_decay flags (0 if none)
        [1] vpin:             max severity of vpin_spike flags (0 if none)
        [2] max_crowd:        max severity of crowding_risk flags (0 if none)
        [3] n_flags_norm:     number of active flags / 10 (soft normalisation)
        """
        block = np.zeros(_ADVERSARIAL_DIM, dtype=np.float32)
        if not flags:
            return block

        decay_sevs = [f.severity for f in flags if f.flag_type == "edge_decay"]
        vpin_sevs = [f.severity for f in flags if f.flag_type == "vpin_spike"]
        crowd_sevs = [f.severity for f in flags if f.flag_type == "crowding_risk"]

        if decay_sevs:
            block[0] = float(np.mean(decay_sevs))
        if vpin_sevs:
            block[1] = float(np.max(vpin_sevs))
        if crowd_sevs:
            block[2] = float(np.max(crowd_sevs))
        block[3] = min(len(flags) / 10.0, 1.0)

        return block


class InstrumentStateAssembler:
    """Assemble multi-asset state tensor with per-instrument surprise block.

    State layout (contiguous float32):
        [0 : N*5]                  → instrument surprise vectors (fixed ordering)
        [N*5 : N*5 + E*5]         → top-E entity surprise vectors
        [... : ... + E*4]          → entity belief features
        [... : ... + M]            → global market features
        [... : ... + 1]            → normalised entity count
        [... : ... + 4]            → adversarial summary

    where N = n_instruments, E = max_entities, M = market_dim.

    The instrument block uses a fixed ordering matching the constructor's
    ``instrument_tickers`` list, so SAC action dimension i maps to
    instrument i.  Missing instruments get zero-padded surprise.
    """

    def __init__(
        self,
        instrument_tickers: list[str],
        max_entities: int = 50,
        surprise_dim: int = 5,
        belief_dim: int = 4,
        market_dim: int = 8,
    ) -> None:
        self._tickers = list(instrument_tickers)
        self._n_instruments = len(instrument_tickers)
        self._ticker_index = {t: i for i, t in enumerate(instrument_tickers)}
        self._max_entities = max_entities
        self._surprise_dim = surprise_dim
        self._belief_dim = belief_dim
        self._market_dim = market_dim

    @property
    def n_instruments(self) -> int:
        return self._n_instruments

    @property
    def state_dim(self) -> int:
        N = self._n_instruments
        E = self._max_entities
        return (
            N * self._surprise_dim
            + E * self._surprise_dim
            + E * self._belief_dim
            + self._market_dim
            + 1
            + _ADVERSARIAL_DIM
        )

    def assemble(
        self,
        instrument_surprises: dict[str, tuple[float, ...]],
        entity_alerts: list[EntityAlert],
        beliefs: list[BeliefState],
        market_features: dict[str, float],
        asset_map: dict[str, str] | None = None,
        adversarial_flags: list[AdversarialFlag] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build a fixed-size state tensor for multi-asset SAC.

        Parameters
        ----------
        instrument_surprises : {ticker → 5-tuple surprise_vector} for instruments.
        entity_alerts : EntityAlerts from the current timestep.
        beliefs : BeliefStates from the world model.
        market_features : global features.
        asset_map : optional {entity_id → ticker} for entity filtering.
        adversarial_flags : optional adversarial flags.

        Returns
        -------
        (state_tensor, metadata)
        """
        N = self._n_instruments
        E = self._max_entities

        # ── Instrument surprise block (N, 5) ──
        inst_block = np.zeros((N, self._surprise_dim), dtype=np.float32)
        n_instruments_active = 0
        for ticker, sv in instrument_surprises.items():
            idx = self._ticker_index.get(ticker)
            if idx is not None:
                inst_block[idx] = sv[:self._surprise_dim]
                n_instruments_active += 1

        # ── Entity surprise block (E, 5) — top-E by composite ──
        if asset_map:
            tradeable_alerts = [
                a for a in entity_alerts if a.entity_id in asset_map
            ]
        else:
            tradeable_alerts = list(entity_alerts)
        tradeable_alerts.sort(key=lambda a: a.composite_surprise, reverse=True)
        selected = tradeable_alerts[:E]
        n_entities_active = len(selected)

        entity_surprise = np.zeros((E, self._surprise_dim), dtype=np.float32)
        for i, alert in enumerate(selected):
            entity_surprise[i] = [
                alert.obs_type_surprise,
                alert.temporal_surprise,
                alert.value_surprise,
                alert.neighborhood_surprise,
                alert.memory_drift,
            ]

        # ── Belief block (E, 4) ──
        belief_by_entity: dict[str, BeliefState] = {}
        for b in beliefs:
            if b.entity_id is not None:
                belief_by_entity[b.entity_id] = b

        belief_block = np.zeros((E, self._belief_dim), dtype=np.float32)
        for i, alert in enumerate(selected):
            b = belief_by_entity.get(alert.entity_id)
            if b is not None:
                belief_block[i] = [
                    b.mean if b.mean is not None else 0.0,
                    b.variance if b.variance is not None else 0.0,
                    b.confidence,
                    1.0 if b.stale else 0.0,
                ]

        # ── Market block (M,) ──
        market_block = np.zeros(self._market_dim, dtype=np.float32)
        market_keys = sorted(market_features.keys())[: self._market_dim]
        for j, key in enumerate(market_keys):
            market_block[j] = market_features.get(key, 0.0)

        # ── Entity count + adversarial ──
        entity_count = np.array(
            [n_entities_active / max(E, 1)], dtype=np.float32
        )
        adv_block = StateAssembler._adversarial_block(adversarial_flags)

        state = np.concatenate(
            [
                inst_block.ravel(),
                entity_surprise.ravel(),
                belief_block.ravel(),
                market_block,
                entity_count,
                adv_block,
            ]
        )

        metadata = {
            "n_instruments_active": n_instruments_active,
            "n_entities_active": n_entities_active,
            "instrument_tickers": list(self._tickers),
            "entity_order": [a.entity_id for a in selected],
        }

        return torch.from_numpy(state), metadata


class DifferentiableStateAssembler:
    """State assembler with gradient-preserving belief block.

    Produces the same state layout as ``InstrumentStateAssembler`` but
    accepts belief means/variances as torch Tensors so autograd can
    back-propagate through the belief segment to upstream Kalman params.

    Non-belief components (instrument surprises, entity surprises, market
    features, adversarial) remain numpy → ``torch.from_numpy()`` (detached).
    Only the belief block carries gradients.

    State layout (identical to InstrumentStateAssembler):
        [0 : N*5]            → instrument surprise (detached)
        [N*5 : N*5 + E*5]   → entity surprise (detached)
        [... : ... + E*4]    → belief features (**differentiable**)
        [... : ... + M]      → market features (detached)
        [... : ... + 1]      → entity count (detached)
        [... : ... + 4]      → adversarial summary (detached)
    """

    def __init__(
        self,
        instrument_tickers: list[str],
        max_entities: int = 50,
        surprise_dim: int = 5,
        belief_dim: int = 4,
        market_dim: int = 8,
    ) -> None:
        self._tickers = list(instrument_tickers)
        self._n_instruments = len(instrument_tickers)
        self._ticker_index = {t: i for i, t in enumerate(instrument_tickers)}
        self._max_entities = max_entities
        self._surprise_dim = surprise_dim
        self._belief_dim = belief_dim
        self._market_dim = market_dim

    @property
    def n_instruments(self) -> int:
        return self._n_instruments

    @property
    def state_dim(self) -> int:
        N = self._n_instruments
        E = self._max_entities
        return (
            N * self._surprise_dim
            + E * self._surprise_dim
            + E * self._belief_dim
            + self._market_dim
            + 1
            + _ADVERSARIAL_DIM
        )

    def assemble(
        self,
        instrument_surprises: dict[str, tuple[float, ...]],
        entity_alerts: list[EntityAlert],
        belief_means: torch.Tensor,
        belief_variances: torch.Tensor,
        market_features: dict[str, float],
        asset_map: dict[str, str] | None = None,
        adversarial_flags: list[Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build state tensor with differentiable belief block.

        Parameters
        ----------
        instrument_surprises : {ticker → 5-tuple} for instruments.
        entity_alerts : EntityAlerts from the current timestep.
        belief_means : Tensor (state_dim_kalman,) — from DiffKalman.
        belief_variances : Tensor (state_dim_kalman,) — from DiffKalman.
        market_features : global features dict.
        asset_map : optional {entity_id → ticker}.
        adversarial_flags : optional adversarial flags.

        Returns
        -------
        (state_tensor, metadata) where state_tensor has gradients through
        the belief block.
        """
        N = self._n_instruments
        E = self._max_entities

        # ── Instrument surprise block (N, 5) — detached ──
        inst_block = np.zeros((N, self._surprise_dim), dtype=np.float32)
        n_instruments_active = 0
        for ticker, sv in instrument_surprises.items():
            idx = self._ticker_index.get(ticker)
            if idx is not None:
                inst_block[idx] = sv[: self._surprise_dim]
                n_instruments_active += 1

        # ── Entity surprise block (E, 5) — detached ──
        if asset_map:
            tradeable_alerts = [
                a for a in entity_alerts if a.entity_id in asset_map
            ]
        else:
            tradeable_alerts = list(entity_alerts)
        tradeable_alerts.sort(key=lambda a: a.composite_surprise, reverse=True)
        selected = tradeable_alerts[:E]
        n_entities_active = len(selected)

        entity_surprise = np.zeros((E, self._surprise_dim), dtype=np.float32)
        for i, alert in enumerate(selected):
            entity_surprise[i] = [
                alert.obs_type_surprise,
                alert.temporal_surprise,
                alert.value_surprise,
                alert.neighborhood_surprise,
                alert.memory_drift,
            ]

        # ── Belief block (E, 4) — DIFFERENTIABLE ──
        # Map Kalman state components directly to belief slots.
        # The first min(n_kalman, E) slots get gradient-connected
        # mean/variance from the Kalman filter.  Remaining slots are zeros.
        # This is independent of n_entities_active — the surprise and
        # belief blocks serve different purposes in the differentiable path.
        n_kalman = belief_means.shape[0]
        belief_rows: list[torch.Tensor] = []
        for i in range(E):
            if i < n_kalman:
                mean_i = belief_means[i]
                var_i = belief_variances[i]
                confidence = torch.tensor(1.0)
                stale = torch.tensor(0.0)
                row = torch.stack([mean_i, var_i, confidence, stale])
            else:
                row = torch.zeros(self._belief_dim)
            belief_rows.append(row)
        belief_block = torch.stack(belief_rows)  # (E, 4) — grad connected

        # ── Market block (M,) — detached ──
        market_block = np.zeros(self._market_dim, dtype=np.float32)
        market_keys = sorted(market_features.keys())[: self._market_dim]
        for j, key in enumerate(market_keys):
            market_block[j] = market_features.get(key, 0.0)

        # ── Entity count + adversarial — detached ──
        entity_count = np.array(
            [n_entities_active / max(E, 1)], dtype=np.float32
        )
        adv_block = StateAssembler._adversarial_block(adversarial_flags)

        # ── Concatenate: detached numpy → tensor, then cat with diff belief ──
        inst_t = torch.from_numpy(inst_block.ravel())
        ent_t = torch.from_numpy(entity_surprise.ravel())
        belief_t = belief_block.reshape(-1)  # (E*4,) — grad connected
        market_t = torch.from_numpy(market_block)
        count_t = torch.from_numpy(entity_count)
        adv_t = torch.from_numpy(adv_block)

        state = torch.cat([inst_t, ent_t, belief_t, market_t, count_t, adv_t])

        metadata = {
            "n_instruments_active": n_instruments_active,
            "n_entities_active": n_entities_active,
            "instrument_tickers": list(self._tickers),
            "entity_order": [a.entity_id for a in selected],
        }

        return state, metadata
