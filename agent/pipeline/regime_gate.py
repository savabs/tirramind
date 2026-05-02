"""TirraMind — Regime Gate (Phase 49b)

Shared helper that reads the latest convergence detection outputs from
PipelineStore and exposes a simple API for the 5 system components that
need to adapt their behaviour based on the current market regime.

Five consumers wired in Phase 49b:
    1. GNN Inference DAG      — force retrain when changepoint_posterior > threshold
    2. RL Training DAG        — raise SAC target entropy in high-changepoint regimes
    3. World Model Update DAG — apply prior decay when regime_label changes
    4. Feature Generation DAG — scale GNN feature trust by stability duration
    5. (This module itself)   — the shared helper

Data sources (read-only, deterministic):
    - ``convergence.*`` signals in the signals table (emitted by
      convergence_detection DAG) for changepoint_posterior and regime_label.
    - ``regime.macro`` beliefs in the beliefs table for MAP regime label.

Design:
    All functions are pure reads — no writes, no LLM calls, no side effects.
    Results are cached within a single call chain by passing a
    :class:`RegimeContext` object; the PipelineStore is not queried twice
    per DAG run.

References:
    - Task: tasks/active/quant_training_ground.md (Phase 49b entry)
    - Convergence signals schema: agent/convergence/signals.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Lookback window for querying convergence signals (7 days)
_LOOKBACK_SECONDS = 7 * 86_400

# Default regime when no data is available
_DEFAULT_REGIME = "expansion"

# Convergence signal name prefix
_CONV_PREFIX = "convergence."

# Belief variable for macro regime
_REGIME_BELIEF = "regime.macro"

# Entropy scale boost applied to SAC in high-changepoint regimes.
# Original target_entropy_scale = -0.5. In high-changepoint mode we
# raise it to -0.3 (less negative → higher target entropy → more exploration).
# Mathematical justification: in regime shifts, the policy distribution
# should be broader (higher entropy) to hedge against model uncertainty.
_ENTROPY_SCALE_NORMAL: float = -0.5
_ENTROPY_SCALE_HIGH_CHANGEPOINT: float = -0.3

# Prior decay factor applied to world model belief update weights when
# regime_label changes.  A value of 0.8 means beliefs from the previous
# regime are down-weighted by 20% before the Kalman predict step.
# This softens priors without resetting them — equivalent to increasing
# process noise Q by 1/decay_factor in the transition step.
_PRIOR_DECAY_REGIME_CHANGE: float = 0.8
_PRIOR_DECAY_STABLE: float = 1.0

# Feature trust scale when stability duration is short (< 3 days)
_TRUST_SCALE_UNSTABLE: float = 0.7
_TRUST_SCALE_STABLE: float = 1.0
_STABILITY_THRESHOLD_DAYS: float = 3.0


@dataclass
class RegimeContext:
    """Snapshot of the current regime as seen by the pipeline.

    Populated once per DAG run by :func:`get_current_regime` and passed
    to the 5 consumer functions so the store is only queried once.

    Attributes
    ----------
    regime_label : str
        MAP regime label from the ``regime.macro`` belief (e.g.
        ``"expansion"``, ``"contraction"``, ``"crisis"``).
        Defaults to ``"expansion"`` when no belief is stored.
    changepoint_posterior : float
        Probability that a structural break occurred in the most recent
        detection window.  Derived from the highest-value convergence
        signal in the last 7 days.  In ``[0, 1]``.
    stability_duration_days : float
        How long the current regime label has been stable (days since
        the last convergence event that changed the regime).
    regime_changed : bool
        True when the latest ``regime.macro`` belief label differs from
        the previous belief stored in the same window.
    as_of : float
        Unix timestamp when this context was computed.
    """

    regime_label: str = _DEFAULT_REGIME
    changepoint_posterior: float = 0.0
    stability_duration_days: float = 30.0
    regime_changed: bool = False
    as_of: float = field(default_factory=time.time)


def get_current_regime(
    store: PipelineStore,
    lookback_seconds: float = _LOOKBACK_SECONDS,
    as_of: float | None = None,
) -> RegimeContext:
    """Query the latest regime state from the pipeline store.

    Reads:
    1. The most recent ``regime.macro`` beliefs (last 2) to detect a
       label change.
    2. The most recent ``convergence.*`` signals to derive
       ``changepoint_posterior``.
    3. Infers ``stability_duration_days`` from the gap since the most
       recent convergence signal with value > 0.5.

    Returns a :class:`RegimeContext` with sensible defaults when no data
    is available (expansion, 0.0 changepoint, 30d stability).

    Parameters
    ----------
    store :
        Open PipelineStore instance.
    lookback_seconds :
        How far back to look for convergence signals.
    as_of :
        Reference time (unix epoch). Defaults to now.
    """
    if as_of is None:
        as_of = time.time()

    ctx = RegimeContext(as_of=as_of)

    # ── 1. Latest regime.macro belief ────────────────────────
    try:
        belief_rows = store.query_beliefs(
            _REGIME_BELIEF,
            since=as_of - lookback_seconds,
            until=as_of,
            limit=2,
        )
        if belief_rows:
            latest = belief_rows[0]
            probs = latest.get("probabilities")
            if probs and isinstance(probs, dict):
                ctx.regime_label = max(probs, key=probs.get)

            # Detect regime change: compare top-2 beliefs
            if len(belief_rows) >= 2:
                prev = belief_rows[1]
                prev_probs = prev.get("probabilities")
                if prev_probs and isinstance(prev_probs, dict):
                    prev_label = max(prev_probs, key=prev_probs.get)
                    ctx.regime_changed = prev_label != ctx.regime_label
    except Exception as exc:
        log.warning("RegimeGate: could not load regime.macro beliefs: %s", exc)

    # ── 2. Changepoint posterior from convergence signals ─────
    try:
        # Query recent convergence signals and take the max value as
        # the changepoint posterior. Convergence signals are in [0, 1].
        since = as_of - lookback_seconds
        # Use a prefix query by querying all signals and filtering
        # (PipelineStore.query_signals requires exact name match)
        # We use the stored signal values from the past week.
        recent_convergence: list[dict[str, Any]] = []
        # Try a few known signal name patterns to find convergence events
        for pattern_signal in [
            "convergence.geopolitical_financial_stress",
            "convergence.commodity_supply_shock",
            "convergence.sovereign_stress",
            "convergence.risk_off_cascade",
            "convergence.unknown_pattern",
        ]:
            try:
                rows = store.query_signals(pattern_signal, since=since, until=as_of, limit=5)
                recent_convergence.extend(rows)
            except Exception:
                pass

        if recent_convergence:
            # Highest convergence score in the window = changepoint posterior
            ctx.changepoint_posterior = max(float(r.get("value", 0.0)) for r in recent_convergence)

            # Stability = time since the most recent high-confidence convergence event
            high_events = [r for r in recent_convergence if float(r.get("value", 0.0)) > 0.5]
            if high_events:
                latest_event_ts = max(r.get("computed_at", 0.0) for r in high_events)
                elapsed = as_of - latest_event_ts
                ctx.stability_duration_days = max(0.0, elapsed / 86_400)
            else:
                ctx.stability_duration_days = lookback_seconds / 86_400

    except Exception as exc:
        log.warning("RegimeGate: could not load convergence signals: %s", exc)

    log.debug(
        "RegimeContext: regime=%s changepoint_posterior=%.3f stability=%.1fd regime_changed=%s",
        ctx.regime_label,
        ctx.changepoint_posterior,
        ctx.stability_duration_days,
        ctx.regime_changed,
    )
    return ctx


def is_high_changepoint(
    store: PipelineStore,
    threshold: float = 0.9,
    *,
    ctx: RegimeContext | None = None,
) -> bool:
    """Return True when the changepoint posterior exceeds *threshold*.

    If *ctx* is provided, it is used directly (avoids a second store
    query).  Otherwise, :func:`get_current_regime` is called.

    Used by the GNN Inference DAG to decide whether to force a full
    retrain regardless of whether a checkpoint exists.
    """
    if ctx is None:
        ctx = get_current_regime(store)
    return ctx.changepoint_posterior >= threshold


def sac_entropy_scale(ctx: RegimeContext) -> float:
    """Return the SAC target_entropy_scale appropriate for the current regime.

    Normal (stable): -0.5  (Haarnoja 2018b heuristic for continuous action spaces)
    High changepoint: -0.3 (less negative → higher target entropy → more exploration)

    The scale is used as: H_target = scale × dim(A).  More negative →
    lower target entropy → more deterministic policy.  During regime shifts
    we want higher entropy to hedge against model uncertainty.

    Consumer: ``rl_training.py`` → SACTrainer.set_regime_entropy_scale()
    """
    if ctx.changepoint_posterior >= 0.5:
        return _ENTROPY_SCALE_HIGH_CHANGEPOINT
    return _ENTROPY_SCALE_NORMAL


def world_model_prior_decay(ctx: RegimeContext) -> float:
    """Return the prior decay factor for the world model update step.

    When the regime label has changed since the previous belief:
        decay = 0.8  (beliefs from previous regime down-weighted 20%)
    When stable:
        decay = 1.0  (no decay — trust current beliefs fully)

    Consumer: ``world_model_update.py`` → applied before belief propagation
    as a multiplicative weight on prior probabilities.

    Mathematical interpretation: equivalent to multiplying prior
    probability masses by *decay*, then renormalizing.  This softens
    (but does not reset) the prior when evidence suggests a regime shift,
    without requiring a hard reset that would discard all accumulated belief.
    """
    if ctx.regime_changed:
        return _PRIOR_DECAY_REGIME_CHANGE
    return _PRIOR_DECAY_STABLE


def feature_trust_scale(ctx: RegimeContext) -> float:
    """Return the GNN feature trust scale based on regime stability.

    When the current regime has been stable for < 3 days:
        trust = 0.7  (GNN embeddings may lag the new regime — down-weight)
    When stable ≥ 3 days:
        trust = 1.0  (GNN features are reliable)

    Consumer: ``feature_generation.py`` → applied as a multiplicative
    scalar to all GNN EngineeredFeature values before world model injection.

    Rationale: after a regime shift, GNN weights (which are updated
    incrementally via EWC) may not yet reflect the new regime structure.
    The trust scale discounts GNN features proportionally until the
    system has had time to re-learn the new regime's entity relationships.
    """
    if ctx.stability_duration_days < _STABILITY_THRESHOLD_DAYS:
        return _TRUST_SCALE_UNSTABLE
    return _TRUST_SCALE_STABLE
