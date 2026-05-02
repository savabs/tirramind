"""
TirraMind — GNN Downstream Alignment Module (Phase 49)

Measures how much world-model belief update is attributable to GNN-derived
features, grouped by GNN entity type.  This produces a per-entity-type
alignment signal that feeds back into GNN training as a loss weight:

    - Entity types whose features sharpen world-model beliefs (high KL delta)
      already have good alignment → down-weight their CE training loss.
    - Entity types whose features produce flat or diffuse belief updates
      (low KL delta) need more signal → up-weight their CE training loss.

The alignment loop:
    DAG update → belief_delta → store alignment signal
    → GNN training loads weights → weighted obs_type CE loss.

Mathematical basis:
    Categorical KL divergence (Kullback & Leibler, 1951):
        KL(p_after || p_before) = Σ_i p_after[i] · log(p_after[i] / p_before[i])
    Measured per world-model variable.  Positive = belief sharpened.
    Equivalently, this is the cross-entropy H(p_after, p_before) − H(p_after),
    so it measures the extra bits needed to encode p_after using p_before as the
    reference.  A large value means the update provided substantial new information.

    Gaussian entropy reduction (for Kalman state beliefs):
        Δentropy = 0.5 · log(var_before / var_after)
    Positive = variance reduced (uncertainty decreased by absorbing observations).
    Zero = no update occurred (NaN observations → filter returned without updating).

    Entity-type aggregation:
        The GNN features observed indices 6–16 in the Kalman observation vector
        are grouped by entity type (see _GNN_FEATURE_ENTITY_TYPES below).
        delta[entity_type] = mean of all variable deltas for that group.
        Unobserved / non-GNN variables are excluded from this aggregation
        (they are stored individually but not used in entity-level weighting).

    Alignment weight (inverse-softplus):
        weight[entity_type] = 1 / (1 + max(delta[entity_type], 0))
    Bounded in (0, 1].  High positive delta → weight near 0 (already aligned).
    Zero or negative delta → weight = 1.0 (needs more gradient signal).
    Softly bounded so weights are never exactly zero (avoids loss starvation).

References:
    Kullback, S. & Leibler, R.A. (1951). "On Information and Sufficiency".
    Annals of Mathematical Statistics. 22(1): 79–86.
    Sarkka, S. (2013). "Bayesian Filtering and Smoothing", Ch. 4.
    Spec: docs/specs/phase49_gnn_alignment_spec.md
"""

from __future__ import annotations

import logging
import math
import time

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────

_ALIGNMENT_SOURCE = "gnn_alignment_signal"
_EPS = 1e-12  # numerical floor for log computations
_DEFAULT_WEIGHT = 1.0  # returned for entity types with no alignment history

# Maps GNN entity-type tags to the world-model continuous-state variable names
# that those features influence via the Kalman observation vector.
# GNN features occupy obs-vector indices 6-16 (see _FEATURE_TO_OBS_INDEX in
# world_model_update.py).  All GNN features contribute to every Kalman state
# via the H matrix, so we aggregate the continuous-state belief deltas and
# attribute them proportionally to each entity type that has a feature in
# the observation vector.
# "cross" represents the cross-entity aggregation feature (index 16).
_GNN_ENTITY_TYPES: frozenset[str] = frozenset({"person", "company", "wallet", "country", "vessel", "cross"})

# World-model variable names that are driven by GNN features (continuous states).
# These are the Kalman-produced belief variables (vs DAG categorical variables).
# Prefix check is used: any variable_name matching one of these prefixes counts.
_KALMAN_STATE_PREFIXES: tuple[str, ...] = ("state.",)


# ── Core functions ─────────────────────────────────────────────


def compute_belief_log_likelihood_delta(
    beliefs_before: list[dict],
    beliefs_after: list[dict],
) -> dict[str, float]:
    """Compute per-variable belief sharpening delta.

    Compares each world-model belief before and after a world-model update
    cycle.  Returns a signed score per variable:
        - Positive: belief sharpened / moved toward data (good alignment)
        - Negative: belief diffused / moved away from data
        - Zero: no change or insufficient data to compare

    For categorical beliefs:
        delta = KL(p_after || p_before) = Σ p_after[i] · log(p_after[i]/p_before[i])
        This equals zero iff p_after == p_before and is maximized when p_after
        is a point mass on the mode that p_before assigned very low probability.

    For Gaussian beliefs:
        delta = 0.5 · log(max(var_before, ε) / max(var_after, ε))
        Positive when variance decreased (filter updated; evidence absorbed).
        Zero when var_after == var_before (no observations available).

    Args:
        beliefs_before: List of belief dicts from ``store.query_all_latest_beliefs()``
            or equivalent.  Each dict must have keys: ``variable_name``,
            ``dist_type``, and either ``probabilities`` (categorical) or
            ``mean`` + ``variance`` (gaussian).
        beliefs_after: List of fresh ``BeliefState`` records returned by
            ``WorldModel.update()`` — converted to dicts via
            ``BeliefState.to_dict()`` before passing here.

    Returns:
        Dict mapping ``variable_name`` → delta scalar.
        Only variables present in both before/after are included.
        Variables with parsing errors or type mismatches are skipped.
    """
    # Index beliefs_before by variable_name (take the most recent if duplicates)
    before_by_name: dict[str, dict] = {}
    for b in beliefs_before:
        name = b.get("variable_name")
        if name:
            before_by_name[name] = b

    deltas: dict[str, float] = {}

    for b_after in beliefs_after:
        name = b_after.get("variable_name")
        if not name:
            continue
        b_before = before_by_name.get(name)
        if b_before is None:
            # No prior belief stored — skip (first run or new variable)
            continue

        dist_type = b_after.get("dist_type", "")

        try:
            if dist_type == "categorical":
                delta = _kl_categorical(b_before, b_after)
            elif dist_type == "gaussian":
                delta = _entropy_reduction_gaussian(b_before, b_after)
            else:
                continue  # unsupported dist type
        except Exception as exc:
            log.debug("Alignment delta computation failed for '%s': %s", name, exc)
            continue

        deltas[name] = delta

    return deltas


def store_entity_alignment(
    store,  # PipelineStore — not imported to avoid circular dependency
    variable_deltas: dict[str, float],
    as_of: float | None = None,
) -> None:
    """Persist belief-delta signals to the pipeline store.

    Two kinds of signals are written:
      1. Per-variable: ``gnn_alignment_signal.<variable_name>`` = delta
      2. Per-entity-type aggregates (for Kalman state variables):
         ``gnn_alignment_signal.entity.<entity_type>`` = mean delta

    Entity-type aggregation uses all Kalman state variable deltas
    (variables whose name starts with ``"state."``).  Since all GNN entity
    types contribute observations to the same Kalman state vector, the
    *aggregate* Kalman delta is stored once for each entity type as a
    uniform signal.  Entity-type differentiation at L3 granularity requires
    counterfactual filtering (out of scope for Phase 49).

    Args:
        store: PipelineStore instance (open).
        variable_deltas: Output of ``compute_belief_log_likelihood_delta``.
        as_of: Unix epoch for metadata.  Defaults to now.
    """
    if as_of is None:
        as_of = time.time()

    if not variable_deltas:
        return

    # ── 1. Store per-variable signals ──────────────────────────────────────
    for var_name, delta in variable_deltas.items():
        try:
            store.store_signal(
                f"{_ALIGNMENT_SOURCE}.{var_name}",
                float(delta),
                metadata={"as_of": as_of},
            )
        except Exception as exc:
            log.debug("Failed to store alignment signal for '%s': %s", var_name, exc)

    # ── 2. Aggregate Kalman state deltas → per-entity-type signals ──────────
    kalman_deltas = [v for k, v in variable_deltas.items() if any(k.startswith(pfx) for pfx in _KALMAN_STATE_PREFIXES)]
    if kalman_deltas:
        mean_kalman_delta = sum(kalman_deltas) / len(kalman_deltas)
        for entity_type in _GNN_ENTITY_TYPES:
            try:
                store.store_signal(
                    f"{_ALIGNMENT_SOURCE}.entity.{entity_type}",
                    float(mean_kalman_delta),
                    metadata={"as_of": as_of, "n_vars": len(kalman_deltas)},
                )
            except Exception as exc:
                log.debug(
                    "Failed to store entity alignment signal for '%s': %s",
                    entity_type,
                    exc,
                )

    log.debug(
        "Stored %d alignment signals (%.0f Kalman → %d entity types).",
        len(variable_deltas),
        len(kalman_deltas),
        len(_GNN_ENTITY_TYPES),
    )


def load_alignment_weights(
    store,  # PipelineStore — not imported to avoid circular dependency
    entity_types: list[str],
    lookback_days: float = 7.0,
) -> dict[str, float] | None:
    """Load per-entity-type alignment weights for GNN training.

    Queries recent ``gnn_alignment_signal.entity.<entity_type>`` signals
    and converts them to training loss weights.

    Weight formula:
        weight[entity_type] = 1 / (1 + max(mean_delta, 0))
    Bounded in (0, 1].  High delta → weight near 0 (already aligned; reduce
    loss emphasis).  Zero or negative delta → weight = 1.0 (needs signal).

    The weights are typically used to scale the per-entity-type contribution
    to the obs_type CE loss during GNN training.

    Args:
        store: PipelineStore instance (open).
        entity_types: List of entity type strings to load weights for.
            Typically all entity types in the GNN (e.g. from IDMap).
        lookback_days: How far back to look for alignment signals.

    Returns:
        Dict mapping entity_type → weight float.
        Returns None if no alignment signals are stored yet
        (first run, or signals have expired).  Callers should treat
        None as "use uniform weights" (no alignment information).
    """
    since = time.time() - lookback_days * 86400.0
    weights: dict[str, float] = {}

    n_found = 0
    for entity_type in entity_types:
        signal_name = f"{_ALIGNMENT_SOURCE}.entity.{entity_type}"
        try:
            rows = store.query_signals(signal_name, since=since, limit=50)
        except Exception as exc:
            log.debug("Failed to load alignment signal for '%s': %s", entity_type, exc)
            continue

        if not rows:
            continue

        # Mean delta over the lookback window
        vals = [r["value"] for r in rows if r.get("value") is not None]
        if not vals:
            continue

        mean_delta = sum(vals) / len(vals)
        weight = 1.0 / (1.0 + max(mean_delta, 0.0))
        weights[entity_type] = weight
        n_found += 1

    if n_found == 0:
        log.debug("No alignment signals found within lookback — returning None.")
        return None

    # Fill missing entity types with the default weight
    for entity_type in entity_types:
        if entity_type not in weights:
            weights[entity_type] = _DEFAULT_WEIGHT

    log.debug(
        "Loaded alignment weights for %d/%d entity types (lookback=%.0fd, min=%.3f, max=%.3f).",
        n_found,
        len(entity_types),
        lookback_days,
        min(weights.values()),
        max(weights.values()),
    )
    return weights


# ── Private helpers ────────────────────────────────────────────


def _kl_categorical(b_before: dict, b_after: dict) -> float:
    """KL(p_after || p_before) for categorical beliefs.

    Returns 0.0 if probabilities are unavailable or empty in either belief.
    Returns the sum Σ p_after[i] · log(p_after[i] / max(p_before[i], ε))
    over all states present in p_after.
    """
    p_before: dict[str, float] | None = b_before.get("probabilities")
    p_after: dict[str, float] | None = b_after.get("probabilities")

    if not p_before or not p_after:
        return 0.0

    kl = 0.0
    for state, q in p_after.items():
        if q <= 0.0:
            continue  # zero mass in after → contributes 0 to KL
        p = max(p_before.get(state, 0.0), _EPS)
        kl += q * math.log(q / p)

    return kl


def _entropy_reduction_gaussian(b_before: dict, b_after: dict) -> float:
    """Entropy reduction for Gaussian beliefs: 0.5 · log(var_before / var_after).

    Returns 0.0 if either variance is None, non-positive, or identical.
    """
    var_before = b_before.get("variance")
    var_after = b_after.get("variance")

    if var_before is None or var_after is None:
        return 0.0
    if var_before <= 0.0 or var_after <= 0.0:
        return 0.0

    # Clamp ratio to avoid extreme values from near-zero variances
    ratio = max(var_before, _EPS) / max(var_after, _EPS)
    return 0.5 * math.log(ratio)
