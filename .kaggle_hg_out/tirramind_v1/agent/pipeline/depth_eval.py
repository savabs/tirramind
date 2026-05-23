"""
TirraMind — Depth Evaluation Module

Measures whether deeper data (L2/L3) adds predictive signal beyond
shallower aggregates (L1). Three metrics:

1. Conditional Mutual Information gain (KSG estimator via sklearn)
2. Belief update magnitude (KL divergence via scipy)
3. Walk-forward Sharpe delta (deferred until full pipeline operational)

Trusted sources:
- MI estimation: Kraskov, Stögbauer, Grassberger (2004). "Estimating Mutual
  Information." Physical Review E 69(6). KSG uses k-nearest-neighbor distances
  to avoid binning artifacts. sklearn.feature_selection uses KSG internally.
- KL divergence: scipy.stats.entropy implements KL(p||q) correctly,
  with the convention 0 * log(0/q) = 0.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Minimum sample size for MI estimation to be meaningful.
_MIN_SAMPLES = 30


def compute_conditional_mi(
    observations_new: np.ndarray,
    observations_existing: np.ndarray | None,
    targets: np.ndarray,
    *,
    discrete_target: bool = False,
    random_state: int = 42,
) -> float:
    """Compute conditional mutual information of new observations given existing ones.

    Estimates I(X_new; Y | X_existing) by computing:
        I([X_existing, X_new]; Y) - I(X_existing; Y)

    When observations_existing is None, returns I(X_new; Y) directly.

    Args:
        observations_new: Array of shape (n_samples,) or (n_samples, n_features)
            for the deeper data.
        observations_existing: Array of shape (n_samples,) or (n_samples, n_features)
            for the shallower data. None if this is the first depth level.
        targets: Array of shape (n_samples,) — the prediction target.
        discrete_target: If True, uses mutual_info_classif (discrete Y).
            If False, uses mutual_info_regression (continuous Y).
        random_state: Random seed for reproducibility.

    Returns:
        Conditional MI in nats. Returns float('nan') if insufficient samples
        or all-constant columns detected.

    Raises:
        ValueError: If array lengths don't match.
    """
    obs_new = np.asarray(observations_new, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)

    if obs_new.ndim == 1:
        obs_new = obs_new.reshape(-1, 1)

    n = obs_new.shape[0]
    if tgt.shape[0] != n:
        raise ValueError(f"observations_new has {n} samples but targets has {tgt.shape[0]}")
    if n < _MIN_SAMPLES:
        log.warning("Insufficient samples for MI: %d < %d", n, _MIN_SAMPLES)
        return float("nan")

    # Filter non-finite values
    mask = np.isfinite(obs_new).all(axis=1) & np.isfinite(tgt)
    if observations_existing is not None:
        obs_ex = np.asarray(observations_existing, dtype=np.float64)
        if obs_ex.ndim == 1:
            obs_ex = obs_ex.reshape(-1, 1)
        if obs_ex.shape[0] != n:
            raise ValueError(f"observations_existing has {obs_ex.shape[0]} samples but observations_new has {n}")
        mask &= np.isfinite(obs_ex).all(axis=1)
        obs_ex = obs_ex[mask]
    else:
        obs_ex = None

    obs_new = obs_new[mask]
    tgt = tgt[mask]

    if len(tgt) < _MIN_SAMPLES:
        log.warning("Insufficient finite samples for MI: %d < %d", len(tgt), _MIN_SAMPLES)
        return float("nan")

    mi_func = mutual_info_classif if discrete_target else mutual_info_regression

    if obs_ex is not None:
        # I([existing, new]; Y)
        combined = np.hstack([obs_ex, obs_new])
        mi_combined = mi_func(combined, tgt, random_state=random_state)
        mi_combined_total = float(np.sum(mi_combined))

        # I(existing; Y)
        mi_existing = mi_func(obs_ex, tgt, random_state=random_state)
        mi_existing_total = float(np.sum(mi_existing))

        # Conditional MI = I(combined; Y) - I(existing; Y)
        cmi = mi_combined_total - mi_existing_total
        # MI can be slightly negative due to estimation noise — clamp to 0
        return max(0.0, cmi)
    else:
        mi_vals = mi_func(obs_new, tgt, random_state=random_state)
        return float(np.sum(mi_vals))


def compute_kl_divergence(
    prior_probs: dict[str, float],
    posterior_probs: dict[str, float],
) -> float:
    """Compute KL(posterior || prior) for discrete probability distributions.

    Both distributions must have the same keys. Uses scipy.stats.entropy
    which handles the convention 0 * log(0/q) = 0.

    Args:
        prior_probs: Dict mapping state names to prior probabilities.
        posterior_probs: Dict mapping state names to posterior probabilities.

    Returns:
        KL divergence in nats (non-negative). Returns 0.0 if distributions
        are identical.

    Raises:
        ValueError: If distributions have different keys or don't sum to ~1.
    """
    if set(prior_probs.keys()) != set(posterior_probs.keys()):
        raise ValueError(f"Distribution keys differ: {set(prior_probs.keys())} vs {set(posterior_probs.keys())}")

    keys = sorted(prior_probs.keys())
    p = np.array([posterior_probs[k] for k in keys], dtype=np.float64)
    q = np.array([prior_probs[k] for k in keys], dtype=np.float64)

    # Validate probabilities sum to ~1
    for name, arr in [("prior", q), ("posterior", p)]:
        s = arr.sum()
        if not math.isclose(s, 1.0, rel_tol=1e-4):
            raise ValueError(f"{name} probabilities sum to {s}, not 1.0")

    return float(scipy_entropy(p, q))


def measure_belief_shift(
    store: PipelineStore,
    variable_name: str,
    before_version: int,
    after_version: int,
) -> float | None:
    """Measure KL divergence between two belief versions in the store.

    Loads the latest belief for each version and computes
    KL(after || before) on their probability distributions.

    Returns None if either belief is not found or not discrete
    (no probabilities field).
    """
    before = store.get_latest_belief(variable_name, version=before_version)
    after = store.get_latest_belief(variable_name, version=after_version)

    if before is None or after is None:
        log.warning(
            "Cannot measure belief shift: missing belief for %s (v%d→v%d)",
            variable_name,
            before_version,
            after_version,
        )
        return None

    prior_probs = before.get("probabilities")
    posterior_probs = after.get("probabilities")

    if prior_probs is None or posterior_probs is None:
        log.warning(
            "Cannot compute KL divergence: belief %s lacks discrete probabilities",
            variable_name,
        )
        return None

    try:
        return compute_kl_divergence(prior_probs, posterior_probs)
    except ValueError as e:
        log.warning("KL divergence computation failed for %s: %s", variable_name, e)
        return None


def run_depth_evaluation(
    store: PipelineStore,
    tool_name: str,
    depth_level: int,
    target_variable: str,
    observations_new: np.ndarray,
    targets: np.ndarray,
    *,
    observations_existing: np.ndarray | None = None,
    discrete_target: bool = False,
    belief_before_version: int | None = None,
    belief_after_version: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a complete depth evaluation and store the result.

    Computes MI gain and optionally KL divergence, stores the result
    in depth_evaluations, and returns the evaluation record.

    Args:
        store: PipelineStore instance.
        tool_name: Name of the tool being evaluated.
        depth_level: Depth level being tested (2 for L2, 3 for L3).
        target_variable: Name of the prediction target.
        observations_new: Data from the deeper level.
        targets: Target variable values.
        observations_existing: Data from the shallower level (None for L1).
        discrete_target: Whether target is discrete.
        belief_before_version: Version of belief before depth evidence.
        belief_after_version: Version of belief after depth evidence.
        metadata: Additional metadata to store.

    Returns:
        Dict with keys: mi_gain, kl_divergence, sample_size, row_id.
    """
    mi_gain = compute_conditional_mi(
        observations_new,
        observations_existing,
        targets,
        discrete_target=discrete_target,
    )

    kl_div: float | None = None
    if belief_before_version is not None and belief_after_version is not None:
        kl_div = measure_belief_shift(store, target_variable, belief_before_version, belief_after_version)

    sample_size = len(targets)
    row_id = store.store_depth_evaluation(
        tool_name=tool_name,
        depth_level=depth_level,
        target_variable=target_variable,
        sample_size=sample_size,
        mi_gain=mi_gain if not math.isnan(mi_gain) else None,
        kl_divergence=kl_div,
        metadata=metadata,
    )

    result = {
        "mi_gain": mi_gain,
        "kl_divergence": kl_div,
        "sample_size": sample_size,
        "row_id": row_id,
    }
    log.info(
        "Depth evaluation: tool=%s depth=%d target=%s mi=%.4f kl=%s n=%d",
        tool_name,
        depth_level,
        target_variable,
        mi_gain,
        kl_div,
        sample_size,
    )
    return result
