"""
TirraMind — Wasserstein Distributionally Robust Portfolio (Idea 16)

Extends the BL+HRP portfolio layer with Wasserstein-ball uncertainty sets.

Problem
-------
The existing Black-Litterman + HRP portfolio (Idea 11) optimises under
the assumption that the empirical return distribution is the true one.
In practice:
  - The empirical covariance is estimated from T ≈ 60–252 samples.
  - Market regimes shift: the distribution tomorrow ≠ distribution today.
  - Standard MVO / HRP are fragile — small perturbations to inputs cause
    large weight swings (the "error maximisation" problem of Michaud 1989).

Solution — Wasserstein Distributionally Robust Optimisation
-------------------------------------------------------------
Blanchet, Chen & Zhou (2022) show that the mean-variance problem:

    min_w  { w^T μ̂ - (λ/2) w^T Σ̂ w }

has a tractable robust counterpart over the Wasserstein ball B_ε(P̂):

    min_w  max_{Q ∈ B_ε(P̂)} E_Q[ -w^T r ]

whose solution is equivalent to standard MVO with:

    Σ_robust = Σ̂ + ε * I          (inflate covariance by uncertainty radius)
    μ_robust = μ̂                   (mean unchanged — robustness enters via Σ)

This inflates eigenvalues uniformly, shrinking the effective bet size on
all factors proportionally — exactly the regularisation institutional
risk managers apply intuitively.

Calibrating ε
-------------
Two methods are provided:

1. bootstrap_epsilon(returns, n_bootstrap=200)
   Compute ε = median Wasserstein-1 distance between bootstrap samples.
   The W1 distance between 1D samples is:
       W1(P, Q) = mean |sort(P) - sort(Q)|   (Vallender 1974)
   Averaged across all N instruments, this gives the data-driven ε.

2. fixed ε passed directly (e.g. ε=0.01 for a 1% uncertainty radius).

References
----------
Blanchet, J., Chen, L., & Zhou, X.Y. (2022). "Distributionally Robust
  Mean-Variance Portfolio Selection with Wasserstein Distances."
  Management Science, 68(2): 1166–1188.
  Direct source for the covariance-inflation closed form.

Kuhn, D., Esfahani, P.M., Nguyen, V.A., & Shafieezadeh-Abadeh, S.
  (2019). "Wasserstein Distributionally Robust Optimization: Theory and
  Applications in Machine Learning." Proc. of INFORMS TutORials.
  General DRO framework via Wasserstein balls.

Michaud, R.O. (1989). "The Markowitz Optimization Enigma: Is Optimized
  Optimal?" Financial Analysts Journal, 45(1): 31–42.
  Documents the input sensitivity problem motivating robustness.

Vallender, S.S. (1974). "Calculation of the Wasserstein Distance Between
  Probability Distributions on the Line." Theory of Probability and
  its Applications, 18(4): 784–786.
  Analytical W1 for empirical distributions = sorted-array L1.

CPU Safety
----------
- All operations are pure NumPy/SciPy.
- Bootstrap capped at n_bootstrap=200.
- Instrument cap: max_instruments=200 (matches portfolio constructor).
- No torch used.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RobustPortfolioWeights:
    """Output of WassersteinRobustPortfolio.build_weights().

    Attributes
    ----------
    weights      : dict[entity_id, weight].  Weights sum to 1.0.
    epsilon      : Wasserstein uncertainty radius used.
    robust_cov   : Inflated covariance trace (scalar diagnostic).
    standard_cov : Original (non-inflated) covariance trace.
    n_assets     : Number of assets in the portfolio.
    built_at     : Unix timestamp.
    method       : "wasserstein_robust" always.
    """

    weights: dict[str, float]
    epsilon: float
    robust_cov: float
    standard_cov: float
    n_assets: int
    built_at: float
    method: str = "wasserstein_robust"


# ═══════════════════════════════════════════════════════════════
# Wasserstein distance helpers
# ═══════════════════════════════════════════════════════════════


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute W1 between two 1D empirical distributions.

    W1(P, Q) = mean |sort(P) - interp(sort(Q), len(P))|

    The analytical formula for 1D empirical distributions is the
    L1 norm between the sorted arrays after interpolating to equal
    length (Vallender 1974).

    Args:
        a: 1D array of samples from distribution P.
        b: 1D array of samples from distribution Q.

    Returns:
        W1 distance (non-negative scalar).
    """
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    if len(a_sorted) == len(b_sorted):
        return float(np.mean(np.abs(a_sorted - b_sorted)))
    # Interpolate b to length of a
    idx = np.linspace(0, len(b_sorted) - 1, len(a_sorted))
    b_interp = np.interp(idx, np.arange(len(b_sorted)), b_sorted)
    return float(np.mean(np.abs(a_sorted - b_interp)))


def bootstrap_epsilon(
    returns: np.ndarray,
    n_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Calibrate ε from bootstrap Wasserstein distances.

    For each instrument, compute the median W1 distance between
    bootstrap resamplings of its return distribution.  The final ε
    is the mean across instruments.

    Args:
        returns     : (T, N) return matrix — T days × N instruments.
        n_bootstrap : Number of bootstrap pairs.  Default 200.
        rng         : NumPy random generator.  None = default.

    Returns:
        Calibrated epsilon (non-negative float).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T, N = returns.shape
    if T < 4:
        return 0.0

    half = T // 2
    eps_per_instrument = []

    for j in range(N):
        col = returns[:, j]
        dists = []
        for _ in range(min(n_bootstrap, 200)):
            a = rng.choice(col, size=half, replace=True)
            b = rng.choice(col, size=half, replace=True)
            dists.append(wasserstein_1d(a, b))
        eps_per_instrument.append(float(np.median(dists)))

    return float(np.mean(eps_per_instrument)) if eps_per_instrument else 0.0


# ═══════════════════════════════════════════════════════════════
# Robust covariance
# ═══════════════════════════════════════════════════════════════


def robust_covariance(
    returns: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Compute the epsilon-inflated robust covariance matrix.

    Σ_robust = Σ̂ + ε * I

    Per Blanchet et al. (2022), this is the closed-form robust
    covariance for the Wasserstein-ball uncertainty set B_ε(P̂).
    The identity inflation shrinks all factor bets by ε uniformly.

    Args:
        returns: (T, N) return matrix.
        epsilon: Wasserstein uncertainty radius.

    Returns:
        Robust covariance (N × N positive semi-definite).
    """
    cov = np.cov(returns.T) if returns.shape[0] > 1 else np.eye(returns.shape[1])
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    return cov + epsilon * np.eye(cov.shape[0])


# ═══════════════════════════════════════════════════════════════
# Minimum-variance weights from robust covariance
# ═══════════════════════════════════════════════════════════════


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio weights.

    Analytical solution: w* = Σ⁻¹ 1 / (1^T Σ⁻¹ 1).
    Falls back to equal weights on singular/ill-conditioned matrix.

    Args:
        cov: (N, N) covariance matrix.

    Returns:
        Weight vector length N summing to 1.
    """
    N = cov.shape[0]
    ones = np.ones(N)
    try:
        cov_inv = scipy.linalg.inv(cov + 1e-8 * np.eye(N))
        raw = cov_inv @ ones
        total = ones @ raw
        if total <= 0 or not math.isfinite(total):
            raise ValueError("degenerate covariance")
        w = raw / total
        # Project to simplex (non-negative + sum = 1)
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s < 1e-12:
            return np.full(N, 1.0 / N)
        return w / s
    except Exception:
        log.warning("min_variance_weights: fallback to equal weights.")
        return np.full(N, 1.0 / N)


def _blend_with_views(
    cov_robust: np.ndarray,
    mean_views: np.ndarray,
    delta: float = 2.5,
) -> np.ndarray:
    """Simple mean-variance tilt on top of robust covariance.

    Maximise: w^T μ - (delta/2) w^T Σ w
    Solution: w* ∝ Σ⁻¹ μ  (then normalised to long-only simplex).
    """
    N = cov_robust.shape[0]
    try:
        cov_inv = scipy.linalg.inv(cov_robust + 1e-8 * np.eye(N))
        raw = cov_inv @ mean_views
        # Project to long-only simplex
        raw = np.maximum(raw, 0.0)
        s = raw.sum()
        if s < 1e-12:
            return _min_variance_weights(cov_robust)
        return raw / s
    except Exception:
        return _min_variance_weights(cov_robust)


# ═══════════════════════════════════════════════════════════════
# WassersteinRobustPortfolio
# ═══════════════════════════════════════════════════════════════


class WassersteinRobustPortfolio:
    """Wasserstein distributionally robust portfolio constructor.

    Produces portfolio weights that are optimal under the worst-case
    return distribution within a Wasserstein ball of radius ε around
    the empirical distribution.

    The result is a mean-variance portfolio with epsilon-inflated
    covariance — more conservative than standard MVO, less sensitive
    to estimation error, and robust to moderate distribution shifts.

    Parameters
    ----------
    epsilon         : Wasserstein uncertainty radius.  0.0 = standard
                      MVO (no robustness).  None = auto-calibrated from
                      bootstrap W1 distances.
    n_bootstrap     : Bootstrap samples for epsilon calibration.
    delta           : Risk-aversion coefficient (higher = more cautious).
    min_history     : Minimum return observations required.
    max_instruments : CPU-safety cap on number of instruments.
    """

    def __init__(
        self,
        epsilon: float | None = None,
        n_bootstrap: int = 200,
        delta: float = 2.5,
        min_history: int = 20,
        max_instruments: int = 200,
    ) -> None:
        self.epsilon = epsilon
        self.n_bootstrap = n_bootstrap
        self.delta = delta
        self.min_history = min_history
        self.max_instruments = max_instruments

    def build_weights(
        self,
        returns: np.ndarray,
        entity_ids: list[str],
        return_views: dict[str, float] | None = None,
    ) -> RobustPortfolioWeights | None:
        """Build Wasserstein-robust portfolio weights.

        Args:
            returns     : (T, N) return matrix.  N = len(entity_ids).
            entity_ids  : Asset identifiers matching columns of returns.
            return_views: Optional GNN return predictions as views.

        Returns:
            RobustPortfolioWeights, or None if insufficient data.
        """
        if returns.shape[0] < self.min_history:
            log.warning(
                "WassersteinRobustPortfolio: only %d observations "
                "(need ≥ %d).",
                returns.shape[0], self.min_history,
            )
            return None

        N = min(returns.shape[1], self.max_instruments)
        returns = returns[:, :N]
        eids = entity_ids[:N]

        # Calibrate or use fixed epsilon
        if self.epsilon is None:
            eps = bootstrap_epsilon(returns, n_bootstrap=self.n_bootstrap)
            log.info("WassersteinRobust: calibrated ε = %.5f", eps)
        else:
            eps = float(self.epsilon)

        # Compute covariance matrices
        cov_standard = np.cov(returns.T) if returns.shape[0] > 1 else np.eye(N)
        if cov_standard.ndim == 0:
            cov_standard = np.array([[float(cov_standard)]])
        cov_robust = cov_standard + eps * np.eye(N)

        # Build weight vector
        if return_views and len(return_views) >= 2:
            mean_vec = np.array(
                [return_views.get(eid, 0.0) for eid in eids], dtype=float
            )
            w = _blend_with_views(cov_robust, mean_vec, self.delta)
        else:
            w = _min_variance_weights(cov_robust)

        weights_dict = {eid: float(w[i]) for i, eid in enumerate(eids)}

        return RobustPortfolioWeights(
            weights=weights_dict,
            epsilon=eps,
            robust_cov=float(np.trace(cov_robust)),
            standard_cov=float(np.trace(cov_standard)),
            n_assets=N,
            built_at=time.time(),
        )

    def build_weights_from_store(
        self,
        store: Any,
        entity_ids: list[str] | None = None,
        return_views: dict[str, float] | None = None,
        lookback_days: int = 90,
        as_of: float | None = None,
    ) -> RobustPortfolioWeights | None:
        """Convenience wrapper: load returns from PipelineStore, then build.

        Args:
            store        : PipelineStore instance.
            entity_ids   : Instruments to include (None = all available).
            return_views : GNN return predictions as views.
            lookback_days: Return history window.
            as_of        : Reference timestamp (None = now).

        Returns:
            RobustPortfolioWeights, or None if insufficient data.
        """
        from agent.convergence.tda_regime import _load_returns  # noqa: PLC0415

        returns = _load_returns(
            store, entity_ids, lookback_days,
            self.max_instruments, as_of,
        )
        if returns is None:
            log.warning("WassersteinRobust: no returns from store.")
            return None

        if entity_ids is None:
            # Use column indices as entity IDs (store loader doesn't return names)
            eids = [f"entity_{i}" for i in range(returns.shape[1])]
        else:
            eids = entity_ids[: returns.shape[1]]

        return self.build_weights(returns, eids, return_views=return_views)

    def store_weights(
        self,
        store: Any,
        result: RobustPortfolioWeights,
    ) -> int:
        """Persist robust portfolio weights and diagnostics to store.

        Signal names:
            robust_portfolio.{entity_id}.weight
            robust_portfolio._epsilon
            robust_portfolio._cov_inflation_ratio

        Returns number of signals written.
        """
        n = 0
        ratio = (
            result.robust_cov / result.standard_cov
            if result.standard_cov > 1e-12 else 1.0
        )
        # Per-asset weights
        for eid, w in result.weights.items():
            try:
                store.store_signal(
                    signal_name=f"robust_portfolio.{eid}.weight",
                    value=w,
                    observed_at=result.built_at,
                    source_tool="wasserstein_robust_portfolio",
                )
                n += 1
            except Exception:
                log.warning("Failed to store weight for %s.", eid, exc_info=True)

        # Diagnostics
        for sig, val in [
            ("robust_portfolio._epsilon", result.epsilon),
            ("robust_portfolio._cov_inflation_ratio", ratio),
        ]:
            try:
                store.store_signal(
                    signal_name=sig,
                    value=val,
                    observed_at=result.built_at,
                    source_tool="wasserstein_robust_portfolio",
                )
                n += 1
            except Exception:
                log.warning("Failed to store diagnostic %s.", sig, exc_info=True)

        return n
