"""
TirraMind — Bayesian Parameter Optimizer (Learning Layer)

Gaussian Process Bayesian Optimization for low-dimensional parameter tuning.
Used by RewardWeightOptimizer (Change 5) and ThresholdOptimizer (Change 7).

Math:
    GP posterior with RBF kernel:
        μ(x) = k_*ᵀ (K + σ_n² I)⁻¹ y
        σ²(x) = k(x,x) - k_*ᵀ (K + σ_n² I)⁻¹ k_*

    Expected Improvement acquisition:
        EI(x) = (μ(x) - f⁺ - ξ) Φ(Z) + σ(x) φ(Z)
        Z = (μ(x) - f⁺ - ξ) / σ(x)

    where Φ, φ are standard normal CDF/PDF.

References:
    - Rasmussen & Williams, "Gaussian Processes for Machine Learning", Ch. 2
    - Snoek, Larochelle & Adams 2012, "Practical Bayesian Optimization"
    - Spec: [[learned_vs_handcoded_architecture_spec]], Changes 5 & 7

Implementation choices:
    - RBF kernel (squared exponential): smooth, differentiable, well-suited
      for low-dim continuous spaces (2-5 dims).
    - Cholesky factorization for numerical stability.
    - Length scale set to 20% of parameter range (heuristic for <30 trials).
    - Noise σ_n = 0.1 (evaluations are noisy — portfolio Sharpe, F1 scores).
    - No hyperparameter optimization of kernel params (insufficient data with
      <30 trials; fixed heuristics are more stable).
    - Why not scikit-optimize: avoids an unmaintained dependency for a
      surface that's <5 dims and <30 evaluations.  numpy/scipy suffice.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParamSpace:
    """Defines a bounded parameter search space.

    Each dimension has a name, lower bound, and upper bound.
    """

    names: list[str]
    bounds: list[tuple[float, float]]  # (lo, hi) per dimension

    def __post_init__(self) -> None:
        if len(self.names) != len(self.bounds):
            raise ValueError("names and bounds must have equal length")
        for name, (lo, hi) in zip(self.names, self.bounds):
            if lo >= hi:
                raise ValueError(f"Invalid bounds for '{name}': lo={lo} >= hi={hi}")

    @property
    def ndim(self) -> int:
        return len(self.names)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Clip parameter vector to bounds."""
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return np.clip(x, lo, hi)

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random point uniformly within bounds."""
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return rng.uniform(lo, hi)

    def to_dict(self, x: np.ndarray) -> dict[str, float]:
        """Convert a parameter vector to a named dict."""
        return {name: float(val) for name, val in zip(self.names, x)}

    def from_dict(self, d: dict[str, float]) -> np.ndarray:
        """Convert a named dict to a parameter vector."""
        return np.array([d[name] for name in self.names])


@dataclass
class Trial:
    """One completed evaluation: parameters → objective value."""

    params: dict[str, float]
    objective: float
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gaussian Process (RBF kernel, Cholesky-based)
# ---------------------------------------------------------------------------


def _rbf_kernel(
    X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float = 1.0
) -> np.ndarray:
    """Squared-exponential (RBF) kernel matrix.

    k(x, x') = σ_f² exp(-‖x - x'‖² / (2 ℓ²))
    """
    # Pairwise squared distances
    sq_dist = (
        np.sum(X1**2, axis=1, keepdims=True)
        + np.sum(X2**2, axis=1, keepdims=True).T
        - 2.0 * X1 @ X2.T
    )
    return signal_var * np.exp(-0.5 * sq_dist / (length_scale**2))


def _gp_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    length_scale: float,
    noise_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GP posterior mean and variance at test points.

    Returns:
        (mu, var) — each shape (n_test,).
    """
    n = X_train.shape[0]
    K = _rbf_kernel(X_train, X_train, length_scale) + noise_var * np.eye(n)

    # Cholesky for numerical stability
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # If Cholesky fails, add jitter
        K += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)

    # α = K⁻¹ y  via  L Lᵀ α = y
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Predictive mean
    K_star = _rbf_kernel(X_test, X_train, length_scale)
    mu = K_star @ alpha

    # Predictive variance
    v = np.linalg.solve(L, K_star.T)  # L⁻¹ K_star.T
    var = 1.0 - np.sum(v**2, axis=0)  # k(x*,x*) = 1.0 (signal_var=1)
    var = np.maximum(var, 1e-10)  # numerical floor

    return mu, var


def _expected_improvement(
    mu: np.ndarray,
    var: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition function.

    EI(x) = (μ(x) - f⁺ - ξ) Φ(Z) + σ(x) φ(Z)
    Z = (μ(x) - f⁺ - ξ) / σ(x)
    """
    sigma = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    # Zero out where sigma is tiny (already evaluated point)
    ei = np.where(sigma < 1e-8, 0.0, ei)
    return ei


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------


class BayesianParamOptimizer:
    """GP-based Bayesian optimizer for low-dimensional parameter spaces.

    Usage::

        space = ParamSpace(["lr", "momentum"], [(1e-4, 1.0), (0.5, 0.99)])
        opt = BayesianParamOptimizer(space, persist_path=Path("opt.json"))
        params = opt.suggest()          # → {"lr": 0.42, "momentum": 0.87}
        opt.record(params, sharpe=1.3)  # record trial result
        best = opt.best()               # → Trial with highest objective

    Design decisions:
        - First ``n_random`` suggestions are uniform random (Latin hypercube
          exploration before the GP has enough data).
        - After that, suggestions maximize Expected Improvement.
        - Candidate pool: 500 random + 20 best-so-far perturbations.
        - Length scale = 20% of mean parameter range (stable for <30 trials).
        - No kernel hyperparameter learning — insufficient data.
    """

    def __init__(
        self,
        space: ParamSpace,
        *,
        persist_path: Path | None = None,
        n_random: int = 5,
        noise_var: float = 0.01,
        n_candidates: int = 500,
        seed: int | None = None,
    ) -> None:
        self._space = space
        self._persist_path = persist_path
        self._n_random = n_random
        self._noise_var = noise_var
        self._n_candidates = n_candidates
        self._rng = np.random.default_rng(seed)
        self._trials: list[Trial] = []

        # Heuristic length scale: 20% of mean parameter range
        ranges = np.array([hi - lo for lo, hi in space.bounds])
        self._length_scale = float(0.2 * np.mean(ranges))

        if persist_path and persist_path.exists():
            self._load()

    @property
    def n_trials(self) -> int:
        return len(self._trials)

    @property
    def trials(self) -> list[Trial]:
        return list(self._trials)

    def suggest(self) -> dict[str, float]:
        """Suggest the next parameter vector to evaluate.

        Returns a dict mapping parameter names to values.
        """
        if self.n_trials < self._n_random:
            # Exploration phase: uniform random
            x = self._space.sample_uniform(self._rng)
            log.info(
                "BO suggest (random %d/%d): %s",
                self.n_trials + 1,
                self._n_random,
                self._space.to_dict(x),
            )
            return self._space.to_dict(x)

        # Build GP from trial history
        X_train, y_train = self._trials_to_arrays()

        # Normalize y to [0, 1] for stable GP (objective can be any scale)
        y_min, y_max = y_train.min(), y_train.max()
        if y_max - y_min > 1e-8:
            y_norm = (y_train - y_min) / (y_max - y_min)
        else:
            y_norm = np.zeros_like(y_train)

        # Normalize X to [0, 1] per dimension
        lo = np.array([b[0] for b in self._space.bounds])
        hi = np.array([b[1] for b in self._space.bounds])
        ranges = hi - lo
        ranges = np.where(ranges < 1e-10, 1.0, ranges)
        X_norm = (X_train - lo) / ranges

        # Generate candidate pool
        n_rand = self._n_candidates - 20
        candidates_rand = self._rng.uniform(0.0, 1.0, (n_rand, self._space.ndim))

        # Perturbations around best point
        best_idx = np.argmax(y_train)
        best_x_norm = X_norm[best_idx]
        perturbations = best_x_norm + self._rng.normal(0, 0.1, (20, self._space.ndim))
        perturbations = np.clip(perturbations, 0.0, 1.0)

        candidates = np.vstack([candidates_rand, perturbations])

        # GP posterior + EI
        mu, var = _gp_posterior(
            X_norm, y_norm, candidates, self._length_scale, self._noise_var
        )
        f_best = y_norm.max()
        ei = _expected_improvement(mu, var, f_best)

        # Pick candidate with highest EI
        best_cand_idx = np.argmax(ei)
        x_norm = candidates[best_cand_idx]

        # Denormalize
        x = x_norm * ranges + lo
        x = self._space.clip(x)

        log.info(
            "BO suggest (EI, trial %d): %s  EI=%.4f",
            self.n_trials + 1,
            self._space.to_dict(x),
            ei[best_cand_idx],
        )
        return self._space.to_dict(x)

    def record(
        self, params: dict[str, float], objective: float, metadata: dict | None = None
    ) -> None:
        """Record a completed trial.

        Args:
            params: The parameter dict (as returned by suggest()).
            objective: The objective value (higher = better).
            metadata: Optional extra info to persist with the trial.
        """
        trial = Trial(params=params, objective=objective, metadata=metadata or {})
        self._trials.append(trial)
        log.info(
            "BO record trial %d: obj=%.4f params=%s",
            self.n_trials,
            objective,
            params,
        )
        self._persist()

    def best(self) -> Trial | None:
        """Return the trial with the highest objective, or None if empty."""
        if not self._trials:
            return None
        return max(self._trials, key=lambda t: t.objective)

    def best_params(self) -> dict[str, float] | None:
        """Return just the params dict of the best trial."""
        b = self.best()
        return b.params if b else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trials_to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert trial list to numpy arrays for GP."""
        X = np.array([self._space.from_dict(t.params) for t in self._trials])
        y = np.array([t.objective for t in self._trials])
        return X, y

    # ------------------------------------------------------------------
    # Persistence (JSON, same pattern as bandit.py)
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "space": {
                "names": self._space.names,
                "bounds": self._space.bounds,
            },
            "trials": [
                {
                    "params": t.params,
                    "objective": t.objective,
                    "metadata": t.metadata,
                }
                for t in self._trials
            ],
        }
        self._persist_path.write_text(json.dumps(state, indent=2))

    def _load(self) -> None:
        try:
            state = json.loads(self._persist_path.read_text())
            for td in state.get("trials", []):
                self._trials.append(
                    Trial(
                        params=td["params"],
                        objective=float(td["objective"]),
                        metadata=td.get("metadata", {}),
                    )
                )
            log.info(
                "Loaded BO state: %d trials from %s",
                len(self._trials),
                self._persist_path,
            )
        except Exception as exc:
            log.warning("Failed to load BO state from %s: %s", self._persist_path, exc)
