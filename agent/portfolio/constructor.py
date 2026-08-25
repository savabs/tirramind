"""
TirraMind — HRP + Black-Litterman Portfolio Construction Layer (Idea 11)

Translates the GNN's per-entity return predictions into actionable portfolio
weights using two complementary methods.

Problem
-------
TirraMind's ``return_pred_head`` produces raw log-return scalars — one per
instrument node.  These are used as training targets (Phase 41 aux loss) but
never converted into actual portfolio allocations.  A raw return scalar is not
a portfolio weight: it ignores covariance structure, cannot be risk-budgeted,
has no turnover controls, and carries no uncertainty information.

Without this layer, TirraMind has **no actionable financial output**.

Solution — Blended BL + HRP
----------------------------

**Step 1 — Black-Litterman Posterior Returns (He & Litterman 1999)**

Skips mean-variance's need for a stable mean estimate (Michaud 1989 argued
this leads to error-maximising portfolios).  Instead:

  - Prior: market equilibrium returns Π = δ · Σ · w_mkt
    where δ = risk aversion ≈ 2.5, w_mkt = equal weight if caps unavailable.
  - Views: GNN predictions Q = [q_1,...,q_n] (absolute, P = I_n).
  - View uncertainty: Ω_ii = (1/c_i - 1) · τ · Σ_ii
    where c_i ∈ (0,1] = quality score.  High c_i → small Ω → views trusted.
  - Posterior:
      Σ_bl = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
      μ_bl  = Σ_bl · [(τΣ)^{-1}Π + P'Ω^{-1}Q]

  τ = 1 / n_history (standard scaling for BL).

**Step 2 — Hierarchical Risk Parity (de Prado 2016)**

Mean-variance maximises Sharpe ratio but is sensitive to estimation error
in the covariance matrix (the "Markowitz curse").  HRP is estimation-error
resistant because it never inverts the covariance matrix:

  1. Build distance matrix: d_ij = √(½(1 − ρ_ij))
     Satisfies triangle inequality; ρ = Pearson correlation of returns.
  2. Hierarchical clustering: Ward linkage on d.
     Produces a dendrogram encoding the natural diversification structure
     of the portfolio — identical in spirit to TirraMind's graph communities.
  3. Quasi-diagonalisation: reorder assets by leaf order of the dendrogram.
  4. Recursive bisection: split the sorted asset list into two halves and
     allocate risk budget proportionally to inverse cluster variance:
       α_L = 1 - Var(L) / (Var(L) + Var(R))
       w_L = α_L · recurse(L),  w_R = (1-α_L) · recurse(R)
     Base case: w_i = 1 / Σ_ii (inverse volatility).

**Step 3 — BL tilt**

Pure HRP ignores return forecasts.  We tilt HRP weights towards assets with
higher BL posterior expected returns:
    w_tilt_i = w_hrp_i · (1 + α · tanh(μ_bl_i / σ_μ))
where α = ``tilt_factor`` (default 0.5) and σ_μ = std(μ_bl).
Weights are then renormalised to sum to 1.

**Step 4 — Turnover constraint**
    w_final = (1 - λ) · w_tilt + λ · w_prev
where λ = ``turnover_lambda`` ∈ [0,1].  When w_prev is None, no smoothing.

Output
------
``PortfolioConstructor.build_weights(store, return_preds, quality_scores, ...)``
returns a ``PortfolioWeights`` object containing:
  - ``weights``: dict[entity_id, float] — final portfolio weights summing to 1
  - ``expected_returns``: dict[entity_id, float] — BL posterior μ_bl
  - ``bl_covariance``: numpy (n, n) — BL posterior covariance Σ_bl
  - ``entity_ids``: ordered list of entities included

Weights are also persisted to the store via ``store_weights()`` for
downstream consumption by the ConvergenceDetector and RL layer.

References
----------
    He, G. & Litterman, R. (1999). The Intuition Behind Black-Litterman
        Model Portfolios. Goldman Sachs Investment Management.

    Black, F. & Litterman, R. (1992). Global Portfolio Optimization.
        Financial Analysts Journal 48(5):28-43.

    de Prado, M.L. (2016). Building Diversified Portfolios that Outperform
        Out of Sample. Journal of Portfolio Management 42(4):59-69.
        https://doi.org/10.3905/jpm.2016.42.4.059
        Core algorithm: getHRP(), getIVP(), getClusterVar(), getRecBipart().

    Michaud, R.O. (1989). The Markowitz Optimization Enigma: Is Optimized
        Optimal? Financial Analysts Journal 45(1):31-42.

    Walters, J. (2014). The Black-Litterman Model in Detail.
        SSRN 1314585 — standard reference for BL formula derivation.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_DAY: float = 86_400.0
_EPS: float = 1e-8

_VALUE_KEYS = (
    "close", "usd_amount", "value", "estimated_value",
    "log_return", "btc_amount",
)


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PortfolioWeights:
    """Portfolio weight output from ``PortfolioConstructor``.

    Attributes
    ----------
    weights : dict[str, float]
        Final portfolio weights, entity_id → weight.  Sum ≈ 1.
    expected_returns : dict[str, float]
        Black-Litterman posterior expected returns per entity.
    hrp_weights : dict[str, float]
        Raw HRP weights before BL tilt (diagnostic).
    bl_covariance : np.ndarray
        BL posterior covariance matrix (n, n).
    entity_ids : list[str]
        Ordered entity IDs included in the portfolio.
    n_assets : int
        Number of assets in the portfolio.
    computed_at : float
        Unix timestamp.
    details : dict[str, Any]
        Intermediate values (tau, delta, etc.) for audit.
    """

    weights: dict[str, float]
    expected_returns: dict[str, float]
    hrp_weights: dict[str, float]
    bl_covariance: np.ndarray
    entity_ids: list[str]
    n_assets: int
    computed_at: float
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# PortfolioConstructor
# ═══════════════════════════════════════════════════════════════════════════


class PortfolioConstructor:
    """Black-Litterman + HRP portfolio constructor.

    Parameters
    ----------
    delta : float
        Risk aversion coefficient for BL equilibrium prior.  Default 2.5.
    tau_scale : float
        Scaling factor: τ = tau_scale / n_history.  Default 1.0.
    tilt_factor : float
        BL-return tilt magnitude α.  0 = pure HRP; 1 = strong tilt.  Default 0.5.
    turnover_lambda : float
        Smoothing factor λ with previous weights.  0 = no smoothing.  Default 0.3.
    min_history : int
        Minimum return observations to include an asset.  Default 20.
    lookback_days : int
        Observation history window.  Default 365.
    n_bins : int
        Time bins for the history window.  Default 60.
    obs_limit : int
        Max observations to load per entity.  Default 2000.
    """

    def __init__(
        self,
        delta: float = 2.5,
        tau_scale: float = 1.0,
        tilt_factor: float = 0.5,
        turnover_lambda: float = 0.3,
        min_history: int = 20,
        lookback_days: int = 365,
        n_bins: int = 60,
        obs_limit: int = 2000,
    ) -> None:
        self.delta = delta
        self.tau_scale = tau_scale
        self.tilt_factor = tilt_factor
        self.turnover_lambda = turnover_lambda
        self.min_history = min_history
        self.lookback_days = lookback_days
        self.n_bins = n_bins
        self.obs_limit = obs_limit

    # ── Public API ─────────────────────────────────────────────────────────

    def build_weights(
        self,
        store: Any,
        return_preds: dict[str, float],
        quality_scores: dict[str, float] | None = None,
        prev_weights: dict[str, float] | None = None,
        as_of: float | None = None,
    ) -> PortfolioWeights | None:
        """Construct portfolio weights from GNN return predictions.

        Args:
            store: PipelineStore instance.
            return_preds: entity_id → predicted log_return.
            quality_scores: entity_id → confidence ∈ (0,1].
                Defaults to 0.5 for missing entities.
            prev_weights: entity_id → previous weight (for turnover smoothing).
            as_of: Reference time.  Defaults to ``time.time()``.

        Returns:
            PortfolioWeights or None if insufficient data.
        """
        if as_of is None:
            as_of = time.time()

        if quality_scores is None:
            quality_scores = {}

        entity_ids = list(return_preds.keys())
        if not entity_ids:
            log.info("PortfolioConstructor: no return predictions provided.")
            return None

        t_start = as_of - self.lookback_days * _DAY

        # Build price/return history for each entity
        price_matrix = self._build_price_matrix(
            store, entity_ids, t_start, as_of
        )   # dict[entity_id, np.ndarray (n_bins,)]

        # Filter entities with sufficient history
        valid_ids = [
            eid for eid in entity_ids
            if eid in price_matrix
            and np.sum(~np.isnan(price_matrix[eid])) >= self.min_history
        ]
        if len(valid_ids) < 2:
            log.info(
                "PortfolioConstructor: only %d entities with sufficient history "
                "(need ≥ 2).",
                len(valid_ids),
            )
            return None

        # Build returns matrix: (n_assets, n_bins-1) log returns
        ret_matrix = self._price_matrix_to_returns(price_matrix, valid_ids)
        n_hist = ret_matrix.shape[1]
        n_assets = len(valid_ids)

        # Sample covariance matrix (annualised by n_hist factor not needed here)
        cov = _sample_covariance(ret_matrix)   # (n, n)

        # BL: equilibrium prior (equal weights)
        w_mkt = np.ones(n_assets) / n_assets
        pi = self.delta * cov @ w_mkt            # equilibrium returns (n,)

        # Views: GNN predicted returns for valid entities
        q = np.array([return_preds.get(eid, 0.0) for eid in valid_ids])

        # View uncertainty: Ω_ii = (1/c_i - 1) * τ * Σ_ii
        tau = self.tau_scale / max(n_hist, 1)
        c = np.array([quality_scores.get(eid, 0.5) for eid in valid_ids])
        c = np.clip(c, 1e-3, 1.0 - 1e-3)
        omega_diag = (1.0 / c - 1.0) * tau * np.diag(cov)

        # BL posterior expected returns and covariance
        mu_bl, cov_bl = _black_litterman(cov, pi, q, omega_diag, tau)

        # HRP weights from BL posterior covariance
        hrp_w = _hrp_weights(cov_bl, valid_ids)   # dict[eid, float]

        # BL tilt: skew HRP weights toward higher-return entities
        if self.tilt_factor > 0.0:
            sigma_mu = float(np.std(mu_bl)) + _EPS
            tilt = np.array([mu_bl[i] / sigma_mu for i in range(n_assets)])
            scale = np.array([hrp_w[eid] for eid in valid_ids])
            scale = scale * (1.0 + self.tilt_factor * np.tanh(tilt))
            scale = np.clip(scale, 0.0, None)
            total = scale.sum()
            if total < _EPS:
                final_w = {eid: 1.0 / n_assets for eid in valid_ids}
            else:
                final_w = {eid: float(scale[i] / total) for i, eid in enumerate(valid_ids)}
        else:
            final_w = hrp_w

        # Turnover smoothing
        if prev_weights and self.turnover_lambda > 0.0:
            lam = self.turnover_lambda
            for eid in final_w:
                prev = prev_weights.get(eid, 0.0)
                final_w[eid] = (1.0 - lam) * final_w[eid] + lam * prev
            # Re-normalise after smoothing (prev weights may not sum to 1 over subset)
            total = sum(final_w.values())
            if total > _EPS:
                final_w = {eid: w / total for eid, w in final_w.items()}

        expected_returns = {eid: float(mu_bl[i]) for i, eid in enumerate(valid_ids)}

        return PortfolioWeights(
            weights=final_w,
            expected_returns=expected_returns,
            hrp_weights=hrp_w,
            bl_covariance=cov_bl,
            entity_ids=valid_ids,
            n_assets=n_assets,
            computed_at=as_of,
            details={
                "tau": tau,
                "delta": self.delta,
                "n_history_bins": n_hist,
                "tilt_factor": self.tilt_factor,
            },
        )

    def store_weights(
        self,
        pw: PortfolioWeights,
        store: Any,
        date: str,
    ) -> int:
        """Persist portfolio weights to the store.

        Stores via ``store.store_portfolio_weights(date, weights)`` for
        downstream RL / paper P&L consumption.

        Returns:
            Number of assets stored.
        """
        try:
            store.store_portfolio_weights(date, pw.weights)
            log.info(
                "PortfolioConstructor: stored %d weights for date %s.",
                pw.n_assets, date,
            )
            return pw.n_assets
        except Exception:
            log.warning(
                "PortfolioConstructor: failed to store weights for %s", date, exc_info=True
            )
            return 0

    # ── Internal: price matrix ─────────────────────────────────────────────

    def _build_price_matrix(
        self,
        store: Any,
        entity_ids: list[str],
        t_start: float,
        t_end: float,
    ) -> dict[str, np.ndarray]:
        """Query observations and bin into a price series per entity."""
        span = max(t_end - t_start, 1.0)
        bin_dur = span / self.n_bins

        price_matrix: dict[str, np.ndarray] = {}

        for eid in entity_ids:
            obs = store.query_entity_observations(
                eid, since=t_start, until=t_end, limit=self.obs_limit
            )
            if not obs:
                continue

            sums = np.zeros(self.n_bins)
            cnts = np.zeros(self.n_bins)
            for o in obs:
                t = float(o.get("observed_at", t_start))
                b = min(int((t - t_start) / bin_dur), self.n_bins - 1)
                v = _extract_price(o.get("value", {}))
                if v is not None and math.isfinite(v) and v > 0:
                    sums[b] += v
                    cnts[b] += 1.0

            with np.errstate(invalid="ignore", divide="ignore"):
                row = np.where(cnts > 0, sums / cnts, np.nan)

            price_matrix[eid] = row

        return price_matrix

    def _price_matrix_to_returns(
        self,
        price_matrix: dict[str, np.ndarray],
        valid_ids: list[str],
    ) -> np.ndarray:
        """Convert price bins to log-returns: (n_assets, n_bins-1)."""
        n = len(valid_ids)
        n_bins = self.n_bins
        ret = np.full((n, n_bins - 1), np.nan)

        for i, eid in enumerate(valid_ids):
            prices = price_matrix[eid]
            # Forward-fill NaNs for continuity
            prices = _forward_fill_prices(prices)
            valid = prices > 0
            if valid.sum() < 2:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ret = np.where(
                    (prices[1:] > 0) & (prices[:-1] > 0),
                    np.log(prices[1:] / prices[:-1]),
                    np.nan,
                )
            ret[i] = log_ret

        # Fill remaining NaNs with 0
        ret = np.nan_to_num(ret, nan=0.0)
        return ret


# ═══════════════════════════════════════════════════════════════════════════
# Black-Litterman
# ═══════════════════════════════════════════════════════════════════════════


def _black_litterman(
    cov: np.ndarray,
    pi: np.ndarray,
    q: np.ndarray,
    omega_diag: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BL posterior expected returns and covariance.

    Uses absolute views (P = I_n) for simplicity — each GNN prediction is
    a direct view on a single asset's return.

    Formula (Walters 2014, eq. 1-3):
        Σ_bl = inv(inv(τΣ) + P' inv(Ω) P)
        μ_bl  = Σ_bl · (inv(τΣ) Π + P' inv(Ω) Q)

    With P = I_n:
        Σ_bl = inv(inv(τΣ) + inv(Ω))
        μ_bl  = Σ_bl · (inv(τΣ) Π + inv(Ω) Q)

    Args:
        cov: (n, n) sample covariance matrix.
        pi: (n,) equilibrium prior returns.
        q: (n,) view returns (GNN predictions).
        omega_diag: (n,) diagonal of view uncertainty matrix Ω.
        tau: BL scaling parameter.

    Returns:
        (mu_bl, cov_bl) — posterior expected returns and covariance.
    """
    n = len(pi)
    tau_cov = tau * cov + np.eye(n) * _EPS  # numerical stability

    try:
        tau_cov_inv = np.linalg.inv(tau_cov)
    except np.linalg.LinAlgError:
        tau_cov_inv = np.linalg.pinv(tau_cov)

    omega_inv = np.diag(1.0 / (omega_diag + _EPS))

    # Posterior covariance
    try:
        cov_bl = np.linalg.inv(tau_cov_inv + omega_inv)
    except np.linalg.LinAlgError:
        cov_bl = np.linalg.pinv(tau_cov_inv + omega_inv)

    # Posterior expected returns
    mu_bl = cov_bl @ (tau_cov_inv @ pi + omega_inv @ q)

    return mu_bl, cov_bl


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical Risk Parity (de Prado 2016)
# ═══════════════════════════════════════════════════════════════════════════


def _hrp_weights(
    cov: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """HRP portfolio weights from covariance matrix.

    Implementation follows de Prado (2016) exactly:
    getHRP → getQuasiDiag → getRecBipart → getIVP.

    Args:
        cov: (n, n) covariance matrix.
        labels: list of n asset labels.

    Returns:
        dict[label, weight] with weights summing to 1.
    """
    n = len(labels)
    if n == 1:
        return {labels[0]: 1.0}
    if n == 0:
        return {}

    # Step 1: correlation matrix and distance
    std = np.sqrt(np.diag(cov)).clip(min=_EPS)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))

    # Step 2: hierarchical clustering (Ward linkage for minimum variance)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list  # noqa: PLC0415

        condensed = _condensed_distance(dist)
        z = linkage(condensed, method="ward")
        leaf_order = leaves_list(z).tolist()
    except ImportError:
        # Fallback: no reordering
        leaf_order = list(range(n))

    # Step 3: quasi-diagonalisation (reorder by cluster leaves)
    sorted_labels = [labels[i] for i in leaf_order]
    sorted_cov = cov[np.ix_(leaf_order, leaf_order)]

    # Step 4: recursive bisection
    raw_w = _rec_bipart(sorted_cov, list(range(n)))

    return {sorted_labels[i]: float(raw_w[i]) for i in range(n)}


def _rec_bipart(cov: np.ndarray, items: list[int]) -> np.ndarray:
    """Recursive bisection weight allocation (de Prado 2016, getRecBipart).

    Returns weight array aligned with ``items`` (0-indexed into cov).
    """
    n = len(items)
    w = np.ones(n)

    if n <= 1:
        return w

    def _cluster_var(idx: list[int]) -> float:
        """Inverse-volatility weighted cluster variance."""
        sub_cov = cov[np.ix_(idx, idx)]
        ivp = 1.0 / np.diag(sub_cov).clip(min=_EPS)
        ivp /= ivp.sum()
        return float(ivp @ sub_cov @ ivp)

    # Split into two halves
    mid = n // 2
    left_idx = items[:mid]
    right_idx = items[mid:]

    var_l = _cluster_var(left_idx)
    var_r = _cluster_var(right_idx)
    alpha_l = 1.0 - var_l / (var_l + var_r + _EPS)

    # Recurse
    w_l = _rec_bipart(cov, left_idx)
    w_r = _rec_bipart(cov, right_idx)

    w[:mid] = alpha_l * w_l
    w[mid:] = (1.0 - alpha_l) * w_r
    return w


def _condensed_distance(dist: np.ndarray) -> np.ndarray:
    """Convert (n, n) distance matrix to condensed upper-triangle form."""
    n = dist.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(dist[i, j])
    return np.array(pairs)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _sample_covariance(ret: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    """Ledoit-Wolf-inspired shrinkage covariance estimator.

    ``Σ_shrunk = (1-α)·Σ + α·μ_var·I``

    where μ_var = mean diagonal variance and α = ``shrinkage``.
    Prevents near-singular matrices when n_assets ≈ n_history.

    Args:
        ret: (n_assets, n_obs) return matrix.
        shrinkage: regularisation weight.  Default 0.05.

    Returns:
        (n_assets, n_assets) regularised covariance matrix.
    """
    n = ret.shape[0]
    cov = np.cov(ret)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    elif cov.shape == ():
        cov = np.array([[float(cov)]])

    mu_var = np.trace(cov) / max(n, 1)
    target = mu_var * np.eye(n)
    return (1.0 - shrinkage) * cov + shrinkage * target


def _extract_price(value_dict: Any) -> float | None:
    """Extract a positive price scalar from an observation value dict."""
    if not isinstance(value_dict, dict):
        return None
    for k in _VALUE_KEYS:
        if k in value_dict:
            try:
                v = float(value_dict[k])
                return v if math.isfinite(v) and v > 0 else None
            except (TypeError, ValueError):
                pass
    return None


def _forward_fill_prices(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values; leading NaNs become the first valid value."""
    out = arr.copy()
    last = 0.0
    # First pass: find first valid
    for v in out:
        if not math.isnan(v) and v > 0:
            last = v
            break
    for i in range(len(out)):
        if math.isnan(out[i]) or out[i] <= 0:
            out[i] = last
        else:
            last = out[i]
    return out
