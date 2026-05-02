"""Pairwise coincidence scoring for convergence detection.

Three lightweight statistical methods detect unusual co-movement between
pairs of aligned, normalised time series:

1. **Rolling Correlation Deviation** — detects when the Pearson
   correlation between two series deviates from its historical baseline.
2. **Joint Exceedance** — counts simultaneous tail events and tests
   against the null hypothesis of independence.
3. **Concordance Index** — measures directional agreement (same-sign
   first-differences) and tests against the null of random direction.

A combined scorer fuses the three via weighted averaging and Fisher's
method for p-values.

All methods are NaN-aware, guard against σ=0 and short-array edge cases,
and never use future information.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

log = logging.getLogger(__name__)

# Minimum non-NaN observations required for any meaningful calculation.
_MIN_VALID = 3

# Guard: treat σ (or any denominator) below this as zero.
_EPSILON = 1e-10

# Smallest p-value we allow before clipping (prevents log(0) in Fisher's).
_P_FLOOR = 1e-300


# ── Result dataclass ───────────────────────────────────────────


@dataclass
class CoincidenceResult:
    """Output of a single coincidence scoring method.

    Attributes
    ----------
    method : str
        One of ``"rolling_corr"``, ``"joint_exceedance"``,
        ``"concordance"``, ``"combined"``.
    score : float
        Measure of unusualness on a z-score-like scale (≥ 0).
        0.0 means no evidence of unusual coincidence.
    p_value : float
        Statistical significance against the null of independence.
        1.0 means no evidence.
    direction : int
        +1 signals converging (moving together more than expected),
        −1 diverging, 0 indeterminate.
    detail : dict
        Method-specific metadata for downstream inspection.
    """

    method: str
    score: float
    p_value: float
    direction: int
    detail: dict = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────


def _no_evidence(method: str, reason: str) -> CoincidenceResult:
    """Return a neutral result when computation is not possible."""
    return CoincidenceResult(
        method=method,
        score=0.0,
        p_value=1.0,
        direction=0,
        detail={"reason": reason},
    )


def _validate_pair(
    a: np.ndarray,
    b: np.ndarray,
    method: str,
    min_len: int,
) -> CoincidenceResult | None:
    """Shared input validation.  Returns a ``_no_evidence`` result on
    failure, or ``None`` if inputs are valid."""
    if len(a) != len(b):
        return _no_evidence(method, "length mismatch")
    if len(a) < min_len:
        return _no_evidence(method, f"need >= {min_len} observations")
    return None


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# ── Rolling Pearson correlation ────────────────────────────────


def _rolling_pearson(
    a: np.ndarray,
    b: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling Pearson ρ with NaN-pair exclusion.

    Returns an array of length ``len(a)``.  Positions ``0 .. window-2``
    are NaN (insufficient history).  Each subsequent position holds ρ
    computed over the trailing *window* observations, skipping any
    index where either *a* or *b* is NaN.

    If a window has fewer than ``_MIN_VALID`` valid pairs, ρ is NaN.
    If either series is constant within the window, ρ is 0.0 by
    convention (no linear relationship when one variable has no
    variance).
    """
    n = len(a)
    rho = np.full(n, np.nan)

    for i in range(window - 1, n):
        lo = i - window + 1
        ca = a[lo : i + 1]
        cb = b[lo : i + 1]

        mask = ~(np.isnan(ca) | np.isnan(cb))
        valid_n = int(mask.sum())
        if valid_n < _MIN_VALID:
            continue

        xa, xb = ca[mask], cb[mask]
        sa = float(np.std(xa, ddof=1))
        sb = float(np.std(xb, ddof=1))
        if sa < _EPSILON or sb < _EPSILON:
            rho[i] = 0.0
            continue

        rho[i] = float(np.corrcoef(xa, xb)[0, 1])

    return rho


# ── Method 1: Rolling Correlation Deviation ────────────────────


def rolling_correlation_score(
    a: np.ndarray,
    b: np.ndarray,
    *,
    corr_window: int = 20,
    baseline_window: int = 100,
) -> CoincidenceResult:
    r"""Detect unusual changes in pairwise correlation.

    Computes a rolling Pearson ρ series, then measures how far the
    most-recent ρ deviates from its historical baseline in standard-
    deviation units.

    Parameters
    ----------
    a, b : np.ndarray
        Aligned time series of identical length.
    corr_window : int
        Trailing window for each ρ estimate.
    baseline_window : int
        Number of prior ρ values used for μ(ρ) and σ(ρ).

    Returns
    -------
    CoincidenceResult
        ``score`` = |ρ_recent − μ_ρ| / σ_ρ  (z-score of correlation).
        ``p_value`` from two-tailed normal test.
        ``direction`` +1 if ρ increased, −1 if decreased.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    err = _validate_pair(a, b, "rolling_corr", corr_window)
    if err is not None:
        return err

    rho = _rolling_pearson(a, b, corr_window)
    valid_rho = rho[~np.isnan(rho)]

    if len(valid_rho) < _MIN_VALID:
        return _no_evidence("rolling_corr", "too few valid correlations")

    rho_current = float(valid_rho[-1])

    # Baseline: all valid ρ values *before* the current one.
    if len(valid_rho) < 2:
        return _no_evidence("rolling_corr", "need >= 2 valid correlations")

    baseline = valid_rho[:-1][-baseline_window:]  # most recent slice
    mu = float(np.mean(baseline))

    if len(baseline) < 2:
        sigma = 0.0
    else:
        sigma = float(np.std(baseline, ddof=1))

    if sigma < _EPSILON:
        return CoincidenceResult(
            method="rolling_corr",
            score=0.0,
            p_value=1.0,
            direction=0,
            detail={
                "rho_current": rho_current,
                "rho_baseline_mean": mu,
                "rho_baseline_std": 0.0,
                "n_valid_corr": int(len(valid_rho)),
            },
        )

    z = abs(rho_current - mu) / sigma
    p = float(2.0 * stats.norm.sf(z))  # two-tailed

    return CoincidenceResult(
        method="rolling_corr",
        score=float(z),
        p_value=p,
        direction=_sign(rho_current - mu),
        detail={
            "rho_current": rho_current,
            "rho_baseline_mean": mu,
            "rho_baseline_std": sigma,
            "n_valid_corr": int(len(valid_rho)),
        },
    )


# ── Method 2: Joint Exceedance ─────────────────────────────────


def joint_exceedance_score(
    z_a: np.ndarray,
    z_b: np.ndarray,
    *,
    z_threshold: float = 2.0,
    window: int = 20,
) -> CoincidenceResult:
    """Detect simultaneous tail events in two z-scored series.

    Counts how often both |z_a| and |z_b| exceed *z_threshold* in the
    most recent *window* observations, then tests whether the observed
    count exceeds what independence would predict.

    Marginal exceedance rates are estimated from the **full** series
    (not just the window) so the test is out-of-window-sample in the
    joint dimension.

    Parameters
    ----------
    z_a, z_b : np.ndarray
        Z-scored aligned series (from the atomic-signal layer).
    z_threshold : float
        Absolute z-score threshold for "exceedance" (default 2.0).
    window : int
        Trailing window for joint exceedance counting.

    Returns
    -------
    CoincidenceResult
        ``score`` = z-score of (observed − expected) joint exceedances.
        ``p_value`` from one-tailed binomial test (greater).
        ``direction`` +1 if excess joint exceedances, 0 otherwise.
    """
    z_a = np.asarray(z_a, dtype=np.float64)
    z_b = np.asarray(z_b, dtype=np.float64)

    err = _validate_pair(z_a, z_b, "joint_exceedance", _MIN_VALID)
    if err is not None:
        return err

    # ── Marginal exceedance rates (full series, NaN-aware) ─────
    valid_mask_a = ~np.isnan(z_a)
    valid_mask_b = ~np.isnan(z_b)
    n_valid_a = int(valid_mask_a.sum())
    n_valid_b = int(valid_mask_b.sum())

    if n_valid_a == 0 or n_valid_b == 0:
        return _no_evidence("joint_exceedance", "all-NaN input")

    p_a = float(np.sum(np.abs(z_a[valid_mask_a]) > z_threshold)) / n_valid_a
    p_b = float(np.sum(np.abs(z_b[valid_mask_b]) > z_threshold)) / n_valid_b
    p_joint_null = p_a * p_b

    if p_a < _EPSILON or p_b < _EPSILON:
        return _no_evidence(
            "joint_exceedance",
            "marginal exceedance rate is zero",
        )

    # ── Observed in recent window ──────────────────────────────
    n = len(z_a)
    win_start = max(0, n - window)
    wa = z_a[win_start:]
    wb = z_b[win_start:]

    pair_mask = ~(np.isnan(wa) | np.isnan(wb))
    n_trials = int(pair_mask.sum())

    if n_trials < _MIN_VALID:
        return _no_evidence("joint_exceedance", "too few valid pairs in window")

    observed = int(np.sum((np.abs(wa[pair_mask]) > z_threshold) & (np.abs(wb[pair_mask]) > z_threshold)))

    expected = n_trials * p_joint_null
    variance = n_trials * p_joint_null * (1.0 - p_joint_null)

    # One-tailed binomial test (excess joint exceedances)
    p_value = float(stats.binomtest(observed, n_trials, p_joint_null, alternative="greater").pvalue)

    # Z-score approximation (normal approx to binomial)
    if variance > _EPSILON:
        z_score = (observed - expected) / math.sqrt(variance)
    else:
        z_score = 0.0

    score = max(0.0, z_score)
    direction = 1 if observed > expected else 0

    return CoincidenceResult(
        method="joint_exceedance",
        score=score,
        p_value=p_value,
        direction=direction,
        detail={
            "observed": observed,
            "expected": round(expected, 4),
            "n_trials": n_trials,
            "p_marginal_a": round(p_a, 6),
            "p_marginal_b": round(p_b, 6),
            "p_joint_null": round(p_joint_null, 6),
        },
    )


# ── Method 3: Concordance Index ────────────────────────────────


def concordance_score(
    a: np.ndarray,
    b: np.ndarray,
    *,
    window: int = 20,
) -> CoincidenceResult:
    """Measure directional agreement between two series.

    Computes the fraction of recent periods where the first-differences
    of *a* and *b* share the same sign, and tests against the null
    hypothesis of random direction (p = 0.5).

    Parameters
    ----------
    a, b : np.ndarray
        Aligned time series of identical length.
    window : int
        Trailing window for concordance counting (applied to the
        first-difference series, so the raw series must have length
        ≥ window + 1).

    Returns
    -------
    CoincidenceResult
        ``score`` = |z| from the binomial approximation.
        ``p_value`` from two-tailed binomial test.
        ``direction`` +1 if concordance > 0.5 (positive coupling),
        −1 if < 0.5 (negative coupling).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Need at least window+1 raw observations to get `window` diffs.
    err = _validate_pair(a, b, "concordance", window + 1)
    if err is not None:
        return err

    da = np.diff(a)
    db = np.diff(b)

    # Most recent `window` first-differences
    recent_da = da[-window:]
    recent_db = db[-window:]

    pair_mask = ~(np.isnan(recent_da) | np.isnan(recent_db))
    n_trials = int(pair_mask.sum())

    if n_trials < _MIN_VALID:
        return _no_evidence("concordance", "too few valid first-differences")

    concordant = int(np.sum(np.sign(recent_da[pair_mask]) == np.sign(recent_db[pair_mask])))

    hit_rate = concordant / n_trials
    expected = n_trials * 0.5
    variance = n_trials * 0.25

    if variance < _EPSILON:
        return _no_evidence("concordance", "zero variance (n_trials too small)")

    z_approx = (concordant - expected) / math.sqrt(variance)

    p_value = float(stats.binomtest(concordant, n_trials, 0.5, alternative="two-sided").pvalue)

    return CoincidenceResult(
        method="concordance",
        score=abs(z_approx),
        p_value=p_value,
        direction=_sign(concordant - expected),
        detail={
            "concordant": concordant,
            "n_trials": n_trials,
            "hit_rate": round(hit_rate, 4),
        },
    )


# ── Combined scorer ────────────────────────────────────────────


_DEFAULT_WEIGHTS: dict[str, float] = {
    "rolling_corr": 1.0 / 3.0,
    "joint_exceedance": 1.0 / 3.0,
    "concordance": 1.0 / 3.0,
}


def combined_coincidence_score(
    a: np.ndarray,
    b: np.ndarray,
    z_a: np.ndarray,
    z_b: np.ndarray,
    *,
    weights: dict[str, float] | None = None,
) -> CoincidenceResult:
    """Fuse all three coincidence methods into a single result.

    Runs :func:`rolling_correlation_score`, :func:`joint_exceedance_score`,
    and :func:`concordance_score`, then combines:

    * **score** — weighted average of individual scores.
    * **p_value** — Fisher's combined probability test on the three
      p-values (χ² = −2 Σ ln pᵢ,  df = 2k).
    * **direction** — score-weighted majority vote.

    Parameters
    ----------
    a, b : np.ndarray
        Aligned raw value series (for rolling-corr and concordance).
    z_a, z_b : np.ndarray
        Aligned z-scored series (for joint-exceedance).
    weights : dict[str, float] | None
        Per-method weights.  Keys must be ``"rolling_corr"``,
        ``"joint_exceedance"``, and/or ``"concordance"``.  Weights are
        normalised to sum to 1.  Default: equal ``1/3`` each.

    Returns
    -------
    CoincidenceResult
        Combined assessment with all three sub-results in ``detail``.
    """
    w = dict(weights) if weights is not None else dict(_DEFAULT_WEIGHTS)
    total_w = sum(w.values())
    if total_w < _EPSILON:
        return _no_evidence("combined", "all weights are zero")
    w = {k: v / total_w for k, v in w.items()}

    # Run the three sub-methods
    r_corr = rolling_correlation_score(a, b)
    r_joint = joint_exceedance_score(z_a, z_b)
    r_conc = concordance_score(a, b)

    sub = {
        "rolling_corr": r_corr,
        "joint_exceedance": r_joint,
        "concordance": r_conc,
    }

    # ── Weighted score ─────────────────────────────────────────
    score = sum(w.get(k, 0.0) * r.score for k, r in sub.items())

    # ── Fisher's combined p-value ──────────────────────────────
    raw_p = [r.p_value for r in sub.values()]
    clipped = [max(p, _P_FLOOR) for p in raw_p]
    chi2_stat = -2.0 * sum(math.log(p) for p in clipped)
    df = 2 * len(clipped)
    combined_p = float(stats.chi2.sf(chi2_stat, df))

    # ── Direction: score-weighted vote ─────────────────────────
    dir_score = sum(w.get(k, 0.0) * r.score * r.direction for k, r in sub.items())
    direction = _sign(dir_score)

    return CoincidenceResult(
        method="combined",
        score=float(score),
        p_value=combined_p,
        direction=direction,
        detail={
            "sub_results": {
                k: {
                    "score": r.score,
                    "p_value": r.p_value,
                    "direction": r.direction,
                }
                for k, r in sub.items()
            },
            "weights": w,
            "fisher_chi2": round(chi2_stat, 4),
            "fisher_df": df,
        },
    )
