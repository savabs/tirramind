"""
Portfolio concentration — how many independent bets is this, really?

WHAT THIS MODULE IS FOR
-----------------------
A holder knows their weights.  Their broker app shows them weights.  What no
broker app shows is the *structure*: that a 4% position can carry 19% of the
portfolio's variance, or that twelve line items move as three things.  Those
are facts about prices that already happened.  This module computes them.

WHAT THIS MODULE IS NOT
-----------------------
It gives no advice and makes no forecast.  "Your 4% position contributes 19%
of your variance over 2023-01-03..2025-09-25 (n=674)" is a fact.  "You should
trim it" is advice, and nothing here says it.  Historical decompositions are
not predictions: every number below describes one specific past window and
would be different over another.  That is stated, per number, in the output.

THE THREE PUBLIC FUNCTIONS
--------------------------
``effective_bets(panel, weights)``
    How many independent sources of variance the money is actually exposed to.
``risk_contributions(panel, weights)``
    Each holding's share of portfolio variance.  Sums to exactly 1.
``variance_concentration(panel, weights)``
    How few names produce most of the variance.

All three take the *same* two arguments and return plain dicts.  All three
report ``n`` (observations), ``p`` (assets), the literal window dates, and an
``excluded`` list naming every holding left out and why.  Nothing is ever
dropped silently — silent exclusion is how you produce a confident wrong
answer about someone's own money, which they will notice and you will not.

THE PANEL
---------
``panel`` is a ``pandas.DataFrame`` of **daily simple returns**: rows are dates
(a DatetimeIndex), columns are tickers.  Use ``returns_from_prices`` to build
one from a price frame.  Passing prices where returns are expected produces a
plausible-looking and completely wrong answer, so ``_prepare`` refuses a frame
that looks like prices rather than guessing.

WHERE ALL OF THIS MISLEADS
--------------------------
Every function here consumes one sample covariance matrix estimated from ``n``
days of ``p`` assets.  That matrix is noisy, and its noise is worst exactly
where this module is most interesting: eigenvalue spread is *biased upward* by
estimation error (Marchenko-Pastur), so a short window makes a portfolio look
like it has more independent bets than it has.  This is why
``effective_bets`` refuses to headline below ``n >= 10p`` and says so instead.
The headline count is the squared diversification ratio, NOT the PCA-entropy
count that is the usual textbook choice: on real data the PCA version scores a
95%-one-stock portfolio 3.69 bets, higher than the same twelve names held
equally. That counterexample, and the reasoning, is in ``effective_bets``.
Correlations are also not stable; they rise in drawdowns, which is when
diversification is being relied on.  Every function that can be recomputed on
sub-windows is, and the spread across them is reported next to the headline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "returns_from_prices",
    "effective_bets",
    "risk_contributions",
    "variance_concentration",
    "format_report",
    "MIN_OBS_PER_ASSET",
    "HEADLINE_N_OVER_P",
    "USABLE_N_OVER_P",
    "SUBPERIOD_N_OVER_P",
]

# --- Thresholds.  Every one of these is a judgement about estimation error, --
# --- not a law, so each is named, exported, and overridable by the caller. --

#: A column with fewer than this many observations in the requested window is
#: excluded outright.  60 trading days is roughly a quarter; below it a
#: correlation estimate is dominated by its own standard error.
MIN_OBS_PER_ASSET = 60

#: ``effective_bets`` will not headline an eigenvalue-derived number unless
#: n >= 10p.  Below that, the sample eigenvalue spectrum is more a picture of
#: the estimation noise than of the portfolio (Marchenko-Pastur edge grows as
#: (1 + sqrt(p/n))^2, which at n = 2p already inflates the top eigenvalue by
#: ~45% for a matrix that is genuinely identity).
HEADLINE_N_OVER_P = 10.0

#: Below n = 5p, even the linear-in-sigma numbers (risk contributions) are
#: flagged unreliable.  Below n = 2p the covariance matrix is singular and
#: nothing is computed at all.
USABLE_N_OVER_P = 5.0
_SINGULAR_N_OVER_P = 2.0

#: A sub-period estimate is only reported when the sub-period itself clears
#: this ratio.  Otherwise the "stability range" would just be measuring how
#: badly a short window estimates a covariance matrix.
SUBPERIOD_N_OVER_P = 3.0

_SUM_TOL = 1e-9


# --------------------------------------------------------------------------
# panel preparation
# --------------------------------------------------------------------------


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from a price frame.

    Computes ``p_t / p_{t-1} - 1`` and drops the first row (which is NaN by
    construction, not by missing data).

    Misleads when: the price frame is not split/dividend adjusted, in which
    case a corporate action appears as a genuine -50% day and will dominate
    every covariance estimate downstream.  Pass adjusted closes.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a DataFrame, got {type(prices).__name__}")
    rets = prices.astype(float).pct_change()
    return rets.iloc[1:]


def _looks_like_prices(frame: pd.DataFrame) -> bool:
    """True when a frame of supposed returns is almost certainly prices.

    Daily simple returns are signed and small.  A frame with no negative value
    anywhere, or with a typical magnitude above 1.0 (i.e. +100% every day),
    is a price frame that someone forgot to difference.  Catching this is the
    difference between a wrong answer and an error message.
    """
    vals = frame.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return False
    if np.nanmedian(np.abs(finite)) > 1.0:
        return True
    return bool((finite >= 0).all()) and finite.size > 5


def _as_frame(panel: Any) -> pd.DataFrame:
    """Accept a returns DataFrame, or any object exposing ``.returns``."""
    if isinstance(panel, pd.DataFrame):
        return panel
    inner = getattr(panel, "returns", None)
    if isinstance(inner, pd.DataFrame):
        return inner
    raise TypeError(
        "panel must be a pandas DataFrame of daily returns (dates x tickers), "
        f"or an object with a .returns DataFrame; got {type(panel).__name__}"
    )


def _fmt_date(value: Any) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except (ValueError, TypeError):
        return str(value)


class _Prepared:
    """Aligned returns matrix, weights, covariance, and the exclusion ledger."""

    __slots__ = (
        "R",
        "tickers",
        "w",
        "cov",
        "n",
        "p",
        "window_start",
        "window_end",
        "excluded",
        "weight_covered",
        "weight_input_sum",
        "rows_dropped_for_alignment",
    )

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    def provenance(self) -> dict:
        """The block every public function stamps onto its result."""
        return {
            "n_observations": int(self.n),
            "n_assets": int(self.p),
            "n_over_p": round(float(self.n) / float(self.p), 2),
            "window_start": self.window_start,
            "window_end": self.window_end,
            "window": f"{self.window_start}..{self.window_end}",
            "excluded": list(self.excluded),
            "weight_of_portfolio_analysed": round(float(self.weight_covered), 6),
            "input_weights_summed_to": round(float(self.weight_input_sum), 6),
            "rows_dropped_for_alignment": int(self.rows_dropped_for_alignment),
            "assets_analysed": list(self.tickers),
        }


def _prepare(
    panel: Any,
    weights: Mapping[str, float],
    *,
    min_obs: int = MIN_OBS_PER_ASSET,
) -> _Prepared:
    """Align panel and weights, excluding loudly and normalising once.

    Exclusion reasons recorded, in the order they are tested:
      ``no_column_in_panel``      - held, but the panel has no series for it
      ``all_values_missing``      - column present, entirely NaN
      ``insufficient_history``    - fewer than ``min_obs`` observations
      ``zero_variance``           - constant series (a suspended or stale feed)
      ``non_positive_weight``     - weight of exactly zero (not held)

    Misleads when: the surviving names are intersected to a complete-case
    window, so one short-history survivor shortens the window for everybody.
    ``rows_dropped_for_alignment`` reports how many rows that cost.
    """
    frame = _as_frame(panel)
    if frame.empty:
        raise ValueError("panel is empty: no returns to analyse")
    if not isinstance(weights, Mapping):
        raise TypeError(f"weights must be a mapping ticker -> weight, got {type(weights).__name__}")
    if not weights:
        raise ValueError("weights is empty: nothing to analyse")

    if _looks_like_prices(frame):
        raise ValueError(
            "panel looks like PRICES, not returns: no negative values and/or a "
            "typical magnitude above 1.0. Pass returns_from_prices(prices). "
            "Refusing rather than silently returning a wrong decomposition."
        )

    excluded: list[dict] = []
    keep: list[str] = []
    input_sum = float(sum(float(v) for v in weights.values()))

    for ticker, raw_w in weights.items():
        w = float(raw_w)
        if w == 0.0:
            excluded.append({"ticker": ticker, "reason": "non_positive_weight", "detail": "weight is 0"})
            continue
        if ticker not in frame.columns:
            excluded.append(
                {
                    "ticker": ticker,
                    "reason": "no_column_in_panel",
                    "detail": "no return series supplied for this holding",
                    "weight": w,
                }
            )
            continue
        col = pd.to_numeric(frame[ticker], errors="coerce")
        obs = int(col.notna().sum())
        if obs == 0:
            excluded.append(
                {
                    "ticker": ticker,
                    "reason": "all_values_missing",
                    "detail": "column is entirely NaN",
                    "weight": w,
                }
            )
            continue
        if obs < min_obs:
            excluded.append(
                {
                    "ticker": ticker,
                    "reason": "insufficient_history",
                    "detail": f"{obs} observations, need >= {min_obs}",
                    "weight": w,
                }
            )
            continue
        if float(col.std(ddof=1)) == 0.0 or not np.isfinite(float(col.std(ddof=1))):
            excluded.append(
                {
                    "ticker": ticker,
                    "reason": "zero_variance",
                    "detail": "series is constant over the window",
                    "weight": w,
                }
            )
            continue
        keep.append(ticker)

    if not keep:
        raise ValueError(
            "no holding survived preparation; exclusions: "
            + "; ".join(f"{e['ticker']}={e['reason']}" for e in excluded)
        )

    sub = frame.loc[:, keep].apply(pd.to_numeric, errors="coerce")
    rows_before = int(len(sub))
    sub = sub.dropna(axis=0, how="any")
    rows_dropped = rows_before - int(len(sub))

    n = int(len(sub))
    if n < 2:
        raise ValueError(
            f"only {n} date(s) have a complete observation across all "
            f"{len(keep)} surviving holdings; cannot estimate a covariance"
        )

    # A holding can lose its variance *after* alignment.
    post_std = sub.std(ddof=1)
    dead = [t for t in keep if not np.isfinite(float(post_std[t])) or float(post_std[t]) == 0.0]
    for t in dead:
        excluded.append(
            {
                "ticker": t,
                "reason": "zero_variance",
                "detail": "constant across the common (complete-case) window",
                "weight": float(weights[t]),
            }
        )
    if dead:
        keep = [t for t in keep if t not in dead]
        if not keep:
            raise ValueError("no holding retained non-zero variance over the common window")
        sub = sub.loc[:, keep]

    kept_w = np.array([float(weights[t]) for t in keep], dtype=float)
    kept_sum = float(kept_w.sum())
    if kept_sum == 0.0:
        raise ValueError("surviving holdings have weights summing to zero")
    w_norm = kept_w / kept_sum
    covered = kept_sum / input_sum if input_sum != 0 else float("nan")

    R = sub.to_numpy(dtype=float)
    cov = np.cov(R, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    cov = 0.5 * (cov + cov.T)  # kill float asymmetry before eigh

    return _Prepared(
        R=R,
        tickers=list(keep),
        w=w_norm,
        cov=cov,
        n=n,
        p=len(keep),
        window_start=_fmt_date(sub.index[0]),
        window_end=_fmt_date(sub.index[-1]),
        excluded=excluded,
        weight_covered=covered,
        weight_input_sum=input_sum,
        rows_dropped_for_alignment=rows_dropped,
    )


# --------------------------------------------------------------------------
# spectrum helpers
# --------------------------------------------------------------------------


def _eigen(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending eigenvalues/vectors of a symmetric matrix, clipped at 0.

    Tiny negative eigenvalues are float noise on a PSD matrix; anything
    materially negative would be a bug and is clipped here, which is why the
    caller also reports ``min_raw_eigenvalue``.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def _entropy_perplexity(p: np.ndarray) -> float:
    """exp(Shannon entropy) of a probability vector: the 'effective count'."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(np.exp(-np.sum(p * np.log(p))))


def _participation_ratio(p: np.ndarray) -> float:
    """1 / sum(p^2): the inverse-Herfindahl effective count."""
    p = np.asarray(p, dtype=float)
    denom = float(np.sum(p**2))
    if denom <= 0:
        return float("nan")
    return 1.0 / denom


def _enb_from(cov: np.ndarray, w: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Meucci (2009) effective number of bets, plus its participation ratio.

    Rotates the portfolio into the principal-component basis of ``cov``,
    splits total variance across those uncorrelated principal portfolios, and
    measures how evenly the split lands.
    """
    vals, vecs = _eigen(cov)
    vals = np.clip(vals, 0.0, None)
    v = vecs.T @ w  # exposure to each principal portfolio
    contrib = (v**2) * vals
    total = float(contrib.sum())
    if total <= 0:
        return float("nan"), float("nan"), np.full(w.shape, np.nan)
    shares = contrib / total
    return _entropy_perplexity(shares), _participation_ratio(shares), shares


# --------------------------------------------------------------------------
# 1. effective bets
# --------------------------------------------------------------------------


def _diversification_ratio(cov, w):
    """Choueifaty-Coignard diversification ratio: weighted-avg vol / portfolio vol.

    Returns (DR, DR**2).  DR**2 is the headline: see ``effective_bets``.
    """
    sd = np.sqrt(np.diag(cov))
    weighted_avg_vol = float(w @ sd)
    port_vol = float(np.sqrt(w @ cov @ w))
    if port_vol <= 0 or weighted_avg_vol <= 0:
        return float("nan"), float("nan")
    dr = weighted_avg_vol / port_vol
    return dr, dr * dr


def effective_bets(
    panel: Any,
    weights: Mapping[str, float],
    *,
    min_obs: int = MIN_OBS_PER_ASSET,
    variance_threshold: float = 0.90,
    n_subperiods: int = 3,
) -> dict:
    """How many independent bets a portfolio actually carries.

    Three different answers are computed, because they answer three different
    questions and a reader deserves to know which one is on screen.

    HEADLINED: ``effective_bets_diversification`` = DR^2
    ------------------------------------------------------
    The squared diversification ratio (Choueifaty & Coignard 2008):

        DR  = (sum_i w_i * sigma_i) / sigma_portfolio
        DR^2 = effective_bets_diversification

    Read it literally: **the portfolio's volatility is what you would get from
    DR^2 equally-sized, equal-volatility, mutually uncorrelated holdings.**
    That interpretation is not an analogy, it is an identity - for N such
    assets the formula returns exactly N (verified in the tests against
    synthetic data for N = 1, 2, 3, 5, 10, and against 4 identical assets,
    which returns 1).  It is bounded below by 1, reaches 1 for a single
    holding or for perfectly correlated holdings, and is weight-aware.

    REPORTED, NOT HEADLINED: ``effective_bets_pca_entropy``
    -------------------------------------------------------
    Meucci (2009): rotate the portfolio into the principal-component basis of
    the covariance matrix, split variance across those uncorrelated principal
    portfolios (``share_k = (e_k'w)^2 lambda_k / w'Sigma w``), and take
    ``exp(-sum share_k log share_k)``.  Also reported as its participation
    ratio ``1/sum share_k^2``.

    **This is computed and shown, but deliberately not headlined, because it
    fails a case this product will actually be handed.**  Measured on real
    data (12 NSE large caps, 2023-01-03..2025-09-25, n=674): a portfolio of
    95% RELIANCE.NS plus eleven 0.45% crumbs scores **3.69** on PCA-entropy -
    the *highest* of any equity portfolio tested, higher than the same twelve
    names held equally (1.13).  A 95%-one-stock book is not four bets.  The
    cause is that PCA-entropy measures how the portfolio's variance spreads
    across principal components, and a single concentrated stock is a mixture
    of many components, so concentration *raises* the score.  DR^2 scores that
    same portfolio 1.07.  The PCA basis is also one arbitrary choice among
    infinitely many uncorrelated rotations (which is why the literature moved
    on to minimum-torsion bets, Meucci/Santangelo/Deguest 2015).

    REPORTED, WEIGHT-BLIND: ``pc_for_90pct_variance``, ``spectrum_effective_rank``
    -----------------------------------------------------------------------------
    Principal components of the holdings' *correlation* matrix (correlation,
    not covariance, so one high-volatility name cannot dominate the spectrum):
    the count needed to reach ``variance_threshold``, and the participation
    ratio ``(sum lambda)^2 / sum lambda^2`` of the spectrum.  These describe
    the *opportunity set* - how many distinct directions the chosen names span
    - and know nothing about how the money is split across them.  A portfolio
    with 95% in one name has the same value as the equal-weighted version.
    Useful context; not a description of the position.

    WHERE THE HEADLINE MISLEADS
    ---------------------------
    1. Sample eigenvalue spread and sample correlations are both biased by
       estimation error, which inflates apparent diversification.  Below
       ``n >= 10p`` this sets ``headline_supported=False`` and explains
       instead of printing.
    2. DR^2 is defined for long-only books.  With any negative weight the
       numerator ``sum w_i sigma_i`` stops being a weighted average of
       volatilities and the ratio loses its meaning; the headline is withheld
       and ``unsupported_reason`` says so.
    3. It is a *volatility* statement, over one window.  It says the realised
       dispersion behaved like that of DR^2 independent assets between the two
       printed dates.  It does not say the next window will.  Correlations
       rise in drawdowns, so this number is typically at its lowest exactly
       when diversification is being relied on - which is why
       ``subperiod_estimates`` recomputes it on equal sub-windows and the
       spread is reported next to it.
    4. A low-volatility holding (a liquid fund, cash) pulls the numerator down
       and can make DR^2 look modest even across genuinely unrelated assets.
       Read it beside ``portfolio_vol_annualised`` from ``risk_contributions``.
    """
    prep = _prepare(panel, weights, min_obs=min_obs)
    out: dict = prep.provenance()
    out["metric"] = "effective_bets"

    vals_raw, _ = np.linalg.eigh(prep.cov)
    out["min_raw_eigenvalue"] = float(vals_raw.min())
    out["variance_threshold"] = float(variance_threshold)
    out["headline_metric"] = "effective_bets_diversification"
    out["headline_metric_reason"] = (
        "DR^2 returns exactly N for N equal-sized uncorrelated holdings, 1 for "
        "a single holding, and 1 for perfectly correlated holdings. The "
        "PCA-entropy alternative fails the concentrated case: it scores a "
        "95%-one-stock book 3.69 against 1.13 for the same names held equally."
    )

    if prep.n < _SINGULAR_N_OVER_P * prep.p:
        out["headline_supported"] = False
        out["unsupported_reason"] = (
            f"n={prep.n} observations for p={prep.p} assets (n/p="
            f"{prep.n / prep.p:.1f}). The sample covariance matrix is singular "
            f"below n/p={_SINGULAR_N_OVER_P:g}; no number is computed at all."
        )
        for k in (
            "effective_bets_diversification",
            "diversification_ratio",
            "effective_bets_pca_entropy",
            "participation_ratio_weighted_pca",
            "pc_for_90pct_variance",
            "spectrum_effective_rank",
        ):
            out[k] = None
        out["pca_eigenvalue_shares"] = None
        out["subperiod_estimates"] = []
        out["subperiod_range"] = None
        out["subperiod_spread"] = None
        return out

    dr, dr2 = _diversification_ratio(prep.cov, prep.w)
    out["diversification_ratio"] = round(float(dr), 4)
    out["effective_bets_diversification"] = round(float(dr2), 4)

    enb, pr_w, shares = _enb_from(prep.cov, prep.w)
    out["effective_bets_pca_entropy"] = round(float(enb), 4)
    out["participation_ratio_weighted_pca"] = round(float(pr_w), 4)
    out["pca_eigenvalue_shares"] = [round(float(s), 6) for s in shares]
    out["pca_not_headlined_reason"] = (
        "PCA-entropy rises with concentration: measured on 12 NSE large caps "
        "(2023-01-03..2025-09-25, n=674) it scores a 95%-one-stock book 3.69 "
        "versus 1.13 for the same twelve names held equally. It is reported "
        "for completeness, not used as the headline."
    )

    sd = np.sqrt(np.diag(prep.cov))
    corr = prep.cov / np.outer(sd, sd)
    corr = 0.5 * (corr + corr.T)
    cvals, _ = _eigen(corr)
    cvals = np.clip(cvals, 0.0, None)
    cshare = cvals / cvals.sum()
    cum = np.cumsum(cshare)
    out["pc_for_90pct_variance"] = int(np.searchsorted(cum, variance_threshold) + 1)
    out["spectrum_effective_rank"] = round(float(_participation_ratio(cshare)), 4)
    out["correlation_eigenvalue_shares"] = [round(float(s), 6) for s in cshare]

    reasons: list[str] = []
    if prep.n < HEADLINE_N_OVER_P * prep.p:
        reasons.append(
            f"n={prep.n} observations for p={prep.p} assets (n/p="
            f"{prep.n / prep.p:.1f}), below the n/p >= {HEADLINE_N_OVER_P:g} "
            "required to headline a covariance-derived count. At this ratio "
            "estimation error alone spreads the correlation structure, which "
            "biases effective-bets UPWARD: the figure would overstate how "
            "diversified the portfolio has been."
        )
    if bool((prep.w < 0).any()):
        reasons.append(
            "at least one holding has a negative weight. DR^2's numerator is a "
            "weighted average of volatilities, which is not meaningful with "
            "short positions, so the headline is withheld."
        )
    out["headline_supported"] = not reasons
    if reasons:
        out["unsupported_reason"] = " Also: ".join(reasons)

    out["subperiod_estimates"] = _subperiod_bets(prep, n_subperiods)
    vals = [s["effective_bets_diversification"] for s in out["subperiod_estimates"]]
    if len(vals) >= 2:
        out["subperiod_range"] = [round(min(vals), 4), round(max(vals), 4)]
        out["subperiod_spread"] = round(max(vals) - min(vals), 4)
    else:
        out["subperiod_range"] = None
        out["subperiod_spread"] = None
        out["subperiod_note"] = (
            f"a sub-period needs n_sub >= {SUBPERIOD_N_OVER_P:g}p = "
            f"{SUBPERIOD_N_OVER_P * prep.p:.0f} observations; with n={prep.n} "
            f"split {n_subperiods} ways there are not enough observations for "
            "a stability check, so none is claimed."
        )
    return out


def _subperiod_bets(prep: _Prepared, n_subperiods: int) -> list[dict]:
    """Recompute the headline on equal, non-overlapping sub-windows.

    Only sub-windows clearing ``SUBPERIOD_N_OVER_P * p`` are reported; below
    that the spread would be measuring how badly a short window estimates a
    covariance matrix rather than genuine instability in the portfolio.
    """
    if n_subperiods < 2:
        return []
    bounds = np.linspace(0, prep.n, n_subperiods + 1).astype(int)
    floor = SUBPERIOD_N_OVER_P * prep.p
    est: list[dict] = []
    for i in range(n_subperiods):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        n_sub = hi - lo
        if n_sub < floor or n_sub < 2:
            continue
        chunk = prep.R[lo:hi, :]
        sd = chunk.std(axis=0, ddof=1)
        if not np.all(np.isfinite(sd)) or np.any(sd == 0):
            continue
        c = np.atleast_2d(np.cov(chunk, rowvar=False, ddof=1))
        c = 0.5 * (c + c.T)
        _, dr2 = _diversification_ratio(c, prep.w)
        enb, _, _ = _enb_from(c, prep.w)
        est.append(
            {
                "subperiod": i + 1,
                "n_observations": int(n_sub),
                "effective_bets_diversification": round(float(dr2), 4),
                "effective_bets_pca_entropy": round(float(enb), 4),
            }
        )
    return est


# --------------------------------------------------------------------------
# 2. risk contributions
# --------------------------------------------------------------------------


def _check_euler_sum(contrib: np.ndarray) -> float:
    """Assert the Euler risk shares sum to 1, and return the total.

    Extracted from ``risk_contributions`` so it can be tested directly. It
    has to be: the identity ``sum_i w_i (Sigma w)_i / (w' Sigma w) == 1`` is
    homogeneous of degree zero in w and holds for *every* input, so no
    portfolio anyone can paste will ever trip this guard. It exists to catch
    a future edit to the decomposition, which means a test has to call it
    with a bad vector rather than hope some input produces one.
    """
    total = float(np.asarray(contrib, dtype=float).sum())
    if not abs(total - 1.0) < _SUM_TOL:
        raise AssertionError(
            f"risk contributions sum to {total!r}, not 1.0 "
            f"(|error|={abs(total - 1.0):.3e} > {_SUM_TOL:g}). "
            "The Euler decomposition is broken."
        )
    return total


def risk_contributions(
    panel: Any,
    weights: Mapping[str, float],
    *,
    min_obs: int = MIN_OBS_PER_ASSET,
) -> dict:
    """Each holding's share of portfolio risk, which is not its weight.

    Portfolio variance is ``w'Sigma w``.  Euler's theorem splits it exactly:
    holding i's contribution is ``w_i * (Sigma w)_i / (w'Sigma w)``, and the
    p contributions sum to exactly 1.  That identity is asserted, not assumed.

    The same percentages describe contribution to *volatility* as well as to
    variance: portfolio vol is homogeneous of degree one in w, so
    ``sigma_p = sum_i w_i * d(sigma_p)/d(w_i)`` and each term divided by
    ``sigma_p`` gives the same fraction.  So "19% of your risk" is unambiguous.

    This is the surprise.  A holding earns a risk share above its weight by
    being volatile, by being correlated with the rest of the book, or both -
    and those two causes are reported separately (``standalone_vol_annualised``
    and ``correlation_with_portfolio``) so the number is explainable rather
    than magical.

    WHERE THIS MISLEADS
    -------------------
    1. It is a *marginal* decomposition at the current weights, valid for an
       infinitesimal change.  It does not say what variance would be if a
       holding were removed - removing 30% of a portfolio changes Sigma w for
       everything else.  Read it as "where the variance sits now".
    2. A holding negatively correlated with the rest can have a NEGATIVE risk
       share.  That is arithmetic, not an error, and it is reported as-is.
       ``has_negative_contributions`` flags it, because the intuitive reading
       of a percentage breaks down when a part is negative.
    3. Variance treats upside and downside alike.  A holding that contributes
       19% of variance contributed 19% of the *movement*, in both directions.
    4. One window, and correlations move.  Both are stamped on the result.
    """
    prep = _prepare(panel, weights, min_obs=min_obs)
    out: dict = prep.provenance()
    out["metric"] = "risk_contributions"

    if prep.n < _SINGULAR_N_OVER_P * prep.p:
        out["reliable"] = False
        out["reliability_reason"] = (
            f"n={prep.n} for p={prep.p} (n/p={prep.n / prep.p:.1f}): the sample "
            f"covariance is singular below n/p={_SINGULAR_N_OVER_P:g}. No "
            "decomposition computed."
        )
        out["contributions"] = []
        return out

    cov = prep.cov
    w = prep.w
    sigma_w = cov @ w
    port_var = float(w @ sigma_w)
    if port_var <= 0:
        raise ValueError(
            f"portfolio variance computed as {port_var:g}, which is not positive; refusing to divide by it"
        )
    port_vol_d = math.sqrt(port_var)
    contrib = w * sigma_w / port_var

    total = _check_euler_sum(contrib)

    sd_d = np.sqrt(np.diag(cov))
    corr_with_port = sigma_w / (sd_d * port_vol_d)
    ann = math.sqrt(252.0)

    rows: list[dict] = []
    for i, t in enumerate(prep.tickers):
        rows.append(
            {
                "ticker": t,
                "weight": round(float(w[i]), 6),
                "risk_share": round(float(contrib[i]), 6),
                "risk_share_minus_weight": round(float(contrib[i] - w[i]), 6),
                "risk_per_unit_weight": (round(float(contrib[i] / w[i]), 4) if w[i] != 0 else None),
                "standalone_vol_annualised": round(float(sd_d[i] * ann), 6),
                "correlation_with_portfolio": round(float(corr_with_port[i]), 4),
            }
        )
    rows.sort(key=lambda r: r["risk_share"], reverse=True)

    out["contributions"] = rows
    out["sums_to"] = round(total, 12)
    out["portfolio_vol_annualised"] = round(float(port_vol_d * ann), 6)
    out["has_negative_contributions"] = bool((contrib < 0).any())

    amp = max(rows, key=lambda r: r["risk_share_minus_weight"])
    out["largest_amplifier"] = {
        "ticker": amp["ticker"],
        "weight": amp["weight"],
        "risk_share": amp["risk_share"],
        "gap": amp["risk_share_minus_weight"],
    }
    dmp = min(rows, key=lambda r: r["risk_share_minus_weight"])
    out["largest_dampener"] = {
        "ticker": dmp["ticker"],
        "weight": dmp["weight"],
        "risk_share": dmp["risk_share"],
        "gap": dmp["risk_share_minus_weight"],
    }

    reliable = prep.n >= USABLE_N_OVER_P * prep.p
    out["reliable"] = bool(reliable)
    if not reliable:
        out["reliability_reason"] = (
            f"n={prep.n} for p={prep.p} (n/p={prep.n / prep.p:.1f}), below "
            f"n/p >= {USABLE_N_OVER_P:g}. Off-diagonal covariance entries carry "
            "standard errors comparable to their own size at this ratio; the "
            "ordering of the contributions is not dependable."
        )
    return out


# --------------------------------------------------------------------------
# 3. variance concentration
# --------------------------------------------------------------------------


def variance_concentration(
    panel: Any,
    weights: Mapping[str, float],
    *,
    min_obs: int = MIN_OBS_PER_ASSET,
    thresholds: Sequence[float] = (0.5, 0.8, 0.9),
) -> dict:
    """How few names produce most of the variance.

    Takes the Euler risk shares from ``risk_contributions``, sorts them, and
    reports how many names are needed to reach each threshold, plus two
    inverse-Herfindahl effective counts: one over weights (what a broker app
    would show) and one over risk shares (what is actually going on).  The gap
    between those two counts is the whole point of this module.

    WHERE THIS MISLEADS
    -------------------
    1. Inherits everything ``risk_contributions`` inherits: one window,
       marginal decomposition, variance is symmetric about zero.
    2. ``effective_n_by_risk`` is only defined when every risk share is
       non-negative.  With a negative contributor, 1/sum(r^2) can exceed p and
       stops meaning "effective count"; it is returned as None with a reason
       rather than printed as a number that looks fine and is nonsense.
    3. ``names_for_50pct`` is a step function of a noisy estimate.  When two
       holdings sit either side of the cutoff it can move by one on a
       re-estimate without anything real having changed.
    """
    rc = risk_contributions(panel, weights, min_obs=min_obs)
    out: dict = {k: v for k, v in rc.items() if k != "contributions"}
    out["metric"] = "variance_concentration"

    rows = rc.get("contributions") or []
    if not rows:
        out["thresholds"] = {}
        out["effective_n_by_risk"] = None
        out["effective_n_by_weight"] = None
        return out

    shares = np.array([r["risk_share"] for r in rows], dtype=float)
    ws = np.array([r["weight"] for r in rows], dtype=float)
    order = np.argsort(shares)[::-1]
    shares_sorted = shares[order]
    tickers_sorted = [rows[i]["ticker"] for i in order]
    cum = np.cumsum(shares_sorted)

    th: dict = {}
    for t in thresholds:
        idx = int(np.searchsorted(cum, float(t)))
        if idx >= len(cum):
            th[f"{int(round(float(t) * 100))}pct"] = {
                "n_names": None,
                "reason": (
                    f"cumulative risk share never reaches {float(t):.0%} "
                    f"(tops out at {float(cum[-1]):.4f}); negative contributions present"
                ),
            }
        else:
            th[f"{int(round(float(t) * 100))}pct"] = {
                "n_names": idx + 1,
                "names": tickers_sorted[: idx + 1],
                "cumulative_risk_share": round(float(cum[idx]), 6),
                "cumulative_weight": round(float(ws[order][: idx + 1].sum()), 6),
            }
    out["thresholds"] = th

    out["top_1_risk_share"] = round(float(shares_sorted[0]), 6)
    out["top_1_ticker"] = tickers_sorted[0]
    k3 = min(3, len(shares_sorted))
    out["top_3_risk_share"] = round(float(shares_sorted[:k3].sum()), 6)
    out["top_3_tickers"] = tickers_sorted[:k3]
    out["top_3_weight"] = round(float(ws[order][:k3].sum()), 6)

    out["hhi_weight"] = round(float(np.sum(ws**2)), 6)
    out["effective_n_by_weight"] = round(float(_participation_ratio(ws)), 4)

    if bool((shares < 0).any()):
        out["hhi_risk"] = None
        out["effective_n_by_risk"] = None
        out["effective_n_by_risk_reason"] = (
            "at least one holding has a negative risk contribution, so the "
            "inverse-Herfindahl of the shares is not an effective count"
        )
    else:
        out["hhi_risk"] = round(float(np.sum(shares**2)), 6)
        out["effective_n_by_risk"] = round(float(_participation_ratio(shares)), 4)

    out["cumulative_risk_curve"] = [
        {"rank": i + 1, "ticker": tickers_sorted[i], "cumulative_risk_share": round(float(c), 6)}
        for i, c in enumerate(cum)
    ]
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

#: Phrases that would turn a statement of fact into personalised investment
#: advice, which is a regulated activity (SEBI Investment Adviser in India)
#: and not one this project is licensed for. ``format_report``'s output is
#: tested against this list.
#:
#: Note what is NOT here: "diversification". Naming the metric
#: ("diversification ratio 1.63") is a label on an arithmetic quantity.
#: Judging it ("you are under-diversified") is the regulated act, and the
#: judgement forms are listed individually below. Banning the stem outright
#: would have forced the metric to be renamed into something less honest.
_ADVICE_WORDS = (
    # imperatives and recommendations
    "you should",
    "we recommend",
    "recommend",
    "you could",
    "consider ",
    "advis",
    "suggest",
    "ought to",
    "need to",
    "worth ",
    # transactions
    "buy ",
    "sell ",
    "trim",
    "rebalance",
    "reduce your",
    "increase your",
    "add to your",
    "cut your",
    "hedge your",
    "swap ",
    # judgements about the holder's position
    "under-diversified",
    "underdiversified",
    "poorly diversified",
    "well diversified",
    "well-diversified",
    "not diversified",
    "over-concentrated",
    "overconcentrated",
    "too much",
    "too few",
    "too many",
    "too little",
    "too high",
    "too low",
    "risky",
    "safer",
    "unsafe",
    "better",
    "worse",
    "healthy",
    "unhealthy",
    "optimal",
    "suboptimal",
    "ideal",
    "problem",
    "warning",
    "danger",
)


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _spct(x: float) -> str:
    """Signed percent, so an amplifier and a dampener are distinguishable."""
    return f"{100.0 * float(x):+.1f}%"


def _coverage_pct(x: float) -> str:
    """Percent that never rounds up to 100.0% unless it is exactly 100%.

    Printing "these figures cover 100.0% of the portfolio" directly beneath a
    list of excluded holdings is the precise shape of confident-wrong-number
    this module exists to avoid, and 0.999994 rounds there by default.
    """
    v = float(x)
    if v >= 1.0:
        return "100%"
    if v > 0.9995:
        return ">99.9%"
    return f"{100.0 * v:.1f}%"


def format_report(
    panel: Any,
    weights: Mapping[str, float],
    *,
    min_obs: int = MIN_OBS_PER_ASSET,
) -> str:
    """The three computations rendered as plain text, stating facts only.

    Every figure carries its window and its n.  No sentence in the output
    tells the reader what to do or what will happen next; ``_ADVICE_WORDS``
    is the list this is tested against.
    """
    eb = effective_bets(panel, weights, min_obs=min_obs)
    rc = risk_contributions(panel, weights, min_obs=min_obs)
    vc = variance_concentration(panel, weights, min_obs=min_obs)

    L: list[str] = []
    n, p = eb["n_observations"], eb["n_assets"]
    held = len(weights)

    if eb["headline_supported"]:
        dr2 = eb["effective_bets_diversification"]
        L.append(f"You hold {held} positions. Over this window they behaved like {dr2:.1f}.")
        L.append("")
        L.append(f"  Your {p} holdings had the volatility of {dr2:.2f} equally-sized,")
        L.append("  equal-volatility, mutually uncorrelated holdings.")
        L.append("")
        L.append(f"  Effective bets (DR^2, headline)      {dr2:.2f}   of {p} holdings analysed")
        L.append(f"  Diversification ratio                {eb['diversification_ratio']:.2f}")
        if eb.get("subperiod_estimates"):
            parts = ", ".join(
                f"{s['effective_bets_diversification']:.2f} (n={s['n_observations']})"
                for s in eb["subperiod_estimates"]
            )
            L.append(f"  Same figure by sub-period            {parts}")
        L.append("")
        L.append("  Other counts, computed and shown for completeness:")
        L.append(
            f"    PCA-entropy bets                   {eb['effective_bets_pca_entropy']:.2f}"
            f"   (not headlined - rises with concentration)"
        )
        L.append(f"    PCA-entropy participation ratio    {eb['participation_ratio_weighted_pca']:.2f}")
        L.append(
            f"    Components for 90% of variance     {eb['pc_for_90pct_variance']}"
            f"      (weight-blind: the names, not the position)"
        )
        L.append(f"    Effective rank of the spectrum     {eb['spectrum_effective_rank']:.2f}   (weight-blind)")
    else:
        L.append(f"You hold {held} positions. The effective-bets figure is not shown.")
        L.append("")
        L.append(f"  Why: {eb['unsupported_reason']}")

    L.append("")
    L.append(f"Window {eb['window']}  ·  n={n} trading days  ·  p={p} holdings  ·  n/p={eb['n_over_p']}")
    L.append("")

    L.append("Share of portfolio variance, by holding")
    L.append(f"  {'holding':<16}{'weight':>9}{'risk':>9}{'gap':>9}{'own vol':>10}{'corr':>7}")
    for r in rc["contributions"]:
        L.append(
            f"  {r['ticker']:<16}{_pct(r['weight']):>9}{_pct(r['risk_share']):>9}"
            f"{_spct(r['risk_share_minus_weight']):>9}"
            f"{_pct(r['standalone_vol_annualised']):>10}"
            f"{r['correlation_with_portfolio']:>7.2f}"
        )
    L.append(f"  {'':<16}{_pct(sum(r['weight'] for r in rc['contributions'])):>9}{_pct(rc['sums_to']):>9}")
    if not rc["reliable"]:
        L.append(f"  Ordering flagged: {rc['reliability_reason']}")

    amp = rc["largest_amplifier"]
    L.append("")
    L.append(
        f"{amp['ticker']} is {_pct(amp['weight'])} of the money and "
        f"{_pct(amp['risk_share'])} of the variance over this window."
    )
    dmp = rc["largest_dampener"]
    L.append(f"{dmp['ticker']} is {_pct(dmp['weight'])} of the money and {_pct(dmp['risk_share'])} of the variance.")

    L.append("")
    for key in ("50pct", "80pct", "90pct"):
        blk = vc["thresholds"].get(key)
        if not blk:
            continue
        if blk.get("n_names") is None:
            L.append(f"  {key[:-3]}% of variance: {blk['reason']}")
        else:
            L.append(
                f"  {blk['n_names']} of {p} holdings carry {key[:-3]}% of the variance "
                f"({', '.join(blk['names'])}) — {_pct(blk['cumulative_weight'])} of the money"
            )
    if vc["effective_n_by_risk"] is not None and eb["headline_supported"]:
        L.append("")
        L.append("Three counts that sound alike and are not")
        L.append(f"  Money spread across holdings (size only)      {vc['effective_n_by_weight']:.2f} of {p}")
        L.append(f"  Variance spread across holdings (size only)   {vc['effective_n_by_risk']:.2f} of {p}")
        L.append(f"  Independent bets, after co-movement           {eb['effective_bets_diversification']:.2f} of {p}")
        L.append("  The first two count holdings and ignore whether they move")
        L.append(f"  together. The third counts directions: {p} overlapping names gave")
        L.append(f"  the volatility of {eb['effective_bets_diversification']:.1f} that do not overlap at all.")

    if eb["excluded"]:
        L.append("")
        L.append("Excluded from every figure above")
        for e in eb["excluded"]:
            L.append(f"  {e['ticker']:<16}{e['reason']:<24}{e['detail']}")
        L.append(
            f"  These figures cover {_coverage_pct(eb['weight_of_portfolio_analysed'])} "
            "of the portfolio by weight; the rest is not represented above."
        )
    if eb["rows_dropped_for_alignment"]:
        L.append(f"  {eb['rows_dropped_for_alignment']} dates dropped: not every holding had an observation on them.")

    L.append("")
    L.append(
        "Every figure above describes price movement between "
        f"{eb['window_start']} and {eb['window_end']} and nothing after it. "
        "Correlations change; these are not forecasts."
    )
    return "\n".join(L)


def _advice_words_in(text: str) -> list[str]:
    """Which forbidden words a rendered report contains. Used by the tests."""
    low = text.lower()
    return [wd for wd in _ADVICE_WORDS if wd in low]
