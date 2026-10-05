"""
TirraMind — Verification Layer: statistical primitives (Layer 2, stateless math).

WHAT THIS MODULE COMPUTES
    The five statistical operations a pattern-verification verdict is built out
    of. Pure functions over arrays: no database, no I/O, no global state, no
    clock. Every one of them is independently checkable by hand, and every one
    of them is a place where a verification product can lie to a customer.

      * ``benjamini_hochberg``     — false-discovery-rate correction. 51 cells
        tested at alpha=0.05 expect ~2.5 false positives; the uncorrected
        p-value of the best cell is not the p-value of the study.
      * ``block_bootstrap_ci``     — confidence interval and two-sided p-value
        for a statistic of a SERIALLY CORRELATED sample. An i.i.d. bootstrap on
        overlapping forward returns understates the interval.
      * ``causal_zscore``          — backward-only standardisation. The F-04
        guard for the statistical half: the point being scored never enters its
        own mean or standard deviation.
      * ``power_estimate``         — the probability this test would have
        detected a stipulated effect. "No detectable effect" is a claim about
        the test; "no effect" is a claim about the world. Only the first is ever
        earned, and only a power figure tells the customer which they bought.
      * ``effective_sample_size``  — pseudo-replication diagnostic. 123 events
        falling on 67 distinct weeks are not 123 independent observations, and
        the published CFTC null lost a factor of five in its headline p-value
        once that was taken seriously.

WHERE THIS MISLEADS (read before quoting any number to a customer)
    1. None of these functions can see WHERE their input came from. A perfectly
       causal z-score computed over a series that was itself assembled with a
       forward-looking join is still leakage. This module guards the arithmetic,
       not the data pipeline (LESSONS F-04).
    2. ``benjamini_hochberg`` controls the false-discovery rate over the family
       you hand it. Hand it 51 of the 54 cells you ran and the correction is a
       fiction: the three you dropped were part of the search. The family must
       be every test performed, including the ones abandoned for being
       uninteresting.
    3. A bootstrap interval is a statement about resampling the sample you have.
       It cannot repair a biased sample, a survivorship filter, or a
       specification that cancels a signed effect by construction. It quantifies
       sampling noise and nothing else.
    4. ``power_estimate`` is only as meaningful as the effect size stipulated
       into it. It is a conditional statement — "if an effect of this size were
       real" — and the effect size is an assumption, never a measurement. Never
       estimate it from the same data you are testing; that is circular and
       produces post-hoc power, which is a known statistical fallacy.
    5. ``effective_sample_size`` counts calendar collisions. Two events in
       different weeks can still be the same event (two correlated contracts
       reacting to one macro release a fortnight apart). Distinct clusters are
       an UPPER bound on independence, never a proof of it.

ZERO DISCIPLINE (the rule this module is built around)
    A zero that means "no effect" and a zero that means "could not compute"
    must never be the same value. So:

      * ``causal_zscore`` returns ``None`` when the z-score is not defined
        (too little history, or zero-variance history). It never returns 0.0
        to mean "could not compute", and it never returns NaN.
      * ``benjamini_hochberg`` returns ``n_tested`` alongside ``n_rejected``.
        ``n_rejected == 0`` with ``n_tested == 0`` means NOTHING WAS TESTED;
        it does not mean nothing survived.
      * ``block_bootstrap_ci`` raises rather than return a p-value from a
        degenerate bootstrap distribution, and floors its p at
        ``1 / n_bootstrap`` because a bootstrap cannot resolve a smaller one.
        A returned p equal to that floor means "at or below the resolution of
        this bootstrap", never "exactly zero".
      * Every function raises ``ValueError`` on NaN or infinite input rather
        than propagating it into a number that looks computed. The reference
        implementation (``scripts/cftc_event_study.py``) would silently return
        NaN from a NaN-containing series; this module will not.

RELATION TO THE REFERENCE IMPLEMENTATION
    ``scripts/cftc_event_study.py`` is the reference and is not modified.
    ``causal_zscore`` is a bit-for-bit replica of its ``zscore_causal`` for
    finite input, including the ``+ 1e-12`` in the denominator and the
    ``MIN_HISTORY = 20`` floor. ``benjamini_hochberg`` reproduces the published
    ``p_BH`` column of ``docs/publications/cot_null_result.md`` exactly (all 51
    cells, tested). Two deliberate divergences, both documented at the function:
    the NaN handling above, and the p-value resampling scheme in
    ``block_bootstrap_ci``: the reference drew its CI from a block bootstrap and
    its p-value from an i.i.d. one, whereas the default here draws both from the
    block resample. The published robustness section made the same move on a
    different correlation axis — it replaced the i.i.d. event resample with a
    cluster resample over as-of weeks — and it cost the headline p-value a
    factor of five, which is a fair measure of what an i.i.d. p is worth on a
    correlated sample. ``p_resample="iid"`` reproduces the reference exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
from scipy.stats import norm

from agent.quant.scoring import block_bootstrap_ci as _quant_block_bootstrap_ci

__all__ = [
    "BHResult",
    "BootstrapResult",
    "MIN_HISTORY",
    "benjamini_hochberg",
    "block_bootstrap_ci",
    "causal_zscore",
    "effective_sample_size",
    "power_estimate",
    "sample_size_for_power",
]

# Matches MIN_HISTORY in scripts/cftc_event_study.py, which in turn matches the
# _zscore_anomaly floor in the live digest. Changing it changes which events
# exist, so it is a named constant rather than a magic default.
MIN_HISTORY = 20

_ZERO_STD = 1e-12


class BHResult(NamedTuple):
    """Outcome of a Benjamini-Hochberg correction.

    ``p_adjusted`` and ``rejected`` are in the INPUT order, not sorted order,
    so they can be zipped straight back onto the rows that produced them.

    ``n_tested`` is carried explicitly so that "nothing survived" can never be
    confused with "nothing was tested": both give ``n_rejected == 0``.
    """

    p_adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float
    n_tested: int
    n_rejected: int
    largest_rejected_p: float | None


class BootstrapResult(NamedTuple):
    """Point estimate, confidence bounds and two-sided p-value.

    Exactly four fields, so ``point, lo, hi, p = block_bootstrap_ci(...)``
    works. The parameters that produced them are the caller's own arguments and
    are deliberately not echoed here.
    """

    point: float
    lo: float
    hi: float
    p: float


def _as_1d_finite(values: Any, name: str) -> np.ndarray:
    """Coerce to a 1-D float array, raising on anything unusable.

    Raises ValueError on a non-numeric element, a non-1-D shape, or any NaN /
    infinite entry. Returning a number computed from a NaN-contaminated array is
    the failure mode this guard exists to prevent.
    """
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: not coercible to a float array ({exc})") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D sequence, got shape {arr.shape}")
    if arr.size and not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(
            f"{name}: {n_bad} of {arr.size} entries are NaN or infinite. "
            "Filter them at the call site and say how many you dropped; this "
            "function will not turn missing data into a number."
        )
    return arr


# --------------------------------------------------------------------------- #
# 1. Multiple testing
# --------------------------------------------------------------------------- #
def benjamini_hochberg(
    pvalues: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> BHResult:
    """Benjamini-Hochberg step-up false-discovery-rate correction.

    WHAT IT COMPUTES
        For m p-values sorted ascending as p_(1) <= ... <= p_(m):

            adjusted p_(k) = min over j >= k of  min(1, m / j * p_(j))

        (the running minimum from the largest rank down is what enforces
        monotonicity, so a small p is never adjusted above a larger one), and
        rejects ranks 1..K where K is the largest k with p_(k) <= k / m * alpha.
        That rejection rule is the canonical BH procedure and is used in
        preference to ``adjusted <= alpha`` so that a float rounding at the
        boundary cannot flip a verdict.

        Identical to ``statsmodels.stats.multitest.multipletests(method="fdr_bh")``
        — cross-checked in the tests — but with numpy only.

    WHAT IT CONTROLS
        The expected proportion of FALSE discoveries among the rejections, at
        level alpha. Not the probability of any false positive (that is
        Bonferroni's family-wise rate), and not a per-test error rate.

    WHERE IT MISLEADS
        * The correction is only as honest as the family. Every test you ran
          belongs in ``pvalues``, including the ones you discarded for looking
          uninteresting and the ones you ran on an earlier specification.
          Dropping tests shrinks m and inflates significance silently.
        * BH assumes the p-values are valid (uniform under the null). A
          bootstrap p from 9 serially correlated observations is not, and no
          correction repairs that.
        * A non-rejection is not evidence of absence. Read it together with
          ``power_estimate``; the published CFTC null had ~10% power, so its 0
          of 51 was close to preordained.
        * ``p_adjusted`` values are NOT p-values of the individual hypotheses.
          They are the smallest alpha at which that hypothesis would be
          rejected by the whole procedure, and they are not comparable across
          families of different size.

    Parameters
    ----------
    pvalues
        Every p-value in the family, in any order. Must all be finite and in
        [0, 1]. An empty family is accepted and returns ``n_tested == 0``.
    alpha
        Target false-discovery rate, strictly between 0 and 1.

    Returns
    -------
    BHResult
        ``p_adjusted`` and ``rejected`` in input order, plus ``n_tested``,
        ``n_rejected`` and ``largest_rejected_p`` (``None`` when nothing was
        rejected — distinguishable from a rejected p of 0.0).

    Raises
    ------
    ValueError
        On a NaN / infinite p-value, a p-value outside [0, 1], a non-1-D input,
        or an alpha outside (0, 1). Never returns a silently-degraded result:
        the reference study filtered NaN cells out BEFORE correcting (and
        reported them as "not testable"), which is the behaviour to copy.
    """
    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")

    p = _as_1d_finite(pvalues, "pvalues")
    m = int(p.size)
    if m == 0:
        # Not an error, but n_tested == 0 makes it impossible to read the empty
        # rejection list as "nothing survived".
        return BHResult((), (), float(alpha), 0, 0, None)
    if np.any(p < 0.0) or np.any(p > 1.0):
        bad = float(p[np.argmax((p < 0.0) | (p > 1.0))])
        raise ValueError(f"pvalues must lie in [0, 1]; found {bad!r}")

    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    scaled = p_sorted * m / ranks
    adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)

    below = p_sorted <= ranks / m * alpha
    k = int(np.nonzero(below)[0][-1]) + 1 if bool(below.any()) else 0
    rej_sorted = np.zeros(m, dtype=bool)
    rej_sorted[:k] = True

    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    rej = np.zeros(m, dtype=bool)
    rej[order] = rej_sorted

    return BHResult(
        p_adjusted=tuple(float(v) for v in adj),
        rejected=tuple(bool(v) for v in rej),
        alpha=float(alpha),
        n_tested=m,
        n_rejected=k,
        largest_rejected_p=float(p_sorted[k - 1]) if k else None,
    )


# --------------------------------------------------------------------------- #
# 2. Block bootstrap
# --------------------------------------------------------------------------- #
def block_bootstrap_ci(
    sample: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    block_len: int | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
    null_value: float = 0.0,
    p_resample: Literal["block", "iid"] = "block",
    p_seed: int | None = None,
    min_n: int = 3,
) -> BootstrapResult:
    """Circular block bootstrap: point estimate, percentile CI, two-sided p.

    IMPLEMENTATION NOTE — THIS IS A WRAPPER, NOT A REIMPLEMENTATION.
        The resampling is done by ``agent.quant.scoring.block_bootstrap_ci``.
        No index-generation logic is duplicated here. The bootstrap
        distribution needed for the p-value is captured by passing that
        function a recording wrapper around ``statistic``, which means the
        p-value and the interval come from the SAME draws rather than two
        independent bootstraps. That coupling relies on the wrapped function
        calling the metric once for the point estimate and then exactly
        ``n_bootstrap`` times; the contract is asserted, and a mismatch raises
        ``RuntimeError`` instead of computing a p from the wrong distribution.

    WHY BLOCKS
        Forward returns over overlapping horizons are serially correlated, and
        events cluster on the calendar. An i.i.d. resample treats each
        observation as an independent draw, which overstates the sample and
        understates the interval. Circular blocks of length ``block_len``
        (default ``round(n ** (1/3))``, capped at ``n // 2`` by the wrapped
        function) keep runs of adjacent observations together.

    THE NULL MATTERS MORE THAN THE INTERVAL
        ``null_value`` is the value the bootstrap distribution is tested
        against, and its default of 0.0 is usually WRONG for a return study.
        The published CFTC study compared each event group against the
        UNCONDITIONAL mean return of every eligible entry, not against zero —
        an asset with a positive drift produces "significant" positive returns
        against a zero null on any subsample whatsoever. Pass the baseline.

    WHERE IT MISLEADS
        * Percentile bootstrap CIs are not bias-corrected (no BCa here) and are
          poor for strongly skewed statistics at small n.
        * With ``p_resample="block"`` the p-value is a two-sided bootstrap
          p: ``2 * min(P(boot <= null), P(boot >= null))``, capped at 1. It is
          not a t-test p and has no closed form.
        * Blocks handle SERIAL correlation within one series. They do not
          handle cross-sectional clustering — several correlated contracts
          firing in the same week. That needs a cluster bootstrap over weeks;
          ``effective_sample_size`` tells you whether you need one (in the
          published study it cost a factor of five in the headline p-value).
        * The p is floored at ``1 / n_bootstrap``. A bootstrap of B draws
          cannot resolve a p below 1/B, so the floor means "at or below this
          resolution", not "zero". Raise ``n_bootstrap`` to resolve further.
        * ``p_resample="iid"`` exists only to reproduce the published Section 4
          column, which drew the interval from blocks and the p from an i.i.d.
          resample (``p_seed=7`` there). It is not the honest default.

    Parameters
    ----------
    sample
        1-D sample, all entries finite. NaN raises rather than being dropped.
    statistic
        ``f(array) -> float``, e.g. ``numpy.mean``. Called ``n_bootstrap + 1``
        times. Must be deterministic; a stochastic statistic makes the recorded
        distribution meaningless.
    block_len
        Block length, or ``None`` for the wrapped default of ``n ** (1/3)``.
    n_bootstrap, seed, confidence
        Passed through. ``seed`` makes the result reproducible.
    null_value
        The comparison value for the p-value. See above: pass a baseline.
    p_resample
        ``"block"`` (default, same draws as the CI) or ``"iid"``.
    p_seed
        Only used when ``p_resample="iid"``; defaults to ``seed``.
    min_n
        Refuse samples smaller than this. Default 3, matching the reference
        study's own ``len(ev) >= 3`` guard. A bootstrap of n=1 or n=2 returns
        a confident-looking interval of zero or near-zero width and a p-value
        of 0, which is the single most dangerous output this module could
        produce, so it is refused rather than returned.

    Returns
    -------
    BootstrapResult
        ``(point, lo, hi, p)`` — unpacks as a 4-tuple.

    Raises
    ------
    ValueError
        NaN / infinite input, non-1-D input, ``n < min_n``, a non-finite
        ``null_value``, a bad ``confidence`` / ``n_bootstrap`` / ``block_len`` /
        ``p_resample``, or a DEGENERATE bootstrap distribution (every resample
        gave the same value — the case of an all-identical sample). A degenerate
        distribution yields an interval of zero width and a p of exactly 0 or 1
        by construction, which would read as overwhelming evidence.
    RuntimeError
        The wrapped resampler did not call the statistic the expected number of
        times, or the statistic returned a non-finite value on some resample.
    """
    x = _as_1d_finite(sample, "sample")
    n = int(x.size)
    if min_n < 3:
        raise ValueError(f"min_n must be at least 3, got {min_n!r}")
    if n < min_n:
        raise ValueError(
            f"sample has n={n}, below min_n={min_n}: a bootstrap this small "
            "produces a zero-width interval and p=0 by construction. Report "
            "the cell as not testable (as the reference study does) instead."
        )
    if not math.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence!r}")
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be at least 2, got {n_bootstrap!r}")
    if block_len is not None and block_len < 1:
        raise ValueError(f"block_len must be at least 1 or None, got {block_len!r}")
    if not math.isfinite(null_value):
        raise ValueError(f"null_value must be finite, got {null_value!r}")
    if p_resample not in ("block", "iid"):
        raise ValueError(f"p_resample must be 'block' or 'iid', got {p_resample!r}")

    recorded: list[float] = []

    def _recording(arr: np.ndarray) -> float:
        value = float(statistic(arr))
        recorded.append(value)
        return value

    point, lo, hi = _quant_block_bootstrap_ci(
        x,
        _recording,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        block_length=block_len,
        seed=seed,
    )

    if len(recorded) != n_bootstrap + 1 or recorded[0] != point:
        raise RuntimeError(
            "agent.quant.scoring.block_bootstrap_ci no longer calls the metric "
            f"once for the point estimate then {n_bootstrap} times for the "
            f"resamples (saw {len(recorded)} calls). The recorded bootstrap "
            "distribution cannot be trusted, so no p-value is returned. Fix "
            "this wrapper rather than the assertion."
        )

    boot = np.asarray(recorded[1:], dtype=float)
    _check_bootstrap_distribution(boot, "block")

    if p_resample == "block":
        dist = boot
    else:
        rng = np.random.default_rng(seed if p_seed is None else p_seed)
        dist = np.asarray(
            [float(statistic(rng.choice(x, size=n, replace=True))) for _ in range(n_bootstrap)],
            dtype=float,
        )
        _check_bootstrap_distribution(dist, "iid")

    p_low = float(np.mean(dist <= null_value))
    p_high = float(np.mean(dist >= null_value))
    p_value = min(1.0, 2.0 * min(p_low, p_high))
    # A bootstrap of B draws cannot resolve a p below 1/B. Reporting 0.0 would
    # claim a precision the method does not have.
    p_value = max(p_value, 1.0 / n_bootstrap)

    return BootstrapResult(point=float(point), lo=float(lo), hi=float(hi), p=float(p_value))


def _check_bootstrap_distribution(dist: np.ndarray, label: str) -> None:
    """Refuse a bootstrap distribution that cannot support a p-value."""
    if not np.all(np.isfinite(dist)):
        n_bad = int(np.count_nonzero(~np.isfinite(dist)))
        raise RuntimeError(
            f"statistic returned a non-finite value on {n_bad} of {dist.size} "
            f"{label} resamples; the p-value would be computed from a partial "
            "distribution."
        )
    if float(dist.min()) == float(dist.max()):
        raise ValueError(
            f"the {label} bootstrap distribution is degenerate (every one of "
            f"{dist.size} resamples gave {float(dist[0])!r}). Its interval has "
            "zero width and its p-value is exactly 0 or 1 by construction, not "
            "by evidence. This is what an all-identical sample looks like."
        )


# --------------------------------------------------------------------------- #
# 3. Causal (backward-only) z-score — the F-04 guard
# --------------------------------------------------------------------------- #
def causal_zscore(
    history: Sequence[float] | np.ndarray,
    *,
    min_history: int = MIN_HISTORY,
) -> float | None:
    """Standardise the LAST point of a series against the points before it.

    WHAT IT COMPUTES
        With ``x = history`` and ``h = x[:-1]``::

            z = (x[-1] - mean(h)) / (std(h) + 1e-12)

        ``std`` is the population standard deviation (numpy's default, ddof=0),
        and the ``1e-12`` is carried from the reference implementation so the
        numbers are bit-identical to the published study rather than merely
        close.

    WHY THE ARGUMENT IS NAMED "history"
        It is the whole series UP TO AND INCLUDING the point being scored, in
        ascending time order. The last element is the observation under test;
        every earlier element is its history. The current point does not enter
        its own mean or standard deviation — that omission is the entire point
        of this function. Including it (an "expanding window" that contains the
        present) shrinks every extreme z toward zero and, worse, makes the z of
        a past point depend on data that did not exist yet. That is LESSONS
        F-04, and it invalidates every downstream number: event selection,
        forward returns, p-values, the verdict.

        A direct consequence, and the property to test: for any k,
        ``causal_zscore(x[:k])`` is unchanged by anything in ``x[k:]``.

    WHAT IT DOES NOT DO
        * It does not know the publication lag. A z-score can be computable
          from data as of Tuesday while the data only became public on Friday.
          Entry timing is the caller's problem (the reference study adds three
          days before it is allowed to trade).
        * It does not detrend or deseasonalise. On a trending series the last
          point is extreme almost by construction, so a large |z| may be a
          statement about the trend, not about an event.
        * With an expanding history the denominator keeps changing, so z values
          from early and late in a series are not on the same scale.

    Parameters
    ----------
    history
        Ascending-time 1-D series, last element = point under test. All entries
        must be finite.
    min_history
        Minimum length of ``history`` (INCLUDING the current point) before a
        z-score is returned. Default ``MIN_HISTORY`` (20), matching
        ``scripts/cftc_event_study.py``, which therefore standardises against
        at least 19 prior points.

    Returns
    -------
    float | None
        ``None`` — never 0.0, never NaN — when the z-score is undefined:
        ``len(history) < min_history``, or the history has no variance. A
        returned 0.0 means the point sits exactly on its historical mean.

    Raises
    ------
    ValueError
        On NaN / infinite entries or a non-1-D input. This is a deliberate
        divergence from the reference, which would have propagated NaN into a
        comparison (``abs(nan) >= 2.0`` is False) and silently dropped the
        event instead of reporting it.
    """
    if min_history < 2:
        raise ValueError(f"min_history must be at least 2, got {min_history!r}")
    x = _as_1d_finite(history, "history")
    if x.size < min_history:
        return None
    hist = x[:-1]
    std = float(np.std(hist))
    if std < _ZERO_STD:
        return None
    return float((x[-1] - float(np.mean(hist))) / (std + _ZERO_STD))


# --------------------------------------------------------------------------- #
# 4. Power
# --------------------------------------------------------------------------- #
def power_estimate(
    n_events: int,
    effect_size: float,
    sigma: float,
    alpha: float = 0.05,
) -> float:
    """Probability of detecting an effect of the stipulated size, if it is real.

    WHAT IT COMPUTES
        Power of a two-sided one-sample test of a mean, normal approximation::

            ncp   = effect_size * sqrt(n_events) / sigma      (expected t)
            power = Phi(ncp - z_crit) + Phi(-ncp - z_crit),  z_crit = Phi^-1(1 - alpha/2)

        The second term is the probability of rejecting in the WRONG tail; it is
        negligible for large effects and is the reason power never falls below
        alpha.

        This reproduces the published power table in
        ``docs/publications/cot_null_result.md``: a reference t of 2.14 measured
        on 1,451 weekly cross-sections implies ``effect_size / sigma =
        2.14 / sqrt(1451) = 0.0562``, which at n=123 gives ncp=0.62 and power
        9.6% — the published figure.

    WHY IT EXISTS
        "No detectable effect" and "no effect" are different claims, and
        conflating them is the most common error in this field. A null result at
        10% power is close to preordained and should barely move a prior; the
        same null at 90% power is real evidence of absence. A verification
        product that reports a verdict without a power figure is selling a
        coin flip as a referee's decision.

    WHERE IT MISLEADS
        * POST-HOC POWER IS NOT A THING. If ``effect_size`` is the effect you
          measured on the same data, the answer is a deterministic function of
          your p-value and tells you nothing. The effect size must come from
          outside: the literature, a customer's stated economic threshold, or a
          stipulated benchmark. Say where it came from whenever you quote the
          number.
        * The normal approximation ignores the estimation of sigma. At n below
          ~30 it is optimistic by a few percentage points; use it to
          distinguish 10% power from 80%, not 78% from 82%.
        * ``n_events`` must be the number of INDEPENDENT observations. Feeding
          it 123 clustered events when there are 67 distinct weeks overstates
          power by a factor of sqrt(123/67) ~ 1.35 in the non-centrality.
          Use ``effective_sample_size`` first.
        * It assumes a two-sided test of a mean with i.i.d. observations. It
          does not describe the power of a bootstrap test, a signed
          specification, or a regression.

    Parameters
    ----------
    n_events
        Number of independent observations, >= 1.
    effect_size
        Stipulated true effect, in the same units as ``sigma``. May be negative
        (only |effect| matters) and may be 0.
    sigma
        Per-observation standard deviation, strictly positive.
    alpha
        Two-sided significance level, strictly between 0 and 1.

    Returns
    -------
    float
        Power in [alpha, 1]. An ``effect_size`` of exactly 0 returns
        (very nearly) ``alpha``: that is the correct answer — the chance of
        rejecting when nothing is there — not "no power".

    Raises
    ------
    ValueError
        ``n_events < 1``, non-integral ``n_events``, ``sigma <= 0``, any
        non-finite argument, or ``alpha`` outside (0, 1). Nothing here returns
        0.0 to mean "could not compute".
    """
    if isinstance(n_events, bool) or not isinstance(n_events, (int, np.integer)):
        raise ValueError(f"n_events must be an integer, got {n_events!r}")
    if n_events < 1:
        raise ValueError(f"n_events must be at least 1, got {n_events!r}")
    for name, value in (("effect_size", effect_size), ("sigma", sigma), ("alpha", alpha)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if float(sigma) <= 0.0:
        raise ValueError(
            f"sigma must be strictly positive, got {sigma!r}: with zero "
            "dispersion every test is trivially significant and 'power' is not "
            "defined."
        )
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")

    z_crit = float(norm.ppf(1.0 - float(alpha) / 2.0))
    ncp = abs(float(effect_size)) * math.sqrt(int(n_events)) / float(sigma)
    power = float(norm.cdf(ncp - z_crit) + norm.cdf(-ncp - z_crit))
    return min(max(power, 0.0), 1.0)


def sample_size_for_power(
    effect_size: float,
    sigma: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Observations needed to reach ``power`` against ``effect_size``.

    WHAT IT COMPUTES
        The textbook inversion of the two-sided normal power function, ignoring
        the wrong-tail term::

            n = ceil( ((z_{1-alpha/2} + z_{power}) * sigma / effect_size) ** 2 )

        Reproduces the published table: effect/sigma = 2.14 / sqrt(1451) at 80%
        power, alpha=0.05 gives n = 2,487.

    WHERE IT MISLEADS
        Same caveats as ``power_estimate``: the answer is only as real as the
        stipulated effect size, it counts INDEPENDENT observations (123 events
        on 67 weeks buys 67), and it is a normal approximation. Because the
        wrong-tail term is dropped, ``power_estimate(n, ...)`` at the returned
        n is at or a hair above the requested power, never below.

    Raises
    ------
    ValueError
        ``effect_size == 0`` (no finite n suffices — this raises rather than
        returning a huge integer that looks computed), ``sigma <= 0``,
        non-finite arguments, or ``power`` / ``alpha`` outside (0, 1). Also
        raises when ``power <= alpha``, where the arithmetic would return a
        meaninglessly small n.
    """
    for name, value in (
        ("effect_size", effect_size),
        ("sigma", sigma),
        ("power", power),
        ("alpha", alpha),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if float(sigma) <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got {sigma!r}")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    if not (0.0 < float(power) < 1.0):
        raise ValueError(f"power must be strictly between 0 and 1, got {power!r}")
    if float(effect_size) == 0.0:
        raise ValueError(
            "effect_size is 0: no finite sample size gives power against a "
            "zero effect. Stipulate the smallest effect worth detecting."
        )
    if float(power) <= float(alpha):
        raise ValueError(
            f"power={power!r} is not above alpha={alpha!r}; any n already "
            "achieves it, so the answer would be meaningless."
        )

    z_alpha = float(norm.ppf(1.0 - float(alpha) / 2.0))
    z_power = float(norm.ppf(float(power)))
    n = ((z_alpha + z_power) * float(sigma) / abs(float(effect_size))) ** 2
    return int(math.ceil(n))


# --------------------------------------------------------------------------- #
# 5. Pseudo-replication
# --------------------------------------------------------------------------- #
def effective_sample_size(
    timestamps: Sequence[float] | np.ndarray,
    *,
    window: float,
    origin: float = 0.0,
) -> dict[str, Any]:
    """Count how many genuinely distinct time buckets a set of events falls in.

    WHAT IT COMPUTES
        Assigns each timestamp to the bucket ``floor((t - origin) / window)``
        and reports how many distinct buckets the events occupy, how big the
        buckets are, and the ratio between the two counts. Timestamps are
        epoch seconds (or any consistent unit shared with ``window``); order
        does not matter and duplicates are kept.

    WHY IT EXISTS
        123 events on 67 distinct weeks are not 123 independent observations. In
        the published CFTC study, four weeks had five correlated contracts
        firing together; re-running every cell with a cluster bootstrap over
        weeks instead of an i.i.d. bootstrap over events moved the headline
        p-value from 0.002 to 0.014 and the adjusted p from 0.102 to 0.294.
        A factor of five, from nothing but counting the calendar. Any n quoted
        to a customer needs this number beside it.

    WHAT IT DELIBERATELY DOES NOT RETURN
        A single "effective n". Converting a cluster count into one requires an
        intra-cluster correlation, and inventing one would be a fabricated
        number wearing a statistical costume. The fix for clustering is a
        cluster bootstrap over the buckets, not a shrunken n. This function is
        the diagnostic that tells you to go and do that.

    WHERE IT MISLEADS
        * Distinct buckets are an UPPER bound on independence. Two events in
          different weeks can still be one event (correlated instruments
          reacting to the same macro release a fortnight apart). Nothing here
          detects that.
        * The grid has a PHASE. With ``origin=0.0`` a weekly grid starts on a
          Thursday, because the Unix epoch was a Thursday. Two events 24 hours
          apart may share a bucket or not depending on that phase, and a
          different ``origin`` gives a different ``n_clusters``. Fix ``origin``
          to something meaningful (the first as-of date, say) and report it —
          it is returned so it can be quoted.
        * Fixed buckets are not clustering. Events at 23:59 and 00:01 on
          consecutive days land in different daily buckets despite being two
          minutes apart.
        * ``max_cluster_size`` is the number to look at, not the mean. One week
          carrying 7 of 123 events is a concentration the ratio alone hides.

    Parameters
    ----------
    timestamps
        1-D finite timestamps, any order, duplicates allowed.
    window
        Bucket width in the same units, strictly positive and finite.
    origin
        Offset of the bucket grid. Default 0.0 (the raw epoch grid).

    Returns
    -------
    dict
        ``n_observations``, ``n_clusters``, ``distinct_ratio``
        (``n_clusters / n_observations``, 1.0 = fully independent),
        ``pseudo_replication`` (``n_observations / n_clusters``, events per
        distinct bucket), ``max_cluster_size``, ``mean_cluster_size``,
        ``singleton_clusters``, ``cluster_sizes`` (descending tuple),
        ``window`` and ``origin``.

    Raises
    ------
    ValueError
        On an EMPTY input (there is no honest ratio for zero events — a
        ``distinct_ratio`` of 0.0 or 1.0 would both be lies), NaN / infinite
        timestamps, a non-1-D input, a non-positive or non-finite ``window``,
        a non-finite ``origin``, or timestamps so large relative to ``window``
        that the bucket index would overflow int64.
    """
    if not math.isfinite(float(window)) or float(window) <= 0.0:
        raise ValueError(f"window must be finite and strictly positive, got {window!r}")
    if not math.isfinite(float(origin)):
        raise ValueError(f"origin must be finite, got {origin!r}")

    t = _as_1d_finite(timestamps, "timestamps")
    n = int(t.size)
    if n == 0:
        raise ValueError(
            "timestamps is empty: 0 events have no meaningful independence "
            "ratio. Report 'no events' upstream; do not let this return a "
            "number that reads like a measurement."
        )

    scaled = (t - float(origin)) / float(window)
    if np.max(np.abs(scaled)) >= 2.0**62:
        raise ValueError(
            "timestamps are too large relative to window: the bucket index "
            "would overflow int64. Check the units (epoch seconds vs "
            "milliseconds) before trusting anything downstream."
        )
    buckets = np.floor(scaled).astype(np.int64)
    _, counts = np.unique(buckets, return_counts=True)
    k = int(counts.size)

    return {
        "n_observations": n,
        "n_clusters": k,
        "distinct_ratio": float(k) / float(n),
        "pseudo_replication": float(n) / float(k),
        "max_cluster_size": int(counts.max()),
        "mean_cluster_size": float(counts.mean()),
        "singleton_clusters": int(np.count_nonzero(counts == 1)),
        "cluster_sizes": tuple(sorted((int(c) for c in counts), reverse=True)),
        "window": float(window),
        "origin": float(origin),
    }
