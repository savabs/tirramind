"""
TirraMind — Portfolio Beliefs: testing what the holder already believes.

WHAT THIS MODULE IS FOR
    A person holds a gold ETF "as a hedge". They have held that belief for six
    years and have never once measured it. This module measures it, over their
    own holdings, and reports what actually happened — per period, with a
    sample size, an interval and a power figure attached to every number.

    Three entry points:

      * ``regime_split``          — cut the window into periods by a stated,
        pre-committed rule (calendar years). The rule is returned in the output
        so the reader can see it was not chosen after looking at the answer.
      * ``test_hedge``            — did the claimed hedge actually move against
        the rest of the book, in each period? Correlation with a block
        bootstrap interval, an n, a weekly-cluster count and a power estimate.
      * ``drawdown_coincidence``  — on the portfolio's worst days, what did each
        holding do? "Your hedge fell with everything else on 7 of your 10 worst
        days" is a fact the holder can check against their own statement.

WHAT THIS MODULE WILL NEVER DO
    Give advice. Every string it emits is past tense and descriptive: "moved
    with your book in 2024", never "is not hedging you", never "consider". It
    describes realised co-movement of public prices over a stated window. It
    makes no forecast, and nothing here should be phrased as one downstream.

WHERE THIS MISLEADS (read before showing any of it to a stranger)
    1. CORRELATION OVER A CALENDAR YEAR IS ABOUT 250 DAILY OBSERVATIONS THAT
       ARE NOT 250 INDEPENDENT ONES. Volatility clusters; a single macro event
       colours a fortnight. Every result carries ``n_independent_weeks`` from
       ``effective_sample_size`` and a power figure computed on the weekly
       count, not the daily one. A year of data is moderately powered to
       detect a |r| of 0.30 and badly powered for anything smaller. "It broke
       in 2023" on one year of data is a claim this module is designed to
       refuse unless the interval supports it.
    2. CALENDAR YEARS ARE ARBITRARY. They are chosen precisely because they are
       arbitrary — a data-driven break finder would locate the break you went
       looking for, and its p-value would be a fiction unless corrected for the
       search. A regime boundary at 31 December has no economic meaning and the
       output says so.
    3. THE WORST 5% OF DAYS ARE A SELECTED SAMPLE. Statistics computed on them
       are conditional on that selection and are not estimates of anything
       unconditional. The counts ("fell on 7 of 10") are reported because a
       count needs no distributional assumption; the binomial p beside them
       assumes days are independent, which they are not, so it is labelled a
       FLOOR and the weekly clustering of those days is printed next to it.
    4. A CORRELATION IS NOT A HEDGE RATIO. A gold ETF at 4% of a book can be
       perfectly negatively correlated and still move the portfolio by
       approximately nothing. This module answers "did it move the other way",
       not "did it matter". Sizing is a different question and a different
       module.
    5. EVERYTHING IS MEASURED ON THE RETURNS PANEL IT IS HANDED. If that panel
       was built with a forward-looking join, a survivorship filter, or prices
       that silently forward-filled across a delisting, nothing here will
       notice. This module guards its own arithmetic, not the data pipeline.

ZERO AND NONE DISCIPLINE
    A correlation of 0.0 and "could not compute a correlation" must never be
    the same value. Every statistic in this module is ``None`` when it could
    not be computed, and carries a reason string saying why. Nothing returns
    0.0, 1.0 or an empty list to mean "not available".

NO SILENT EXCLUSION
    Every ticker dropped — missing from the panel, too few overlapping
    observations, zero variance — appears in the ``excluded`` field with a
    reason and a count. A portfolio result that quietly measured 9 of 12
    holdings is the shape of the confident wrong answer this product exists
    to refuse.

REUSE
    ``block_bootstrap_ci``, ``power_estimate`` and ``effective_sample_size``
    come from ``agent.verify.stats`` and are not reimplemented here. The
    correlation bootstrap resamples PAIRED observations by bootstrapping an
    index array through that same function, so the day-alignment of the two
    series survives resampling.

Layer: 3 (world model — beliefs about structure). Stateless; no I/O, no clock.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from agent.verify.stats import (
    benjamini_hochberg,
    block_bootstrap_ci,
    effective_sample_size,
    power_estimate,
)

__all__ = [
    "Regime",
    "RegimeHedge",
    "HedgeResult",
    "regime_split",
    "scrub_suspect_returns",
    "test_hedge",
    "drawdown_coincidence",
    "render_hedge_result",
    "render_drawdown_coincidence",
    "MIN_REGIME_DAYS",
    "MIN_OVERLAP_DAYS",
    "REFERENCE_HEDGE_CORRELATION",
    "SUSPECT_RETURN_THRESHOLD",
    "SECONDS_PER_WEEK",
]


# --------------------------------------------------------------------------- #
# Constants — every threshold is named, because each one silently decides
# whether a claim is made at all.
# --------------------------------------------------------------------------- #

#: A period shorter than this is reported, but never given a correlation.
#: 60 trading days is about a quarter, or ~12 distinct weeks; at 12 weekly
#: observations the power to detect |r| = 0.30 is under 20%, so any number
#: produced there would be noise wearing a decimal point.
MIN_REGIME_DAYS = 60

#: Minimum overlapping non-missing observations between two series before a
#: correlation is computed at all. Matches the regime floor deliberately: a
#: pair that overlaps on 40 days inside a 250-day regime is a data problem,
#: not a short regime, and is reported as an exclusion either way.
MIN_OVERLAP_DAYS = 60

#: The stipulated effect the power figure is computed against: the correlation
#: a position would need for "it moves against my book" to be a description of
#: the position rather than of the noise. It is an ASSUMPTION, stated in the
#: output, never estimated from the data being tested (post-hoc power is not a
#: thing — see agent.verify.stats.power_estimate).
REFERENCE_HEDGE_CORRELATION = -0.30

#: Bucket width for the weekly clustering diagnostic, in seconds.
SECONDS_PER_WEEK = 7 * 24 * 3600

#: Fraction of days that counts as "the worst days".
_WORST_FRACTION = 0.05

#: Fewest worst-days that can carry a countable claim. Two bad days is an
#: anecdote; "fell on 2 of 2" is not a statistic.
_MIN_WORST_DAYS = 5

#: Bootstrap settings. Fixed so results are reproducible and quotable.
_N_BOOTSTRAP = 2000
_BOOTSTRAP_SEED = 42
_CONFIDENCE = 0.95

#: A returns panel whose values look like this is almost certainly prices.
#: A daily simple return above 100% in the 99th percentile across a whole
#: column does not happen for a listed instrument.
_RETURN_SANITY_QUANTILE = 0.99
_RETURN_SANITY_MAX = 1.0

#: A single-day absolute simple return at or above this is treated as a
#: suspected unadjusted corporate action rather than a price move.
#:
#: This is not hypothetical. GOLDBEES.NS — the most widely held gold ETF in
#: India — split 1:100 on 2019-12-19, and yfinance's auto-adjusted close does
#: NOT adjust for it: the series shows -99.0% on one day and +9,900% on the
#: next. Left in, that pair puts the fund's mean daily return at +5.5% and
#: would have told a stranger their gold ETF returns 5.5% a DAY. Every such
#: cell is dropped and REPORTED WITH ITS DATE, so a genuine 50% move that gets
#: caught by this rule is visible and checkable rather than silently removed.
SUSPECT_RETURN_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Panel adaptation
# --------------------------------------------------------------------------- #
def _coerce_panel(panel: Any) -> pd.DataFrame:
    """Coerce the caller's panel into a DataFrame of DAILY SIMPLE RETURNS.

    WHAT IT ACCEPTS
        * a ``pandas.DataFrame`` of returns, indexed by date, one column per
          ticker;
        * any object exposing a ``.returns`` attribute that is such a frame —
          notably ``agent.portfolio.holdings.PricePanel``, whose ``.returns``
          is ``prices.pct_change().dropna()``;
        * a mapping of ticker -> return series.

    WHY IT CHECKS THE MAGNITUDES
        Handing this module a PRICE panel instead of a RETURNS panel is the
        single most likely integration mistake, and it fails silently and
        spectacularly: two price levels that both drift upward correlate at
        0.95 regardless of whether the instruments have anything to do with
        each other, and the product would tell a stranger their gold ETF
        tracks their bank stock. So a panel whose 99th-percentile absolute
        value exceeds 1.0 (a 100% daily move) is refused rather than measured.

    WHERE IT MISLEADS
        It cannot tell a return panel from a panel of small numbers that are
        not returns (basis points divided by 10,000, say). It catches the loud
        mistake, not every mistake.
    """
    frame = getattr(panel, "returns", panel)
    if isinstance(frame, Mapping):
        frame = pd.DataFrame(frame)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "panel must be a pandas DataFrame of daily simple returns "
            "(index=date, columns=tickers), an object exposing such a frame as "
            f".returns, or a mapping of ticker -> series; got {type(panel).__name__}."
        )
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("panel is empty: there is nothing to measure.")

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("panel contains no finite values: there is nothing to measure.")

    q = float(np.quantile(np.abs(finite), _RETURN_SANITY_QUANTILE))
    if q > _RETURN_SANITY_MAX:
        raise ValueError(
            "panel does not look like daily simple returns: the "
            f"{_RETURN_SANITY_QUANTILE:.0%} percentile of |value| is {q:.4g}, "
            "which is a move of "
            f"{q:.0%} in one day. This is what a PRICE panel looks like, and "
            "correlating price levels produces ~0.95 between any two drifting "
            "instruments. Convert to returns before calling this module."
        )

    index = _coerce_index(numeric.index)
    out = pd.DataFrame(values, index=index, columns=[str(c) for c in numeric.columns])
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    return out


def scrub_suspect_returns(
    frame: pd.DataFrame,
    *,
    threshold: float = SUSPECT_RETURN_THRESHOLD,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Blank single-day returns large enough to be a corporate action, and say so.

    WHAT IT COMPUTES
        Returns a copy of the frame with every cell whose absolute value is at
        or above ``threshold`` (default 0.5 — a 50% move in one day) replaced
        by NaN, plus one ``(ticker, reason)`` line per affected ticker naming
        every date and value removed.

    WHY IT EXISTS
        yfinance's auto-adjusted close does not adjust every corporate action.
        GOLDBEES.NS split 1:100 in December 2019 and the adjusted series still
        carries a -99.0% day followed by a +9,900% day. Those two cells alone
        move the fund's mean daily return to +5.5%. Nothing downstream would
        have flagged it: the correlation, the mean, the drawdown table would
        all have been computed and printed to a stranger who owns the fund and
        would have known instantly that the number was nonsense.

    WHERE IT MISLEADS
        A genuine 50%+ one-day move — a fraud revelation, a delisting gap, a
        micro-cap — is indistinguishable from a bad split by magnitude alone,
        and this function will remove it too, which FLATTERS the holding. That
        is why every removed cell is returned with its date and value rather
        than counted: the reader can check whether 2019-12-23 was a real day.
    """
    if not (0.0 < float(threshold) <= 10.0):
        raise ValueError(f"threshold must be in (0, 10], got {threshold!r}")
    values = frame.to_numpy(dtype=float)
    suspect = np.abs(values) >= float(threshold)
    if not suspect.any():
        return frame, []
    cleaned = frame.mask(pd.DataFrame(suspect, index=frame.index, columns=frame.columns))
    notes: list[tuple[str, str]] = []
    for j, ticker in enumerate(frame.columns):
        rows = np.flatnonzero(suspect[:, j])
        if rows.size == 0:
            continue
        detail = ", ".join(f"{frame.index[i].date().isoformat()} {values[i, j]:+.1%}" for i in rows[:6])
        if rows.size > 6:
            detail += f", and {rows.size - 6} more"
        notes.append(
            (
                str(ticker),
                f"{rows.size} day(s) blanked as a suspected unadjusted corporate "
                f"action (|daily return| >= {threshold:.0%}): {detail}. Check "
                "those dates: a genuine move that large would also be removed.",
            )
        )
    return cleaned, notes


def _coerce_weights(weights: Any, *, name: str = "weights") -> dict[str, float]:
    """Turn a weight mapping OR a ``pandas.Series`` into a plain dict.

    WHY THIS IS NOT JUST ``dict(weights)``
        ``agent.portfolio.holdings.PricePanel.weights`` is a ``pandas.Series``,
        and a Series is not a ``Mapping``. Iterating one yields its VALUES, not
        its index, so ``for t in weights`` would silently walk a list of floats
        and look up tickers called "0.18". ``if not weights`` on a Series
        raises "truth value is ambiguous" outright. Both failure modes are the
        kind that reach a stranger as a confidently wrong table, so the
        conversion is explicit and the type is checked.
    """
    if isinstance(weights, (pd.Series, Mapping)):
        pairs = [(str(k), float(v)) for k, v in weights.items()]
    else:
        raise TypeError(
            f"{name} must be a mapping of ticker -> weight or a pandas Series, got {type(weights).__name__}."
        )
    out: dict[str, float] = {}
    for ticker, value in pairs:
        if not math.isfinite(value):
            raise ValueError(f"{name}[{ticker!r}] is {value!r}, which is not finite.")
        if ticker in out:
            raise ValueError(
                f"{name} names {ticker!r} twice. Merge the lots upstream; this "
                "module will not guess which line the reader meant."
            )
        out[ticker] = value
    return out


def _coerce_index(index: Any) -> pd.DatetimeIndex:
    """Turn the panel's index into a DatetimeIndex, refusing an unusable one."""
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(index))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"panel index is not convertible to dates ({exc}). Every claim this "
            "module makes carries a date range, so an unusable index is fatal "
            "rather than ignorable."
        ) from exc
    if idx.hasnans:
        raise ValueError("panel index contains NaT: some rows have no date.")
    if idx.has_duplicates:
        n_dup = int(idx.duplicated().sum())
        raise ValueError(
            f"panel index has {n_dup} duplicated dates. A duplicated day is "
            "double-counted in every statistic below; de-duplicate upstream."
        )
    return idx


# --------------------------------------------------------------------------- #
# 1. Regimes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Regime:
    """One period of the window, produced by a stated rule.

    ``method`` and ``method_note`` travel WITH the regime rather than being
    documented elsewhere, so that a rendered output physically cannot show a
    period without showing how the period was chosen.
    """

    label: str
    start: date
    end: date
    n_days: int
    method: str
    method_note: str
    testable: bool
    not_testable_reason: str | None = None

    @property
    def window(self) -> str:
        """Human window string: an actual date range, never 'since 2024'."""
        return f"{self.start.isoformat()} to {self.end.isoformat()} (n={self.n_days} trading days)"


_REGIME_METHODS: dict[str, str] = {
    "calendar_year": (
        "Calendar years. The boundary is 31 December, chosen BEFORE looking at "
        "the data and for no economic reason at all. That is the point: a "
        "data-driven break finder locates the break it was sent to find, and "
        "its p-value is a fiction unless corrected for the search."
    ),
    "calendar_half": (
        "Calendar half-years (Jan-Jun, Jul-Dec). Boundaries fixed in advance "
        "for the same reason as calendar years; use only when the window is "
        "too short to give calendar years a usable sample."
    ),
    "whole_window": (
        "No split: one period covering the whole panel. Reported as a regime "
        "so that a single-period result carries the same window and n as a "
        "split one."
    ),
}


def regime_split(panel: Any, *, method: str = "calendar_year") -> list[Regime]:
    """Cut the panel's window into periods by a pre-committed calendar rule.

    WHAT IT COMPUTES
        Groups the panel's trading days by calendar year (default), calendar
        half-year, or not at all, and returns one ``Regime`` per non-empty
        group, in chronological order. Each carries its true first and last
        TRADING day (not 1 January), its day count, the method name, a note
        explaining the method, and whether it is long enough to support a
        correlation claim (``MIN_REGIME_DAYS`` = 60).

    WHY NO DATA-DRIVEN SPLIT IS OFFERED
        Because it would work. A change-point search over a single pair of
        series will always return a change point, and the period either side
        of it will always look different — that is what the search maximised.
        Reporting the resulting "break" without correcting for the search is
        how a product tells a stranger their hedge broke in March 2023 when
        nothing happened in March 2023. Calendar boundaries cannot be tuned,
        so they cannot be tuned in your favour.

    WHERE IT MISLEADS
        * A calendar year is not a regime. Markets do not change state on 31
          December, and a real regime change in July is split across two
          periods here, diluting both. This method is legible and honest, not
          sensitive.
        * A partial year at either end of the panel is a short period and will
          usually be marked not testable. It is still returned, because
          dropping it would hide the fact that the window does not cover it.

    Parameters
    ----------
    panel
        Returns panel (see ``_coerce_panel``).
    method
        ``"calendar_year"`` (default), ``"calendar_half"`` or
        ``"whole_window"``.

    Returns
    -------
    list[Regime]
        Chronological, never empty for a non-empty panel. Periods too short to
        test are included with ``testable=False`` and a reason.

    Raises
    ------
    ValueError
        Unknown ``method``, or an unusable panel.
    """
    if method not in _REGIME_METHODS:
        raise ValueError(
            f"unknown regime method {method!r}; supported: "
            f"{sorted(_REGIME_METHODS)}. No data-driven split is offered on "
            "purpose: a change-point search always finds a change point, and "
            "the period either side of it always differs, because that is what "
            "the search maximised."
        )
    frame = _coerce_panel(panel)
    index = frame.index
    note = _REGIME_METHODS[method]

    if method == "whole_window":
        keys = pd.Series(["whole window"] * len(index), index=index)
    elif method == "calendar_year":
        keys = pd.Series([str(ts.year) for ts in index], index=index)
    else:
        keys = pd.Series(
            [f"{ts.year} H{1 if ts.month <= 6 else 2}" for ts in index],
            index=index,
        )

    regimes: list[Regime] = []
    for label in dict.fromkeys(keys.tolist()):
        days = index[keys.to_numpy() == label]
        n = int(days.size)
        testable = n >= MIN_REGIME_DAYS
        reason = (
            None
            if testable
            else (
                f"{n} trading days is below the {MIN_REGIME_DAYS}-day floor "
                f"(~{MIN_REGIME_DAYS // 5} distinct weeks). A correlation on "
                "this much data has under 20% power against |r| = "
                f"{abs(REFERENCE_HEDGE_CORRELATION):.2f}, so no correlation is "
                "reported for this period."
            )
        )
        regimes.append(
            Regime(
                label=str(label),
                start=days[0].date(),
                end=days[-1].date(),
                n_days=n,
                method=method,
                method_note=note,
                testable=testable,
                not_testable_reason=reason,
            )
        )
    return regimes


# --------------------------------------------------------------------------- #
# 2. Hedge testing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegimeHedge:
    """What the claimed hedge did against the book during one period.

    Every statistical field is ``None`` when it could not be computed, with
    ``stat_reason`` saying why. There is no sentinel number.
    """

    regime: Regime
    n_days_used: int
    n_days_dropped: int
    n_independent_weeks: int | None
    correlation: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    p_adjusted: float | None
    survives_multiplicity: bool | None
    n_periods_tested: int
    stat_reason: str | None
    power_if_days_independent: float | None
    power_if_only_weeks_independent: float | None
    reference_correlation: float
    verdict: str
    detail: str


@dataclass(frozen=True)
class HedgeResult:
    """Full answer to 'did my hedge actually hedge?', period by period."""

    hedge_ticker: str
    book_tickers: tuple[str, ...]
    book_weighting: str
    window_start: date
    window_end: date
    n_days_total: int
    regime_method: str
    regime_method_note: str
    bootstrap_note: str
    regimes: tuple[RegimeHedge, ...] = ()
    excluded: tuple[tuple[str, str], ...] = ()
    headline: str = ""
    caveats: tuple[str, ...] = field(default_factory=tuple)


def _book_series(
    frame: pd.DataFrame,
    book_tickers: Sequence[str],
    weights: dict[str, float] | None,
) -> tuple[pd.Series, tuple[str, ...], str, list[tuple[str, str]]]:
    """Build the book's return series, reporting every ticker it could not use."""
    excluded: list[tuple[str, str]] = []
    usable: list[str] = []
    for ticker in dict.fromkeys(str(t) for t in book_tickers):
        if ticker not in frame.columns:
            excluded.append((ticker, "not present in the returns panel"))
            continue
        n_obs = int(frame[ticker].notna().sum())
        if n_obs < MIN_OVERLAP_DAYS:
            excluded.append(
                (
                    ticker,
                    f"only {n_obs} non-missing daily returns in the panel, below the {MIN_OVERLAP_DAYS}-day floor",
                )
            )
            continue
        usable.append(ticker)

    if not usable:
        raise ValueError(
            "no book ticker survived: "
            + "; ".join(f"{t} ({why})" for t, why in excluded)
            + ". There is no book to measure the hedge against."
        )

    if weights is None:
        w = pd.Series(1.0 / len(usable), index=usable)
        description = f"equal-weighted across the {len(usable)} usable book holdings"
    else:
        raw = {t: float(weights[t]) for t in usable if t in weights}
        missing = [t for t in usable if t not in weights]
        for ticker in missing:
            excluded.append((ticker, "no weight supplied for this ticker"))
        usable = [t for t in usable if t in raw]
        if not usable:
            raise ValueError("no book ticker had both data and a weight.")
        total = sum(abs(v) for v in raw.values())
        if total <= 0.0:
            raise ValueError("supplied book weights sum to zero in absolute value.")
        w = pd.Series({t: raw[t] / total for t in usable})
        description = (
            f"weighted by the supplied holdings across {len(usable)} book "
            f"positions, renormalised to sum to 1 (input absolute sum {total:.4g})"
        )

    sub = frame[usable]
    # A day is usable for the book only if every book holding traded. Partial
    # rows are dropped rather than treated as zero returns, and the count of
    # dropped days is reported by the caller.
    series = sub.mul(w, axis=1).sum(axis=1, skipna=False)
    return series, tuple(usable), description, excluded


def _paired_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float | None, float | None, float | None, str | None]:
    """Pearson correlation with a PAIRED circular-block bootstrap interval.

    HOW THE PAIRING SURVIVES THE BOOTSTRAP
        ``agent.verify.stats.block_bootstrap_ci`` resamples a 1-D sample. So
        the sample handed to it is the INDEX ARRAY ``[0, 1, ..., n-1]``, and
        the statistic looks both series up at the resampled indices. Blocks of
        adjacent indices therefore become blocks of adjacent (x, y) PAIRS, and
        the day alignment that the whole claim rests on is never broken.

    WHERE IT MISLEADS
        Percentile bootstrap intervals for a correlation are not
        bias-corrected and are poor near |r| = 1. They also cannot repair a
        sample that is unrepresentative; they describe resampling noise only.

    Returns ``(point, lo, hi, p, reason)``; the last four are ``None`` with a
    reason string when the bootstrap could not produce an honest interval.
    """
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return (
            float("nan"),
            None,
            None,
            None,
            "one of the two series has zero variance over this period: a correlation is undefined, not zero",
        )

    point = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(point):
        return (
            float("nan"),
            None,
            None,
            None,
            "the Pearson correlation is not finite over this period",
        )

    def _stat(idx: np.ndarray) -> float:
        take = idx.astype(np.int64)
        a, b = x[take], y[take]
        sa, sb = float(np.std(a)), float(np.std(b))
        if sa == 0.0 or sb == 0.0:
            # A resample that happens to be constant in one series carries no
            # information about correlation. Returning 0.0 would pull the
            # interval toward the null; NaN makes the wrapper refuse, which is
            # the behaviour we want to surface rather than hide.
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    try:
        result = block_bootstrap_ci(
            np.arange(x.size, dtype=float),
            _stat,
            n_bootstrap=_N_BOOTSTRAP,
            seed=_BOOTSTRAP_SEED,
            confidence=_CONFIDENCE,
            null_value=0.0,
        )
    except (ValueError, RuntimeError) as exc:
        return (
            point,
            None,
            None,
            None,
            f"no bootstrap interval: {exc.__class__.__name__}: {exc}",
        )
    return point, float(result.lo), float(result.hi), float(result.p), None


def _weekly_clusters(index: pd.DatetimeIndex) -> tuple[int | None, str | None]:
    """Distinct calendar weeks the given trading days fall in.

    The grid origin is pinned to the FIRST day in the set rather than the Unix
    epoch, because the epoch grid starts on a Thursday and a Thursday phase
    would make the count depend on an accident of 1970.
    """
    if index.size == 0:
        return None, "no days"
    seconds = np.asarray([pd.Timestamp(ts).to_pydatetime().timestamp() for ts in index], dtype=float)
    try:
        info = effective_sample_size(
            seconds,
            window=float(SECONDS_PER_WEEK),
            origin=float(seconds[0]),
        )
    except ValueError as exc:
        return None, str(exc)
    return int(info["n_clusters"]), None


def _correlation_power(n_independent: int | None, reference_correlation: float) -> float | None:
    """Power to detect the stipulated correlation, on the Fisher-z scale.

    ``atanh(r)`` is approximately normal with standard error ``1/sqrt(n-3)``,
    so the non-centrality of the two-sided test is ``|atanh(r)| * sqrt(n-3)``.
    That is exactly the form ``power_estimate`` computes with
    ``effect_size = |atanh(r)|``, ``sigma = 1`` and ``n_events = n - 3``.

    WHERE IT MISLEADS
        The effect size is STIPULATED, never measured from these data.
        Post-hoc power — plugging in the correlation you just observed — is a
        deterministic restatement of your own p-value and means nothing.
    """
    if n_independent is None or n_independent <= 3:
        return None
    effect = abs(math.atanh(max(min(reference_correlation, 0.999999), -0.999999)))
    if effect == 0.0:
        return None
    return float(power_estimate(int(n_independent - 3), effect, 1.0))


def test_hedge(
    panel: Any,
    hedge_ticker: str,
    book_tickers: Sequence[str],
    *,
    regimes: Sequence[Regime],
    weights: Mapping[str, float] | pd.Series | None = None,
    reference_correlation: float = REFERENCE_HEDGE_CORRELATION,
) -> HedgeResult:
    """Did the claimed hedge move against the rest of the book, period by period?

    WHAT IT COMPUTES
        For each supplied regime, the Pearson correlation between the hedge's
        daily return and the book's daily return (book = the other holdings,
        equal-weighted unless ``weights`` is given), with:

          * a 95% circular-block-bootstrap confidence interval over PAIRED
            days (2,000 resamples, seed 42) and a two-sided bootstrap p
            against a null of zero;
          * ``n_days_used`` and ``n_days_dropped`` — days where either side was
            missing are dropped and counted, never zero-filled;
          * ``n_independent_weeks`` — the distinct calendar weeks those days
            fall in, because ~250 daily observations inside a year are not 250
            independent ones;
          * two power figures against the stipulated ``reference_correlation``:
            one assuming daily observations are independent (optimistic) and
            one assuming only weeks are (conservative). Both are printed
            because the truth is between them.

        The verdict for a period is read off the INTERVAL, not the point
        estimate: the whole interval below zero means it moved against the
        book, the whole interval above zero means it moved with it, and an
        interval straddling zero means no detectable relationship — reported
        together with the power, so "we could not tell" is distinguishable
        from "there was nothing there".

    WHAT IT DOES NOT COMPUTE
        Whether the hedge was worth holding. A 4% position can be perfectly
        negatively correlated and move the portfolio by nothing. This answers
        direction of co-movement only, over the past, on a stated window.

    WHERE IT MISLEADS
        * The book is measured only on days when EVERY usable book holding
          traded. A holding that is frequently suspended shortens the sample
          for everything; the dropped-day count is reported per period so this
          is visible rather than silent.
        * Pearson correlation is a linear, symmetric measure. A position that
          is flat in calm markets and rallies hard in crashes can show a near
          zero correlation while being exactly the thing the holder wanted.
          ``drawdown_coincidence`` is the companion that looks at the tail.
        * The bootstrap interval assumes the sample is representative of the
          period. It is a statement about resampling, not about the future.

    Parameters
    ----------
    panel
        Returns panel (see ``_coerce_panel``).
    hedge_ticker
        The holding whose hedging claim is being tested.
    book_tickers
        The rest of the book. ``hedge_ticker`` is removed if present, and the
        removal is reported in ``excluded``.
    regimes
        Periods from ``regime_split``. Passed in rather than computed so that
        the split rule is the caller's stated choice, visible in the output.
    weights
        Optional ticker -> weight for the book. Default equal weight, stated
        in ``book_weighting``.
    reference_correlation
        The stipulated effect the power figures are computed against. Default
        -0.30. It is an assumption and is echoed in every ``RegimeHedge``.

    Returns
    -------
    HedgeResult

    Raises
    ------
    ValueError
        ``hedge_ticker`` absent from the panel or with too little data, no
        usable book ticker, or an empty ``regimes``.
    """
    frame = _coerce_panel(panel)
    hedge_ticker = str(hedge_ticker)
    if not regimes:
        raise ValueError("regimes is empty: call regime_split first and pass its output.")

    frame, excluded = scrub_suspect_returns(frame)

    if hedge_ticker not in frame.columns:
        raise ValueError(
            f"hedge ticker {hedge_ticker!r} is not in the returns panel "
            f"(panel has {len(frame.columns)} columns). Nothing is measured "
            "rather than measuring a different instrument."
        )
    n_hedge = int(frame[hedge_ticker].notna().sum())
    if n_hedge < MIN_OVERLAP_DAYS:
        raise ValueError(
            f"hedge ticker {hedge_ticker!r} has only {n_hedge} non-missing "
            f"daily returns, below the {MIN_OVERLAP_DAYS}-day floor."
        )

    book_in = [str(t) for t in book_tickers]
    if hedge_ticker in book_in:
        book_in = [t for t in book_in if t != hedge_ticker]
        excluded.append((hedge_ticker, "removed from the book: it is the instrument being tested"))
    book, used, weighting, book_excluded = _book_series(
        frame, book_in, None if weights is None else _coerce_weights(weights)
    )
    excluded.extend(book_excluded)

    hedge = frame[hedge_ticker]
    # PASS 1 — measure every period. No verdict is assigned yet, because a
    # verdict depends on how many periods were tested in total.
    measured: list[dict[str, Any]] = []
    for regime in regimes:
        mask = (frame.index.date >= regime.start) & (frame.index.date <= regime.end)
        idx = frame.index[mask]
        h = hedge.loc[idx]
        b = book.loc[idx]
        both = h.notna() & b.notna()
        n_used = int(both.sum())
        n_dropped = int(idx.size - n_used)
        used_idx = idx[both.to_numpy()]

        n_weeks, week_reason = _weekly_clusters(used_idx)
        row: dict[str, Any] = {
            "regime": regime,
            "n_days_used": n_used,
            "n_days_dropped": n_dropped,
            "n_independent_weeks": n_weeks,
            "power_days": _correlation_power(n_used, reference_correlation),
            "power_weeks": _correlation_power(n_weeks, reference_correlation),
            "correlation": None,
            "lo": None,
            "hi": None,
            "p": None,
            "stat_reason": None,
        }

        if not regime.testable or n_used < MIN_OVERLAP_DAYS:
            reason = regime.not_testable_reason or (
                f"only {n_used} days where both the hedge and the whole book "
                f"traded, below the {MIN_OVERLAP_DAYS}-day floor"
            )
            row["stat_reason"] = reason if not week_reason else f"{reason}; {week_reason}"
            measured.append(row)
            continue

        x = h.loc[used_idx].to_numpy(dtype=float)
        y = b.loc[used_idx].to_numpy(dtype=float)
        point, lo, hi, p_value, stat_reason = _paired_correlation_ci(x, y)
        row["stat_reason"] = stat_reason
        row["correlation"] = None if not math.isfinite(point) else float(point)
        row["lo"], row["hi"], row["p"] = lo, hi, p_value
        measured.append(row)

    # PASS 2 — correct for having tested every period.
    #
    # Eight calendar years is EIGHT TESTS. At 95% confidence roughly one year
    # in twenty crosses the line with nothing behind it, and the one that does
    # is exactly the year a reader would screenshot as "the year it broke".
    # Benjamini-Hochberg over the family of periods actually tested is what
    # stops this module manufacturing that year. The raw p and the raw
    # interval are still reported beside the adjusted decision, so nothing is
    # hidden by the correction.
    tested = [i for i, row in enumerate(measured) if row["p"] is not None]
    n_tested = len(tested)
    adjusted: dict[int, float] = {}
    survives: dict[int, bool] = {}
    if tested:
        bh = benjamini_hochberg([measured[i]["p"] for i in tested], alpha=0.05)
        for slot, i in enumerate(tested):
            adjusted[i] = float(bh.p_adjusted[slot])
            survives[i] = bool(bh.rejected[slot])

    results: list[RegimeHedge] = []
    for i, row in enumerate(measured):
        lo, hi, p_value = row["lo"], row["hi"], row["p"]
        corr = row["correlation"]
        p_adj = adjusted.get(i)
        survived = survives.get(i)

        if p_value is None or lo is None or hi is None:
            verdict = "not enough data to say"
            detail = row["stat_reason"] or "no interval available"
        elif not survived:
            verdict = "no relationship this window can resolve"
            detail = (
                f"r = {corr:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}], "
                f"bootstrap p = {p_value:.3f}, p after correcting for the "
                f"{n_tested} periods tested = {p_adj:.3f}, on "
                f"{row['n_days_used']} days"
                + (f" falling in {row['n_independent_weeks']} distinct weeks" if row["n_independent_weeks"] else "")
            )
            if row["power_weeks"] is not None:
                detail += (
                    f". Power to detect r = {reference_correlation:+.2f} here was "
                    f"{row['power_weeks']:.0%} (treating weeks as the independent "
                    "unit), so this is 'could not tell', not 'nothing there'."
                )
        elif hi < 0.0:
            verdict = "moved AGAINST your book"
            detail = _decided_detail(corr, lo, hi, p_value, p_adj, n_tested, row)
        elif lo > 0.0:
            verdict = "moved WITH your book"
            detail = _decided_detail(corr, lo, hi, p_value, p_adj, n_tested, row)
        else:
            # The bootstrap p and the percentile interval can disagree at the
            # boundary. When they do, the weaker claim wins.
            verdict = "no relationship this window can resolve"
            detail = (
                f"r = {corr:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}] straddles zero "
                f"even though the bootstrap p is {p_value:.3f}; the interval is "
                "the weaker claim and it is the one reported."
            )

        results.append(
            RegimeHedge(
                regime=row["regime"],
                n_days_used=row["n_days_used"],
                n_days_dropped=row["n_days_dropped"],
                n_independent_weeks=row["n_independent_weeks"],
                correlation=corr,
                ci_low=lo,
                ci_high=hi,
                p_value=p_value,
                p_adjusted=p_adj,
                survives_multiplicity=survived,
                n_periods_tested=n_tested,
                stat_reason=row["stat_reason"],
                power_if_days_independent=row["power_days"],
                power_if_only_weeks_independent=row["power_weeks"],
                reference_correlation=reference_correlation,
                verdict=verdict,
                detail=detail,
            )
        )

    headline = _hedge_headline(hedge_ticker, results)
    caveats = (
        "Correlation describes direction of co-movement over the stated days. "
        "It is not a statement about size, and not a forecast.",
        "Periods are calendar blocks fixed in advance, not detected from the "
        "data. A real change of behaviour inside a year is split across two "
        "periods here.",
        f"Power figures are computed against a STIPULATED correlation of "
        f"{reference_correlation:+.2f}. That number is an assumption, not a "
        "measurement, and was not taken from these data.",
        f"{n_tested} period(s) were tested, so every decision here is made on "
        "the Benjamini-Hochberg adjusted p, not the raw one. At 95% confidence "
        "roughly one period in twenty crosses the line with nothing behind it, "
        "and that period is exactly the one that would get screenshotted as "
        "'the year it broke'.",
    )

    return HedgeResult(
        hedge_ticker=hedge_ticker,
        book_tickers=used,
        book_weighting=weighting,
        window_start=frame.index[0].date(),
        window_end=frame.index[-1].date(),
        n_days_total=int(frame.index.size),
        regime_method=regimes[0].method,
        regime_method_note=regimes[0].method_note,
        bootstrap_note=(
            f"95% interval from a circular block bootstrap over paired days "
            f"({_N_BOOTSTRAP} resamples, seed {_BOOTSTRAP_SEED}); p is a "
            "two-sided bootstrap p against r = 0, floored at "
            f"{1.0 / _N_BOOTSTRAP:.4f}."
        ),
        regimes=tuple(results),
        excluded=tuple(excluded),
        headline=headline,
        caveats=caveats,
    )


def _decided_detail(
    corr: float | None,
    lo: float,
    hi: float,
    p_value: float,
    p_adj: float | None,
    n_tested: int,
    row: Mapping[str, Any],
) -> str:
    """The evidence line for a period whose direction survived the correction."""
    weeks = f" falling in {row['n_independent_weeks']} distinct weeks" if row["n_independent_weeks"] else ""
    adj = f", p after correcting for the {n_tested} periods tested = {p_adj:.3f}" if p_adj is not None else ""
    return (
        f"r = {corr:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}], bootstrap "
        f"p = {p_value:.3f}{adj}, on {row['n_days_used']} days{weeks}"
    )


# ``test_hedge`` is a public API function, not a pytest test. Without this,
# any test module that imports it has it collected as a test and errors on a
# missing ``panel`` fixture.
test_hedge.__test__ = False


def _hedge_headline(hedge_ticker: str, results: Sequence[RegimeHedge]) -> str:
    """One factual sentence: how many periods went each way, and which.

    "Could not tell" is counted and named alongside the two decided outcomes
    rather than being left out of the sentence. A headline that reports two
    decided periods out of eight and stays silent about the other six reads as
    if six periods agreed with it.
    """
    against = [r.regime.label for r in results if r.verdict.startswith("moved AGAINST")]
    with_book = [r.regime.label for r in results if r.verdict.startswith("moved WITH")]
    unclear = [
        r.regime.label for r in results if r.verdict.startswith("no relationship") or r.verdict.startswith("not enough")
    ]
    total = len(results)
    if total == 0:
        return f"{hedge_ticker}: no period was supplied, so nothing was measured."
    unit = "period" if total == 1 else "periods"
    parts: list[str] = []
    if against:
        parts.append(f"moved AGAINST your book in {len(against)} of {total} {unit} ({', '.join(against)})")
    if with_book:
        parts.append(f"moved WITH your book in {len(with_book)} ({', '.join(with_book)})")
    if unclear:
        parts.append(
            f"and in the remaining {len(unclear)} ({', '.join(unclear)}) this much data could not tell either way"
        )
    if not parts:
        return f"{hedge_ticker}: no period in this window could be measured."
    if against or with_book:
        return f"{hedge_ticker} " + ", ".join(parts) + "."
    return (
        f"{hedge_ticker}: in none of the {total} {unit} "
        f"({', '.join(unclear)}) did this much data resolve whether it moved "
        "with your book or against it."
    )


# --------------------------------------------------------------------------- #
# 3. Drawdown coincidence
# --------------------------------------------------------------------------- #
def drawdown_coincidence(
    panel: Any,
    weights: Mapping[str, float] | pd.Series,
    *,
    worst_fraction: float = _WORST_FRACTION,
) -> dict[str, Any]:
    """On the portfolio's worst days, what did each holding actually do?

    WHAT IT COMPUTES
        Builds the portfolio's daily return from ``weights``, takes the worst
        ``worst_fraction`` of days (default 5%, at least
        ``_MIN_WORST_DAYS`` = 5 of them or nothing is reported), and for every
        holding reports:

          * ``n_fell`` — on how many of those worst days it closed down. This
            is a COUNT. It needs no distribution, no model and no assumption,
            and the holder can check it against their own broker statement.
          * ``mean_return_on_worst_days`` vs ``mean_return_on_other_days`` —
            what it did then versus the rest of the window.
          * ``fell_frequency_all_days`` — how often it closed down across the
            whole window, which is the only honest comparison for ``n_fell``.
            A stock that falls on 48% of all days falling on 7 of 10 crash
            days is unremarkable; one that falls on 20% of all days doing the
            same is not.
          * ``binomial_p_floor`` — the exact binomial tail for ``n_fell``
            given ``fell_frequency_all_days``. Labelled a FLOOR because it
            assumes the worst days are independent draws, and they are not:
            ``worst_day_clustering`` reports how few distinct weeks they fall
            in, and a p computed on clustered days is too small.

    WHY THE COUNT IS THE HEADLINE AND THE p IS THE FOOTNOTE
        "Your hedge fell with everything else on 7 of your 10 worst days" is
        checkable by a stranger against their own account in about a minute.
        A p-value is not. The count is the claim; the p-value is only there to
        stop a reader over-reading a count of 6 of 10 on a coin flip.

    WHERE IT MISLEADS
        * THE WORST DAYS ARE A SELECTED SAMPLE. Every number here is
          conditional on that selection and estimates nothing unconditional.
        * The worst days cluster. Ten worst days routinely fall in three or
          four weeks — often inside one crash — so they are nothing like ten
          independent observations, and the reported clustering should be read
          before the p.
        * A day is "down" if its simple return is strictly negative. A holding
          that is flat on a crash day (a bond fund that did not trade) counts
          as not falling, which flatters it.
        * A day on which ANY holding did not trade is dropped from the
          portfolio series entirely, never zero-filled, and the number of days
          dropped is reported in ``excluded``. One thin holding therefore
          shortens the window for everything — visibly, not silently. It also
          means every holding shares one denominator, so no row in the table
          is counted out of a different number of days.

    Parameters
    ----------
    panel
        Returns panel (see ``_coerce_panel``).
    weights
        Ticker -> weight. Renormalised to sum to 1 in absolute value; the
        input sum is reported so a typo is visible.
    worst_fraction
        Fraction of days treated as "the worst days". Default 0.05.

    Returns
    -------
    dict
        Keys: ``method``, ``window``, ``n_days``, ``n_worst_days``,
        ``worst_day_threshold``, ``worst_days`` (date, portfolio return),
        ``portfolio_weighting``, ``holdings`` (one dict each, sorted by
        ``n_fell`` descending), ``worst_day_clustering``, ``excluded``,
        ``headline``, ``caveats``.

    Raises
    ------
    ValueError
        Unusable panel, no usable holding, ``worst_fraction`` outside (0, 0.5],
        or too few days to produce ``_MIN_WORST_DAYS`` worst days.
    """
    if not (0.0 < float(worst_fraction) <= 0.5):
        raise ValueError(
            f"worst_fraction must be in (0, 0.5], got {worst_fraction!r}: "
            "beyond a half the 'worst days' are just the days."
        )
    frame = _coerce_panel(panel)
    weights = _coerce_weights(weights)
    if not weights:
        raise ValueError("weights is empty: there is no portfolio to measure.")

    frame, excluded = scrub_suspect_returns(frame)
    usable: list[str] = []
    for ticker in dict.fromkeys(str(t) for t in weights):
        if ticker not in frame.columns:
            excluded.append((ticker, "not present in the returns panel"))
            continue
        n_obs = int(frame[ticker].notna().sum())
        if n_obs < MIN_OVERLAP_DAYS:
            excluded.append(
                (
                    ticker,
                    f"only {n_obs} non-missing daily returns, below the {MIN_OVERLAP_DAYS}-day floor",
                )
            )
            continue
        usable.append(ticker)
    if not usable:
        raise ValueError(
            "no holding survived: "
            + "; ".join(f"{t} ({why})" for t, why in excluded)
            + ". There is no portfolio to measure."
        )

    raw = {t: float(weights[t]) for t in usable}
    total = sum(abs(v) for v in raw.values())
    if total <= 0.0:
        raise ValueError("supplied weights sum to zero in absolute value.")
    w = pd.Series({t: raw[t] / total for t in usable})

    sub = frame[usable]
    port = sub.mul(w, axis=1).sum(axis=1, skipna=False)
    port = port.dropna()
    n_days = int(port.size)
    n_days_dropped = int(frame.index.size - n_days)
    if n_days == 0:
        raise ValueError(
            "there is no day on which every holding traded, so no portfolio "
            "return series exists. Shorten the window or drop the holding with "
            "the sparsest history."
        )

    n_worst = int(round(float(worst_fraction) * n_days))
    if n_worst < _MIN_WORST_DAYS:
        raise ValueError(
            f"{n_days} days at {worst_fraction:.0%} gives {n_worst} worst days, "
            f"below the {_MIN_WORST_DAYS}-day floor. A count out of two or "
            "three days is an anecdote, not a statistic, so nothing is "
            f"reported. About {int(math.ceil(_MIN_WORST_DAYS / worst_fraction))} "
            "trading days are needed."
        )

    worst_idx = port.nsmallest(n_worst).sort_index().index
    threshold = float(port.loc[worst_idx].max())
    other_idx = port.index.difference(worst_idx)
    n_weeks, week_reason = _weekly_clusters(pd.DatetimeIndex(worst_idx))

    holdings: list[dict[str, Any]] = []
    for ticker in usable:
        col = frame[ticker]
        on_worst = col.loc[worst_idx].dropna()
        on_other = col.loc[other_idx].dropna()
        all_obs = col.dropna()
        n_obs_worst = int(on_worst.size)
        n_fell = int((on_worst.to_numpy() < 0.0).sum())
        fell_freq = float((all_obs.to_numpy() < 0.0).mean()) if all_obs.size else None

        # INVARIANT: the portfolio series only contains days on which EVERY
        # usable holding traded, so every worst day has a price for every
        # holding and the denominator below is always the full n_worst. If the
        # day-dropping policy above is ever relaxed, this stops the table
        # printing counts against a denominator that silently shrank.
        if n_obs_worst != n_worst:
            raise RuntimeError(
                f"{ticker} has {n_obs_worst} prices across the {n_worst} worst "
                "days. The portfolio series is supposed to contain only days "
                "on which every holding traded, so this cannot happen without "
                "the construction above having changed. Refusing to print a "
                "count against an unstated denominator."
            )

        p_floor: float | None = None
        if fell_freq is not None and 0.0 < fell_freq < 1.0:
            p_floor = float(_scipy_stats.binomtest(n_fell, n_obs_worst, fell_freq, alternative="greater").pvalue)

        statement = (
            f"{ticker} fell on {n_fell} of your {n_obs_worst} worst days"
            + (
                f"; it fell on {fell_freq:.0%} of all {all_obs.size} days in the window"
                if fell_freq is not None
                else ""
            )
            + "."
        )
        holdings.append(
            {
                "ticker": ticker,
                "weight": float(w[ticker]),
                "n_worst_days_with_data": n_obs_worst,
                "n_fell": n_fell,
                "mean_return_on_worst_days": float(on_worst.mean()),
                "mean_return_on_other_days": (float(on_other.mean()) if on_other.size else None),
                "fell_frequency_all_days": fell_freq,
                "binomial_p_floor": p_floor,
                "statement": statement,
            }
        )

    holdings.sort(key=lambda h: (-(h["n_fell"] or -1), h["ticker"]))
    if n_days_dropped:
        excluded.append(
            (
                "(calendar)",
                f"{n_days_dropped} of {frame.index.size} panel days had at "
                "least one holding without a price and were dropped from the "
                "portfolio series rather than zero-filled",
            )
        )

    headline = _drawdown_headline(holdings, n_worst, n_weeks)

    return {
        "method": (
            f"The worst {worst_fraction:.0%} of days by realised portfolio "
            f"return ({n_worst} of {n_days} days on which every holding "
            "traded), ranked on the portfolio series built from the supplied "
            "weights. Days are selected AFTER the fact; every number below is "
            "conditional on that selection."
        ),
        "window": (
            f"{port.index[0].date().isoformat()} to {port.index[-1].date().isoformat()} (n={n_days} trading days)"
        ),
        "n_days": n_days,
        "n_worst_days": n_worst,
        "worst_day_threshold": threshold,
        "worst_days": tuple((ts.date(), float(port.loc[ts])) for ts in worst_idx),
        "portfolio_weighting": (f"{len(usable)} holdings, renormalised to sum to 1 (input absolute sum {total:.4g})"),
        "holdings": tuple(holdings),
        "worst_day_clustering": {
            "n_distinct_weeks": n_weeks,
            "reason": week_reason,
            "note": (
                f"Those {n_worst} worst days fall in "
                f"{n_weeks if n_weeks is not None else '?'} distinct calendar "
                "weeks. They are not that many independent events, which is "
                "why the binomial p beside each holding is a FLOOR."
            ),
        },
        "excluded": tuple(excluded),
        "headline": headline,
        "caveats": (
            "These days were chosen because they were the worst. Nothing here estimates an unconditional quantity.",
            "A 'fall' is a strictly negative simple return. A holding that did "
            "not trade that day counts as not falling, which flatters it.",
            "The binomial p assumes the worst days are independent draws. They "
            "are not; see the clustering note. Read it as a lower bound.",
            "This describes days that already happened. It is not a forecast.",
        ),
    }


#: A holding that fell on at least this share of the worst days is described
#: as having "fallen with the book". Stated rather than tuned: it is a
#: reporting threshold for prose, and every underlying count is printed beside
#: it so a reader never has to trust the threshold.
_TOGETHER_SHARE = 0.80


def _drawdown_headline(
    holdings: Sequence[Mapping[str, Any]],
    n_worst: int,
    n_weeks: int | None,
) -> str:
    """State how much of the book moved as one block, and name the exception.

    The interesting fact on a crash day is not that a stock fell — most things
    fall — but HOW MANY of the holdings fell together, and whether the one the
    holder keeps for protection was among them. Both halves are counts the
    reader can check line by line in the table below the headline.
    """
    scored = [h for h in holdings if h.get("n_fell") is not None]
    if not scored:
        return f"No holding had a price on any of the {n_worst} worst days."
    together = [
        h
        for h in scored
        if h["n_worst_days_with_data"] > 0 and h["n_fell"] / h["n_worst_days_with_data"] >= _TOGETHER_SHARE
    ]
    weeks = f" Those {n_worst} days fall in {n_weeks} distinct calendar weeks." if n_weeks is not None else ""
    if len(together) == len(scored):
        return (
            f"On your {n_worst} worst days, all {len(scored)} of your holdings "
            f"fell together on at least {_TOGETHER_SHARE:.0%} of them. Nothing "
            f"in this portfolio was doing anything different.{weeks}"
        )
    if not together:
        return f"On your {n_worst} worst days, no holding fell on as many as {_TOGETHER_SHARE:.0%} of them.{weeks}"
    exceptions = sorted(
        (h for h in scored if h not in together),
        key=lambda h: h["n_fell"] / max(h["n_worst_days_with_data"], 1),
    )
    names = ", ".join(f"{h['ticker']} ({h['n_fell']}/{h['n_worst_days_with_data']})" for h in exceptions[:3])
    return (
        f"On your {n_worst} worst days, {len(together)} of your {len(scored)} "
        f"holdings fell together on at least {_TOGETHER_SHARE:.0%} of them. The "
        f"exception{'s were' if len(exceptions) > 1 else ' was'}: {names}."
        f"{weeks}"
    )


# --------------------------------------------------------------------------- #
# 4. Rendering — plain text, so the caller can see exactly what a reader sees
# --------------------------------------------------------------------------- #
def render_hedge_result(result: HedgeResult) -> str:
    """Render a ``HedgeResult`` as plain text, with every caveat attached.

    Rendering lives beside the computation on purpose: a renderer elsewhere
    could drop the sample size, and a claim without its n is the thing this
    module exists to prevent.
    """
    lines: list[str] = []
    lines.append(result.headline)
    lines.append("")
    lines.append(
        f"Tested: {result.hedge_ticker} against a book of "
        f"{len(result.book_tickers)} holdings ({', '.join(result.book_tickers)})."
    )
    lines.append(f"Book construction: {result.book_weighting}.")
    lines.append(
        f"Window: {result.window_start.isoformat()} to "
        f"{result.window_end.isoformat()} ({result.n_days_total} trading days)."
    )
    lines.append(f"Periods: {result.regime_method} — {result.regime_method_note}")
    lines.append(f"Interval: {result.bootstrap_note}")
    lines.append("")
    for r in result.regimes:
        lines.append(f"  {r.regime.label}  [{r.regime.window}]")
        lines.append(f"      {r.verdict}")
        lines.append(f"      {r.detail}")
        if r.n_days_dropped:
            lines.append(
                f"      {r.n_days_dropped} day(s) in this period dropped: the hedge or some book holding had no price."
            )
        if r.power_if_days_independent is not None:
            pw = f"{r.power_if_only_weeks_independent:.0%}" if r.power_if_only_weeks_independent is not None else "n/a"
            lines.append(
                f"      power vs a stipulated r = {r.reference_correlation:+.2f}: "
                f"{r.power_if_days_independent:.0%} if the "
                f"{r.n_days_used} days are independent, {pw} if only the "
                f"{r.n_independent_weeks} weeks are."
            )
        lines.append("")
    if result.excluded:
        lines.append("Excluded (nothing is dropped silently):")
        for ticker, why in result.excluded:
            lines.append(f"  - {ticker}: {why}")
        lines.append("")
    lines.append("Read this with:")
    for c in result.caveats:
        lines.append(f"  * {c}")
    return "\n".join(lines)


def render_drawdown_coincidence(result: Mapping[str, Any], *, max_days_shown: int = 10) -> str:
    """Render ``drawdown_coincidence`` output as plain text, caveats attached.

    Only the ``max_days_shown`` deepest days are listed, and the line says so
    explicitly; the per-holding counts below are always over ALL the worst
    days, never over the shortened list. Truncating the evidence without
    saying so is how a table starts lying.
    """
    lines: list[str] = []
    lines.append(str(result["headline"]))
    lines.append("")
    lines.append(f"Window: {result['window']}")
    lines.append(f"Method: {result['method']}")
    lines.append(
        f"Worst-day threshold: a day counted as one of the worst if the "
        f"portfolio returned {result['worst_day_threshold']:+.2%} or less."
    )
    lines.append(f"Portfolio: {result['portfolio_weighting']}")
    lines.append("")
    shown = sorted(result["worst_days"], key=lambda dr: dr[1])[:max_days_shown]
    n_worst = int(result["n_worst_days"])
    if n_worst <= max_days_shown:
        lines.append(f"The {n_worst} worst days:")
    else:
        lines.append(f"The {len(shown)} deepest of those {n_worst} days (the full list is in result['worst_days']):")
    for day, ret in shown:
        lines.append(f"  {day.isoformat()}  {ret:+.2%}")
    lines.append(f"  {result['worst_day_clustering']['note']}")
    lines.append("")
    lines.append("What each holding did on those days:")
    for h in result["holdings"]:
        if h["n_fell"] is None:
            lines.append(f"  {h['ticker']:<14} {h['statement']}")
            continue
        p = h["binomial_p_floor"]
        lines.append(
            f"  {h['ticker']:<14} weight {h['weight']:>6.1%}  "
            f"fell on {h['n_fell']}/{h['n_worst_days_with_data']} worst days  "
            f"(fell on {h['fell_frequency_all_days']:.0%} of all days)"
        )
        other = h["mean_return_on_other_days"]
        lines.append(
            f"  {'':<14} mean on those days {h['mean_return_on_worst_days']:+.2%}"
            + (f", mean on every other day {other:+.2%}" if other is not None else "")
            + (
                ""
                if p is None
                # "p >= 0.000" reads as a claim of exactly zero. A p below the
                # print precision is reported as below it, never as zero.
                else (", binomial p >= 0.001" if p < 0.001 else f", binomial p >= {p:.3f}")
            )
        )
    lines.append("")
    if result["excluded"]:
        lines.append("Excluded (nothing is dropped silently):")
        for ticker, why in result["excluded"]:
            lines.append(f"  - {ticker}: {why}")
        lines.append("")
    lines.append("Read this with:")
    for c in result["caveats"]:
        lines.append(f"  * {c}")
    return "\n".join(lines)
