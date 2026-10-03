"""TirraMind — Portfolio Attribution: which decisions cost what, over many windows.

WHAT THIS MODULE IS FOR
    A broker app shows profit and loss against your BUY PRICE. It never shows
    it against the alternative of having done nothing — of having owned the
    index instead of picking names out of it. That second number is the one
    the holder has never computed, and it is arithmetic, not a forecast.

    This module computes it, per holding, and refuses to compute it on one
    window.

      * ``holding_attribution``      — one window. Per holding: its own total
        return, the benchmark's over the SAME days, the difference, and the
        WEIGHTED contribution to the book's gap versus the benchmark. Ranked by
        contribution, because a name that lost 50% at a 1% weight is a footnote
        and a name that lost 20% at a 30% weight is the story.
      * ``decision_summary``         — "6 of 7 holdings trailed the index", plus
        the two counterfactuals that stop the headline being a smear: what the
        book returned WITHOUT its worst contributor, and without its BEST one.
      * ``multi_window_attribution`` — the same computation over every window a
        pre-committed rule produces, with the count of windows that contradict
        the full-window result printed first.
      * ``survivorship_note``        — the bias that cannot be measured from the
        data at hand, stated rather than buried.

WHY MULTI-WINDOW IS NOT AN OPTION
    A single window is a cherry-pick whether or not anybody cherry-picked it.
    Pick a different three years and the sign flips. The whole claim of this
    product is "we do not let you fool yourself"; publishing a one-window
    number would be that exact failure, in public, on the front page. So the
    single-window entry point exists to be *called by* the multi-window one,
    and the multi-window result reports the windows that disagree BEFORE the
    ones that agree.

    The window rules offered are pre-committed and untunable: calendar years,
    calendar halves, or fixed-length windows stepped at a fixed interval. No
    rule is offered that searches for the window that maximises anything,
    because such a search always succeeds.

WHAT THIS MODULE WILL NEVER DO
    Give advice. Every string it emits is past tense and descriptive: "trailed
    NIFTYBEES.NS by 14.2% over 2023-10-04..2026-09-26", never "you should
    index", never "consider", never a word about what happens next. Facts
    about realised prices over a stated window carry no forecast. That is the
    only reason this is publishable by people who are not registered advisers.

WHERE THIS MISLEADS — read before showing any of it to a stranger
    1. SURVIVORSHIP. The book contains what the holder STILL holds. Names they
       sold are invisible here, and people sell losers and keep winners about
       as often as the reverse. This biases the measured gap in an UNKNOWN
       direction and by an unknown amount. ``survivorship_note`` says so in
       words meant to be printed, not logged.
    2. WEIGHTS ARE A SNAPSHOT. ``PricePanel.weights`` is computed at the LAST
       aligned close. The attribution needs START weights. They are derived
       exactly — ``w_start ∝ w_end / (1 + R)`` — under the assumption that the
       holder held the same share counts for the whole window. If they added to
       a position halfway through, that assumption is wrong and this module
       cannot tell. ``AttributionResult.weights_start`` is exposed so the
       assumption is checkable rather than hidden.
    3. THE GAP IS NOT A COUNTERFACTUAL P&L. "Owning the index instead" ignores
       every tax, every cost, and every reason the holder had. It is the
       arithmetic difference between two realised price paths.
    4. RETURN PER UNIT OF VOLATILITY IS NOT A SHARPE RATIO. No risk-free rate
       is subtracted. It is annualised return divided by annualised volatility
       and nothing more.
    5. ANNUALISING A SHORT WINDOW IS EXTRAPOLATION. Any window under one
       trading year carries ``annualisation_is_extrapolation=True``; a 60-day
       window annualised reports a number nothing in the data supports.
    6. A BLANKED CORPORATE ACTION IS AN ASSUMPTION. ``scrub_suspect_returns``
       removes >=50% single-day moves (GOLDBEES.NS's 1:100 split survives
       yfinance's adjustment). A blanked day is then compounded as 0.0%, which
       is a guess. Every blanked cell is reported with its date and value.
    7. INTERSECTING WITH THE BENCHMARK COSTS DAYS. The book and the benchmark
       are compounded over the days BOTH traded, never over different ones. A
       window that loses more than ``MAX_BENCHMARK_DAY_LOSS`` of its days to
       that intersection is refused, not measured.

ZERO AND NONE DISCIPLINE
    A contribution of 0.0 and "could not be computed" are never the same
    value. Anything uncomputable is ``None`` with a reason beside it.

NO SILENT EXCLUSION
    Every holding dropped — absent from the panel, zero weight, no price, a
    total return of -100% that makes the start weight undefined — appears in
    ``excluded`` with a reason. A result that quietly measured 5 of 7 holdings
    is the confident wrong answer this product exists to refuse.

RECONCILIATION
    Weighted contributions sum to the book's gap versus the benchmark. Under
    the buy-and-hold basis this is an algebraic identity, and a residual above
    ``RECONCILIATION_TOLERANCE`` raises rather than prints. Under the
    fixed-weight basis it is NOT an identity — the difference is the
    rebalancing-and-compounding effect, and it is reported as that.

REUSE
    ``scrub_suspect_returns`` and ``regime_split`` come from
    ``agent.portfolio.beliefs`` and are not reimplemented here. Price fetching
    and its disk cache come from ``agent.portfolio.holdings``.

Layer: 3 (world model — facts about a held book). Stateless apart from the
price cache that ``fetch_benchmark`` inherits from ``holdings``.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from agent.portfolio.beliefs import (
    _coerce_panel as _coerce_returns,  # noqa: PLC2701 — see NOTE below
)
from agent.portfolio.beliefs import (
    _coerce_weights as _coerce_weight_map,  # noqa: PLC2701
)
from agent.portfolio.beliefs import (
    regime_split,
    scrub_suspect_returns,
)

# NOTE on the two underscore imports above. They are deliberate reuse, not a
# convenience. ``_coerce_panel`` carries the guard that refuses a PRICE panel
# handed in where a RETURNS panel belongs — the integration mistake that makes
# every correlation 0.95 and every number wrong in the same direction. If this
# module grew its own copy, the two copies would drift and one of them would be
# the lenient one. ``_coerce_weights`` carries the ``pandas.Series`` trap
# (iterating a Series yields its VALUES, so ``for t in weights`` silently walks
# floats). Both are tested in tests/test_portfolio_beliefs.py. Neither file is
# edited by this module.

__all__ = [
    "MAX_BENCHMARK_DAY_LOSS",
    "MIN_WINDOW_DAYS",
    "RECONCILIATION_TOLERANCE",
    "TRADING_DAYS_PER_YEAR",
    "AttributionResult",
    "Exclusion",
    "HoldingResult",
    "MultiWindowResult",
    "SeriesStats",
    "Window",
    "WindowAttribution",
    "decision_summary",
    "fetch_benchmark",
    "format_attribution_report",
    "format_multi_window_report",
    "holding_attribution",
    "multi_window_attribution",
    "survivorship_note",
]


# --------------------------------------------------------------------------- #
# Constants. Each one silently decides whether a claim is made at all, so each
# one is named and justified here rather than inlined at its use site.
# --------------------------------------------------------------------------- #

#: Trading days per year, for annualising a total return and a daily standard
#: deviation. 252 is the convention; NSE runs ~246-250, so an annualised figure
#: here is very slightly overstated in magnitude. The error is under 2% of the
#: figure and does not change any sign.
TRADING_DAYS_PER_YEAR = 252

#: Below this many daily returns, a window is reported but carries no return
#: figure. 60 trading days is about a quarter. A quarter's total return
#: annualised is a number with no information in it, and the product's failure
#: mode is exactly a confident number with no information in it. This happens
#: to equal ``beliefs.MIN_REGIME_DAYS``, for a different reason: that floor is
#: about correlation power, this one is about extrapolation.
MIN_WINDOW_DAYS = 60

#: Fraction of the book's trading days that may be lost to intersecting with
#: the benchmark's calendar before the window is refused. 5% of a year is 12
#: days; a 12-day hole in a total return is not a rounding error.
MAX_BENCHMARK_DAY_LOSS = 0.05

#: Absolute tolerance on ``sum(contributions) - gap`` under the buy-and-hold
#: basis, where the two are algebraically identical and differ only by float64
#: accumulation over a few thousand multiplications. Measured residuals on real
#: books are ~1e-16; 1e-9 is four orders of magnitude of headroom and still far
#: tighter than any residual a real bug would produce.
RECONCILIATION_TOLERANCE = 1e-9

#: A return of exactly -100% makes ``w_start ∝ w_end / (1 + R)`` undefined. Any
#: growth factor at or below this is treated as "this position went to zero",
#: which is an exclusion with a reason, not a division.
_MIN_GROWTH_FACTOR = 1e-9

_BASES = {
    # NOTE: this prose is RENDERED, and tests/test_portfolio_attribution.py
    # runs it through concentration._advice_words_in. "Buy and hold" and
    # "trimming winners" both trip that list — correctly, since the list cannot
    # know a transaction verb is being used descriptively, and a list that made
    # exceptions for this module would make them for the next one too. So the
    # basis is described without transaction verbs. The KEY stays
    # "buy_and_hold" because it is an API identifier and is never rendered as
    # prose.
    "buy_and_hold": (
        "Held unchanged. The share counts implied by your stated weights stay "
        "the same for the whole window; the weights themselves drift as prices "
        "move. Contributions sum to the gap exactly, because they are the same "
        "arithmetic rearranged."
    ),
    "fixed_weight": (
        "Fixed weight. The book is returned to your stated weights at every "
        "close. Nobody holds a book this way, but it isolates what the WEIGHTS "
        "did from what the drift did. Contributions do not sum to the gap; the "
        "difference is the rebalancing effect and is reported as that."
    ),
}


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Window:
    """The dated sample a number was computed on. Never optional, never a
    phrase like "the last three years" — an actual pair of trading days."""

    label: str
    start: dt.date
    end: dt.date
    n_days: int

    def __str__(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()} (n={self.n_days} trading days)"

    @property
    def years(self) -> float:
        return self.n_days / TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class Exclusion:
    """Something not used, and why. Silent exclusion is how a confident wrong
    answer gets made, so nothing is dropped without one of these."""

    name: str
    reason: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.name}: {self.reason}" + (f" ({self.detail})" if self.detail else "")


@dataclass(frozen=True)
class SeriesStats:
    """Total, annualised, volatility and return-per-unit-volatility for one
    daily return series.

    ``return_per_vol`` is annualised return divided by annualised volatility.
    It is NOT a Sharpe ratio: no risk-free rate is subtracted, so it is not
    comparable to a published Sharpe and must never be labelled as one.

    ``annualisation_is_extrapolation`` is True whenever the window is shorter
    than one trading year. The annualised figure is still computed — hiding it
    would be its own dishonesty — but it is flagged everywhere it is shown.
    """

    n_days: int
    total_return: float
    annualised_return: float
    volatility: float
    return_per_vol: float | None
    return_per_vol_reason: str = ""
    annualisation_is_extrapolation: bool = False

    def line(self, name: str) -> str:
        rpv = "  n/a" if self.return_per_vol is None else f"{self.return_per_vol:5.2f}"
        flag = " *extrapolated*" if self.annualisation_is_extrapolation else ""
        return (
            f"{name:<16} {self.total_return:+8.1%} total  {self.annualised_return:+7.1%}/yr{flag}  "
            f"vol {self.volatility:6.1%}  return-per-vol {rpv}"
        )


@dataclass(frozen=True)
class HoldingResult:
    """One holding's arithmetic against the benchmark over one window.

    ``contribution`` is ``weight_start * (total_return - benchmark_return)``
    and is THE number to rank by. ``excess_return`` alone ranks a 1%-weight
    disaster above a 30%-weight drag and tells the story backwards.

    ``weight_start`` is the share of the book's value this position held on the
    window's FIRST day — derived, not pasted. ``weight_end`` is what was
    handed in. They differ by exactly the position's own performance, which is
    why both are shown.
    """

    ticker: str
    weight_start: float
    weight_end: float
    total_return: float
    benchmark_return: float
    excess_return: float
    contribution: float
    n_days: int
    scrubbed_days: int = 0
    notes: tuple[str, ...] = ()

    @property
    def beat_benchmark(self) -> bool:
        return self.excess_return > 0.0

    def statement(self, benchmark: str) -> str:
        """One past-tense factual sentence. No verb implying a future."""
        verb = "returned" if self.total_return >= 0 else "returned"
        rel = "ahead of" if self.beat_benchmark else "behind"
        return (
            f"{self.ticker} {verb} {self.total_return:+.1%} while {benchmark} returned "
            f"{self.benchmark_return:+.1%} — {abs(self.excess_return):.1%} {rel} it. "
            f"At {self.weight_start:.1%} of the book on day one that moved the book's "
            f"gap by {self.contribution:+.1%}."
        )


class AttributionResult(list):
    """``list[HoldingResult]`` that also carries the window, the totals, the
    reconciliation and everything excluded.

    It is a real ``list``, so ``holding_attribution`` honours its documented
    ``list[HoldingResult]`` contract and a caller can iterate it without
    knowing this type exists. The extra attributes exist because a bare list
    has nowhere to put an exclusion, and an exclusion with nowhere to go
    becomes a silent one.

    Ordering is by ``contribution`` ASCENDING — the position that cost the most
    is first. That is deliberate: the reader's eye starts at the top, and the
    top of this list should be the fact they did not know.
    """

    def __init__(
        self,
        holdings: Sequence[HoldingResult] = (),
        *,
        window: Window,
        basis: str,
        basis_note: str,
        benchmark_ticker: str,
        book: SeriesStats,
        benchmark: SeriesStats,
        weights_start: Mapping[str, float] | None = None,
        weights_end: Mapping[str, float] | None = None,
        excluded: Sequence[Exclusion] = (),
        notes: Sequence[str] = (),
        usable: bool = True,
        unusable_reason: str = "",
    ) -> None:
        super().__init__(holdings)
        self.window = window
        self.basis = basis
        self.basis_note = basis_note
        self.benchmark_ticker = benchmark_ticker
        self.book = book
        self.benchmark = benchmark
        self.weights_start = dict(weights_start or {})
        self.weights_end = dict(weights_end or {})
        self.excluded = tuple(excluded)
        self.notes = tuple(notes)
        self.usable = usable
        self.unusable_reason = unusable_reason

    # -- the three numbers the headline is made of -------------------------
    @property
    def gap_total(self) -> float:
        """Book total return minus benchmark total return, same days."""
        return self.book.total_return - self.benchmark.total_return

    @property
    def gap_annualised(self) -> float:
        return self.book.annualised_return - self.benchmark.annualised_return

    @property
    def contribution_sum(self) -> float:
        return float(sum(h.contribution for h in self))

    @property
    def residual(self) -> float:
        """``gap_total - sum(contributions)``. Zero by identity under
        buy-and-hold; the rebalancing effect under fixed weights."""
        return self.gap_total - self.contribution_sum

    @property
    def residual_explanation(self) -> str:
        if self.basis == "buy_and_hold":
            return (
                f"residual {self.residual:+.2e} — float64 accumulation only. Under buy-and-hold "
                "the weighted contributions ARE the gap, rearranged; anything above "
                f"{RECONCILIATION_TOLERANCE:.0e} would be a bug and raises."
            )
        return (
            f"residual {self.residual:+.2%} of the {self.gap_total:+.1%} gap. Under fixed weights "
            "the book is returned to your stated weights every close, so the book's compounded "
            "return is not the weighted sum of the holdings' compounded returns. The difference "
            "is the rebalancing-and-compounding effect: the arithmetic of moving money out of "
            "whatever rose and into whatever fell at every close, not any holding's doing."
        )

    @property
    def n_beat(self) -> int:
        return sum(1 for h in self if h.beat_benchmark)

    @property
    def n_trailed(self) -> int:
        return sum(1 for h in self if not h.beat_benchmark)


@dataclass(frozen=True)
class WindowAttribution:
    """One window's result inside a multi-window run, flattened to the few
    numbers the stability verdict is computed from. The full
    ``AttributionResult`` is kept so the per-holding detail is never lost."""

    window: Window
    result: AttributionResult | None
    skipped_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.result is not None and self.result.usable

    @property
    def gap_total(self) -> float | None:
        return None if not self.usable else self.result.gap_total  # type: ignore[union-attr]

    @property
    def gap_annualised(self) -> float | None:
        return None if not self.usable else self.result.gap_annualised  # type: ignore[union-attr]


@dataclass(frozen=True)
class MultiWindowResult:
    """Every window the pre-committed rule produced, and what disagrees.

    ``windows_contradicting`` is listed and rendered BEFORE
    ``windows_agreeing``. That ordering is the product: a reader who stops
    reading after one line should have read the exception, not the headline.

    ``direction_consistent`` is True only when every usable window has the same
    sign of gap as the full window. It is the ONLY condition under which the
    single-window headline may be published without the qualifier beside it.
    """

    method: str
    method_note: str
    full_window: AttributionResult
    windows: tuple[WindowAttribution, ...]
    benchmark_ticker: str
    independent_windows: int
    overlap_note: str
    notes: tuple[str, ...] = ()
    excluded: tuple[Exclusion, ...] = ()

    @property
    def usable_windows(self) -> tuple[WindowAttribution, ...]:
        return tuple(w for w in self.windows if w.usable)

    @property
    def full_sign(self) -> int:
        g = self.full_window.gap_total
        return 0 if g == 0 else (1 if g > 0 else -1)

    @property
    def windows_agreeing(self) -> tuple[WindowAttribution, ...]:
        s = self.full_sign
        return tuple(w for w in self.usable_windows if _sign(w.gap_total) == s)

    @property
    def windows_contradicting(self) -> tuple[WindowAttribution, ...]:
        s = self.full_sign
        return tuple(w for w in self.usable_windows if _sign(w.gap_total) != s)

    @property
    def direction_consistent(self) -> bool:
        return bool(self.usable_windows) and not self.windows_contradicting

    @property
    def gaps(self) -> list[float]:
        return [w.gap_total for w in self.usable_windows if w.gap_total is not None]

    @property
    def headline_qualifier(self) -> str:
        """The sentence that must travel with the full-window number.

        Past tense, no recommendation, no forecast. This is the string the web
        layer is expected to print underneath the headline; it exists so that
        printing the headline without it is an omission somebody has to commit
        on purpose.
        """
        n = len(self.usable_windows)
        if n == 0:
            return (
                f"No {self.method} window in this panel was long enough to measure "
                f"({MIN_WINDOW_DAYS} trading days minimum), so the full-window figure stands alone "
                "and has not been checked against any other window."
            )
        k = len(self.windows_agreeing)
        if self.direction_consistent:
            return (
                f"The same direction held in all {n} {self.method} windows measured "
                f"({_window_span(self.usable_windows)})."
            )
        worst = max(self.windows_contradicting, key=lambda w: abs(w.gap_total or 0.0))
        return (
            f"The same direction held in {k} of {n} {self.method} windows. It did NOT hold in "
            f"{len(self.windows_contradicting)}: over {_window_cell(worst.window, pad=False)} the book was "
            f"{worst.gap_total:+.1%} against {self.benchmark_ticker}, the other way round."
        )


def _contributor_role(contribution: float, *, extreme: str) -> str:
    """Name a contributor by what it actually DID, not by where it sorted.

    The ranked list's first row is the lowest contribution and its last row the
    highest, but "lowest" is not "negative". On a book where every holding beat
    the benchmark, the bottom row still ADDED to the gap, and calling it "the
    largest negative contributor" — as an earlier draft of this module did —
    is a sentence that is false about a real book on the page of a product
    whose entire pitch is that its numbers are checkable. The label is chosen
    from the sign, so it cannot be right only when the reader lost.
    """
    if contribution < 0:
        return "the largest single negative contributor" if extreme == "lowest" else "the least negative contributor"
    if contribution > 0:
        return "the smallest positive contributor" if extreme == "lowest" else "the largest single positive contributor"
    return "a holding that moved the gap by exactly zero"


def _sign(x: float | None) -> int:
    if x is None:
        return 0
    return 0 if x == 0 else (1 if x > 0 else -1)


def _window_cell(w: Window, *, pad: bool = True) -> str:
    """Render a window without saying the same dates twice.

    A rolling window's label IS its date range, a calendar window's label is
    "2024". Printing ``label + str(window)`` unconditionally produced
    "2024-01-04..2025-01-13 (2024-01-04..2025-01-13 (n=252 trading days))" in
    the rolling report — noise in exactly the column a reader scans to find
    the window that disagrees.
    """
    span = f"{w.start.isoformat()}..{w.end.isoformat()}"
    if w.label == span:
        return f"{span} (n={w.n_days})"
    if pad:  # column form, for the tables
        return f"{w.label:<10} {span} (n={w.n_days})"
    return f"{w.label} ({span}, n={w.n_days})"  # prose form, for a sentence


def _window_span(ws: Sequence[WindowAttribution]) -> str:
    if not ws:
        return "no windows"
    return f"{ws[0].window.start.isoformat()}..{ws[-1].window.end.isoformat()}"


# --------------------------------------------------------------------------- #
# Panel / benchmark plumbing
# --------------------------------------------------------------------------- #
def _as_date(ts: Any) -> dt.date:
    return pd.Timestamp(ts).date()


def _benchmark_returns(benchmark: Any, *, name_hint: str = "") -> tuple[pd.Series, str]:
    """Coerce the benchmark argument into (daily returns series, ticker name).

    ACCEPTS
        * a ``pandas.Series`` of daily simple returns (its ``name`` is used as
          the ticker if set);
        * a single-column ``DataFrame``;
        * anything exposing ``.returns`` that is one of the above — notably a
          one-name ``PricePanel`` from ``fetch_benchmark``;
        * a ``(name, series)`` pair.

    WHERE IT MISLEADS
        It cannot verify that what it was handed is the benchmark the caller
        thinks it is. A caller that passes a returns series of the wrong
        instrument gets a perfectly reconciled, perfectly wrong answer. The
        ticker name is carried through into every rendered string precisely so
        that mistake is visible on the page.
    """
    if isinstance(benchmark, tuple) and len(benchmark) == 2:
        name_hint, benchmark = str(benchmark[0]), benchmark[1]
    frame = getattr(benchmark, "returns", benchmark)
    if isinstance(frame, pd.DataFrame):
        if frame.shape[1] != 1:
            raise ValueError(
                f"benchmark frame has {frame.shape[1]} columns; a benchmark is one instrument. "
                "Pass a Series, a one-column frame, or a one-name PricePanel."
            )
        name_hint = name_hint or str(frame.columns[0])
        frame = frame.iloc[:, 0]
    if not isinstance(frame, pd.Series):
        raise TypeError(
            "benchmark must be a pandas Series of daily simple returns, a one-column DataFrame, "
            f"a (name, series) pair, or an object exposing one as .returns; got {type(benchmark).__name__}."
        )
    ticker = name_hint or (str(frame.name) if frame.name is not None else "benchmark")
    series = pd.to_numeric(frame, errors="coerce").astype(float)
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index))
    series = series[~series.index.duplicated(keep="last")].sort_index()
    finite = series.to_numpy()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(f"benchmark {ticker} has no finite observations.")
    q = float(np.quantile(np.abs(finite), 0.99))
    if q > 1.0:
        raise ValueError(
            f"benchmark {ticker} does not look like daily simple returns: the 99th percentile of "
            f"|value| is {q:.4g}, a {q:.0%} move in one day. That is what a PRICE series looks "
            "like. Convert to returns before passing it."
        )
    return series.rename(ticker), ticker


def fetch_benchmark(
    ticker: str,
    *,
    period: str = "5y",
    base_currency: str | None = None,
    fetcher: Callable[[str, str], tuple[pd.Series, str] | None] | None = None,
) -> Any:
    """Fetch one benchmark instrument as a one-name ``PricePanel``.

    This is a thin wrapper over ``agent.portfolio.holdings.fetch_prices`` and
    exists only so the benchmark inherits the same disk cache, the same
    suffix probing (``NIFTYBEES`` -> ``NIFTYBEES.NS``), the same currency
    conversion and the same exclusion reporting as the holdings do. A public
    demo will be rate-limited; a second, uncached fetch path would be the thing
    that rate-limits it.

    WHERE THIS MISLEADS
        The benchmark is whatever ticker was asked for. ``NIFTYBEES.NS`` is an
        ETF, so its return is after its expense ratio and tracking error — it
        is what a holder could actually have owned, which is the right
        comparison, and NOT the Nifty 50 index level, which nobody can own.
        Saying "the Nifty returned X" when X came from the ETF would be wrong
        by roughly the expense ratio, every year, in the flattering direction.
    """
    from agent.portfolio.holdings import fetch_prices, parse_holdings  # noqa: PLC0415 — network-y

    held = parse_holdings(f"{ticker} 100%")
    if not held.entries:
        raise ValueError(f"{ticker!r} did not parse as a ticker.")
    return fetch_prices(held, period=period, base_currency=base_currency, fetcher=fetcher)


def _align(
    returns: pd.DataFrame,
    bench: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, list[Exclusion], list[str], float]:
    """Intersect the book's calendar with the benchmark's.

    Both sides are compounded over the SAME days or the comparison is not a
    comparison. The days lost are counted and reported; losing more than
    ``MAX_BENCHMARK_DAY_LOSS`` of them is a refusal, upstream.
    """
    excluded: list[Exclusion] = []
    notes: list[str] = []
    common = returns.index.intersection(bench.dropna().index)
    lost = len(returns.index) - len(common)
    frac = lost / max(1, len(returns.index))
    if lost:
        notes.append(
            f"{lost} of the book's {len(returns.index)} trading days ({frac:.1%}) are not in "
            f"{bench.name}'s calendar and were dropped from BOTH sides. Both totals below are "
            "compounded over the days they shared, not over every day the book traded."
        )
    return returns.loc[common], bench.loc[common], excluded, notes, frac


def _series_stats(daily: pd.Series) -> SeriesStats:
    """Total / annualised / volatility / return-per-vol for a daily series."""
    values = daily.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    total = float(np.prod(1.0 + values) - 1.0)
    growth = 1.0 + total
    if n == 0 or growth <= 0.0:
        annualised = float("nan") if n == 0 else -1.0
    else:
        annualised = float(growth ** (TRADING_DAYS_PER_YEAR / n) - 1.0)
    vol = float(np.std(values, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if n > 1 else float("nan")
    if not math.isfinite(vol) or vol <= 0.0:
        rpv: float | None = None
        reason = (
            "volatility could not be computed (fewer than 2 observations)"
            if n <= 1
            else "volatility is zero: this series did not move, so return per unit of it is undefined"
        )
    else:
        rpv, reason = annualised / vol, ""
    return SeriesStats(
        n_days=n,
        total_return=total,
        annualised_return=annualised,
        volatility=vol,
        return_per_vol=rpv,
        return_per_vol_reason=reason,
        annualisation_is_extrapolation=n < TRADING_DAYS_PER_YEAR,
    )


def _prepare(
    panel: Any,
    weights: Any,
    *,
    scrub: bool,
) -> tuple[pd.DataFrame, dict[str, float], list[Exclusion], list[str], dict[str, int]]:
    """Coerce the panel and weights, scrub corporate actions, drop what cannot
    be measured — and return a reason for every drop."""
    frame = _coerce_returns(panel)
    wmap = _coerce_weight_map(weights, name="weights")
    excluded: list[Exclusion] = []
    notes: list[str] = []

    missing_from_panel = [t for t in wmap if t not in frame.columns]
    for t in missing_from_panel:
        excluded.append(
            Exclusion(t, "carried a weight but has no column in the returns panel", "nothing can be said about it")
        )
    unweighted = [str(c) for c in frame.columns if str(c) not in wmap]
    for t in unweighted:
        excluded.append(Exclusion(t, "is in the panel but carries no weight", "not part of the book as stated"))

    keep = [str(c) for c in frame.columns if str(c) in wmap]
    for t in list(keep):
        if not math.isfinite(wmap[t]) or wmap[t] <= 0.0:
            excluded.append(Exclusion(t, f"weight is {wmap[t]:g}", "a zero or negative weight is not a holding"))
            keep.remove(t)
    if not keep:
        return frame.iloc[:, :0], {}, excluded, notes, {}

    frame = frame[keep]

    scrubbed: dict[str, int] = dict.fromkeys(keep, 0)
    if scrub:
        cleaned, scrub_notes = scrub_suspect_returns(frame)
        if scrub_notes:
            blanked = cleaned.isna() & frame.notna()
            for t in keep:
                scrubbed[t] = int(blanked[t].sum())
            notes += [f"{t}: {why}" for t, why in scrub_notes]
            notes.append(
                "Blanked days are compounded as 0.0% — an assumption, not a measurement. A holding "
                "with blanked days is understated if the real move was up and overstated if it was down."
            )
            frame = cleaned.fillna(0.0)

    # A column that is entirely missing after alignment says nothing.
    for t in list(keep):
        col = frame[t]
        if int(col.notna().sum()) == 0:
            excluded.append(Exclusion(t, "has no finite return in this window", ""))
            keep.remove(t)
    frame = frame[keep].fillna(0.0)
    wmap = {t: float(wmap[t]) for t in keep}
    return frame, wmap, excluded, notes, {t: scrubbed.get(t, 0) for t in keep}


def _start_weights(
    weights_end: Mapping[str, float],
    growth: Mapping[str, float],
) -> tuple[dict[str, float], list[Exclusion]]:
    """Back out day-one weights from end weights and each holding's growth.

    ``value_start_i = value_end_i / (1 + R_i)`` exactly, so
    ``w_start_i ∝ w_end_i / (1 + R_i)``. This is the arithmetic that makes the
    reconciliation an identity rather than an approximation.

    WHERE IT MISLEADS
        It assumes the SAME SHARE COUNT was held for the whole window. A
        position that was added to, trimmed, or opened mid-window has a
        day-one weight this cannot recover, and the error flows straight into
        that holding's contribution. Nothing in a pasted holdings list reveals
        it, so the derived weights are exposed for the reader to check.
    """
    excluded: list[Exclusion] = []
    raw: dict[str, float] = {}
    for t, w in weights_end.items():
        g = float(growth[t])
        if not math.isfinite(g) or g <= _MIN_GROWTH_FACTOR:
            excluded.append(
                Exclusion(
                    t,
                    f"total return over this window is {g - 1.0:+.1%}",
                    "a position that went to zero has no recoverable day-one weight",
                )
            )
            continue
        raw[t] = float(w) / g
    total = sum(raw.values())
    if total <= 0.0:
        return {}, excluded
    return {t: v / total for t, v in raw.items()}, excluded


def _book_daily(returns: pd.DataFrame, weights_start: Mapping[str, float], basis: str) -> pd.Series:
    """The book's daily return series under the stated basis.

    buy_and_hold: ``V_t = sum_i w_start_i * cumprod(1+r_i)_t`` with ``V_0 = 1``
    because the start weights sum to 1. Compounding the resulting daily series
    reproduces ``sum_i w_start_i * R_i`` to float64 precision — which is what
    makes the reconciliation assertion meaningful rather than circular.

    fixed_weight: ``r_t = sum_i w_i * r_it``, i.e. traded back to the stated
    weights at every close.
    """
    cols = list(weights_start)
    w = pd.Series({t: float(weights_start[t]) for t in cols}, dtype=float)
    sub = returns[cols]
    if basis == "fixed_weight":
        return (sub * w).sum(axis=1)
    cum = (1.0 + sub).cumprod()
    value = cum.mul(w, axis=1).sum(axis=1)
    prev = value.shift(1)
    prev.iloc[0] = float(w.sum())
    return value / prev - 1.0


# --------------------------------------------------------------------------- #
# 1. Single window
# --------------------------------------------------------------------------- #
def holding_attribution(
    panel: Any,
    weights: Any,
    benchmark: Any,
    *,
    basis: str = "buy_and_hold",
    scrub: bool = True,
    weights_as_of: str = "end",
    window_label: str = "full window",
    min_days: int = MIN_WINDOW_DAYS,
) -> AttributionResult:
    """Per holding: its return, the benchmark's over the same days, and what
    the difference cost the book.

    WHAT IT COMPUTES
        For each holding: ``total_return`` (compounded daily returns over the
        shared calendar), ``benchmark_return`` (the same benchmark total for
        every holding — it is the one alternative, not a per-name yardstick),
        ``excess_return = total_return - benchmark_return`` and
        ``contribution = weight_start * excess_return``.

        Summed over holdings, those contributions equal the book's total return
        minus the benchmark's, exactly, under the buy-and-hold basis — because
        ``sum_i w_i (R_i - R_b) = sum_i w_i R_i - R_b`` when the weights sum to
        one. The result is sorted by contribution ascending, worst first.

    WHY CONTRIBUTION AND NOT EXCESS RETURN
        A 1%-weight position that halved has an excess return that dominates
        every list it appears in and moved the book by 0.5%. A 30%-weight
        position that trailed by 20% moved it by 6%. Ranking by excess return
        tells the reader the small disaster was the story. It was not.

    WHERE THIS MISLEADS
        Everything in the module docstring, and in particular: the start
        weights are DERIVED from the end weights on the assumption of an
        unchanged share count (see ``_start_weights``), and the comparison
        ignores costs and taxes on both sides.

    Parameters
    ----------
    panel
        Returns panel, or anything exposing one as ``.returns`` — a
        ``PricePanel`` from ``agent.portfolio.holdings`` is the intended input.
    weights
        Mapping or ``pandas.Series`` of ticker -> weight. Interpreted as of the
        window's END by default, matching ``PricePanel.weights``.
    benchmark
        Daily returns of the one alternative. See ``_benchmark_returns``.
    basis
        ``"buy_and_hold"`` (default) or ``"fixed_weight"``. See ``_BASES``.
    scrub
        Blank and report >=50% single-day moves as suspected unadjusted
        corporate actions. Default True.
    weights_as_of
        ``"end"`` (default) or ``"start"``. ``"start"`` skips the derivation
        and takes the weights as day-one weights, which is right when the
        caller genuinely knows them and wrong when they pasted a broker
        snapshot.
    min_days
        Below this many shared trading days the result is returned with
        ``usable=False`` and a reason instead of a number.

    Returns
    -------
    AttributionResult
        A ``list[HoldingResult]`` carrying the window, both series' statistics,
        the reconciliation residual and every exclusion.

    Raises
    ------
    ValueError
        Unknown ``basis`` or ``weights_as_of``; a price panel passed where
        returns belong; a benchmark that is not one instrument; or a
        buy-and-hold reconciliation residual above
        ``RECONCILIATION_TOLERANCE``, which is an internal arithmetic bug and
        must never be rendered to anybody.
    """
    if basis not in _BASES:
        raise ValueError(f"unknown basis {basis!r}; supported: {sorted(_BASES)}")
    if weights_as_of not in {"end", "start"}:
        raise ValueError(f"weights_as_of must be 'end' or 'start', got {weights_as_of!r}")

    bench_ret, bench_name = _benchmark_returns(benchmark)
    frame, wmap, excluded, notes, scrubbed = _prepare(panel, weights, scrub=scrub)

    if not wmap:
        return _unusable(
            window_label,
            frame,
            bench_name,
            basis,
            excluded,
            notes,
            "no holding in the panel carried a usable weight",
        )

    frame, bench_ret, align_excl, align_notes, lost_frac = _align(frame, bench_ret)
    excluded += align_excl
    notes += align_notes
    if lost_frac > MAX_BENCHMARK_DAY_LOSS:
        return _unusable(
            window_label,
            frame,
            bench_name,
            basis,
            excluded,
            notes,
            f"{lost_frac:.1%} of the book's trading days are missing from {bench_name}'s calendar, "
            f"above the {MAX_BENCHMARK_DAY_LOSS:.0%} ceiling. Compounding the two sides over "
            "calendars that differ by that much compares two different windows and calls the "
            "difference a decision. Use a benchmark listed on the same exchange as the book.",
        )

    n = int(len(frame.index))
    if n == 0:
        return _unusable(
            window_label, frame, bench_name, basis, excluded, notes, "the book and the benchmark share no trading day"
        )
    window = Window(window_label, _as_date(frame.index[0]), _as_date(frame.index[-1]), n)

    if n < min_days:
        return _unusable(
            window_label,
            frame,
            bench_name,
            basis,
            excluded,
            notes,
            f"only {n} shared trading days; {min_days} is the floor below which a total return "
            "annualises into a number the data does not support",
            window=window,
        )

    growth = {t: float(np.prod(1.0 + frame[t].to_numpy(dtype=float))) for t in wmap}
    if weights_as_of == "end":
        w_start, drop = _start_weights(wmap, growth)
        excluded += drop
        weights_end_shown = dict(wmap)
    else:
        total = sum(wmap.values())
        w_start = {t: v / total for t, v in wmap.items()} if total > 0 else {}
        weights_end_shown = {t: w_start[t] * growth[t] for t in w_start}
        tot_e = sum(weights_end_shown.values())
        weights_end_shown = {t: v / tot_e for t, v in weights_end_shown.items()} if tot_e > 0 else {}

    if not w_start:
        return _unusable(
            window_label,
            frame,
            bench_name,
            basis,
            excluded,
            notes,
            "no day-one weight could be derived for any holding",
            window=window,
        )
    if len(w_start) != len(wmap):
        notes.append(
            f"{len(wmap) - len(w_start)} holding(s) were excluded after the returns were computed, "
            "so the remaining day-one weights were renormalised to 1.0 and the book below is the "
            "book WITHOUT them."
        )

    book_daily = _book_daily(frame, w_start, basis)
    book = _series_stats(book_daily)
    bench = _series_stats(bench_ret)
    r_b = bench.total_return

    results = [
        HoldingResult(
            ticker=t,
            weight_start=w_start[t],
            weight_end=weights_end_shown.get(t, float("nan")),
            total_return=growth[t] - 1.0,
            benchmark_return=r_b,
            excess_return=(growth[t] - 1.0) - r_b,
            contribution=w_start[t] * ((growth[t] - 1.0) - r_b),
            n_days=n,
            scrubbed_days=scrubbed.get(t, 0),
            notes=(),
        )
        for t in w_start
    ]
    results.sort(key=lambda h: h.contribution)

    out = AttributionResult(
        results,
        window=window,
        basis=basis,
        basis_note=_BASES[basis],
        benchmark_ticker=bench_name,
        book=book,
        benchmark=bench,
        weights_start=w_start,
        weights_end=weights_end_shown,
        excluded=excluded,
        notes=notes,
        usable=True,
    )

    if basis == "buy_and_hold" and abs(out.residual) > RECONCILIATION_TOLERANCE:
        raise ValueError(
            "attribution failed to reconcile under the buy-and-hold basis: the weighted "
            f"contributions sum to {out.contribution_sum:.12f} but the book's gap versus "
            f"{bench_name} is {out.gap_total:.12f}, a residual of {out.residual:.3e} against a "
            f"tolerance of {RECONCILIATION_TOLERANCE:.0e}. Under this basis the two are the same "
            "arithmetic rearranged, so a residual this large is a bug in this module, not a "
            "property of the book. Nothing here may be shown to anybody until it is found."
        )
    return out


def _unusable(
    label: str,
    frame: pd.DataFrame,
    bench_name: str,
    basis: str,
    excluded: Sequence[Exclusion],
    notes: Sequence[str],
    reason: str,
    window: Window | None = None,
) -> AttributionResult:
    if window is None:
        if len(frame.index):
            window = Window(label, _as_date(frame.index[0]), _as_date(frame.index[-1]), int(len(frame.index)))
        else:
            today = dt.date.today()
            window = Window(label, today, today, 0)
    empty = SeriesStats(0, float("nan"), float("nan"), float("nan"), None, "no window")
    return AttributionResult(
        (),
        window=window,
        basis=basis,
        basis_note=_BASES.get(basis, ""),
        benchmark_ticker=bench_name,
        book=empty,
        benchmark=empty,
        excluded=tuple(excluded),
        notes=tuple(notes),
        usable=False,
        unusable_reason=reason,
    )


# --------------------------------------------------------------------------- #
# 2. The summary and its counterfactuals
# --------------------------------------------------------------------------- #
def decision_summary(attr: AttributionResult | Sequence[HoldingResult]) -> dict:
    """Turn a per-holding attribution into the counted facts and the two
    counterfactuals that keep it honest.

    WHAT IT COMPUTES
        * how many holdings trailed the benchmark and how many beat it;
        * the worst and best contributors by ``contribution``;
        * ``without_worst`` — the book's total return with the single worst
          contributor removed and the remaining day-one weights renormalised;
        * ``without_best`` — the same for the best contributor.

    WHY ``without_best`` IS NOT OPTIONAL
        On the book this product was built from, six of seven holdings trailed
        the index and one — the gold ETF the holder thinks of as the boring
        hedge — returned +144.8%. Reporting only "six of seven trailed" would
        be true and would be a smear: remove that one position and the book is
        far worse still. "One position carried this book" is the honest read
        and it is a DIFFERENT statement. ``one_position_carried`` is True when
        the best contributor alone exceeds the sum of every other positive
        contribution, and ``without_best_flips_sign`` is True when removing it
        turns a book that beat the benchmark into one that did not.

    WHERE THIS MISLEADS
        Removing a position and renormalising is arithmetic on the same price
        history, not a simulation of a different past. A holder who had not
        owned the gold ETF would have owned something else with that money, and
        this says nothing about what. It answers "how much of this book's
        result came from that one line", and only that.

    Returns
    -------
    dict
        All keys are always present. Anything that could not be computed is
        ``None`` with a ``*_reason`` key beside it — never 0.0.
    """
    holdings = list(attr)
    res = attr if isinstance(attr, AttributionResult) else None
    bench_name = res.benchmark_ticker if res is not None else "the benchmark"

    base: dict[str, Any] = {
        "n_holdings": len(holdings),
        "n_trailed": None,
        "n_beat": None,
        "benchmark_ticker": bench_name,
        "window": str(res.window) if res is not None else None,
        "book_total_return": None,
        "benchmark_total_return": None,
        "gap_total": None,
        "gap_annualised": None,
        "worst_contributor": None,
        "worst_contribution": None,
        "worst_contributor_role": None,
        "best_contributor": None,
        "best_contribution": None,
        "best_contributor_role": None,
        "without_worst_book_return": None,
        "without_worst_reason": "",
        "without_worst_gap": None,
        "without_best_book_return": None,
        "without_best_reason": "",
        "without_best_gap": None,
        "one_position_carried": None,
        "without_best_flips_sign": None,
        "contribution_sum": None,
        "residual": None,
        "reconciles": None,
        "residual_explanation": None,
        "statement": "",
        "unavailable_reason": "",
    }

    if res is not None and not res.usable:
        base["unavailable_reason"] = res.unusable_reason
        base["statement"] = f"No attribution was computed for this window: {res.unusable_reason}."
        return base
    if not holdings:
        base["unavailable_reason"] = "no holdings were measured"
        base["statement"] = "No holding in this book could be measured against a benchmark."
        return base

    r_b = holdings[0].benchmark_return
    w = {h.ticker: h.weight_start for h in holdings}
    r = {h.ticker: h.total_return for h in holdings}
    total_w = sum(w.values())
    book_bh = sum(w[t] * r[t] for t in w) / total_w if total_w > 0 else float("nan")

    def _without(drop: str) -> tuple[float | None, str]:
        rest = total_w - w[drop]
        if rest <= 1e-12:
            return None, f"{drop} was the whole book; there is nothing left to compute"
        return sum(w[t] * r[t] for t in w if t != drop) / rest, ""

    worst = min(holdings, key=lambda h: h.contribution)
    best = max(holdings, key=lambda h: h.contribution)
    worst_role = _contributor_role(worst.contribution, extreme="lowest")
    best_role = _contributor_role(best.contribution, extreme="highest")
    wo_worst, wo_worst_why = _without(worst.ticker)
    wo_best, wo_best_why = _without(best.ticker)

    positives = [h.contribution for h in holdings if h.contribution > 0]
    other_positive = sum(positives) - max(positives) if positives else 0.0
    carried = bool(positives) and max(positives) > other_positive

    book_total = res.book.total_return if res is not None else book_bh
    gap = book_total - r_b

    base.update(
        {
            "n_trailed": sum(1 for h in holdings if not h.beat_benchmark),
            "n_beat": sum(1 for h in holdings if h.beat_benchmark),
            "book_total_return": book_total,
            "benchmark_total_return": r_b,
            "gap_total": gap,
            "gap_annualised": res.gap_annualised if res is not None else None,
            "worst_contributor": worst.ticker,
            "worst_contribution": worst.contribution,
            "worst_contributor_role": worst_role,
            "best_contributor": best.ticker,
            "best_contribution": best.contribution,
            "best_contributor_role": best_role,
            "without_worst_book_return": wo_worst,
            "without_worst_reason": wo_worst_why,
            "without_worst_gap": None if wo_worst is None else wo_worst - r_b,
            "without_best_book_return": wo_best,
            "without_best_reason": wo_best_why,
            "without_best_gap": None if wo_best is None else wo_best - r_b,
            "one_position_carried": carried,
            "without_best_flips_sign": (None if wo_best is None else _sign(gap) != _sign(wo_best - r_b)),
            "contribution_sum": res.contribution_sum if res is not None else sum(h.contribution for h in holdings),
            "residual": res.residual if res is not None else None,
            "reconciles": (None if res is None else abs(res.residual) <= RECONCILIATION_TOLERANCE),
            "residual_explanation": res.residual_explanation if res is not None else None,
        }
    )

    n = len(holdings)
    n_trailed = base["n_trailed"]
    parts = [f"{n_trailed} of {n} holdings returned less than {bench_name} over this window."]
    if wo_best is not None:
        parts.append(
            f"Without {best.ticker}, {best_role}, the book returned {wo_best:+.1%} against {bench_name}'s {r_b:+.1%}."
        )
    if wo_worst is not None:
        parts.append(f"Without {worst.ticker}, {worst_role}, it returned {wo_worst:+.1%}.")
    if carried:
        parts.append(
            f"{best.ticker} contributed more than every other positive contribution combined: "
            "one position accounts for the book's upside over this window."
        )
    base["statement"] = " ".join(parts)
    return base


# --------------------------------------------------------------------------- #
# 3. Survivorship
# --------------------------------------------------------------------------- #
def survivorship_note(panel: Any) -> str:
    """State the bias the data at hand cannot measure.

    WHAT IT COMPUTES
        Nothing numeric about the bias — that is the point, and pretending
        otherwise would be the dishonesty. It reports what IS observable: how
        many names are in the panel, over what window, and how many symbols the
        ingestion layer already had to exclude for having ended (delisted,
        suspended, or too short a history), since those are the only
        disappeared names this system ever sees.

    WHY IT IS A PARAGRAPH AND NOT A FOOTNOTE
        A pasted portfolio is a list of survivors. Names sold at a loss and
        names sold at a profit are equally invisible, and the literature does
        not agree on which dominates for retail holders — disposition effect
        says losers are HELD and winners sold, which pushes this measured gap
        DOWNWARD; tax-loss harvesting and capitulation push the other way.
        The direction is unknown, so the honest statement names both and claims
        neither.

    WHERE THIS MISLEADS
        It can only count exclusions the ingestion layer recorded. A name sold
        two years ago was never typed in and leaves no trace anywhere in this
        system.
    """
    frame = None
    try:
        frame = _coerce_returns(panel)
    except (TypeError, ValueError):
        frame = None

    lines: list[str] = []
    if frame is not None and not frame.empty:
        start, end = _as_date(frame.index[0]), _as_date(frame.index[-1])
        lines.append(
            f"These {frame.shape[1]} names are the ones still held, measured over "
            f"{start.isoformat()}..{end.isoformat()}."
        )
    else:
        lines.append("These are the names still held.")

    lines.append(
        "Anything sold before today is absent from this analysis and from this system: it was "
        "never typed in, so nothing here measured it."
    )
    lines.append(
        "That biases every figure above in a direction nobody can determine from this data. Selling "
        "losers and keeping winners pushes these figures upward; holding losers and taking profits "
        "early pushes them downward. Both are ordinary behaviours, they push opposite ways, and the "
        "net direction is unknown."
    )

    excluded = getattr(panel, "excluded", ())
    ended = [e for e in excluded if _looks_like_ending(getattr(e, "reason", ""))]
    if ended:
        names = ", ".join(str(getattr(e, "symbol", getattr(e, "name", "?"))) for e in ended)
        lines.append(
            f"{len(ended)} symbol(s) in what was pasted did end within or before this window and were "
            f"excluded for it ({names}). Those are the only disappearances this system can see at all."
        )
    return " ".join(lines)


_ENDING_MARKERS = ("delisted", "suspended", "last price is", "no price history", "trading days of history")


def _looks_like_ending(reason: str) -> bool:
    low = str(reason).lower()
    return any(m in low for m in _ENDING_MARKERS)


# --------------------------------------------------------------------------- #
# 4. Multi-window — the part that is the product
# --------------------------------------------------------------------------- #
_ROLLING_NOTE = (
    "Fixed-length windows of {length} trading days, started every {step} trading days from the "
    "first day of the panel. Length and step are fixed in advance and are not tuned: a search over "
    "window lengths always finds the length that maximises whatever it was pointed at."
)


def _rolling_windows(index: pd.DatetimeIndex, length: int, step: int) -> list[tuple[str, pd.DatetimeIndex]]:
    out: list[tuple[str, pd.DatetimeIndex]] = []
    n = len(index)
    if length <= 0 or step <= 0:
        raise ValueError(f"window length and step must both be positive, got {length} and {step}")
    for start in range(0, max(1, n - length + 1), step):
        stop = start + length
        if stop > n:
            break
        days = index[start:stop]
        out.append((f"{days[0].date().isoformat()}..{days[-1].date().isoformat()}", days))
    return out


def multi_window_attribution(
    panel: Any,
    weights: Any,
    benchmark: Any,
    *,
    method: str = "calendar_year",
    basis: str = "buy_and_hold",
    scrub: bool = True,
    window_days: int = TRADING_DAYS_PER_YEAR,
    step_days: int = 21,
    min_days: int = MIN_WINDOW_DAYS,
) -> MultiWindowResult:
    """Run the attribution over every window a pre-committed rule produces.

    WHAT IT COMPUTES
        The full-window attribution, plus one attribution per sub-window, plus
        the count of sub-windows whose gap has the OPPOSITE sign to the full
        window's. ``headline_qualifier`` renders the contradicting windows
        first; ``direction_consistent`` is the only flag under which a
        single-window headline is safe to publish unqualified.

    THE SAME BOOK IN EVERY WINDOW
        Each sub-window is measured with the share counts implied by the stated
        end weights, re-valued at that sub-window's first close. That is the
        only coherent reading of "this book, over that period": fixing the
        WEIGHTS instead would silently rebalance the book at the start of every
        window and flatter it, because it would be buying each position back
        down after it ran.

    WINDOW RULES
        ``"calendar_year"`` / ``"calendar_half"`` delegate to
        ``agent.portfolio.beliefs.regime_split``, whose boundaries are 31
        December and 30 June — chosen before looking at the data and for no
        economic reason, which is precisely why they cannot be tuned.
        ``"rolling"`` uses fixed-length windows at a fixed step. No data-driven
        rule is offered.

    WHERE THIS MISLEADS
        * ROLLING WINDOWS ARE NOT INDEPENDENT. Consecutive 252-day windows 21
          days apart share 92% of their days. "The book trailed in 38 of 41
          windows" is ONE observation restated 41 times, not 41 pieces of
          evidence. ``independent_windows`` reports how many non-overlapping
          windows the panel actually contains, and that is the number that
          counts.
        * CALENDAR YEARS ARE NOT REGIMES. Markets do not change state on 31
          December, and a real turn in July is split across two periods,
          diluting both.
        * A PARTIAL YEAR AT EITHER END is usually below ``min_days`` and is
          returned with a skip reason rather than dropped, because dropping it
          would hide that the panel does not cover it.
        * EVERY SUB-WINDOW INHERITS THE SURVIVORSHIP BIAS of the book it was
          computed on. Measuring the same survivors over more windows does not
          reduce it — it repeats it.

    Raises
    ------
    ValueError
        Unknown ``method``, or a full window that is itself unusable (there is
        nothing to compare sub-windows against).
    """
    if method not in {"calendar_year", "calendar_half", "rolling"}:
        raise ValueError(
            f"unknown method {method!r}; supported: ['calendar_half', 'calendar_year', 'rolling']. "
            "No data-driven window rule is offered: a search over windows always finds one."
        )

    full = holding_attribution(
        panel,
        weights,
        benchmark,
        basis=basis,
        scrub=scrub,
        window_label="full window",
        min_days=min_days,
    )
    if not full.usable:
        raise ValueError(
            f"the full window is not usable ({full.unusable_reason}), so there is nothing for "
            "sub-windows to be checked against."
        )

    frame, wmap, _excl, _notes, _scr = _prepare(panel, weights, scrub=scrub)
    bench_ret, bench_name = _benchmark_returns(benchmark)
    frame, bench_ret, _e2, _n2, _lost = _align(frame, bench_ret)
    index = frame.index

    # Share counts implied by the stated END weights, expressed as a value per
    # unit of day-one price. Re-valuing these at each sub-window's first close
    # is what keeps "the same book" the same book across windows.
    cum = (1.0 + frame[list(wmap)]).cumprod()
    growth_full = {t: float(cum[t].iloc[-1]) for t in wmap}
    shares = {t: wmap[t] / growth_full[t] for t in wmap if growth_full[t] > _MIN_GROWTH_FACTOR}
    prev_cum = cum.shift(1)
    prev_cum.iloc[0] = 1.0

    if method == "rolling":
        spans = _rolling_windows(index, window_days, step_days)
        note = _ROLLING_NOTE.format(length=window_days, step=step_days)
        independent = len(index) // window_days if window_days else 0
        overlap = (
            f"{len(spans)} windows of {window_days} trading days at a {step_days}-day step. "
            f"Consecutive windows share {max(0, window_days - step_days) / window_days:.0%} of their "
            f"days, so these are not {len(spans)} independent tests — the panel holds at most "
            f"{independent} non-overlapping window(s) of this length."
        )
    else:
        regimes = regime_split(frame, method=method)
        note = regimes[0].method_note if regimes else ""
        spans = []
        for reg in regimes:
            mask = (index.date >= reg.start) & (index.date <= reg.end)
            spans.append((reg.label, index[mask]))
        independent = len(spans)
        overlap = f"{len(spans)} non-overlapping {method.replace('_', ' ')} windows; each day is counted once."

    windows: list[WindowAttribution] = []
    for label, days in spans:
        if len(days) == 0:
            continue
        win = Window(label, _as_date(days[0]), _as_date(days[-1]), int(len(days)))
        if len(days) < min_days:
            windows.append(
                WindowAttribution(
                    window=win,
                    result=None,
                    skipped_reason=(
                        f"{len(days)} trading days is below the {min_days}-day floor; a total return "
                        "over this little data annualises into a figure nothing supports"
                    ),
                )
            )
            continue
        sub = frame.loc[days]
        # Day-one value of each position in THIS window = shares * price at its start.
        start_value = {t: shares[t] * float(prev_cum[t].loc[days[0]]) for t in shares}
        sv_total = sum(start_value.values())
        if sv_total <= 0:
            windows.append(WindowAttribution(win, None, "the book had no positive value at this window's start"))
            continue
        w_start_win = {t: v / sv_total for t, v in start_value.items()}
        try:
            res = holding_attribution(
                sub,
                w_start_win,
                (bench_name, bench_ret.loc[days]),
                basis=basis,
                scrub=False,  # already scrubbed once on the full panel
                weights_as_of="start",
                window_label=label,
                min_days=min_days,
            )
        except ValueError as exc:  # a reconciliation failure must not be swallowed
            raise ValueError(f"window {label}: {exc}") from exc
        windows.append(WindowAttribution(win, res, None if res.usable else res.unusable_reason))

    notes: list[str] = [
        "Every sub-window measures the same share counts, re-valued at that window's first close. "
        "The book is not returned to its stated weights at a window boundary.",
        "Sub-windows were not scrubbed again; the corporate-action scrub was applied once to the "
        "whole panel and every sub-window inherits it.",
    ]
    if method == "rolling" and len(spans) > independent:
        notes.append(
            "Counting overlapping windows as separate evidence is the mistake this note exists to "
            f"prevent: {len(spans)} windows here contain at most {independent} independent one(s)."
        )

    return MultiWindowResult(
        method=method,
        method_note=note,
        full_window=full,
        windows=tuple(windows),
        benchmark_ticker=bench_name,
        independent_windows=int(independent),
        overlap_note=overlap,
        notes=tuple(notes),
        excluded=full.excluded,
    )


# --------------------------------------------------------------------------- #
# 5. Rendering (terminal / review; the web layer owns real presentation)
# --------------------------------------------------------------------------- #
def format_attribution_report(attr: AttributionResult, summary: Mapping[str, Any] | None = None) -> str:
    """Plain-text rendering of one window. Facts only, past tense, no advice.

    This exists so the numbers can be eyeballed in a terminal and pasted into a
    review. It is not the product surface, and it deliberately prints the
    exclusions and the residual next to the headline rather than under it.
    """
    L: list[str] = []
    if not attr.usable:
        return f"NOT COMPUTED: {attr.unusable_reason}\nWINDOW  {attr.window}"
    L.append(f"WINDOW    {attr.window}")
    L.append(f"BASIS     {attr.basis} — {attr.basis_note}")
    L.append("")
    L.append(attr.book.line("your book"))
    L.append(attr.benchmark.line(attr.benchmark_ticker + " alone"))
    L.append(
        f"{'gap':<16} {attr.gap_total:+8.1%} total  {attr.gap_annualised:+7.1%}/yr  "
        f"(book minus {attr.benchmark_ticker}, same {attr.window.n_days} days)"
    )
    if attr.book.return_per_vol is not None and attr.benchmark.return_per_vol is not None:
        L.append(
            "                 return-per-vol is annualised return / annualised volatility. "
            "It is not a Sharpe ratio: no risk-free rate is subtracted."
        )
    L.append("")
    L.append("PER HOLDING, RANKED BY WHAT IT MOVED THE GAP (worst first)")
    L.append(f"  {'ticker':<14}{'wt day 1':>9}{'wt today':>9}{'return':>10}{'vs bench':>10}{'contribution':>14}")
    for h in attr:
        L.append(
            f"  {h.ticker:<14}{h.weight_start:>9.1%}{h.weight_end:>9.1%}"
            f"{h.total_return:>10.1%}{h.excess_return:>10.1%}{h.contribution:>14.2%}"
            + ("   [scrubbed days]" if h.scrubbed_days else "")
        )
    L.append(
        f"  {'SUM':<14}{sum(attr.weights_start.values()):>9.1%}{'':>9}{'':>10}{'':>10}{attr.contribution_sum:>14.2%}"
    )
    L.append(f"  RECONCILIATION  {attr.residual_explanation}")

    if summary is not None:
        L.append("")
        L.append("WHAT THE DECISIONS ADDED UP TO")
        L.append(f"  {summary.get('statement', '')}")
        if summary.get("without_best_flips_sign"):
            L.append(
                f"  Removing {summary['best_contributor']} reverses the sign of the gap: the book beat "
                f"{attr.benchmark_ticker} only because of that one position."
            )
    if attr.notes:
        L.append("")
        L.append("NOTES")
        L += [f"  - {n}" for n in attr.notes]
    if attr.excluded:
        L.append("")
        L.append("EXCLUDED (nothing is dropped silently)")
        L += [f"  - {e}" for e in attr.excluded]
    return "\n".join(L)


def format_multi_window_report(mw: MultiWindowResult, *, max_windows: int = 40) -> str:
    """Plain-text rendering of the multi-window run, contradictions first."""
    L: list[str] = []
    L.append(f"BENCHMARK    {mw.benchmark_ticker}")
    L.append(f"WINDOW RULE  {mw.method} — {mw.method_note}")
    L.append(f"OVERLAP      {mw.overlap_note}")
    L.append("")
    L.append(f"FULL WINDOW  {mw.full_window.window}")
    L.append(f"  {mw.full_window.book.line('your book')}")
    L.append(f"  {mw.full_window.benchmark.line(mw.benchmark_ticker + ' alone')}")
    L.append(f"  gap {mw.full_window.gap_total:+.1%} total, {mw.full_window.gap_annualised:+.1%}/yr")
    L.append("")
    L.append("DOES IT HOLD IN OTHER WINDOWS?")
    L.append(f"  {mw.headline_qualifier}")
    if mw.windows_contradicting:
        L.append("")
        L.append("  WINDOWS THAT CONTRADICT THE FULL-WINDOW RESULT")
        for w in mw.windows_contradicting:
            L.append(f"    {_window_cell(w.window):<42}  gap {w.gap_total:+8.1%}")
    agree = mw.windows_agreeing
    if agree:
        L.append("")
        L.append("  WINDOWS THAT AGREE")
        for w in agree[:max_windows]:
            L.append(f"    {_window_cell(w.window):<42}  gap {w.gap_total:+8.1%}")
        if len(agree) > max_windows:
            L.append(f"    ... and {len(agree) - max_windows} more")
    skipped = [w for w in mw.windows if not w.usable]
    if skipped:
        L.append("")
        L.append("  WINDOWS NOT MEASURED (reported, not dropped)")
        for w in skipped:
            L.append(f"    {_window_cell(w.window):<42}  {w.skipped_reason}")

    gaps = mw.gaps
    if gaps:
        L.append("")
        L.append(
            f"  GAP ACROSS WINDOWS  min {min(gaps):+.1%}  median {float(np.median(gaps)):+.1%}  max {max(gaps):+.1%}"
        )
    if mw.notes:
        L.append("")
        L.append("NOTES")
        L += [f"  - {n}" for n in mw.notes]
    return "\n".join(L)


# Keep a module-level marker of the fields a renderer must never omit. Used by
# the tests, which assert that a rendered report carries the window, the
# benchmark name and the survivorship statement — the three things whose
# absence turns a fact into a claim.
_REQUIRED_IN_ANY_RENDER: tuple[str, ...] = ("WINDOW", "gap")
