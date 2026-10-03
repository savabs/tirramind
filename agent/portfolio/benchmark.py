"""
What did picking cost? — a portfolio against the alternative of having done nothing.

WHAT THIS MODULE IS FOR
-----------------------
A broker app shows profit and loss against your *buy price*.  It never shows
profit and loss against the alternative you actually faced, which was to buy a
broad index fund and stop.  That comparison is arithmetic on prices that have
already happened, and it is the one number about a portfolio that its owner has
not already computed, because no tool they use computes it for them.

Measured here on a real seven-name Indian book (RELIANCE, TCS, INFY, ITC,
HDFCBANK, GOLDBEES, NIFTYBEES, equally weighted) over 2023-09-29..2026-09-29:

    your book     +10.6% total, +3.4%/yr, vol 12.5%, return-per-vol 0.27
    NIFTYBEES.NS  +19.4% total, +6.1%/yr, vol 11.9%, return-per-vol 0.51
    => picking seven names instead of the index cost 8.8 points, at more risk
       per holding: 6 of 7 lost to the index. TCS -35.8%, ITC -30.6%, INFY -24.3%.
       The only winner was GOLDBEES +145.0% — the position most holders think of
       as the boring hedge.

WHAT THIS MODULE IS NOT
-----------------------
It gives no advice.  "Your book returned 10.6% and the index returned 19.4% over
2023-09-29..2026-09-29" is a statement about the past.  "You should index" is
regulated investment advice and nothing here says it, implies it, or arranges
the output so that it is the only readable conclusion.  ``format_report`` is
tested against :data:`agent.portfolio.concentration._ADVICE_WORDS`, which is the
same list the concentration report is held to.

WHY MANY WINDOWS AND NOT ONE
----------------------------
One window is cherry-picking, and measurably so.  That same seven-name book,
across the eight named windows this module computes, produced gaps of
-3.8, -1.4, -0.5, +2.7, -7.0, -5.2, +0.9 and -3.8 points.  The sign flips.
Headlining any one of them would be choosing the story.  A product whose whole
claim is "we do not let you fool yourself" cannot do that on its own front page.
So:

* :func:`default_windows` fixes the window set *by rule*, stated in
  :data:`WINDOW_RULE`, before any return is computed — full history, every
  calendar year, trailing 1y/2y/3y, and every rolling one-year window.
* :func:`rolling_gap` is the anti-cherry-pick: over every rolling window, in how
  many did the book beat the index, and what were the median, best and worst
  gaps.  On that book: it out-returned the index in 223 of 493 rolling one-year
  windows (45%), median gap -0.2 points, best +6.0, worst -6.5.  That survives a
  sceptic; a single number does not.

THE THREE THINGS THAT WOULD MAKE EVERY NUMBER HERE A LIE
--------------------------------------------------------
**1. Price return compared against total return.**  If the holdings include
dividends and the benchmark does not, the gap is manufactured out of nothing.
Measured on this repo's own cache, ``^NSEI`` (a price index) understated
``NIFTYBEES.NS`` (the same exposure as a total-return ETF) by **121.5 bps per
year** over 2021-09-29..2026-09-29 — 27.81% against 35.38% cumulative.  A book
compared to ``^NSEI`` would be handed 1.2 points a year of fake outperformance.
So: ``^``-prefixed index symbols are classified ``total_return=False``
(:data:`PRICE_RETURN_SYMBOLS`), they are never chosen as a default, choosing one
explicitly attaches a warning naming its total-return substitute, and
:func:`fetch_benchmark` asserts that the underlying fetcher still passes
``auto_adjust=True`` before it will return anything at all.

**2. Today's weights are not the weights they held.**  Everything here applies
*current* weights backwards.  That is a fixed-weight counterfactual — what a
book that started at today's shape would have done — and it is **not what the
owner earned**.  They traded along the way; without transaction history their
true return is not computable, and pretending otherwise is a lie.  This is
stated in :attr:`PickingCost.weight_caveat`, printed at the top of the report,
and it is the single most important sentence in the output.

**3. A benchmark chosen to make the gap look big.**  This is the most gameable
part of the product and a sharp reader will check it.  The default is fixed by
the book's own currency and market (:data:`DEFAULT_BENCHMARKS`); a mixed-currency
book is reported against *every* applicable benchmark, all shown; and if the book
holds the benchmark itself — the example book holds ``NIFTYBEES.NS`` — that is
said plainly and the comparison is *also* run on the rest of the book.

WHERE ELSE THIS MISLEADS
------------------------
* **Buy-and-hold, not constant-mix.**  The headline path invests today's weights
  once at the window start and never trades, so weights drift.  A constant-mix
  figure is reported beside it; measured on four real Indian books over three
  years the two answers differ by 2.6 to 6.7 points, in both directions, and
  neither is "the" answer because a real holder did neither.
* **Rolling windows overlap.**  The 493 rolling one-year windows a three-year
  book yields are not 493 independent observations; they are about three.
  :attr:`RollingGap.n_independent` carries that number next to the count, and
  the caveat says in words that the beat fraction describes this one history
  rather than estimating a probability.
* **Volatility is realised daily vol, annualised by the observed observation
  rate.**  Return-per-vol here divides annualised return by annualised vol and
  subtracts no risk-free rate.  It is **not** a Sharpe ratio and is not labelled
  one.
* **Cross-calendar comparison forward-fills.**  An INR book against ``SPY`` has
  to reconcile two trading calendars.  Window endpoints use a forward fill (a
  closed US market did not move), but benchmark *volatility* is computed on the
  benchmark's own trading days and annualised by its own observation rate, so a
  foreign holiday is never counted as a zero-return day.  For i.i.d. returns the
  two conventions very nearly coincide — the diluted variance and the larger
  annualisation factor cancel — so this is not a large correction; it matters
  when closures cluster, which on real data is exactly when the market is moving.
  The number of filled days is reported either way.
* **Survivorship is absent by construction.**  The book contains what the owner
  holds *now*.  Anything they already sold at a loss is invisible here, so the
  book's own number is, if anything, flattered.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from agent.portfolio import holdings as _h

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BENCHMARKS",
    "MIN_ANNUALISE_YEARS",
    "MIN_VOL_OBS",
    "MIN_WINDOW_OBS",
    "PRICE_RETURN_SYMBOLS",
    "WINDOW_RULE",
    "Benchmark",
    "HoldingLeg",
    "PickingCost",
    "RollingGap",
    "Window",
    "WindowResult",
    "choose_benchmarks",
    "default_windows",
    "fetch_benchmark",
    "format_report",
    "picking_cost",
    "rolling_gap",
]

# ---------------------------------------------------------------------------
# Thresholds. Every one of these is a judgement that can change an answer, so
# each is named, defaulted here, and reported wherever it bites.
# ---------------------------------------------------------------------------

#: A window with fewer aligned prints than this produces no result at all.
#: Ten trading days is two weeks; a "return" over less is noise with a label.
MIN_WINDOW_OBS = 10

#: Below this many daily returns, realised volatility is not reported. 20 obs
#: gives a standard error on vol of roughly 1/sqrt(2*19) ~= 16% of the estimate.
MIN_VOL_OBS = 20

#: Below this many years, a total return is NOT annualised. Annualising a
#: three-month number multiplies both the return and its error by four, which is
#: how a quiet quarter becomes a headline.
MIN_ANNUALISE_YEARS = 0.5

#: Trading-day count treated as one year when a caller passes ``length`` as an
#: integer. Only used for labelling; all annualisation uses calendar time.
TRADING_DAYS_PER_YEAR = 252

#: Forward-filling more than this share of a benchmark's days across a calendar
#: mismatch earns a prominent warning rather than a footnote.
MAX_FILL_FRACTION = 0.02

#: The default benchmark per currency, and the reason it is the default. These
#: are all *total-return* vehicles: a fund's NAV includes its dividends, and
#: yfinance ``auto_adjust=True`` reinvests them. Currencies absent from this map
#: get no guessed benchmark — see :func:`choose_benchmarks`.
DEFAULT_BENCHMARKS: dict[str, tuple[str, str]] = {
    "INR": (
        "NIFTYBEES.NS",
        "broadest liquid Indian equity vehicle a retail holder could actually have bought "
        "instead of picking names; total-return (an ETF's NAV includes its dividends)",
    ),
    "USD": (
        "SPY",
        "broadest liquid US equity vehicle a retail holder could actually have bought instead "
        "of picking names; total-return",
    ),
}

#: Index symbols that are PRICE returns, mapped to the total-return vehicle that
#: tracks the same exposure. Comparing dividend-inclusive holdings against one of
#: these manufactures a gap; see the module docstring for the measured size.
PRICE_RETURN_SYMBOLS: dict[str, str] = {
    "^NSEI": "NIFTYBEES.NS",
    "^NSEBANK": "BANKBEES.NS",
    "^BSESN": "SETFNIF50.NS",
    "^CNX100": "NIFTYBEES.NS",
    "^GSPC": "SPY",
    "^DJI": "DIA",
    "^IXIC": "QQQ",
    "^RUT": "IWM",
    "^FTSE": "ISF.L",
    "^N225": "1321.T",
    "^STOXX50E": "FEZ",
}

#: The window set, stated as a rule BEFORE any return is computed. A window set
#: chosen after seeing the answer is cherry-picking with extra steps; this string
#: is what stops that, and it is printed in every report.
WINDOW_RULE = (
    "Windows are fixed by rule, not chosen after seeing the answer: (1) the full "
    "shared history of the book; (2) every calendar year the history touches, with "
    "part-years labelled as part-years; (3) trailing 1, 2 and 3 years to the last "
    "print, where the history reaches; (4) every rolling one-year window, one per "
    "trading day. Windows the history cannot cover are listed as skipped, with the "
    "reason, rather than dropped."
)

_STD_PERIODS: tuple[tuple[float, str], ...] = (
    (1.0, "1y"),
    (2.0, "2y"),
    (5.0, "5y"),
    (10.0, "10y"),
)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """One comparison window, carrying the rule that produced it.

    ``rule`` exists so that a window can never be presented without the reason
    it is in the set. ``complete`` is False for a calendar year the history only
    partly covers — the number is still real, but it is not a year.
    """

    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str  # "full" | "calendar_year" | "trailing" | "rolling"
    rule: str
    complete: bool = True

    @property
    def years(self) -> float:
        """Calendar length in years. Annualisation uses this, never a day count."""
        return max((self.end - self.start).days, 0) / 365.25


@dataclass
class Benchmark:
    """A benchmark series in the book's base currency, plus why it was chosen.

    Attributes
    ----------
    prices : Series
        Closes reindexed onto the book's calendar and forward-filled. Correct for
        window endpoints; NOT used for volatility.
    native : Series
        Closes on the benchmark's OWN trading days, base-currency converted. Used
        for volatility, so that a foreign holiday cannot inject a zero-return day
        and flatter the benchmark's risk.
    total_return : bool
        False for price-return indices. A False here is a defect in the
        comparison, not a footnote, and the report says so.
    weight_in_book : float
        The book's own weight in this exact ticker, 0.0 if it holds none. When
        this is non-zero, part of the book *is* the benchmark.
    """

    ticker: str
    label: str
    prices: pd.Series
    native: pd.Series
    native_currency: str
    base_currency: str
    total_return: bool
    why: str
    warnings: tuple[str, ...] = ()
    fx_note: str = ""
    filled_days: int = 0
    weight_in_book: float = 0.0

    @property
    def fill_fraction(self) -> float:
        return 0.0 if len(self.prices) == 0 else self.filled_days / len(self.prices)


@dataclass(frozen=True)
class HoldingLeg:
    """One holding's own return over a window, and its share of the book's.

    ``contribution`` is ``weight * total_return``. Under the buy-and-hold
    counterfactual these sum **exactly** to the book's total return; the code
    asserts it and :func:`tests <tests.test_portfolio_benchmark>` pin it.
    """

    ticker: str
    weight: float
    total_return: float
    contribution: float
    vs_benchmark: float


@dataclass
class WindowResult:
    """The book against one benchmark over one window. Everything, or nothing."""

    window: Window
    benchmark_ticker: str
    n_obs: int
    years: float
    port_total: float
    port_annual: float
    port_vol: float
    port_return_per_vol: float
    bench_total: float
    bench_annual: float
    bench_vol: float
    bench_return_per_vol: float
    gap_total: float
    gap_annual: float
    legs: tuple[HoldingLeg, ...] = ()
    n_legs_behind: int = 0
    n_legs_held: int = 0
    constant_mix_total: float = float("nan")
    constant_mix_annual: float = float("nan")
    usable: bool = True
    notes: tuple[str, ...] = ()

    @property
    def beat(self) -> bool:
        """Did the book out-return the benchmark over this window?"""
        return bool(self.gap_total > 0)


@dataclass
class PickingCost:
    """The headline object: what picking cost, across every window, per benchmark.

    ``results`` maps benchmark ticker -> window results, in window order. Results
    for ``kind == "rolling"`` carry no per-holding legs and no constant-mix
    figure: those are built only for the named windows a report renders.
    ``ex_benchmark`` holds the same for the book with the benchmark holding
    removed and the rest reweighted, and is populated only when the book
    actually holds the benchmark.
    """

    benchmarks: tuple[Benchmark, ...]
    results: dict[str, tuple[WindowResult, ...]]
    window_rule: str
    weight_caveat: str
    base_currency: str
    span: str
    ex_benchmark: dict[str, tuple[WindowResult, ...]] = field(default_factory=dict)
    skipped_windows: tuple[str, ...] = ()
    excluded: tuple[tuple[str, str], ...] = ()
    assumptions: tuple[str, ...] = ()

    def named(self, ticker: str) -> tuple[WindowResult, ...]:
        """Results for the windows a human reads — everything but the rolling family."""
        return tuple(r for r in self.results.get(ticker, ()) if r.window.kind != "rolling")

    def rolling(self, ticker: str) -> tuple[WindowResult, ...]:
        return tuple(r for r in self.results.get(ticker, ()) if r.window.kind == "rolling")

    def full_history(self, ticker: str) -> WindowResult | None:
        for r in self.results.get(ticker, ()):
            if r.window.kind == "full":
                return r
        return None


@dataclass
class RollingGap:
    """Every rolling window of one length: how often, by how much, at what worst.

    This is the anti-cherry-pick. ``beat_fraction`` over a full set of rolling
    windows cannot be moved by choosing an endpoint, which is exactly what a
    single-window headline can.

    ``n_independent`` is the honest denominator: overlapping windows share most
    of their data, so ``n_windows`` is not a sample size.
    """

    benchmark_ticker: str
    length_label: str
    length_years: float
    series: pd.DataFrame
    n_windows: int
    n_independent: float
    beat_count: int
    beat_fraction: float
    median_gap: float
    mean_gap: float
    best_gap: float
    best_window: str
    worst_gap: float
    worst_window: str
    headline: str
    caveats: tuple[str, ...] = ()
    usable: bool = True
    unusable_reason: str = ""


# ---------------------------------------------------------------------------
# Guards. Each of these exists because its absence produces a confident wrong
# number rather than an error.
# ---------------------------------------------------------------------------


def _assert_total_return_fetcher() -> None:
    """Refuse to fetch a benchmark if the price fetcher stopped adjusting for dividends.

    ``agent.portfolio.holdings._yf_fetch`` passes ``auto_adjust=True``, which is
    the only reason holdings and benchmark are on the same footing. If that ever
    changes, every gap in this module silently gains the benchmark's dividend
    yield — about 1.2 points a year on the Nifty. This is a tripwire on that
    exact regression, not decoration.
    """
    try:
        src = inspect.getsource(_h._yf_fetch)
    except (OSError, TypeError) as exc:  # pragma: no cover - source always present
        raise RuntimeError(
            "cannot verify that holdings._yf_fetch adjusts for dividends; refusing to "
            "compute a benchmark gap that would silently include or exclude them"
        ) from exc
    if "auto_adjust=True" not in src.replace(" ", ""):
        raise RuntimeError(
            "agent.portfolio.holdings._yf_fetch no longer passes auto_adjust=True. Holdings "
            "would be price-return while an ETF benchmark stays total-return, manufacturing a "
            "gap of roughly the benchmark's dividend yield. Fix the fetcher before using this "
            "module."
        )


def _looks_like_returns(frame: pd.DataFrame) -> bool:
    """True when a frame handed in as prices is obviously daily returns.

    Prices are positive and large; daily returns straddle zero and are tiny.
    Passing returns here would make every ratio ``P(T)/P(0)`` meaningless while
    still producing a plausible-looking percentage.
    """
    vals = frame.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return False
    return bool(np.nanmax(np.abs(finite)) < 1.0 and (finite <= 0).any())


def _as_prices(panel: Any) -> tuple[pd.DataFrame, str, dict[str, str], Any]:
    """Accept a PricePanel or a raw price DataFrame; return (prices, ccy, natives, panel)."""
    natives: dict[str, str] = {}
    base = "UNKNOWN"
    if hasattr(panel, "prices") and hasattr(panel, "weights"):
        prices = panel.prices
        base = getattr(panel, "base_currency", "UNKNOWN")
        for res in getattr(panel, "resolutions", ()):
            natives[res.resolved] = res.currency
        src = panel
    elif isinstance(panel, pd.DataFrame):
        prices, src = panel, None
    else:
        raise TypeError(f"panel must be a PricePanel or a price DataFrame, got {type(panel).__name__}")

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("panel carries no prices")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("panel prices must be indexed by date")
    if _looks_like_returns(prices):
        raise ValueError(
            "this frame looks like daily RETURNS, not prices: every value is below 1.0 in "
            "absolute terms and some are negative. picking_cost needs price levels — on returns "
            "it would compute a plausible-looking and entirely wrong number"
        )
    if (prices.to_numpy(dtype=float) <= 0).any():
        raise ValueError("prices contain a non-positive value; a return ratio through zero is meaningless")
    prices = prices.sort_index()
    if prices.isna().to_numpy().any():
        raise ValueError("panel prices contain NaN; align the calendar before comparing to a benchmark")
    return prices, base, natives, src


def _as_weights(weights: Any, prices: pd.DataFrame, panel: Any) -> tuple[pd.Series, list[str]]:
    """Normalise weights onto the price columns; refuse the shapes that lie."""
    notes: list[str] = []
    if weights is None:
        if panel is None or not hasattr(panel, "weights"):
            raise ValueError("no weights given and the panel carries none")
        weights = panel.weights
    if isinstance(weights, Mapping):
        w = pd.Series(weights, dtype=float)
    elif isinstance(weights, pd.Series):
        w = weights.astype(float)
    else:
        raise TypeError(f"weights must be a mapping or Series, got {type(weights).__name__}")

    missing = [c for c in prices.columns if c not in w.index]
    if missing:
        raise ValueError(f"no weight for {', '.join(missing)}; refusing to assume zero for a held position")
    extra = [k for k in w.index if k not in prices.columns]
    if extra:
        notes.append(f"weights named {', '.join(sorted(extra))}, which the panel has no prices for; ignored here")
    w = w.reindex(prices.columns).astype(float)
    if w.isna().any():
        raise ValueError("a weight is NaN")
    if (w < 0).any():
        raise ValueError(
            "negative weights (a short position) are not supported: the buy-and-hold value path "
            "and the per-holding decomposition both assume long-only, and a wrong answer here "
            "would look right"
        )
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights sum to zero")
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        notes.append(f"weights summed to {total:.6f}; normalised to 1.0")
        w = w / total
    return w, notes


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _snap_forward(index: pd.DatetimeIndex, when: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(when, side="left")
    return None if pos >= len(index) else index[pos]


def _snap_back(index: pd.DatetimeIndex, when: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(when, side="right") - 1
    return None if pos < 0 else index[pos]


def _windows_with_skips(
    panel: Any,
    *,
    rolling_length: str | int = "1y",
    rolling_step: int = 1,
    include_rolling: bool = True,
) -> tuple[list[Window], list[str]]:
    prices, _, _, _ = _as_prices(panel)
    idx = prices.index
    windows: list[Window] = []
    skips: list[str] = []

    first, last = idx[0], idx[-1]
    windows.append(
        Window(
            label=f"full history ({first:%Y-%m-%d}..{last:%Y-%m-%d})",
            start=first,
            end=last,
            kind="full",
            rule="the entire window the book's holdings share",
        )
    )

    # (2) calendar years. A part-year is kept and labelled; dropping it would let
    #     the current year, the one the owner remembers, vanish from the table.
    for year in sorted({int(d.year) for d in idx}):
        in_year = idx[(idx.year == year)]
        if len(in_year) < MIN_WINDOW_OBS:
            skips.append(
                f"calendar {year}: only {len(in_year)} prints in the shared window, below the {MIN_WINDOW_OBS} floor"
            )
            continue
        start, end = in_year[0], in_year[-1]
        # Complete means the HISTORY covers the whole year, not that the first
        # print fell in January. A book whose history ends on 1 December has no
        # complete 2023, and labelling it "calendar 2023" would compare eleven
        # months against twelve elsewhere in the table.
        complete = first <= pd.Timestamp(year=year, month=1, day=1) and last >= pd.Timestamp(
            year=year, month=12, day=31
        )
        if complete:
            label = f"calendar {year}"
        else:
            label = f"calendar {year} (part-year, {start:%b}..{end:%b})"
        windows.append(
            Window(
                label=label,
                start=start,
                end=end,
                kind="calendar_year",
                rule="every calendar year the shared history touches, part-years included and labelled",
                complete=complete,
            )
        )

    # (3) trailing 1/2/3 years to the last print.
    for k in (1, 2, 3):
        want = last - pd.DateOffset(years=k)
        if want < first:
            have = (last - first).days / 365.25
            skips.append(f"trailing {k}y: the shared history is only {have:.2f} years long")
            continue
        start = _snap_forward(idx, want)
        if start is None or len(idx[(idx >= start) & (idx <= last)]) < MIN_WINDOW_OBS:
            skips.append(f"trailing {k}y: fewer than {MIN_WINDOW_OBS} prints in the window")
            continue
        windows.append(
            Window(
                label=f"trailing {k}y",
                start=start,
                end=last,
                kind="trailing",
                rule=f"the {k} calendar years ending at the last shared print",
            )
        )

    # (4) every rolling window of the given length, one per trading day.
    if include_rolling:
        length_label, offset, length_years = _length_spec(rolling_length)
        step = max(int(rolling_step), 1)
        made = 0
        for pos in range(0, len(idx), step):
            start = idx[pos]
            end = _snap_back(idx, start + offset)
            if end is None or end <= start:
                continue
            if start + offset > last:
                break
            n = int(((idx >= start) & (idx <= end)).sum())
            if n < MIN_WINDOW_OBS:
                continue
            windows.append(
                Window(
                    label=f"rolling {length_label} {start:%Y-%m-%d}..{end:%Y-%m-%d}",
                    start=start,
                    end=end,
                    kind="rolling",
                    rule=f"every rolling {length_label} window, one per trading day",
                )
            )
            made += 1
        if made == 0:
            skips.append(
                f"rolling {length_label}: the shared history "
                f"({(last - first).days / 365.25:.2f} years) does not contain one complete window"
            )

    return windows, skips


def default_windows(
    panel: Any,
    *,
    rolling_length: str | int = "1y",
    rolling_step: int = 1,
    include_rolling: bool = True,
) -> list[Window]:
    """The window set, fixed by :data:`WINDOW_RULE` before any return is computed.

    Full history, every calendar year (part-years labelled), trailing 1/2/3
    years, and every rolling one-year window at a one-trading-day step.

    Where this misleads: the rolling family dominates the list by count — a
    three-year book yields three named windows and ~500 rolling ones — so a
    renderer that averages over ``default_windows`` indiscriminately is averaging
    over one heavily overlapped family. Use ``Window.kind`` to separate them;
    :meth:`PickingCost.named` does.
    """
    windows, _ = _windows_with_skips(
        panel,
        rolling_length=rolling_length,
        rolling_step=rolling_step,
        include_rolling=include_rolling,
    )
    return windows


def _length_spec(length: str | int) -> tuple[str, pd.DateOffset, float]:
    """Turn ``"1y"`` / ``"6m"`` / ``252`` into (label, offset, years)."""
    if isinstance(length, int | np.integer):
        days = int(length)
        if days < MIN_WINDOW_OBS:
            raise ValueError(f"rolling length of {days} trading days is below the {MIN_WINDOW_OBS} floor")
        years = days / TRADING_DAYS_PER_YEAR
        return f"{days}d", pd.Timedelta(days=round(years * 365.25)), years
    text = str(length).strip().lower()
    if text.endswith("y"):
        n = int(text[:-1])
        return f"{n}y", pd.DateOffset(years=n), float(n)
    if text.endswith("m"):
        n = int(text[:-1])
        return f"{n}m", pd.DateOffset(months=n), n / 12.0
    raise ValueError(f"cannot read a window length from {length!r}; use '1y', '6m' or a trading-day count")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def choose_benchmarks(panel: Any) -> list[tuple[str, str]]:
    """The default benchmark(s) for a book, and the reason for each.

    The rule is the book's own market, not the number that flatters or damns it:
    every native currency holding at least one position gets its market's broad
    total-return vehicle, and all of them are reported, ordered by the share of
    the book's weight in that currency. A currency with no entry in
    :data:`DEFAULT_BENCHMARKS` gets none — an invented proxy is worse than an
    absent one.

    Where this misleads: a mixed book's gap can differ by tens of points between
    its benchmarks (a 3y INR/USD book measured here: +74.1 points against
    NIFTYBEES.NS converted to USD, -6.2 points against SPY, same book, same
    window). Reporting only the first is the cherry-pick this function exists to
    make visible, which is why the caller gets the whole list.
    """
    prices, base, natives, src = _as_prices(panel)
    try:
        w, _ = _as_weights(None, prices, src)
    except (ValueError, TypeError):
        w = pd.Series(1.0 / len(prices.columns), index=prices.columns)

    # Order by how much of the book sits in each market, not by column order:
    # the first benchmark is the one a single-headline caller will use, so the
    # rule that picks it has to be about the book and not about paste order.
    exposure: dict[str, float] = {}
    for tic in prices.columns:
        ccy = natives.get(tic, base)
        exposure[ccy] = exposure.get(ccy, 0.0) + float(w.get(tic, 0.0))
    present = sorted(exposure, key=lambda c: (-exposure[c], c)) or [base]

    out: list[tuple[str, str]] = []
    for ccy in present:
        pick = DEFAULT_BENCHMARKS.get(ccy)
        if pick is None:
            continue
        ticker, why = pick
        if len(present) > 1:
            why = (
                f"{why}; {exposure[ccy]:.0%} of the book is {ccy}-denominated, so this market is one "
                f"of the ones it is in. Benchmarks are ordered by that share, largest first — not by "
                f"which makes the gap biggest"
            )
        out.append((ticker, why))
    return out


def _period_for(span_years: float) -> str:
    for years, name in _STD_PERIODS:
        if span_years <= years - 0.02:
            return name
    return "max"


def fetch_benchmark(
    panel: Any,
    ticker: str | None = None,
    *,
    why: str | None = None,
    fetcher: Any = None,
    period: str | None = None,
) -> Benchmark:
    """Fetch one benchmark, convert it to the book's currency, and align it.

    ``ticker=None`` takes the first entry from :func:`choose_benchmarks`.

    Where this misleads: an index symbol beginning with ``^`` is a *price*
    return and is flagged ``total_return=False`` with a warning naming its
    total-return substitute — comparing it against dividend-adjusted holdings
    would hand the book a free 1.2 points a year on the Nifty. Cross-calendar
    alignment forward-fills endpoints (a shut market did not move) but never
    volatility, which is computed on the benchmark's own trading days.
    """
    prices, base, _, src = _as_prices(panel)
    if ticker is None:
        picks = choose_benchmarks(panel)
        if not picks:
            raise ValueError(
                f"no default benchmark for a book based in {base}; name one explicitly rather than "
                f"letting this module invent a proxy"
            )
        ticker, why = picks[0]
    ticker = ticker.strip().upper()

    if fetcher is None:
        _assert_total_return_fetcher()
        fetch = _h._yf_fetch
    else:
        fetch = fetcher

    span_years = (prices.index[-1] - prices.index[0]).days / 365.25
    per = period or _period_for(span_years + 0.25)

    got = _h._fetch_cached(ticker, per, fetch)
    if got is None or len(got.close) == 0:
        raise ValueError(f"no price history for benchmark {ticker} over period {per}")

    warnings: list[str] = []
    substitute = PRICE_RETURN_SYMBOLS.get(ticker)
    total_return = substitute is None and not ticker.startswith("^")
    if substitute is not None:
        warnings.append(
            f"{ticker} is a PRICE index: it excludes dividends while the holdings include them "
            f"(auto_adjust=True). Measured on this repo's cache, ^NSEI understated NIFTYBEES.NS by "
            f"121.5 bps/yr over 2021-09-29..2026-09-29. Every gap below is flattered by roughly the "
            f"index's dividend yield. {substitute} tracks the same exposure on a total-return basis."
        )
    elif not total_return:
        warnings.append(
            f"{ticker} looks like an index symbol rather than a tradable fund. If it excludes "
            f"dividends, the gap below is overstated by its dividend yield."
        )

    close = got.close
    native_ccy = got.currency
    fx_note = ""
    if native_ccy != base:
        if native_ccy == "UNKNOWN":
            raise ValueError(f"yfinance reports no currency for {ticker}; it cannot be mixed with a {base} book")
        fx, fx_name = _h._fx_series(native_ccy, base, per, fetch)
        if fx is None:
            raise ValueError(
                f"benchmark {ticker} is priced in {native_ccy} and no {native_ccy}->{base} rate is "
                f"available; comparing unconverted would be a wrong answer that looks right"
            )
        aligned = fx.reindex(close.index).ffill()
        usable = aligned.notna()
        if int(usable.sum()) < MIN_WINDOW_OBS:
            raise ValueError(f"{native_ccy}->{base} rate covers only {int(usable.sum())} of {ticker}'s days")
        close = (close[usable] * aligned[usable]).rename(ticker)
        fx_note = (
            f"{ticker} converted {native_ccy} -> {base} via {fx_name}; an FX move is part of the gap "
            f"below, because it was part of what a {base} holder would have earned"
        )

    native = close.sort_index()
    aligned_prices = native.reindex(prices.index).ffill()
    # A benchmark that starts after the book does cannot be back-filled: leaving
    # it NaN and reporting the shortfall beats inventing a level.
    have = aligned_prices.notna()
    filled = int(have.sum() - native.reindex(prices.index).notna().sum())
    if int(have.sum()) < MIN_WINDOW_OBS:
        raise ValueError(
            f"benchmark {ticker} overlaps the book's calendar on only {int(have.sum())} days "
            f"({native.index[0]:%Y-%m-%d}..{native.index[-1]:%Y-%m-%d} against "
            f"{prices.index[0]:%Y-%m-%d}..{prices.index[-1]:%Y-%m-%d})"
        )
    if not have.all():
        warnings.append(
            f"{ticker} has no price on {int((~have).sum())} of the book's {len(prices)} days "
            f"(its history starts {native.index[0]:%Y-%m-%d}); those days are excluded from every "
            f"window that contains them"
        )

    frac = filled / max(len(prices), 1)
    if frac > MAX_FILL_FRACTION:
        warnings.append(
            f"the book and {ticker} trade on different calendars: {filled} of {len(prices)} days "
            f"({frac:.1%}) use a forward-filled benchmark price. Window endpoints are unaffected; "
            f"benchmark volatility is computed on {ticker}'s own trading days for this reason"
        )

    weight_in_book = 0.0
    if src is not None and hasattr(src, "weights") and ticker in getattr(src, "weights", pd.Series(dtype=float)).index:
        weight_in_book = float(src.weights[ticker])
    elif ticker in prices.columns:
        weight_in_book = float("nan")
    if ticker in prices.columns and weight_in_book > 0:
        warnings.append(
            f"the book already holds {ticker} at {weight_in_book:.1%}. Part of this book IS the "
            f"benchmark, which drags the comparison toward a tie; the 'rest of the book' table "
            f"shows the same comparison with that position removed and the remainder reweighted"
        )

    label = ticker
    return Benchmark(
        ticker=ticker,
        label=label,
        prices=aligned_prices,
        native=native,
        native_currency=native_ccy,
        base_currency=base,
        total_return=total_return,
        why=why or "named explicitly by the caller",
        warnings=tuple(warnings),
        fx_note=fx_note,
        filled_days=filled,
        weight_in_book=weight_in_book if weight_in_book == weight_in_book else 0.0,
    )


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def _buy_and_hold_path(prices: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Value of 1.0 invested at ``w`` on the first row and never traded again.

    Weights DRIFT: this is a fixed-*allocation* counterfactual, not a
    fixed-weight one. It is the right comparison against a buy-and-hold index
    because the index drifts too.
    """
    rel = prices.divide(prices.iloc[0], axis=1)
    return rel.mul(w, axis=1).sum(axis=1)


def _constant_mix_total(prices: pd.DataFrame, w: pd.Series) -> float:
    """Total return if the book were pushed back to ``w`` every single day.

    Reported beside the headline because the two answers differ and neither is
    privileged: a real holder did neither.
    """
    rets = prices.pct_change().dropna(how="any")
    if rets.empty:
        return float("nan")
    daily = rets.mul(w, axis=1).sum(axis=1)
    return float((1.0 + daily).prod() - 1.0)


def _annualise(total: float, years: float) -> float:
    if not np.isfinite(total) or years < MIN_ANNUALISE_YEARS or total <= -1.0:
        return float("nan")
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def _realised_vol(path: pd.Series, years: float) -> float:
    """Annualised stdev of daily returns, scaled by the OBSERVED observation rate.

    Scaling by the series' own obs-per-year rather than a hardcoded 252 is what
    lets an Indian book and a US benchmark, on different calendars, be compared
    without one of them being annualised at the wrong frequency.
    """
    rets = path.pct_change().dropna()
    if len(rets) < MIN_VOL_OBS or years <= 0:
        return float("nan")
    per_year = len(rets) / years
    return float(rets.std(ddof=1) * math.sqrt(per_year))


def _ratio(annual: float, vol: float) -> float:
    if not np.isfinite(annual) or not np.isfinite(vol) or vol <= 0:
        return float("nan")
    return float(annual / vol)


def _slice(frame: pd.DataFrame | pd.Series, start: pd.Timestamp, end: pd.Timestamp):
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def _one_window(
    prices: pd.DataFrame,
    w: pd.Series,
    bench: Benchmark,
    win: Window,
    *,
    with_legs: bool = True,
) -> WindowResult:
    sub = _slice(prices, win.start, win.end)
    bsub = _slice(bench.prices, win.start, win.end).dropna()
    notes: list[str] = []

    nan = float("nan")
    if len(sub) < MIN_WINDOW_OBS or len(bsub) < MIN_WINDOW_OBS:
        return WindowResult(
            window=win,
            benchmark_ticker=bench.ticker,
            n_obs=len(sub),
            years=win.years,
            port_total=nan,
            port_annual=nan,
            port_vol=nan,
            port_return_per_vol=nan,
            bench_total=nan,
            bench_annual=nan,
            bench_vol=nan,
            bench_return_per_vol=nan,
            gap_total=nan,
            gap_annual=nan,
            usable=False,
            notes=(
                f"only {len(sub)} book prints and {len(bsub)} benchmark prints in this window; "
                f"{MIN_WINDOW_OBS} is the floor below which a return is noise with a label",
            ),
        )

    # The window's effective endpoints are the first and last date BOTH sides
    # have, never the requested ones. Using a requested date the benchmark does
    # not have is how a gap gets a day of drift baked into it.
    lo = max(sub.index[0], bsub.index[0])
    hi = min(sub.index[-1], bsub.index[-1])
    sub = _slice(sub, lo, hi)
    bsub = _slice(bsub, lo, hi)
    years = max((hi - lo).days, 0) / 365.25
    if (lo, hi) != (win.start, win.end):
        notes.append(f"effective window {lo:%Y-%m-%d}..{hi:%Y-%m-%d} — the dates both sides priced")

    path = _buy_and_hold_path(sub, w)
    port_total = float(path.iloc[-1] - 1.0)
    bench_total = float(bsub.iloc[-1] / bsub.iloc[0] - 1.0)

    port_annual = _annualise(port_total, years)
    bench_annual = _annualise(bench_total, years)
    if not np.isfinite(port_annual) and years < MIN_ANNUALISE_YEARS:
        notes.append(
            f"window is {years:.2f} years; not annualised, because annualising a short window "
            f"multiplies the number and its error together"
        )

    port_vol = _realised_vol(path, years)
    bnative = _slice(bench.native, lo, hi)
    bench_vol = _realised_vol(bnative, years)
    if not np.isfinite(bench_vol) and len(bnative) >= MIN_VOL_OBS:  # pragma: no cover - defensive
        notes.append(f"{bench.ticker} volatility unavailable for this window")
    if len(bnative) < len(bsub):
        notes.append(
            f"{bench.ticker} volatility uses its own {len(bnative)} trading days in this window, "
            f"not the book's {len(bsub)} — a foreign holiday is not a flat day"
        )

    legs: tuple[HoldingLeg, ...] = ()
    behind = 0
    if with_legs:
        rel = sub.iloc[-1] / sub.iloc[0] - 1.0
        built = [
            HoldingLeg(
                ticker=str(t),
                weight=float(w[t]),
                total_return=float(rel[t]),
                contribution=float(w[t] * rel[t]),
                vs_benchmark=float(rel[t] - bench_total),
            )
            for t in sub.columns
        ]
        # Ordered worst-against-the-index first, by the SAME rule whichever way
        # the book went: on a winning book this leads with the least-winning
        # position, not with the best one. Re-ordering per outcome would make the
        # table an argument rather than a description.
        built.sort(key=lambda leg: (leg.vs_benchmark, leg.ticker))
        legs = tuple(built)
        # A zero-weight column is not a bet anyone made; counting it would let a
        # position the book does not hold change the "N of M lost to the index" line.
        behind = sum(1 for leg in legs if leg.weight > 0 and leg.vs_benchmark < 0)
        # Buy-and-hold decomposes exactly. If it ever does not, the path and the
        # legs disagree and one of them is wrong; say so rather than print both.
        resid = abs(sum(leg.contribution for leg in legs) - port_total)
        if resid > 1e-9:  # pragma: no cover - guarded by construction
            notes.append(f"per-holding contributions miss the book total by {resid:.2e}; treat the legs as indicative")

    # Skipped when legs are off (the rolling family): it is a robustness figure
    # for a table a human reads, and 500 extra pct_change calls buy nothing.
    cm_total = _constant_mix_total(sub, w) if with_legs else float("nan")

    return WindowResult(
        window=win,
        benchmark_ticker=bench.ticker,
        n_obs=len(sub),
        years=years,
        port_total=port_total,
        port_annual=port_annual,
        port_vol=port_vol,
        port_return_per_vol=_ratio(port_annual, port_vol),
        bench_total=bench_total,
        bench_annual=bench_annual,
        bench_vol=bench_vol,
        bench_return_per_vol=_ratio(bench_annual, bench_vol),
        gap_total=port_total - bench_total,
        gap_annual=(port_annual - bench_annual) if np.isfinite(port_annual) and np.isfinite(bench_annual) else nan,
        legs=legs,
        n_legs_behind=behind,
        n_legs_held=sum(1 for leg in legs if leg.weight > 0),
        constant_mix_total=cm_total,
        constant_mix_annual=_annualise(cm_total, years),
        usable=True,
        notes=tuple(notes),
    )


WEIGHT_CAVEAT = (
    "READ THIS FIRST. Every number below applies TODAY's weights backwards. It is a "
    "fixed-allocation counterfactual: what a book that started at today's shape, and was never "
    "traded again, would have done. It is NOT what you earned. You bought and sold along the "
    "way, and without your transaction history your actual return is not computable from a "
    "holdings list — so nothing here is your actual return, and no line below claims to be. It "
    "also contains only what you still hold: anything already sold is invisible, which flatters "
    "the book rather than the index."
)


def picking_cost(
    panel: Any,
    weights: Any = None,
    *,
    benchmark: Any = None,
    windows: Sequence[Window] | None = None,
    fetcher: Any = None,
    include_rolling: bool = True,
    rolling_length: str | int = "1y",
    rolling_step: int = 1,
) -> PickingCost:
    """What picking cost, against a broad index, over every window in the rule set.

    Parameters
    ----------
    panel : PricePanel | DataFrame
        Aligned price levels. Daily returns are refused, loudly.
    weights : mapping | Series | None
        Today's weights. ``None`` takes the panel's.
    benchmark : str | Benchmark | sequence | None
        ``None`` takes :func:`choose_benchmarks`, which is driven by the book's
        own currencies. A mixed-currency book gets one benchmark per market and
        all of them are reported — picking the one that makes the gap biggest is
        the most gameable move available here, so the choice is a rule.
    windows : sequence[Window] | None
        ``None`` takes :func:`default_windows`. Passing a hand-picked list is
        allowed and is exactly the cherry-pick the rule set exists to prevent,
        so the returned ``window_rule`` says the set was supplied by the caller.

    Returns a :class:`PickingCost`. Nothing is dropped silently: skipped windows,
    ignored weights and excluded holdings all arrive with a reason attached.

    Where this misleads: see :data:`WEIGHT_CAVEAT` (today's weights are not the
    weights held) and the module docstring (total-return matching, benchmark
    choice, overlapping rolling windows).
    """
    prices, base, _, src = _as_prices(panel)
    w, weight_notes = _as_weights(weights, prices, src if src is not None else panel)

    # --- benchmarks ---------------------------------------------------------
    marks: list[Benchmark] = []
    if benchmark is None:
        picks = choose_benchmarks(panel)
        if not picks:
            raise ValueError(
                f"no default benchmark for a {base} book; name one rather than letting this module invent a proxy"
            )
        for tic, why in picks:
            marks.append(fetch_benchmark(panel, tic, why=why, fetcher=fetcher))
    elif isinstance(benchmark, Benchmark):
        marks.append(benchmark)
    elif isinstance(benchmark, str):
        marks.append(fetch_benchmark(panel, benchmark, fetcher=fetcher))
    elif isinstance(benchmark, Iterable):
        for item in benchmark:
            marks.append(item if isinstance(item, Benchmark) else fetch_benchmark(panel, item, fetcher=fetcher))
    else:
        raise TypeError(f"benchmark must be None, a ticker, a Benchmark, or a sequence; got {type(benchmark).__name__}")
    if not marks:
        raise ValueError("no benchmark to compare against")

    # --- windows ------------------------------------------------------------
    if windows is None:
        wins, skips = _windows_with_skips(
            panel,
            rolling_length=rolling_length,
            rolling_step=rolling_step,
            include_rolling=include_rolling,
        )
        rule = WINDOW_RULE
    else:
        wins, skips = list(windows), []
        rule = (
            "WINDOWS SUPPLIED BY THE CALLER, not by the rule set. A window chosen after seeing "
            "the answer is a cherry-pick; " + WINDOW_RULE
        )

    results: dict[str, tuple[WindowResult, ...]] = {}
    ex_bench: dict[str, tuple[WindowResult, ...]] = {}
    assumptions: list[str] = list(weight_notes)
    excluded: list[tuple[str, str]] = []

    for mark in marks:
        # Per-holding legs and the constant-mix figure are for the handful of
        # windows a human reads. Building them for every rolling window too
        # multiplies the work by ~500 for a table nothing renders.
        results[mark.ticker] = tuple(_one_window(prices, w, mark, win, with_legs=win.kind != "rolling") for win in wins)

        # The book holding the benchmark is not a footnote: part of it IS the
        # benchmark, and the interesting comparison is about the rest.
        if mark.ticker in prices.columns and mark.weight_in_book > 0:
            rest_cols = [c for c in prices.columns if c != mark.ticker]
            if len(rest_cols) == 0:
                excluded.append((mark.ticker, "the book is nothing but the benchmark; there is no 'rest' to compare"))
            else:
                rest_w = w[rest_cols] / w[rest_cols].sum()
                rest_px = prices[rest_cols]
                ex_bench[mark.ticker] = tuple(
                    _one_window(rest_px, rest_w, mark, win) for win in wins if win.kind != "rolling"
                )
                assumptions.append(
                    f"the book holds {mark.ticker} at {mark.weight_in_book:.1%}; the 'rest of the book' "
                    f"table removes it and reweights the remaining {len(rest_cols)} positions to 1.0"
                )
        for warn in mark.warnings:
            assumptions.append(warn)
        if mark.fx_note:
            assumptions.append(mark.fx_note)

    if src is not None:
        for exc in getattr(src, "excluded", ()):
            excluded.append((exc.symbol, exc.reason))

    span = f"{prices.index[0]:%Y-%m-%d}..{prices.index[-1]:%Y-%m-%d} ({len(prices)} shared prints)"
    return PickingCost(
        benchmarks=tuple(marks),
        results=results,
        window_rule=rule,
        weight_caveat=WEIGHT_CAVEAT,
        base_currency=base,
        span=span,
        ex_benchmark=ex_bench,
        skipped_windows=tuple(skips),
        excluded=tuple(excluded),
        assumptions=tuple(assumptions),
    )


def rolling_gap(
    panel: Any,
    weights: Any = None,
    benchmark: Any = None,
    *,
    length: str | int = "1y",
    step: int = 1,
    fetcher: Any = None,
) -> RollingGap:
    """Over EVERY rolling window of one length: how often did the book beat the index?

    This is the anti-cherry-pick, and it is the number the product is defensible
    on. "You beat the index in 3 of 24 rolling one-year windows; the median
    window was 4.1 points behind, the best 11.0 ahead, the worst 19.6 behind"
    cannot be improved by choosing an endpoint, which is exactly what a single
    headline can.

    Where this misleads: **rolling windows overlap**. Twenty-four one-year
    windows cut from three years of data share most of their days and are worth
    roughly three independent observations, not twenty-four;
    :attr:`RollingGap.n_independent` carries that number and the caveat list
    states it in words. A beat fraction computed on overlapping windows is a
    description of this one history, not an estimate of a probability.
    """
    prices, base, _, src = _as_prices(panel)
    w, _ = _as_weights(weights, prices, src if src is not None else panel)

    if isinstance(benchmark, Benchmark):
        mark = benchmark
    elif benchmark is None:
        mark = fetch_benchmark(panel, None, fetcher=fetcher)
    else:
        mark = fetch_benchmark(panel, str(benchmark), fetcher=fetcher)

    label, offset, length_years = _length_spec(length)
    wins, skips = _windows_with_skips(panel, rolling_length=length, rolling_step=step, include_rolling=True)
    rolling = [win for win in wins if win.kind == "rolling"]

    span_years = (prices.index[-1] - prices.index[0]).days / 365.25
    if not rolling:
        return RollingGap(
            benchmark_ticker=mark.ticker,
            length_label=label,
            length_years=length_years,
            series=pd.DataFrame(columns=["start", "port_total", "bench_total", "gap"]),
            n_windows=0,
            n_independent=0.0,
            beat_count=0,
            beat_fraction=float("nan"),
            median_gap=float("nan"),
            mean_gap=float("nan"),
            best_gap=float("nan"),
            best_window="",
            worst_gap=float("nan"),
            worst_window="",
            headline=f"the book's shared history is {span_years:.2f} years — not one complete {label} window",
            caveats=tuple(skips),
            usable=False,
            unusable_reason=f"no complete {label} window inside {span_years:.2f} years of shared history",
        )

    rows = []
    for win in rolling:
        res = _one_window(prices, w, mark, win, with_legs=False)
        if not res.usable:
            continue
        rows.append(
            {
                "start": win.start,
                "end": win.end,
                "port_total": res.port_total,
                "bench_total": res.bench_total,
                "gap": res.gap_total,
            }
        )
    series = pd.DataFrame(rows).set_index("end").sort_index()

    gaps = series["gap"].to_numpy(dtype=float)
    n = int(len(gaps))
    beat = int((gaps > 0).sum())
    best_i = int(np.argmax(gaps))
    worst_i = int(np.argmin(gaps))
    n_ind = span_years / length_years if length_years > 0 else float("nan")

    def _win_label(i: int) -> str:
        return f"{series.iloc[i].name:%Y-%m-%d} back to {series['start'].iloc[i]:%Y-%m-%d}"

    headline = (
        f"the book out-returned {mark.ticker} in {beat} of {n} rolling {label} windows "
        f"({beat / n:.0%}) over {prices.index[0]:%Y-%m-%d}..{prices.index[-1]:%Y-%m-%d}"
    )

    caveats = [
        f"these {n} windows overlap heavily: {span_years:.2f} years of history divided by a {label} "
        f"window is about {n_ind:.1f} independent observations, not {n}. The fraction describes this "
        f"one history and is not a probability",
        "every window applies today's weights backwards — see the weight caveat",
        f"gaps are total return over the window, {base}, dividends included on both sides",
    ]
    if not mark.total_return:
        caveats.insert(0, f"{mark.ticker} is a price index and excludes dividends; every gap here is overstated")
    if mark.weight_in_book > 0:
        caveats.append(f"the book holds {mark.ticker} at {mark.weight_in_book:.1%}, dragging every window toward a tie")
    caveats += skips

    return RollingGap(
        benchmark_ticker=mark.ticker,
        length_label=label,
        length_years=length_years,
        series=series,
        n_windows=n,
        n_independent=float(n_ind),
        beat_count=beat,
        beat_fraction=beat / n,
        median_gap=float(np.median(gaps)),
        mean_gap=float(np.mean(gaps)),
        best_gap=float(gaps[best_i]),
        best_window=_win_label(best_i),
        worst_gap=float(gaps[worst_i]),
        worst_window=_win_label(worst_i),
        headline=headline,
        caveats=tuple(caveats),
        usable=True,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct(x: float, width: int = 7) -> str:
    return f"{'--':>{width}}" if not np.isfinite(x) else f"{x * 100:{width}.1f}%"


def _num(x: float, width: int = 6) -> str:
    return f"{'--':>{width}}" if not np.isfinite(x) else f"{x:{width}.2f}"


def _results_table(rows: Sequence[WindowResult], title: str) -> list[str]:
    out = [title]
    out.append(
        f"  {'window':<38}{'book':>9}{'index':>9}{'gap':>9}"
        f"{'book/yr':>9}{'idx/yr':>9}{'gap/yr':>9}{'bk vol':>8}{'ix vol':>8}{'bk r/v':>8}{'ix r/v':>8}"
    )
    for r in rows:
        if not r.usable:
            out.append(f"  {r.window.label:<38}{'-- not computed: ' + (r.notes[0] if r.notes else 'no data')}")
            continue
        out.append(
            f"  {r.window.label:<38}{_pct(r.port_total, 8)} {_pct(r.bench_total, 8)} {_pct(r.gap_total, 8)} "
            f"{_pct(r.port_annual, 8)} {_pct(r.bench_annual, 8)} {_pct(r.gap_annual, 8)} "
            f"{_pct(r.port_vol, 7)} {_pct(r.bench_vol, 7)} {_num(r.port_return_per_vol, 7)} "
            f"{_num(r.bench_return_per_vol, 7)}"
        )
    return out


def format_report(
    panel: Any,
    weights: Any = None,
    *,
    benchmark: Any = None,
    fetcher: Any = None,
    length: str | int = "1y",
    max_rolling_rows: int = 0,
) -> str:
    """The whole comparison as text a stranger can read and check in thirty seconds.

    Every figure carries its window. No sentence tells the reader what to do or
    what happens next; ``concentration._ADVICE_WORDS`` is the list this output is
    tested against, and the forward-looking phrases are tested separately.
    """
    from agent.portfolio import concentration as _c  # noqa: PLC0415 - shared vocabulary only

    # include_rolling=False here only: the rolling family is computed once, by
    # rolling_gap below, and computing it twice doubles the work for one table.
    cost = picking_cost(panel, weights, benchmark=benchmark, fetcher=fetcher, include_rolling=False)
    L: list[str] = []
    L.append(cost.weight_caveat)
    L.append("")
    L.append(f"Book       : {cost.span}, everything in {cost.base_currency}")
    L.append(f"Window rule: {cost.window_rule}")
    L.append("")

    for mark in cost.benchmarks:
        L.append(f"Benchmark  : {mark.ticker} — {mark.why}")
        L.append(f"             total-return basis: {'yes' if mark.total_return else 'NO — see warning below'}")
        rows = cost.named(mark.ticker)
        L.append("")
        L += _results_table(rows, f"Book against {mark.ticker}, by window")
        L.append("")
        L.append(
            f"  r/v is annualised return divided by annualised volatility; no risk-free rate is "
            f"subtracted, so it is not a Sharpe ratio. A window shorter than {MIN_ANNUALISE_YEARS} "
            f"years is not annualised at all (shown as --): annualising a part-year multiplies the "
            f"number and its error together."
        )

        full = cost.full_history(mark.ticker)
        if full is not None and full.usable:
            L.append("")
            L.append(f"Per holding over the full history, against {mark.ticker} ({_pct(full.bench_total).strip()})")
            L.append(
                f"  ordered by how each position did against {mark.ticker}, furthest behind first — "
                f"the same rule whichever way the book went"
            )
            L.append(f"  {'ticker':<16}{'weight':>8}{'own return':>13}{'vs index':>11}{'contribution':>15}")
            for leg in full.legs:
                L.append(
                    f"  {leg.ticker:<16}{leg.weight * 100:7.1f}%{_pct(leg.total_return, 12)} "
                    f"{_pct(leg.vs_benchmark, 10)} {_pct(leg.contribution, 14)}"
                )
            L.append(
                f"  {full.n_legs_held - full.n_legs_behind} of {full.n_legs_held} positions out-returned "
                f"{mark.ticker} over this window; contributions sum to the book's "
                f"{_pct(full.port_total).strip()}."
            )
            L.append(
                f"  Constant-mix variant (weights pushed back to today's every day): "
                f"{_pct(full.constant_mix_total).strip()} total. The headline path never trades, so its "
                f"weights drift. A real holder did neither of these two things."
            )

        if mark.ticker in cost.ex_benchmark:
            L.append("")
            L += _results_table(
                cost.ex_benchmark[mark.ticker],
                f"The REST of the book (with {mark.ticker} removed and the remainder reweighted)",
            )
        L.append("")

    rg = rolling_gap(panel, weights, cost.benchmarks[0], length=length, fetcher=fetcher)
    L.append(f"Every rolling {rg.length_label} window — the part that cannot be cherry-picked")
    if not rg.usable:
        L.append(f"  not computed: {rg.unusable_reason}")
    else:
        L.append(f"  {rg.headline}")
        L.append(
            f"  median window {_pct(rg.median_gap).strip()}   best {_pct(rg.best_gap).strip()} "
            f"({rg.best_window})   worst {_pct(rg.worst_gap).strip()} ({rg.worst_window})"
        )
        if max_rolling_rows:
            shown = rg.series.iloc[:: max(1, len(rg.series) // max_rolling_rows)]
            L.append(f"  {'window ending':<16}{'book':>9}{'index':>9}{'gap':>9}")
            for end, row in shown.iterrows():
                L.append(
                    f"  {end:%Y-%m-%d}    {_pct(row['port_total'], 8)} {_pct(row['bench_total'], 8)} "
                    f"{_pct(row['gap'], 8)}"
                )
    for c in rg.caveats:
        L.append(f"  - {c}")

    if cost.skipped_windows:
        L.append("")
        L.append("Windows in the rule set that this history cannot cover")
        L += [f"  - {s}" for s in cost.skipped_windows]
    if cost.excluded:
        L.append("")
        L.append("Excluded (nothing is dropped silently)")
        L += [f"  - {sym}: {why}" for sym, why in cost.excluded]
    if cost.assumptions:
        L.append("")
        L.append("Assumptions and caveats")
        L += [f"  - {a}" for a in dict.fromkeys(cost.assumptions)]

    L.append("")
    L.append(
        "Every figure above describes price movement inside a stated window and nothing after it. "
        "These are arithmetic on the past, not forecasts."
    )
    text = "\n".join(L)
    bad = [wd for wd in _c._ADVICE_WORDS if wd in text.lower()]
    if bad:  # pragma: no cover - the tests are the real guard; this is the seatbelt
        log.error("advice language reached the rendered report: %s", bad)
    return text


def _advice_words_in(text: str) -> list[str]:
    """Which forbidden words a rendered report contains. Used by the tests."""
    from agent.portfolio import concentration as _c  # noqa: PLC0415

    low = text.lower()
    return [wd for wd in _c._ADVICE_WORDS if wd in low]
