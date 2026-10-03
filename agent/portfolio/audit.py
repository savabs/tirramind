"""The product object: one pasted book in, one report out.

``audit(text)`` composes the tested portfolio modules into a single report;
``render_text(audit)`` turns it into the demo output. This module is *assembly*.
It computes almost nothing itself. It owns which modules are called, in what
order their answers are shown, what is withheld when a piece is missing, and
the wording that surrounds the numbers — and nothing else.

WHAT THE REPORT SAYS, AND WHY IN THIS ORDER
-------------------------------------------
1. **THE NUMBER** — the book's total return against one benchmark over the
   longest window the data supports, *with the rolling-window record attached
   to it*. A broker shows profit against a purchase price and never against
   the alternative of having done nothing; that gap is the finding.
2. **WHICH DECISIONS** — per holding, ranked by what it moved the book's gap
   by, worst first, plus the two counterfactuals (drop the worst line, drop the
   best line) that stop the ranking being a smear.
3. **WHY IT MOVED THAT WAY** — risk share against weight, correlation groups,
   worst-day behaviour. Structure, demoted to explanation. Facts about a
   portfolio's STRUCTURE do not surprise the person who chose that structure,
   which is why structure is third and not first.
4. **WHAT THIS IS NOT** — a section, in plain words, not a footnote.

THE ONE RULE THAT SHAPES THE WHOLE FILE
---------------------------------------
**A single window is a cherry-pick.** Pick a different three years and the sign
flips. A product whose pitch is "we do not let you fool yourself" cannot ship a
one-window number on its own page. So:

* :attr:`Audit.headline_publishable` is False unless a multi-window record was
  computed, and :func:`render_text` refuses to print the headline table
  without it. Printing the number alone has to be something a caller does on
  purpose, against the grain of this API.
* Windows come from ``benchmark.WINDOW_RULE`` and
  ``attribution.multi_window_attribution``, both of which fix the window set
  before any return is computed. This module never selects a window.
* Contradicting windows are rendered before agreeing ones, inheriting the
  ordering ``MultiWindowResult.headline_qualifier`` already enforces.

WHERE THIS MISLEADS — read before quoting any number out of here
----------------------------------------------------------------
Each composed module documents its own distortions and this file repeats none
of them. What is specific to the *assembly*:

* **One benchmark reaches the page.** ``benchmark.choose_benchmarks`` can
  return several for a mixed-currency book. This report renders the first and
  names the others as not rendered. Choosing among benchmarks is the most
  gameable move available in this product, so the choice is a rule (the book's
  own market) and the rule is printed.
* **Three modules, three window definitions.** The headline table's window is
  the shared book-and-benchmark calendar from ``benchmark``; the attribution's
  window is ``attribution``'s own alignment; ``concentration`` and ``overlap``
  run on the book's panel with no benchmark in it at all. They usually differ
  by a handful of days. Every block therefore prints its own window and its
  own n, and no number from one block is arithmetic with a number from another.
* **A degraded section is still a rendered section.** If a sibling module is
  missing or raises, that block says "not computed" and the reason. It never
  silently disappears, and the rest of the report still renders. This repo has
  eighteen recorded failure modes and nearly all of them were silent.
* **Nothing here is advice.** No line recommends, suggests, forecasts, or
  implies an action. ``advice_words_in`` is the list the tests assert the
  rendered report against, and it is exported so a caller can assert it too.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agent.portfolio.holdings import (
    Holdings,
    PricePanel,
    fetch_prices,
    parse_holdings,
)

__all__ = [
    "ADVICE_WORDS",
    "DEFAULT_PERIOD",
    "DEFAULT_ROLLING_LENGTHS",
    "DEFAULT_ROLLING_STEP",
    "MAX_SERIES_ROWS",
    "MIN_WINDOWS_FOR_TALLY",
    "Audit",
    "Section",
    "BENIGN_PHRASES",
    "LIABILITY_WORDS",
    "advice_words_in",
    "audit",
    "render_text",
]

# ---------------------------------------------------------------------------
# Constants. Each one can change what the reader concludes, so each is named,
# exported and overridable rather than buried in a signature default.
# ---------------------------------------------------------------------------

#: yfinance period fetched for both book and benchmark. Five years is the
#: longest span over which a typical retail Indian book's holdings all exist;
#: "max" truncates to the youngest ETF anyway and costs a much slower fetch.
DEFAULT_PERIOD = "5y"

#: Rolling-window lengths reported beside the headline. One year is short
#: enough to be numerous and long enough not to be noise; three years is the
#: horizon a headline like this is usually quoted on, and is the one that most
#: often disagrees with the full history.
DEFAULT_ROLLING_LENGTHS: tuple[str, ...] = ("1y", "3y")

#: Step between rolling windows, in trading days. Every rolling window is
#: computed (step=1 inside ``benchmark.rolling_gap`` would be one per trading
#: day); 21 is one month and is what the *displayed* series is thinned to.
#: Thinning changes the row count, not the tally — the tally is computed on
#: every window at this step and the step is printed.
DEFAULT_ROLLING_STEP = 21

#: Rows of a rolling series printed in full before the display thins further.
#: The series is never summarised away entirely: F-18 in this repo's log is a
#: number reported as a result that was in fact a constant across every run,
#: and the only defence that works is showing the values.
MAX_SERIES_ROWS = 40

#: Below this many windows, a rolling family is reported as the individual
#: windows it is and no tally, fraction or median is quoted from it. "Behind in
#: 1 of 1 windows (0%)" reads as evidence and is a single number wearing a
#: percentage sign.
MIN_WINDOWS_FOR_TALLY = 3

#: Words that would turn arithmetic into regulated investment advice (SEBI
#: Investment Adviser), plus the forward-looking forms that would turn a fact
#: about the past into a forecast. The first blocks are kept in step with
#: ``concentration._ADVICE_WORDS`` so every report in this package is held to
#: one standard.
ADVICE_WORDS: tuple[str, ...] = (
    # imperatives and recommendations
    "you should",
    "we recommend",
    "recommend",
    "you could",
    "you must",
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
    "switch to",
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
    # forecasts: a fact about the past carries no liability, a forecast does
    "expect",
    "forecast",
    "predict",
    "going forward",
    "next year",
    "in future",
    "likely to",
    "will continue",
    "should continue",
)

#: The subset of :data:`ADVICE_WORDS` whose presence would make this a
#: regulated recommendation or a forecast rather than arithmetic: imperatives,
#: transaction verbs, and forward-looking forms. Every word here is asserted
#: absent from the WHOLE rendered report, sibling-supplied strings included,
#: because SEBI does not care which module wrote the sentence.
LIABILITY_WORDS: tuple[str, ...] = (
    "you should",
    "we recommend",
    "recommend",
    "you could",
    "you must",
    "consider ",
    "advis",
    "suggest",
    "ought to",
    "need to",
    "buy ",
    "sell ",
    "trim",
    "rebalance",
    "reduce your",
    "increase your",
    "add to your",
    "cut your",
    "hedge your",
    "switch to",
    "expect",
    "forecast",
    "predict",
    "going forward",
    "next year",
    "in future",
    "likely to",
    "will continue",
    "should continue",
)

#: Exact phrases in which a listed word is a technical term rather than an
#: instruction, paired with the module that produces them. The exemption is by
#: PHRASE, not by word: "buy and hold" is the name of a measurement basis, so
#: it is exempt, while "buy NIFTYBEES" would still be caught. Each entry is a
#: standing claim that a specific sibling string is benign, and it fails loudly
#: if that string changes.
BENIGN_PHRASES: tuple[tuple[str, str], ...] = (
    ("buy and hold", "attribution.AttributionResult.basis_note — the name of the counterfactual"),
    (
        "look better here than it was",
        "attribution.survivorship_note — the direction of an unmeasurable bias, not a judgement of the book",
    ),
    ("make it look worse", "attribution.survivorship_note — the opposite direction of the same bias"),
)

#: Characters of preceding context scanned for a negator. "This is not a
#: forecast" must not be flagged as a forecast, and the negator is always
#: within a few words of the term in the phrasings this package uses.
_NEGATION_WINDOW = 40
_NEGATORS = ("not ", "never ", "no ", "nothing ", "none of ", "cannot ", "n't ")


def advice_words_in(
    text: str,
    *,
    words: Sequence[str] | None = None,
    allow_negated: bool = True,
    allow_benign: bool = True,
) -> list[str]:
    """Which forbidden words a rendered report contains.

    ``words`` defaults to :data:`ADVICE_WORDS`; pass :data:`LIABILITY_WORDS` to
    scan only for the forms that would make the page regulated advice.

    ``allow_negated`` skips an occurrence with a negator in the preceding
    ``_NEGATION_WINDOW`` characters, so a disclaimer ("none of it is a
    recommendation", "this is not a forecast") does not register as the thing
    it disclaims. ``allow_benign`` skips an occurrence that falls inside one of
    the exact phrases in :data:`BENIGN_PHRASES`.

    WHERE THIS MISLEADS
        It is a substring scan with two escape hatches. The negation heuristic
        would forgive "not only should you..."; the benign list is a standing
        assertion that four specific sibling strings are technical terms, and
        it would forgive a new sentence that happened to contain one of them.
        This is a floor on discipline, not a compliance proof, and it is never
        a substitute for reading the output.
    """
    low = text.lower()
    benign: list[tuple[int, int]] = []
    if allow_benign:
        for phrase, _source in BENIGN_PHRASES:
            at = low.find(phrase)
            while at != -1:
                benign.append((at, at + len(phrase)))
                at = low.find(phrase, at + 1)
    hits: list[str] = []
    for word in words if words is not None else ADVICE_WORDS:
        pos = low.find(word)
        while pos != -1:
            excused = (allow_negated and _is_negated(low, pos)) or any(a <= pos < b for a, b in benign)
            if not excused:
                hits.append(word)
                break
            pos = low.find(word, pos + 1)
    return hits


def _is_negated(low: str, pos: int) -> bool:
    context = low[max(0, pos - _NEGATION_WINDOW) : pos]
    return any(n in context for n in _NEGATORS)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """A block that either computed, or explicitly did not and says why.

    ``computed=False`` with an empty ``reason`` raises. A block that vanished
    without saying why is the silent-failure shape behind most of this repo's
    recorded defects, and it is made unrepresentable here rather than
    discouraged in a comment.
    """

    key: str
    title: str
    computed: bool
    reason: str = ""
    lines: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.computed and not self.reason:
            raise ValueError(f"section {self.key!r} is not computed but carries no reason")


@dataclass(frozen=True)
class Audit:
    """Everything the report needs, already computed, in render order.

    The sibling result objects are held as-is (``benchmark.PickingCost``,
    ``benchmark.RollingGap``, ``attribution.AttributionResult``,
    ``attribution.MultiWindowResult``) rather than flattened into fields of
    this class. Flattening them would mean re-deriving numbers those modules
    already computed and tested, and a re-derivation that drifts from its
    source is exactly how a report starts disagreeing with itself.
    """

    holdings: Holdings
    panel: PricePanel
    benchmark_ticker: str
    benchmark_why: str
    other_benchmarks: tuple[tuple[str, str], ...]
    picking: Any | None
    picking_reason: str
    records: tuple[Any, ...]
    records_reason: str
    attribution: Any | None
    summary: Mapping[str, Any]
    attribution_reason: str
    multi: Any | None
    multi_reason: str
    structure: tuple[Section, ...]
    survivorship: str
    limitations: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return bool(self.panel.usable)

    @property
    def full_window(self) -> Any | None:
        """The full-history ``WindowResult``, or None if it was not computed."""
        if self.picking is None:
            return None
        return self.picking.full_history(self.benchmark_ticker)

    @property
    def usable_records(self) -> tuple[Any, ...]:
        return tuple(r for r in self.records if getattr(r, "usable", False) and r.n_windows > 0)

    @property
    def basis_check(self) -> dict[str, Any] | None:
        """The same book, the same window, two counterfactuals, two answers.

        ``agent.portfolio.benchmark`` reads the pasted weights as the
        ALLOCATION AT THE START of each window: "a book that started at today's
        shape and was never traded again". ``agent.portfolio.attribution``
        reads them as END weights and backs out day one from them: "the share
        counts today's weights imply, held throughout". Both are documented,
        both are defensible, and on a book with one large winner they differ by
        a lot — the first hands that winner its post-run-up weight from day
        one, so it systematically flatters a book that had one.

        On a 15-name Indian book measured 2023-09..2026-09 the two bases gave
        +5.7 points and -8.6 points against the same index: opposite SIGNS.
        Rendering one of them alone would be the cherry-pick this product
        exists to refuse, so this property exists to force both onto the page.

        Returns None when either basis is missing — never a half-comparison.
        """
        fw = self.full_window
        attr = self.attribution
        if fw is None or attr is None or not attr.usable:
            return None
        alloc_gap = float(fw.gap_total)
        shares_gap = float(attr.gap_total)
        return {
            "allocation_book_total": float(fw.port_total),
            "allocation_gap": alloc_gap,
            "allocation_window": f"{fw.window.start:%Y-%m-%d}..{fw.window.end:%Y-%m-%d} (n={fw.n_obs})",
            "shares_book_total": float(attr.book.total_return),
            "shares_gap": shares_gap,
            "shares_window": str(attr.window),
            "book_total_difference": float(fw.port_total) - float(attr.book.total_return),
            "gap_difference": alloc_gap - shares_gap,
            "signs_agree": (alloc_gap > 0) == (shares_gap > 0),
        }

    @property
    def headline_publishable(self) -> bool:
        """False unless the headline has a multi-window record to travel with.

        One window is a selected window. This product's entire claim is that it
        does not select, so the headline is withheld — not footnoted, withheld
        — when nothing was computed to check it against. A caller that wants
        the number anyway must reach past ``render_text`` for it.
        """
        if self.full_window is None:
            return False
        return bool(self.usable_records) or (self.multi is not None and bool(self.multi.usable_windows))


# ---------------------------------------------------------------------------
# Composition helpers. Each one returns a value and a reason; never raises.
# ---------------------------------------------------------------------------


def _weights_map(panel: PricePanel) -> dict[str, float]:
    return {str(k): float(v) for k, v in panel.weights.items()}


def _why(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _section_concentration(panel: PricePanel) -> Section:
    title = "Where the movement sat, against where the money sat"
    try:
        from agent.portfolio import concentration as conc  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return Section("concentration", title, False, f"agent.portfolio.concentration — {_why(exc)}")
    try:
        rc = conc.risk_contributions(panel, _weights_map(panel))
    except Exception as exc:  # noqa: BLE001
        return Section("concentration", title, False, f"risk_contributions — {_why(exc)}")
    rows = rc.get("contributions") or []
    if not rows:
        return Section(
            "concentration", title, False, rc.get("reliability_reason") or "no contribution row was returned"
        )
    lines = [
        f"    {r['ticker']:<14}{r['weight']:>7.1%} of the money{r['risk_share']:>10.1%} of the variance"
        f"    (its own vol {r['standalone_vol_annualised']:.1%}/yr, correlation with the rest "
        f"{r['correlation_with_portfolio']:+.2f})"
        for r in rows
    ]
    lines.append(
        f"    n={rc['n_observations']} daily returns over {rc['window']}, p={rc['n_assets']} holdings "
        f"(n/p={rc['n_over_p']})"
    )
    if not rc.get("reliable", True):
        lines.append(f"    NOT RELIABLE: {rc.get('reliability_reason', '')}")
    if rc.get("has_negative_contributions"):
        lines.append(
            "    A negative share means that holding moved against the rest of the book. "
            "That is arithmetic, not an error."
        )
    return Section("concentration", title, True, lines=tuple(lines), data=rc)


def _section_overlap(panel: PricePanel) -> Section:
    title = "Line items that moved as one"
    try:
        from agent.portfolio import overlap as ov  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return Section("overlap", title, False, f"agent.portfolio.overlap — {_why(exc)}")
    try:
        clusters = ov.correlation_clusters(panel, _weights_map(panel))
    except Exception as exc:  # noqa: BLE001
        return Section("overlap", title, False, f"correlation_clusters — {_why(exc)}")
    lines = [f"    {c.statement()}" for c in clusters]
    singles = tuple(getattr(clusters, "singletons", ()))
    if singles:
        lines.append(f"    Joined no group at this cutoff: {', '.join(singles)}.")
    stable = getattr(clusters, "stable_between", None)
    if stable:
        lines.append(
            f"    This exact grouping is unchanged for every correlation cutoff between "
            f"{stable[0]:.2f} and {stable[1]:.2f}, so it is a property of the book rather than "
            "of the cutoff."
        )
    lines += [f"    {n}" for n in getattr(clusters, "notes", ())]
    lines += [f"    left out — {e}" for e in getattr(clusters, "excluded", ())]
    if not lines:
        return Section("overlap", title, False, "no group and no exclusion was produced")
    return Section("overlap", title, True, lines=tuple(lines), data={"n_groups": clusters.n_groups})


def _section_worst_days(panel: PricePanel) -> Section:
    title = "What each holding did on the book's worst days"
    try:
        from agent.portfolio import beliefs  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return Section("worst_days", title, False, f"agent.portfolio.beliefs — {_why(exc)}")
    try:
        res = beliefs.drawdown_coincidence(panel, _weights_map(panel))
    except Exception as exc:  # noqa: BLE001
        return Section("worst_days", title, False, f"drawdown_coincidence — {_why(exc)}")
    n_worst = int(res.get("n_worst_days", 0))
    lines = [f"    The worst {n_worst} of {res.get('n_days', 0)} days in {res.get('window', 'the window')}:"]
    for h in res.get("holdings", []):
        lines.append(
            f"    {h['ticker']:<14}fell on {h['n_fell']:>2} of those {n_worst}"
            f"   (it falls on {h['fell_frequency_all_days']:.0%} of all days)"
            f"   mean {h['mean_return_on_worst_days']:+.2%} then, {h['mean_return_on_other_days']:+.2%} otherwise"
        )
    clustering = res.get("worst_day_clustering") or {}
    note = clustering.get("note") if isinstance(clustering, Mapping) else None
    if note:
        lines.append(f"    {note}")
    lines += [f"    {c}" for c in res.get("caveats", ())]
    return Section("worst_days", title, True, lines=tuple(lines), data=res)


def _limitations(panel: PricePanel, bench: str, other: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """The 'what this is not' section, built from this book's own facts."""
    items = [
        "TWO WEIGHTING BASES, TWO ANSWERS, BOTH PRINTED. A holdings list says what is held now. "
        "Read as the allocation a book STARTED with, it gives one figure; read as end weights with "
        "day one backed out of them, it gives another. Section 1 uses the first and section 2 the "
        "second, each labelled, and on a book with one large winner they differ by more than ten "
        "points. Neither is what was actually earned, because that needs a transaction history and "
        "a holdings list does not contain one.",
        "FIXED WEIGHTS, APPLIED BACKWARDS. There is one weight vector in this report: what the "
        f"book looks like on its last date ({panel.weight_basis or 'as pasted'}). Every earlier "
        "window is measured with the share counts those weights imply, re-valued at that window's "
        "first close. So a 2023 window here is 'this book, over 2023' — it is not what was held in "
        "2023. Real purchase dates, additions and withdrawals are not in the input.",
        "OVERLAPPING WINDOWS ARE NOT INDEPENDENT OBSERVATIONS. Rolling windows one month apart "
        "share about 92% of their days. A tally like 'behind in 38 of 41' is one history looked at "
        "41 times. Every record below prints how many NON-overlapping windows the history actually "
        "contains, and that is the number the tally means.",
        "SURVIVORSHIP, IN BOTH DIRECTIONS. A pasted book is a list of what is still held. Positions "
        "sold at a loss and positions sold at a gain are equally absent, and which dominates for a "
        "retail holder is not settled. The direction of that bias is unknown here, so no correction "
        "is applied and none is claimed.",
        f"ONE BENCHMARK, ONE MARKET, ONE CURRENCY. Everything is in {panel.base_currency} and "
        f"against {bench} alone"
        + (f" ({', '.join(t for t, _ in other)} also applies to this book and is not rendered here)." if other else ".")
        + " A different benchmark can flip the sign of every gap below. This is one comparison, "
        "not a verdict.",
        "NO COSTS AND NO TAX, ON EITHER SIDE. No brokerage, no STT, no stamp duty, no capital-gains "
        "tax, no dividend tax, and no cost of the switching this report does not describe. An "
        "after-tax outcome is not any number here.",
        "TOTAL RETURNS ON BOTH SIDES. Adjusted closes fold dividends back in, for the book and for "
        "the benchmark alike. A broker app showing price-only profit and loss will disagree with "
        "these figures, and a high-yield holding reads stronger here than it does there.",
        f"ONE HISTORY. Nothing here is a statement about any period outside {panel.window}. The "
        "windows were fixed by rule before any return was computed, which limits selection inside "
        "this history; it does nothing about the history itself being the one that happened.",
        "IT IS ARITHMETIC, AND ONLY ARITHMETIC. Every figure is a past return of the basket as "
        "pasted. Nothing here is a recommendation, a rating, or a statement about any future period.",
    ]
    return tuple(items)


# ---------------------------------------------------------------------------
# The product call
# ---------------------------------------------------------------------------


def audit(
    holdings_text: str,
    *,
    benchmark: str | None = None,
    period: str = DEFAULT_PERIOD,
    rolling_lengths: Sequence[str] = DEFAULT_ROLLING_LENGTHS,
    rolling_step: int = DEFAULT_ROLLING_STEP,
    multi_window_method: str = "calendar_year",
    fetcher: Any = None,
) -> Audit:
    """Parse a pasted book, measure it against one benchmark, explain the gap.

    WHAT IT COMPUTES
        Nothing, directly. It calls, in order:
        ``holdings.parse_holdings`` and ``holdings.fetch_prices`` for the
        panel; ``benchmark.choose_benchmarks`` / ``fetch_benchmark`` /
        ``picking_cost`` / ``rolling_gap`` for the headline and its record;
        ``attribution.holding_attribution`` / ``decision_summary`` /
        ``multi_window_attribution`` / ``survivorship_note`` for the
        per-holding block; ``concentration``, ``overlap`` and ``beliefs`` for
        the structural block. It then arranges the answers.

    WHERE IT MISLEADS
        See the module docstring, all of it, and each composed module's own.
        The three that most change a reader's conclusion: today's weights are
        applied backwards through every window, overlapping windows are not
        independent observations, and the benchmark choice can flip the sign of
        the headline.

    Parameters
    ----------
    benchmark
        Ticker to compare against. ``None`` applies ``choose_benchmarks``,
        which is driven by the book's own currencies, and the reason is printed.
    rolling_lengths
        Window lengths for the anti-cherry-pick record. A length the history
        cannot cover is reported as not covered, not dropped.
    rolling_step
        Trading days between consecutive rolling windows.
    multi_window_method
        Passed to ``attribution.multi_window_attribution``: ``"calendar_year"``,
        ``"calendar_half"`` or ``"rolling"``. Calendar boundaries were chosen
        for no economic reason, which is exactly why they cannot be tuned.
    fetcher
        Injection point passed to every fetch, so a test never touches network.

    Returns
    -------
    Audit
        Never raises for an unusable book or a missing sibling. Every block
        that could not be computed carries the reason, because a caller has to
        render that either way.
    """
    parsed = parse_holdings(holdings_text)
    panel = fetch_prices(parsed, period=period, fetcher=fetcher)
    notes: list[str] = []

    if not panel.usable:
        reason = f"the price panel is unusable — {panel.unusable_reason}"
        return _blank(parsed, panel, "", "", reason, notes)

    # --- benchmark choice, which is a rule and is printed as one ----------
    try:
        from agent.portfolio import benchmark as bm  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        bm = None  # type: ignore[assignment]
        bench_reason = f"agent.portfolio.benchmark — {_why(exc)}"
    else:
        bench_reason = ""

    bench_ticker, bench_why = (benchmark or ""), ("named by the caller" if benchmark else "")
    other: tuple[tuple[str, str], ...] = ()
    mark = None
    if bm is not None:
        try:
            picks = bm.choose_benchmarks(panel)
        except Exception as exc:  # noqa: BLE001
            picks = []
            notes.append(f"choose_benchmarks — {_why(exc)}")
        if not benchmark:
            if picks:
                bench_ticker, bench_why = picks[0]
                other = tuple(picks[1:])
            else:
                bench_reason = (
                    f"no default benchmark exists for a {panel.base_currency} book, and inventing a "
                    "proxy would be less honest than reporting none. Name one with benchmark=."
                )
        else:
            other = tuple(p for p in picks if p[0] != bench_ticker)
        if bench_ticker and not bench_reason:
            try:
                mark = bm.fetch_benchmark(panel, bench_ticker, fetcher=fetcher)
            except Exception as exc:  # noqa: BLE001
                bench_reason = f"fetch_benchmark({bench_ticker!r}) — {_why(exc)}"

    structure = (
        _section_concentration(panel),
        _section_overlap(panel),
        _section_worst_days(panel),
    )

    if mark is None:
        return _blank(
            parsed, panel, bench_ticker, bench_why, bench_reason or "no benchmark was fetched", notes, structure, other
        )

    for w in getattr(mark, "warnings", ()):
        notes.append(w)
    if getattr(mark, "fx_note", ""):
        notes.append(mark.fx_note)
    # ``Benchmark.warnings`` already carries the "you hold the benchmark" note when
    # it applies; adding our own would print the same fact twice, and a reader who
    # sees the same caveat twice starts skipping caveats.

    # --- 1. the number, and its record ------------------------------------
    picking, picking_reason = None, ""
    try:
        picking = bm.picking_cost(panel, benchmark=mark, include_rolling=False, fetcher=fetcher)
    except Exception as exc:  # noqa: BLE001
        picking_reason = f"picking_cost — {_why(exc)}"

    records: list[Any] = []
    record_problems: list[str] = []
    for length in rolling_lengths:
        try:
            gap = bm.rolling_gap(panel, benchmark=mark, length=length, step=rolling_step, fetcher=fetcher)
        except Exception as exc:  # noqa: BLE001
            record_problems.append(f"{length} — {_why(exc)}")
            continue
        if gap.usable and gap.n_windows > 0:
            records.append(gap)
        else:
            record_problems.append(f"{length} — {gap.unusable_reason or 'no usable window'}")

    # --- 2. which decisions -----------------------------------------------
    attribution, summary, attribution_reason = None, {}, ""
    multi, multi_reason = None, ""
    try:
        from agent.portfolio import attribution as at  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        attribution_reason = multi_reason = f"agent.portfolio.attribution — {_why(exc)}"
        survivorship = ""
    else:
        bench_panel = _benchmark_panel(at, mark, panel, period, fetcher)
        try:
            attribution = at.holding_attribution(panel, panel.weights, bench_panel)
            summary = at.decision_summary(attribution)
        except Exception as exc:  # noqa: BLE001
            attribution, attribution_reason = None, f"holding_attribution — {_why(exc)}"
        try:
            multi = at.multi_window_attribution(panel, panel.weights, bench_panel, method=multi_window_method)
        except Exception as exc:  # noqa: BLE001
            multi_reason = f"multi_window_attribution({multi_window_method!r}) — {_why(exc)}"
        try:
            survivorship = at.survivorship_note(panel)
        except Exception as exc:  # noqa: BLE001
            survivorship = f"survivorship note not computed — {_why(exc)}"

    return Audit(
        holdings=parsed,
        panel=panel,
        benchmark_ticker=mark.ticker,
        benchmark_why=bench_why or getattr(mark, "why", ""),
        other_benchmarks=other,
        picking=picking,
        picking_reason=picking_reason,
        records=tuple(records),
        records_reason="; ".join(record_problems),
        attribution=attribution,
        summary=summary,
        attribution_reason=attribution_reason,
        multi=multi,
        multi_reason=multi_reason,
        structure=structure,
        survivorship=survivorship,
        limitations=_limitations(panel, mark.ticker, other),
        notes=tuple(notes),
    )


def _benchmark_panel(at: Any, mark: Any, panel: PricePanel, period: str, fetcher: Any) -> Any:
    """The benchmark as ``attribution`` wants it, reusing the already-fetched series.

    ``benchmark.Benchmark.native`` is the benchmark on its OWN trading days in
    the book's base currency — the series that module deliberately keeps
    un-forward-filled for exactly this kind of reuse. Handing it over avoids a
    second network fetch, which on a public demo is the fetch that gets
    rate-limited. If it is not usable, we fall back to ``attribution``'s own
    fetch rather than pass something half-right.
    """
    native = getattr(mark, "native", None)
    if isinstance(native, pd.Series) and len(native) > 2:
        return (mark.ticker, native.pct_change().dropna())
    return at.fetch_benchmark(mark.ticker, period=period, base_currency=panel.base_currency, fetcher=fetcher)


def _blank(
    parsed: Holdings,
    panel: PricePanel,
    ticker: str,
    why: str,
    reason: str,
    notes: list[str],
    structure: tuple[Section, ...] = (),
    other: tuple[tuple[str, str], ...] = (),
) -> Audit:
    """An ``Audit`` in which the comparison could not be made, and says why."""
    return Audit(
        holdings=parsed,
        panel=panel,
        benchmark_ticker=ticker,
        benchmark_why=why,
        other_benchmarks=other,
        picking=None,
        picking_reason=reason,
        records=(),
        records_reason=reason,
        attribution=None,
        summary={},
        attribution_reason=reason,
        multi=None,
        multi_reason=reason,
        structure=structure,
        survivorship="",
        limitations=_limitations(panel, ticker or "no benchmark", other),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct(x: float | None, width: int = 9, places: int = 1) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return f"{'--':>{width}}"
    return f"{float(x) * 100:{width}.{places}f}%"


def _pts(x: float | None, places: int = 1) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "--"
    return f"{float(x) * 100:+.{places}f} pts"


def _ratio(x: float | None) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return f"{'--':>6}"
    return f"{float(x):6.2f}"


def _wrap(text: str, width: int = 74) -> list[str]:
    out: list[str] = []
    cur = ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def _render_record(rec: Any) -> list[str]:
    """One rolling-window record: the tally, the honest denominator, the series.

    The series is printed. F-18 in this repo's log is a number that was written
    up as a result and was in fact a constant across every run; the only
    defence that works is showing the values, so a record whose gap never moves
    says so in capitals instead of quoting a median.
    """
    L: list[str] = ["", f"  Every rolling {rec.length_label} window ({rec.n_windows} of them):"]
    gaps = rec.series["gap"].to_numpy(dtype=float)
    if rec.n_windows < MIN_WINDOWS_FOR_TALLY:
        L.append(
            f"    This history holds {rec.n_windows} complete {rec.length_label} window(s). Below "
            f"{MIN_WINDOWS_FOR_TALLY} no fraction or median is quoted from them: a tally of one is "
            "one number with a percentage sign on it. The window(s) themselves:"
        )
        for r in rec.series.itertuples():
            L.append(
                f"    {r.start:%Y-%m-%d}..{r.Index:%Y-%m-%d}  book {_pct(r.port_total, 7)}  "
                f"benchmark {_pct(r.bench_total, 7)}  gap {_pts(r.gap)}"
            )
        return L
    if len(gaps) > 1 and float(gaps.max() - gaps.min()) <= 1e-12:
        L.append(
            f"    THE GAP IS IDENTICAL IN ALL {rec.n_windows} WINDOWS TO MACHINE PRECISION. "
            "That is not a result about a portfolio, it is a frozen computation. No median is quoted."
        )
        return L
    L.append(
        f"    Ahead in {rec.beat_count} of {rec.n_windows} windows ({rec.beat_fraction:.0%}), "
        f"behind in {rec.n_windows - rec.beat_count}."
    )
    L.append(
        f"    Median window {_pts(rec.median_gap)}.  Widest ahead {_pts(rec.best_gap)} "
        f"({rec.best_window}).  Widest behind {_pts(rec.worst_gap)} ({rec.worst_window})."
    )
    if rec.beat_count and rec.beat_count < rec.n_windows:
        L.append("    The sign of the gap is not the same in every window: both directions occur in this history.")
    else:
        L.append(f"    The sign of the gap is the same in all {rec.n_windows} windows of this length in this history.")
    for c in rec.caveats:
        L += [f"    {line}" for line in _wrap(c, 72)]
    L.append("")
    L.append(f"    {'window ending':<14}{'book':>10}{'benchmark':>11}{'gap':>13}")
    rows = list(rec.series.itertuples())
    stride = max(1, math.ceil(len(rows) / MAX_SERIES_ROWS))
    shown = rows[::stride]
    if rows and shown[-1] is not rows[-1]:
        shown.append(rows[-1])
    for r in shown:
        L.append(f"    {r.Index:%Y-%m-%d}  {_pct(r.port_total, 8)}{_pct(r.bench_total, 11)}{_pts(r.gap):>13}")
    if stride > 1:
        L.append(
            f"    ({len(shown)} of {len(rows)} windows shown, every {stride}th — the tally above is "
            f"computed on all {len(rows)}.)"
        )
    return L


def _render_named_windows(rows: Sequence[Any], title: str) -> list[str]:
    L: list[str] = ["", f"  {title}", ""]
    L.append(f"    {'window':<40}{'book':>9}{'benchmark':>11}{'gap':>13}{'n':>7}")
    for r in rows:
        if not r.usable:
            L.append(f"    {r.window.label:<40}not computed — {'; '.join(r.notes) or 'unusable window'}")
            continue
        flag = "" if r.window.complete else "   part period"
        L.append(
            f"    {r.window.label:<40}{_pct(r.port_total, 9)}{_pct(r.bench_total, 11)}"
            f"{_pts(r.gap_total):>13}{r.n_obs:>7}{flag}"
        )
    return L


def render_text(audit_result: Audit) -> str:
    """The demo output: the number with its record, the decisions, the caveats.

    WHERE THIS MISLEADS
        It is a string, and a reader can quote one line of it. The headline
        table is therefore not emitted at all unless a multi-window record was
        computed to sit beneath it — see ``Audit.headline_publishable``. That
        stops this function producing the cherry-pick; it cannot stop a reader
        performing one with scissors.
    """
    a = audit_result
    bench = a.benchmark_ticker or "a benchmark"
    L: list[str] = []

    L.append("=" * 78)
    L.append("WHAT YOUR BOOK DID, AGAINST THE ALTERNATIVE OF HAVING DONE NOTHING")
    L.append("=" * 78)
    L.append("")
    L.append(f"Read       : {len(a.holdings.entries)} position(s) from the paste, {len(a.panel.tickers)} priced.")
    L.append(f"Window     : {a.panel.window}, everything in {a.panel.base_currency}.")
    if a.benchmark_ticker:
        L.append(f"Compared to: {bench}")
        for line in _wrap(a.benchmark_why, 64):
            L.append(f"             {line}")
    for t, w in a.other_benchmarks:
        L.append(f"             {t} also applies to this book by the same rule and is not rendered here ({w}).")
    for note in a.notes:
        for i, line in enumerate(_wrap(note, 64)):
            L.append(f"{'Note       : ' if i == 0 else '             '}{line}")
    if a.panel.excluded:
        L.append("")
        L.append("Not measured (nothing is dropped silently):")
        for e in a.panel.excluded:
            L.append(f"  {e.symbol}: {e.reason}")
    for line in a.holdings.assumptions:
        L.append(f"Assumed    : {line}")

    # --- 1 -----------------------------------------------------------------
    L.append("")
    L.append("-" * 78)
    L.append("1. THE NUMBER")
    L.append("-" * 78)
    fw = a.full_window
    if fw is None:
        L.append(f"  Not computed — {a.picking_reason or 'no full-history window was produced'}")
    elif not a.headline_publishable:
        L.append("  WITHHELD. The full-history figures were computed, but no second window was:")
        L.append(f"    {a.records_reason or a.multi_reason or 'no multi-window check was available'}")
        for line in _wrap(
            "A single window is a selected window, and a number with nothing to check it against is "
            "the exact failure this report exists to catch. It is not shown here.",
            72,
        ):
            L.append(f"    {line}")
    else:
        L.append("")
        L.append(f"  {'':<20}{'total':>9}{'per year':>10}{'vol/yr':>9}{'return per vol':>16}")
        L.append(
            f"  {'your book':<20}{_pct(fw.port_total, 9)}{_pct(fw.port_annual, 10)}"
            f"{_pct(fw.port_vol, 9)}{_ratio(fw.port_return_per_vol):>16}"
        )
        L.append(
            f"  {bench + ' alone':<20}{_pct(fw.bench_total, 9)}{_pct(fw.bench_annual, 10)}"
            f"{_pct(fw.bench_vol, 9)}{_ratio(fw.bench_return_per_vol):>16}"
        )
        L.append("")
        L.append(
            f"  Over {fw.window.start:%Y-%m-%d} to {fw.window.end:%Y-%m-%d} "
            f"({fw.n_obs} aligned daily returns, {fw.years:.2f} years): "
            f"{_pts(fw.gap_total)} total, {_pts(fw.gap_annual)} a year."
        )
        L.append(
            "  'Return per vol' is annualised return divided by annualised volatility with no "
            "risk-free rate subtracted. It is not a Sharpe ratio."
        )
        if math.isfinite(getattr(fw, "constant_mix_total", float("nan"))):
            L.append(
                f"  Held at these weights and returned to them every close instead of left alone: "
                f"{_pct(fw.constant_mix_total, 0)} total. The figure above leaves the book alone."
            )
        for n in fw.notes:
            for line in _wrap(n, 72):
                L.append(f"  {line}")

        bc = a.basis_check
        if bc is not None:
            L.append("")
            if bc["signs_agree"]:
                L.append("  THAT ROW ANSWERS ONE OF TWO QUESTIONS, AND THIS REPORT ASKS BOTH.")
            else:
                L.append("  THE TWO QUESTIONS THIS CAN MEAN GIVE OPPOSITE ANSWERS ON THIS BOOK.")
            for line in _wrap(
                f"Above: a book that STARTED at the shape you pasted and was never traded again — "
                f"{_pct(bc['allocation_book_total'], 0)} total, {_pts(bc['allocation_gap'])} against "
                f"{bench}. Section 2 asks a different question: the share counts those same weights "
                f"imply, held for the whole window — {_pct(bc['shares_book_total'], 0)} total, "
                f"{_pts(bc['shares_gap'])} against {bench}.",
                74,
            ):
                L.append(f"  {line}")
            for line in _wrap(
                "The difference is which day the weights belong to. Handing a position that ran up "
                "its FINAL weight from day one credits the book with money it did not have in that "
                "position at the time, so the first reading runs high on a book that had one large "
                "winner. Neither figure is what was actually earned: that needs a transaction "
                "history, and a holdings list does not contain one.",
                74,
            ):
                L.append(f"  {line}")
            if not bc["signs_agree"]:
                for line in _wrap(
                    f"On this book the two readings do not even agree on direction: "
                    f"{_pts(bc['allocation_gap'])} on the first, {_pts(bc['shares_gap'])} on the "
                    "second. Both are printed here for that reason. One of them on its own would be "
                    "a choice this report has no basis for making.",
                    74,
                ):
                    L.append(f"  {line}")
        elif a.attribution is not None or a.attribution_reason:
            L.append("")
            L.append(
                "  Only one weighting basis was computed, so the two readings could not be "
                f"cross-checked: {a.attribution_reason or 'the per-holding block is unusable'}."
            )

        L.append("")
        L.append("  AND HERE IS THAT NUMBER IN EVERY OTHER WINDOW.")
        L.append(
            "  Every window below reads the pasted weights as a starting allocation, the first of "
            "the two readings. The second reading has its own window-by-window check, in section 2."
        )

        if a.picking is not None:
            named = [r for r in a.picking.named(a.benchmark_ticker)]
            if named:
                L.extend(_render_named_windows(named, "Windows fixed by rule before any return was computed:"))
            ex = [r for r in a.picking.ex_benchmark.get(a.benchmark_ticker, ()) if r.window.kind != "rolling"]
            if ex:
                L.extend(
                    _render_named_windows(
                        ex,
                        f"The same windows with the book's own {a.benchmark_ticker} position removed and the "
                        f"rest reweighted — because holding the benchmark drags the gap toward zero:",
                    )
                )
            for skipped in a.picking.skipped_windows:
                L.append(f"    skipped — {skipped}")
        for rec in a.usable_records:
            L.extend(_render_record(rec))
        if a.records_reason:
            L.append("")
            L.append(f"  Rolling lengths not covered by this history: {a.records_reason}")

    # --- 2 -----------------------------------------------------------------
    L.append("")
    L.append("-" * 78)
    L.append("2. WHICH DECISIONS")
    L.append("-" * 78)
    if a.attribution is None:
        L.append(f"  Not computed — {a.attribution_reason or 'no per-holding figures were produced'}")
    elif not a.attribution.usable:
        L.append(f"  Not computed — {a.attribution.unusable_reason}")
    else:
        attr = a.attribution
        if a.multi is not None:
            L.append("  THE SAME CHECK ON THIS READING: every window, not just the full one.")
            for line in _wrap(a.multi.headline_qualifier, 74):
                L.append(f"  {line}")
            L.append(
                f"  ({len(a.multi.usable_windows)} {a.multi.method} windows measured; "
                f"{a.multi.independent_windows} of them do not overlap. {a.multi.overlap_note})"
            )
            for w in a.multi.windows:
                if w.skipped_reason:
                    L.append(f"    {w.window.label}: not measured — {w.skipped_reason}")
                elif w.usable:
                    L.append(f"    {w.window.label:<22}{_pts(w.gap_total):>12}   ({w.window})")
            L.append("")
        elif a.multi_reason:
            L.append(f"  Window-by-window check on this reading not computed — {a.multi_reason}")
            L.append("")
        L.append(
            f"  Over {attr.window}, against {attr.benchmark_ticker}. Worst first. "
            "Weights here are day one, backed out of the ones you pasted."
        )
        L.append("")
        L.append(
            f"  {'holding':<14}{'weight':>8}{'its return':>12}{'benchmark':>11}"
            f"{'difference':>12}{'moved the gap by':>19}"
        )
        for h in attr:
            L.append(
                f"  {h.ticker:<14}{h.weight_start:>8.1%}{_pct(h.total_return, 12)}"
                f"{_pct(h.benchmark_return, 11)}{_pts(h.excess_return):>12}{_pts(h.contribution):>19}"
            )
        L.append("")
        L.append(f"  Weights shown are day-one weights, derived from the pasted weights ({attr.basis_note}).")
        s = a.summary
        if s.get("statement"):
            for line in _wrap(str(s["statement"]), 74):
                L.append(f"  {line}")
        if s.get("one_position_carried"):
            L.append("")
            for line in _wrap(
                f"{s['best_contributor']} contributed more than every other positive contribution in "
                "this book combined. A count of how many lines trailed is true and is not the whole "
                "of it: removing that one line changes the book's result by "
                f"{_pts((s.get('without_best_gap') or 0) - (s.get('gap_total') or 0))}.",
                74,
            ):
                L.append(f"  {line}")
        if s.get("residual_explanation"):
            L.append(f"  Reconciliation: {s['residual_explanation']}")
        for n in attr.notes:
            for line in _wrap(n, 74):
                L.append(f"  {line}")
        for e in attr.excluded:
            L.append(f"  left out — {e}")

    # --- 3 -----------------------------------------------------------------
    L.append("")
    L.append("-" * 78)
    L.append("3. WHY IT MOVED THAT WAY")
    L.append("-" * 78)
    for line in _wrap(
        "Structure, as explanation for the gap above. On its own it is not a finding: the shape of "
        "a book does not surprise the person who chose the shape, which is why it is here and not "
        "at the top.",
        74,
    ):
        L.append(f"  {line}")
    for sec in a.structure:
        L.append("")
        L.append(f"  {sec.title}")
        if not sec.computed:
            L.append(f"    Not computed — {sec.reason}")
            continue
        L.extend(sec.lines)

    # --- 4 -----------------------------------------------------------------
    L.append("")
    L.append("-" * 78)
    L.append("4. WHAT THIS IS NOT")
    L.append("-" * 78)
    for item in a.limitations:
        L.append("")
        L += [f"  {c}" for c in _wrap(item)]
    if a.survivorship:
        L.append("")
        for para in a.survivorship.split("\n"):
            if not para.strip():
                L.append("")
                continue
            L += [f"  {c}" for c in _wrap(para.strip())]
    L.append("")
    L.append("=" * 78)
    return "\n".join(L)
