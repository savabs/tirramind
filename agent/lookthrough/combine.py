"""Look-through: add up what a user owns directly AND through their index funds.

The whole product in one sentence
--------------------------------
You think you own 5% HDFCBANK. You also own a Nifty 50 fund, and HDFCBANK is
about 11% of that fund. So your real HDFCBANK exposure is 5% + (fund weight x
11%). The method is **addition**. There is no statistic, no window, no p-value.

Public surface
--------------
``look_through(holdings_text) -> LookThrough``
``render_text(lt) -> str``

What it computes
----------------
1. Every pasted line is classified as one of three things:
   * a **direct stock** — contributes its own weight to itself;
   * an **unpackable index fund** — a fund whose index has a published
     constituent list, so its weight is spread over the constituents;
   * an **opaque fund** — an actively managed fund, a debt or gold fund, or
     anything with no constituent list. Its weight stays on its own line,
     named, with the reason we could not see into it. It is never silently
     dropped and never silently redistributed over the names we *can* see,
     which would inflate every one of them.
2. For every underlying company:
   ``actual = direct_weight + sum over funds of (fund_weight x that company's
   weight in that fund's index)``
3. Lines are ranked by **surprise** (``actual - listed``), not by size, because
   a position you never listed showing up at 2.1% is the finding; a position
   that moved 0.1pp is not.

Where the numbers come from
---------------------------
Parsing: ``agent.portfolio.holdings.parse_holdings`` (reused, not reimplemented).
Index weights: ``agent.lookthrough.indices.fetch_index``, which is the **only**
source — this module deliberately holds no second implementation of an index
weight. Two modules computing the same weight by different routes is two
different answers to the same question, and the first time they disagreed the
user would be right to leave. If ``indices`` cannot be imported, the
look-through reports ``not computed`` with that reason.
Remote-fetch caching therefore lives in those two modules, both of which cache
every fetch to disk; this module issues no network request of its own.

WHERE THIS MISLEADS — read this before trusting a number out of here
-------------------------------------------------------------------
* **Index weights are estimated, not published.** NSE publishes index
  *membership* in a free keyless CSV but **not** index *weights*, so
  ``indices.py`` reconstructs them from free-float market cap. It measures its
  own error against a published ETF and returns that measurement as
  ``Index.weight_accuracy_note``; ``render_text`` prints that note in the same
  block as the numbers it qualifies, never as a footnote. At the time of
  writing the sibling reports a mean absolute error of ~0.4pp on NIFTY 50.
* **The arithmetic is exact; the inputs are approximate.** A "+4.3 via
  NIFTYBEES" line is exactly ``fund_weight x constituent_weight`` — nothing is
  inferred or back-solved, and ``Contribution.explain()`` prints the two
  factors. If a user checks that line against a factsheet the multiplication
  will reconcile; the *constituent weight* it was multiplied by may differ from
  the factsheet by the error above.
* **One index per fund, matched by name.** "UTI Nifty 50 Index Fund" is matched
  to NIFTY 50 by its name. A fund tracking an equal-weight, capped, or custom
  variant will be looked through against the plain index and will be wrong. The
  matched index is printed next to every fund so that is catchable.
* **Tracking error and cash are ignored.** A real index fund holds ~99.x% of
  the index plus cash and futures. We treat it as holding exactly the index.
* **Weights are a snapshot.** Index membership changes a few times a year and
  constituent prices change daily; ``indices.py`` caches both, so a number can
  be as stale as its caches allow.
* **Share counts cannot be mixed with funds.** A fund's *unit* count cannot be
  turned into a weight without its NAV, which nothing here fetches. A basket
  sized in share counts that contains a fund is reported ``not computed`` with
  that reason, not guessed.
* **Two number shapes are refused outright.** The shared parser reads
  "Rs 1,20,000" as 1 and "2 lakh" as 2 — silently, and wrong by five orders of
  magnitude. Rather than propagate that, ``look_through`` refuses such a paste
  and says what to change. See ``_unsupported_number_shapes``.
* **Nothing here is advice.** "You hold 9.9% of HDFCBANK" is arithmetic.
  Whether that is too much is a judgement this module does not make and must
  not make: we are not a SEBI-registered Investment Adviser. No string a user
  sees recommends, suggests, forecasts, or implies an action.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from agent.portfolio.holdings import Holdings, parse_holdings

log = logging.getLogger(__name__)

__all__ = [
    "WEIGHT_SUM_TOLERANCE",
    "Contribution",
    "FundLine",
    "IndexWeightProvider",
    "IndexWeights",
    "LookThrough",
    "LookThroughError",
    "PositionLine",
    "look_through",
    "render_text",
    "resolve_line",
]


class LookThroughError(RuntimeError):
    """Raised only when the arithmetic itself is broken (weights do not sum)."""


# ---------------------------------------------------------------------------
# Constants — every threshold that can silently change an answer lives here.
# ---------------------------------------------------------------------------

#: Post-look-through weights must sum to 1.0 to within this. This is pure
#: addition of floats, so anything above float noise is a bug, not drift.
WEIGHT_SUM_TOLERANCE = 1e-9

#: Rows printed in the company table by default. The remainder is counted and
#: described, never silently cut.
DEFAULT_MAX_ROWS = 15

#: Unit words we treat as part of a position's *size*, not its name. Kept
#: local rather than imported from holdings.py, whose equivalents are private:
#: a silent upstream rename must not change which token we read as a quantity.
#: "lakh" and "crore" are deliberately absent — see _SCALE_WORD_RE below.
_SIZE_WORDS = {
    "SHARES",
    "SHARE",
    "SHS",
    "SH",
    "QTY",
    "QUANTITY",
    "UNITS",
    "UNIT",
    "NOS",
    "NO",
    "VALUE",
    "AMOUNT",
    "WORTH",
    "INVESTED",
    "COST",
    "MKTVALUE",
    "PERCENT",
    "PCT",
    "WEIGHT",
    "ALLOCATION",
    "INR",
    "RS",
    "RS.",
    "RUPEES",
    "USD",
    "EUR",
    "GBP",
    "JPY",
}
_CURRENCY_GLYPHS = set("\u20b9$\u20ac\u00a3\u00a5%")

#: Two shapes of number that ``agent.portfolio.holdings.parse_holdings``
#: mis-reads today, verified 2026-10-02 against the installed version:
#:   "Rs 1,20,000"  -> read as 1 share   (Indian lakh digit grouping)
#:   "2 lakh"       -> read as 2 shares  ("lakh" dropped as a stray word)
#: Both are silent and both are off by five or six orders of magnitude, which
#: is exactly the kind of wrong number that loses a user for good. We do not
#: edit holdings.py; we refuse the paste and say what to change.
_INDIAN_GROUPING_RE = re.compile(r"\d{1,2},\d{2},\d{3}\b")
_SCALE_WORD_RE = re.compile(r"\b(lakhs?|lacs?|crores?)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Fund recognition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pattern:
    """One name pattern -> what the line is.

    ``index`` is the index whose constituent list unpacks the fund, or ``None``
    when the pattern identifies something we explicitly cannot see into.
    """

    rx: re.Pattern[str]
    index: str | None
    ticker: str | None = None
    why: str = ""


def _p(src: str, index: str | None, ticker: str | None = None, why: str = "") -> _Pattern:
    return _Pattern(re.compile(src, re.IGNORECASE), index, ticker, why)


#: Order is load-bearing: first match wins, so the specific index must be tried
#: before the general one. "UTI Nifty Next 50" must not match "nifty 50".
_INDEX_PATTERNS: tuple[_Pattern, ...] = (
    _p(r"\bjunior\s*bees\b", "NIFTY NEXT 50", "JUNIORBEES"),
    _p(r"\bbank\s*bees\b", "NIFTY BANK", "BANKBEES"),
    _p(r"\bit\s*bees\b", "NIFTY IT", "ITBEES"),
    _p(r"\bmid\s*150\s*bees\b", "NIFTY MIDCAP 150", "MID150BEES"),
    _p(r"\bnifty\s*bees\b", "NIFTY 50", "NIFTYBEES"),
    _p(r"\bsetfnifbk\b", "NIFTY BANK", "SETFNIFBK"),
    _p(r"\bsetfnif50\b", "NIFTY 50", "SETFNIF50"),
    _p(r"\butiniftetf\b", "NIFTY 50", "UTINIFTETF"),
    _p(r"\bnifty\s*next\s*50\b", "NIFTY NEXT 50"),
    _p(r"\bnifty\s*midcap\s*150\b", "NIFTY MIDCAP 150"),
    _p(r"\bnifty\s*(?:bank|financial\s*services)\b|\bbank\s*nifty\b", "NIFTY BANK"),
    _p(r"\bnifty\s*it\b", "NIFTY IT"),
    _p(r"\bnifty\s*100\b", "NIFTY 100"),
    _p(r"\bnifty\s*500\b", "NIFTY 500"),
    _p(r"\bnifty\s*50\b", "NIFTY 50"),
    _p(
        r"\bnifty\b(?=.*\b(?:index|etf|fund)\b)",
        "NIFTY 50",
        None,
        "the line says 'nifty' and 'index/etf/fund' but no index number, so we "
        "read it as NIFTY 50 — if it tracks a different Nifty index this row is wrong",
    ),
)

#: Things we can name but cannot unpack. The ``why`` is shown to the user and
#: has to be something they could act on.
_OPAQUE_PATTERNS: tuple[_Pattern, ...] = (
    _p(r"\bflexi\s*-?\s*cap\b", None, None, "an actively managed flexi-cap fund"),
    _p(r"\bmulti\s*-?\s*cap\b", None, None, "an actively managed multi-cap fund"),
    _p(r"\bsmall\s*-?\s*cap\b", None, None, "an actively managed small-cap fund"),
    _p(r"\blarge\s*(?:&|and)\s*mid\s*-?\s*cap\b", None, None, "an actively managed large-and-mid-cap fund"),
    _p(r"\bmid\s*-?\s*cap\b", None, None, "an actively managed mid-cap fund"),
    _p(r"\blarge\s*-?\s*cap\b", None, None, "an actively managed large-cap fund"),
    _p(r"\bblue\s*-?\s*chip\b", None, None, "an actively managed bluechip fund"),
    _p(r"\bvalue\s*fund\b|\bcontra\b", None, None, "an actively managed value/contra fund"),
    _p(r"\bfocus(?:ed|sed)\b", None, None, "an actively managed focused fund"),
    _p(r"\belss\b|\btax\s*saver\b", None, None, "an actively managed ELSS fund"),
    _p(r"\bbalanced\s*advantage\b|\bhybrid\b|\bbaf\b", None, None, "a hybrid fund that also holds debt"),
    _p(
        r"\bgilt\b|\bdebt\s*fund\b|\bliquid\s*fund\b|\bcorporate\s*bond\b",
        None,
        None,
        "a debt fund, which holds no equities to look through to",
    ),
    _p(
        r"\bgold\s*(?:etf|bees|fund)\b|\bsgb\b",
        None,
        None,
        "a gold instrument, which holds no equities to look through to",
    ),
    _p(r"\bsmall\s*case\b|\bsmallcase\b", None, None, "a smallcase basket"),
    _p(r"\bmutual\s*fund\b|\bfund\b|\bscheme\b", None, None, "a fund we have no constituent list for"),
)

_OPAQUE_FALLBACK_WHY = (
    "we have no constituent list for it — SEBI requires monthly portfolio "
    "disclosure, so the holdings exist, but this module does not parse them yet"
)


@dataclass(frozen=True)
class LineKind:
    """How one pasted line was classified, and why — before any arithmetic."""

    line_no: int
    raw: str
    label: str
    kind: str  # "direct" | "index_fund" | "opaque_fund"
    index: str | None = None
    note: str = ""

    @property
    def is_fund(self) -> bool:
        return self.kind != "direct"


def _is_unit_word(tok: str) -> bool:
    return tok.strip(".,:").upper() in _SIZE_WORDS


def _has_digit(tok: str) -> bool:
    return any(ch.isdigit() for ch in tok)


def _split_size(text: str) -> tuple[str, str]:
    """Split one line into ``(name, size)``. Nothing is discarded.

    ``parse_holdings`` takes the *first* token as the ticker and the
    biggest-looking number as the quantity, so on "UTI Nifty 50 Index Fund 5%"
    it reads the ticker as "UTI" and the quantity as 50 — the 50 in "Nifty 50".
    Multi-word Indian fund names are therefore unusable raw. We find the size
    ourselves and hand ``parse_holdings`` a one-token line.

    The size is a run of tokens at one end of the line containing **at most one
    number** plus any adjacent unit words ("Rs", "shares", "%"). Stopping at
    the first number is what keeps "Nifty 50" and "Midcap 150" intact.

    Misleads: a bare "Nifty 50" with no size attached is ambiguous and this
    reads the 50 as a size. Such a line has no weight, so ``parse_holdings``
    rejects it as mixed-unit and it is reported, not guessed at.
    """
    toks = text.split()
    j, seen_num = len(toks), False
    while j > 0:
        t = toks[j - 1]
        if _is_unit_word(t):
            j -= 1
        elif _has_digit(t) and not seen_num:
            j -= 1
            seen_num = True
        else:
            break
    i, seen_num = 0, False
    while i < j:
        t = toks[i]
        if _is_unit_word(t):
            i += 1
        elif _has_digit(t) and not seen_num:
            i += 1
            seen_num = True
        else:
            break
    name = " ".join(toks[i:j])
    size = " ".join(toks[:i] + toks[j:])
    return name, size


def resolve_line(line_no: int, raw: str) -> tuple[LineKind, str]:
    """Classify one pasted line.

    Returns ``(kind, rewritten_line)``. ``rewritten_line`` is what we hand
    ``parse_holdings``: unchanged for a direct stock, and
    ``"FUNDLINE<n> <size>"`` for a fund, so a multi-word fund name cannot be
    misread as a ticker plus a quantity. ``kind.label`` is the fund name
    exactly as the user typed it, which is what gets printed back to them.

    Misleads: classification is by *name only*. A stock whose name contained
    "fund" would be classified as an opaque fund. The classification is
    printed next to every line, so that is catchable rather than hidden.
    """
    text = " ".join(raw.split()).strip(" ,;\t-\u2022*")
    name, size = _split_size(text)
    for pat in (*_INDEX_PATTERNS, *_OPAQUE_PATTERNS):
        if not pat.rx.search(name):
            continue
        label = name or text
        token = f"FUNDLINE{line_no}"
        if pat.index is not None:
            kind = LineKind(line_no, text, label, "index_fund", pat.index, pat.why)
        else:
            kind = LineKind(line_no, text, label, "opaque_fund", None, pat.why or _OPAQUE_FALLBACK_WHY)
        return kind, f"{token} {size}".strip()
    return LineKind(line_no, text, name or text, "direct"), text


# ---------------------------------------------------------------------------
# Index weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexWeights:
    """Constituent weights for one index, summing to 1.0, plus its caveats.

    ``note`` is not decoration. It carries the measured error of these specific
    weights and :func:`render_text` prints it in the same block as the numbers
    it qualifies.
    """

    index: str
    weights: Mapping[str, float]
    names: Mapping[str, str] = field(default_factory=dict)
    note: str = ""
    excluded: tuple[tuple[str, str], ...] = ()
    asof: str = ""
    #: Measured error of these weights, in percentage points, when the source
    #: measured itself against published holdings. None = unmeasured, which is
    #: different from zero and is rendered as such.
    error_mean_pp: float | None = None
    error_max_pp: float | None = None

    def check(self) -> None:
        total = sum(self.weights.values())
        if self.weights and abs(total - 1.0) > 1e-6:
            raise LookThroughError(f"{self.index}: constituent weights sum to {total!r}, not 1.0")


class IndexWeightProvider(Protocol):
    """What ``look_through`` needs from an index-weight source.

    ``agent/lookthrough/indices.py`` is the source. This module holds **no**
    second implementation of index weights on purpose: two modules computing
    the same weight by different routes is two different answers to the same
    question, and the first time they disagreed the user would be right to
    leave. If ``indices`` cannot be imported, the look-through is reported
    ``not computed`` with that reason rather than guessed at here.
    """

    def index_weights(self, index: str) -> IndexWeights: ...


def _opt_float(value: object) -> float | None:
    """``float(value)`` or None — None means unmeasured, never zero."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class _SiblingProvider:
    """Adapt ``agent.lookthrough.indices`` to :class:`IndexWeightProvider`.

    Prefers an explicit ``index_weights`` if that module ever grows one, and
    otherwise uses ``fetch_index(name) -> Index``, taking ``Index.weights()``,
    the company names, ``Index.weight_accuracy_note`` (the measured error) and
    ``Index.excluded`` (constituents dropped for missing data, each with its
    reason) straight through.

    Misleads: nothing here re-checks the sibling's arithmetic beyond asserting
    the weights sum to 1.0. The accuracy claim in the rendered output is the
    sibling's own measurement, not an independent one.
    """

    def __init__(self, module: object) -> None:
        self._module = module
        self._explicit = getattr(module, "index_weights", None)
        self._fetch = getattr(module, "fetch_index", None)
        if not callable(self._explicit) and not callable(self._fetch):
            raise LookThroughError(
                "agent.lookthrough.indices exposes neither index_weights() nor "
                "fetch_index(); cannot obtain index weights"
            )

    @property
    def describe(self) -> str:
        which = "index_weights()" if callable(self._explicit) else "fetch_index()"
        return f"agent.lookthrough.indices.{which}"

    def index_weights(self, index: str) -> IndexWeights:
        if callable(self._explicit):
            got = self._explicit(index)
            if isinstance(got, IndexWeights):
                got.check()
                return got
            return self._coerce(index, got)
        idx = self._fetch(index)  # type: ignore[misc]
        acc = getattr(idx, "accuracy", None)
        weights = idx.weights() if callable(getattr(idx, "weights", None)) else idx.weights
        names = {str(c.symbol).upper(): str(getattr(c, "company", c.symbol)) for c in getattr(idx, "constituents", ())}
        iw = IndexWeights(
            index=str(getattr(idx, "name", index)).upper(),
            weights={str(k).upper(): float(v) for k, v in weights.items()},
            names=names,
            note=str(getattr(idx, "weight_accuracy_note", "") or ""),
            excluded=tuple((str(a), str(b)) for a, b in tuple(getattr(idx, "excluded", ()) or ())),
            asof=str(getattr(idx, "as_of", "") or ""),
            error_mean_pp=_opt_float(getattr(acc, "mean_abs_pp", None)),
            error_max_pp=_opt_float(getattr(acc, "max_abs_pp", None)),
        )
        iw.check()
        return iw

    @staticmethod
    def _coerce(index: str, got: object) -> IndexWeights:
        """Accept a (mapping, note) pair or a bare mapping; refuse anything else."""
        note = ""
        weights: object = got
        if isinstance(got, tuple) and len(got) == 2:
            weights, note = got[0], str(got[1])
        elif hasattr(got, "weights"):
            weights = got.weights
            note = str(getattr(got, "note", "") or "")
        if not isinstance(weights, Mapping):
            raise LookThroughError(
                "agent.lookthrough.indices.index_weights returned "
                f"{type(got).__name__}, which is not an IndexWeights, a "
                "(mapping, note) pair, or a mapping"
            )
        total = sum(float(v) for v in weights.values())
        if total <= 0:
            raise LookThroughError(f"index_weights({index!r}) weights sum to {total}")
        iw = IndexWeights(
            index=str(index).upper(),
            weights={str(k).upper(): float(v) / total for k, v in weights.items()},
            note=note or "weights from agent.lookthrough.indices (no note supplied)",
        )
        iw.check()
        return iw


def _default_provider() -> tuple[IndexWeightProvider | None, str]:
    """The sibling, or ``None`` plus the reason it could not be used."""
    try:
        from agent.lookthrough import indices as sibling
    except ImportError as exc:
        return None, (
            "agent.lookthrough.indices could not be imported "
            f"({type(exc).__name__}: {exc}), and this module holds no second "
            "implementation of index weights"
        )
    try:
        prov = _SiblingProvider(sibling)
    except LookThroughError as exc:
        return None, str(exc)
    return prov, prov.describe


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contribution:
    """One fund's exact contribution to one company's real weight."""

    source: str  # the fund label, as the user typed it
    index: str  # the index it was unpacked against
    fund_weight: float  # the fund's weight in the portfolio
    constituent_weight: float  # the company's weight in that index
    weight: float  # fund_weight * constituent_weight — exactly

    def explain(self) -> str:
        return (
            f"{self.weight:.4%} = {self.fund_weight:.4%} of the book in {self.source} "
            f"x {self.constituent_weight:.4%} of {self.index}"
        )


@dataclass(frozen=True)
class PositionLine:
    """One company (or one opaque fund), listed vs actually held."""

    symbol: str
    name: str
    listed: float | None  # None = the user never wrote this name down
    actual: float
    via: tuple[Contribution, ...] = ()
    kind: str = "company"  # "company" | "opaque_fund"
    note: str = ""

    @property
    def surprise(self) -> float:
        """How much of ``actual`` the user did not write down."""
        return self.actual - (self.listed or 0.0)


@dataclass(frozen=True)
class FundLine:
    """A pasted line we treated as a fund, and what we did with it."""

    label: str
    weight: float
    index: str | None
    unpacked: bool
    reason: str = ""
    n_constituents: int = 0


@dataclass(frozen=True)
class LookThrough:
    """The whole answer. ``computed=False`` means we refused rather than guessed."""

    lines: tuple[PositionLine, ...] = ()  # ranked by surprise, descending
    n_listed: int = 0
    n_unpacked: int = 0
    n_companies: int = 0
    funds: tuple[FundLine, ...] = ()
    residual: float = 0.0
    weight_basis: str = ""
    weight_source: str = ""
    approximation_notes: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    exclusions: tuple[tuple[str, str], ...] = ()
    computed: bool = True
    not_computed_reason: str = ""

    @property
    def total(self) -> float:
        return sum(ln.actual for ln in self.lines)

    def top_n_share(self, n: int = 3) -> float:
        return sum(sorted((ln.actual for ln in self.lines), reverse=True)[:n])

    def by_symbol(self, symbol: str) -> PositionLine | None:
        want = symbol.strip().upper()
        return next((ln for ln in self.lines if ln.symbol == want), None)


# ---------------------------------------------------------------------------
# Weighting the pasted lines
# ---------------------------------------------------------------------------


def _price_weights_via_holdings(holdings: Holdings) -> tuple[dict[str, float], str]:
    """Share counts and mixed/foreign-currency amounts -> weights, via prices.

    Delegates to the tested ``agent.portfolio.holdings.fetch_prices``, which
    values share counts at the last aligned close and converts foreign-currency
    amounts at a real FX rate.

    Refuses one specific output of that function: its documented fallback of
    "could not value any position; fell back to EQUAL WEIGHT". An equal weight
    nobody asked for is a fabricated number, and this product's whole claim is
    that its numbers are the user's own arithmetic.
    """
    from agent.portfolio.holdings import fetch_prices

    panel = fetch_prices(holdings)
    if not panel.usable or panel.weights.empty:
        raise LookThroughError(panel.unusable_reason or "no usable price panel")
    if "EQUAL WEIGHT" in panel.weight_basis:
        raise LookThroughError(
            "no position could be valued, so the price layer fell back to equal "
            "weight — we will not show you a number you did not give us. "
            "Re-paste as percentages."
        )
    weights = {str(k).split(".")[0].upper(): float(v) for k, v in panel.weights.items()}
    return weights, panel.weight_basis


def _line_weights(
    holdings: Holdings,
    kinds: Mapping[str, LineKind],
    price_weights: Callable[[Holdings], tuple[dict[str, float], str]],
) -> tuple[dict[str, float], str, list[str]]:
    """Turn parsed entries into weights summing to 1.0.

    Percent and single-currency amount are pure arithmetic and keep every line.
    Share counts and mixed/foreign-currency amounts need prices and FX, so they
    go through ``fetch_prices`` — except that a fund has no price, so a basket
    needing prices that also contains a fund is refused rather than guessed.

    Misleads: in the percent and amount paths the user's own numbers are simply
    renormalised, so a typo in them is carried straight through. The
    renormalisation divisor is reported whenever it is not 100.
    """
    notes: list[str] = []
    entries = holdings.entries
    if not entries:
        raise LookThroughError(
            "no readable positions in the pasted text"
            + (
                " — every line was rejected: "
                + "; ".join(f"line {p.line_no} ({p.reason})" for p in holdings.unreadable)
                if holdings.unreadable
                else ""
            )
        )
    mode = holdings.unit_mode

    if mode.startswith("percent"):
        total = sum(e.quantity for e in entries)
        if total <= 0:
            raise LookThroughError("the percentages in the paste sum to zero")
        if abs(total - 100.0) > 0.05:
            notes.append(
                f"your percentages sum to {total:.2f}, not 100 — every number below "
                f"is your figure divided by {total:.2f}"
            )
        basis = f"your stated percentages, normalised from {total:.2f} to 100"
        if "equal weight" in mode:
            basis = "EQUAL WEIGHT, because no line carried a size — this is our assumption, not your portfolio"
        return {e.symbol.upper(): e.quantity / total for e in entries}, basis, notes

    currencies = {(e.currency or "").upper() for e in entries if e.currency}
    if mode == "amount" and len(currencies) <= 1:
        total = sum(e.quantity for e in entries)
        if total <= 0:
            raise LookThroughError("the amounts in the paste sum to zero")
        cur = next(iter(currencies), "") or "the stated currency"
        return (
            {e.symbol.upper(): e.quantity / total for e in entries},
            f"your stated amounts in {cur}, totalling {total:,.0f}",
            notes,
        )

    # Everything else needs a price: share counts, or amounts in more than one
    # currency, or a basket mixing the two.
    fund_names = sorted({k.label for k in kinds.values() if k.is_fund})
    if fund_names:
        raise LookThroughError(
            f"this basket is sized in {mode} and includes a fund "
            f"({', '.join(fund_names)}). Turning that into a weight needs the "
            "fund's NAV or unit price, which nothing here fetches. Re-paste as "
            "percentages, or as amounts in one currency."
        )
    weights, basis = price_weights(holdings)
    if not weights:
        raise LookThroughError(f"no position sized in {mode} could be priced")
    missing = sorted({e.symbol.upper() for e in entries} - {k.upper() for k in weights})
    total = sum(weights.values())
    if total <= 0:
        raise LookThroughError(f"every position sized in {mode} priced to zero")
    if missing:
        notes.append("could not price, so excluded from the weights: " + ", ".join(missing))
    return {s: w / total for s, w in weights.items()}, basis, notes


# ---------------------------------------------------------------------------
# Pre-flight: number shapes we refuse rather than mis-read
# ---------------------------------------------------------------------------


def _unsupported_number_shapes(raw_lines: Sequence[str]) -> list[str]:
    """Name every line whose number ``parse_holdings`` would read wrong.

    Returns one message per offending line, each saying what to change. This is
    a refusal, not a repair: we do not rewrite the user's number, because
    guessing whether "1,20,000" means 120000 or 1.2 is exactly the guess that
    produces a confidently wrong headline.
    """
    out: list[str] = []
    for i, raw in enumerate(raw_lines, start=1):
        if _INDIAN_GROUPING_RE.search(raw):
            out.append(
                f'line {i} "{raw.strip()}" uses Indian digit grouping, which the '
                "shared parser reads as the digits before the first comma (so "
                "1,20,000 becomes 1) — re-paste the number without commas"
            )
        m = _SCALE_WORD_RE.search(raw)
        if m:
            out.append(
                f'line {i} "{raw.strip()}" is sized in {m.group(0)}, which the shared '
                f'parser drops (so "2 {m.group(0)}" becomes 2) — write the number '
                "out in full, or use percentages"
            )
    return out


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------


def look_through(
    holdings_text: str,
    *,
    provider: IndexWeightProvider | None = None,
    price_weights: Callable[[Holdings], tuple[dict[str, float], str]] = _price_weights_via_holdings,
) -> LookThrough:
    """Add up direct holdings and fund holdings into one real weight per company.

    Pass ``provider`` to supply index weights from somewhere else (the sibling
    ``agent.lookthrough.indices`` is picked up automatically when present).

    What it computes: for every company, ``direct + sum over funds of
    fund_weight x constituent_weight``. Post-look-through weights sum to 1.0
    and the check is enforced, not assumed.

    Where it misleads: see the module docstring. The short version is that the
    *arithmetic* is exact and the *index weights* are an approximation whose
    note travels with the numbers.
    """
    if provider is None:
        provider, source_label = _default_provider()
    else:
        source_label = getattr(provider, "describe", type(provider).__name__)

    raw_lines = [ln for ln in holdings_text.splitlines() if ln.strip()]
    n_listed = len(raw_lines)

    bad = _unsupported_number_shapes(raw_lines)
    if bad:
        return LookThrough(
            n_listed=n_listed,
            weight_source=source_label,
            computed=False,
            not_computed_reason=(
                "we will not put a number on this paste because one of its numbers "
                "would be read wrong by five or six orders of magnitude: " + "; ".join(bad)
            ),
        )

    kinds: dict[str, LineKind] = {}
    rewritten: list[str] = []
    assumptions: list[str] = []
    for i, raw in enumerate(raw_lines, start=1):
        kind, new_line = resolve_line(i, raw)
        rewritten.append(new_line)
        if kind.is_fund:
            token = new_line.split()[0]
            kinds[token] = kind
            what = kind.index or "no constituent list"
            assumptions.append(f'line {i}: read "{kind.raw}" as the fund "{kind.label}" -> {what}')
            if kind.note and kind.kind == "index_fund":
                assumptions.append(f"line {i}: {kind.note}")

    holdings = parse_holdings("\n".join(rewritten))
    assumptions = list(holdings.assumptions) + assumptions
    exclusions: list[tuple[str, str]] = [(f"line {p.line_no}: {p.raw}", p.reason) for p in holdings.unreadable]

    try:
        weights, basis, weight_notes = _line_weights(holdings, kinds, price_weights)
    except LookThroughError as exc:
        return LookThrough(
            n_listed=n_listed,
            funds=tuple(FundLine(k.label, 0.0, k.index, False, "weights not computed") for k in kinds.values()),
            weight_source=source_label,
            assumptions=tuple(assumptions),
            exclusions=tuple(exclusions),
            computed=False,
            not_computed_reason=str(exc),
        )
    assumptions += weight_notes

    direct: dict[str, float] = {}
    names: dict[str, str] = {}
    contribs: dict[str, list[Contribution]] = {}
    fund_lines: list[FundLine] = []
    opaque: list[PositionLine] = []
    notes: list[str] = []
    seen_index_notes: set[str] = set()
    index_error: dict[str, float] = {}
    index_exposure: dict[str, float] = {}

    for symbol, weight in weights.items():
        kind = kinds.get(symbol)
        if kind is None:
            direct[symbol] = direct.get(symbol, 0.0) + weight
            continue
        if kind.kind == "opaque_fund":
            fund_lines.append(FundLine(kind.label, weight, None, False, kind.note))
            opaque.append(
                PositionLine(
                    symbol=kind.label,
                    name=kind.label,
                    listed=weight,
                    actual=weight,
                    kind="opaque_fund",
                    note=kind.note,
                )
            )
            continue
        index = kind.index or ""
        try:
            if provider is None:
                raise LookThroughError(source_label)
            iw = provider.index_weights(index)
            # Re-check here, not only inside the adapter: an injected provider
            # must not be able to put weights that do not sum to 1.0 into the
            # addition. A bad index becomes a reported line, not a crash.
            iw.check()
            if not iw.weights:
                raise LookThroughError(f"{index}: the source returned no constituents")
        except Exception as exc:  # noqa: BLE001 - becomes a reported exclusion
            reason = (
                f"could not fetch {index} weights ({type(exc).__name__}: {exc}); its "
                f"{weight:.1%} is left on its own line and NOT spread over the index"
            )
            fund_lines.append(FundLine(kind.label, weight, index, False, reason))
            opaque.append(
                PositionLine(
                    symbol=kind.label,
                    name=kind.label,
                    listed=weight,
                    actual=weight,
                    kind="opaque_fund",
                    note=reason,
                )
            )
            continue
        if iw.note and iw.note not in seen_index_notes:
            seen_index_notes.add(iw.note)
            notes.append(f"{iw.index}: {iw.note}")
        if iw.error_max_pp is not None:
            index_error[iw.index] = max(index_error.get(iw.index, 0.0), iw.error_max_pp)
        index_exposure[iw.index] = index_exposure.get(iw.index, 0.0) + weight
        for csym, cw in iw.weights.items():
            contribs.setdefault(csym, []).append(Contribution(kind.label, iw.index, weight, cw, weight * cw))
            names.setdefault(csym, iw.names.get(csym, csym))
        fund_lines.append(FundLine(kind.label, weight, iw.index, True, "", len(iw.weights)))
        for csym, reason in iw.excluded:
            exclusions.append((f"{iw.index} constituent {csym}", reason))

    # merge contributions that came from two lines of the same fund
    lines: list[PositionLine] = []
    for symbol in sorted(set(direct) | set(contribs)):
        merged: dict[tuple[str, str], Contribution] = {}
        for c in contribs.get(symbol, ()):
            key = (c.source, c.index)
            prev = merged.get(key)
            if prev is None:
                merged[key] = c
            else:
                merged[key] = Contribution(
                    c.source,
                    c.index,
                    prev.fund_weight + c.fund_weight,
                    prev.constituent_weight,
                    prev.weight + c.weight,
                )
        via = tuple(sorted(merged.values(), key=lambda c: -c.weight))
        listed = direct.get(symbol)
        lines.append(
            PositionLine(
                symbol=symbol,
                name=names.get(symbol, symbol),
                listed=listed,
                actual=(listed or 0.0) + sum(c.weight for c in via),
                via=via,
            )
        )
    lines += opaque

    # surprise first, then size; a never-listed name outranks a 0.1pp move
    lines.sort(key=lambda ln: (-ln.surprise, -ln.actual, ln.symbol))

    for idx_name, exposure in sorted(index_exposure.items()):
        err = index_error.get(idx_name)
        if err is None:
            notes.append(
                f"{idx_name}: its source did not measure its own error, so the "
                f"{exposure:.1%} of your money in {idx_name} funds carries an "
                "unquantified error, not a zero one"
            )
            continue
        notes.append(
            f"{idx_name}: {exposure:.1%} of your money sits in {idx_name} funds, so "
            f"that worst-case {err:.2f}pp error in a constituent's index weight moves "
            f"your figure for that company by up to {exposure * err:.2f}pp "
            f"({exposure:.4f} x {err:.2f}pp)"
        )

    total = sum(ln.actual for ln in lines)
    residual = total - 1.0
    if abs(residual) > WEIGHT_SUM_TOLERANCE:
        raise LookThroughError(
            f"post-look-through weights sum to {total!r} (residual {residual:.3e}), "
            f"not 1.0 within {WEIGHT_SUM_TOLERANCE:g} — the addition is wrong, "
            "refusing to show a number"
        )

    return LookThrough(
        lines=tuple(lines),
        n_listed=n_listed,
        n_unpacked=sum(1 for f in fund_lines if f.unpacked),
        n_companies=sum(1 for ln in lines if ln.kind == "company"),
        funds=tuple(fund_lines),
        residual=residual,
        weight_basis=basis,
        weight_source=source_label,
        approximation_notes=tuple(notes),
        assumptions=tuple(assumptions),
        exclusions=tuple(exclusions),
    )


# ---------------------------------------------------------------------------
# Rendering — this text IS the product
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _via_text(ln: PositionLine) -> str:
    """The "+X via Y" clause. Exact: each X is ``fund_weight x index_weight``."""
    if ln.kind == "opaque_fund":
        return "(one line, not looked through)"
    if not ln.via:
        return "(held directly only)"
    if ln.listed is None:
        if len(ln.via) == 1:
            return f"(all via {ln.via[0].source})"
        inner = ", ".join(f"{c.weight * 100:.1f} via {c.source}" for c in ln.via)
        return f"(all of it: {inner})"
    inner = ", ".join(f"+{c.weight * 100:.1f} via {c.source}" for c in ln.via)
    return f"({inner})"


def render_text(lt: LookThrough, *, max_rows: int = DEFAULT_MAX_ROWS) -> str:
    """Render the look-through as plain text a stranger can read.

    Contains no advice: no "over-concentrated", no "consider", no "should".
    Every number that rests on an approximation is printed with the note that
    says so, in the same block, not in a footnote the reader can skip.
    """
    out: list[str] = []
    if not lt.computed:
        out.append(f"Look-through NOT COMPUTED: {lt.not_computed_reason}")
        if lt.assumptions:
            out.append("")
            out.append("How we read your paste")
            out += [f"  - {a}" for a in lt.assumptions]
        if lt.exclusions:
            out.append("")
            out.append("Lines we could not read")
            out += [f"  - {w}: {r}" for w, r in lt.exclusions]
        return "\n".join(out)

    if lt.n_unpacked:
        out.append(
            f"You listed {lt.n_listed} positions. Looking through {lt.n_unpacked} of them, "
            f"you hold {lt.n_companies} companies."
        )
    else:
        out.append(
            f"You listed {lt.n_listed} positions and you hold {lt.n_companies} companies. "
            "There was nothing to look through: no line resolved to an index "
            "whose constituents we could unpack, so what you listed is what you hold."
        )
    out.append("")

    rows = [ln for ln in lt.lines if ln.kind == "company"]
    shown = rows[:max_rows]
    width = max([13] + [len(ln.symbol) for ln in shown])
    for ln in shown:
        listed = "    none" if ln.listed is None else f"{_pct(ln.listed):>8}"
        out.append(
            f"{ln.symbol:<{width}}  you listed {listed}    you actually hold {_pct(ln.actual):>7}   {_via_text(ln)}"
        )
    if len(rows) > len(shown):
        rest = rows[len(shown) :]
        biggest = max(rest, key=lambda ln: ln.actual)
        out.append(
            f"... and {len(rest)} more companies you hold, ranked below these "
            f"because less of each was unlisted. The largest of them is "
            f"{biggest.symbol} at {_pct(biggest.actual)}. All {len(rows)} are in "
            "LookThrough.lines — none was dropped."
        )

    out.append("")
    top3 = lt.top_n_share(3)
    out.append(f"Your 3 largest real positions are {_pct(top3)} of the money.")
    out.append(f"Weights sum to {lt.total:.10f} (residual {lt.residual:.2e}). Basis: {lt.weight_basis}.")

    unpacked = [f for f in lt.funds if f.unpacked]
    if unpacked:
        out.append("")
        out.append("Looked through")
        for f in unpacked:
            out.append(
                f"  {f.label} — {_pct(f.weight)} of your money, spread over the "
                f"{f.n_constituents} constituents of {f.index}"
            )

    blocked = [f for f in lt.funds if not f.unpacked]
    if blocked:
        out.append("")
        out.append(f"{len(blocked)} of the {lt.n_listed} lines could not be looked through")
        for f in blocked:
            out.append(f"  {f.label} — {_pct(f.weight)} of your money — {f.reason}")

    if lt.approximation_notes:
        out.append("")
        out.append("How the fund numbers above were obtained")
        out += [f"  - {n}" for n in lt.approximation_notes]
        out.append(f"  - weight source: {lt.weight_source}")

    if lt.assumptions:
        out.append("")
        out.append("How we read your paste")
        out += [f"  - {a}" for a in lt.assumptions]

    if lt.exclusions:
        out.append("")
        out.append("Excluded, with the reason (nothing is dropped silently)")
        for what, why in lt.exclusions:
            out.append(f"  - {what}: {why}")

    return "\n".join(out)
