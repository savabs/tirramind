"""Map what a user pastes to an index, so that a fund can be unpacked.

What this computes
------------------
``resolve_fund("NIFTYBEES")`` answers one question: *which published index, if
any, does this thing track?* It returns a :class:`FundMatch` naming an index
that :mod:`agent.lookthrough.indices` can actually fetch a constituent list
for, or ``None``.

It never answers "what does this fund hold". An index fund that tracks the
Nifty 50 holds the Nifty 50 — that is knowable from public data. An actively
managed fund does not, and inferring its holdings from its category would be
inventing data. For those, this module returns ``None`` and says why, in a
sentence naming what the user typed.

How the match is made
---------------------
Not by searching for ``"nifty 50"`` in the text. That is the obvious approach
and it is wrong: ``"Motilal Oswal Nifty 500 Momentum 50 Index Fund"`` contains
both ``"nifty 500"`` and ``"nifty 50"``, and it tracks neither — it tracks
NIFTY500 MOMENTUM 50, a 50-stock momentum screen whose constituents are not in
any CSV we hold. A substring matcher unpacks it into the wrong 501 companies
and reports success.

Instead, the scheme name is reduced to an **index phrase** by removing the
fund house, the product-type boilerplate (``Index Fund``, ``ETF``, ``Fund of
Fund``, ``BeES``, plan and option suffixes) and nothing else, and the
*remainder* must equal a known index **exactly**::

    HDFC Nifty 50 Index Fund            -> "nifty 50"                 -> MATCH
    HDFC NIFTY 100 Equal Weight Index Fund -> "nifty 100 equal weight" -> no match
    Motilal Oswal Nifty 500 Momentum 50 Index Fund
                                        -> "nifty 500 momentum 50"    -> no match
    Nippon India ETF Nifty Next 50 Junior BeES -> "nifty next 50"      -> MATCH

Leftover words are the signal that this is a *different* index. The failure
mode of exact matching is a miss, which is reported; the failure mode of
substring matching is a confident wrong answer, which is not.

Measured on the live AMFI scheme master (2026-10-02): of 3,386 distinct scheme
names, 753 sit in an Index-Fund or ETF category. 199 of those reduce to one of
the seven indices we can fetch. The other 554 are named indices we hold no
constituent list for (BSE Sensex, Nifty Smallcap 250, Nifty 200 Momentum 30),
gold and silver ETFs, and debt index funds. All 554 are reported as
un-unpacked with the index named, never silently skipped.

WHERE THIS MISLEADS — read before trusting a number out of here
---------------------------------------------------------------
* **This names the index. It does not weight it.** A match says "this tracks
  NIFTY 50", nothing more. The constituent weights come from
  :mod:`agent.lookthrough.indices`, and that module's weights are *computed*
  from free-float market cap, not published by NSE. Its accuracy note, not
  this one, governs how wrong a final percentage can be.
* **An index fund is not the index.** It holds a cash balance for redemptions,
  it rebalances a day or two after the index does, and it has tracking error.
  Treating 100% of a fund's value as index exposure overstates equity exposure
  by roughly the cash drag — usually well under 1% of the fund, but it is not
  zero, and it is not measured here.
* **ETF ticker to index is read off the listed name, and some listed names are
  opaque.** ``NIFTYBEES`` resolves because Yahoo reports its name as "Nippon
  India ETF Nifty 50 BeES". ``MIDCAPIETF`` does **not** resolve, because its
  listed name is the string "ICICIPRAMC - ICICIM150", which names no index.
  The issuer code strongly suggests Midcap 150; suggestion is not knowledge, so
  it is returned as un-unpacked. See :data:`OPAQUE_TICKERS`.
* **A one-year return correlation cannot be used to repair that.** It was
  tried. Two ETFs that genuinely track NIFTY 100 (``NIF100IETF``,
  ``HDFCNIF100``) correlate only 0.665 with each other on daily returns, while
  ``NIFTYBEES`` and ``BANKBEES`` — different indices — correlate 0.914. Thin
  ETF trading moves the printed price away from NAV by more than the
  difference between two indices. The test does not discriminate and is not
  used.
* **Exact matching has a known false-negative rate, and it is not zero.**
  Measured on the live file: 2 of the 165 schemes that genuinely track one of
  our seven indices are refused, both ELSS wrappers — "360 ONE ELSS Tax Saver
  Nifty 50 Index Fund" and "NAVI ELSS TAX SAVER NIFTY50 INDEX FUND" really do
  track the Nifty 50, but their phrases reduce to "elss tax saver nifty 50",
  which is not the Nifty 50's name. That is ~1.2% of resolvable schemes, and
  the trade is deliberate: a miss is reported to the user as un-unpacked,
  whereas the loose matching that would catch these also unpacks "Nifty 500
  Momentum 50" into the wrong 501 companies.
* **Bare "Nifty" is read as NIFTY 50.** "UTI Nifty" and "Nifty Index Fund"
  carry no index number. In Indian usage an unqualified "Nifty" is the Nifty 50
  and every AMFI scheme whose phrase reduces to bare "nifty" is a Nifty 50
  tracker, but this *is* an assumption and every match made this way says so in
  :attr:`FundMatch.caveats`.
* **Fund-of-funds add a wrapper.** "Nippon India Nifty Next 50 Junior BeES FoF"
  holds an ETF which holds the index. The look-through is still arithmetic, but
  there are two layers of fee and two layers of tracking error, and the FoF's
  own cash balance is invisible here.
* **The AMFI master is a daily NAV file, not a holdings file.** It gives scheme
  names, codes, ISINs, categories and NAVs. It contains no portfolio. Nothing
  in this module reads or infers a single fund holding from it.
* **Membership is current, not historical.** A fund that tracked a different
  index in the past resolves to its index of today.
* **Nothing here is advice.** "This tracks the Nifty 50" is a statement about a
  published mandate. No string in this module recommends, suggests, forecasts
  or implies an action.

Caching
-------
The AMFI file is 1.5 MB and is republished once a day on business days, so it
is cached to disk for 24 hours; a fetch failure is cached for 1 hour so a
public demo cannot hammer AMFI with a retry loop. On a failure with a stale
file on disk, the stale file is used and :attr:`SchemeMaster.staleness_note`
says so — degraded, never silent. Cache root: ``$TIRRA_LOOKTHROUGH_CACHE``,
else ``$TIRRA_PORTFOLIO_CACHE/lookthrough``, else
``~/.cache/tirramind/lookthrough``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "AMFI_NAV_ALL_URL",
    "OPAQUE_TICKERS",
    "SUPPORTED_INDICES",
    "AmfiUnavailable",
    "FundLookup",
    "FundMatch",
    "FundSet",
    "Scheme",
    "SchemeMaster",
    "classify",
    "index_phrase",
    "load_amfi_schemes",
    "load_scheme_master",
    "look_up_fund",
    "resolve_fund",
    "resolve_funds",
    "supported_indices",
]

# ---------------------------------------------------------------------------
# Constants — every threshold or table that can silently change an answer.
# ---------------------------------------------------------------------------

#: AMFI's scheme master: every open scheme with code, ISINs, category and NAV.
#: Free, keyless, no login. Verified 200 OK, 1,519,831 bytes, 14,366 scheme
#: rows on 2026-10-02.
AMFI_NAV_ALL_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"

#: AMFI republishes once per business day. A longer TTL would serve a stale NAV
#: date; a shorter one re-downloads 1.5 MB for nothing.
CACHE_TTL_AMFI = 24 * 3600

#: A failure is cached too, so a public demo cannot retry-loop against AMFI.
CACHE_TTL_FAIL = 3600

#: Failures are cached under their own key so that recording one cannot
#: destroy the last good copy of the file.
_FAIL_KEY = "{key}#fetch-failed"

#: The indices :mod:`agent.lookthrough.indices` can fetch a constituent list
#: for. Keys are spelled exactly as that module's ``fetch_index`` expects, so a
#: :attr:`FundMatch.index_id` can be passed straight to it.
#:
#: ``aliases`` are *post-normalisation* index phrases (see :func:`index_phrase`)
#: and are matched EXACTLY. Adding a loose alias here is the one edit that can
#: make this module unpack the wrong index, so each one is justified:
SUPPORTED_INDICES: dict[str, tuple[str, ...]] = {
    # "nifty" alone: Indian usage for the Nifty 50. Flagged in caveats.
    # "cnx nifty" / "s p cnx nifty": the index's pre-2015 names, still in older
    # scheme names and in what long-term investors type.
    "NIFTY 50": ("nifty 50", "nifty", "cnx nifty", "s p cnx nifty", "nse nifty 50"),
    # "nifty junior" / "cnx nifty junior": the Next 50's former names.
    "NIFTY NEXT 50": ("nifty next 50", "next 50", "nifty junior", "cnx nifty junior"),
    "NIFTY 100": ("nifty 100",),
    "NIFTY 500": ("nifty 500",),
    # "bank nifty" is the universal spoken form; "nifty banking" appears in a
    # few scheme names. NOT bare "bank" — "SBI Banking Fund" is active.
    "NIFTY BANK": ("nifty bank", "bank nifty", "nifty banking"),
    # NOT bare "it" — too short to be anything but a false positive.
    "NIFTY IT": ("nifty it", "nifty information technology"),
    "NIFTY MIDCAP 150": ("nifty midcap 150", "nifty mid cap 150"),
}

#: NSE ETF symbol -> the name under which it is listed, as Yahoo Finance
#: reports it (``Ticker("X.NS").info["longName"]``). Every one was read live on
#: 2026-10-02 and is recorded verbatim. The index is NOT asserted here — it is
#: derived by running the listed name through :func:`index_phrase`, the same
#: path a pasted scheme name takes. That is deliberate: a hand-written
#: ``ticker -> index`` table is an unverifiable claim, whereas this is a quote
#: from the issuer's listed name plus arithmetic a test can re-run.
TICKER_LISTED_NAMES: dict[str, str] = {
    "NIFTYBEES": "Nippon India ETF Nifty 50 BeES",
    "SETFNIF50": "SBI Nifty 50 ETF",
    "NIFTYIETF": "ICICI Prudential Nifty 50 ETF",
    "NIFTY1": "Kotak Nifty 50 ETF",
    "LICNETFN50": "LIC MF Nifty 50 ETF",
    "NIFTYETF": "Mirae Asset Nifty 50 ETF",
    "JUNIORBEES": "Nippon India ETF Nifty Next 50 Junior BeES",
    "SETFNN50": "SBI Nifty Next 50 ETF",
    "HDFCNEXT50": "HDFC Nifty Next 50 ETF",
    "BANKBEES": "Nippon India ETF Nifty Bank BeES",
    "SETFNIFBK": "SBI Nifty Bank ETF",
    "BANKIETF": "ICICI Prudential Nifty Bank ETF",
    "HDFCNIFBAN": "HDFC Nifty Bank ETF - Growth",
    "ITBEES": "Nippon India ETF Nifty IT",
    "ITETF": "Mirae Asset Nifty IT ETF",
    "MID150BEES": "Nippon India ETF Nifty Midcap 150",
    "HDFCMID150": "HDFC NIFTY Midcap 150 ETF",
    "NIF100IETF": "ICICI Prudential Nifty 100 ETF",
    "HDFCNIF100": "HDFC Nifty 100 ETF",
    "LICNFNHGP": "LIC MF Nifty 100 ETF",
}

#: Tickers that exist and trade, whose listed name names no index. Recorded so
#: the refusal can quote the name rather than say "unknown". The issuer codes
#: are suggestive (``ICICIM150`` → Midcap 150) and are deliberately NOT acted
#: on: an inferred mapping that is wrong unpacks a position into 150 wrong
#: companies and reports success.
OPAQUE_TICKERS: dict[str, str] = {
    "HDFCNIFTY": "HDFCAMC - HDFCNIFTY",
    "NEXT50IETF": "ICICIPRAMC - ICICINXT50",
    "ITIETF": "ICICIPRAMC - ICICITECH",
    "MIDCAPIETF": "ICICIPRAMC - ICICIM150",
    "MIDCAPETF": "MIRAEAMC - MAM150ETF",
}

#: NSE tickers for ETFs that hold NO EQUITIES — gold, silver, debt. Listed
#: rather than inferred from the spelling, because inference on glued ticker
#: names is unreliable in both directions and the two failure modes are not
#: symmetric:
#:
#: * ``\bgold\b`` does not match ``GOLDBEES`` (no word boundary before BEES),
#:   so the strict form misses the real ticker and a gold ETF is counted among
#:   the reader's equity holdings — the F-19 failure, silently wrong.
#: * dropping the boundary to ``\bgold`` matches ``GOLDIAM`` — Goldiam
#:   International, a real NSE-listed company — turning a stock into a fund.
#:
#: There is no regex that reliably separates "a ticker that starts with the
#: name of a metal" from "a ticker for a fund that holds that metal", so this
#: is a list. It is short, it is checkable against NSE, and a ticker missing
#: from it fails safe: the free-text patterns in ``combine`` still catch any
#: name containing a separate "gold"/"silver"/"debt" word.
NON_EQUITY_TICKERS: dict[str, str] = {
    "GOLDBEES": "a gold ETF, which holds metal rather than company shares",
    "GOLDSHARE": "a gold ETF, which holds metal rather than company shares",
    "GOLDETF": "a gold ETF, which holds metal rather than company shares",
    "GOLDIETF": "a gold ETF, which holds metal rather than company shares",
    "GOLD1": "a gold ETF, which holds metal rather than company shares",
    "GOLDCASE": "a gold ETF, which holds metal rather than company shares",
    "AXISGOLD": "a gold ETF, which holds metal rather than company shares",
    "HDFCGOLD": "a gold ETF, which holds metal rather than company shares",
    "SETFGOLD": "a gold ETF, which holds metal rather than company shares",
    "IVZINGOLD": "a gold ETF, which holds metal rather than company shares",
    "LICMFGOLD": "a gold ETF, which holds metal rather than company shares",
    "QGOLDHALF": "a gold ETF, which holds metal rather than company shares",
    "SILVERBEES": "a silver ETF, which holds metal rather than company shares",
    "SILVERETF": "a silver ETF, which holds metal rather than company shares",
    "SILVERIETF": "a silver ETF, which holds metal rather than company shares",
    "HDFCSILVER": "a silver ETF, which holds metal rather than company shares",
    "AXISILVER": "a silver ETF, which holds metal rather than company shares",
    "LIQUIDBEES": "a liquid-debt ETF, which holds no company shares",
    "LIQUIDETF": "a liquid-debt ETF, which holds no company shares",
    "LIQUIDCASE": "a liquid-debt ETF, which holds no company shares",
    "GILT5YBEES": "a government-bond ETF, which holds no company shares",
    "EBBETF0430": "a bond ETF (Bharat Bond), which holds no company shares",
    "EBBETF0431": "a bond ETF (Bharat Bond), which holds no company shares",
    "EBBETF0433": "a bond ETF (Bharat Bond), which holds no company shares",
}

#: Fund-house prefixes, longest first at use. Taken from the 55 AMC names in
#: the live AMFI file (with "Mutual Fund" removed) plus former names and
#: sub-brands that appear in scheme names but not in the AMC list ("Parag
#: Parikh" is a PPFAS scheme; "IDFC" became Bandhan; "Reliance" became Nippon).
_HOUSE_PREFIXES: tuple[str, ...] = (
    "360 one",
    "abakkus",
    "aditya birla sun life",
    "absl",
    "alphagrep",
    "angel one",
    "ask",
    "axis",
    "bajaj finserv",
    "bandhan",
    "bank of india",
    "baroda bnp paribas",
    "baroda pioneer",
    "birla sun life",
    "canara robeco",
    "capitalmind",
    "choice",
    "dsp blackrock",
    "dsp",
    "edelweiss",
    "franklin india",
    "franklin templeton",
    "groww",
    "hdfcamc",
    "hdfc",
    "helios",
    "hsbc",
    "icici prudential",
    "icici pru",
    "icicipramc",
    "idfc",
    "il&fs",
    "invesco india",
    "invesco",
    "iti",
    "jio blackrock",
    "jioblackrock",
    "jm financial",
    "kotak mahindra",
    "kotak",
    "lakshya",
    "lic mf",
    "lic",
    "mahindra manulife",
    "miraeamc",
    "mirae asset",
    "monarch",
    "most",
    "motilal oswal",
    "navi",
    "nippon india",
    "nippon life india",
    "nj",
    "old bridge",
    "parag parikh",
    "pgim india",
    "ppfas",
    "principal",
    "quant",
    "quantum",
    "reliance",
    "samco",
    "sbi",
    "shriram",
    "sundaram",
    "tata",
    "taurus",
    "the wealth company",
    "trust",
    "union",
    "unifi",
    "uti",
    "whiteoak capital",
    "zerodha",
)

#: Product-type boilerplate, removed anywhere in the phrase. Every token here
#: is a word about the *wrapper*, never about the index. "Growth" is NOT here
#: (it is part of "NIFTY GROWTH SECTORS 15"); plan and option words are
#: stripped as trailing suffixes only, by :func:`_strip_plan_suffix`.
_BOILERPLATE_PHRASES: tuple[str, ...] = (
    "exchange traded fund",
    "exchange traded scheme",
    "fund of funds",
    "fund of fund",
    "junior bees",
    "total return index",
    "total returns index",
    "index fund",
    "index funds",
    "open ended",
    "open-ended",
    "passive",
    "tracker",
)

_BOILERPLATE_WORDS: frozenset[str] = frozenset(
    {
        "bees",
        "etf",
        "etfs",
        "fof",
        "fund",
        "funds",
        "index",
        "mf",
        "scheme",
        "schemes",
        "tri",
    }
)

#: Trailing plan/option tails, stripped from the end only. AMFI keeps plan and
#: option in their own columns, but a handful of scheme names carry them inline
#: ("SBI NIFTY 1D Rate Liquid ETF - Growth") and a human pastes them constantly.
_PLAN_SUFFIX_RE = re.compile(
    r"(?:[\s\-–—(|,]+)"
    r"(?:direct|regular|growth|idcw|dividend|payout|reinvestment|bonus|"
    r"cumulative|option|plan|g|d|dp|rp)"
    r"[\s\-–—)|,.]*$",
    re.IGNORECASE,
)

#: Trailing quantity tails, stripped from the end only: a user pastes
#: "NIFTYBEES 200 units" or "NIFTYBEES ₹50,000". A bare trailing number is NOT
#: stripped, because "NIFTY 50" ends in one.
_QTY_SUFFIX_RE = re.compile(
    r"[\s,;|]+(?:(?:rs\.?|inr|₹|\$|usd)\s*)?[\d,]+(?:\.\d+)?\s*"
    r"(?:units?|unit|shares?|shs?|qty|nos?|%|pct|percent)?\s*$",
    re.IGNORECASE,
)
#: A currency amount anywhere in the text ("NIFTYBEES Rs 50,000"). The leading
#: ``(?<![a-z0-9])`` is load-bearing: without it, "NIFTY GROWTH SECTORS 15"
#: matches on "RS 15" and reduces to "nifty growth secto", silently corrupting
#: an index name into a different string.
_CURRENCY_LEAD_RE = re.compile(r"(?<![a-z0-9])(?:rs\.?|inr|₹|\$|usd)\s*[\d,]+(?:\.\d+)?", re.IGNORECASE)

#: An AMFI ISIN. Mutual-fund ISINs in India start INF; listed equity is INE.
_ISIN_RE = re.compile(r"^IN[EF][0-9A-Z]{9}$", re.IGNORECASE)

#: A bare exchange ticker: no spaces, short, no lowercase words in it. Used
#: only to decide whether to try the ticker table before the name matcher.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9&]{1,14}$")

#: Name fragments that mark a scheme as passively tracking something. Checked
#: BEFORE the active markers, because "Nifty 500 Value 50 Index Fund" contains
#: "value" and is not an active value fund.
_PASSIVE_MARKERS: tuple[str, ...] = (
    "index fund",
    "index funds",
    "etf",
    "exchange traded",
    "bees",
    " index",
    "index ",
    "passive",
    "tracker",
)

#: Index families whose name alone marks a passive product even with no
#: "index"/"etf" word: a scheme phrase starting with one of these is tracking a
#: published benchmark.
_INDEX_FAMILY_PREFIXES: tuple[str, ...] = (
    "nifty",
    "bse",
    "sensex",
    "s p bse",
    "msci",
    "crisil",
    "crisil ibx",
)

#: SEBI scheme-category words that mark a manager picking holdings. Only
#: consulted when no passive marker was found.
_ACTIVE_MARKERS: tuple[str, ...] = (
    "flexi cap",
    "flexicap",
    "flexi-cap",
    "multi cap",
    "multicap",
    "large cap",
    "largecap",
    "mid cap",
    "midcap fund",
    "small cap",
    "smallcap fund",
    "large & mid",
    "large and mid",
    "focused",
    "focussed",
    "contra",
    "value fund",
    "dividend yield",
    "elss",
    "tax saver",
    "bluechip",
    "blue chip",
    "opportunities",
    "emerging",
    "discovery",
    "prima",
    "aggressive hybrid",
    "conservative hybrid",
    "balanced advantage",
    "dynamic asset allocation",
    "equity savings",
    "multi asset",
    "arbitrage",
    "liquid fund",
    "overnight fund",
    "money market",
    "ultra short",
    "low duration",
    "short duration",
    "medium duration",
    "long duration",
    "corporate bond",
    "credit risk",
    "dynamic bond",
    "gilt fund",
    "banking and psu",
    "banking & psu",
    "retirement",
    "children",
    "childrens",
    "equity fund",
    "income fund",
    "savings fund",
)

#: AMFI category fragments that mark the whole category as passive.
_PASSIVE_CATEGORY_MARKERS: tuple[str, ...] = (
    "index funds",
    "index fund",
    "exchange traded fund",
    "etf",
)

_NON_EQUITY_PHRASES: tuple[str, ...] = (
    "gold",
    "silver",
    "g sec",
    "gsec",
    "g-sec",
    "sdl",
    "gilt",
    "liquid",
    "1d rate",
    "debt",
    "bond",
    "ibx",
    "treasury",
    "money market",
)


class AmfiUnavailable(RuntimeError):
    """AMFI could not be reached and no cached copy of the file exists."""


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scheme:
    """One row of AMFI's scheme master.

    This carries no portfolio. AMFI's NAVAll file publishes identity and price,
    not holdings, so nothing downstream may infer what a scheme owns from a
    :class:`Scheme`.
    """

    code: int
    name: str
    isin_growth: str | None
    isin_reinvest: str | None
    plan: str
    option: str
    nav: float | None
    nav_date: str
    category: str
    amc: str

    @property
    def isins(self) -> tuple[str, ...]:
        return tuple(i for i in (self.isin_growth, self.isin_reinvest) if i)


@dataclass(frozen=True)
class SchemeMaster:
    """The parsed AMFI file, plus an honest account of where it came from."""

    schemes: tuple[Scheme, ...]
    as_of: str
    source: str
    fetched_at: float
    stale: bool
    #: Set when the file's dates are not uniform — e.g. forward-dated rows.
    as_of_note: str | None = None
    unparsed_lines: tuple[str, ...] = ()
    _by_isin: dict[str, Scheme] = field(default_factory=dict, repr=False, compare=False)
    _by_phrase: dict[str, tuple[Scheme, ...]] = field(default_factory=dict, repr=False, compare=False)
    _by_name: dict[str, tuple[Scheme, ...]] = field(default_factory=dict, repr=False, compare=False)

    @property
    def staleness_note(self) -> str | None:
        """Why this data may be out of date, or ``None`` when it is fresh."""
        if not self.stale:
            return None
        age_h = (time.time() - self.fetched_at) / 3600.0
        return (
            f"AMFI could not be reached; using a cached copy downloaded "
            f"{age_h:.1f}h ago with NAVs dated {self.as_of}. A scheme launched "
            f"since then will not be found."
        )

    def by_isin(self, isin: str) -> Scheme | None:
        return self._by_isin.get((isin or "").strip().upper())

    def by_name(self, name: str) -> tuple[Scheme, ...]:
        """Schemes whose name matches ``name`` after whitespace/case folding."""
        return self._by_name.get(_fold(name), ())

    def by_phrase(self, phrase: str) -> tuple[Scheme, ...]:
        """Schemes whose :func:`index_phrase` equals ``phrase`` exactly."""
        return self._by_phrase.get(phrase, ())

    def amc_names(self) -> tuple[str, ...]:
        return tuple(sorted({s.amc for s in self.schemes if s.amc}))


@dataclass(frozen=True)
class FundMatch:
    """A fund we can unpack, and the chain of reasoning that got us there.

    ``index_id`` is spelled exactly as :func:`agent.lookthrough.indices.fetch_index`
    expects it. This object says *which* index; it says nothing about weights.
    """

    typed: str
    index_id: str
    kind: str  # "etf" | "index_fund" | "fund_of_funds" | "passive"
    matched_on: str
    index_phrase: str
    scheme_name: str | None = None
    scheme_code: int | None = None
    isin: str | None = None
    category: str | None = None
    aliased_from: str | None = None
    caveats: tuple[str, ...] = ()

    def describe(self) -> str:  # pragma: no cover - display only
        bits = [f'"{self.typed}" tracks {self.index_id} ({self.matched_on})']
        bits += [f"  caveat: {c}" for c in self.caveats]
        return "\n".join(bits)


@dataclass(frozen=True)
class FundLookup:
    """The full result of one lookup: the match, or the reason there is none.

    ``reason`` is always populated, including on success, because a user who
    pasted "UTI Nifty" deserves to be told we read it as the Nifty 50.
    """

    typed: str
    match: FundMatch | None
    reason: str
    classification: str  # "index" | "active" | "unknown"
    index_phrase: str
    tried: tuple[str, ...] = ()

    @property
    def unpacked(self) -> bool:
        return self.match is not None


@dataclass(frozen=True)
class FundSet:
    """Several lookups at once, with the counts the headline needs.

    The point of this type is that the un-unpacked funds are *counted*, not
    dropped. A headline that says "I looked through your funds" when two of
    four were skipped is the failure this exists to prevent.
    """

    lookups: tuple[FundLookup, ...]
    master_note: str | None = None

    @property
    def matched(self) -> tuple[FundLookup, ...]:
        return tuple(x for x in self.lookups if x.match is not None)

    @property
    def unmatched(self) -> tuple[FundLookup, ...]:
        return tuple(x for x in self.lookups if x.match is None)

    def headline(self) -> str:
        """One sentence stating the coverage. States a count, never a judgement."""
        n, k = len(self.lookups), len(self.matched)
        if n == 0:
            return "No funds were listed."
        fund_word = "fund" if n == 1 else "funds"
        if k == 0:
            return f"Of the {n} {fund_word} you hold, I could look through none."
        if k == n:
            return f"Of the {n} {fund_word} you hold, I could look through {'it' if n == 1 else f'all {n}'}."
        return f"Of the {n} {fund_word} you hold, I could look through {k}."

    def describe(self) -> str:  # pragma: no cover - display only
        out = [self.headline()]
        if self.master_note:
            out.append(f"  ! {self.master_note}")
        for x in self.matched:
            out.append(f"  unpacked  {x.typed} -> {x.match.index_id}  ({x.match.matched_on})")
            out += [f"              caveat: {c}" for c in x.match.caveats]
        for x in self.unmatched:
            out.append(f"  left out  {x.typed} — {x.reason}")
        return "\n".join(out)


def supported_indices() -> tuple[str, ...]:
    """The indices a :class:`FundMatch` can name, in canonical spelling."""
    return tuple(SUPPORTED_INDICES)


# ---------------------------------------------------------------------------
# Disk cache — mandatory: AMFI is 1.5 MB and this sits behind a public demo.
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    root = os.environ.get("TIRRA_LOOKTHROUGH_CACHE")
    if root:
        path = Path(root)
    elif os.environ.get("TIRRA_PORTFOLIO_CACHE"):
        path = Path(os.environ["TIRRA_PORTFOLIO_CACHE"]) / "lookthrough"
    else:
        path = Path.home() / ".cache" / "tirramind" / "lookthrough"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    return _cache_dir() / f"amfi_{digest}.json"


def _cache_read(key: str) -> dict | None:
    """Return the cached blob, or ``None`` when absent or unreadable.

    A blob past its TTL is still returned, with ``expired`` set, so that a live
    fetch failure can fall back to it rather than failing outright.
    """
    try:
        blob = json.loads(_cache_path(key).read_text())
    except (OSError, ValueError):
        return None
    ttl = CACHE_TTL_AMFI if blob.get("ok") else CACHE_TTL_FAIL
    blob["expired"] = (time.time() - float(blob.get("fetched_at", 0))) > ttl
    return blob


def _cache_write(key: str, blob: dict) -> None:
    blob = {**blob, "fetched_at": time.time(), "key": key}
    try:
        _cache_path(key).write_text(json.dumps(blob))
    except OSError as exc:  # pragma: no cover - disk problems must not kill a demo
        log.warning("lookthrough funds cache write failed for %s: %s", key, exc)


def _http_get_text(url: str, *, timeout: float) -> str:
    import httpx  # noqa: PLC0415 — keep network imports out of module import time

    resp = httpx.get(
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; tirramind-lookthrough)"},
    )
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# AMFI scheme master
# ---------------------------------------------------------------------------


def _clean_cell(tok: str) -> str:
    return _nfkc(tok).strip()


def parse_amfi_text(text: str) -> tuple[list[Scheme], list[str]]:
    """Parse AMFI's NAVAll layout into schemes plus the lines we could not read.

    The file is a flat text report, not a CSV: category headings and AMC names
    appear as bare lines between the semicolon-delimited scheme rows, and the
    row layout has changed over the years (6 columns historically, 8 today with
    Plan and Option split out). Both are handled; a row matching neither is
    returned in the second list rather than skipped, so a layout change shows
    up as a count instead of as missing schemes.
    """
    schemes: list[Scheme] = []
    unparsed: list[str] = []
    category = ""
    amc = ""
    header: list[str] | None = None

    for raw in text.splitlines():
        line = _nfkc(raw).rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if ";" not in line:
            # Category headings always carry a parenthesised scheme type;
            # everything else at this level is a fund house.
            if "schemes(" in stripped.lower() or "schemes (" in stripped.lower():
                category = stripped
            else:
                amc = stripped
            continue

        cells = [_clean_cell(c) for c in line.split(";")]
        if not cells[0].isdigit():
            if header is None and cells[0].lower().startswith("scheme code"):
                header = [c.lower() for c in cells]
            else:
                unparsed.append(stripped)
            continue

        if len(cells) >= 8:
            code, isin_g, isin_r, name, plan, option, nav, date = cells[:8]
        elif len(cells) >= 6:
            code, isin_g, isin_r, name, nav, date = cells[:6]
            plan = option = ""
        else:
            unparsed.append(stripped)
            continue

        if not name:
            unparsed.append(stripped)
            continue

        schemes.append(
            Scheme(
                code=int(code),
                name=name,
                isin_growth=_isin_or_none(isin_g),
                isin_reinvest=_isin_or_none(isin_r),
                plan=plan,
                option=option,
                nav=_float_or_none(nav),
                nav_date=date,
                category=category,
                amc=amc,
            )
        )
    return schemes, unparsed


def _nav_as_of(schemes: Sequence[Scheme]) -> tuple[str, str | None]:
    """The date the file is as of, plus a note when later dates also appear.

    Returns ``(as_of, note)``. ``as_of`` is the *modal* NAV date — the one most
    schemes were priced on.

    Two traps live here, both of which produce a confidently wrong date:

    * **Lexicographic max.** AMFI writes ``01-Oct-2026``. ``max()`` over those
      strings answers ``31-Oct-2025`` for a file dated October 2026, because
      "3" sorts above "0". Dates are therefore compared as dates.
    * **Forward-dated rows.** Even compared correctly, the maximum is not the
      file's date: on 2026-10-02 the live file priced 8,122 schemes at
      01-Oct-2026 and 377 at 04-Oct-2026, three days in the future. Taking the
      maximum would report a date that has not happened. The mode is what the
      file is as of; the outliers are reported in the note rather than hidden.
    """
    counts: dict[tuple[int, int, int], int] = {}
    raw_for: dict[tuple[int, int, int], str] = {}
    unparsed = 0
    for s in schemes:
        raw = (s.nav_date or "").strip()
        if not raw:
            continue
        parsed = _parse_amfi_date(raw)
        if parsed is None:
            unparsed += 1
            continue
        counts[parsed] = counts.get(parsed, 0) + 1
        raw_for.setdefault(parsed, raw)
    if not counts:
        return "", ("no NAV date in the file could be parsed" if unparsed else None)

    modal = max(counts, key=lambda k: (counts[k], k))
    later = sum(n for k, n in counts.items() if k > modal)
    bits: list[str] = []
    if later:
        newest = max(counts)
        bits.append(
            f"{later} of {sum(counts.values())} schemes carry a NAV date after "
            f"it (latest {raw_for[newest]}); the file's date is the one most "
            f"schemes share, not the maximum"
        )
    if unparsed:
        bits.append(f"{unparsed} rows had an unreadable NAV date")
    return raw_for[modal], ("; ".join(bits) if bits else None)


_MONTHS = {
    m: i
    for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}


def _parse_amfi_date(raw: str) -> tuple[int, int, int] | None:
    parts = raw.split("-")
    if len(parts) != 3:
        return None
    day, mon, year = parts
    month = _MONTHS.get(mon[:3].lower())
    if month is None or not day.isdigit() or not year.isdigit():
        return None
    return (int(year), month, int(day))


def _isin_or_none(tok: str) -> str | None:
    tok = (tok or "").strip().upper()
    return tok if _ISIN_RE.match(tok) else None


def _float_or_none(tok: str) -> float | None:
    try:
        return float((tok or "").replace(",", ""))
    except ValueError:
        return None


def load_scheme_master(*, refresh: bool = False, timeout: float = 60.0) -> SchemeMaster:
    """Fetch (or read from cache) and index AMFI's scheme master.

    Raises :class:`AmfiUnavailable` only when the network fails AND no cached
    copy exists at all. With a stale copy on disk, it is used and
    :attr:`SchemeMaster.staleness_note` says so — degraded, never silent.
    """
    key = AMFI_NAV_ALL_URL
    blob = None if refresh else _cache_read(key)
    text: str | None = None
    fetched_at = time.time()
    stale = False

    if blob and blob.get("ok") and not blob.get("expired"):
        text = blob["text"]
        fetched_at = float(blob["fetched_at"])
    else:
        try:
            text = _http_get_text(key, timeout=timeout)
            _cache_write(key, {"ok": True, "text": text})
            fetched_at = time.time()
        except Exception as exc:  # noqa: BLE001 - any transport error degrades the same way
            # The failure marker goes on its OWN key. Writing it to `key` would
            # overwrite the last good copy of the file, which is the very thing
            # the fallback below needs — the stale-fallback path then looks
            # correct and never runs.
            _cache_write(_FAIL_KEY.format(key=key), {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            good = blob if (blob and blob.get("ok")) else _cache_read(key)
            if good and good.get("ok") and good.get("text"):
                text = good["text"]
                fetched_at = float(good["fetched_at"])
                stale = True
                log.warning("AMFI fetch failed (%s); using cached copy", exc)
            else:
                raise AmfiUnavailable(
                    f"could not fetch {key} ({type(exc).__name__}: {exc}) and no cached copy exists"
                ) from exc

    schemes, unparsed = parse_amfi_text(text or "")
    if not schemes:
        raise AmfiUnavailable(
            f"{key} returned {len(text or '')} bytes but no scheme rows could be parsed — the file layout has changed"
        )

    by_isin: dict[str, Scheme] = {}
    by_phrase: dict[str, list[Scheme]] = {}
    by_name: dict[str, list[Scheme]] = {}
    for s in schemes:
        for i in s.isins:
            by_isin.setdefault(i, s)
        by_name.setdefault(_fold(s.name), []).append(s)
        ph = index_phrase(s.name)
        if ph:
            by_phrase.setdefault(ph, []).append(s)

    as_of, as_of_note = _nav_as_of(schemes)
    return SchemeMaster(
        schemes=tuple(schemes),
        as_of=as_of,
        source=key,
        as_of_note=as_of_note,
        fetched_at=fetched_at,
        stale=stale,
        unparsed_lines=tuple(unparsed),
        _by_isin=by_isin,
        _by_phrase={k: tuple(v) for k, v in by_phrase.items()},
        _by_name={k: tuple(v) for k, v in by_name.items()},
    )


def load_amfi_schemes(*, refresh: bool = False, timeout: float = 60.0) -> list[Scheme]:
    """Every scheme in AMFI's master file.

    Thin wrapper over :func:`load_scheme_master` for callers that only want the
    rows. Prefer the master when you care whether the data is stale.
    """
    return list(load_scheme_master(refresh=refresh, timeout=timeout).schemes)


# ---------------------------------------------------------------------------
# Normalisation — the whole correctness of this module lives here
# ---------------------------------------------------------------------------


def _nfkc(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    for dash in "‐‑‒–—―−":
        s = s.replace(dash, "-")
    return s.replace("‘", "'").replace("’", "'").replace(" ", " ")


def _fold(s: str) -> str:
    return re.sub(r"\s+", " ", _nfkc(s).strip().lower())


def _strip_plan_suffix(s: str) -> str:
    """Remove trailing plan/option tails, repeatedly ("- Direct Plan - Growth")."""
    prev = None
    while prev != s:
        prev = s
        s = _PLAN_SUFFIX_RE.sub("", s).strip()
    return s


def _strip_qty_suffix(s: str) -> str:
    """Remove a trailing quantity or amount, but never a bare trailing number.

    "NIFTYBEES 200 units" and "NIFTYBEES Rs 50,000" lose their tail.
    "NIFTY 50" and "Nifty 500" keep theirs, which is the whole point.
    """
    s = _CURRENCY_LEAD_RE.sub(" ", s)
    prev = None
    while prev != s:
        prev = s
        cut = _QTY_SUFFIX_RE.sub("", s).strip()
        # Only accept the cut if it removed a unit word or a currency amount;
        # a bare number at the end may be part of the index name.
        if cut != s and re.search(
            r"(?:units?|unit|shares?|shs?|qty|nos?|%|pct|percent|rs\.?|inr|₹|\$|usd)\s*$",
            s,
            re.IGNORECASE,
        ):
            s = cut
    return s.strip()


def _strip_exchange_suffix(s: str) -> str:
    low = s.lower()
    for suf in (".ns", ".bo", ".nse", ".bse", "-eq", ".eq"):
        if low.endswith(suf):
            return s[: -len(suf)]
    return s


def _strip_house(s: str) -> tuple[str, str | None]:
    """Drop a leading fund-house name. Returns (remainder, house or None)."""
    low = s
    for house in sorted(_HOUSE_PREFIXES, key=len, reverse=True):
        if low == house:
            # The whole text is a fund house. Nothing to match on; keep it so
            # the caller reports "that is a fund house, not a scheme".
            return s, None
        for sep in (" ", " - ", "- ", "-", " | "):
            lead = house + sep
            if low.startswith(lead):
                return low[len(lead) :].strip(), house
    return s, None


def _split_glued_digits(s: str) -> str:
    """``nifty50`` -> ``nifty 50``; ``midcap150`` -> ``midcap 150``.

    Only a letter-then-digit boundary is split. A digit-then-letter boundary
    ("1D Rate", "10yr") is left alone, because splitting it invents a token.
    """
    return re.sub(r"(?<=[a-z])(?=\d)", " ", s)


def index_phrase(text: str) -> str:
    """Reduce a scheme name or ticker to the index it names, and nothing else.

    What it computes: lowercase, strip the exchange suffix, strip a trailing
    quantity and a trailing plan/option tail, drop a leading fund-house name,
    delete product-type boilerplate (``Index Fund``, ``ETF``, ``BeES``, ``Fund
    of Fund``), split glued digits, normalise "bank nifty" to "nifty bank", and
    collapse whitespace.

    Where it misleads: this is a *lossy* reduction tuned for Indian scheme
    names, and it is not a parser. "HDFC Nifty G-Sec Dec 2026 Index Fund"
    reduces to "nifty g sec dec 2026" — correctly unmatched, but the phrase
    itself is not a real index name. Only an EXACT hit against
    :data:`SUPPORTED_INDICES` means anything; the phrase is otherwise just a
    string to quote back to the user. It also strips the fund house before
    looking for the index, so a scheme whose index name happens to begin with a
    fund-house word would be mangled — none of the 3,386 live scheme names do.
    """
    s = _fold(text)
    if not s:
        return ""
    s = _strip_exchange_suffix(s)
    s = _strip_qty_suffix(s)
    s = _strip_plan_suffix(s)
    s, _house = _strip_house(s)
    s = _strip_plan_suffix(s)

    for phrase in _BOILERPLATE_PHRASES:
        s = s.replace(phrase, " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = _split_glued_digits(s)
    words = [w for w in s.split() if w and w not in _BOILERPLATE_WORDS]
    s = " ".join(words)
    if s.startswith("bank nifty"):
        s = "nifty bank" + s[len("bank nifty") :]
    return s.strip()


_ALIAS_TO_INDEX: dict[str, str] = {alias: index for index, aliases in SUPPORTED_INDICES.items() for alias in aliases}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(scheme_name: str, *, category: str | None = None) -> str:
    """``"index"``, ``"active"`` or ``"unknown"`` for one scheme.

    What it computes: whether a manager picks this scheme's holdings. ``index``
    means it tracks a published benchmark, so its holdings are knowable from
    public data; ``active`` means a manager chooses them, so they are not.
    ``category`` is AMFI's SEBI category string when you have it, and it is
    authoritative — pass it.

    Where it misleads: ``index`` does NOT mean we can unpack it. A gold ETF, a
    G-Sec index fund and a Nifty Smallcap 250 index fund are all ``index`` here
    and all un-unpackable: two hold no equities and the third has no
    constituent list in this repo. ``index`` also says nothing about whether we
    know the *weights*. Name-only classification is the weak path: it reads
    marketing copy, and a scheme named without any of the words below comes
    back ``unknown`` rather than guessed.
    """
    name = _fold(scheme_name)
    cat = _fold(category or "")

    if cat and any(m in cat for m in _PASSIVE_CATEGORY_MARKERS):
        return "index"

    # The NAME is checked before falling back to the category, because AMFI
    # files an index-tracking fund-of-funds ("Groww Nifty 200 ETF FOF") under
    # "Fund of Funds Scheme (Domestic)", a category that says nothing about
    # whether a manager picks the holdings. Its name does.
    if any(m in f" {name} " for m in _PASSIVE_MARKERS):
        return "index"

    if cat and "(" in cat:
        return "active"

    phrase = index_phrase(scheme_name)
    if phrase and any(phrase == fam or phrase.startswith(fam + " ") for fam in _INDEX_FAMILY_PREFIXES):
        return "index"

    if any(m in name for m in _ACTIVE_MARKERS):
        return "active"
    return "unknown"


def _kind_of(name: str, category: str | None) -> str:
    low = _fold(name) + " " + _fold(category or "")
    if "fund of fund" in low or "fof" in low.split() or "fund of funds" in low:
        return "fund_of_funds"
    if "etf" in low.split() or "exchange traded" in low or "bees" in low:
        return "etf"
    if "index fund" in low or "index" in low:
        return "index_fund"
    return "passive"


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

_CAVEAT_NOT_THE_INDEX = (
    "an index fund is not the index: it holds a cash balance and rebalances a "
    "day or two late, so treating all of it as index exposure overstates the "
    "equity by the cash drag (usually under 1%, not measured here)"
)
_CAVEAT_WEIGHTS_ELSEWHERE = (
    "this names the index only — the constituent weights come from "
    "agent.lookthrough.indices, which computes them from free-float market cap "
    "rather than reading them from NSE"
)
_CAVEAT_BARE_NIFTY = (
    'the text says "nifty" with no index number; read as NIFTY 50, which is '
    "what an unqualified Nifty means in Indian usage — wrong if it tracks a "
    "different Nifty index"
)
_CAVEAT_FOF = (
    "this is a fund of funds: it holds an ETF which holds the index, so there "
    "are two wrappers of fee and tracking error, and the outer fund's own cash "
    "balance is not visible here"
)


def _assumption_suffix(match: FundMatch) -> str:
    """The part of a success message that names an assumption we had to make.

    A match built on an alias rather than the index's own name is still a
    match, but a user who typed "UTI Nifty" must be told we read it as the
    Nifty 50 in the same sentence that reports the result — not only in a
    caveats tuple a caller might not render.
    """
    bits: list[str] = []
    if match.index_phrase in {"nifty", "cnx nifty", "s p cnx nifty"}:
        bits.append('that text carries no index number, and an unqualified "Nifty" is read here as the Nifty 50')
    elif match.aliased_from:
        bits.append(f'matched on the alias "{match.aliased_from}"')
    if match.kind == "fund_of_funds":
        bits.append("it reaches the index through another fund, not directly")
    return (" (" + "; ".join(bits) + ")") if bits else ""


def _match_from_phrase(
    typed: str,
    phrase: str,
    *,
    matched_on: str,
    source_name: str | None,
    scheme: Scheme | None,
    isin: str | None = None,
) -> FundMatch | None:
    index_id = _ALIAS_TO_INDEX.get(phrase)
    if index_id is None:
        return None
    name_for_kind = source_name or typed
    category = scheme.category if scheme else None
    caveats = [_CAVEAT_WEIGHTS_ELSEWHERE, _CAVEAT_NOT_THE_INDEX]
    if phrase in {"nifty", "cnx nifty", "s p cnx nifty"}:
        caveats.insert(0, _CAVEAT_BARE_NIFTY)
    kind = _kind_of(name_for_kind, category)
    if kind == "fund_of_funds":
        caveats.insert(0, _CAVEAT_FOF)
    aliased = None
    canonical_alias = SUPPORTED_INDICES[index_id][0]
    if phrase != canonical_alias:
        aliased = phrase
    return FundMatch(
        typed=typed,
        index_id=index_id,
        kind=kind,
        matched_on=matched_on,
        index_phrase=phrase,
        scheme_name=scheme.name if scheme else source_name,
        scheme_code=scheme.code if scheme else None,
        isin=isin or (scheme.isins[0] if scheme and scheme.isins else None),
        category=category,
        aliased_from=aliased,
        caveats=tuple(caveats),
    )


def _why_not(typed: str, phrase: str, classification: str, name: str | None) -> str:
    """The sentence a user reads when their fund was left un-unpacked.

    States what they typed, what we understood it to be, and what is missing.
    Never recommends anything.
    """
    shown = name or typed
    if not phrase:
        return (
            f'I could not read "{typed}" as a fund or an index at all — nothing '
            f"was left of it after removing the fund house and the words "
            f"index/fund/ETF. It was left un-unpacked."
        )
    if any(p in phrase for p in _NON_EQUITY_PHRASES) and classification == "index":
        return (
            f'"{shown}" tracks "{phrase}", which is not an equity index, so it '
            f"holds no company shares to add to your stock positions. It was "
            f"left un-unpacked."
        )
    if classification == "index":
        return (
            f'"{shown}" is a passive product tracking "{phrase}". I hold '
            f"published constituent lists for only {len(SUPPORTED_INDICES)} "
            f"indices ({', '.join(SUPPORTED_INDICES)}), and that is not one of "
            f"them, so it was left un-unpacked rather than mapped to a nearby "
            f"index."
        )
    if classification == "active":
        return (
            f'"{shown}" is an actively managed fund — a manager picks its '
            f"holdings. SEBI requires it to publish them monthly, but they are "
            f"not knowable from its name, and guessing them from its category "
            f"would be inventing data. It was left un-unpacked."
        )
    return (
        f'I could not tell what "{typed}" is. It reduced to "{phrase}", which '
        f"matches no index I hold a constituent list for and carries none of "
        f"the words that mark a fund (index, ETF, flexi cap, large cap, ...). "
        f"It was left un-unpacked."
    )


def _master_or_none(master: SchemeMaster | None, *, offline: bool) -> tuple[SchemeMaster | None, str | None]:
    if master is not None:
        return master, master.staleness_note
    if offline:
        return None, None
    try:
        m = load_scheme_master()
    except AmfiUnavailable as exc:
        log.warning("AMFI master unavailable: %s", exc)
        return None, (
            f"AMFI's scheme master could not be loaded ({exc}), so funds were "
            f"matched on the text you typed alone — a fund whose name does not "
            f"itself name its index could not be resolved."
        )
    return m, m.staleness_note


def look_up_fund(
    text: str,
    *,
    master: SchemeMaster | None = None,
    offline: bool = False,
) -> FundLookup:
    """Resolve one pasted fund, always with a reason.

    Tries, in order: AMFI ISIN, NSE ETF ticker (via the issuer's listed name),
    an exact AMFI scheme-name hit, then the text's own index phrase. The first
    that yields a SUPPORTED index wins; otherwise the lookup carries the reason
    it did not.

    ``offline=True`` skips the AMFI fetch entirely, which is what tests want.
    """
    typed = _nfkc(text).strip()
    tried: list[str] = []
    m, _note = _master_or_none(master, offline=offline)

    if not typed:
        return FundLookup(
            typed=typed,
            match=None,
            reason="an empty line was passed where a fund name was expected.",
            classification="unknown",
            index_phrase="",
            tried=(),
        )

    # 1. ISIN — unambiguous when it hits.
    bare = _strip_exchange_suffix(typed).strip().upper()
    if _ISIN_RE.match(bare):
        tried.append(f"ISIN lookup in AMFI's scheme master for {bare}")
        scheme = m.by_isin(bare) if m else None
        if scheme:
            phrase = index_phrase(scheme.name)
            match = _match_from_phrase(
                typed,
                phrase,
                matched_on=f'ISIN {bare} is scheme "{scheme.name}" in AMFI',
                source_name=scheme.name,
                scheme=scheme,
                isin=bare,
            )
            cls = classify(scheme.name, category=scheme.category)
            return FundLookup(
                typed=typed,
                match=match,
                reason=(
                    f'ISIN {bare} is "{scheme.name}", which tracks {match.index_id}{_assumption_suffix(match)}.'
                    if match
                    else _why_not(typed, phrase, cls, scheme.name)
                ),
                classification=cls,
                index_phrase=phrase,
                tried=tuple(tried),
            )
        return FundLookup(
            typed=typed,
            match=None,
            reason=(
                f"{bare} looks like an ISIN but is not in AMFI's scheme master"
                + (f" (as of {m.as_of})" if m else " — and the master could not be loaded to check")
                + ". It was left un-unpacked."
            ),
            classification="unknown",
            index_phrase="",
            tried=tuple(tried),
        )

    # 2. A bare exchange ticker: resolve via the name it is LISTED under.
    if _TICKER_RE.match(bare) and " " not in typed:
        if bare in OPAQUE_TICKERS:
            tried.append(f"NSE ticker table for {bare}")
            return FundLookup(
                typed=typed,
                match=None,
                reason=(
                    f"{bare} is a traded ETF, but the name it is listed under is "
                    f'"{OPAQUE_TICKERS[bare]}", which names no index. Its issuer '
                    f"code hints at one; a hint is not a holdings list, so it was "
                    f"left un-unpacked rather than mapped to a guess."
                ),
                classification="index",
                index_phrase="",
                tried=tuple(tried),
            )
        listed = TICKER_LISTED_NAMES.get(bare)
        if listed:
            tried.append(f"NSE ticker table for {bare}")
            phrase = index_phrase(listed)
            match = _match_from_phrase(
                typed,
                phrase,
                matched_on=f'ticker {bare} is listed as "{listed}"',
                source_name=listed,
                scheme=None,
            )
            if match:
                return FundLookup(
                    typed=typed,
                    match=match,
                    reason=(
                        f'{bare} is listed as "{listed}", which tracks {match.index_id}{_assumption_suffix(match)}.'
                    ),
                    classification="index",
                    index_phrase=phrase,
                    tried=tuple(tried),
                )

    # 3. An exact AMFI scheme-name hit gives us AMFI's category, which is a far
    #    better classifier than the name.
    scheme: Scheme | None = None
    if m:
        tried.append("exact scheme-name match in AMFI's scheme master")
        hits = m.by_name(_strip_plan_suffix(_fold(typed))) or m.by_name(_fold(typed))
        if hits:
            scheme = hits[0]

    source_name = scheme.name if scheme else typed
    phrase = index_phrase(source_name)
    if not phrase and not scheme:
        phrase = index_phrase(typed)
    tried.append(f'index phrase "{phrase}"')

    match = _match_from_phrase(
        typed,
        phrase,
        matched_on=(
            f'scheme "{scheme.name}" in AMFI reduces to "{phrase}"'
            if scheme
            else f'the text you typed reduces to "{phrase}"'
        ),
        source_name=source_name,
        scheme=scheme,
    )
    cls = classify(source_name, category=scheme.category if scheme else None)
    if match is None and not scheme and m:
        # The text itself did not name a supported index. Before giving up, see
        # whether exactly one AMFI scheme shares its phrase — that resolves
        # "UTI Nifty" to "UTI Nifty 50 Index Fund" only when it is unambiguous.
        sibling = m.by_phrase(phrase)
        if sibling:
            cls = classify(sibling[0].name, category=sibling[0].category)

    return FundLookup(
        typed=typed,
        match=match,
        reason=(
            f'"{source_name}" tracks {match.index_id}{_assumption_suffix(match)}.'
            if match
            else _why_not(typed, phrase, cls, scheme.name if scheme else None)
        ),
        classification=cls if match is None else "index",
        index_phrase=phrase,
        tried=tuple(tried),
    )


def resolve_fund(
    text: str,
    *,
    master: SchemeMaster | None = None,
    offline: bool = False,
) -> FundMatch | None:
    """The index this fund tracks, or ``None`` when we do not genuinely know it.

    ``None`` is returned for an actively managed fund, for a passive product
    tracking an index we hold no constituent list for, for a gold/silver/debt
    tracker, and for anything unrecognised. Use :func:`look_up_fund` when you
    need the reason — and a user-facing caller always needs the reason, because
    a fund silently dropped is a wrong total.
    """
    return look_up_fund(text, master=master, offline=offline).match


def resolve_funds(
    texts: Iterable[str],
    *,
    master: SchemeMaster | None = None,
    offline: bool = False,
) -> FundSet:
    """Resolve several funds, keeping the un-unpacked ones counted.

    Loads the AMFI master once for the whole batch. The returned
    :class:`FundSet` is what a headline should be built from: it knows both how
    many funds were given and how many were understood.
    """
    items = [_nfkc(t).strip() for t in texts]
    m, note = _master_or_none(master, offline=offline)
    lookups = tuple(look_up_fund(t, master=m, offline=True) for t in items)
    return FundSet(lookups=lookups, master_note=note)


def coverage_report(schemes: Sequence[Scheme]) -> dict[str, object]:
    """Counts over a scheme master: how many schemes, how many we can unpack.

    Used by the verification script and the tests. Returns plain counts, so a
    regression in :func:`index_phrase` shows up as a number that moved.
    """
    names: dict[str, Scheme] = {}
    for s in schemes:
        names.setdefault(s.name, s)
    per_index: dict[str, int] = {k: 0 for k in SUPPORTED_INDICES}
    cls_counts = {"index": 0, "active": 0, "unknown": 0}
    unsupported_phrases: dict[str, int] = {}
    for name, s in names.items():
        cls = classify(name, category=s.category)
        cls_counts[cls] += 1
        idx = _ALIAS_TO_INDEX.get(index_phrase(name))
        if idx and cls == "index":
            per_index[idx] += 1
        elif cls == "index":
            unsupported_phrases[index_phrase(name)] = unsupported_phrases.get(index_phrase(name), 0) + 1
    return {
        "rows": len(schemes),
        "distinct_names": len(names),
        "classification": cls_counts,
        "unpackable": sum(per_index.values()),
        "per_index": per_index,
        "unsupported_index_phrases": len(unsupported_phrases),
    }
