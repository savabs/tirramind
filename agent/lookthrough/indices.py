"""Indian index membership and weights — the foundation of look-through.

What this computes
------------------
``fetch_index("NIFTY 50")`` returns every constituent of an NSE index together
with the fraction of the index each one represents. Downstream look-through
multiplies those fractions by what a user holds in an index fund, so that
"5% NIFTYBEES" becomes "0.54% HDFCBANK, 0.46% ICICIBANK, ...".

Where the numbers come from
---------------------------
* **Membership** — NSE publishes a constituent CSV per index at
  ``nsearchives.nseindia.com``. Free, keyless, no login. It carries Company
  Name, Industry, Symbol, Series and ISIN. It does **not** carry weights.
* **Weights** — computed here, from Yahoo Finance (``yfinance``) company data.

WHERE THIS MISLEADS — read before trusting a number out of here
---------------------------------------------------------------
* **The weights are computed, not published.** NSE does not publish index
  weights on any free endpoint (checked 2026-10-02: the ``Daily_Snapshot``
  weightage paths return the site's HTML, and ``/api/equity-stockIndices``
  returns 404). Every weight in this module is an estimate, and
  :attr:`Index.weight_accuracy_note` states its measured error.
* **Free-float is the only honest basis.** NSE indices are *free-float*
  market-cap weighted: they count only shares available to trade and exclude
  promoter and government blocks. India's large caps are promoter-heavy —
  Reliance is ~51% promoter-held, TCS ~72% — so weighting by **full** market
  cap gets them badly wrong in both directions. Measured against published ETF
  weights on 2026-10-02:

  ===================  ===============  ==================================
  basis                mean abs. error  worst constituent
  ===================  ===============  ==================================
  free-float mcap      0.37 pp          HDFCBANK, 0.64 pp (10.78% vs 10.14%)
  full mcap            1.76 pp          ICICIBANK, 4.52 pp (5.21% vs 9.73%)
  ===================  ===============  ==================================

  Full market cap is therefore **not** shippable: it puts HDFC Bank at 6.2%
  of the Nifty when the real weight is 10.1%, and HDFC Bank is the single name
  this product exists to talk about. It also fails its own coverage check —
  the nine benchmark names come to 40.6% of a full-cap-weighted index but
  51.8% of the real one. ``weight_method="full_mcap"`` is kept only so the comparison
  above can be reproduced, and it carries a louder accuracy note.
* **Free-float factors are Yahoo's, not NSE's.** We use
  ``floatShares / sharesOutstanding`` as a stand-in for NSE's published
  Investible Weight Factor. They agree closely for the names we could check
  (Reliance 49.1% vs an IWF near 0.50; TCS 28.2% vs ~0.28) but they are a
  different vendor's estimate, refreshed on Yahoo's own schedule, not NSE's
  quarterly IWF revision.
* **No caps or index divisors.** Real index construction applies capping rules
  (NIFTY BANK and NIFTY IT cap a single stock at 33% and the top-3 at 62%) and
  a rounded share count fixed at the last rebalance. We apply none of that, so
  concentrated indices (BANK, IT) are the least accurate of the seven. Their
  error is **not** quantified — the 0.37 pp figure above was measured on
  NIFTY 50 only.
* **Weights are a snapshot, and they drift daily.** They are derived from
  today's market caps, not from the share counts fixed at the last rebalance.
  Two runs a week apart will differ. ``as_of`` dates every answer.
* **Membership is as-published, not as-of-a-date.** The NSE CSV is current
  membership. There is no history here, so this module cannot answer "what was
  in the Nifty last March".
* **Nothing here is advice.** These are published memberships and arithmetic
  on public market caps.

Caching
-------
Every remote fetch is written to disk, because NSE and Yahoo both rate-limit
and this is meant to sit behind a public demo. Constituent lists get a 7-day
TTL (membership changes a handful of times a year), market caps 1 day (they
move with price), failures 1 hour (so one bad ticker cannot be re-hammered).
Root: ``$TIRRA_LOOKTHROUGH_CACHE``, else ``$TIRRA_PORTFOLIO_CACHE/lookthrough``,
else ``~/.cache/tirramind/lookthrough``.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "CapQuote",
    "Constituent",
    "Index",
    "IndexUnavailable",
    "WeightComparison",
    "compare_to_published",
    "fetch_index",
    "known_indices",
]


# ---------------------------------------------------------------------------
# Constants — every threshold that can silently change an answer lives here.
# ---------------------------------------------------------------------------

#: NSE's keyless constituent CSVs. Verified 200 OK with the expected row counts
#: on 2026-10-02. The count is asserted on fetch: NSE silently serving a
#: truncated or different list is exactly the kind of failure that would ship a
#: wrong number, so a count far from expectation is reported, not accepted.
_INDEX_SOURCES: dict[str, tuple[str, int]] = {
    "NIFTY 50": ("ind_nifty50list", 50),
    "NIFTY NEXT 50": ("ind_niftynext50list", 50),
    "NIFTY 100": ("ind_nifty100list", 100),
    "NIFTY 500": ("ind_nifty500list", 501),
    "NIFTY BANK": ("ind_niftybanklist", 14),
    "NIFTY IT": ("ind_niftyitlist", 10),
    "NIFTY MIDCAP 150": ("ind_niftymidcap150list", 150),
}

_CSV_BASE = "https://nsearchives.nseindia.com/content/indices/{slug}.csv"

#: Short names a caller might reasonably type.
_ALIASES: dict[str, str] = {
    "NIFTY": "NIFTY 50",
    "NIFTY50": "NIFTY 50",
    "NIFTYNEXT50": "NIFTY NEXT 50",
    "NEXT50": "NIFTY NEXT 50",
    "JUNIOR": "NIFTY NEXT 50",
    "NIFTY100": "NIFTY 100",
    "NIFTY500": "NIFTY 500",
    "NIFTYBANK": "NIFTY BANK",
    "BANKNIFTY": "NIFTY BANK",
    "BANK": "NIFTY BANK",
    "NIFTYIT": "NIFTY IT",
    "IT": "NIFTY IT",
    "NIFTYMIDCAP150": "NIFTY MIDCAP 150",
    "MIDCAP150": "NIFTY MIDCAP 150",
}

#: An ETF that tracks NIFTY 50 and whose real holdings Yahoo publishes. Used
#: only to measure our own error, never as a weight source.
PUBLISHED_BENCHMARK_ETF = "INDY"

#: Tickers inside an ETF's holdings that are not index constituents (cash
#: sweeps, FX forwards). Excluded before renormalising.
_NON_EQUITY_HOLDINGS = frozenset({"XTSLA", "MVRXX", "USD", "CASH", "-", ""})

CACHE_TTL_LIST = 7 * 24 * 3600
CACHE_TTL_CAPS = 24 * 3600
CACHE_TTL_PUBLISHED = 24 * 3600
CACHE_TTL_FAIL = 3600

#: Above this gap (percentage points) between the overlap's share of our index
#: and of the benchmark's book, a direct index-level comparison is mixing a
#: weighting error with a scale mismatch and says so.
COVERAGE_GAP_OK_PP = 2.0

#: Weights summing further than this from 1.0 are refused outright.
WEIGHT_SUM_TOLERANCE = 1e-9

#: Yahoo's own suffix for NSE listings.
_NSE_SUFFIX = ".NS"

#: If the fetched row count differs from expectation by more than this, say so.
#: Membership drifts by a name or two between our constant and NSE's file after
#: a reconstitution; a wholesale mismatch means we fetched the wrong thing.
_COUNT_TOLERANCE = 3

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

WEIGHT_METHODS = ("free_float_mcap", "full_mcap")

_NOTE_FREE_FLOAT = (
    "APPROXIMATION. NSE publishes which stocks are in this index but not their "
    "weights, so these are computed from free-float market cap "
    "(shares available to trade x price), which is the same basis NSE uses. "
    "Measured against the published holdings of {etf}, an ETF tracking NIFTY 50, "
    "on {date}: mean error {mean:.2f} percentage points, worst "
    "{worst_sym} off by {max:.2f}pp. Free-float share counts are Yahoo "
    "Finance's estimate, not NSE's quarterly Investible Weight Factor. "
    "Index capping rules are not applied."
)

_NOTE_FULL_MCAP = (
    "APPROXIMATION, AND A POOR ONE — do not show these to a user. Weights are "
    "computed from FULL market cap, which counts promoter and government "
    "holdings that NSE's free-float index excludes. Measured against the "
    "published holdings of {etf} on {date}: mean error {mean:.2f} percentage "
    "points, worst {worst_sym} off by {max:.2f}pp. Use "
    "weight_method='free_float_mcap' instead."
)

_NOTE_UNMEASURED = (
    "APPROXIMATION, ERROR NOT MEASURED FOR THIS INDEX. NSE publishes this "
    "index's membership but not its weights; these are computed from "
    "{basis}. The only published weights we can obtain for free track "
    "NIFTY 50, where this method's mean error is {ref}. This index is not "
    "NIFTY 50 and its error could be larger — concentrated indices "
    "(NIFTY BANK, NIFTY IT) apply capping rules we do not."
)

#: The NIFTY 50 free-float error measured on 2026-10-02, quoted when a
#: different index has no benchmark of its own. Regenerate with
#: ``compare_to_published`` and update this string if the method changes.
_REFERENCE_ERROR_FREE_FLOAT = "0.37pp (worst 0.64pp)"
_REFERENCE_ERROR_FULL = "1.76pp (worst 4.52pp)"


class IndexUnavailable(RuntimeError):
    """Raised when an index cannot be built into a usable weight vector.

    Carries every reason, so a caller can show the user what failed rather
    than an empty result. Never raised for a *partial* index — a few excluded
    constituents are reported in :attr:`Index.excluded` and the remaining
    weights are renormalised.
    """

    def __init__(self, name: str, reasons: Sequence[str]) -> None:
        self.index_name = name
        self.reasons = tuple(reasons)
        detail = "; ".join(self.reasons) if self.reasons else "no reason recorded"
        super().__init__(f"cannot build weights for {name}: {detail}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapQuote:
    """What a cap source returned for one symbol.

    ``float_shares`` and ``shares_outstanding`` are None when the vendor has
    no figure. A None float with a present market cap is usable only on the
    ``full_mcap`` basis; on the free-float basis it is a fallback that is
    labelled per constituent, never silently substituted.
    """

    symbol: str
    market_cap: float | None
    float_shares: float | None
    shares_outstanding: float | None
    currency: str | None = None


@dataclass(frozen=True)
class Constituent:
    """One index member and its computed share of the index.

    Fields are ordered so ``(symbol, company, industry, isin, weight)`` is the
    leading tuple, which is what consumers of this module were specified
    against; iterating a Constituent yields exactly those five, in that order.

    * ``weight`` is a FRACTION in [0, 1], not a percentage. Weights across an
      :class:`Index` sum to 1.0.
    * ``market_cap`` is the cap actually used to compute ``weight`` — free-float
      on the default basis, which is *smaller* than the company's market cap.
    * ``free_float_factor`` is ``float_shares / shares_outstanding``, or None
      when the vendor had no float figure. ``cap_basis`` says which happened.
    """

    symbol: str
    company: str
    industry: str
    isin: str
    weight: float
    market_cap: float
    free_float_factor: float | None
    cap_basis: str
    yahoo_symbol: str

    def __iter__(self) -> Iterator[object]:
        yield from (self.symbol, self.company, self.industry, self.isin, self.weight)


@dataclass(frozen=True)
class Index:
    """An index's membership plus a weight vector that sums to 1.0.

    ``weight_accuracy_note`` is not decoration. It carries the measured error
    of these specific weights and is meant to be rendered wherever they are.
    """

    name: str
    as_of: str
    constituents: tuple[Constituent, ...]
    weight_method: str
    weight_accuracy_note: str
    excluded: tuple[tuple[str, str], ...]
    source_list_url: str
    accuracy: WeightComparison | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.constituents)

    def weights(self) -> dict[str, float]:
        """Bare NSE symbol -> fraction of the index. Sums to 1.0."""
        return {c.symbol: c.weight for c in self.constituents}

    def weight_of(self, symbol: str) -> float:
        """Fraction of the index held in ``symbol``; 0.0 if it is not a member.

        0.0 genuinely means "not in this index". A constituent that was
        *excluded* for missing data is in :attr:`excluded`, and also reads 0.0
        here — check ``excluded`` before telling a user an index holds nothing.
        """
        key = _bare_symbol(symbol)
        for c in self.constituents:
            if c.symbol == key:
                return c.weight
        return 0.0

    def top(self, n: int = 10) -> tuple[Constituent, ...]:
        """The n heaviest constituents, heaviest first."""
        return tuple(sorted(self.constituents, key=lambda c: -c.weight)[:n])


@dataclass(frozen=True)
class WeightComparison:
    """Our computed weights vs. a published set, in percentage points.

    Two errors are reported, because they answer different questions and the
    difference between them is large enough to mislead:

    * ``mean_abs_pp`` / ``max_abs_pp`` — **index-level**, and the honest
      headline. "We say HDFC Bank is 10.78% of the Nifty; the ETF's book says
      10.14%" is an error of 0.64pp. This is the error in the number this
      product actually shows a user, so it is the one that belongs in the
      accuracy note.
    * ``mean_abs_rel_pp`` / ``max_abs_rel_pp`` — the same comparison with both
      sides renormalised onto the overlapping names only. It isolates *relative*
      weighting from any coverage mismatch, but it divides by the overlap's
      share of the index (about 0.52 here), so it roughly doubles every figure
      and describes a quantity nobody is shown. Kept as a cross-check, never as
      the headline.

    ``coverage_ours`` and ``coverage_published`` are the overlap's share of each
    side. They must be close; if they diverge, the two sides are not describing
    the same universe and neither error figure means anything.

    ``rows`` is every name compared, so the full series can be printed rather
    than trusted (LESSONS F-18: a single summary statistic hides a constant).
    """

    benchmark: str
    as_of: str
    n_compared: int
    mean_abs_pp: float
    max_abs_pp: float
    max_abs_symbol: str
    rows: tuple[tuple[str, float, float, float], ...]
    mean_abs_rel_pp: float = 0.0
    max_abs_rel_pp: float = 0.0
    coverage_ours: float = 0.0
    coverage_published: float = 0.0
    note: str = ""

    def table(self) -> str:
        """Fixed-width rendering of every compared row, for eyeballing."""
        head = f"{'symbol':<14}{'published%':>11}{'computed%':>11}{'diff pp':>9}"
        lines = [head, "-" * len(head)]
        for sym, pub, got, diff in self.rows:
            lines.append(f"{sym:<14}{pub * 100:11.3f}{got * 100:11.3f}{diff:+9.3f}")
        lines.append("-" * len(head))
        lines.append(
            f"n={self.n_compared}  mean|diff|={self.mean_abs_pp:.3f}pp  "
            f"max|diff|={self.max_abs_pp:.3f}pp ({self.max_abs_symbol})"
        )
        lines.append(
            f"these {self.n_compared} names are {self.coverage_ours * 100:.2f}% of our "
            f"index and {self.coverage_published * 100:.2f}% of the benchmark's book "
            f"-- {self.coverage_verdict()}; renormalised onto them alone, "
            f"mean|diff|={self.mean_abs_rel_pp:.3f}pp"
        )
        return "\n".join(lines)

    def coverage_gap_pp(self) -> float:
        """How far apart the two sides' coverage of the overlap is, in points."""
        return abs(self.coverage_ours - self.coverage_published) * 100.0

    def coverage_verdict(self) -> str:
        """Whether a direct, index-level comparison is sound. Computed, not asserted.

        A hardcoded "close => same universe" printed green at a 11.2pp gap on
        the full-market-cap basis, which is exactly the kind of string this
        repo has been burned by. This one reads the number.
        """
        gap = self.coverage_gap_pp()
        if gap <= COVERAGE_GAP_OK_PP:
            return f"same universe, gap {gap:.2f}pp"
        return (
            f"COVERAGE GAP {gap:.2f}pp -- the two sides do not cover the same "
            f"share of the index, so the per-name differences above mix a "
            f"weighting error with a scale mismatch and understate neither "
            f"cleanly"
        )


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def known_indices() -> list[str]:
    """The indices this module can fetch, in the canonical spelling."""
    return list(_INDEX_SOURCES)


def _canonical(name: str) -> str:
    raw = (name or "").strip()
    squashed = "".join(ch for ch in raw.upper() if ch.isalnum())
    for canon in _INDEX_SOURCES:
        if "".join(canon.split()) == squashed:
            return canon
    if squashed in _ALIASES:
        return _ALIASES[squashed]
    raise KeyError(f"unknown index {name!r} — known: {', '.join(known_indices())}")


def _bare_symbol(symbol: str) -> str:
    """Strip a Yahoo exchange suffix so ``HDFCBANK.NS`` and ``HDFCBANK`` match."""
    s = (symbol or "").strip().upper()
    for suf in (".NS", ".BO", ".NSE"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


# ---------------------------------------------------------------------------
# Disk cache — mandatory, because a public demo will hit NSE and Yahoo hard.
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    explicit = os.environ.get("TIRRA_LOOKTHROUGH_CACHE")
    if explicit:
        path = Path(explicit)
    else:
        shared = os.environ.get("TIRRA_PORTFOLIO_CACHE")
        path = Path(shared) / "lookthrough" if shared else Path.home() / ".cache" / "tirramind" / "lookthrough"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(kind: str, key: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)[:80]
    return _cache_dir() / f"{kind}__{safe}.json"


def _cache_read(kind: str, key: str, ttl_ok: float) -> dict | None:
    p = _cache_path(kind, key)
    try:
        blob = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict):
        return None
    ttl = ttl_ok if blob.get("ok") else CACHE_TTL_FAIL
    try:
        age = time.time() - float(blob.get("fetched_at", 0))
    except (TypeError, ValueError):
        return None
    if age > ttl:
        return None
    return blob


def _cache_write(kind: str, key: str, blob: dict) -> None:
    payload = {**blob, "fetched_at": time.time(), "key": key}
    try:
        _cache_path(kind, key).write_text(json.dumps(payload))
    except OSError as exc:  # pragma: no cover - disk problems shouldn't kill a demo
        log.warning("lookthrough cache write failed for %s/%s: %s", kind, key, exc)


# ---------------------------------------------------------------------------
# Remote fetchers. Each is injectable so tests never touch the network.
# ---------------------------------------------------------------------------

ListFetcher = Callable[[str], str]
CapFetcher = Callable[[Sequence[str]], Mapping[str, CapQuote]]
PublishedFetcher = Callable[[str], Mapping[str, float]]


def _http_get_text(url: str, *, timeout: float = 30.0) -> str:
    import httpx  # noqa: PLC0415 — network lib, keep out of import time

    resp = httpx.get(
        url,
        headers={"User-Agent": _UA, "Accept": "text/csv,text/plain,*/*"},
        timeout=timeout,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.text


def _yf_caps(symbols: Sequence[str]) -> dict[str, CapQuote]:
    """Market cap and float shares per Yahoo symbol, one ``.info`` call each.

    ``.info`` is used rather than ``fast_info`` because it is the only one that
    carries ``floatShares``, and free-float is the whole point. It costs about
    0.5s per symbol uncached, so NIFTY 500 takes ~4 minutes on a cold cache and
    is instant afterwards.
    """
    import yfinance as yf  # noqa: PLC0415

    yf_log = logging.getLogger("yfinance")
    prior = yf_log.level
    yf_log.setLevel(logging.CRITICAL)
    out: dict[str, CapQuote] = {}
    try:
        for sym in symbols:
            try:
                info = yf.Ticker(sym).info or {}
            except Exception as exc:  # noqa: BLE001 — yfinance raises many types
                log.warning("yfinance info failed for %s: %s", sym, exc)
                out[sym] = CapQuote(sym, None, None, None, None)
                continue
            out[sym] = CapQuote(
                symbol=sym,
                market_cap=_as_pos_float(info.get("marketCap")),
                float_shares=_as_pos_float(info.get("floatShares")),
                shares_outstanding=_as_pos_float(info.get("sharesOutstanding")),
                currency=(info.get("currency") or None),
            )
    finally:
        yf_log.setLevel(prior)
    return out


def _yf_published_weights(etf: str) -> dict[str, float]:
    """Published top holdings of ``etf`` as {bare NSE symbol: fraction of equity}.

    Yahoo exposes only the top ten holdings, so this can measure the top of an
    index and nothing else. Cash sweeps are dropped and the remainder is
    renormalised to the ETF's equity book, because an ETF holding 3.8% cash
    scales every equity weight down by that much relative to the index it
    tracks — comparing without that correction would invent a uniform negative
    bias.
    """
    import yfinance as yf  # noqa: PLC0415

    yf_log = logging.getLogger("yfinance")
    prior = yf_log.level
    yf_log.setLevel(logging.CRITICAL)
    try:
        frame = yf.Ticker(etf).funds_data.top_holdings
    finally:
        yf_log.setLevel(prior)
    if frame is None or len(frame) == 0:
        return {}
    col = "Holding Percent"
    if col not in frame.columns:  # pragma: no cover - shape change upstream
        raise IndexUnavailable(etf, [f"{etf} holdings table has no {col!r} column"])
    raw: dict[str, float] = {}
    non_equity = 0.0
    for sym, row in frame.iterrows():
        pct = _as_pos_float(row[col])
        if pct is None:
            continue
        ticker = str(sym).strip().upper()
        if ticker in _NON_EQUITY_HOLDINGS:
            non_equity += pct
            continue
        raw[_bare_symbol(ticker)] = pct
    equity_share = 1.0 - non_equity
    if equity_share <= 0:  # pragma: no cover - would mean an all-cash ETF
        return {}
    return {k: v / equity_share for k, v in raw.items()}


def _as_pos_float(value: object) -> float | None:
    """A strictly positive finite float, or None. Zero and NaN are None.

    A zero market cap is not a small company, it is a missing figure, and a
    zero-weight constituent would vanish from the output without a reason.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0:
        return None
    return out


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    symbol: str
    company: str
    industry: str
    isin: str


def _parse_constituent_csv(text: str) -> tuple[list[_Row], list[str]]:
    """Parse NSE's constituent CSV. Returns (rows, problems).

    NSE's header is ``Company Name,Industry,Symbol,Series,ISIN Code``. We read
    by header name, not position, so a reordered column does not silently shift
    every company's industry.
    """
    problems: list[str] = []
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not reader.fieldnames:
        return [], ["constituent CSV was empty or had no header row"]
    norm = {"".join(ch for ch in (f or "").lower() if ch.isalnum()): f for f in reader.fieldnames}
    need = {"symbol": "symbol", "companyname": "company", "industry": "industry", "isincode": "isin"}
    missing = [k for k in need if k not in norm]
    if "symbol" in missing:
        return [], [f"constituent CSV has no Symbol column; saw {reader.fieldnames}"]
    rows: list[_Row] = []
    seen: set[str] = set()
    for i, rec in enumerate(reader, start=2):
        sym = (rec.get(norm["symbol"]) or "").strip().upper()
        if not sym:
            problems.append(f"row {i} of the constituent CSV had no symbol — skipped")
            continue
        if sym in seen:
            problems.append(f"{sym} appeared twice in the constituent CSV — kept the first")
            continue
        seen.add(sym)
        rows.append(
            _Row(
                symbol=sym,
                company=(rec.get(norm.get("companyname", "")) or "").strip(),
                industry=(rec.get(norm.get("industry", "")) or "").strip(),
                isin=(rec.get(norm.get("isincode", "")) or "").strip().upper(),
            )
        )
    for key in missing:
        problems.append(f"constituent CSV had no {key!r} column — {need[key]} is blank for every row")
    return rows, problems


def _fetch_constituents(canon: str, *, fetcher: ListFetcher, use_cache: bool) -> tuple[list[_Row], str, list[str], str]:
    """(rows, as_of, problems, url). Cached for CACHE_TTL_LIST."""
    slug, expected = _INDEX_SOURCES[canon]
    url = _CSV_BASE.format(slug=slug)
    if use_cache:
        blob = _cache_read("list", slug, CACHE_TTL_LIST)
        if blob is not None:
            if not blob.get("ok"):
                raise IndexUnavailable(canon, [str(blob.get("reason", "cached failure"))])
            rows = [_Row(**r) for r in blob["rows"]]
            return rows, str(blob["as_of"]), list(blob.get("problems", [])), url

    try:
        text = fetcher(url)
    except Exception as exc:  # noqa: BLE001 — httpx/urllib raise many types
        reason = f"could not download the constituent list for {canon} from NSE ({exc})"
        if use_cache:
            _cache_write("list", slug, {"ok": False, "reason": reason})
        raise IndexUnavailable(canon, [reason]) from exc

    rows, problems = _parse_constituent_csv(text)
    if not rows:
        reason = problems[0] if problems else "constituent CSV contained no rows"
        if use_cache:
            _cache_write("list", slug, {"ok": False, "reason": reason})
        raise IndexUnavailable(canon, [reason])
    if abs(len(rows) - expected) > _COUNT_TOLERANCE:
        problems.append(
            f"NSE returned {len(rows)} constituents for {canon}; this index "
            f"normally has about {expected} — the downloaded list may be truncated "
            f"or be a different index. The weights here cover only the "
            f"{len(rows)} names that were returned"
        )
    as_of = datetime.now(UTC).strftime("%Y-%m-%d")
    if use_cache:
        _cache_write(
            "list",
            slug,
            {
                "ok": True,
                "as_of": as_of,
                "problems": problems,
                "rows": [
                    {"symbol": r.symbol, "company": r.company, "industry": r.industry, "isin": r.isin} for r in rows
                ],
            },
        )
    return rows, as_of, problems, url


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


def _fetch_caps(yahoo_symbols: Sequence[str], *, fetcher: CapFetcher, use_cache: bool) -> dict[str, CapQuote]:
    """Caps for every symbol, reading the per-symbol disk cache first.

    Cached per symbol rather than per index so NIFTY 100 reuses the NIFTY 50
    work, and so one unknown ticker does not expire a whole index's caps.
    """
    found: dict[str, CapQuote] = {}
    todo: list[str] = []
    for sym in yahoo_symbols:
        blob = _cache_read("cap", sym, CACHE_TTL_CAPS) if use_cache else None
        if blob is None:
            todo.append(sym)
            continue
        if not blob.get("ok"):
            found[sym] = CapQuote(sym, None, None, None, None)
            continue
        found[sym] = CapQuote(
            symbol=sym,
            market_cap=_as_pos_float(blob.get("market_cap")),
            float_shares=_as_pos_float(blob.get("float_shares")),
            shares_outstanding=_as_pos_float(blob.get("shares_outstanding")),
            currency=blob.get("currency"),
        )
    if todo:
        fresh = fetcher(todo)
        for sym in todo:
            q = fresh.get(sym) or CapQuote(sym, None, None, None, None)
            found[sym] = q
            if use_cache:
                _cache_write(
                    "cap",
                    sym,
                    {
                        "ok": q.market_cap is not None,
                        "market_cap": q.market_cap,
                        "float_shares": q.float_shares,
                        "shares_outstanding": q.shares_outstanding,
                        "currency": q.currency,
                    },
                )
    return found


def _effective_cap(q: CapQuote, method: str) -> tuple[float | None, float | None, str, str | None]:
    """(cap, free_float_factor, cap_basis, exclusion_reason).

    On the free-float basis a missing float figure does NOT disqualify a
    constituent — dropping Reliance from the Nifty would be a far bigger error
    than weighting it by full cap — but the fallback is recorded on the
    constituent and counted in the index's notes.

    Every figure is re-sanitised here rather than trusted from the CapQuote.
    This is the one chokepoint every weight passes through, and a cap of 0.0 or
    NaN arriving from any fetcher is a MISSING figure, not a tiny company: 0.0
    would ship a 0%-weight constituent with no reason attached, and NaN would
    poison the whole vector.
    """
    market_cap = _as_pos_float(q.market_cap)
    float_shares = _as_pos_float(q.float_shares)
    shares_outstanding = _as_pos_float(q.shares_outstanding)
    q = CapQuote(q.symbol, market_cap, float_shares, shares_outstanding, q.currency)
    if q.market_cap is None:
        return (
            None,
            None,
            "none",
            f"{_bare_symbol(q.symbol)}: Yahoo Finance has no market cap for "
            f"{q.symbol}, so its index weight cannot be computed — it is left out "
            f"and the other weights are renormalised",
        )
    if method == "full_mcap":
        return q.market_cap, None, "full_mcap", None
    if q.float_shares is None or q.shares_outstanding is None:
        return q.market_cap, None, "full_mcap_fallback", None
    factor = q.float_shares / q.shares_outstanding
    if not math.isfinite(factor) or factor <= 0:
        return q.market_cap, None, "full_mcap_fallback", None
    # Yahoo occasionally reports more float than shares outstanding (different
    # as-of dates on the two fields). Cap at 1.0 rather than inventing a
    # free-float larger than the company.
    factor = min(factor, 1.0)
    return q.market_cap * factor, factor, "free_float", None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _normalisation_error(constituents: Sequence[Constituent]) -> float:
    """How far the weight vector is from summing to exactly 1.0."""
    return abs(math.fsum(c.weight for c in constituents) - 1.0)


def _assert_normalised(name: str, constituents: Sequence[Constituent]) -> None:
    """Refuse to return a weight vector that does not sum to 1.0.

    Split out from :func:`fetch_index` so a test can stub
    :func:`_normalisation_error` and prove this guard is live. Deleting an
    inline assert is invisible to tests that compute the sum themselves.
    """
    err = _normalisation_error(constituents)
    if err > WEIGHT_SUM_TOLERANCE:
        raise AssertionError(f"{name} weights are off 1.0 by {err!r} — refusing to return them")


def fetch_index(
    name: str,
    *,
    weight_method: str = "free_float_mcap",
    list_fetcher: ListFetcher | None = None,
    cap_fetcher: CapFetcher | None = None,
    published_fetcher: PublishedFetcher | None = None,
    measure_accuracy: bool = True,
    use_cache: bool = True,
) -> Index:
    """Constituents of an NSE index with computed weights that sum to 1.0.

    Membership is published by NSE; the weights are **ours**, estimated from
    free-float market cap because NSE publishes no free weights. Read
    :attr:`Index.weight_accuracy_note` — it carries the measured error of the
    numbers you are about to use, and it is written to be shown to a user.

    Parameters
    ----------
    name:
        Canonical name or a common alias (``"nifty50"``, ``"banknifty"``).
    weight_method:
        ``"free_float_mcap"`` (default, mean error 0.37pp on NIFTY 50) or
        ``"full_mcap"`` (1.76pp, worst 4.52pp — kept for comparison only).
    measure_accuracy:
        When True and the index is NIFTY 50, compare against a published ETF
        and put the live measurement in the note. Costs one extra request.

    Raises
    ------
    KeyError
        Unknown index name.
    ValueError
        Unknown ``weight_method``.
    IndexUnavailable
        The constituent list could not be read, or no constituent had a market
        cap. Carries every reason.
    """
    canon = _canonical(name)
    if weight_method not in WEIGHT_METHODS:
        raise ValueError(f"weight_method must be one of {WEIGHT_METHODS}, got {weight_method!r}")
    lf = list_fetcher or _http_get_text
    cf = cap_fetcher or _yf_caps

    rows, as_of, notes, url = _fetch_constituents(canon, fetcher=lf, use_cache=use_cache)
    yahoo = {r.symbol: r.symbol + _NSE_SUFFIX for r in rows}
    caps = _fetch_caps(list(yahoo.values()), fetcher=cf, use_cache=use_cache)

    kept: list[tuple[_Row, float, float | None, str]] = []
    excluded: list[tuple[str, str]] = []
    fallbacks: list[str] = []
    for r in rows:
        ysym = yahoo[r.symbol]
        q = caps.get(ysym) or CapQuote(ysym, None, None, None, None)
        cap, factor, basis, reason = _effective_cap(q, weight_method)
        if cap is None:
            excluded.append((r.symbol, reason or "no market cap available"))
            continue
        if basis == "full_mcap_fallback":
            fallbacks.append(r.symbol)
        kept.append((r, cap, factor, basis))

    if not kept:
        raise IndexUnavailable(
            canon,
            [f"no market cap for any of the {len(rows)} constituents"] + [reason for _, reason in excluded[:5]],
        )

    total = math.fsum(cap for _, cap, _, _ in kept)
    if total <= 0:  # pragma: no cover - _as_pos_float makes this unreachable
        raise IndexUnavailable(canon, ["total market cap of kept constituents was not positive"])

    constituents = tuple(
        Constituent(
            symbol=r.symbol,
            company=r.company,
            industry=r.industry,
            isin=r.isin,
            weight=cap / total,
            market_cap=cap,
            free_float_factor=factor,
            cap_basis=basis,
            yahoo_symbol=yahoo[r.symbol],
        )
        for r, cap, factor, basis in sorted(kept, key=lambda t: -t[1])
    )

    # Weights must sum to 1.0. This is the one invariant every downstream
    # look-through number depends on; an off-by-a-dropped-name here would
    # understate every pass-through holding without any visible symptom.
    _assert_normalised(canon, constituents)

    if excluded:
        notes.append(
            f"{len(excluded)} of {len(rows)} constituents excluded for missing market "
            f"cap; the remaining {len(constituents)} weights are renormalised to 1.0, "
            f"so each is slightly overstated relative to the real index"
        )
    if fallbacks:
        shown = ", ".join(fallbacks[:8]) + (" ..." if len(fallbacks) > 8 else "")
        notes.append(
            f"{len(fallbacks)} constituents had no free-float share count and fell "
            f"back to FULL market cap, which overstates promoter-heavy names: {shown}"
        )

    accuracy = None
    if measure_accuracy:
        try:
            accuracy = compare_to_published(
                constituents,
                published_fetcher=published_fetcher or _yf_published_weights,
                index_name=canon,
                use_cache=use_cache,
            )
        except Exception as exc:  # noqa: BLE001 — accuracy is a bonus, not the answer
            notes.append(
                f"could not measure weight accuracy against published ETF holdings "
                f"({exc}); the note below quotes the last offline measurement instead"
            )

    note = _accuracy_note(canon, weight_method, accuracy, as_of)
    return Index(
        name=canon,
        as_of=as_of,
        constituents=constituents,
        weight_method=weight_method,
        weight_accuracy_note=note,
        excluded=tuple(excluded),
        source_list_url=url,
        accuracy=accuracy,
        notes=tuple(notes),
    )


def _accuracy_note(canon: str, method: str, accuracy: WeightComparison | None, as_of: str) -> str:
    basis = (
        "free-float market cap (shares available to trade x price)"
        if method == "free_float_mcap"
        else "FULL market cap, including promoter holdings the index excludes"
    )
    if accuracy is None or accuracy.n_compared == 0:
        ref = _REFERENCE_ERROR_FREE_FLOAT if method == "free_float_mcap" else _REFERENCE_ERROR_FULL
        return _NOTE_UNMEASURED.format(basis=basis, ref=ref)
    template = _NOTE_FREE_FLOAT if method == "free_float_mcap" else _NOTE_FULL_MCAP
    return template.format(
        etf=accuracy.benchmark,
        date=accuracy.as_of or as_of,
        mean=accuracy.mean_abs_pp,
        max=accuracy.max_abs_pp,
        worst_sym=accuracy.max_abs_symbol,
    )


def compare_to_published(
    constituents: Sequence[Constituent] | Index,
    *,
    published_fetcher: PublishedFetcher | None = None,
    benchmark: str = PUBLISHED_BENCHMARK_ETF,
    index_name: str = "NIFTY 50",
    use_cache: bool = True,
) -> WeightComparison | None:
    """Measure our computed weights against a published ETF's real holdings.

    Returns None when there is nothing to compare against — only NIFTY 50 has
    a free published tracker we can read, and the tracker publishes only its
    top ten holdings, so this measures the top of the index and says nothing
    about the tail.

    The ETF's weights are rescaled off its cash position, which turns "3.8% of
    this fund is a treasury sweep" into index weights an index-tracking fund's
    equity book implies. They are then compared DIRECTLY to ours, with no
    second renormalisation, because the directly-comparable index-level number
    is the one the product shows. The overlap-renormalised figure is computed
    too and reported alongside; it is about twice as large simply because it
    divides by the overlap's ~52% share of the index, and reporting it as the
    headline would overstate the error in every number a user sees.

    The safety check against the no-renormalisation choice is
    ``coverage_ours`` vs ``coverage_published``: if the overlapping names are
    51.3% of our index and 51.8% of the benchmark's, the two sides cover the
    same universe and a direct comparison is sound. A wide gap there means it
    is not, and the caller should disbelieve both figures.
    """
    if _canonical(index_name) != "NIFTY 50":
        return None
    rows = constituents.constituents if isinstance(constituents, Index) else tuple(constituents)
    if not rows:
        return None
    ours = {c.symbol: c.weight for c in rows}

    pf = published_fetcher or _yf_published_weights
    pub: Mapping[str, float] | None = None
    if use_cache:
        blob = _cache_read("published", benchmark, CACHE_TTL_PUBLISHED)
        if blob is not None and blob.get("ok"):
            pub = {str(k): float(v) for k, v in (blob.get("weights") or {}).items()}
    if pub is None:
        pub = pf(benchmark)
        if use_cache:
            _cache_write("published", benchmark, {"ok": bool(pub), "weights": dict(pub or {})})
    if not pub:
        return None

    shared = [s for s in pub if s in ours]
    missing = [s for s in pub if s not in ours]
    if not shared:
        return None

    pub_sum = math.fsum(pub[s] for s in shared)
    our_sum = math.fsum(ours[s] for s in shared)
    if pub_sum <= 0 or our_sum <= 0:  # pragma: no cover
        return None

    out_rows: list[tuple[str, float, float, float]] = []
    diffs: list[float] = []
    rel_diffs: list[float] = []
    for s in sorted(shared, key=lambda k: -pub[k]):
        # Index-level: both already normalised over the whole index.
        d = (ours[s] - pub[s]) * 100.0
        out_rows.append((s, pub[s], ours[s], d))
        diffs.append(abs(d))
        # Relative: both renormalised onto the overlap, as a cross-check.
        rel_diffs.append(abs((ours[s] / our_sum - pub[s] / pub_sum) * 100.0))
    worst = max(range(len(diffs)), key=lambda i: diffs[i])
    note = (
        f"{benchmark} publishes only its top {len(pub)} holdings, so this measures "
        f"the top of the index, not the tail. The benchmark's weights are rescaled "
        f"off its cash position so both sides are shares of an all-equity index."
    )
    if missing:
        note += f" Not matched to a constituent: {', '.join(sorted(missing))}."
    return WeightComparison(
        benchmark=benchmark,
        as_of=datetime.now(UTC).strftime("%Y-%m-%d"),
        n_compared=len(shared),
        mean_abs_pp=math.fsum(diffs) / len(diffs),
        max_abs_pp=max(diffs),
        max_abs_symbol=out_rows[worst][0],
        rows=tuple(out_rows),
        mean_abs_rel_pp=math.fsum(rel_diffs) / len(rel_diffs),
        max_abs_rel_pp=max(rel_diffs),
        coverage_ours=our_sum,
        coverage_published=pub_sum,
        note=note,
    )
