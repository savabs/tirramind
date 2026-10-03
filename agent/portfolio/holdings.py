"""Holdings ingestion: turn pasted text into an aligned, currency-consistent price panel.

This is the foundation layer of the public portfolio-structure demo. Everything
downstream (factor projection, risk attribution, clustering) consumes
:class:`PricePanel`. It computes three things:

1. ``parse_holdings(text)`` — read what a human actually pastes.
2. ``fetch_prices(holdings)`` — resolve tickers, fetch daily closes from yfinance
   (disk-cached), convert to one base currency, intersect calendars.
3. ``PricePanel`` — the aligned returns matrix, plus a full account of what was
   dropped and why.

WHERE THIS MISLEADS — read before trusting a number out of here
---------------------------------------------------------------
* **Intersection, not interpolation.** Different listings keep different
  holidays. We take the *intersection* of trading dates. Adding one US name to
  an all-India portfolio typically costs ~10-15 days a year. ``coverage`` reports
  exactly how many days each ticker lost; it is never silent.
* **FX is forward-filled.** Currency conversion uses a daily FX close reindexed
  onto the equity calendar with forward-fill. On a day the FX market was shut but
  the equity market was open, yesterday's rate is used. ``fx_report`` counts
  those days. A forward-filled rate makes that day's converted return a *local*
  return with no FX component — it is not wrong, but it is not a true
  cross-currency return either.
* **Returns are simple, not log.** ``returns`` is ``pct_change`` on adjusted
  closes. Adjusted closes already fold dividends and splits back in, so these are
  total returns, not price returns — a high-yield holding will look better here
  than in a broker app that shows price only.
* **Weights are a snapshot.** Share and amount weights are computed from the
  price on the *last aligned date*, not from what the user paid. They drift the
  moment the market moves.
* **No survivorship correction.** If a holding was delisted, yfinance's history
  ends there and we exclude it with a reason. We do not reconstruct it.
* **Nothing here is advice.** These are arithmetic facts about a stated basket
  over a stated window.

Caching
-------
Every network fetch is written to a JSON file under ``TIRRA_PORTFOLIO_CACHE``
(default ``~/.cache/tirramind/yfinance``) with a TTL. Failures are cached too, at
a shorter TTL, so a public demo cannot hammer yfinance with the same bad ticker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "MIN_ALIGNED_DAYS",
    "Exclusion",
    "HoldingEntry",
    "Holdings",
    "ParseProblem",
    "PricePanel",
    "Resolution",
    "Unit",
    "fetch_prices",
    "parse_holdings",
]

# ---------------------------------------------------------------------------
# Constants — every threshold that can silently change an answer lives here.
# ---------------------------------------------------------------------------

#: Below this many aligned trading days, no correlation / beta / risk
#: decomposition downstream is worth showing. 60 days ~= one quarter; a
#: correlation on fewer than that has a standard error wide enough to flip sign.
MIN_ALIGNED_DAYS = 60

#: A ticker whose own history is shorter than this is excluded before alignment,
#: so that one new listing cannot silently truncate the window for everyone else.
MIN_OWN_DAYS = 60

#: If a ticker's last print is more than this many calendar days before the
#: newest print in the basket, treat it as delisted/suspended rather than
#: quietly forward-filling a dead price.
STALE_DAYS = 15

#: Cache lifetimes (seconds). Daily closes change once a day; a miss is cheap
#: to re-try, a rate-limit is not.
CACHE_TTL_OK = 6 * 3600
CACHE_TTL_FAIL = 30 * 60

_SUFFIX_CANDIDATES: tuple[str, ...] = (".NS", ".BO")

_CURRENCY_SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
_CURRENCY_WORDS = {
    "INR": "INR",
    "RS": "INR",
    "RS.": "INR",
    "RUPEES": "INR",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "JPY": "JPY",
}
_QTY_WORDS = {"SHARES", "SHARE", "SHS", "SH", "QTY", "QUANTITY", "UNITS", "UNIT", "NOS", "NO"}
_VALUE_WORDS = {"VALUE", "AMOUNT", "WORTH", "INVESTED", "COST", "MKTVALUE"}
_PCT_WORDS = {"PERCENT", "PCT", "WEIGHT", "ALLOCATION"}

_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9&\-]{0,18}(\.[A-Za-z]{1,4})?$")
_SUFFIXED_RE = re.compile(r"^[A-Za-z0-9&\-]{1,18}\.[A-Za-z]{1,4}$")

#: Header tokens → logical column. Broker exports are the realistic paste.
_HEADER_MAP: dict[str, str] = {
    "symbol": "symbol",
    "ticker": "symbol",
    "scrip": "symbol",
    "scripname": "symbol",
    "instrument": "symbol",
    "stock": "symbol",
    "security": "symbol",
    "isin": "ignore",
    "qty": "shares",
    "quantity": "shares",
    "shares": "shares",
    "units": "shares",
    "holdingqty": "shares",
    "weight": "percent",
    "weightage": "percent",
    "allocation": "percent",
    "percent": "percent",
    "pct": "percent",
    "%": "percent",
    "value": "amount",
    "amount": "amount",
    "marketvalue": "amount",
    "currentvalue": "amount",
    "invested": "amount",
}


class Unit(str, Enum):
    """How the user expressed the size of a position."""

    SHARES = "shares"
    PERCENT = "percent"
    AMOUNT = "amount"


@dataclass(frozen=True)
class HoldingEntry:
    """One position exactly as we understood it, with the line it came from."""

    symbol: str
    quantity: float
    unit: Unit
    raw: str
    line_no: int
    currency: str | None = None

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.unit is Unit.PERCENT:
            return f"{self.symbol} {self.quantity:g}%"
        if self.unit is Unit.AMOUNT:
            return f"{self.symbol} {self.currency or '?'} {self.quantity:,.0f}"
        return f"{self.symbol} {self.quantity:g} sh"


@dataclass(frozen=True)
class ParseProblem:
    """A line we could not read. Never dropped silently — always surfaced."""

    line_no: int
    raw: str
    reason: str


@dataclass(frozen=True)
class Holdings:
    """What we read out of the pasted text.

    ``assumptions`` is the load-bearing field: every guess the parser made (which
    token was the ticker, which of three numbers on a line was the quantity,
    whether a bare list meant equal weight) is recorded there in plain English so
    the user can catch us being wrong.
    """

    entries: tuple[HoldingEntry, ...] = ()
    unreadable: tuple[ParseProblem, ...] = ()
    assumptions: tuple[str, ...] = ()
    unit_mode: str = "none"

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(e.symbol for e in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def describe(self) -> str:  # pragma: no cover - display only
        lines = [f"Read {len(self.entries)} position(s), sized in {self.unit_mode}."]
        lines += [f"  {e}" for e in self.entries]
        if self.assumptions:
            lines.append("Assumptions:")
            lines += [f"  - {a}" for a in self.assumptions]
        if self.unreadable:
            lines.append("Could not read:")
            lines += [f'  - line {p.line_no}: "{p.raw}" — {p.reason}' for p in self.unreadable]
        return "\n".join(lines)


@dataclass(frozen=True)
class Resolution:
    """What a typed symbol was resolved to, so a wrong guess is catchable."""

    typed: str
    resolved: str
    currency: str
    note: str


@dataclass(frozen=True)
class Exclusion:
    """A holding that did NOT make it into the panel, and why."""

    symbol: str
    reason: str
    resolved: str | None = None


@dataclass
class PricePanel:
    """Aligned daily closes and returns for a basket, in one base currency.

    Attributes
    ----------
    prices : DataFrame
        Adjusted closes, index = the common trading calendar, columns = resolved
        tickers, expressed in ``base_currency``. No NaNs by construction.
    returns : DataFrame
        ``prices.pct_change().dropna()`` — one row shorter than ``prices``.
    weights : Series
        Normalised portfolio weights summing to 1.0, on the same columns.
    coverage : DataFrame
        Per ticker: own_days, own_start, own_end, days_lost_to_alignment.
        This is how you see which holding is costing the panel its history.
    excluded : tuple[Exclusion, ...]
        Every symbol that did not make it, with a reason. Never empty-by-silence.
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.Series
    base_currency: str
    coverage: pd.DataFrame
    excluded: tuple[Exclusion, ...] = ()
    resolutions: tuple[Resolution, ...] = ()
    assumptions: tuple[str, ...] = ()
    fx_report: tuple[str, ...] = ()
    weight_basis: str = ""
    window_note: tuple[str, ...] = ()
    usable: bool = False
    unusable_reason: str = ""

    # -- derived facts, each carrying its own sample size ------------------
    @property
    def n_days(self) -> int:
        """Aligned trading days of *price*."""
        return int(len(self.prices))

    @property
    def n_returns(self) -> int:
        """Aligned *return* observations — always ``n_days - 1``. This, not
        ``n_days``, is the sample size behind any correlation or beta."""
        return int(len(self.returns))

    @property
    def start(self) -> pd.Timestamp | None:
        return None if self.prices.empty else self.prices.index[0]

    @property
    def end(self) -> pd.Timestamp | None:
        return None if self.prices.empty else self.prices.index[-1]

    @property
    def window(self) -> str:
        """A citable window string: a date range and an n, never 'since 2024'."""
        if self.prices.empty:
            return "no aligned window"
        return f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d} ({self.n_returns} daily returns)"

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(self.prices.columns)

    def describe(self) -> str:
        """Human-readable audit of the panel. Safe to show a stranger verbatim."""
        out: list[str] = []
        if not self.usable:
            out.append(f"PANEL UNUSABLE: {self.unusable_reason}")
        out.append(f"Window     : {self.window}")
        out += [f"             {n}" for n in self.window_note]
        out.append(f"Currency   : everything converted to {self.base_currency}")
        out.append(f"Holdings in: {len(self.tickers)}    excluded: {len(self.excluded)}")
        out.append("")
        out.append("Resolved tickers")
        for r in self.resolutions:
            w = self.weights.get(r.resolved, float("nan"))
            wt = "  --  " if pd.isna(w) else f"{w:6.1%}"
            out.append(f"  {r.typed:<14} -> {r.resolved:<14} {r.currency}  {wt}   {r.note}")
        out.append("")
        out.append("Coverage (days each ticker lost to the shared calendar)")
        out.append(f"  {'ticker':<14}{'own days':>9}{'own start':>13}{'own end':>13}{'lost':>7}")
        for tic, row in self.coverage.iterrows():
            out.append(
                f"  {tic:<14}{int(row['own_days']):>9}{str(row['own_start'])[:10]:>13}"
                f"{str(row['own_end'])[:10]:>13}{int(row['days_lost_to_alignment']):>7}"
            )
        if self.excluded:
            out.append("")
            out.append("EXCLUDED (nothing is dropped silently)")
            for e in self.excluded:
                tag = f"{e.symbol}" + (f" -> {e.resolved}" if e.resolved else "")
                out.append(f"  {tag:<28} {e.reason}")
        if self.fx_report:
            out.append("")
            out.append("FX")
            out += [f"  {line}" for line in self.fx_report]
        if self.assumptions:
            out.append("")
            out.append("Assumptions")
            out += [f"  - {a}" for a in self.assumptions]
        if self.weight_basis:
            out.append("")
            out.append(f"Weights    : {self.weight_basis}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _protect_thousands(line: str) -> str:
    """Replace a thousands-comma with \\x00 so field-splitting cannot cut a number."""
    return re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "\x00", line)


def _restore(tok: str) -> str:
    return tok.replace("\x00", ",")


def _split_fields(line: str) -> list[str]:
    protected = _protect_thousands(line)
    for delim in ("\t", ";", "|"):
        if delim in protected:
            return [_restore(f).strip() for f in protected.split(delim)]
    if "," in protected:
        return [_restore(f).strip() for f in protected.split(",")]
    return [_restore(f).strip() for f in protected.split()]


def _norm_header(tok: str) -> str:
    return re.sub(r"[^a-z%]", "", tok.lower())


def _detect_header(fields: Sequence[str]) -> dict[str, int] | None:
    """Map logical column -> index, if this row looks like a header."""
    mapping: dict[str, int] = {}
    hits = 0
    for i, f in enumerate(fields):
        key = _HEADER_MAP.get(_norm_header(f))
        if key is None:
            continue
        hits += 1
        if key != "ignore" and key not in mapping:
            mapping[key] = i
    if hits >= 2 and "symbol" in mapping:
        return mapping
    return None


def _parse_number(tok: str) -> tuple[float, str | None, str | None] | None:
    """Parse one token into (value, unit_hint, currency).

    unit_hint is "percent", "amount" or None (meaning: not self-describing).
    Returns None if the token is not a number at all.
    """
    t = tok.strip().replace(",", "").replace("\x00", "")
    if not t:
        return None
    cur = None
    hint = None
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in t:
            cur, hint = code, "amount"
            t = t.replace(sym, "")
    if t.endswith("%"):
        hint = "percent"
        t = t[:-1]
    t = t.strip()
    # "Rs." / "INR" glued to the number, e.g. "Rs50000"
    m = re.match(r"^(?i:rs\.?|inr|usd)\s*(.*)$", t)
    if m and m.group(1):
        cur = cur or ("USD" if t.upper().startswith("USD") else "INR")
        hint = "amount"
        t = m.group(1)
    if not re.fullmatch(r"-?\d*\.?\d+(e-?\d+)?", t, flags=re.IGNORECASE):
        return None
    try:
        return float(t), hint, cur
    except ValueError:  # pragma: no cover - guarded by the regex
        return None


def _looks_like_symbol(tok: str) -> bool:
    t = tok.strip()
    if not t or _parse_number(t) is not None:
        return False
    if t.upper() in _QTY_WORDS or t.upper() in _VALUE_WORDS or t.upper() in _PCT_WORDS:
        return False
    if t.upper() in _CURRENCY_WORDS:
        return False
    return bool(_SYMBOL_RE.match(t))


def _pick_symbol(tokens: Sequence[str]) -> tuple[str | None, str | None]:
    """Choose the ticker token. Returns (symbol, assumption-or-None)."""
    cands = [t for t in tokens if _looks_like_symbol(t)]
    if not cands:
        return None, None
    suffixed = [c for c in cands if _SUFFIXED_RE.match(c)]
    if suffixed:
        note = None
        if len(cands) > 1:
            note = f"took '{suffixed[0]}' as the ticker (it carries an exchange suffix); ignored {cands[1:]!r}"
        return suffixed[0].upper(), note
    upper = [c for c in cands if c.isupper() and len(c) >= 2]
    chosen = upper[0] if upper else cands[0]
    note = None
    if len(cands) > 1:
        note = f"took the first token '{chosen}' as the ticker; ignored the rest of {cands!r}"
    return chosen.upper(), note


def parse_holdings(text: str) -> Holdings:
    """Read a pasted portfolio into structured entries.

    Accepts, in one blob and mixed: ``RELIANCE 30%`` / ``AAPL, 50 shares`` /
    ``INFY.NS 120`` / ``HDFCBANK ₹50,000`` / a CSV with a broker header row /
    a bare list of symbols.

    Sizing rules, which are the part that can quietly be wrong:

    * If **every** entry is a percentage → ``unit_mode="percent"``, weights are
      taken as given and renormalised to 1.0 (a report is emitted if they did not
      already sum to ~100).
    * If entries mix shares and amounts → ``unit_mode="value"``; both become a
      money value at the last aligned price, so they can be compared.
    * **Percentages are never mixed with shares/amounts.** A basket that says
      "RELIANCE 30%, AAPL 50 shares" is genuinely ambiguous — there is no honest
      way to combine them — so the minority form is rejected with a reason rather
      than guessed at.
    * If **no** entry carries a number → equal weight, reported as an assumption.

    Where this misleads: the parser picks the *first* number on a line as the
    quantity. A broker export whose first numeric column is average price, not
    quantity, will be read wrongly — which is why every such line adds an entry
    to ``assumptions`` naming the numbers it ignored.
    """
    raw_lines = [ln.rstrip() for ln in (text or "").splitlines()]
    entries: list[HoldingEntry] = []
    problems: list[ParseProblem] = []
    assumptions: list[str] = []

    # --- header-driven path (broker CSV) ---------------------------------
    header: dict[str, int] | None = None
    header_line_no = -1
    for i, ln in enumerate(raw_lines):
        if not ln.strip():
            continue
        maybe = _detect_header(_split_fields(ln))
        if maybe:
            header, header_line_no = maybe, i
        break

    for i, ln in enumerate(raw_lines):
        if i == header_line_no or not ln.strip():
            continue
        line_no = i + 1
        if ln.strip().startswith("#"):
            continue
        fields = _split_fields(ln)
        parsed = _parse_line_with_header(fields, header) if header else _parse_line_freeform(fields)
        if parsed is None:
            problems.append(ParseProblem(line_no, ln.strip(), "no ticker-like token found"))
            continue
        symbol, qty, unit, cur, notes = parsed
        if symbol is None:
            problems.append(ParseProblem(line_no, ln.strip(), "no ticker-like token found"))
            continue
        if qty is not None and (not np.isfinite(qty) or qty <= 0):
            problems.append(ParseProblem(line_no, ln.strip(), f"quantity {qty} is not positive"))
            continue
        entries.append(
            HoldingEntry(
                symbol=symbol,
                quantity=float(qty) if qty is not None else float("nan"),
                unit=unit or Unit.SHARES,
                raw=ln.strip(),
                line_no=line_no,
                currency=cur,
            )
        )
        assumptions += [f"line {line_no}: {n}" for n in notes]

    entries, mode, mode_notes, rejected = _reconcile_units(entries)
    assumptions += mode_notes
    problems += rejected

    # Duplicate symbols: combine rather than silently keep the last row.
    entries, dup_notes = _merge_duplicates(entries)
    assumptions += dup_notes

    return Holdings(
        entries=tuple(entries),
        unreadable=tuple(problems),
        assumptions=tuple(assumptions),
        unit_mode=mode,
    )


def _parse_line_with_header(
    fields: Sequence[str], header: dict[str, int]
) -> tuple[str | None, float | None, Unit | None, str | None, list[str]] | None:
    notes: list[str] = []
    si = header.get("symbol", 0)
    if si >= len(fields):
        return None
    sym_tokens = fields[si].split()
    symbol, note = _pick_symbol(sym_tokens) if sym_tokens else (None, None)
    if note:
        notes.append(note)
    if symbol is None:
        return None
    for key, unit in (("percent", Unit.PERCENT), ("shares", Unit.SHARES), ("amount", Unit.AMOUNT)):
        idx = header.get(key)
        if idx is None or idx >= len(fields):
            continue
        num = _parse_number(fields[idx])
        if num is None:
            continue
        val, _hint, cur = num
        return symbol, val, unit, cur, notes
    return symbol, None, None, None, notes


def _parse_line_freeform(
    fields: Sequence[str],
) -> tuple[str | None, float | None, Unit | None, str | None, list[str]] | None:
    tokens: list[str] = []
    for f in fields:
        tokens.extend(f.split())
    if not tokens:
        return None
    symbol, note = _pick_symbol(tokens)
    notes = [note] if note else []
    if symbol is None:
        return None

    numbers: list[tuple[float, str | None, str | None, int]] = []
    for i, tok in enumerate(tokens):
        num = _parse_number(tok)
        if num is not None:
            numbers.append((num[0], num[1], num[2], i))
    if not numbers:
        return symbol, None, None, None, notes

    val, hint, cur, pos = numbers[0]
    if len(numbers) > 1:
        ignored = [f"{n[0]:g}" for n in numbers[1:]]
        notes.append(f"{len(numbers)} numbers on this line; used {val:g} as the quantity, ignored {ignored}")

    # A trailing word can re-label the number: "50 shares", "50000 INR", "5 pct".
    unit = Unit.SHARES
    if hint == "percent":
        unit = Unit.PERCENT
    elif hint == "amount":
        unit = Unit.AMOUNT
    for tok in tokens[pos + 1 : pos + 3]:
        up = tok.upper().strip(".")
        if up in _QTY_WORDS:
            unit = Unit.SHARES
            break
        if up in _VALUE_WORDS:
            unit = Unit.AMOUNT
            break
        if up in _PCT_WORDS:
            unit = Unit.PERCENT
            break
        if up in _CURRENCY_WORDS:
            unit, cur = Unit.AMOUNT, _CURRENCY_WORDS[up]
            break
    return symbol, val, unit, cur, notes


def _reconcile_units(
    entries: list[HoldingEntry],
) -> tuple[list[HoldingEntry], str, list[str], list[ParseProblem]]:
    """Decide one sizing mode for the whole basket. Never mixes % with value."""
    notes: list[str] = []
    rejected: list[ParseProblem] = []
    sized = [e for e in entries if np.isfinite(e.quantity)]
    unsized = [e for e in entries if not np.isfinite(e.quantity)]

    if not sized:
        if not entries:
            return [], "none", notes, rejected
        notes.append(f"no quantities given for any of the {len(entries)} symbols — assuming EQUAL WEIGHT")
        eq = 100.0 / len(entries)
        return (
            [HoldingEntry(e.symbol, eq, Unit.PERCENT, e.raw, e.line_no, None) for e in entries],
            "percent (equal weight assumed)",
            notes,
            rejected,
        )

    for e in unsized:
        rejected.append(ParseProblem(e.line_no, e.raw, "no quantity found on this line while other lines had one"))

    pct = [e for e in sized if e.unit is Unit.PERCENT]
    val = [e for e in sized if e.unit is not Unit.PERCENT]
    if pct and val:
        if len(pct) >= len(val):
            for e in val:
                rejected.append(
                    ParseProblem(
                        e.line_no,
                        e.raw,
                        "sized in shares/amount while the rest of the basket is in percent — "
                        "the two cannot be combined without knowing the portfolio's total value",
                    )
                )
            sized, mode = pct, "percent"
        else:
            for e in pct:
                rejected.append(
                    ParseProblem(
                        e.line_no,
                        e.raw,
                        "sized in percent while the rest of the basket is in shares/amount — "
                        "the two cannot be combined without knowing the portfolio's total value",
                    )
                )
            sized, mode = val, "value (shares and/or amounts)"
        notes.append(
            f"basket mixed percentages ({len(pct)}) with shares/amounts ({len(val)}); "
            f"kept the {mode.split()[0]} entries and rejected the other {len(rejected)}"
        )
    elif pct:
        sized, mode = pct, "percent"
        total = sum(e.quantity for e in pct)
        if abs(total - 100.0) > 1.0:
            notes.append(f"percentages sum to {total:.1f}%, not 100% — renormalised to 100%")
    else:
        sized = val
        kinds = {e.unit.value for e in val}
        mode = "value (shares and/or amounts)" if len(kinds) > 1 else f"{next(iter(kinds))}"
    return sized, mode, notes, rejected


def _merge_duplicates(entries: list[HoldingEntry]) -> tuple[list[HoldingEntry], list[str]]:
    seen: dict[str, HoldingEntry] = {}
    notes: list[str] = []
    for e in entries:
        prev = seen.get(e.symbol)
        if prev is None:
            seen[e.symbol] = e
            continue
        if prev.unit is e.unit:
            merged = prev.quantity + e.quantity
            notes.append(
                f"{e.symbol} appeared on lines {prev.line_no} and {e.line_no}; "
                f"added them together ({prev.quantity:g} + {e.quantity:g} = {merged:g})"
            )
            seen[e.symbol] = HoldingEntry(e.symbol, merged, e.unit, prev.raw, prev.line_no, prev.currency or e.currency)
        else:
            notes.append(f"{e.symbol} appeared twice in different units; kept line {prev.line_no}")
    return list(seen.values()), notes


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


@dataclass
class _Series:
    """One ticker's raw close history plus its native currency."""

    symbol: str
    close: pd.Series
    currency: str


def _cache_dir() -> Path:
    root = os.environ.get("TIRRA_PORTFOLIO_CACHE")
    path = Path(root) if root else Path.home() / ".cache" / "tirramind" / "yfinance"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(symbol: str, period: str) -> Path:
    digest = hashlib.sha256(f"{symbol}|{period}".encode()).hexdigest()[:20]
    return _cache_dir() / f"{digest}.json"


def _cache_read(symbol: str, period: str) -> dict | None:
    p = _cache_key(symbol, period)
    try:
        blob = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    ttl = CACHE_TTL_OK if blob.get("ok") else CACHE_TTL_FAIL
    if time.time() - float(blob.get("fetched_at", 0)) > ttl:
        return None
    return blob


def _cache_write(symbol: str, period: str, blob: dict) -> None:
    blob = {**blob, "fetched_at": time.time(), "symbol": symbol, "period": period}
    try:
        _cache_key(symbol, period).write_text(json.dumps(blob))
    except OSError as exc:  # pragma: no cover - disk problems shouldn't kill a demo
        log.warning("portfolio price cache write failed for %s: %s", symbol, exc)


def _yf_fetch(symbol: str, period: str) -> tuple[pd.Series, str] | None:
    """Fetch adjusted daily closes + native currency for one symbol.

    Returns None when yfinance has no usable history (unknown ticker, delisted,
    or the request failed). Callers must treat None as "exclude with a reason",
    never as "zero".
    """
    import yfinance as yf  # noqa: PLC0415 — heavy, network-y; keep out of import time

    # Suffix probing ("RELIANCE" -> .NS, .BO) makes 404s a NORMAL, expected part
    # of resolution. yfinance prints them at ERROR on its own logger, which in a
    # web demo looks like a fault. Silence only this call; the miss is still
    # returned as None and still reaches the user as a reported exclusion.
    yf_log = logging.getLogger("yfinance")
    prior = yf_log.level
    yf_log.setLevel(logging.CRITICAL)
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period=period, auto_adjust=True)
    except Exception as exc:  # noqa: BLE001 — yfinance raises many unrelated types
        log.warning("yfinance history failed for %s: %s", symbol, exc)
        return None
    finally:
        yf_log.setLevel(prior)
    if hist is None or hist.empty or "Close" not in hist:
        return None
    close = hist["Close"].dropna()
    if close.empty:
        return None
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    currency = "UNKNOWN"
    try:
        currency = (dict(tk.fast_info).get("currency") or "UNKNOWN").upper()
    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance fast_info failed for %s: %s", symbol, exc)
    return close, currency


def _fetch_cached(
    symbol: str, period: str, fetcher: Callable[[str, str], tuple[pd.Series, str] | None]
) -> _Series | None:
    blob = _cache_read(symbol, period)
    if blob is not None:
        if not blob.get("ok"):
            return None
        idx = pd.to_datetime(blob["dates"])
        return _Series(symbol, pd.Series(blob["close"], index=idx, name=symbol), blob["currency"])
    got = fetcher(symbol, period)
    if got is None:
        _cache_write(symbol, period, {"ok": False})
        return None
    close, currency = got
    _cache_write(
        symbol,
        period,
        {
            "ok": True,
            "currency": currency,
            "dates": [d.strftime("%Y-%m-%d") for d in close.index],
            "close": [float(v) for v in close.to_numpy()],
        },
    )
    close = close.copy()
    close.name = symbol
    return _Series(symbol, close, currency)


def _candidates(symbol: str) -> list[str]:
    s = symbol.upper().strip()
    if "." in s or "=" in s or "^" in s:
        return [s]
    return [s + suf for suf in _SUFFIX_CANDIDATES] + [s]


def _resolve(
    symbol: str, period: str, fetcher: Callable[[str, str], tuple[pd.Series, str] | None]
) -> tuple[_Series | None, Resolution | None, str]:
    """Try exchange suffixes in order; report which one matched."""
    tried: list[str] = []
    for cand in _candidates(symbol):
        tried.append(cand)
        got = _fetch_cached(cand, period, fetcher)
        if got is not None and len(got.close) > 0:
            if len(tried) == 1 and cand == symbol.upper():
                note = "used exactly as typed"
            else:
                note = f"tried {', '.join(tried)} — matched {cand}. If that is the wrong listing, type the full ticker."
            return got, Resolution(symbol, cand, got.currency, note), ""
    return (
        None,
        None,
        f"no price history on yfinance for {' or '.join(tried)} — check the spelling or the exchange suffix",
    )


def _fx_series(
    quote: str, base: str, period: str, fetcher: Callable[[str, str], tuple[pd.Series, str] | None]
) -> tuple[pd.Series | None, str]:
    """Daily FX close converting 1 unit of ``quote`` into ``base``."""
    direct = f"{quote}{base}=X"
    got = _fetch_cached(direct, period, fetcher)
    if got is not None and len(got.close) > 0:
        return got.close, direct
    inverse = f"{base}{quote}=X"
    got = _fetch_cached(inverse, period, fetcher)
    if got is not None and len(got.close) > 0:
        inv = 1.0 / got.close
        return inv, f"1/{inverse}"
    return None, ""


def fetch_prices(
    holdings: Holdings,
    *,
    period: str = "3y",
    base_currency: str | None = None,
    min_days: int = MIN_ALIGNED_DAYS,
    fetcher: Callable[[str, str], tuple[pd.Series, str] | None] | None = None,
) -> PricePanel:
    """Resolve tickers, fetch closes, convert to one currency, and align calendars.

    Parameters
    ----------
    period : str
        yfinance period string ("1y", "3y", "5y", "max").
    base_currency : str | None
        Currency to express everything in. ``None`` infers it from the most
        common native currency in the basket, which is reported.
    min_days : int
        Aligned-return floor. A panel below it is returned with ``usable=False``
        rather than handed downstream to produce a confident wrong number.
    fetcher : callable
        Injection point for tests: ``(symbol, period) -> (close_series, currency)``
        or ``None``. Defaults to the disk-cached yfinance fetcher.

    Where this misleads: see the module docstring — intersection costs days, FX
    is forward-filled, and weights are a last-date snapshot.
    """
    fetch = fetcher or _yf_fetch
    excluded: list[Exclusion] = []
    resolutions: list[Resolution] = []
    assumptions: list[str] = list(holdings.assumptions)
    fx_report: list[str] = []

    for problem in holdings.unreadable:
        excluded.append(Exclusion(problem.raw[:40] or f"line {problem.line_no}", problem.reason))

    series: dict[str, _Series] = {}
    entry_by_ticker: dict[str, HoldingEntry] = {}
    for entry in holdings.entries:
        got, res, why = _resolve(entry.symbol, period, fetch)
        if got is None or res is None:
            excluded.append(Exclusion(entry.symbol, why))
            continue
        if res.resolved in series:
            excluded.append(Exclusion(entry.symbol, f"resolves to {res.resolved}, already in the basket", res.resolved))
            continue
        series[res.resolved] = got
        resolutions.append(res)
        entry_by_ticker[res.resolved] = entry

    if not series:
        return _empty_panel(
            base_currency or "INR", excluded, resolutions, assumptions, "no ticker resolved to any price history"
        )

    # --- drop short and stale histories BEFORE alignment ------------------
    newest = max(s.close.index[-1] for s in series.values())
    for tic in list(series):
        s = series[tic]
        if len(s.close) < MIN_OWN_DAYS:
            excluded.append(
                Exclusion(
                    entry_by_ticker[tic].symbol,
                    f"only {len(s.close)} trading days of history "
                    f"({s.close.index[0]:%Y-%m-%d} to {s.close.index[-1]:%Y-%m-%d}); "
                    f"{MIN_OWN_DAYS} needed before anything can be computed about it",
                    tic,
                )
            )
            del series[tic]
            continue
        gap = (newest - s.close.index[-1]).days
        if gap > STALE_DAYS:
            excluded.append(
                Exclusion(
                    entry_by_ticker[tic].symbol,
                    f"last price is {s.close.index[-1]:%Y-%m-%d}, {gap} days before the rest of the "
                    f"basket — delisted, suspended, or a wrong ticker",
                    tic,
                )
            )
            del series[tic]

    if not series:
        return _empty_panel(
            base_currency or "INR", excluded, resolutions, assumptions, "every holding was excluded before alignment"
        )

    # --- currency ---------------------------------------------------------
    currencies = [s.currency for s in series.values()]
    if base_currency is None:
        base = max(set(currencies), key=currencies.count)
        if len(set(currencies)) > 1:
            assumptions.append(
                f"basket spans {' and '.join(sorted(set(currencies)))}; converted everything to {base} "
                f"(the currency of {currencies.count(base)} of {len(currencies)} holdings)"
            )
    else:
        base = base_currency.upper()

    converted: dict[str, pd.Series] = {}
    for tic, s in series.items():
        if s.currency == base:
            converted[tic] = s.close
            continue
        if s.currency == "UNKNOWN":
            excluded.append(
                Exclusion(
                    entry_by_ticker[tic].symbol,
                    f"yfinance did not report a currency for it, so it cannot be safely mixed with {base}",
                    tic,
                )
            )
            continue
        fx, fx_name = _fx_series(s.currency, base, period, fetch)
        if fx is None:
            excluded.append(
                Exclusion(
                    entry_by_ticker[tic].symbol,
                    f"priced in {s.currency}; no {s.currency}->{base} rate available, and mixing "
                    f"currencies without converting would be a wrong answer that looks right",
                    tic,
                )
            )
            continue
        aligned_fx = fx.reindex(s.close.index).ffill()
        filled = int(aligned_fx.notna().sum() - fx.reindex(s.close.index).notna().sum())
        usable = aligned_fx.notna()
        if int(usable.sum()) < MIN_OWN_DAYS:
            excluded.append(
                Exclusion(
                    entry_by_ticker[tic].symbol,
                    f"{s.currency}->{base} rate covers only {int(usable.sum())} of its {len(s.close)} days",
                    tic,
                )
            )
            continue
        converted[tic] = (s.close[usable] * aligned_fx[usable]).rename(tic)
        fx_report.append(
            f"{tic}: {s.currency} -> {base} via {fx_name}; "
            f"{filled} of {int(usable.sum())} days used a forward-filled rate "
            f"(FX market shut while the equity market was open)"
        )

    for tic in list(series):
        if tic not in converted:
            del series[tic]
    if not converted:
        return _empty_panel(base, excluded, resolutions, assumptions, "no holding could be expressed in one currency")

    # --- alignment --------------------------------------------------------
    own = {t: s for t, s in converted.items()}
    common = _intersect(own)
    binding_notes: list[str] = []
    # Never drop below two holdings to buy window length: a one-name "portfolio"
    # is not a portfolio, and at that point the honest answer is "not enough
    # shared history", not a longer window for a basket of one.
    while len(common) < min_days + 1 and len(own) > 2:
        victim = _binding_constraint(own, common)
        if victim is None:
            break
        gained_from = len(common)
        del own[victim]
        common = _intersect(own)
        binding_notes.append(f"dropping {victim} lifted the shared window from {gained_from} to {len(common)} days")
        excluded.append(
            Exclusion(
                entry_by_ticker[victim].symbol,
                f"its history was the binding constraint: with it the whole basket shared only "
                f"{gained_from} days, below the {min_days}-day floor",
                victim,
            )
        )
    assumptions += binding_notes

    prices = pd.DataFrame({t: own[t].reindex(common) for t in own}).sort_index()
    coverage = pd.DataFrame(
        {
            "own_days": {t: int(len(converted[t])) for t in own},
            "own_start": {t: converted[t].index[0].date() for t in own},
            "own_end": {t: converted[t].index[-1].date() for t in own},
            "days_lost_to_alignment": {t: int(len(converted[t]) - len(common)) for t in own},
        }
    )
    returns = prices.pct_change().dropna(how="any")

    weights, basis = _resolve_weights({t: entry_by_ticker[t] for t in own}, prices, base, series, period, fetch)

    usable = len(returns) >= min_days and len(own) >= 1
    reason = (
        ""
        if usable
        else f"only {len(returns)} aligned daily returns across {len(own)} holdings; "
        f"{min_days} is the floor below which correlations are not worth showing"
    )
    return PricePanel(
        prices=prices,
        returns=returns,
        weights=weights,
        base_currency=base,
        coverage=coverage,
        excluded=tuple(excluded),
        resolutions=tuple(r for r in resolutions if r.resolved in own),
        assumptions=tuple(assumptions),
        fx_report=tuple(fx_report),
        weight_basis=basis,
        window_note=_explain_window(own, common),
        usable=usable,
        unusable_reason=reason,
    )


def _intersect(series_map: dict[str, pd.Series]) -> pd.DatetimeIndex:
    idx: pd.DatetimeIndex | None = None
    for s in series_map.values():
        valid = s.dropna().index
        idx = valid if idx is None else idx.intersection(valid)
    return idx if idx is not None else pd.DatetimeIndex([])


def _explain_window(series_map: dict[str, pd.Series], common: pd.DatetimeIndex) -> tuple[str, ...]:
    """Name the tickers that decide each edge of the shared window.

    Without this, a user whose broker shows a price for yesterday sees a window
    that ends three days ago and reasonably concludes the numbers are stale or
    broken. Usually one ETF simply did not trade that day. Say which one.
    """
    if len(common) == 0 or not series_map:
        return ()
    notes: list[str] = []
    first, last = common[0], common[-1]

    starters = sorted(t for t, s in series_map.items() if s.dropna().index[0] >= first)
    if starters and len(starters) == len(series_map):
        notes.append(f"starts {first:%Y-%m-%d}, the start of the requested history window — no holding limits it")
    elif starters:
        notes.append(
            f"starts {first:%Y-%m-%d} because that is the first day "
            f"{', '.join(starters)} {'has' if len(starters) == 1 else 'have'} a price"
        )

    after = sorted({d for s in series_map.values() for d in s.dropna().index if d > last})
    if after:
        nxt = after[0]
        missing = sorted(t for t, s in series_map.items() if nxt not in s.dropna().index)
        notes.append(
            f"ends {last:%Y-%m-%d} because the next day any holding traded is {nxt:%Y-%m-%d}, "
            f"and {', '.join(missing)} {'has' if len(missing) == 1 else 'have'} no price that day"
        )
    return tuple(notes)


def _binding_constraint(series_map: dict[str, pd.Series], common: pd.DatetimeIndex) -> str | None:
    """Which single ticker, removed, most extends the shared window?"""
    best, best_gain = None, 0
    for tic in series_map:
        rest = {t: s for t, s in series_map.items() if t != tic}
        if not rest:
            continue
        gain = len(_intersect(rest)) - len(common)
        if gain > best_gain:
            best, best_gain = tic, gain
    return best


def _resolve_weights(
    entries: dict[str, HoldingEntry],
    prices: pd.DataFrame,
    base: str,
    series: dict[str, _Series],
    period: str,
    fetch: Callable[[str, str], tuple[pd.Series, str] | None],
) -> tuple[pd.Series, str]:
    """Normalise shares / percentages / amounts into weights summing to 1.

    Shares are valued at the close on the **last aligned date** — not at cost, and
    not at today's live price if today did not survive alignment.
    """
    tickers = list(prices.columns)
    if not tickers:
        return pd.Series(dtype=float), "no holdings"
    last_date = prices.index[-1]
    last = prices.iloc[-1]

    units = {entries[t].unit for t in tickers}
    if units == {Unit.PERCENT}:
        raw = pd.Series({t: entries[t].quantity for t in tickers}, dtype=float)
        total = float(raw.sum())
        note = f"as given, renormalised from {total:.1f}% to 100%"
        return raw / total, f"percentages you typed, {note}"

    values: dict[str, float] = {}
    notes: list[str] = []
    for t in tickers:
        e = entries[t]
        if e.unit is Unit.SHARES:
            values[t] = e.quantity * float(last[t])
        elif e.unit is Unit.AMOUNT:
            amt_cur = (e.currency or base).upper()
            if amt_cur == base:
                values[t] = e.quantity
            else:
                fx, fx_name = _fx_series(amt_cur, base, period, fetch)
                rate = float(fx.reindex(prices.index).ffill().iloc[-1]) if fx is not None else float("nan")
                if not np.isfinite(rate):
                    values[t] = float("nan")
                    notes.append(f"{t}: amount given in {amt_cur} but no {amt_cur}->{base} rate; excluded from weights")
                else:
                    values[t] = e.quantity * rate
                    notes.append(f"{t}: amount converted {amt_cur}->{base} at {rate:.4f} ({fx_name})")
        else:  # a percentage inside an otherwise value-sized basket
            values[t] = float("nan")
            notes.append(f"{t}: percentage in a value-sized basket; excluded from weights")

    ser = pd.Series(values, dtype=float)
    ser = ser[ser.notna()]
    if ser.empty or float(ser.sum()) <= 0:
        eq = pd.Series(1.0 / len(tickers), index=tickers)
        return eq, "could not value any position; fell back to EQUAL WEIGHT"
    weights = ser / float(ser.sum())
    weights = weights.reindex(tickers).fillna(0.0)
    basis = (
        f"share counts valued at the {last_date:%Y-%m-%d} close in {base} "
        f"(a snapshot — not what you paid, and it moves with the market)"
    )
    if notes:
        basis += "; " + "; ".join(notes)
    return weights, basis


def _empty_panel(
    base: str,
    excluded: Iterable[Exclusion],
    resolutions: Iterable[Resolution],
    assumptions: Iterable[str],
    reason: str,
) -> PricePanel:
    return PricePanel(
        prices=pd.DataFrame(),
        returns=pd.DataFrame(),
        weights=pd.Series(dtype=float),
        base_currency=base,
        coverage=pd.DataFrame(columns=["own_days", "own_start", "own_end", "days_lost_to_alignment"]),
        excluded=tuple(excluded),
        resolutions=tuple(resolutions),
        assumptions=tuple(assumptions),
        usable=False,
        unusable_reason=reason,
    )
