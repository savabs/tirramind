"""The HTTP boundary for the portfolio audit: paste in, structured facts out.

This module computes NOTHING. ``agent.portfolio.audit`` already assembles the
whole report from the tested sibling modules; this file owns three things and
only three:

1. **Serialisation.** The ``Audit`` object into JSON a page can lay out —
   sections with their numbers, their window and their sample size, never a
   pre-rendered paragraph the page has to parse back apart.
2. **The production limits.** Rate limit per IP, a cap on the paste, a timeout,
   an admission cap, and a 429/413/504 body a human can read. A free public
   demo that shares one yfinance quota with every visitor falls over on the
   first abusive client unless those exist before it is exposed.
3. **Saying what it does with the input.** The holdings text is the most
   sensitive thing a visitor will ever hand us. It is read from the socket into
   a local, passed to ``audit``, and dropped when the request ends. It is never
   written to disk, never logged, and never stored. Every response repeats that
   in :data:`PRIVACY`, so the page can display it rather than us claiming it in
   a README nobody opens.

WHAT IS AND IS NOT ON THE WIRE
------------------------------
Every string in a response is either (a) authored here and scanned by
:func:`advice_words_in` in the test suite, or (b) copied verbatim from a
sibling module whose own tests already scan it. Nothing on the wire
recommends, suggests, forecasts or implies an action, because a sentence that
did would make this a regulated investment recommendation (SEBI Investment
Adviser) and we hold no such registration. Facts about the past are the entire
product, and the reason it is legal for us to publish it.

Three disclosures ride on EVERY successful response, in :data:`DISCLOSURES`,
as first-class fields rather than a footnote the page can choose to omit:
today's weights applied backwards is a counterfactual and not what was earned;
only what is still held is visible; every figure carries its window and n.

WHERE THIS MISLEADS
-------------------
* **The timeout does not cancel the work.** ``concurrent.futures`` cannot kill
  a running thread. On a timeout the client gets 504 and the audit keeps
  running to completion in the background — which is deliberate, because it
  warms ``holdings``' disk cache and the retry is then fast. The admission slot
  is held until that thread finishes, so a timed-out request still counts
  against concurrency. Anything else would let a slow client multiply load.
* **The rate limiter is in-process and in-memory.** Correct for one
  ``ThreadingHTTPServer`` process, which is what this is. Behind more than one
  process it is bypassable per process, and it would have to move to a shared
  store. It is also keyed on the socket peer address, so every visitor behind
  one NAT shares one bucket.
* **``X-Forwarded-For`` is deliberately NOT trusted.** A client can set it
  freely, so honouring it would hand any caller an unlimited supply of rate
  limit buckets. Behind a real proxy this therefore rate-limits the proxy.
  Fixing that needs a trusted-proxy list, which is a deployment fact this
  module does not have.
* **One file is served from disk, chosen by exact path match.** ``/`` and
  ``/index.html`` map to one configured file. There is no static directory and
  no path joining from user input, so there is no traversal to get wrong.

Usage::

    PYTHONPATH=. .venv/bin/python -m agent.portfolio.server --port 8800
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agent.portfolio.audit import audit as run_audit
from agent.portfolio.audit import render_text
from agent.portfolio.holdings import parse_holdings

log = logging.getLogger(__name__)

#: Bumped by hand when the response SHAPE changes, so a page pinned to a shape
#: can tell. It is not a git revision: this module has no business shelling out
#: to git, and a version that only changes when the contract changes is the
#: more useful of the two.
API_VERSION = "1"

#: Where the sibling agent's page lives. Exactly one file is served, by exact
#: path match — see the module docstring on why there is no static directory.
DEFAULT_PAGE = Path(__file__).resolve().parents[2] / "products" / "audit" / "index.html"


# ---------------------------------------------------------------------------
# Limits. Every one of these is a cap that keeps one client from taking the
# demo down for everyone, and every one is readable at /api/health so the page
# can state the limit instead of discovering it with a 413.
# ---------------------------------------------------------------------------


def _int_env(name: str, default: int) -> int:
    """An int from the environment that never raises on junk."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    return value if value > 0 else default


#: Holdings this will price. Forty lines is a large real book; past that the
#: correlation work is O(n^2) on a shared free quota, and the caller is either
#: a professional (who should run the library) or a stress test.
MAX_HOLDINGS = _int_env("TIRRA_AUDIT_MAX_HOLDINGS", 40)

#: Non-blank lines accepted before anything is parsed. This is the cap that
#: refuses a 10,000-line paste in microseconds, ahead of MAX_HOLDINGS, because
#: the point is not to chew through it and then decline.
MAX_LINES = _int_env("TIRRA_AUDIT_MAX_LINES", 200)

#: Request body ceiling. Forty holdings is well under a kilobyte; 32 KB leaves
#: room for a messy broker export without leaving room for an upload.
MAX_BODY_BYTES = _int_env("TIRRA_AUDIT_MAX_BODY_BYTES", 32 * 1024)

#: An oversized body is drained into nothing up to here so the 413 reaches the
#: client as a sentence rather than a connection reset — see
#: :meth:`AuditHandler._discard`. Past this the connection is simply closed.
MAX_DISCARD_BYTES = _int_env("TIRRA_AUDIT_MAX_DISCARD_BYTES", 4 * 1024 * 1024)

#: How long a caller waits before we hand back a sentence instead of a spinner.
#: A first fetch of seven uncached Indian tickers plus the benchmark runs a few
#: seconds; a cold cache with retries can run long. See the module docstring on
#: what happens to the work after this fires.
AUDIT_TIMEOUT_S = _int_env("TIRRA_AUDIT_TIMEOUT_S", 45)

#: Audits running at once. yfinance is one shared free quota, so this is the
#: real protection; the rate limiter only shapes who gets to queue for it.
AUDIT_CONCURRENCY = _int_env("TIRRA_AUDIT_CONCURRENCY", 4)

#: Per-IP buckets: a short one so a loop cannot spin, a long one so a patient
#: loop cannot either. Sized for a person trying a few variants of their book.
RATE_BURST_CALLS = _int_env("TIRRA_AUDIT_RATE_BURST", 5)
RATE_BURST_WINDOW_S = _int_env("TIRRA_AUDIT_RATE_BURST_WINDOW_S", 60)
RATE_HOUR_CALLS = _int_env("TIRRA_AUDIT_RATE_HOUR", 30)
RATE_HOUR_WINDOW_S = _int_env("TIRRA_AUDIT_RATE_HOUR_WINDOW_S", 3600)

#: A ticker a caller may name as the benchmark. Deliberately narrow: this
#: string reaches a network client, and `^NSEI`-style index symbols are the
#: only reason punctuation is allowed at all.
BENCHMARK_RE = re.compile(r"^[A-Za-z0-9.^=&:-]{1,24}$")


# ---------------------------------------------------------------------------
# Authored copy. Everything a visitor reads that did not come from a sibling
# module is here, in one block, so the advice scan in the tests has one place
# to point at and a reviewer has one place to read.
# ---------------------------------------------------------------------------

#: What is done with the pasted holdings, returned on every response so the
#: page states it rather than us asserting it somewhere nobody looks.
PRIVACY: Mapping[str, Any] = {
    "holdings_stored": False,
    "holdings_logged": False,
    "accounts": False,
    "cookies": False,
    "analytics": False,
    "statement": (
        "What you pasted was held in memory for this one request and dropped when it ended. "
        "It was not written to disk, not written to any log, and not stored anywhere. There is "
        "no account, no cookie and no analytics on this page. Daily closing prices for the "
        "tickers are cached on the server; your list of them is not."
    ),
}

#: The three limits that must be legible ON the result, not behind a link. A
#: visitor who later works out that we overstated will not come back, and they
#: would be right not to. Keyed so the page can place each one deliberately.
DISCLOSURES: tuple[Mapping[str, str], ...] = (
    {
        "key": "weights_applied_backwards",
        "title": "These are today's weights, applied backwards",
        "body": (
            "There is one weight vector in this report: the shape of the book as pasted. Every "
            "earlier window is measured with the share counts those weights imply. So this is a "
            "fixed-allocation counterfactual — 'this basket, over that period' — and it is NOT "
            "what you earned. Your purchase dates, additions and withdrawals are not in a "
            "holdings list, so the return you actually got cannot be computed from one."
        ),
    },
    {
        "key": "survivorship",
        "title": "Only what you still hold is visible here",
        "body": (
            "A pasted book is a list of what survived. Anything closed out is absent from every "
            "figure below, at a gain or at a loss alike, and the closures are usually the losses. "
            "That absence flatters this book against the index by an amount no number here "
            "measures. No correction is applied and none is claimed."
        ),
    },
    {
        "key": "window_and_sample",
        "title": "Every figure carries its window and its sample size",
        "body": (
            "Each block below prints the dates it covers and how many observations it had, "
            "because three modules align the same history slightly differently and a figure "
            "without its window is not checkable. Numbers from two different blocks are not "
            "arithmetic with each other. Rolling windows overlap, so a tally across them is one "
            "history looked at many times, and the count of non-overlapping windows is printed "
            "beside it."
        ),
    },
)

#: Plain-English bodies for every refusal. Kept together because a stack trace
#: reaching a visitor is the failure this dictionary exists to make impossible,
#: and because every one of these strings is scanned by the tests.
MESSAGES: Mapping[str, str] = {
    "rate_limited": (
        "More audits came from your address in a short period than this endpoint accepts. The "
        "price data behind it is a single shared free quota, so the limit keeps one visitor from "
        "taking the page down for everyone. Wait the number of seconds in retry_after and send "
        "the same paste again."
    ),
    "busy": (
        "Every audit slot on this server is in use right now. Nothing is wrong with your paste. "
        "Wait the number of seconds in retry_after and send it again."
    ),
    "timeout": (
        "This audit passed the time limit and was stopped before it finished. The usual cause is "
        "a first-time ticker whose prices are not cached yet. The fetch continues on the server "
        "and a second attempt in a minute is normally fast."
    ),
    "body_too_large": (
        "That paste is larger than this endpoint accepts. Send the holdings themselves — one "
        "line per position, a symbol and a quantity — rather than a whole exported file."
    ),
    "too_many_lines": "That paste has more lines than this endpoint reads.",
    "too_many_holdings": "That paste has more positions than this endpoint prices.",
    "no_body": (
        'No request body arrived. POST JSON of the form {"holdings": "RELIANCE 40\\nTCS 12"} '
        "with a Content-Type of application/json."
    ),
    "bad_json": ('That body is not valid JSON. The form this endpoint reads is {"holdings": "RELIANCE 40\\nTCS 12"}.'),
    "holdings_missing": (
        'The body has no "holdings" field with text in it. One line per position, a symbol and a quantity: RELIANCE 40.'
    ),
    "holdings_not_string": 'The "holdings" field has to be text, one line per position.',
    "nothing_parsed": (
        "No line in that paste read as a position. Each line wants a symbol and a quantity, like "
        "RELIANCE 40 or TCS 12. The lines that could not be read are listed in unreadable."
    ),
    "bad_benchmark": (
        'That "benchmark" value is not a ticker this endpoint will pass on. Leave it out and '
        "the benchmark is chosen by rule from the book's own market, and the rule is returned."
    ),
    "not_found": "No such endpoint. This server answers GET /, GET /api/health and POST /api/audit.",
    "method_not_allowed": "That method is not accepted on this path.",
    "bad_request": (
        "This server could not read that as an HTTP request. Nothing was audited. The endpoints "
        "are GET /, GET /api/health and POST /api/audit."
    ),
    "internal": (
        "Something inside the audit failed. The details are in the server log under the error_id "
        "below; nothing about your holdings is in that log. Sending the same paste again is safe."
    ),
    "page_missing": (
        "The page file is not present on this server, so there is nothing to show at this "
        "address. The API underneath it is running: POST /api/audit works, and GET /api/health "
        "reports the limits."
    ),
    "unusable": (
        "The audit ran but could not price this book well enough to measure it. The reason from "
        "each block is in the response, and nothing was guessed to fill a gap."
    ),
}


# ---------------------------------------------------------------------------
# Rate limiting and admission
# ---------------------------------------------------------------------------


class RateLimiter:
    """In-memory sliding window keyed by an arbitrary string.

    Same shape as ``agent.brief_server._RateLimiter`` and correct under the
    same single-process condition. Kept local rather than imported because that
    module pulls in the payments and delivery stack on import, and this server
    must start with nothing but the portfolio package present.

    WHERE THIS MISLEADS
        Memory grows with the number of distinct keys seen. Stale buckets are
        dropped opportunistically on each call and wholesale once the map
        passes :attr:`max_keys`, which bounds it without a background thread.
    """

    def __init__(self, max_calls: int, window_s: float, *, max_keys: int = 10_000) -> None:
        self.max_calls = max_calls
        self.window_s = float(window_s)
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """``(allowed, retry_after_seconds)``. Consumes one call when allowed."""
        now = time.time() if now is None else now
        cutoff = now - self.window_s
        with self._lock:
            if len(self._hits) > self.max_keys:
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > cutoff}
            hits = [h for h in self._hits.get(key, []) if h > cutoff]
            if len(hits) >= self.max_calls:
                self._hits[key] = hits
                return False, max(0.0, hits[0] + self.window_s - now)
            hits.append(now)
            self._hits[key] = hits
            return True, 0.0


_BURST_LIMITER = RateLimiter(RATE_BURST_CALLS, RATE_BURST_WINDOW_S)
_HOUR_LIMITER = RateLimiter(RATE_HOUR_CALLS, RATE_HOUR_WINDOW_S)

#: Admission. Acquired non-blocking, released only when the audit thread
#: actually finishes — see the module docstring on timeouts.
_AUDIT_SLOTS = threading.BoundedSemaphore(AUDIT_CONCURRENCY)

#: Two spare threads over the admission cap so an admitted request never waits
#: on the pool, and the pool is never the thing that queues.
_POOL = ThreadPoolExecutor(max_workers=AUDIT_CONCURRENCY + 2, thread_name_prefix="audit")


# ---------------------------------------------------------------------------
# JSON sanitising. Section.data comes straight from sibling modules and holds
# numpy scalars, pandas timestamps, dataclasses and NaN. json.dumps will emit
# `NaN`, which is not JSON and which JSON.parse rejects, so nothing reaches
# json.dumps without passing through here.
# ---------------------------------------------------------------------------

_MAX_DEPTH = 12


def _num(x: Any) -> float | int | None:
    """A JSON-safe number, or None. NaN and infinity become None, never 0.0."""
    if x is None or isinstance(x, bool):
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if isinstance(x, int):
        return int(x)
    return value


def jsonable(obj: Any, depth: int = 0) -> Any:
    """Anything a sibling module put in a ``data`` mapping, made JSON-safe.

    WHERE THIS MISLEADS
        A DataFrame or Series is replaced by its shape, not its contents. The
        only ones that reach here are ``RollingGap.series`` and the panel's
        coverage frame, both of which are per-window tables this API summarises
        instead of shipping. An unrecognised object becomes ``str(obj)``, so a
        new sibling field arrives as a readable string rather than a 500.
    """
    if depth > _MAX_DEPTH:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return _num(obj)
    if isinstance(obj, Enum):
        return jsonable(obj.value, depth + 1)
    # pd.Timestamp subclasses datetime, so this catches it too. A midnight
    # timestamp is a trading DATE and is emitted as one; anything with a real
    # time of day keeps it.
    if isinstance(obj, dt.datetime):
        return obj.date().isoformat() if (obj.hour, obj.minute, obj.second) == (0, 0, 0) else obj.isoformat()
    if isinstance(obj, dt.date):
        return obj.isoformat()
    if hasattr(obj, "shape") and hasattr(obj, "columns"):  # DataFrame
        return {"_table_omitted": True, "n_rows": int(getattr(obj, "shape", (0,))[0])}
    if hasattr(obj, "dtype") and hasattr(obj, "item") and getattr(obj, "shape", None) == ():
        try:
            return jsonable(obj.item(), depth + 1)
        except Exception:  # noqa: BLE001
            return str(obj)
    if hasattr(obj, "shape") and hasattr(obj, "index"):  # Series
        return {"_series_omitted": True, "n": int(len(obj))}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name, None), depth + 1) for f in dataclass_fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(str(v) for v in obj)
    if isinstance(obj, (list, tuple)) or (isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))):
        return [jsonable(v, depth + 1) for v in obj]
    return str(obj)


def _date(x: Any) -> str | None:
    if x is None:
        return None
    try:
        return f"{x:%Y-%m-%d}"
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# Serialising the Audit. One function per sibling value object, each carrying
# that object's own window and sample size.
# ---------------------------------------------------------------------------


def _benchmark_window(w: Any) -> dict[str, Any]:
    """``benchmark.Window`` — the window rule is part of the window."""
    return {
        "label": getattr(w, "label", ""),
        "start": _date(getattr(w, "start", None)),
        "end": _date(getattr(w, "end", None)),
        "kind": getattr(w, "kind", ""),
        "rule": getattr(w, "rule", ""),
        "complete": bool(getattr(w, "complete", True)),
        "years": _num(getattr(w, "years", None)),
    }


def _attribution_window(w: Any) -> dict[str, Any]:
    """``attribution.Window`` — a different alignment of the same history."""
    return {
        "label": getattr(w, "label", ""),
        "start": _date(getattr(w, "start", None)),
        "end": _date(getattr(w, "end", None)),
        "n_days": int(getattr(w, "n_days", 0) or 0),
        "years": _num(getattr(w, "years", None)),
        "text": str(w),
    }


def _series_stats(s: Any) -> dict[str, Any]:
    return {
        "n_days": int(getattr(s, "n_days", 0) or 0),
        "total_return": _num(getattr(s, "total_return", None)),
        "annualised_return": _num(getattr(s, "annualised_return", None)),
        "volatility": _num(getattr(s, "volatility", None)),
        "return_per_vol": _num(getattr(s, "return_per_vol", None)),
        "return_per_vol_reason": getattr(s, "return_per_vol_reason", ""),
        "annualisation_is_extrapolation": bool(getattr(s, "annualisation_is_extrapolation", False)),
    }


def _headline(a: Any) -> dict[str, Any]:
    """Section 1: the book against the alternative of having done nothing.

    ``published`` mirrors ``Audit.headline_publishable`` exactly. When it is
    False the numbers are still absent from this block and the reason is in
    ``withheld_reason``: one window is a selected window, and this API refuses
    to hand a page a number with nothing to check it against, for the same
    reason ``render_text`` refuses to print one.
    """
    fw = a.full_window
    if fw is None:
        return {
            "published": False,
            "withheld_reason": a.picking_reason or "no full-history window was produced",
        }
    if not a.headline_publishable:
        return {
            "published": False,
            "withheld_reason": (
                "The full-history figures were computed but no second window was, so there is "
                "nothing to check them against and they are withheld rather than footnoted. "
                + (a.records_reason or a.multi_reason or "no multi-window check was available")
            ),
        }
    return {
        "published": True,
        "withheld_reason": "",
        "window": _benchmark_window(fw.window),
        "n_observations": int(fw.n_obs),
        "years": _num(fw.years),
        "book": {
            "total_return": _num(fw.port_total),
            "annualised_return": _num(fw.port_annual),
            "volatility": _num(fw.port_vol),
            "return_per_vol": _num(fw.port_return_per_vol),
        },
        "benchmark": {
            "ticker": fw.benchmark_ticker,
            "total_return": _num(fw.bench_total),
            "annualised_return": _num(fw.bench_annual),
            "volatility": _num(fw.bench_vol),
            "return_per_vol": _num(fw.bench_return_per_vol),
        },
        "gap_total": _num(fw.gap_total),
        "gap_annualised": _num(fw.gap_annual),
        "beat": bool(getattr(fw, "beat", False)),
        "n_holdings_behind_benchmark": int(getattr(fw, "n_legs_behind", 0) or 0),
        "n_holdings_held": int(getattr(fw, "n_legs_held", 0) or 0),
        "rebalanced_to_these_weights_total": _num(getattr(fw, "constant_mix_total", None)),
        "return_per_vol_note": (
            "Annualised return divided by annualised volatility, with no risk-free rate taken "
            "off. It is not a Sharpe ratio."
        ),
        "notes": list(getattr(fw, "notes", ())),
    }


def _rolling(r: Any) -> dict[str, Any]:
    """One rolling-window family: the record the headline has to travel with."""
    return {
        "length": r.length_label,
        "length_years": _num(r.length_years),
        "benchmark_ticker": r.benchmark_ticker,
        "n_windows": int(r.n_windows),
        "n_independent_windows": _num(r.n_independent),
        "beat_count": int(r.beat_count),
        "beat_fraction": _num(r.beat_fraction),
        "median_gap": _num(r.median_gap),
        "mean_gap": _num(r.mean_gap),
        "best_gap": _num(r.best_gap),
        "best_window": r.best_window,
        "worst_gap": _num(r.worst_gap),
        "worst_window": r.worst_window,
        "statement": r.headline,
        "caveats": list(getattr(r, "caveats", ())),
    }


def _holding_result(h: Any, bench: str) -> dict[str, Any]:
    return {
        "ticker": h.ticker,
        "weight_start": _num(h.weight_start),
        "weight_end": _num(h.weight_end),
        "total_return": _num(h.total_return),
        "benchmark_return": _num(h.benchmark_return),
        "excess_return": _num(h.excess_return),
        "contribution_to_gap": _num(h.contribution),
        "n_days": int(h.n_days),
        "scrubbed_days": int(getattr(h, "scrubbed_days", 0) or 0),
        "beat_benchmark": bool(h.beat_benchmark),
        "statement": h.statement(bench),
        "notes": list(getattr(h, "notes", ())),
    }


def _decisions(a: Any) -> dict[str, Any]:
    """Section 2: which line moved the gap, worst first.

    Ordering is ``attribution``'s own — contribution ascending — and is not
    re-sorted here. The position that cost the most is first because that is
    the fact the holder did not already know, and re-ranking it in a
    presentation layer would quietly change what the block says.
    """
    attr = a.attribution
    if attr is None or not getattr(attr, "usable", False):
        return {
            "computed": False,
            "reason": a.attribution_reason or getattr(attr, "unusable_reason", "") or "not computed",
            "holdings": [],
            "summary": jsonable(a.summary),
        }
    bench = attr.benchmark_ticker
    return {
        "computed": True,
        "reason": "",
        "window": _attribution_window(attr.window),
        "basis": attr.basis,
        "basis_note": attr.basis_note,
        "benchmark_ticker": bench,
        "book": _series_stats(attr.book),
        "benchmark": _series_stats(attr.benchmark),
        "gap_total": _num(attr.gap_total),
        "gap_annualised": _num(attr.gap_annualised),
        "n_beat": int(attr.n_beat),
        "n_trailed": int(attr.n_trailed),
        "holdings": [_holding_result(h, bench) for h in attr],
        "contribution_sum": _num(attr.contribution_sum),
        "residual": _num(attr.residual),
        "residual_explanation": attr.residual_explanation,
        "excluded": [jsonable(e) for e in getattr(attr, "excluded", ())],
        "notes": list(getattr(attr, "notes", ())),
        "summary": jsonable(a.summary),
    }


def _multi_window(a: Any) -> dict[str, Any]:
    """The same book over fixed calendar windows: does the sign hold up?

    Contradicting windows come first, inheriting the order
    ``MultiWindowResult.headline_qualifier`` already enforces. A page that
    rendered the agreeing windows first would be performing the cherry-pick
    this block exists to prevent.
    """
    m = a.multi
    if m is None:
        return {"computed": False, "reason": a.multi_reason or "not computed", "windows": []}

    def _row(wa: Any, agrees: bool | None) -> dict[str, Any]:
        return {
            "window": _attribution_window(wa.window),
            "usable": bool(wa.usable),
            "skipped_reason": wa.skipped_reason or "",
            "gap_total": _num(wa.gap_total),
            "gap_annualised": _num(wa.gap_annualised),
            "agrees_with_full_window": agrees,
        }

    contradicting = list(m.windows_contradicting)
    agreeing = list(m.windows_agreeing)
    ordered = [_row(w, False) for w in contradicting] + [_row(w, True) for w in agreeing]
    seen = {r["window"]["label"] for r in ordered}
    ordered += [_row(w, None) for w in m.windows if w.window.label not in seen]
    return {
        "computed": True,
        "reason": "",
        "method": m.method,
        "method_note": m.method_note,
        "benchmark_ticker": m.benchmark_ticker,
        "n_windows": len(m.windows),
        "n_usable_windows": len(m.usable_windows),
        "n_independent_windows": int(m.independent_windows),
        "overlap_note": m.overlap_note,
        "direction_consistent": bool(m.direction_consistent),
        "qualifier": m.headline_qualifier,
        "full_window_gap_total": _num(getattr(m.full_window, "gap_total", None)),
        "windows": ordered,
        "notes": list(getattr(m, "notes", ())),
        "excluded": [str(e) for e in getattr(m, "excluded", ())],
    }


def _clusters(panel: Any) -> dict[str, Any]:
    """The same-bet groups as fields rather than sentences.

    ``audit``'s overlap section carries the finished sentences and only a group
    count in its ``data``, and a page needs the members, the weight and the
    correlation as numbers to lay them out. This calls
    ``overlap.correlation_clusters`` with the identical arguments ``audit``
    passes it — the same panel, the same weights, both defaults — so the two
    cannot disagree; it is the same pure function on the same input. On any
    failure this degrades to the section's own sentences and says so, rather
    than the page losing the finding.
    """
    try:
        from agent.portfolio import overlap as ov  # noqa: PLC0415

        weights = {str(k): float(v) for k, v in panel.weights.items()}
        cs = ov.correlation_clusters(panel, weights)
    except Exception as exc:  # noqa: BLE001
        return {"computed": False, "reason": f"{type(exc).__name__}: {exc}", "groups": []}
    stable = getattr(cs, "stable_between", None)
    return {
        "computed": True,
        "reason": "",
        "n_groups": int(cs.n_groups),
        "groups": [
            {
                "members": list(c.members),
                "size": int(c.size),
                "weight_of_book": _num(c.weight),
                "mean_correlation": _num(c.mean_correlation),
                "min_correlation": _num(c.min_correlation),
                "min_correlation_ci": [_num(c.min_correlation_ci[0]), _num(c.min_correlation_ci[1])],
                "window": {
                    "start": _date(c.window.start),
                    "end": _date(c.window.end),
                    "n_days": int(c.window.n_days),
                },
                "surprise": _num(c.surprise),
                "statement": c.statement(),
            }
            for c in cs
        ],
        "singletons": list(getattr(cs, "singletons", ())),
        "stable_between": None if not stable else [_num(stable[0]), _num(stable[1])],
        "notes": list(getattr(cs, "notes", ())),
        "excluded": [str(e) for e in getattr(cs, "excluded", ())],
    }


def _section(s: Any) -> dict[str, Any]:
    """A structural block, computed or explicitly not, with its reason.

    ``lines`` is kept beside ``data`` on purpose. The numbers are what the page
    lays out; the lines are the wording those numbers were already tested
    against, and a page that needs a sentence should use that one rather than
    invent a new one over the same figure.
    """
    return {
        "key": s.key,
        "title": s.title,
        "computed": bool(s.computed),
        "reason": s.reason,
        "lines": [line.strip() for line in s.lines],
        "data": jsonable(s.data),
    }


def audit_payload(a: Any, *, include_text: bool = False, holdings_text: str = "") -> dict[str, Any]:
    """The whole ``Audit`` as JSON-safe structured data.

    Field order here is render order: what was read, the window, the
    benchmark and its rule, the headline with its rolling record attached,
    the two weighting bases side by side, the per-holding decisions, the
    calendar-window check, the structural explanation, then the limits. A page
    can reorder it; this is the order the report argues in.
    """
    panel = a.panel
    structure = {s.key: _section(s) for s in a.structure}
    if "overlap" in structure:
        structure["overlap"]["clusters"] = _clusters(panel)
    payload: dict[str, Any] = {
        "ok": True,
        "version": API_VERSION,
        "usable": bool(a.usable),
        "unusable_reason": panel.unusable_reason if not a.usable else "",
        "privacy": dict(PRIVACY),
        "disclosures": [dict(d) for d in DISCLOSURES],
        "input": {
            "positions_read": len(a.holdings.entries),
            "positions_priced": len(panel.tickers),
            "unit_mode": a.holdings.unit_mode,
            "positions": [
                {
                    "symbol": e.symbol,
                    "quantity": _num(e.quantity),
                    "unit": e.unit.value if isinstance(e.unit, Enum) else str(e.unit),
                    "line_no": int(e.line_no),
                    "currency": e.currency,
                }
                for e in a.holdings.entries
            ],
            "unreadable": [
                {"line_no": int(p.line_no), "raw": p.raw, "reason": p.reason} for p in a.holdings.unreadable
            ],
            "assumptions": list(a.holdings.assumptions),
            "resolutions": [
                {"typed": r.typed, "resolved": r.resolved, "currency": r.currency, "note": r.note}
                for r in getattr(panel, "resolutions", ())
            ],
            "not_measured": [
                {"symbol": e.symbol, "reason": e.reason, "resolved": e.resolved} for e in getattr(panel, "excluded", ())
            ],
        },
        "window": {
            "text": panel.window,
            "start": _date(panel.start),
            "end": _date(panel.end),
            "n_trading_days": int(panel.n_days),
            "n_daily_returns": int(panel.n_returns),
            "currency": panel.base_currency,
            "weight_basis": panel.weight_basis,
            "notes": list(getattr(panel, "window_note", ())),
        },
        "weights": {str(k): _num(v) for k, v in panel.weights.items()},
        "benchmark": {
            "ticker": a.benchmark_ticker,
            "why": a.benchmark_why,
            "also_applies_not_rendered": [{"ticker": t, "why": w} for t, w in getattr(a, "other_benchmarks", ())],
        },
        "notes": list(a.notes),
        "fx": list(getattr(panel, "fx_report", ())),
        "panel_assumptions": list(getattr(panel, "assumptions", ())),
        "look_through": (
            _look_through(holdings_text)
            if holdings_text
            else {"computed": False, "reason": "caller did not pass the original paste"}
        ),
        "headline": _headline(a),
        "rolling_record": [_rolling(r) for r in a.usable_records],
        "rolling_record_reason": a.records_reason,
        "basis_check": jsonable(a.basis_check),
        "decisions": _decisions(a),
        "calendar_windows": _multi_window(a),
        "structure": structure,
        "survivorship": a.survivorship,
        "limitations": list(a.limitations),
    }
    if include_text:
        payload["report_text"] = render_text(a)
    return payload


def _look_through(holdings_text: str) -> dict:
    """The look-through block: what the book really holds once index funds unpack.

    Leads the response because it is the only finding that is both simple and
    surprising. Everything else in this payload describes positions the user
    chose; this one tells them about exposure they did not.

    Never raises into the response. A look-through failure must not take the
    rest of the audit down with it — the section reports ``computed: false``
    with a reason, which is also what it does for a book of only direct stocks.
    """
    try:
        from agent.lookthrough.combine import look_through
    except Exception as exc:  # noqa: BLE001
        return {"computed": False, "reason": f"look-through unavailable: {type(exc).__name__}"}
    try:
        lt = look_through(holdings_text)
    except Exception as exc:  # noqa: BLE001
        log.exception("look_through failed")
        return {"computed": False, "reason": f"could not look through this book: {type(exc).__name__}"}
    if not getattr(lt, "computed", False):
        return {"computed": False, "reason": getattr(lt, "not_computed_reason", "") or "nothing to unpack"}

    # Rank by SURPRISE — how much of the holding the user did not list — rather
    # than by size. A 2.1% position they never listed is more interesting than a
    # large one that moved 0.1pp.
    def surprise(line) -> float:
        listed = line.listed if line.listed is not None else 0.0
        return float(line.actual) - float(listed)

    ranked = sorted(lt.lines, key=surprise, reverse=True)
    rows = [
        {
            "symbol": ln.symbol,
            "name": ln.name,
            "listed": _num(ln.listed) if ln.listed is not None else None,
            "actual": _num(ln.actual),
            "added": _num(surprise(ln)),
            "via": [
                {
                    "source": c.source,
                    "index": c.index,
                    "fund_weight": _num(c.fund_weight),
                    "constituent_weight": _num(c.constituent_weight),
                    "weight": _num(c.weight),
                }
                for c in (ln.via or ())
            ],
            "kind": ln.kind,
            "note": ln.note,
        }
        for ln in ranked
    ]
    top3 = sum(float(ln.actual) for ln in sorted(lt.lines, key=lambda x: -float(x.actual))[:3])
    biggest = ranked[0] if ranked else None
    return {
        "computed": True,
        "n_listed": lt.n_listed,
        "n_unpacked": lt.n_unpacked,
        "n_companies": lt.n_companies,
        "headline": (
            f"You listed {lt.n_listed} positions. Looking through {lt.n_unpacked} of them, "
            f"you hold {lt.n_companies} companies."
        ),
        "biggest_surprise": (
            {
                "symbol": biggest.symbol,
                "listed": _num(biggest.listed) if biggest.listed is not None else None,
                "actual": _num(biggest.actual),
                "added": _num(surprise(biggest)),
            }
            if biggest is not None and surprise(biggest) > 0
            else None
        ),
        "top3_share": _num(top3),
        "rows": rows,
        "funds": [
            {
                "label": f.label,
                "weight": _num(f.weight),
                "index": f.index,
                "unpacked": f.unpacked,
                "reason": f.reason,
                "n_constituents": f.n_constituents,
            }
            for f in (lt.funds or ())
        ],
        "residual": _num(lt.residual),
        "weight_basis": lt.weight_basis,
        "weight_source": lt.weight_source,
        # Carried to the surface on purpose: index weights are computed from
        # free-float market cap, not published, and the user sees that where
        # they see the number rather than in a footer.
        "approximation_notes": list(lt.approximation_notes or ()),
        "assumptions": list(lt.assumptions or ()),
        "exclusions": [{"symbol": sym, "reason": why} for sym, why in (lt.exclusions or ())],
    }


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------


def _count_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


class AuditHandler(BaseHTTPRequestHandler):
    """The three endpoints, and nothing else.

    Every handler body is wrapped so that no exception can reach the socket as
    a traceback: :meth:`do_GET` and :meth:`do_POST` catch everything, log it
    with an id, and return :data:`MESSAGES`\\ ``["internal"]`` with that id.
    """

    server_version = f"tirra-audit/{API_VERSION}"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    #: Set by :func:`serve`. The one file served at "/".
    page_path: Path = DEFAULT_PAGE

    # -- plumbing ---------------------------------------------------------
    def _send(self, code: int, ctype: str, body: bytes, extra: Mapping[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the visitor closed the tab; not an error worth a log line

    def _json(self, code: int, obj: Mapping[str, Any], extra: Mapping[str, str] | None = None) -> None:
        body = json.dumps(obj, allow_nan=False, indent=2, sort_keys=False).encode("utf-8")
        self._send(code, "application/json; charset=utf-8", body, extra)

    def _refuse(
        self,
        code: int,
        key: str,
        *,
        detail: str = "",
        extra_fields: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Every non-200 goes through here, so every one is a readable sentence."""
        obj: dict[str, Any] = {
            "ok": False,
            "version": API_VERSION,
            "error": key,
            "message": MESSAGES[key] + (f" {detail}" if detail else ""),
        }
        obj.update(extra_fields or {})
        self._json(code, obj, headers)

    def _client_key(self) -> str:
        """The rate-limit bucket. The socket peer, never a client-set header."""
        try:
            return str(self.client_address[0])
        except Exception:  # noqa: BLE001
            return "unknown"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        sys.stderr.write(f"[audit-server] {fmt % args}\n")

    # -- GET --------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        try:
            path = self.path.split("?", 1)[0]
            if path == "/api/health":
                self._health()
            elif path in ("/", "/index.html"):
                self._page()
            elif path == "/favicon.ico":
                self._send(204, "text/plain", b"")
            else:
                self._refuse(404, "not_found")
        except Exception:  # noqa: BLE001
            self._internal()

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Answer the preflight explicitly.

        The page is served from this same origin and needs no CORS at all. The
        header is here for the case where a reviewer opens the HTML off disk,
        and it is configurable so a deployment can narrow it; the default is
        wide because there is nothing behind this endpoint to protect — no
        auth, no cookie, no session, nothing stored.
        """
        self._send(
            204,
            "text/plain",
            b"",
            {
                "Access-Control-Allow-Origin": os.getenv("TIRRA_AUDIT_CORS_ORIGIN", "*"),
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "600",
            },
        )

    #: Every method this server answers. Returned in ``Allow`` on a 405 so a
    #: client is told what to use instead of guessing.
    ALLOWED_METHODS = "GET, HEAD, POST, OPTIONS"

    def _method_not_allowed(self) -> None:
        """405, as JSON, with the body drained so keep-alive stays in step."""
        self._drain()
        self._refuse(
            405,
            "method_not_allowed",
            detail=f"This server answers {self.ALLOWED_METHODS}.",
            extra_fields={"allowed": self.ALLOWED_METHODS},
            headers={"Allow": self.ALLOWED_METHODS},
        )

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Answer JSON where the stdlib would have answered an HTML page.

        ``BaseHTTPRequestHandler`` calls this for everything it rejects before
        any ``do_*`` method runs: a method with no handler, an over-long request
        line, too many headers. Its default body is an HTML document that also
        skips every header :meth:`_send` sets, so a JSON client got
        ``text/html`` with no ``nosniff`` and no ``no-store`` — and the echoed
        method name in that page is the only place this server ever reflected
        input back. Routing it through :meth:`_refuse` closes both.

        The stdlib's ``message``/``explain`` are deliberately DROPPED rather
        than forwarded: they are the internal wording, and MESSAGES is the only
        copy a visitor is meant to read.
        """
        key = "method_not_allowed" if code in (405, 501) else "not_found" if code == 404 else "bad_request"
        self.close_connection = True
        try:
            self._refuse(
                code,
                key,
                extra_fields={"allowed": self.ALLOWED_METHODS} if key == "method_not_allowed" else None,
                headers={"Allow": self.ALLOWED_METHODS, "Connection": "close"}
                if key == "method_not_allowed"
                else {"Connection": "close"},
            )
        except Exception:  # noqa: BLE001
            # The request was malformed enough that even the status line cannot
            # be written (HTTP/0.9, or the peer is already gone). There is
            # nowhere left to report to, and an exception here would be logged
            # as a crash rather than the bad request it is.
            pass

    def _health(self) -> None:
        self._json(
            200,
            {
                "ok": True,
                "version": API_VERSION,
                "page_present": self.page_path.is_file(),
                "limits": {
                    "max_holdings": MAX_HOLDINGS,
                    "max_lines": MAX_LINES,
                    "max_body_bytes": MAX_BODY_BYTES,
                    "audit_timeout_s": AUDIT_TIMEOUT_S,
                    "concurrent_audits": AUDIT_CONCURRENCY,
                    "rate_limit": {
                        "burst": f"{RATE_BURST_CALLS} per {RATE_BURST_WINDOW_S}s per address",
                        "sustained": f"{RATE_HOUR_CALLS} per {RATE_HOUR_WINDOW_S}s per address",
                    },
                },
                "privacy": dict(PRIVACY),
            },
        )

    def _page(self) -> None:
        path = self.page_path
        if not path.is_file():
            self._refuse(503, "page_missing", extra_fields={"page_path_configured": str(path)})
            return
        try:
            body = path.read_bytes()
        except OSError as exc:
            log.warning("page unreadable at %s: %s", path, exc)
            self._refuse(503, "page_missing", extra_fields={"page_path_configured": str(path)})
            return
        self._send(200, "text/html; charset=utf-8", body)

    # -- POST -------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        try:
            path = self.path.split("?", 1)[0]
            if path != "/api/audit":
                self._drain()
                self._refuse(404, "not_found")
                return
            self._audit()
        except Exception:  # noqa: BLE001
            self._internal()

    def _drain(self) -> None:
        """Read and discard a body so a keep-alive connection stays in step."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if 0 < length <= MAX_BODY_BYTES:
            try:
                self.rfile.read(length)
            except OSError:
                pass

    def _discard(self, length: int) -> bool:
        """Read and throw away ``length`` bytes, up to :data:`MAX_DISCARD_BYTES`.

        This exists so an oversized paste is refused POLITELY. A server that
        answers 413 and closes while the client is still writing gives that
        client a connection reset instead of the sentence — which is what
        happens in a browser, and a reset is not an explanation. So the body is
        drained into nothing (never parsed, never held) up to a ceiling, the
        413 is written, and the client reads it.

        Above the ceiling the connection is closed mid-write instead. At tens
        of megabytes the caller is not a person pasting a portfolio, and
        draining it would be the denial of service the cap exists to stop.

        Returns True when the body was fully drained and a response can be
        delivered cleanly.
        """
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                return False
            remaining -= len(chunk)
        return True

    def _read_body(self) -> bytes | None:
        """The request body, or None after a refusal has already been sent.

        The size decision is made from ``Content-Length`` BEFORE the body is
        read, so an oversized paste is never parsed and never held. See
        :meth:`_discard` for why it is still drained up to a ceiling.
        """
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            # No length and no chunked encoding means no body. A chunked client
            # is read through rfile up to the cap plus one byte, so an overrun
            # is caught the same way.
            if not self.headers.get("Transfer-Encoding"):
                self._refuse(400, "no_body")
                return None
            body = self.rfile.read(MAX_BODY_BYTES + 1)
            if len(body) > MAX_BODY_BYTES:
                self._refuse(
                    413,
                    "body_too_large",
                    extra_fields={"max_body_bytes": MAX_BODY_BYTES},
                    headers={"Connection": "close"},
                )
                self.close_connection = True
                return None
            return body
        try:
            length = int(str(raw_len).strip())
        except (TypeError, ValueError):
            self._refuse(400, "no_body")
            return None
        if length < 0:
            self._refuse(400, "no_body")
            return None
        if length > MAX_BODY_BYTES:
            drained = length <= MAX_DISCARD_BYTES and self._discard(length)
            self._refuse(
                413,
                "body_too_large",
                detail=f"It is {length} bytes and the limit is {MAX_BODY_BYTES}.",
                extra_fields={"max_body_bytes": MAX_BODY_BYTES, "received_bytes": length},
                headers={} if drained else {"Connection": "close"},
            )
            self.close_connection = not drained
            return None
        if length == 0:
            self._refuse(400, "no_body")
            return None
        body = self.rfile.read(length)
        if len(body) < length:  # client hung up mid-body
            self._refuse(400, "no_body")
            self.close_connection = True
            return None
        return body

    def _audit(self) -> None:
        # The rate limit is checked BEFORE the body is read, so a client
        # flooding malformed requests is cut off for the same cost as a client
        # flooding valid ones. Checking it after would leave an unmetered path:
        # every refusal decided inside _read_body would be free to repeat.
        allowed, retry = _BURST_LIMITER.allow(self._client_key())
        if allowed:
            allowed, retry = _HOUR_LIMITER.allow(self._client_key())
        if not allowed:
            wait = int(math.ceil(retry)) or 1
            # Closed rather than drained: a client already over its limit has
            # no claim on our reading its body politely.
            self._refuse(
                429,
                "rate_limited",
                extra_fields={"retry_after": wait},
                headers={"Retry-After": str(wait), "Connection": "close"},
            )
            self.close_connection = True
            return

        body = self._read_body()
        if body is None:
            return

        try:
            parsed_body = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._refuse(400, "bad_json", detail=f"({exc.__class__.__name__})")
            return
        if not isinstance(parsed_body, Mapping):
            self._refuse(400, "bad_json")
            return

        text = parsed_body.get("holdings")
        if text is None or (isinstance(text, str) and not text.strip()):
            self._refuse(400, "holdings_missing")
            return
        if not isinstance(text, str):
            self._refuse(400, "holdings_not_string")
            return

        n_lines = _count_lines(text)
        if n_lines > MAX_LINES:
            self._refuse(
                413,
                "too_many_lines",
                detail=f"It has {n_lines} and the limit is {MAX_LINES}.",
                extra_fields={"lines": n_lines, "max_lines": MAX_LINES},
            )
            return

        benchmark = parsed_body.get("benchmark")
        if benchmark is not None:
            if not isinstance(benchmark, str) or not BENCHMARK_RE.match(benchmark.strip()):
                self._refuse(400, "bad_benchmark")
                return
            benchmark = benchmark.strip()

        # Parse before fetching anything. parse_holdings is pure string work,
        # so the position cap is enforced ahead of the first network call and a
        # 500-name paste costs nobody a quota.
        try:
            parsed = parse_holdings(text)
        except Exception:  # noqa: BLE001
            self._internal()
            return
        if len(parsed.entries) > MAX_HOLDINGS:
            self._refuse(
                413,
                "too_many_holdings",
                detail=f"It has {len(parsed.entries)} and the limit is {MAX_HOLDINGS}.",
                extra_fields={"positions": len(parsed.entries), "max_holdings": MAX_HOLDINGS},
            )
            return
        if not parsed.entries:
            self._refuse(
                422,
                "nothing_parsed",
                extra_fields={
                    "unreadable": [
                        {"line_no": int(p.line_no), "raw": p.raw, "reason": p.reason} for p in parsed.unreadable
                    ][:MAX_LINES],
                },
            )
            return

        include_text = bool(parsed_body.get("include_text"))

        if not _AUDIT_SLOTS.acquire(blocking=False):
            self._refuse(
                503,
                "busy",
                extra_fields={"retry_after": 10},
                headers={"Retry-After": "10"},
            )
            return

        future: Future = _POOL.submit(self._compute, text, benchmark, include_text)
        # Released when the WORK finishes, not when this request returns — a
        # timed-out audit is still running and still holds its slot.
        future.add_done_callback(lambda _f: _AUDIT_SLOTS.release())

        started = time.time()
        try:
            payload = future.result(timeout=AUDIT_TIMEOUT_S)
        except FutureTimeout:
            self._refuse(
                504,
                "timeout",
                extra_fields={"timeout_s": AUDIT_TIMEOUT_S, "retry_after": 60},
                headers={"Retry-After": "60"},
            )
            return
        except Exception:  # noqa: BLE001
            self._internal()
            return

        payload["elapsed_s"] = round(time.time() - started, 2)
        if not payload.get("usable"):
            # A 200 with usable=false, not a 4xx: the audit ran, every block
            # carries its own reason, and the page has something true to show.
            payload["message"] = MESSAGES["unusable"]
        self._json(200, payload)

    def _compute(self, text: str, benchmark: str | None, include_text: bool) -> dict[str, Any]:
        """The audit, on a worker thread. The only place holdings text is used."""
        result = run_audit(text, benchmark=benchmark)
        # `text` is passed on so look-through can re-parse it: it needs the raw
        # paste to tell an index fund from a direct holding, which the finished
        # Audit no longer distinguishes. It stays on this thread and is not
        # retained anywhere — see the privacy note on every response.
        return audit_payload(result, include_text=include_text, holdings_text=text)

    def _internal(self) -> None:
        """A 500 with an id and no traceback.

        The traceback goes to the server log via ``log.exception``; the pasted
        holdings do not, because they are never passed to the logger anywhere
        in this module.
        """
        error_id = uuid.uuid4().hex[:12]
        log.exception("audit request failed [%s] path=%s", error_id, self.path.split("?", 1)[0])
        try:
            self._refuse(500, "internal", extra_fields={"error_id": error_id})
        except Exception:  # noqa: BLE001
            pass  # the connection is already gone; there is nowhere to report to


def serve(port: int = 8800, host: str = "127.0.0.1", page: str | os.PathLike[str] | None = None) -> None:
    """Run the audit server (blocking)."""
    logging.basicConfig(level=logging.INFO, format="[audit-server] %(levelname)s %(message)s", stream=sys.stderr)

    class _Handler(AuditHandler):
        pass

    _Handler.page_path = Path(page) if page else Path(os.getenv("TIRRA_AUDIT_PAGE") or DEFAULT_PAGE)

    httpd = ThreadingHTTPServer((host, port), _Handler)
    sys.stderr.write(
        f"[audit-server] http://{host}:{port}  page={_Handler.page_path} "
        f"(present={_Handler.page_path.is_file()})\n"
        f"[audit-server] limits: {MAX_HOLDINGS} holdings, {MAX_LINES} lines, "
        f"{MAX_BODY_BYTES}B body, {AUDIT_TIMEOUT_S}s timeout, "
        f"{AUDIT_CONCURRENCY} concurrent, "
        f"{RATE_BURST_CALLS}/{RATE_BURST_WINDOW_S}s and {RATE_HOUR_CALLS}/{RATE_HOUR_WINDOW_S}s per address\n"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[audit-server] stopped\n")
    finally:
        httpd.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the portfolio audit over HTTP")
    parser.add_argument("--port", type=int, default=_int_env("TIRRA_AUDIT_PORT", 8800))
    parser.add_argument("--host", type=str, default=os.getenv("TIRRA_AUDIT_HOST", "127.0.0.1"))
    parser.add_argument("--page", type=str, default=None, help=f"HTML file served at / (default {DEFAULT_PAGE})")
    args = parser.parse_args(argv)
    serve(port=args.port, host=args.host, page=args.page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
