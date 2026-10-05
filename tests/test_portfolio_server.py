"""Tests for ``agent.portfolio.server`` — the HTTP boundary of the audit.

WHAT THESE TESTS ARE FOR
    The server computes nothing, so testing it means testing the four things a
    boundary can get wrong, each of which is visible to a stranger on the
    public internet:

    1. **A number that is not JSON.** ``json.dumps`` emits a bare ``NaN`` for a
       float NaN, which ``JSON.parse`` rejects — the whole page would go blank
       over one unusable volatility. Every response here is dumped with
       ``allow_nan=False``, so a NaN that escaped :func:`server._num` fails the
       test rather than the page.
    2. **A sentence that is advice.** ``test_no_advice_anywhere_in_a_response``
       walks EVERY string in a full response — ours and every one copied from a
       sibling module — through ``audit.advice_words_in``. This is the test that
       makes the product legal to ship, and it is deliberately the broadest one
       in the file.
    3. **A refusal that is a stack trace.** Every 4xx and 5xx asserts a plain
       sentence, and ``test_internal_error_leaks_nothing`` asserts that a
       handler blowing up mid-request yields an id and no traceback.
    4. **A cap that is not enforced.** One caller must not be able to spend the
       shared yfinance quota: body size, line count, position count, rate limit,
       admission and timeout each have a test that a 200 would fail.

NO NETWORK, AND NO SHARED CACHE
    ``server.run_audit`` is monkeypatched in every test that reaches the audit,
    to the real ``audit.audit`` with an injected fetcher, and
    ``TIRRA_PORTFOLIO_CACHE`` is redirected to a tmp directory. Both are
    needed: ``holdings._fetch_cached`` reads the disk cache BEFORE the injected
    fetcher, so a test with only a fetcher would silently measure whatever real
    prices happen to be on the machine.
"""

from __future__ import annotations

import json
import math
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.portfolio import audit as A
from agent.portfolio import server as S

# --------------------------------------------------------------------------
# Synthetic market. Same construction as tests/test_portfolio_audit.py: the
# noise is mean-centred in log space so every endpoint is exact and no assertion
# below is really asserting a seed.
# --------------------------------------------------------------------------

N_DAYS = 800
LAST_DAY = "2026-09-29"
BENCH = "NIFTYBEES.NS"


def _path(start: float, end: float, *, vol: float, seed: int, n: int = N_DAYS) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=LAST_DAY, periods=n)
    mu = np.log(end / start) / (n - 1)
    eps = rng.normal(0.0, vol, n - 1)
    eps = eps - eps.mean()
    return pd.Series(start * np.exp(np.concatenate([[0.0], np.cumsum(mu + eps)])), index=idx, name="close")


class FakeMarket:
    """A price source with no network. Records what was asked for."""

    def __init__(self) -> None:
        self.series: dict[str, tuple[pd.Series, str]] = {}
        self.calls: list[tuple[str, str]] = []

    def add(self, symbol: str, series: pd.Series, currency: str = "INR") -> FakeMarket:
        self.series[symbol] = (series, currency)
        return self

    def __call__(self, symbol: str, period: str):
        self.calls.append((symbol, period))
        got = self.series.get(symbol)
        return None if got is None else (got[0], got[1])


def ordinary_market() -> FakeMarket:
    """Five names and an index, none degenerate, one clear laggard."""
    m = FakeMarket()
    m.add(BENCH, _path(100.0, 150.0, vol=0.008, seed=1))
    m.add("AAA.NS", _path(100.0, 190.0, vol=0.013, seed=2))
    m.add("BBB.NS", _path(100.0, 80.0, vol=0.015, seed=3))
    m.add("CCC.NS", _path(100.0, 130.0, vol=0.010, seed=4))
    m.add("DDD.NS", _path(100.0, 115.0, vol=0.020, seed=5))
    return m


BOOK = "AAA 100\nBBB 250\nCCC 40\nDDD 300"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_PORTFOLIO_CACHE", str(tmp_path / "yf"))


@pytest.fixture
def market() -> FakeMarket:
    return ordinary_market()


@pytest.fixture
def offline(monkeypatch, market):
    """Point the server's audit at the fake market. Returns the market."""

    def _run(text: str, *, benchmark: str | None = None):
        return A.audit(text, period="3y", benchmark=benchmark, fetcher=market)

    monkeypatch.setattr(S, "run_audit", _run)
    return market


def _audit_result(market: FakeMarket, book: str = BOOK, **kw):
    return A.audit(book, period="3y", fetcher=market, **kw)


# --------------------------------------------------------------------------
# Walking every string a caller sees
# --------------------------------------------------------------------------


def _strings(obj, path="$"):
    """Every string in a payload, with the path it sits at."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")


def _numbers(obj, path="$"):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{path}[{i}]")


# ==========================================================================
# 1. Serialisation: JSON-safe, and nothing invented to fill a gap
# ==========================================================================


def test_payload_is_strict_json(market):
    """``allow_nan=False`` is the whole point: a NaN here blanks the page."""
    payload = S.audit_payload(_audit_result(market))
    text = json.dumps(payload, allow_nan=False)  # raises ValueError on NaN/Inf
    assert "NaN" not in text and "Infinity" not in text
    for path, value in _numbers(payload):
        assert not isinstance(value, float) or math.isfinite(value), path


def test_num_turns_nan_into_none_not_zero():
    """A missing number must read as missing. 0.0 would be a false fact."""
    assert S._num(float("nan")) is None
    assert S._num(float("inf")) is None
    assert S._num(None) is None
    assert S._num("not a number") is None
    assert S._num(0.0) == 0.0
    assert S._num(3) == 3


def test_jsonable_handles_what_siblings_actually_put_in_data():
    @dataclass
    class Leg:
        ticker: str
        weight: float

    blob = {
        "ts": pd.Timestamp("2024-01-02"),
        "when": pd.Timestamp("2024-01-02 15:30:00"),
        "np_float": np.float64(1.5),
        "np_int": np.int64(7),
        "nan": float("nan"),
        "leg": Leg("AAA.NS", 0.25),
        "frame": pd.DataFrame({"a": [1, 2, 3]}),
        "series": pd.Series([1.0, 2.0]),
        "members": ("AAA.NS", "BBB.NS"),
        "set": {"b", "a"},
    }
    out = S.jsonable(blob)
    json.dumps(out, allow_nan=False)
    assert out["ts"] == "2024-01-02"
    assert out["when"] == "2024-01-02T15:30:00"
    assert out["np_float"] == 1.5
    assert out["np_int"] == 7
    assert out["nan"] is None
    assert out["leg"] == {"ticker": "AAA.NS", "weight": 0.25}
    assert out["frame"]["_table_omitted"] is True and out["frame"]["n_rows"] == 3
    assert out["series"]["_series_omitted"] is True
    assert out["members"] == ["AAA.NS", "BBB.NS"]
    assert out["set"] == ["a", "b"]


def test_payload_carries_the_findings_the_product_is_for(market):
    """risk-against-weight, the same-bet group, and the gap against the index.

    Each of the three has to arrive as FIELDS. A page that had to regex a
    sentence to lay out "6.0% of the money, 0.5% of the variance" would be
    re-deriving a number a tested module already computed.
    """
    payload = S.audit_payload(_audit_result(market))

    rows = payload["structure"]["concentration"]["data"]["contributions"]
    assert rows and all({"ticker", "weight", "risk_share"} <= set(r) for r in rows)
    assert payload["structure"]["concentration"]["data"]["n_observations"] > 0

    clusters = payload["structure"]["overlap"]["clusters"]
    assert clusters["computed"] is True
    for g in clusters["groups"]:
        assert set(g) >= {"members", "weight_of_book", "mean_correlation", "window"}
        assert g["window"]["n_days"] > 0

    head = payload["headline"]
    assert head["published"] is True
    assert head["gap_total"] is not None
    assert head["benchmark"]["ticker"] == BENCH
    assert head["n_observations"] > 0


def test_every_block_carries_its_own_window_and_sample_size(market):
    """Three modules align the same history differently; a figure without its
    window is not checkable, and this is the disclosure we promise."""
    payload = S.audit_payload(_audit_result(market))
    assert payload["headline"]["window"]["start"] and payload["headline"]["n_observations"] > 0
    assert payload["decisions"]["window"]["n_days"] > 0
    assert payload["structure"]["concentration"]["data"]["window"]
    assert payload["structure"]["worst_days"]["data"]["window"]
    for rec in payload["rolling_record"]:
        assert rec["n_windows"] > 0
        assert rec["n_independent_windows"] is not None
    assert payload["window"]["n_daily_returns"] > 0


def test_holdings_ordering_is_attributions_own_worst_first(market):
    """Worst contributor first. Re-sorting in a presentation layer would change
    what the block says about which line moved the gap."""
    result = _audit_result(market)
    payload = S.audit_payload(result)
    contributions = [h["contribution_to_gap"] for h in payload["decisions"]["holdings"]]
    assert contributions == sorted(contributions)
    assert [h["ticker"] for h in payload["decisions"]["holdings"]] == [h.ticker for h in result.attribution]


def test_a_missing_block_says_why_and_does_not_vanish(market, monkeypatch):
    """Eighteen recorded failure modes in this repo, nearly all silent."""
    monkeypatch.setattr(
        "agent.portfolio.concentration.risk_contributions",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    payload = S.audit_payload(_audit_result(market))
    block = payload["structure"]["concentration"]
    assert block["computed"] is False
    assert "boom" in block["reason"]
    assert block["key"] == "concentration" and block["title"]


def test_headline_is_withheld_without_a_second_window(market, monkeypatch):
    """One window is a selected window. The API withholds the number rather
    than footnoting it, mirroring ``Audit.headline_publishable`` exactly."""
    result = _audit_result(market)
    assert result.headline_publishable
    stripped = S.audit_payload(_stripped_of_records(result))
    assert stripped["headline"]["published"] is False
    assert stripped["headline"]["withheld_reason"]
    assert "gap_total" not in stripped["headline"]
    assert "book" not in stripped["headline"]


def _stripped_of_records(result):
    """The same audit with every multi-window check removed."""
    import dataclasses

    return dataclasses.replace(result, records=(), multi=None, records_reason="removed in a test")


def test_unusable_book_is_a_reasoned_payload_not_an_exception():
    """A book whose prices do not exist still renders, with reasons."""
    payload = S.audit_payload(A.audit("ZZZZ 10", period="3y", fetcher=FakeMarket()))
    assert payload["usable"] is False
    assert payload["unusable_reason"]
    assert payload["headline"]["published"] is False
    json.dumps(payload, allow_nan=False)


def test_report_text_is_opt_in(market):
    result = _audit_result(market)
    assert "report_text" not in S.audit_payload(result)
    assert len(S.audit_payload(result, include_text=True)["report_text"]) > 1000


# ==========================================================================
# 2. The constraint that shapes every word on screen
# ==========================================================================


def test_no_advice_anywhere_in_a_response(market):
    """The test that makes this legal to publish.

    Every string in a full response, ours and every sibling's, through the same
    scan ``render_text`` is held to. "Your 4% position carries 19% of the risk"
    is arithmetic; anything that recommends, suggests, forecasts or implies an
    action is a regulated investment recommendation we are not registered to
    give.
    """
    payload = S.audit_payload(_audit_result(market), include_text=True)
    offenders = [(path, A.advice_words_in(text)) for path, text in _strings(payload)]
    offenders = [(p, h) for p, h in offenders if h]
    assert offenders == [], offenders


#: ``audit._NEGATION_WINDOW`` is 40 characters, which would forgive "not only
#: should you sell" — the negator is there, but it negates something else.
#: Every real disclaimer in this package puts the negator immediately before
#: the word ("is not a forecast", "none of it is a recommendation"), so this
#: file holds the wire to a much tighter window than the module's own. 24 is
#: the length of the longest real one, "Nothing here is a recommendation".
_TIGHT_NEGATION = 24


def test_every_liability_word_on_the_wire_is_negated_right_where_it_sits(market):
    """The regulated subset, with a stricter negation rule than the module's.

    ``advice_words_in`` forgives a negator anywhere in the preceding 40
    characters. SEBI reads the sentence. So this runs the scan with the hatches
    OFF and then re-admits an occurrence only when it is inside one of the
    phrases ``audit.BENIGN_PHRASES`` justifies by name, or when the negator is
    directly in front of it. "This is not a forecast" passes; "not only should
    you sell" would not, and the module's own scan would let it through.
    """
    payload = S.audit_payload(_audit_result(market), include_text=True)
    justified = tuple(p for p, _why in A.BENIGN_PHRASES)
    negators = A._NEGATORS
    offenders: list[str] = []
    for path, text in _strings(payload):
        low = text.lower()
        for word in A.LIABILITY_WORDS:
            start = low.find(word)
            while start != -1:
                before = low[max(0, start - _TIGHT_NEGATION) : start]
                in_benign = any(p in low and low.find(p) <= start < low.find(p) + len(p) for p in justified)
                if not in_benign and not any(n in before for n in negators):
                    offenders.append(f"{path}: {word!r} in {text[max(0, start - 60) : start + 60]!r}")
                start = low.find(word, start + 1)
    assert offenders == [], offenders


def test_authored_copy_is_scanned_even_when_no_audit_runs():
    """Every refusal, every disclosure, the privacy note. These reach a caller
    on paths where no audit object exists, so they need their own pass."""
    authored = [("PRIVACY.statement", S.PRIVACY["statement"])]
    authored += [(f"MESSAGES.{k}", v) for k, v in S.MESSAGES.items()]
    for d in S.DISCLOSURES:
        authored += [(f"{d['key']}.title", d["title"]), (f"{d['key']}.body", d["body"])]
    for name, text in authored:
        assert A.advice_words_in(text) == [], f"{name}: {A.advice_words_in(text)}"
        assert A.advice_words_in(text, words=A.LIABILITY_WORDS, allow_negated=False, allow_benign=False) == [], name


def test_the_three_disclosures_are_on_the_result_not_behind_a_link(market):
    payload = S.audit_payload(_audit_result(market))
    keys = [d["key"] for d in payload["disclosures"]]
    assert keys == ["weights_applied_backwards", "survivorship", "window_and_sample"]
    for d in payload["disclosures"]:
        assert d["title"] and len(d["body"]) > 80
    body = " ".join(d["body"] for d in payload["disclosures"]).lower()
    assert "not what you earned" in body  # the counterfactual, said plainly
    assert "survived" in body  # only what is still held is visible
    assert payload["limitations"], "the full 'what this is not' list travels too"


def test_privacy_note_is_on_every_response_and_is_true(market):
    payload = S.audit_payload(_audit_result(market))
    assert payload["privacy"]["holdings_stored"] is False
    assert payload["privacy"]["holdings_logged"] is False
    assert payload["privacy"]["cookies"] is False
    assert payload["privacy"]["analytics"] is False
    assert "memory" in payload["privacy"]["statement"]


def test_both_weighting_bases_travel_together(market):
    """``basis_check`` is never a half-comparison: the two bases gave opposite
    signs on a real book, and rendering one alone is the cherry-pick."""
    payload = S.audit_payload(_audit_result(market))
    check = payload["basis_check"]
    assert check is not None
    assert {"allocation_gap", "shares_gap", "signs_agree"} <= set(check)
    assert check["allocation_window"] and check["shares_window"]


def test_contradicting_calendar_windows_come_first(market):
    """Inherits ``MultiWindowResult.headline_qualifier``'s order. A page that
    showed the agreeing windows first would perform the cherry-pick."""
    payload = S.audit_payload(_audit_result(market))
    cal = payload["calendar_windows"]
    assert cal["computed"] is True
    agreement = [w["agrees_with_full_window"] for w in cal["windows"] if w["agrees_with_full_window"] is not None]
    assert agreement == sorted(agreement, key=lambda a: a is True)
    assert cal["n_independent_windows"] >= 1
    assert cal["qualifier"]


# ==========================================================================
# 3. The HTTP surface
# ==========================================================================


@pytest.fixture
def server(offline, tmp_path, monkeypatch):
    """A real ThreadingHTTPServer on an ephemeral port.

    Rate limiters are replaced per test so one test's calls never exhaust
    another's budget, and the admission semaphore is replaced so a leaked slot
    cannot leak across tests.
    """
    monkeypatch.setattr(S, "_BURST_LIMITER", S.RateLimiter(50, 60))
    monkeypatch.setattr(S, "_HOUR_LIMITER", S.RateLimiter(200, 3600))
    monkeypatch.setattr(S, "_AUDIT_SLOTS", threading.BoundedSemaphore(4))

    page = tmp_path / "index.html"
    page.write_text("<!doctype html><title>Audit</title><h1>paste your book</h1>", encoding="utf-8")

    class _H(S.AuditHandler):
        page_path = page

        def log_message(self, fmt, *args):  # keep the pytest output readable
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield _Client(base, page)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@dataclass
class _Response:
    code: int
    body: str
    headers: dict

    @property
    def json(self):
        return json.loads(self.body)


class _Client:
    def __init__(self, base: str, page: Path) -> None:
        self.base = base
        self.page = page

    def get(self, path: str) -> _Response:
        return self._send(urllib.request.Request(self.base + path))

    def post(self, path: str, body, *, content_type: str = "application/json", headers=None) -> _Response:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method="POST")
        req.add_header("Content-Type", content_type)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        return self._send(req)

    @staticmethod
    def _send(req) -> _Response:
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return _Response(resp.status, resp.read().decode("utf-8"), dict(resp.headers))
        except urllib.error.HTTPError as exc:
            return _Response(exc.code, exc.read().decode("utf-8"), dict(exc.headers))


def test_health_states_the_limits(server):
    r = server.get("/api/health")
    assert r.code == 200
    body = r.json
    assert body["ok"] is True and body["version"] == S.API_VERSION
    limits = body["limits"]
    assert limits["max_holdings"] == S.MAX_HOLDINGS
    assert limits["max_lines"] == S.MAX_LINES
    assert limits["max_body_bytes"] == S.MAX_BODY_BYTES
    assert limits["audit_timeout_s"] == S.AUDIT_TIMEOUT_S
    assert limits["rate_limit"]["burst"] and limits["rate_limit"]["sustained"]
    assert body["privacy"]["holdings_stored"] is False


def test_post_audit_returns_the_structured_report(server):
    r = server.post("/api/audit", {"holdings": BOOK})
    assert r.code == 200
    body = r.json
    assert body["ok"] is True
    assert body["input"]["positions_read"] == 4
    assert body["input"]["positions_priced"] == 4
    assert body["headline"]["published"] is True
    assert body["benchmark"]["ticker"] == BENCH
    assert body["disclosures"] and body["privacy"]
    assert body["elapsed_s"] >= 0
    assert "report_text" not in body
    assert r.headers.get("Content-Type", "").startswith("application/json")
    assert r.headers.get("Cache-Control") == "no-store"


def test_named_benchmark_is_honoured_and_junk_is_refused(server):
    ok = server.post("/api/audit", {"holdings": BOOK, "benchmark": BENCH})
    assert ok.code == 200 and ok.json["benchmark"]["ticker"] == BENCH
    for junk in ["; rm -rf /", "a" * 40, "<script>", 42, ["NIFTYBEES.NS"]]:
        bad = server.post("/api/audit", {"holdings": BOOK, "benchmark": junk})
        assert bad.code == 400, junk
        assert bad.json["error"] == "bad_benchmark"


def test_empty_body_and_garbage_are_sentences_not_tracebacks(server):
    empty = server.post("/api/audit", b"")
    assert empty.code == 400 and empty.json["error"] == "no_body"

    garbage = server.post("/api/audit", b"this is not json at all {{{")
    assert garbage.code == 400 and garbage.json["error"] == "bad_json"
    assert "JSON" in garbage.json["message"]

    array = server.post("/api/audit", b"[1, 2, 3]")
    assert array.code == 400 and array.json["error"] == "bad_json"

    missing = server.post("/api/audit", {"portfolio": BOOK})
    assert missing.code == 400 and missing.json["error"] == "holdings_missing"

    blank = server.post("/api/audit", {"holdings": "   \n  \n"})
    assert blank.code == 400 and blank.json["error"] == "holdings_missing"

    wrong_type = server.post("/api/audit", {"holdings": ["AAA 100"]})
    assert wrong_type.code == 400 and wrong_type.json["error"] == "holdings_not_string"

    for r in (empty, garbage, array, missing, blank, wrong_type):
        _assert_no_traceback(r)


def test_a_line_that_was_dropped_says_so_on_the_response(server):
    """``parse_holdings`` rejects a quantity-less line when other lines have
    one. The response names that line and the reason, so the page can show the
    holder what of their paste did not make it into the numbers."""
    r = server.post("/api/audit", {"holdings": "AAA\nBBB 250\nCCC 40"})
    assert r.code == 200
    unread = r.json["input"]["unreadable"]
    assert [u["raw"] for u in unread] == ["AAA"]
    assert unread[0]["reason"] and unread[0]["line_no"] == 1
    assert r.json["input"]["positions_read"] == 2


def test_assumed_quantities_are_stated_not_made_silently(server):
    """When NO line carries a quantity the parser assumes one, and says so."""
    r = server.post("/api/audit", {"holdings": "AAA\nBBB\nCCC"})
    assert r.code == 200
    assert r.json["input"]["assumptions"], "an assumed quantity must be stated"
    assert r.json["input"]["positions_read"] == 3


def test_one_holding_is_a_valid_book(server):
    r = server.post("/api/audit", {"holdings": "AAA 100"})
    assert r.code == 200
    body = r.json
    assert body["input"]["positions_read"] == 1
    # A one-name book has no correlation group to report, and the block says so
    # rather than disappearing.
    overlap = body["structure"]["overlap"]
    assert overlap["computed"] is False or overlap["clusters"]["computed"] in (True, False)
    assert body["disclosures"]


def test_unparseable_paste_is_422_with_the_lines_that_failed(server):
    """``parse_holdings`` is deliberately lenient — "the quick brown fox" reads
    as one position with an assumed quantity, and that assumption is returned
    in ``input.assumptions``. 422 is for a paste where NO line held a symbol."""
    r = server.post("/api/audit", {"holdings": "???\n!!!\n###"})
    assert r.code == 422
    assert r.json["error"] == "nothing_parsed"
    assert len(r.json["unreadable"]) >= 1
    assert all({"line_no", "raw", "reason"} <= set(u) for u in r.json["unreadable"])
    _assert_no_traceback(r)


def test_five_hundred_holdings_is_refused_before_any_fetch(server, offline):
    before = len(offline.calls)
    r = server.post("/api/audit", {"holdings": "\n".join(f"SYM{i} 10" for i in range(500))})
    assert r.code == 413
    assert r.json["error"] == "too_many_lines"
    assert r.json["lines"] == 500 and r.json["max_lines"] == S.MAX_LINES
    assert len(offline.calls) == before, "a refused paste must not spend the price quota"
    _assert_no_traceback(r)


def test_more_positions_than_the_cap_is_refused_before_any_fetch(server, offline):
    before = len(offline.calls)
    n = S.MAX_HOLDINGS + 5
    r = server.post("/api/audit", {"holdings": "\n".join(f"SYM{i} 10" for i in range(n))})
    assert r.code == 413
    assert r.json["error"] == "too_many_holdings"
    assert r.json["positions"] == n and r.json["max_holdings"] == S.MAX_HOLDINGS
    assert len(offline.calls) == before


def test_oversized_body_is_refused_without_reading_it(server):
    """A 10 MB paste must not cross the socket. The refusal is decided from
    Content-Length and the connection is closed."""
    huge = json.dumps({"holdings": "AAA 100\n" * 200_000}).encode("utf-8")
    assert len(huge) > S.MAX_BODY_BYTES
    r = server.post("/api/audit", huge)
    assert r.code == 413
    assert r.json["error"] == "body_too_large"
    assert r.json["max_body_bytes"] == S.MAX_BODY_BYTES
    assert r.json["received_bytes"] == len(huge)
    _assert_no_traceback(r)


def test_a_body_past_the_drain_ceiling_is_cut_off_rather_than_swallowed(server, monkeypatch):
    """Draining is a courtesy with a limit. Past the ceiling the connection is
    closed mid-write, because reading tens of megabytes to be polite about
    refusing them is the denial of service the cap exists to stop."""
    monkeypatch.setattr(S, "MAX_DISCARD_BYTES", 1024)
    huge = json.dumps({"holdings": "AAA 100\n" * 50_000}).encode("utf-8")
    try:
        r = server.post("/api/audit", huge)
    except urllib.error.URLError as exc:  # the reset is the point
        assert isinstance(exc.reason, (ConnectionResetError, BrokenPipeError)), exc
    else:
        assert r.code == 413 and r.json["error"] == "body_too_large"


def test_a_delisted_ticker_is_named_not_swallowed(server):
    r = server.post("/api/audit", {"holdings": "AAA 100\nDELISTED 50\nBBB 250"})
    assert r.code == 200
    body = r.json
    named = {e["symbol"] for e in body["input"]["not_measured"]}
    assert "DELISTED" in named
    assert (body["input"]["not_measured"][0]["reason"] or "").strip()
    assert body["input"]["positions_read"] == 3
    assert body["input"]["positions_priced"] == 2


def test_a_book_of_only_unknown_tickers_is_200_with_reasons(server):
    r = server.post("/api/audit", {"holdings": "NOPE 10\nNADA 20"})
    assert r.code == 200
    body = r.json
    assert body["usable"] is False
    assert body["unusable_reason"]
    assert body["message"] == S.MESSAGES["unusable"]
    assert body["headline"]["published"] is False and body["headline"]["withheld_reason"]
    _assert_no_traceback(r)


def test_rate_limit_is_429_with_retry_after(server, monkeypatch):
    monkeypatch.setattr(S, "_BURST_LIMITER", S.RateLimiter(2, 60))
    codes = [server.post("/api/audit", {"holdings": BOOK}).code for _ in range(3)]
    assert codes[:2] == [200, 200]
    assert codes[2] == 429
    limited = server.post("/api/audit", {"holdings": BOOK})
    assert limited.code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    assert limited.json["retry_after"] >= 1
    assert "quota" in limited.json["message"]
    _assert_no_traceback(limited)


def test_a_malformed_request_still_spends_rate_budget(server, monkeypatch):
    """Otherwise there is an unmetered path: every refusal decided while
    reading the body would be free to repeat as fast as a socket allows."""
    monkeypatch.setattr(S, "_BURST_LIMITER", S.RateLimiter(2, 60))
    assert server.post("/api/audit", b"garbage").code == 400
    assert server.post("/api/audit", b"").code == 400
    assert server.post("/api/audit", {"holdings": BOOK}).code == 429


def test_sustained_limit_applies_even_when_the_burst_bucket_is_open(server, monkeypatch):
    monkeypatch.setattr(S, "_BURST_LIMITER", S.RateLimiter(100, 60))
    monkeypatch.setattr(S, "_HOUR_LIMITER", S.RateLimiter(1, 3600))
    assert server.post("/api/audit", {"holdings": BOOK}).code == 200
    assert server.post("/api/audit", {"holdings": BOOK}).code == 429


def test_rate_limiter_forgets_old_hits():
    limiter = S.RateLimiter(2, 10)
    assert limiter.allow("ip", now=1000.0)[0] is True
    assert limiter.allow("ip", now=1001.0)[0] is True
    blocked, retry = limiter.allow("ip", now=1002.0)
    assert blocked is False and 0 < retry <= 10
    assert limiter.allow("ip", now=1011.5)[0] is True, "the window slid"
    assert limiter.allow("other", now=1002.0)[0] is True, "buckets are per key"


def test_timeout_is_504_with_a_useful_message(server, monkeypatch):
    def _slow(text, *, benchmark=None):
        time.sleep(1.5)
        raise AssertionError("never reached in this test")

    monkeypatch.setattr(S, "run_audit", _slow)
    monkeypatch.setattr(S, "AUDIT_TIMEOUT_S", 0.3)
    r = server.post("/api/audit", {"holdings": BOOK})
    assert r.code == 504
    assert r.json["error"] == "timeout"
    assert r.json["timeout_s"] == 0.3
    assert int(r.headers["Retry-After"]) > 0
    assert "cached" in r.json["message"]
    _assert_no_traceback(r)


def test_a_timed_out_audit_keeps_its_slot_until_the_work_ends(server, monkeypatch):
    """The work cannot be cancelled, so releasing the slot on timeout would let
    a slow client run more audits than the concurrency cap allows."""
    started = threading.Event()
    release = threading.Event()

    def _slow(text, *, benchmark=None):
        started.set()
        release.wait(timeout=10)
        return A.audit(text, period="3y", fetcher=ordinary_market())

    monkeypatch.setattr(S, "run_audit", _slow)
    monkeypatch.setattr(S, "AUDIT_TIMEOUT_S", 0.3)
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(S, "_AUDIT_SLOTS", slots)

    assert server.post("/api/audit", {"holdings": BOOK}).code == 504
    assert started.is_set()
    assert slots.acquire(blocking=False) is False, "the slot is still held by the running audit"
    release.set()
    for _ in range(100):
        if slots.acquire(blocking=False):
            slots.release()
            break
        time.sleep(0.05)
    else:
        pytest.fail("the slot was never released after the work finished")


def test_no_free_slot_is_503_not_a_queue(server, monkeypatch):
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(S, "_AUDIT_SLOTS", slots)
    r = server.post("/api/audit", {"holdings": BOOK})
    assert r.code == 503
    assert r.json["error"] == "busy"
    assert int(r.headers["Retry-After"]) > 0
    _assert_no_traceback(r)


def test_internal_error_leaks_nothing(server, monkeypatch, caplog):
    """A handler blowing up yields an id and a sentence, never a traceback —
    and the pasted holdings are in neither the response nor the log."""
    pasted_line = "MYSECRETHOLDING 12345"

    def _boom(text, *, benchmark=None):
        raise RuntimeError("internal detail: /Users/someone/secret/path.py line 42")

    monkeypatch.setattr(S, "run_audit", _boom)
    with caplog.at_level("ERROR"):
        r = server.post("/api/audit", {"holdings": f"AAA 100\n{pasted_line}"})
    assert r.code == 500
    assert r.json["error"] == "internal"
    assert re.fullmatch(r"[0-9a-f]{12}", r.json["error_id"])
    assert "Traceback" not in r.body
    assert "internal detail" not in r.body
    assert "/Users/" not in r.body
    logged = "\n".join(rec.getMessage() for rec in caplog.records)
    assert r.json["error_id"] in logged, "the id has to be findable in the log"
    assert pasted_line not in logged, "holdings must never reach a log"
    assert "MYSECRETHOLDING" not in r.body


def test_holdings_never_appear_in_the_access_log(server, capsys):
    """The default ``log_message`` prints the request line. A GET with holdings
    in the query string would put them there, so audits are POST only."""
    r = server.get("/api/audit")
    assert r.code == 405 or r.code == 404


def test_unknown_paths_and_methods(server):
    missing = server.get("/does/not/exist")
    assert missing.code == 404 and missing.json["error"] == "not_found"
    assert "POST /api/audit" in missing.json["message"]

    wrong_post = server.post("/api/health", {"holdings": BOOK})
    assert wrong_post.code == 404
    _assert_no_traceback(missing)
    _assert_no_traceback(wrong_post)


def test_every_other_method_is_json_not_the_stdlib_html_page(server):
    """A method with no handler must not fall through to BaseHTTPRequestHandler.

    Its default ``send_error`` answers ``text/html`` with the method name echoed
    into the document and none of the headers ``_send`` sets. That is a contract
    break for a JSON client and the one place this server would ever reflect
    input back, so both are asserted here rather than assumed.
    """
    host, port = server.base.rsplit(":", 1)
    for method in ("PUT", "DELETE", "PATCH", "TRACE", "FROBNICATE"):
        with socket.create_connection(("127.0.0.1", int(port)), timeout=30) as sock:
            sock.sendall(
                f"{method} /api/audit HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode()
            )
            chunks = []
            while True:
                b = sock.recv(65536)
                if not b:
                    break
                chunks.append(b)
        raw = b"".join(chunks).decode("utf-8", "replace")
        head, _, body = raw.partition("\r\n\r\n")

        assert "text/html" not in head.lower(), f"{method} answered HTML: {head}"
        assert "application/json" in head.lower(), f"{method} did not answer JSON: {head}"
        assert "<html" not in body.lower() and "<!doctype" not in body.lower(), method
        # the header block _send guarantees, which the stdlib page skipped
        assert "X-Content-Type-Options: nosniff" in head, method
        assert "Cache-Control: no-store" in head, method

        payload = json.loads(body)
        assert payload["ok"] is False, method
        assert payload["error"] == "method_not_allowed", method
        assert payload["allowed"] == "GET, HEAD, POST, OPTIONS", method
        assert "Allow: GET, HEAD, POST, OPTIONS" in head, method
        # the method name must not be reflected back to the caller
        assert method not in body, f"{method} was echoed into the response body"
        # and the internal stdlib wording is not the visitor's copy
        assert "Unsupported method" not in raw, method


def test_a_malformed_request_line_is_json_too(server):
    """``send_error`` is also the stdlib's path for a request it cannot parse."""
    host, port = server.base.rsplit(":", 1)
    with socket.create_connection(("127.0.0.1", int(port)), timeout=30) as sock:
        sock.sendall(b"GET " + b"/" + b"A" * 70000 + b" HTTP/1.1\r\nHost: x\r\n\r\n")
        chunks = []
        try:
            while True:
                b = sock.recv(65536)
                if not b:
                    break
                chunks.append(b)
        except OSError:
            pass
    raw = b"".join(chunks).decode("utf-8", "replace")
    if not raw:
        pytest.skip("the peer closed before any response, which leaks nothing either")
    assert "<html" not in raw.lower(), raw[:400]
    assert "Traceback" not in raw


def test_root_serves_the_page_and_says_so_when_it_is_missing(server):
    r = server.get("/")
    assert r.code == 200
    assert r.headers["Content-Type"].startswith("text/html")
    assert "paste your book" in r.body

    server.page.unlink()
    gone = server.get("/")
    assert gone.code == 503
    assert gone.json["error"] == "page_missing"
    assert "POST /api/audit works" in gone.json["message"]
    assert gone.json["page_path_configured"].endswith("index.html")


def test_only_the_configured_file_is_reachable(server):
    """No static directory, no path joining, so no traversal."""
    for path in ["/../../etc/passwd", "/products/audit/index.html", "/agent/portfolio/server.py", "/%2e%2e/etc/passwd"]:
        r = server.get(path)
        assert r.code == 404, path


def test_options_preflight_is_answered(server):
    req = urllib.request.Request(server.base + "/api/audit", method="OPTIONS")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 204
        assert resp.headers["Access-Control-Allow-Methods"] == "GET, POST, OPTIONS"


def test_no_response_anywhere_contains_a_traceback_or_a_path(server):
    """One sweep over every refusal this endpoint can produce."""
    calls = [
        server.post("/api/audit", b""),
        server.post("/api/audit", b"{"),
        server.post("/api/audit", {"holdings": 5}),
        server.post("/api/audit", {"holdings": "prose only"}),
        server.post("/api/audit", {"holdings": "\n".join(f"S{i} 1" for i in range(500))}),
        server.post("/api/audit", {"holdings": BOOK, "benchmark": "!!"}),
        server.get("/nope"),
        server.get("/api/health"),
        server.post("/api/audit", {"holdings": BOOK}),
    ]
    for r in calls:
        _assert_no_traceback(r)
        assert r.body.strip(), "every response has a body"
        json.loads(r.body) if r.headers.get("Content-Type", "").startswith("application/json") else None


def _assert_no_traceback(r: _Response) -> None:
    for forbidden in ("Traceback", 'File "/', '.py", line', "agent/portfolio/", "site-packages"):
        assert forbidden not in r.body, f"{forbidden!r} leaked in a {r.code}: {r.body[:300]}"


def test_concurrent_requests_do_not_interfere(server):
    """Four audits at once, four correct answers, no crossed wires."""
    results: list[_Response] = []
    lock = threading.Lock()

    def _one(book: str) -> None:
        r = server.post("/api/audit", {"holdings": book})
        with lock:
            results.append(r)

    books = ["AAA 100", "AAA 100\nBBB 250", "CCC 40\nDDD 300", BOOK]
    threads = [threading.Thread(target=_one, args=(b,)) for b in books]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert len(results) == 4
    assert all(r.code == 200 for r in results), [r.code for r in results]
    assert sorted(r.json["input"]["positions_read"] for r in results) == [1, 2, 2, 4]


def test_socket_close_mid_body_does_not_wedge_the_server(server):
    """A client that promises 5000 bytes and sends 10 gets a sentence, and the
    next request still works."""
    host, port = server.base.rsplit(":", 1)
    sock = socket.create_connection(("127.0.0.1", int(port)), timeout=10)
    sock.sendall(b"POST /api/audit HTTP/1.1\r\nHost: x\r\nContent-Length: 5000\r\n\r\nshort")
    sock.shutdown(socket.SHUT_WR)
    sock.recv(4096)
    sock.close()
    assert server.get("/api/health").code == 200


# ==========================================================================
# The look-through block — the lead finding, and the page's contract with it
# ==========================================================================
#
# These test the SERVER's mapping, not the arithmetic (that is
# tests/test_lookthrough_combine.py). The mapping is where the page's
# assumptions live, and one of them was already wrong in the HTML: it counted
# ``len(rows)`` and called the result "companies", printing "53 companies"
# directly below a headline that said 51. Two numbers for the same thing on one
# card is how a reader decides the whole page is unreliable, so the invariant
# that separates them is pinned here rather than left to the next reader.
#
# No network: ``combine._default_provider`` is replaced, which keeps the real
# combine logic and the real server mapping and only fakes the index weights.


@pytest.fixture
def fake_index(monkeypatch):
    """Index weights with no NSE and no Yahoo behind them."""
    from agent.lookthrough import combine as C

    class _Prov:
        describe = "a test provider; these weights are made up"

        def index_weights(self, index: str) -> C.IndexWeights:
            if index != "NIFTY 50":
                raise C.LookThroughError(f"test provider has no {index}")
            return C.IndexWeights(
                index=index,
                weights={"HDFCBANK": 0.5, "RELIANCE": 0.25, "ICICIBANK": 0.25},
                names={k: f"{k} Ltd." for k in ("HDFCBANK", "RELIANCE", "ICICIBANK")},
                note="TEST NOTE: made up",
                asof="2026-10-02",
            )

    prov = _Prov()
    monkeypatch.setattr(C, "_default_provider", lambda: (prov, prov.describe))
    return prov


# NIFTYBEES at 40% over the fake index: HDFCBANK gets 0.40 x 0.5 = 0.20 added
# to the 10% listed, so actual is 30%. Numbers checkable by eye on purpose.
FUND_BOOK = "HDFCBANK 10%\nRELIANCE 20%\nNIFTYBEES 40%\nGOLDBEES 30%"


def test_n_companies_is_not_the_row_count(fake_index):
    """``rows`` carries the funds we could NOT unpack; ``n_companies`` must not.

    The page prints ``n_companies`` under the table and ``len(rows)`` is the
    wrong number for it. If this ever becomes an equality, the HTML's fallback
    (filter on ``kind == "company"``) and the server disagree silently.
    """
    lt = S._look_through(FUND_BOOK)
    assert lt["computed"] is True

    companies = [r for r in lt["rows"] if r["kind"] == "company"]
    opaque = [r for r in lt["rows"] if r["kind"] != "company"]

    assert lt["n_companies"] == len(companies)
    assert opaque, "GOLDBEES should be carried as a non-company row, not dropped"
    assert len(lt["rows"]) > lt["n_companies"]


def test_every_row_kind_is_one_the_page_knows(fake_index):
    """The page branches on ``kind``; a new value would render as a company.

    ``lookThroughCard`` filters ``kind === "company"`` to count companies, so a
    third kind added upstream would be counted as one and inflate the figure
    under the table without failing anything.
    """
    lt = S._look_through(FUND_BOOK)
    kinds = {r["kind"] for r in lt["rows"]}
    assert kinds <= {"company", "opaque_fund"}, kinds


def test_added_is_actual_minus_listed_so_it_is_percentage_points(fake_index):
    """The headline number. It is a DIFFERENCE, which is why the page says pp.

    Pinned because the page renders this field with a ``pp`` suffix: were it
    ever to become a ratio (actual/listed) the label would be wrong and the
    number would still look plausible.
    """
    lt = S._look_through(FUND_BOOK)
    by_sym = {r["symbol"]: r for r in lt["rows"]}

    hdfc = by_sym["HDFCBANK"]
    assert hdfc["listed"] == pytest.approx(0.10)
    assert hdfc["actual"] == pytest.approx(0.30)
    assert hdfc["added"] == pytest.approx(0.20)

    for r in lt["rows"]:
        listed = r["listed"] or 0.0
        assert r["added"] == pytest.approx(r["actual"] - listed, abs=1e-9)


def test_rows_are_ranked_by_surprise_not_by_size(fake_index):
    """A small position they never listed beats a large one that barely moved.

    This is the whole reason the table is worth reading, and sorting by
    ``actual`` instead would still produce a sensible-looking table.
    """
    lt = S._look_through(FUND_BOOK)
    added = [r["added"] for r in lt["rows"]]
    assert added == sorted(added, reverse=True)


def test_a_book_of_only_direct_stocks_computes_but_unpacks_nothing(fake_index):
    """``computed`` means the engine RAN, not that it found anything.

    The page gates on ``n_unpacked > 0`` to decide whether this card leads,
    because a four-stock book computes perfectly well and has nothing hidden.
    Leading with it produced a table whose every row read "30.0% / 30.0% / —".
    """
    lt = S._look_through("HDFCBANK 30%\nRELIANCE 30%\nINFY 20%\nTCS 20%")
    assert lt["computed"] is True
    assert lt["n_unpacked"] == 0
    assert all(r["added"] == pytest.approx(0.0) for r in lt["rows"])


def test_n_listed_counts_funds_too_so_it_is_not_the_direct_count(fake_index):
    """Pinned because the page subtracts the opaque funds from it.

    The empty-state card says "your N direct holdings are exactly what you
    listed", and N is ``n_listed`` minus the funds it could not unpack. If
    ``n_listed`` ever stops counting funds, that subtraction double-counts.
    """
    lt = S._look_through("HDFCBANK 40%\nRELIANCE 30%\nGOLDBEES 30%")
    assert lt["n_listed"] == 3
    opaque = [f for f in lt["funds"] if not f["unpacked"]]
    assert len(opaque) == 1
    assert lt["n_listed"] - len(opaque) == 2


def test_an_unpackable_fund_is_named_and_never_guessed(fake_index):
    """The refusal is the trustworthy part, so it must carry a reason."""
    lt = S._look_through(FUND_BOOK)
    opaque = [f for f in lt["funds"] if not f["unpacked"]]
    assert opaque, "GOLDBEES holds no equities and must not be unpacked"
    for f in opaque:
        assert f["reason"].strip(), f"{f['label']} refused without saying why"
        assert f["n_constituents"] == 0


def test_the_approximation_note_travels_with_the_numbers(fake_index):
    """A reader who finds this in a footer later would be right to distrust us."""
    lt = S._look_through(FUND_BOOK)
    assert lt["approximation_notes"], "unpacked an index and said nothing about how"


def test_look_through_failure_does_not_take_the_audit_down(monkeypatch):
    """The rest of the payload is worth more than this block."""
    from agent.lookthrough import combine as C

    def _boom(*a, **k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(C, "look_through", _boom)
    lt = S._look_through(FUND_BOOK)
    assert lt["computed"] is False
    assert lt["reason"]
    # The reason is for a stranger on the internet: a type name, not a trace.
    assert "Traceback" not in lt["reason"]
    assert "provider exploded" not in lt["reason"]


def test_no_paste_means_no_claim(fake_index):
    """Absent input must not render as a book with nothing hidden."""
    lt = S._look_through("")
    assert lt["computed"] is False
    assert lt["reason"]


# ==========================================================================
# MUTATIONS CAUGHT
# ==========================================================================
#
# Each line is a one-line change to server.py and the test that fails for it.
# Written by making the change and running the file, not by reasoning about it.
#
# _num returns 0.0 instead of None for NaN        -> test_num_turns_nan_into_none_not_zero
# json.dumps without allow_nan=False              -> test_payload_is_strict_json
# _headline returns the numbers when not publishable -> test_headline_is_withheld_without_a_second_window
# _decisions sorts holdings by weight             -> test_holdings_ordering_is_attributions_own_worst_first
# _multi_window renders agreeing windows first    -> test_contradicting_calendar_windows_come_first
# DISCLOSURES dropped from audit_payload          -> test_the_three_disclosures_are_on_the_result_not_behind_a_link
# a "you should" added to any MESSAGES entry      -> test_authored_copy_is_scanned_even_when_no_audit_runs
# _refuse sends str(exc) instead of MESSAGES[key] -> test_internal_error_leaks_nothing
# the MAX_LINES check moved after parse_holdings  -> (still passes) ...
# the MAX_LINES check removed                     -> test_five_hundred_holdings_is_refused_before_any_fetch
# the MAX_HOLDINGS check removed                  -> test_more_positions_than_the_cap_is_refused_before_any_fetch
# _read_body reads an oversized body anyway       -> test_oversized_body_is_refused_without_reading_it
# the hour limiter dropped from _audit            -> test_sustained_limit_applies_even_when_the_burst_bucket_is_open
# the slot released on timeout instead of on done -> test_a_timed_out_audit_keeps_its_slot_until_the_work_ends
# _AUDIT_SLOTS.acquire(blocking=True)             -> test_no_free_slot_is_503_not_a_queue
# _page joins the request path onto a directory   -> test_only_the_configured_file_is_reachable
# n_companies set to len(rows)                    -> test_n_companies_is_not_the_row_count
# rows sorted by actual instead of by added       -> test_rows_are_ranked_by_surprise_not_by_size
# added computed as actual/listed                 -> test_added_is_actual_minus_listed_so_it_is_percentage_points
# an opaque fund dropped from rows                -> test_n_companies_is_not_the_row_count
# a refusal reason left empty                     -> test_an_unpackable_fund_is_named_and_never_guessed
# approximation_notes dropped from the block      -> test_the_approximation_note_travels_with_the_numbers
# _look_through re-raising instead of reporting   -> test_look_through_failure_does_not_take_the_audit_down
# str(exc) used as the look-through reason        -> test_look_through_failure_does_not_take_the_audit_down
