"""Tests for ``agent.portfolio.audit`` — the assembled public report.

WHAT THESE TESTS ARE FOR
    ``audit.py`` computes almost nothing. Testing it means testing the things
    assembly can get wrong: a block that vanishes without a reason, a headline
    published with nothing to check it against, two composed modules answering
    different questions under one heading, and a sentence that turns arithmetic
    into regulated advice. Each of those is a test below, and each was written
    to fail against a specific one-line mutation of the module (listed in
    ``MUTATIONS_CAUGHT`` at the foot of this file).

NO NETWORK, AND NO SHARED CACHE
    Every test injects a fetcher and redirects ``TIRRA_PORTFOLIO_CACHE`` to a
    tmp directory. Both are required, not one: ``holdings._fetch_cached`` reads
    the disk cache BEFORE calling the injected fetcher, so a test that only
    injects a fetcher would silently measure whatever real prices happen to be
    cached on the machine — and would also write its fake prices into the real
    cache for the next real run to pick up.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
import pytest

from agent.portfolio import audit as A

# --------------------------------------------------------------------------
# Synthetic market
# --------------------------------------------------------------------------

N_DAYS = 800
LAST_DAY = "2026-09-29"


def _dates(n: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.bdate_range(end=LAST_DAY, periods=n)


def _path(start: float, end: float, *, vol: float, seed: int, n: int = N_DAYS) -> pd.Series:
    """A price path from ``start`` to EXACTLY ``end`` with deterministic noise.

    The noise is mean-centred in log space, so the endpoint is exact and every
    total return asserted below is a number chosen here rather than a number
    that came out of a random draw. Without the centring a test asserting
    "-31.8 points" would be asserting the seed.
    """
    rng = np.random.default_rng(seed)
    mu = np.log(end / start) / (n - 1)
    eps = rng.normal(0.0, vol, n - 1)
    eps = eps - eps.mean()
    log_rel = np.concatenate([[0.0], np.cumsum(mu + eps)])
    return pd.Series(start * np.exp(log_rel), index=_dates(n), name="close")


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


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_PORTFOLIO_CACHE", str(tmp_path / "yf"))


BENCH = "NIFTYBEES.NS"


def ordinary_market() -> FakeMarket:
    """A book of four names and an index, none of them degenerate."""
    m = FakeMarket()
    m.add(BENCH, _path(100.0, 150.0, vol=0.008, seed=1))
    m.add("AAA.NS", _path(100.0, 190.0, vol=0.013, seed=2))
    m.add("BBB.NS", _path(100.0, 80.0, vol=0.015, seed=3))
    m.add("CCC.NS", _path(100.0, 132.0, vol=0.011, seed=4))
    m.add("DDD.NS", _path(100.0, 120.0, vol=0.009, seed=5))
    return m


ORDINARY_BOOK = "AAA 100\nBBB 100\nCCC 100\nDDD 100"


def run(book: str, market: FakeMarket, **kw) -> A.Audit:
    return A.audit(book, period="3y", fetcher=market, **kw)


# --------------------------------------------------------------------------
# 1. The report is whole
# --------------------------------------------------------------------------


def test_report_contains_all_four_sections_in_order():
    text = A.render_text(run(ORDINARY_BOOK, ordinary_market()))
    positions = [
        text.index("1. THE NUMBER"),
        text.index("2. WHICH DECISIONS"),
        text.index("3. WHY IT MOVED THAT WAY"),
        text.index("4. WHAT THIS IS NOT"),
    ]
    assert positions == sorted(positions)


def test_headline_carries_its_window_and_its_n():
    text = A.render_text(run(ORDINARY_BOOK, ordinary_market()))
    headline_block = text[text.index("1. THE NUMBER") : text.index("2. WHICH DECISIONS")]
    assert "aligned daily returns" in headline_block
    assert LAST_DAY in headline_block


def test_no_network_is_touched_and_every_fetch_goes_through_the_injected_fetcher():
    market = ordinary_market()
    run(ORDINARY_BOOK, market)
    assert market.calls, "the fetcher was never called; the panel came from somewhere else"
    assert all(sym.endswith((".NS", ".BO")) or "=" in sym or sym.isupper() for sym, _ in market.calls)


# --------------------------------------------------------------------------
# 2. Nothing is dropped silently
# --------------------------------------------------------------------------


def test_a_symbol_that_does_not_price_is_named_in_the_report_with_a_reason():
    market = ordinary_market()
    audit_result = run(ORDINARY_BOOK + "\nZZZZ 100", market)
    excluded = {e.symbol for e in audit_result.panel.excluded}
    assert "ZZZZ" in excluded
    text = A.render_text(audit_result)
    assert "ZZZZ" in text
    reason = next(e.reason for e in audit_result.panel.excluded if e.symbol == "ZZZZ")
    assert reason and reason in text


def test_every_parsed_position_is_either_measured_or_excluded():
    market = ordinary_market()
    audit_result = run(ORDINARY_BOOK + "\nZZZZ 100", market)
    measured = {r.typed for r in audit_result.panel.resolutions}
    excluded = {e.symbol for e in audit_result.panel.excluded}
    for entry in audit_result.holdings.entries:
        assert entry.symbol in measured or entry.symbol in excluded, f"{entry.symbol} vanished"


def test_a_section_that_did_not_compute_must_say_why():
    with pytest.raises(ValueError, match="carries no reason"):
        A.Section("k", "t", computed=False)
    A.Section("k", "t", computed=False, reason="the module raised")


def test_a_structure_module_that_raises_degrades_to_a_named_reason():
    from agent.portfolio import overlap as ov

    def boom(*_a, **_k):
        raise RuntimeError("deliberate")

    market = ordinary_market()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ov, "correlation_clusters", boom)
        audit_result = run(ORDINARY_BOOK, market)
    section = next(s for s in audit_result.structure if s.key == "overlap")
    assert not section.computed
    assert "deliberate" in section.reason
    text = A.render_text(audit_result)
    assert "Not computed" in text
    assert "3. WHY IT MOVED THAT WAY" in text and "4. WHAT THIS IS NOT" in text


def test_a_structure_module_that_cannot_be_imported_degrades_rather_than_raising():
    import agent.portfolio as pkg

    market = ordinary_market()
    with pytest.MonkeyPatch.context() as mp:
        # Both are needed: ``from agent.portfolio import beliefs`` resolves the
        # package ATTRIBUTE first and only consults sys.modules if that fails.
        mp.delattr(pkg, "beliefs", raising=False)
        mp.setitem(sys.modules, "agent.portfolio.beliefs", None)
        audit_result = run(ORDINARY_BOOK, market)
    section = next(s for s in audit_result.structure if s.key == "worst_days")
    assert not section.computed and section.reason
    A.render_text(audit_result)


def test_a_book_that_prices_nothing_returns_an_audit_with_reasons_not_an_exception():
    audit_result = run("QQQQ 100\nRRRR 100", FakeMarket())
    assert not audit_result.usable
    assert audit_result.picking_reason and audit_result.attribution_reason
    text = A.render_text(audit_result)
    assert "Not computed" in text
    assert "4. WHAT THIS IS NOT" in text


def test_a_panel_with_too_little_shared_history_is_refused_before_any_figure_is_computed():
    """The dangerous case is not zero data, it is a LITTLE data.

    Two holdings with 70 sessions each but only 35 in common will happily
    produce a return, a volatility and a gap — all of them noise with a decimal
    point. ``fetch_prices`` marks the panel unusable at 60 aligned returns and
    this test pins that ``audit`` stops there rather than measuring anyway.
    """
    market = FakeMarket()
    market.add(BENCH, _path(100.0, 140.0, vol=0.01, seed=31, n=200))
    market.add("AAA.NS", _path(100.0, 150.0, vol=0.01, seed=32, n=70))
    market.add("BBB.NS", _path(100.0, 120.0, vol=0.01, seed=33, n=140).iloc[::2])

    audit_result = run("AAA 10\nBBB 10", market)
    assert not audit_result.panel.usable
    assert len(audit_result.panel.prices) > 0, "this is the little-data case, not the no-data case"
    assert audit_result.full_window is None
    assert "unusable" in audit_result.picking_reason
    assert "60" in audit_result.picking_reason

    text = A.render_text(audit_result)
    headline = text[text.index("1. THE NUMBER") : text.index("2. WHICH DECISIONS")]
    assert "Not computed" in headline
    assert "return per vol" not in headline
    assert A.advice_words_in(text, words=A.LIABILITY_WORDS) == []


def test_no_default_benchmark_is_reported_rather_than_invented():
    from agent.portfolio import benchmark as bm

    market = ordinary_market()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bm, "choose_benchmarks", lambda _p: [])
        audit_result = run(ORDINARY_BOOK, market)
    assert audit_result.full_window is None
    assert "no default benchmark" in audit_result.picking_reason
    assert "no default benchmark" in A.render_text(audit_result)


# --------------------------------------------------------------------------
# 3. The headline never travels alone
# --------------------------------------------------------------------------


def test_headline_is_publishable_only_with_a_record():
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    assert audit_result.headline_publishable
    assert audit_result.usable_records


def test_headline_is_withheld_when_no_other_window_could_be_measured():
    from agent.portfolio import benchmark as bm

    market = ordinary_market()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bm, "rolling_gap", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no windows")))
        audit_result = run(ORDINARY_BOOK, market, multi_window_method="rolling")
        object.__setattr__(audit_result, "multi", None)
    assert audit_result.full_window is not None
    assert not audit_result.headline_publishable
    text = A.render_text(audit_result)
    assert "WITHHELD" in text
    assert "%" not in text[text.index("1. THE NUMBER") : text.index("2. WHICH DECISIONS")]


def test_the_record_prints_the_series_and_not_just_its_endpoints():
    """F-18: a summarised series hides a frozen one. The values go on the page.

    Asserted against the SERIES, not against ``MAX_SERIES_ROWS`` — a test that
    reads the same constant the renderer reads cannot catch that constant being
    turned down to 2.
    """
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    record = next(r for r in audit_result.usable_records if r.length_label == "1y")
    text = A.render_text(audit_result)
    block = text[text.index(f"Every rolling {record.length_label} window") :]
    row = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+\S.*pts$")
    dates = [m.group(1) for ln in block.splitlines()[:200] if (m := row.match(ln.strip()))]
    assert len(dates) >= record.n_windows // 2, f"{len(dates)} rows for {record.n_windows} windows"
    assert dates[0] == f"{record.series.index[0]:%Y-%m-%d}"
    assert dates[-1] == f"{record.series.index[-1]:%Y-%m-%d}"


def test_the_record_states_how_many_windows_do_not_overlap():
    text = A.render_text(run(ORDINARY_BOOK, ordinary_market()))
    assert "independent observations" in text


def test_a_record_too_short_to_be_a_record_quotes_no_fraction():
    """F-18's shape: a tally of one reads as evidence and is one number."""
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    text = A.render_text(audit_result)
    short = [r for r in audit_result.usable_records if r.n_windows < A.MIN_WINDOWS_FOR_TALLY]
    if not short:
        pytest.skip("this history produced no short rolling family")
    block = text[text.index(f"Every rolling {short[0].length_label} window") :].splitlines()
    joined = "\n".join(block[:12])
    assert "no fraction or median is quoted" in joined
    assert "Ahead in" not in joined


def test_a_gap_that_never_moves_is_called_out_instead_of_being_averaged():
    """The F-18 tripwire: a constant series is a frozen computation, not a result."""
    market = FakeMarket()
    shared = _path(100.0, 150.0, vol=0.01, seed=11)
    market.add(BENCH, shared)
    market.add("AAA.NS", shared * 3.0)  # the same instrument at a different price level
    market.add("BBB.NS", shared * 7.0)
    audit_result = run("AAA 10\nBBB 10", market)
    records = [r for r in audit_result.usable_records if r.n_windows >= A.MIN_WINDOWS_FOR_TALLY]
    assert records, "expected at least one rolling family on 800 days"
    for rec in records:
        gaps = rec.series["gap"].to_numpy(dtype=float)
        assert float(gaps.max() - gaps.min()) < 1e-9
    text = A.render_text(audit_result)
    assert "IDENTICAL IN ALL" in text.upper()
    assert "Median window" not in text


def test_a_normal_books_gap_series_actually_varies():
    """The other half of F-18: confirm the live path is not frozen either."""
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    rec = next(r for r in audit_result.usable_records if r.n_windows >= A.MIN_WINDOWS_FOR_TALLY)
    gaps = rec.series["gap"].to_numpy(dtype=float)
    assert len(set(np.round(gaps, 9))) > 1
    assert float(gaps.max() - gaps.min()) > 0.01


# --------------------------------------------------------------------------
# 4. Two composed modules, two counterfactuals, both on the page
# --------------------------------------------------------------------------


def sign_disagreement_market() -> tuple[str, FakeMarket]:
    """A book engineered so the two weighting bases disagree on DIRECTION.

    One share of a 5x winner and ten of a name that fell 20%, against an index
    up 50%. Read as a starting allocation the book returns +141.5% (the winner
    is handed its FINAL 38% weight from day one); read as the share counts the
    stated weights imply, it returns +18.2%. Against +50% those are +91.5
    points and -31.8 points: opposite signs, same book, same window.
    """
    m = FakeMarket()
    m.add(BENCH, _path(100.0, 150.0, vol=0.008, seed=21))
    m.add("WIN.NS", _path(100.0, 500.0, vol=0.016, seed=22))
    m.add("LOS.NS", _path(100.0, 80.0, vol=0.012, seed=23))
    return "WIN 1\nLOS 10", m


def test_both_weighting_bases_are_computed_and_reported():
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    bc = audit_result.basis_check
    assert bc is not None
    assert bc["allocation_book_total"] != pytest.approx(bc["shares_book_total"], abs=1e-9)
    text = A.render_text(audit_result)
    assert f"{bc['allocation_book_total'] * 100:.1f}%" in text
    assert f"{bc['shares_book_total'] * 100:.1f}%" in text


def test_the_two_bases_are_the_numbers_the_two_modules_actually_produced():
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    bc = audit_result.basis_check
    assert bc["allocation_gap"] == pytest.approx(audit_result.full_window.gap_total)
    assert bc["shares_gap"] == pytest.approx(audit_result.attribution.gap_total)


def test_a_sign_disagreement_between_the_bases_is_stated_not_resolved():
    """Integration: whatever the two modules answer, BOTH answers reach the page.

    The arithmetic belongs to ``benchmark`` and ``attribution`` and may legally
    change, so this asserts the assembly's job — both numbers reported, and the
    disagreement named when there is one — rather than pinning their values.
    ``test_render_states_a_sign_disagreement_it_is_handed`` pins the rendering
    itself against fixed inputs, independent of either module.
    """
    book, market = sign_disagreement_market()
    audit_result = run(book, market)
    bc = audit_result.basis_check
    assert bc is not None
    text = A.render_text(audit_result)
    assert A._pct(bc["allocation_book_total"], 0) in text
    assert A._pct(bc["shares_book_total"], 0) in text
    if bc["signs_agree"]:
        assert "do not even agree on direction" not in text
        assert "ANSWERS ONE OF TWO QUESTIONS" in text
    else:
        assert "do not even agree on direction" in text
        assert A._pts(bc["allocation_gap"]) in text
        assert A._pts(bc["shares_gap"]) in text


def _stub_audit(*, alloc_gap: float, shares_gap: float, panel, holdings_obj):
    """An ``Audit`` whose two bases are handed in, so the renderer is tested alone."""
    from types import SimpleNamespace as NS

    window = NS(
        start=pd.Timestamp("2023-09-29"), end=pd.Timestamp(LAST_DAY), kind="full", label="full history", complete=True
    )
    full = NS(
        window=window,
        n_obs=742,
        years=3.0,
        port_total=0.30,
        port_annual=0.091,
        port_vol=0.13,
        port_return_per_vol=0.70,
        bench_total=0.30 - alloc_gap,
        bench_annual=0.06,
        bench_vol=0.12,
        bench_return_per_vol=0.50,
        gap_total=alloc_gap,
        gap_annual=alloc_gap / 3.0,
        constant_mix_total=float("nan"),
        notes=(),
        usable=True,
    )
    picking = NS(
        full_history=lambda _t: full,
        named=lambda _t: [full],
        ex_benchmark={},
        skipped_windows=(),
    )

    class _Attr(list):
        usable = True
        unusable_reason = ""
        gap_total = shares_gap
        book = NS(total_return=0.10)
        window = "2023-10-03..2026-09-29 (n=741 trading days)"
        benchmark_ticker = BENCH
        basis = "buy_and_hold"
        basis_note = "share counts held unchanged"
        notes = ()
        excluded = ()

    attribution = _Attr()
    record = NS(
        length_label="1y",
        n_windows=12,
        beat_count=5,
        beat_fraction=5 / 12,
        median_gap=0.01,
        best_gap=0.05,
        best_window="w1",
        worst_gap=-0.05,
        worst_window="w2",
        caveats=(),
        usable=True,
        series=pd.DataFrame(
            {
                "start": pd.bdate_range(end="2026-03-01", periods=12),
                "port_total": np.linspace(0.05, 0.20, 12),
                "bench_total": np.linspace(0.10, 0.12, 12),
                "gap": np.linspace(-0.05, 0.08, 12),
            }
        ).set_index(pd.bdate_range(end=LAST_DAY, periods=12)),
    )
    return A.Audit(
        holdings=holdings_obj,
        panel=panel,
        benchmark_ticker=BENCH,
        benchmark_why="fixed for this test",
        other_benchmarks=(),
        picking=picking,
        picking_reason="",
        records=(record,),
        records_reason="",
        attribution=attribution,
        summary={"statement": "", "reconciles": True, "residual_explanation": "exact"},
        attribution_reason="",
        multi=None,
        multi_reason="",
        structure=(),
        survivorship="",
        limitations=("ONE HISTORY. Fixed for this test.",),
        notes=(),
    )


@pytest.mark.parametrize(
    "alloc_gap, shares_gap, disagree",
    [(0.057, -0.086, True), (-0.20, -0.31, False), (0.91, 0.30, False), (-0.02, 0.15, True)],
)
def test_render_states_a_sign_disagreement_it_is_handed(alloc_gap, shares_gap, disagree):
    """Pins the renderer against fixed inputs, so no sibling change can move it."""
    source = run(ORDINARY_BOOK, ordinary_market())
    stub = _stub_audit(alloc_gap=alloc_gap, shares_gap=shares_gap, panel=source.panel, holdings_obj=source.holdings)
    assert stub.basis_check["signs_agree"] is not disagree
    text = A.render_text(stub)
    assert A._pts(alloc_gap) in text
    assert A._pts(shares_gap) in text
    assert ("do not even agree on direction" in text) is disagree
    assert A.advice_words_in(text, words=A.LIABILITY_WORDS) == []


def test_the_basis_check_is_withheld_rather_than_half_reported():
    from agent.portfolio import attribution as at

    market = ordinary_market()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(at, "holding_attribution", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
        audit_result = run(ORDINARY_BOOK, market)
    assert audit_result.basis_check is None
    text = A.render_text(audit_result)
    assert "could not be cross-checked" in text
    assert "nope" in text


# --------------------------------------------------------------------------
# 5. Which decisions
# --------------------------------------------------------------------------


def test_every_priced_holding_appears_in_the_decisions_table():
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    text = A.render_text(audit_result)
    block = text[text.index("2. WHICH DECISIONS") : text.index("3. WHY IT MOVED THAT WAY")]
    for ticker in audit_result.panel.tickers:
        assert ticker in block


def test_contributions_are_reconciled_and_the_reconciliation_is_shown():
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    assert audit_result.summary["reconciles"] is True
    assert "Reconciliation:" in A.render_text(audit_result)


def test_the_window_by_window_check_on_the_second_basis_is_rendered_under_it():
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    text = A.render_text(audit_result)
    assert audit_result.multi is not None
    here = text.index("THE SAME CHECK ON THIS READING")
    assert text.index("2. WHICH DECISIONS") < here < text.index("3. WHY IT MOVED THAT WAY")


# --------------------------------------------------------------------------
# 6. No advice, and no forecast
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "book, market_factory",
    [
        (ORDINARY_BOOK, ordinary_market),
        ("AAA 100", ordinary_market),
        ("AAA 5\nBBB 500\nCCC 1\nDDD 40\nZZZZ 3", ordinary_market),
    ],
)
def test_no_word_that_would_make_this_regulated_advice_reaches_the_page(book, market_factory):
    text = A.render_text(run(book, market_factory()))
    assert A.advice_words_in(text, words=A.LIABILITY_WORDS) == []


def test_the_prose_this_module_writes_is_clean_under_the_full_list():
    """Sibling modules own their own wording; this asserts ours."""
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    ours = "\n".join(audit_result.limitations)
    assert A.advice_words_in(ours) == []


def test_any_judgement_word_on_the_page_is_one_we_have_already_accounted_for():
    """A NEW advice word appearing from any module fails this, with its line."""
    text = A.render_text(run(ORDINARY_BOOK, ordinary_market()))
    hits = A.advice_words_in(text)
    if hits:
        offending = [ln.strip() for ln in text.splitlines() if any(h in ln.lower() for h in hits)]
        pytest.fail(f"unaccounted advice words {hits} in:\n" + "\n".join(offending[:10]))


def test_the_scanner_catches_real_advice():
    assert "you should" in A.advice_words_in("You should index instead.")
    assert "rebalance" in A.advice_words_in("Rebalance the book each quarter.")
    assert "expect" in A.advice_words_in("We expect this to continue.")


def test_the_scanner_forgives_a_disclaimer_but_not_the_thing_it_disclaims():
    assert A.advice_words_in("Nothing here is a recommendation.") == []
    assert A.advice_words_in("This is not a forecast.") == []
    assert "recommend" in A.advice_words_in("Nothing here is a recommendation.", allow_negated=False)


def test_the_benign_exemption_is_by_phrase_and_not_by_word():
    assert A.advice_words_in("Measured on a buy and hold basis.") == []
    assert "buy " in A.advice_words_in("Buy NIFTYBEES instead of these names.")
    assert "buy " in A.advice_words_in("Measured on a buy and hold basis.", allow_benign=False)


def test_every_benign_exemption_names_the_module_that_produces_it():
    for phrase, source in A.BENIGN_PHRASES:
        assert phrase and source, "an exemption with no stated source is an unexplained hole"
        assert any(w in phrase for w in A.ADVICE_WORDS), f"{phrase!r} exempts nothing"


# --------------------------------------------------------------------------
# 7. What this is not
# --------------------------------------------------------------------------


def test_the_limitations_are_a_section_and_not_a_footnote():
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    assert len(audit_result.limitations) >= 7
    text = A.render_text(audit_result)
    tail = text[text.index("4. WHAT THIS IS NOT") :]
    assert len(tail.splitlines()) > 30


@pytest.mark.parametrize(
    "must_mention",
    ["WEIGHTING BASES", "OVERLAPPING", "SURVIVORSHIP", "ONE BENCHMARK", "NO COSTS AND NO TAX", "TOTAL RETURNS"],
)
def test_each_named_distortion_is_stated_in_plain_words(must_mention):
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    assert any(must_mention in item for item in audit_result.limitations)


def test_the_limitations_name_this_books_own_currency_and_benchmark():
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    joined = "\n".join(audit_result.limitations)
    assert audit_result.panel.base_currency in joined
    assert audit_result.benchmark_ticker in joined


def test_the_survivorship_paragraph_is_present_and_names_no_direction():
    audit_result = run(ORDINARY_BOOK, ordinary_market())
    assert audit_result.survivorship
    assert A.advice_words_in(A.render_text(audit_result), words=A.LIABILITY_WORDS) == []


# --------------------------------------------------------------------------
# 8. Determinism
# --------------------------------------------------------------------------


def test_the_same_book_twice_gives_the_same_report():
    first = A.render_text(run(ORDINARY_BOOK, ordinary_market()))
    second = A.render_text(run(ORDINARY_BOOK, ordinary_market()))
    assert first == second


#: Mutations these tests were checked against. Each was applied to
#: ``agent/portfolio/audit.py``, the suite was run, and the named test failed.
MUTATIONS_CAUGHT = (
    "headline_publishable returns True unconditionally "
    "-> test_headline_is_withheld_when_no_other_window_could_be_measured",
    "render_text prints the headline table regardless of headline_publishable "
    "-> test_headline_is_withheld_when_no_other_window_could_be_measured",
    "basis_check returns only the allocation basis -> test_both_weighting_bases_are_computed_and_reported",
    "basis_check computes signs_agree as True always -> test_render_states_a_sign_disagreement_it_is_handed",
    "render_text drops the one-basis-only notice when the cross-check is unavailable "
    "-> test_the_basis_check_is_withheld_rather_than_half_reported",
    "Section.__post_init__ drops the reason requirement -> test_a_section_that_did_not_compute_must_say_why",
    "_section_overlap swallows the exception and returns computed=True with no lines "
    "-> test_a_structure_module_that_raises_degrades_to_a_named_reason",
    "_render_record quotes a median for a one-window family "
    "-> test_a_record_too_short_to_be_a_record_quotes_no_fraction",
    "_render_record drops the constant-series branch "
    "-> test_a_gap_that_never_moves_is_called_out_instead_of_being_averaged",
    "_render_record prints only the first and last window of the series "
    "-> test_the_record_prints_the_series_and_not_just_its_endpoints",
    "advice_words_in excuses every occurrence with allow_benign "
    "-> test_the_benign_exemption_is_by_phrase_and_not_by_word",
    "advice_words_in returns [] unconditionally -> test_the_scanner_catches_real_advice",
    "audit() lets fetch_prices raise instead of returning a reason "
    "-> test_a_book_that_prices_nothing_returns_an_audit_with_reasons_not_an_exception",
    "audit() measures a panel fetch_prices marked unusable instead of stopping "
    "-> test_a_panel_with_too_little_shared_history_is_refused_before_any_figure_is_computed",
    "audit() invents SPY when choose_benchmarks returns nothing "
    "-> test_no_default_benchmark_is_reported_rather_than_invented",
    "_limitations drops the two-bases item -> test_each_named_distortion_is_stated_in_plain_words",
)
