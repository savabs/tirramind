"""Tests for agent.portfolio.benchmark.

Three kinds of test, deliberately:

1. **Calibration against closed-form truth.** A price series that exactly
   doubles over exactly two calendar years must report ``total = 1.0`` and
   ``annual = sqrt(2) - 1``. A book whose only holding is the benchmark must
   report a gap of exactly zero. These are the only tests that can say the
   numbers *mean* what the docstrings claim rather than merely being computed.

2. **Tests a defect would actually fail.** LESSONS F-09/F-13 record four tests
   in this repo that asserted the bug they were meant to catch. Every assertion
   below is written so that the obvious way to break the function makes it red;
   the mutation run is recorded in the task output. In particular
   ``test_weights_are_actually_used`` exists because a function that silently
   ignored ``weights`` would still pass every sums-to-one and sign check here,
   and ``test_price_return_benchmark_is_flagged_and_never_default`` exists
   because the single most embarrassing possible bug in this module is comparing
   dividend-inclusive holdings against a dividend-excluding index.

3. **Integrity tests on the output itself.** No advice language, no
   forward-looking claim, the weight caveat present, the window rule present,
   and — F-18 — the headline series must be shown to *vary* rather than being
   one favourable number.

Everything runs offline: ``fake_fetcher`` replaces yfinance and the price cache
is redirected to ``tmp_path`` so the tests cannot read or poison a real cache.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agent.portfolio import benchmark as B
from agent.portfolio import holdings as H

SEED = 20260929
INR = "INR"


# --------------------------------------------------------------------------
# synthetic world
# --------------------------------------------------------------------------


def _cal(n: int, start: str = "2021-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def const_series(n: int, level: float = 100.0, start: str = "2021-01-04") -> pd.Series:
    return pd.Series(level, index=_cal(n, start), dtype=float)


def exact_growth(n: int, total_over_span: float, start: str = "2021-01-04") -> pd.Series:
    """A series that grows smoothly to exactly ``1 + total_over_span`` by its last print."""
    idx = _cal(n, start)
    frac = np.linspace(0.0, 1.0, n)
    return pd.Series(100.0 * (1.0 + total_over_span) ** frac, index=idx)


def wobble(
    n: int, total_over_span: float, period: int, phase: float = 0.0, amp: float = 0.15, start: str = "2021-01-04"
) -> pd.Series:
    """The benchmark's own path times a slow sine — so rolling gaps straddle zero
    by construction rather than by luck of a seed."""
    base = exact_growth(n, total_over_span, start)
    t = np.arange(n, dtype=float)
    return base * (1.0 + amp * np.sin(2 * np.pi * t / period + phase))


def gbm(n: int, drift: float = 0.08, vol: float = 0.18, seed: int = SEED, start: str = "2021-01-04") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = _cal(n, start)
    step = rng.normal(drift / 252.0 - 0.5 * (vol**2) / 252.0, vol / math.sqrt(252.0), size=n)
    step[0] = 0.0
    return pd.Series(100.0 * np.exp(np.cumsum(step)), index=idx)


def fake_fetcher(book: dict[str, tuple[pd.Series, str]]):
    """A ``(symbol, period) -> (close, currency) | None`` stand-in for yfinance."""

    def _f(symbol: str, period: str):
        got = book.get(symbol)
        return None if got is None else (got[0].copy(), got[1])

    return _f


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """No test may read or write the real ``~/.cache/tirramind`` price cache."""
    monkeypatch.setenv("TIRRA_PORTFOLIO_CACHE", str(tmp_path / "px"))


def panel_from(book: dict[str, tuple[pd.Series, str]], text: str, **kw) -> H.PricePanel:
    return H.fetch_prices(H.parse_holdings(text), fetcher=fake_fetcher(book), period="5y", **kw)


@pytest.fixture
def two_asset_world() -> dict[str, tuple[pd.Series, str]]:
    n = 760  # ~3 calendar years of business days
    return {
        "NIFTYBEES.NS": (exact_growth(n, 0.30), INR),
        "AAA.NS": (exact_growth(n, 0.60), INR),
        "BBB.NS": (exact_growth(n, -0.20), INR),
        "NOISY.NS": (gbm(n, drift=0.05, vol=0.20, seed=1), INR),
        "NOISY2.NS": (gbm(n, drift=0.05, vol=0.20, seed=2), INR),
        "TRACK.NS": (wobble(n, 0.30, period=400), INR),
        "TRACK2.NS": (wobble(n, 0.30, period=400, phase=np.pi / 3), INR),
        "FLAT.NS": (const_series(n), INR),
    }


def _panel(world, text):
    p = panel_from(world, text)
    assert p.usable, p.unusable_reason
    return p


def _bench(world, panel, ticker="NIFTYBEES.NS"):
    return B.fetch_benchmark(panel, ticker, fetcher=fake_fetcher(world))


# ==========================================================================
# 1. CALIBRATION — the numbers must mean what the docstrings say
# ==========================================================================


def test_book_of_only_the_benchmark_has_exactly_zero_gap(two_asset_world):
    """A book that IS the index cannot have out- or under-performed it.

    This is the tightest calibration available: it pins the return convention,
    the window endpoints and the benchmark alignment all at once. Any off-by-one
    in the window, or a price/total-return mismatch, moves it off zero.
    """
    p = _panel(two_asset_world, "NIFTYBEES.NS 100")
    cost = B.picking_cost(p, benchmark=_bench(two_asset_world, p), include_rolling=False)
    for r in cost.named("NIFTYBEES.NS"):
        assert abs(r.gap_total) < 1e-12, f"{r.window.label}: gap {r.gap_total}"
        assert abs(r.port_total - r.bench_total) < 1e-12


def test_total_and_annualised_return_match_closed_form(two_asset_world):
    """Exactly +60% over the full span must annualise to (1.6)**(1/years) - 1."""
    p = _panel(two_asset_world, "AAA.NS 100")
    cost = B.picking_cost(p, benchmark=_bench(two_asset_world, p), include_rolling=False)
    full = cost.full_history("NIFTYBEES.NS")
    assert full is not None
    assert full.port_total == pytest.approx(0.60, abs=1e-9)
    assert full.bench_total == pytest.approx(0.30, abs=1e-9)
    assert full.gap_total == pytest.approx(0.30, abs=1e-9)
    assert full.port_annual == pytest.approx(1.60 ** (1.0 / full.years) - 1.0, rel=1e-12)
    assert full.gap_annual == pytest.approx(full.port_annual - full.bench_annual, rel=1e-12)


def test_flat_book_has_zero_return_and_zero_volatility(two_asset_world):
    p = _panel(two_asset_world, "FLAT.NS 50\nNIFTYBEES.NS 50")
    b = _bench(two_asset_world, p)
    r = B.picking_cost(p, {"FLAT.NS": 1.0, "NIFTYBEES.NS": 0.0}, benchmark=b, include_rolling=False).full_history(
        "NIFTYBEES.NS"
    )
    assert r.port_total == pytest.approx(0.0, abs=1e-12)
    assert r.port_vol == pytest.approx(0.0, abs=1e-12)
    assert not np.isfinite(r.port_return_per_vol), "return-per-vol on zero vol must be undefined, not infinite"
    assert r.gap_total == pytest.approx(-r.bench_total, abs=1e-12)


def test_per_holding_contributions_sum_exactly_to_the_book(two_asset_world):
    """Buy-and-hold decomposes exactly. If it does not, the legs and the headline
    are telling a reader two different stories about the same book."""
    p = _panel(two_asset_world, "AAA.NS 30\nBBB.NS 25\nNOISY.NS 20\nNOISY2.NS 25")
    b = _bench(two_asset_world, p)
    cost = B.picking_cost(p, benchmark=b, include_rolling=False)
    for r in cost.named("NIFTYBEES.NS"):
        if not r.usable:
            continue
        assert sum(leg.contribution for leg in r.legs) == pytest.approx(r.port_total, abs=1e-12)
        assert sum(leg.weight for leg in r.legs) == pytest.approx(1.0, abs=1e-12)
        for leg in r.legs:
            assert leg.contribution == pytest.approx(leg.weight * leg.total_return, abs=1e-15)
            assert leg.vs_benchmark == pytest.approx(leg.total_return - r.bench_total, abs=1e-15)


def test_volatility_is_calibrated_against_a_known_generating_vol(two_asset_world):
    """A 20%-vol generator must come back at roughly 20%, not 1.2% or 320%."""
    p = _panel(two_asset_world, "NOISY.NS 100")
    b = _bench(two_asset_world, p)
    r = B.picking_cost(p, benchmark=b, include_rolling=False).full_history("NIFTYBEES.NS")
    assert 0.15 < r.port_vol < 0.25, r.port_vol
    assert r.port_return_per_vol == pytest.approx(r.port_annual / r.port_vol, rel=1e-12)


def test_constant_mix_and_buy_and_hold_agree_on_one_asset_and_differ_on_two(two_asset_world):
    """With one holding there is nothing to drift, so the two paths must coincide.
    With two dispersed holdings they must not — if they did, the constant-mix
    figure is not being computed and the 'robustness' column is decoration."""
    p1 = _panel(two_asset_world, "NOISY.NS 100")
    r1 = B.picking_cost(p1, benchmark=_bench(two_asset_world, p1), include_rolling=False).full_history("NIFTYBEES.NS")
    assert r1.constant_mix_total == pytest.approx(r1.port_total, rel=1e-9)

    p2 = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    r2 = B.picking_cost(p2, benchmark=_bench(two_asset_world, p2), include_rolling=False).full_history("NIFTYBEES.NS")
    assert abs(r2.constant_mix_total - r2.port_total) > 1e-3


# ==========================================================================
# 2. TESTS A DEFECT WOULD FAIL
# ==========================================================================


def test_weights_are_actually_used(two_asset_world):
    """A function that ignored ``weights`` would pass every other test in this
    file. Two very different weightings of the same two names must not agree."""
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    b = _bench(two_asset_world, p)
    heavy_a = B.picking_cost(p, {"AAA.NS": 0.9, "BBB.NS": 0.1}, benchmark=b, include_rolling=False)
    heavy_b = B.picking_cost(p, {"AAA.NS": 0.1, "BBB.NS": 0.9}, benchmark=b, include_rolling=False)
    ra = heavy_a.full_history("NIFTYBEES.NS")
    rb = heavy_b.full_history("NIFTYBEES.NS")
    assert ra.port_total == pytest.approx(0.9 * 0.60 + 0.1 * -0.20, abs=1e-9)
    assert rb.port_total == pytest.approx(0.1 * 0.60 + 0.9 * -0.20, abs=1e-9)
    assert ra.port_total > rb.port_total


def test_gap_sign_points_the_right_way(two_asset_world):
    """A book that beat the index must report a positive gap, and one that lost
    a negative one. A sign flip here would invert the entire product."""
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    b = _bench(two_asset_world, p)
    winner = B.picking_cost(p, {"AAA.NS": 1.0, "BBB.NS": 0.0}, benchmark=b, include_rolling=False).full_history(
        "NIFTYBEES.NS"
    )
    loser = B.picking_cost(p, {"AAA.NS": 0.0, "BBB.NS": 1.0}, benchmark=b, include_rolling=False).full_history(
        "NIFTYBEES.NS"
    )
    assert winner.gap_total > 0 and winner.beat
    assert loser.gap_total < 0 and not loser.beat
    # A zero-weight column is not a bet anyone made, so it must not be counted.
    assert winner.n_legs_held == 1 and winner.n_legs_behind == 0
    assert loser.n_legs_held == 1 and loser.n_legs_behind == 1
    assert len(winner.legs) == 2, "both columns are still reported; only the count is held-only"


def test_returns_frame_passed_as_prices_is_refused(two_asset_world):
    """Handing daily returns to a price function produces a plausible-looking and
    entirely wrong percentage. It must raise, not guess."""
    p = _panel(two_asset_world, "NOISY.NS 50\nNOISY2.NS 50")
    with pytest.raises(ValueError, match="RETURNS"):
        B.picking_cost(p.returns, {c: 0.5 for c in p.returns.columns}, benchmark="NIFTYBEES.NS")


def test_negative_weight_is_refused(two_asset_world):
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    with pytest.raises(ValueError, match="negative weights"):
        B.picking_cost(p, {"AAA.NS": 1.4, "BBB.NS": -0.4}, benchmark=_bench(two_asset_world, p))


def test_missing_weight_is_refused_rather_than_assumed_zero(two_asset_world):
    """Assuming zero for a held position silently shrinks the book and changes
    every number in the report."""
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    with pytest.raises(ValueError, match="no weight for"):
        B.picking_cost(p, {"AAA.NS": 1.0}, benchmark=_bench(two_asset_world, p))


def test_unused_weight_is_reported_not_dropped_silently(two_asset_world):
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    cost = B.picking_cost(
        p, {"AAA.NS": 0.5, "BBB.NS": 0.4, "ZZZ.NS": 0.1}, benchmark=_bench(two_asset_world, p), include_rolling=False
    )
    joined = " ".join(cost.assumptions)
    assert "ZZZ.NS" in joined and "ignored" in joined
    assert "normalised to 1.0" in joined


def test_non_positive_and_nan_prices_are_refused():
    idx = _cal(300)
    bad = pd.DataFrame({"A": np.linspace(100, 200, 300), "B": np.linspace(10, -5, 300)}, index=idx)
    with pytest.raises(ValueError, match="non-positive"):
        B._as_prices(bad)
    holey = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=idx)
    holey.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        B._as_prices(holey)


def test_auto_adjust_tripwire_fires_when_the_fetcher_stops_adjusting(monkeypatch):
    """The single most embarrassing possible bug: holdings go price-return while
    the ETF benchmark stays total-return, manufacturing the benchmark's dividend
    yield as a gap. This tripwire is the only thing standing between that change
    and a public number."""

    def price_return_fetch(symbol, period):
        # deliberately does NOT pass the dividend-adjusting flag
        return None

    monkeypatch.setattr(B._h, "_yf_fetch", price_return_fetch)
    with pytest.raises(RuntimeError, match="auto_adjust=True"):
        B._assert_total_return_fetcher()

    def adjusted_fetch(symbol, period):
        hist = ("history", dict(auto_adjust=True))  # noqa: C408 - the literal is the point
        return hist

    monkeypatch.setattr(B._h, "_yf_fetch", adjusted_fetch)
    B._assert_total_return_fetcher()


def test_the_real_holdings_fetcher_still_adjusts_for_dividends():
    """Live tripwire against the actual module, not a stand-in."""
    B._assert_total_return_fetcher()


def test_price_return_benchmark_is_flagged_and_never_default(two_asset_world):
    """``^NSEI`` excludes dividends. Measured on this repo's cache it understated
    NIFTYBEES.NS by 121.5 bps/yr. It must be classified, warned about by name,
    pointed at its total-return substitute, and never chosen as a default."""
    world = dict(two_asset_world)
    world["^NSEI"] = (exact_growth(760, 0.24), INR)  # same exposure, dividends stripped
    p = _panel(world, "AAA.NS 50\nBBB.NS 50")

    assert all(t not in B.PRICE_RETURN_SYMBOLS for t, _ in B.choose_benchmarks(p))
    for ticker, _ in B.DEFAULT_BENCHMARKS.values():
        assert not ticker.startswith("^")

    bad = B.fetch_benchmark(p, "^NSEI", fetcher=fake_fetcher(world))
    assert bad.total_return is False
    assert any("NIFTYBEES.NS" in w and "PRICE index" in w for w in bad.warnings)

    good = B.fetch_benchmark(p, "NIFTYBEES.NS", fetcher=fake_fetcher(world))
    assert good.total_return is True

    manufactured = (
        B.picking_cost(p, benchmark=bad, include_rolling=False).full_history("^NSEI").gap_total
        - B.picking_cost(p, benchmark=good, include_rolling=False).full_history("NIFTYBEES.NS").gap_total
    )
    assert manufactured == pytest.approx(0.30 - 0.24, abs=1e-9), "the flagged difference must be the dividend gap"

    rg = B.rolling_gap(p, benchmark=bad)
    assert any("price index" in c for c in rg.caveats)


def test_book_holding_the_benchmark_is_named_and_the_rest_compared(two_asset_world):
    """Part of this book IS the index. Saying so, and re-running on the rest, is
    the difference between an honest comparison and a rigged one."""
    p = _panel(two_asset_world, "NIFTYBEES.NS 50\nBBB.NS 25\nAAA.NS 25")
    b = _bench(two_asset_world, p)
    assert b.weight_in_book > 0
    assert any("already holds NIFTYBEES.NS" in w for w in b.warnings)

    cost = B.picking_cost(p, benchmark=b, include_rolling=False)
    assert "NIFTYBEES.NS" in cost.ex_benchmark
    rest = {r.window.label: r for r in cost.ex_benchmark["NIFTYBEES.NS"]}
    whole = {r.window.label: r for r in cost.named("NIFTYBEES.NS")}
    full_label = next(lbl for lbl in whole if lbl.startswith("full history"))
    assert abs(rest[full_label].gap_total - whole[full_label].gap_total) > 1e-6, (
        "removing the benchmark holding must move the gap; if it does not, the ex-benchmark "
        "table is the same table with a different title"
    )
    # Holding the index drags the whole-book gap toward zero.
    assert abs(whole[full_label].gap_total) < abs(rest[full_label].gap_total)


def test_holdings_are_ordered_by_one_rule_whichever_way_the_book_went(two_asset_world):
    """Leg order is an editorial choice, so it must be a rule and must be stated.
    Furthest-behind-the-index first, on a losing book AND on a winning one — a
    table re-sorted to suit the outcome is an argument, not a description."""
    b = None
    for text in ("AAA.NS 30\nBBB.NS 30\nNOISY.NS 40", "AAA.NS 60\nTRACK.NS 40"):
        p = _panel(two_asset_world, text)
        b = _bench(two_asset_world, p)
        r = B.picking_cost(p, benchmark=b, include_rolling=False).full_history("NIFTYBEES.NS")
        order = [leg.vs_benchmark for leg in r.legs]
        assert order == sorted(order), f"legs out of order for {text!r}: {order}"
        assert r.legs[0].vs_benchmark <= r.legs[-1].vs_benchmark

    winning = _panel(two_asset_world, "AAA.NS 60\nTRACK.NS 40")
    rw = B.picking_cost(winning, benchmark=b, include_rolling=False).full_history("NIFTYBEES.NS")
    assert rw.port_total > rw.bench_total, "this fixture must be a winning book"
    assert rw.legs[0].vs_benchmark < rw.legs[-1].vs_benchmark, (
        "a winning book must still lead with its least-winning position; flipping the sort for "
        "a winner is what turns an analysis engine into a rhetoric engine"
    )
    text = B.format_report(winning, fetcher=fake_fetcher(two_asset_world))
    assert "furthest behind first" in text, "the ordering rule must be stated, not silently applied"


# ==========================================================================
# 3. WINDOWS — the set is a rule, not a choice
# ==========================================================================


def test_default_windows_contain_every_promised_family(two_asset_world):
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    wins = B.default_windows(p)
    kinds = {w.kind for w in wins}
    assert kinds == {"full", "calendar_year", "trailing", "rolling"}
    assert len({w.label for w in wins}) == len(wins), "window labels must be unique"
    lo, hi = p.prices.index[0], p.prices.index[-1]
    for w in wins:
        assert lo <= w.start < w.end <= hi, w.label
        assert w.rule, f"{w.label} carries no rule"
    years = sorted({w.label for w in wins if w.kind == "calendar_year"})
    assert len(years) == len({d.year for d in p.prices.index})
    assert any("part-year" in y for y in years), "a part-year must be labelled, not hidden"
    rolling = [w for w in wins if w.kind == "rolling"]
    assert len(rolling) > 100
    for w in rolling[:: max(1, len(rolling) // 10)]:
        assert 0.97 <= w.years <= 1.01, f"{w.label} is {w.years:.3f} years, not one year"


def test_short_history_skips_windows_with_a_stated_reason():
    world = {
        "NIFTYBEES.NS": (exact_growth(300, 0.10), INR),
        "AAA.NS": (exact_growth(300, 0.20), INR),
    }
    p = _panel(world, "AAA.NS 100")
    cost = B.picking_cost(p, benchmark=_bench(world, p))
    labels = {r.window.label for r in cost.named("NIFTYBEES.NS")}
    assert "trailing 2y" not in labels and "trailing 3y" not in labels
    joined = " ".join(cost.skipped_windows)
    assert "trailing 2y" in joined and "trailing 3y" in joined
    assert "years" in joined, "a skip must carry the measurement that caused it"


def test_caller_supplied_windows_are_labelled_as_a_cherry_pick_risk(two_asset_world):
    """Hand-picking windows is allowed and must be visibly flagged, because it is
    exactly the failure the rule set exists to prevent."""
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    picked = [w for w in B.default_windows(p, include_rolling=False) if w.kind == "full"]
    cost = B.picking_cost(p, benchmark=_bench(two_asset_world, p), windows=picked)
    assert "SUPPLIED BY THE CALLER" in cost.window_rule
    assert B.WINDOW_RULE in cost.window_rule

    auto = B.picking_cost(p, benchmark=_bench(two_asset_world, p), include_rolling=False)
    assert auto.window_rule == B.WINDOW_RULE


def test_short_windows_are_not_annualised_and_thin_ones_have_no_volatility(two_asset_world):
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    b = _bench(two_asset_world, p)
    idx = p.prices.index
    short = B.Window("q", idx[0], idx[40], "trailing", "a deliberately short window")
    thin = B.Window("thin", idx[0], idx[14], "trailing", "a deliberately thin window")
    cost = B.picking_cost(p, benchmark=b, windows=[short, thin])
    res = {r.window.label: r for r in cost.results["NIFTYBEES.NS"]}
    assert np.isfinite(res["q"].port_total)
    assert not np.isfinite(res["q"].port_annual), "a 2-month window must not be annualised"
    assert any("not annualised" in n for n in res["q"].notes)
    assert not np.isfinite(res["thin"].port_vol), "15 prints is below the volatility floor"


def test_window_below_the_observation_floor_is_unusable_with_a_reason(two_asset_world):
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    idx = p.prices.index
    tiny = B.Window("tiny", idx[0], idx[3], "trailing", "below the floor on purpose")
    cost = B.picking_cost(p, benchmark=_bench(two_asset_world, p), windows=[tiny])
    r = cost.results["NIFTYBEES.NS"][0]
    assert r.usable is False
    assert not np.isfinite(r.port_total)
    assert r.notes and str(B.MIN_WINDOW_OBS) in r.notes[0]


# ==========================================================================
# 4. ROLLING GAP — the anti-cherry-pick
# ==========================================================================


def test_rolling_gap_counts_match_the_series_it_reports(two_asset_world):
    p = _panel(two_asset_world, "NOISY.NS 50\nNOISY2.NS 50")
    rg = B.rolling_gap(p, benchmark=_bench(two_asset_world, p))
    assert rg.usable and rg.n_windows == len(rg.series)
    gaps = rg.series["gap"].to_numpy()
    assert rg.beat_count == int((gaps > 0).sum())
    assert rg.beat_fraction == pytest.approx(rg.beat_count / rg.n_windows)
    assert rg.median_gap == pytest.approx(float(np.median(gaps)))
    assert rg.best_gap == pytest.approx(gaps.max())
    assert rg.worst_gap == pytest.approx(gaps.min())
    assert rg.worst_gap <= rg.median_gap <= rg.best_gap
    assert (rg.series["port_total"] - rg.series["bench_total"]).sub(rg.series["gap"]).abs().max() < 1e-12


def test_rolling_gap_is_zero_for_the_index_and_total_for_a_dominating_book(two_asset_world):
    """Two ends of the scale. A book that is the index beats it 0% of the time;
    a book that dominates it every single day beats it 100% of the time."""
    p_same = _panel(two_asset_world, "NIFTYBEES.NS 100")
    rg_same = B.rolling_gap(p_same, benchmark=_bench(two_asset_world, p_same))
    assert rg_same.beat_count == 0
    assert abs(rg_same.series["gap"]).max() < 1e-12

    p_win = _panel(two_asset_world, "AAA.NS 100")
    rg_win = B.rolling_gap(p_win, benchmark=_bench(two_asset_world, p_win))
    assert rg_win.beat_fraction == 1.0
    assert rg_win.worst_gap > 0


def test_rolling_gap_reports_its_overlap_honestly(two_asset_world):
    """493 overlapping one-year windows from three years of data are worth about
    three observations. Reporting the count without the caveat is the exact
    dishonesty this module sells against."""
    p = _panel(two_asset_world, "NOISY.NS 50\nNOISY2.NS 50")
    rg = B.rolling_gap(p, benchmark=_bench(two_asset_world, p))
    span_years = (p.prices.index[-1] - p.prices.index[0]).days / 365.25
    assert rg.n_independent == pytest.approx(span_years, rel=1e-9)
    assert rg.n_independent < rg.n_windows / 10
    assert any("overlap" in c and "not a probability" in c for c in rg.caveats)
    assert str(rg.n_windows) in rg.headline


def test_rolling_gap_on_too_short_a_history_says_so_instead_of_returning_a_number():
    world = {"NIFTYBEES.NS": (exact_growth(150, 0.05), INR), "AAA.NS": (exact_growth(150, 0.09), INR)}
    p = _panel(world, "AAA.NS 100")
    rg = B.rolling_gap(p, benchmark=_bench(world, p), length="1y")
    assert rg.usable is False
    assert rg.n_windows == 0
    assert not np.isfinite(rg.beat_fraction)
    assert "no complete 1y window" in rg.unusable_reason
    assert "0.57 years" in rg.unusable_reason


def test_rolling_length_accepts_years_months_and_day_counts(two_asset_world):
    p = _panel(two_asset_world, "NOISY.NS 50\nNOISY2.NS 50")
    b = _bench(two_asset_world, p)
    one_y = B.rolling_gap(p, benchmark=b, length="1y")
    six_m = B.rolling_gap(p, benchmark=b, length="6m")
    d = B.rolling_gap(p, benchmark=b, length=126)
    assert one_y.length_years == 1.0 and six_m.length_years == pytest.approx(0.5)
    assert six_m.n_windows > one_y.n_windows
    assert d.length_label == "126d"
    with pytest.raises(ValueError, match="cannot read a window length"):
        B.rolling_gap(p, benchmark=b, length="a fortnight")


# ==========================================================================
# 5. BENCHMARK CHOICE — the most gameable part of the product
# ==========================================================================


def test_benchmark_default_follows_the_market_not_the_flattering_number(two_asset_world):
    p = _panel(two_asset_world, "AAA.NS 50\nBBB.NS 50")
    picks = B.choose_benchmarks(p)
    assert [t for t, _ in picks] == ["NIFTYBEES.NS"]
    assert picks[0][1], "a benchmark must arrive with the reason it was chosen"


def test_mixed_currency_book_gets_every_market_ordered_by_exposure():
    n = 760
    world = {
        "NIFTYBEES.NS": (exact_growth(n, 0.30), INR),
        "AAA.NS": (exact_growth(n, 0.40), INR),
        "SPY": (exact_growth(n, 0.50), "USD"),
        "USU.NS": (exact_growth(n, 0.55), "USD"),
        "USDINR=X": (const_series(n, 83.0), INR),
        "INRUSD=X": (const_series(n, 1 / 83.0), "USD"),
    }
    p = panel_from(world, "AAA.NS 10\nUSU.NS 90", base_currency="INR")
    assert p.usable, p.unusable_reason
    picks = B.choose_benchmarks(p)
    tickers = [t for t, _ in picks]
    assert set(tickers) == {"NIFTYBEES.NS", "SPY"}
    assert tickers[0] == "SPY", "the market the book is mostly in comes first"
    assert "ordered by that share" in picks[0][1]

    cost = B.picking_cost(
        p, benchmark=[B.fetch_benchmark(p, t, fetcher=fake_fetcher(world)) for t in tickers], include_rolling=False
    )
    assert set(cost.results) == {"NIFTYBEES.NS", "SPY"}
    gaps = {t: cost.full_history(t).gap_total for t in tickers}
    assert gaps["NIFTYBEES.NS"] != pytest.approx(gaps["SPY"]), (
        "two benchmarks must give two answers; reporting only one of them is the cherry-pick"
    )


def test_foreign_benchmark_is_converted_and_the_fx_move_is_part_of_the_gap():
    """An INR holder comparing against SPY earned the S&P AND the rupee move.
    Skipping the conversion leaves the comparison in the wrong currency, which on
    a moving rate is a wrong answer that looks right."""
    n = 760
    spy = exact_growth(n, 0.50)
    fx = exact_growth(n, 0.20) * (83.0 / 100.0)  # USD/INR drifts 83 -> 99.6
    world = {
        "AAA.NS": (exact_growth(n, 0.60), INR),
        "NIFTYBEES.NS": (exact_growth(n, 0.30), INR),
        "SPY": (spy, "USD"),
        "USDINR=X": (fx, INR),
    }
    p = _panel(world, "AAA.NS 100")
    b = B.fetch_benchmark(p, "SPY", fetcher=fake_fetcher(world))
    assert b.native_currency == "USD" and b.base_currency == INR
    assert "USDINR=X" in b.fx_note and "INR" in b.fx_note

    r = B.picking_cost(p, benchmark=b, include_rolling=False).full_history("SPY")
    # 1.50 of S&P times 1.20 of rupee depreciation = 1.80 for an INR holder.
    assert r.bench_total == pytest.approx(1.50 * 1.20 - 1.0, rel=1e-9)
    assert r.bench_total != pytest.approx(0.50, rel=1e-6), "unconverted SPY would report 50%"
    assert r.gap_total == pytest.approx(0.60 - (1.50 * 1.20 - 1.0), abs=1e-9)


def test_unknown_currency_gets_no_invented_benchmark():
    n = 760
    world = {"XXX.L": (exact_growth(n, 0.2), "GBP"), "YYY.L": (exact_growth(n, 0.3), "GBP")}
    p = panel_from(world, "XXX.L 50\nYYY.L 50")
    assert B.choose_benchmarks(p) == []
    with pytest.raises(ValueError, match="no default benchmark"):
        B.picking_cost(p)


def test_benchmark_shorter_than_the_book_is_reported_not_backfilled():
    n = 760
    short = exact_growth(n, 0.30)
    short = short.iloc[200:]
    world = {"NIFTYBEES.NS": (short, INR), "AAA.NS": (exact_growth(n, 0.60), INR)}
    p = _panel(world, "AAA.NS 100")
    b = B.fetch_benchmark(p, "NIFTYBEES.NS", fetcher=fake_fetcher(world))
    assert any("has no price on" in w for w in b.warnings)
    r = B.picking_cost(p, benchmark=b, include_rolling=False).full_history("NIFTYBEES.NS")
    assert any("effective window" in n for n in r.notes)
    assert r.window.start < pd.Timestamp(short.index[0]) <= pd.Timestamp(r.window.end)


def test_cross_calendar_benchmark_volatility_uses_its_own_trading_days():
    """A market that was shut did not have a flat day. Computing benchmark vol on
    the book's calendar would inject zero-return days and understate its risk."""
    n = 760
    idx = _cal(n)
    bench = pd.Series(gbm(n, drift=0.07, vol=0.16, seed=7).to_numpy(), index=idx)
    bench = bench[bench.index.dayofweek != 2]  # a market closed every Wednesday
    world = {
        "NIFTYBEES.NS": (bench, INR),
        "AAA.NS": (gbm(n, drift=0.07, vol=0.16, seed=8), INR),
    }
    p = _panel(world, "AAA.NS 100")
    b = B.fetch_benchmark(p, "NIFTYBEES.NS", fetcher=fake_fetcher(world))
    assert b.filled_days > 100
    assert any("different calendars" in w for w in b.warnings)
    r = B.picking_cost(p, benchmark=b, include_rolling=False).full_history("NIFTYBEES.NS")
    own = b.native[(b.native.index >= r.window.start) & (b.native.index <= r.window.end)]
    # Computed here from first principles, NOT via B._realised_vol: a test that
    # calls the function it is checking cannot catch a mutation inside it.
    own_rets = own.pct_change().dropna()
    obs_per_year = len(own_rets) / r.years
    assert 200 < obs_per_year < 215, f"the fixture must NOT trade 252 days a year: {obs_per_year:.0f}"
    expected = float(own_rets.std(ddof=1) * math.sqrt(obs_per_year))
    assert r.bench_vol == pytest.approx(expected, rel=1e-12), (
        "benchmark volatility must use the benchmark's own trading days AND its own observation "
        "rate; a hardcoded 252 would inflate this fixture by sqrt(252/210) = 1.10x"
    )
    assert r.bench_vol != pytest.approx(float(own_rets.std(ddof=1) * math.sqrt(252.0)), rel=1e-6)
    filled_vol = B._realised_vol(b.prices.dropna(), r.years)
    # For i.i.d. returns the diluted variance and the larger annualisation factor
    # very nearly cancel, so the two conventions are close but never identical.
    # An implementation that used b.prices would land exactly on filled_vol.
    assert r.bench_vol != pytest.approx(filled_vol, abs=1e-9), (
        f"own-calendar {r.bench_vol:.6f} is indistinguishable from forward-filled "
        f"{filled_vol:.6f}; this test cannot tell the two implementations apart"
    )
    assert any("own" in n and "trading days" in n for n in r.notes)


# ==========================================================================
# 6. THE OUTPUT ITSELF
# ==========================================================================


@pytest.fixture
def report(two_asset_world) -> str:
    p = _panel(two_asset_world, "AAA.NS 30\nBBB.NS 30\nNOISY.NS 20\nNIFTYBEES.NS 20")
    return B.format_report(p, fetcher=fake_fetcher(two_asset_world))


def test_report_contains_no_advice_language(report):
    """The product may state facts about holdings; personalised investment advice
    is regulated and unlicensed here. This test is the line."""
    found = B._advice_words_in(report)
    assert found == [], f"advice language in rendered report: {found}"


def test_report_contains_no_forward_looking_claim(report):
    low = report.lower()
    for phrase in ("will ", "expect", "predict", "going forward", "outlook", "likely", "should"):
        assert phrase not in low, f"forward-looking or prescriptive phrase in report: {phrase!r}"
    assert low.count("forecast") == low.count("not forecasts") == 1


def test_report_leads_with_the_weight_caveat_and_states_the_window_rule(report):
    first = report.split("\n", 1)[0]
    assert "TODAY's weights" in first and "NOT what you earned" in first
    assert "transaction history" in first
    assert B.WINDOW_RULE in report
    assert "not a Sharpe ratio" in report


def test_report_names_the_benchmark_and_why_it_was_chosen(report):
    assert "NIFTYBEES.NS" in report
    assert "broadest liquid Indian equity vehicle" in report
    assert "total-return basis: yes" in report
    assert "already holds NIFTYBEES.NS" in report
    assert "The REST of the book" in report


def test_report_carries_the_rolling_overlap_caveat(report):
    """The rolling headline is the product's defensible claim, and it is only
    defensible next to its overlap caveat. A report that prints "beat the index
    in 223 of 493 windows" without saying those 493 are worth about three is
    doing the thing this module exists to prevent."""
    assert "rolling 1y windows" in report
    assert "overlap" in report and "not a probability" in report
    assert "independent observations, not" in report
    assert "today's weights backwards" in report


def test_report_is_deterministic(two_asset_world, report):
    p = _panel(two_asset_world, "AAA.NS 30\nBBB.NS 30\nNOISY.NS 20\nNIFTYBEES.NS 20")
    assert B.format_report(p, fetcher=fake_fetcher(two_asset_world)) == report


def test_report_reads_the_same_for_a_winning_book(two_asset_world):
    """If the output only reads well when the user lost, this is a rhetoric
    engine and not an analysis engine."""
    p = _panel(two_asset_world, "AAA.NS 100")
    text = B.format_report(p, fetcher=fake_fetcher(two_asset_world))
    assert B._advice_words_in(text) == []
    assert "out-returned NIFTYBEES.NS in" in text
    rg = B.rolling_gap(p, benchmark=_bench(two_asset_world, p))
    assert rg.beat_fraction == 1.0
    assert f"{rg.beat_count} of {rg.n_windows}" in text


# ==========================================================================
# 7. F-18 — a number that never moves is not a result
# ==========================================================================


def test_the_headline_gap_varies_across_windows(two_asset_world):
    """LESSONS F-18: before calling a number good, print its full series and
    confirm it varies. A gap identical across every window would mean the window
    machinery is disconnected and the multi-window claim is decoration."""
    p = _panel(two_asset_world, "NOISY.NS 40\nNOISY2.NS 30\nAAA.NS 15\nBBB.NS 15")
    cost = B.picking_cost(p, benchmark=_bench(two_asset_world, p), include_rolling=False)
    named = [r for r in cost.named("NIFTYBEES.NS") if r.usable]
    gaps = np.array([r.gap_total for r in named], dtype=float)
    assert len(gaps) >= 5
    # Full history and trailing 3y coincide by construction on a 3y panel; every
    # other pair must not.
    assert len(np.unique(np.round(gaps, 10))) >= len(gaps) - 1, [(r.window.label, r.gap_total) for r in named]
    assert gaps.std() > 1e-3, f"gap series is flat across windows: {gaps}"

    rg = B.rolling_gap(p, benchmark=_bench(two_asset_world, p))
    assert rg.series["gap"].std() > 1e-3
    assert rg.best_gap > rg.worst_gap + 1e-3


def test_the_sign_of_the_gap_actually_flips_somewhere(two_asset_world):
    """The whole argument for many windows is that one window can lie. If no
    book in this suite ever flips sign across windows, the argument is untested."""
    p = _panel(two_asset_world, "TRACK.NS 50\nTRACK2.NS 50")
    rg = B.rolling_gap(p, benchmark=_bench(two_asset_world, p))
    assert rg.best_gap > 0 > rg.worst_gap, (
        f"rolling gaps never change sign ({rg.worst_gap:.4f}..{rg.best_gap:.4f}); "
        "the anti-cherry-pick is untested on this fixture"
    )
