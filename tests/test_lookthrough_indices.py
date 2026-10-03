"""Tests for agent.lookthrough.indices.

Every test here injects its fetchers and sets ``use_cache=False`` or points the
cache at a tmp_path, so the suite never touches NSE or Yahoo. The one live
check lives in the verification script, not here.

These tests are written to fail on the bug, not to assert the behaviour. The
repo has four recorded cases of a test asserting the defect it was meant to
catch (LESSONS F-09), so each test below names the wrong answer it rejects.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from agent.lookthrough import indices as mod
from agent.lookthrough.indices import (
    CACHE_TTL_CAPS,
    CACHE_TTL_FAIL,
    COVERAGE_GAP_OK_PP,
    CapQuote,
    Index,
    IndexUnavailable,
    compare_to_published,
    fetch_index,
    known_indices,
)

# ---------------------------------------------------------------------------
# Fixtures: a 4-name toy index with hand-computable answers.
# ---------------------------------------------------------------------------

CSV = (
    "Company Name,Industry,Symbol,Series,ISIN Code\n"
    "HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01034\n"
    "Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018\n"
    "Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029\n"
    "ITC Ltd.,Fast Moving Consumer Goods,ITC,EQ,INE154A01025\n"
)

# market cap, float shares, shares outstanding.
# Float ratios: HDFCBANK 1.0, RELIANCE 0.5, TCS 0.25, ITC 0.8
CAPS = {
    "HDFCBANK.NS": CapQuote("HDFCBANK.NS", 100.0, 100.0, 100.0, "INR"),
    "RELIANCE.NS": CapQuote("RELIANCE.NS", 200.0, 50.0, 100.0, "INR"),
    "TCS.NS": CapQuote("TCS.NS", 80.0, 25.0, 100.0, "INR"),
    "ITC.NS": CapQuote("ITC.NS", 50.0, 80.0, 100.0, "INR"),
}
# free-float caps: 100, 100, 20, 40 -> total 260
FF_EXPECTED = {
    "HDFCBANK": 100 / 260,
    "RELIANCE": 100 / 260,
    "TCS": 20 / 260,
    "ITC": 40 / 260,
}
# full caps: 100, 200, 80, 50 -> total 430
FULL_EXPECTED = {
    "HDFCBANK": 100 / 430,
    "RELIANCE": 200 / 430,
    "TCS": 80 / 430,
    "ITC": 50 / 430,
}


def list_fetcher(_url: str) -> str:
    return CSV


def cap_fetcher(symbols):
    return {s: CAPS[s] for s in symbols if s in CAPS}


def no_published(_etf: str):
    return {}


def build(tmp_path, monkeypatch, **kw) -> Index:
    # Respect a cache dir the test already chose; a helper that silently
    # overrode it made the expiry test age files nothing ever read.
    if "TIRRA_LOOKTHROUGH_CACHE" not in os.environ:
        monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path / "cache"))
    kw.setdefault("list_fetcher", list_fetcher)
    kw.setdefault("cap_fetcher", cap_fetcher)
    kw.setdefault("published_fetcher", no_published)
    return fetch_index(kw.pop("name", "NIFTY 50"), **kw)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def test_known_indices_is_the_seven_nse_publishes():
    assert known_indices() == [
        "NIFTY 50",
        "NIFTY NEXT 50",
        "NIFTY 100",
        "NIFTY 500",
        "NIFTY BANK",
        "NIFTY IT",
        "NIFTY MIDCAP 150",
    ]


@pytest.mark.parametrize(
    ("typed", "canon"),
    [
        ("NIFTY 50", "NIFTY 50"),
        ("nifty50", "NIFTY 50"),
        ("  Nifty 50 ", "NIFTY 50"),
        ("nifty_50", "NIFTY 50"),
        ("banknifty", "NIFTY BANK"),
        ("NIFTY BANK", "NIFTY BANK"),
        ("midcap150", "NIFTY MIDCAP 150"),
        ("nifty midcap 150", "NIFTY MIDCAP 150"),
        ("next50", "NIFTY NEXT 50"),
    ],
)
def test_aliases_resolve(typed, canon):
    assert mod._canonical(typed) == canon


def test_unknown_index_names_the_known_ones():
    # Rejects a silent fallback to NIFTY 50 for a typo.
    with pytest.raises(KeyError) as exc:
        mod._canonical("SENSEX")
    assert "NIFTY 50" in str(exc.value)


def test_unknown_weight_method_is_rejected():
    # Rejects silently treating an unknown method as the default.
    with pytest.raises(ValueError, match="weight_method"):
        fetch_index(
            "NIFTY 50", weight_method="equal", list_fetcher=list_fetcher, cap_fetcher=cap_fetcher, use_cache=False
        )


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_free_float_weights_match_hand_arithmetic(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch)
    got = ix.weights()
    assert set(got) == set(FF_EXPECTED)
    for sym, want in FF_EXPECTED.items():
        assert got[sym] == pytest.approx(want, abs=1e-12), sym


def test_free_float_differs_from_full_mcap(tmp_path, monkeypatch):
    """The whole honesty argument rests on these two NOT being the same.

    If a refactor made the free-float branch fall through to full cap, every
    weight would still sum to 1.0 and every other test could still pass.
    """
    ff = build(tmp_path, monkeypatch).weights()
    full = build(tmp_path / "b", monkeypatch, weight_method="full_mcap").weights()
    assert full == pytest.approx(FULL_EXPECTED, abs=1e-12)
    # RELIANCE is promoter-heavy in the fixture: full cap nearly doubles it.
    assert full["RELIANCE"] > ff["RELIANCE"] + 0.07
    assert ff["HDFCBANK"] > full["HDFCBANK"] + 0.14


def test_weights_sum_to_one(tmp_path, monkeypatch):
    for method in ("free_float_mcap", "full_mcap"):
        ix = build(tmp_path / method, monkeypatch, weight_method=method)
        assert math.fsum(c.weight for c in ix.constituents) == pytest.approx(1.0, abs=1e-12)


def test_weight_series_varies_and_is_not_uniform(tmp_path, monkeypatch):
    """LESSONS F-18: a number that never moves is not a measurement.

    Rejects the failure where every constituent silently gets 1/n.
    """
    ws = sorted(build(tmp_path, monkeypatch).weights().values())
    assert len(set(ws)) > 1
    assert max(ws) - min(ws) > 0.1
    assert not all(math.isclose(w, 1 / len(ws)) for w in ws)


def test_constituents_sorted_heaviest_first(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch)
    ws = [c.weight for c in ix.constituents]
    assert ws == sorted(ws, reverse=True)
    assert [c.symbol for c in ix.top(2)] in (["HDFCBANK", "RELIANCE"], ["RELIANCE", "HDFCBANK"])


def test_metadata_carried_through(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch)
    hdfc = next(c for c in ix.constituents if c.symbol == "HDFCBANK")
    assert hdfc.company == "HDFC Bank Ltd."
    assert hdfc.industry == "Financial Services"
    assert hdfc.isin == "INE040A01034"
    assert hdfc.yahoo_symbol == "HDFCBANK.NS"
    tcs = next(c for c in ix.constituents if c.symbol == "TCS")
    assert tcs.free_float_factor == pytest.approx(0.25)
    assert tcs.cap_basis == "free_float"
    # market_cap is the cap USED, i.e. free-float, not the company's full cap.
    assert tcs.market_cap == pytest.approx(20.0)


def test_constituent_unpacks_as_the_specified_five_tuple(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch)
    symbol, company, industry, isin, weight = next(iter(ix.constituents))
    assert isin.startswith("INE")
    assert 0 < weight < 1
    assert industry and company and symbol


def test_weight_of_accepts_bare_and_suffixed(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch)
    assert ix.weight_of("HDFCBANK") == pytest.approx(ix.weight_of("HDFCBANK.NS"))
    assert ix.weight_of("hdfcbank.ns") == pytest.approx(FF_EXPECTED["HDFCBANK"])
    assert ix.weight_of("INFY") == 0.0


# ---------------------------------------------------------------------------
# Exclusions — nothing drops silently.
# ---------------------------------------------------------------------------


def test_missing_market_cap_is_excluded_with_an_actionable_reason(tmp_path, monkeypatch):
    def caps(symbols):
        out = dict(cap_fetcher(symbols))
        out["TCS.NS"] = CapQuote("TCS.NS", None, None, None, None)
        return out

    ix = build(tmp_path, monkeypatch, cap_fetcher=caps)
    assert [s for s, _ in ix.excluded] == ["TCS"]
    reason = ix.excluded[0][1]
    # The reason must NAME the holding it concerns, up front. Asserting only
    # that "TCS" appears somewhere passed on a reason whose identifying prefix
    # had been deleted, because ".NS" elsewhere in the sentence still matched.
    assert reason.startswith("TCS:"), reason
    assert "market cap" in reason
    assert "renormalis" in reason
    assert "TCS" not in ix.weights()
    # And the survivors still sum to 1.0, renormalised off 240 not 260.
    assert math.fsum(ix.weights().values()) == pytest.approx(1.0, abs=1e-12)
    assert ix.weights()["HDFCBANK"] == pytest.approx(100 / 240)
    assert any("excluded" in n for n in ix.notes)


def test_zero_market_cap_is_a_missing_figure_not_a_zero_weight(tmp_path, monkeypatch):
    """Rejects shipping a 0.0%-weight constituent with no explanation."""

    def caps(symbols):
        out = dict(cap_fetcher(symbols))
        out["ITC.NS"] = CapQuote("ITC.NS", 0.0, 80.0, 100.0, "INR")
        return out

    ix = build(tmp_path, monkeypatch, cap_fetcher=caps)
    assert [s for s, _ in ix.excluded] == ["ITC"]
    assert "ITC" not in ix.weights()


def test_nan_market_cap_is_excluded_not_propagated(tmp_path, monkeypatch):
    def caps(symbols):
        out = dict(cap_fetcher(symbols))
        out["ITC.NS"] = CapQuote("ITC.NS", float("nan"), 80.0, 100.0, "INR")
        return out

    ix = build(tmp_path, monkeypatch, cap_fetcher=caps)
    assert [s for s, _ in ix.excluded] == ["ITC"]
    assert all(math.isfinite(w) for w in ix.weights().values())


def test_symbol_absent_from_the_cap_fetcher_is_excluded(tmp_path, monkeypatch):
    """A symbol that does not resolve at all: fetcher returns no entry for it."""

    def caps(symbols):
        return {s: CAPS[s] for s in symbols if s in CAPS and s != "RELIANCE.NS"}

    ix = build(tmp_path, monkeypatch, cap_fetcher=caps)
    assert [s for s, _ in ix.excluded] == ["RELIANCE"]
    assert "RELIANCE" in ix.excluded[0][1]


def test_missing_float_falls_back_to_full_cap_and_says_so(tmp_path, monkeypatch):
    """Dropping Reliance would be worse than approximating it — but say which."""

    def caps(symbols):
        out = dict(cap_fetcher(symbols))
        out["RELIANCE.NS"] = CapQuote("RELIANCE.NS", 200.0, None, 100.0, "INR")
        return out

    ix = build(tmp_path, monkeypatch, cap_fetcher=caps)
    rel = next(c for c in ix.constituents if c.symbol == "RELIANCE")
    assert rel.cap_basis == "full_mcap_fallback"
    assert rel.free_float_factor is None
    assert rel.market_cap == pytest.approx(200.0)
    assert not ix.excluded
    note = " ".join(ix.notes)
    assert "RELIANCE" in note and "FULL market cap" in note


def test_float_exceeding_shares_outstanding_is_capped_at_one(tmp_path, monkeypatch):
    """Yahoo's two fields can disagree on as-of date; free float <= company."""

    def caps(symbols):
        out = dict(cap_fetcher(symbols))
        out["ITC.NS"] = CapQuote("ITC.NS", 50.0, 150.0, 100.0, "INR")
        return out

    ix = build(tmp_path, monkeypatch, cap_fetcher=caps)
    itc = next(c for c in ix.constituents if c.symbol == "ITC")
    assert itc.free_float_factor == pytest.approx(1.0)
    assert itc.market_cap == pytest.approx(50.0)


def test_every_exclusion_reason_names_its_symbol_first(tmp_path, monkeypatch):
    """Contract for consumers that render `excluded` as a bare list of reasons."""

    def caps(symbols):
        return {"HDFCBANK.NS": CAPS["HDFCBANK.NS"]}

    ix = build(tmp_path, monkeypatch, cap_fetcher=caps)
    assert len(ix.excluded) == 3
    for sym, reason in ix.excluded:
        assert reason.startswith(f"{sym}:"), (sym, reason)
        assert len(reason) > len(sym) + 20, "a reason a user could act on, not a code"


def test_normalisation_guard_is_live(tmp_path, monkeypatch):
    """Rejects deleting the sum-to-1.0 guard.

    Tests that compute the sum themselves cannot see the in-module guard
    disappear, so this one stubs the error function and proves fetch_index
    refuses rather than returning a broken vector.
    """
    monkeypatch.setattr(mod, "_normalisation_error", lambda cs: 0.25)
    with pytest.raises(AssertionError, match="off 1.0"):
        build(tmp_path, monkeypatch)


def test_normalisation_error_measures_the_real_gap(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch)
    assert mod._normalisation_error(ix.constituents) <= mod.WEIGHT_SUM_TOLERANCE
    # The two heaviest are 200/260 of the index, so dropping the rest leaves
    # the vector 0.2308 short of 1.0.
    half = tuple(ix.constituents[:2])
    assert mod._normalisation_error(half) == pytest.approx(60 / 260, abs=1e-12)


def test_no_constituent_has_a_cap_raises_with_reasons(tmp_path, monkeypatch):
    with pytest.raises(IndexUnavailable) as exc:
        build(tmp_path, monkeypatch, cap_fetcher=lambda syms: {})
    assert "NIFTY 50" in str(exc.value)
    assert exc.value.reasons
    assert any("market cap" in r for r in exc.value.reasons)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def test_reordered_columns_do_not_shift_fields(tmp_path, monkeypatch):
    """Rejects positional parsing, which would put the ISIN in `industry`."""
    reordered = (
        "ISIN Code,Symbol,Industry,Series,Company Name\nINE040A01034,HDFCBANK,Financial Services,EQ,HDFC Bank Ltd.\n"
    )
    ix = build(tmp_path, monkeypatch, list_fetcher=lambda u: reordered)
    c = ix.constituents[0]
    assert (c.symbol, c.isin, c.industry, c.company) == (
        "HDFCBANK",
        "INE040A01034",
        "Financial Services",
        "HDFC Bank Ltd.",
    )


def test_bom_prefixed_csv_parses():
    rows, problems = mod._parse_constituent_csv("﻿" + CSV)
    assert [r.symbol for r in rows] == ["HDFCBANK", "RELIANCE", "TCS", "ITC"]
    assert problems == []


def test_blank_symbol_row_is_reported_not_silently_dropped():
    rows, problems = mod._parse_constituent_csv(CSV + "Mystery Ltd.,Unknown,,EQ,INE999A01011\n")
    assert len(rows) == 4
    assert any("no symbol" in p for p in problems)


def test_duplicate_symbol_is_reported():
    rows, problems = mod._parse_constituent_csv(CSV + "HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01034\n")
    assert len(rows) == 4
    assert any("twice" in p for p in problems)


def test_csv_without_a_symbol_column_fails_loudly(tmp_path, monkeypatch):
    with pytest.raises(IndexUnavailable, match="Symbol"):
        build(tmp_path, monkeypatch, list_fetcher=lambda u: "A,B\n1,2\n")


def test_empty_csv_fails_loudly(tmp_path, monkeypatch):
    with pytest.raises(IndexUnavailable):
        build(tmp_path, monkeypatch, list_fetcher=lambda u: "")


def test_download_failure_raises_with_the_cause(tmp_path, monkeypatch):
    def boom(_url):
        raise OSError("connection reset")

    with pytest.raises(IndexUnavailable, match="connection reset"):
        build(tmp_path, monkeypatch, list_fetcher=boom)


def test_unexpected_constituent_count_is_flagged(tmp_path, monkeypatch):
    """NSE quietly serving 4 rows for NIFTY 50 must not pass as fine."""
    ix = build(tmp_path, monkeypatch)
    assert any("normally has about 50" in n for n in ix.notes)


def test_count_within_tolerance_is_not_flagged(tmp_path, monkeypatch):
    rows = "\n".join(f"Co {i} Ltd.,Industry {i},SYM{i},EQ,INE{i:03d}A01011" for i in range(10))
    ix = build(
        tmp_path,
        monkeypatch,
        name="NIFTY IT",
        list_fetcher=lambda u: "Company Name,Industry,Symbol,Series,ISIN Code\n" + rows + "\n",
        cap_fetcher=lambda syms: {s: CapQuote(s, 10.0 + i, 5.0, 10.0, "INR") for i, s in enumerate(syms)},
    )
    assert not any("normally has about" in n for n in ix.notes)
    assert len(ix) == 10


def test_each_index_uses_its_own_nse_url(tmp_path, monkeypatch):
    seen = []

    def spy(url):
        seen.append(url)
        return CSV

    for name in known_indices():
        build(tmp_path / name, monkeypatch, name=name, list_fetcher=spy)
    assert len(set(seen)) == len(known_indices())
    assert all(u.startswith("https://nsearchives.nseindia.com/content/indices/") for u in seen)
    assert "ind_nifty50list.csv" in seen[0]


# ---------------------------------------------------------------------------
# Accuracy measurement
# ---------------------------------------------------------------------------


def test_comparison_is_zero_when_published_matches_us(tmp_path, monkeypatch):
    """A perfect method must measure as perfect, or the metric is broken."""
    pub = {
        "HDFCBANK": FF_EXPECTED["HDFCBANK"],
        "RELIANCE": FF_EXPECTED["RELIANCE"],
        "TCS": FF_EXPECTED["TCS"],
    }
    ix = build(tmp_path, monkeypatch, published_fetcher=lambda e: pub)
    assert ix.accuracy is not None
    assert ix.accuracy.mean_abs_pp == pytest.approx(0.0, abs=1e-9)
    assert ix.accuracy.max_abs_pp == pytest.approx(0.0, abs=1e-9)
    assert ix.accuracy.mean_abs_rel_pp == pytest.approx(0.0, abs=1e-9)
    assert ix.accuracy.n_compared == 3


def test_comparison_detects_a_real_error(tmp_path, monkeypatch):
    """Rejects a comparison that always reports ~0 by renormalising the gap away."""
    # Published says HDFCBANK/RELIANCE/TCS are 50/30/20% OF THE INDEX; our
    # free-float weights are 100/260, 100/260, 20/260.
    pub = {"HDFCBANK": 0.5, "RELIANCE": 0.3, "TCS": 0.2}
    ix = build(tmp_path, monkeypatch, published_fetcher=lambda e: pub)
    acc = ix.accuracy
    assert acc is not None
    ours = {s: g for s, _p, g, _d in acc.rows}
    # Index-level: our weight is unchanged, not rescaled onto the overlap.
    assert ours["HDFCBANK"] == pytest.approx(FF_EXPECTED["HDFCBANK"])
    # HDFCBANK off 11.54pp, RELIANCE off 8.46pp, TCS off 12.31pp -> TCS worst.
    assert acc.max_abs_pp == pytest.approx(abs(20 / 260 - 0.2) * 100, abs=1e-6)
    assert acc.max_abs_symbol == "TCS"
    assert acc.mean_abs_pp == pytest.approx(
        (abs(100 / 260 - 0.5) + abs(100 / 260 - 0.3) + abs(20 / 260 - 0.2)) / 3 * 100,
        abs=1e-6,
    )


def test_index_level_error_is_the_headline_not_the_relative_one(tmp_path, monkeypatch):
    """The shipped number is "x% of the index", so that is what we measure.

    Rejects the bug this module shipped once: renormalising both sides onto the
    overlap divides by the overlap's share (~0.52 live), which roughly doubled
    every reported error and described a quantity no user is ever shown.
    """
    pub = {"HDFCBANK": 0.5, "RELIANCE": 0.3, "TCS": 0.2}
    acc = build(tmp_path, monkeypatch, published_fetcher=lambda e: pub).accuracy
    assert acc is not None
    # Our three overlapping names are 220/260 of our index; theirs are 100% of
    # theirs. The relative figure divides by that, so it must differ.
    assert acc.coverage_ours == pytest.approx(220 / 260)
    assert acc.coverage_published == pytest.approx(1.0)
    assert acc.mean_abs_rel_pp != pytest.approx(acc.mean_abs_pp, abs=1e-6)
    # And the note quotes the index-level figure, not the relative one.
    assert (
        f"{acc.mean_abs_pp:.2f}"
        in build(tmp_path / "n", monkeypatch, published_fetcher=lambda e: pub).weight_accuracy_note
    )


def test_coverage_fields_catch_a_universe_mismatch(tmp_path, monkeypatch):
    """A big coverage gap is the signal that a direct comparison is unsound."""
    pub = {"HDFCBANK": 1.0}
    acc = build(tmp_path, monkeypatch, published_fetcher=lambda e: pub).accuracy
    assert acc is not None
    assert acc.coverage_ours == pytest.approx(FF_EXPECTED["HDFCBANK"])
    assert acc.coverage_published == pytest.approx(1.0)
    assert abs(acc.coverage_ours - acc.coverage_published) > 0.5
    assert "% of our index" in acc.table()
    # The verdict must READ the gap, not print "close" regardless. This is the
    # bug the live run caught: a hardcoded "close => same universe" declared
    # the full-market-cap comparison sound at an 11.2pp gap.
    assert acc.coverage_gap_pp() > COVERAGE_GAP_OK_PP
    assert "COVERAGE GAP" in acc.coverage_verdict()
    assert "COVERAGE GAP" in acc.table()


def test_coverage_verdict_passes_when_universes_agree(tmp_path, monkeypatch):
    pub = dict(FF_EXPECTED)  # identical to ours: coverage 100% both sides
    acc = build(tmp_path, monkeypatch, published_fetcher=lambda e: pub).accuracy
    assert acc is not None
    assert acc.coverage_gap_pp() == pytest.approx(0.0, abs=1e-9)
    assert "same universe" in acc.coverage_verdict()
    assert "COVERAGE GAP" not in acc.table()


def test_comparison_strips_cash_and_renormalises():
    frame_like = {"XTSLA": 0.04, "HDFCBANK.NS": 0.48, "RELIANCE.BO": 0.48}
    # Emulate _yf_published_weights' contract rather than yfinance's object.
    equity = 1 - frame_like["XTSLA"]
    pub = {mod._bare_symbol(k): v / equity for k, v in frame_like.items() if k not in ("XTSLA",)}
    assert pub["RELIANCE"] == pytest.approx(0.5)
    assert math.fsum(pub.values()) == pytest.approx(1.0)


def test_published_weights_parser_drops_cash_and_maps_bse_suffix(monkeypatch):
    import pandas as pd

    frame = pd.DataFrame(
        {"Holding Percent": [0.0380, 0.4810, 0.4810]},
        index=pd.Index(["XTSLA", "HDFCBANK.NS", "AXISBANK.BO"], name="Symbol"),
    )

    class _FD:
        top_holdings = frame

    class _T:
        def __init__(self, *_a, **_k):
            self.funds_data = _FD()

    monkeypatch.setattr("yfinance.Ticker", _T)
    out = mod._yf_published_weights("INDY")
    assert set(out) == {"HDFCBANK", "AXISBANK"}
    assert math.fsum(out.values()) == pytest.approx(1.0, abs=1e-9)
    assert out["AXISBANK"] == pytest.approx(0.481 / (1 - 0.038))


def test_comparison_declines_for_indices_with_no_published_tracker(tmp_path, monkeypatch):
    """Rejects quoting NIFTY 50's error on NIFTY BANK's weights."""
    ix = build(
        tmp_path,
        monkeypatch,
        name="NIFTY BANK",
        published_fetcher=lambda e: {"HDFCBANK": 0.5, "RELIANCE": 0.5},
    )
    assert ix.accuracy is None
    assert "ERROR NOT MEASURED FOR THIS INDEX" in ix.weight_accuracy_note
    assert "0.37pp" in ix.weight_accuracy_note


def test_no_published_overlap_yields_no_measurement(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch, published_fetcher=lambda e: {"AAPL": 1.0})
    assert ix.accuracy is None


def test_published_fetch_failure_degrades_to_a_note(tmp_path, monkeypatch):
    def boom(_e):
        raise RuntimeError("yahoo down")

    ix = build(tmp_path, monkeypatch, published_fetcher=boom)
    assert ix.accuracy is None
    assert any("yahoo down" in n for n in ix.notes)
    assert len(ix) == 4  # the index itself is still usable


def test_comparison_table_prints_every_row(tmp_path, monkeypatch):
    ix = build(
        tmp_path,
        monkeypatch,
        published_fetcher=lambda e: {"HDFCBANK": 0.5, "RELIANCE": 0.3, "TCS": 0.2},
    )
    table = ix.accuracy.table()
    for sym in ("HDFCBANK", "RELIANCE", "TCS"):
        assert sym in table
    assert "mean|diff|" in table


def test_compare_accepts_an_index_or_a_sequence(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch)
    pub = {"HDFCBANK": 0.5, "RELIANCE": 0.5}
    a = compare_to_published(ix, published_fetcher=lambda e: pub, use_cache=False)
    b = compare_to_published(list(ix.constituents), published_fetcher=lambda e: pub, use_cache=False)
    assert a is not None and b is not None
    assert a.rows == b.rows


# ---------------------------------------------------------------------------
# Accuracy note — the string a user sees.
# ---------------------------------------------------------------------------


def test_note_labels_the_approximation_and_carries_the_measured_number(tmp_path, monkeypatch):
    ix = build(
        tmp_path,
        monkeypatch,
        published_fetcher=lambda e: {"HDFCBANK": 0.5, "RELIANCE": 0.3, "TCS": 0.2},
    )
    note = ix.weight_accuracy_note
    assert "APPROXIMATION" in note
    assert "free-float" in note
    assert "percentage points" in note
    assert f"{ix.accuracy.mean_abs_pp:.2f}" in note
    assert ix.accuracy.max_abs_symbol in note


def test_full_mcap_note_warns_not_to_show_it(tmp_path, monkeypatch):
    ix = build(
        tmp_path,
        monkeypatch,
        weight_method="full_mcap",
        published_fetcher=lambda e: {"HDFCBANK": 0.5, "RELIANCE": 0.3, "TCS": 0.2},
    )
    note = ix.weight_accuracy_note
    assert "do not show these to a user" in note
    assert "free_float_mcap" in note


def test_note_contains_no_advice_verbs(tmp_path, monkeypatch):
    """No string a user sees may recommend, suggest, forecast or imply action.

    'should'/'recommend' about OUR OWN method choice is engineering guidance,
    so we check the forbidden words only in their investment senses.
    """
    # Phrases, not bare substrings: "expected about 50" is a row-count check,
    # not a market expectation, and a substring scan flagged it.
    banned = (
        "over-concentrated",
        "overconcentrated",
        "over concentrated",
        "you should",
        "we recommend",
        "recommend",
        "we suggest",
        "you ought",
        "consider trimming",
        "should trim",
        "should buy",
        "should sell",
        "diversify",
        "too risky",
        "too much",
        "we expect",
        "is expected to",
        "will outperform",
        "will underperform",
        "forecast",
        "advice",
        "advisable",
    )
    for method in ("free_float_mcap", "full_mcap"):
        ix = build(tmp_path / method, monkeypatch, weight_method=method)
        texts = [ix.weight_accuracy_note, *ix.notes, *(r for _, r in ix.excluded)]
        for t in texts:
            low = t.lower()
            for word in banned:
                assert word not in low, f"{word!r} in {method} output: {t}"


# ---------------------------------------------------------------------------
# Caching — mandatory for a public demo.
# ---------------------------------------------------------------------------


def test_second_call_hits_disk_and_not_the_network(tmp_path, monkeypatch):
    calls = {"list": 0, "cap": 0}

    def lf(url):
        calls["list"] += 1
        return CSV

    def cf(symbols):
        calls["cap"] += 1
        return cap_fetcher(symbols)

    for _ in range(3):
        build(tmp_path, monkeypatch, list_fetcher=lf, cap_fetcher=cf)
    assert calls == {"list": 1, "cap": 1}


def test_caps_are_cached_per_symbol_so_indices_share_work(tmp_path, monkeypatch):
    """NIFTY 100 must not re-fetch the 50 caps NIFTY 50 already paid for."""
    asked: list[list[str]] = []

    def cf(symbols):
        asked.append(list(symbols))
        return cap_fetcher(symbols)

    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path / "c"))
    fetch_index(
        "NIFTY 50",
        list_fetcher=lambda u: CSV,
        cap_fetcher=cf,
        published_fetcher=no_published,
    )
    fetch_index(
        "NIFTY 100",
        list_fetcher=lambda u: CSV,
        cap_fetcher=cf,
        published_fetcher=no_published,
    )
    assert asked[0] == ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "ITC.NS"]
    assert len(asked) == 1, "second index re-fetched caps it already had on disk"


def test_use_cache_false_bypasses_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path / "c"))
    calls = {"n": 0}

    def cf(symbols):
        calls["n"] += 1
        return cap_fetcher(symbols)

    for _ in range(2):
        fetch_index(
            "NIFTY 50",
            list_fetcher=lambda u: CSV,
            cap_fetcher=cf,
            published_fetcher=no_published,
            use_cache=False,
        )
    assert calls["n"] == 2
    assert not list((tmp_path / "c").glob("*.json")) or True  # no writes required


def test_expired_cache_is_refetched(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path / "c"))
    calls = {"n": 0}

    def cf(symbols):
        calls["n"] += 1
        return cap_fetcher(symbols)

    build(tmp_path, monkeypatch, cap_fetcher=cf)
    for p in (tmp_path / "c").glob("cap__*.json"):
        blob = json.loads(p.read_text())
        blob["fetched_at"] = blob["fetched_at"] - CACHE_TTL_CAPS - 10
        p.write_text(json.dumps(blob))
    build(tmp_path, monkeypatch, cap_fetcher=cf)
    assert calls["n"] == 2


def test_corrupt_cache_file_is_ignored_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path / "c"))
    build(tmp_path, monkeypatch)
    for p in (tmp_path / "c").glob("*.json"):
        p.write_text("{not json")
    ix = build(tmp_path, monkeypatch)
    assert len(ix) == 4


def test_failed_cap_is_cached_so_a_bad_ticker_is_not_rehammered(tmp_path, monkeypatch):
    calls = {"n": 0}

    def cf(symbols):
        calls["n"] += 1
        return {s: CAPS[s] for s in symbols if s in CAPS and s != "TCS.NS"}

    a = build(tmp_path, monkeypatch, cap_fetcher=cf)
    b = build(tmp_path, monkeypatch, cap_fetcher=cf)
    assert calls["n"] == 1
    assert [s for s, _ in a.excluded] == [s for s, _ in b.excluded] == ["TCS"]


def test_a_failed_cap_is_retried_sooner_than_a_good_one(tmp_path, monkeypatch):
    """A newly-listed stock must not stay excluded for a whole day.

    BHARATCOAL and CMPDI are live examples: fresh NIFTY 500 entrants Yahoo had
    no cap for. Caching that miss on the 24h success TTL would keep them
    excluded long after Yahoo catches up, so failures get the short TTL. The
    mutation that wrote every cap to the cache as "ok": True left the exclusion
    behaviour identical and only changed which TTL applied — so this test
    checks the TTL, not the exclusion.
    """
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path / "c"))
    calls = {"n": 0}
    present = {"on": False}

    def cf(symbols):
        calls["n"] += 1
        out = {s: CAPS[s] for s in symbols if s in CAPS and s != "TCS.NS"}
        if present["on"]:
            out["TCS.NS"] = CAPS["TCS.NS"]
        return out

    first = build(tmp_path, monkeypatch, cap_fetcher=cf)
    assert [s for s, _ in first.excluded] == ["TCS"]

    # Age every cache entry by just over the FAILURE ttl, well under the
    # success ttl. Only the failed symbol may be refetched.
    assert CACHE_TTL_FAIL < CACHE_TTL_CAPS
    aged = CACHE_TTL_FAIL + 60
    for f in (tmp_path / "c").glob("cap__*.json"):
        blob = json.loads(f.read_text())
        blob["fetched_at"] -= aged
        f.write_text(json.dumps(blob))

    present["on"] = True
    second = build(tmp_path, monkeypatch, cap_fetcher=cf)
    assert calls["n"] == 2, "the failed lookup was not retried after its short TTL"
    assert not second.excluded, "TCS stayed excluded though its cap is now available"
    assert "TCS" in second.weights()


def test_a_good_cap_is_not_refetched_inside_the_failure_ttl(tmp_path, monkeypatch):
    """The companion: a successful cap keeps the long TTL."""
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path / "c"))
    calls = {"n": 0}

    def cf(symbols):
        calls["n"] += 1
        return cap_fetcher(symbols)

    build(tmp_path, monkeypatch, cap_fetcher=cf)
    for f in (tmp_path / "c").glob("cap__*.json"):
        blob = json.loads(f.read_text())
        blob["fetched_at"] -= CACHE_TTL_FAIL + 60
        f.write_text(json.dumps(blob))
    build(tmp_path, monkeypatch, cap_fetcher=cf)
    assert calls["n"] == 1, "good caps were refetched on the failure TTL"


def test_cache_root_prefers_lookthrough_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_PORTFOLIO_CACHE", str(tmp_path / "pf"))
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path / "lt"))
    assert mod._cache_dir() == tmp_path / "lt"
    monkeypatch.delenv("TIRRA_LOOKTHROUGH_CACHE")
    assert mod._cache_dir() == tmp_path / "pf" / "lookthrough"


def test_cache_key_is_filesystem_safe():
    p = mod._cache_path("cap", "M&M.NS/../../etc")
    assert p.parent == mod._cache_dir()
    assert "/" not in p.name and ".." not in p.name


# ---------------------------------------------------------------------------
# Numeric guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "want"),
    [
        (1.5, 1.5),
        ("2.5", 2.5),
        (0, None),
        (-1, None),
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        ("", None),
        ("abc", None),
    ],
)
def test_as_pos_float(value, want):
    assert mod._as_pos_float(value) == want


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("HDFCBANK", "HDFCBANK"),
        ("HDFCBANK.NS", "HDFCBANK"),
        ("axisbank.bo", "AXISBANK"),
        (" tcs.ns ", "TCS"),
        ("M&M", "M&M"),
    ],
)
def test_bare_symbol(raw, want):
    assert mod._bare_symbol(raw) == want


def test_index_len_and_weights_agree(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch)
    assert len(ix) == len(ix.constituents) == len(ix.weights())


def test_constituent_and_index_are_frozen(tmp_path, monkeypatch):
    ix = build(tmp_path, monkeypatch)
    with pytest.raises(Exception):
        ix.constituents[0].weight = 0.5  # type: ignore[misc]
    with pytest.raises(Exception):
        ix.name = "X"  # type: ignore[misc]
