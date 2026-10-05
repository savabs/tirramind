"""Tests for agent.portfolio.beliefs.

These are written against the two ways this module can hurt a stranger:

  1. It computes a number the data cannot support and prints it with a
     decimal point ("your hedge broke in 2023", n=30).
  2. It quietly drops a holding, a day or a corporate action and reports a
     confident answer about the rest.

So most of the assertions here are about REFUSALS, ``None``s and the contents
of ``excluded`` — not about whether a correlation is correct to four places.
The arithmetic is delegated to agent.verify.stats, which has its own suite.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from agent.portfolio import beliefs
from agent.portfolio.beliefs import (
    MIN_OVERLAP_DAYS,
    SUSPECT_RETURN_THRESHOLD,
    drawdown_coincidence,
    regime_split,
    render_drawdown_coincidence,
    render_hedge_result,
    scrub_suspect_returns,
    test_hedge,
)

# --------------------------------------------------------------------------- #
# Fixtures — deterministic synthetic panels. No network, no clock.
# --------------------------------------------------------------------------- #


def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    """n business days starting at `start`."""
    return pd.bdate_range(start=start, periods=n)


def _panel(n: int = 500, seed: int = 7, start: str = "2020-01-01") -> pd.DataFrame:
    """A book of three correlated equities, one true hedge, one noise asset.

    HEDGE is constructed as ``-0.9 * market + idiosyncratic noise`` so its
    correlation with the equal-weighted book is strongly and unambiguously
    negative. NOISE is independent of everything.
    """
    rng = np.random.default_rng(seed)
    idx = _dates(n, start)
    market = rng.normal(0.0, 0.012, n)
    frame = pd.DataFrame(
        {
            "AAA": market + rng.normal(0.0, 0.006, n),
            "BBB": market + rng.normal(0.0, 0.006, n),
            "CCC": market + rng.normal(0.0, 0.006, n),
            "HEDGE": -0.9 * market + rng.normal(0.0, 0.004, n),
            "NOISE": rng.normal(0.0, 0.010, n),
        },
        index=idx,
    )
    return frame


BOOK = ["AAA", "BBB", "CCC"]


# --------------------------------------------------------------------------- #
# Panel coercion
# --------------------------------------------------------------------------- #


def test_price_panel_is_refused_not_correlated():
    """A price panel must raise, not produce r=0.95 between unrelated names."""
    prices = pd.DataFrame(
        {"AAA": np.linspace(100.0, 200.0, 300), "BBB": np.linspace(50.0, 90.0, 300)},
        index=_dates(300),
    )
    with pytest.raises(ValueError, match="does not look like daily simple returns"):
        regime_split(prices)


def test_duplicated_dates_are_refused():
    """A duplicated day is double-counted in every statistic below it."""
    frame = _panel(200)
    frame = pd.concat([frame, frame.iloc[[10]]])
    with pytest.raises(ValueError, match="duplicated dates"):
        regime_split(frame)


def test_empty_panel_is_refused():
    with pytest.raises(ValueError, match="empty"):
        regime_split(pd.DataFrame())


def test_panel_accepts_object_with_returns_attribute():
    """The sibling panel module is expected to expose `.returns`."""

    class _Panel:
        def __init__(self, frame):
            self.returns = frame

    frame = _panel(300)
    assert regime_split(_Panel(frame)) == regime_split(frame)


# --------------------------------------------------------------------------- #
# scrub_suspect_returns — the GOLDBEES 1:100 split guard
# --------------------------------------------------------------------------- #


def test_unadjusted_split_is_blanked_and_reported_with_its_date():
    """A -99% / +9900% pair must be removed AND named, never silently kept."""
    frame = _panel(300)
    frame.iloc[100, frame.columns.get_loc("HEDGE")] = -0.99
    frame.iloc[101, frame.columns.get_loc("HEDGE")] = 99.0

    cleaned, notes = scrub_suspect_returns(frame)
    assert np.isnan(cleaned.iloc[100]["HEDGE"])
    assert np.isnan(cleaned.iloc[101]["HEDGE"])
    # Nothing else touched.
    assert cleaned.drop(index=cleaned.index[[100, 101]]).equals(frame.drop(index=frame.index[[100, 101]]))

    assert len(notes) == 1
    ticker, reason = notes[0]
    assert ticker == "HEDGE"
    assert frame.index[100].date().isoformat() in reason
    assert frame.index[101].date().isoformat() in reason
    assert "-99.0%" in reason


def test_scrub_is_a_no_op_on_a_clean_panel():
    frame = _panel(300)
    cleaned, notes = scrub_suspect_returns(frame)
    assert notes == []
    assert cleaned is frame


def test_scrub_threshold_boundary_is_inclusive():
    frame = _panel(120)
    frame.iloc[5, 0] = SUSPECT_RETURN_THRESHOLD
    frame.iloc[6, 0] = SUSPECT_RETURN_THRESHOLD - 1e-9
    cleaned, notes = scrub_suspect_returns(frame)
    assert np.isnan(cleaned.iloc[5, 0])
    assert not np.isnan(cleaned.iloc[6, 0])
    assert len(notes) == 1


def test_split_contamination_reaches_the_hedge_result_excluded_list():
    """The whole point: the caller sees the corporate action, not just a number."""
    frame = _panel(600)
    frame.iloc[300, frame.columns.get_loc("HEDGE")] = -0.99
    frame.iloc[301, frame.columns.get_loc("HEDGE")] = 99.0
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    reasons = {t: why for t, why in result.excluded}
    assert "HEDGE" in reasons
    assert "corporate action" in reasons["HEDGE"]


# --------------------------------------------------------------------------- #
# regime_split
# --------------------------------------------------------------------------- #


def test_calendar_year_split_boundaries_are_real_trading_days():
    frame = _panel(800, start="2020-01-01")
    regimes = regime_split(frame)
    assert [r.label for r in regimes] == ["2020", "2021", "2022", "2023"]
    for r in regimes:
        assert r.start.year == int(r.label)
        assert r.end.year == int(r.label)
        assert r.start in {d.date() for d in frame.index}
        assert r.end in {d.date() for d in frame.index}
    assert sum(r.n_days for r in regimes) == len(frame)


def test_short_trailing_period_is_returned_but_marked_not_testable():
    """A 20-day stub must be visible AND refused, not silently dropped."""
    frame = _panel(280, start="2020-01-01")  # spills ~20 days into 2021
    regimes = regime_split(frame)
    last = regimes[-1]
    assert last.label == "2021"
    assert last.n_days < beliefs.MIN_REGIME_DAYS
    assert last.testable is False
    assert last.not_testable_reason is not None
    assert str(last.n_days) in last.not_testable_reason


def test_method_travels_with_every_regime():
    """A rendered period can never appear without how the period was chosen."""
    for r in regime_split(_panel(500)):
        assert r.method == "calendar_year"
        assert "31 December" in r.method_note
    for r in regime_split(_panel(500), method="whole_window"):
        assert r.method == "whole_window"
        assert len(r.method_note) > 0


def test_window_string_always_carries_a_date_range_and_an_n():
    for r in regime_split(_panel(500)):
        assert re.match(r"^\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2} \(n=\d+ trading days\)$", r.window)


def test_data_driven_split_is_refused_by_name():
    with pytest.raises(ValueError, match="No data-driven split is offered"):
        regime_split(_panel(300), method="changepoint")


def test_calendar_half_split():
    regimes = regime_split(_panel(520, start="2020-01-01"), method="calendar_half")
    assert [r.label for r in regimes][:3] == ["2020 H1", "2020 H2", "2021 H1"]


# --------------------------------------------------------------------------- #
# test_hedge — direction
# --------------------------------------------------------------------------- #


def test_true_hedge_is_detected_in_every_year():
    frame = _panel(750)
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    testable = [r for r in result.regimes if r.regime.testable]
    assert len(testable) >= 2
    for r in testable:
        assert r.verdict == "moved AGAINST your book", (r.regime.label, r.detail)
        assert r.correlation is not None and r.correlation < -0.5
        assert r.ci_high is not None and r.ci_high < 0.0


def test_a_book_member_moves_with_the_book():
    frame = _panel(750)
    result = test_hedge(frame, "AAA", ["BBB", "CCC"], regimes=regime_split(frame))
    for r in result.regimes:
        if r.regime.testable:
            assert r.verdict == "moved WITH your book"
            assert r.ci_low is not None and r.ci_low > 0.0


def test_independent_asset_is_reported_as_unresolvable_not_as_zero():
    """The differentiating output: 'we could not tell', with its power."""
    frame = _panel(750)
    result = test_hedge(frame, "NOISE", BOOK, regimes=regime_split(frame))
    testable = [r for r in result.regimes if r.regime.testable]
    assert testable
    for r in testable:
        assert r.verdict == "no relationship this window can resolve"
        assert "could not tell" in r.detail
        assert r.power_if_only_weeks_independent is not None


def test_pairing_survives_the_bootstrap():
    """If the bootstrap broke day-alignment, a real hedge would look like noise.

    Shuffling the hedge column destroys the pairing; the verdict must flip from
    'moved AGAINST' to unresolvable. A bootstrap that resampled the two series
    independently would give the same (null) answer for both panels.
    """
    frame = _panel(750)
    regimes = regime_split(frame)
    aligned = test_hedge(frame, "HEDGE", BOOK, regimes=regimes)

    shuffled = frame.copy()
    rng = np.random.default_rng(3)
    shuffled["HEDGE"] = rng.permutation(shuffled["HEDGE"].to_numpy())
    broken = test_hedge(shuffled, "HEDGE", BOOK, regimes=regime_split(shuffled))

    a = [r for r in aligned.regimes if r.regime.testable]
    b = [r for r in broken.regimes if r.regime.testable]
    assert all(r.verdict == "moved AGAINST your book" for r in a)
    assert all(r.verdict != "moved AGAINST your book" for r in b)


# --------------------------------------------------------------------------- #
# test_hedge — refusals, exclusions and zero discipline
# --------------------------------------------------------------------------- #


def test_short_regime_yields_none_not_zero():
    """A regime too short for a correlation must not report 0.0."""
    frame = _panel(280, start="2020-01-01")
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    stub = result.regimes[-1]
    assert stub.regime.testable is False
    assert stub.correlation is None
    assert stub.ci_low is None and stub.ci_high is None and stub.p_value is None
    assert stub.verdict == "not enough data to say"
    assert stub.stat_reason is not None


def test_missing_book_ticker_is_reported_not_dropped():
    frame = _panel(500)
    result = test_hedge(frame, "HEDGE", BOOK + ["GHOST"], regimes=regime_split(frame))
    reasons = dict(result.excluded)
    assert reasons["GHOST"] == "not present in the returns panel"
    assert "GHOST" not in result.book_tickers


def test_book_ticker_with_too_little_history_is_reported():
    frame = _panel(500)
    frame["THIN"] = np.nan
    frame.iloc[:30, frame.columns.get_loc("THIN")] = 0.001
    result = test_hedge(frame, "HEDGE", BOOK + ["THIN"], regimes=regime_split(frame))
    reasons = dict(result.excluded)
    assert "THIN" in reasons
    assert str(MIN_OVERLAP_DAYS) in reasons["THIN"]
    assert "THIN" not in result.book_tickers


def test_hedge_ticker_inside_the_book_is_removed_and_said_so():
    """Otherwise the hedge is correlated partly with itself."""
    frame = _panel(500)
    result = test_hedge(frame, "HEDGE", BOOK + ["HEDGE"], regimes=regime_split(frame))
    assert "HEDGE" not in result.book_tickers
    assert any(t == "HEDGE" and "being tested" in why for t, why in result.excluded)


def test_absent_hedge_ticker_raises():
    frame = _panel(500)
    with pytest.raises(ValueError, match="not in the returns panel"):
        test_hedge(frame, "GHOST", BOOK, regimes=regime_split(frame))


def test_hedge_with_too_little_history_raises():
    frame = _panel(500)
    frame["THIN"] = np.nan
    frame.iloc[:30, frame.columns.get_loc("THIN")] = 0.001
    with pytest.raises(ValueError, match="below the .* floor"):
        test_hedge(frame, "THIN", BOOK, regimes=regime_split(frame))


def test_no_usable_book_ticker_raises():
    frame = _panel(500)
    with pytest.raises(ValueError, match="no book ticker survived"):
        test_hedge(frame, "HEDGE", ["GHOST1", "GHOST2"], regimes=regime_split(frame))


def test_empty_regimes_raises():
    frame = _panel(500)
    with pytest.raises(ValueError, match="regimes is empty"):
        test_hedge(frame, "HEDGE", BOOK, regimes=[])


def test_missing_days_are_counted_not_zero_filled():
    frame = _panel(500)
    frame.iloc[10:20, frame.columns.get_loc("AAA")] = np.nan
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    assert sum(r.n_days_dropped for r in result.regimes) == 10
    first = result.regimes[0]
    assert first.n_days_used + first.n_days_dropped == first.regime.n_days


def test_zero_variance_series_gives_none_with_a_reason():
    frame = _panel(500)
    frame["FLAT"] = 0.0
    result = test_hedge(frame, "FLAT", BOOK, regimes=regime_split(frame))
    for r in result.regimes:
        assert r.correlation is None or not np.isfinite(r.correlation)
        assert r.stat_reason is not None
        assert "zero variance" in r.stat_reason or "floor" in r.stat_reason


# --------------------------------------------------------------------------- #
# test_hedge — sample size and power, attached to every claim
# --------------------------------------------------------------------------- #


def test_every_testable_claim_carries_n_and_a_week_count():
    frame = _panel(750)
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    for r in result.regimes:
        if not r.regime.testable:
            continue
        assert r.n_days_used >= MIN_OVERLAP_DAYS
        assert r.n_independent_weeks is not None
        # Daily data: fewer weeks than days, and at least one week per 7 days.
        assert r.n_independent_weeks < r.n_days_used
        assert r.n_independent_weeks >= r.n_days_used // 7
        assert f"on {r.n_days_used} days" in r.detail


def test_weekly_power_is_lower_than_daily_power():
    """Both are printed because the truth is between them."""
    frame = _panel(750)
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    for r in result.regimes:
        if r.power_if_days_independent is None:
            continue
        assert r.power_if_only_weeks_independent is not None
        assert r.power_if_only_weeks_independent < r.power_if_days_independent


def test_reference_correlation_is_echoed_and_is_not_measured_from_the_data():
    frame = _panel(750)
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame), reference_correlation=-0.5)
    for r in result.regimes:
        assert r.reference_correlation == -0.5
        if r.correlation is not None and np.isfinite(r.correlation):
            assert r.reference_correlation != pytest.approx(r.correlation)
    assert "-0.50" in " ".join(result.caveats)


def test_headline_counts_the_unresolved_periods_too():
    frame = _panel(750)
    result = test_hedge(frame, "NOISE", BOOK, regimes=regime_split(frame))
    assert "could not tell" in result.headline or "resolve" in result.headline


def test_weights_change_the_book_and_are_reported():
    frame = _panel(500)
    equal = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    tilted = test_hedge(
        frame,
        "HEDGE",
        BOOK,
        regimes=regime_split(frame),
        weights={"AAA": 8.0, "BBB": 1.0, "CCC": 1.0},
    )
    assert "equal-weighted" in equal.book_weighting
    assert "renormalised" in tilted.book_weighting
    assert equal.regimes[0].correlation != tilted.regimes[0].correlation


# --------------------------------------------------------------------------- #
# drawdown_coincidence
# --------------------------------------------------------------------------- #


def _weights(**kw: float) -> dict[str, float]:
    return kw


def test_worst_days_are_the_worst_five_percent():
    frame = _panel(600)
    w = _weights(AAA=0.3, BBB=0.3, CCC=0.3, HEDGE=0.1)
    out = drawdown_coincidence(frame, w)
    assert out["n_worst_days"] == round(0.05 * out["n_days"])
    rets = [r for _, r in out["worst_days"]]
    assert max(rets) == pytest.approx(out["worst_day_threshold"])
    port = frame[list(w)].mul(pd.Series(w) / sum(w.values()), axis=1).sum(axis=1)
    assert min(sorted(port)[out["n_worst_days"] :]) > max(rets)


def test_a_true_hedge_falls_on_far_fewer_worst_days():
    frame = _panel(600)
    out = drawdown_coincidence(frame, _weights(AAA=0.3, BBB=0.3, CCC=0.3, HEDGE=0.1))
    by_ticker = {h["ticker"]: h for h in out["holdings"]}
    n_worst = out["n_worst_days"]
    assert by_ticker["AAA"]["n_fell"] > 0.85 * n_worst
    assert by_ticker["HEDGE"]["n_fell"] < 0.30 * n_worst
    assert by_ticker["HEDGE"]["mean_return_on_worst_days"] > 0.0
    assert by_ticker["AAA"]["mean_return_on_worst_days"] < 0.0


def test_holdings_sorted_by_how_often_they_fell():
    frame = _panel(600)
    out = drawdown_coincidence(frame, _weights(AAA=0.3, BBB=0.3, CCC=0.3, HEDGE=0.1))
    counts = [h["n_fell"] for h in out["holdings"]]
    assert counts == sorted(counts, reverse=True)


def test_headline_names_the_exception_and_the_week_count():
    frame = _panel(600)
    out = drawdown_coincidence(frame, _weights(AAA=0.3, BBB=0.3, CCC=0.3, HEDGE=0.1))
    assert "HEDGE" in out["headline"]
    assert "distinct calendar weeks" in out["headline"]
    assert out["worst_day_clustering"]["n_distinct_weeks"] is not None
    assert out["worst_day_clustering"]["n_distinct_weeks"] <= out["n_worst_days"]


def test_fall_frequency_baseline_is_reported_beside_every_count():
    """7 of 10 means nothing without knowing how often it falls in general."""
    frame = _panel(600)
    out = drawdown_coincidence(frame, _weights(AAA=0.3, BBB=0.3, CCC=0.3, HEDGE=0.1))
    for h in out["holdings"]:
        assert 0.0 < h["fell_frequency_all_days"] < 1.0
        assert h["binomial_p_floor"] is not None
        assert 0.0 <= h["binomial_p_floor"] <= 1.0


def test_too_few_days_refuses_rather_than_reporting_two_of_two():
    frame = _panel(80)
    with pytest.raises(ValueError, match="below the .* floor"):
        drawdown_coincidence(frame, _weights(AAA=0.5, HEDGE=0.5))


def test_missing_holding_is_reported_not_dropped():
    frame = _panel(600)
    out = drawdown_coincidence(frame, _weights(AAA=0.5, HEDGE=0.4, GHOST=0.1))
    assert any(t == "GHOST" for t, _ in out["excluded"])
    assert "GHOST" not in {h["ticker"] for h in out["holdings"]}


def test_dropped_calendar_days_are_reported():
    frame = _panel(600)
    frame.iloc[5:12, frame.columns.get_loc("AAA")] = np.nan
    out = drawdown_coincidence(frame, _weights(AAA=0.5, BBB=0.3, HEDGE=0.2))
    assert out["n_days"] == len(frame) - 7
    assert any("dropped from the portfolio series" in why for _, why in out["excluded"])


def test_weights_are_renormalised_and_the_input_sum_is_shown():
    frame = _panel(600)
    out = drawdown_coincidence(frame, _weights(AAA=30.0, BBB=30.0, HEDGE=40.0))
    assert sum(h["weight"] for h in out["holdings"]) == pytest.approx(1.0)
    assert "100" in out["portfolio_weighting"]


def test_empty_weights_and_bad_fraction_are_refused():
    frame = _panel(600)
    with pytest.raises(ValueError, match="weights is empty"):
        drawdown_coincidence(frame, {})
    with pytest.raises(ValueError, match="worst_fraction"):
        drawdown_coincidence(frame, _weights(AAA=1.0), worst_fraction=0.9)


def test_every_count_shares_one_stated_denominator():
    """The portfolio series only holds days when everything traded, so the
    denominator behind every "fell on X of Y" is the same Y. A table whose
    rows quietly have different denominators is unreadable and misleading."""
    frame = _panel(600)
    frame.iloc[5:12, frame.columns.get_loc("AAA")] = np.nan
    out = drawdown_coincidence(frame, _weights(AAA=0.4, BBB=0.3, CCC=0.2, HEDGE=0.1))
    denominators = {h["n_worst_days_with_data"] for h in out["holdings"]}
    assert denominators == {out["n_worst_days"]}
    for h in out["holdings"]:
        assert 0 <= h["n_fell"] <= out["n_worst_days"]


# --------------------------------------------------------------------------- #
# The regulatory line — facts only, never advice
# --------------------------------------------------------------------------- #

_ADVICE = re.compile(
    r"\b(you should|we recommend|recommend|advise|advice|consider (?:trimming|selling|buying)"
    r"|rebalance|trim|diversify|hedge more|reduce your|increase your|we suggest"
    r"|is too (?:big|concentrated)|overweight|underweight)\b",
    re.IGNORECASE,
)

_FORECAST = re.compile(
    r"\b(will (?:fall|rise|protect|continue|outperform|underperform)"
    r"|expected to|going to|is likely to|predicts?)\b",
    re.IGNORECASE,
)


def _emitted_text() -> str:
    frame = _panel(600)
    regimes = regime_split(frame)
    hedge = test_hedge(frame, "HEDGE", BOOK, regimes=regimes)
    noise = test_hedge(frame, "NOISE", BOOK, regimes=regimes)
    dd = drawdown_coincidence(frame, _weights(AAA=0.3, BBB=0.3, CCC=0.3, HEDGE=0.1))
    return "\n".join(
        [
            render_hedge_result(hedge),
            render_hedge_result(noise),
            render_drawdown_coincidence(dd),
        ]
    )


def test_no_emitted_string_gives_advice():
    text = _emitted_text()
    found = _ADVICE.findall(text)
    assert not found, f"advice-shaped language in output: {found}"


def test_no_emitted_string_makes_a_forecast():
    text = _emitted_text()
    found = _FORECAST.findall(text)
    assert not found, f"forecast-shaped language in output: {found}"


def test_rendered_output_always_carries_a_window_and_an_n():
    text = _emitted_text()
    assert re.search(r"\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}", text)
    assert "trading days" in text
    assert "Read this with:" in text


def test_rendered_drawdown_says_when_it_truncates_the_day_list():
    frame = _panel(600)
    out = drawdown_coincidence(frame, _weights(AAA=0.3, BBB=0.3, CCC=0.3, HEDGE=0.1))
    text = render_drawdown_coincidence(out, max_days_shown=5)
    assert "deepest of those" in text
    assert str(out["n_worst_days"]) in text
    # The per-holding counts are over ALL worst days, not the shown five.
    assert f"/{out['n_worst_days']} worst days" in text


# --------------------------------------------------------------------------- #
# Multiplicity — eight calendar years is eight tests
# --------------------------------------------------------------------------- #


def test_a_lone_lucky_period_does_not_survive_the_correction():
    """NOISE is independent of the book, yet 2020 crosses p<0.05 on raw p.

    That single year is exactly what a reader would screenshot as "the year it
    broke". The Benjamini-Hochberg correction over the periods actually tested
    must demote it, and the raw p must still be visible beside the adjusted one.
    """
    frame = _panel(750)
    result = test_hedge(frame, "NOISE", BOOK, regimes=regime_split(frame))
    lucky = next(r for r in result.regimes if r.regime.label == "2020")
    assert lucky.p_value is not None and lucky.p_value < 0.05  # raw: "significant"
    assert lucky.survives_multiplicity is False
    assert lucky.verdict == "no relationship this window can resolve"
    assert f"{lucky.p_value:.3f}" in lucky.detail
    assert "correcting for the 3 periods tested" in lucky.detail


def test_adjusted_p_is_never_below_the_raw_p():
    frame = _panel(750)
    for ticker in ("HEDGE", "NOISE", "AAA"):
        book = [t for t in BOOK if t != ticker]
        result = test_hedge(frame, ticker, book, regimes=regime_split(frame))
        for r in result.regimes:
            if r.p_value is None:
                assert r.p_adjusted is None and r.survives_multiplicity is None
                continue
            assert r.p_adjusted >= r.p_value - 1e-12


def test_family_size_counts_only_the_periods_actually_tested():
    """Dropping untestable periods from the family would inflate significance."""
    frame = _panel(800, start="2020-01-01")  # 2023 is a short stub
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    n_testable = sum(1 for r in result.regimes if r.p_value is not None)
    assert n_testable < len(result.regimes)
    for r in result.regimes:
        assert r.n_periods_tested == n_testable
    assert f"{n_testable} period(s) were tested" in " ".join(result.caveats)


def test_a_real_hedge_still_survives_the_correction():
    """The correction must not be so blunt that a true effect cannot show."""
    frame = _panel(750)
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    testable = [r for r in result.regimes if r.p_value is not None]
    assert testable
    assert all(r.survives_multiplicity is True for r in testable)


# --------------------------------------------------------------------------- #
# Gaps found by mutation testing
# --------------------------------------------------------------------------- #


def test_a_pandas_series_of_weights_is_accepted():
    """``PricePanel.weights`` is a Series, and a Series is not a Mapping.

    Iterating one yields its VALUES, so ``for t in weights`` would walk a list
    of floats and look for a ticker called "0.4"; ``if not weights`` raises
    "truth value is ambiguous" outright. Both reach a reader as a wrong table.
    """
    frame = _panel(600)
    as_dict = _weights(AAA=0.4, BBB=0.3, CCC=0.2, HEDGE=0.1)
    as_series = pd.Series(as_dict)
    assert drawdown_coincidence(frame, as_series)["holdings"] == (drawdown_coincidence(frame, as_dict)["holdings"])
    a = test_hedge(
        frame, "HEDGE", BOOK, regimes=regime_split(frame), weights=pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    )
    b = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame), weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    assert [r.correlation for r in a.regimes] == [r.correlation for r in b.regimes]


def test_duplicate_and_non_finite_weights_are_refused():
    frame = _panel(600)
    with pytest.raises(ValueError, match="not finite"):
        drawdown_coincidence(frame, {"AAA": float("nan"), "BBB": 0.5})
    with pytest.raises(TypeError, match="mapping of ticker"):
        drawdown_coincidence(frame, ["AAA", "BBB"])


def test_a_verdict_never_contradicts_its_interval():
    """M6: deciding direction from the point estimate instead of the interval.

    The two agree only because the multiplicity gate implies the interval
    excludes zero. Locking the implication makes that dependency explicit: if
    the gate is ever loosened, a verdict could outrun its interval, and this
    goes red.
    """
    frame = _panel(750)
    for seed in (1, 5, 11, 23, 41):
        panel = _panel(750, seed=seed)
        for ticker, book in (("HEDGE", BOOK), ("NOISE", BOOK), ("AAA", ["BBB", "CCC"])):
            result = test_hedge(panel, ticker, book, regimes=regime_split(panel))
            for r in result.regimes:
                if r.verdict == "moved AGAINST your book":
                    assert r.ci_high is not None and r.ci_high < 0.0
                    assert r.survives_multiplicity is True
                elif r.verdict == "moved WITH your book":
                    assert r.ci_low is not None and r.ci_low > 0.0
                    assert r.survives_multiplicity is True
    assert frame is not None


def test_power_against_the_reference_effect_is_actually_high_on_a_full_year():
    """M14: a reference effect silently collapsed to ~0 makes power ~= alpha.

    A full calendar year at |r| = 0.30 is ~100% powered on daily observations
    and ~55-65% on weekly ones. If either figure comes back near 5%, the
    stipulated effect has been lost and every "we could not tell" beside it is
    meaningless.
    """
    frame = _panel(750)
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    full_years = [r for r in result.regimes if r.n_days_used > 200]
    assert full_years
    for r in full_years:
        assert r.power_if_days_independent > 0.95
        assert 0.45 < r.power_if_only_weeks_independent < 0.85


def test_weekly_power_uses_weeks_not_days():
    """M13: reporting the daily power figure in the weekly slot."""
    frame = _panel(750)
    result = test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame))
    for r in result.regimes:
        if r.power_if_days_independent is None:
            continue
        assert r.power_if_only_weeks_independent != pytest.approx(r.power_if_days_independent, abs=1e-6)


def test_a_flat_day_does_not_count_as_a_fall():
    """M18: 'fell' must be a strictly negative return.

    A bond fund that simply did not trade on a crash day has not fallen with
    the book, and counting it as though it had would overstate exactly the
    number this product puts in its headline.
    """
    frame = _panel(600)
    w = _weights(AAA=0.4, BBB=0.3, CCC=0.2, HEDGE=0.1)
    base = drawdown_coincidence(frame, w)
    worst = [d for d, _ in sorted(base["worst_days"], key=lambda dr: dr[1])]

    flat = frame.copy()
    targets = [ts for ts in flat.index if ts.date() in set(worst[:4])]
    assert len(targets) == 4
    flat.loc[targets, "HEDGE"] = 0.0
    out = drawdown_coincidence(flat, w)
    hedge = next(h for h in out["holdings"] if h["ticker"] == "HEDGE")
    on_those = [float(flat.loc[ts, "HEDGE"]) for ts in flat.index if ts.date() in set(worst[:4])]
    assert on_those == [0.0, 0.0, 0.0, 0.0]
    assert hedge["n_fell"] == sum(1 for d, _ in out["worst_days"] if float(flat.loc[pd.Timestamp(d), "HEDGE"]) < 0.0)


def test_fall_frequency_is_the_real_frequency_not_a_constant():
    """M19: a hardcoded baseline makes every binomial p a fiction."""
    frame = _panel(600)
    rng = np.random.default_rng(99)
    # An asset that falls on ~20% of days: large positive drift, small noise.
    frame["RARELY_FALLS"] = 0.02 + rng.normal(0.0, 0.02, len(frame))
    w = _weights(AAA=0.4, BBB=0.3, CCC=0.2, RARELY_FALLS=0.1)
    out = drawdown_coincidence(frame, w)
    rare = next(h for h in out["holdings"] if h["ticker"] == "RARELY_FALLS")
    expected = float((frame["RARELY_FALLS"].to_numpy() < 0.0).mean())
    assert rare["fell_frequency_all_days"] == pytest.approx(expected)
    assert 0.10 < expected < 0.30  # genuinely different from a 0.5 constant


def test_drawdown_weights_are_renormalised_in_the_drawdown_path():
    """M21: the portfolio series must not be scaled by un-normalised weights."""
    frame = _panel(600)
    small = drawdown_coincidence(frame, _weights(AAA=0.5, BBB=0.3, HEDGE=0.2))
    big = drawdown_coincidence(frame, _weights(AAA=50.0, BBB=30.0, HEDGE=20.0))
    assert small["worst_day_threshold"] == pytest.approx(big["worst_day_threshold"])
    assert [r for _, r in small["worst_days"]] == pytest.approx([r for _, r in big["worst_days"]])
    assert sum(h["weight"] for h in big["holdings"]) == pytest.approx(1.0)


def test_a_duplicated_ticker_in_the_weights_is_refused():
    """M28: a Series index can repeat a ticker; a dict literal cannot.

    Two lots of the same holding pasted as two lines must be merged upstream
    with a stated rule, not silently collapsed to whichever one happened to be
    last — that is a weight the reader never chose.
    """
    frame = _panel(600)
    dup = pd.Series([0.5, 0.3, 0.2], index=["AAA", "AAA", "BBB"])
    with pytest.raises(ValueError, match="names 'AAA' twice"):
        drawdown_coincidence(frame, dup)
    with pytest.raises(ValueError, match="names 'AAA' twice"):
        test_hedge(frame, "HEDGE", BOOK, regimes=regime_split(frame), weights=dup)
