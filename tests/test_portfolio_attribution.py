"""Tests for agent.portfolio.attribution.

These are written against the specific ways this module can hurt a stranger who
pastes a portfolio they know by heart:

  1. It prints a number that does not reconcile — the contributions do not add
     up to the gap, so one of the two is wrong and neither is labelled.
  2. It ranks the story backwards, putting a 1%-weight disaster above a
     30%-weight drag, so the reader fixes on the wrong line.
  3. It reads well only when the reader lost. A module whose sentences are
     grammatical only for a losing book is a rhetoric engine.
  4. It quietly drops a holding, a day or a window and reports confidently
     about what is left.
  5. It publishes one window. A single window is a cherry-pick whether or not
     anybody cherry-picked it, and this product's whole claim is that it does
     not let you fool yourself.
  6. It gives advice, which we are not licensed to give.

So most assertions here are about IDENTITIES, REFUSALS, the contents of
``excluded``, and the wording that survives a winning book.

Everything is synthetic and deterministic. No network, no clock: the one test
that exercises ``fetch_benchmark`` injects a fetcher.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agent.portfolio import attribution as A
from agent.portfolio.attribution import (
    MAX_BENCHMARK_DAY_LOSS,
    MIN_WINDOW_DAYS,
    RECONCILIATION_TOLERANCE,
    TRADING_DAYS_PER_YEAR,
    AttributionResult,
    decision_summary,
    fetch_benchmark,
    format_attribution_report,
    format_multi_window_report,
    holding_attribution,
    multi_window_attribution,
    survivorship_note,
)

# The forbidden-word list is owned by concentration.py and is deliberately not
# duplicated here: two copies would drift and one of them would be the lenient
# one. It is private because it is a test fixture, not an API.
from agent.portfolio.concentration import _advice_words_in

# --------------------------------------------------------------------------- #
# Fixtures — deterministic synthetic panels. No network, no clock.
# --------------------------------------------------------------------------- #


def _dates(n: int, start: str = "2021-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _panel(n: int = 760, seed: int = 11, start: str = "2021-01-01") -> pd.DataFrame:
    """Five names with genuinely different drifts, plus a benchmark column.

    Drifts are set far apart on purpose so that the ORDER of contributions is
    not a coin flip that a different seed would reverse; the tests below assert
    on ordering and would be flaky otherwise.
    """
    rng = np.random.default_rng(seed)
    idx = _dates(n, start)
    mkt = rng.normal(0.0004, 0.010, n)
    cols = {
        "BIGLOSER": mkt - 0.0009 + rng.normal(0.0, 0.007, n),
        "MIDLOSER": mkt - 0.0004 + rng.normal(0.0, 0.006, n),
        "FLAT": mkt + rng.normal(0.0, 0.005, n),
        "WINNER": mkt + 0.0011 + rng.normal(0.0, 0.009, n),
        "TINYDISASTER": mkt - 0.0030 + rng.normal(0.0, 0.012, n),
        "BENCH": mkt,
    }
    return pd.DataFrame(cols, index=idx)


def _book(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.drop(columns=["BENCH"])


def _bench(panel: pd.DataFrame) -> pd.Series:
    return panel["BENCH"]


def _equal_end_weights(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / frame.shape[1], index=frame.columns)


def _growth(frame: pd.DataFrame) -> dict[str, float]:
    return {c: float(np.prod(1.0 + frame[c].to_numpy(dtype=float))) for c in frame.columns}


@pytest.fixture(autouse=True)
def _isolated_price_cache(tmp_path, monkeypatch) -> None:
    """Point the holdings disk cache at a fresh directory for every test.

    Without this, ``test_fetch_benchmark_...`` passed while its injected
    fetcher was never called: a previous real run had left NIFTYBEES.NS in
    ~/.cache/tirramind/yfinance, the cache served it, and the test asserted
    nothing. A test that passes because of a file on the developer's laptop is
    a test that will pass in CI for a different reason, or fail there for no
    reason at all.
    """
    monkeypatch.setenv("TIRRA_PORTFOLIO_CACHE", str(tmp_path / "yf"))


@pytest.fixture
def panel() -> pd.DataFrame:
    return _panel()


# --------------------------------------------------------------------------- #
# 1. Reconciliation — the identity the whole module rests on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13, 21])
def test_contributions_sum_to_the_gap_exactly_under_buy_and_hold(seed: int) -> None:
    """sum_i w_start_i * (R_i - R_b) == book_total - bench_total, to float64.

    This is the assertion the brief asks for. It is an identity, not an
    approximation, so it is tested at ``RECONCILIATION_TOLERANCE`` and not at
    "close enough for a percentage".
    """
    p = _panel(seed=seed)
    attr = holding_attribution(_book(p), _equal_end_weights(_book(p)), _bench(p))
    assert attr.usable
    assert abs(attr.residual) <= RECONCILIATION_TOLERANCE, (
        f"residual {attr.residual:.3e} exceeds {RECONCILIATION_TOLERANCE:.0e}"
    )
    assert attr.contribution_sum == pytest.approx(attr.gap_total, abs=RECONCILIATION_TOLERANCE)


def test_book_total_equals_weighted_sum_of_holding_returns(panel: pd.DataFrame) -> None:
    """The book's compounded DAILY series must reproduce the weighted sum of
    the holdings' compounded returns. If it does not, the headline return and
    the attribution table are describing two different portfolios."""
    book = _book(panel)
    attr = holding_attribution(book, _equal_end_weights(book), _bench(panel))
    direct = sum(h.weight_start * h.total_return for h in attr)
    assert attr.book.total_return == pytest.approx(direct, abs=1e-12)


def test_start_weights_reproduce_the_end_weights_that_were_handed_in(panel: pd.DataFrame) -> None:
    """w_start * (1+R), renormalised, must return the pasted weights. This is
    the round trip that makes the derivation checkable rather than asserted."""
    book = _book(panel)
    w_end = pd.Series([0.35, 0.25, 0.20, 0.15, 0.05], index=book.columns)
    attr = holding_attribution(book, w_end, _bench(panel))
    growth = _growth(book)
    rebuilt = {t: attr.weights_start[t] * growth[t] for t in attr.weights_start}
    total = sum(rebuilt.values())
    for t, v in rebuilt.items():
        assert v / total == pytest.approx(float(w_end[t]), abs=1e-12)


def test_weights_as_of_start_is_not_silently_the_same_as_end(panel: pd.DataFrame) -> None:
    """Passing the same numbers as START weights must give a different answer
    from passing them as END weights, or the parameter is decorative."""
    book = _book(panel)
    w = pd.Series([0.35, 0.25, 0.20, 0.15, 0.05], index=book.columns)
    as_end = holding_attribution(book, w, _bench(panel), weights_as_of="end")
    as_start = holding_attribution(book, w, _bench(panel), weights_as_of="start")
    assert as_end.book.total_return != pytest.approx(as_start.book.total_return, abs=1e-6)
    for t in book.columns:
        assert as_start.weights_start[t] == pytest.approx(float(w[t]), abs=1e-12)
    assert abs(as_start.residual) <= RECONCILIATION_TOLERANCE


def test_fixed_weight_basis_leaves_a_residual_and_names_it(panel: pd.DataFrame) -> None:
    """Under daily rebalancing the contributions are NOT the gap. The module
    must report that difference as the rebalancing effect rather than hiding it
    or, worse, forcing it to zero."""
    book = _book(panel)
    attr = holding_attribution(book, _equal_end_weights(book), _bench(panel), basis="fixed_weight")
    assert abs(attr.residual) > 1e-4, "a 3-year daily rebalance cannot have a zero rebalancing effect"
    assert "rebalancing" in attr.residual_explanation.lower()
    assert attr.contribution_sum != pytest.approx(attr.gap_total, abs=1e-6)


def test_a_broken_reconciliation_raises_rather_than_renders(panel: pd.DataFrame, monkeypatch) -> None:
    """The reconciliation check must be load-bearing. If the book series is
    corrupted, the module must refuse — not print a residual in the footer that
    nobody reads. This is the mutation guard for the assertion itself."""
    book = _book(panel)
    real = A._book_daily

    def sabotaged(returns, weights_start, basis):
        return real(returns, weights_start, basis) + 0.0001

    monkeypatch.setattr(A, "_book_daily", sabotaged)
    with pytest.raises(ValueError, match="failed to reconcile"):
        holding_attribution(book, _equal_end_weights(book), _bench(panel))


# --------------------------------------------------------------------------- #
# 2. Ranking — the story must be told in the right order
# --------------------------------------------------------------------------- #


def test_ranked_by_contribution_not_by_excess_return() -> None:
    """A tiny position that collapsed must NOT outrank a large position that
    merely dragged. Ranking by excess return is the wrong story told
    confidently, which is the failure mode this product exists to avoid."""
    n = 400
    idx = _dates(n)
    frame = pd.DataFrame(
        {
            "BIGDRAG": np.full(n, -0.0008),  # large weight, modest loss
            "TINYBOMB": np.full(n, -0.0060),  # tiny weight, catastrophic loss
            "BENCH": np.zeros(n),
        },
        index=idx,
    )
    book = frame[["BIGDRAG", "TINYBOMB"]]
    attr = holding_attribution(book, pd.Series({"BIGDRAG": 0.97, "TINYBOMB": 0.03}), frame["BENCH"])
    by_excess = sorted(attr, key=lambda h: h.excess_return)
    assert by_excess[0].ticker == "TINYBOMB", "fixture is wrong: TINYBOMB must have the worse return"
    assert attr[0].ticker == "BIGDRAG", "ranked by excess return instead of contribution"
    assert abs(attr[0].contribution) > abs(attr[1].contribution)


def test_result_is_a_real_list_of_holding_results(panel: pd.DataFrame) -> None:
    book = _book(panel)
    attr = holding_attribution(book, _equal_end_weights(book), _bench(panel))
    assert isinstance(attr, list)
    assert isinstance(attr, AttributionResult)
    assert len(attr) == book.shape[1]
    assert {h.ticker for h in attr} == set(book.columns)


def test_weights_start_sums_to_one(panel: pd.DataFrame) -> None:
    book = _book(panel)
    attr = holding_attribution(book, _equal_end_weights(book), _bench(panel))
    assert sum(attr.weights_start.values()) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# 3. It must read correctly for a book that WON
# --------------------------------------------------------------------------- #


def _winning_panel(n: int = 400) -> pd.DataFrame:
    idx = _dates(n)
    return pd.DataFrame(
        {
            "GOOD": np.full(n, 0.0010),
            "BETTER": np.full(n, 0.0014),
            "BEST": np.full(n, 0.0030),
            "BENCH": np.full(n, 0.0004),
        },
        index=idx,
    )


def test_every_holding_beating_the_benchmark_is_described_as_such() -> None:
    p = _winning_panel()
    book = p[["GOOD", "BETTER", "BEST"]]
    attr = holding_attribution(book, _equal_end_weights(book), p["BENCH"])
    summary = decision_summary(attr)
    assert summary["n_trailed"] == 0
    assert summary["n_beat"] == 3
    assert attr.gap_total > 0
    assert all(h.beat_benchmark for h in attr)


def test_the_lowest_contributor_on_a_winning_book_is_not_called_negative() -> None:
    """REGRESSION. The first draft labelled the bottom row of the ranked list
    "the largest single negative contributor" unconditionally. On a book where
    every holding beat the benchmark that sentence is FALSE about a real book,
    printed on the page of a product whose pitch is that its numbers are
    checkable. The label must come from the sign, not from the sort order."""
    p = _winning_panel()
    book = p[["GOOD", "BETTER", "BEST"]]
    summary = decision_summary(holding_attribution(book, _equal_end_weights(book), p["BENCH"]))
    assert summary["worst_contribution"] > 0, "fixture is wrong: every contribution should be positive"
    assert "negative" not in summary["worst_contributor_role"]
    assert "negative" not in summary["statement"].lower()
    assert summary["worst_contributor_role"] == "the smallest positive contributor"


def test_a_losing_book_still_names_its_negative_contributor_negative(panel: pd.DataFrame) -> None:
    """The mirror of the test above: the sign-aware label must not have been
    fixed by deleting the word 'negative' everywhere."""
    book = _book(panel)
    summary = decision_summary(holding_attribution(book, _equal_end_weights(book), _bench(panel)))
    assert summary["worst_contribution"] < 0
    assert summary["worst_contributor_role"] == "the largest single negative contributor"


# --------------------------------------------------------------------------- #
# 4. decision_summary and its counterfactuals
# --------------------------------------------------------------------------- #


def test_counterfactual_arithmetic_is_a_renormalised_weighted_sum(panel: pd.DataFrame) -> None:
    book = _book(panel)
    attr = holding_attribution(book, _equal_end_weights(book), _bench(panel))
    s = decision_summary(attr)
    w = {h.ticker: h.weight_start for h in attr}
    r = {h.ticker: h.total_return for h in attr}
    for key, drop in (
        ("without_worst_book_return", s["worst_contributor"]),
        ("without_best_book_return", s["best_contributor"]),
    ):
        rest = 1.0 - w[drop]
        expected = sum(w[t] * r[t] for t in w if t != drop) / rest
        assert s[key] == pytest.approx(expected, abs=1e-12)


def test_without_best_can_be_worse_and_that_is_reported(panel: pd.DataFrame) -> None:
    """The counterfactual that stops 'you picked badly' being a smear: if one
    name carried the book, removing it must make the book look WORSE, and the
    summary must say so rather than only counting the losers."""
    book = _book(panel)
    attr = holding_attribution(book, _equal_end_weights(book), _bench(panel))
    s = decision_summary(attr)
    assert s["without_best_book_return"] < s["book_total_return"]
    assert s["best_contributor"] in s["statement"]
    assert "Without" in s["statement"]


def test_one_position_carried_is_true_only_when_it_actually_did() -> None:
    n = 300
    idx = _dates(n)
    # One large winner, several small ones: the large one alone exceeds the rest.
    carried = pd.DataFrame(
        {
            "HUGE": np.full(n, 0.0040),
            "SMALL1": np.full(n, 0.00042),
            "SMALL2": np.full(n, 0.00041),
            "BENCH": np.full(n, 0.0004),
        },
        index=idx,
    )
    book = carried[["HUGE", "SMALL1", "SMALL2"]]
    s = decision_summary(holding_attribution(book, _equal_end_weights(book), carried["BENCH"]))
    assert s["one_position_carried"] is True

    # Three equal winners: no single one carried it.
    even = pd.DataFrame(
        {
            "A": np.full(n, 0.0020),
            "B": np.full(n, 0.0020),
            "C": np.full(n, 0.0020),
            "BENCH": np.full(n, 0.0004),
        },
        index=idx,
    )
    ebook = even[["A", "B", "C"]]
    s2 = decision_summary(holding_attribution(ebook, _equal_end_weights(ebook), even["BENCH"]))
    assert s2["one_position_carried"] is False


def test_without_best_flips_sign_is_detected() -> None:
    """A book that beat the index only because of one name must be flagged."""
    n = 300
    idx = _dates(n)
    frame = pd.DataFrame(
        {
            "STAR": np.full(n, 0.0050),
            "DRAG1": np.full(n, 0.0000),
            "DRAG2": np.full(n, 0.0000),
            "BENCH": np.full(n, 0.0008),
        },
        index=idx,
    )
    book = frame[["STAR", "DRAG1", "DRAG2"]]
    attr = holding_attribution(book, _equal_end_weights(book), frame["BENCH"])
    s = decision_summary(attr)
    assert s["gap_total"] > 0
    assert s["without_best_gap"] < 0
    assert s["without_best_flips_sign"] is True


def test_summary_keys_are_always_present_even_when_nothing_was_computed() -> None:
    """Zero and None discipline: a caller must never have to guess whether a
    missing key means zero."""
    n = 30  # below MIN_WINDOW_DAYS
    idx = _dates(n)
    frame = pd.DataFrame({"A": np.full(n, 0.001), "BENCH": np.zeros(n)}, index=idx)
    attr = holding_attribution(frame[["A"]], pd.Series({"A": 1.0}), frame["BENCH"])
    assert not attr.usable
    s = decision_summary(attr)
    assert s["unavailable_reason"]
    for key in ("gap_total", "worst_contributor", "without_best_book_return", "one_position_carried"):
        assert s[key] is None
    assert s["n_holdings"] == 0


# --------------------------------------------------------------------------- #
# 5. Nothing is dropped silently
# --------------------------------------------------------------------------- #


def test_a_weight_with_no_price_column_is_excluded_with_a_reason(panel: pd.DataFrame) -> None:
    book = _book(panel)
    w = _equal_end_weights(book)
    w["GHOST"] = 0.2
    attr = holding_attribution(book, w, _bench(panel))
    names = {e.name for e in attr.excluded}
    assert "GHOST" in names
    assert any("no column" in e.reason for e in attr.excluded if e.name == "GHOST")


def test_a_price_column_with_no_weight_is_excluded_with_a_reason(panel: pd.DataFrame) -> None:
    book = _book(panel)
    w = _equal_end_weights(book).drop("FLAT")
    attr = holding_attribution(book, w, _bench(panel))
    assert "FLAT" in {e.name for e in attr.excluded}
    assert "FLAT" not in {h.ticker for h in attr}


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_a_zero_or_negative_weight_is_excluded_with_a_reason(panel: pd.DataFrame, bad: float) -> None:
    book = _book(panel)
    w = _equal_end_weights(book)
    w["FLAT"] = bad
    attr = holding_attribution(book, w, _bench(panel))
    assert any(e.name == "FLAT" and "weight is" in e.reason for e in attr.excluded)


def test_a_holding_that_went_to_zero_is_excluded_not_divided_by() -> None:
    """w_start ∝ w_end / (1 + R) is undefined at R = -100%. That must be an
    exclusion with a reason, not an inf that poisons every other weight."""
    n = 300
    idx = _dates(n)
    # -10% a day for 300 days compounds to a growth factor of ~1e-14. No single
    # day trips the corporate-action scrub, so this reaches the weight
    # derivation as a genuine wipeout rather than as a suspected split.
    frame = pd.DataFrame(
        {"DEAD": np.full(n, -0.10), "ALIVE": np.full(n, 0.0005), "BENCH": np.full(n, 0.0003)},
        index=idx,
    )
    book = frame[["DEAD", "ALIVE"]]
    attr = holding_attribution(book, pd.Series({"DEAD": 0.5, "ALIVE": 0.5}), frame["BENCH"])
    assert "DEAD" in {e.name for e in attr.excluded}
    assert all(math.isfinite(v) for v in attr.weights_start.values())
    assert sum(attr.weights_start.values()) == pytest.approx(1.0, abs=1e-12)
    assert any("renormalised" in n for n in attr.notes)


def test_days_lost_to_the_benchmark_calendar_are_counted_in_the_notes(panel: pd.DataFrame) -> None:
    book = _book(panel)
    bench = _bench(panel).drop(panel.index[[5, 60, 200]])
    attr = holding_attribution(book, _equal_end_weights(book), bench)
    assert attr.usable
    assert any("not in" in n and "calendar" in n for n in attr.notes)
    assert attr.window.n_days == len(panel.index) - 3


def test_a_benchmark_on_a_wildly_different_calendar_is_refused(panel: pd.DataFrame) -> None:
    """Compounding two sides over calendars that differ by more than
    MAX_BENCHMARK_DAY_LOSS compares two different windows and calls the
    difference a decision."""
    book = _book(panel)
    keep = panel.index[::2]  # drops half the days
    attr = holding_attribution(book, _equal_end_weights(book), _bench(panel).loc[keep])
    assert not attr.usable
    assert f"{MAX_BENCHMARK_DAY_LOSS:.0%}" in attr.unusable_reason


def test_a_short_window_refuses_rather_than_annualising_noise() -> None:
    n = MIN_WINDOW_DAYS - 1
    idx = _dates(n)
    frame = pd.DataFrame({"A": np.full(n, 0.002), "BENCH": np.zeros(n)}, index=idx)
    attr = holding_attribution(frame[["A"]], pd.Series({"A": 1.0}), frame["BENCH"])
    assert not attr.usable
    assert str(MIN_WINDOW_DAYS) in attr.unusable_reason
    assert format_attribution_report(attr).startswith("NOT COMPUTED")


# --------------------------------------------------------------------------- #
# 6. Input guards inherited from beliefs, and benchmark coercion
# --------------------------------------------------------------------------- #


def test_a_price_panel_where_returns_belong_is_refused(panel: pd.DataFrame) -> None:
    """The single most likely integration mistake. The guard lives in
    beliefs._coerce_panel and is imported, not reimplemented."""
    prices = (1.0 + _book(panel)).cumprod() * 100.0
    with pytest.raises(ValueError, match="daily simple returns"):
        holding_attribution(prices, _equal_end_weights(prices), _bench(panel))


def test_a_benchmark_price_series_is_refused(panel: pd.DataFrame) -> None:
    book = _book(panel)
    bench_prices = (1.0 + _bench(panel)).cumprod() * 250.0
    with pytest.raises(ValueError, match="PRICE series"):
        holding_attribution(book, _equal_end_weights(book), bench_prices)


def test_a_multi_column_benchmark_is_refused(panel: pd.DataFrame) -> None:
    book = _book(panel)
    with pytest.raises(ValueError, match="a benchmark is one instrument"):
        holding_attribution(book, _equal_end_weights(book), panel[["BENCH", "FLAT"]])


def test_benchmark_accepts_a_name_series_pair_and_carries_the_name(panel: pd.DataFrame) -> None:
    book = _book(panel)
    attr = holding_attribution(book, _equal_end_weights(book), ("NIFTYBEES.NS", _bench(panel).rename(None)))
    assert attr.benchmark_ticker == "NIFTYBEES.NS"
    assert "NIFTYBEES.NS" in format_attribution_report(attr)


def test_unknown_basis_and_weights_as_of_are_refused(panel: pd.DataFrame) -> None:
    book = _book(panel)
    with pytest.raises(ValueError, match="unknown basis"):
        holding_attribution(book, _equal_end_weights(book), _bench(panel), basis="equal")
    with pytest.raises(ValueError, match="weights_as_of"):
        holding_attribution(book, _equal_end_weights(book), _bench(panel), weights_as_of="middle")


def test_weights_as_a_plain_dict_and_as_a_series_agree(panel: pd.DataFrame) -> None:
    """A pandas Series is not a Mapping and iterating one yields its VALUES.
    Both paths must give the same answer or one of them is walking floats."""
    book = _book(panel)
    w = {"BIGLOSER": 0.3, "MIDLOSER": 0.2, "FLAT": 0.2, "WINNER": 0.2, "TINYDISASTER": 0.1}
    a = holding_attribution(book, w, _bench(panel))
    b = holding_attribution(book, pd.Series(w), _bench(panel))
    assert a.gap_total == pytest.approx(b.gap_total, abs=1e-15)


def test_fetch_benchmark_uses_the_holdings_fetcher_and_returns_one_name() -> None:
    """fetch_benchmark must go through holdings.fetch_prices — that is how it
    inherits the disk cache a public demo needs. Injecting a fetcher proves the
    path without touching the network."""
    idx = pd.bdate_range("2022-01-03", periods=400)
    closes = pd.Series(np.linspace(100.0, 160.0, len(idx)), index=idx)

    calls: list[tuple[str, str]] = []

    def fake(symbol: str, period: str):
        calls.append((symbol, period))
        if symbol != "NIFTYBEES.NS":
            return None
        return closes, "INR"

    bench = fetch_benchmark("NIFTYBEES.NS", period="2y", fetcher=fake)
    assert bench.usable
    assert bench.tickers == ("NIFTYBEES.NS",)
    assert calls and calls[0] == ("NIFTYBEES.NS", "2y")


# --------------------------------------------------------------------------- #
# 7. Corporate-action scrub
# --------------------------------------------------------------------------- #


def test_a_suspected_split_is_blanked_and_reported_not_silently_compounded() -> None:
    """GOLDBEES.NS's 1:100 split survives yfinance's adjustment as a -99% day
    followed by a +9900% one. Compounding that silently would put a nonsense
    number in front of somebody who owns the fund."""
    n = 400
    idx = _dates(n)
    col = np.full(n, 0.0005)
    col[150] = -0.99  # the split day, unadjusted
    col[151] = 50.0  # and the print that undoes it, imperfectly
    frame = pd.DataFrame({"SPLIT": col, "OTHER": np.full(n, 0.0004), "BENCH": np.full(n, 0.0004)}, index=idx)
    book = frame[["SPLIT", "OTHER"]]

    attr = holding_attribution(book, _equal_end_weights(book), frame["BENCH"], scrub=True)
    split = next(h for h in attr if h.ticker == "SPLIT")
    assert split.scrubbed_days == 2
    assert any("corporate" in note for note in attr.notes)
    assert any("compounded as 0.0%" in note for note in attr.notes)
    assert any("2023-" not in note for note in attr.notes)  # dates are quoted, not counted away

    unscrubbed = holding_attribution(book, _equal_end_weights(book), frame["BENCH"], scrub=False)
    raw = next(h for h in unscrubbed if h.ticker == "SPLIT").total_return
    # 0.01 * 51 = 0.51 against the +22% the other 398 days earned: the pair
    # leaves a ~-38% hole that no real price move made.
    assert raw == pytest.approx(-0.3777, abs=0.01)
    assert split.total_return > 0.0
    assert split.total_return - raw > 0.4


# --------------------------------------------------------------------------- #
# 8. Multi-window — the part that is the product
# --------------------------------------------------------------------------- #


def test_multi_window_surfaces_windows_that_contradict_the_full_window() -> None:
    """A book that trails overall but beat the benchmark in one calendar year
    must have that year reported, and reported FIRST."""
    idx = pd.bdate_range("2021-01-01", periods=1000)
    year = pd.Series(idx.year, index=idx)
    book_drift = np.where(year.to_numpy() == 2022, 0.0025, -0.0008)
    frame = pd.DataFrame(
        {"A": book_drift, "B": book_drift - 0.0001, "BENCH": np.full(len(idx), 0.0004)},
        index=idx,
    )
    book = frame[["A", "B"]]
    mw = multi_window_attribution(book, _equal_end_weights(book), frame["BENCH"], method="calendar_year")
    assert mw.full_window.gap_total < 0
    assert mw.direction_consistent is False
    labels = [w.window.label for w in mw.windows_contradicting]
    assert "2022" in labels
    text = format_multi_window_report(mw)
    assert text.index("CONTRADICT") < text.index("WINDOWS THAT AGREE")
    assert "did NOT hold" in mw.headline_qualifier


def test_direction_consistent_when_every_window_agrees(panel: pd.DataFrame) -> None:
    n = 1000
    idx = pd.bdate_range("2021-01-01", periods=n)
    frame = pd.DataFrame(
        {"A": np.full(n, -0.0010), "B": np.full(n, -0.0009), "BENCH": np.full(n, 0.0004)},
        index=idx,
    )
    book = frame[["A", "B"]]
    mw = multi_window_attribution(book, _equal_end_weights(book), frame["BENCH"], method="calendar_year")
    assert mw.direction_consistent is True
    assert mw.windows_contradicting == ()
    assert "all" in mw.headline_qualifier


def test_the_gap_varies_across_windows_it_is_not_one_number_repeated(panel: pd.DataFrame) -> None:
    """LESSONS F-18. A value identical across every window is not a robust
    result, it is a disconnected computation. Before any of these numbers is
    called good it must be shown to move."""
    book = _book(panel)
    mw = multi_window_attribution(book, _equal_end_weights(book), _bench(panel), method="calendar_year")
    gaps = mw.gaps
    assert len(gaps) >= 3, "fixture must span at least three calendar years"
    assert len(set(round(g, 9) for g in gaps)) == len(gaps), f"gaps do not vary: {gaps}"
    assert max(gaps) - min(gaps) > 1e-3


def test_short_windows_are_reported_not_dropped() -> None:
    """A stub year at the edge of the panel must appear with a reason. Dropping
    it would hide that the window does not cover it."""
    idx = pd.bdate_range("2021-11-01", periods=700)
    n = len(idx)
    frame = pd.DataFrame(
        {"A": np.full(n, -0.0005), "B": np.full(n, -0.0004), "BENCH": np.full(n, 0.0004)},
        index=idx,
    )
    book = frame[["A", "B"]]
    mw = multi_window_attribution(book, _equal_end_weights(book), frame["BENCH"], method="calendar_year")
    stub = [w for w in mw.windows if not w.usable]
    assert stub, "the 2021 stub must be present"
    assert all(w.skipped_reason for w in stub)
    assert str(MIN_WINDOW_DAYS) in stub[0].skipped_reason
    assert "not measured" in format_multi_window_report(mw).lower()


def test_rolling_windows_report_how_few_of_them_are_independent(panel: pd.DataFrame) -> None:
    """24 overlapping 252-day windows are not 24 pieces of evidence. The count
    that matters is the non-overlapping one, and it must be stated."""
    book = _book(panel)
    mw = multi_window_attribution(
        book, _equal_end_weights(book), _bench(panel), method="rolling", window_days=252, step_days=21
    )
    assert len(mw.usable_windows) > mw.independent_windows
    assert mw.independent_windows == len(panel.index) // 252
    assert "independent" in mw.overlap_note
    assert any("independent" in n for n in mw.notes)


def test_sub_windows_use_the_same_share_counts_not_rebalanced_weights(panel: pd.DataFrame) -> None:
    """If each window started from the pasted weights, the book would be
    silently rebalanced at every boundary — buying each winner back down after
    it ran, which flatters it. The day-one weights of the FIRST sub-window must
    match the full window's, and later ones must have drifted."""
    book = _book(panel)
    mw = multi_window_attribution(book, _equal_end_weights(book), _bench(panel), method="calendar_year")
    usable = mw.usable_windows
    assert len(usable) >= 2
    first, later = usable[0], usable[-1]
    for t, v in mw.full_window.weights_start.items():
        assert first.result.weights_start[t] == pytest.approx(v, abs=1e-9)
    drift = max(
        abs(later.result.weights_start[t] - mw.full_window.weights_start[t]) for t in mw.full_window.weights_start
    )
    assert drift > 1e-3, "later windows show no weight drift: the book is being rebalanced"


def test_every_sub_window_reconciles(panel: pd.DataFrame) -> None:
    book = _book(panel)
    mw = multi_window_attribution(book, _equal_end_weights(book), _bench(panel), method="calendar_year")
    for w in mw.usable_windows:
        assert abs(w.result.residual) <= RECONCILIATION_TOLERANCE


def test_unknown_window_method_is_refused_and_says_why(panel: pd.DataFrame) -> None:
    book = _book(panel)
    with pytest.raises(ValueError, match="No data-driven window rule is offered"):
        multi_window_attribution(book, _equal_end_weights(book), _bench(panel), method="best_window")


def test_an_unusable_full_window_refuses_rather_than_reporting_sub_windows() -> None:
    n = 40
    idx = _dates(n)
    frame = pd.DataFrame({"A": np.full(n, 0.001), "BENCH": np.zeros(n)}, index=idx)
    with pytest.raises(ValueError, match="nothing for"):
        multi_window_attribution(frame[["A"]], pd.Series({"A": 1.0}), frame["BENCH"])


def test_calendar_half_is_available_and_gives_more_windows_than_years(panel: pd.DataFrame) -> None:
    book = _book(panel)
    years = multi_window_attribution(book, _equal_end_weights(book), _bench(panel), method="calendar_year")
    halves = multi_window_attribution(book, _equal_end_weights(book), _bench(panel), method="calendar_half")
    assert len(halves.windows) > len(years.windows)
    assert halves.independent_windows == len(halves.windows)


# --------------------------------------------------------------------------- #
# 9. Survivorship
# --------------------------------------------------------------------------- #


def test_survivorship_note_names_both_directions_and_claims_neither(panel: pd.DataFrame) -> None:
    text = survivorship_note(_book(panel))
    low = text.lower()
    # Both directions must be named...
    assert "upward" in low and "downward" in low
    assert "opposite" in low
    assert "sold" in low
    assert str(_book(panel).shape[1]) in text
    # ...and neither may be claimed as the net one.
    assert "the net direction is unknown" in low
    for claimed in ("therefore", "so the book", "which means the book"):
        assert claimed not in low


def test_survivorship_note_picks_up_delisted_exclusions_from_the_panel() -> None:
    class FakePanel:
        returns = pd.DataFrame({"A": np.full(300, 0.001), "B": np.full(300, 0.001)}, index=_dates(300))
        excluded = (
            type("Exc", (), {"symbol": "GONE", "reason": "last price is 2024-02-01, 400 days before the rest"})(),
            type("Exc", (), {"symbol": "TYPO", "reason": "no price history on yfinance for TYPO.NS"})(),
        )

    text = survivorship_note(FakePanel())
    assert "GONE" in text
    assert "2 symbol(s)" in text


def test_survivorship_note_never_raises_on_an_unusable_panel() -> None:
    """It is printed next to a refusal as often as next to a result."""
    assert survivorship_note(object())
    assert survivorship_note(pd.DataFrame())


# --------------------------------------------------------------------------- #
# 10. No advice, ever
# --------------------------------------------------------------------------- #


def test_no_rendered_string_contains_advice(panel: pd.DataFrame) -> None:
    """SEBI Investment Adviser regulations. "Your book returned less than the
    index" is arithmetic; "you should index" is regulated advice and we are not
    licensed. The word list is concentration.py's, shared deliberately."""
    book = _book(panel)
    attr = holding_attribution(book, _equal_end_weights(book), _bench(panel))
    summary = decision_summary(attr)
    mw = multi_window_attribution(book, _equal_end_weights(book), _bench(panel), method="calendar_year")

    texts = {
        "attribution report": format_attribution_report(attr, summary),
        "multi-window report": format_multi_window_report(mw),
        "survivorship note": survivorship_note(book),
        "summary statement": summary["statement"],
        "headline qualifier": mw.headline_qualifier,
        "residual explanation": attr.residual_explanation,
    }
    for name, text in texts.items():
        found = _advice_words_in(text)
        assert not found, f"{name} contains advice words {found}"


def test_no_holding_statement_contains_advice_or_a_future_tense(panel: pd.DataFrame) -> None:
    book = _book(panel)
    attr = holding_attribution(book, _equal_end_weights(book), _bench(panel))
    for h in attr:
        s = h.statement(attr.benchmark_ticker)
        assert not _advice_words_in(s), s
        for forecast in (" will ", " expect", " likely", " going to "):
            assert forecast not in s.lower(), s


# --------------------------------------------------------------------------- #
# 11. Statistics and labelling
# --------------------------------------------------------------------------- #


def test_return_per_vol_is_annualised_return_over_annualised_vol() -> None:
    n = 500
    rng = np.random.default_rng(3)
    idx = _dates(n)
    col = rng.normal(0.0005, 0.01, n)
    frame = pd.DataFrame({"A": col, "BENCH": np.zeros(n)}, index=idx)
    attr = holding_attribution(frame[["A"]], pd.Series({"A": 1.0}), frame["BENCH"])
    book = attr.book
    expected_vol = float(np.std(col, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    expected_ann = float((1.0 + book.total_return) ** (TRADING_DAYS_PER_YEAR / n) - 1.0)
    assert book.volatility == pytest.approx(expected_vol, rel=1e-9)
    assert book.annualised_return == pytest.approx(expected_ann, rel=1e-9)
    assert book.return_per_vol == pytest.approx(expected_ann / expected_vol, rel=1e-9)


def test_annualisation_uses_252_trading_days_and_not_calendar_days() -> None:
    """MUTATION GUARD. The test above computes its expectation FROM
    ``TRADING_DAYS_PER_YEAR``, so changing that constant to 365 leaves it
    passing — the expectation moves with the code. This one pins the
    convention with literals: one trading year of data must annualise to
    itself, and 252 must be the number.
    """
    assert TRADING_DAYS_PER_YEAR == 252
    n = 252
    idx = _dates(n)
    frame = pd.DataFrame({"A": np.full(n, 0.001), "BENCH": np.zeros(n)}, index=idx)
    attr = holding_attribution(frame[["A"]], pd.Series({"A": 1.0}), frame["BENCH"], min_days=60)
    assert attr.book.total_return == pytest.approx(1.001**252 - 1.0, rel=1e-12)
    # Exactly one trading year: annualising it must be the identity.
    assert attr.book.annualised_return == pytest.approx(attr.book.total_return, rel=1e-12)
    assert attr.book.annualisation_is_extrapolation is False


def test_return_per_vol_is_none_with_a_reason_when_the_series_never_moved() -> None:
    n = 300
    idx = _dates(n)
    frame = pd.DataFrame({"A": np.full(n, 0.0005), "BENCH": np.zeros(n)}, index=idx)
    attr = holding_attribution(frame[["A"]], pd.Series({"A": 1.0}), frame["BENCH"])
    assert attr.benchmark.return_per_vol is None
    assert "zero" in attr.benchmark.return_per_vol_reason


def test_a_sub_year_window_is_flagged_as_extrapolated() -> None:
    n = 120
    rng = np.random.default_rng(4)
    idx = _dates(n)
    frame = pd.DataFrame({"A": rng.normal(0.001, 0.01, n), "BENCH": np.zeros(n)}, index=idx)
    attr = holding_attribution(frame[["A"]], pd.Series({"A": 1.0}), frame["BENCH"])
    assert attr.book.annualisation_is_extrapolation is True
    assert "extrapolated" in format_attribution_report(attr)


def test_the_report_never_calls_return_per_vol_a_sharpe_ratio(panel: pd.DataFrame) -> None:
    book = _book(panel)
    text = format_attribution_report(holding_attribution(book, _equal_end_weights(book), _bench(panel)))
    assert "not a Sharpe ratio" in text
    assert "Sharpe ratio of" not in text


# --------------------------------------------------------------------------- #
# 12. Determinism
# --------------------------------------------------------------------------- #


def test_the_same_inputs_give_byte_identical_reports(panel: pd.DataFrame) -> None:
    book = _book(panel)
    a = format_attribution_report(
        holding_attribution(book, _equal_end_weights(book), _bench(panel)),
        decision_summary(holding_attribution(book, _equal_end_weights(book), _bench(panel))),
    )
    b = format_attribution_report(
        holding_attribution(book, _equal_end_weights(book), _bench(panel)),
        decision_summary(holding_attribution(book, _equal_end_weights(book), _bench(panel))),
    )
    assert a == b
