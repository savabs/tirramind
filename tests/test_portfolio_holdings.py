"""Tests for agent.portfolio.holdings.

Design rule taken from LESSONS F-16/F-18: a test that cannot go red when the
behaviour it names is broken is worse than no test. Every assertion here was
checked by deliberately breaking the corresponding line in holdings.py and
confirming the test failed (see MUTATIONS section of the task report).

All price fetching is injected. Nothing in this file touches the network, so
these tests cannot turn green because of a cache hit or red because yfinance is
rate-limiting.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from agent.portfolio.holdings import (
    MIN_ALIGNED_DAYS,
    Holdings,
    Unit,
    fetch_prices,
    parse_holdings,
)

# ---------------------------------------------------------------------------
# Synthetic price fixtures — deterministic, no network.
# ---------------------------------------------------------------------------

NSE_HOLIDAYS = {"2024-03-08", "2024-03-25", "2024-04-11"}
US_HOLIDAYS = {"2024-03-29", "2024-05-27", "2024-07-04"}


def _calendar(start: str, end: str, holidays: set[str]) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    return pd.DatetimeIndex([d for d in days if d.strftime("%Y-%m-%d") not in holidays])


def _walk(index: pd.DatetimeIndex, seed: int, start_price: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.012, len(index))
    return pd.Series(start_price * np.exp(np.cumsum(steps)), index=index)


def make_fetcher(spec: dict[str, tuple[pd.Series, str]], calls: list[str] | None = None):
    """Build a (symbol, period) -> (close, currency) | None fetcher from a dict."""

    def _fetch(symbol: str, period: str):
        if calls is not None:
            calls.append(symbol)
        return spec.get(symbol)

    return _fetch


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_PORTFOLIO_CACHE", str(tmp_path / "yfcache"))
    return tmp_path / "yfcache"


@pytest.fixture
def three_year_spec():
    """Two INR names on the NSE calendar, one USD name on the US calendar."""
    nse = _calendar("2023-01-02", "2025-12-31", NSE_HOLIDAYS)
    us = _calendar("2023-01-02", "2025-12-31", US_HOLIDAYS)
    fx = _calendar("2023-01-02", "2025-12-31", set())
    return {
        "RELIANCE.NS": (_walk(nse, 1, 1200.0), "INR"),
        "TCS.NS": (_walk(nse, 2, 3400.0), "INR"),
        "AAPL": (_walk(us, 3, 180.0), "USD"),
        "USDINR=X": (pd.Series(np.linspace(82.0, 88.0, len(fx)), index=fx), "INR"),
    }


# ===========================================================================
# PARSING
# ===========================================================================


class TestParseFreeform:
    def test_percent_shares_and_suffixed_forms(self):
        h = parse_holdings("RELIANCE 30%\nAAPL, 50 shares\nINFY.NS 120")
        # percent and value cannot be combined; percent is the minority here (1 vs 2)
        assert h.unit_mode.startswith("value")
        assert h.symbols == ("AAPL", "INFY.NS")
        assert [e.quantity for e in h.entries] == [50.0, 120.0]
        assert all(e.unit is Unit.SHARES for e in h.entries)
        # the rejected percentage line is REPORTED, not silently dropped
        assert any("RELIANCE" in p.raw for p in h.unreadable)
        assert any("percent" in p.reason for p in h.unreadable)

    def test_all_percent_basket_renormalises_and_says_so(self):
        h = parse_holdings("RELIANCE 30%\nTCS 25%\nINFY 20%")
        assert h.unit_mode == "percent"
        assert len(h) == 3
        assert any("75.0%" in a for a in h.assumptions)

    def test_amounts_with_currency_symbols(self):
        h = parse_holdings("HDFCBANK ₹50,000\nAAPL $2,000")
        by_sym = {e.symbol: e for e in h.entries}
        assert by_sym["HDFCBANK"].quantity == 50_000.0
        assert by_sym["HDFCBANK"].currency == "INR"
        assert by_sym["HDFCBANK"].unit is Unit.AMOUNT
        assert by_sym["AAPL"].quantity == 2_000.0
        assert by_sym["AAPL"].currency == "USD"

    def test_thousands_comma_is_not_a_field_delimiter(self):
        h = parse_holdings("HDFCBANK, 1,250 shares")
        assert len(h) == 1
        assert h.entries[0].quantity == 1250.0

    def test_trailing_unit_word_relabels_the_number(self):
        h = parse_holdings("TCS 25000 INR")
        assert h.entries[0].unit is Unit.AMOUNT
        assert h.entries[0].currency == "INR"
        assert h.entries[0].quantity == 25000.0

    def test_bare_symbol_list_assumes_equal_weight_and_reports_it(self):
        h = parse_holdings("RELIANCE\nTCS\nINFY\nHDFCBANK")
        assert len(h) == 4
        assert all(e.unit is Unit.PERCENT for e in h.entries)
        assert all(e.quantity == pytest.approx(25.0) for e in h.entries)
        assert any("EQUAL WEIGHT" in a for a in h.assumptions)

    def test_unreadable_line_is_surfaced_not_swallowed(self):
        h = parse_holdings("RELIANCE 10\n???\nTCS 5")
        assert len(h) == 2
        assert len(h.unreadable) == 1
        assert h.unreadable[0].raw == "???"
        assert h.unreadable[0].line_no == 2

    def test_missing_quantity_among_sized_lines_is_rejected_with_reason(self):
        h = parse_holdings("RELIANCE 10\nTCS\nINFY 5")
        assert h.symbols == ("RELIANCE", "INFY")
        assert any(p.raw == "TCS" and "no quantity" in p.reason for p in h.unreadable)

    def test_duplicate_symbol_is_summed_and_reported(self):
        h = parse_holdings("RELIANCE 10\nRELIANCE 15")
        assert len(h) == 1
        assert h.entries[0].quantity == 25.0
        assert any("added them together" in a for a in h.assumptions)

    def test_multi_number_line_names_the_numbers_it_ignored(self):
        h = parse_holdings("RELIANCE 10 1234.50 12345.00")
        assert h.entries[0].quantity == 10.0
        note = " ".join(h.assumptions)
        assert "4 numbers" in note or "3 numbers" in note
        assert "1234.5" in note

    def test_negative_quantity_is_rejected(self):
        h = parse_holdings("RELIANCE -10")
        assert len(h) == 0
        assert any("not positive" in p.reason for p in h.unreadable)

    def test_empty_input_is_empty_not_an_exception(self):
        h = parse_holdings("")
        assert h == Holdings()


class TestParseCsvAndBrokerExport:
    def test_header_driven_csv(self):
        text = "Symbol,Quantity,Avg Price\nRELIANCE.NS,10,1200\nTCS.NS,5,3400\n"
        h = parse_holdings(text)
        assert h.symbols == ("RELIANCE.NS", "TCS.NS")
        assert [e.quantity for e in h.entries] == [10.0, 5.0]
        assert all(e.unit is Unit.SHARES for e in h.entries)

    def test_header_picks_quantity_column_not_the_first_number(self):
        """The whole point of header detection: without it, 'Avg Price' first
        would be read as the quantity."""
        text = "Instrument,Avg Price,Qty\nRELIANCE,1200.50,10\n"
        h = parse_holdings(text)
        assert h.entries[0].quantity == 10.0

    def test_header_weight_column_gives_percentages(self):
        text = "Ticker,Weightage\nRELIANCE.NS,40\nTCS.NS,60\n"
        h = parse_holdings(text)
        assert h.unit_mode == "percent"
        assert [e.unit for e in h.entries] == [Unit.PERCENT, Unit.PERCENT]

    def test_pipe_delimited_broker_dump_with_grouped_numbers(self):
        text = "RELIANCE INDUSTRIES LTD | RELIANCE.NS | 10 | 1,234.50 | 12,345.00"
        h = parse_holdings(text)
        assert h.symbols == ("RELIANCE.NS",)
        assert h.entries[0].quantity == 10.0
        assert any("exchange suffix" in a for a in h.assumptions)


# ===========================================================================
# FETCH / ALIGN
# ===========================================================================


class TestResolution:
    def test_bare_symbol_resolves_to_ns_and_reports_what_it_tried(self, three_year_spec, isolated_cache):
        calls: list[str] = []
        panel = fetch_prices(
            parse_holdings("RELIANCE 50\nTCS 10"),
            fetcher=make_fetcher(three_year_spec, calls),
        )
        assert panel.tickers == ("RELIANCE.NS", "TCS.NS")
        note = {r.typed: r.note for r in panel.resolutions}
        assert "RELIANCE.NS" in note["RELIANCE"]
        assert "wrong listing" in note["RELIANCE"]
        assert "RELIANCE.NS" in calls

    def test_unknown_ticker_is_excluded_with_the_list_of_attempts(self, three_year_spec, isolated_cache):
        panel = fetch_prices(
            parse_holdings("RELIANCE 50\nNOTATICKER 10"),
            fetcher=make_fetcher(three_year_spec),
        )
        assert panel.tickers == ("RELIANCE.NS",)
        bad = [e for e in panel.excluded if e.symbol == "NOTATICKER"]
        assert len(bad) == 1
        assert "NOTATICKER.NS" in bad[0].reason and "NOTATICKER.BO" in bad[0].reason
        assert "[" not in bad[0].reason, "a python list repr leaked into a user-facing reason"


class TestAlignment:
    def test_intersection_and_per_ticker_day_loss(self, three_year_spec, isolated_cache):
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS 50\nTCS.NS 10\nAAPL 5"),
            fetcher=make_fetcher(three_year_spec),
        )
        nse_days = len(three_year_spec["RELIANCE.NS"][0])
        us_days = len(three_year_spec["AAPL"][0])
        assert panel.n_days < min(nse_days, us_days), "intersecting two calendars must lose days"
        assert panel.n_returns == panel.n_days - 1
        # every ticker's loss is reported, and matches its own history length
        for tic in panel.tickers:
            own = int(panel.coverage.loc[tic, "own_days"])
            lost = int(panel.coverage.loc[tic, "days_lost_to_alignment"])
            assert lost == own - panel.n_days
            assert lost >= 0
        assert int(panel.coverage.loc["RELIANCE.NS", "days_lost_to_alignment"]) > 0

    def test_no_nans_survive_into_returns(self, three_year_spec, isolated_cache):
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS 50\nAAPL 5"),
            fetcher=make_fetcher(three_year_spec),
        )
        assert not panel.prices.isna().to_numpy().any()
        assert not panel.returns.isna().to_numpy().any()

    def test_window_string_carries_a_date_range_and_an_n(self, three_year_spec, isolated_cache):
        panel = fetch_prices(parse_holdings("RELIANCE.NS 50\nTCS.NS 1"), fetcher=make_fetcher(three_year_spec))
        assert " to " in panel.window
        assert f"{panel.n_returns} daily returns" in panel.window

    def test_window_end_names_the_ticker_that_did_not_trade(self, isolated_cache):
        """The single most confusing thing a stranger can see: a window ending
        days before their broker's last price. It must name the culprit."""
        idx = _calendar("2024-01-01", "2024-12-31", set())
        lazy = idx.drop(idx[-1])  # ETFBEES simply did not print on the last day
        spec = {"A.NS": (_walk(idx, 1), "INR"), "ETFBEES.NS": (_walk(lazy, 2), "INR")}
        panel = fetch_prices(parse_holdings("A.NS 10\nETFBEES.NS 10"), fetcher=make_fetcher(spec))
        assert panel.end == idx[-2]
        note = " ".join(panel.window_note)
        assert "ETFBEES.NS" in note and "no price that day" in note
        assert idx[-1].strftime("%Y-%m-%d") in note

    def test_window_start_names_the_shortest_history(self, isolated_cache):
        idx = _calendar("2024-01-01", "2024-12-31", set())
        late = idx[100:]
        spec = {"A.NS": (_walk(idx, 1), "INR"), "LATE.NS": (_walk(late, 2), "INR")}
        panel = fetch_prices(parse_holdings("A.NS 10\nLATE.NS 10"), fetcher=make_fetcher(spec))
        note = " ".join(panel.window_note)
        assert "LATE.NS" in note
        assert f"starts {late[0]:%Y-%m-%d}" in note

    def test_returns_actually_vary(self, three_year_spec, isolated_cache):
        """LESSONS F-18: a value identical across the series is disconnected,
        not converged. Confirm the return series is not a constant."""
        panel = fetch_prices(parse_holdings("RELIANCE.NS 50\nTCS.NS 1"), fetcher=make_fetcher(three_year_spec))
        for tic in panel.tickers:
            assert panel.returns[tic].std() > 1e-6
            assert panel.returns[tic].nunique() > panel.n_returns * 0.9


class TestExclusionRules:
    def test_short_history_is_excluded_with_its_day_count(self, three_year_spec, isolated_cache):
        short_idx = _calendar("2025-11-03", "2025-12-31", set())
        spec = {**three_year_spec, "NEWIPO.NS": (_walk(short_idx, 9, 500.0), "INR")}
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS 50\nTCS.NS 10\nNEWIPO.NS 100"),
            fetcher=make_fetcher(spec),
        )
        assert "NEWIPO.NS" not in panel.tickers
        exc = [e for e in panel.excluded if e.resolved == "NEWIPO.NS"]
        assert len(exc) == 1
        assert f"only {len(short_idx)} trading days" in exc[0].reason
        # and it did NOT drag the survivors' window down with it
        assert panel.n_returns > 400

    def test_delisted_name_is_excluded_with_its_last_price_date(self, three_year_spec, isolated_cache):
        dead_idx = _calendar("2023-01-02", "2024-06-28", NSE_HOLIDAYS)
        spec = {**three_year_spec, "DEAD.NS": (_walk(dead_idx, 11, 90.0), "INR")}
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS 50\nTCS.NS 10\nDEAD.NS 100"),
            fetcher=make_fetcher(spec),
        )
        assert "DEAD.NS" not in panel.tickers
        exc = [e for e in panel.excluded if e.resolved == "DEAD.NS"][0]
        assert "2024-06-28" in exc.reason
        assert "delisted" in exc.reason

    def test_every_excluded_item_carries_a_nonempty_reason(self, three_year_spec, isolated_cache):
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS 50\nGARBAGE 1\n???"),
            fetcher=make_fetcher(three_year_spec),
        )
        assert panel.excluded
        for e in panel.excluded:
            assert e.reason.strip(), f"{e.symbol} excluded with no reason — this is the silent-exclusion bug"

    def test_unparsed_lines_propagate_into_panel_exclusions(self, three_year_spec, isolated_cache):
        panel = fetch_prices(parse_holdings("RELIANCE.NS 50\n%%%\n"), fetcher=make_fetcher(three_year_spec))
        assert any("%%%" in e.symbol for e in panel.excluded)


class TestCurrency:
    def test_usd_holding_is_converted_not_silently_mixed(self, three_year_spec, isolated_cache):
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS 50\nAAPL 100"),
            base_currency="INR",
            fetcher=make_fetcher(three_year_spec),
        )
        assert panel.base_currency == "INR"
        # AAPL at ~180 USD must land near 180*~85 INR, i.e. >> its USD level
        aapl_inr = float(panel.prices["AAPL"].iloc[-1])
        aapl_usd = float(three_year_spec["AAPL"][0].reindex(panel.prices.index).iloc[-1])
        assert aapl_inr / aapl_usd > 50, "USD series was not converted to INR"
        assert any("AAPL" in line and "USDINR=X" in line for line in panel.fx_report)

    def test_fx_forward_fill_count_is_reported(self, three_year_spec, isolated_cache):
        # Knock a week out of the FX series: those days must be reported as filled.
        fx, cur = three_year_spec["USDINR=X"]
        gapped = fx.drop(fx.index[100:105])
        spec = {**three_year_spec, "USDINR=X": (gapped, cur)}
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS 50\nAAPL 100"), base_currency="INR", fetcher=make_fetcher(spec)
        )
        line = next(x for x in panel.fx_report if x.startswith("AAPL"))
        m = re.search(r"(\d+) of (\d+) days used a forward-filled rate", line)
        assert m is not None, f"fx_report line did not report a fill count: {line}"
        assert int(m.group(1)) > 0, f"forward-filled days not counted: {line}"

    def test_missing_fx_rate_excludes_rather_than_mixes(self, three_year_spec, isolated_cache):
        spec = {k: v for k, v in three_year_spec.items() if k != "USDINR=X"}
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS 50\nAAPL 100"), base_currency="INR", fetcher=make_fetcher(spec)
        )
        assert "AAPL" not in panel.tickers
        exc = [e for e in panel.excluded if e.resolved == "AAPL"][0]
        assert "USD->INR" in exc.reason

    def test_base_currency_inferred_from_the_majority_and_reported(self, three_year_spec, isolated_cache):
        panel = fetch_prices(parse_holdings("RELIANCE.NS 50\nTCS.NS 10\nAAPL 5"), fetcher=make_fetcher(three_year_spec))
        assert panel.base_currency == "INR"
        assert any("converted everything to INR" in a for a in panel.assumptions)
        assert any("INR and USD" in a for a in panel.assumptions)

    def test_unknown_currency_is_excluded(self, three_year_spec, isolated_cache):
        nse = three_year_spec["RELIANCE.NS"][0].index
        spec = {**three_year_spec, "MYSTERY.NS": (_walk(nse, 21, 50.0), "UNKNOWN")}
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS 50\nMYSTERY.NS 10"), base_currency="INR", fetcher=make_fetcher(spec)
        )
        assert "MYSTERY.NS" not in panel.tickers
        assert any("did not report a currency" in e.reason for e in panel.excluded)


class TestWeights:
    def test_share_weights_use_the_last_aligned_close(self, three_year_spec, isolated_cache):
        panel = fetch_prices(parse_holdings("RELIANCE.NS 50\nTCS.NS 10"), fetcher=make_fetcher(three_year_spec))
        last = panel.prices.iloc[-1]
        expected_r = 50 * last["RELIANCE.NS"]
        expected_t = 10 * last["TCS.NS"]
        total = expected_r + expected_t
        assert panel.weights["RELIANCE.NS"] == pytest.approx(expected_r / total)
        assert panel.weights.sum() == pytest.approx(1.0)
        assert "close" in panel.weight_basis

    def test_percent_weights_are_renormalised(self, three_year_spec, isolated_cache):
        panel = fetch_prices(parse_holdings("RELIANCE.NS 30%\nTCS.NS 10%"), fetcher=make_fetcher(three_year_spec))
        assert panel.weights["RELIANCE.NS"] == pytest.approx(0.75)
        assert panel.weights["TCS.NS"] == pytest.approx(0.25)
        assert panel.weights.sum() == pytest.approx(1.0)

    def test_amount_in_foreign_currency_is_converted_for_weighting(self, three_year_spec, isolated_cache):
        panel = fetch_prices(
            parse_holdings("RELIANCE.NS ₹100000\nAAPL $1000"),
            base_currency="INR",
            fetcher=make_fetcher(three_year_spec),
        )
        rate = float(three_year_spec["USDINR=X"][0].reindex(panel.prices.index).ffill().iloc[-1])
        expected = (1000 * rate) / (100000 + 1000 * rate)
        assert panel.weights["AAPL"] == pytest.approx(expected, rel=1e-6)
        assert "USD->INR" in panel.weight_basis

    def test_weights_sum_to_one_across_every_sizing_mode(self, three_year_spec, isolated_cache):
        for text in ("RELIANCE.NS 50\nTCS.NS 10", "RELIANCE.NS 60%\nTCS.NS 40%", "RELIANCE.NS\nTCS.NS"):
            panel = fetch_prices(parse_holdings(text), fetcher=make_fetcher(three_year_spec))
            assert panel.weights.sum() == pytest.approx(1.0), text


class TestUsabilityFloor:
    def test_panel_below_the_floor_is_marked_unusable_with_its_n(self, isolated_cache):
        """Two holdings, 80 shared days, a 200-day floor. We must NOT buy window
        length by dropping one of only two names — we say the panel is too short."""
        idx_a = _calendar("2025-01-02", "2025-12-31", set())
        idx_b = idx_a[-80:]  # 80 own days, enough to pass MIN_OWN_DAYS
        spec = {"A.NS": (_walk(idx_a, 1), "INR"), "B.NS": (_walk(idx_b, 2), "INR")}
        panel = fetch_prices(parse_holdings("A.NS 10\nB.NS 10"), min_days=200, fetcher=make_fetcher(spec))
        assert panel.usable is False
        assert set(panel.tickers) == {"A.NS", "B.NS"}, "dropped a name from a two-name basket"
        assert panel.n_returns == 79
        assert "79" in panel.unusable_reason
        assert "200 is the floor" in panel.unusable_reason

    def test_binding_constraint_is_dropped_and_named(self, isolated_cache):
        """With three names and a 200-day floor, the one short history that is
        costing everyone else their window is dropped — and named."""
        long_idx = _calendar("2023-01-02", "2025-12-31", set())
        short_idx = long_idx[-150:]
        spec = {
            "A.NS": (_walk(long_idx, 1), "INR"),
            "B.NS": (_walk(long_idx, 2), "INR"),
            "C.NS": (_walk(short_idx, 3), "INR"),
        }
        panel = fetch_prices(parse_holdings("A.NS 10\nB.NS 10\nC.NS 10"), min_days=200, fetcher=make_fetcher(spec))
        assert "C.NS" not in panel.tickers
        exc = [e for e in panel.excluded if e.resolved == "C.NS"][0]
        assert "binding constraint" in exc.reason
        assert "150 days" in exc.reason
        assert panel.usable is True
        assert panel.n_returns == len(long_idx) - 1
        assert any("lifted the shared window from 150 to" in a for a in panel.assumptions)

    def test_a_long_enough_shared_window_drops_nobody(self, isolated_cache):
        long_idx = _calendar("2023-01-02", "2025-12-31", set())
        short_idx = long_idx[-150:]
        spec = {
            "A.NS": (_walk(long_idx, 1), "INR"),
            "B.NS": (_walk(long_idx, 2), "INR"),
            "C.NS": (_walk(short_idx, 3), "INR"),
        }
        panel = fetch_prices(
            parse_holdings("A.NS 10\nB.NS 10\nC.NS 10"),
            min_days=MIN_ALIGNED_DAYS,
            fetcher=make_fetcher(spec),
        )
        assert set(panel.tickers) == {"A.NS", "B.NS", "C.NS"}
        assert panel.n_returns == 149
        assert panel.usable is True

    def test_all_holdings_unknown_gives_an_unusable_panel_not_a_crash(self, isolated_cache):
        panel = fetch_prices(parse_holdings("NOPE 10\nALSONOPE 5"), fetcher=make_fetcher({}))
        assert panel.usable is False
        assert panel.prices.empty
        assert panel.n_returns == 0
        assert len(panel.excluded) == 2
        assert panel.unusable_reason


class TestCaching:
    def test_second_call_does_not_refetch(self, three_year_spec, isolated_cache):
        calls: list[str] = []
        fetcher = make_fetcher(three_year_spec, calls)
        text = "RELIANCE.NS 50\nTCS.NS 10"
        fetch_prices(parse_holdings(text), fetcher=fetcher)
        first = len(calls)
        assert first > 0
        fetch_prices(parse_holdings(text), fetcher=fetcher)
        assert len(calls) == first, "cache did not prevent a refetch — a public demo would be rate-limited"

    def test_failures_are_cached_too(self, isolated_cache):
        calls: list[str] = []
        fetcher = make_fetcher({}, calls)
        fetch_prices(parse_holdings("NOPE 1"), fetcher=fetcher)
        first = len(calls)
        fetch_prices(parse_holdings("NOPE 1"), fetcher=fetcher)
        assert len(calls) == first

    def test_cached_series_round_trips_exactly(self, three_year_spec, isolated_cache):
        text = "RELIANCE.NS 50\nTCS.NS 10"
        warm = fetch_prices(parse_holdings(text), fetcher=make_fetcher(three_year_spec))
        cold = fetch_prices(parse_holdings(text), fetcher=make_fetcher({}))  # cache only
        pd.testing.assert_frame_equal(warm.prices, cold.prices)

    def test_expired_cache_entry_is_refetched(self, three_year_spec, isolated_cache, monkeypatch):
        calls: list[str] = []
        fetcher = make_fetcher(three_year_spec, calls)
        fetch_prices(parse_holdings("RELIANCE.NS 50\nTCS.NS 1"), fetcher=fetcher)
        n = len(calls)
        for p in isolated_cache.glob("*.json"):
            blob = json.loads(p.read_text())
            blob["fetched_at"] = 0.0
            p.write_text(json.dumps(blob))
        fetch_prices(parse_holdings("RELIANCE.NS 50\nTCS.NS 1"), fetcher=fetcher)
        assert len(calls) > n


class TestDescribe:
    def test_describe_names_every_exclusion(self, three_year_spec, isolated_cache):
        panel = fetch_prices(parse_holdings("RELIANCE.NS 50\nAAPL 10\nNOPE 1"), fetcher=make_fetcher(three_year_spec))
        text = panel.describe()
        assert "NOPE" in text
        assert "EXCLUDED" in text
        assert panel.window in text
        for tic in panel.tickers:
            assert tic in text
