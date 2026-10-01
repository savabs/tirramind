"""Tests for agent.portfolio.overlap.

The bar here is set by this repo's failure log: nearly every recorded fuckup
was silent, and four of the tests written to catch one of them asserted the
bug instead. So these tests are written to fail when the code is *wrong*, not
when it merely changes:

* Planted structure. Synthetic panels are built from known latent factors, and
  the tests assert the exact recovered grouping, not "some clusters exist".
* Negative controls. A pure-noise portfolio must yield no factors and no
  clusters; a null that preserves autocorrelation must not fool the selection
  gate. A test suite with no negative control cannot tell a working detector
  from one that always says yes.
* Exclusion accounting. Every input ticker must appear either in the output or
  in ``excluded`` with a reason. This is asserted as a conservation law, which
  is the only form that catches a *new* silent drop path added later.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from agent.portfolio import overlap
from agent.portfolio.overlap import (
    Cluster,
    correlation_clusters,
    factor_exposure,
    format_overlap_report,
    load_reference_returns,
    threshold_sweep,
)

LIVE_DB = "/Users/becmachlean/Projects/tirramind/.tirra_pipeline/pipeline.db"


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _business_days(n: int, start: str = "2023-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _panel_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Prices whose pct_change reproduces `returns` (first row is the base)."""
    return 100.0 * (1.0 + returns).cumprod()


def planted_panel(
    n_days: int = 500,
    *,
    seed: int = 0,
    groups: tuple[tuple[str, ...], ...] = (
        ("IT_A", "IT_B", "IT_C"),
        ("BANK_A", "BANK_B"),
        ("LONE",),
    ),
    loading: float = 0.94,
) -> pd.DataFrame:
    """Prices with an exactly known correlation structure.

    Each group shares one latent factor at `loading`; the residual is
    independent. Within-group correlation is therefore ~loading**2 and
    between-group correlation is ~0, so the correct clustering is `groups`.
    """
    rng = np.random.default_rng(seed)
    idx = _business_days(n_days)
    columns: dict[str, np.ndarray] = {}
    for group in groups:
        latent = rng.standard_normal(n_days)
        for name in group:
            noise = rng.standard_normal(n_days)
            columns[name] = 0.01 * (loading * latent + math.sqrt(1 - loading**2) * noise)
    return _panel_from_returns(pd.DataFrame(columns, index=idx))


def equal_weights(panel: pd.DataFrame) -> dict[str, float]:
    return {c: 1.0 for c in panel.columns}


def write_reference_db(path, tickers: dict[str, np.ndarray], dates: pd.DatetimeIndex) -> str:
    """A stand-in pipeline DB with the live schema, so the fast tests do not
    depend on (or risk touching) the real one."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL, created_at REAL NOT NULL, metadata_json TEXT);
        CREATE TABLE entity_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, entity_id TEXT NOT NULL,
            source_tool TEXT NOT NULL, observed_at REAL NOT NULL, ingested_at REAL NOT NULL,
            observation_type TEXT NOT NULL, depth_level INTEGER NOT NULL DEFAULT 1,
            value_json TEXT NOT NULL, metadata_json TEXT);
        """
    )
    for i, (ticker, simple_returns) in enumerate(tickers.items()):
        eid = f"e{i:04d}"
        conn.execute(
            "INSERT INTO entities VALUES (?,?,?,?,?)",
            (
                eid,
                "instrument",
                f"{ticker} name",
                0.0,
                json.dumps({"ticker": ticker, "asset_class": "equity_etf", "region": "X"}),
            ),
        )
        for day, value in zip(dates, simple_returns):
            observed = dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()
            conn.execute(
                "INSERT INTO entity_observations "
                "(entity_id, source_tool, observed_at, ingested_at, observation_type, value_json) "
                "VALUES (?,?,?,?,?,?)",
                (eid, "t", observed, 0.0, "instrument_daily", json.dumps({"log_return": math.log1p(value)})),
            )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def clear_reference_cache():
    overlap._load_reference_returns_cached.cache_clear()
    yield
    overlap._load_reference_returns_cached.cache_clear()


# ==========================================================================
# correlation_clusters — structure recovery
# ==========================================================================


def test_planted_groups_are_recovered_exactly():
    panel = planted_panel()
    result = correlation_clusters(panel, equal_weights(panel))
    recovered = {frozenset(c.members) for c in result}
    assert recovered == {frozenset({"IT_A", "IT_B", "IT_C"}), frozenset({"BANK_A", "BANK_B"})}
    assert result.singletons == ("LONE",)
    assert result.n_groups == 3


def test_independent_holdings_form_no_group():
    """Negative control. If everything is independent there is no same-bet,
    and a detector that cannot return nothing is not a detector."""
    panel = planted_panel(groups=(("A",), ("B",), ("C",), ("D",)))
    result = correlation_clusters(panel, equal_weights(panel))
    assert list(result) == []
    assert result.n_groups == 4
    assert set(result.singletons) == {"A", "B", "C", "D"}


def test_cluster_correlations_match_a_direct_computation():
    """Guards against the numbers being plausible but not the thing named."""
    panel = planted_panel()
    result = correlation_clusters(panel, equal_weights(panel))
    cluster = next(c for c in result if len(c.members) == 3)
    direct = panel.pct_change().dropna()[list(cluster.members)].corr().to_numpy()
    pairs = direct[np.triu_indices_from(direct, k=1)]
    assert cluster.mean_correlation == pytest.approx(float(pairs.mean()), abs=1e-9)
    assert cluster.min_correlation == pytest.approx(float(pairs.min()), abs=1e-9)
    assert cluster.min_correlation_ci[0] < cluster.min_correlation < cluster.min_correlation_ci[1]


def test_threshold_is_a_choice_and_moves_the_group_count():
    panel = planted_panel()
    tight = correlation_clusters(panel, equal_weights(panel), correlation_threshold=0.95)
    default = correlation_clusters(panel, equal_weights(panel))
    loose = correlation_clusters(panel, equal_weights(panel), correlation_threshold=-0.5)
    assert (tight.n_groups, default.n_groups, loose.n_groups) == (6, 3, 1)


def test_stability_range_brackets_the_threshold_and_holds_the_partition():
    """The defence against 'you picked the cutoff to get that answer'. The
    reported range must actually be a plateau: re-running anywhere inside it
    gives the identical grouping, and just outside it does not."""
    panel = planted_panel()
    weights = equal_weights(panel)
    result = correlation_clusters(panel, weights, correlation_threshold=0.6)
    low, high = result.stable_between
    assert low <= 0.6 <= high
    target = {frozenset(c.members) for c in result}
    for probe in (low, (low + high) / 2, high):
        again = correlation_clusters(panel, weights, correlation_threshold=probe)
        assert {frozenset(c.members) for c in again} == target
    outside = correlation_clusters(panel, weights, correlation_threshold=high + 0.02)
    assert {frozenset(c.members) for c in outside} != target, "range is not a maximal plateau"


def test_a_grouping_that_exists_at_only_one_cutoff_reports_a_narrow_plateau():
    """Negative control for the stability claim: manufactured near-ties must
    produce a narrow range, not a wide one, or the number means nothing."""
    panel = planted_panel()
    wide = correlation_clusters(panel, equal_weights(panel), correlation_threshold=0.6)
    narrow = correlation_clusters(panel, equal_weights(panel), correlation_threshold=0.885)
    wide_span = wide.stable_between[1] - wide.stable_between[0]
    narrow_span = narrow.stable_between[1] - narrow.stable_between[0]
    assert wide_span > narrow_span
    # A plateau must always contain the cutoff it was measured at, including
    # when it is one step wide. A range that does not bracket its own
    # threshold is not a range, it is a number pointing somewhere else.
    for result, threshold in ((wide, 0.6), (narrow, 0.885)):
        low, high = result.stable_between
        assert low <= threshold <= high, f"{result.stable_between} does not bracket {threshold}"
        assert low >= -1.0 and high <= 1.0


def test_threshold_sweep_exposes_the_whole_curve():
    panel = planted_panel()
    rows = threshold_sweep(panel, equal_weights(panel), thresholds=(0.95, 0.9, 0.8, 0.6, 0.4, 0.0))
    counts = [r["n_groups"] for r in rows]
    assert counts == sorted(counts, reverse=True), "group count must fall as the cutoff loosens"
    assert counts[0] == 6, "at rho>=0.95 nothing in a planted 0.88 structure groups"
    assert counts[-1] == 1, "at rho>=0 everything merges"
    # The planted structure is clean and well separated, so it must own a wide
    # plateau. If it did not, stable_between would be measuring nothing.
    middle = next(r for r in rows if r["threshold"] == 0.6)
    assert middle["n_groups"] == 3
    low, high = middle["stable_between"]
    assert high - low > 0.5


def test_weight_is_the_share_of_the_book_the_group_carries():
    panel = planted_panel()
    weights = {"IT_A": 30, "IT_B": 20, "IT_C": 10, "BANK_A": 20, "BANK_B": 10, "LONE": 10}
    result = correlation_clusters(panel, weights)
    it = next(c for c in result if "IT_A" in c.members)
    assert it.weight == pytest.approx(0.60, abs=1e-9)


def test_ranking_puts_the_big_heavy_group_above_the_tight_small_one():
    """'A cluster of 2 is not interesting; a cluster of 8 carrying 60% is.'
    The tight pair has the HIGHER correlation, so ranking by statistical
    magnitude would invert this. That inversion is the bug being guarded."""
    rng = np.random.default_rng(3)
    idx = _business_days(600)
    big = rng.standard_normal(600)
    pair = rng.standard_normal(600)
    cols = {}
    for i in range(8):
        cols[f"BIG{i}"] = 0.01 * (0.88 * big + 0.475 * rng.standard_normal(600))
    for i in range(2):
        cols[f"TWIN{i}"] = 0.01 * (0.995 * pair + 0.0999 * rng.standard_normal(600))
    panel = _panel_from_returns(pd.DataFrame(cols, index=idx))
    weights = {f"BIG{i}": 5.0 for i in range(8)} | {f"TWIN{i}": 30.0 for i in range(2)}
    result = correlation_clusters(panel, weights)
    twin = next(c for c in result if "TWIN0" in c.members)
    big = next(c for c in result if "BIG0" in c.members)
    # The pair is both MORE correlated and HEAVIER. Any ranking that weighs
    # only correlation, or only weight, or the product of the two, puts it
    # first. Only counting how many holdings the person thought were separate
    # puts the group of eight first, which is the finding.
    assert twin.mean_correlation > big.mean_correlation
    assert twin.weight > big.weight
    assert result[0] is big
    assert len(big.members) == 8
    assert big.weight == pytest.approx(0.40, abs=1e-9)


def test_near_identical_holdings_are_called_out_separately():
    panel = planted_panel()
    panel = panel.assign(IT_A_DUP=panel["IT_A"] * 1.0001)
    result = correlation_clusters(panel, equal_weights(panel))
    pairs = {frozenset((a, b)) for a, b, _ in result.near_duplicates}
    assert frozenset({"IT_A", "IT_A_DUP"}) in pairs
    assert all(r >= overlap.NEAR_DUPLICATE_THRESHOLD for _, _, r in result.near_duplicates)


# ==========================================================================
# correlation_clusters — honesty bar
# ==========================================================================


def test_short_history_holding_is_excluded_with_a_reason_not_used():
    panel = planted_panel()
    panel["NEWLY_LISTED"] = np.nan
    panel.iloc[-30:, panel.columns.get_loc("NEWLY_LISTED")] = np.linspace(100, 110, 30)
    result = correlation_clusters(panel, equal_weights(panel))
    assert "NEWLY_LISTED" not in result.holdings_used
    assert all("NEWLY_LISTED" not in c.members for c in result)
    reason = next(e for e in result.excluded if e.name == "NEWLY_LISTED")
    assert reason.reason == "insufficient history"
    assert "29" in reason.detail or "30" in reason.detail  # the actual count, not a vague phrase


def test_twenty_days_of_history_yields_no_correlation_at_all():
    """'20 days of history cannot yield a meaningful correlation; say so and
    show nothing rather than showing something shaky.'"""
    panel = planted_panel(n_days=21)
    result = correlation_clusters(panel, equal_weights(panel))
    assert list(result) == []
    assert result.window is None
    assert any("120" in n for n in result.notes)
    text = format_overlap_report(result)
    assert "correlation 0." not in text, "a correlation was printed on 20 days of data"
    assert "none at this threshold" in text
    assert "20 trading days of returns, need 120" in text


def test_a_holding_on_a_disjoint_calendar_is_dropped_and_named():
    """A holding that never trades on the same days as the rest would cap the
    shared window for everybody. It must be dropped loudly, not quietly kept
    on a pairwise-complete basis."""
    panel = planted_panel(n_days=400)
    offset = planted_panel(n_days=400, seed=99, groups=(("FOREIGN",),))
    offset.index = offset.index + pd.Timedelta(days=1000)
    panel = pd.concat([panel, offset], axis=1)
    result = correlation_clusters(panel, equal_weights(panel))
    names = {e.name for e in result.excluded}
    assert "FOREIGN" in names
    assert "FOREIGN" not in result.holdings_used
    assert result.window is not None and result.window.n_days >= overlap.MIN_TRADING_DAYS


def test_every_input_ticker_is_either_used_or_explained():
    """Conservation law. This is the test that catches a *future* silent drop."""
    panel = planted_panel()
    panel["NEWLY_LISTED"] = np.nan
    panel.iloc[-30:, panel.columns.get_loc("NEWLY_LISTED")] = np.linspace(100, 110, 30)
    weights = equal_weights(panel) | {"DELISTED.NS": 500.0, "ZERO_POSITION": 0.0}
    result = correlation_clusters(panel, weights)
    accounted = set(result.holdings_used) | {e.name for e in result.excluded}
    assert set(weights) <= accounted


def test_absent_ticker_is_reported_rather_than_ignored():
    panel = planted_panel()
    result = correlation_clusters(panel, equal_weights(panel) | {"NOT_A_TICKER": 100.0})
    entry = next(e for e in result.excluded if e.name == "NOT_A_TICKER")
    assert "no price history" in entry.reason


def test_weight_renormalisation_is_reported():
    panel = planted_panel()
    result = correlation_clusters(panel, {c: 50_000.0 for c in panel.columns})
    assert any("renormalised" in n for n in result.notes)


def test_dropping_part_of_the_book_is_flagged_because_weights_change_meaning():
    panel = planted_panel()
    panel["NEWLY_LISTED"] = np.nan
    panel.iloc[-30:, panel.columns.get_loc("NEWLY_LISTED")] = np.linspace(100, 110, 30)
    result = correlation_clusters(panel, equal_weights(panel))
    assert any("usable overlapping history" in n for n in result.notes)
    # The dropped holding's weight must be redistributed, not left as a hole:
    # shares are of the measured book, and they have to add up.
    assert sum(result.weights_used.values()) == pytest.approx(1.0, abs=1e-12)
    assert set(result.weights_used) == set(result.holdings_used)
    grouped = sum(c.weight for c in result)
    singles = sum(result.weights_used[s] for s in result.singletons)
    assert grouped + singles == pytest.approx(1.0, abs=1e-12)
    assert "NEWLY_LISTED" not in result.weights_used
    # Six equal holdings survive a seventh being dropped, so each is 1/6.
    assert all(v == pytest.approx(1 / 6, abs=1e-12) for v in result.weights_used.values())


def test_every_cluster_carries_its_window_and_n():
    panel = planted_panel()
    result = correlation_clusters(panel, equal_weights(panel))
    for c in result:
        assert c.window.n_days == result.window.n_days
        assert c.window.start < c.window.end
        assert str(c.window.n_days) in c.statement()
        assert c.window.start.isoformat() in c.statement()


def test_returns_handed_over_as_prices_raise_rather_than_silently_wrong():
    panel = planted_panel()
    returns = panel.pct_change().dropna()
    with pytest.raises(ValueError, match="not a price panel"):
        correlation_clusters(returns, equal_weights(returns))
    explicit = correlation_clusters(returns, equal_weights(returns), kind="returns")
    assert len(explicit) == 2


def test_returns_contract_is_a_real_list_of_clusters():
    panel = planted_panel()
    result = correlation_clusters(panel, equal_weights(panel))
    assert isinstance(result, list)
    assert all(isinstance(c, Cluster) for c in result)


def test_output_contains_no_advice():
    """Personalised investment advice is regulated and we are not licensed.
    Every rendered string must be a fact about the past."""
    banned = [
        "should",
        "recommend",
        "advis",
        "suggest",
        "consider",
        "you must",
        "buy",
        "sell",
        "trim",
        "rebalance",
        "overweight",
        "underweight",
        "will ",
        "expect",
        "forecast",
        "predict",
        "outlook",
        "opportunity",
        "risky",
        "too much",
        "safe",
    ]
    panel = planted_panel()
    result = correlation_clusters(panel, equal_weights(panel))
    text = format_overlap_report(result).lower()
    text += " ".join(c.statement() for c in result).lower()
    offenders = [w for w in banned if w in text]
    assert offenders == [], f"advice-shaped language in output: {offenders}"


# ==========================================================================
# factor_exposure
# ==========================================================================


def test_planted_factor_is_recovered_with_the_right_beta(tmp_path, clear_reference_cache):
    rng = np.random.default_rng(11)
    n = 600
    dates = _business_days(n)
    market = 0.01 * rng.standard_normal(n)
    others = {f"NOISE{i}": 0.01 * rng.standard_normal(n) for i in range(10)}
    db = write_reference_db(tmp_path / "ref.db", {"MKT": market, **others}, dates)

    holdings = {f"H{i}": 1.3 * market + 0.004 * rng.standard_normal(n) for i in range(4)}
    panel = _panel_from_returns(pd.DataFrame(holdings, index=dates))

    out = factor_exposure(panel, equal_weights(panel), db_path=db)
    assert out["sufficient"] is True
    assert [f["ticker"] for f in out["factors"]] == ["MKT"]
    assert out["factors"][0]["beta"] == pytest.approx(1.3, abs=0.05)
    assert out["r2"] > 0.85
    assert out["factors"][0]["role"] == "exposure"
    assert out["factors"][0]["material"] is True


def test_variance_shares_sum_to_r_squared(tmp_path, clear_reference_cache):
    rng = np.random.default_rng(12)
    n = 700
    dates = _business_days(n)
    a = 0.01 * rng.standard_normal(n)
    b = 0.01 * rng.standard_normal(n)
    db = write_reference_db(
        tmp_path / "ref.db",
        {"A": a, "B": b, **{f"N{i}": 0.01 * rng.standard_normal(n) for i in range(8)}},
        dates,
    )
    holdings = {"H0": 0.9 * a + 0.5 * b + 0.004 * rng.standard_normal(n)}
    panel = _panel_from_returns(pd.DataFrame(holdings, index=dates))
    out = factor_exposure(panel, {"H0": 1.0}, db_path=db)
    total = sum(f["variance_share"] for f in out["factors"])
    assert total == pytest.approx(out["r2"], abs=1e-9)
    assert out["unexplained_share"] == pytest.approx(1.0 - out["r2"], abs=1e-12)


def test_a_noise_portfolio_gets_no_factors(tmp_path, clear_reference_cache):
    """Negative control. 89 candidates against noise must yield nothing, and
    the output must say so rather than reporting the prettiest of 89."""
    rng = np.random.default_rng(13)
    n = 600
    dates = _business_days(n)
    db = write_reference_db(tmp_path / "ref.db", {f"F{i}": 0.01 * rng.standard_normal(n) for i in range(40)}, dates)
    panel = _panel_from_returns(pd.DataFrame({"H0": 0.01 * rng.standard_normal(n)}, index=dates))
    out = factor_exposure(panel, {"H0": 1.0}, db_path=db)
    assert out["sufficient"] is True
    assert out["factors"] == []
    assert out["r2"] == 0.0
    assert out["unexplained_share"] == 1.0
    assert any("nothing in the" in w for w in out["warnings"])


def test_selection_is_capped_and_the_cap_is_reported(tmp_path, clear_reference_cache):
    rng = np.random.default_rng(14)
    n = 800
    dates = _business_days(n)
    factors = {f"F{i}": 0.01 * rng.standard_normal(n) for i in range(12)}
    db = write_reference_db(tmp_path / "ref.db", factors, dates)
    blend = sum(0.3 * v for v in factors.values()) + 0.002 * rng.standard_normal(n)
    panel = _panel_from_returns(pd.DataFrame({"H0": blend}, index=dates))
    out = factor_exposure(panel, {"H0": 1.0}, db_path=db, max_factors=2)
    assert len(out["factors"]) <= 2
    assert "max 2 of" in out["selection"]
    assert out["candidates_examined"] == 12


def test_holdout_is_scored_on_days_the_selection_never_saw(tmp_path, clear_reference_cache):
    rng = np.random.default_rng(15)
    n = 700
    dates = _business_days(n)
    market = 0.01 * rng.standard_normal(n)
    db = write_reference_db(
        tmp_path / "ref.db",
        {"MKT": market, **{f"N{i}": 0.01 * rng.standard_normal(n) for i in range(10)}},
        dates,
    )
    panel = _panel_from_returns(pd.DataFrame({"H0": market + 0.003 * rng.standard_normal(n)}, index=dates))
    out = factor_exposure(panel, {"H0": 1.0}, db_path=db)
    holdout = out["holdout"]
    train_end = holdout["train_window"].split("..")[1].split(" ")[0]
    test_start = holdout["test_window"].split("..")[0]
    assert test_start > train_end, "holdout overlaps the slice selection was run on"
    assert holdout["selected_on_train"] == ["MKT"]
    assert holdout["r2"] > 0.7


def test_holdout_selection_cannot_see_the_holdout(tmp_path, clear_reference_cache):
    """F-04 in miniature. The relationship here exists ONLY in the final 40%.
    An honest holdout selects on the first 60%, sees nothing, and says so. A
    holdout that peeks at the full window finds the factor and reports a fit
    it could never have predicted — which is exactly what a leaked eval looks
    like: a good number produced by a broken wire."""
    rng = np.random.default_rng(23)
    n = 700
    dates = _business_days(n)
    signal = 0.01 * rng.standard_normal(n)
    db = write_reference_db(
        tmp_path / "ref.db",
        {"LATE": signal, **{f"N{i}": 0.01 * rng.standard_normal(n) for i in range(10)}},
        dates,
    )
    cut = int(n * 0.6)
    holding = 0.01 * rng.standard_normal(n)
    holding[cut:] = 3.0 * signal[cut:] + 0.001 * rng.standard_normal(n - cut)
    panel = _panel_from_returns(pd.DataFrame({"H0": holding}, index=dates))
    out = factor_exposure(panel, {"H0": 1.0}, db_path=db)
    assert out["holdout"]["selected_on_train"] == [], (
        "the holdout selected a factor that is only detectable in the held-out slice, so selection saw the holdout"
    )


def test_selection_gate_holds_under_circular_shift_null(tmp_path, clear_reference_cache):
    """The strongest cheap null: shift the portfolio in time. Its entire
    autocorrelation structure survives; only its alignment with the factors
    dies. If forward selection still finds factors here, the whole
    factor_exposure output is a garden of forking paths."""
    rng = np.random.default_rng(16)
    n = 700
    dates = _business_days(n)
    factors = pd.DataFrame({f"F{i}": 0.01 * rng.standard_normal(n) for i in range(60)}, index=dates)
    y = 0.01 * rng.standard_normal(n)
    y = pd.Series(y).rolling(3).mean().bfill().to_numpy()  # give it autocorrelation
    picks = []
    for shift in (97, 211, 313, 421, 557):
        shifted = np.concatenate([y[shift:], y[:shift]])
        chosen, _ = overlap._forward_select(shifted, factors, 3, 0.05)
        picks.append(len(chosen))
    assert sum(picks) <= 1, f"gate leaks under the null: {picks}"


def test_insufficient_history_reports_no_numbers_at_all(tmp_path, clear_reference_cache):
    rng = np.random.default_rng(17)
    n = 40
    dates = _business_days(n)
    db = write_reference_db(tmp_path / "ref.db", {"MKT": 0.01 * rng.standard_normal(n)}, dates)
    panel = _panel_from_returns(pd.DataFrame({"H0": 0.01 * rng.standard_normal(n)}, index=dates))
    out = factor_exposure(panel, {"H0": 1.0}, db_path=db)
    assert out["sufficient"] is False
    assert "120" in out["reason"]
    for forbidden in ("r2", "adjusted_r2", "factors", "beta"):
        assert forbidden not in out, f"{forbidden} reported despite insufficient data"


def test_degenerate_factor_is_excluded_with_a_reason(tmp_path, clear_reference_cache):
    rng = np.random.default_rng(18)
    n = 600
    dates = _business_days(n)
    db = write_reference_db(
        tmp_path / "ref.db",
        {"FLAT": np.zeros(n), "MKT": 0.01 * rng.standard_normal(n)},
        dates,
    )
    panel = _panel_from_returns(pd.DataFrame({"H0": 0.01 * rng.standard_normal(n)}, index=dates))
    out = factor_exposure(panel, {"H0": 1.0}, db_path=db)
    entry = next(e for e in out["excluded"] if e["name"] == "FLAT")
    assert entry["reason"] == "reference factor has no variation"


def test_ragged_factor_is_excluded_rather_than_shrinking_everyone_else(tmp_path, clear_reference_cache):
    """Admitting a factor with a torn calendar costs *every* number days,
    because all candidates share one sample. A factor missing a third of the
    window must be refused by name, not quietly admitted."""
    rng = np.random.default_rng(24)
    n = 600
    dates = _business_days(n)
    market = 0.01 * rng.standard_normal(n)
    ragged = 0.01 * rng.standard_normal(n)
    ragged[rng.choice(n, size=int(n * 0.35), replace=False)] = np.nan
    db = write_reference_db(tmp_path / "ref.db", {"MKT": market, "RAGGED": ragged}, dates)
    # NaNs are not written as observations at all, mirroring a real gap.
    panel = _panel_from_returns(pd.DataFrame({"H0": market + 0.003 * rng.standard_normal(n)}, index=dates))
    out = factor_exposure(panel, {"H0": 1.0}, db_path=db)
    entry = next(e for e in out["excluded"] if e["name"] == "RAGGED")
    assert entry["reason"] == "reference factor does not cover the window"
    assert "%" in entry["detail"], "coverage excluded without saying how bad it was"
    assert out["window"]["n_days"] > int(n * 0.9), "a ragged factor still shrank the sample"


def test_significant_but_tiny_factor_is_flagged_immaterial(tmp_path, clear_reference_cache):
    """With enough days a factor worth a quarter of a percent of variance is
    overwhelmingly significant. On a public page that line reads as a finding.
    It must be reported, and reported as not one."""
    rng = np.random.default_rng(25)
    n = 2000
    dates = _business_days(n)
    big = 0.01 * rng.standard_normal(n)
    tiny = 0.01 * rng.standard_normal(n)
    db = write_reference_db(
        tmp_path / "ref.db",
        {"BIG": big, "TINY": tiny, **{f"N{i}": 0.01 * rng.standard_normal(n) for i in range(6)}},
        dates,
    )
    holding = big + 0.05 * tiny + 0.0004 * rng.standard_normal(n)
    panel = _panel_from_returns(pd.DataFrame({"H0": holding}, index=dates))
    out = factor_exposure(panel, {"H0": 1.0}, db_path=db)
    picked = {f["ticker"]: f for f in out["factors"]}
    assert "TINY" in picked, "test is void unless the tiny factor is actually selected"
    assert picked["TINY"]["hac_p_value"] < 1e-6, "test is void unless it is significant"
    assert abs(picked["TINY"]["variance_share"]) < overlap.MIN_MATERIAL_VARIANCE_SHARE
    assert picked["TINY"]["material"] is False
    assert picked["BIG"]["material"] is True
    assert any("TINY" in w and "material=False" in w for w in out["warnings"])


def test_every_excluded_entry_has_a_reason(tmp_path, clear_reference_cache):
    rng = np.random.default_rng(19)
    n = 600
    dates = _business_days(n)
    db = write_reference_db(tmp_path / "ref.db", {"MKT": 0.01 * rng.standard_normal(n)}, dates)
    panel = _panel_from_returns(pd.DataFrame({"H0": 0.01 * rng.standard_normal(n)}, index=dates))
    out = factor_exposure(panel, {"H0": 1.0, "GHOST": 5.0}, db_path=db)
    assert {e["name"] for e in out["excluded"]} >= {"GHOST"}
    for entry in out["excluded"]:
        assert entry["reason"].strip(), f"{entry['name']} excluded with an empty reason"


def test_the_live_db_is_only_ever_opened_read_only(tmp_path, monkeypatch, clear_reference_cache):
    """Rule 2 of the brief. Asserted at the sqlite3 boundary so it holds for
    any future query path, not just the one written today."""
    rng = np.random.default_rng(20)
    n = 300
    dates = _business_days(n)
    db = write_reference_db(tmp_path / "ref.db", {"MKT": 0.01 * rng.standard_normal(n)}, dates)

    seen: list[tuple] = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        seen.append((target, kwargs))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(overlap.sqlite3, "connect", spy)
    load_reference_returns(db)
    assert seen, "no connection was opened"
    for target, kwargs in seen:
        assert kwargs.get("uri") is True
        assert "mode=ro" in str(target)


def test_reference_loader_returns_varying_series(clear_reference_cache):
    """F-18: before calling any number good, confirm the series varies."""
    pytest.importorskip("pandas")
    try:
        returns, meta = load_reference_returns(LIVE_DB)
    except (FileNotFoundError, sqlite3.Error):  # pragma: no cover
        pytest.skip("live pipeline DB not present")
    assert returns.shape[1] >= 80
    stds = returns.std(skipna=True)
    assert (stds > 1e-5).all(), f"flat reference series: {list(stds[stds <= 1e-5].index)}"
    assert returns.notna().sum().min() > 500
    assert set(meta.columns) == {"name", "asset_class", "quote_currency"}
    assert meta.loc["USDINR=X", "quote_currency"] == "USD/INR"
    assert meta.loc["SPY", "quote_currency"] == "USD"


def test_live_db_end_to_end_on_a_real_shaped_portfolio(clear_reference_cache):
    """Integration: the real 89-instrument reference set, a synthetic book
    built to track the India ETF. Catches schema drift in the live DB."""
    try:
        reference, _ = load_reference_returns(LIVE_DB)
    except (FileNotFoundError, sqlite3.Error):  # pragma: no cover
        pytest.skip("live pipeline DB not present")
    inda = reference["INDA"].dropna()
    assert len(inda) > 500
    rng = np.random.default_rng(21)
    frame = pd.DataFrame(
        {f"H{i}": 1.1 * inda.to_numpy() + 0.004 * rng.standard_normal(len(inda)) for i in range(3)},
        index=pd.to_datetime(inda.index),
    )
    panel = _panel_from_returns(frame)
    out = factor_exposure(panel, equal_weights(panel), db_path=LIVE_DB)
    assert out["sufficient"] is True
    assert out["candidates_examined"] >= 80
    assert "INDA" in [f["ticker"] for f in out["factors"]]
    assert out["r2"] > 0.5


def test_report_renders_without_advice_and_shows_exclusions(tmp_path, clear_reference_cache):
    rng = np.random.default_rng(22)
    n = 600
    dates = _business_days(n)
    market = 0.01 * rng.standard_normal(n)
    db = write_reference_db(
        tmp_path / "ref.db",
        {"MKT": market, **{f"N{i}": 0.01 * rng.standard_normal(n) for i in range(5)}},
        dates,
    )
    holdings = {f"H{i}": market + 0.004 * rng.standard_normal(n) for i in range(3)}
    panel = _panel_from_returns(pd.DataFrame(holdings, index=dates))
    weights = equal_weights(panel) | {"GHOST": 10.0}
    text = format_overlap_report(correlation_clusters(panel, weights), factor_exposure(panel, weights, db_path=db))
    assert "GHOST" in text
    assert "trading days" in text
    assert "Unexplained" in text
    assert "should" not in text.lower()
