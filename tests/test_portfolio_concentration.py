"""Tests for agent.portfolio.concentration.

Two kinds of test here, deliberately:

1. **Calibration against synthetic ground truth.**  For N equally-weighted,
   equal-volatility, mutually uncorrelated assets the headline must return
   exactly N; for perfectly correlated assets it must return 1.  These are the
   only tests in the file that can tell you the headline *means* what the
   docstring claims, rather than merely being computed without crashing.

2. **Tests that a defect would actually fail.**  LESSONS F-09/F-13 record four
   tests in this repo that asserted the bug they were meant to catch.  Every
   assertion below is written so that the obvious way to break the function
   makes it red - the accompanying mutation run is recorded in the task
   output.  In particular ``test_metrics_respond_to_weights`` exists because a
   function that silently ignored ``weights`` would still pass every
   sums-to-one and non-negativity check in this file.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agent.portfolio import concentration as C

SEED = 20260929


# --------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------


def _panel(array: np.ndarray, names: list[str], start: str = "2023-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=array.shape[0])
    return pd.DataFrame(array, index=idx, columns=names)


def independent_panel(n: int, p: int, vol: float = 0.01, seed: int = SEED) -> pd.DataFrame:
    """p mutually independent, identically-scaled return series."""
    rng = np.random.default_rng(seed)
    return _panel(rng.normal(0.0, vol, size=(n, p)), [f"A{i}" for i in range(p)])


def identical_panel(n: int, p: int, vol: float = 0.01, seed: int = SEED) -> pd.DataFrame:
    """p copies of one series: one asset wearing p different names."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, vol, size=(n, 1))
    return _panel(np.repeat(base, p, axis=1), [f"A{i}" for i in range(p)])


def factor_panel(n: int, p: int, beta: float = 0.8, idio: float = 0.01, seed: int = SEED) -> pd.DataFrame:
    """One common factor plus idiosyncratic noise: a realistic equity book."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 0.01, size=(n, 1))
    e = rng.normal(0.0, idio, size=(n, p))
    return _panel(beta * f + e, [f"A{i}" for i in range(p)])


def equal_weights(panel: pd.DataFrame) -> dict[str, float]:
    p = panel.shape[1]
    return {c: 1.0 / p for c in panel.columns}


# --------------------------------------------------------------------------
# 1. CALIBRATION - the headline must mean what the docstring says it means
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 3, 5, 10])
def test_headline_equals_n_for_n_independent_equal_assets(p: int) -> None:
    """DR^2 must return exactly p for p uncorrelated equal-vol equal-weight assets.

    This is the claim the product headline makes in words ("your 12 positions
    had the volatility of 2.7 independent ones"). If this test does not hold,
    the sentence is false and the number is decoration.
    """
    panel = independent_panel(n=6000, p=p)
    out = C.effective_bets(panel, equal_weights(panel))
    assert out["headline_supported"] is True
    assert out["effective_bets_diversification"] == pytest.approx(p, rel=0.05)


@pytest.mark.parametrize("p", [2, 4, 8])
def test_headline_is_one_for_perfectly_correlated_assets(p: int) -> None:
    """p names that are literally the same series are one bet, not p."""
    panel = identical_panel(n=3000, p=p)
    out = C.effective_bets(panel, equal_weights(panel))
    assert out["effective_bets_diversification"] == pytest.approx(1.0, abs=1e-6)


def test_headline_is_one_for_a_single_holding() -> None:
    panel = independent_panel(n=1000, p=1)
    out = C.effective_bets(panel, {"A0": 1.0})
    assert out["effective_bets_diversification"] == pytest.approx(1.0, abs=1e-9)
    assert out["pc_for_90pct_variance"] == 1


def test_headline_is_bounded_below_by_one() -> None:
    """DR >= 1 always; a value below 1 would mean the portfolio is more
    volatile than every holding in it, which is arithmetically impossible."""
    rng = np.random.default_rng(SEED)
    for _ in range(25):
        p = int(rng.integers(2, 9))
        panel = factor_panel(n=900, p=p, beta=float(rng.uniform(0.0, 2.0)))
        w = rng.dirichlet(np.ones(p))
        out = C.effective_bets(panel, dict(zip(panel.columns, w)))
        assert out["effective_bets_diversification"] >= 1.0 - 1e-9
        assert out["effective_bets_diversification"] <= p + 1e-6


def test_concentrating_into_one_name_drives_headline_to_one() -> None:
    """The case that disqualified the PCA-entropy alternative.

    A book that is 95% one name is ~1 bet regardless of how many other names
    are nominally present. The headline must say so; PCA-entropy does not,
    and that difference is asserted here so the choice cannot be silently
    reverted to the textbook metric.
    """
    panel = factor_panel(n=2000, p=12)
    cols = list(panel.columns)
    conc = {cols[0]: 0.95, **{c: 0.05 / 11 for c in cols[1:]}}
    spread = equal_weights(panel)

    out_conc = C.effective_bets(panel, conc)
    out_spread = C.effective_bets(panel, spread)

    assert out_conc["effective_bets_diversification"] < 1.15
    assert out_conc["effective_bets_diversification"] < out_spread["effective_bets_diversification"]
    # And the documented reason the PCA version is not headlined: it moves the
    # WRONG way, scoring the concentrated book above the spread one.
    assert out_conc["effective_bets_pca_entropy"] > out_spread["effective_bets_pca_entropy"]
    assert out_conc["headline_metric"] == "effective_bets_diversification"


# --------------------------------------------------------------------------
# 2. RISK CONTRIBUTIONS
# --------------------------------------------------------------------------


def test_risk_contributions_sum_to_exactly_one() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(30):
        p = int(rng.integers(2, 10))
        panel = factor_panel(n=800, p=p, beta=float(rng.uniform(0.0, 1.5)))
        w = rng.dirichlet(np.ones(p))
        out = C.risk_contributions(panel, dict(zip(panel.columns, w)))
        total = sum(r["risk_share"] for r in out["contributions"])
        assert total == pytest.approx(1.0, abs=1e-5)
        assert out["sums_to"] == pytest.approx(1.0, abs=C._SUM_TOL)


def test_risk_contributions_match_covariance_with_portfolio_series() -> None:
    """Independent check by a completely different route.

    Holding i's Euler share equals w_i * Cov(r_i, r_portfolio) / Var(r_portfolio),
    computed from the raw return series with no covariance matrix involved.
    A bug in the matrix path (wrong axis, missing weight, using correlation
    where covariance belongs) breaks the agreement.
    """
    panel = factor_panel(n=1200, p=6, beta=0.7)
    rng = np.random.default_rng(SEED)
    w = rng.dirichlet(np.ones(6))
    weights = dict(zip(panel.columns, w))

    out = C.risk_contributions(panel, weights)
    got = {r["ticker"]: r["risk_share"] for r in out["contributions"]}

    R = panel.to_numpy()
    wv = np.array([weights[c] for c in panel.columns])
    rp = R @ wv
    var_p = float(np.var(rp, ddof=1))
    for i, col in enumerate(panel.columns):
        expected = wv[i] * float(np.cov(R[:, i], rp, ddof=1)[0, 1]) / var_p
        assert got[col] == pytest.approx(expected, abs=1e-6)


def test_risk_contributions_match_numerical_euler_derivative() -> None:
    """Third route: finite-difference d(sigma_p)/d(w_i), scaled by w_i."""
    panel = factor_panel(n=1000, p=5, beta=0.6)
    weights = {c: v for c, v in zip(panel.columns, [0.4, 0.25, 0.15, 0.12, 0.08])}
    out = C.risk_contributions(panel, weights)
    got = {r["ticker"]: r["risk_share"] for r in out["contributions"]}

    cov = np.cov(panel.to_numpy(), rowvar=False, ddof=1)
    wv = np.array([weights[c] for c in panel.columns])

    def vol(v: np.ndarray) -> float:
        return float(np.sqrt(v @ cov @ v))

    h = 1e-7
    for i, col in enumerate(panel.columns):
        e = np.zeros(len(wv))
        e[i] = h
        deriv = (vol(wv + e) - vol(wv - e)) / (2 * h)
        assert got[col] == pytest.approx(wv[i] * deriv / vol(wv), abs=1e-5)


def test_equal_weight_identical_assets_have_equal_risk_shares() -> None:
    panel = identical_panel(n=800, p=4)
    out = C.risk_contributions(panel, equal_weights(panel))
    shares = [r["risk_share"] for r in out["contributions"]]
    assert all(s == pytest.approx(0.25, abs=1e-9) for s in shares)


def test_a_small_volatile_holding_can_exceed_its_weight_in_risk() -> None:
    """The product's core claim, on constructed data where the answer is known."""
    rng = np.random.default_rng(SEED)
    n = 1500
    f = rng.normal(0, 0.01, size=(n, 1))
    calm = 0.8 * f + rng.normal(0, 0.004, size=(n, 3))
    wild = 3.0 * f + rng.normal(0, 0.030, size=(n, 1))
    panel = _panel(np.hstack([calm, wild]), ["C0", "C1", "C2", "WILD"])
    weights = {"C0": 0.32, "C1": 0.32, "C2": 0.32, "WILD": 0.04}

    out = C.risk_contributions(panel, weights)
    wild_row = next(r for r in out["contributions"] if r["ticker"] == "WILD")
    assert wild_row["weight"] == pytest.approx(0.04)
    # Assert the PROPERTY (large amplification), not a hand-picked magnitude.
    # On this construction WILD is 4% of the money and ~14.8% of the variance,
    # a 3.7x amplification. The threshold is deliberately below what the data
    # happens to produce rather than tuned up to meet it.
    assert wild_row["risk_share"] > 3 * wild_row["weight"]
    assert wild_row["risk_per_unit_weight"] > 3.0
    assert out["largest_amplifier"]["ticker"] == "WILD"


def test_negative_risk_contribution_is_reported_not_hidden() -> None:
    """A hedge genuinely has a negative share. Clipping it would make the
    percentages lie and would break the sum-to-one identity silently."""
    rng = np.random.default_rng(SEED)
    n = 1500
    f = rng.normal(0, 0.012, size=(n, 1))
    longs = f + rng.normal(0, 0.003, size=(n, 2))
    hedge = -1.0 * f + rng.normal(0, 0.002, size=(n, 1))
    panel = _panel(np.hstack([longs, hedge]), ["L0", "L1", "HEDGE"])
    out = C.risk_contributions(panel, {"L0": 0.45, "L1": 0.45, "HEDGE": 0.10})

    hedge_row = next(r for r in out["contributions"] if r["ticker"] == "HEDGE")
    assert hedge_row["risk_share"] < 0
    assert out["has_negative_contributions"] is True
    assert out["sums_to"] == pytest.approx(1.0, abs=C._SUM_TOL)
    assert out["largest_dampener"]["ticker"] == "HEDGE"


def test_negative_contribution_suppresses_effective_n_by_risk() -> None:
    """1/sum(r^2) is not an effective count when a share is negative."""
    rng = np.random.default_rng(SEED)
    n = 1200
    f = rng.normal(0, 0.012, size=(n, 1))
    panel = _panel(
        np.hstack([f + rng.normal(0, 0.003, size=(n, 2)), -f + rng.normal(0, 0.002, size=(n, 1))]),
        ["L0", "L1", "HEDGE"],
    )
    out = C.variance_concentration(panel, {"L0": 0.45, "L1": 0.45, "HEDGE": 0.10})
    assert out["effective_n_by_risk"] is None
    assert "negative" in out["effective_n_by_risk_reason"]


# --------------------------------------------------------------------------
# 3. VARIANCE CONCENTRATION
# --------------------------------------------------------------------------


def test_variance_concentration_curve_is_monotone_and_ends_at_one() -> None:
    panel = factor_panel(n=900, p=8, beta=0.7)
    rng = np.random.default_rng(SEED)
    w = rng.dirichlet(np.ones(8))
    out = C.variance_concentration(panel, dict(zip(panel.columns, w)))
    cum = [c["cumulative_risk_share"] for c in out["cumulative_risk_curve"]]
    assert cum == sorted(cum)
    assert cum[-1] == pytest.approx(1.0, abs=1e-5)
    assert len(cum) == 8


def test_thresholds_are_nested_and_carry_their_names() -> None:
    panel = factor_panel(n=900, p=10, beta=0.7)
    rng = np.random.default_rng(SEED)
    w = rng.dirichlet(np.ones(10) * 0.4)  # deliberately lumpy
    out = C.variance_concentration(panel, dict(zip(panel.columns, w)))
    n50 = out["thresholds"]["50pct"]["n_names"]
    n80 = out["thresholds"]["80pct"]["n_names"]
    n90 = out["thresholds"]["90pct"]["n_names"]
    assert n50 <= n80 <= n90
    assert out["thresholds"]["50pct"]["cumulative_risk_share"] >= 0.5
    assert len(out["thresholds"]["80pct"]["names"]) == n80
    assert out["top_1_ticker"] == out["thresholds"]["50pct"]["names"][0]


def test_effective_n_by_weight_is_p_for_equal_weights() -> None:
    panel = factor_panel(n=600, p=6)
    out = C.variance_concentration(panel, equal_weights(panel))
    assert out["effective_n_by_weight"] == pytest.approx(6.0, abs=1e-6)


# --------------------------------------------------------------------------
# 4. THE DEFECT-DETECTION TESTS
#    Each of these fails for a mutation that every other test above survives.
# --------------------------------------------------------------------------


def test_metrics_respond_to_weights() -> None:
    """A function that ignored ``weights`` would pass sum-to-one, monotonicity
    and non-negativity everywhere else in this file. It fails here."""
    panel = factor_panel(n=1200, p=6, beta=0.5)
    cols = list(panel.columns)
    a = equal_weights(panel)
    b = {cols[0]: 0.80, **{c: 0.20 / 5 for c in cols[1:]}}

    eb_a, eb_b = C.effective_bets(panel, a), C.effective_bets(panel, b)
    assert eb_a["effective_bets_diversification"] != eb_b["effective_bets_diversification"]

    rc_a = {r["ticker"]: r["risk_share"] for r in C.risk_contributions(panel, a)["contributions"]}
    rc_b = {r["ticker"]: r["risk_share"] for r in C.risk_contributions(panel, b)["contributions"]}
    assert rc_b[cols[0]] > 0.7
    assert rc_a[cols[0]] < 0.3

    vc_a, vc_b = C.variance_concentration(panel, a), C.variance_concentration(panel, b)
    assert vc_b["top_1_risk_share"] > vc_a["top_1_risk_share"]


def test_metrics_respond_to_correlation_structure() -> None:
    """A function that ignored the off-diagonal covariance (using only
    variances) would return the same headline for independent and for
    identical assets. It fails here."""
    ind = independent_panel(n=2500, p=6)
    ident = identical_panel(n=2500, p=6)
    a = C.effective_bets(ind, equal_weights(ind))["effective_bets_diversification"]
    b = C.effective_bets(ident, equal_weights(ident))["effective_bets_diversification"]
    assert a > 5.0
    assert b == pytest.approx(1.0, abs=1e-6)


def test_risk_shares_are_not_just_the_weights() -> None:
    """Guards against the degenerate 'return the weights' implementation."""
    panel = factor_panel(n=1200, p=5, beta=0.6, idio=0.02)
    weights = dict(zip(panel.columns, [0.30, 0.25, 0.20, 0.15, 0.10]))
    out = C.risk_contributions(panel, weights)
    gaps = [abs(r["risk_share_minus_weight"]) for r in out["contributions"]]
    assert max(gaps) > 0.01, "risk shares are indistinguishable from weights"


def test_headline_varies_across_portfolios_f18() -> None:
    """LESSONS F-18: a value identical across inputs is disconnected, not
    converged. Eight structurally different portfolios must give eight
    different headline numbers."""
    panels = {
        "ind4": independent_panel(n=2000, p=4),
        "ind10": independent_panel(n=2000, p=10),
        "fac6": factor_panel(n=2000, p=6, beta=0.9),
        "fac6_weak": factor_panel(n=2000, p=6, beta=0.2),
        "ident5": identical_panel(n=2000, p=5),
    }
    vals = []
    for panel in panels.values():
        cols = list(panel.columns)
        vals.append(C.effective_bets(panel, equal_weights(panel))["effective_bets_diversification"])
        vals.append(
            C.effective_bets(panel, {cols[0]: 0.9, **{c: 0.1 / (len(cols) - 1) for c in cols[1:]}})[
                "effective_bets_diversification"
            ]
        )
    assert len(set(vals)) >= 8, f"headline barely varies across portfolios: {vals}"


# --------------------------------------------------------------------------
# 5. STATISTICAL HONESTY GATES
# --------------------------------------------------------------------------


def test_short_window_refuses_to_headline_and_says_why() -> None:
    """n < 10p must withhold the headline, and the reason must name n, p and
    the direction of the bias."""
    panel = independent_panel(n=40, p=10)  # n/p = 4
    out = C.effective_bets(panel, equal_weights(panel), min_obs=20)
    assert out["headline_supported"] is False
    reason = out["unsupported_reason"]
    assert "n=40" in reason and "p=10" in reason
    assert "UPWARD" in reason or "overstate" in reason
    # the number is still computed and available, just not headlined
    assert out["effective_bets_diversification"] is not None


def test_headline_is_supported_when_n_over_p_clears_the_gate() -> None:
    panel = independent_panel(n=10 * 6, p=6)  # n/p = exactly 10
    out = C.effective_bets(panel, equal_weights(panel))
    assert out["headline_supported"] is True
    assert "unsupported_reason" not in out


def test_singular_covariance_computes_nothing_at_all() -> None:
    panel = independent_panel(n=15, p=10)  # n/p = 1.5 < 2
    out = C.effective_bets(panel, equal_weights(panel), min_obs=10)
    assert out["headline_supported"] is False
    assert out["effective_bets_diversification"] is None
    assert out["subperiod_estimates"] == []
    rc = C.risk_contributions(panel, equal_weights(panel), min_obs=10)
    assert rc["reliable"] is False
    assert rc["contributions"] == []


def test_every_result_carries_its_sample_size_and_window() -> None:
    panel = factor_panel(n=500, p=5)
    w = equal_weights(panel)
    for out in (
        C.effective_bets(panel, w),
        C.risk_contributions(panel, w),
        C.variance_concentration(panel, w),
    ):
        assert out["n_observations"] == 500
        assert out["n_assets"] == 5
        assert out["window_start"] == "2023-01-02"
        assert len(out["window_end"]) == 10
        assert ".." in out["window"]
        assert "excluded" in out


def test_subperiod_estimates_are_reported_and_differ() -> None:
    panel = factor_panel(n=1200, p=5, beta=0.7)
    out = C.effective_bets(panel, equal_weights(panel), n_subperiods=3)
    subs = out["subperiod_estimates"]
    assert len(subs) == 3
    assert sum(s["n_observations"] for s in subs) == 1200
    vals = [s["effective_bets_diversification"] for s in subs]
    assert len(set(vals)) == 3, "sub-period estimates are identical: not recomputed"
    assert out["subperiod_range"] == [min(vals), max(vals)]


def test_subperiods_are_withheld_when_too_short_to_estimate() -> None:
    """With 3 splits this branch is unreachable: the headline gate (n >= 10p)
    already guarantees each third clears 3p. It becomes reachable when the
    caller asks for more splits than the window can support, which is exactly
    when a stability claim would be measuring estimation noise."""
    panel = independent_panel(n=110, p=10)  # n/p=11 ok; each fifth is 22 < 3p
    out = C.effective_bets(panel, equal_weights(panel), n_subperiods=5)
    assert out["headline_supported"] is True
    assert out["subperiod_estimates"] == []
    assert out["subperiod_range"] is None
    assert "n_sub >=" in out["subperiod_note"]


# --------------------------------------------------------------------------
# 6. NO SILENT EXCLUSION
# --------------------------------------------------------------------------


def test_missing_ticker_is_excluded_with_a_reason_and_coverage() -> None:
    panel = factor_panel(n=500, p=4)
    w = {**equal_weights(panel), "DELISTED.NS": 0.20}
    out = C.effective_bets(panel, w)
    exc = [e for e in out["excluded"] if e["ticker"] == "DELISTED.NS"]
    assert len(exc) == 1
    assert exc[0]["reason"] == "no_column_in_panel"
    assert exc[0]["weight"] == 0.20
    assert out["n_assets"] == 4
    assert out["weight_of_portfolio_analysed"] == pytest.approx(1.0 / 1.2, abs=1e-6)


def test_short_history_holding_is_excluded_not_silently_padded() -> None:
    panel = factor_panel(n=500, p=4)
    panel["NEWLISTING"] = np.nan
    panel.iloc[-30:, -1] = 0.01
    w = {c: 0.2 for c in panel.columns}
    out = C.effective_bets(panel, w)
    exc = [e for e in out["excluded"] if e["ticker"] == "NEWLISTING"]
    assert exc and exc[0]["reason"] == "insufficient_history"
    assert "30 observations" in exc[0]["detail"]
    assert f"{C.MIN_OBS_PER_ASSET}" in exc[0]["detail"]
    assert "NEWLISTING" not in out["assets_analysed"]


def test_constant_series_is_excluded() -> None:
    panel = factor_panel(n=400, p=3)
    panel["SUSPENDED"] = 0.0
    w = {c: 0.25 for c in panel.columns}
    out = C.risk_contributions(panel, w)
    exc = [e for e in out["excluded"] if e["ticker"] == "SUSPENDED"]
    assert exc and exc[0]["reason"] == "zero_variance"
    assert out["n_assets"] == 3


def test_partial_overlap_reports_rows_dropped() -> None:
    panel = factor_panel(n=400, p=3)
    panel.iloc[:50, 1] = np.nan
    out = C.risk_contributions(panel, equal_weights(panel))
    assert out["rows_dropped_for_alignment"] == 50
    assert out["n_observations"] == 350


def test_zero_weight_holding_is_recorded_as_excluded() -> None:
    panel = factor_panel(n=400, p=3)
    w = {**equal_weights(panel), "A0": 0.0}
    out = C.effective_bets(panel, w)
    assert any(e["ticker"] == "A0" and e["reason"] == "non_positive_weight" for e in out["excluded"])
    assert out["n_assets"] == 2


def test_weights_are_renormalised_and_the_original_sum_reported() -> None:
    panel = factor_panel(n=400, p=4)
    w = {c: 25.0 for c in panel.columns}  # percentages, or rupees
    out = C.risk_contributions(panel, w)
    assert out["input_weights_summed_to"] == pytest.approx(100.0)
    assert sum(r["weight"] for r in out["contributions"]) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# 7. INPUT GUARDS
# --------------------------------------------------------------------------


def test_prices_passed_as_returns_is_refused_not_computed() -> None:
    """The single most likely caller error, and the one that would produce a
    confident wrong answer rather than a crash."""
    prices = _panel(
        np.cumprod(1 + np.random.default_rng(SEED).normal(0, 0.01, size=(400, 4)), axis=0) * 100,
        ["A0", "A1", "A2", "A3"],
    )
    with pytest.raises(ValueError, match="looks like PRICES"):
        C.effective_bets(prices, equal_weights(prices))


def test_returns_from_prices_round_trips() -> None:
    rng = np.random.default_rng(SEED)
    true_rets = rng.normal(0, 0.01, size=(300, 3))
    prices = _panel(100 * np.cumprod(1 + true_rets, axis=0), ["A", "B", "C"])
    got = C.returns_from_prices(prices)
    assert len(got) == 299
    np.testing.assert_allclose(got.to_numpy(), true_rets[1:], atol=1e-12)


def test_empty_and_malformed_inputs_raise_clearly() -> None:
    panel = factor_panel(n=300, p=3)
    with pytest.raises(ValueError, match="weights is empty"):
        C.effective_bets(panel, {})
    with pytest.raises(TypeError, match="mapping"):
        C.effective_bets(panel, ["A0", "A1"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DataFrame"):
        C.effective_bets([[0.1, 0.2]], {"A0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="panel is empty"):
        C.effective_bets(pd.DataFrame(), {"A0": 1.0})


def test_no_surviving_holding_raises_rather_than_returning_a_number() -> None:
    panel = factor_panel(n=300, p=3)
    with pytest.raises(ValueError, match="no holding survived"):
        C.effective_bets(panel, {"NOPE.NS": 0.5, "ALSONOPE.NS": 0.5})


def test_panel_object_with_returns_attribute_is_accepted() -> None:
    panel = factor_panel(n=400, p=3)

    class Loaded:
        returns = panel

    out = C.risk_contributions(Loaded(), equal_weights(panel))
    assert out["n_assets"] == 3


# --------------------------------------------------------------------------
# 8. THE REGULATORY LINE
# --------------------------------------------------------------------------


def test_rendered_report_contains_no_advice_language() -> None:
    """The product may state facts about holdings; personalised investment
    advice is regulated and unlicensed here. This test is the line."""
    panel = factor_panel(n=900, p=6, beta=0.7)
    text = C.format_report(panel, equal_weights(panel))
    found = C._advice_words_in(text)
    assert found == [], f"advice language in rendered report: {found}"


def test_rendered_report_contains_no_forward_looking_claim() -> None:
    panel = factor_panel(n=900, p=6, beta=0.7)
    text = C.format_report(panel, equal_weights(panel)).lower()
    for phrase in ("will ", "expect", "predict", "going forward", "outlook", "likely"):
        assert phrase not in text, f"forward-looking phrase in report: {phrase!r}"
    # "forecast" may appear only inside the disclaimer that denies making one
    assert text.count("forecast") == text.count("not forecasts") == 1


def test_rendered_report_states_its_window_and_sample_size() -> None:
    panel = factor_panel(n=900, p=6)
    text = C.format_report(panel, equal_weights(panel))
    assert "n=900 trading days" in text
    assert "2023-01-02" in text
    assert "p=6 holdings" in text


def test_report_never_prints_full_coverage_while_excluding_a_holding() -> None:
    """0.999994 rounds to '100.0%'. Printed under a list of exclusions that is
    a lie, and it is the exact shape of error this module exists to avoid."""
    panel = factor_panel(n=500, p=5)
    w = {**{c: 0.2 for c in panel.columns}, "TINY.NS": 1e-6}
    text = C.format_report(panel, w)
    assert "Excluded from every figure above" in text
    assert "TINY.NS" in text
    cover = [ln for ln in text.splitlines() if "cover" in ln][0]
    assert "100.0%" not in cover and "100%" not in cover
    assert ">99.9%" in cover


def test_report_shows_the_refusal_when_the_window_is_too_short() -> None:
    panel = independent_panel(n=40, p=10)
    text = C.format_report(panel, equal_weights(panel), min_obs=20)
    assert "not shown" in text
    assert "Why:" in text


# --------------------------------------------------------------------------
# 9. DETERMINISM
# --------------------------------------------------------------------------


def test_results_are_deterministic_and_column_order_invariant() -> None:
    panel = factor_panel(n=800, p=6, beta=0.6)
    rng = np.random.default_rng(SEED)
    w = dict(zip(panel.columns, rng.dirichlet(np.ones(6))))

    a = C.effective_bets(panel, w)
    b = C.effective_bets(panel, w)
    assert a["effective_bets_diversification"] == b["effective_bets_diversification"]

    shuffled = panel[list(reversed(list(panel.columns)))]
    c = C.effective_bets(shuffled, w)
    assert c["effective_bets_diversification"] == pytest.approx(a["effective_bets_diversification"], abs=1e-9)

    ra = {r["ticker"]: r["risk_share"] for r in C.risk_contributions(panel, w)["contributions"]}
    rc = {r["ticker"]: r["risk_share"] for r in C.risk_contributions(shuffled, w)["contributions"]}
    for k in ra:
        assert ra[k] == pytest.approx(rc[k], abs=1e-9)


def test_headline_is_scale_invariant_in_weights() -> None:
    """Weights in rupees, in percent, or as fractions must give one answer."""
    panel = factor_panel(n=800, p=4)
    base = {"A0": 0.4, "A1": 0.3, "A2": 0.2, "A3": 0.1}
    for mult in (1.0, 100.0, 47321.55):
        out = C.effective_bets(panel, {k: v * mult for k, v in base.items()})
        assert out["effective_bets_diversification"] == pytest.approx(
            C.effective_bets(panel, base)["effective_bets_diversification"], abs=1e-9
        )


def test_headline_is_invariant_to_a_units_change_in_returns() -> None:
    """Scaling every return by a constant scales portfolio and asset vols
    equally, so a *ratio* of volatilities must not move."""
    panel = factor_panel(n=800, p=5, beta=0.6)
    w = equal_weights(panel)
    a = C.effective_bets(panel, w)["effective_bets_diversification"]
    b = C.effective_bets(panel * 3.7, w)["effective_bets_diversification"]
    assert a == pytest.approx(b, abs=1e-9)
    assert not math.isnan(a)


# --------------------------------------------------------------------------
# 10. GAPS FOUND BY MUTATION TESTING
#     Each test below exists because a deliberate defect survived the suite
#     as originally written. They are kept separate so the reason is legible.
# --------------------------------------------------------------------------


def test_headline_numerator_is_weighted_not_a_plain_average() -> None:
    """Mutation M05: ``sum_i w_i sigma_i`` replaced by ``mean(sigma_i)``.

    Every calibration test above uses equal weights, where the two are
    identical, so all of them survived the defect. This uses deliberately
    unequal weights over deliberately unequal volatilities, and checks the
    headline against DR^2 computed by hand from the definition.
    """
    rng = np.random.default_rng(SEED)
    n = 4000
    vols = np.array([0.004, 0.010, 0.030, 0.060])
    panel = _panel(rng.normal(0, 1, size=(n, 4)) * vols, ["A0", "A1", "A2", "A3"])
    weights = {"A0": 0.70, "A1": 0.20, "A2": 0.07, "A3": 0.03}

    out = C.effective_bets(panel, weights)

    cov = np.cov(panel.to_numpy(), rowvar=False, ddof=1)
    wv = np.array([weights[c] for c in panel.columns])
    sd = np.sqrt(np.diag(cov))
    expected = (float(wv @ sd) / float(np.sqrt(wv @ cov @ wv))) ** 2
    assert out["effective_bets_diversification"] == pytest.approx(expected, abs=1e-4)

    # And the plain-average version is a materially different number, so the
    # assertion above genuinely discriminates between them.
    wrong = (float(sd.mean()) / float(np.sqrt(wv @ cov @ wv))) ** 2
    assert abs(wrong - expected) > 1.0


def test_euler_sum_guard_rejects_a_vector_that_does_not_sum_to_one() -> None:
    """Mutation M20: the sum-to-one assertion removed, and nothing went red.

    Nothing could. The identity is homogeneous of degree zero in the weights,
    so it holds for every possible input and no pasted portfolio will ever
    trip the guard. The guard is there to catch a future edit to the
    decomposition, which means the only honest test is to call it directly.
    """
    assert C._check_euler_sum(np.array([0.6, 0.3, 0.1])) == pytest.approx(1.0)
    for bad in ([0.5, 0.4], [0.6, 0.5], [1.0, 1e-8], [np.nan, 0.5]):
        with pytest.raises(AssertionError, match="Euler decomposition is broken"):
            C._check_euler_sum(np.array(bad, dtype=float))


def test_risk_contributions_actually_invokes_the_euler_guard(monkeypatch) -> None:
    """The companion to the test above: the guard's logic being correct is
    worthless if ``risk_contributions`` stops calling it. This fails if the
    call is deleted."""
    calls: list[float] = []
    real = C._check_euler_sum

    def spy(contrib):
        calls.append(float(np.asarray(contrib).sum()))
        return real(contrib)

    monkeypatch.setattr(C, "_check_euler_sum", spy)
    panel = factor_panel(n=600, p=4)
    C.risk_contributions(panel, equal_weights(panel))
    assert len(calls) == 1, "risk_contributions did not call the Euler guard"
    assert calls[0] == pytest.approx(1.0, abs=C._SUM_TOL)


def test_effective_n_by_risk_is_an_inverse_hhi_not_a_head_count() -> None:
    """Mutation M23: ``effective_n_by_risk`` returning ``len(shares)``.

    For unequal risk shares the effective count must be strictly below the
    number of holdings, and must equal p only when the shares are equal.
    """
    panel = factor_panel(n=1200, p=6, beta=0.5, idio=0.02)
    lumpy = dict(zip(panel.columns, [0.50, 0.20, 0.12, 0.08, 0.06, 0.04]))
    out = C.variance_concentration(panel, lumpy)
    assert out["effective_n_by_risk"] < 5.0
    assert out["effective_n_by_risk"] < out["n_assets"]
    assert out["effective_n_by_risk"] == pytest.approx(1.0 / out["hhi_risk"], abs=1e-3)

    even = C.variance_concentration(identical_panel(n=800, p=4), {f"A{i}": 0.25 for i in range(4)})
    assert even["effective_n_by_risk"] == pytest.approx(4.0, abs=1e-6)


def test_window_dates_describe_the_rows_actually_used() -> None:
    """Mutation M24: ``window_start`` taken from the raw panel rather than the
    aligned complete-case window.

    Claiming a window that starts before the first row used is a false
    provenance stamp - the sort this whole module exists to prevent - and it
    only shows up when rows are dropped at the front.
    """
    panel = factor_panel(n=400, p=3)
    panel.iloc[:50, 1] = np.nan
    out = C.risk_contributions(panel, equal_weights(panel))

    assert out["rows_dropped_for_alignment"] == 50
    assert out["n_observations"] == 350
    assert out["window_start"] == panel.index[50].date().isoformat()
    assert out["window_start"] != panel.index[0].date().isoformat()
    assert out["window_end"] == panel.index[-1].date().isoformat()
    # the report must not advertise the wider window either
    text = C.format_report(panel, equal_weights(panel))
    assert panel.index[0].date().isoformat() not in text
