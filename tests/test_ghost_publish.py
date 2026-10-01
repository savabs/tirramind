"""Tests for `agent.ghost.publish` — the scoreboard that cannot hide its denominator.

These are not smoke tests. Each one names a specific way a chain-discovery page
could lie, and asserts that this module makes that lie impossible or visible:

  * a funnel that does not reconcile          -> ScoreboardError
  * a chain tested below its declared floors  -> ScoreboardError
  * survivors on a constant p-value series    -> ScoreboardError (F-18)
  * a page of survivors with no graveyard     -> unrenderable by construction
  * a null result rendered as a failure       -> asserted to read as a finding
  * a minimum detectable effect that does not
    agree with the substrate's power curve    -> asserted against power_estimate
"""

from __future__ import annotations

import math
import re
from dataclasses import fields

import pytest

from agent.ghost.publish import (
    ChainResult,
    FilterStep,
    Scoreboard,
    ScoreboardError,
    SeriesCheck,
    UntestableReason,
    mde_at_power,
    render_scoreboard,
    series_check,
    summarise,
)
from agent.verify.stats import power_estimate

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def chain(
    chain_id: str = "c1",
    *,
    p: float = 0.2,
    p_adj: float = 0.9,
    rejected: bool = False,
    n: int = 120,
    clusters: int = 60,
    power: float = 0.061,
    effect: float = 0.0035,
    sigma: float = 0.02,
    hops: int = 2,
    independent: int = 1,
    name: str | None = None,
    falsifier: str = "the next 20 firings failing to show a positive 5-step effect",
    notes: tuple[str, ...] = (),
    market: str = "instrument daily close",
    scaffold_only: bool = False,
    mechanism: str = "country/sovereign_debt -> instrument.close",
) -> ChainResult:
    return ChainResult(
        chain_id=chain_id,
        name=name or f"chain {chain_id}",
        route=("Russia", "[produced_in/eia]", "WTI Crude Oil") if hops else (),
        hops=hops,
        independent_sources=independent,
        evidence_sources=("gdelt",) if independent else (),
        lag_steps=5,
        lag_label="trading days",
        market=market,
        mechanism=mechanism,
        n_events=n,
        n_clusters=clusters,
        sigma=sigma,
        power=power,
        p_uncorrected=p,
        p_adjusted=p_adj,
        rejected=rejected,
        effect=effect,
        event_mean=effect - 0.0001,
        baseline=-0.0001,
        ci_low=-0.002,
        ci_high=0.009,
        falsifier=falsifier,
        route_is_scaffold_only=scaffold_only,
        notes=notes,
    )


def board(
    *,
    tested: tuple[ChainResult, ...] = (),
    untestable: tuple[UntestableReason, ...] | None = None,
    enumerated: int | None = None,
    min_events: int = 30,
    min_clusters: int = 20,
    alpha: float = 0.05,
    filters: tuple[FilterStep, ...] | None = None,
    checks: tuple[SeriesCheck, ...] = (),
) -> Scoreboard:
    untestable = untestable if untestable is not None else (UntestableReason("too few events", 100),)
    n_untestable = sum(u.count for u in untestable)
    enumerated = enumerated if enumerated is not None else n_untestable + len(tested)
    if filters is None:
        # a default funnel, but only when it is itself arithmetically legal: these
        # tests deliberately build boards with a wrong `enumerated`, and the
        # FilterStep guard must not pre-empt the Scoreboard guard under test.
        filters = (
            (FilterStep("direct link between entity types", enumerated, n_untestable, "no link"),)
            if enumerated >= n_untestable >= 0
            else ()
        )
    return Scoreboard(
        title="Ghost chain discovery — test",
        generated_at="2026-09-30T00:00:00+00:00",
        db_path=".tirra_pipeline/pipeline.db",
        as_of_label="none (the live graph)",
        alpha=alpha,
        min_events=min_events,
        min_clusters=min_clusters,
        enumerated=enumerated,
        filters=filters,
        untestable=untestable,
        tested=tested,
        grid={"z": "1.5, 2, 2.5, 3", "product": "2 x 2 = 4"},
        caveats=("one market, one window",),
        provenance={"engine": "agent.verify.study"},
        checks=checks,
    )


# ---------------------------------------------------------------------------
# the funnel must reconcile
# ---------------------------------------------------------------------------


def test_funnel_reconciles_when_counts_add_up():
    b = board(tested=(chain(),), untestable=(UntestableReason("too few events", 9),))
    assert b.enumerated == 10
    assert b.untestable_total == 9
    assert len(b.tested) == 1


@pytest.mark.parametrize("enumerated", [9, 11, 0, 1000])
def test_funnel_that_does_not_reconcile_is_refused(enumerated):
    with pytest.raises(ScoreboardError, match="does not reconcile"):
        board(
            tested=(chain(),),
            untestable=(UntestableReason("too few events", 9),),
            enumerated=enumerated,
        )


def test_unaccounted_count_is_named_in_the_error():
    with pytest.raises(ScoreboardError) as exc:
        board(tested=(chain(),), untestable=(UntestableReason("x", 9),), enumerated=40)
    assert "30 chains are unaccounted for" in str(exc.value)


def test_enumerated_may_not_be_negative():
    with pytest.raises(ScoreboardError, match="enumerated must be >= 0"):
        board(enumerated=-1, untestable=())


def test_empty_run_is_legal_and_reconciles():
    b = board(tested=(), untestable=(), enumerated=0, filters=())
    assert b.enumerated == 0 and b.untestable_total == 0 and b.tested == ()


# ---------------------------------------------------------------------------
# the floors must be honoured
# ---------------------------------------------------------------------------


def test_chain_below_the_event_floor_cannot_be_in_tested():
    with pytest.raises(ScoreboardError, match="below the declared floors"):
        board(tested=(chain(n=29, clusters=25),), min_events=30, min_clusters=20)


def test_chain_below_the_cluster_floor_cannot_be_in_tested():
    with pytest.raises(ScoreboardError, match="below the declared floors"):
        board(tested=(chain(n=310, clusters=16),), min_events=30, min_clusters=20)


def test_a_chain_exactly_on_the_floors_is_accepted():
    b = board(tested=(chain(n=30, clusters=20),), min_events=30, min_clusters=20)
    assert b.tested[0].n_events == 30


def test_more_clusters_than_events_is_impossible():
    with pytest.raises(ScoreboardError, match="distinct event dates from"):
        chain(n=10, clusters=11)


# ---------------------------------------------------------------------------
# p-value integrity
# ---------------------------------------------------------------------------


def test_adjusted_p_below_uncorrected_p_is_refused():
    with pytest.raises(ScoreboardError, match="never shrinks a p"):
        chain(p=0.40, p_adj=0.20)


def test_adjusted_p_equal_to_uncorrected_is_allowed():
    c = chain(p=0.04, p_adj=0.04)
    assert c.p_adjusted == pytest.approx(c.p_uncorrected)


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_non_p_values_are_refused(bad):
    with pytest.raises(ScoreboardError, match="not a p-value"):
        chain(p=bad, p_adj=1.0)


def test_rejected_chain_with_adjusted_p_above_alpha_is_refused():
    with pytest.raises(ScoreboardError, match="marked rejected"):
        board(tested=(chain(p=0.001, p_adj=0.30, rejected=True),), alpha=0.05)


def test_rejected_chain_at_exactly_alpha_is_accepted():
    b = board(tested=(chain(p=0.01, p_adj=0.05, rejected=True),), alpha=0.05)
    assert len(b.survivors) == 1


def test_duplicate_chain_ids_would_double_count_the_family():
    with pytest.raises(ScoreboardError, match="duplicate chain_id"):
        board(tested=(chain("dup"), chain("dup", p=0.3)), untestable=(UntestableReason("x", 8),))


def test_power_outside_zero_one_is_refused():
    with pytest.raises(ScoreboardError, match="power="):
        chain(power=1.4)


# ---------------------------------------------------------------------------
# a falsifier is mandatory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_a_chain_without_a_falsifier_is_not_publishable(empty):
    with pytest.raises(ScoreboardError, match="no falsifier"):
        chain(falsifier=empty)


@pytest.mark.parametrize(("cid", "nm"), [("", "a name"), ("c1", "")])
def test_a_chain_without_an_id_or_a_name_is_refused(cid, nm):
    kw = {f.name: getattr(chain(), f.name) for f in fields(ChainResult)}
    kw["chain_id"], kw["name"] = cid, nm
    with pytest.raises(ScoreboardError, match="needs a chain_id and a name"):
        ChainResult(**kw)


# ---------------------------------------------------------------------------
# F-18: a constant series is a defect, not a discovery
# ---------------------------------------------------------------------------


def test_series_check_flags_a_constant_series():
    c = series_check("p", [0.5] * 20)
    assert c.degenerate is True
    assert c.distinct == 1 and c.n == 20
    assert "DEGENERATE" in c.line()


def test_series_check_accepts_a_varying_series():
    c = series_check("p", [0.1, 0.2, 0.3])
    assert c.degenerate is False
    assert (c.minimum, c.median, c.maximum) == (0.1, 0.2, 0.3)
    assert "DEGENERATE" not in c.line()


def test_series_check_median_of_an_even_series_is_interpolated():
    assert series_check("x", [1.0, 2.0, 3.0, 4.0]).median == pytest.approx(2.5)


def test_series_check_drops_non_finite_values_visibly():
    c = series_check("p", [0.1, float("nan"), float("inf"), 0.3])
    assert c.n == 2, "the two unusable values must not be counted as computed"
    assert c.minimum == 0.1 and c.maximum == 0.3


def test_series_check_of_nothing_says_so():
    c = series_check("p", [])
    assert c.n == 0 and c.distinct == 0 and c.minimum is None
    assert "no values computed" in c.line()


def test_single_value_series_is_not_called_degenerate():
    assert series_check("p", [0.4]).degenerate is False


def test_survivors_on_a_constant_p_series_are_refused():
    tested = tuple(chain(f"c{i}", p=0.001, p_adj=0.001, rejected=True) for i in range(5))
    with pytest.raises(ScoreboardError, match="broken pipeline, not a"):
        board(tested=tested, untestable=(UntestableReason("x", 5),))


def test_constant_p_series_without_survivors_is_rendered_not_refused():
    tested = tuple(chain(f"c{i}", p=0.5, p_adj=0.99) for i in range(5))
    b = board(tested=tested, untestable=(UntestableReason("x", 5),))
    out = render_scoreboard(b)
    assert "DEGENERATE" in out
    assert "defect claim about this pipeline" in out


# ---------------------------------------------------------------------------
# filter arithmetic
# ---------------------------------------------------------------------------


def test_filter_step_kept_is_examined_minus_dropped():
    f = FilterStep("x", examined=100, dropped=40, reason="r")
    assert f.kept == 60


def test_filter_step_cannot_drop_more_than_it_examined():
    with pytest.raises(ScoreboardError, match="dropped 101 of 100"):
        FilterStep("x", examined=100, dropped=101, reason="r")


def test_filter_step_rejects_negative_counts():
    with pytest.raises(ScoreboardError, match="negative count"):
        FilterStep("x", examined=-1, dropped=0, reason="r")


def test_untestable_reason_rejects_negative_count():
    with pytest.raises(ScoreboardError, match="has count -1"):
        UntestableReason("x", -1)


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------


def test_nominal_survivors_counts_uncorrected_significance():
    tested = (
        chain("a", p=0.001, p_adj=0.05, rejected=True),
        chain("b", p=0.03, p_adj=0.6),
        chain("c", p=0.05, p_adj=0.7),
        chain("d", p=0.06, p_adj=0.8),
    )
    b = board(tested=tested, untestable=(UntestableReason("x", 6),))
    assert b.nominal_survivors == 3, "0.05 is inclusive; 0.06 is not nominal"
    assert len(b.survivors) == 1


def test_survivors_are_ordered_by_adjusted_p_and_kills_by_uncorrected_p():
    tested = (
        chain("s2", p=0.002, p_adj=0.04, rejected=True),
        chain("s1", p=0.001, p_adj=0.01, rejected=True),
        chain("k2", p=0.40, p_adj=0.9),
        chain("k1", p=0.10, p_adj=0.8),
    )
    b = board(tested=tested, untestable=(UntestableReason("x", 6),))
    assert [c.chain_id for c in b.survivors] == ["s1", "s2"]
    assert [c.chain_id for c in b.kills] == ["k1", "k2"]


def test_median_power_and_markets_are_computed_from_the_family():
    tested = (
        chain("a", power=0.05, market="instrument daily close"),
        chain("b", power=0.10, market="sovereign bond yield"),
        chain("c", power=0.30, market="instrument daily close"),
    )
    b = board(tested=tested, untestable=(UntestableReason("x", 7),))
    assert b.median_power == pytest.approx(0.10)
    assert b.markets == ("instrument daily close", "sovereign bond yield")
    assert b.n_adequately_powered == 0


def test_adequately_powered_counts_at_eighty_percent():
    tested = (chain("a", power=0.80), chain("b", power=0.79))
    b = board(tested=tested, untestable=(UntestableReason("x", 8),))
    assert b.n_adequately_powered == 1
    assert tested[0].underpowered is False and tested[1].underpowered is True


def test_median_power_of_an_empty_family_is_none():
    assert board(tested=(), untestable=(UntestableReason("x", 4),)).median_power is None


# ---------------------------------------------------------------------------
# minimum detectable effect, checked against the substrate's own power curve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "sigma"), [(30, 0.02), (120, 0.015), (1000, 0.04), (70, 0.01)])
def test_mde_at_eighty_percent_round_trips_through_power_estimate(n, sigma):
    mde = mde_at_power(n, sigma)
    got = power_estimate(n_events=n, effect_size=mde, sigma=sigma, alpha=0.05)
    assert got == pytest.approx(0.80, abs=2e-3), (
        "the MDE must invert agent.verify.stats.power_estimate, or the page and the "
        "substrate disagree about what was detectable"
    )


def test_mde_shrinks_with_n_and_grows_with_sigma():
    assert mde_at_power(400, 0.02) < mde_at_power(100, 0.02)
    assert mde_at_power(100, 0.04) > mde_at_power(100, 0.02)
    assert mde_at_power(100, 0.02) == pytest.approx(2 * mde_at_power(400, 0.02))


def test_mde_at_a_non_default_power_uses_the_quantile_path():
    got = power_estimate(n_events=100, effect_size=mde_at_power(100, 0.02, power=0.5), sigma=0.02)
    assert got == pytest.approx(0.50, abs=5e-3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0, "sigma": 0.02}, "n >= 1"),
        ({"n": 10, "sigma": -1.0}, "sigma >= 0"),
        ({"n": 10, "sigma": float("nan")}, "sigma >= 0"),
        ({"n": 10, "sigma": 0.02, "power": 0.0}, "power in"),
        ({"n": 10, "sigma": 0.02, "alpha": 1.0}, "alpha in"),
    ],
)
def test_mde_rejects_nonsense(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mde_at_power(**kwargs)


def test_chain_mde_is_none_when_sigma_is_unusable():
    assert chain(sigma=float("nan")).mde_80 is None
    assert chain(sigma=0.0).mde_80 is None
    assert chain(sigma=0.02, n=100).mde_80 == pytest.approx(mde_at_power(100, 0.02))


# ---------------------------------------------------------------------------
# rendering: the denominator leads, and the graveyard is not optional
# ---------------------------------------------------------------------------


def test_render_leads_with_the_denominator_before_any_survivor():
    tested = (chain("s", p=0.0001, p_adj=0.004, rejected=True),) + tuple(
        chain(f"k{i}", p=0.3 + i / 100, p_adj=0.9) for i in range(5)
    )
    b = board(tested=tested, untestable=(UntestableReason("too few events", 994),))
    out = render_scoreboard(b)
    assert out.index("CHAINS ENUMERATED") < out.index("## Survivors")
    assert out.index("TESTED") < out.index("## Survivors")
    assert "1,000" in out, "the enumerated count must be printed"
    assert "SURVIVED BH @ 0.05" in out


def test_render_always_includes_the_graveyard_and_the_limits():
    tested = (chain("s", p=0.0001, p_adj=0.004, rejected=True), chain("k", p=0.3, p_adj=0.9))
    b = board(tested=tested, untestable=(UntestableReason("x", 8),))
    out = render_scoreboard(b)
    assert "## The graveyard" in out
    assert "## Where these claims stop applying" in out
    assert "## Provenance" in out
    assert "## How the space narrowed" in out


def test_render_reports_how_many_would_have_passed_uncorrected():
    tested = (
        chain("s", p=0.0001, p_adj=0.004, rejected=True),
        chain("k1", p=0.01, p_adj=0.9),
        chain("k2", p=0.02, p_adj=0.9),
        chain("k3", p=0.60, p_adj=0.99),
    )
    b = board(tested=tested, untestable=(UntestableReason("x", 6),))
    out = render_scoreboard(b)
    assert "2 of these 3 killed chains" in out
    assert "uncorrected p at or below 0.05" in out


def test_survivor_rows_carry_n_clusters_power_lag_and_falsifier():
    b = board(
        tested=(chain("s", p=0.0001, p_adj=0.004, rejected=True, n=310, clusters=64, power=0.62),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "310" in out and "64" in out
    assert "62.0%" in out
    assert "trading days" in out
    assert "what would falsify it" in out
    assert "independent time-varying sources" in out
    assert "upper bound" in out


def test_underpowered_survivors_are_labelled_as_such():
    b = board(
        tested=(chain("s", p=0.0001, p_adj=0.004, rejected=True, power=0.056, clusters=64),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "UNDERPOWERED" in out
    assert "1 of 1 survivors are underpowered" in out
    assert "chance excursion" in out


def test_thin_cluster_survivors_are_labelled_as_such():
    b = board(
        tested=(chain("s", p=0.0001, p_adj=0.004, rejected=True, clusters=25),),
        untestable=(UntestableReason("x", 9),),
        min_clusters=20,
    )
    out = render_scoreboard(b)
    assert "fewer than 30 distinct" in out


def test_graveyard_truncates_but_states_the_exact_count():
    tested = tuple(chain(f"k{i}", p=min(0.99, 0.01 + i / 200), p_adj=0.99) for i in range(60))
    b = board(tested=tested, untestable=(UntestableReason("x", 40),))
    out = render_scoreboard(b)
    assert "60 tested and killed" in out
    assert "20 further killed" in out


def test_render_escapes_pipes_so_a_chain_name_cannot_break_a_table():
    b = board(
        tested=(chain("k", name="a|b -> c|d", p=0.4, p_adj=0.9),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "a\\|b" in out
    assert "| a|b" not in out


def test_render_rejects_a_non_scoreboard():
    with pytest.raises(TypeError, match="expects a Scoreboard"):
        render_scoreboard({"enumerated": 10})


def test_zero_hop_chains_are_not_described_as_direct():
    b = board(
        tested=(chain("s", hops=0, independent=0, p=0.001, p_adj=0.01, rejected=True),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "no mechanism route computed" in out
    assert "not a claim of directness" in out


def test_single_market_is_called_out_by_name():
    b = board(tested=(chain("k", p=0.4, p_adj=0.9),), untestable=(UntestableReason("x", 9),))
    out = render_scoreboard(b)
    assert "One market." in out
    assert "instrument daily close" in out


def test_multiple_markets_are_not_pooled():
    tested = (chain("a", p=0.4, p_adj=0.9), chain("b", p=0.5, p_adj=0.9, market="protocol TVL"))
    b = board(tested=tested, untestable=(UntestableReason("x", 8),))
    out = render_scoreboard(b)
    assert "2 target markets" in out
    assert "not pooled" in out


def test_untestable_reasons_are_itemised_with_counts_and_a_total():
    b = board(
        tested=(chain("k", p=0.4, p_adj=0.9),),
        untestable=(
            UntestableReason("no direct graph link", 12000),
            UntestableReason("fewer than 30 gradeable events", 800),
        ),
    )
    out = render_scoreboard(b)
    assert "12,000" in out and "800" in out
    assert "12,800" in out and "total untestable" in out
    assert "never tested, not because they disappointed" in out


def test_the_grid_is_printed_so_the_denominator_is_recomputable():
    b = board(tested=(chain("k", p=0.4, p_adj=0.9),), untestable=(UntestableReason("x", 9),))
    out = render_scoreboard(b)
    assert "grid.z" in out
    assert "recomputed by hand" in out


# ---------------------------------------------------------------------------
# the zero-survivor page must read as a finding — the likely branch
# ---------------------------------------------------------------------------


def test_zero_survivors_reads_as_a_bounded_finding_with_a_power_analysis():
    tested = tuple(chain(f"k{i}", p=0.02 + i / 500, p_adj=0.9, power=0.06, n=120, sigma=0.02) for i in range(40))
    b = board(tested=tested, untestable=(UntestableReason("too few events", 41168),))
    out = render_scoreboard(b)
    assert "## Result: no chain survived correction" in out
    assert "## Survivors" not in out
    # the finding, stated
    assert "0 survived" in out
    # the boundary of the claim
    assert "### What we could have detected" in out
    assert "median smallest effect detectable at 80% power" in out
    assert "median power across the tested family" in out
    assert "No tested chain reached 80% power." in out
    # the expected-by-chance arithmetic
    assert "expected to clear 0.05 by chance alone" in out
    # and the graveyard is still there
    assert "## The graveyard" in out
    assert "41,208" in out


def test_zero_survivors_states_the_uncorrected_count_that_was_not_published():
    tested = tuple(chain(f"k{i}", p=0.001 + i / 1000, p_adj=0.9) for i in range(30))
    b = board(tested=tested, untestable=(UntestableReason("x", 70),))
    out = render_scoreboard(b)
    n_nominal = b.nominal_survivors
    assert n_nominal == 30
    assert f"{n_nominal} of the 30 tested chains had an" in out


def test_nothing_testable_at_all_is_a_finding_about_the_graph():
    b = board(tested=(), untestable=(UntestableReason("fewer than 30 gradeable events", 14400),))
    out = render_scoreboard(b)
    assert "none was testable" in out
    assert "nothing to correct and nothing to report as a discovery" in out
    assert "about this graph's density, not about the markets" in out
    assert "14,400" in out
    assert "## The graveyard" in out


def test_adequately_powered_null_does_not_claim_nothing_was_powered():
    tested = tuple(chain(f"k{i}", p=0.2 + i / 100, p_adj=0.9, power=0.9) for i in range(10))
    b = board(tested=tested, untestable=(UntestableReason("x", 90),))
    out = render_scoreboard(b)
    assert "No tested chain reached 80% power." not in out
    assert "tested chains with power at or above 80%: **10 of 10**" in out


# ---------------------------------------------------------------------------
# formatting must never emit a blank or a bare NaN
# ---------------------------------------------------------------------------


def test_non_finite_effect_prints_not_computed_rather_than_nan():
    b = board(
        tested=(chain("k", p=0.4, p_adj=0.9, effect=float("nan"), sigma=float("nan")),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    # "provenance" contains the letters n-a-n, so scan for nan as a rendered VALUE
    assert not re.search(r"(?<![a-z])nan(?![a-z])", out, re.IGNORECASE), (
        "a non-finite number reached the page as a bare nan"
    )
    assert "n/a" in out, "the missing effect must render as an explicit n/a"


def test_render_output_is_markdown_and_ends_with_a_newline():
    b = board(tested=(chain("k", p=0.4, p_adj=0.9),), untestable=(UntestableReason("x", 9),))
    out = render_scoreboard(b)
    assert out.startswith("# Ghost chain discovery")
    assert out.endswith("\n")
    assert "```" in out


def test_summarise_is_a_terminal_view_with_the_same_counts():
    b = board(
        tested=(chain("s", p=0.001, p_adj=0.01, rejected=True), chain("k", p=0.04, p_adj=0.9)),
        untestable=(UntestableReason("too few events", 998),),
    )
    text = summarise(b)
    assert "enumerated   1,000" in text
    assert "tested       2" in text
    assert "survived     1" in text
    assert "nominal      2" in text


def test_headline_block_survives_being_read_as_plain_text():
    b = board(tested=(chain("k", p=0.4, p_adj=0.9),), untestable=(UntestableReason("x", 41207),))
    out = render_scoreboard(b)
    block = out.split("```")[1]
    for label in ("CHAINS ENUMERATED", "UNTESTABLE", "TESTED", "SURVIVED BH", "MEDIAN POWER"):
        assert label in block
    assert "41,208" in block
    assert "<- the denominator" in block
    assert "<- the BH family size, m" in block


def test_alpha_outside_zero_one_is_refused():
    with pytest.raises(ScoreboardError, match="alpha must be in"):
        board(alpha=0.0)


def test_negative_hop_or_source_counts_are_refused():
    with pytest.raises(ScoreboardError, match="negative hop/source count"):
        chain(hops=-1)
    with pytest.raises(ScoreboardError, match="negative hop/source count"):
        chain(independent=-1)


def test_series_check_objects_passed_in_are_the_ones_rendered():
    custom = (SeriesCheck("my series", n=3, distinct=3, minimum=0.1, median=0.2, maximum=0.3),)
    b = board(
        tested=(chain("k", p=0.4, p_adj=0.9),),
        untestable=(UntestableReason("x", 9),),
        checks=custom,
    )
    out = render_scoreboard(b)
    assert "my series: n=3, 3 distinct" in out


def test_math_import_is_used_for_finiteness_not_swallowed():
    """A NaN power must be rejected, not quietly compared as False."""
    with pytest.raises(ScoreboardError, match="power="):
        chain(power=float("nan"))
    assert math.isfinite(chain().power)


# ---------------------------------------------------------------------------
# clustering, p-of-zero and scaffold routes
# ---------------------------------------------------------------------------


def test_clustering_factor_is_events_per_distinct_date():
    c = chain(n=6156, clusters=76)
    assert c.clustering_factor == pytest.approx(6156 / 76)
    assert c.heavily_clustered is True


def test_one_event_per_date_is_not_heavily_clustered():
    assert chain(n=76, clusters=76).heavily_clustered is False
    assert chain(n=380, clusters=76).heavily_clustered is False  # exactly 5.0
    assert chain(n=381, clusters=76).heavily_clustered is True


def test_clustering_factor_is_none_without_clusters():
    b = board(tested=(), untestable=(UntestableReason("x", 1),))
    assert b.median_power is None
    # a chain with zero clusters can only exist below the floors, so build it bare
    from dataclasses import fields as dc_fields

    kw = {f.name: getattr(chain(), f.name) for f in dc_fields(ChainResult)}
    kw["n_events"], kw["n_clusters"] = 0, 0
    assert ChainResult(**kw).clustering_factor is None


def test_heavily_clustered_survivors_get_the_cftc_warning():
    b = board(
        tested=(chain("s", p=0.0001, p_adj=0.004, rejected=True, n=6156, clusters=76),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "heavily clustered" in out.lower()
    assert "0.102 to 0.294" in out, "the measured precedent must be cited, not asserted"
    assert "81.0 events per date" in out


def test_a_bootstrap_p_of_zero_is_never_printed_as_a_bare_zero():
    b = board(
        tested=(chain("s", p=0.0, p_adj=0.0, rejected=True),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "no bootstrap resample reached the observed value" in out
    assert "< 1/n_resamples" in out
    assert "**p** 0 uncorrected, 0 BH-adjusted" not in out


def test_the_ci_is_labelled_as_being_on_the_event_mean_not_the_effect():
    b = board(
        tested=(chain("s", p=0.001, p_adj=0.01, rejected=True),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "95% CI of the event mean" in out
    assert "NOT on the effect" in out
    assert "an event mean of" in out


def test_a_scaffold_only_top_route_is_disclosed():
    b = board(
        tested=(chain("s", p=0.001, p_adj=0.01, rejected=True, scaffold_only=True),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "pure static geography" in out
    assert "routes a signal without witnessing anything" in out


def test_an_evidenced_top_route_carries_no_scaffold_disclaimer():
    b = board(
        tested=(chain("s", p=0.001, p_adj=0.01, rejected=True, scaffold_only=False),),
        untestable=(UntestableReason("x", 9),),
    )
    assert "pure static geography" not in render_scoreboard(b)


def test_graveyard_table_carries_the_events_per_date_column():
    b = board(
        tested=(chain("k", p=0.4, p_adj=0.9, n=6156, clusters=76),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "ev/date" in out
    assert "| 81 |" in out


def test_correlated_candidate_caveat_is_always_present():
    b = board(tested=(chain("k", p=0.4, p_adj=0.9),), untestable=(UntestableReason("x", 9),))
    out = render_scoreboard(b)
    assert "Correlated candidates." in out
    assert "valid under positive dependence" in out


# ---------------------------------------------------------------------------
# gaps found by mutation testing: both of these escaped a first pass
# ---------------------------------------------------------------------------


def test_a_two_value_constant_series_is_degenerate():
    """Mutation M16: a `n >= 3` floor let a 2-chain constant p series through.

    Two tested chains returning the identical p is already a defect signal, and a
    family of two is exactly the size at which someone would be tempted to
    publish both.
    """
    c = series_check("p", [0.5, 0.5])
    assert c.n == 2 and c.distinct == 1
    assert c.degenerate is True
    assert "DEGENERATE" in c.line()


def test_a_two_value_varying_series_is_not_degenerate():
    assert series_check("p", [0.5, 0.6]).degenerate is False


def test_survivors_on_a_two_chain_constant_p_series_are_refused():
    tested = (
        chain("a", p=0.001, p_adj=0.002, rejected=True),
        chain("b", p=0.001, p_adj=0.002, rejected=True),
    )
    with pytest.raises(ScoreboardError, match="broken pipeline, not a"):
        board(tested=tested, untestable=(UntestableReason("x", 8),))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_every_formatter_refuses_to_print_a_bare_non_finite(bad):
    """Mutation M43: `_fmt` dropped its non-finite branch and nothing noticed.

    These three are the only functions that turn a number into a page cell, so
    they are the whole surface on which a NaN could reach a reader.
    """
    from agent.ghost.publish import _bp, _fmt, _fmt_p, _pct

    for fn in (_fmt, _pct, _bp, _fmt_p):
        rendered = fn(bad)
        assert not re.search(r"(?<![a-z])(nan|inf)(?![a-z])", rendered, re.IGNORECASE), (
            f"{fn.__name__} printed {rendered!r} for {bad}"
        )
        assert rendered in ("n/a", "not computed"), f"{fn.__name__} -> {rendered!r}"


def test_formatters_still_render_real_numbers():
    from agent.ghost.publish import _bp, _fmt, _fmt_p, _pct

    assert _fmt(0.123456) == "0.1235"
    assert _pct(0.061) == "6.1%"
    assert _bp(0.0035) == "+0.350%"
    assert _fmt_p(0.0123) == "0.0123"
    assert _fmt(None) == "n/a"


# ---------------------------------------------------------------------------
# rows are not discoveries: near-duplicate survivors must be collapsed
# ---------------------------------------------------------------------------


def test_survivor_rows_are_counted_as_mechanisms_not_rows():
    tested = (
        chain("a", p=0.001, p_adj=0.004, rejected=True, mechanism="curve -> fx"),
        chain("b", p=0.001, p_adj=0.005, rejected=True, mechanism="curve -> fx"),
        chain("c", p=0.001, p_adj=0.006, rejected=True, mechanism="curve -> fx"),
        chain("d", p=0.002, p_adj=0.007, rejected=True, mechanism="cftc -> platinum"),
    )
    b = board(tested=tested, untestable=(UntestableReason("x", 6),))
    assert b.distinct_survivor_mechanisms == 2
    out = render_scoreboard(b)
    assert "4 rows, 2 distinct mechanisms" in out
    assert "**4 surviving rows are 2 mechanisms.**" in out
    assert "Count mechanisms, not rows" in out
    assert "`curve -> fx` x3" in out


def test_one_mechanism_per_survivor_carries_no_collapse_banner():
    tested = (
        chain("a", p=0.001, p_adj=0.004, rejected=True, mechanism="m1"),
        chain("b", p=0.002, p_adj=0.005, rejected=True, mechanism="m2"),
    )
    b = board(tested=tested, untestable=(UntestableReason("x", 8),))
    assert b.distinct_survivor_mechanisms == 2
    out = render_scoreboard(b)
    assert "2 rows, 2 distinct mechanisms" in out
    assert "surviving rows are" not in out


def test_a_single_survivor_reads_as_one_mechanism_singular():
    b = board(
        tested=(chain("a", p=0.001, p_adj=0.004, rejected=True),),
        untestable=(UntestableReason("x", 9),),
    )
    assert "1 rows, 1 distinct mechanism," in render_scoreboard(b)


def test_distinct_tested_mechanisms_appears_on_the_null_page():
    tested = tuple(chain(f"k{i}", p=0.2 + i / 100, p_adj=0.9, mechanism=f"m{i % 3}") for i in range(9))
    b = board(tested=tested, untestable=(UntestableReason("x", 91),))
    assert b.distinct_tested_mechanisms == 3
    assert "3 distinct event-stream/target" in render_scoreboard(b)


def test_zero_independent_sources_is_stated_as_no_witness():
    b = board(
        tested=(chain("s", p=0.001, p_adj=0.004, rejected=True, independent=0),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "nothing in the graph independently witnesses this chain" in out
    assert "a static link is not a mechanism" in out
    assert "an upper bound; two pipelines" not in out


def test_one_independent_source_keeps_the_upper_bound_caveat():
    b = board(
        tested=(chain("s", p=0.001, p_adj=0.004, rejected=True, independent=1),),
        untestable=(UntestableReason("x", 9),),
    )
    out = render_scoreboard(b)
    assert "an upper bound; two pipelines" in out
    assert "nothing in the graph independently witnesses" not in out


def test_the_untestable_label_describes_every_reason_not_only_the_floors():
    """Most untestable chains never had a pair at all; the headline must not imply otherwise."""
    b = board(
        tested=(chain("k", p=0.4, p_adj=0.9),),
        untestable=(UntestableReason("no direct graph link", 4000),),
    )
    out = render_scoreboard(b)
    block = out.split("```")[1]
    assert "no gradeable pair at all" in block
    assert "itemised below" in block


def test_an_absent_as_of_bound_is_explained_rather_than_printed_raw():
    b = board(tested=(chain("k", p=0.4, p_adj=0.9),), untestable=(UntestableReason("x", 9),))
    out = render_scoreboard(b)
    assert "ending none" not in out
    assert "No as-of bound was applied" in out
    assert "most recent observation" in out


def test_an_explicit_as_of_bound_is_named():
    b = board(tested=(chain("k", p=0.4, p_adj=0.9),), untestable=(UntestableReason("x", 9),))
    b = Scoreboard(
        **{
            **{f.name: getattr(b, f.name) for f in __import__("dataclasses").fields(Scoreboard)},
            "as_of_label": "2025-06-30",
        }
    )
    out = render_scoreboard(b)
    assert "bounded at 2025-06-30" in out
    assert "No as-of bound was applied" not in out


def test_a_survivor_with_a_saturated_bootstrap_p_is_flagged_as_unrankable():
    """A p of exactly 0 is rejected at any alpha, so the correction cannot rank it."""
    b = board(
        tested=(
            chain("s", p=0.0, p_adj=0.0, rejected=True),
            chain("t", p=0.001, p_adj=0.01, rejected=True),
        ),
        untestable=(UntestableReason("x", 8),),
    )
    out = render_scoreboard(b)
    assert "1 of 2 survivors have a bootstrap p of exactly 0." in out
    assert "rejected at ANY alpha" in out
    assert "lowering alpha will never remove it" in out


def test_survivors_without_a_saturated_p_carry_no_such_banner():
    b = board(
        tested=(chain("s", p=0.001, p_adj=0.01, rejected=True),),
        untestable=(UntestableReason("x", 9),),
    )
    assert "bootstrap p of exactly 0." not in render_scoreboard(b)


def test_expected_by_chance_keeps_a_decimal_for_a_small_family():
    """A family of 8 expects 0.4 false positives; "roughly 0" would invert the argument."""
    tested = tuple(chain(f"k{i}", p=0.2 + i / 100, p_adj=0.9) for i in range(8))
    b = board(tested=tested, untestable=(UntestableReason("x", 82),))
    out = render_scoreboard(b)
    assert "0.4 are expected to clear 0.05 by chance alone" in out
    assert "roughly 0 are expected" not in out


def test_expected_by_chance_rounds_for_a_large_family():
    tested = tuple(chain(f"k{i}", p=0.2 + i / 1000, p_adj=0.9) for i in range(400))
    b = board(tested=tested, untestable=(UntestableReason("x", 600),))
    out = render_scoreboard(b)
    assert "about 20 are expected to clear 0.05 by chance alone" in out


def test_a_graveyard_with_no_nominal_hits_says_they_failed_outright():
    tested = tuple(chain(f"k{i}", p=0.3 + i / 100, p_adj=0.9) for i in range(5))
    b = board(tested=tested, untestable=(UntestableReason("x", 95),))
    out = render_scoreboard(b)
    assert "failed outright" in out
    assert "would have looked like discoveries" not in out


# ---------------------------------------------------------------------------
# freezing a run: re-rendering must not mean re-running
# ---------------------------------------------------------------------------


def _runner():
    """Import scripts/ghost_discovery_run.py without executing a run."""
    import importlib.util
    import pathlib
    import sys

    if "ghost_discovery_run" in sys.modules:
        return sys.modules["ghost_discovery_run"]
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "ghost_discovery_run.py"
    spec = importlib.util.spec_from_file_location("ghost_discovery_run", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through sys.modules, so the module has to
    # be registered BEFORE it is executed or the decorator raises.
    sys.modules["ghost_discovery_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rich_board():
    tested = (
        chain("s", p=0.0, p_adj=0.0, rejected=True, mechanism="m1", scaffold_only=True),
        chain("k1", p=0.04, p_adj=0.9, mechanism="m1", notes=("a note",)),
        chain("k2", p=0.50, p_adj=0.99, mechanism="m2", independent=0),
    )
    return board(
        tested=tested,
        untestable=(UntestableReason("no link", 900), UntestableReason("too few dates", 97)),
        checks=(series_check("p", [0.0, 0.04, 0.5]),),
    )


def test_a_frozen_board_re_renders_byte_for_byte():
    """A wording change must be re-renderable without re-running 20 minutes of studies.

    If re-rendering required a re-run, every edit would be an invitation to keep
    whichever run produced the nicer numbers.
    """
    mod = _runner()
    original = _rich_board()
    restored = mod.board_from_json(mod.board_to_json(original))
    assert render_scoreboard(restored) == render_scoreboard(original)


def test_a_frozen_board_keeps_every_field_that_drives_the_page():
    mod = _runner()
    original = _rich_board()
    restored = mod.board_from_json(mod.board_to_json(original))
    assert restored.enumerated == original.enumerated
    assert restored.untestable_total == original.untestable_total
    assert [c.chain_id for c in restored.tested] == [c.chain_id for c in original.tested]
    assert restored.distinct_survivor_mechanisms == original.distinct_survivor_mechanisms
    assert restored.tested[0].route == original.tested[0].route
    assert restored.tested[0].route_is_scaffold_only is True
    assert restored.tested[1].notes == ("a note",)
    assert restored.checks[0].distinct == original.checks[0].distinct


def test_a_hand_edited_frozen_board_cannot_be_rendered_into_a_smaller_family():
    """The invariants run again on load, so shrinking the denominator by hand fails."""
    import json as _json

    mod = _runner()
    doc = _json.loads(mod.board_to_json(_rich_board()))
    doc["scalars"]["enumerated"] = 3  # "3 chains tested, 1 survived"
    with pytest.raises(ScoreboardError, match="does not reconcile"):
        mod.board_from_json(_json.dumps(doc))


def test_a_hand_edited_frozen_board_cannot_smuggle_a_chain_below_the_floors():
    import json as _json

    mod = _runner()
    doc = _json.loads(mod.board_to_json(_rich_board()))
    doc["tested"][0]["n_clusters"] = 2
    with pytest.raises(ScoreboardError, match="below the declared floors"):
        mod.board_from_json(_json.dumps(doc))
