"""Tests for agent.lookthrough.combine.

Design notes
------------
Almost every test uses a HAND-COMPUTED fake index provider, so the assertions
are against arithmetic a reader can verify in their head rather than against
whatever NSE and Yahoo say today. Two tests marked ``network`` run the real
thing end to end; they are the only ones that can be flaky and they assert
*structure* (weights sum to 1, the attribution reconciles) rather than
particular index weights.

What these tests are FOR (and the mutation each one kills):
  * the sum-to-1.0 invariant, including with an opaque fund holding weight;
  * the exactness of "+X via Y" — fund_weight x constituent_weight, not an
    inferred residual;
  * ranking by SURPRISE, not by size;
  * the three shapes the product must handle: fund+stocks, all-direct,
    funds-only;
  * every refusal path producing a reason, never a wrong number;
  * no advice language in any string a user sees.
"""

from __future__ import annotations

import re

import pytest

from agent.lookthrough import combine as C

# ---------------------------------------------------------------------------
# A hand-computed index. Three names, weights chosen so every product is
# exact in binary and can be checked by eye.
# ---------------------------------------------------------------------------

FAKE_NIFTY = {"HDFCBANK": 0.5, "RELIANCE": 0.25, "ICICIBANK": 0.25}
FAKE_NEXT = {"DMART": 0.5, "ZOMATO": 0.5}


class FakeProvider:
    """Deterministic index weights, with a recorded call count."""

    def __init__(self, note: str = "TEST NOTE: these weights are made up", **err):
        self.note = note
        self.calls: list[str] = []
        self.err = err

    def index_weights(self, index: str) -> C.IndexWeights:
        self.calls.append(index)
        table = {"NIFTY 50": FAKE_NIFTY, "NIFTY NEXT 50": FAKE_NEXT}
        if index not in table:
            raise C.LookThroughError(f"fake provider has no {index}")
        return C.IndexWeights(
            index=index,
            weights=dict(table[index]),
            names={k: f"{k} Ltd." for k in table[index]},
            note=self.note,
            asof="2026-10-02",
            **self.err,
        )


def lt(text: str, provider=None, **kw) -> C.LookThrough:
    return C.look_through(text, provider=provider or FakeProvider(), **kw)


# ---------------------------------------------------------------------------
# Shape 1 — the common case: an index fund plus direct stocks
# ---------------------------------------------------------------------------

COMMON = "HDFCBANK 5%\nRELIANCE 8%\nINFY 37%\nNIFTYBEES 40%\nParag Parikh Flexi Cap 10%"


def test_common_case_actual_weights_are_direct_plus_fund_share():
    r = lt(COMMON)
    assert r.computed, r.not_computed_reason
    # 0.05 direct + 0.40 * 0.50 = 0.25
    assert r.by_symbol("HDFCBANK").actual == pytest.approx(0.25)
    # 0.08 direct + 0.40 * 0.25 = 0.18
    assert r.by_symbol("RELIANCE").actual == pytest.approx(0.18)
    # never listed: 0.40 * 0.25 = 0.10
    icici = r.by_symbol("ICICIBANK")
    assert icici.listed is None
    assert icici.actual == pytest.approx(0.10)
    # INFY is not in the index, so look-through leaves it alone
    assert r.by_symbol("INFY").actual == pytest.approx(0.37)
    assert r.by_symbol("INFY").via == ()


def test_weights_sum_to_one_with_an_opaque_fund_holding_weight():
    r = lt(COMMON)
    assert r.total == pytest.approx(1.0, abs=C.WEIGHT_SUM_TOLERANCE)
    assert abs(r.residual) <= C.WEIGHT_SUM_TOLERANCE
    # the opaque fund's 10% is a line of its own, not redistributed
    opaque = [x for x in r.lines if x.kind == "opaque_fund"]
    assert len(opaque) == 1
    assert opaque[0].actual == pytest.approx(0.10)
    assert "flexi-cap" in opaque[0].note


def test_attribution_is_the_exact_product_not_an_inferred_residual():
    r = lt(COMMON)
    c = r.by_symbol("HDFCBANK").via[0]
    assert c.source == "NIFTYBEES"
    assert c.index == "NIFTY 50"
    assert c.fund_weight == pytest.approx(0.40)
    assert c.constituent_weight == pytest.approx(0.50)
    assert c.weight == pytest.approx(c.fund_weight * c.constituent_weight)
    # and the reconciliation a user would do by hand against a factsheet
    assert c.weight == pytest.approx(0.20)
    # explain() must print BOTH factors, so a user can redo the multiplication
    assert "NIFTYBEES" in c.explain() and "NIFTY 50" in c.explain()
    assert "40.0000%" in c.explain() and "50.0000%" in c.explain()
    assert "20.0000%" in c.explain()


def test_ranking_is_by_surprise_not_by_size():
    r = lt(COMMON)
    order = [x.symbol for x in r.lines if x.kind == "company"]
    # HDFCBANK surprise 0.20, ICICIBANK 0.10, RELIANCE 0.10 (but larger),
    # INFY 0.0 — so INFY is last despite being the second-largest holding.
    assert order[0] == "HDFCBANK"
    assert order[-1] == "INFY"
    assert r.by_symbol("INFY").actual > r.by_symbol("ICICIBANK").actual
    assert order.index("ICICIBANK") < order.index("INFY")
    surprises = [r.by_symbol(s).surprise for s in order]
    assert surprises == sorted(surprises, reverse=True)


def test_never_listed_position_outranks_a_tiny_move():
    # ICICIBANK was never listed and arrives at 10%; TCS was listed and moves 0.
    r = lt("TCS 50%\nNIFTYBEES 40%\nINFY 10%")
    order = [x.symbol for x in r.lines if x.kind == "company"]
    assert order.index("ICICIBANK") < order.index("TCS")


def test_render_is_the_product_and_names_the_surprise():
    text = C.render_text(lt(COMMON))
    assert "You listed 5 positions. Looking through 1 of them, you hold 4 companies." in text
    assert re.search(r"HDFCBANK\s+you listed\s+5\.0%\s+you actually hold\s+25\.0%", text)
    assert "(+20.0 via NIFTYBEES)" in text
    assert re.search(r"ICICIBANK\s+you listed\s+none\s+you actually hold\s+10\.0%", text)
    assert "(all via NIFTYBEES)" in text
    assert "(held directly only)" in text  # INFY
    assert "1 of the 5 lines could not be looked through" in text
    assert "Parag Parikh Flexi Cap" in text


def test_top_three_is_arithmetic_over_post_lookthrough_weights():
    r = lt(COMMON)
    assert r.top_n_share(3) == pytest.approx(0.37 + 0.25 + 0.18)
    assert f"{(0.37 + 0.25 + 0.18) * 100:.1f}%" in C.render_text(r)


# ---------------------------------------------------------------------------
# Shape 2 — all direct stocks: look-through changes nothing, and says so
# ---------------------------------------------------------------------------

ALL_DIRECT = "RELIANCE 25%\nHDFCBANK 20%\nINFY 20%\nTCS 20%\nITC 15%"


def test_all_direct_is_an_identity_and_touches_no_index():
    p = FakeProvider()
    r = lt(ALL_DIRECT, provider=p)
    assert p.calls == []
    assert r.n_unpacked == 0
    assert r.n_companies == 5
    assert r.total == pytest.approx(1.0)
    for line in r.lines:
        assert line.listed == pytest.approx(line.actual)
        assert line.surprise == pytest.approx(0.0)
        assert line.via == ()


def test_all_direct_output_reads_sensibly_and_states_that_nothing_changed():
    text = C.render_text(lt(ALL_DIRECT))
    assert "There was nothing to look through" in text
    assert "what you listed is what you hold" in text
    assert "(held directly only)" in text
    # no empty "Looked through" / "could not be looked through" sections
    assert "Looked through\n" not in text
    assert "could not be looked through" not in text
    assert "Your 3 largest real positions are 65.0% of the money." in text


# ---------------------------------------------------------------------------
# Shape 3 — funds only
# ---------------------------------------------------------------------------


def test_funds_only_holds_the_union_of_the_indices():
    r = lt("NIFTYBEES 50%\nJUNIORBEES 30%\nSBI Bluechip Fund 20%")
    assert r.n_unpacked == 2
    assert r.n_companies == 5  # 3 in NIFTY 50 + 2 in NIFTY NEXT 50
    assert r.total == pytest.approx(1.0)
    assert all(x.listed is None for x in r.lines if x.kind == "company")
    assert r.by_symbol("HDFCBANK").actual == pytest.approx(0.50 * 0.50)
    assert r.by_symbol("DMART").actual == pytest.approx(0.30 * 0.50)
    text = C.render_text(r)
    assert "you hold 5 companies" in text
    assert "SBI Bluechip Fund" in text and "20.0%" in text


def test_two_funds_tracking_the_same_index_merge_into_one_attribution_line():
    r = lt("NIFTYBEES 30%\nUTI Nifty 50 Index Fund 70%")
    hdfc = r.by_symbol("HDFCBANK")
    assert hdfc.actual == pytest.approx(0.5)
    assert len(hdfc.via) == 2
    assert {c.source for c in hdfc.via} == {"NIFTYBEES", "UTI Nifty 50 Index Fund"}
    assert sum(c.weight for c in hdfc.via) == pytest.approx(0.5)


def test_the_same_fund_listed_twice_is_one_attribution_source():
    r = lt("NIFTYBEES 40%\nNIFTYBEES 60%")
    hdfc = r.by_symbol("HDFCBANK")
    assert len(hdfc.via) == 1
    assert hdfc.via[0].fund_weight == pytest.approx(1.0)
    assert hdfc.actual == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Line classification — the part that reads what a human pastes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,kind,index",
    [
        ("HDFCBANK 5%", "direct", None),
        ("NIFTYBEES 40%", "index_fund", "NIFTY 50"),
        ("UTI Nifty 50 Index Fund 5%", "index_fund", "NIFTY 50"),
        ("HDFC Nifty 50 Index Fund Direct Growth 12%", "index_fund", "NIFTY 50"),
        ("ICICI Prudential Nifty Next 50 Index Fund 6%", "index_fund", "NIFTY NEXT 50"),
        ("JUNIORBEES 10%", "index_fund", "NIFTY NEXT 50"),
        ("Motilal Oswal Nifty Midcap 150 Index Fund 8%", "index_fund", "NIFTY MIDCAP 150"),
        ("BANKBEES 7%", "index_fund", "NIFTY BANK"),
        ("Nippon India Nifty 500 Index Fund 4%", "index_fund", "NIFTY 500"),
        ("Parag Parikh Flexi Cap 2.7%", "opaque_fund", None),
        ("SBI Bluechip Fund 4%", "opaque_fund", None),
        ("Nippon India Gold ETF 3%", "opaque_fund", None),
        ("HDFC Balanced Advantage Fund 6%", "opaque_fund", None),
        ("Quant Small Cap Fund 5%", "opaque_fund", None),
    ],
)
def test_line_classification(line, kind, index):
    got, _ = C.resolve_line(1, line)
    assert got.kind == kind
    assert got.index == index


@pytest.mark.parametrize(
    "line,name,size",
    [
        # the exact failure that makes parse_holdings unusable raw: it reads
        # the ticker as "UTI" and the quantity as the 50 in "Nifty 50".
        ("UTI Nifty 50 Index Fund 5%", "UTI Nifty 50 Index Fund", "5%"),
        ("Motilal Oswal Nifty Midcap 150 Index Fund 8%", "Motilal Oswal Nifty Midcap 150 Index Fund", "8%"),
        # the number ends the NAME, with the size right after it: the trailing
        # scan must stop at the first number or it eats "150" as well.
        ("Motilal Oswal Nifty Midcap 150 10%", "Motilal Oswal Nifty Midcap 150", "10%"),
        ("HDFC Nifty 50 20%", "HDFC Nifty 50", "20%"),
        ("UTI Nifty 50 Rs 20000", "UTI Nifty 50", "Rs 20000"),
        ("40% NIFTYBEES", "NIFTYBEES", "40%"),
        ("Rs 80000 BANKBEES", "BANKBEES", "Rs 80000"),
        ("HDFCBANK 100 shares", "HDFCBANK", "100 shares"),
        ("HDFCBANK", "HDFCBANK", ""),
        # a numbered list, which brokers and forum posts both produce
        ("1. HDFCBANK 5%", "HDFCBANK", "1. 5%"),
        # degenerate, but it pins the stated rule at BOTH ends of the line:
        # a size run carries at most one number, so the second number here
        # belongs to the name and not to the size.
        ("5 10 HDFCBANK", "10 HDFCBANK", "5"),
    ],
)
def test_size_split_keeps_numbers_that_are_part_of_the_name(line, name, size):
    assert C._split_size(line) == (name, size)


def test_the_number_in_the_index_name_is_never_read_as_the_quantity():
    r = lt("UTI Nifty 50 Index Fund 100%")
    # if "50" had been read as the quantity the fund weight would still be 1.0
    # after normalisation, so assert the arithmetic that only holds if the
    # fund was recognised at all:
    assert r.n_unpacked == 1
    assert r.by_symbol("HDFCBANK").actual == pytest.approx(0.5)
    assert r.funds[0].label == "UTI Nifty 50 Index Fund"


def test_fund_label_is_what_the_user_typed_not_our_internal_token():
    text = C.render_text(lt(COMMON))
    assert "FUNDLINE" not in text


# ---------------------------------------------------------------------------
# Refusals — every one carries a reason, and none carries a number
# ---------------------------------------------------------------------------


def test_indian_digit_grouping_is_refused_not_misread():
    r = lt("HDFCBANK Rs 1,20,000\nRELIANCE Rs 80,000")
    assert not r.computed
    assert "1,20,000 becomes 1" in r.not_computed_reason
    assert "without commas" in r.not_computed_reason
    assert r.lines == ()
    assert "NOT COMPUTED" in C.render_text(r)


def test_lakh_and_crore_are_refused_not_misread():
    r = lt("HDFCBANK 2 lakh\nRELIANCE 3 lakh")
    assert not r.computed
    assert "lakh" in r.not_computed_reason
    assert "becomes 2" in r.not_computed_reason


def test_share_counts_mixed_with_a_fund_are_refused_with_the_nav_reason():
    r = lt("HDFCBANK 100 shares\nNIFTYBEES 50 units")
    assert not r.computed
    assert "NAV" in r.not_computed_reason
    assert "NIFTYBEES" in r.not_computed_reason
    assert r.lines == ()


def test_amounts_in_two_currencies_go_through_the_priced_fx_path():
    """Two currencies cannot be added; they must be routed to fetch_prices."""
    seen = {}

    def fake_prices(holdings):
        seen["mode"] = holdings.unit_mode
        return {"HDFCBANK": 0.25, "RELIANCE": 0.75}, "fake FX-converted basis"

    r = C.look_through(
        "HDFCBANK $5000\nRELIANCE \u20b9400000",
        provider=FakeProvider(),
        price_weights=fake_prices,
    )
    assert r.computed, r.not_computed_reason
    assert seen["mode"] == "amount"
    assert r.by_symbol("RELIANCE").actual == pytest.approx(0.75)
    assert "fake FX-converted basis" in C.render_text(r)


def test_a_priced_basket_containing_a_fund_is_refused_with_the_nav_reason():
    r = lt("HDFCBANK $5000\nNIFTYBEES \u20b9400000")
    assert not r.computed
    assert "NAV" in r.not_computed_reason
    assert "NIFTYBEES" in r.not_computed_reason


def test_an_equal_weight_fallback_from_the_price_layer_is_refused():
    class Panel:
        usable = True
        unusable_reason = ""
        weight_basis = "could not value any position; fell back to EQUAL WEIGHT"

        class weights:  # noqa: N801
            empty = False

    import agent.portfolio.holdings as H

    real = H.fetch_prices
    H.fetch_prices = lambda h: Panel()
    try:
        with pytest.raises(C.LookThroughError, match="will not show you a number"):
            C._price_weights_via_holdings(C.parse_holdings("HDFCBANK 10 shares"))
    finally:
        H.fetch_prices = real


def test_a_bare_list_with_no_sizes_says_equal_weight_is_our_assumption():
    r = lt("HDFCBANK\nRELIANCE\nINFY\nTCS")
    assert r.computed
    assert r.total == pytest.approx(1.0)
    assert r.by_symbol("TCS").actual == pytest.approx(0.25)
    assert "EQUAL WEIGHT" in r.weight_basis
    assert "our \nassumption, not your portfolio" in r.weight_basis.replace("  ", " ") or (
        "not your portfolio" in r.weight_basis
    )
    assert "not your portfolio" in C.render_text(r)


def test_empty_paste_is_refused_with_a_reason():
    r = lt("   \n\n")
    assert not r.computed
    assert "no readable positions" in r.not_computed_reason


def test_a_fund_whose_index_cannot_be_fetched_keeps_its_own_weight():
    class Broken(FakeProvider):
        def index_weights(self, index):
            raise RuntimeError("NSE returned 503")

    r = lt("HDFCBANK 50%\nNIFTYBEES 50%", provider=Broken())
    assert r.computed
    assert r.total == pytest.approx(1.0)  # nothing vanished
    assert r.n_unpacked == 0
    blocked = [f for f in r.funds if not f.unpacked]
    assert len(blocked) == 1
    assert "503" in blocked[0].reason
    assert "NOT spread over the index" in blocked[0].reason
    assert "503" in C.render_text(r)


def test_missing_sibling_module_degrades_to_not_computed_with_the_reason(monkeypatch):
    """indices.py is a concurrently written sibling. Its absence must degrade."""
    import sys

    import agent.lookthrough as pkg

    monkeypatch.setitem(sys.modules, "agent.lookthrough.indices", None)
    monkeypatch.delattr(pkg, "indices", raising=False)
    prov, reason = C._default_provider()
    assert prov is None
    assert "could not be imported" in reason
    r = C.look_through("HDFCBANK 50%\nNIFTYBEES 50%")
    assert r.computed  # the direct stock is still arithmetic
    assert r.total == pytest.approx(1.0)
    assert r.by_symbol("HDFCBANK").actual == pytest.approx(0.5)
    blocked = [f for f in r.funds if not f.unpacked]
    assert len(blocked) == 1 and "indices" in blocked[0].reason
    assert "NOT spread over the index" in C.render_text(r)


def test_a_sibling_without_either_entry_point_degrades_with_a_named_reason(monkeypatch):
    import sys
    import types

    stub = types.ModuleType("agent.lookthrough.indices")
    monkeypatch.setitem(sys.modules, "agent.lookthrough.indices", stub)
    import agent.lookthrough as pkg

    monkeypatch.setattr(pkg, "indices", stub, raising=False)
    prov, reason = C._default_provider()
    assert prov is None
    assert "neither index_weights() nor fetch_index()" in reason


def test_percentages_that_do_not_sum_to_100_are_normalised_and_said_so():
    r = lt("HDFCBANK 30%\nRELIANCE 30%\nINFY 30%")
    assert r.total == pytest.approx(1.0)
    assert r.by_symbol("HDFCBANK").actual == pytest.approx(1 / 3)
    assert any("sum to 90.00" in a for a in r.assumptions)
    assert "divided by 90.00" in C.render_text(r)


def test_a_broken_provider_whose_weights_do_not_sum_to_one_is_rejected():
    class Bad:
        def index_weights(self, index):
            return C.IndexWeights(index=index, weights={"A": 0.4, "B": 0.4})

    r = C.look_through("NIFTYBEES 100%", provider=Bad())
    assert r.computed
    blocked = [f for f in r.funds if not f.unpacked]
    assert len(blocked) == 1
    assert "not 1.0" in blocked[0].reason


def test_the_tolerance_is_tight_enough_to_see_a_real_drift():
    """A provider 5e-7 off passes its own check but must break the addition.

    This is what stops WEIGHT_SUM_TOLERANCE from being quietly widened into a
    tolerance that would accept a genuinely wrong total.
    """

    class SlightlyOff:
        def index_weights(self, index):
            return C.IndexWeights(index=index, weights={"A": 0.5, "B": 0.5000005})

    SlightlyOff().index_weights("NIFTY 50").check()  # its own check passes
    with pytest.raises(C.LookThroughError, match="refusing to show a number"):
        C.look_through("NIFTYBEES 100%", provider=SlightlyOff())


def test_a_lookthrough_with_a_fund_at_full_weight_still_sums_to_one():
    r = lt("NIFTYBEES 100%")
    assert r.total == pytest.approx(1.0, abs=C.WEIGHT_SUM_TOLERANCE)
    assert abs(r.residual) <= C.WEIGHT_SUM_TOLERANCE


def test_residual_breach_raises_rather_than_rendering_a_number(monkeypatch):
    """The sum-to-1 check must be a wall, not a warning."""
    monkeypatch.setattr(C, "WEIGHT_SUM_TOLERANCE", -1.0)
    with pytest.raises(C.LookThroughError, match="refusing to show a number"):
        lt(COMMON)


# ---------------------------------------------------------------------------
# The approximation note must travel with the numbers
# ---------------------------------------------------------------------------


def test_the_index_note_is_rendered_next_to_the_numbers():
    r = lt(COMMON, provider=FakeProvider(note="MADE UP WEIGHTS, 9.9pp error"))
    assert any("MADE UP WEIGHTS" in n for n in r.approximation_notes)
    assert "MADE UP WEIGHTS, 9.9pp error" in C.render_text(r)


def test_a_measured_error_is_propagated_to_the_users_own_exposure():
    r = lt(COMMON, provider=FakeProvider(error_mean_pp=0.5, error_max_pp=2.0))
    # 40% of the book in the fund x 2.0pp worst index error = 0.80pp
    assert any("0.80pp" in n for n in r.approximation_notes), r.approximation_notes
    assert "0.80pp" in C.render_text(r)


def test_an_unmeasured_error_is_said_to_be_unquantified_not_zero():
    r = lt(COMMON, provider=FakeProvider())
    joined = " ".join(r.approximation_notes)
    assert "unquantified error, not a zero one" in joined
    assert "0.00pp" not in joined


# ---------------------------------------------------------------------------
# No advice, ever
# ---------------------------------------------------------------------------

_BANNED = (
    "over-concentrated",
    "overconcentrated",
    "concentrated",
    "you should",
    "consider ",
    "we recommend",
    "recommend",
    "suggest",
    "advice",
    "advise",
    "trim",
    "rebalance",
    "diversify",
    "too much",
    "risky",
    "warning",
    "forecast",
    "expect to",
    "will outperform",
    "buy",
    "sell",
)


@pytest.mark.parametrize("text", [COMMON, ALL_DIRECT, "NIFTYBEES 60%\nSBI Bluechip Fund 40%"])
def test_no_rendered_string_contains_advice(text):
    out = C.render_text(lt(text)).lower()
    for word in _BANNED:
        assert word not in out, f"advice-adjacent word {word!r} in rendered output"


def test_refusal_text_contains_no_advice():
    for bad in ("HDFCBANK Rs 1,20,000", "HDFCBANK 100 shares\nNIFTYBEES 50 units"):
        out = C.render_text(lt(bad)).lower()
        for word in _BANNED:
            assert word not in out, f"{word!r} in refusal for {bad!r}"


# ---------------------------------------------------------------------------
# Live sources. These are the only tests here that touch the network; each
# skips (with the reason printed) rather than failing when a source is down,
# because a flaky red test teaches people to ignore red tests.
# ---------------------------------------------------------------------------


def _live(text: str) -> C.LookThrough:
    r = C.look_through(text)
    if not r.computed:
        pytest.skip(f"live sources unavailable: {r.not_computed_reason}")
    blocked = [f.reason for f in r.funds if not f.unpacked]
    if blocked:
        pytest.skip(f"live index fetch failed: {blocked[0]}")
    return r


def test_live_lookthrough_sums_to_one_and_reconciles_by_hand():
    r = _live("HDFCBANK 5%\nRELIANCE 8%\nINFY 37%\nNIFTYBEES 50%")
    assert r.total == pytest.approx(1.0, abs=C.WEIGHT_SUM_TOLERANCE)
    assert r.n_unpacked == 1
    assert r.n_companies == 50
    hdfc = r.by_symbol("HDFCBANK")
    assert hdfc.listed == pytest.approx(0.05)
    # the user's hand check: listed + fund_weight x index_weight
    c = hdfc.via[0]
    assert c.fund_weight == pytest.approx(0.50)
    assert hdfc.actual == pytest.approx(0.05 + c.fund_weight * c.constituent_weight)
    assert 0.05 < c.constituent_weight < 0.20  # HDFCBANK's real share of NIFTY 50
    assert r.approximation_notes and "APPROXIMATION" in " ".join(r.approximation_notes)


def test_live_series_varies_rather_than_being_a_constant():
    """LESSONS F-18: print the whole series and confirm it is not a constant."""
    r = _live("NIFTYBEES 100%")
    series = [x.actual for x in r.lines]
    print("\n".join(f"{x.symbol:<14}{x.actual * 100:7.3f}%" for x in r.lines))
    assert len(series) == 50
    assert len(set(round(v, 6) for v in series)) > 40, "weights look quantised/constant"
    assert max(series) / min(series) > 5, "a real cap-weighted index is not flat"
    assert sum(series) == pytest.approx(1.0, abs=C.WEIGHT_SUM_TOLERANCE)
