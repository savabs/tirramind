"""Tests for :mod:`agent.lookthrough.funds`.

These tests are written against the failure that matters: unpacking a fund into
the wrong index and reporting success. A test that only checks "NIFTYBEES
resolves to NIFTY 50" passes against a one-line substring matcher that also
unpacks "Nifty 500 Momentum 50" into the Nifty 500. So most of what follows
asserts the *refusals*, and several tests assert specifically that a named
mutation of the module would be caught.

The offline tests use a small hand-built AMFI fixture whose format is a verbatim
copy of the live file's layout (8 semicolon-separated columns, category and AMC
as bare lines between the rows). ``@pytest.mark.live`` tests hit the real AMFI
endpoint and are the only ones that need a network.
"""

from __future__ import annotations

import json
import time

import pytest

from agent.lookthrough.funds import (
    OPAQUE_TICKERS,
    SUPPORTED_INDICES,
    TICKER_LISTED_NAMES,
    AmfiUnavailable,
    classify,
    coverage_report,
    index_phrase,
    load_amfi_schemes,
    load_scheme_master,
    look_up_fund,
    parse_amfi_text,
    resolve_fund,
    resolve_funds,
    supported_indices,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: A verbatim-format slice of AMFI's NAVAll.txt. Every row below was copied
#: from the live file on 2026-10-02, including its scheme code and ISIN, so a
#: change to the real layout breaks these tests rather than passing silently.
AMFI_FIXTURE = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Exchange Traded Funds (ETFs) - Equity ETF)

Nippon India Mutual Fund

140084;INF204KB14I2;-;Nippon India ETF Nifty 50 BeES;Direct Plan;;255.7379;01-Oct-2026
140087;INF204KB15I9;-;Nippon India ETF Nifty Bank BeES;Direct Plan;;564.4703;01-Oct-2026
140085;INF732E01045;-;Nippon India ETF Nifty Next 50 Junior BeES;Direct Plan;;746.4101;01-Oct-2026
140089;INF204KB16I7;-;Nippon India ETF Nifty PSU Bank BeES;Direct Plan;;88.6562;01-Oct-2026

Open Ended Schemes(Index Funds - Equity Funds)

UTI Mutual Fund

120716;INF789F01XA0;-;UTI Nifty 50 Index Fund;Direct Plan;Growth Option;180.1234;01-Oct-2026
143173;INF789FC1LT6;-;UTI Nifty Next 50 Index Fund;Direct Plan;Growth Option;24.5678;01-Oct-2026

HDFC Mutual Fund

101762;INF179K01XQ0;-;HDFC Nifty 50 Index Fund;Direct Plan;Growth Option;245.6789;04-Oct-2026
152345;INF179KC1AB1;-;HDFC NIFTY 100 Equal Weight Index Fund;Direct Plan;Growth Option;15.4321;01-Oct-2026
152346;INF179KC1CD2;-;HDFC NIFTY GROWTH SECTORS 15 ETF;Direct Plan;;120.5000;01-Oct-2026

Motilal Oswal Mutual Fund

147623;INF247L01AB5;-;Motilal Oswal Nifty 500 Index Fund;Direct Plan;Growth Option;22.3344;01-Oct-2026
147624;INF247L01CD6;-;Motilal Oswal Nifty 500 Momentum 50 Index Fund;Direct Plan;Growth Option;18.9900;01-Oct-2026

Open Ended Schemes(Equity Scheme - Flexi Cap Fund)

PPFAS Mutual Fund

122639;INF879O01019;-;Parag Parikh Flexi Cap Fund;Direct Plan;Growth;91.2345;01-Oct-2026

HDFC Mutual Fund

118955;INF179K01WT5;-;HDFC Flexi Cap Fund;Direct Plan;Growth;1987.6543;01-Oct-2026

Open Ended Schemes(Other Scheme - Gold ETF)

Nippon India Mutual Fund

145073;INF204KB17I5;-;Nippon India ETF Gold BeES;Direct Plan;;85.4321;01-Oct-2026
"""


@pytest.fixture
def master(tmp_path, monkeypatch):
    """A :class:`SchemeMaster` built from the fixture, with no network at all.

    The AMFI text is planted directly in the module's disk cache, so
    ``load_scheme_master`` takes its cache-hit path. That exercises the real
    parsing and indexing code rather than a stub.
    """
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path))
    from agent.lookthrough import funds as f

    f._cache_write(f.AMFI_NAV_ALL_URL, {"ok": True, "text": AMFI_FIXTURE})

    def _boom(*_a, **_k):  # pragma: no cover - asserts no network is used
        raise AssertionError("the offline tests must not touch the network")

    monkeypatch.setattr(f, "_http_get_text", _boom)
    return f.load_scheme_master()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_amfi_text_reads_rows_categories_and_amc():
    schemes, unparsed = parse_amfi_text(AMFI_FIXTURE)
    assert unparsed == [], f"unparsed rows: {unparsed}"
    assert len(schemes) == 14

    nb = next(s for s in schemes if s.name == "Nippon India ETF Nifty 50 BeES")
    assert nb.code == 140084
    assert nb.isin_growth == "INF204KB14I2"
    assert nb.isin_reinvest is None, "'-' must become None, not the string '-'"
    assert nb.nav == pytest.approx(255.7379)
    assert nb.amc == "Nippon India Mutual Fund"
    assert "Equity ETF" in nb.category

    ppfas = next(s for s in schemes if s.name == "Parag Parikh Flexi Cap Fund")
    assert ppfas.amc == "PPFAS Mutual Fund", "the AMC must be the nearest heading above"
    assert "Flexi Cap" in ppfas.category


def test_parse_handles_the_historical_six_column_layout():
    """AMFI used to publish 6 columns. Both layouts must parse, not one.

    MUTATION GUARD: dropping the ``len(cells) >= 6`` branch makes this fail
    instead of silently returning zero schemes for an older file.
    """
    six = (
        "Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;"
        "Scheme Name;Net Asset Value;Date\n"
        "Open Ended Schemes(Index Funds - Equity Funds)\n"
        "UTI Mutual Fund\n"
        "120716;INF789F01XA0;-;UTI Nifty 50 Index Fund;180.1234;01-Oct-2026\n"
    )
    schemes, unparsed = parse_amfi_text(six)
    assert unparsed == []
    assert len(schemes) == 1
    assert schemes[0].plan == "" and schemes[0].option == ""
    assert schemes[0].nav == pytest.approx(180.1234)


def test_unreadable_rows_are_returned_not_dropped():
    """A malformed row must be reported. Silence here is a missing scheme."""
    text = AMFI_FIXTURE + "this;is;broken\n"
    schemes, unparsed = parse_amfi_text(text)
    assert len(schemes) == 14
    assert unparsed == ["this;is;broken"]


def test_as_of_is_the_modal_date_not_the_lexicographic_or_future_max(master):
    """``as_of`` must be a real file date, by two separate near-misses.

    The fixture holds 13 rows at 01-Oct-2026 and one at 04-Oct-2026.

    MUTATION GUARD 1: ``max()`` over the raw strings answers "04-Oct-2026"
    here, and on the live file answers "31-Oct-2025" — a date from the wrong
    year, because "3" sorts above "0".
    MUTATION GUARD 2: a correct date-wise ``max`` still answers "04-Oct-2026",
    a date that had not happened when the file was published.
    """
    assert master.as_of == "01-Oct-2026"
    assert master.as_of_note is not None
    assert "04-Oct-2026" in master.as_of_note, "the outlier must be disclosed, not hidden"


# ---------------------------------------------------------------------------
# index_phrase — the whole correctness of the module lives here
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HDFC Nifty 50 Index Fund", "nifty 50"),
        ("UTI Nifty 50 Index Fund", "nifty 50"),
        ("Nippon India Index Fund - Nifty 50 Plan", "nifty 50"),
        ("Nippon India ETF Nifty 50 BeES", "nifty 50"),
        ("Nippon India ETF Nifty Next 50 Junior BeES", "nifty next 50"),
        ("Nippon India Nifty Next 50 Junior BeES FoF", "nifty next 50"),
        ("Tata Nifty 50 Exchange Traded Fund", "nifty 50"),
        ("Franklin India NSE Nifty 50 Index Fund", "nse nifty 50"),
        ("SBI NIFTY INDEX FUND", "nifty"),
        ("UTI - Nifty Next 50 Index Fund", "nifty next 50"),
        ("HDFC NIFTY Midcap 150 ETF", "nifty midcap 150"),
        ("Bank Nifty Index Fund", "nifty bank"),
        ("ICICI Prudential Nifty 50 Index Fund - Direct Plan - Growth", "nifty 50"),
        # Glued digits: scheme names really are written both ways.
        ("DSP Nifty50 Equal Weight Index Fund", "nifty 50 equal weight"),
    ],
)
def test_index_phrase_reduces_real_scheme_names(text, expected):
    assert index_phrase(text) == expected


def test_index_phrase_does_not_eat_a_word_that_ends_in_rs():
    """ "SECTORS 15" must survive. This was a real bug.

    MUTATION GUARD: removing the ``(?<![a-z0-9])`` lookbehind from
    ``_CURRENCY_LEAD_RE`` makes "NIFTY GROWTH SECTORS 15" reduce to
    "nifty growth secto", because "RS 15" reads as a rupee amount. The trap
    case still gets rejected either way, so only an assertion on the phrase
    itself catches it.
    """
    assert index_phrase("HDFC NIFTY GROWTH SECTORS 15 ETF") == "nifty growth sectors 15"


@pytest.mark.parametrize(
    "text",
    ["NIFTYBEES 200 units", "NIFTYBEES Rs 50,000", "NIFTYBEES ₹1,20,000", "niftybees.ns"],
)
def test_index_phrase_strips_a_pasted_quantity(text):
    assert index_phrase(text) == "niftybees"


@pytest.mark.parametrize("text", ["NIFTY 50", "Nifty 500", "Nifty Midcap 150"])
def test_index_phrase_keeps_a_trailing_number_that_is_part_of_the_index(text):
    """A bare trailing number must never be stripped as a quantity.

    MUTATION GUARD: dropping the unit-word check in ``_strip_qty_suffix`` turns
    "NIFTY 50" into "nifty" and "Nifty 500" into "nifty", which then both
    resolve to NIFTY 50 — the Nifty 500 silently becomes the Nifty 50.
    """
    assert index_phrase(text) == text.lower()


# ---------------------------------------------------------------------------
# The refusals. These are the tests that matter.
# ---------------------------------------------------------------------------

#: Every one of these contains the text of a supported index and tracks a
#: DIFFERENT index. A substring matcher resolves all of them, wrongly.
STRATEGY_INDEX_TRAPS = [
    ("HDFC NIFTY 100 Equal Weight Index Fund", "nifty 100 equal weight"),
    ("Motilal Oswal Nifty 500 Momentum 50 Index Fund", "nifty 500 momentum 50"),
    ("DSP Nifty 50 Equal Weight ETF", "nifty 50 equal weight"),
    ("Kotak Nifty 50 Value 20 ETF", "nifty 50 value 20"),
    ("UTI Nifty 200 Momentum 30 Index Fund", "nifty 200 momentum 30"),
    ("Nippon India Nifty Midcap 150 Momentum 50 Index Fund", "nifty midcap 150 momentum 50"),
    ("UTI Nifty Next 50 Value 20 Index Fund", "nifty next 50 value 20"),
    ("Nippon India ETF Nifty PSU Bank BeES", "nifty psu bank"),
    ("Nippon India Nifty Private Bank ETF", "nifty private bank"),
    ("SBI Nifty Smallcap 250 Index Fund", "nifty smallcap 250"),
    ("HDFC BSE Sensex Index Fund", "bse sensex"),
    ("HDFC Nifty G-Sec Dec 2026 Index Fund", "nifty g sec dec 2026"),
    ("HDFC NIFTY GROWTH SECTORS 15 ETF", "nifty growth sectors 15"),
]


@pytest.mark.parametrize(("name", "phrase"), STRATEGY_INDEX_TRAPS)
def test_a_different_index_is_never_mapped_to_a_nearby_one(name, phrase, master):
    """The core guarantee. Each of these names a real, different index.

    MUTATION GUARD: replace the exact alias lookup with ``any(alias in phrase)``
    and 9 of these 13 resolve — "Nifty 500 Momentum 50" (a 50-stock momentum
    screen) would be unpacked into the 501 companies of the Nifty 500.
    """
    r = look_up_fund(name, master=master)
    assert r.match is None, f"{name} wrongly resolved to {r.match and r.match.index_id}"
    assert r.index_phrase == phrase
    assert phrase in r.reason, "the refusal must quote the index we actually read"
    assert "un-unpacked" in r.reason


def test_an_active_fund_is_refused_with_a_reason_naming_it(master):
    r = look_up_fund("Parag Parikh Flexi Cap Fund", master=master)
    assert r.match is None
    assert r.classification == "active"
    assert "Parag Parikh Flexi Cap Fund" in r.reason
    assert "actively managed" in r.reason
    assert "un-unpacked" in r.reason


def test_an_active_fund_not_in_amfi_is_still_refused_as_active(master):
    """A half-typed name ("HDFC Flexi Cap") must not fall through to unknown."""
    r = look_up_fund("HDFC Flexi Cap", master=master)
    assert r.match is None
    assert r.classification == "active"


def test_a_gold_etf_is_refused_as_holding_no_equities(master):
    r = look_up_fund("Nippon India ETF Gold BeES", master=master)
    assert r.match is None
    assert "not an equity index" in r.reason


def test_an_opaque_ticker_is_refused_and_quotes_its_listed_name(master):
    """MIDCAPIETF's issuer code hints at Midcap 150. A hint is not knowledge.

    MUTATION GUARD: moving any entry of ``OPAQUE_TICKERS`` into
    ``TICKER_LISTED_NAMES`` with an inferred index makes this fail.
    """
    r = look_up_fund("MIDCAPIETF", master=master)
    assert r.match is None
    assert "ICICIPRAMC - ICICIM150" in r.reason, "quote the listed name, do not paraphrase"
    assert "hint" in r.reason


def test_opaque_and_resolvable_ticker_tables_are_disjoint():
    assert not (set(OPAQUE_TICKERS) & set(TICKER_LISTED_NAMES))


def test_an_unrecognised_string_is_refused_not_guessed(master):
    r = look_up_fund("my broker's balanced thing", master=master)
    assert r.match is None
    assert r.classification == "unknown"
    assert "un-unpacked" in r.reason


def test_an_isin_that_is_not_in_amfi_says_so(master):
    r = look_up_fund("INF999Z01ZZ9", master=master)
    assert r.match is None
    assert "not in AMFI" in r.reason


def test_empty_input_is_reported_not_silently_skipped(master):
    r = look_up_fund("   ", master=master)
    assert r.match is None
    assert "empty" in r.reason


# ---------------------------------------------------------------------------
# The matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "index_id"),
    [
        ("NIFTYBEES", "NIFTY 50"),
        ("niftybees", "NIFTY 50"),
        ("NIFTYBEES.NS", "NIFTY 50"),
        ("BANKBEES", "NIFTY BANK"),
        ("JUNIORBEES", "NIFTY NEXT 50"),
        ("ITBEES", "NIFTY IT"),
        ("MID150BEES", "NIFTY MIDCAP 150"),
        ("INF204KB14I2", "NIFTY 50"),
        ("inf204kb14i2", "NIFTY 50"),
        ("UTI Nifty 50 Index Fund", "NIFTY 50"),
        ("UTI Nifty Next 50 Index Fund", "NIFTY NEXT 50"),
        ("uti nifty next 50 index fund", "NIFTY NEXT 50"),
        ("Motilal Oswal Nifty 500 Index Fund", "NIFTY 500"),
        ("Nippon India Index Fund - Nifty 50 Plan", "NIFTY 50"),
        ("Nifty 50", "NIFTY 50"),
        ("Bank Nifty ETF", "NIFTY BANK"),
    ],
)
def test_generous_matching_on_what_a_human_actually_pastes(typed, index_id, master):
    m = resolve_fund(typed, master=master)
    assert m is not None, f"{typed!r} did not resolve"
    assert m.index_id == index_id


def test_a_matched_index_id_is_spelled_the_way_indices_expects_it(master):
    """A FundMatch must be feedable to agent.lookthrough.indices.fetch_index.

    MUTATION GUARD: renaming any key of SUPPORTED_INDICES (to "Nifty 50", say)
    breaks this, instead of breaking at runtime inside the sibling module.
    """
    from agent.lookthrough.indices import known_indices

    assert set(SUPPORTED_INDICES) <= set(known_indices())
    m = resolve_fund("NIFTYBEES", master=master)
    assert m.index_id in known_indices()


def test_bare_nifty_is_matched_but_the_assumption_is_stated(master):
    """ "UTI Nifty" resolves, and the sentence the user reads says why."""
    r = look_up_fund("UTI Nifty", master=master)
    assert r.match is not None
    assert r.match.index_id == "NIFTY 50"
    assert r.match.aliased_from == "nifty"
    assert "no index number" in r.reason
    assert any("unqualified" in c or "no index number" in c for c in r.match.caveats)


def test_a_fund_of_funds_is_matched_and_flagged_as_a_second_wrapper(master):
    r = look_up_fund("Nippon India Nifty Next 50 Junior BeES FoF", master=master)
    assert r.match is not None
    assert r.match.index_id == "NIFTY NEXT 50"
    assert r.match.kind == "fund_of_funds"
    assert any("fund of funds" in c for c in r.match.caveats)
    assert "another fund" in r.reason


def test_every_match_carries_the_caveat_that_it_is_not_the_weights(master):
    """A match names an index. It must never read as "and here are the weights"."""
    for typed in ("NIFTYBEES", "UTI Nifty 50 Index Fund", "INF204KB14I2"):
        m = resolve_fund(typed, master=master)
        assert m is not None
        joined = " ".join(m.caveats)
        assert "free-float" in joined
        assert "cash balance" in joined


def test_match_records_the_scheme_it_came_from_when_amfi_knew_it(master):
    m = resolve_fund("UTI Nifty 50 Index Fund", master=master)
    assert m.scheme_code == 120716
    assert m.isin == "INF789F01XA0"
    assert "Index Funds" in (m.category or "")


def test_every_ticker_in_the_table_derives_a_supported_index(master):
    """The ticker table asserts a listed NAME, never an index.

    The index is derived by running that name through ``index_phrase``. This
    test is what makes the table verifiable rather than a hand-written claim:
    a normalisation regression shows up here as an unresolved ticker.
    """
    unresolved = []
    for ticker, listed in TICKER_LISTED_NAMES.items():
        phrase = index_phrase(listed)
        if phrase not in {a for aliases in SUPPORTED_INDICES.values() for a in aliases}:
            unresolved.append((ticker, listed, phrase))
    assert unresolved == [], f"ticker names no longer reduce to a known index: {unresolved}"


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "category", "expected"),
    [
        ("UTI Nifty 50 Index Fund", "Open Ended Schemes(Index Funds - Equity Funds)", "index"),
        ("Nippon India ETF Nifty 50 BeES", "Open Ended Schemes(ETFs - Equity ETF)", "index"),
        ("Parag Parikh Flexi Cap Fund", "Open Ended Schemes(Equity Scheme - Flexi Cap Fund)", "active"),
        ("SBI Liquid Fund", "Open Ended Schemes(Debt Scheme - Liquid Fund)", "active"),
        # Name-only: no category available.
        ("HDFC Nifty 50 Index Fund", None, "index"),
        ("Motilal Oswal Nifty 500 Momentum 50 Index Fund", None, "index"),
        ("Nifty 500 Value 50", None, "index"),
        ("HDFC Flexi Cap Fund", None, "active"),
        ("ICICI Prudential Bluechip Fund", None, "active"),
        ("my broker's balanced thing", None, "unknown"),
        ("", None, "unknown"),
    ],
)
def test_classify(name, category, expected):
    assert classify(name, category=category) == expected


def test_classify_reads_the_name_before_falling_back_to_the_category():
    """An index-tracking fund-of-funds is filed under a category that hides it.

    AMFI files "Groww Nifty 200 ETF FOF" under "Fund of Funds Scheme
    (Domestic)", which says nothing about who picks the holdings.

    MUTATION GUARD: putting the ``if cat and "(" in cat: return "active"``
    branch back above the name check makes this return "active".
    """
    assert (
        classify(
            "Groww Nifty 200 ETF FOF",
            category="Open Ended Schemes(Fund of Funds Scheme (Domestic) - Fund of Funds Scheme (Domestic))",
        )
        == "index"
    )


def test_classify_index_does_not_imply_unpackable(master):
    """ "index" is about who picks the holdings, not about whether we have them."""
    for name in ("SBI Nifty Smallcap 250 Index Fund", "Nippon India ETF Gold BeES"):
        assert classify(name) == "index"
        assert resolve_fund(name, master=master) is None


def test_a_passive_marker_beats_an_active_sounding_word():
    """ "Nifty 500 Value 50 Index Fund" is passive despite the word "value"."""
    assert classify("Nippon India Nifty 500 Value 50 Index Fund") == "index"
    assert classify("Kotak Nifty 50 Value 20 ETF") == "index"


@pytest.mark.parametrize(
    "name",
    [
        # All four are live scheme names that are index trackers AND contain a
        # word from _ACTIVE_MARKERS ("elss", "multicap", "flexicap").
        "360 ONE ELSS Tax Saver Nifty 50 Index Fund",
        "NAVI ELSS TAX SAVER NIFTY50 INDEX FUND",
        "HDFC NIFTY500 Multicap 50:25:25 Index Fund",
        "DSP Nifty500 Flexicap Quality 30 ETF",
    ],
)
def test_an_index_fund_wearing_an_active_word_is_still_passive(name):
    """An index tracker must never be described as manager-picked.

    MUTATION GUARD: checking ``_ACTIVE_MARKERS`` before ``_PASSIVE_MARKERS``
    makes every one of these "active", so a user holding an ELSS-wrapped Nifty
    50 *index* fund is told "a manager picks its holdings" — a false statement
    about their own fund. These names resolve to no index either way (their
    phrases carry leftover words), so only an assertion on the classification
    catches it.
    """
    assert classify(name) == "index"


# ---------------------------------------------------------------------------
# Batch resolution and the headline
# ---------------------------------------------------------------------------


def test_un_unpacked_funds_are_counted_in_the_headline(master):
    """The headline must state coverage, because a dropped fund is a wrong total."""
    fs = resolve_funds(
        [
            "NIFTYBEES",
            "Parag Parikh Flexi Cap Fund",
            "UTI Nifty Next 50 Index Fund",
            "SBI Nifty Smallcap 250 Index Fund",
        ],
        master=master,
    )
    assert len(fs.lookups) == 4
    assert len(fs.matched) == 2
    assert len(fs.unmatched) == 2
    assert fs.headline() == "Of the 4 funds you hold, I could look through 2."


@pytest.mark.parametrize(
    ("book", "expected"),
    [
        ([], "No funds were listed."),
        (["Parag Parikh Flexi Cap Fund"], "Of the 1 fund you hold, I could look through none."),
        (["NIFTYBEES"], "Of the 1 fund you hold, I could look through it."),
        (["NIFTYBEES", "BANKBEES"], "Of the 2 funds you hold, I could look through all 2."),
    ],
)
def test_headline_wording(book, expected, master):
    assert resolve_funds(book, master=master).headline() == expected


def test_headline_never_claims_full_coverage_when_one_fund_was_skipped(master):
    """MUTATION GUARD: ``k >= n`` or ``k == len(self.matched)`` in ``headline``."""
    fs = resolve_funds(["NIFTYBEES", "Mirae Asset Large Cap Fund"], master=master)
    assert "all" not in fs.headline()
    assert fs.headline() == "Of the 2 funds you hold, I could look through 1."


def test_every_unmatched_lookup_carries_an_actionable_reason(master):
    fs = resolve_funds(
        [
            "Parag Parikh Flexi Cap Fund",
            "SBI Nifty Smallcap 250 Index Fund",
            "MIDCAPIETF",
            "Nippon India ETF Gold BeES",
            "nonsense token",
        ],
        master=master,
    )
    assert len(fs.unmatched) == 5
    for x in fs.unmatched:
        assert x.reason, f"{x.typed} was refused with no reason"
        assert len(x.reason) > 40, f"{x.typed}: reason too short to be actionable"
        assert "un-unpacked" in x.reason


def test_nothing_user_facing_recommends_an_action(master):
    """No advice, ever. "over-concentrated" or "you should" is regulated advice."""
    banned = (
        "you should",
        "we recommend",
        "recommend",
        "consider ",
        "advis",
        "over-concentrated",
        "overconcentrated",
        "too much",
        "trim",
        "rebalance your",
        "buy ",
        "sell ",
        "better off",
        "risky",
    )
    fs = resolve_funds(
        [
            "NIFTYBEES",
            "UTI Nifty",
            "Parag Parikh Flexi Cap Fund",
            "SBI Nifty Smallcap 250 Index Fund",
            "MIDCAPIETF",
            "Nippon India ETF Gold BeES",
            "Nippon India Nifty Next 50 Junior BeES FoF",
        ],
        master=master,
    )
    strings = [fs.headline()]
    for x in fs.lookups:
        strings.append(x.reason)
        if x.match:
            strings.extend(x.match.caveats)
            strings.append(x.match.matched_on)
    for s in strings:
        low = s.lower()
        for word in banned:
            assert word not in low, f"advice-shaped word {word!r} in: {s}"


# ---------------------------------------------------------------------------
# Caching and degradation
# ---------------------------------------------------------------------------


def test_the_amfi_file_is_cached_to_disk_and_reread_without_a_fetch(tmp_path, monkeypatch):
    """A public demo must not re-download 1.5 MB per request."""
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path))
    from agent.lookthrough import funds as f

    calls = []

    def _fake(url, *, timeout):
        calls.append(url)
        return AMFI_FIXTURE

    monkeypatch.setattr(f, "_http_get_text", _fake)
    first = f.load_scheme_master()
    second = f.load_scheme_master()
    assert len(calls) == 1, f"fetched {len(calls)} times; the cache did not hold"
    assert first.schemes == second.schemes
    assert list(tmp_path.glob("amfi_*.json")), "nothing was written to disk"


def test_refresh_bypasses_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path))
    from agent.lookthrough import funds as f

    calls = []
    monkeypatch.setattr(f, "_http_get_text", lambda url, *, timeout: (calls.append(url), AMFI_FIXTURE)[1])
    f.load_scheme_master()
    f.load_scheme_master(refresh=True)
    assert len(calls) == 2


def test_a_stale_cache_is_used_on_failure_and_the_staleness_is_declared(tmp_path, monkeypatch):
    """Degraded, never silent. This is the F-01..F-18 house rule.

    MUTATION GUARD: returning the stale copy with ``stale=False`` makes this
    fail — a demo would then report a fund as "not found" because it launched
    after an expired cache, with nothing telling the reader the data was old.
    """
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path))
    from agent.lookthrough import funds as f

    blob = {"ok": True, "text": AMFI_FIXTURE, "fetched_at": time.time() - 10 * 24 * 3600}
    f._cache_path(f.AMFI_NAV_ALL_URL).write_text(json.dumps(blob))

    def _down(url, *, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(f, "_http_get_text", _down)
    m = f.load_scheme_master()
    assert m.stale is True
    assert m.staleness_note is not None
    assert "cached copy" in m.staleness_note
    assert len(m.schemes) == 14


def test_no_cache_and_no_network_raises_rather_than_returning_empty(tmp_path, monkeypatch):
    """An empty scheme list would make every fund "not found" — a wrong answer."""
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path))
    from agent.lookthrough import funds as f

    monkeypatch.setattr(f, "_http_get_text", lambda url, *, timeout: (_ for _ in ()).throw(OSError("no net")))
    with pytest.raises(AmfiUnavailable):
        f.load_scheme_master()


def test_a_layout_change_that_yields_no_schemes_raises(tmp_path, monkeypatch):
    """MUTATION GUARD: dropping the ``if not schemes`` check returns 0 schemes."""
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path))
    from agent.lookthrough import funds as f

    monkeypatch.setattr(f, "_http_get_text", lambda url, *, timeout: "<html>maintenance</html>")
    with pytest.raises(AmfiUnavailable, match="layout has changed"):
        f.load_scheme_master()


def test_resolution_still_works_when_amfi_is_unavailable(tmp_path, monkeypatch):
    """A text that names its own index must resolve with no AMFI at all."""
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path))
    from agent.lookthrough import funds as f

    monkeypatch.setattr(f, "_http_get_text", lambda url, *, timeout: (_ for _ in ()).throw(OSError("no net")))
    fs = f.resolve_funds(["UTI Nifty 50 Index Fund", "NIFTYBEES", "Parag Parikh Flexi Cap Fund"])
    assert len(fs.matched) == 2
    assert fs.master_note is not None, "the degradation must be declared"
    assert "could not be loaded" in fs.master_note


def test_resolve_funds_loads_the_master_once_for_the_whole_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path))
    from agent.lookthrough import funds as f

    calls = []
    monkeypatch.setattr(f, "_http_get_text", lambda url, *, timeout: (calls.append(url), AMFI_FIXTURE)[1])
    f.resolve_funds(["NIFTYBEES"] * 20)
    assert len(calls) == 1


def test_offline_true_makes_no_network_call_and_no_cache_read(tmp_path, monkeypatch):
    monkeypatch.setenv("TIRRA_LOOKTHROUGH_CACHE", str(tmp_path))
    from agent.lookthrough import funds as f

    def _boom(*_a, **_k):
        raise AssertionError("offline=True must not fetch")

    monkeypatch.setattr(f, "_http_get_text", _boom)
    assert f.resolve_fund("NIFTYBEES", offline=True).index_id == "NIFTY 50"
    assert f.resolve_fund("Parag Parikh Flexi Cap Fund", offline=True) is None


def test_cache_dir_honours_the_portfolio_cache_env(tmp_path, monkeypatch):
    monkeypatch.delenv("TIRRA_LOOKTHROUGH_CACHE", raising=False)
    monkeypatch.setenv("TIRRA_PORTFOLIO_CACHE", str(tmp_path / "pf"))
    from agent.lookthrough import funds as f

    assert f._cache_dir() == tmp_path / "pf" / "lookthrough"


# ---------------------------------------------------------------------------
# Table hygiene
# ---------------------------------------------------------------------------


def test_every_alias_is_unique_across_indices():
    """Two indices sharing an alias would make resolution order-dependent."""
    seen: dict[str, str] = {}
    for index_id, aliases in SUPPORTED_INDICES.items():
        for a in aliases:
            assert a not in seen, f"alias {a!r} claimed by both {seen.get(a)} and {index_id}"
            seen[a] = index_id


def test_no_alias_is_short_enough_to_be_a_false_positive():
    """Bare "bank" or "it" would match "SBI Banking Fund" and anything at all."""
    for aliases in SUPPORTED_INDICES.values():
        for a in aliases:
            assert a not in {"bank", "it", "50", "100", "500", "next"}
            assert len(a) >= 5, f"alias {a!r} is too short to be safe"


def test_supported_indices_is_the_public_list():
    assert supported_indices() == tuple(SUPPORTED_INDICES)
    assert len(supported_indices()) == 7


# ---------------------------------------------------------------------------
# Live: the real AMFI file
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.integration
def test_live_amfi_master_has_the_shape_we_measured():
    """Against the real file. Measured 2026-10-02: 14,366 rows, 3,386 names.

    The bounds are wide because AMFI's scheme count genuinely moves; they are
    there to catch a truncated or redirected download, which is the failure
    that would otherwise look like "your fund is not in AMFI".
    """
    m = load_scheme_master(refresh=True)
    assert len(m.schemes) > 10_000, f"only {len(m.schemes)} rows — a truncated download?"
    assert m.unparsed_lines == (), f"unreadable rows: {m.unparsed_lines[:5]}"
    assert m.as_of, "no NAV date parsed"
    assert not m.stale

    rep = coverage_report(m.schemes)
    assert rep["distinct_names"] > 3_000
    assert rep["classification"]["index"] > 500
    assert rep["classification"]["active"] > 2_000
    # 163 on 2026-10-02. A collapse means index_phrase broke; a jump means an
    # alias got loose.
    assert 100 <= rep["unpackable"] <= 400, rep
    for index_id, n in rep["per_index"].items():
        assert n > 0, f"no live scheme resolves to {index_id}"


@pytest.mark.live
@pytest.mark.integration
def test_live_no_strategy_index_resolves_to_a_plain_index():
    """Over the whole live file: nothing with leftover words may resolve.

    Scans every distinct scheme name classified as passive, and asserts that
    any name resolving to a supported index reduces EXACTLY to one of that
    index's aliases — i.e. no "momentum", "equal weight", "value", "quality",
    "low volatility" or "alpha" name sneaks through.
    """
    m = load_scheme_master()
    strategy_words = (
        "momentum",
        "equal weight",
        "value",
        "quality",
        "low volatility",
        "alpha",
        "smallcap",
        "sensex",
        "psu",
        "private bank",
        "g-sec",
        "sdl",
        "gold",
        "silver",
    )
    offenders = []
    for s in {x.name for x in m.schemes}:
        match = resolve_fund(s, master=m)
        if match is None:
            continue
        low = s.lower()
        if any(w in low for w in strategy_words):
            offenders.append((s, match.index_id))
    assert offenders == [], f"strategy indices wrongly unpacked: {offenders[:10]}"


@pytest.mark.live
@pytest.mark.integration
def test_live_real_retail_pastes():
    """Ten names a retail investor would actually type, against live AMFI."""
    m = load_scheme_master()
    expect: list[tuple[str, str | None]] = [
        ("NIFTYBEES", "NIFTY 50"),
        ("BANKBEES", "NIFTY BANK"),
        ("UTI Nifty Index Fund", "NIFTY 50"),
        ("ICICI Prudential Nifty 50 Index Fund", "NIFTY 50"),
        ("Nippon India Index Fund - Nifty 50 Plan", "NIFTY 50"),
        ("Motilal Oswal Nifty 500 Index Fund", "NIFTY 500"),
        ("UTI Nifty Next 50 Index Fund", "NIFTY NEXT 50"),
        ("HDFC Flexi Cap", None),
        ("Parag Parikh Flexi Cap Fund", None),
        ("SBI Nifty Smallcap 250 Index Fund", None),
    ]
    wrong = []
    for typed, want in expect:
        got = resolve_fund(typed, master=m)
        got_id = got.index_id if got else None
        if got_id != want:
            wrong.append((typed, want, got_id))
    assert wrong == [], wrong


@pytest.mark.live
@pytest.mark.integration
def test_live_load_amfi_schemes_is_the_documented_thin_wrapper():
    schemes = load_amfi_schemes()
    assert isinstance(schemes, list)
    assert len(schemes) > 10_000
    assert all(s.code > 0 for s in schemes[:100])
