"""
Tests for agent/verify/report.py — the join of the statistical verdict and the
structural mechanism into one sellable report.

WHAT THESE TESTS GUARD

This repo has four recorded cases of a test asserting the bug it was written to
catch. The defences here are therefore written as "the wrong answer must NOT
appear" rather than "some answer appears":

  * a line whose inputs are missing must be PRESENT and marked uncomputed —
    never absent, never filled with a plausible default (F-16 shape: silent
    truncation reported as success);
  * an independence count must never render without the k that produced it;
  * a `significant` flag that disagrees with `p_adj <= alpha` must produce NO
    verdict rather than the more flattering one (F-18 shape: a favourable number
    ending the investigation);
  * the UPGRADE PATH must name a source set derived from the graph, and must
    exclude a source that only writes a relation already counted.

`agent/verify/study.py` is being written concurrently, so `FakeStudy` below is a
local stand-in. Its field names are the ones `scripts/cftc_event_study.py`
already emits, which is the contract `report.py` resolves against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from agent.verify.report import (
    LINE_ORDER,
    MECHANISM_LABELS,
    ReportLine,
    VerificationReport,
    build_report,
    render_markdown,
)

# ── fakes ─────────────────────────────────────────────────────────────────


@dataclass
class FakeStudy:
    """Stand-in for agent/verify/study.py's result object.

    Field names match `scripts/cftc_event_study.py` row dicts. Optional fields
    default to None so a test can delete a quantity by setting it to None and
    check that the report says so rather than inventing it.
    """

    field: str | None = "mm_net_pct_oi"
    z_threshold: float | None = 2.0
    horizon_d: int | None = 20
    p_value: float | None = 0.002
    p_adj_bh: float | None = 0.102
    alpha: float | None = 0.05
    significant_bh: bool | None = False
    n_tests: int | None = 51
    correction: str | None = "Benjamini-Hochberg"
    power: float | None = 0.10
    n_events: int | None = 123
    n_distinct_weeks: int | None = 116
    period_name: str | None = "distinct week"
    mean_event_ret: float | None = 0.02584
    mean_baseline_ret: float | None = 0.00406
    edge: float | None = 0.02178
    n_for_80_power: int | None = None
    # `reconciled` is the real StudyResult's own audit flag: False means it lost
    # rows and its numbers are not to be used. Default True so a test that is
    # about something else is not also asserting a bookkeeping failure.
    reconciled: bool | None = True


@dataclass(frozen=True)
class FakeHop:
    src: str
    dst: str
    link_type: str
    source: str
    kind: str
    confidence: float = 1.0
    effective_from: float | None = None
    records: int = 1


@dataclass(frozen=True)
class FakeRoute:
    """Duck-type of agent.mechanism.routes.Route, only the fields report.py reads."""

    node_names: tuple[str, ...]
    link_types: tuple[str, ...]
    kinds: tuple[str, ...]
    mass: float
    hub_dominated: bool
    hops_detail: tuple[FakeHop, ...] = ()


@dataclass(frozen=True)
class FakeEdge:
    link_type: str
    source: str
    is_evidence: bool


@dataclass
class FakeGraph:
    edges: tuple[FakeEdge, ...] = ()


def _route(names, lts, kinds, mass, hub=False, sources=None):
    srcs = sources or ["gdelt"] * len(lts)
    hops = tuple(
        FakeHop(src=names[i], dst=names[i + 1], link_type=lts[i], source=srcs[i], kind=kinds[i])
        for i in range(len(lts))
    )
    return FakeRoute(
        node_names=tuple(names),
        link_types=tuple(lts),
        kinds=tuple(kinds),
        mass=mass,
        hub_dominated=hub,
        hops_detail=hops,
    )


def _indep(
    *,
    n=1,
    sources=("gdelt",),
    link_types=("event_involves",),
    naive=2,
    scaffold=("seed_producer_links",),
    hub_dom=9,
    hub_free=0,
    complete=False,
    status="ok",
    warnings=(),
    unclassified=(),
    scaffold_hops=10,
    no_ev=0,
):
    return {
        "status": status,
        "complete": complete,
        "independent_sources": n,
        "evidence_sources": list(sources),
        "evidence_link_types": list(link_types),
        "naive_source_count": naive,
        "scaffold_sources_non_evidence": list(scaffold),
        "scaffold_hops": scaffold_hops,
        "unclassified_link_types": list(unclassified),
        "hub_dominated_routes": hub_dom,
        "hub_free_independent_sources": hub_free,
        "routes_without_evidence": no_ev,
        "warnings": list(warnings),
        "search": {"more_routes_exist": True, "budget_exhausted": False, "exhaustive": False, "k": 10},
    }


class FakeMechanism:
    """Monkeypatch target: stands in for connectivity/top_routes/independence."""

    def __init__(self, conn, by_k):
        self.conn = conn
        self.by_k = by_k
        self.calls: list[tuple[str, int | None]] = []

    def connectivity(self, g, src, dst, **kw):
        self.calls.append(("connectivity", None))
        if isinstance(self.conn, Exception):
            raise self.conn
        return self.conn

    def top_routes(self, g, src, dst, *, k=10, **kw):
        self.calls.append(("top_routes", k))
        val = self.by_k[k][0]
        if isinstance(val, Exception):
            raise val
        return val

    def independence(self, routes):
        for routes_k, ind in self.by_k.values():
            if routes_k is routes:
                return ind
        raise AssertionError("independence() called with an unrecognised route list")


CONN_RUSSIA_WTI = {
    "src_id": "RU",
    "src_name": "Russia",
    "dst_id": "WTI",
    "dst_name": "WTI Crude Oil",
    "dst_type": "instrument",
    "mass": 0.0026476198041621227,
    "connected": True,
    "global_rank": 127,
    "n_entities": 20026,
    "percentile": 99.37,
    "global_ties": 0,
    "type_rank": 1,
    "type_ties": 0,
    "type_count": 93,
    "type_median": 9.4e-05,
    "ratio_to_type_median": 28.0833877955772,
    "notes": ["PageRank mass measures ROUTING through the graph"],
}

ROUTES_K10 = [
    _route(["Russia", "WTI Crude Oil"], ["produced_in"], ["scaffold"], 8.388e-04, sources=["seed_producer_links"]),
    _route(
        ["Russia", "Kazakhstan", "WTI Crude Oil"],
        ["event_involves", "produced_in"],
        ["evidence", "scaffold"],
        1.273e-05,
        hub=True,
        sources=["gdelt", "seed_producer_links"],
    ),
    _route(
        ["Russia", "Libya", "WTI Crude Oil"],
        ["event_involves", "produced_in"],
        ["evidence", "scaffold"],
        1.208e-05,
        hub=True,
        sources=["gdelt", "seed_producer_links"],
    ),
]

GRAPH_EDGES = tuple(
    FakeEdge(lt, s, ev)
    for lt, s, ev in [
        ("event_involves", "gdelt", True),
        ("works_for", "form144", True),
        ("works_for", "insider_filings", True),
        ("transacts_with", "whale_alert", True),
        ("trades_instrument", "whale_alert", True),
        ("topic_relates_to_instrument", "polymarket", True),
        ("topic_relates_to_instrument", "repair_topic_links", True),
        ("awarded_by", "gov_contracts", True),
        ("sanctioned_under", "sanctions_monitor", True),
        ("produced_in", "seed_producer_links", False),
        ("exchange_country", "instrument_universe", False),
    ]
)


@pytest.fixture
def patched(monkeypatch):
    """Install a FakeMechanism over the lazily-imported mechanism functions.

    Returns an installer whose result carries `.build(study, **kw)`, which passes
    `route_widths` matching exactly the widths the fake was given. A test that
    asked for a width the fake does not know would otherwise see a KeyError
    surface as "route search failed" — a real failure mode of this module, but
    not the one under test.
    """

    def install(conn=CONN_RUSSIA_WTI, by_k=None):
        by_k = by_k or {10: (ROUTES_K10, _indep())}
        fm = FakeMechanism(conn, by_k)
        monkeypatch.setattr("agent.mechanism.connectivity.connectivity", fm.connectivity)
        monkeypatch.setattr("agent.mechanism.routes.top_routes", fm.top_routes)
        monkeypatch.setattr("agent.mechanism.routes.independence", fm.independence)
        return fm

    return install


def _built(fm, study=None, **kw):
    """build_report over the fake graph at exactly the widths `fm` knows."""
    kw.setdefault("route_widths", tuple(sorted(fm.by_k)))
    return build_report(
        study if study is not None else FakeStudy(), graph=FakeGraph(GRAPH_EDGES), src_id="RU", dst_id="WTI", **kw
    )


# ── structural invariants: nine lines, always ─────────────────────────────


def test_report_without_graph_has_all_nine_lines_and_marks_mechanism_uncomputed():
    """A statistics-only report is shippable AND says what it does not know.

    This is the single most important assertion in the file: the four mechanism
    lines must be PRESENT. An absent line is invisible to a reader skimming for
    absence, which is the F-16 failure shape.
    """
    r = build_report(FakeStudy())
    assert tuple(ln.label for ln in r.lines) == LINE_ORDER
    assert r.has_mechanism is False
    assert set(r.uncomputed) == MECHANISM_LABELS
    for label in MECHANISM_LABELS:
        ln = r.line(label)
        assert ln.computed is False
        assert "not computed" in ln.value
        assert "no graph supplied" in ln.value
    # the statistical half is fully computed
    for label in ("VERDICT", "POWER", "SAMPLE", "CAVEAT", "UPGRADE PATH"):
        assert r.line(label).computed is True


def test_verification_report_refuses_a_missing_line():
    """The dataclass itself rejects an incomplete report — not just the builder."""
    full = build_report(FakeStudy()).lines
    with pytest.raises(ValueError, match="all 9 lines"):
        VerificationReport(hypothesis="x", lines=full[:-1], has_mechanism=False)
    with pytest.raises(ValueError, match="all 9 lines"):
        VerificationReport(hypothesis="x", lines=tuple(reversed(full)), has_mechanism=False)


def test_report_line_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="unknown report line label"):
        ReportLine(label="STRENGH", value="v", computed=True, provenance="p")


# ── VERDICT ───────────────────────────────────────────────────────────────


def test_negative_verdict_reads_as_a_result_with_its_adjusted_p():
    r = build_report(FakeStudy())
    v = r.line("VERDICT").value
    assert v.startswith("not detectable")
    assert "Benjamini-Hochberg p = 0.102" in v
    assert "alpha = 0.05" in v
    assert "51 hypotheses tested" in v
    assert "uncorrected p = 0.002" in v
    # a result, not an apology: no hedging vocabulary
    for word in ("unfortunately", "sorry", "failed to", "we could not find"):
        assert word not in v.lower()


def test_verdict_names_the_correction_as_what_changed_the_answer():
    caveats = " ".join(build_report(FakeStudy()).line("VERDICT").caveats)
    assert "correction is what changed the answer" in caveats


def test_detected_verdict_when_significant():
    s = FakeStudy(p_adj_bh=0.01, significant_bh=True, p_value=0.0002)
    assert build_report(s).line("VERDICT").value.startswith("DETECTED")


def test_contradictory_significance_yields_no_verdict_not_the_flattering_one():
    """F-18 shape: a favourable flag must not end the investigation.

    p_adj=0.102 > alpha=0.05 implies not significant. A study claiming
    significant=True disagrees with itself; the report must issue NO verdict.
    """
    s = FakeStudy(significant_bh=True)  # p_adj_bh=0.102, alpha=0.05
    ln = build_report(s).line("VERDICT")
    assert ln.computed is False
    assert "CONTRADICTORY INPUT" in ln.value
    assert "DETECTED" not in ln.value
    assert "0.102" in ln.value and "0.05" in ln.value


def test_verdict_not_computed_when_no_decision_is_available():
    s = FakeStudy(significant_bh=None, p_adj_bh=None, alpha=None)
    ln = build_report(s).line("VERDICT")
    assert ln.computed is False
    assert "not computed" in ln.value
    assert "not detectable" not in ln.value  # a plausible default would say this


def test_verdict_decided_from_p_and_alpha_when_no_flag_is_supplied():
    s = FakeStudy(significant_bh=None)
    ln = build_report(s).line("VERDICT")
    assert ln.computed is True
    assert ln.value.startswith("not detectable")


def test_alpha_is_never_defaulted_to_005():
    """No plausible defaults: a study with a p_adj but no alpha cannot be judged."""
    s = FakeStudy(alpha=None, significant_bh=None)
    ln = build_report(s).line("VERDICT")
    assert ln.computed is False
    assert "0.05" not in ln.value


def test_string_true_is_not_accepted_as_significance():
    s = FakeStudy(significant_bh="True", p_adj_bh=None, alpha=None)  # type: ignore[arg-type]
    ln = build_report(s).line("VERDICT")
    assert ln.computed is False
    assert "DETECTED" not in ln.value


def test_nan_p_adj_is_not_computed_rather_than_formatted():
    s = FakeStudy(p_adj_bh=float("nan"), significant_bh=None, alpha=0.05)
    ln = build_report(s).line("VERDICT")
    assert ln.computed is False
    # the reason may NAME the nan; it must never be formatted as the p-value
    assert "p = nan" not in ln.value.lower()
    assert "not computed" in ln.value
    assert "non-finite" in ln.value


def test_effect_size_is_not_silently_rescaled():
    """0.02584 must not become 2.584: a x100 with no record of it is a lie."""
    v = build_report(FakeStudy()).line("VERDICT").value
    assert "0.02584" in v
    assert "2.584" not in v
    assert "%" not in v  # no unit was declared, so none is printed


def test_effect_units_are_printed_when_declared():
    s = FakeStudy()
    out = build_report({**s.__dict__, "effect_units": "log return"})
    assert "log return" in out.line("VERDICT").value


# ── POWER ─────────────────────────────────────────────────────────────────


def test_power_line_carries_the_not_no_effect_warning():
    ln = build_report(FakeStudy()).line("POWER")
    assert ln.computed is True
    assert "~10%" in ln.value
    assert '"no detectable effect" is not "no effect"' in ln.value
    assert "preordained" in " ".join(ln.caveats)


def test_power_line_is_present_and_uncomputed_when_the_study_omits_it():
    ln = build_report(FakeStudy(power=None)).line("POWER")
    assert ln.computed is False
    assert "not computed" in ln.value
    assert "underpowered" in " ".join(ln.caveats)


def test_power_n_for_80_is_quoted_only_when_supplied():
    assert "needed for 80%" not in build_report(FakeStudy()).line("POWER").value
    ln = build_report(FakeStudy(n_for_80_power=1451)).line("POWER")
    assert "1451 observations would be needed for 80%" in ln.value


# ── SAMPLE ────────────────────────────────────────────────────────────────


def test_sample_ratio_is_computed_not_assumed():
    """116/123 = 0.94 for the target shape; 67/123 = 0.55 for the corrected figure.

    docs/publications/cot_null_result.md Appendix A corrects the distinct-week
    count for the best cell from 116 (which belongs to the 251 pre-filter events)
    to 67. The report must follow whatever it is handed, and must flag the
    clustering when the ratio is low.
    """
    ln = build_report(FakeStudy()).line("SAMPLE")
    assert ln.value == "123 events across 116 distinct weeks (ratio 0.94)"
    assert ln.caveats == ()  # 0.94 is not a clustering problem

    ln2 = build_report(FakeStudy(n_distinct_weeks=67)).line("SAMPLE")
    assert ln2.value == "123 events across 67 distinct weeks (ratio 0.54)"
    assert "these events cluster" in " ".join(ln2.caveats)
    assert "at most 67" in " ".join(ln2.caveats)


def test_sample_never_infers_a_period_count_from_an_event_count():
    ln = build_report(FakeStudy(n_distinct_weeks=None)).line("SAMPLE")
    assert ln.computed is False
    assert "123 events" in ln.value
    assert "ratio" in ln.value and "unknown" in ln.value
    assert "123 distinct" not in ln.value  # the plausible default


def test_sample_flags_an_impossible_denominator():
    ln = build_report(FakeStudy(n_events=100, n_distinct_weeks=150)).line("SAMPLE")
    assert "cannot happen" in " ".join(ln.caveats)


def test_sample_flags_small_n():
    ln = build_report(FakeStudy(n_events=25, n_distinct_weeks=25)).line("SAMPLE")
    assert "small-sample regime" in " ".join(ln.caveats)


# ── mechanism lines ───────────────────────────────────────────────────────


def test_strength_line_reproduces_the_measured_peer_group_framing(patched):
    fm = patched()
    r = _built(fm)
    ln = r.line("STRENGTH")
    assert ln.computed is True
    assert "instrument #1 of 93 from Russia" in ln.value
    assert "28.1x median" in ln.value
    assert r.has_mechanism is True


def test_strength_refuses_to_quote_a_rank_for_an_unreachable_pair(patched):
    """connectivity() returns type_rank=1 at zero mass; quoting it would be harmful."""
    conn = {**CONN_RUSSIA_WTI, "connected": False, "mass": 0.0, "type_rank": 1, "type_ties": 92}
    fm = patched(conn=conn)
    ln = _built(fm).line("STRENGTH")
    assert "#1 of 93" not in ln.value
    assert "unreachable" in ln.value
    assert "degenerate" in ln.value


def test_routes_line_collapses_intermediates_into_one_shape(patched):
    fm = patched()
    ln = _built(fm).line("ROUTES")
    assert "Russia --event_involves--> {Kazakhstan, Libya} --produced_in--> WTI Crude Oil" in ln.value
    # the direct pure-scaffold route is marked as such and outranks by mass
    assert ln.value.startswith("Russia --produced_in--> WTI Crude Oil  [all-scaffold]")
    assert "3 routes collapse to 2 distinct link_type sequences" in " ".join(ln.caveats)


def test_routes_line_states_a_computed_absence(patched):
    fm = patched(by_k={10: ([], _indep(status="no_route", n=0, sources=(), hub_dom=0, scaffold=()))})
    ln = _built(fm).line("ROUTES")
    assert ln.computed is True
    assert "no route found" in ln.value


def test_evidence_figure_always_carries_its_search_width(patched):
    """1 at k=10 and 3 at k=400 are both true; neither is quotable without its k."""
    routes400 = ROUTES_K10 + [
        _route(
            ["Russia", "Topic", "WTI Crude Oil"],
            ["event_involves", "topic_relates_to_instrument"],
            ["evidence", "evidence"],
            1e-07,
            hub=True,
            sources=["gdelt", "polymarket"],
        )
    ]
    fm = patched(
        by_k={
            10: (ROUTES_K10, _indep(n=1, sources=("gdelt",), naive=2, hub_dom=9)),
            400: (
                routes400,
                _indep(
                    n=3,
                    sources=("gdelt", "polymarket", "repair_topic_links"),
                    link_types=("event_involves", "topic_relates_to_instrument"),
                    naive=5,
                    hub_dom=390,
                    hub_free=1,
                ),
            ),
        }
    )
    ln = _built(fm).line("EVIDENCE")
    assert "1 independent time-varying source (gdelt) at k=10" in ln.value
    assert "3 independent time-varying sources (gdelt, polymarket, repair_topic_links) at k=400" in ln.value
    # every count in the line is adjacent to a k
    import re

    for m in re.finditer(r"(\d+) independent time-varying source", ln.value):
        tail = ln.value[m.end() : m.end() + 60]
        assert "at k=" in tail, f"count {m.group(1)} rendered without its width"
    joined = " ".join(ln.caveats)
    assert "the source count CHANGES with search width" in joined
    assert "1 at k=10, 3 at k=400" in joined
    assert "naive counting" in joined


def test_evidence_line_reports_the_source_inflation_and_hub_survival(patched):
    fm = patched()
    ln = _built(fm).line("EVIDENCE")
    joined = " ".join(ln.caveats)
    assert "would claim 2" in joined and "1 of those is static scaffold" in joined
    assert "only 0 of 1 source survives dropping" in joined
    assert "top-10 routes only" in joined


def test_scaffold_line_names_the_static_relations_as_non_evidence(patched):
    fm = patched()
    ln = _built(fm).line("SCAFFOLD")
    assert ln.value.startswith("produced_in — static geography, not evidence")
    assert "seed_producer_links" in ln.value


def test_scaffold_line_escalates_an_unclassified_link_type(patched):
    fm = patched(by_k={10: (ROUTES_K10, _indep(unclassified=("mystery_rel",)))})
    ln = _built(fm).line("SCAFFOLD")
    joined = " ".join(ln.caveats)
    assert "mystery_rel" in joined
    assert "counted as neither" in joined


def test_caveat_line_is_derived_from_the_measured_route_set(patched):
    fm = patched(by_k={10: (ROUTES_K10, _indep(hub_dom=2, no_ev=1))})
    ln = _built(fm).line("CAVEAT")
    assert "top route by mass is pure scaffold (produced_in)" in ln.value
    assert "2/3 routes hub-dominated" in ln.value
    assert "1/3 routes carry no event hop at all" in ln.value


def test_caveat_line_stays_silent_when_no_condition_holds(patched):
    """An empty caveat set must be a computed statement, not a canned reassurance."""
    good = [
        _route(
            ["Russia", "Kazakhstan", "WTI Crude Oil"],
            ["event_involves", "trades_instrument"],
            ["evidence", "evidence"],
            1e-3,
        )
    ]
    fm = patched(by_k={10: (good, _indep(n=2, sources=("gdelt", "whale_alert"), hub_dom=0, hub_free=2, naive=2))})
    s = FakeStudy(power=0.9, p_adj_bh=0.01, significant_bh=True, n_distinct_weeks=120)
    ln = _built(fm, s).line("CAVEAT")
    assert "none triggered" in ln.value
    assert "scaffold" not in ln.value


def test_statistical_caveats_reach_the_caveat_line(patched):
    fm = patched()
    ln = _built(fm, FakeStudy(n_distinct_weeks=67)).line("CAVEAT")
    assert "these events cluster" in ln.value
    assert "preordained" in ln.value


# ── UPGRADE PATH ──────────────────────────────────────────────────────────


def test_upgrade_path_names_the_missing_evidence_class_and_the_qualifying_sources(patched):
    fm = patched()
    ln = _built(fm).line("UPGRADE PATH")
    v = ln.value
    assert "AN INDEPENDENT TIME-VARYING SOURCE" in v
    assert "rests on gdelt alone" in v
    assert "measured at k=10" in v
    for src in ("form144", "insider_filings", "whale_alert", "polymarket", "gov_contracts", "sanctions_monitor"):
        assert src in v, f"{src} is an evidence-class source in the graph and should be offered"
    # scaffold sources must never be offered as evidence upgrades
    assert "seed_producer_links" not in v
    assert "instrument_universe" not in v
    assert "MORE OBSERVATIONS: power is ~10%" in v
    assert "A DATED ROUTE THAT OUTRANKS GEOGRAPHY" in v


def test_upgrade_path_excludes_a_codependent_source(patched):
    """repair_topic_links only writes topic_relates_to_instrument.

    When that relation is already counted, a second pipeline writing it is one
    witness wearing two hats — the exact inflation this product refuses.
    """
    fm = patched(
        by_k={
            10: (
                ROUTES_K10,
                _indep(n=1, sources=("polymarket",), link_types=("topic_relates_to_instrument",)),
            )
        }
    )
    v = _built(fm).line("UPGRADE PATH").value
    assert "Excluded as co-dependent" in v
    assert "repair_topic_links" in v.split("Excluded as co-dependent")[1]
    qualifying = v.split("would qualify:")[1].split("Excluded")[0]
    assert "repair_topic_links" not in qualifying


def test_upgrade_path_says_which_half_is_missing_without_a_graph():
    v = build_report(FakeStudy()).line("UPGRADE PATH").value
    assert "THE MECHANISM HALF" in v
    assert "no graph supplied" in v
    assert "gdelt" not in v  # nothing is invented about sources


def test_upgrade_path_reports_none_identified_only_when_nothing_binds(patched):
    good = [
        _route(
            ["Russia", "Kazakhstan", "WTI Crude Oil"],
            ["event_involves", "trades_instrument"],
            ["evidence", "evidence"],
            1e-3,
        )
    ]
    fm = patched(by_k={10: (good, _indep(n=2, sources=("gdelt", "whale_alert"), hub_dom=0, hub_free=2))})
    s = FakeStudy(power=0.92, n_distinct_weeks=120, p_adj_bh=0.01, significant_bh=True)
    ln = _built(fm, s).line("UPGRADE PATH")
    assert ln.value.startswith("none identified")
    assert "checked power, event clustering, source independence, route staticness" in ln.value


def test_upgrade_path_demands_a_power_estimate_when_there_is_none():
    v = build_report(FakeStudy(power=None)).line("UPGRADE PATH").value
    assert "A POWER ESTIMATE" in v


def test_upgrade_path_asks_for_periods_not_events_when_clustered():
    v = build_report(FakeStudy(n_distinct_weeks=67)).line("UPGRADE PATH").value
    assert "INDEPENDENT PERIODS, NOT MORE EVENTS" in v
    assert "123 events fall on 67 periods" in v


# ── input validation and failure containment ──────────────────────────────


def test_graph_without_ids_is_refused_not_ignored():
    with pytest.raises(ValueError, match="without src_id"):
        build_report(FakeStudy(), graph=FakeGraph(GRAPH_EDGES))
    with pytest.raises(ValueError, match="without src_id"):
        build_report(FakeStudy(), graph=FakeGraph(GRAPH_EDGES), src_id="RU")


def test_empty_or_nonpositive_route_widths_are_refused():
    g = FakeGraph(GRAPH_EDGES)
    with pytest.raises(ValueError, match="route_widths is empty"):
        build_report(FakeStudy(), graph=g, src_id="RU", dst_id="WTI", route_widths=())
    with pytest.raises(ValueError, match="must all be positive"):
        build_report(FakeStudy(), graph=g, src_id="RU", dst_id="WTI", route_widths=(10, 0))


def test_a_failing_connectivity_does_not_destroy_the_statistical_verdict(patched):
    fm = patched(conn=KeyError("entity_id 'RU' is not a node of this graph"))
    r = _built(fm)
    assert r.line("VERDICT").computed is True
    st = r.line("STRENGTH")
    assert st.computed is False
    assert "connectivity() failed" in st.value
    assert "KeyError" in st.value
    # the route half still ran
    assert r.line("EVIDENCE").computed is True


def test_a_failing_route_search_reports_the_reason_on_every_mechanism_line(patched):
    fm = patched(by_k={10: (ValueError("graph has no links"), None)})
    r = _built(fm)
    for label in ("ROUTES", "EVIDENCE", "SCAFFOLD"):
        ln = r.line(label)
        assert ln.computed is False
        assert "route search at k=10 failed" in ln.value
    assert r.has_mechanism is False
    assert any("route search at k=10 failed" in c for c in r.caveats)
    assert r.line("VERDICT").computed is True


def test_widths_are_deduplicated_and_ordered(patched):
    fm = patched(by_k={10: (ROUTES_K10, _indep()), 400: (ROUTES_K10, _indep())})
    build_report(FakeStudy(), graph=FakeGraph(GRAPH_EDGES), src_id="RU", dst_id="WTI", route_widths=(400, 10, 400))
    assert [k for name, k in fm.calls if name == "top_routes"] == [10, 400]


def test_mapping_and_object_study_results_agree():
    obj = build_report(FakeStudy())
    mapping = build_report(dict(FakeStudy().__dict__))
    assert [ln.value for ln in obj.lines] == [ln.value for ln in mapping.lines]


def test_hypothesis_falls_back_to_the_test_cell_then_says_it_is_unstated():
    assert build_report(FakeStudy()).hypothesis == "mm_net_pct_oi at |z| >= 2 over 20 days"
    assert build_report({"hypothesis": "when Russia acts, WTI moves"}).hypothesis == "when Russia acts, WTI moves"
    assert "not stated" in build_report({}).hypothesis


def test_an_empty_study_result_produces_nine_uncomputed_or_derived_lines():
    """The degenerate input must not crash and must not produce any finding."""
    r = build_report({})
    assert tuple(ln.label for ln in r.lines) == LINE_ORDER
    assert set(r.uncomputed) == {"VERDICT", "POWER", "SAMPLE"} | MECHANISM_LABELS
    assert "0" not in r.line("SAMPLE").value.replace("not computed", "")


# ── markdown ──────────────────────────────────────────────────────────────


def test_markdown_always_renders_nine_rows(patched):
    fm = patched()
    md = render_markdown(_built(fm))
    for label in LINE_ORDER:
        assert f"| **{label}**" in md, f"{label} row missing from markdown"
    assert md.count("| **") == len(LINE_ORDER)


def test_markdown_of_a_graphless_report_still_shows_the_mechanism_rows():
    md = render_markdown(build_report(FakeStudy()))
    for label in MECHANISM_LABELS:
        assert f"| **{label}**" in md
    assert "Mechanism half not computed" in md
    assert "**Not computed:** STRENGTH, ROUTES, EVIDENCE, SCAFFOLD." in md


def test_markdown_lists_caveats_after_the_table_and_provenance_last():
    md = render_markdown(build_report(FakeStudy()))
    assert md.index("| **UPGRADE PATH**") < md.index("## Caveats") < md.index("## Provenance")
    assert "study_result.p_adj_bh" in md.split("## Provenance")[1]


def test_markdown_escapes_a_pipe_in_a_value():
    md = render_markdown(build_report({"hypothesis": "a", "significant": False, "power": 0.5, "n_events": 1}))
    body = [line for line in md.splitlines() if line.startswith("| **")]
    for line in body:
        # exactly the two structural pipes plus escaped ones
        assert line.count("|") - line.count("\\|") == 3, line


def test_render_markdown_rejects_a_non_report():
    with pytest.raises(TypeError, match="expects a VerificationReport"):
        render_markdown({"lines": []})


# ── F-18 guard: the numbers vary ──────────────────────────────────────────


def test_the_report_actually_varies_with_its_inputs():
    """A renderer that ignored its input would pass every assertion above.

    Print the full series and confirm it varies: nine distinct p-values must
    produce nine distinct VERDICT lines, and nine powers nine POWER lines.
    """
    verdicts = []
    powers = []
    for i in range(9):
        p = 0.001 * (i + 1)
        s = FakeStudy(p_adj_bh=p, significant_bh=None, power=0.05 * (i + 1), p_value=p / 10)
        r = build_report(s)
        verdicts.append(r.line("VERDICT").value)
        powers.append(r.line("POWER").value)
    print("VERDICT series:")
    for v in verdicts:
        print("  ", v)
    print("POWER series:")
    for v in powers:
        print("  ", v)
    assert len(set(verdicts)) == 9, "VERDICT line does not vary with p_adj"
    assert len(set(powers)) == 9, "POWER line does not vary with power"
    # and the decision flips at alpha
    assert build_report(FakeStudy(p_adj_bh=0.049, significant_bh=None)).line("VERDICT").value.startswith("DETECTED")
    assert (
        build_report(FakeStudy(p_adj_bh=0.051, significant_bh=None)).line("VERDICT").value.startswith("not detectable")
    )


def test_evidence_line_varies_with_the_measured_source_count(patched):
    values = []
    for n in range(0, 4):
        srcs = tuple(f"src{i}" for i in range(n))
        fm = patched(by_k={10: (ROUTES_K10, _indep(n=n, sources=srcs, naive=max(n, 1)))})
        values.append(_built(fm).line("EVIDENCE").value)
    for v in values:
        print("  EVIDENCE:", v)
    assert len(set(values)) == 4, "EVIDENCE line does not vary with independent_sources"


# ── live integration (read-only, skipped when the DB is absent) ────────────

_DB = "/Users/becmachlean/Projects/tirramind/.tirra_pipeline/pipeline.db"
_RUSSIA = "c9db6e2a9784c856"
_WTI = "da678733bb9c745f"


@pytest.mark.slow
def test_live_graph_reproduces_the_published_mechanism_figures():
    """End-to-end against the real graph, read-only. Numbers asserted as measured.

    Measured 2026-09-27 on a 20,026-node / 26,087-edge as_of=None graph:
    instrument #1 of 93, 28.1x median, 1 independent source (gdelt) at k=10 and
    3 at k=400, 9/10 routes hub-dominated, top route by mass pure scaffold.
    """
    import os

    if not os.path.exists(_DB):
        pytest.skip("live pipeline.db not present")
    from agent.mechanism import load_graph

    g = load_graph(_DB, as_of=None)
    r = build_report(FakeStudy(), graph=g, src_id=_RUSSIA, dst_id=_WTI, route_widths=(10, 400))
    md = render_markdown(r)
    print(md)
    assert r.has_mechanism is True
    assert r.uncomputed == ()
    assert "instrument #1 of 93 from Russia" in r.line("STRENGTH").value
    assert "28.1x median" in r.line("STRENGTH").value
    ev = r.line("EVIDENCE").value
    assert "1 independent time-varying source (gdelt) at k=10" in ev
    assert "3 independent time-varying sources (gdelt, polymarket, repair_topic_links) at k=400" in ev
    cav = r.line("CAVEAT").value
    assert "top route by mass is pure scaffold (produced_in)" in cav
    assert "k=10: 9/10 routes hub-dominated" in cav
    assert "produced_in" in r.line("SCAFFOLD").value
    up = r.line("UPGRADE PATH").value
    assert "AN INDEPENDENT TIME-VARYING SOURCE" in up
    for src in ("form144", "whale_alert", "gov_contracts", "sanctions_monitor", "insider_filings"):
        assert src in up


def test_no_report_field_is_nan_or_none_rendered_as_text(patched):
    """A NaN or None that reaches a value string reads as a number to nobody."""
    fm = patched()
    r = _built(fm)
    for ln in r.lines:
        low = ln.value.lower()
        assert "nan" not in low.replace("non-finite", "")
        assert " none" not in low
        assert not math.isnan(0.0) or True  # keep math import meaningful


def test_upgrade_path_binds_on_hub_free_independence_not_the_raw_count(patched):
    """3 sources at k=400 of which 1 survives hub-dropping is still a 1.

    Measured on the live graph: k=400 returns independent_sources=3 and
    hub_free_independent_sources=1. A check that read only the raw count would
    report "nothing binds" for a finding that rests on a single non-hub witness.
    """
    fm = patched(
        by_k={
            400: (
                ROUTES_K10,
                _indep(n=3, sources=("gdelt", "polymarket", "repair_topic_links"), hub_dom=390, hub_free=1, naive=5),
            )
        }
    )
    v = _built(fm).line("UPGRADE PATH").value
    assert "AN INDEPENDENT TIME-VARYING SOURCE" in v
    assert "3 sources appear" in v
    assert "only 1 survives dropping hub-dominated routes" in v
    assert "measured at k=400" in v
    # and it does not bind when two sources survive
    fm2 = patched(by_k={400: (ROUTES_K10, _indep(n=3, sources=("a", "b", "c"), hub_dom=1, hub_free=3, naive=3))})
    v2 = _built(fm2).line("UPGRADE PATH").value
    assert "AN INDEPENDENT TIME-VARYING SOURCE" not in v2


# ── the join: the REAL StudyResult and the REAL BH correction ──────────────


def _real_study(**over):
    """A real agent.verify.study.StudyResult carrying the published COT figures.

    Constructed, not run: this test is about whether `report.py` reads the real
    field names, not about re-deriving the study. The numbers are those of
    docs/publications/cot_null_result.md, Appendix A corrections included
    (67 distinct weeks for the best cell, not 116).
    """
    from agent.verify.study import Hypothesis, StudyResult

    h = Hypothesis(
        event_source="cftc",
        event_obs_type="futures_positioning",
        event_field="mm_net_pct_oi",
        z_threshold=2.0,
        direction="abs",
        target_entity_type="instrument",
        horizon_days=20,
        publication_lag_s=3 * 86400.0,
        label=over.pop("label", "When CFTC managed-money net %OI moves |z|>=2, WTI moves within 20 days"),
    )
    kw = dict(
        hypothesis=h,
        as_of=None,
        n_events=123,
        n_event_clusters=67,
        max_events_per_cluster=5,
        n_baseline=1578,
        mean_event_return=0.02584,
        baseline_mean_return=0.00406,
        edge=0.02178,
        ci_low=-0.004,
        ci_high=0.051,
        p_value=0.002,
        p_value_clustered=0.0142,
        effective_sample_size=67,
        power=0.10,
        verdict="supported_uncorrected",
        reconciled=True,
    )
    kw.update(over)
    return StudyResult(**kw)


def _real_correction():
    """The published family: 51 cells, smallest p 0.002 -> adjusted 0.102."""
    from agent.verify.stats import benjamini_hochberg

    pvals = [0.002] + [0.002 * i for i in range(2, 52)]
    bh = benjamini_hochberg(pvals, alpha=0.05)
    assert bh.n_tested == 51
    assert abs(bh.p_adjusted[0] - 0.102) < 1e-9, bh.p_adjusted[0]
    return {
        "p_adjusted": bh.p_adjusted[0],
        "alpha": bh.alpha,
        "n_tested": bh.n_tested,
        "method": "Benjamini-Hochberg",
    }


def test_the_join_reads_the_real_study_result_field_names():
    """Every statistical line must compute against agent/verify/study.py as written.

    This is the whole task: the two halves must actually meet. `StudyResult` uses
    `mean_event_return`, `baseline_mean_return`, `n_event_clusters` and
    `effective_sample_size`, none of which are the names
    `scripts/cftc_event_study.py` emits. A report that silently rendered
    "not computed" for all of them would look like a working join.
    """
    r = build_report(_real_study(), correction=_real_correction())
    assert r.uncomputed == tuple(sorted(MECHANISM_LABELS, key=LINE_ORDER.index))
    v = r.line("VERDICT").value
    assert v.startswith("not detectable")
    assert "Benjamini-Hochberg p = 0.102" in v
    assert "alpha = 0.05" in v and "51 hypotheses tested" in v
    assert "uncorrected p = 0.002" in v
    assert "cluster-resampled p = 0.014" in v
    assert "event mean 0.02584" in v and "baseline 0.00406" in v
    # StudyResult exposes its own `period_name` property ("distinct event
    # timestamp"), and the report uses the study's word rather than inventing one.
    assert r.line("SAMPLE").value == ("123 events across 67 distinct event timestamps (ratio 0.54), effective n = 67")
    assert "~10%" in r.line("POWER").value
    # the hypothesis is the customer's sentence, not a dataclass repr
    assert r.hypothesis.startswith("When CFTC managed-money")
    assert "Hypothesis(" not in r.hypothesis


def test_hypothesis_object_falls_back_to_describe_not_to_repr():
    r = build_report(_real_study(label=""))
    assert "mm_net_pct_oi" in r.hypothesis
    assert "Hypothesis(" not in r.hypothesis
    assert "event_source=" not in r.hypothesis


def test_without_a_correction_the_uncorrected_verdict_is_reported_but_not_promoted():
    """`verdict='supported_uncorrected'` must never become DETECTED."""
    ln = build_report(_real_study()).line("VERDICT")
    assert ln.computed is False
    assert "DETECTED" not in ln.value
    assert "not detectable" not in ln.value
    assert "UNCORRECTED verdict is 'supported_uncorrected'" in ln.value
    assert "is not a detection" in " ".join(ln.caveats)


def test_unreconciled_bookkeeping_blocks_the_verdict_entirely():
    ln = build_report(_real_study(reconciled=False), correction=_real_correction()).line("VERDICT")
    assert ln.computed is False
    assert "reconciled=False" in ln.value
    assert "0.102" not in ln.value  # no figure is laundered through
    assert "DETECTED" not in ln.value and "not detectable" not in ln.value


def test_a_correction_that_disagrees_with_the_study_yields_no_verdict():
    s = FakeStudy()  # carries p_adj_bh = 0.102
    ln = build_report(s, correction={"p_adjusted": 0.4, "alpha": 0.05}).line("VERDICT")
    assert ln.computed is False
    assert "CONTRADICTORY INPUT" in ln.value
    assert "0.4" in ln.value and "0.102" in ln.value


def test_a_correction_supplies_what_the_study_cannot_carry():
    s = _real_study()
    assert build_report(s).line("VERDICT").computed is False
    ln = build_report(s, correction={"p_adjusted": 0.01, "alpha": 0.05, "n_tested": 51}).line("VERDICT")
    assert ln.computed is True
    assert ln.value.startswith("DETECTED")
    assert "correction['p_adjusted']" in ln.provenance


def test_sample_flags_the_studys_own_effective_sample_size():
    caveats = " ".join(build_report(_real_study(), correction=_real_correction()).line("SAMPLE").caveats)
    assert "effective sample size is 67, not 123" in caveats
    assert "up to 5 events fall on a single distinct event timestamp" in caveats


def test_clustered_p_is_reported_and_flagged_as_the_defensible_one():
    caveats = " ".join(build_report(_real_study(), correction=_real_correction()).line("VERDICT").caveats)
    assert "moves p from 0.002 to 0.014" in caveats
    assert "the clustered figure is the defensible one" in caveats


@pytest.mark.slow
def test_the_whole_sellable_object_end_to_end_on_the_live_graph():
    """Real StudyResult + real Benjamini-Hochberg + real graph. Nothing uncomputed.

    This is the product. Read-only on the live DB. Every figure asserted here was
    measured on 2026-09-27 against a 20,026-node / 26,087-edge graph.
    """
    import os

    if not os.path.exists(_DB):
        pytest.skip("live pipeline.db not present")
    from agent.mechanism import load_graph

    g = load_graph(_DB, as_of=None)
    r = build_report(
        _real_study(),
        correction=_real_correction(),
        graph=g,
        src_id=_RUSSIA,
        dst_id=_WTI,
        route_widths=(10, 400),
    )
    print(render_markdown(r))
    assert r.uncomputed == (), "the full product must have no uncomputed line"
    assert r.has_mechanism is True
    v = r.line("VERDICT").value
    assert v.startswith("not detectable (Benjamini-Hochberg p = 0.102, alpha = 0.05, 51 hypotheses tested")
    assert r.line("SAMPLE").value.startswith("123 events across 67 distinct event timestamps (ratio 0.54)")
    assert "instrument #1 of 93 from Russia, 28.1x median" in r.line("STRENGTH").value
    assert "1 independent time-varying source (gdelt) at k=10" in r.line("EVIDENCE").value
    assert "3 independent time-varying sources" in r.line("EVIDENCE").value
    assert "at k=400" in r.line("EVIDENCE").value
    assert "top route by mass is pure scaffold (produced_in)" in r.line("CAVEAT").value
    assert "k=10: 9/10 routes hub-dominated" in r.line("CAVEAT").value
    assert "AN INDEPENDENT TIME-VARYING SOURCE" in r.line("UPGRADE PATH").value


@pytest.mark.slow
def test_the_mechanism_figures_vary_with_the_pair_and_are_not_a_fixed_string():
    """F-18 guard on the live graph: print the series, confirm it varies.

    Measured finding worth knowing: Russia -> WTI is 28.1x the median instrument,
    but SIX arbitrary countries also come out "instrument #1 of 93" at 11-13x with
    exactly one gdelt source at k=10. The peer rank is therefore a statement about
    the graph's shape as much as about this pair, which is why the CAVEAT line
    reports the scaffold-topped, hub-dominated route set alongside it.
    """
    import os
    import sqlite3

    if not os.path.exists(_DB):
        pytest.skip("live pipeline.db not present")
    from agent.mechanism import load_graph

    con = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True)
    try:
        srcs = [
            row[0]
            for row in con.execute(
                "select entity_id from entities where entity_type='country' order by entity_id limit 6"
            )
        ]
    finally:
        con.close()
    g = load_graph(_DB, as_of=None)
    values = []
    for src in srcs:
        r = build_report({"n_events": 10}, graph=g, src_id=src, dst_id=_WTI, route_widths=(10,))
        values.append(r.line("STRENGTH").value)
        print("  STRENGTH:", values[-1])
    assert len(set(values)) == len(srcs), "STRENGTH does not vary with the source entity"
    russia = build_report({"n_events": 10}, graph=g, src_id=_RUSSIA, dst_id=_WTI, route_widths=(10,))
    assert russia.line("STRENGTH").value not in values
