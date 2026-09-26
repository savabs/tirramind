"""
Tests for agent/mechanism/routes.py.

Each test here is written to go RED against a specific wrong implementation, and
every one of those breakages was applied on purpose and observed to fail before
this file was committed. The mutations checked:

    * counting every hop's source instead of only event hops          -> test_independence_ignores_scaffold_sources
    * treating an unclassified link_type as evidence                  -> test_unclassified_link_type_is_neither
    * returning [] for an unknown id instead of raising               -> test_unknown_entity_raises
    * returning [] for "not connected" with no provenance             -> test_independence_rejects_bare_empty_list
    * dropping the k cap silently                                     -> test_truncation_is_reported
    * ranking without the degree split (no hub discount)              -> test_hub_route_is_flagged_and_discounted
    * ignoring as_of (2026 graph explaining a 2019 pattern, F-04)     -> test_as_of_excludes_future_links
    * opening the live DB writable                                    -> test_load_route_graph_opens_read_only
    * calling "no path" and "one source" the same number              -> test_no_route_status_is_not_a_source_count

Tests whose name starts with ``test_live_`` read the real pipeline DB read-only
and skip when it is absent. Their asserted numbers were measured on 2026-09-26.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.mechanism import routes as R

LIVE_DB = Path(__file__).resolve().parents[1] / ".tirra_pipeline" / "pipeline.db"
live_only = pytest.mark.skipif(not LIVE_DB.exists(), reason=f"live pipeline DB not present at {LIVE_DB}")

# Live entity ids (stable primary keys in the entities table).
RUSSIA = "c9db6e2a9784c856"
WTI = "da678733bb9c745f"
WALMART = "2ea7003ca7d7996c"  # company, degree 21, NOT in WTI's component
WTI_PHYSICAL_CFTC = "5643d6e25da448b8"  # exists in entities, zero links
WALLET_MULTI = "9f4f4f0fa25d1105"  # bitcoin wallet
NASDAQ100 = "df791da46a7221ac"

T2000 = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
T2019 = datetime(2019, 6, 1, tzinfo=UTC).timestamp()
T2026 = datetime(2026, 6, 1, tzinfo=UTC).timestamp()


def link(a, b, lt, src, *, conf=1.0, eff=T2000, created=T2000):
    return {
        "entity_id_a": a,
        "entity_id_b": b,
        "link_type": lt,
        "source": src,
        "confidence": conf,
        "effective_from": eff,
        "created_at": created,
    }


def graph_of(links, names=None):
    return R.RouteGraph.from_links(links, names=names or {})


# ── classification ────────────────────────────────────────────────────────


def test_link_kind_splits_event_from_structural():
    assert R.link_kind("event_involves") == "evidence"
    assert R.link_kind("works_for") == "evidence"
    assert R.link_kind("produced_in") == "scaffold"
    assert R.link_kind("exchange_country") == "scaffold"
    assert R.link_kind("no_such_relation") == "unclassified"
    assert not (EVENT := R.EVENT_RELATIONS) & R.STRUCTURAL_RELATIONS
    assert "produced_in" not in EVENT


def test_relation_sets_have_not_drifted_from_graph_builder():
    """The copies in routes.py must equal graph_builder's source of truth."""
    pytest.importorskip("torch_geometric")
    from agent.models.gnn import graph_builder as gb

    assert R.EVENT_RELATIONS == gb.EVENT_RELATIONS
    assert R.STRUCTURAL_RELATIONS == gb.STRUCTURAL_RELATIONS


# ── independence: the load-bearing number ─────────────────────────────────


def test_independence_ignores_scaffold_sources():
    """One gdelt event hop + one geography hop is ONE source, not two."""
    g = graph_of(
        [
            link("ru", "kz", "event_involves", "gdelt"),
            link("kz", "wti", "produced_in", "seed_producer_links"),
        ]
    )
    verdict = R.independence(R.top_routes(g, "ru", "wti"))
    assert verdict["status"] == "ok"
    assert verdict["independent_sources"] == 1
    assert verdict["evidence_sources"] == ["gdelt"]
    assert verdict["scaffold_sources_non_evidence"] == ["seed_producer_links"]
    assert verdict["naive_source_count"] == 2
    assert verdict["source_inflation_factor"] == pytest.approx(2.0)
    assert "seed_producer_links" not in verdict["evidence_sources"]


def test_independence_counts_two_genuinely_distinct_event_sources():
    g = graph_of(
        [
            link("ru", "topic", "event_involves", "gdelt"),
            link("topic", "wti", "topic_relates_to_instrument", "polymarket"),
        ]
    )
    verdict = R.independence(R.top_routes(g, "ru", "wti"))
    assert verdict["independent_sources"] == 2
    assert verdict["evidence_sources"] == ["gdelt", "polymarket"]
    assert verdict["evidence_channels"] == [
        "event_involves/gdelt",
        "topic_relates_to_instrument/polymarket",
    ]
    assert verdict["shared_relation_sources"] == {}


def test_independence_reports_zero_when_every_hop_is_geography():
    g = graph_of(
        [
            link("ru", "usa", "located_in", "instrument_universe"),
            link("usa", "wti", "produced_in", "seed_producer_links"),
        ]
    )
    verdict = R.independence(R.top_routes(g, "ru", "wti"))
    assert verdict["status"] == "ok"  # a route exists; it just witnesses nothing
    assert verdict["independent_sources"] == 0
    assert verdict["routes_without_evidence"] == 1
    assert verdict["naive_source_count"] == 2
    assert "NO TIME-VARYING EVIDENCE" in verdict["verdict"]


def test_unclassified_link_type_is_neither_evidence_nor_scaffold():
    g = graph_of([link("a", "b", "brand_new_relation", "some_collector")])
    verdict = R.independence(R.top_routes(g, "a", "b"))
    assert verdict["independent_sources"] == 0
    assert verdict["unclassified_link_types"] == ["brand_new_relation"]
    assert verdict["unclassified_sources_non_evidence"] == ["some_collector"]
    assert verdict["scaffold_sources_non_evidence"] == []
    assert any("UNCLASSIFIED" in w for w in verdict["warnings"])


def test_independence_flags_two_sources_writing_one_relation():
    """polymarket + repair_topic_links on the same link_type is one witness."""
    g = graph_of(
        [
            link("t1", "wti", "topic_relates_to_instrument", "polymarket"),
            link("a", "t1", "event_involves", "gdelt"),
            link("t2", "wti", "topic_relates_to_instrument", "repair_topic_links"),
            link("a", "t2", "event_involves", "gdelt"),
        ]
    )
    verdict = R.independence(R.top_routes(g, "a", "wti"))
    assert verdict["independent_sources"] == 3
    assert verdict["shared_relation_sources"] == {"topic_relates_to_instrument": ["polymarket", "repair_topic_links"]}
    assert any("CO-DEPENDENCE RISK" in w for w in verdict["warnings"])


def test_independence_rejects_bare_empty_list():
    """An empty list with no provenance cannot be scored, so it raises."""
    with pytest.raises(R.RouteIndeterminateError):
        R.independence([])


def test_independence_rejects_non_routes():
    with pytest.raises(R.RouteError):
        R.independence(["not a route"])


def test_no_route_status_is_not_a_source_count():
    """Two components: status says no_route, and the 0 is explained."""
    g = graph_of(
        [
            link("a", "b", "event_involves", "gdelt"),
            link("x", "y", "event_involves", "gdelt"),
        ]
    )
    result = R.top_routes(g, "a", "y")
    assert list(result) == []
    assert result.search.connected is False
    assert result.search.shortest_hops is None
    assert "NO ROUTE" in result.search.reason
    verdict = R.independence(result)
    assert verdict["status"] == "no_route"
    assert verdict["independent_sources"] == 0
    assert "NO ROUTE" in verdict["verdict"]
    assert verdict["status"] != R.independence(R.top_routes(g, "a", "b"))["status"]


def test_hub_free_count_drops_sources_that_only_survive_via_hubs():
    hub_links = [link("hub", f"pad{i}", "works_for", "form144") for i in range(12)]
    g = graph_of(
        [
            link("a", "hub", "event_involves", "gdelt"),
            link("hub", "z", "produced_in", "seed_producer_links"),
            *hub_links,
        ]
    )
    verdict = R.independence(R.top_routes(g, "a", "z"))
    assert verdict["independent_sources"] == 1
    assert verdict["hub_dominated_routes"] == 1
    assert verdict["hub_free_independent_sources"] == 0
    assert any("passes through a hub" in w for w in verdict["warnings"])


# ── degenerate input raises, it does not return a plausible zero ───────────


def test_unknown_entity_raises():
    g = graph_of([link("a", "b", "event_involves", "gdelt")])
    with pytest.raises(R.RouteUnknownEntityError):
        R.top_routes(g, "nope", "b")
    with pytest.raises(R.RouteUnknownEntityError):
        R.top_routes(g, "a", "nope")


def test_isolated_entity_raises_and_is_not_confused_with_unknown():
    g = R.RouteGraph.from_links(
        [link("a", "b", "event_involves", "gdelt")],
        names={"a": "A", "b": "B", "lonely": "Lonely"},
    )
    with pytest.raises(R.RouteIsolatedEntityError):
        R.top_routes(g, "lonely", "b")
    assert not issubclass(R.RouteIsolatedEntityError, R.RouteUnknownEntityError)


def test_empty_graph_raises():
    g = graph_of([])
    with pytest.raises(R.RouteEmptyGraphError):
        R.top_routes(g, "a", "b")


def test_self_pair_raises():
    g = graph_of([link("a", "b", "event_involves", "gdelt")])
    with pytest.raises(R.RouteError):
        R.top_routes(g, "a", "a")


def test_bad_parameters_raise():
    g = graph_of([link("a", "b", "event_involves", "gdelt")])
    for kwargs in (
        {"k": 0},
        {"max_hops": 0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"hub_degree_percentile": 101.0},
        {"max_expansions": 0},
    ):
        with pytest.raises(R.RouteError):
            R.top_routes(g, "a", "b", **kwargs)


def test_link_record_missing_source_raises():
    with pytest.raises(R.RouteError):
        R.RouteGraph.from_links(
            [
                {
                    "entity_id_a": "a",
                    "entity_id_b": "b",
                    "link_type": "event_involves",
                }
            ]
        )


def test_graph_without_edge_metadata_raises_rather_than_guessing():
    class MatrixOnlyGraph:
        A = object()
        idx = {"a": 0}
        ents = ["a"]

    with pytest.raises(R.RouteMissingEdgeMetadataError):
        R.top_routes(MatrixOnlyGraph(), "a", "b")


def test_mechanism_graph_shaped_object_is_adapted():
    """Row-indexed edges (MechanismGraph's shape) are mapped back to ids."""

    class Edge:
        def __init__(self, a, b, lt, src, eff):
            self.a, self.b = a, b
            self.link_type, self.source = lt, src
            self.confidence, self.effective_at = 1.0, eff
            self.is_evidence = True  # deliberately WRONG for produced_in

    class Report:
        rows_read = 3
        rows_with_effective_from = 2
        rows_dropped_by_as_of = 7

    class FakeMechanismGraph:
        entity_ids = ("ru", "kz", "wti")
        canonical_names = ("Russia", "Kazakhstan", "WTI Crude Oil")
        entity_types = ("country", "country", "instrument")
        edges = (
            Edge(0, 1, "event_involves", "gdelt", T2019),
            Edge(1, 2, "produced_in", "seed_producer_links", T2000),
        )
        as_of = T2026
        report = Report()

    result = R.top_routes(FakeMechanismGraph(), "ru", "wti")
    assert len(result) == 1
    assert result[0].node_names == ("Russia", "Kazakhstan", "WTI Crude Oil")
    # is_evidence=True on the produced_in edge must NOT override link_kind
    assert result[0].kinds == ("evidence", "scaffold")
    assert R.independence(result)["independent_sources"] == 1
    assert result.search.as_of == T2026
    assert result.search.links_excluded_by_as_of == 7
    assert result.search.links_with_undated_fallback == 1


def test_foreign_graph_exposing_links_is_accepted():
    class ForeignGraph:
        links = [link("a", "b", "event_involves", "gdelt")]
        name = {"a": "A", "b": "B"}

    result = R.top_routes(ForeignGraph(), "a", "b")
    assert len(result) == 1
    assert result[0].node_names == ("A", "B")


# ── truncation and completeness are always reported ───────────────────────


def test_truncation_is_reported():
    g = graph_of(
        [link("a", f"m{i}", "event_involves", "gdelt") for i in range(4)]
        + [link(f"m{i}", "z", "produced_in", "seed_producer_links") for i in range(4)]
    )
    full = R.top_routes(g, "a", "z", k=10)
    assert len(full) == 4
    assert full.search.more_routes_exist is False
    assert full.search.exhaustive is True
    assert full.search.warnings() == []

    capped = R.top_routes(g, "a", "z", k=1)
    assert len(capped) == 1
    assert capped.search.more_routes_exist is True
    assert capped.search.exhaustive is False
    assert any("TRUNCATED" in w for w in capped.search.warnings())
    assert R.independence(capped)["complete"] is False
    assert R.independence(full)["complete"] is True


def test_budget_exhaustion_is_loud_and_not_a_zero():
    g = graph_of(
        [link("a", f"m{i}", "event_involves", "gdelt") for i in range(50)]
        + [link(f"m{i}", "z", "produced_in", "seed_producer_links") for i in range(50)]
    )
    starved = R.top_routes(g, "a", "z", k=50, max_expansions=1)
    assert starved.search.budget_exhausted is True
    assert starved.search.exhaustive is False
    assert any("BUDGET EXHAUSTED" in w for w in starved.search.warnings())


def test_shortest_path_beyond_max_hops_is_distinguishable_from_disconnection():
    g = graph_of(
        [
            link("a", "b", "event_involves", "gdelt"),
            link("b", "c", "event_involves", "gdelt"),
            link("c", "z", "produced_in", "seed_producer_links"),
        ]
    )
    short = R.top_routes(g, "a", "z", max_hops=2)
    assert list(short) == []
    assert short.search.connected is True  # NOT the disconnected case
    assert short.search.shortest_hops == 3
    assert "max_hops" in short.search.reason
    assert R.independence(short)["status"] == "no_route"
    assert len(R.top_routes(g, "a", "z", max_hops=3)) == 1


def test_duplicate_reciprocal_rows_collapse_into_one_route():
    g = graph_of(
        [
            link("a", "b", "event_involves", "gdelt"),
            link("b", "a", "event_involves", "gdelt"),  # same relation, other way
        ]
    )
    result = R.top_routes(g, "a", "b")
    assert len(result) == 1
    assert result[0].hops_detail[0].records == 2
    assert R.independence(result)["independent_sources"] == 1


def test_parallel_link_types_are_separate_routes_but_flagged_as_one_node_path():
    g = graph_of(
        [
            link("a", "m", "event_involves", "gdelt"),
            link("m", "z", "produced_in", "seed_producer_links"),
            link("m", "z", "exchange_country", "instrument_universe"),
        ]
    )
    result = R.top_routes(g, "a", "z")
    assert len(result) == 2
    verdict = R.independence(result)
    assert verdict["routes_sharing_node_sequence"] == 1
    assert verdict["independent_sources"] == 1  # still one gdelt witness
    assert any("retrace a node sequence" in w for w in verdict["warnings"])


# ── mass, ordering, hubs ──────────────────────────────────────────────────


def test_mass_equals_hand_computed_ppr_path_contribution():
    g = graph_of(
        [
            link("a", "m", "event_involves", "gdelt"),
            link("m", "z", "produced_in", "seed_producer_links"),
            link("m", "pad", "works_for", "form144"),
        ]
    )
    alpha = 0.15
    route = R.top_routes(g, "a", "z", alpha=alpha)[0]
    # deg(a) = 1, deg(m) = 3
    expected = alpha * ((1 - alpha) / 1) * ((1 - alpha) / 3)
    assert route.mass == pytest.approx(expected)
    assert route.hops == 2
    assert route.intermediate_degrees == (3,)


def test_hub_route_is_flagged_and_discounted():
    hub_pad = [link("hub", f"p{i}", "works_for", "form144") for i in range(20)]
    g = graph_of(
        [
            link("a", "hub", "event_involves", "gdelt"),
            link("hub", "z", "produced_in", "seed_producer_links"),
            link("a", "quiet", "event_involves", "gdelt"),
            link("quiet", "z", "produced_in", "seed_producer_links"),
            *hub_pad,
        ]
    )
    result = R.top_routes(g, "a", "z", k=2)
    assert len(result) == 2
    quiet, hub = result[0], result[1]
    assert quiet.nodes == ("a", "quiet", "z")
    assert hub.nodes == ("a", "hub", "z")
    assert quiet.mass > hub.mass  # the hub route is discounted, not repeated
    assert hub.hub_dominated is True
    assert hub.hub_nodes == ("hub",)
    assert hub.max_intermediate_degree == 22
    assert quiet.hub_dominated is False
    assert quiet.max_intermediate_degree == 2


def test_routes_are_ordered_by_descending_mass():
    g = graph_of(
        [link("a", f"m{i}", "event_involves", "gdelt") for i in range(5)]
        + [link(f"m{i}", "z", "produced_in", "seed_producer_links") for i in range(5)]
        + [link(f"m{i}", f"pad{i}_{j}", "works_for", "form144") for i in range(5) for j in range(i)]
    )
    masses = [r.mass for r in R.top_routes(g, "a", "z", k=5)]
    assert masses == sorted(masses, reverse=True)
    assert len(set(masses)) == 5


def test_direct_edge_route_has_no_intermediates():
    g = graph_of([link("a", "z", "produced_in", "seed_producer_links")])
    route = R.top_routes(g, "a", "z", max_hops=1)[0]
    assert route.hops == 1
    assert route.intermediate_degrees == ()
    assert route.max_intermediate_degree == 0
    assert route.hub_dominated is False
    assert "SCAF" in route.describe()


def test_describe_marks_evidence_and_scaffold_hops():
    g = graph_of(
        [
            link("a", "m", "event_involves", "gdelt"),
            link("m", "z", "produced_in", "seed_producer_links"),
        ]
    )
    text = R.describe_routes(R.top_routes(g, "a", "z"))
    assert "EVID" in text and "SCAF" in text
    assert "independent_sources=1" in text
    assert "VERDICT:" in text


# ── time honesty (F-04) ───────────────────────────────────────────────────


def test_as_of_excludes_future_links():
    """A 2026 link must not explain a 2019 pattern."""
    links = [
        link("a", "m", "event_involves", "gdelt", eff=T2019),
        link("m", "z", "produced_in", "seed_producer_links", eff=T2026),
    ]
    today = graph_of(links)
    assert len(R.top_routes(today, "a", "z")) == 1

    as_of_2019 = R.RouteGraph.from_links(links, as_of="2019-12-31")
    assert as_of_2019.stats["links_excluded_by_as_of"] == 1
    # 'z' has no links at all as of 2019 — that is an isolated node, not a
    # missing route and not a bad id, and the error must say the cutoff did it.
    with pytest.raises(R.RouteIsolatedEntityError, match="as_of"):
        R.top_routes(as_of_2019, "a", "z")


def test_as_of_is_reported_on_every_search():
    links = [
        link("a", "m", "event_involves", "gdelt", eff=T2019),
        link("m", "z", "produced_in", "seed_producer_links", eff=T2019),
        link("a", "z", "event_involves", "gdelt", eff=T2026),
    ]
    g = R.RouteGraph.from_links(links, as_of=datetime(2020, 1, 1, tzinfo=UTC))
    result = R.top_routes(g, "a", "z")
    assert len(result) == 1  # the 2026 shortcut is gone
    assert result[0].hops == 2
    assert result.search.links_excluded_by_as_of == 1
    assert result.search.as_of == pytest.approx(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    assert any("as_of=" in w for w in result.search.warnings())


def test_undated_links_are_counted_as_fallback_dated():
    g = R.RouteGraph.from_links([link("a", "b", "event_involves", "gdelt", eff=None, created=T2019)])
    assert g.stats["links_with_undated_fallback"] == 1
    assert any("no effective_from" in w for w in R.top_routes(g, "a", "b").search.warnings())


def test_as_of_forms_agree():
    ts = datetime(2020, 1, 1, tzinfo=UTC).timestamp()
    assert R._normalise_as_of("2020-01-01") == ts
    assert R._normalise_as_of(datetime(2020, 1, 1)) == ts
    assert R._normalise_as_of(ts) == ts
    with pytest.raises(R.RouteError):
        R._normalise_as_of("last tuesday")
    with pytest.raises(R.RouteError):
        R._normalise_as_of(True)


# ── the DB is opened read-only, full stop ─────────────────────────────────


def test_load_route_graph_opens_read_only(tmp_path, monkeypatch):
    db = tmp_path / "toy.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE entities (entity_id TEXT PRIMARY KEY, entity_type TEXT,
                               canonical_name TEXT, created_at REAL);
        CREATE TABLE entity_links (link_id INTEGER PRIMARY KEY, entity_id_a TEXT,
                               entity_id_b TEXT, link_type TEXT, confidence REAL,
                               source TEXT, created_at REAL, effective_from REAL);
        INSERT INTO entities VALUES ('a','country','A',0), ('b','instrument','B',0);
        INSERT INTO entity_links VALUES
            (1,'a','b','event_involves',1.0,'gdelt',0,0);
        """
    )
    conn.commit()
    conn.close()

    seen: list[tuple[tuple, dict]] = []
    real_connect = sqlite3.connect

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(R.sqlite3, "connect", spy)
    g = R.load_route_graph(db)
    assert len(seen) == 1
    (dsn, *_), kwargs = seen[0]
    assert "mode=ro" in dsn
    assert kwargs.get("uri") is True
    assert g.n_nodes == 2
    assert len(R.top_routes(g, "a", "b")) == 1


def test_load_route_graph_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        R.load_route_graph(tmp_path / "nope.db")


# ── live graph (measured 2026-09-26) ──────────────────────────────────────


@pytest.fixture(scope="module")
def live_graph():
    if not LIVE_DB.exists():
        pytest.skip("live DB absent")
    return R.load_route_graph(LIVE_DB)


@live_only
def test_live_graph_shape(live_graph):
    assert live_graph.stats["entities_in_db"] == 20026
    assert live_graph.stats["records_in"] == 30131
    assert live_graph.n_nodes == 18302  # 1,724 entities have no links at all
    assert live_graph.degree["f07b20bdd8b5f9bb"] == 522  # United States, the hub


@live_only
def test_live_russia_wti_has_exactly_one_independent_source(live_graph):
    """The acceptance number: 1 time-varying source, not the naive count."""
    result = R.top_routes(live_graph, RUSSIA, WTI)
    assert len(result) == 10
    assert result.search.connected is True
    assert result.search.shortest_hops == 1
    verdict = R.independence(result)
    assert verdict["status"] == "ok"
    assert verdict["independent_sources"] == 1
    assert verdict["evidence_sources"] == ["gdelt"]
    assert verdict["evidence_channels"] == ["event_involves/gdelt"]
    assert verdict["scaffold_sources_non_evidence"] == ["seed_producer_links"]
    assert verdict["naive_source_count"] > verdict["independent_sources"]
    assert verdict["complete"] is False  # more routes exist; k=10 truncated
    assert any("TRUNCATED" in w for w in verdict["warnings"])
    # every evidenced route is Russia -> {producer country} -> WTI
    evidenced = [r for r in result if r.has_evidence]
    assert len(evidenced) == 9
    assert {r.link_types for r in evidenced} == {("event_involves", "produced_in")}
    assert all(r.hub_dominated for r in evidenced)


@live_only
def test_live_russia_wti_scaffold_only_top_route(live_graph):
    """Russia produces oil, so the highest-mass route is pure geography."""
    top = R.top_routes(live_graph, RUSSIA, WTI)[0]
    assert top.nodes == (RUSSIA, WTI)
    assert top.kinds == ("scaffold",)
    assert top.sources == ("seed_producer_links",)
    assert top.has_evidence is False


@live_only
def test_live_wider_route_set_raises_the_count_and_says_why(live_graph):
    """Independence is a property of the route set, and the dict admits it."""
    wide = R.top_routes(live_graph, RUSSIA, WTI, k=400, max_expansions=500_000)
    verdict = R.independence(wide)
    assert verdict["independent_sources"] == 3
    assert verdict["evidence_sources"] == ["gdelt", "polymarket", "repair_topic_links"]
    assert verdict["naive_source_count"] == 5
    assert verdict["shared_relation_sources"] == {"topic_relates_to_instrument": ["polymarket", "repair_topic_links"]}
    assert any("CO-DEPENDENCE RISK" in w for w in verdict["warnings"])
    assert verdict["hub_free_independent_sources"] == 1


@live_only
def test_live_multi_sourced_pair(live_graph):
    """A pair with genuinely different collectors behind it."""
    verdict = R.independence(R.top_routes(live_graph, WALLET_MULTI, NASDAQ100))
    assert verdict["independent_sources"] == 3
    assert verdict["evidence_sources"] == [
        "polymarket",
        "repair_topic_links",
        "whale_alert",
    ]
    assert set(verdict["evidence_link_types"]) == {
        "topic_relates_to_instrument",
        "trades_instrument",
        "transacts_with",
    }
    assert verdict["scaffold_sources_non_evidence"] == []


@live_only
def test_live_unconnected_pair_returns_no_route_not_zero_evidence(live_graph):
    result = R.top_routes(live_graph, WALMART, WTI)
    assert list(result) == []
    assert result.search.connected is False
    assert result.search.shortest_hops is None
    verdict = R.independence(result)
    assert verdict["status"] == "no_route"
    assert "NO ROUTE" in verdict["verdict"]


@live_only
def test_live_isolated_entity_raises(live_graph):
    """The WTI-PHYSICAL CFTC contract exists but has zero links."""
    with pytest.raises(R.RouteIsolatedEntityError):
        R.top_routes(live_graph, WTI_PHYSICAL_CFTC, WTI)


@live_only
def test_live_graph_load_does_not_touch_the_db_file():
    before = LIVE_DB.stat().st_mtime_ns
    g = R.load_route_graph(LIVE_DB)
    R.top_routes(g, RUSSIA, WTI)
    assert LIVE_DB.stat().st_mtime_ns == before


@live_only
def test_live_as_of_1990_shrinks_the_graph(live_graph):
    """F-04: a historical as_of must change the answer, or it is not filtering."""
    old = R.load_route_graph(LIVE_DB, as_of="1995-01-01")
    assert old.stats["links_excluded_by_as_of"] > 0
    assert old.n_nodes < live_graph.n_nodes
