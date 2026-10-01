"""Regression tests for the three graph_builder input defects (audit 2026-09-23).

Each class pins one defect that made the model "a per-node MLP over a type
one-hot plus a count and a recency scalar" rather than a graph network:

    A. Node value features were hard zero for 99.8% of observations —
       `_compute_obs_stats` / `_compute_distributional_features` probed six key
       names that no collector writes (736 of 384,285 live rows matched).
    B. Edges were gated on `entity_links.created_at`, an INGEST stamp, so the
       F-14 fix closed the F-04 leak by deleting the graph: 86.9% of
       observations predate the earliest edge, and `HetTGN.forward` skips the
       entire HGT loop when `edge_index_dict` is empty.
    C. The graph was strictly directed, so 64% of edges — including all 9,487
       GDELT country edges, which point *out* of instruments — could never
       propagate to the nodes the objective is computed on.

Every test here fails against the pre-fix `graph_builder.py`.

See LESSONS.md F-04 (leakage) and F-14 (the fix nobody called): nothing in this
file may weaken the `until` gate. The event-relation tests exist precisely to
prove it is still closed.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path

import pytest

from agent.models.gnn.graph_builder import (
    EVENT_RELATIONS,
    OBS_TYPES_WITHOUT_VALUE,
    OBS_VALUE_KEYS,
    OBSERVATION_TYPES,
    STRUCTURAL_RELATIONS,
    GraphBuilder,
    IDMap,
    SchemaDriftError,
    _build_edge_data,
    _compute_distributional_features,
    _compute_obs_stats,
    _links_as_of,
    extract_obs_value,
    reverse_relation,
    scale_obs_value,
    unclassified_link_relations,
    unmapped_observation_types,
    validate_value_keys_against_store,
)
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore

# The six key names the dead probe looked for. Kept here so the tests can state
# what used to happen, and so nobody quietly reintroduces them as a fallback.
_DEAD_PROBE_KEYS = (
    "usd_amount",
    "btc_amount",
    "value",
    "estimated_value",
    "goldstein_scale",
    "num_articles",
)

# observation_types present in the live DB on 2026-09-23, snapshotted so this
# guard runs in CI, where `.tirra_pipeline/` does not exist (audit P7.4: the
# country regression guards never ran because they skipped without the DB).
_DB_OBSERVATION_TYPES_20260923 = (
    "area_daily_activity",
    "baltic_activity_proxy",
    "bankruptcy_status",
    "border_throughput",
    "btc_transfer",
    "campaign_finance",
    "capital_flow",
    "cb_balance_sheet",
    "cb_policy_rate",
    "cert_issued",
    "consumer_confidence",
    "contract_award",
    "creditor_filing",
    "dividend",
    "dns_change",
    "drug_approval",
    "economic_activity",
    "futures_positioning",
    "futures_positioning_derived",
    "geopolitical_event",
    "insider_trade",
    "instrument_daily",
    "instrument_return",
    "instrument_volatility",
    "instrument_volume",
    "market_probability",
    "options_chain_eod",
    "pageview_spike",
    "pathogen_level",
    "petroleum_inventory",
    "price_movement",
    "regulatory_velocity",
    "research_velocity",
    "sanctions_listing",
    "sell_intent",
    "short_interest",
    "sovereign_yield",
    "trade_flow",
    "tvl_change",
    "vessel_position",
)

_LIVE_DB = Path(__file__).resolve().parents[1] / ".tirra_pipeline" / "pipeline.db"

# Store-backed tests need plausible epoch timestamps: store.store_entity_observation
# rejects anything outside [1990-01-01, now + 1d].
_T_OBS = 1_577_836_800.0  # 2020-01-01 — the observation, and the window
_T_WINDOW = 1_580_000_000.0  # 2020-01-26 — window end, well before ingestion
_T_INGEST = 1_767_225_600.0  # 2026-01-01 — when the link row was INSERTed


@pytest.fixture()
def store() -> PipelineStore:
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


def _obs(obs_type: str, value: dict, entity_id: str = "e1", ts: float = 1000.0) -> dict:
    return {
        "entity_id": entity_id,
        "observed_at": ts,
        "observation_type": obs_type,
        "source_tool": "test",
        "value": value,
    }


# ═══════════════════════════════════════════════════════════════
# A. Node value features
# ═══════════════════════════════════════════════════════════════


class TestValueKeyMapCoverage:
    """Every observation_type must resolve to a documented decision."""

    def test_every_registry_type_is_mapped(self) -> None:
        missing = [t for t in OBSERVATION_TYPES if t not in OBS_VALUE_KEYS]
        assert not missing, (
            f"observation_types with no OBS_VALUE_KEYS entry: {missing}. "
            "An unmapped type contributes an all-zero value feature for every "
            "one of its rows — the exact failure this map exists to end."
        )

    def test_map_holds_nothing_unregistered(self) -> None:
        extra = [t for t in OBS_VALUE_KEYS if t not in OBSERVATION_TYPES]
        assert not extra, f"OBS_VALUE_KEYS entries not in OBSERVATION_TYPES: {extra}"

    def test_map_is_alphabetical(self) -> None:
        """Same reviewability rule as the registries: insertions must be obvious."""
        assert list(OBS_VALUE_KEYS) == sorted(OBS_VALUE_KEYS)

    def test_value_less_types_are_explicit(self) -> None:
        """A type with no numeric payload says so; it is not just left out."""
        assert "drug_approval" in OBS_TYPES_WITHOUT_VALUE
        assert OBS_VALUE_KEYS["drug_approval"] == ()
        assert "tvl_change" not in OBS_TYPES_WITHOUT_VALUE

    def test_dead_probe_keys_are_not_a_fallback(self) -> None:
        """The six near-miss names must not be reintroduced as a catch-all.

        `goldstein_scale` / `num_articles` are what the old probe looked for;
        GDELT writes `goldstein` / `num_mentions`. A fallback to the old names
        would mask the next near-miss instead of surfacing it.
        """
        for key in _DEAD_PROBE_KEYS:
            if key == "value":
                continue  # a real key for petroleum_inventory / economic_activity
            users = [t for t, keys in OBS_VALUE_KEYS.items() if key in keys]
            assert not users, f"dead probe key {key!r} reintroduced for {users}"


class TestExtractObsValue:
    """The extractor itself. Each case returns None or 0.0 pre-fix."""

    def test_geopolitical_event_uses_the_key_gdelt_actually_writes(self) -> None:
        # 92,211 live rows. The old probe wanted "goldstein_scale" and got nothing.
        val = extract_obs_value(_obs("geopolitical_event", {"goldstein": -5.0, "num_mentions": 10}))
        assert val is not None
        assert val == pytest.approx(scale_obs_value(-5.0))
        assert val < 0, "sign must survive scaling — goldstein is signed"

    @pytest.mark.parametrize(
        ("obs_type", "value", "expected_raw"),
        [
            ("tvl_change", {"tvl_usd": 1.5e9}, 1.5e9),
            ("instrument_daily", {"close": 431.2, "log_return": 0.004}, 431.2),
            ("btc_transfer", {"value_btc": 12.5}, 12.5),
            ("contract_award", {"amount_usd": 4_200_000.0}, 4_200_000.0),
            ("market_probability", {"yes_price": 0.62}, 0.62),
            ("sovereign_yield", {"yield_pct": 4.31}, 4.31),
            ("dividend", {"amount": 0.94}, 0.94),
            ("insider_trade", {"shares": 1200, "price": 51.0}, 1200),
            ("petroleum_inventory", {"value": 421_000.0}, 421_000.0),
        ],
    )
    def test_real_keys_resolve(self, obs_type: str, value: dict, expected_raw: float) -> None:
        val = extract_obs_value(_obs(obs_type, value))
        assert val is not None, f"{obs_type} still yields no value"
        assert math.isfinite(val)
        assert val == pytest.approx(scale_obs_value(expected_raw))

    def test_absent_value_is_none_not_zero(self) -> None:
        """None means 'omit'. 0.0 would be a claim about the world."""
        assert extract_obs_value(_obs("tvl_change", {"chain": "ethereum"})) is None
        assert extract_obs_value(_obs("tvl_change", {"tvl_usd": None})) is None

    def test_value_less_type_is_none(self) -> None:
        assert extract_obs_value(_obs("drug_approval", {"application_number": "ANDA076907"})) is None

    def test_non_numeric_and_bool_are_skipped(self) -> None:
        assert extract_obs_value(_obs("contract_award", {"amount_usd": "not-a-number"})) is None
        assert extract_obs_value(_obs("cert_issued", {"is_expired": True})) is None

    def test_falls_through_to_the_next_mapped_key(self) -> None:
        """A present-but-unusable first key must not abort the lookup.

        The old probe `break`s on the first key name it finds, usable or not.
        """
        val = extract_obs_value(_obs("insider_trade", {"shares": None, "price": 51.0}))
        assert val == pytest.approx(scale_obs_value(51.0))

    def test_unmapped_type_is_loud(self, caplog) -> None:
        with caplog.at_level(logging.ERROR, logger="agent.models.gnn.graph_builder"):
            val = extract_obs_value(_obs("brand_new_collector_type", {"amount": 5.0}))
        assert val is None
        assert "brand_new_collector_type" in unmapped_observation_types()
        assert any("OBS_VALUE_KEYS" in r.message for r in caplog.records), (
            "an unmapped observation_type must be reported, not silently zeroed"
        )


class TestValueScaling:
    """Raw magnitudes span 0.1 to 1.5e9; one column cannot carry them unscaled."""

    def test_scaling_is_signed_and_monotone(self) -> None:
        assert scale_obs_value(0.0) == 0.0
        assert scale_obs_value(-100.0) == -scale_obs_value(100.0)
        assert scale_obs_value(10.0) < scale_obs_value(1000.0)

    def test_a_tvl_row_cannot_drown_a_price_row(self) -> None:
        tvl = extract_obs_value(_obs("tvl_change", {"tvl_usd": 1.5e9}))
        px = extract_obs_value(_obs("instrument_daily", {"close": 431.2}))
        assert tvl is not None and px is not None
        raw_ratio = 1.5e9 / 431.2
        assert raw_ratio > 3_000_000
        assert tvl / px < 5.0, (
            f"scaled ratio {tvl / px:.1f} — a single DeFi row still dominates every price observation in the window"
        )

    def test_scaled_values_stay_in_a_trainable_range(self) -> None:
        for raw in (0.1, 1.0, 1e6, 1.5e9):
            assert abs(scale_obs_value(raw)) < 25.0


class TestObsStatsUseTheMap:
    """The feature-tensor call sites, not just the helper."""

    def test_mean_value_is_non_zero_for_real_observations(self) -> None:
        obs = [
            _obs("geopolitical_event", {"goldstein": -5.0, "num_mentions": 10}),
            _obs("geopolitical_event", {"goldstein": -3.0, "num_mentions": 4}),
        ]
        stats = _compute_obs_stats(obs, "e1", 2000.0)
        assert stats["count"] == 2.0
        expected = (scale_obs_value(-5.0) + scale_obs_value(-3.0)) / 2
        assert stats["mean_value"] == pytest.approx(expected)
        assert stats["mean_value"] != 0.0

    def test_valueless_rows_are_excluded_from_the_mean_not_zeroed(self) -> None:
        obs = [
            _obs("tvl_change", {"tvl_usd": 1_000_000.0}),
            _obs("tvl_change", {"chain": "ethereum"}),  # no value in this row
        ]
        stats = _compute_obs_stats(obs, "e1", 2000.0)
        assert stats["mean_value"] == pytest.approx(scale_obs_value(1_000_000.0)), (
            "a row with no extractable value was averaged in as 0.0, halving the feature"
        )

    def test_entity_with_no_extractable_values_is_zero(self) -> None:
        stats = _compute_obs_stats([_obs("drug_approval", {"brand_names": ["X"]})], "e1", 2000.0)
        assert stats["mean_value"] == 0.0

    def test_distributional_features_use_the_same_quantity(self) -> None:
        obs = [
            _obs("tvl_change", {"tvl_usd": 1_000.0}),
            _obs("tvl_change", {"tvl_usd": 1_000_000.0}),
            _obs("tvl_change", {"tvl_usd": 1_000_000_000.0}),
        ]
        feats = _compute_distributional_features(obs)
        assert feats["variance"] > 0.0, "value distribution block was all-zero"
        assert feats["min"] == pytest.approx(scale_obs_value(1_000.0))
        assert feats["max"] == pytest.approx(scale_obs_value(1_000_000_000.0))
        assert feats["max"] < 25.0, "distributional block still carries raw 1e9 magnitudes"


class TestValueKeyStoreGuard:
    def test_unmapped_type_in_store_raises(self, store: PipelineStore) -> None:
        eid = entity_id_from_key("company", "acme")
        store.register_entity("company", "Acme", eid)
        store.store_entity_observation(
            entity_id=eid,
            source_tool="test",
            observed_at=_T_OBS,
            observation_type="brand_new_collector_type",
            value={"amount": 1.0},
        )
        with pytest.raises(SchemaDriftError, match="brand_new_collector_type"):
            validate_value_keys_against_store(store)

    def test_mapped_store_passes(self, store: PipelineStore) -> None:
        eid = entity_id_from_key("company", "acme")
        store.register_entity("company", "Acme", eid)
        store.store_entity_observation(
            entity_id=eid,
            source_tool="test",
            observed_at=_T_OBS,
            observation_type="contract_award",
            value={"amount_usd": 10.0},
        )
        assert validate_value_keys_against_store(store) == {"unmapped_observation_types": []}


class TestLiveDatabaseCoverage:
    """The map is a claim about a real database. Check it against one."""

    def test_snapshot_of_live_types_is_fully_mapped(self) -> None:
        """Runs everywhere, including CI, where .tirra_pipeline/ is absent."""
        missing = [t for t in _DB_OBSERVATION_TYPES_20260923 if t not in OBS_VALUE_KEYS]
        assert not missing, f"live observation_types with no mapping: {missing}"

    @pytest.mark.skipif(not _LIVE_DB.exists(), reason="live pipeline.db not present")
    def test_live_rows_resolve_to_finite_values(self) -> None:
        conn = sqlite3.connect(f"file:{_LIVE_DB}?mode=ro", uri=True)  # READ-ONLY, always
        try:
            rows = conn.execute("SELECT observation_type, value_json FROM entity_observations").fetchall()
        finally:
            conn.close()

        assert rows, "live DB has no observations to check"
        unmapped: set[str] = set()
        total = 0
        resolved = 0
        for obs_type, value_json in rows:
            if obs_type not in OBS_VALUE_KEYS:
                unmapped.add(obs_type)
                continue
            total += 1
            try:
                value = json.loads(value_json or "{}")
            except (TypeError, json.JSONDecodeError):
                value = {}
            val = extract_obs_value({"observation_type": obs_type, "value": value})
            if val is not None:
                assert math.isfinite(val)
                resolved += 1

        assert not unmapped, f"live observation_types with no mapping: {sorted(unmapped)}"
        coverage = resolved / total
        assert coverage > 0.90, (
            f"only {coverage:.2%} of live observations yield a value "
            f"({resolved}/{total}) — the old six-key probe managed 0.19%"
        )


# ═══════════════════════════════════════════════════════════════
# B. Edge time basis
# ═══════════════════════════════════════════════════════════════


def _link(link_type: str, created_at: float, **extra) -> dict:
    return {
        "entity_id_a": "a",
        "entity_id_b": "b",
        "link_type": link_type,
        "confidence": 1.0,
        "created_at": created_at,
        **extra,
    }


class TestRelationTimeBasis:
    def test_the_two_sets_are_disjoint(self) -> None:
        assert not (STRUCTURAL_RELATIONS & EVENT_RELATIONS)

    def test_live_link_types_are_all_classified(self) -> None:
        """Every link_type in the live DB on 2026-09-23 has a stated basis."""
        live = {
            "awarded_by",
            "cftc_tracks",
            "domain_owned_by",
            "event_involves",
            "exchange_country",
            "fx_base_country",
            "fx_quote_country",
            "located_in",
            "market_authorized_in",
            "operates_in",
            "produced_in",
            "sanctioned_under",
            "topic_relates_to_instrument",
            "tracks_issuer",
            "tracks_protocol",
            "trades_instrument",
            "transacts_with",
            "works_for",
        }
        assert not (live - STRUCTURAL_RELATIONS - EVENT_RELATIONS)

    def test_structural_link_is_in_scope_for_a_historical_window(self) -> None:
        """An instrument's country of production was true before we ingested it.

        `created_at` is an INSERT stamp (live range 2026-04-19 .. 2026-08-27)
        and 86.9% of observations predate it, so gating structural relations on
        it empties the graph for ~87-100% of training windows.
        """
        links = [_link("produced_in", created_at=9_000.0)]
        assert len(_links_as_of(links, until=2_000.0)) == 1

    @pytest.mark.parametrize("relation", sorted(STRUCTURAL_RELATIONS))
    def test_every_structural_relation_survives(self, relation: str) -> None:
        assert len(_links_as_of([_link(relation, created_at=9_000.0)], until=2_000.0)) == 1

    @pytest.mark.parametrize("relation", sorted(EVENT_RELATIONS))
    def test_event_relations_stay_gated(self, relation: str) -> None:
        """F-04/F-14 must stay closed: this is the half that is a real leak."""
        assert _links_as_of([_link(relation, created_at=9_000.0)], until=2_000.0) == []

    def test_event_relation_is_kept_once_the_window_reaches_it(self) -> None:
        links = [_link("event_involves", created_at=9_000.0)]
        assert len(_links_as_of(links, until=10_000.0)) == 1

    def test_event_relation_prefers_the_underlying_event_time(self) -> None:
        """Ingest lag must not hide an event that had already happened."""
        links = [_link("event_involves", created_at=9_000.0, metadata={"event_time": 1_500.0})]
        assert len(_links_as_of(links, until=2_000.0)) == 1

    def test_event_time_in_metadata_cannot_smuggle_in_a_future_event(self) -> None:
        links = [_link("event_involves", created_at=100.0, metadata={"event_time": 9_000.0})]
        assert _links_as_of(links, until=2_000.0) == []

    def test_unclassified_relation_is_gated_and_logged(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="agent.models.gnn.graph_builder"):
            kept = _links_as_of([_link("some_new_relation", created_at=9_000.0)], until=2_000.0)
        assert kept == [], "an unclassified relation must be gated — the safe side"
        assert "some_new_relation" in unclassified_link_relations()
        assert any("some_new_relation" in r.message for r in caplog.records)

    def test_until_none_keeps_everything(self) -> None:
        links = [_link("event_involves", created_at=9e9), _link("produced_in", created_at=9e9)]
        assert len(_links_as_of(links, until=None)) == 2

    def test_link_without_created_at_is_still_kept(self) -> None:
        """Pre-column links are kept, not silently dropped (documented trade)."""
        assert len(_links_as_of([{"entity_id_a": "a", "entity_id_b": "b"}], until=1_000.0)) == 1


class TestHistoricalWindowKeepsItsGraph:
    """The integration half — F-14's lesson: testing the filter is not testing the gate."""

    @staticmethod
    def _seed(store: PipelineStore) -> tuple[str, str]:
        inst = entity_id_from_key("instrument", "XOM")
        ctry = entity_id_from_key("country", "US")
        store.register_entity("instrument", "XOM", inst)
        store.register_entity("country", "US", ctry)
        store.store_entity_observation(
            entity_id=inst,
            source_tool="test",
            observed_at=_T_OBS,
            observation_type="instrument_daily",
            value={"close": 100.0, "log_return": 0.01, "volume": 1000},
        )
        store.store_entity_observation(
            entity_id=ctry,
            source_tool="test",
            observed_at=_T_OBS,
            observation_type="geopolitical_event",
            value={"goldstein": -4.0, "num_mentions": 7},
        )
        return inst, ctry

    @staticmethod
    def _link_at(store: PipelineStore, a: str, b: str, link_type: str, created_at: float) -> None:
        conn = store._get_conn()  # noqa: SLF001 — link_entities stamps now()
        conn.execute(
            "INSERT OR IGNORE INTO entity_links "
            "(entity_id_a, entity_id_b, link_type, confidence, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (a, b, link_type, 1.0, "test", created_at),
        )
        conn.commit()

    @staticmethod
    def _edges(data) -> int:
        return sum(int(data[et].edge_index.shape[1]) for et in data.edge_types)

    def test_structural_edge_survives_and_event_edge_does_not(self, store: PipelineStore) -> None:
        inst, ctry = self._seed(store)
        self._link_at(store, inst, ctry, "produced_in", created_at=_T_INGEST)
        self._link_at(store, ctry, inst, "event_involves", created_at=_T_INGEST)

        builder = GraphBuilder(store)
        id_map, _, links = builder.prepare_static()
        obs = builder.prefetch_observations()
        data, _, _ = builder.build_from_cached(id_map, links, until=_T_WINDOW, observations=obs)

        relations = {et[1] for et in data.edge_types}
        assert "produced_in" in relations, (
            "the historical window has no structural edges — HetTGN skips the "
            "whole HGT loop on an empty edge_index_dict, so this is the graph "
            "network silently becoming an MLP"
        )
        assert "event_involves" not in relations, "a future event edge leaked (F-04)"
        assert "rev_event_involves" not in relations
        assert self._edges(data) > 0

    def test_event_edge_appears_once_the_window_reaches_it(self, store: PipelineStore) -> None:
        inst, ctry = self._seed(store)
        self._link_at(store, ctry, inst, "event_involves", created_at=_T_INGEST)

        builder = GraphBuilder(store)
        id_map, _, links = builder.prepare_static()
        obs = builder.prefetch_observations()
        data, _, _ = builder.build_from_cached(id_map, links, until=_T_INGEST + 1.0, observations=obs)

        assert "event_involves" in {et[1] for et in data.edge_types}

    def test_omitting_until_is_still_a_typeerror(self, store: PipelineStore) -> None:
        """F-14's rule: never restore a default to build_from_cached(until=...)."""
        self._seed(store)
        builder = GraphBuilder(store)
        id_map, _, links = builder.prepare_static()
        with pytest.raises(TypeError, match="until"):
            builder.build_from_cached(id_map, links)  # type: ignore[call-arg]


# ═══════════════════════════════════════════════════════════════
# C. Reverse edges
# ═══════════════════════════════════════════════════════════════


class TestReverseEdges:
    @staticmethod
    def _id_map() -> IDMap:
        m = IDMap()
        m.add("instrument", "xom")
        m.add("country", "us")
        return m

    @staticmethod
    def _links() -> list[dict]:
        return [
            {
                "entity_id_a": "xom",
                "entity_id_b": "us",
                "link_type": "produced_in",
                "confidence": 0.9,
                "created_at": 1_000.0,
            }
        ]

    def test_reverse_relation_is_emitted(self) -> None:
        result = _build_edge_data(self._links(), self._id_map(), reference_time=1_000.0)
        fwd = ("instrument", "produced_in", "country")
        rev = ("country", "rev_produced_in", "instrument")
        assert fwd in result
        assert rev in result, (
            "no reverse relation — HGT passes messages src→dst only, so the "
            "country node can never influence the instrument the loss is on"
        )
        assert result[rev]["edge_index"][0].tolist() == result[fwd]["edge_index"][1].tolist()
        assert result[rev]["edge_index"][1].tolist() == result[fwd]["edge_index"][0].tolist()

    def test_reverse_carries_the_same_edge_attr(self) -> None:
        result = _build_edge_data(self._links(), self._id_map(), reference_time=1_000.0)
        fwd = result[("instrument", "produced_in", "country")]["edge_attr"]
        rev = result[("country", "rev_produced_in", "instrument")]["edge_attr"]
        assert rev.shape == fwd.shape
        assert rev.tolist() == fwd.tolist()

    def test_reverse_edges_can_be_disabled(self) -> None:
        result = _build_edge_data(
            self._links(),
            self._id_map(),
            reference_time=1_000.0,
            add_reverse_edges=False,
        )
        assert list(result) == [("instrument", "produced_in", "country")]

    def test_no_reverse_of_a_reverse(self) -> None:
        links = [
            {
                "entity_id_a": "us",
                "entity_id_b": "xom",
                "link_type": "rev_produced_in",
                "confidence": 1.0,
                "created_at": 1_000.0,
            }
        ]
        result = _build_edge_data(links, self._id_map(), reference_time=1_000.0)
        assert ("instrument", "rev_rev_produced_in", "country") not in result

    def test_reverse_naming_follows_pyg_convention(self) -> None:
        assert reverse_relation("produced_in") == "rev_produced_in"

    def test_gdelt_country_edges_can_reach_the_instrument(self) -> None:
        """The concrete 9,487-edge case: country→instrument had no return path.

        `event_involves` points country→instrument, and every instrument-side
        relation points out of the instrument, so before the inverse existed no
        message from a country could arrive at an instrument node at any depth.
        """
        m = IDMap()
        m.add("instrument", "xom")
        m.add("country", "us")
        links = [
            {
                "entity_id_a": "xom",
                "entity_id_b": "us",
                "link_type": "produced_in",
                "confidence": 1.0,
                "created_at": 1_000.0,
            }
        ]
        result = _build_edge_data(links, m, reference_time=1_000.0)
        into_instrument = [et for et in result if et[2] == "instrument"]
        assert into_instrument, "nothing in the graph can deliver a message to an instrument node"
