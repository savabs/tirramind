"""Phase 11a.5 — Edge case test suite for entity_links + co-occurrences.

Covers:
- link_entities: create, dedup/idempotent, self-link rejection, metadata round-trip
- query_entity_links: direction filter, link_type filter, confidence threshold,
  empty results, limit, multiple links
- query_co_occurrences: basic pair, window boundary precision, empty, single-tool
  filter, since filter, multiple matches, time_delta sign, large gap
- Integration: real store round-trip, backward compat (existing entity methods
  unaffected)
"""

from __future__ import annotations

import time

import pytest

from agent.pipeline.store import PipelineStore

# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def store() -> PipelineStore:
    return PipelineStore(":memory:")


def _seed_entities(store: PipelineStore) -> tuple[str, str, str]:
    """Create three entities: company, country, vessel. Return their IDs."""
    cid = "c" * 16
    kid = "k" * 16
    vid = "v" * 16
    store.register_entity("company", "Acme Corp", cid)
    store.register_entity("country", "US", kid)
    store.register_entity("vessel", "MV Test", vid)
    return cid, kid, vid


# ── link_entities tests ──────────────────────────────────────


class TestLinkEntities:
    def test_create_link(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        link_id = store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        assert link_id is not None
        assert isinstance(link_id, int)
        assert link_id > 0

    def test_idempotent_on_duplicate(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        id1 = store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        id2 = store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        assert id1 is not None
        assert id2 is None  # already existed

    def test_same_pair_different_type(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        id1 = store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        id2 = store.link_entities(cid, kid, "operates_in", "manual")
        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

    def test_self_link_rejected(self, store: PipelineStore) -> None:
        cid, _, _ = _seed_entities(store)
        with pytest.raises(ValueError, match="Cannot link an entity to itself"):
            store.link_entities(cid, cid, "related_to", "test")

    def test_metadata_round_trip(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        meta = {"source_file": "tickers.json", "year": 2024}
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers", metadata=meta)
        links = store.query_entity_links(cid, link_type="headquartered_in")
        assert len(links) == 1
        assert links[0]["metadata"] == meta

    def test_confidence_stored(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers", confidence=0.8)
        links = store.query_entity_links(cid, link_type="headquartered_in")
        assert links[0]["confidence"] == pytest.approx(0.8)

    def test_default_confidence_is_one(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        links = store.query_entity_links(cid, link_type="headquartered_in")
        assert links[0]["confidence"] == pytest.approx(1.0)


# ── query_entity_links tests ────────────────────────────────


class TestQueryEntityLinks:
    def test_outgoing_direction(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        out = store.query_entity_links(cid, direction="outgoing")
        inc = store.query_entity_links(cid, direction="incoming")
        assert len(out) == 1
        assert len(inc) == 0

    def test_incoming_direction(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        inc = store.query_entity_links(kid, direction="incoming")
        assert len(inc) == 1
        assert inc[0]["entity_id_a"] == cid

    def test_both_direction(self, store: PipelineStore) -> None:
        cid, kid, vid = _seed_entities(store)
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        store.link_entities(vid, cid, "docked_at", "ais_data")
        both = store.query_entity_links(cid, direction="both")
        assert len(both) == 2

    def test_invalid_direction_rejected(self, store: PipelineStore) -> None:
        cid, _, _ = _seed_entities(store)
        with pytest.raises(ValueError, match="direction must be"):
            store.query_entity_links(cid, direction="sideways")

    def test_link_type_filter(self, store: PipelineStore) -> None:
        cid, kid, vid = _seed_entities(store)
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        store.link_entities(cid, vid, "cargo_of", "ais_data")
        hq_only = store.query_entity_links(cid, link_type="headquartered_in", direction="outgoing")
        assert len(hq_only) == 1
        assert hq_only[0]["link_type"] == "headquartered_in"

    def test_confidence_filter(self, store: PipelineStore) -> None:
        cid, kid, vid = _seed_entities(store)
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers", confidence=0.9)
        store.link_entities(cid, vid, "cargo_of", "ais_data", confidence=0.3)
        high = store.query_entity_links(cid, direction="outgoing", min_confidence=0.5)
        assert len(high) == 1
        assert high[0]["entity_id_b"] == kid

    def test_empty_result(self, store: PipelineStore) -> None:
        cid, _, _ = _seed_entities(store)
        links = store.query_entity_links(cid)
        assert links == []

    def test_limit(self, store: PipelineStore) -> None:
        cid, kid, vid = _seed_entities(store)
        store.link_entities(cid, kid, "type_a", "src")
        store.link_entities(cid, vid, "type_b", "src")
        limited = store.query_entity_links(cid, direction="outgoing", limit=1)
        assert len(limited) == 1

    def test_link_dict_keys(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")
        link = store.query_entity_links(cid)[0]
        for key in (
            "link_id",
            "entity_id_a",
            "entity_id_b",
            "link_type",
            "confidence",
            "source",
            "created_at",
            "metadata",
        ):
            assert key in link, f"Missing key: {key}"
        assert "metadata_json" not in link  # should be decoded


# ── query_co_occurrences tests ──────────────────────────────


def _seed_obs(
    store: PipelineStore, entity_id: str, tool: str, ts: float, obs_type: str, value: dict | None = None
) -> int:
    """Helper to store an observation and return row id."""
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool=tool,
        observed_at=ts,
        observation_type=obs_type,
        value=value or {},
        depth_level=2,
    )


class TestQueryCoOccurrences:
    def test_basic_pair(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        _seed_obs(store, cid, "insider_filings", t0, "insider_trade", {"action": "sell"})
        _seed_obs(store, kid, "gdelt", t0 + 3600, "geopolitical_event", {"goldstein": -5})

        cooccs = store.query_co_occurrences(cid, kid, window_seconds=7200)
        assert len(cooccs) == 1
        assert cooccs[0]["obs_a"]["entity_id"] == cid
        assert cooccs[0]["obs_b"]["entity_id"] == kid
        # a is 3600s before b → time_delta = a.observed_at - b.observed_at = -3600
        assert cooccs[0]["time_delta_seconds"] == pytest.approx(-3600.0)

    def test_window_boundary_exact(self, store: PipelineStore) -> None:
        """Observation pair at exactly window_seconds apart should be included."""
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        _seed_obs(store, cid, "insider_filings", t0, "insider_trade")
        _seed_obs(store, kid, "gdelt", t0 + 100, "geopolitical_event")

        included = store.query_co_occurrences(cid, kid, window_seconds=100)
        assert len(included) == 1

    def test_window_boundary_excluded(self, store: PipelineStore) -> None:
        """Observation pair just outside window should be excluded."""
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        _seed_obs(store, cid, "insider_filings", t0, "insider_trade")
        _seed_obs(store, kid, "gdelt", t0 + 101, "geopolitical_event")

        excluded = store.query_co_occurrences(cid, kid, window_seconds=100)
        assert len(excluded) == 0

    def test_empty_no_observations(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        cooccs = store.query_co_occurrences(cid, kid)
        assert cooccs == []

    def test_source_tool_filter_a(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        _seed_obs(store, cid, "insider_filings", t0, "insider_trade")
        _seed_obs(store, cid, "whale_alert", t0 + 10, "btc_transfer")
        _seed_obs(store, kid, "gdelt", t0 + 60, "geopolitical_event")

        # Only insider_filings side for entity A
        cooccs = store.query_co_occurrences(cid, kid, source_tool_a="insider_filings", window_seconds=7200)
        assert len(cooccs) == 1
        assert cooccs[0]["obs_a"]["source_tool"] == "insider_filings"

    def test_source_tool_filter_b(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        _seed_obs(store, cid, "insider_filings", t0, "insider_trade")
        _seed_obs(store, kid, "gdelt", t0 + 60, "geopolitical_event")
        _seed_obs(store, kid, "ais_vessel", t0 + 60, "vessel_position")

        cooccs = store.query_co_occurrences(cid, kid, source_tool_b="gdelt", window_seconds=7200)
        assert len(cooccs) == 1
        assert cooccs[0]["obs_b"]["source_tool"] == "gdelt"

    def test_since_filter(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        old = 1_600_000_000.0
        recent = 1_700_000_000.0
        _seed_obs(store, cid, "insider_filings", old, "insider_trade")
        _seed_obs(store, kid, "gdelt", old + 60, "geopolitical_event")
        _seed_obs(store, cid, "insider_filings", recent, "insider_trade")
        _seed_obs(store, kid, "gdelt", recent + 60, "geopolitical_event")

        cooccs = store.query_co_occurrences(cid, kid, since=1_650_000_000.0, window_seconds=7200)
        assert len(cooccs) == 1
        assert cooccs[0]["obs_a"]["observed_at"] == pytest.approx(recent)

    def test_multiple_matches(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        for i in range(3):
            _seed_obs(store, cid, "insider_filings", t0 + i * 100, "insider_trade")
        _seed_obs(store, kid, "gdelt", t0 + 150, "geopolitical_event")

        cooccs = store.query_co_occurrences(cid, kid, window_seconds=200)
        # All three insider obs are within 200s of the GDELT event
        assert len(cooccs) == 3

    def test_time_delta_sign_positive(self, store: PipelineStore) -> None:
        """When obs_a happens AFTER obs_b, time_delta is positive."""
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        _seed_obs(store, cid, "insider_filings", t0 + 1000, "insider_trade")
        _seed_obs(store, kid, "gdelt", t0, "geopolitical_event")

        cooccs = store.query_co_occurrences(cid, kid, window_seconds=2000)
        assert len(cooccs) == 1
        assert cooccs[0]["time_delta_seconds"] == pytest.approx(1000.0)

    def test_large_gap_excluded(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        _seed_obs(store, cid, "insider_filings", 1_000_000.0, "insider_trade")
        _seed_obs(store, kid, "gdelt", 2_000_000.0, "geopolitical_event")

        cooccs = store.query_co_occurrences(cid, kid, window_seconds=72 * 3600)
        assert len(cooccs) == 0

    def test_obs_value_decoded(self, store: PipelineStore) -> None:
        """Values in co-occurrence results should be JSON-decoded dicts."""
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        _seed_obs(store, cid, "insider_filings", t0, "insider_trade", {"shares": 5000})
        _seed_obs(store, kid, "gdelt", t0 + 60, "geopolitical_event", {"goldstein": -3.5})

        cooccs = store.query_co_occurrences(cid, kid, window_seconds=7200)
        assert cooccs[0]["obs_a"]["value"] == {"shares": 5000}
        assert cooccs[0]["obs_b"]["value"] == {"goldstein": -3.5}

    def test_limit_respected(self, store: PipelineStore) -> None:
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        for i in range(5):
            _seed_obs(store, cid, "insider_filings", t0 + i, "insider_trade")
        _seed_obs(store, kid, "gdelt", t0 + 2, "geopolitical_event")

        cooccs = store.query_co_occurrences(cid, kid, window_seconds=10, limit=2)
        assert len(cooccs) == 2

    def test_ordered_by_abs_time_delta(self, store: PipelineStore) -> None:
        """Results ordered by closest temporal match first."""
        cid, kid, _ = _seed_entities(store)
        t0 = 1_700_000_000.0
        _seed_obs(store, cid, "insider_filings", t0, "insider_trade")  # delta=100
        _seed_obs(store, cid, "insider_filings", t0 + 90, "insider_trade")  # delta=10
        _seed_obs(store, kid, "gdelt", t0 + 100, "geopolitical_event")

        cooccs = store.query_co_occurrences(cid, kid, window_seconds=200)
        assert len(cooccs) == 2
        # Closest first
        assert abs(cooccs[0]["time_delta_seconds"]) <= abs(cooccs[1]["time_delta_seconds"])


# ── integration / backward compat ────────────────────────────


class TestIntegration:
    def test_entity_links_round_trip(self, store: PipelineStore) -> None:
        """Full round-trip: register entities, link, query back."""
        cid = store.register_entity("company", "Test Co", "a" * 16)
        kid = store.register_entity("country", "DE", "b" * 16)
        link_id = store.link_entities(
            cid, kid, "headquartered_in", "sec_tickers", confidence=0.95, metadata={"region": "EU"}
        )
        assert link_id is not None
        links = store.query_entity_links(cid, link_type="headquartered_in")
        assert len(links) == 1
        assert links[0]["entity_id_a"] == cid
        assert links[0]["entity_id_b"] == kid
        assert links[0]["confidence"] == pytest.approx(0.95)
        assert links[0]["metadata"]["region"] == "EU"

    def test_existing_entity_methods_unaffected(self, store: PipelineStore) -> None:
        """Backward compat: existing entity CRUD still works after schema change."""
        eid = store.register_entity("company", "Old Corp", "x" * 16)
        store.add_entity_alias(eid, "sec", "CIK000123")
        resolved = store.resolve_entity("sec", "CIK000123")
        assert resolved == eid

        row_id = store.store_entity_observation(
            entity_id=eid,
            source_tool="insider_filings",
            observed_at=time.time(),
            observation_type="insider_trade",
            value={"action": "buy"},
            depth_level=2,
        )
        assert row_id > 0
        obs = store.query_entity_observations(eid)
        assert len(obs) == 1
        assert obs[0]["value"]["action"] == "buy"

    def test_co_occurrence_with_links(self, store: PipelineStore) -> None:
        """End-to-end: link entities, add observations, query co-occurrences."""
        cid = store.register_entity("company", "Acme", "a" * 16)
        kid = store.register_entity("country", "RU", "b" * 16)
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        store.store_entity_observation(
            cid, "insider_filings", t0, "insider_trade", {"action": "sell", "shares": 10000}, 2
        )
        store.store_entity_observation(
            kid, "gdelt", t0 + 7200, "geopolitical_event", {"goldstein": -8.0, "event_code": "190"}, 2
        )

        # Query links to find paired entity
        links = store.query_entity_links(cid, link_type="headquartered_in")
        assert len(links) == 1
        country_id = links[0]["entity_id_b"]

        # Query co-occurrences
        cooccs = store.query_co_occurrences(cid, country_id, window_seconds=72 * 3600)
        assert len(cooccs) == 1
        assert cooccs[0]["obs_a"]["value"]["action"] == "sell"
        assert cooccs[0]["obs_b"]["value"]["goldstein"] == -8.0
