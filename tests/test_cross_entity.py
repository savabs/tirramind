"""Phase 11b.5 — Edge case test suite for cross_entity module.

Covers:
- seed_company_country_links: basic seeding, missing company entities skip,
  duplicate link idempotency, empty tickers file
- CrossEntityDetector.detect_insider_gdelt: basic hit, no-hit (no links),
  no-hit (no obs), multiple co-occurrences, Goldstein filter, score calc,
  boundary window, since filter
- store_l3_observations: basic store, min_score filter, empty list,
  depth_level=3 verification, pattern value round-trip
- Integration: full end-to-end pipeline
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent.pipeline.cross_entity import (
    DEFAULT_WINDOW_SECONDS,
    GOLDSTEIN_THRESHOLD,
    CrossEntityDetector,
    seed_company_country_links,
    seed_vessel_country_links,
    resolve_port_country,
    SANCTIONS_ROOT_CODES,
    VESSEL_WINDOW_SECONDS,
    WHALE_GOLDSTEIN_THRESHOLD,
    WHALE_VALUE_SCALE,
    WHALE_WINDOW_SECONDS,
    resolve_wallet_exchange,
    seed_whale_country_links,
)
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore


# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def store() -> PipelineStore:
    return PipelineStore(":memory:")


@pytest.fixture()
def tickers_file(tmp_path: Path) -> Path:
    """Minimal SEC tickers JSON for testing."""
    data = {
        "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": "789019", "ticker": "MSFT", "title": "Microsoft Corp"},
        "2": {"cik_str": "1652044", "ticker": "GOOGL", "title": "Alphabet Inc."},
    }
    p = tmp_path / "tickers.json"
    p.write_text(json.dumps(data))
    return p


def _register_company(store: PipelineStore, cik: str, name: str) -> str:
    eid = entity_id_from_key("company", cik)
    store.register_entity("company", name, eid, metadata={"cik": cik})
    store.add_entity_alias(eid, "sec_cik", cik)
    return eid


def _register_country(store: PipelineStore, fips: str, name: str) -> str:
    eid = entity_id_from_key("country", fips)
    store.register_entity("country", name, eid, metadata={"fips": fips})
    store.add_entity_alias(eid, "fips_country", fips)
    return eid


def _add_insider_obs(
    store: PipelineStore, entity_id: str, ts: float, value: dict | None = None
) -> int:
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool="insider_filings",
        observed_at=ts,
        observation_type="insider_trade",
        value=value or {"action": "sell", "shares": 5000},
        depth_level=2,
    )


def _add_gdelt_obs(
    store: PipelineStore,
    entity_id: str,
    ts: float,
    goldstein: float = -5.0,
    value: dict | None = None,
) -> int:
    v = value or {"goldstein": goldstein, "event_code": "190"}
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool="gdelt",
        observed_at=ts,
        observation_type="geopolitical_event",
        value=v,
        depth_level=2,
    )


# ── seed_company_country_links tests ──────────────────────────


class TestSeedCompanyCountryLinks:
    def test_basic_seeding(self, store: PipelineStore, tickers_file: Path) -> None:
        # Register the companies first (seeder only links existing entities)
        _register_company(store, "320193", "apple")
        _register_company(store, "789019", "microsoft")
        _register_company(store, "1652044", "alphabet")

        count = seed_company_country_links(store, str(tickers_file))
        assert count == 3

        # Verify the US country entity was created
        us_id = entity_id_from_key("country", "US")
        us = store.get_entity(us_id)
        assert us is not None
        assert us["entity_type"] == "country"

    def test_skips_unregistered_companies(
        self, store: PipelineStore, tickers_file: Path
    ) -> None:
        # Only register one of the three
        _register_company(store, "320193", "apple")
        count = seed_company_country_links(store, str(tickers_file))
        assert count == 1

    def test_idempotent(self, store: PipelineStore, tickers_file: Path) -> None:
        _register_company(store, "320193", "apple")
        c1 = seed_company_country_links(store, str(tickers_file))
        c2 = seed_company_country_links(store, str(tickers_file))
        assert c1 == 1
        assert c2 == 0  # all links already exist

    def test_empty_tickers(self, store: PipelineStore, tmp_path: Path) -> None:
        empty = tmp_path / "empty.json"
        empty.write_text("{}")
        count = seed_company_country_links(store, str(empty))
        assert count == 0

    def test_link_type_and_source(
        self, store: PipelineStore, tickers_file: Path
    ) -> None:
        cid = _register_company(store, "320193", "apple")
        seed_company_country_links(store, str(tickers_file))
        links = store.query_entity_links(cid, direction="outgoing")
        assert len(links) == 1
        assert links[0]["link_type"] == "headquartered_in"
        assert links[0]["source"] == "sec_tickers"


# ── detect_insider_gdelt tests ──────────────────────────────


class TestDetectInsiderGdelt:
    def test_basic_hit(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        _add_gdelt_obs(store, kid, t0 + 3600, goldstein=-5.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        assert len(patterns) == 1
        assert patterns[0]["pattern_type"] == "insider_x_gdelt"
        assert patterns[0]["entity_a"] == cid
        assert patterns[0]["entity_b"] == kid
        assert patterns[0]["goldstein"] == -5.0
        assert patterns[0]["score"] > 0

    def test_no_link_returns_empty(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        assert patterns == []

    def test_no_observations_returns_empty(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        assert patterns == []

    def test_positive_goldstein_filtered(self, store: PipelineStore) -> None:
        """Cooperative events (positive Goldstein) should not trigger patterns."""
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        _add_gdelt_obs(store, kid, t0 + 3600, goldstein=5.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        assert patterns == []

    def test_goldstein_at_threshold_included(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        _add_gdelt_obs(store, kid, t0 + 3600, goldstein=GOLDSTEIN_THRESHOLD)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        assert len(patterns) == 1

    def test_goldstein_just_above_threshold_excluded(
        self, store: PipelineStore
    ) -> None:
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        _add_gdelt_obs(store, kid, t0 + 3600, goldstein=GOLDSTEIN_THRESHOLD + 0.1)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        assert patterns == []

    def test_multiple_cooccurrences(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        _add_insider_obs(store, cid, t0 + 1000)
        _add_gdelt_obs(store, kid, t0 + 500, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid, window_seconds=3600)
        assert len(patterns) == 2

    def test_score_increases_with_proximity(self, store: PipelineStore) -> None:
        """Closer co-occurrences should score higher."""
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        _add_insider_obs(store, cid, t0 + 3500)
        _add_gdelt_obs(store, kid, t0 + 3600, goldstein=-5.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid, window_seconds=7200)

        # Sort by abs(time_delta) to identify close vs far
        by_delta = sorted(patterns, key=lambda p: abs(p["time_delta_hours"]))
        assert by_delta[0]["score"] > by_delta[1]["score"]

    def test_score_increases_with_severity(self, store: PipelineStore) -> None:
        """More negative Goldstein should score higher (same proximity)."""
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        _add_gdelt_obs(store, kid, t0 + 100, goldstein=-3.0)
        _add_gdelt_obs(store, kid, t0 + 200, goldstein=-9.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid, window_seconds=7200)
        assert len(patterns) == 2
        # Find each by goldstein
        g3 = [p for p in patterns if p["goldstein"] == -3.0][0]
        g9 = [p for p in patterns if p["goldstein"] == -9.0][0]
        assert g9["score"] > g3["score"]

    def test_since_filter(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        old = 1_600_000_000.0
        recent = 1_700_000_000.0
        _add_insider_obs(store, cid, old)
        _add_gdelt_obs(store, kid, old + 3600, goldstein=-5.0)
        _add_insider_obs(store, cid, recent)
        _add_gdelt_obs(store, kid, recent + 3600, goldstein=-5.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid, since=1_650_000_000.0)
        assert len(patterns) == 1

    def test_missing_goldstein_skipped(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        # GDELT event without goldstein field
        _add_gdelt_obs(store, kid, t0 + 100, value={"event_code": "190"})

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        assert patterns == []

    def test_custom_goldstein_threshold(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        _add_gdelt_obs(store, kid, t0 + 3600, goldstein=-3.0)

        detector = CrossEntityDetector(store)
        # Default threshold (-2) would include -3
        with_default = detector.detect_insider_gdelt(cid)
        assert len(with_default) == 1
        # Stricter threshold excludes it
        with_strict = detector.detect_insider_gdelt(cid, goldstein_threshold=-5.0)
        assert len(with_strict) == 0

    def test_obs_ids_in_pattern(self, store: PipelineStore) -> None:
        """Patterns should include observation IDs for traceability."""
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        insider_id = _add_insider_obs(store, cid, t0)
        gdelt_id = _add_gdelt_obs(store, kid, t0 + 3600, goldstein=-5.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        assert patterns[0]["obs_a_id"] == insider_id
        assert patterns[0]["obs_b_id"] == gdelt_id


# ── store_l3_observations tests ──────────────────────────────


class TestStoreL3Observations:
    def test_basic_store(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        patterns = [
            {
                "pattern_type": "insider_x_gdelt",
                "entity_a": cid,
                "entity_b": "b" * 16,
                "insider_event": {"action": "sell"},
                "gdelt_event": {"goldstein": -5},
                "time_delta_hours": -1.0,
                "goldstein": -5.0,
                "score": 0.45,
                "obs_a_id": 1,
                "obs_b_id": 2,
            }
        ]

        detector = CrossEntityDetector(store)
        count = detector.store_l3_observations(patterns)
        assert count == 1

    def test_min_score_filter(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        patterns = [
            {"pattern_type": "t", "entity_a": cid, "score": 0.1},
            {"pattern_type": "t", "entity_a": cid, "score": 0.5},
            {"pattern_type": "t", "entity_a": cid, "score": 0.8},
        ]
        detector = CrossEntityDetector(store)
        count = detector.store_l3_observations(patterns, min_score=0.4)
        assert count == 2

    def test_empty_list(self, store: PipelineStore) -> None:
        detector = CrossEntityDetector(store)
        count = detector.store_l3_observations([])
        assert count == 0

    def test_depth_level_3(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        patterns = [
            {
                "pattern_type": "insider_x_gdelt",
                "entity_a": cid,
                "entity_b": "b" * 16,
                "score": 0.5,
            }
        ]
        detector = CrossEntityDetector(store)
        detector.store_l3_observations(patterns)

        obs = store.query_entity_observations(cid, depth_level=3)
        assert len(obs) == 1
        assert obs[0]["depth_level"] == 3
        assert obs[0]["observation_type"] == "cross_entity_pattern"
        assert obs[0]["source_tool"] == "cross_entity"

    def test_pattern_value_round_trip(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        patterns = [
            {
                "pattern_type": "insider_x_gdelt",
                "entity_a": cid,
                "entity_b": "b" * 16,
                "insider_event": {"action": "sell", "shares": 10000},
                "gdelt_event": {"goldstein": -8.0},
                "time_delta_hours": -2.0,
                "goldstein": -8.0,
                "score": 0.72,
                "obs_a_id": 1,
                "obs_b_id": 2,
            }
        ]
        detector = CrossEntityDetector(store)
        detector.store_l3_observations(patterns)

        obs = store.query_entity_observations(cid, depth_level=3)
        val = obs[0]["value"]
        assert val["pattern_type"] == "insider_x_gdelt"
        assert val["score"] == 0.72
        assert val["insider_event"]["shares"] == 10000

    def test_metadata_has_pattern_type(self, store: PipelineStore) -> None:
        cid = _register_company(store, "320193", "apple")
        patterns = [
            {
                "pattern_type": "insider_x_gdelt",
                "entity_a": cid,
                "score": 0.5,
            }
        ]
        detector = CrossEntityDetector(store)
        detector.store_l3_observations(patterns)

        obs = store.query_entity_observations(cid, depth_level=3)
        assert obs[0]["metadata"]["pattern_type"] == "insider_x_gdelt"


# ── integration ──────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline(self, store: PipelineStore, tickers_file: Path) -> None:
        """End-to-end: register → seed links → add obs → detect → store L3."""
        # 1. Register entities
        cid = _register_company(store, "320193", "apple")

        # 2. Seed company→country links
        count = seed_company_country_links(store, str(tickers_file))
        assert count == 1  # only apple is registered

        # 3. Add observations
        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0, {"action": "sell", "shares": 50000})
        us_id = entity_id_from_key("country", "US")
        _add_gdelt_obs(store, us_id, t0 + 7200, goldstein=-6.0)

        # 4. Detect patterns
        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        assert len(patterns) == 1
        assert patterns[0]["goldstein"] == -6.0

        # 5. Store L3 observations
        stored = detector.store_l3_observations(patterns)
        assert stored == 1

        # 6. Query back
        l3_obs = store.query_entity_observations(cid, depth_level=3)
        assert len(l3_obs) == 1
        assert l3_obs[0]["value"]["pattern_type"] == "insider_x_gdelt"
        assert l3_obs[0]["value"]["insider_event"]["shares"] == 50000

    def test_no_cross_contamination(self, store: PipelineStore) -> None:
        """L3 observations should not appear in L2 queries."""
        cid = _register_company(store, "320193", "apple")
        kid = _register_country(store, "US", "United States")
        store.link_entities(cid, kid, "headquartered_in", "sec_tickers")

        t0 = 1_700_000_000.0
        _add_insider_obs(store, cid, t0)
        _add_gdelt_obs(store, kid, t0 + 3600, goldstein=-5.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(cid)
        detector.store_l3_observations(patterns)

        # L2 query should only return the original insider obs
        l2_obs = store.query_entity_observations(cid, depth_level=2)
        assert all(o["observation_type"] == "insider_trade" for o in l2_obs)

        # L3 query returns the pattern
        l3_obs = store.query_entity_observations(cid, depth_level=3)
        assert len(l3_obs) == 1
        assert l3_obs[0]["observation_type"] == "cross_entity_pattern"


# ══════════════════════════════════════════════════════════════
# Phase 11c: Vessel × Sanctions
# ══════════════════════════════════════════════════════════════


def _register_vessel(store: PipelineStore, imo: str, name: str) -> str:
    eid = entity_id_from_key("vessel", imo)
    store.register_entity("vessel", name, eid, metadata={"imo": imo})
    store.add_entity_alias(eid, "imo", imo)
    return eid


def _add_port_call_obs(
    store: PipelineStore,
    entity_id: str,
    ts: float,
    port: str = "HELSINKI",
    prev_port: str = "",
    next_port: str = "",
) -> int:
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool="ais_vessel",
        observed_at=ts,
        observation_type="port_call",
        value={
            "port": port,
            "prev_port": prev_port,
            "next_port": next_port,
            "arrival_with_cargo": True,
        },
        depth_level=2,
    )


def _add_vessel_pos_obs(
    store: PipelineStore,
    entity_id: str,
    ts: float,
    lat: float = 60.0,
    lon: float = 25.0,
) -> int:
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool="ais_vessel",
        observed_at=ts,
        observation_type="vessel_position",
        value={"lat": lat, "lon": lon, "sog": 12.0, "cog": 180.0},
        depth_level=2,
    )


def _add_sanctions_gdelt(
    store: PipelineStore,
    entity_id: str,
    ts: float,
    goldstein: float = -7.0,
    event_root_code: str = "17",
    quad_class: int = 4,
) -> int:
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool="gdelt",
        observed_at=ts,
        observation_type="geopolitical_event",
        value={
            "goldstein": goldstein,
            "event_root_code": event_root_code,
            "quad_class": quad_class,
            "event_code": "172",
        },
        depth_level=2,
    )


# ── resolve_port_country tests ──────────────────────────────


class TestResolvePortCountry:
    def test_un_locode_space_separated(self) -> None:
        assert resolve_port_country("RU LED") == "RS"

    def test_un_locode_known_prefix(self) -> None:
        assert resolve_port_country("RULED") == "RS"

    def test_baltic_port_helsinki(self) -> None:
        assert resolve_port_country("HELSINKI") == "FI"

    def test_baltic_port_st_petersburg(self) -> None:
        assert resolve_port_country("ST PETERSBURG") == "RS"

    def test_baltic_port_gdansk(self) -> None:
        assert resolve_port_country("GDANSK") == "PL"

    def test_case_insensitive(self) -> None:
        assert resolve_port_country("helsinki") == "FI"
        assert resolve_port_country("ru led") == "RS"

    def test_unknown_port_returns_none(self) -> None:
        assert resolve_port_country("UNKNOWN PORT") is None

    def test_empty_returns_none(self) -> None:
        assert resolve_port_country("") is None
        assert resolve_port_country(None) is None
        assert resolve_port_country("  ") is None

    def test_short_string_returns_none(self) -> None:
        assert resolve_port_country("AB") is None

    def test_iso_to_fips_germany(self) -> None:
        assert resolve_port_country("DE HAM") == "GM"

    def test_iso_to_fips_sweden(self) -> None:
        assert resolve_port_country("SE GOT") == "SW"


# ── seed_vessel_country_links tests ─────────────────────────


class TestSeedVesselCountryLinks:
    def test_basic_seeding(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        t0 = 1_700_000_000.0
        _add_port_call_obs(
            store,
            vid,
            t0,
            port="HELSINKI",
            prev_port="ST PETERSBURG",
            next_port="GDANSK",
        )

        count = seed_vessel_country_links(store)
        assert count == 3  # FI, RS, PL

        links = store.query_entity_links(
            vid, direction="outgoing", link_type="port_call_to"
        )
        countries = {lk["entity_id_b"] for lk in links}
        assert len(countries) == 3

    def test_no_observations_no_links(self, store: PipelineStore) -> None:
        _register_vessel(store, "9000001", "MV Test")
        count = seed_vessel_country_links(store)
        assert count == 0

    def test_idempotent(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        _add_port_call_obs(store, vid, 1_700_000_000.0, port="HELSINKI")

        c1 = seed_vessel_country_links(store)
        c2 = seed_vessel_country_links(store)
        assert c1 == 1
        assert c2 == 0

    def test_unresolvable_port_skipped(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        _add_port_call_obs(store, vid, 1_700_000_000.0, port="MYSTERY HARBOR")

        count = seed_vessel_country_links(store)
        assert count == 0

    def test_multiple_ports_same_country(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        t0 = 1_700_000_000.0
        _add_port_call_obs(store, vid, t0, port="HELSINKI")
        _add_port_call_obs(store, vid, t0 + 3600, port="KOTKA")

        count = seed_vessel_country_links(store)
        assert count == 1  # both Finnish → one link to FI

    def test_link_type_is_port_call_to(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        _add_port_call_obs(store, vid, 1_700_000_000.0, port="RIGA")

        seed_vessel_country_links(store)
        links = store.query_entity_links(vid, direction="outgoing")
        assert links[0]["link_type"] == "port_call_to"
        assert links[0]["source"] == "ais_vessel_obs"


# ── detect_vessel_sanctions tests ────────────────────────────


class TestDetectVesselSanctions:
    def test_basic_hit(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        t0 = 1_700_000_000.0
        _add_port_call_obs(store, vid, t0, port="ST PETERSBURG")
        _add_sanctions_gdelt(store, kid, t0 + 3600, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert len(patterns) == 1
        assert patterns[0]["pattern_type"] == "vessel_x_sanctions"
        assert patterns[0]["entity_a"] == vid
        assert patterns[0]["entity_b"] == kid
        assert patterns[0]["goldstein"] == -7.0
        assert patterns[0]["score"] > 0

    def test_no_links_returns_empty(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert patterns == []

    def test_no_observations_returns_empty(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert patterns == []

    def test_non_sanctions_root_code_filtered(self, store: PipelineStore) -> None:
        """CAMEO root code 01 (public statement) should not trigger."""
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        t0 = 1_700_000_000.0
        _add_port_call_obs(store, vid, t0, port="ST PETERSBURG")
        # Root code 01 = Make Public Statement, not sanctions
        _add_sanctions_gdelt(
            store, kid, t0 + 3600, goldstein=-3.0, event_root_code="01", quad_class=3
        )

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert patterns == []

    def test_quad_class_4_included_even_without_sanctions_code(
        self, store: PipelineStore
    ) -> None:
        """Material conflict (quad_class=4) should pass even with non-sanctions root code."""
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        t0 = 1_700_000_000.0
        _add_port_call_obs(store, vid, t0, port="ST PETERSBURG")
        # Root code 19 = Fight, but quad_class=4 should still match
        _add_sanctions_gdelt(
            store, kid, t0 + 3600, goldstein=-8.0, event_root_code="19", quad_class=4
        )

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert len(patterns) == 1

    def test_positive_goldstein_filtered(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        t0 = 1_700_000_000.0
        _add_port_call_obs(store, vid, t0, port="ST PETERSBURG")
        _add_sanctions_gdelt(store, kid, t0 + 3600, goldstein=5.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert patterns == []

    def test_multiple_cooccurrences(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        t0 = 1_700_000_000.0
        _add_port_call_obs(store, vid, t0, port="ST PETERSBURG")
        _add_port_call_obs(store, vid, t0 + 1000, port="ST PETERSBURG")
        _add_sanctions_gdelt(store, kid, t0 + 500, goldstein=-6.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid, window_seconds=3600)
        assert len(patterns) == 2

    def test_score_increases_with_proximity(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        t0 = 1_700_000_000.0
        _add_port_call_obs(store, vid, t0, port="ST PETERSBURG")
        _add_port_call_obs(store, vid, t0 + 47 * 3600, port="ST PETERSBURG")
        _add_sanctions_gdelt(store, kid, t0 + 47.5 * 3600, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        by_delta = sorted(patterns, key=lambda p: abs(p["time_delta_hours"]))
        assert by_delta[0]["score"] > by_delta[1]["score"]

    def test_since_filter(self, store: PipelineStore) -> None:
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        old = 1_600_000_000.0
        recent = 1_700_000_000.0
        _add_port_call_obs(store, vid, old, port="ST PETERSBURG")
        _add_sanctions_gdelt(store, kid, old + 3600, goldstein=-7.0)
        _add_port_call_obs(store, vid, recent, port="ST PETERSBURG")
        _add_sanctions_gdelt(store, kid, recent + 3600, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid, since=1_650_000_000.0)
        assert len(patterns) == 1

    def test_pattern_has_vessel_obs_type(self, store: PipelineStore) -> None:
        """Pattern dict should include the vessel observation type."""
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        t0 = 1_700_000_000.0
        _add_port_call_obs(store, vid, t0, port="ST PETERSBURG")
        _add_sanctions_gdelt(store, kid, t0 + 3600, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert patterns[0]["vessel_obs_type"] == "port_call"
        assert patterns[0]["event_root_code"] == "17"

    def test_missing_root_code_with_quad4_still_matches(
        self, store: PipelineStore
    ) -> None:
        """Event with no event_root_code but quad_class=4."""
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        t0 = 1_700_000_000.0
        _add_port_call_obs(store, vid, t0, port="ST PETERSBURG")
        store.store_entity_observation(
            entity_id=kid,
            source_tool="gdelt",
            observed_at=t0 + 3600,
            observation_type="geopolitical_event",
            value={"goldstein": -5.0, "quad_class": 4},
            depth_level=2,
        )

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert len(patterns) == 1


# ── Vessel × Sanctions integration ──────────────────────────


class TestVesselSanctionsIntegration:
    def test_full_pipeline(self, store: PipelineStore) -> None:
        """End-to-end: register → seed links → add obs → detect → store L3."""
        # 1. Register vessel entity
        vid = _register_vessel(store, "9000001", "MV Arctic")

        # 2. Add port call observation (vessel visited Russia)
        t0 = 1_700_000_000.0
        _add_port_call_obs(
            store,
            vid,
            t0,
            port="HELSINKI",
            prev_port="ST PETERSBURG",
            next_port="GDANSK",
        )

        # 3. Seed vessel→country links from observations
        link_count = seed_vessel_country_links(store)
        assert link_count == 3  # FI, RS, PL

        # 4. Add GDELT sanctions event for Russia
        rs_id = entity_id_from_key("country", "RS")
        _add_sanctions_gdelt(
            store, rs_id, t0 + 7200, goldstein=-8.0, event_root_code="17"
        )

        # 5. Detect patterns
        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert len(patterns) == 1
        assert patterns[0]["goldstein"] == -8.0
        assert patterns[0]["pattern_type"] == "vessel_x_sanctions"

        # 6. Store L3 observations
        stored = detector.store_l3_observations(patterns)
        assert stored == 1

        # 7. Verify L3 obs stored correctly
        l3_obs = store.query_entity_observations(vid, depth_level=3)
        assert len(l3_obs) == 1
        assert l3_obs[0]["value"]["pattern_type"] == "vessel_x_sanctions"
        assert l3_obs[0]["source_tool"] == "cross_entity"

    def test_vessel_position_also_triggers(self, store: PipelineStore) -> None:
        """vessel_position observations should also co-occur with GDELT events."""
        vid = _register_vessel(store, "9000001", "MV Test")
        kid = _register_country(store, "RS", "Russia")
        store.link_entities(vid, kid, "port_call_to", "ais_vessel_obs")

        t0 = 1_700_000_000.0
        _add_vessel_pos_obs(store, vid, t0)
        _add_sanctions_gdelt(store, kid, t0 + 3600, goldstein=-6.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_vessel_sanctions(vid)
        assert len(patterns) == 1
        assert patterns[0]["vessel_obs_type"] == "vessel_position"


# ══════════════════════════════════════════════════════════════
# Phase 11d: Whale Crypto × Geopolitical
# ══════════════════════════════════════════════════════════════

_TEST_EXCHANGE_WALLETS: dict[str, tuple[str, str]] = {
    "1ExchangeUSaddr0001": ("test_exchange_us", "US"),
    "1ExchangeCJaddr0001": ("test_exchange_cj", "CJ"),
    "1ExchangeLUaddr0001": ("test_exchange_lu", "LU"),
}


def _register_wallet(store: PipelineStore, addr: str) -> str:
    eid = entity_id_from_key("wallet", addr)
    store.register_entity("wallet", addr, eid)
    store.add_entity_alias(eid, "btc_address", addr)
    return eid


def _add_btc_transfer_obs(
    store: PipelineStore,
    entity_id: str,
    ts: float,
    value_btc: float = 500.0,
    direction: str = "out",
) -> int:
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool="whale_alert",
        observed_at=ts,
        observation_type="btc_transfer",
        value={
            "tx_hash": "deadbeef0123456789",
            "value_btc": value_btc,
            "direction": direction,
            "counterparty_count": 1,
            "confirmed": True,
            "block_height": 800000,
        },
        depth_level=2,
    )


def _add_highimpact_gdelt(
    store: PipelineStore,
    entity_id: str,
    ts: float,
    goldstein: float = -7.0,
) -> int:
    return store.store_entity_observation(
        entity_id=entity_id,
        source_tool="gdelt",
        observed_at=ts,
        observation_type="geopolitical_event",
        value={"goldstein": goldstein, "event_root_code": "19", "event_code": "190"},
        depth_level=2,
    )


# ── resolve_wallet_exchange tests ────────────────────────────


class TestResolveWalletExchange:
    def test_hit(self) -> None:
        result = resolve_wallet_exchange("1ExchangeUSaddr0001", _TEST_EXCHANGE_WALLETS)
        assert result == ("test_exchange_us", "US")

    def test_miss(self) -> None:
        result = resolve_wallet_exchange("1UnknownAddr", _TEST_EXCHANGE_WALLETS)
        assert result is None

    def test_none_address(self) -> None:
        assert resolve_wallet_exchange(None, _TEST_EXCHANGE_WALLETS) is None

    def test_empty_address(self) -> None:
        assert resolve_wallet_exchange("", _TEST_EXCHANGE_WALLETS) is None
        assert resolve_wallet_exchange("  ", _TEST_EXCHANGE_WALLETS) is None

    def test_default_dict_empty(self) -> None:
        # With no override, uses module-level KNOWN_EXCHANGE_WALLETS (empty)
        assert resolve_wallet_exchange("anything") is None


# ── seed_whale_country_links tests ───────────────────────────


class TestSeedWhaleCountryLinks:
    def test_basic_seeding(self, store: PipelineStore) -> None:
        _register_wallet(store, "1ExchangeUSaddr0001")
        count = seed_whale_country_links(store, _TEST_EXCHANGE_WALLETS)
        assert count == 1

        wid = entity_id_from_key("wallet", "1ExchangeUSaddr0001")
        links = store.query_entity_links(
            wid, direction="outgoing", link_type="exchange_based_in"
        )
        assert len(links) == 1
        us_eid = entity_id_from_key("country", "US")
        assert links[0]["entity_id_b"] == us_eid

    def test_no_wallets_no_links(self, store: PipelineStore) -> None:
        count = seed_whale_country_links(store, _TEST_EXCHANGE_WALLETS)
        assert count == 0

    def test_idempotent(self, store: PipelineStore) -> None:
        _register_wallet(store, "1ExchangeUSaddr0001")
        c1 = seed_whale_country_links(store, _TEST_EXCHANGE_WALLETS)
        c2 = seed_whale_country_links(store, _TEST_EXCHANGE_WALLETS)
        assert c1 == 1
        assert c2 == 0

    def test_unknown_address_skipped(self, store: PipelineStore) -> None:
        _register_wallet(store, "1RandomWalletAddr")
        count = seed_whale_country_links(store, _TEST_EXCHANGE_WALLETS)
        assert count == 0

    def test_empty_exchange_dict(self, store: PipelineStore) -> None:
        _register_wallet(store, "1ExchangeUSaddr0001")
        count = seed_whale_country_links(store, {})
        assert count == 0

    def test_link_has_exchange_metadata(self, store: PipelineStore) -> None:
        _register_wallet(store, "1ExchangeUSaddr0001")
        seed_whale_country_links(store, _TEST_EXCHANGE_WALLETS)

        wid = entity_id_from_key("wallet", "1ExchangeUSaddr0001")
        links = store.query_entity_links(wid, direction="outgoing")
        assert links[0]["link_type"] == "exchange_based_in"
        assert links[0]["source"] == "exchange_wallet_match"


# ── detect_whale_geopolitical tests ──────────────────────────


class TestDetectWhaleGeopolitical:
    def test_basic_hit(self, store: PipelineStore) -> None:
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, t0, value_btc=200.0)
        _add_highimpact_gdelt(store, kid, t0 + 3600, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_whale_geopolitical(wid)
        assert len(patterns) == 1
        p = patterns[0]
        assert p["pattern_type"] == "whale_x_geopolitical"
        assert p["entity_a"] == wid
        assert p["entity_b"] == kid
        assert p["goldstein"] == -7.0
        assert p["value_btc"] == 200.0
        assert p["score"] > 0

    def test_no_links_returns_empty(self, store: PipelineStore) -> None:
        wid = _register_wallet(store, "1SomeWallet")
        detector = CrossEntityDetector(store)
        assert detector.detect_whale_geopolitical(wid) == []

    def test_no_observations_returns_empty(self, store: PipelineStore) -> None:
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        detector = CrossEntityDetector(store)
        assert detector.detect_whale_geopolitical(wid) == []

    def test_goldstein_above_threshold_filtered(self, store: PipelineStore) -> None:
        """Goldstein -3 should be filtered at the default -5 threshold."""
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, t0, value_btc=200.0)
        _add_highimpact_gdelt(store, kid, t0 + 3600, goldstein=-3.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_whale_geopolitical(wid)
        assert patterns == []

    def test_positive_goldstein_filtered(self, store: PipelineStore) -> None:
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, t0, value_btc=200.0)
        _add_highimpact_gdelt(store, kid, t0 + 3600, goldstein=5.0)

        detector = CrossEntityDetector(store)
        assert detector.detect_whale_geopolitical(wid) == []

    def test_small_transfer_lower_score(self, store: PipelineStore) -> None:
        """A 10 BTC transfer should score lower than a 200 BTC transfer."""
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, t0, value_btc=10.0)
        _add_highimpact_gdelt(store, kid, t0 + 3600, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_whale_geopolitical(wid)
        assert len(patterns) == 1
        small_score = patterns[0]["score"]
        assert small_score > 0
        # value_weight = min(10/100, 1) = 0.1 — should be significantly less than 1.0
        assert small_score < 0.1  # 0.1 × 0.7 × proximity < 0.1

    def test_large_transfer_capped_score(self, store: PipelineStore) -> None:
        """Transfers >= 100 BTC should have value_weight capped at 1.0."""
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, t0, value_btc=500.0)
        _add_highimpact_gdelt(store, kid, t0 + 100, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_whale_geopolitical(wid)
        assert len(patterns) == 1
        # value_weight = min(500/100, 1) = 1.0, severity = 0.7, proximity ≈ 1.0
        assert patterns[0]["score"] > 0.6

    def test_since_filter(self, store: PipelineStore) -> None:
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        old = 1_600_000_000.0
        recent = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, old, value_btc=200.0)
        _add_highimpact_gdelt(store, kid, old + 3600, goldstein=-7.0)
        _add_btc_transfer_obs(store, wid, recent, value_btc=200.0)
        _add_highimpact_gdelt(store, kid, recent + 3600, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_whale_geopolitical(wid, since=1_650_000_000.0)
        assert len(patterns) == 1

    def test_multiple_cooccurrences(self, store: PipelineStore) -> None:
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, t0, value_btc=200.0)
        _add_btc_transfer_obs(store, wid, t0 + 1000, value_btc=300.0)
        _add_highimpact_gdelt(store, kid, t0 + 500, goldstein=-6.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_whale_geopolitical(wid, window_seconds=3600)
        assert len(patterns) == 2

    def test_direction_in_pattern(self, store: PipelineStore) -> None:
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, t0, value_btc=200.0, direction="in")
        _add_highimpact_gdelt(store, kid, t0 + 3600, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_whale_geopolitical(wid)
        assert patterns[0]["direction"] == "in"

    def test_zero_value_btc(self, store: PipelineStore) -> None:
        """Transfer with 0 BTC should result in score 0."""
        wid = _register_wallet(store, "1ExchangeUSaddr0001")
        kid = _register_country(store, "US", "United States")
        store.link_entities(wid, kid, "exchange_based_in", "exchange_wallet_match")

        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, t0, value_btc=0.0)
        _add_highimpact_gdelt(store, kid, t0 + 3600, goldstein=-7.0)

        detector = CrossEntityDetector(store)
        patterns = detector.detect_whale_geopolitical(wid)
        assert len(patterns) == 1
        assert patterns[0]["score"] == 0.0


# ── Whale × Geopolitical integration ────────────────────────


class TestWhaleGeopoliticalIntegration:
    def test_full_pipeline(self, store: PipelineStore) -> None:
        """End-to-end: register → seed links → add obs → detect → store L3."""
        # 1. Register wallet entity (known exchange)
        wid = _register_wallet(store, "1ExchangeUSaddr0001")

        # 2. Seed exchange → country links
        link_count = seed_whale_country_links(store, _TEST_EXCHANGE_WALLETS)
        assert link_count == 1

        # 3. Add whale transfer + GDELT event
        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid, t0, value_btc=250.0, direction="out")
        us_eid = entity_id_from_key("country", "US")
        _add_highimpact_gdelt(store, us_eid, t0 + 7200, goldstein=-8.0)

        # 4. Detect patterns
        detector = CrossEntityDetector(store)
        patterns = detector.detect_whale_geopolitical(wid)
        assert len(patterns) == 1
        assert patterns[0]["value_btc"] == 250.0
        assert patterns[0]["goldstein"] == -8.0

        # 5. Store L3 observations
        stored = detector.store_l3_observations(patterns)
        assert stored == 1

        # 6. Verify
        l3_obs = store.query_entity_observations(wid, depth_level=3)
        assert len(l3_obs) == 1
        assert l3_obs[0]["value"]["pattern_type"] == "whale_x_geopolitical"
        assert l3_obs[0]["source_tool"] == "cross_entity"

    def test_multiple_exchanges_same_country(self, store: PipelineStore) -> None:
        """Two exchange wallets for the same country should not duplicate links."""
        wid1 = _register_wallet(store, "1ExchangeUSaddr0001")
        wid2 = _register_wallet(store, "1AnotherUSExchange")

        wallets = {
            "1ExchangeUSaddr0001": ("exchange_a", "US"),
            "1AnotherUSExchange": ("exchange_b", "US"),
        }
        count = seed_whale_country_links(store, wallets)
        assert count == 2  # two wallets each get a link

        us_eid = entity_id_from_key("country", "US")
        t0 = 1_700_000_000.0
        _add_btc_transfer_obs(store, wid1, t0, value_btc=100.0)
        _add_highimpact_gdelt(store, us_eid, t0 + 3600, goldstein=-6.0)

        detector = CrossEntityDetector(store)
        p1 = detector.detect_whale_geopolitical(wid1)
        p2 = detector.detect_whale_geopolitical(wid2)
        assert len(p1) == 1
        assert len(p2) == 0  # wid2 has no btc_transfer obs
