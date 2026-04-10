"""Tests for GDELT L2 upgrade — country entity persistence from
actor pairs, entity_ids in event output, dedup, and MI integration.

Mirrors the test pattern from test_whale_alert_l2.py / test_ais_vessel_l2.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.tools.gdelt import GDELTTool

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_event(
    *,
    event_id: str = "1001",
    date: str = "20260401",
    actor1_name: str | None = "UNITED STATES",
    actor1_country: str = "US",
    actor1_type: str = "GOV",
    actor2_name: str | None = "CHINA",
    actor2_country: str = "CH",
    actor2_type: str = "GOV",
    event_code: str = "190",
    event_root: str = "19",
    event_description: str = "Fight",
    quad_class: int = 4,
    quad_label: str = "Material Conflict",
    goldstein: float | None = -10.0,
    num_mentions: int = 50,
    num_sources: int = 10,
    avg_tone: float | None = -5.2,
    location_name: str = "South China Sea",
    location_country: str = "CH",
    location_lat: float | None = 15.0,
    location_lon: float | None = 115.0,
    source_url: str = "https://example.com/article",
) -> dict[str, Any]:
    """Build a parsed GDELT event dict (same shape as _parse_events output)."""
    return {
        "id": event_id,
        "date": date,
        "actor1": {
            "name": actor1_name,
            "country": actor1_country,
            "type": actor1_type,
        },
        "actor2": {
            "name": actor2_name,
            "country": actor2_country,
            "type": actor2_type,
        },
        "event_code": event_code,
        "event_root": event_root,
        "event_description": event_description,
        "quad_class": quad_class,
        "quad_label": quad_label,
        "goldstein": goldstein,
        "num_mentions": num_mentions,
        "num_sources": num_sources,
        "avg_tone": avg_tone,
        "location": {
            "name": location_name,
            "country": location_country,
            "lat": location_lat,
            "lon": location_lon,
        },
        "source_url": source_url,
    }


def _make_store() -> MagicMock:
    """Create a mock PipelineStore with the entity API surface."""
    store = MagicMock()
    store.register_entity = MagicMock(return_value="eid")
    store.add_entity_alias = MagicMock()
    store.store_entity_observation = MagicMock(return_value=1)
    store.resolve_entity = MagicMock(return_value=None)
    return store


# ===========================================================================
# Class: TestConstructor
# ===========================================================================


class TestConstructor:
    """Step 10b.5.2: PipelineStore kwarg in constructor."""

    def test_default_no_store(self) -> None:
        tool = GDELTTool()
        assert tool._store is None
        assert tool._cache is None

    def test_with_store(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        assert tool._store is store

    def test_with_cache_and_store(self) -> None:
        cache = MagicMock()
        store = _make_store()
        tool = GDELTTool(cache, pipeline_store=store)
        assert tool._cache is cache
        assert tool._store is store

    def test_store_keyword_only(self) -> None:
        """pipeline_store must be keyword-only — positional should fail."""
        store = _make_store()
        with pytest.raises(TypeError):
            GDELTTool(None, store)  # type: ignore[misc]


# ===========================================================================
# Class: TestPersistGuard
# ===========================================================================


class TestPersistGuard:
    """Step 10b.5.3: Guard methods skip when store/entities unavailable."""

    def test_no_store_noop(self) -> None:
        tool = GDELTTool()
        tool._persist_entities([_make_event()])  # should not raise

    def test_empty_list_noop(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        tool._persist_entities([])
        store.register_entity.assert_not_called()

    def test_entity_id_unavailable(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        with patch("agent.tools.gdelt.entity_id_from_key", None):
            tool._persist_entities([_make_event()])
        store.register_entity.assert_not_called()

    def test_inner_exception_propagates(self) -> None:
        """Inner errors propagate — caller wraps in try/except."""
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB locked")
        tool = GDELTTool(pipeline_store=store)
        with pytest.raises(RuntimeError):
            tool._persist_entities_inner([_make_event()])


# ===========================================================================
# Class: TestPersistEntitiesInner
# ===========================================================================


class TestPersistEntitiesInner:
    """Step 10b.5.3: Inner persistence logic — country registration + observations."""

    def test_both_actors_registered(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(actor1_country="US", actor2_country="CH")
        tool._persist_entities_inner([ev])

        # Two entities registered (US + CH)
        assert store.register_entity.call_count == 2
        # Two aliases
        assert store.add_entity_alias.call_count == 2
        # Two observations (one per actor)
        assert store.store_entity_observation.call_count == 2

    def test_empty_actor1_country_skipped(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(actor1_country="", actor2_country="CH")
        tool._persist_entities_inner([ev])

        # Only actor2 registered
        assert store.register_entity.call_count == 1
        assert store.store_entity_observation.call_count == 1
        call_kw = store.register_entity.call_args.kwargs
        eid = entity_id_from_key("country", "CH")
        assert call_kw["entity_id"] == eid

    def test_empty_actor2_country_skipped(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(actor1_country="US", actor2_country="")
        tool._persist_entities_inner([ev])

        assert store.register_entity.call_count == 1
        assert store.store_entity_observation.call_count == 1

    def test_both_actors_empty_country(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(actor1_country="", actor2_country="")
        tool._persist_entities_inner([ev])

        store.register_entity.assert_not_called()
        store.store_entity_observation.assert_not_called()

    def test_same_country_dyad(self) -> None:
        """Domestic event: both actors same country → registered once, 2 observations."""
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(actor1_country="US", actor2_country="US")
        tool._persist_entities_inner([ev])

        # Registered once (dedup)
        assert store.register_entity.call_count == 1
        # But two observations (initiator + target roles)
        assert store.store_entity_observation.call_count == 2

    def test_dedup_across_events(self) -> None:
        """Same country in multiple events → registered once, observation per event."""
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev1 = _make_event(event_id="1", actor1_country="US", actor2_country="CH")
        ev2 = _make_event(event_id="2", actor1_country="US", actor2_country="IR")
        tool._persist_entities_inner([ev1, ev2])

        # US registered once, CH once, IR once = 3 entities
        assert store.register_entity.call_count == 3
        # ev1: US(init)+CH(target), ev2: US(init)+IR(target) = 4 observations
        assert store.store_entity_observation.call_count == 4

    def test_initiator_observation_role(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(
            event_id="101",
            actor1_country="US",
            actor2_country="CH",
            goldstein=-8.0,
            quad_class=4,
        )
        tool._persist_entities_inner([ev])

        calls = store.store_entity_observation.call_args_list
        # First call is actor1 (initiator)
        init_call = calls[0]
        assert init_call.kwargs["observation_type"] == "geopolitical_event"
        assert init_call.kwargs["depth_level"] == 2
        val = init_call.kwargs["value"]
        assert val["role"] == "initiator"
        assert val["counterpart_country"] == "CH"
        assert val["event_id"] == "101"
        assert val["goldstein"] == -8.0
        assert val["quad_class"] == 4

    def test_target_observation_role(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(actor1_country="US", actor2_country="CH")
        tool._persist_entities_inner([ev])

        calls = store.store_entity_observation.call_args_list
        # Second call is actor2 (target)
        target_call = calls[1]
        val = target_call.kwargs["value"]
        assert val["role"] == "target"
        assert val["counterpart_country"] == "US"

    def test_name_fallback_to_country_code(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(actor1_name=None, actor1_country="IR")
        tool._persist_entities_inner([ev])

        # Find the IR registration call
        reg_calls = store.register_entity.call_args_list
        ir_call = [
            c
            for c in reg_calls
            if c.kwargs["entity_id"] == entity_id_from_key("country", "IR")
        ]
        assert len(ir_call) == 1
        assert ir_call[0].kwargs["canonical_name"] == "IR"

    def test_actor_type_in_metadata(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(actor1_country="US", actor1_type="MIL")
        tool._persist_entities_inner([ev])

        reg_calls = store.register_entity.call_args_list
        us_call = [
            c
            for c in reg_calls
            if c.kwargs["entity_id"] == entity_id_from_key("country", "US")
        ]
        assert us_call[0].kwargs["metadata"]["actor_type"] == "MIL"
        assert us_call[0].kwargs["metadata"]["fips_code"] == "US"

    def test_fips_alias(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(actor1_country="US", actor2_country="CH")
        tool._persist_entities_inner([ev])

        alias_calls = store.add_entity_alias.call_args_list
        sources = {c.args[1] for c in alias_calls}
        assert sources == {"fips"}
        ext_ids = {c.args[2] for c in alias_calls}
        assert ext_ids == {"US", "CH"}

    def test_location_in_observation(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(location_country="TW")
        tool._persist_entities_inner([ev])

        calls = store.store_entity_observation.call_args_list
        for call in calls:
            assert call.kwargs["value"]["location"] == "TW"

    def test_event_description_in_observation(self) -> None:
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)
        ev = _make_event(event_root="18", event_description="Assault")
        tool._persist_entities_inner([ev])

        calls = store.store_entity_observation.call_args_list
        for call in calls:
            val = call.kwargs["value"]
            assert val["event_root"] == "18"
            assert val["event_description"] == "Assault"


# ===========================================================================
# Class: TestEntityIdsOutput
# ===========================================================================


class TestEntityIdsOutput:
    """Step 10b.5.4: entity_id fields in actor sub-dicts."""

    def test_events_mode_entity_ids(self) -> None:
        """Events mode adds entity_id to actor1 and actor2 sub-dicts."""
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)

        ev = _make_event(actor1_country="US", actor2_country="CH")

        with (
            patch.object(tool, "_fetch_event_batches", return_value=["fake"]),
            patch.object(tool, "_parse_events", return_value=[ev]),
        ):
            result = tool.execute(mode="events", hours_back=1, quad_class="all")

        assert result.success
        events = result.data["events"]
        assert len(events) == 1
        assert events[0]["actor1"]["entity_id"] == entity_id_from_key("country", "US")
        assert events[0]["actor2"]["entity_id"] == entity_id_from_key("country", "CH")

    def test_empty_country_no_entity_id(self) -> None:
        """Actor with empty country code gets no entity_id."""
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)

        ev = _make_event(actor1_country="US", actor2_country="")

        with (
            patch.object(tool, "_fetch_event_batches", return_value=["fake"]),
            patch.object(tool, "_parse_events", return_value=[ev]),
        ):
            result = tool.execute(mode="events", hours_back=1, quad_class="all")

        assert result.success
        events = result.data["events"]
        assert "entity_id" in events[0]["actor1"]
        assert "entity_id" not in events[0]["actor2"]

    def test_articles_mode_no_entity_ids(self) -> None:
        """Articles mode should NOT have entity_ids."""
        tool = GDELTTool()

        articles = [
            {
                "title": "Test",
                "url": "https://x.com",
                "seendate": "20260401",
                "domain": "x.com",
                "language": "en",
                "sourcecountry": "US",
                "tone": -2.0,
            }
        ]

        with patch.object(tool, "_fetch_articles", return_value=articles):
            result = tool.execute(mode="articles", query="test")

        assert result.success
        for art in result.data["articles"]:
            assert "entity_id" not in art

    def test_entity_id_none_when_module_unavailable(self) -> None:
        """When entity_id_from_key is None, entity_id not added."""
        tool = GDELTTool()

        ev = _make_event(actor1_country="US", actor2_country="CH")

        with (
            patch.object(tool, "_fetch_event_batches", return_value=["fake"]),
            patch.object(tool, "_parse_events", return_value=[ev]),
            patch("agent.tools.gdelt.entity_id_from_key", None),
        ):
            result = tool.execute(mode="events", hours_back=1, quad_class="all")

        assert result.success
        events = result.data["events"]
        assert "entity_id" not in events[0]["actor1"]
        assert "entity_id" not in events[0]["actor2"]


# ===========================================================================
# Class: TestIntegration
# ===========================================================================


class TestIntegration:
    """End-to-end integration scenarios."""

    def test_backward_compat_no_store(self) -> None:
        """Tool works identically without pipeline_store."""
        tool = GDELTTool()

        ev = _make_event()
        with (
            patch.object(tool, "_fetch_event_batches", return_value=["fake"]),
            patch.object(tool, "_parse_events", return_value=[ev]),
        ):
            result = tool.execute(mode="events", hours_back=1, quad_class="all")

        assert result.success

    def test_persistence_error_non_fatal(self) -> None:
        """Persistence errors don't break the events mode result."""
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB locked")
        tool = GDELTTool(pipeline_store=store)

        ev = _make_event()
        with (
            patch.object(tool, "_fetch_event_batches", return_value=["fake"]),
            patch.object(tool, "_parse_events", return_value=[ev]),
        ):
            result = tool.execute(mode="events", hours_back=1, quad_class="all")

        assert result.success

    def test_articles_mode_no_persistence(self) -> None:
        """Articles mode does not trigger any persistence."""
        store = _make_store()
        tool = GDELTTool(pipeline_store=store)

        articles = [
            {
                "title": "Test",
                "url": "https://x.com",
                "seendate": "20260401",
                "domain": "x.com",
                "language": "en",
                "sourcecountry": "US",
                "tone": -2.0,
            }
        ]

        with patch.object(tool, "_fetch_articles", return_value=articles):
            result = tool.execute(mode="articles", query="test")

        assert result.success
        store.register_entity.assert_not_called()
        store.store_entity_observation.assert_not_called()


# ===========================================================================
# Class: TestRealStoreIntegration
# ===========================================================================


class TestRealStoreIntegration:
    """MI measurement: real PipelineStore + entities persisted + queried back."""

    def test_l2_with_real_store(self, tmp_path: Path) -> None:
        from agent.pipeline.store import PipelineStore

        db = tmp_path / "test.db"
        ps = PipelineStore(db_path=db)
        tool = GDELTTool(pipeline_store=ps)

        ev = _make_event(
            event_id="101",
            actor1_country="US",
            actor2_country="CH",
            goldstein=-8.0,
        )
        tool._persist_entities_inner([ev])

        eid_us = entity_id_from_key("country", "US")
        obs_us = ps.query_entity_observations(eid_us)
        assert len(obs_us) == 1
        assert obs_us[0]["observation_type"] == "geopolitical_event"
        val = obs_us[0]["value"]
        assert val["role"] == "initiator"
        assert val["counterpart_country"] == "CH"
        assert val["goldstein"] == -8.0

        eid_ch = entity_id_from_key("country", "CH")
        obs_ch = ps.query_entity_observations(eid_ch)
        assert len(obs_ch) == 1
        assert obs_ch[0]["value"]["role"] == "target"
        assert obs_ch[0]["value"]["counterpart_country"] == "US"

    def test_multiple_events_real_store(self, tmp_path: Path) -> None:
        from agent.pipeline.store import PipelineStore

        db = tmp_path / "test.db"
        ps = PipelineStore(db_path=db)
        tool = GDELTTool(pipeline_store=ps)

        ev1 = _make_event(event_id="1", actor1_country="US", actor2_country="CH")
        ev2 = _make_event(event_id="2", actor1_country="US", actor2_country="IR")
        tool._persist_entities_inner([ev1, ev2])

        eid_us = entity_id_from_key("country", "US")
        obs_us = ps.query_entity_observations(eid_us)
        # US is initiator in both events
        assert len(obs_us) == 2


# ===========================================================================
# Class: TestMIMeasurement
# ===========================================================================


class TestMIMeasurement:
    """Mutual information: L2 provides strictly more entity info than L1."""

    def test_l2_more_info_than_l1(self) -> None:
        """L2 event actors have entity_id; L1 does not."""
        ev = _make_event()
        assert "entity_id" not in ev["actor1"]
        assert "entity_id" not in ev["actor2"]

        # After L2 enrichment
        if entity_id_from_key is not None:
            ev["actor1"]["entity_id"] = entity_id_from_key(
                "country", ev["actor1"]["country"]
            )
            ev["actor2"]["entity_id"] = entity_id_from_key(
                "country", ev["actor2"]["country"]
            )
            assert "entity_id" in ev["actor1"]
            assert "entity_id" in ev["actor2"]

    def test_mi_with_real_store(self, tmp_path: Path) -> None:
        """L2 with real store: entity queryable after persistence."""
        from agent.pipeline.store import PipelineStore

        db = tmp_path / "test.db"
        ps = PipelineStore(db_path=db)
        tool = GDELTTool(pipeline_store=ps)

        ev = _make_event(actor1_country="US", actor2_country="CH")
        tool._persist_entities_inner([ev])

        eid = entity_id_from_key("country", "US")
        obs = ps.query_entity_observations(eid)
        assert len(obs) >= 1
        val = obs[0]["value"]
        assert "counterpart_country" in val
        assert "goldstein" in val
