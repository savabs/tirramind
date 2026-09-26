"""Phase 33 edge-case tests for Organization + Grid Enrichment L2 persistence.

Covers: guard checks (no store, no entity_id_from_key), exception safety,
empty/missing data, agency resolution, graph builder obs types.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from agent.models.gnn.graph_builder import ENRICHMENT_DIM, OBSERVATION_TYPES
from agent.tools.electricity_monitor import (
    KNOWN_REGIONS,
    ElectricityMonitorTool,
    PersistStats,
)
from agent.tools.regulatory_gazette import RegulatoryGazetteTool

# ── Helpers ──────────────────────────────────────────────────


def _store() -> MagicMock:
    s = MagicMock()
    s.register_entity = MagicMock()
    s.store_entity_observation = MagicMock(return_value=1)
    return s


# ── Electricity fixtures ─────────────────────────────────────

# Periods safely in the past: _observation_from_record filters anything dated
# beyond now + 12h as a day-ahead forecast.
_PERIODS = ("2026-09-23T12", "2026-09-23T13", "2026-09-23T14")


def _epoch(period: str) -> float:
    """Expected observed_at, computed independently of the collector's parser."""
    return datetime.strptime(period, "%Y-%m-%dT%H").replace(tzinfo=UTC).timestamp()


def _demand_records() -> list[dict]:
    """One EIA actual-demand row per hour, the shape persistence now consumes."""
    return [{"period": p, "value": 1000.0 + i, "type": "D", "type-name": "Demand"} for i, p in enumerate(_PERIODS)]


# =====================================================================
# Graph builder constants
# =====================================================================


class TestGraphBuilderPhase33:
    def test_obs_types_sorted(self):
        assert sorted(OBSERVATION_TYPES) == OBSERVATION_TYPES

    def test_new_obs_types_present(self):
        for ot in ("grid_demand", "regulatory_velocity"):
            assert ot in OBSERVATION_TYPES, f"{ot} missing from OBSERVATION_TYPES"

    def test_obs_count(self):
        # 52 since 2026-08-26 (was asserted 46 while the list held 48 — this
        # assertion had drifted and was failing). Registry growth shifts
        # one-hot positions and invalidates checkpoints — retrain on change.
        assert len(OBSERVATION_TYPES) == 52

    def test_enrichment_dim(self):
        assert 9 + len(OBSERVATION_TYPES) == ENRICHMENT_DIM
        # Derived, not hardcoded: 9 scalars + one slot per OBSERVATION_TYPES entry.
        # Was pinned at 55 (correct only at 46 obs types); once the registry
        # grew, obs_type_dist wrote past the block and crashed entity_scoring.
        assert 9 + len(OBSERVATION_TYPES) == ENRICHMENT_DIM


# =====================================================================
# Regulatory Gazette L2
# =====================================================================


class TestRegulatoryGazetteL2:
    def test_no_store_returns_zero(self):
        tool = RegulatoryGazetteTool()
        assert tool._persist_entities({"documents": []}, "recent") == {"regulatory_velocity_obs": 0}

    @patch("agent.tools.regulatory_gazette._entity_id_from_key", None)
    def test_no_entity_id_returns_zero(self):
        tool = RegulatoryGazetteTool(pipeline_store=_store())
        assert tool._persist_entities({"documents": []}, "recent") == {"regulatory_velocity_obs": 0}

    def test_exception_caught(self):
        store = _store()
        store.register_entity.side_effect = RuntimeError("boom")
        tool = RegulatoryGazetteTool(pipeline_store=store)
        result = tool._persist_entities(
            {"documents": [{"agencies": ["Securities and Exchange Commission"], "type": "RULE"}]},
            "recent",
        )
        assert result == {"regulatory_velocity_obs": 0}

    def test_empty_documents(self):
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        result = tool._persist_entities_inner({"documents": []}, "recent")
        assert result == {"regulatory_velocity_obs": 0}
        assert store.register_entity.call_count == 0

    def test_no_documents_key(self):
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        result = tool._persist_entities_inner({}, "recent")
        assert result == {"regulatory_velocity_obs": 0}

    def test_single_known_agency(self):
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        data = {
            "documents": [
                {
                    "agencies": ["Securities and Exchange Commission"],
                    "type": "RULE",
                    "significant": True,
                }
            ]
        }
        result = tool._persist_entities_inner(data, "recent")
        assert result == {"regulatory_velocity_obs": 1}
        assert store.register_entity.call_args.args[:2] == ("organization", "sec")
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["observation_type"] == "regulatory_velocity"
        assert obs["depth_level"] == 2
        assert obs["value"]["doc_count"] == 1
        assert obs["value"]["significant_count"] == 1
        assert "RULE" in obs["value"]["types"]

    def test_multiple_docs_same_agency_aggregated(self):
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        data = {
            "documents": [
                {
                    "agencies": ["Food and Drug Administration"],
                    "type": "RULE",
                    "significant": False,
                },
                {
                    "agencies": ["Food and Drug Administration"],
                    "type": "PRORULE",
                    "significant": True,
                },
                {
                    "agencies": ["Food and Drug Administration"],
                    "type": "RULE",
                    "significant": False,
                },
            ]
        }
        result = tool._persist_entities_inner(data, "search")
        assert result == {"regulatory_velocity_obs": 1}
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["value"]["doc_count"] == 3
        assert obs["value"]["significant_count"] == 1
        assert sorted(obs["value"]["types"]) == ["PRORULE", "RULE"]

    def test_multiple_agencies_separate_obs(self):
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        data = {
            "documents": [
                {"agencies": ["Securities and Exchange Commission"], "type": "RULE"},
                {"agencies": ["Environmental Protection Agency"], "type": "PRORULE"},
            ]
        }
        result = tool._persist_entities_inner(data, "recent")
        assert result == {"regulatory_velocity_obs": 2}

    def test_unknown_agency_best_effort_key(self):
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        data = {
            "documents": [
                {
                    "agencies": ["Bureau of Unusual Findings"],
                    "type": "NOTICE",
                    "significant": False,
                }
            ]
        }
        result = tool._persist_entities_inner(data, "recent")
        assert result == {"regulatory_velocity_obs": 1}
        key = store.register_entity.call_args.args[1]
        assert key == "bureau_of_unusual_findings"

    def test_empty_agency_name_skipped(self):
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        data = {
            "documents": [
                {"agencies": ["", "  "], "type": "RULE"},
            ]
        }
        result = tool._persist_entities_inner(data, "recent")
        assert result == {"regulatory_velocity_obs": 0}

    def test_doc_with_no_agencies_list(self):
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        data = {
            "documents": [
                {"type": "RULE", "significant": True},
            ]
        }
        result = tool._persist_entities_inner(data, "recent")
        assert result == {"regulatory_velocity_obs": 0}

    def test_doc_with_multiple_agencies(self):
        """Single doc from multiple agencies creates separate obs."""
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        data = {
            "documents": [
                {
                    "agencies": [
                        "Securities and Exchange Commission",
                        "Commodity Futures Trading Commission",
                    ],
                    "type": "RULE",
                    "significant": True,
                }
            ]
        }
        result = tool._persist_entities_inner(data, "recent")
        assert result == {"regulatory_velocity_obs": 2}
        keys = sorted(c.args[1] for c in store.register_entity.call_args_list)
        assert "cftc" in keys or "sec" in keys

    def test_mode_passed_through(self):
        store = _store()
        tool = RegulatoryGazetteTool(pipeline_store=store)
        data = {
            "documents": [
                {"agencies": ["Federal Reserve System"], "type": "RULE"},
            ]
        }
        tool._persist_entities_inner(data, "upcoming")
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["value"]["mode"] == "upcoming"


# =====================================================================
# Electricity Monitor L2
# =====================================================================


class TestElectricityMonitorL2:
    # The collector used to write ONE metadata row per (region, mode), stamped
    # with wall-clock time, and these tests asserted that shape. It now writes
    # one observation per hourly reading with observed_at taken from the
    # reading's own EIA period: wall-clock stamping of historical readings is
    # what made entity_links useless for time-windowed graphs. Call sites and
    # assertions below track the per-reading contract.

    def test_no_store_returns_zero(self):
        tool = ElectricityMonitorTool()
        # Was: == {"grid_demand_obs": 0}. The pass now reports a PersistStats,
        # which keeps written/skipped/malformed/rejected apart — one integer
        # could not tell "nothing to store" from "every row unparseable".
        assert tool._persist_entities("PJM", "demand", _demand_records()) == PersistStats()

    @patch("agent.tools.electricity_monitor._entity_id_from_key", None)
    def test_no_entity_id_returns_zero(self):
        tool = ElectricityMonitorTool(pipeline_store=_store())
        assert tool._persist_entities("PJM", "demand", _demand_records()) == PersistStats()

    def test_exception_caught(self):
        store = _store()
        store.register_entity.side_effect = RuntimeError("boom")
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities("PJM", "demand", _demand_records())
        assert result == PersistStats()
        # Nothing may be written once entity registration failed — an
        # observation whose entity row does not exist is an orphan.
        store.store_entity_observation.assert_not_called()

    def test_empty_region_returns_zero(self):
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities_inner("", "demand", _demand_records())
        assert result == PersistStats()
        store.register_entity.assert_not_called()

    def test_no_records_returns_zero(self):
        """An empty window registers nothing — the guard is now on records too."""
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        assert tool._persist_entities_inner("PJM", "demand", []) == PersistStats()
        store.register_entity.assert_not_called()

    def test_known_region_persisted(self):
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities_inner("PJM", "demand", _demand_records())
        # Was: result == {"grid_demand_obs": 1} — one metadata row for the
        # whole window regardless of how many hours came back.
        assert result == PersistStats(written=len(_PERIODS))
        assert store.register_entity.call_args.args[:2] == ("organization", "PJM")

        calls = store.store_entity_observation.call_args_list
        assert len(calls) == len(_PERIODS)
        # The reason the signature changed: each row is stamped with its own
        # EIA period, not with time.time() at collection.
        assert [c.kwargs["observed_at"] for c in calls] == [_epoch(p) for p in _PERIODS]
        assert {c.kwargs["observation_type"] for c in calls} == {"grid_demand"}
        assert {c.kwargs["depth_level"] for c in calls} == {2}

        first = calls[0].kwargs["value"]
        assert first["region"] == "PJM"
        assert first["mw"] == 1000.0
        assert first["type"] == "Demand"
        # "region_name" and "mode" were fields of the old metadata row; the
        # per-reading value carries the measurement instead. KNOWN_REGIONS
        # naming is still covered by test_all_known_regions_have_names.
        assert "region_name" not in first
        assert "mode" not in first

    def test_unknown_region_still_persisted(self):
        """BAs not in KNOWN_REGIONS should still persist — EIA has more than we list."""
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        records = [{"period": _PERIODS[0], "value": 500.0, "fueltype": "WND"}]
        result = tool._persist_entities_inner("UNKNOWN_BA", "generation", records)
        assert result == PersistStats(written=1)
        obs = store.store_entity_observation.call_args.kwargs
        # Was: "grid_demand" for every mode. Generation and interchange now
        # carry their own observation types instead of all three landing in
        # one series.
        assert obs["observation_type"] == "grid_generation"
        assert obs["observed_at"] == _epoch(_PERIODS[0])
        assert obs["value"]["region"] == "UNKNOWN_BA"
        assert obs["value"]["mwh"] == 500.0

    def test_generation_mode(self):
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        records = [
            {"period": _PERIODS[0], "value": 900.0, "fueltype": "COL"},
            {"period": _PERIODS[0], "value": 120.0, "fueltype": "SUN"},
        ]
        result = tool._persist_entities_inner("CISO", "generation", records)
        # One observation per fuel per hour, not one row for the window.
        assert result == PersistStats(written=2)
        values = [c.kwargs["value"] for c in store.store_entity_observation.call_args_list]
        assert [v["fuel_code"] for v in values] == ["COL", "SUN"]
        # Was: obs["value"]["region_name"] == "California ISO" on a metadata
        # row. The fuel code is resolved through EIA_FUEL_TYPES instead.
        assert [v["fuel"] for v in values] == ["Coal", "Solar"]
        assert all(v["region"] == "CISO" for v in values)

    def test_interchange_mode(self):
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        records = [{"period": _PERIODS[1], "value": -300.0, "fromba": "ERCO", "toba": "SWPP"}]
        result = tool._persist_entities_inner("ERCO", "interchange", records)
        assert result == PersistStats(written=1)
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["observation_type"] == "grid_interchange"
        assert obs["observed_at"] == _epoch(_PERIODS[1])
        # Was: value["mode"] == "interchange" on the metadata row. The flow's
        # two ends are the fact worth storing.
        assert obs["value"]["from"] == "ERCO"
        assert obs["value"]["to"] == "SWPP"
        assert obs["value"]["mwh"] == -300.0

    def test_forecast_and_future_rows_filtered_not_malformed(self):
        """DF forecasts and future-dated periods are deliberate skips."""
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        records = [
            {"period": _PERIODS[0], "value": 1000.0, "type": "D"},
            {"period": _PERIODS[1], "value": 1010.0, "type": "DF"},
            {"period": "2099-01-01T00", "value": 1020.0, "type": "D"},
        ]
        result = tool._persist_entities_inner("PJM", "demand", records)
        assert result == PersistStats(written=1, skipped=2)
        assert store.store_entity_observation.call_count == 1

    def test_missing_value_counted_malformed(self):
        """A row with no `value` is the data[0]=value defect — never a routine skip."""
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        records = [
            {"period": _PERIODS[0], "value": 1000.0, "type": "D"},
            {"period": _PERIODS[1], "type": "D"},
            {"period": "not-a-period", "value": 1020.0, "type": "D"},
        ]
        result = tool._persist_entities_inner("PJM", "demand", records)
        # Counted separately from `skipped`: these are rows we meant to store
        # and could not parse, and execute() fails the run on them. Storing
        # them as 0 MW is what once reported "Peak: 0 MW" for every hour.
        assert result == PersistStats(written=1, malformed=2)

    def test_store_rejection_counted_not_swallowed(self):
        """One refused row must not cost the rest of the window, but must be counted."""
        store = _store()
        store.store_entity_observation.side_effect = [1, RuntimeError("refused"), 1]
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities_inner("PJM", "demand", _demand_records())
        assert result == PersistStats(written=2, rejected=1)

    def test_all_known_regions_have_names(self):
        """KNOWN_REGIONS maps BA codes to human-readable names."""
        for code, name in KNOWN_REGIONS.items():
            assert isinstance(name, str)
            assert len(name) > 0
            assert len(code) >= 2
