"""Phase 33 edge-case tests for Organization + Grid Enrichment L2 persistence.

Covers: guard checks (no store, no entity_id_from_key), exception safety,
empty/missing data, agency resolution, graph builder obs types.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.models.gnn.graph_builder import ENRICHMENT_DIM, OBSERVATION_TYPES
from agent.tools.electricity_monitor import KNOWN_REGIONS, ElectricityMonitorTool
from agent.tools.regulatory_gazette import RegulatoryGazetteTool

# ── Helpers ──────────────────────────────────────────────────


def _store() -> MagicMock:
    s = MagicMock()
    s.register_entity = MagicMock()
    s.store_entity_observation = MagicMock(return_value=1)
    return s


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
    def test_no_store_returns_zero(self):
        tool = ElectricityMonitorTool()
        assert tool._persist_entities("PJM", "demand") == {"grid_demand_obs": 0}

    @patch("agent.tools.electricity_monitor._entity_id_from_key", None)
    def test_no_entity_id_returns_zero(self):
        tool = ElectricityMonitorTool(pipeline_store=_store())
        assert tool._persist_entities("PJM", "demand") == {"grid_demand_obs": 0}

    def test_exception_caught(self):
        store = _store()
        store.register_entity.side_effect = RuntimeError("boom")
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities("PJM", "demand")
        assert result == {"grid_demand_obs": 0}

    def test_empty_region_returns_zero(self):
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities_inner("", "demand")
        assert result == {"grid_demand_obs": 0}

    def test_known_region_persisted(self):
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities_inner("PJM", "demand")
        assert result == {"grid_demand_obs": 1}
        assert store.register_entity.call_args.args[:2] == ("organization", "PJM")
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["observation_type"] == "grid_demand"
        assert obs["depth_level"] == 2
        assert obs["value"]["region"] == "PJM"
        assert obs["value"]["region_name"] == "PJM Interconnection"
        assert obs["value"]["mode"] == "demand"

    def test_unknown_region_still_persisted(self):
        """BAs not in KNOWN_REGIONS should still persist — EIA has more than we list."""
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities_inner("UNKNOWN_BA", "generation")
        assert result == {"grid_demand_obs": 1}
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["value"]["region"] == "UNKNOWN_BA"
        # region_name falls back to code when not in KNOWN_REGIONS
        assert obs["value"]["region_name"] == "UNKNOWN_BA"

    def test_generation_mode(self):
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities_inner("CISO", "generation")
        assert result == {"grid_demand_obs": 1}
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["value"]["mode"] == "generation"
        assert obs["value"]["region_name"] == "California ISO"

    def test_interchange_mode(self):
        store = _store()
        tool = ElectricityMonitorTool(pipeline_store=store)
        result = tool._persist_entities_inner("ERCO", "interchange")
        assert result == {"grid_demand_obs": 1}
        obs = store.store_entity_observation.call_args.kwargs
        assert obs["value"]["mode"] == "interchange"
        assert obs["value"]["region_name"] == "Electric Reliability Council of Texas"

    def test_all_known_regions_have_names(self):
        """KNOWN_REGIONS maps BA codes to human-readable names."""
        for code, name in KNOWN_REGIONS.items():
            assert isinstance(name, str)
            assert len(name) > 0
            assert len(code) >= 2
