"""Tests for CFTC tool L2 entity persistence — Phase 25b.

Tests cover:
- Entity registration for CFTC contracts
- Observation storage for futures_positioning
- Instrument<->contract linking via cftc_code_to_ticker mapping
- Edge cases: unmapped codes, missing fields, empty rows, duplicates
- Integration with graph builder types
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.cftc import CFTCTool, _report_date_to_ts


# ── Helpers ───────────────────────────────────────────────────


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    store.link_entities = MagicMock(return_value=1)
    return store


def _make_row(
    cftc_code: str = "088691",
    market_name: str = "GOLD - COMMODITY EXCHANGE INC.",
    report_date: str = "2025-01-14",
    oi: int = 500000,
    mm_long: int = 200000,
    mm_short: int = 150000,
    pm_long: int = 100000,
    pm_short: int = 120000,
    swap_long: int = 80000,
    swap_short: int = 90000,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a minimal enriched CFTC row (post-_compute_signals)."""
    row = {
        "Market_and_Exchange_Names": market_name,
        "Report_Date_as_YYYY-MM-DD": report_date,
        "CFTC_Contract_Market_Code": cftc_code,
        "Open_Interest_All": oi,
        "M_Money_Positions_Long_All": mm_long,
        "M_Money_Positions_Short_All": mm_short,
        "Prod_Merc_Positions_Long_All": pm_long,
        "Prod_Merc_Positions_Short_All": pm_short,
        "Swap_Positions_Long_All": swap_long,
        "Swap__Positions_Short_All": swap_short,
        # Computed signals (normally added by _compute_signals)
        "_mm_net": mm_long - mm_short,
        "_pm_net": pm_long - pm_short,
        "_swap_net": swap_long - swap_short,
        "_mm_net_pct_oi": round((mm_long - mm_short) / oi * 100, 2) if oi else 0.0,
        "_mm_weekly_flow": 5000,
        "_oi_change": 10000,
        "_conc_top4_long": 25.0,
        "_conc_top4_short": 30.0,
    }
    row.update(overrides)
    return row


def _make_tool(store: MagicMock | None = None) -> CFTCTool:
    return CFTCTool(cache=None, pipeline_store=store)


# ── Report date conversion ───────────────────────────────────


class TestReportDateToTs:
    def test_valid_date(self):
        ts = _report_date_to_ts("2025-01-14")
        dt = datetime.utcfromtimestamp(ts)
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 14

    def test_empty_string(self):
        assert _report_date_to_ts("") == 0.0

    def test_none_like(self):
        assert _report_date_to_ts("") == 0.0

    def test_malformed(self):
        assert _report_date_to_ts("not-a-date") == 0.0

    def test_whitespace(self):
        assert _report_date_to_ts("  ") == 0.0

    def test_wrong_format(self):
        # DD/MM/YYYY instead of YYYY-MM-DD
        assert _report_date_to_ts("14/01/2025") == 0.0


# ── Persist entities: basic behavior ─────────────────────────


class TestCFTCPersistEntitiesBasic:
    def test_no_store_returns_zeros(self):
        tool = _make_tool(store=None)
        result = tool._persist_entities([_make_row()])
        assert result == {"observations": 0, "contracts": 0, "links": 0}

    def test_empty_rows_returns_zeros(self):
        store = _make_store()
        tool = _make_tool(store=store)
        result = tool._persist_entities([])
        assert result == {"observations": 0, "contracts": 0, "links": 0}
        store.register_entity.assert_not_called()

    def test_returns_count_dict(self):
        store = _make_store()
        tool = _make_tool(store=store)
        result = tool._persist_entities([_make_row()])
        assert isinstance(result, dict)
        assert "observations" in result
        assert "contracts" in result
        assert "links" in result

    def test_single_row_registers_contract(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([_make_row(cftc_code="088691")])
        contract_calls = [
            c
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "cftc_contract"
        ]
        assert len(contract_calls) == 1
        kw = contract_calls[0].kwargs
        assert kw["metadata"]["cftc_code"] == "088691"

    def test_single_row_stores_observation(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([_make_row()])
        assert store.store_entity_observation.call_count == 1
        kw = store.store_entity_observation.call_args.kwargs
        assert kw["observation_type"] == "futures_positioning"
        assert kw["depth_level"] == 2
        assert kw["source_tool"] == "cftc"

    def test_observation_value_contains_signals(self):
        store = _make_store()
        tool = _make_tool(store=store)
        row = _make_row(oi=100000, mm_long=60000, mm_short=40000)
        tool._persist_entities([row])
        val = store.store_entity_observation.call_args.kwargs["value"]
        assert val["open_interest"] == 100000
        assert val["mm_net"] == 20000
        assert val["pm_net"] is not None
        assert val["swap_net"] is not None

    def test_mapped_code_creates_instrument_link(self):
        store = _make_store()
        tool = _make_tool(store=store)
        # 088691 = Gold = GC=F
        tool._persist_entities([_make_row(cftc_code="088691")])
        link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "cftc_tracks"
        ]
        assert len(link_calls) == 1
        kw = link_calls[0].kwargs
        assert kw["metadata"]["ticker"] == "GC=F"
        assert kw["confidence"] == 1.0

    def test_unmapped_code_no_instrument_link(self):
        store = _make_store()
        tool = _make_tool(store=store)
        # 999999 is not in the instrument universe
        tool._persist_entities([_make_row(cftc_code="999999")])
        link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "cftc_tracks"
        ]
        assert len(link_calls) == 0
        # But contract and observation should still be created
        assert store.register_entity.call_count == 1
        assert store.store_entity_observation.call_count == 1


# ── Persist entities: dedup and multi-row ─────────────────────


class TestCFTCPersistEntitiesMulti:
    def test_two_rows_same_code_one_contract(self):
        store = _make_store()
        tool = _make_tool(store=store)
        rows = [
            _make_row(cftc_code="088691", report_date="2025-01-14"),
            _make_row(cftc_code="088691", report_date="2025-01-07"),
        ]
        result = tool._persist_entities(rows)
        assert result["contracts"] == 1  # deduped
        assert result["observations"] == 2  # one per row

    def test_two_different_codes(self):
        store = _make_store()
        tool = _make_tool(store=store)
        rows = [
            _make_row(cftc_code="088691"),  # Gold
            _make_row(cftc_code="023651"),  # Natural Gas
        ]
        result = tool._persist_entities(rows)
        assert result["contracts"] == 2
        assert result["observations"] == 2
        assert result["links"] == 2  # both mapped

    def test_mixed_mapped_and_unmapped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        rows = [
            _make_row(cftc_code="088691"),  # Gold → mapped
            _make_row(cftc_code="999999"),  # Unknown → no link
        ]
        result = tool._persist_entities(rows)
        assert result["links"] == 1  # only gold


# ── Edge cases ────────────────────────────────────────────────


class TestCFTCPersistEdgeCases:
    def test_empty_cftc_code_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        row = _make_row(cftc_code="")
        result = tool._persist_entities([row])
        assert result["observations"] == 0
        store.register_entity.assert_not_called()

    def test_whitespace_cftc_code_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        row = _make_row(cftc_code="   ")
        result = tool._persist_entities([row])
        assert result["observations"] == 0

    def test_missing_cftc_code_key_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        row = _make_row()
        del row["CFTC_Contract_Market_Code"]
        result = tool._persist_entities([row])
        assert result["observations"] == 0

    def test_missing_report_date_uses_zero_ts(self):
        store = _make_store()
        tool = _make_tool(store=store)
        row = _make_row(report_date="")
        tool._persist_entities([row])
        kw = store.store_entity_observation.call_args.kwargs
        assert kw["observed_at"] == 0.0

    def test_link_entities_returns_zero(self):
        """Duplicate link (returns 0) should not increment count."""
        store = _make_store()
        store.link_entities = MagicMock(return_value=0)
        tool = _make_tool(store=store)
        result = tool._persist_entities([_make_row(cftc_code="088691")])
        assert result["links"] == 0

    def test_persist_exception_is_nonfatal(self):
        """_persist_entities should catch exceptions and return zeros."""
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB error")
        tool = _make_tool(store=store)
        # Should not raise
        result = tool._persist_entities([_make_row()])
        assert result == {"observations": 0, "contracts": 0, "links": 0}

    def test_none_signal_values_stored(self):
        """Rows with None signal values should still be stored."""
        store = _make_store()
        tool = _make_tool(store=store)
        row = _make_row()
        row["_conc_top4_long"] = None
        row["_conc_top4_short"] = None
        tool._persist_entities([row])
        val = store.store_entity_observation.call_args.kwargs["value"]
        assert val["conc_top4_long"] is None
        assert val["conc_top4_short"] is None


# ── Constructor ──────────────────────────────────────────────


class TestCFTCConstructor:
    def test_accepts_pipeline_store(self):
        store = _make_store()
        tool = CFTCTool(cache=None, pipeline_store=store)
        assert tool._store is store

    def test_pipeline_store_defaults_none(self):
        tool = CFTCTool(cache=None)
        assert tool._store is None

    def test_backward_compatible_positional(self):
        """Old callers: CFTCTool(cache)."""
        tool = CFTCTool(None)
        assert tool._store is None


# ── Graph builder integration ────────────────────────────────


class TestCFTCGraphIntegration:
    def test_cftc_contract_in_entity_types(self):
        from agent.models.gnn.graph_builder import ENTITY_TYPES

        assert "cftc_contract" in ENTITY_TYPES

    def test_futures_positioning_in_observation_types(self):
        from agent.models.gnn.graph_builder import OBSERVATION_TYPES

        assert "futures_positioning" in OBSERVATION_TYPES

    def test_cftc_contract_in_seed_types(self):
        from agent.pipeline.entity import SEED_ENTITY_TYPES

        assert "cftc_contract" in SEED_ENTITY_TYPES


# ── Execute integration (mocked fetch) ───────────────────────


class TestCFTCExecuteWithPersistence:
    """Verify execute() calls _persist_entities when store is present."""

    @patch.object(CFTCTool, "_fetch_latest")
    def test_execute_calls_persist(self, mock_fetch):
        # Build a minimal valid CSV row (191 fields)
        fields = [""] * 191
        fields[0] = "GOLD - COMMODITY EXCHANGE INC."  # Market_and_Exchange_Names
        fields[1] = "250114"  # As_of_Date
        fields[2] = "2025-01-14"  # Report_Date
        fields[3] = "088691"  # CFTC_Contract_Market_Code
        fields[7] = "500000"  # Open_Interest_All
        fields[8] = "100000"  # Prod_Merc_Long
        fields[9] = "120000"  # Prod_Merc_Short
        fields[10] = "80000"  # Swap_Long
        fields[11] = "90000"  # Swap_Short
        fields[13] = "200000"  # MM_Long
        fields[14] = "150000"  # MM_Short
        mock_fetch.return_value = ",".join(fields)

        store = _make_store()
        tool = _make_tool(store=store)
        result = tool.execute(mode="latest")

        assert result.success
        # Should have called store methods
        assert store.register_entity.call_count >= 1
        assert store.store_entity_observation.call_count >= 1

    @patch.object(CFTCTool, "_fetch_latest")
    def test_execute_persist_failure_nonfatal(self, mock_fetch):
        """Even if persistence fails, execute should still return data."""
        fields = [""] * 191
        fields[0] = "GOLD"
        fields[1] = "250114"
        fields[2] = "2025-01-14"
        fields[3] = "088691"
        fields[7] = "500000"
        fields[8] = "100000"
        fields[9] = "120000"
        fields[10] = "80000"
        fields[11] = "90000"
        fields[13] = "200000"
        fields[14] = "150000"
        mock_fetch.return_value = ",".join(fields)

        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB down")
        tool = _make_tool(store=store)

        result = tool.execute(mode="latest")
        assert result.success  # persist failure is non-fatal

    @patch.object(CFTCTool, "_fetch_latest")
    def test_execute_without_store(self, mock_fetch):
        """Execute works fine without a pipeline store (L1 mode)."""
        fields = [""] * 191
        fields[0] = "GOLD"
        fields[1] = "250114"
        fields[2] = "2025-01-14"
        fields[3] = "088691"
        fields[7] = "500000"
        fields[13] = "200000"
        fields[14] = "150000"
        mock_fetch.return_value = ",".join(fields)

        tool = CFTCTool(cache=None)  # No store
        result = tool.execute(mode="latest")
        assert result.success
