"""L2 persistence tests for finra_short_volume tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.tools.finra_short_volume import FinraShortVolumeTool


# ── Helpers ───────────────────────────────────────────────────


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock()
    store.store_entity_observation = MagicMock()
    return store


def _make_tool(store: MagicMock | None = None) -> FinraShortVolumeTool:
    return FinraShortVolumeTool(cache=None, pipeline_store=store)


# ── Store absent → no-op ─────────────────────────────────────


class TestNoPersistenceWithoutStore:
    def test_no_store_no_crash(self):
        tool = _make_tool(store=None)
        tool._persist_entities(
            [{"ticker": "AAPL", "short_ratio": 0.3, "date": "2025-01-01"}]
        )
        # No exception, no calls

    def test_empty_records_no_crash(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([])
        store.register_entity.assert_not_called()
        store.store_entity_observation.assert_not_called()


# ── Basic entity registration ────────────────────────────────


class TestEntityRegistration:
    def test_single_ticker_registered(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [{"ticker": "AAPL", "short_ratio": 0.42, "date": "2025-06-01"}]
        )

        store.register_entity.assert_called_once()
        call = store.register_entity.call_args
        assert call.kwargs["entity_type"] == "company"
        assert call.kwargs["canonical_name"] == "AAPL"
        assert isinstance(call.kwargs["entity_id"], str)
        assert len(call.kwargs["entity_id"]) > 0

    def test_multiple_tickers_each_registered(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [
                {"ticker": "AAPL", "short_ratio": 0.3, "date": "2025-06-01"},
                {"ticker": "TSLA", "short_ratio": 0.5, "date": "2025-06-01"},
                {"ticker": "GME", "short_ratio": 0.7, "date": "2025-06-01"},
            ]
        )

        assert store.register_entity.call_count == 3
        assert store.store_entity_observation.call_count == 3

    def test_duplicate_ticker_registered_once(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [
                {"ticker": "AAPL", "short_ratio": 0.3, "date": "2025-06-01"},
                {"ticker": "AAPL", "short_ratio": 0.4, "date": "2025-06-02"},
            ]
        )

        # Entity registered once, but two observations
        store.register_entity.assert_called_once()
        assert store.store_entity_observation.call_count == 2

    def test_ticker_uppercased(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [{"ticker": "aapl", "short_ratio": 0.3, "date": "2025-06-01"}]
        )

        call = store.register_entity.call_args
        assert call.kwargs["canonical_name"] == "AAPL"


# ── Observation storage ──────────────────────────────────────


class TestObservationStorage:
    def test_observation_type_is_short_interest(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [{"ticker": "AAPL", "short_ratio": 0.42, "date": "2025-06-01"}]
        )

        call = store.store_entity_observation.call_args
        assert call.kwargs["observation_type"] == "short_interest"
        assert call.kwargs["depth_level"] == 2

    def test_short_volume_fields_in_value(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [
                {
                    "ticker": "AAPL",
                    "short_ratio": 0.42,
                    "total_volume": 1_000_000,
                    "short_volume": 420_000,
                    "zscore": 2.1,
                    "trend": "rising",
                    "is_anomaly": True,
                    "date": "2025-06-01",
                }
            ]
        )

        value = store.store_entity_observation.call_args.kwargs["value"]
        assert value["short_ratio"] == 0.42
        assert value["total_volume"] == 1_000_000
        assert value["short_volume"] == 420_000
        assert value["zscore"] == 2.1
        assert value["trend"] == "rising"
        assert value["is_anomaly"] is True

    def test_short_interest_fields_in_value(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [
                {
                    "symbol": "GME",
                    "settlement_date": "2025-05-31",
                    "current_short_position": 80_000_000,
                    "previous_short_position": 60_000_000,
                    "change_percent": 33.3,
                    "days_to_cover": 8.5,
                    "avg_daily_volume": 9_400_000,
                }
            ]
        )

        value = store.store_entity_observation.call_args.kwargs["value"]
        assert value["current_short_position"] == 80_000_000
        assert value["days_to_cover"] == 8.5
        assert value["change_percent"] == 33.3

    def test_timestamp_parsed_from_date(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [{"ticker": "AAPL", "short_ratio": 0.3, "date": "2025-06-15"}]
        )

        ts = store.store_entity_observation.call_args.kwargs["observed_at"]
        from datetime import datetime, timezone

        expected = datetime(2025, 6, 15, tzinfo=timezone.utc).timestamp()
        assert ts == expected

    def test_timestamp_fallback_for_bad_date(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [{"ticker": "AAPL", "short_ratio": 0.3, "date": "bad-date"}]
        )

        ts = store.store_entity_observation.call_args.kwargs["observed_at"]
        assert isinstance(ts, float)
        assert ts > 0


# ── Graceful edge cases ──────────────────────────────────────


class TestEdgeCases:
    def test_missing_ticker_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [{"short_ratio": 0.3, "date": "2025-06-01"}]
        )  # no ticker

        store.register_entity.assert_not_called()

    def test_empty_ticker_skipped(self):
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([{"ticker": "  ", "short_ratio": 0.3}])

        store.register_entity.assert_not_called()

    def test_record_with_no_value_fields_skipped(self):
        """Record with only ticker/date but no short data → no observation."""
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities([{"ticker": "AAPL", "date": "2025-06-01"}])

        # Entity registered but no observation (no value fields matched)
        store.store_entity_observation.assert_not_called()

    def test_persist_exception_non_fatal(self):
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB error")
        tool = _make_tool(store=store)

        # Should not raise — exception is caught
        tool._persist_entities(
            [{"ticker": "AAPL", "short_ratio": 0.3, "date": "2025-06-01"}]
        )

    def test_symbol_field_used_as_fallback(self):
        """Short interest records use 'symbol' instead of 'ticker'."""
        store = _make_store()
        tool = _make_tool(store=store)
        tool._persist_entities(
            [
                {
                    "symbol": "GME",
                    "settlement_date": "2025-05-31",
                    "current_short_position": 80_000_000,
                }
            ]
        )

        call = store.register_entity.call_args
        assert call.kwargs["canonical_name"] == "GME"
