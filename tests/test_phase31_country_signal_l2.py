"""Phase 31 edge-case tests for remaining country-signal L2 persistence."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.tools.consumer_sentiment import ConsumerSentimentTool
from agent.tools.food_security import FoodSecurityTool
from agent.tools.internet_outages import InternetOutagesTool
from agent.tools.migration_flows import MigrationFlowsTool, _normalize_country_code


def _store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock()
    store.store_entity_observation = MagicMock(return_value=1)
    return store


class TestConsumerSentimentL2:
    def test_eu_confidence_skips_aggregates(self):
        store = _store()
        tool = ConsumerSentimentTool(pipeline_store=store)
        tool._persist_entities_inner(
            {
                "data": {
                    "EU27_2020": [{"period": "2026-01", "value": -10.0}],
                    "DE": [{"period": "2026-01", "value": -3.0}],
                },
                "signals": {
                    "countries": {
                        "DE": {
                            "mom_change": 0.5,
                            "trend": "STABLE",
                            "consecutive_decline": False,
                        }
                    }
                },
            },
            "eu_confidence",
        )
        assert store.register_entity.call_count == 1
        assert (
            store.store_entity_observation.call_args.kwargs["observation_type"]
            == "consumer_confidence"
        )

    def test_us_sentiment_persists_to_us(self):
        store = _store()
        tool = ConsumerSentimentTool(pipeline_store=store)
        counts = tool._persist_entities_inner(
            {
                "signals": {
                    "sentiment_latest": 72.0,
                    "sentiment_date": "2026-01-01",
                    "sentiment_mom": 1.0,
                    "inflation_exp_1yr": 3.2,
                    "inflation_anchor": "ELEVATED",
                }
            },
            "us_sentiment",
        )
        assert counts["consumer_confidence_obs"] == 1
        assert store.register_entity.call_args.args[:2] == ("country", "US")

    def test_no_store_guard(self):
        tool = ConsumerSentimentTool()
        assert tool._persist_entities({}, "eu_confidence") == {
            "consumer_confidence_obs": 0
        }


class TestFoodSecurityL2:
    def test_wld_skipped(self):
        store = _store()
        tool = FoodSecurityTool(pipeline_store=store)
        counts = tool._persist_entities_inner(
            {"country": "WLD", "records": [], "signals": {}},
            "production",
        )
        assert counts["food_security_obs"] == 0
        assert store.register_entity.call_count == 0

    def test_country_obs_persisted(self):
        store = _store()
        tool = FoodSecurityTool(pipeline_store=store)
        counts = tool._persist_entities_inner(
            {
                "country": "US",
                "indicator": "AG.PRD.FOOD.XD",
                "records": [{"year": "2025", "value": 101.0}],
                "signals": {"yoy_change_pct": 2.0, "trend_direction": "up"},
            },
            "production",
        )
        assert counts["food_security_obs"] == 1
        assert (
            store.store_entity_observation.call_args.kwargs["observation_type"]
            == "food_security"
        )


class TestInternetOutagesL2:
    def test_all_country_skipped(self):
        store = _store()
        tool = InternetOutagesTool(pipeline_store=store)
        counts = tool._persist_entities_inner(
            {"country": "ALL", "signals": {}},
            "censorship",
        )
        assert counts["internet_disruption_obs"] == 0

    def test_network_health_persisted(self):
        store = _store()
        tool = InternetOutagesTool(pipeline_store=store)
        counts = tool._persist_entities_inner(
            {
                "country": "IR",
                "signals": {"disconnect_rate_pct": 55.0, "alert": "CRITICAL"},
            },
            "network_health",
        )
        assert counts["internet_disruption_obs"] == 1
        assert store.register_entity.call_args.args[:2] == ("country", "IR")


class TestMigrationFlowsL2:
    def test_normalize_country_code(self):
        assert _normalize_country_code("DEU") == "DE"
        assert _normalize_country_code("PH") == "PH"
        assert _normalize_country_code("WLD") is None

    def test_global_displacement_skipped(self):
        store = _store()
        tool = MigrationFlowsTool(pipeline_store=store)
        counts = tool._persist_entities_inner(
            {"country": "", "signals": {}},
            "displacement",
        )
        assert counts["migration_pressure_obs"] == 0

    def test_iso3_country_persisted(self):
        store = _store()
        tool = MigrationFlowsTool(pipeline_store=store)
        counts = tool._persist_entities_inner(
            {
                "country": "DEU",
                "role": "asylum",
                "year": 2025,
                "signals": {"acceptance_rate": 42.0, "alert": None},
            },
            "asylum",
        )
        assert counts["migration_pressure_obs"] == 1
        assert store.register_entity.call_args.args[:2] == ("country", "DE")
