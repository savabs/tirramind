"""Edge-case tests for pre-world-model convergence fixes.

Covers:
  F1 — consumer_sentiment, food_security, political_risk extractors
  F2 — direction propagation from coincidence → DetectionResult → ConvergenceSignal
  F4 — persistence_count wiring in DAG callback
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.convergence.evidence import Evidence
from agent.convergence.extractors import extract_evidence, registered_tools


# ══════════════════════════════════════════════════════════════
#  F1: consumer_sentiment extractor
# ══════════════════════════════════════════════════════════════


class TestConsumerSentimentExtractor:
    """Tests for the consumer_sentiment extractor."""

    def test_registered(self):
        assert "consumer_sentiment" in registered_tools()

    # ── eu_confidence mode ─────────────────────────────────────

    def test_eu_confidence_basic(self):
        data = {
            "mode": "eu_confidence",
            "signals": {
                "countries": {
                    "DE": {
                        "latest": -5.2,
                        "mom_change": -2.1,
                        "trend": "DETERIORATING",
                    },
                    "FR": {"latest": 3.0, "mom_change": 1.5, "trend": "IMPROVING"},
                },
                "synchronized_decline": False,
            },
        }
        result = extract_evidence("consumer_sentiment", data)
        assert len(result) == 2
        ids = {e.signal_id for e in result}
        assert "consumer_sentiment.eu.de" in ids
        assert "consumer_sentiment.eu.fr" in ids

    def test_eu_confidence_direction(self):
        data = {
            "mode": "eu_confidence",
            "signals": {
                "countries": {
                    "DE": {"latest": -10.0, "mom_change": -3.0},
                    "FR": {"latest": 5.0, "mom_change": 2.0},
                    "IT": {"latest": 0.0, "mom_change": 0.0},
                },
            },
        }
        result = extract_evidence("consumer_sentiment", data)
        by_id = {e.signal_id: e for e in result}
        assert by_id["consumer_sentiment.eu.de"].direction == -1
        assert by_id["consumer_sentiment.eu.fr"].direction == 1
        assert by_id["consumer_sentiment.eu.it"].direction == 0

    def test_eu_synchronized_decline(self):
        data = {
            "mode": "eu_confidence",
            "signals": {
                "countries": {
                    "DE": {"latest": -10.0, "mom_change": -3.0},
                },
                "synchronized_decline": True,
            },
        }
        result = extract_evidence("consumer_sentiment", data)
        sync = [e for e in result if "synchronized_decline" in e.signal_id]
        assert len(sync) == 1
        assert sync[0].direction == -1
        assert sync[0].value == 1.0
        assert sync[0].confidence == 0.9

    def test_eu_no_synchronized_decline_when_false(self):
        data = {
            "mode": "eu_confidence",
            "signals": {
                "countries": {"DE": {"latest": 5.0, "mom_change": 2.0}},
                "synchronized_decline": False,
            },
        }
        result = extract_evidence("consumer_sentiment", data)
        sync = [e for e in result if "synchronized_decline" in e.signal_id]
        assert len(sync) == 0

    def test_eu_empty_countries(self):
        data = {
            "mode": "eu_confidence",
            "signals": {"countries": {}},
        }
        assert extract_evidence("consumer_sentiment", data) == []

    def test_eu_missing_mom_change(self):
        data = {
            "mode": "eu_confidence",
            "signals": {
                "countries": {"DE": {"latest": -5.0}},
            },
        }
        result = extract_evidence("consumer_sentiment", data)
        assert len(result) == 1
        assert result[0].direction == 0

    def test_eu_none_latest_skipped(self):
        data = {
            "mode": "eu_confidence",
            "signals": {
                "countries": {"DE": {"latest": None, "mom_change": -1.0}},
            },
        }
        assert extract_evidence("consumer_sentiment", data) == []

    def test_eu_category_is_macro_momentum(self):
        data = {
            "mode": "eu_confidence",
            "signals": {"countries": {"DE": {"latest": -5.0, "mom_change": 1.0}}},
        }
        result = extract_evidence("consumer_sentiment", data)
        assert all(e.category == "macro_momentum" for e in result)

    # ── us_sentiment mode ──────────────────────────────────────

    def test_us_sentiment_basic(self):
        data = {
            "mode": "us_sentiment",
            "signals": {
                "sentiment_latest": 65.0,
                "sentiment_mom_change": -2.5,
            },
        }
        result = extract_evidence("consumer_sentiment", data)
        assert len(result) >= 1
        headline = [e for e in result if "headline" in e.signal_id]
        assert len(headline) == 1
        assert headline[0].value == 65.0
        assert headline[0].direction == -1  # mom_change < 0

    def test_us_sentiment_no_mom(self):
        data = {
            "mode": "us_sentiment",
            "signals": {"sentiment_latest": 80.0},
        }
        result = extract_evidence("consumer_sentiment", data)
        headline = [e for e in result if "headline" in e.signal_id]
        assert headline[0].direction == 1  # >= 70, optimistic

    def test_us_sentiment_low_level(self):
        data = {
            "mode": "us_sentiment",
            "signals": {"sentiment_latest": 45.0},
        }
        result = extract_evidence("consumer_sentiment", data)
        headline = [e for e in result if "headline" in e.signal_id]
        assert headline[0].direction == -1  # <= 50, pessimistic

    def test_us_inflation_expectations(self):
        data = {
            "mode": "us_sentiment",
            "signals": {
                "sentiment_latest": 65.0,
                "inflation_exp_latest": 4.5,
            },
        }
        result = extract_evidence("consumer_sentiment", data)
        inf = [e for e in result if "inflation_expectations" in e.signal_id]
        assert len(inf) == 1
        assert inf[0].direction == 1  # > 3.0 = stress

    def test_us_inflation_low(self):
        data = {
            "mode": "us_sentiment",
            "signals": {
                "sentiment_latest": 65.0,
                "inflation_exp_latest": 2.0,
            },
        }
        result = extract_evidence("consumer_sentiment", data)
        inf = [e for e in result if "inflation_expectations" in e.signal_id]
        assert inf[0].direction == -1  # <= 3.0

    def test_us_no_signals(self):
        data = {"mode": "us_sentiment", "signals": {}}
        assert extract_evidence("consumer_sentiment", data) == []

    # ── inflation_reality mode ─────────────────────────────────

    def test_cpi_mom_positive(self):
        data = {
            "mode": "inflation_reality",
            "signals": {"cpi_mom_change": 0.3},
        }
        result = extract_evidence("consumer_sentiment", data)
        cpi = [e for e in result if "cpi.mom" in e.signal_id]
        assert len(cpi) == 1
        assert cpi[0].direction == 1

    def test_cpi_mom_negative(self):
        data = {
            "mode": "inflation_reality",
            "signals": {"cpi_mom_change": -0.2},
        }
        result = extract_evidence("consumer_sentiment", data)
        cpi = [e for e in result if "cpi.mom" in e.signal_id]
        assert cpi[0].direction == -1

    def test_expectations_gap(self):
        data = {
            "mode": "inflation_reality",
            "signals": {"expectations_gap": 1.5},
        }
        result = extract_evidence("consumer_sentiment", data)
        gap = [e for e in result if "expectations_gap" in e.signal_id]
        assert len(gap) == 1
        assert gap[0].direction == 1  # positive gap = inflation fears

    def test_expectations_gap_negative(self):
        data = {
            "mode": "inflation_reality",
            "signals": {"expectations_gap": -0.5},
        }
        result = extract_evidence("consumer_sentiment", data)
        gap = [e for e in result if "expectations_gap" in e.signal_id]
        assert gap[0].direction == -1

    # ── Edge cases ─────────────────────────────────────────────

    def test_unknown_mode_returns_empty(self):
        data = {"mode": "unknown_mode", "signals": {}}
        assert extract_evidence("consumer_sentiment", data) == []

    def test_none_data(self):
        assert extract_evidence("consumer_sentiment", None) == []

    def test_non_dict_data(self):
        assert extract_evidence("consumer_sentiment", "string") == []

    def test_missing_signals_key(self):
        data = {"mode": "eu_confidence"}
        assert extract_evidence("consumer_sentiment", data) == []

    def test_non_dict_country_entry_skipped(self):
        data = {
            "mode": "eu_confidence",
            "signals": {"countries": {"DE": "not_a_dict"}},
        }
        assert extract_evidence("consumer_sentiment", data) == []


# ══════════════════════════════════════════════════════════════
#  F1: food_security extractor
# ══════════════════════════════════════════════════════════════


class TestFoodSecurityExtractor:
    """Tests for the food_security extractor."""

    def test_registered(self):
        assert "food_security" in registered_tools()

    def test_production_food(self):
        data = {
            "records": [
                {
                    "country": "IN",
                    "year": "2021",
                    "value": 105.0,
                    "indicator": "AG.PRD.FOOD.XD",
                },
                {
                    "country": "IN",
                    "year": "2022",
                    "value": 102.0,
                    "indicator": "AG.PRD.FOOD.XD",
                },
            ],
            "valid_count": 2,
            "country": "IN",
            "indicator": "AG.PRD.FOOD.XD",
            "signals": {"trend_direction": "down", "consecutive_years": 1},
        }
        result = extract_evidence("food_security", data)
        assert len(result) >= 1
        main = result[0]
        assert main.signal_id == "food_security.in.production.food"
        assert main.category == "biological"
        assert main.direction == -1
        assert main.value == 102.0

    def test_cereal_yield(self):
        data = {
            "records": [
                {
                    "country": "US",
                    "year": "2023",
                    "value": 8500.0,
                    "indicator": "AG.YLD.CREL.KG",
                }
            ],
            "country": "US",
            "indicator": "AG.YLD.CREL.KG",
            "signals": {"trend_direction": "up"},
        }
        result = extract_evidence("food_security", data)
        assert result[0].signal_id == "food_security.us.cereal_yield"
        assert result[0].direction == 1

    def test_food_import(self):
        data = {
            "records": [
                {
                    "country": "EG",
                    "year": "2023",
                    "value": 35.0,
                    "indicator": "TM.VAL.FOOD.ZS.UN",
                }
            ],
            "country": "EG",
            "indicator": "TM.VAL.FOOD.ZS.UN",
            "signals": {},
        }
        result = extract_evidence("food_security", data)
        assert result[0].signal_id == "food_security.eg.food_import_pct"
        assert result[0].direction == 0  # no trend

    def test_food_export(self):
        data = {
            "records": [
                {
                    "country": "BR",
                    "year": "2023",
                    "value": 25.0,
                    "indicator": "TX.VAL.FOOD.ZS.UN",
                }
            ],
            "country": "BR",
            "indicator": "TX.VAL.FOOD.ZS.UN",
            "signals": {"trend_direction": "up"},
        }
        result = extract_evidence("food_security", data)
        assert "food_export_pct" in result[0].signal_id

    def test_stress_alert_emits_extra_evidence(self):
        data = {
            "records": [
                {
                    "country": "SD",
                    "year": "2023",
                    "value": 80.0,
                    "indicator": "AG.PRD.FOOD.XD",
                }
            ],
            "country": "SD",
            "indicator": "AG.PRD.FOOD.XD",
            "signals": {
                "trend_direction": "down",
                "stress_alert": "Production declining 2+ consecutive years",
            },
        }
        result = extract_evidence("food_security", data)
        assert len(result) == 2
        stress = [e for e in result if "stress" in e.signal_id]
        assert len(stress) == 1
        assert stress[0].direction == -1
        assert stress[0].confidence == 0.9

    def test_stress_alert_bumps_main_confidence(self):
        data = {
            "records": [
                {
                    "country": "SD",
                    "year": "2023",
                    "value": 80.0,
                    "indicator": "AG.PRD.FOOD.XD",
                }
            ],
            "country": "SD",
            "indicator": "AG.PRD.FOOD.XD",
            "signals": {"stress_alert": "Production declining"},
        }
        result = extract_evidence("food_security", data)
        main = [e for e in result if "stress" not in e.signal_id]
        assert main[0].confidence == 0.9

    def test_no_stress_default_confidence(self):
        data = {
            "records": [
                {
                    "country": "US",
                    "year": "2023",
                    "value": 110.0,
                    "indicator": "AG.PRD.FOOD.XD",
                }
            ],
            "country": "US",
            "indicator": "AG.PRD.FOOD.XD",
            "signals": {},
        }
        result = extract_evidence("food_security", data)
        assert result[0].confidence == 0.8

    def test_empty_records(self):
        data = {
            "records": [],
            "country": "US",
            "indicator": "AG.PRD.FOOD.XD",
            "signals": {},
        }
        assert extract_evidence("food_security", data) == []

    def test_all_null_values(self):
        data = {
            "records": [
                {
                    "country": "US",
                    "year": "2023",
                    "value": None,
                    "indicator": "AG.PRD.FOOD.XD",
                }
            ],
            "country": "US",
            "indicator": "AG.PRD.FOOD.XD",
            "signals": {},
        }
        assert extract_evidence("food_security", data) == []

    def test_none_data(self):
        assert extract_evidence("food_security", None) == []

    def test_non_dict_data(self):
        assert extract_evidence("food_security", [1, 2]) == []

    def test_missing_records_key(self):
        data = {"country": "US", "indicator": "AG.PRD.FOOD.XD", "signals": {}}
        assert extract_evidence("food_security", data) == []

    def test_unknown_indicator_fallback(self):
        data = {
            "records": [
                {
                    "country": "US",
                    "year": "2023",
                    "value": 42.0,
                    "indicator": "XX.YY.ZZ",
                }
            ],
            "country": "US",
            "indicator": "XX.YY.ZZ",
            "signals": {},
        }
        result = extract_evidence("food_security", data)
        # Falls back to cleaned indicator
        assert (
            "xx_yy_zz" in result[0].signal_id.lower()
            or "food_security" in result[0].signal_id
        )


# ══════════════════════════════════════════════════════════════
#  F1: political_risk extractor
# ══════════════════════════════════════════════════════════════


class TestPoliticalRiskExtractor:
    """Tests for the political_risk extractor."""

    def test_registered(self):
        assert "political_risk" in registered_tools()

    # ── expenditures mode ──────────────────────────────────────

    def test_expenditures_basic(self):
        data = {
            "result_type": "expenditures",
            "records": [{"support_oppose": "S", "expenditure_amount": 100000}],
            "signals": {
                "support_total": 100000,
                "oppose_total": 50000,
                "oppose_ratio": 0.333,
                "top_targets": [
                    {"candidate": "Smith, John", "total_spent": 80000},
                ],
            },
        }
        result = extract_evidence("political_risk", data)
        ids = {e.signal_id for e in result}
        assert "political_risk.ie_total_spend" in ids
        assert "political_risk.oppose_ratio" in ids

    def test_expenditures_total_spend_value(self):
        data = {
            "result_type": "expenditures",
            "records": [{"x": 1}],
            "signals": {"support_total": 500000, "oppose_total": 300000},
        }
        result = extract_evidence("political_risk", data)
        total = [e for e in result if "ie_total_spend" in e.signal_id]
        assert total[0].value == 800000.0

    def test_expenditures_high_oppose_ratio(self):
        data = {
            "result_type": "expenditures",
            "records": [{"x": 1}],
            "signals": {"oppose_ratio": 0.7},
        }
        result = extract_evidence("political_risk", data)
        opp = [e for e in result if "oppose_ratio" in e.signal_id]
        assert opp[0].direction == 1  # > 0.5

    def test_expenditures_low_oppose_ratio(self):
        data = {
            "result_type": "expenditures",
            "records": [{"x": 1}],
            "signals": {"oppose_ratio": 0.3},
        }
        result = extract_evidence("political_risk", data)
        opp = [e for e in result if "oppose_ratio" in e.signal_id]
        assert opp[0].direction == -1  # < 0.5

    def test_top_target_slug(self):
        data = {
            "result_type": "expenditures",
            "records": [{"x": 1}],
            "signals": {
                "top_targets": [{"candidate": "Biden, Joe", "total_spent": 50000}],
            },
        }
        result = extract_evidence("political_risk", data)
        target = [e for e in result if "target." in e.signal_id]
        assert len(target) == 1
        assert "biden" in target[0].signal_id

    def test_zero_total_spend_skipped(self):
        data = {
            "result_type": "expenditures",
            "records": [{"x": 1}],
            "signals": {"support_total": 0, "oppose_total": 0},
        }
        result = extract_evidence("political_risk", data)
        total = [e for e in result if "ie_total_spend" in e.signal_id]
        assert len(total) == 0

    def test_top_target_zero_spent_skipped(self):
        data = {
            "result_type": "expenditures",
            "records": [{"x": 1}],
            "signals": {"top_targets": [{"candidate": "X", "total_spent": 0}]},
        }
        result = extract_evidence("political_risk", data)
        target = [e for e in result if "target." in e.signal_id]
        assert len(target) == 0

    # ── filings mode ───────────────────────────────────────────

    def test_filings_avg_cash(self):
        data = {
            "result_type": "filings",
            "records": [{"cash_on_hand_end": 1000000}],
            "signals": {"avg_cash_on_hand": 1000000.0},
        }
        result = extract_evidence("political_risk", data)
        cash = [e for e in result if "avg_cash_on_hand" in e.signal_id]
        assert len(cash) == 1
        assert cash[0].value == 1000000.0

    def test_filings_zero_cash_skipped(self):
        data = {
            "result_type": "filings",
            "records": [{"x": 1}],
            "signals": {"avg_cash_on_hand": 0},
        }
        result = extract_evidence("political_risk", data)
        assert len(result) == 0

    # ── candidates mode ────────────────────────────────────────

    def test_candidates_fundraisers(self):
        data = {
            "result_type": "candidates",
            "records": [{"has_raised_funds": True}],
            "signals": {"active_fundraisers": 10},
        }
        result = extract_evidence("political_risk", data)
        fund = [e for e in result if "active_fundraisers" in e.signal_id]
        assert len(fund) == 1
        assert fund[0].direction == 1  # > 5

    def test_candidates_few_fundraisers(self):
        data = {
            "result_type": "candidates",
            "records": [{"has_raised_funds": True}],
            "signals": {"active_fundraisers": 3},
        }
        result = extract_evidence("political_risk", data)
        fund = [e for e in result if "active_fundraisers" in e.signal_id]
        assert fund[0].direction == 0  # <= 5

    def test_candidates_zero_fundraisers_skipped(self):
        data = {
            "result_type": "candidates",
            "records": [{"x": 1}],
            "signals": {"active_fundraisers": 0},
        }
        assert extract_evidence("political_risk", data) == []

    # ── Edge cases ─────────────────────────────────────────────

    def test_unknown_result_type(self):
        data = {"result_type": "unknown", "records": [{"x": 1}], "signals": {}}
        assert extract_evidence("political_risk", data) == []

    def test_empty_records(self):
        data = {"result_type": "expenditures", "records": [], "signals": {}}
        assert extract_evidence("political_risk", data) == []

    def test_none_data(self):
        assert extract_evidence("political_risk", None) == []

    def test_non_dict_data(self):
        assert extract_evidence("political_risk", 42) == []

    def test_category_is_geopolitical(self):
        data = {
            "result_type": "expenditures",
            "records": [{"x": 1}],
            "signals": {"support_total": 100, "oppose_total": 100},
        }
        result = extract_evidence("political_risk", data)
        assert all(e.category == "geopolitical" for e in result)

    def test_non_dict_top_target_skipped(self):
        data = {
            "result_type": "expenditures",
            "records": [{"x": 1}],
            "signals": {"top_targets": ["not_a_dict"]},
        }
        result = extract_evidence("political_risk", data)
        target = [e for e in result if "target." in e.signal_id]
        assert len(target) == 0


# ══════════════════════════════════════════════════════════════
#  F2: Direction propagation
# ══════════════════════════════════════════════════════════════


class TestDirectionPropagation:
    """Verify direction flows from coincidence → DetectionResult → ConvergenceSignal."""

    def test_detection_result_has_direction_field(self):
        from agent.convergence.detector import DetectionResult
        from agent.convergence.graph import ConvergenceClique

        clique = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["cat1", "cat2"],
            edges=[("a", "b", 0.8)],
            score=0.7,
        )
        dr = DetectionResult(
            clique=clique,
            event_type="test",
            template_match=0.0,
            boosted_score=0.7,
            lead_signal=None,
            direction=-1,
        )
        assert dr.direction == -1

    def test_detection_result_default_direction(self):
        from agent.convergence.detector import DetectionResult
        from agent.convergence.graph import ConvergenceClique

        clique = ConvergenceClique(
            signals=["a", "b"],
            categories=["cat1", "cat2"],
            edges=[],
            score=0.5,
        )
        dr = DetectionResult(
            clique=clique,
            event_type="test",
            template_match=0.0,
            boosted_score=0.5,
            lead_signal=None,
        )
        assert dr.direction == 1  # default

    def test_signal_uses_detection_direction(self):
        from agent.convergence.detector import DetectionResult
        from agent.convergence.graph import ConvergenceClique
        from agent.convergence.signals import from_detection_result

        clique = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["cat1", "cat2"],
            edges=[("a", "b", 0.8)],
            score=0.7,
        )
        dr = DetectionResult(
            clique=clique,
            event_type="test_pattern",
            template_match=0.5,
            boosted_score=0.7,
            lead_signal="a",
            direction=-1,
        )
        sig = from_detection_result(dr, as_of=1700000000.0)
        assert sig.direction == -1

    def test_aggregate_direction_positive(self):
        from agent.convergence.coincidence import CoincidenceResult
        from agent.convergence.detector import ConvergenceDetector
        from agent.convergence.graph import ConvergenceClique

        clique = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["cat1", "cat2"],
            edges=[("a", "b", 0.8), ("b", "c", 0.6)],
            score=0.7,
        )
        scores = {
            ("a", "b"): CoincidenceResult(
                method="combined", score=0.8, p_value=0.01, direction=1
            ),
            ("b", "c"): CoincidenceResult(
                method="combined", score=0.6, p_value=0.02, direction=1
            ),
        }
        assert ConvergenceDetector._aggregate_direction(clique, scores) == 1

    def test_aggregate_direction_negative(self):
        from agent.convergence.coincidence import CoincidenceResult
        from agent.convergence.detector import ConvergenceDetector
        from agent.convergence.graph import ConvergenceClique

        clique = ConvergenceClique(
            signals=["a", "b"],
            categories=["cat1", "cat2"],
            edges=[("a", "b", 0.8)],
            score=0.7,
        )
        scores = {
            ("a", "b"): CoincidenceResult(
                method="combined", score=0.9, p_value=0.01, direction=-1
            ),
        }
        assert ConvergenceDetector._aggregate_direction(clique, scores) == -1

    def test_aggregate_direction_mixed(self):
        from agent.convergence.coincidence import CoincidenceResult
        from agent.convergence.detector import ConvergenceDetector
        from agent.convergence.graph import ConvergenceClique

        clique = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["cat1", "cat2"],
            edges=[],
            score=0.5,
        )
        # Opposite directions, but +1 has more weight
        scores = {
            ("a", "b"): CoincidenceResult(
                method="combined", score=0.9, p_value=0.01, direction=1
            ),
            ("b", "c"): CoincidenceResult(
                method="combined", score=0.3, p_value=0.1, direction=-1
            ),
        }
        assert ConvergenceDetector._aggregate_direction(clique, scores) == 1

    def test_aggregate_direction_no_matching_pairs(self):
        from agent.convergence.coincidence import CoincidenceResult
        from agent.convergence.detector import ConvergenceDetector
        from agent.convergence.graph import ConvergenceClique

        clique = ConvergenceClique(
            signals=["a", "b"],
            categories=["cat1", "cat2"],
            edges=[],
            score=0.5,
        )
        # No pairs match the clique's signals
        scores = {
            ("x", "y"): CoincidenceResult(
                method="combined", score=0.9, p_value=0.01, direction=-1
            ),
        }
        assert ConvergenceDetector._aggregate_direction(clique, scores) == 1  # default

    def test_aggregate_direction_zero_scores(self):
        from agent.convergence.coincidence import CoincidenceResult
        from agent.convergence.detector import ConvergenceDetector
        from agent.convergence.graph import ConvergenceClique

        clique = ConvergenceClique(
            signals=["a", "b"],
            categories=["cat1", "cat2"],
            edges=[],
            score=0.5,
        )
        scores = {
            ("a", "b"): CoincidenceResult(
                method="combined", score=0.0, p_value=1.0, direction=-1
            ),
        }
        assert (
            ConvergenceDetector._aggregate_direction(clique, scores) == 1
        )  # default (zero weight)


# ══════════════════════════════════════════════════════════════
#  F4: persistence_count wiring
# ══════════════════════════════════════════════════════════════


class TestPersistenceCountWiring:
    """Verify the DAG callback passes persistence_count to from_detection_result."""

    def test_from_detection_result_uses_persistence(self):
        from agent.convergence.detector import DetectionResult
        from agent.convergence.graph import ConvergenceClique
        from agent.convergence.signals import from_detection_result

        clique = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["cat1", "cat2"],
            edges=[],
            score=0.7,
        )
        dr = DetectionResult(
            clique=clique,
            event_type="test_pattern",
            template_match=0.0,
            boosted_score=0.7,
            lead_signal=None,
        )
        sig = from_detection_result(dr, persistence_count=5, as_of=1700000000.0)
        assert sig.persistence_days == 5

    def test_default_persistence_is_zero(self):
        from agent.convergence.detector import DetectionResult
        from agent.convergence.graph import ConvergenceClique
        from agent.convergence.signals import from_detection_result

        clique = ConvergenceClique(
            signals=["a", "b", "c"],
            categories=["cat1", "cat2"],
            edges=[],
            score=0.7,
        )
        dr = DetectionResult(
            clique=clique,
            event_type="test_pattern",
            template_match=0.0,
            boosted_score=0.7,
            lead_signal=None,
        )
        sig = from_detection_result(dr, as_of=1700000000.0)
        assert sig.persistence_days == 0

    def test_dag_callback_passes_persistence(self):
        """Integration: DAG callback reads persistence_history from detector."""
        from agent.pipeline.dags.convergence_detection import run_convergence_detection

        # This is a structural test — verify the code path exists
        # by checking the function source references persistence_history.
        import inspect

        src = inspect.getsource(run_convergence_detection)
        assert "persistence_history" in src
        assert "persistence_count" in src
        assert "fingerprint" in src


# ══════════════════════════════════════════════════════════════
#  All evidence validation
# ══════════════════════════════════════════════════════════════


class TestAllNewExtractorsProduceValidEvidence:
    """Every Evidence produced must pass Evidence's own validation."""

    @pytest.mark.parametrize(
        "tool,data",
        [
            (
                "consumer_sentiment",
                {
                    "mode": "eu_confidence",
                    "signals": {
                        "countries": {"DE": {"latest": -5.0, "mom_change": -1.0}}
                    },
                },
            ),
            (
                "consumer_sentiment",
                {
                    "mode": "us_sentiment",
                    "signals": {"sentiment_latest": 65.0, "inflation_exp_latest": 4.0},
                },
            ),
            (
                "consumer_sentiment",
                {
                    "mode": "inflation_reality",
                    "signals": {"cpi_mom_change": 0.3, "expectations_gap": 1.0},
                },
            ),
            (
                "food_security",
                {
                    "records": [
                        {
                            "country": "US",
                            "year": "2023",
                            "value": 105.0,
                            "indicator": "AG.PRD.FOOD.XD",
                        }
                    ],
                    "country": "US",
                    "indicator": "AG.PRD.FOOD.XD",
                    "signals": {"trend_direction": "up"},
                },
            ),
            (
                "political_risk",
                {
                    "result_type": "expenditures",
                    "records": [{"x": 1}],
                    "signals": {
                        "support_total": 100000,
                        "oppose_total": 50000,
                        "oppose_ratio": 0.333,
                    },
                },
            ),
        ],
    )
    def test_evidence_valid(self, tool, data):
        results = extract_evidence(tool, data)
        assert len(results) >= 1
        for ev in results:
            assert isinstance(ev, Evidence)
            assert ev.source == tool
            assert ev.signal_id
            assert ev.category in {
                "behavioral_intent",
                "biological",
                "financial_stress",
                "geopolitical",
                "macro_momentum",
                "monetary_policy",
                "physical_disruption",
                "physical_flow",
                "positioning",
                "regulatory_action",
                "supply_chain",
            }
            assert ev.direction in (-1, 0, 1)
            assert 0.0 <= ev.confidence <= 1.0
            assert ev.ttl > 0
