"""Tests for agent.convergence.extractors — framework + first 10 tool extractors."""

from __future__ import annotations

import time

import pytest

from agent.convergence.evidence import Evidence
from agent.convergence.extractors import (
    _REGISTRY,
    _safe_float,
    _safe_int,
    extract_evidence,
    register_extractor,
    registered_tools,
)


# ══════════════════════════════════════════════════════════════
# Framework tests
# ══════════════════════════════════════════════════════════════


class TestFramework:
    """Tests for the extractor registry and dispatch."""

    def test_registered_tools_returns_sorted_list(self):
        names = registered_tools()
        assert isinstance(names, list)
        assert names == sorted(names)

    def test_all_49_tools_registered(self):
        expected = {
            "academic_preprints",
            "ais_vessel_tracking",
            "bankruptcy_court",
            "building_permits",
            "capital_flows",
            "central_bank_balance",
            "cert_transparency",
            "cftc",
            "comtrade",
            "consumer_sentiment",
            "creditor_filings",
            "defi_flows",
            "disease_surveillance",
            "dns_monitor",
            "drug_regulatory",
            "earthquake_proximity",
            "electricity_monitor",
            "energy_supply",
            "finra_short_volume",
            "foia_requests",
            "food_security",
            "form144",
            "gdelt",
            "global_pmi",
            "gov_contracts",
            "insider_filings",
            "interconnection_queue",
            "internet_infrastructure",
            "job_postings",
            "labor_disruptions",
            "liquidity_regime",
            "lobbying",
            "macro_data",
            "market_data",
            "patent_filings",
            "political_risk",
            "polymarket",
            "polymarket_whales",
            "power_grid",
            "regulatory_gazette",
            "sanctions_monitor",
            "satellite_activity",
            "sovereign_debt",
            "supply_chain_prices",
            "transport_throughput",
            "treasury_receipts",
            "weather_alerts",
            "whale_alert",
            "wikipedia_pageviews",
        }
        assert expected == set(registered_tools())

    def test_extract_evidence_returns_list(self):
        result = extract_evidence("cftc", {"contracts": []})
        assert isinstance(result, list)

    def test_extract_evidence_unknown_tool(self):
        result = extract_evidence("no_such_tool", {"x": 1})
        assert result == []

    def test_extract_evidence_none_data(self):
        result = extract_evidence("cftc", None)
        assert result == []

    def test_extract_evidence_non_dict_data(self):
        result = extract_evidence("cftc", "not a dict")
        assert result == []

    def test_extract_evidence_catches_exception(self):
        """Extractor that raises should not propagate."""

        def _boom(tool_name, data):
            raise RuntimeError("boom")

        # Use a unique name so it won't collide
        _REGISTRY["__test_boom"] = _boom
        try:
            result = extract_evidence("__test_boom", {"x": 1})
            assert result == []
        finally:
            del _REGISTRY["__test_boom"]

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_extractor("cftc", lambda t, d: [])

    def test_all_evidence_objects_valid(self):
        """Smoke: every extractor with minimal valid data returns valid Evidence."""
        # Provide just enough data to produce at least 0 items without error
        for name in registered_tools():
            result = extract_evidence(name, {})
            assert isinstance(result, list)
            for e in result:
                assert isinstance(e, Evidence)


class TestHelpers:

    def test_safe_float_normal(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float("2.5") == 2.5

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, -1.0) == -1.0

    def test_safe_float_invalid(self):
        assert _safe_float("abc") == 0.0

    def test_safe_int_normal(self):
        assert _safe_int(42) == 42
        assert _safe_int("7") == 7

    def test_safe_int_none(self):
        assert _safe_int(None) == 0
        assert _safe_int(None, -1) == -1

    def test_safe_int_invalid(self):
        assert _safe_int("abc") == 0


# ══════════════════════════════════════════════════════════════
# Per-tool extractor tests
# ══════════════════════════════════════════════════════════════


class TestCFTC:

    def _data(self, **overrides):
        base = {
            "contracts": [
                {
                    "Market_and_Exchange_Names": "CRUDE OIL - NEW YORK MERCANTILE EXCHANGE",
                    "_mm_net_pct_oi": 12.5,
                    "_mm_net": 30000,
                    "_pm_net": -5000,
                    "Open_Interest_All": 100000,
                },
            ],
        }
        base.update(overrides)
        return base

    def test_basic_extraction(self):
        evs = extract_evidence("cftc", self._data())
        assert len(evs) == 1
        e = evs[0]
        assert e.source == "cftc"
        assert e.signal_id.startswith("cftc.")
        assert "mm_net_pct_oi" in e.signal_id
        assert e.value == 12.5
        assert e.direction == 1  # positive positioning
        assert e.category == "positioning"

    def test_negative_positioning(self):
        data = self._data()
        data["contracts"][0]["_mm_net_pct_oi"] = -8.3
        evs = extract_evidence("cftc", data)
        assert evs[0].direction == -1

    def test_zero_positioning(self):
        data = self._data()
        data["contracts"][0]["_mm_net_pct_oi"] = 0
        evs = extract_evidence("cftc", data)
        assert evs[0].direction == 0

    def test_multiple_contracts(self):
        data = {
            "contracts": [
                {"Market_and_Exchange_Names": "GOLD - COMEX", "_mm_net_pct_oi": 5.0},
                {"Market_and_Exchange_Names": "SILVER - COMEX", "_mm_net_pct_oi": -3.0},
            ]
        }
        evs = extract_evidence("cftc", data)
        assert len(evs) == 2
        ids = {e.signal_id for e in evs}
        assert len(ids) == 2  # unique signal IDs

    def test_missing_pct_oi(self):
        data = {"contracts": [{"Market_and_Exchange_Names": "GOLD", "_mm_net": 1000}]}
        evs = extract_evidence("cftc", data)
        assert evs == []

    def test_empty_contracts(self):
        evs = extract_evidence("cftc", {"contracts": []})
        assert evs == []

    def test_no_contracts_key(self):
        evs = extract_evidence("cftc", {"something_else": 123})
        assert evs == []

    def test_contracts_not_list(self):
        evs = extract_evidence("cftc", {"contracts": "invalid"})
        assert evs == []

    def test_non_dict_contract_entry(self):
        evs = extract_evidence("cftc", {"contracts": [None, 42, "bad"]})
        assert evs == []

    def test_ttl_is_weekly(self):
        evs = extract_evidence("cftc", self._data())
        assert evs[0].ttl == 604_800


class TestWeatherAlerts:

    def test_summary_mode(self):
        data = {"alert_count": 15, "fire_count_infra": 3}
        evs = extract_evidence("weather_alerts", data)
        assert any("alert_count" in e.signal_id for e in evs)
        assert any("fire_count_infra" in e.signal_id for e in evs)

    def test_alerts_mode(self):
        data = {
            "alerts": [
                {
                    "event": "Tornado Warning",
                    "severity": "Extreme",
                    "market_relevant": True,
                },
                {
                    "event": "Frost Advisory",
                    "severity": "Minor",
                    "market_relevant": False,
                },
                {
                    "event": "Hurricane Watch",
                    "severity": "Severe",
                    "market_relevant": True,
                },
            ]
        }
        evs = extract_evidence("weather_alerts", data)
        severe = next(e for e in evs if "severe" in e.signal_id)
        assert severe.value == 2.0  # Extreme + Severe
        market = next(e for e in evs if "market_relevant" in e.signal_id)
        assert market.value == 2.0

    def test_alerts_mode_no_severe(self):
        data = {
            "alerts": [
                {
                    "event": "Frost Advisory",
                    "severity": "Minor",
                    "market_relevant": False,
                },
            ]
        }
        evs = extract_evidence("weather_alerts", data)
        severe = next(e for e in evs if "severe" in e.signal_id)
        assert severe.value == 0.0
        assert severe.direction == 0

    def test_fires_mode(self):
        data = {
            "fires": [{"lat": 40, "lon": -100}],
            "zones_affected": ["oil_refining_gulf", "power_grid_western"],
        }
        evs = extract_evidence("weather_alerts", data)
        zone_ev = next(e for e in evs if "fire_zones" in e.signal_id)
        assert zone_ev.value == 2.0

    def test_zero_alerts(self):
        data = {"alert_count": 0}
        evs = extract_evidence("weather_alerts", data)
        e = evs[0]
        assert e.direction == 0

    def test_empty_dict(self):
        evs = extract_evidence("weather_alerts", {})
        assert evs == []


class TestSanctionsMonitor:

    def test_recent_mode(self):
        data = {"results": [{"name": "Entity1"}], "count": 1, "days_back": 7}
        evs = extract_evidence("sanctions_monitor", data)
        assert len(evs) == 1
        assert evs[0].signal_id == "sanctions.global.recent_additions"
        assert evs[0].value == 1.0

    def test_programs_mode(self):
        data = {
            "programs": [
                {"name": "SDGT", "count": 500},
                {"name": "IRAN", "count": 200},
            ],
            "count": 2,
        }
        evs = extract_evidence("sanctions_monitor", data)
        assert any("program_count" in e.signal_id for e in evs)
        assert any("total_entries" in e.signal_id for e in evs)
        total_ev = next(e for e in evs if "total_entries" in e.signal_id)
        assert total_ev.value == 700.0

    def test_search_mode_ignored(self):
        data = {"query": "some entity", "results": [{"name": "X"}], "count": 1}
        evs = extract_evidence("sanctions_monitor", data)
        # Search mode has results but no days_back or programs → empty
        assert evs == []

    def test_empty(self):
        evs = extract_evidence("sanctions_monitor", {})
        assert evs == []


class TestAISVessel:

    def test_area_mode(self):
        data = {
            "area": "Strait of Hormuz",
            "total_vessels": 42,
            "type_counts": {"tanker": 18, "cargo": 12, "other": 12},
        }
        evs = extract_evidence("ais_vessel_tracking", data)
        count_ev = next(e for e in evs if "vessel_count" in e.signal_id)
        assert count_ev.value == 42.0
        assert count_ev.category == "physical_flow"
        ratio_ev = next(e for e in evs if "tanker_ratio" in e.signal_id)
        assert abs(ratio_ev.value - 18 / 42) < 1e-6

    def test_destination_flow_mode(self):
        data = {"strategic": {"suez": 30, "russia": 5, "rotterdam": 20}}
        evs = extract_evidence("ais_vessel_tracking", data)
        assert len(evs) == 3
        assert all("ais.destination." in e.signal_id for e in evs)

    def test_zero_vessels(self):
        data = {"area": "Empty Zone", "total_vessels": 0, "type_counts": {"tanker": 0}}
        evs = extract_evidence("ais_vessel_tracking", data)
        count_ev = next(e for e in evs if "vessel_count" in e.signal_id)
        assert count_ev.value == 0.0
        # No tanker_ratio when total=0 (div by zero guard)
        assert not any("tanker_ratio" in e.signal_id for e in evs)

    def test_empty(self):
        evs = extract_evidence("ais_vessel_tracking", {})
        assert evs == []


class TestFINRAShortVolume:

    def test_short_volume_mode(self):
        data = {
            "ticker": "SPY",
            "signals": {
                "latest_ratio": 0.55,
                "zscore": 2.1,
                "trend": "rising",
                "is_anomaly": True,
            },
        }
        evs = extract_evidence("finra_short_volume", data)
        ratio_ev = next(e for e in evs if "short_ratio" in e.signal_id)
        assert ratio_ev.value == 0.55
        assert ratio_ev.direction == 1  # > 0.5 = bearish
        assert ratio_ev.confidence == 0.8  # anomaly → higher conf

    def test_short_volume_non_anomaly(self):
        data = {
            "ticker": "SPY",
            "signals": {
                "latest_ratio": 0.42,
                "zscore": 0.5,
                "is_anomaly": False,
            },
        }
        evs = extract_evidence("finra_short_volume", data)
        ratio_ev = next(e for e in evs if "short_ratio" in e.signal_id)
        assert ratio_ev.direction == -1  # < 0.5
        assert ratio_ev.confidence == 0.6

    def test_short_interest_mode(self):
        data = {
            "ticker": "GME",
            "signals": {
                "squeeze_risk": True,
                "days_to_cover": 7.5,
            },
        }
        evs = extract_evidence("finra_short_volume", data)
        dtc_ev = next(e for e in evs if "days_to_cover" in e.signal_id)
        assert dtc_ev.value == 7.5
        assert dtc_ev.direction == 1  # > 5.0

    def test_no_signals(self):
        evs = extract_evidence("finra_short_volume", {"ticker": "SPY"})
        assert evs == []


class TestDiseaseSurveillance:

    def test_wastewater_mode(self):
        data = {
            "pathogen": "SARS-CoV-2",
            "total_samples": 1000,
            "hot_states": 8,
            "summaries": [
                {"state": "CA", "detections": 300},
                {"state": "TX", "detections": 200},
            ],
        }
        evs = extract_evidence("disease_surveillance", data)
        rate_ev = next(e for e in evs if "detection_rate" in e.signal_id)
        assert rate_ev.value == 0.5  # 500/1000
        assert rate_ev.direction == 1  # > 0.3
        hot_ev = next(e for e in evs if "hot_states" in e.signal_id)
        assert hot_ev.value == 8.0

    def test_outbreak_mode(self):
        data = {
            "entries": [
                {"title": "Ebola - DRC", "date": "2024-01-01"},
                {"title": "Cholera - Haiti", "date": "2024-01-02"},
                {"title": "MERS - Saudi", "date": "2024-01-03"},
                {"title": "Mpox - Ghana", "date": "2024-01-04"},
            ]
        }
        evs = extract_evidence("disease_surveillance", data)
        count_ev = next(e for e in evs if "outbreak_count" in e.signal_id)
        assert count_ev.value == 4.0
        assert count_ev.direction == 1  # > 3

    def test_low_detection_rate(self):
        data = {
            "pathogen": "influenza",
            "total_samples": 1000,
            "hot_states": 2,
            "summaries": [{"state": "NY", "detections": 50}],
        }
        evs = extract_evidence("disease_surveillance", data)
        rate_ev = next(e for e in evs if "detection_rate" in e.signal_id)
        assert rate_ev.direction == 0  # <= 0.3

    def test_empty(self):
        evs = extract_evidence("disease_surveillance", {})
        assert evs == []


class TestEarthquakeProximity:

    def test_recent_mode(self):
        data = {
            "quakes": [
                {"magnitude": 6.2, "place": "Chile", "lat": -33.0, "lon": -71.0},
                {"magnitude": 3.1, "place": "Alaska", "lat": 61.0, "lon": -149.0},
            ],
            "count": 2,
            "near_infrastructure": 1,
        }
        evs = extract_evidence("earthquake_proximity", data)
        count_ev = next(e for e in evs if "count" in e.signal_id)
        assert count_ev.value == 2.0
        mag_ev = next(e for e in evs if "max_magnitude" in e.signal_id)
        assert mag_ev.value == 6.2
        assert mag_ev.direction == 1  # >= 6.0
        infra_ev = next(e for e in evs if "infrastructure" in e.signal_id)
        assert infra_ev.value == 1.0

    def test_monitor_mode(self):
        data = {
            "zone": {"name": "Gulf Coast", "sector": "oil"},
            "quakes": [{"magnitude": 4.5}],
            "count": 1,
        }
        evs = extract_evidence("earthquake_proximity", data)
        assert len(evs) == 2
        assert any("gulf_coast" in e.signal_id for e in evs)

    def test_empty_quakes(self):
        evs = extract_evidence("earthquake_proximity", {"quakes": []})
        assert evs == []

    def test_no_quakes_key(self):
        evs = extract_evidence("earthquake_proximity", {"count": 0})
        assert evs == []


class TestGlobalPMI:

    def test_basic_signals(self):
        data = {
            "mode": "cli",
            "signals": {
                "USA": {
                    "latest_value": 101.5,
                    "mom_change": 0.3,
                    "regime": "expanding",
                },
                "DEU": {
                    "latest_value": 98.2,
                    "mom_change": -0.5,
                    "regime": "contracting",
                },
            },
        }
        evs = extract_evidence("global_pmi", data)
        usa_ev = next(e for e in evs if "pmi.usa.cli" == e.signal_id)
        assert usa_ev.value == 101.5
        assert usa_ev.direction == 1  # > 100
        deu_ev = next(e for e in evs if "pmi.deu.cli" == e.signal_id)
        assert deu_ev.direction == -1  # < 100

    def test_mom_momentum_signals(self):
        data = {
            "signals": {
                "USA": {"latest_value": 100.5, "mom_change": 0.3, "regime": "stable"},
            },
        }
        evs = extract_evidence("global_pmi", data)
        mom_ev = next(e for e in evs if "mom" in e.signal_id)
        assert mom_ev.value == 0.3
        assert mom_ev.direction == 1  # positive change

    def test_skips_internal_keys(self):
        data = {
            "signals": {
                "_spreads": {"US_EU": 3.2},
                "USA": {"latest_value": 100.0, "mom_change": 0.0},
            },
        }
        evs = extract_evidence("global_pmi", data)
        assert not any("_spreads" in e.signal_id for e in evs)

    def test_no_signals_key(self):
        evs = extract_evidence("global_pmi", {"records": []})
        assert evs == []


class TestTreasuryReceipts:

    def test_cash_balance_mode(self):
        data = {"signals": {"tga_daily_change_pct": -2.5}}
        evs = extract_evidence("treasury_receipts", data)
        ev = next(e for e in evs if "tga" in e.signal_id)
        assert ev.value == -2.5
        assert ev.direction == 1  # TGA drop = liquidity injection

    def test_tga_rise(self):
        data = {"signals": {"tga_daily_change_pct": 3.0}}
        evs = extract_evidence("treasury_receipts", data)
        ev = evs[0]
        assert ev.direction == -1  # TGA rise = liquidity drain

    def test_deposits_mode(self):
        data = {
            "signals": {
                "net_flow_today": 500_000_000,
                "total_deposits_today": 800_000_000,
            },
        }
        evs = extract_evidence("treasury_receipts", data)
        flow_ev = next(e for e in evs if "net_flow" in e.signal_id)
        assert flow_ev.direction == 1  # positive flow
        dep_ev = next(e for e in evs if "deposits" in e.signal_id)
        assert dep_ev.direction == 0  # neutral

    def test_empty_signals(self):
        evs = extract_evidence("treasury_receipts", {"signals": {}})
        assert evs == []


class TestJobPostings:

    def test_jolts_mode(self):
        data = {
            "summary": {
                "JTSJOL": {"latest_value": 9_500_000, "trend": "falling"},
                "JTSLDR": {"latest_value": 1_800_000, "trend": "rising"},
            }
        }
        evs = extract_evidence("job_postings", data)
        open_ev = next(e for e in evs if "openings" in e.signal_id)
        assert open_ev.value == 9_500_000.0
        assert open_ev.direction == 1  # falling openings = stress (flip_sign)
        assert open_ev.category == "behavioral_intent"

        lay_ev = next(e for e in evs if "layoffs" in e.signal_id)
        assert lay_ev.direction == 1  # rising layoffs = stress (no flip)
        assert lay_ev.category == "macro_momentum"

    def test_labor_market_mode(self):
        data = {
            "summary": {
                "UNRATE": {"latest_value": 4.1, "trend": "rising"},
                "ICSA": {"latest_value": 230_000, "trend": "falling"},
                "PAYEMS": {"latest_value": 157_000_000, "trend": "rising"},
            }
        }
        evs = extract_evidence("job_postings", data)
        unemp = next(e for e in evs if "unemployment" in e.signal_id)
        assert unemp.direction == 1  # rising unemployment = stress
        claims = next(e for e in evs if "initial_claims" in e.signal_id)
        assert claims.direction == -1  # falling claims = less stress
        payrolls = next(e for e in evs if "payrolls" in e.signal_id)
        assert payrolls.direction == -1  # rising payrolls = less stress (flip)

    def test_no_summary(self):
        evs = extract_evidence("job_postings", {"records": []})
        assert evs == []

    def test_missing_latest_value(self):
        data = {"summary": {"JTSJOL": {"trend": "falling"}}}
        evs = extract_evidence("job_postings", data)
        assert evs == []


# ══════════════════════════════════════════════════════════════
# Per-tool tests — Batch 2 (new extractors)
# ══════════════════════════════════════════════════════════════


class TestTransportThroughput:

    def test_recent_mode(self):
        data = {
            "records": [
                {"border": "Canada", "total": 15000},
                {"border": "Mexico", "total": 12000},
            ]
        }
        evs = extract_evidence("transport_throughput", data)
        total_ev = next(e for e in evs if "border_total" in e.signal_id)
        assert total_ev.value == 27000.0
        assert total_ev.category == "physical_flow"

    def test_comparison_mode(self):
        data = {
            "comparison": [
                {"date": "2025-01", "canada": 15000, "mexico": 12000, "ratio": 1.25}
            ]
        }
        evs = extract_evidence("transport_throughput", data)
        ratio_ev = next(e for e in evs if "ratio" in e.signal_id)
        assert ratio_ev.value == 1.25

    def test_trend_series(self):
        data = {
            "series": [
                {"date": "2024-12", "total": 10000},
                {"date": "2025-01", "total": 12000},
            ]
        }
        evs = extract_evidence("transport_throughput", data)
        change_ev = next(e for e in evs if "volume_change" in e.signal_id)
        assert abs(change_ev.value - 0.2) < 1e-6
        assert change_ev.direction == 1  # >5% increase

    def test_declining_trend(self):
        data = {
            "series": [
                {"date": "2024-12", "total": 10000},
                {"date": "2025-01", "total": 9000},
            ]
        }
        evs = extract_evidence("transport_throughput", data)
        change_ev = next(e for e in evs if "volume_change" in e.signal_id)
        assert change_ev.direction == -1  # >5% decline

    def test_empty(self):
        assert extract_evidence("transport_throughput", {}) == []

    def test_malformed(self):
        assert extract_evidence("transport_throughput", {"records": "bad"}) == []


class TestCapitalFlows:

    def test_coordinated_selling(self):
        data = {
            "mode": "holdings",
            "coordination": {
                "coordinated_selling": True,
                "coordinated_buying": False,
                "sellers": ["CN", "JP"],
                "buyers": [],
            },
            "holdings": [],
        }
        evs = extract_evidence("capital_flows", data)
        sell_ev = next(e for e in evs if "coordinated_selling" in e.signal_id)
        assert sell_ev.direction == 1
        assert sell_ev.value == 2.0

    def test_coordinated_buying(self):
        data = {
            "coordination": {
                "coordinated_selling": False,
                "coordinated_buying": True,
                "sellers": [],
                "buyers": ["UK", "DE"],
            }
        }
        evs = extract_evidence("capital_flows", data)
        buy_ev = next(e for e in evs if "coordinated_buying" in e.signal_id)
        assert buy_ev.direction == -1

    def test_holdings_mom(self):
        data = {"holdings": [{"country": "China", "mom_change_pct": -5.0}]}
        evs = extract_evidence("capital_flows", data)
        ev = next(e for e in evs if "holdings_mom_pct" in e.signal_id)
        assert ev.direction == 1  # >3% drop

    def test_flow_reversal(self):
        data = {
            "flows": [{"key": "tic_net", "flow_reversal": True, "latest_value": -50000}]
        }
        evs = extract_evidence("capital_flows", data)
        assert any("reversal" in e.signal_id for e in evs)

    def test_reserve_stress(self):
        data = {"stress_alerts": [{"series": "CN_reserves", "stress": True}]}
        evs = extract_evidence("capital_flows", data)
        assert any("stress_count" in e.signal_id for e in evs)

    def test_empty(self):
        assert extract_evidence("capital_flows", {}) == []


class TestSovereignDebt:

    def test_us_yields_inversion(self):
        data = {"records": [{"curve_2s10s": -0.15, "curve_3m10y": -0.3}]}
        evs = extract_evidence("sovereign_debt", data)
        curve_ev = next(e for e in evs if "curve_2s10s" in e.signal_id)
        assert curve_ev.direction == 1  # Inverted
        assert curve_ev.category == "financial_stress"

    def test_us_yields_positive(self):
        data = {"records": [{"curve_2s10s": 1.5, "curve_3m10y": 2.0}]}
        evs = extract_evidence("sovereign_debt", data)
        assert all(e.direction == -1 for e in evs)  # Normal curve

    def test_spreads_mode(self):
        data = {"spreads": [{"country": "Italy", "spread_vs_de": 250}]}
        evs = extract_evidence("sovereign_debt", data)
        ev = evs[0]
        assert ev.direction == 1  # >200bps = stress

    def test_low_spread(self):
        data = {"spreads": [{"country": "France", "spread_vs_de": 50}]}
        evs = extract_evidence("sovereign_debt", data)
        assert evs[0].direction == 0

    def test_empty(self):
        assert extract_evidence("sovereign_debt", {}) == []


class TestCreditorFilings:

    def test_sec_filing_count(self):
        data = {"sec_count": 12}
        evs = extract_evidence("creditor_filings", data)
        ev = evs[0]
        assert ev.direction == 1  # >5

    def test_red_flags(self):
        data = {"red_flags": 3}
        evs = extract_evidence("creditor_filings", data)
        ev = next(e for e in evs if "red_flags" in e.signal_id)
        assert ev.direction == 1

    def test_stress_scan_clusters(self):
        data = {
            "clusters": [
                {"entity": "X"},
                {"entity": "Y"},
                {"entity": "Z"},
                {"entity": "W"},
            ]
        }
        evs = extract_evidence("creditor_filings", data)
        ev = next(e for e in evs if "cluster_count" in e.signal_id)
        assert ev.value == 4.0
        assert ev.direction == 1  # >3

    def test_empty(self):
        assert extract_evidence("creditor_filings", {}) == []


class TestBankruptcyCourt:

    def test_us_bankruptcy(self):
        data = {"mode": "us_bankruptcy", "count": 25, "chapter_breakdown": {"11": 5}}
        evs = extract_evidence("bankruptcy_court", data)
        count_ev = next(e for e in evs if "filing_count" in e.signal_id)
        assert count_ev.direction == 1  # >10
        ch11_ev = next(e for e in evs if "chapter_11" in e.signal_id)
        assert ch11_ev.value == 5.0

    def test_sec_enforcement(self):
        data = {"mode": "sec_enforcement", "count": 8, "type_breakdown": {"fraud": 3}}
        evs = extract_evidence("bankruptcy_court", data)
        assert any("enforcement_count" in e.signal_id for e in evs)

    def test_empty(self):
        assert extract_evidence("bankruptcy_court", {}) == []


class TestLiquidityRegime:

    def test_contraction_regime(self):
        data = {
            "current_regime": "contraction",
            "current_state": 0,
            "composite_zscore": -2.0,
            "n_changepoints": 3,
        }
        evs = extract_evidence("liquidity_regime", data)
        regime_ev = next(
            e for e in evs if "regime" in e.signal_id and "zscore" not in e.signal_id
        )
        assert regime_ev.direction == 1  # Contraction = stress

    def test_expansion_regime(self):
        data = {
            "current_regime": "expansion",
            "current_state": 2,
            "composite_zscore": 2.0,
        }
        evs = extract_evidence("liquidity_regime", data)
        regime_ev = next(
            e for e in evs if "regime" in e.signal_id and "zscore" not in e.signal_id
        )
        assert regime_ev.direction == -1  # Expansion = relief

    def test_zscore_stress(self):
        data = {"composite_zscore": -2.5}
        evs = extract_evidence("liquidity_regime", data)
        ev = next(e for e in evs if "zscore" in e.signal_id)
        assert ev.direction == 1  # Below -1.5

    def test_empty(self):
        assert extract_evidence("liquidity_regime", {}) == []


class TestCentralBankBalance:

    def test_balance_sheet_contraction(self):
        data = {"banks": [{"code": "FED", "wow_pct": -1.2}]}
        evs = extract_evidence("central_bank_balance", data)
        ev = evs[0]
        assert ev.direction == 1  # Contraction = tightening = stress

    def test_balance_sheet_expansion(self):
        data = {"banks": [{"code": "ECB", "wow_pct": 0.8}]}
        evs = extract_evidence("central_bank_balance", data)
        assert evs[0].direction == -1  # Expansion = easing

    def test_net_liquidity(self):
        data = {"net_usd": 5.5}
        evs = extract_evidence("central_bank_balance", data)
        assert any("net_liquidity" in e.signal_id for e in evs)

    def test_policy_divergence(self):
        data = {
            "synchronized": False,
            "divergences": [
                {"pair": "FED-ECB"},
                {"pair": "FED-BOJ"},
                {"pair": "ECB-BOE"},
            ],
        }
        evs = extract_evidence("central_bank_balance", data)
        sync_ev = next(e for e in evs if "synchronized" in e.signal_id)
        assert sync_ev.value == 0.0
        div_ev = next(e for e in evs if "divergence_count" in e.signal_id)
        assert div_ev.direction == 1  # >= 3 divergences

    def test_empty(self):
        assert extract_evidence("central_bank_balance", {}) == []


class TestDrugRegulatory:

    def test_approvals(self):
        data = {"mode": "approvals", "total": 5, "results": []}
        evs = extract_evidence("drug_regulatory", data)
        ev = evs[0]
        assert "approvals" in ev.signal_id
        assert ev.value == 5.0

    def test_adverse_events_serious(self):
        data = {
            "mode": "adverse_events",
            "signals": {"seriousness_ratio": 0.7, "serious_count": 25},
        }
        evs = extract_evidence("drug_regulatory", data)
        sr_ev = next(e for e in evs if "seriousness_ratio" in e.signal_id)
        assert sr_ev.direction == 1  # >0.5
        sc_ev = next(e for e in evs if "serious_count" in e.signal_id)
        assert sc_ev.direction == 1  # >10

    def test_labels_boxed_warning(self):
        data = {
            "mode": "labels",
            "results": [
                {"has_boxed_warning": True},
                {"has_boxed_warning": False},
                {"has_boxed_warning": True},
            ],
        }
        evs = extract_evidence("drug_regulatory", data)
        ev = next(e for e in evs if "boxed_warning" in e.signal_id)
        assert ev.value == 2.0

    def test_empty(self):
        assert extract_evidence("drug_regulatory", {}) == []


class TestRegulatoryGazette:

    def test_documents_with_significant(self):
        data = {
            "documents": [
                {"title": "Rule A", "significant": True},
                {"title": "Rule B", "significant": False},
                {"title": "Rule C", "significant": True},
            ],
            "count": 3,
        }
        evs = extract_evidence("regulatory_gazette", data)
        doc_ev = next(e for e in evs if "document_count" in e.signal_id)
        assert doc_ev.value == 3.0
        sig_ev = next(e for e in evs if "significant_count" in e.signal_id)
        assert sig_ev.value == 2.0
        assert sig_ev.direction == 1

    def test_no_significant(self):
        data = {"documents": [{"title": "Rule A", "significant": False}], "count": 1}
        evs = extract_evidence("regulatory_gazette", data)
        assert len(evs) == 1  # Only document_count
        assert "document_count" in evs[0].signal_id

    def test_empty_documents_still_emits_count(self):
        evs = extract_evidence("regulatory_gazette", {"documents": []})
        assert len(evs) == 1
        assert evs[0].value == 0.0

    def test_no_documents_key(self):
        assert extract_evidence("regulatory_gazette", {"agencies": {}}) == []


class TestBuildingPermits:

    def test_mom_decline(self):
        data = {"summary": {"PERMIT1": {"mom_pct": -8.0, "label": "Total Permits"}}}
        evs = extract_evidence("building_permits", data)
        ev = evs[0]
        assert ev.direction == 1  # <-5% = decline stress

    def test_consecutive_declines(self):
        data = {"summary": {"PERMIT1": {"mom_pct": -2.0, "consecutive_declines": 4}}}
        evs = extract_evidence("building_permits", data)
        decline_ev = next(e for e in evs if "consecutive_declines" in e.signal_id)
        assert decline_ev.direction == 1

    def test_growth(self):
        data = {"summary": {"PERMIT1": {"mom_pct": 7.0}}}
        evs = extract_evidence("building_permits", data)
        assert evs[0].direction == -1  # >5% = expansion

    def test_empty(self):
        assert extract_evidence("building_permits", {"summary": {}}) == []


class TestPatentFilings:

    def test_trends_with_growth(self):
        data = {
            "cpc_class": "H01L",
            "yearly_counts": {"2023": 1000, "2024": 1300},
            "total_count": 2300,
        }
        evs = extract_evidence("patent_filings", data)
        total_ev = next(e for e in evs if "total_count" in e.signal_id)
        assert total_ev.value == 2300.0
        growth_ev = next(e for e in evs if "yoy_growth" in e.signal_id)
        assert abs(growth_ev.value - 0.3) < 1e-6
        assert growth_ev.direction == 1  # >20% growth

    def test_search_mode(self):
        data = {"mode": "search", "total_count": 50}
        evs = extract_evidence("patent_filings", data)
        assert any("search.total_count" in e.signal_id for e in evs)

    def test_empty(self):
        assert extract_evidence("patent_filings", {}) == []


class TestLobbying:

    def test_spending_anomaly(self):
        data = {
            "mode": "spending",
            "target": "Pharma Inc",
            "anomaly": {"anomaly": True, "ratio": 3.5},
        }
        evs = extract_evidence("lobbying", data)
        ev = evs[0]
        assert ev.direction == 1
        assert ev.confidence == 0.7

    def test_spending_normal(self):
        data = {
            "mode": "spending",
            "target": "Normal Corp",
            "anomaly": {"anomaly": False, "ratio": 1.1},
        }
        evs = extract_evidence("lobbying", data)
        ev = evs[0]
        assert ev.direction == 0
        assert ev.confidence == 0.4

    def test_search_volume(self):
        data = {"mode": "search", "total_count": 150}
        evs = extract_evidence("lobbying", data)
        assert any("filing_count" in e.signal_id for e in evs)

    def test_empty(self):
        assert extract_evidence("lobbying", {}) == []


class TestWikipediaPageviews:

    def test_spike_mode(self):
        data = {
            "spikes": [
                {"article": "Federal Reserve", "z_score": 5.0, "latest_views": 50000}
            ]
        }
        evs = extract_evidence("wikipedia_pageviews", data)
        ev = evs[0]
        assert ev.direction == 1
        assert ev.confidence == min(0.3 + 5.0 * 0.1, 0.9)  # 0.8

    def test_low_zscore_filtered(self):
        data = {"spikes": [{"article": "Boring Page", "z_score": 1.5}]}
        evs = extract_evidence("wikipedia_pageviews", data)
        assert evs == []  # z < 2.0 filtered

    def test_series_mode_with_anomaly(self):
        data = {"article": "Bitcoin", "stats": {"mean": 1000, "std": 200, "max": 2000}}
        evs = extract_evidence("wikipedia_pageviews", data)
        # z = (2000-1000)/200 = 5.0 > 2.0 → should emit
        assert len(evs) == 1
        assert evs[0].direction == 1

    def test_empty(self):
        assert extract_evidence("wikipedia_pageviews", {}) == []


class TestCertTransparency:

    def test_search_mode(self):
        data = {"domain": "example.com", "count": 42, "active": 30, "expired": 12}
        evs = extract_evidence("cert_transparency", data)
        count_ev = next(
            e for e in evs if ".count" in e.signal_id and "subdomain" not in e.signal_id
        )
        assert count_ev.value == 42.0
        ratio_ev = next(e for e in evs if "active_ratio" in e.signal_id)
        assert abs(ratio_ev.value - 30 / 42) < 1e-6

    def test_subdomains_mode(self):
        data = {
            "domain": "big.co",
            "subdomains": [{"subdomain": "a"}, {"subdomain": "b"}],
        }
        evs = extract_evidence("cert_transparency", data)
        assert any("subdomain_count" in e.signal_id for e in evs)

    def test_empty(self):
        assert extract_evidence("cert_transparency", {}) == []


class TestDNSMonitor:

    def test_diff_with_changes(self):
        data = {
            "domain": "target.com",
            "changes": [{"type": "A", "old": "1.1.1.1", "new": "2.2.2.2"}],
        }
        evs = extract_evidence("dns_monitor", data)
        ev = evs[0]
        assert "change_count" in ev.signal_id
        assert ev.direction == 1

    def test_diff_no_changes(self):
        data = {"domain": "stable.com", "changes": []}
        evs = extract_evidence("dns_monitor", data)
        assert evs == []

    def test_bulk_resolve(self):
        data = {"total_records": 100, "domain_count": 5}
        evs = extract_evidence("dns_monitor", data)
        assert any("total_records" in e.signal_id for e in evs)

    def test_empty(self):
        assert extract_evidence("dns_monitor", {}) == []


class TestPolymarket:

    def test_extreme_probability_high_volume(self):
        data = {
            "markets": [
                {
                    "slug": "will-x-happen",
                    "yes_price": 0.92,
                    "volume_24h": 50000,
                    "price_change_24h": 0.05,
                }
            ]
        }
        evs = extract_evidence("polymarket", data)
        prob_ev = next(e for e in evs if "probability" in e.signal_id)
        assert prob_ev.direction == 1  # >0.85

    def test_low_probability(self):
        data = {
            "markets": [
                {
                    "slug": "unlikely-event",
                    "yes_price": 0.08,
                    "volume_24h": 5000,
                    "price_change_24h": -0.02,
                }
            ]
        }
        evs = extract_evidence("polymarket", data)
        prob_ev = next(e for e in evs if "probability" in e.signal_id)
        assert prob_ev.direction == -1  # <0.15

    def test_large_price_move(self):
        data = {
            "markets": [
                {
                    "slug": "big-move",
                    "yes_price": 0.50,
                    "volume_24h": 20000,
                    "price_change_24h": 0.15,
                }
            ]
        }
        evs = extract_evidence("polymarket", data)
        move_ev = next(e for e in evs if "price_change" in e.signal_id)
        assert move_ev.direction == 1  # >10% up

    def test_low_volume_filtered(self):
        data = {
            "markets": [
                {
                    "slug": "tiny",
                    "yes_price": 0.99,
                    "volume_24h": 100,
                    "price_change_24h": 0,
                }
            ]
        }
        evs = extract_evidence("polymarket", data)
        assert evs == []  # Volume < 1000

    def test_empty(self):
        assert extract_evidence("polymarket", {"markets": []}) == []


class TestPolymarketWhales:

    def test_top_wallets(self):
        data = {
            "wallets": [
                {"wallet": "0xabc", "composite": 0.85},
                {"wallet": "0xdef", "composite": 0.72},
            ]
        }
        evs = extract_evidence("polymarket_whales", data)
        ev = next(e for e in evs if "avg_composite" in e.signal_id)
        assert abs(ev.value - 0.785) < 1e-6

    def test_market_whales(self):
        data = {
            "whales": [
                {"wallet": "0xabc", "total_usdc": 50000},
                {"wallet": "0xdef", "total_usdc": 30000},
            ]
        }
        evs = extract_evidence("polymarket_whales", data)
        ev = next(e for e in evs if "concentration" in e.signal_id)
        assert ev.value == 80000.0

    def test_empty(self):
        assert extract_evidence("polymarket_whales", {}) == []


class TestInsiderFilings:

    def test_cluster_buying(self):
        data = {
            "clusters": [
                {
                    "ticker": "AAPL",
                    "insider_count": 3,
                    "total_value": 500000,
                    "conviction": 0.8,
                }
            ],
            "total_purchases": 5,
        }
        evs = extract_evidence("insider_filings", data)
        purchase_ev = next(e for e in evs if "purchase_count" in e.signal_id)
        assert purchase_ev.direction == -1  # Buying = bullish
        cluster_ev = next(e for e in evs if "cluster" in e.signal_id)
        assert cluster_ev.direction == -1
        assert cluster_ev.confidence == min(0.5 + 0.8 * 0.4, 0.95)

    def test_single_insider_no_cluster(self):
        data = {
            "clusters": [
                {
                    "ticker": "MSFT",
                    "insider_count": 1,
                    "total_value": 10000,
                    "conviction": 0.5,
                }
            ],
            "total_purchases": 1,
        }
        evs = extract_evidence("insider_filings", data)
        # insider_count < 2 → no cluster evidence, but still total_purchases
        assert not any("cluster" in e.signal_id for e in evs)

    def test_empty(self):
        assert extract_evidence("insider_filings", {"clusters": []}) == []


class TestForm144:

    def test_sell_cluster(self):
        data = {
            "clusters": [
                {
                    "ticker": "TSLA",
                    "urgency": 0.9,
                    "total_value": 2000000,
                    "pct_of_outstanding": 2.5,
                }
            ],
            "total_filings": 10,
        }
        evs = extract_evidence("form144", data)
        filing_ev = next(e for e in evs if "filing_count" in e.signal_id)
        assert filing_ev.direction == 1  # Selling intent = bearish
        cluster_ev = next(e for e in evs if "sell_cluster" in e.signal_id)
        assert cluster_ev.direction == 1
        pct_ev = next(e for e in evs if "pct_outstanding" in e.signal_id)
        assert pct_ev.value == 2.5

    def test_small_pct_no_significant_signal(self):
        data = {
            "clusters": [
                {
                    "ticker": "IBM",
                    "urgency": 0.3,
                    "total_value": 50000,
                    "pct_of_outstanding": 0.5,
                }
            ],
            "total_filings": 1,
        }
        evs = extract_evidence("form144", data)
        assert not any("pct_outstanding" in e.signal_id for e in evs)  # <1%

    def test_empty(self):
        assert extract_evidence("form144", {"clusters": []}) == []


class TestGDELT:

    def test_events_conflict(self):
        data = {
            "events": [
                {
                    "goldstein": -8.0,
                    "quad_label": "Material Conflict",
                    "num_mentions": 50,
                },
                {
                    "goldstein": -5.0,
                    "quad_label": "Material Conflict",
                    "num_mentions": 30,
                },
                {
                    "goldstein": 2.0,
                    "quad_label": "Verbal Cooperation",
                    "num_mentions": 10,
                },
            ]
        }
        evs = extract_evidence("gdelt", data)
        goldstein_ev = next(e for e in evs if "avg_goldstein" in e.signal_id)
        assert goldstein_ev.direction == 1  # Avg below -3
        count_ev = next(e for e in evs if "event_count" in e.signal_id)
        assert count_ev.value == 3.0
        conflict_ev = next(e for e in evs if "material_conflict_ratio" in e.signal_id)
        assert abs(conflict_ev.value - 2 / 3) < 1e-6

    def test_cooperative_events(self):
        data = {"events": [{"goldstein": 5.0, "quad_label": "Verbal Cooperation"}]}
        evs = extract_evidence("gdelt", data)
        goldstein_ev = next(e for e in evs if "avg_goldstein" in e.signal_id)
        assert goldstein_ev.direction == -1  # >3 = cooperation

    def test_empty_events(self):
        assert extract_evidence("gdelt", {"events": []}) == []


class TestWhaleAlert:

    def test_transactions(self):
        data = {
            "transactions": [{"hash": "abc"}, {"hash": "def"}],
            "summary": {"total_btc": 150.5},
        }
        evs = extract_evidence("whale_alert", data)
        count_ev = next(e for e in evs if "large_tx_count" in e.signal_id)
        assert count_ev.value == 2.0
        vol_ev = next(e for e in evs if "whale_volume" in e.signal_id)
        assert vol_ev.value == 150.5

    def test_empty_transactions(self):
        assert extract_evidence("whale_alert", {"transactions": []}) == []

    def test_empty(self):
        assert extract_evidence("whale_alert", {}) == []


class TestComtrade:

    def test_flows_mode(self):
        data = {
            "reporter": "USA",
            "partner": "China",
            "flow": "import",
            "records": [{"trade_value_usd": 1000000}, {"trade_value_usd": 2000000}],
        }
        evs = extract_evidence("comtrade", data)
        ev = next(e for e in evs if "import_value_usd" in e.signal_id)
        assert ev.value == 3000000.0
        assert ev.category == "supply_chain"

    def test_commodity_mode(self):
        data = {
            "commodity_code": "2709",
            "commodity_name": "Crude petroleum",
            "records": [{"trade_value_usd": 500000}],
        }
        evs = extract_evidence("comtrade", data)
        assert any("2709" in e.signal_id for e in evs)

    def test_empty_records(self):
        assert extract_evidence("comtrade", {"records": []}) == []


class TestEnergySupply:

    def test_stocks_signals(self):
        data = {
            "label": "petroleum_stocks",
            "signals": {
                "crude_stocks": {"latest": 450000, "wow_change": -3000},
                "gasoline_stocks": {"latest": 220000, "wow_change": 1500},
            },
        }
        evs = extract_evidence("energy_supply", data)
        crude_level = next(e for e in evs if "crude_stocks.level" in e.signal_id)
        assert crude_level.value == 450000.0
        crude_wow = next(e for e in evs if "crude_stocks.wow_change" in e.signal_id)
        assert crude_wow.direction == 1  # Stock decline = tightening
        gas_wow = next(e for e in evs if "gasoline_stocks.wow_change" in e.signal_id)
        assert gas_wow.direction == -1  # Stock build

    def test_rig_count(self):
        data = {"count": 580, "records": [{"date": "2025-01", "rigs": 580}]}
        evs = extract_evidence("energy_supply", data)
        assert any("rig_count" in e.signal_id for e in evs)

    def test_empty(self):
        assert extract_evidence("energy_supply", {}) == []


class TestSupplyChainPrices:

    def test_producer_prices(self):
        data = {
            "mode": "producer_prices",
            "signals": {"semiconductors": {"mom_change": 1.2, "latest": 115.0}},
        }
        evs = extract_evidence("supply_chain_prices", data)
        ev = evs[0]
        assert ev.direction == 1  # >0.5% = inflationary

    def test_pressure_index(self):
        data = {"mode": "pressure_index", "pressure": {"score": 0.75}}
        evs = extract_evidence("supply_chain_prices", data)
        ev = next(e for e in evs if "pressure_index" in e.signal_id)
        assert ev.direction == 1  # >0.6

    def test_empty(self):
        assert extract_evidence("supply_chain_prices", {}) == []


class TestMacroData:

    def test_fred_series(self):
        data = {
            "GDP": [
                {"date": "2024-Q3", "value": "25000"},
                {"date": "2024-Q4", "value": "25500"},
            ]
        }
        evs = extract_evidence("macro_data", data)
        ev = evs[0]
        assert ev.value == 25500.0
        assert ev.direction == 0  # Generic
        assert ev.category == "macro_momentum"

    def test_multiple_series(self):
        data = {
            "CPIAUCSL": [{"date": "2024-12", "value": "310.5"}],
            "FEDFUNDS": [{"date": "2024-12", "value": "5.33"}],
        }
        evs = extract_evidence("macro_data", data)
        assert len(evs) == 2

    def test_empty_series(self):
        data = {"GDP": []}
        assert extract_evidence("macro_data", data) == []


class TestMarketData:

    def test_return_calculation(self):
        data = {
            "SPY": [
                {"Close": 500.0, "Volume": 1000000},
                {"Close": 510.0, "Volume": 1200000},
            ]
        }
        evs = extract_evidence("market_data", data)
        ret_ev = next(e for e in evs if "return" in e.signal_id)
        assert abs(ret_ev.value - 0.02) < 1e-6
        vol_ev = next(e for e in evs if "volume" in e.signal_id)
        assert vol_ev.value == 1200000.0

    def test_large_drop(self):
        data = {
            "SPY": [
                {"Close": 500.0, "Volume": 1000000},
                {"Close": 480.0, "Volume": 2000000},
            ]
        }
        evs = extract_evidence("market_data", data)
        ret_ev = next(e for e in evs if "return" in e.signal_id)
        assert ret_ev.direction == 1  # -4% = stress

    def test_single_record_no_return(self):
        data = {"SPY": [{"Close": 500.0, "Volume": 1000000}]}
        evs = extract_evidence("market_data", data)
        assert not any("return" in e.signal_id for e in evs)
        assert any("volume" in e.signal_id for e in evs)

    def test_empty(self):
        assert extract_evidence("market_data", {}) == []


class TestStubExtractors:
    """Output-only tools return [] for any input."""

    @pytest.mark.parametrize(
        "tool",
        [
            "satellite_activity",
            "foia_requests",
            "interconnection_queue",
            "internet_infrastructure",
            "electricity_monitor",
        ],
    )
    def test_stub_returns_empty(self, tool):
        assert extract_evidence(tool, {"anything": "here"}) == []

    @pytest.mark.parametrize(
        "tool",
        [
            "satellite_activity",
            "foia_requests",
            "interconnection_queue",
            "internet_infrastructure",
            "electricity_monitor",
        ],
    )
    def test_stub_handles_none(self, tool):
        assert extract_evidence(tool, None) == []


class TestEdgeCases:

    def test_all_extractors_handle_int_data(self):
        """Passing an int instead of dict should not crash."""
        for name in registered_tools():
            result = extract_evidence(name, 42)
            assert isinstance(result, list)

    def test_all_extractors_handle_string_data(self):
        for name in registered_tools():
            result = extract_evidence(name, "garbage string")
            assert isinstance(result, list)

    def test_all_extractors_handle_list_data(self):
        for name in registered_tools():
            result = extract_evidence(name, [1, 2, 3])
            assert isinstance(result, list)

    def test_all_evidence_has_valid_categories(self):
        """Any Evidence produced must pass validation (category check)."""
        from agent.convergence.taxonomy import CATEGORIES

        test_cases = {
            "cftc": {
                "contracts": [
                    {"Market_and_Exchange_Names": "GOLD", "_mm_net_pct_oi": 5}
                ]
            },
            "weather_alerts": {"alert_count": 10},
            "earthquake_proximity": {
                "quakes": [{"magnitude": 5.0}],
                "count": 1,
                "near_infrastructure": 0,
            },
            "capital_flows": {
                "coordination": {"coordinated_selling": True, "sellers": ["CN"]}
            },
            "sovereign_debt": {"records": [{"curve_2s10s": 0.5}]},
            "bankruptcy_court": {"mode": "us_bankruptcy", "count": 10},
            "liquidity_regime": {"current_regime": "expansion", "current_state": 2},
            "gdelt": {"events": [{"goldstein": -5, "quad_label": "Material Conflict"}]},
            "polymarket": {
                "markets": [
                    {
                        "slug": "test",
                        "yes_price": 0.95,
                        "volume_24h": 5000,
                        "price_change_24h": 0,
                    }
                ]
            },
            "comtrade": {"reporter": "USA", "records": [{"trade_value_usd": 1000}]},
        }
        for tool, data in test_cases.items():
            for ev in extract_evidence(tool, data):
                assert (
                    ev.category in CATEGORIES
                ), f"{tool}: {ev.category} not in CATEGORIES"

    def test_all_evidence_direction_valid(self):
        test_cases = {
            "cftc": {
                "contracts": [{"Market_and_Exchange_Names": "X", "_mm_net_pct_oi": 0}]
            },
            "global_pmi": {"signals": {"USA": {"latest_value": 100, "mom_change": 0}}},
        }
        for tool, data in test_cases.items():
            for ev in extract_evidence(tool, data):
                assert ev.direction in (-1, 0, 1)

    def test_all_evidence_confidence_in_range(self):
        test_cases = {
            "cftc": {
                "contracts": [{"Market_and_Exchange_Names": "X", "_mm_net_pct_oi": 10}]
            },
            "finra_short_volume": {
                "ticker": "A",
                "signals": {"latest_ratio": 0.5, "is_anomaly": True},
            },
        }
        for tool, data in test_cases.items():
            for ev in extract_evidence(tool, data):
                assert 0.0 <= ev.confidence <= 1.0

    def test_all_evidence_ttl_positive(self):
        for tool, data in [
            (
                "cftc",
                {
                    "contracts": [
                        {"Market_and_Exchange_Names": "X", "_mm_net_pct_oi": 1}
                    ]
                },
            ),
            ("weather_alerts", {"alert_count": 1}),
            ("disease_surveillance", {"entries": [{"title": "X"}]}),
        ]:
            for ev in extract_evidence(tool, data):
                assert ev.ttl > 0

    def test_performance_100_calls(self):
        """100 extract_evidence calls should complete in < 0.5s."""
        data = {
            "contracts": [{"Market_and_Exchange_Names": "GOLD", "_mm_net_pct_oi": 5}]
        }
        start = time.monotonic()
        for _ in range(100):
            extract_evidence("cftc", data)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"100 calls took {elapsed:.2f}s"
