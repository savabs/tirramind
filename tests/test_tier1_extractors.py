"""Edge-case tests for Tier 1 extractors: internet_infrastructure, power_grid, defi_flows.

Covers: None/empty/wrong-type data, missing keys, partial dicts, zero-length lists,
boundary floats, each mode individually, direction/confidence sanity.
"""

from __future__ import annotations

import math

import pytest

from agent.convergence.evidence import Evidence
from agent.convergence.extractors import extract_evidence

# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════


def _ids(results: list[Evidence]) -> set[str]:
    return {e.signal_id for e in results}


def _by_id(results: list[Evidence], signal_id: str) -> Evidence | None:
    for e in results:
        if e.signal_id == signal_id:
            return e
    return None


# ══════════════════════════════════════════════════════════════
#  Internet Infrastructure — common edge cases
# ══════════════════════════════════════════════════════════════


class TestInternetInfrastructureEdgeCases:
    """Defensive handling for the internet_infrastructure extractor."""

    def test_none_data(self):
        assert extract_evidence("internet_infrastructure", None) == []

    def test_empty_dict(self):
        assert extract_evidence("internet_infrastructure", {}) == []

    def test_non_dict_data(self):
        assert extract_evidence("internet_infrastructure", "text") == []
        assert extract_evidence("internet_infrastructure", 42) == []
        assert extract_evidence("internet_infrastructure", [1, 2]) == []

    def test_unknown_mode(self):
        assert extract_evidence("internet_infrastructure", {"mode": "bogus"}) == []

    def test_mode_missing(self):
        # No mode key → returns empty
        assert extract_evidence("internet_infrastructure", {"alerts": []}) == []


# ══════════════════════════════════════════════════════════════
#  Internet Infrastructure — outages mode
# ══════════════════════════════════════════════════════════════


class TestInternetOutages:
    def test_empty_alerts_and_events(self):
        data = {"mode": "outages", "alerts": [], "events": [], "country": ""}
        results = extract_evidence("internet_infrastructure", data)
        ids = _ids(results)
        assert "internet.outage.critical_count" in ids
        assert "internet.outage.event_breadth" in ids
        # No events → no event_max_score
        assert "internet.outage.event_max_score" not in ids

    def test_critical_alerts_counted(self):
        data = {
            "mode": "outages",
            "alerts": [
                {"level": "critical", "country": "IR"},
                {"level": "critical", "country": "RU"},
                {"level": "warning", "country": "CN"},
                {"level": "critical", "country": "IR"},  # duplicate country
            ],
            "events": [],
        }
        results = extract_evidence("internet_infrastructure", data)
        crit = _by_id(results, "internet.outage.critical_count")
        assert crit is not None
        assert crit.value == 2.0  # IR + RU, deduped
        assert crit.direction == 1
        assert crit.category == "physical_disruption"

    def test_events_breadth_and_max_score(self):
        data = {
            "mode": "outages",
            "alerts": [],
            "events": [
                {"country": "EG", "score": 80.0},
                {"country": "SY", "score": 120.5},
                {"country": "EG", "score": 30.0},  # duplicate country
            ],
        }
        results = extract_evidence("internet_infrastructure", data)
        breadth = _by_id(results, "internet.outage.event_breadth")
        max_score = _by_id(results, "internet.outage.event_max_score")
        assert breadth is not None
        assert breadth.value == 2.0  # EG + SY
        assert max_score is not None
        assert max_score.value == 120.5
        assert max_score.direction == 1  # > 50

    def test_alerts_not_a_list(self):
        data = {"mode": "outages", "alerts": "bad", "events": []}
        results = extract_evidence("internet_infrastructure", data)
        # Should still produce event_breadth from empty events list
        assert _by_id(results, "internet.outage.event_breadth") is not None
        # Should NOT produce critical_count (alerts not iterable)
        assert _by_id(results, "internet.outage.critical_count") is None

    def test_events_with_missing_score(self):
        data = {
            "mode": "outages",
            "alerts": [],
            "events": [{"country": "XX"}],  # no score key
        }
        results = extract_evidence("internet_infrastructure", data)
        max_score = _by_id(results, "internet.outage.event_max_score")
        assert max_score is not None
        assert max_score.value == 0.0  # _safe_float default

    def test_non_dict_items_in_alerts(self):
        data = {
            "mode": "outages",
            "alerts": ["not_a_dict", 42, None],
            "events": [],
        }
        results = extract_evidence("internet_infrastructure", data)
        crit = _by_id(results, "internet.outage.critical_count")
        assert crit is not None
        assert crit.value == 0.0


# ══════════════════════════════════════════════════════════════
#  Internet Infrastructure — censorship mode
# ══════════════════════════════════════════════════════════════


class TestInternetCensorship:
    def test_basic_censorship(self):
        data = {
            "mode": "censorship",
            "country": "CN",
            "test": "web_connectivity",
            "rows": [
                {"confirmed": 5, "anomaly": 10, "ok": 85},
                {"confirmed": 3, "anomaly": 8, "ok": 89},
            ],
            "trend": "rising",
            "avg_rate": 0.15,
            "max_rate": 0.25,
        }
        results = extract_evidence("internet_infrastructure", data)
        ids = _ids(results)
        assert "internet.censorship.anomaly_rate" in ids
        assert "internet.censorship.trend_rising" in ids
        assert "internet.censorship.confirmed_total" in ids

        rate = _by_id(results, "internet.censorship.anomaly_rate")
        assert rate.value == 0.15
        assert rate.direction == 1  # > 0.1

        trend = _by_id(results, "internet.censorship.trend_rising")
        assert trend.value == 1.0
        assert trend.direction == 1

        confirmed = _by_id(results, "internet.censorship.confirmed_total")
        assert confirmed.value == 8.0  # 5 + 3

    def test_falling_trend(self):
        data = {
            "mode": "censorship",
            "trend": "falling",
            "avg_rate": 0.05,
            "max_rate": 0.08,
            "rows": [],
        }
        results = extract_evidence("internet_infrastructure", data)
        trend = _by_id(results, "internet.censorship.trend_rising")
        assert trend.value == 0.0
        assert trend.direction == -1

    def test_stable_trend(self):
        data = {
            "mode": "censorship",
            "trend": "stable",
            "avg_rate": 0.02,
            "max_rate": 0.03,
            "rows": [],
        }
        results = extract_evidence("internet_infrastructure", data)
        trend = _by_id(results, "internet.censorship.trend_rising")
        assert trend.direction == 0

    def test_rows_not_a_list(self):
        data = {
            "mode": "censorship",
            "trend": "stable",
            "avg_rate": 0.0,
            "max_rate": 0.0,
            "rows": "bad",
        }
        results = extract_evidence("internet_infrastructure", data)
        # Should produce anomaly_rate and trend but not confirmed_total
        assert _by_id(results, "internet.censorship.anomaly_rate") is not None
        assert _by_id(results, "internet.censorship.confirmed_total") is None

    def test_missing_avg_rate(self):
        data = {"mode": "censorship", "trend": "stable", "rows": []}
        results = extract_evidence("internet_infrastructure", data)
        rate = _by_id(results, "internet.censorship.anomaly_rate")
        assert rate.value == 0.0


# ══════════════════════════════════════════════════════════════
#  Internet Infrastructure — signals mode
# ══════════════════════════════════════════════════════════════


class TestInternetSignals:
    def test_critical_severity(self):
        data = {
            "mode": "signals",
            "current": 0.3,
            "severity": "critical",
            "drops": [{"time": "t1", "value": 0.2}],
        }
        results = extract_evidence("internet_infrastructure", data)
        conn = _by_id(results, "internet.signals.connectivity_level")
        assert conn is not None
        assert conn.value == 0.3
        assert conn.direction == -1
        assert conn.category == "physical_disruption"

        drops = _by_id(results, "internet.signals.drop_count")
        assert drops.value == 1.0

    def test_warning_severity(self):
        data = {"mode": "signals", "current": 0.7, "severity": "warning", "drops": []}
        results = extract_evidence("internet_infrastructure", data)
        conn = _by_id(results, "internet.signals.connectivity_level")
        assert conn.direction == -1

    def test_normal_severity(self):
        data = {"mode": "signals", "current": 0.95, "severity": "normal", "drops": []}
        results = extract_evidence("internet_infrastructure", data)
        conn = _by_id(results, "internet.signals.connectivity_level")
        assert conn.direction == 0

    def test_drops_not_a_list(self):
        data = {"mode": "signals", "current": 0.9, "severity": "normal", "drops": None}
        results = extract_evidence("internet_infrastructure", data)
        assert _by_id(results, "internet.signals.drop_count") is None
        assert _by_id(results, "internet.signals.connectivity_level") is not None

    def test_many_drops_direction(self):
        drops = [{"time": f"t{i}", "value": 0.4} for i in range(10)]
        data = {
            "mode": "signals",
            "current": 0.5,
            "severity": "warning",
            "drops": drops,
        }
        results = extract_evidence("internet_infrastructure", data)
        dc = _by_id(results, "internet.signals.drop_count")
        assert dc.value == 10.0
        assert dc.direction == 1  # > 3

    def test_default_current_when_missing(self):
        data = {"mode": "signals", "severity": "normal", "drops": []}
        results = extract_evidence("internet_infrastructure", data)
        conn = _by_id(results, "internet.signals.connectivity_level")
        assert conn.value == 1.0  # _safe_float default for "current"


# ══════════════════════════════════════════════════════════════
#  Internet Infrastructure — incidents mode
# ══════════════════════════════════════════════════════════════


class TestInternetIncidents:
    def test_basic_incidents(self):
        data = {
            "mode": "incidents",
            "incidents": [
                {"title": "A", "countries": ["IR", "SY"], "start": "2025-01-01"},
                {"title": "B", "countries": ["IR"], "start": "2025-02-01"},
                {
                    "title": "C",
                    "countries": ["RU", "BY", "UA", "PL", "DE", "FR"],
                    "start": "2025-03-01",
                },
            ],
            "country_frequency": {
                "IR": 2,
                "SY": 1,
                "RU": 1,
                "BY": 1,
                "UA": 1,
                "PL": 1,
                "DE": 1,
                "FR": 1,
            },
        }
        results = extract_evidence("internet_infrastructure", data)
        active = _by_id(results, "internet.incidents.active_count")
        breadth = _by_id(results, "internet.incidents.country_breadth")
        assert active.value == 3.0
        assert breadth.value == 8.0  # IR, SY, RU, BY, UA, PL, DE, FR
        assert breadth.direction == 1  # > 5

    def test_empty_incidents(self):
        data = {"mode": "incidents", "incidents": [], "country_frequency": {}}
        results = extract_evidence("internet_infrastructure", data)
        active = _by_id(results, "internet.incidents.active_count")
        assert active.value == 0.0

    def test_incidents_not_a_list(self):
        data = {"mode": "incidents", "incidents": "bad"}
        results = extract_evidence("internet_infrastructure", data)
        assert _by_id(results, "internet.incidents.active_count") is None

    def test_incident_with_non_list_countries(self):
        data = {
            "mode": "incidents",
            "incidents": [
                {"title": "A", "countries": "not_a_list", "start": "2025-01-01"},
            ],
        }
        results = extract_evidence("internet_infrastructure", data)
        breadth = _by_id(results, "internet.incidents.country_breadth")
        assert breadth.value == 0.0  # countries not iterable → empty set


# ══════════════════════════════════════════════════════════════
#  Power Grid — common edge cases
# ══════════════════════════════════════════════════════════════


class TestPowerGridEdgeCases:
    def test_none_data(self):
        assert extract_evidence("power_grid", None) == []

    def test_empty_dict(self):
        assert extract_evidence("power_grid", {}) == []

    def test_non_dict_data(self):
        assert extract_evidence("power_grid", "text") == []

    def test_unknown_mode(self):
        assert extract_evidence("power_grid", {"mode": "bogus"}) == []


# ══════════════════════════════════════════════════════════════
#  Power Grid — demand mode
# ══════════════════════════════════════════════════════════════


class TestPowerGridDemand:
    def test_basic_demand(self):
        data = {
            "total_peak_mw": 25000.0,
            "zones": [
                {"zone": "N.Y.C.", "peak_mw": 10000, "avg_mw": 8000, "readings": 288},
                {"zone": "LONGIL", "peak_mw": 5000, "avg_mw": 4000, "readings": 288},
            ],
            "date": "2025-06-15",
        }
        results = extract_evidence("power_grid", data)
        peak = _by_id(results, "power_grid.demand.total_peak_mw")
        zones = _by_id(results, "power_grid.demand.zone_count")
        assert peak.value == 25000.0
        assert peak.category == "physical_flow"
        assert zones.value == 2.0

    def test_no_zones_key(self):
        data = {"total_peak_mw": 1000.0}
        results = extract_evidence("power_grid", data)
        zones = _by_id(results, "power_grid.demand.zone_count")
        assert zones.value == 0.0

    def test_zones_not_a_list(self):
        data = {"total_peak_mw": 1000.0, "zones": "bad"}
        results = extract_evidence("power_grid", data)
        zones = _by_id(results, "power_grid.demand.zone_count")
        assert zones.value == 0.0


# ══════════════════════════════════════════════════════════════
#  Power Grid — fuel_mix mode
# ══════════════════════════════════════════════════════════════


class TestPowerGridFuelMix:
    def test_basic_fuel_mix(self):
        data = {
            "fuels": [
                {"fuel_type": "Natural Gas", "mw": 6000, "pct": 60.0},
                {"fuel_type": "Nuclear", "mw": 2000, "pct": 20.0},
                {"fuel_type": "Wind", "mw": 1000, "pct": 10.0},
                {"fuel_type": "Hydro", "mw": 1000, "pct": 10.0},
            ],
            "total_mw": 10000.0,
            "date": "2025-06-15",
        }
        results = extract_evidence("power_grid", data)
        gas = _by_id(results, "power_grid.fuel.gas_share_pct")
        renew = _by_id(results, "power_grid.fuel.renewable_share_pct")
        assert gas.value == 60.0
        assert gas.direction == 1  # > 50
        assert renew.value == 20.0  # wind 10 + hydro 10

    def test_zero_total_mw(self):
        data = {"fuels": [{"fuel_type": "Wind", "mw": 0}], "total_mw": 0.0}
        results = extract_evidence("power_grid", data)
        # Zero total → no fuel signals (division guard)
        assert _by_id(results, "power_grid.fuel.gas_share_pct") is None

    def test_fuels_not_a_list(self):
        data = {"fuels": "bad", "total_mw": 1000.0}
        results = extract_evidence("power_grid", data)
        assert _by_id(results, "power_grid.fuel.gas_share_pct") is None

    def test_gas_below_50_direction(self):
        data = {
            "fuels": [
                {"fuel_type": "Natural Gas", "mw": 3000},
                {"fuel_type": "Nuclear", "mw": 7000},
            ],
            "total_mw": 10000.0,
        }
        results = extract_evidence("power_grid", data)
        gas = _by_id(results, "power_grid.fuel.gas_share_pct")
        assert gas.direction == 0  # 30% < 50


# ══════════════════════════════════════════════════════════════
#  Power Grid — pricing mode
# ══════════════════════════════════════════════════════════════


class TestPowerGridPricing:
    def test_basic_pricing(self):
        data = {
            "stressed_zones": ["N.Y.C.", "LONGIL"],
            "zones": [
                {"zone": "N.Y.C.", "da_lbmp": 45.50, "rt_lbmp": 55.50, "spread": 10.0},
                {"zone": "LONGIL", "da_lbmp": 42.00, "rt_lbmp": 35.00, "spread": -7.0},
                {"zone": "CAPITL", "da_lbmp": 38.00, "rt_lbmp": 39.00, "spread": 1.0},
            ],
            "date": "2025-06-15",
        }
        results = extract_evidence("power_grid", data)
        stressed = _by_id(results, "power_grid.pricing.stressed_zone_count")
        max_spread = _by_id(results, "power_grid.pricing.max_spread")
        avg_da = _by_id(results, "power_grid.pricing.avg_da_lbmp")

        assert stressed.value == 2.0
        assert stressed.direction == 1
        assert max_spread.value == 10.0  # abs(10.0)
        assert max_spread.direction == 1  # > 5
        assert abs(avg_da.value - 41.83) < 0.01

    def test_no_stressed_zones(self):
        data = {
            "stressed_zones": [],
            "zones": [{"zone": "X", "da_lbmp": 30.0, "spread": 1.0}],
        }
        results = extract_evidence("power_grid", data)
        stressed = _by_id(results, "power_grid.pricing.stressed_zone_count")
        assert stressed.value == 0.0
        assert stressed.direction == 0

    def test_zones_with_none_spread(self):
        data = {
            "stressed_zones": [],
            "zones": [
                {"zone": "X", "da_lbmp": 30.0, "spread": None},
            ],
        }
        results = extract_evidence("power_grid", data)
        # spread is None → filtered out → no max_spread
        assert _by_id(results, "power_grid.pricing.max_spread") is None

    def test_zones_not_a_list(self):
        data = {"stressed_zones": ["X"], "zones": "bad"}
        results = extract_evidence("power_grid", data)
        stressed = _by_id(results, "power_grid.pricing.stressed_zone_count")
        assert stressed.value == 1.0
        assert _by_id(results, "power_grid.pricing.max_spread") is None


# ══════════════════════════════════════════════════════════════
#  Power Grid — forecast mode
# ══════════════════════════════════════════════════════════════


class TestPowerGridForecast:
    def test_basic_forecast(self):
        data = {
            "persistent_deviation_zones": ["CAPITL", "WEST"],
            "zones": [
                {
                    "zone": "CAPITL",
                    "hours": 24,
                    "avg_deviation_pct": 4.5,
                    "significant_deviations": 8,
                },
                {
                    "zone": "WEST",
                    "hours": 24,
                    "avg_deviation_pct": 3.2,
                    "significant_deviations": 3,
                },
                {
                    "zone": "N.Y.C.",
                    "hours": 24,
                    "avg_deviation_pct": 1.0,
                    "significant_deviations": 0,
                },
            ],
        }
        results = extract_evidence("power_grid", data)
        persist = _by_id(results, "power_grid.forecast.persistent_deviation_count")
        max_sig = _by_id(results, "power_grid.forecast.max_significant_deviations")

        assert persist.value == 2.0
        assert persist.direction == 1
        assert max_sig.value == 8.0
        assert max_sig.direction == 1  # > 5

    def test_no_persistent_deviations(self):
        data = {
            "persistent_deviation_zones": [],
            "zones": [{"zone": "X", "significant_deviations": 1}],
        }
        results = extract_evidence("power_grid", data)
        persist = _by_id(results, "power_grid.forecast.persistent_deviation_count")
        assert persist.value == 0.0
        assert persist.direction == 0

    def test_zones_missing(self):
        data = {"persistent_deviation_zones": []}
        results = extract_evidence("power_grid", data)
        assert _by_id(results, "power_grid.forecast.max_significant_deviations") is None


# ══════════════════════════════════════════════════════════════
#  DeFi Flows — common edge cases
# ══════════════════════════════════════════════════════════════


class TestDefiFlowsEdgeCases:
    def test_none_data(self):
        assert extract_evidence("defi_flows", None) == []

    def test_empty_dict(self):
        assert extract_evidence("defi_flows", {}) == []

    def test_non_dict_data(self):
        assert extract_evidence("defi_flows", 99) == []

    def test_unknown_mode(self):
        assert extract_evidence("defi_flows", {"mode": "bogus"}) == []


# ══════════════════════════════════════════════════════════════
#  DeFi Flows — tvl mode
# ══════════════════════════════════════════════════════════════


class TestDefiTVL:
    def test_basic_tvl(self):
        data = {
            "total_tvl": 50_000_000_000.0,
            "protocols": [
                {"name": "Lido", "tvl_usd": 20_000_000_000.0, "change_1d_pct": -2.0},
                {"name": "Aave", "tvl_usd": 10_000_000_000.0, "change_1d_pct": -6.0},
                {"name": "SmallFi", "tvl_usd": 500_000.0, "change_1d_pct": -8.0},
            ],
            "count": 3,
        }
        results = extract_evidence("defi_flows", data)
        total = _by_id(results, "defi.tvl.total_usd")
        drawdown = _by_id(results, "defi.tvl.drawdown_breadth")
        conc = _by_id(results, "defi.tvl.top_concentration_pct")

        assert total.value == 50_000_000_000.0
        assert drawdown.value == 2.0  # Aave (-6%) + SmallFi (-8%)
        assert conc.value == 40.0  # 20B / 50B
        assert conc.direction == 1  # > 30%

    def test_no_drawdowns(self):
        data = {
            "total_tvl": 1000.0,
            "protocols": [
                {"name": "X", "tvl_usd": 500.0, "change_1d_pct": 2.0},
            ],
            "count": 1,
        }
        results = extract_evidence("defi_flows", data)
        drawdown = _by_id(results, "defi.tvl.drawdown_breadth")
        assert drawdown.value == 0.0

    def test_zero_total_tvl_concentration(self):
        data = {"total_tvl": 0.0, "protocols": [{"tvl_usd": 0.0}], "count": 1}
        results = extract_evidence("defi_flows", data)
        # Zero total → no concentration signal
        assert _by_id(results, "defi.tvl.top_concentration_pct") is None

    def test_empty_protocols(self):
        data = {"total_tvl": 1000.0, "protocols": [], "count": 0}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.tvl.drawdown_breadth") is not None
        assert _by_id(results, "defi.tvl.top_concentration_pct") is None  # empty list

    def test_protocols_not_a_list(self):
        data = {"total_tvl": 1000.0, "protocols": "bad"}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.tvl.total_usd") is not None
        assert _by_id(results, "defi.tvl.drawdown_breadth") is None

    def test_change_1d_pct_none(self):
        data = {
            "total_tvl": 1000.0,
            "protocols": [
                {"name": "X", "tvl_usd": 1000.0, "change_1d_pct": None},
            ],
        }
        results = extract_evidence("defi_flows", data)
        drawdown = _by_id(results, "defi.tvl.drawdown_breadth")
        assert drawdown.value == 0.0


# ══════════════════════════════════════════════════════════════
#  DeFi Flows — stablecoins mode
# ══════════════════════════════════════════════════════════════


class TestDefiStablecoins:
    def test_basic_stablecoins(self):
        data = {
            "total_supply": 150_000_000_000.0,
            "stablecoins": [
                {
                    "name": "Tether",
                    "symbol": "USDT",
                    "circulating_usd": 100_000_000_000.0,
                },
                {"name": "USDC", "symbol": "USDC", "circulating_usd": 30_000_000_000.0},
            ],
            "count": 2,
        }
        results = extract_evidence("defi_flows", data)
        supply = _by_id(results, "defi.stablecoin.total_supply")
        share = _by_id(results, "defi.stablecoin.top_share_pct")

        assert supply.value == 150_000_000_000.0
        assert abs(share.value - 66.67) < 0.01
        assert share.direction == 1  # > 60%

    def test_zero_total_supply(self):
        data = {"total_supply": 0.0, "stablecoins": [{"circulating_usd": 0.0}]}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.stablecoin.top_share_pct") is None

    def test_empty_stablecoins_list(self):
        data = {"total_supply": 1000.0, "stablecoins": []}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.stablecoin.top_share_pct") is None

    def test_stablecoins_not_a_list(self):
        data = {"total_supply": 1000.0, "stablecoins": 42}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.stablecoin.total_supply") is not None
        assert _by_id(results, "defi.stablecoin.top_share_pct") is None


# ══════════════════════════════════════════════════════════════
#  DeFi Flows — dex_volume mode
# ══════════════════════════════════════════════════════════════


class TestDefiDexVolume:
    def test_basic_dex_volume(self):
        data = {
            "total_volume_24h": 5_000_000_000.0,
            "dexes": [
                {"name": "Uniswap", "volume_24h_usd": 2e9, "change_1d_pct": 10.0},
                {"name": "PancakeSwap", "volume_24h_usd": 1e9, "change_1d_pct": 60.0},
                {"name": "SushiSwap", "volume_24h_usd": 0.5e9, "change_1d_pct": 80.0},
            ],
            "count": 3,
        }
        results = extract_evidence("defi_flows", data)
        vol = _by_id(results, "defi.dex.total_volume_24h")
        panic = _by_id(results, "defi.dex.panic_breadth")

        assert vol.value == 5_000_000_000.0
        assert panic.value == 2.0  # PancakeSwap 60% + SushiSwap 80%

    def test_no_panic(self):
        data = {
            "total_volume_24h": 1000.0,
            "dexes": [{"name": "X", "change_1d_pct": 10.0}],
        }
        results = extract_evidence("defi_flows", data)
        panic = _by_id(results, "defi.dex.panic_breadth")
        assert panic.value == 0.0
        assert panic.direction == 0

    def test_dexes_not_a_list(self):
        data = {"total_volume_24h": 1000.0, "dexes": None}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.dex.total_volume_24h") is not None
        assert _by_id(results, "defi.dex.panic_breadth") is None

    def test_change_1d_pct_none(self):
        data = {
            "total_volume_24h": 1000.0,
            "dexes": [{"name": "X", "change_1d_pct": None}],
        }
        results = extract_evidence("defi_flows", data)
        panic = _by_id(results, "defi.dex.panic_breadth")
        assert panic.value == 0.0


# ══════════════════════════════════════════════════════════════
#  DeFi Flows — chain mode
# ══════════════════════════════════════════════════════════════


class TestDefiChain:
    def test_basic_chain(self):
        data = {
            "grand_total_tvl": 80_000_000_000.0,
            "chains": [
                {
                    "chain": "Ethereum",
                    "tvl_usd": 50_000_000_000.0,
                    "protocol_count": 500,
                },
                {"chain": "BSC", "tvl_usd": 10_000_000_000.0, "protocol_count": 200},
            ],
            "count": 2,
        }
        results = extract_evidence("defi_flows", data)
        total = _by_id(results, "defi.chain.total_tvl")
        conc = _by_id(results, "defi.chain.top_concentration_pct")

        assert total.value == 80_000_000_000.0
        assert conc.value == 62.5  # 50B / 80B
        assert conc.direction == 1  # > 50%

    def test_zero_grand_total(self):
        data = {"grand_total_tvl": 0.0, "chains": [{"tvl_usd": 0.0}]}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.chain.top_concentration_pct") is None

    def test_empty_chains(self):
        data = {"grand_total_tvl": 1000.0, "chains": []}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.chain.total_tvl") is not None
        assert _by_id(results, "defi.chain.top_concentration_pct") is None

    def test_chains_not_a_list(self):
        data = {"grand_total_tvl": 1000.0, "chains": "bad"}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.chain.total_tvl") is not None
        assert _by_id(results, "defi.chain.top_concentration_pct") is None


# ══════════════════════════════════════════════════════════════
#  Mode inference tests (tools don't include "mode" key)
# ══════════════════════════════════════════════════════════════


class TestModeInference:
    """PowerGrid and DeFi tools don't include a 'mode' key — verify inference."""

    def test_power_grid_infers_demand(self):
        data = {"total_peak_mw": 1000, "zones": []}
        results = extract_evidence("power_grid", data)
        assert _by_id(results, "power_grid.demand.total_peak_mw") is not None

    def test_power_grid_infers_fuel_mix(self):
        data = {"fuels": [{"fuel_type": "Natural Gas", "mw": 500}], "total_mw": 1000}
        results = extract_evidence("power_grid", data)
        assert _by_id(results, "power_grid.fuel.gas_share_pct") is not None

    def test_power_grid_infers_pricing(self):
        data = {"stressed_zones": [], "zones": []}
        results = extract_evidence("power_grid", data)
        assert _by_id(results, "power_grid.pricing.stressed_zone_count") is not None

    def test_power_grid_infers_forecast(self):
        data = {"persistent_deviation_zones": [], "zones": []}
        results = extract_evidence("power_grid", data)
        assert _by_id(results, "power_grid.forecast.persistent_deviation_count") is not None

    def test_defi_infers_tvl(self):
        data = {"protocols": [], "total_tvl": 0, "count": 0}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.tvl.total_usd") is not None

    def test_defi_infers_stablecoins(self):
        data = {"stablecoins": [], "total_supply": 0}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.stablecoin.total_supply") is not None

    def test_defi_infers_dex_volume(self):
        data = {"dexes": [], "total_volume_24h": 0}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.dex.total_volume_24h") is not None

    def test_defi_infers_chain(self):
        data = {"chains": [], "grand_total_tvl": 0}
        results = extract_evidence("defi_flows", data)
        assert _by_id(results, "defi.chain.total_tvl") is not None


# ══════════════════════════════════════════════════════════════
#  Evidence field sanity
# ══════════════════════════════════════════════════════════════


class TestEvidenceSanity:
    """All produced Evidence objects have valid field ranges."""

    @pytest.fixture()
    def all_results(self):
        """Collect evidence from representative data across all three tools."""
        inet_outages = extract_evidence(
            "internet_infrastructure",
            {
                "mode": "outages",
                "alerts": [{"level": "critical", "country": "IR"}],
                "events": [{"country": "SY", "score": 75.0}],
            },
        )
        inet_censor = extract_evidence(
            "internet_infrastructure",
            {
                "mode": "censorship",
                "avg_rate": 0.3,
                "trend": "rising",
                "rows": [{"confirmed": 10}],
            },
        )
        inet_signals = extract_evidence(
            "internet_infrastructure",
            {
                "mode": "signals",
                "current": 0.4,
                "severity": "critical",
                "drops": [{"time": "t1"}],
            },
        )
        inet_incidents = extract_evidence(
            "internet_infrastructure",
            {
                "mode": "incidents",
                "incidents": [{"title": "A", "countries": ["US"]}],
            },
        )
        power = extract_evidence(
            "power_grid",
            {
                "total_peak_mw": 25000,
                "zones": [{"zone": "X"}],
            },
        )
        defi = extract_evidence(
            "defi_flows",
            {
                "total_tvl": 50e9,
                "protocols": [{"tvl_usd": 20e9, "change_1d_pct": -10.0}],
            },
        )
        return inet_outages + inet_censor + inet_signals + inet_incidents + power + defi

    def test_all_are_evidence(self, all_results):
        for e in all_results:
            assert isinstance(e, Evidence)

    def test_direction_in_range(self, all_results):
        for e in all_results:
            assert e.direction in (-1, 0, 1), f"{e.signal_id}: direction={e.direction}"

    def test_confidence_in_range(self, all_results):
        for e in all_results:
            assert 0.0 <= e.confidence <= 1.0, f"{e.signal_id}: confidence={e.confidence}"

    def test_value_is_finite(self, all_results):
        for e in all_results:
            assert math.isfinite(e.value), f"{e.signal_id}: value={e.value}"

    def test_ttl_positive(self, all_results):
        for e in all_results:
            assert e.ttl > 0, f"{e.signal_id}: ttl={e.ttl}"

    def test_signal_ids_are_dotted(self, all_results):
        for e in all_results:
            assert "." in e.signal_id, f"signal_id missing dot: {e.signal_id}"

    def test_categories_are_valid(self, all_results):
        valid = {
            "physical_flow",
            "physical_disruption",
            "financial_stress",
            "monetary_policy",
            "regulatory_action",
            "behavioral_intent",
            "positioning",
            "macro_momentum",
            "biological",
            "geopolitical",
            "supply_chain",
        }
        for e in all_results:
            assert e.category in valid, f"{e.signal_id}: category={e.category}"
