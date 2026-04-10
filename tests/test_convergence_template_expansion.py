"""Tests for 10 advanced convergence templates (#13–#22).

Covers:
- Template structural integrity (names, step counts, min_match, regexes, categories)
- All-22-template uniqueness invariants
- Synthetic evidence matching for each new template
- Direction constraint testing (especially smart_money_divergence)
- Temporal ordering across long windows (stealth_accumulation 90d)
- Partial match scoring
- Cross-template regression (existing 12 templates unchanged)
"""

from __future__ import annotations

import re
import time

import pytest

from agent.convergence.evidence import Evidence
from agent.convergence.graph import ConvergenceClique
from agent.convergence.taxonomy import CATEGORIES
from agent.convergence.templates import (
    TEMPLATE_LIBRARY,
    CausalTemplate,
    TemplateStep,
    match_template,
)

# ── Helpers ────────────────────────────────────────────────────

_DAY = 86_400  # seconds


def _ev(
    signal_id: str,
    category: str,
    timestamp: float = 1_000_000.0,
    direction: int = 1,
    value: float = 1.0,
) -> Evidence:
    return Evidence(
        source="test",
        signal_id=signal_id,
        timestamp=timestamp,
        value=value,
        direction=direction,
        confidence=0.8,
        category=category,
        tags=(),
        ttl=86_400,
    )


def _clique(signals: list[str], categories: list[str]) -> ConvergenceClique:
    edges = []
    for i, a in enumerate(signals):
        for b in signals[i + 1 :]:
            edges.append((a, b, 1.0))
    return ConvergenceClique(
        signals=sorted(signals),
        categories=sorted(set(categories)),
        edges=edges,
        score=0.5,
        p_values=[0.01] * len(edges),
    )


def _template_by_name(name: str) -> CausalTemplate:
    """Find template by name or fail."""
    for t in TEMPLATE_LIBRARY:
        if t.name == name:
            return t
    raise ValueError(f"Template {name!r} not found in TEMPLATE_LIBRARY")


# ── New template names for parametrised tests ──────────────────

NEW_TEMPLATE_NAMES = [
    "silent_nationalization",
    "defi_canary",
    "pandemic_physical_evidence",
    "capital_flight_crypto",
    "infrastructure_decay_cascade",
    "commodity_hoarding",
    "smart_money_divergence",
    "sanctions_evasion_network",
    "carry_trade_unwind",
    "stealth_accumulation",
]


# ═══════════════════════════════════════════════════════════════
# Structural integrity tests
# ═══════════════════════════════════════════════════════════════


class TestLibraryStructure:
    """Invariants that must hold across the full 50-template library."""

    def test_total_count(self):
        assert len(TEMPLATE_LIBRARY) == 50

    def test_unique_names(self):
        names = [t.name for t in TEMPLATE_LIBRARY]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_all_new_templates_present(self):
        names = {t.name for t in TEMPLATE_LIBRARY}
        for n in NEW_TEMPLATE_NAMES:
            assert n in names, f"Missing template: {n}"

    def test_existing_12_unchanged(self):
        expected = [
            "supply_chain_disruption",
            "monetary_policy_shift",
            "geopolitical_escalation",
            "health_crisis",
            "agricultural_shock",
            "energy_crisis",
            "credit_stress_cascade",
            "tech_disruption",
            "labor_market_shift",
            "trade_war_escalation",
            "construction_cycle",
            "digital_infrastructure_crisis",
        ]
        for i, name in enumerate(expected):
            assert TEMPLATE_LIBRARY[i].name == name, (
                f"Template #{i} expected {name}, got {TEMPLATE_LIBRARY[i].name}"
            )


class TestNewTemplateProperties:
    """Property tests on each new template."""

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_has_steps(self, name: str):
        t = _template_by_name(name)
        assert len(t.steps) >= 4, f"{name} has only {len(t.steps)} steps"

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_step_zero_within_days_zero(self, name: str):
        t = _template_by_name(name)
        assert t.steps[0].within_days == 0, (
            f"{name} step 0 within_days={t.steps[0].within_days}, expected 0"
        )

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_within_days_non_decreasing(self, name: str):
        t = _template_by_name(name)
        days = [s.within_days for s in t.steps]
        for i in range(1, len(days)):
            assert days[i] >= days[i - 1], (
                f"{name} within_days not non-decreasing at step {i}: "
                f"{days[i - 1]} → {days[i]}"
            )

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_valid_categories(self, name: str):
        t = _template_by_name(name)
        for idx, step in enumerate(t.steps):
            for cat in step.category_pattern.split("|"):
                assert cat in CATEGORIES, (
                    f"{name} step {idx} has invalid category {cat!r}"
                )

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_regex_compiles(self, name: str):
        t = _template_by_name(name)
        for idx, step in enumerate(t.steps):
            try:
                re.compile(step.signal_pattern)
            except re.error as e:
                pytest.fail(
                    f"{name} step {idx} has invalid regex "
                    f"{step.signal_pattern!r}: {e}"
                )

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_direction_valid(self, name: str):
        t = _template_by_name(name)
        for idx, step in enumerate(t.steps):
            assert step.direction in (+1, -1, None), (
                f"{name} step {idx} direction={step.direction}"
            )

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_description_nonempty(self, name: str):
        t = _template_by_name(name)
        assert len(t.description.strip()) > 20, (
            f"{name} description too short"
        )


class TestMinMatchValues:
    """Verify min_match settings match spec."""

    def test_five_step_templates_min_match_3(self):
        five_step = [
            n for n in NEW_TEMPLATE_NAMES
            if len(_template_by_name(n).steps) == 5
        ]
        for n in five_step:
            t = _template_by_name(n)
            assert t.min_match == 3, (
                f"{n} has {len(t.steps)} steps but min_match={t.min_match}"
            )

    def test_carry_trade_unwind_six_steps_min_match_4(self):
        t = _template_by_name("carry_trade_unwind")
        assert len(t.steps) == 6
        assert t.min_match == 4
        assert t.effective_min_match == 4

    def test_effective_min_match_all_new(self):
        for n in NEW_TEMPLATE_NAMES:
            t = _template_by_name(n)
            assert t.effective_min_match >= 3, (
                f"{n} effective_min_match={t.effective_min_match}"
            )


# ═══════════════════════════════════════════════════════════════
# Category span tests — each new template should be cross-category
# ═══════════════════════════════════════════════════════════════


class TestCategorySpan:
    """Each new template should span at least 3 distinct categories."""

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_min_3_categories(self, name: str):
        t = _template_by_name(name)
        cats = set()
        for step in t.steps:
            cats.update(step.category_pattern.split("|"))
        assert len(cats) >= 3, (
            f"{name} only spans {len(cats)} categories: {sorted(cats)}"
        )


# ═══════════════════════════════════════════════════════════════
# Synthetic matching tests — each template gets a perfect-match test
# ═══════════════════════════════════════════════════════════════


class TestSilentNationalization:
    def test_full_match(self):
        t = _template_by_name("silent_nationalization")
        t0 = 1_000_000.0
        sigs = [
            "lobbying.energy.spend",
            "satellite.fire.hotspot_count",
            "regulatory_gazette.ng.export_ban",
            "insider.XYZ.cluster_sales",
            "ais.tanker.vessel_count",
        ]
        cats = [
            "behavioral_intent",
            "physical_disruption",
            "regulatory_action",
            "positioning",
            "physical_flow",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 10 * _DAY, +1),
            _ev(sigs[2], cats[2], t0 + 18 * _DAY, +1),
            _ev(sigs[3], cats[3], t0 + 25 * _DAY, -1),
            _ev(sigs[4], cats[4], t0 + 40 * _DAY),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.matched_steps == 5
        assert result.lead_signal == sigs[0]
        assert result.temporal_order_valid is True

    def test_partial_match_3_of_5(self):
        t = _template_by_name("silent_nationalization")
        t0 = 1_000_000.0
        sigs = ["lobbying.energy.spend", "satellite.fire.hotspot_count",
                "regulatory_gazette.ng.export_ban"]
        cats = ["behavioral_intent", "physical_disruption", "regulatory_action"]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 10 * _DAY, +1),
            _ev(sigs[2], cats[2], t0 + 18 * _DAY, +1),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == pytest.approx(3 / 5)
        assert result.matched_steps == 3


class TestDefiCanary:
    def test_full_match(self):
        t = _template_by_name("defi_canary")
        t0 = 1_000_000.0
        sigs = [
            "defi.total_tvl_delta",
            "whale_alert.btc.large_tx",
            "wikipedia.bitcoin_crash.views",
            "polymarket.crypto_default.probability",
            "bankruptcy.crypto_exchange.filing",
        ]
        cats = [
            "financial_stress",
            "financial_stress",
            "behavioral_intent",
            "positioning",
            "financial_stress",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 2 * _DAY),
            _ev(sigs[2], cats[2], t0 + 5 * _DAY, +1),
            _ev(sigs[3], cats[3], t0 + 12 * _DAY),
            _ev(sigs[4], cats[4], t0 + 25 * _DAY, +1),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.temporal_order_valid is True

    def test_wrong_trigger_direction(self):
        """DeFi trigger must be direction=+1 (stress up)."""
        t = _template_by_name("defi_canary")
        t0 = 1_000_000.0
        timeline = [
            _ev("defi.total_tvl_delta", "financial_stress", t0, -1),
        ]
        clique = _clique(["defi.total_tvl_delta"], ["financial_stress"])
        result = match_template(clique, timeline, t)
        # Direction mismatch means step 0 doesn't match
        assert result.matched_steps == 0


class TestPandemicPhysicalEvidence:
    def test_full_match(self):
        t = _template_by_name("pandemic_physical_evidence")
        t0 = 1_000_000.0
        sigs = [
            "disease.covid.case_count",
            "satellite.events.active_count",
            "wikipedia.pandemic.views",
            "transport.container.throughput",
            "cftc.crude_oil.mm_net_long",
        ]
        cats = [
            "biological",
            "physical_disruption",
            "behavioral_intent",
            "physical_flow",
            "positioning",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(sigs[2], cats[2], t0 + 10 * _DAY, +1),
            _ev(sigs[3], cats[3], t0 + 18 * _DAY, -1),
            _ev(sigs[4], cats[4], t0 + 28 * _DAY),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.matched_steps == 5


class TestCapitalFlightCrypto:
    def test_full_match(self):
        t = _template_by_name("capital_flight_crypto")
        t0 = 1_000_000.0
        sigs = [
            "political_risk.turkey.instability",
            "capital_flows.em.outflow",
            "defi.stablecoin.minting",
            "sovereign_debt.turkey.spread",
            "central_bank.turkey.rate_change",
        ]
        cats = [
            "geopolitical",
            "monetary_policy",
            "financial_stress",
            "financial_stress",
            "monetary_policy",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(sigs[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(sigs[3], cats[3], t0 + 18 * _DAY, +1),
            _ev(sigs[4], cats[4], t0 + 28 * _DAY),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.temporal_order_valid is True


class TestInfrastructureDecayCascade:
    def test_full_match_60_day_window(self):
        t = _template_by_name("infrastructure_decay_cascade")
        t0 = 1_000_000.0
        sigs = [
            "power_grid.frequency.deviation",
            "internet.outage.count",
            "building_permits.us.new_private",
            "jobs.construction.postings",
            "consumer_sentiment.umich.index",
        ]
        cats = [
            "physical_flow",
            "physical_disruption",
            "macro_momentum",
            "behavioral_intent",
            "macro_momentum",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 12 * _DAY, +1),
            _ev(sigs[2], cats[2], t0 + 28 * _DAY, -1),
            _ev(sigs[3], cats[3], t0 + 42 * _DAY, -1),
            _ev(sigs[4], cats[4], t0 + 55 * _DAY, -1),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.temporal_order_valid is True

    def test_temporal_order_invalid_beyond_60_days(self):
        """Step 4 fires at day 65 — exceeds within_days=60."""
        t = _template_by_name("infrastructure_decay_cascade")
        t0 = 1_000_000.0
        sigs = [
            "power_grid.frequency.deviation",
            "consumer_sentiment.umich.index",
        ]
        cats = ["physical_flow", "macro_momentum"]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 65 * _DAY, -1),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        # The step still matches (category + regex) but temporal_order_valid
        # should be False because 65 > 60
        if result.matched_steps >= 2:
            assert result.temporal_order_valid is False


class TestCommodityHoarding:
    def test_full_match(self):
        t = _template_by_name("commodity_hoarding")
        t0 = 1_000_000.0
        sigs = [
            "weather.drought.intensity",
            "satellite.vegetation.anomaly_pct",
            "ais.grain.vessel_count",
            "cftc.wheat.mm_net_long",
            "food_security.fao.alert_count",
        ]
        cats = [
            "physical_disruption",
            "supply_chain",
            "physical_flow",
            "positioning",
            "biological",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(sigs[2], cats[2], t0 + 12 * _DAY),
            _ev(sigs[3], cats[3], t0 + 18 * _DAY, +1),
            _ev(sigs[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.matched_steps == 5


class TestSmartMoneyDivergence:
    """Specifically tests opposed direction constraints."""

    def test_full_match_with_diverging_directions(self):
        t = _template_by_name("smart_money_divergence")
        t0 = 1_000_000.0
        sigs = [
            "wikipedia.tesla.views",
            "polymarket.tesla.probability",
            "form144.tesla.ceo_sale",
            "finra.TSLA.short_ratio",
            "defi.whale.exit_volume",
        ]
        cats = [
            "behavioral_intent",
            "positioning",
            "positioning",
            "positioning",
            "financial_stress",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),        # retail euphoria ↑
            _ev(sigs[1], cats[1], t0 + 5 * _DAY, +1),  # public bullish ↑
            _ev(sigs[2], cats[2], t0 + 12 * _DAY, -1),  # insiders selling ↓
            _ev(sigs[3], cats[3], t0 + 18 * _DAY, +1),  # shorts increasing ↑
            _ev(sigs[4], cats[4], t0 + 28 * _DAY, +1),  # whale exits ↑
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.temporal_order_valid is True

    def test_insider_wrong_direction_no_match(self):
        """Step 2 requires direction=-1 (insiders selling). If +1, no match."""
        t = _template_by_name("smart_money_divergence")
        t0 = 1_000_000.0
        # Just the insider step with wrong direction
        sigs = [
            "wikipedia.tesla.views",
            "form144.tesla.ceo_sale",
        ]
        cats = ["behavioral_intent", "positioning"]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 12 * _DAY, +1),  # wrong: should be -1
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        # step 0 matches, step 2 doesn't match due to direction
        assert result.matched_steps < 2 or result.match_score < 0.5


class TestSanctionsEvasionNetwork:
    def test_full_match(self):
        t = _template_by_name("sanctions_evasion_network")
        t0 = 1_000_000.0
        sigs = [
            "sanctions.iran.additions",
            "ais.strait_hormuz.dark_ships",
            "cert_trans.iranian_bank.new_cert",
            "defi.stablecoin.circumvention",
            "gdelt.iran.event_intensity",
        ]
        cats = [
            "regulatory_action",
            "physical_flow",
            "behavioral_intent",
            "financial_stress",
            "geopolitical",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 5 * _DAY),
            _ev(sigs[2], cats[2], t0 + 12 * _DAY),
            _ev(sigs[3], cats[3], t0 + 18 * _DAY, +1),
            _ev(sigs[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.temporal_order_valid is True

    def test_step2_dns_match_physical_disruption(self):
        """Step 2 category_pattern is 'behavioral_intent|physical_disruption'."""
        t = _template_by_name("sanctions_evasion_network")
        t0 = 1_000_000.0
        sigs = [
            "sanctions.iran.additions",
            "dns.sanctioned.zone_change",
        ]
        cats = ["regulatory_action", "physical_disruption"]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 12 * _DAY),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.matched_steps >= 2


class TestCarryTradeUnwind:
    def test_full_match_6_steps(self):
        t = _template_by_name("carry_trade_unwind")
        t0 = 1_000_000.0
        sigs = [
            "central_bank.boj.rate_change",
            "capital_flows.em.outflow",
            "defi.yield.liquidations",
            "sovereign_debt.em.spread",
            "cftc.yen.mm_net_long",
            "pmi.global.manufacturing",
        ]
        cats = [
            "monetary_policy",
            "monetary_policy",
            "financial_stress",
            "financial_stress",
            "positioning",
            "macro_momentum",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0),
            _ev(sigs[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(sigs[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(sigs[3], cats[3], t0 + 18 * _DAY, +1),
            _ev(sigs[4], cats[4], t0 + 28 * _DAY, -1),
            _ev(sigs[5], cats[5], t0 + 42 * _DAY, -1),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.matched_steps == 6
        assert result.total_steps == 6

    def test_min_match_4_sufficient(self):
        """4 of 6 steps matching should exceed effective_min_match."""
        t = _template_by_name("carry_trade_unwind")
        assert t.effective_min_match == 4
        t0 = 1_000_000.0
        sigs = [
            "central_bank.boj.rate_change",
            "capital_flows.em.outflow",
            "defi.yield.liquidations",
            "sovereign_debt.em.spread",
        ]
        cats = [
            "monetary_policy",
            "monetary_policy",
            "financial_stress",
            "financial_stress",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0),
            _ev(sigs[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(sigs[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(sigs[3], cats[3], t0 + 18 * _DAY, +1),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.matched_steps == 4
        assert result.match_score == pytest.approx(4 / 6)


class TestStealthAccumulation:
    def test_full_match_90_day_window(self):
        t = _template_by_name("stealth_accumulation")
        t0 = 1_000_000.0
        sigs = [
            "wikipedia.lithium.views",
            "lobbying.mining.spend",
            "patent.battery.filings",
            "cftc.lithium.mm_net_long",
            "regulatory_gazette.mining.approval",
        ]
        cats = [
            "behavioral_intent",
            "behavioral_intent",
            "behavioral_intent",
            "positioning",
            "regulatory_action",
        ]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 25 * _DAY, +1),
            _ev(sigs[2], cats[2], t0 + 40 * _DAY),
            _ev(sigs[3], cats[3], t0 + 55 * _DAY, +1),
            _ev(sigs[4], cats[4], t0 + 85 * _DAY),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score == 1.0
        assert result.temporal_order_valid is True

    def test_temporal_exceeds_90_days(self):
        """Regulatory step at day 95 exceeds within_days=90."""
        t = _template_by_name("stealth_accumulation")
        t0 = 1_000_000.0
        sigs = [
            "wikipedia.lithium.views",
            "regulatory_gazette.mining.approval",
        ]
        cats = ["behavioral_intent", "regulatory_action"]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 95 * _DAY),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        if result.matched_steps >= 2:
            assert result.temporal_order_valid is False

    def test_drug_regulatory_also_matches_step4(self):
        """Step 4 regex includes drug_regulatory."""
        t = _template_by_name("stealth_accumulation")
        t0 = 1_000_000.0
        sigs = [
            "wikipedia.pharma.views",
            "drug_regulatory.fda.approval",
        ]
        cats = ["behavioral_intent", "regulatory_action"]
        timeline = [
            _ev(sigs[0], cats[0], t0, +1),
            _ev(sigs[1], cats[1], t0 + 80 * _DAY),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.matched_steps >= 2


# ═══════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_clique_returns_zero(self):
        for name in NEW_TEMPLATE_NAMES:
            t = _template_by_name(name)
            clique = _clique([], [])
            result = match_template(clique, [], t)
            assert result.match_score == 0.0

    def test_single_signal_partial(self):
        """A clique with one signal can match at most one step."""
        t = _template_by_name("defi_canary")
        t0 = 1_000_000.0
        timeline = [_ev("defi.total_tvl", "financial_stress", t0, +1)]
        clique = _clique(["defi.total_tvl"], ["financial_stress"])
        result = match_template(clique, timeline, t)
        assert result.matched_steps <= 2  # might match step 0 and step 4
        assert result.match_score <= 0.5

    def test_wrong_category_no_match(self):
        """Signal matches regex but wrong category → no match."""
        t = _template_by_name("commodity_hoarding")
        t0 = 1_000_000.0
        # weather.drought.intensity should be physical_disruption, not positioning
        timeline = [_ev("weather.drought.intensity", "positioning", t0, +1)]
        clique = _clique(["weather.drought.intensity"], ["positioning"])
        result = match_template(clique, timeline, t)
        assert result.matched_steps == 0

    def test_reverse_temporal_order_invalid(self):
        """Lag signals firing BEFORE trigger → temporal_order_valid=False."""
        t = _template_by_name("capital_flight_crypto")
        t0 = 1_000_000.0
        sigs = [
            "political_risk.test.instability",
            "capital_flows.test.outflow",
        ]
        cats = ["geopolitical", "monetary_policy"]
        timeline = [
            # Trigger fires AFTER the lag signal
            _ev(sigs[0], cats[0], t0 + 20 * _DAY, +1),  # trigger at day 20
            _ev(sigs[1], cats[1], t0, -1),  # lag at day 0 (before trigger!)
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        if result.matched_steps >= 2:
            assert result.temporal_order_valid is False

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_no_crash_with_unrelated_signals(self, name: str):
        """Templates should gracefully handle signals from unrelated domains."""
        t = _template_by_name(name)
        sigs = ["unrelated.foo.bar", "unrelated.baz.qux"]
        cats = ["positioning", "macro_momentum"]
        timeline = [
            _ev(sigs[0], cats[0], 1_000_000.0),
            _ev(sigs[1], cats[1], 1_000_001.0),
        ]
        clique = _clique(sigs, cats)
        result = match_template(clique, timeline, t)
        assert result.match_score < 1.0
        # Must not crash

    @pytest.mark.parametrize("name", NEW_TEMPLATE_NAMES)
    def test_match_score_bounded_0_1(self, name: str):
        """Match score must always be in [0, 1]."""
        t = _template_by_name(name)
        clique = _clique(["x.y.z"], ["positioning"])
        result = match_template(clique, [_ev("x.y.z", "positioning")], t)
        assert 0.0 <= result.match_score <= 1.0


# ═══════════════════════════════════════════════════════════════
# Regression — existing templates structure unchanged
# ═══════════════════════════════════════════════════════════════


class TestExistingTemplatesRegression:
    """Ensure original 12 templates were not modified."""

    ORIGINAL_SPECS = {
        "supply_chain_disruption": (4, 3),
        "monetary_policy_shift": (4, 3),
        "geopolitical_escalation": (4, 3),
        "health_crisis": (4, 3),
        "agricultural_shock": (4, 3),
        "energy_crisis": (4, 3),
        "credit_stress_cascade": (4, 3),
        "tech_disruption": (4, 3),
        "labor_market_shift": (4, 3),
        "trade_war_escalation": (4, 3),
        "construction_cycle": (4, 3),
        "digital_infrastructure_crisis": (4, 3),
    }

    @pytest.mark.parametrize(
        "name,expected",
        ORIGINAL_SPECS.items(),
    )
    def test_step_count_and_min_match(self, name: str, expected: tuple):
        t = _template_by_name(name)
        n_steps, eff_min = expected
        assert len(t.steps) == n_steps, f"{name} steps changed"
        assert t.effective_min_match == eff_min, f"{name} min_match changed"
