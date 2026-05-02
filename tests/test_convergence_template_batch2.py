"""Tests for 28 convergence templates (#23–#50).

Covers:
- Template structural integrity (names, step counts, min_match, regexes, categories)
- Full 50-template library invariants
- Synthetic evidence matching for every new template
- Direction constraint testing (pharma_pipeline_collapse step-0 direction=-1)
- Temporal ordering across varied windows (45-day templates)
- Partial match scoring
- Cross-template regression (existing 22 templates unchanged)
"""

from __future__ import annotations

import re

import pytest

from agent.convergence.evidence import Evidence
from agent.convergence.graph import ConvergenceClique
from agent.convergence.taxonomy import CATEGORIES
from agent.convergence.templates import (
    TEMPLATE_LIBRARY,
    CausalTemplate,
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
    for t in TEMPLATE_LIBRARY:
        if t.name == name:
            return t
    raise ValueError(f"Template {name!r} not found in TEMPLATE_LIBRARY")


# ── Batch-2 template names ────────────────────────────────────

BATCH2_NAMES = [
    "currency_crisis_em",
    "dollar_squeeze",
    "twin_deficit_crisis",
    "sovereign_debt_spiral",
    "fiscal_dominance",
    "real_estate_bubble",
    "construction_bust_banking",
    "inflation_persistence",
    "deflation_trap",
    "chokepoint_disruption",
    "dark_fleet_expansion",
    "shipping_regime_change",
    "liquidity_freeze",
    "bank_run_digital",
    "contagion_cascade",
    "climate_insurance_cascade",
    "water_stress_food_crisis",
    "stablecoin_depeg",
    "crypto_energy_nexus",
    "drug_safety_crisis",
    "pharma_pipeline_collapse",
    "election_positioning",
    "regime_change_market",
    "critical_mineral_bottleneck",
    "supply_chain_decoupling",
    "bond_equity_divergence",
    "commodity_demand_collapse",
    "internet_censorship_escalation",
]


# ═══════════════════════════════════════════════════════════════
# Full library invariants
# ═══════════════════════════════════════════════════════════════


class TestLibraryInvariants:
    """Invariants across the full 50-template library."""

    def test_total_count(self):
        assert len(TEMPLATE_LIBRARY) == 50

    def test_unique_names(self):
        names = [t.name for t in TEMPLATE_LIBRARY]
        assert len(names) == len(set(names)), f"Duplicate: {names}"

    def test_batch2_count(self):
        assert len(BATCH2_NAMES) == 28

    def test_all_batch2_present(self):
        names = {t.name for t in TEMPLATE_LIBRARY}
        for n in BATCH2_NAMES:
            assert n in names, f"Missing template: {n}"


# ═══════════════════════════════════════════════════════════════
# Structural integrity — parametrised over all 28
# ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name", BATCH2_NAMES)
class TestBatch2Structure:
    def test_has_at_least_5_steps(self, name: str):
        t = _template_by_name(name)
        assert len(t.steps) >= 5, f"{name} has {len(t.steps)} steps"

    def test_step0_within_days_is_0(self, name: str):
        t = _template_by_name(name)
        assert t.steps[0].within_days == 0

    def test_within_days_non_decreasing(self, name: str):
        t = _template_by_name(name)
        for i in range(1, len(t.steps)):
            assert t.steps[i].within_days >= t.steps[i - 1].within_days, f"{name} step {i} within_days decreases"

    def test_valid_categories(self, name: str):
        t = _template_by_name(name)
        for i, s in enumerate(t.steps):
            for cat in s.category_pattern.split("|"):
                assert cat in CATEGORIES, f"{name} step {i} has unknown category {cat!r}"

    def test_valid_regex(self, name: str):
        t = _template_by_name(name)
        for i, s in enumerate(t.steps):
            try:
                re.compile(s.signal_pattern)
            except re.error as exc:
                pytest.fail(f"{name} step {i} bad regex: {exc}")

    def test_direction_values(self, name: str):
        t = _template_by_name(name)
        for i, s in enumerate(t.steps):
            assert s.direction in (+1, -1, None), f"{name} step {i} direction={s.direction}"

    def test_effective_min_match(self, name: str):
        t = _template_by_name(name)
        eff = t.effective_min_match
        assert 1 <= eff <= len(t.steps), f"{name} eff_min_match={eff}"

    def test_at_least_3_distinct_categories(self, name: str):
        t = _template_by_name(name)
        cats = set()
        for s in t.steps:
            cats.update(s.category_pattern.split("|"))
        assert len(cats) >= 3, f"{name} only {len(cats)} distinct categories"


# ═══════════════════════════════════════════════════════════════
# Synthetic matching — full-match for every template
# ═══════════════════════════════════════════════════════════════


class TestCurrencyFiscalCluster:
    """Templates 23–27."""

    def test_currency_crisis_em_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("currency_crisis_em")
        signals = [
            "central_bank.tr.balance_wow_pct",
            "capital_flows.tr.holdings_mom_pct",
            "sovereign.tr.spread_vs_de",
            "gdelt.global.material_conflict_ratio",
            "cftc.eur.mm_net_pct_oi",
        ]
        cats = [
            "monetary_policy",
            "monetary_policy",
            "financial_stress",
            "geopolitical",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 18 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_dollar_squeeze_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("dollar_squeeze")
        signals = [
            "central_bank.us.balance_wow_pct",
            "sovereign.br.spread_vs_de",
            "capital_flows.br.holdings_mom_pct",
            "defi.tvl.drawdown_breadth",
            "pmi.br.manufacturing",
        ]
        cats = [
            "monetary_policy",
            "financial_stress",
            "monetary_policy",
            "financial_stress",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, -1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_twin_deficit_crisis_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("twin_deficit_crisis")
        signals = [
            "treasury.us.net_flow_today",
            "capital_flows.global.reserves_stress",
            "sovereign.us.curve_2s10s",
            "political_risk.oppose_ratio",
            "cftc.tbond.mm_net_pct_oi",
        ]
        cats = [
            "macro_momentum",
            "monetary_policy",
            "financial_stress",
            "geopolitical",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, -1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_sovereign_debt_spiral_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("sovereign_debt_spiral")
        signals = [
            "sovereign.gr.spread_vs_de",
            "capital_flows.gr.holdings_mom_pct",
            "creditor.sec.filing_count",
            "gdelt.global.material_conflict_ratio",
            "consumer_sentiment.eu.headline",
        ]
        cats = [
            "financial_stress",
            "monetary_policy",
            "financial_stress",
            "geopolitical",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_fiscal_dominance_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("fiscal_dominance")
        signals = [
            "central_bank.us.net_liquidity_usd",
            "treasury.tga.daily_change_pct",
            "sovereign.us.curve_2s10s",
            "consumer_sentiment.us.inflation_expectations",
            "cftc.tbond.mm_net_pct_oi",
        ]
        cats = [
            "monetary_policy",
            "macro_momentum",
            "financial_stress",
            "macro_momentum",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestRealEstateInflationCluster:
    """Templates 28–31."""

    def test_real_estate_bubble_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("real_estate_bubble")
        signals = [
            "permits.us.single_family.mom_pct",
            "central_bank.us.balance_wow_pct",
            "lobbying.real_estate.spend_anomaly",
            "creditor.sec.filing_count",
            "consumer_sentiment.us.headline",
        ]
        cats = [
            "macro_momentum",
            "monetary_policy",
            "behavioral_intent",
            "financial_stress",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 12 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 19 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 28 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 40 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_construction_bust_banking_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("construction_bust_banking")
        signals = [
            "permits.us.total.mom_pct",
            "jobs.us.construction",
            "bankruptcy.us.chapter_11",
            "liquidity.us.regime",
            "pmi.us.manufacturing",
        ]
        cats = [
            "macro_momentum",
            "behavioral_intent",
            "financial_stress",
            "financial_stress",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, -1),
            _ev(signals[1], cats[1], t0 + 12 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 19 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 28 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 40 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_inflation_persistence_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("inflation_persistence")
        signals = [
            "consumer_sentiment.us.inflation_expectations",
            "jobs.us.openings",
            "supply_chain.freight.mom_pct",
            "pmi.us.manufacturing",
            "central_bank.us.net_liquidity_usd",
        ]
        cats = [
            "macro_momentum",
            "behavioral_intent",
            "supply_chain",
            "macro_momentum",
            "monetary_policy",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_deflation_trap_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("deflation_trap")
        signals = [
            "consumer_sentiment.eu.headline",
            "pmi.de.manufacturing",
            "jobs.us.claims",
            "bankruptcy.us.chapter_11",
            "central_bank.ecb.balance_wow_pct",
        ]
        cats = [
            "macro_momentum",
            "macro_momentum",
            "behavioral_intent",
            "financial_stress",
            "monetary_policy",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, -1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, -1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestChokepointShippingCluster:
    """Templates 32–34."""

    def test_chokepoint_disruption_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("chokepoint_disruption")
        signals = [
            "gdelt.global.material_conflict_ratio",
            "ais.suez.vessel_count",
            "supply_chain.freight.mom_pct",
            "energy.crude.wow_change",
            "cftc.crude_oil.mm_net_pct_oi",
        ]
        cats = [
            "geopolitical",
            "physical_flow",
            "supply_chain",
            "physical_flow",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_dark_fleet_expansion_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("dark_fleet_expansion")
        signals = [
            "sanctions.global.recent_additions",
            "ais.hormuz.tanker_ratio",
            "gdelt.global.event_count",
            "cftc.crude_oil.mm_net_pct_oi",
            "energy.crude.level",
        ]
        cats = [
            "regulatory_action",
            "physical_flow",
            "geopolitical",
            "positioning",
            "physical_flow",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_shipping_regime_change_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("shipping_regime_change")
        signals = [
            "ais.baltic.vessel_count",
            "comtrade.steel.trade_volume",
            "pmi.cn.manufacturing",
            "cftc.copper.mm_net_pct_oi",
            "wiki.baltic_dry_index.spike_zscore",
        ]
        cats = [
            "physical_flow",
            "supply_chain",
            "macro_momentum",
            "positioning",
            "behavioral_intent",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestLiquidityBankingCluster:
    """Templates 35–37."""

    def test_liquidity_freeze_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("liquidity_freeze")
        signals = [
            "liquidity.us.regime",
            "sovereign.us.curve_2s10s",
            "central_bank.us.net_liquidity_usd",
            "finra.spy.short_ratio",
            "pmi.us.manufacturing",
        ]
        cats = [
            "financial_stress",
            "financial_stress",
            "monetary_policy",
            "positioning",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 2 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 5 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 12 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_bank_run_digital_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("bank_run_digital")
        signals = [
            "defi.tvl.drawdown_breadth",
            "wiki.silicon_valley_bank.spike_zscore",
            "creditor.sec.filing_count",
            "liquidity.us.composite_zscore",
            "polymarket.bank_crisis.probability",
        ]
        cats = [
            "financial_stress",
            "behavioral_intent",
            "financial_stress",
            "financial_stress",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 2 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 5 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 12 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 19 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_contagion_cascade_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("contagion_cascade")
        signals = [
            "bankruptcy.us.chapter_11",
            "sovereign.it.spread_vs_de",
            "defi.tvl.drawdown_breadth",
            "capital_flows.ust.coordinated_selling",
            "consumer_sentiment.us.headline",
        ]
        cats = [
            "financial_stress",
            "financial_stress",
            "financial_stress",
            "monetary_policy",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, -1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestClimateEnvironmentCluster:
    """Templates 38–39."""

    def test_climate_insurance_cascade_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("climate_insurance_cascade")
        signals = [
            "weather.us.severe_alert_count",
            "satellite.fire.hotspot_count",
            "creditor.sec.filing_count",
            "permits.us.total.mom_pct",
            "consumer_sentiment.us.headline",
        ]
        cats = [
            "physical_disruption",
            "physical_disruption",
            "financial_stress",
            "macro_momentum",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 28 * _DAY, -1),
            _ev(signals[4], cats[4], t0 + 40 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_water_stress_food_crisis_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("water_stress_food_crisis")
        signals = [
            "weather.global.fire_count_infra",
            "satellite.vegetation.anomaly_pct",
            "food_security.eg.stress",
            "ais.suez.vessel_count",
            "gdelt.global.avg_goldstein",
        ]
        cats = [
            "physical_disruption",
            "supply_chain",
            "biological",
            "physical_flow",
            "geopolitical",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 12 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 19 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 28 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 40 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestCryptoDigitalCluster:
    """Templates 40–41."""

    def test_stablecoin_depeg_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("stablecoin_depeg")
        signals = [
            "defi.stablecoin.total_supply",
            "defi.dex.total_volume_24h",
            "crypto.btc.whale_volume",
            "wiki.terra_luna.spike_zscore",
            "polymarket.stablecoin.probability",
        ]
        cats = [
            "financial_stress",
            "financial_stress",
            "financial_stress",
            "behavioral_intent",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 2 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 5 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 12 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 19 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_crypto_energy_nexus_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("crypto_energy_nexus")
        signals = [
            "power_grid.demand.total_peak_mw",
            "defi.tvl.total_usd",
            "energy.nat_gas.wow_change",
            "regulatory.us.significant_count",
            "wiki.crypto_mining.spike_zscore",
        ]
        cats = [
            "physical_flow",
            "financial_stress",
            "physical_flow",
            "regulatory_action",
            "behavioral_intent",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestPharmaCluster:
    """Templates 42–43."""

    def test_drug_safety_crisis_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("drug_safety_crisis")
        signals = [
            "fda.adverse_events.serious_count",
            "wiki.vioxx.spike_zscore",
            "form144.pfizer.sell_cluster",
            "creditor.sec.filing_count",
            "finra.pfe.short_ratio",
        ]
        cats = [
            "regulatory_action",
            "behavioral_intent",
            "positioning",
            "financial_stress",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, -1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_pharma_pipeline_collapse_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("pharma_pipeline_collapse")
        signals = [
            "fda.approvals.count",
            "form144.mrna.sell_cluster",
            "patent.H04L.total_count",
            "finra.mrna.short_ratio",
            "bankruptcy.sec.enforcement_count",
        ]
        cats = [
            "regulatory_action",
            "positioning",
            "behavioral_intent",
            "positioning",
            "financial_stress",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, -1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, -1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_pharma_pipeline_collapse_wrong_direction_step0(self):
        """Step 0 requires direction=-1; sending +1 should miss it."""
        t0 = 1_000_000.0
        tmpl = _template_by_name("pharma_pipeline_collapse")
        signals = [
            "fda.approvals.count",
            "form144.mrna.sell_cluster",
            "patent.H04L.total_count",
            "finra.mrna.short_ratio",
            "bankruptcy.sec.enforcement_count",
        ]
        cats = [
            "regulatory_action",
            "positioning",
            "behavioral_intent",
            "positioning",
            "financial_stress",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),  # WRONG direction
            _ev(signals[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, -1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score < 1.0
        assert result.lead_signal is None  # step 0 didn't match


class TestPoliticalCluster:
    """Templates 44–45."""

    def test_election_positioning_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("election_positioning")
        signals = [
            "political_risk.ie_total_spend",
            "lobbying.defense.spend_anomaly",
            "polymarket.election.probability",
            "regulatory.us.document_count",
            "consumer_sentiment.us.headline",
        ]
        cats = [
            "geopolitical",
            "behavioral_intent",
            "positioning",
            "regulatory_action",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 12 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 19 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 28 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 40 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_regime_change_market_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("regime_change_market")
        signals = [
            "gdelt.global.material_conflict_ratio",
            "wiki.coup.spike_zscore",
            "capital_flows.ust.coordinated_selling",
            "sanctions.global.recent_additions",
            "cftc.eur.mm_net_pct_oi",
        ]
        cats = [
            "geopolitical",
            "behavioral_intent",
            "monetary_policy",
            "regulatory_action",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, -1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestMaterialsTradeCluster:
    """Templates 46–47."""

    def test_critical_mineral_bottleneck_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("critical_mineral_bottleneck")
        signals = [
            "sanctions.global.program_count",
            "comtrade.lithium.trade_volume",
            "patent.C01G.total_count",
            "ais.pacific.vessel_count",
            "cftc.copper.mm_net_pct_oi",
        ]
        cats = [
            "regulatory_action",
            "supply_chain",
            "behavioral_intent",
            "physical_flow",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_supply_chain_decoupling_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("supply_chain_decoupling")
        signals = [
            "sanctions.global.recent_additions",
            "comtrade.semiconductors.trade_volume",
            "lobbying.chips.spend_anomaly",
            "ais.pacific.vessel_count",
            "permits.us.total.mom_pct",
        ]
        cats = [
            "regulatory_action",
            "supply_chain",
            "behavioral_intent",
            "physical_flow",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 12 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 19 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 28 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 40 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestCrossMarketCluster:
    """Templates 48–49."""

    def test_bond_equity_divergence_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("bond_equity_divergence")
        signals = [
            "sovereign.us.curve_2s10s",
            "liquidity.us.composite_zscore",
            "pmi.us.manufacturing",
            "cftc.tbond.mm_net_pct_oi",
            "consumer_sentiment.us.headline",
        ]
        cats = [
            "financial_stress",
            "financial_stress",
            "macro_momentum",
            "positioning",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, -1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, -1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid

    def test_commodity_demand_collapse_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("commodity_demand_collapse")
        signals = [
            "energy.crude.wow_change",
            "ais.global.vessel_count",
            "pmi.cn.manufacturing",
            "supply_chain.freight.mom_pct",
            "cftc.crude_oil.mm_net_pct_oi",
        ]
        cats = [
            "physical_flow",
            "physical_flow",
            "macro_momentum",
            "supply_chain",
            "positioning",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, -1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, -1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, -1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestInternetCensorshipTemplate:
    """Template 50."""

    def test_internet_censorship_escalation_full_match(self):
        t0 = 1_000_000.0
        tmpl = _template_by_name("internet_censorship_escalation")
        signals = [
            "internet.censorship.anomaly_rate",
            "dns.cloudflare.change_count",
            "gdelt.global.material_conflict_ratio",
            "capital_flows.cn.holdings_mom_pct",
            "defi.tvl.total_usd",
        ]
        cats = [
            "physical_disruption",
            "behavioral_intent",
            "geopolitical",
            "monetary_policy",
            "financial_stress",
        ]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, -1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


# ═══════════════════════════════════════════════════════════════
# Edge case & partial match tests
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_partial_match_3_of_5(self):
        """A clique with only 3 matching signals scores 0.6."""
        t0 = 1_000_000.0
        tmpl = _template_by_name("bond_equity_divergence")
        signals = [
            "sovereign.us.curve_2s10s",
            "liquidity.us.composite_zscore",
            "pmi.us.manufacturing",
        ]
        cats = ["financial_stress", "financial_stress", "macro_momentum"]
        clique = _clique(signals, cats)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == pytest.approx(0.6)

    def test_empty_clique_scores_zero(self):
        tmpl = _template_by_name("currency_crisis_em")
        clique = _clique([], [])
        result = match_template(clique, [], tmpl)
        assert result.match_score == 0.0

    def test_wrong_categories_score_zero(self):
        """Signals matching regex but wrong categories should not match."""
        t0 = 1_000_000.0
        tmpl = _template_by_name("deflation_trap")
        # Use signal_ids that match the regexes but assign wrong categories
        signals = [
            "consumer_sentiment.us.headline",
            "pmi.us.manufacturing",
            "jobs.us.claims",
            "bankruptcy.us.chapter_11",
            "central_bank.us.net_liquidity_usd",
        ]
        # All wrong categories
        wrong_cats = [
            "physical_flow",
            "physical_flow",
            "physical_flow",
            "physical_flow",
            "physical_flow",
        ]
        clique = _clique(signals, wrong_cats)
        timeline = [_ev(signals[i], wrong_cats[i], t0 + i * 5 * _DAY, -1) for i in range(5)]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 0.0

    def test_temporal_order_violation(self):
        """Lag signals firing before trigger invalidates temporal order."""
        t0 = 1_000_000.0
        tmpl = _template_by_name("liquidity_freeze")
        signals = [
            "liquidity.us.regime",
            "sovereign.us.curve_2s10s",
            "central_bank.us.net_liquidity_usd",
            "finra.spy.short_ratio",
            "pmi.us.manufacturing",
        ]
        cats = [
            "financial_stress",
            "financial_stress",
            "monetary_policy",
            "positioning",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        # All signals fire at same time (step 0) — lag before trigger
        timeline = [
            _ev(signals[0], cats[0], t0 + 30 * _DAY, +1),  # trigger LATE
            _ev(signals[1], cats[1], t0, +1),  # fires before trigger
            _ev(signals[2], cats[2], t0 + 1 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 2 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 3 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert not result.temporal_order_valid

    def test_direction_none_accepts_any(self):
        """Steps with direction=None should accept both +1 and -1."""
        t0 = 1_000_000.0
        tmpl = _template_by_name("shipping_regime_change")
        # All steps have direction=None except step 4 (+1)
        signals = [
            "ais.baltic.vessel_count",
            "comtrade.steel.trade_volume",
            "pmi.cn.manufacturing",
            "cftc.copper.mm_net_pct_oi",
            "wiki.shipping.spike_zscore",
        ]
        cats = [
            "physical_flow",
            "supply_chain",
            "macro_momentum",
            "positioning",
            "behavioral_intent",
        ]
        clique = _clique(signals, cats)
        # Mix of directions for None-direction steps
        timeline = [
            _ev(signals[0], cats[0], t0, -1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, -1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 19 * _DAY, -1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0

    def test_drug_safety_crisis_insider_sell_direction(self):
        """drug_safety_crisis step 2 requires direction=-1 (insider selling)."""
        t0 = 1_000_000.0
        tmpl = _template_by_name("drug_safety_crisis")
        signals = [
            "fda.adverse_events.serious_count",
            "wiki.drug_recall.spike_zscore",
            "form144.pfizer.sell_cluster",
            "creditor.sec.filing_count",
            "finra.pfe.short_ratio",
        ]
        cats = [
            "regulatory_action",
            "behavioral_intent",
            "positioning",
            "financial_stress",
            "positioning",
        ]
        clique = _clique(signals, cats)
        # Step 2 with WRONG direction (+1 instead of -1)
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 5 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 12 * _DAY, +1),  # Wrong: buying not selling
            _ev(signals[3], cats[3], t0 + 19 * _DAY, +1),
            _ev(signals[4], cats[4], t0 + 28 * _DAY, +1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score < 1.0  # step 2 should miss

    def test_45_day_window_template(self):
        """Templates with 45-day windows should accept signals within range."""
        t0 = 1_000_000.0
        tmpl = _template_by_name("climate_insurance_cascade")
        signals = [
            "weather.us.severe_alert_count",
            "satellite.fire.hotspot_count",
            "creditor.sec.filing_count",
            "permits.us.total.mom_pct",
            "consumer_sentiment.us.headline",
        ]
        cats = [
            "physical_disruption",
            "physical_disruption",
            "financial_stress",
            "macro_momentum",
            "macro_momentum",
        ]
        clique = _clique(signals, cats)
        # Push step 4 right to the 45-day limit
        timeline = [
            _ev(signals[0], cats[0], t0, +1),
            _ev(signals[1], cats[1], t0 + 6 * _DAY, +1),
            _ev(signals[2], cats[2], t0 + 13 * _DAY, +1),
            _ev(signals[3], cats[3], t0 + 29 * _DAY, -1),
            _ev(signals[4], cats[4], t0 + 44 * _DAY, -1),
        ]
        result = match_template(clique, timeline, tmpl)
        assert result.match_score == 1.0
        assert result.temporal_order_valid


class TestCrossTemplateRegression:
    """Ensure batch 1 templates are not broken by batch 2 additions."""

    BATCH1_NAMES = [
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

    def test_batch1_templates_still_present(self):
        names = {t.name for t in TEMPLATE_LIBRARY}
        for n in self.BATCH1_NAMES:
            assert n in names, f"Batch-1 template {n} missing after batch-2 addition"

    def test_batch1_template_steps_unchanged(self):
        """Verify step counts haven't changed for original templates."""
        expected_steps = {
            "supply_chain_disruption": 4,
            "monetary_policy_shift": 4,
            "geopolitical_escalation": 4,
            "health_crisis": 4,
            "agricultural_shock": 4,
            "energy_crisis": 4,
            "credit_stress_cascade": 4,
            "tech_disruption": 4,
            "labor_market_shift": 4,
            "trade_war_escalation": 4,
            "construction_cycle": 4,
            "digital_infrastructure_crisis": 4,
        }
        for name, count in expected_steps.items():
            t = _template_by_name(name)
            assert len(t.steps) == count, f"{name} step count changed"
