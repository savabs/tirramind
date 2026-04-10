"""Tests for causal chain template library + matcher (Phase 7c-D.1).

Covers: TemplateStep matching helpers, CausalTemplate parsing/effective_min_match,
TEMPLATE_LIBRARY completeness, match_template (full match, partial match,
direction constraint, temporal order, reverse order, empty inputs),
match_all_templates sorting, best_match threshold, unknown_pattern fallback.
"""

from __future__ import annotations

import time

import pytest

from agent.convergence.evidence import Evidence
from agent.convergence.graph import ConvergenceClique
from agent.convergence.templates import (
    TEMPLATE_LIBRARY,
    CausalTemplate,
    TemplateMatchResult,
    TemplateStep,
    best_match,
    match_all_templates,
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
    source: str = "test",
) -> Evidence:
    """Shorthand Evidence constructor for tests."""
    return Evidence(
        source=source,
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
    """Minimal clique for template matching tests."""
    # Build dummy edges between all pairs
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


# ═══════════════════════════════════════════════════════════════
# TemplateStep — matches_category / matches_signal
# ═══════════════════════════════════════════════════════════════


class TestTemplateStep:
    def test_single_category_match(self):
        step = TemplateStep("positioning", r"cftc\.", 0)
        assert step.matches_category("positioning")
        assert not step.matches_category("macro_momentum")

    def test_pipe_separated_categories(self):
        step = TemplateStep("physical_disruption|regulatory_action", r".*", 0)
        assert step.matches_category("physical_disruption")
        assert step.matches_category("regulatory_action")
        assert not step.matches_category("positioning")

    def test_signal_regex_match(self):
        step = TemplateStep("positioning", r"cftc\..*\.mm_net_long", 0)
        assert step.matches_signal("cftc.crude_oil.mm_net_long")
        assert not step.matches_signal("finra.AAPL.short_ratio")

    def test_signal_regex_partial_match(self):
        step = TemplateStep("positioning", r"cftc\.", 0)
        assert step.matches_signal("cftc.anything.here")

    def test_signal_regex_no_match(self):
        step = TemplateStep("positioning", r"^cftc$", 0)
        assert not step.matches_signal("cftc.crude_oil")

    def test_invalid_regex_returns_false(self):
        step = TemplateStep("positioning", r"[invalid", 0)
        assert not step.matches_signal("anything")

    def test_direction_field(self):
        step_with = TemplateStep("positioning", r"cftc\.", 0, direction=+1)
        step_without = TemplateStep("positioning", r"cftc\.", 0, direction=None)
        assert step_with.direction == 1
        assert step_without.direction is None


# ═══════════════════════════════════════════════════════════════
# CausalTemplate — construction and effective_min_match
# ═══════════════════════════════════════════════════════════════


class TestCausalTemplate:
    def test_construction(self):
        t = CausalTemplate(
            name="test",
            description="Test template",
            steps=(
                TemplateStep("positioning", r"cftc\.", 0),
                TemplateStep("macro_momentum", r"pmi\.", 14),
            ),
        )
        assert t.name == "test"
        assert len(t.steps) == 2

    def test_effective_min_match_auto(self):
        """min_match=0 (sentinel) → len(steps) - 1."""
        t = CausalTemplate(
            name="test",
            description="",
            steps=(
                TemplateStep("a", r".", 0),
                TemplateStep("b", r".", 7),
                TemplateStep("c", r".", 14),
                TemplateStep("d", r".", 21),
            ),
        )
        assert t.effective_min_match == 3

    def test_effective_min_match_explicit(self):
        t = CausalTemplate(
            name="test",
            description="",
            steps=(
                TemplateStep("a", r".", 0),
                TemplateStep("b", r".", 7),
            ),
            min_match=2,
        )
        assert t.effective_min_match == 2

    def test_effective_min_match_single_step(self):
        t = CausalTemplate(
            name="test",
            description="",
            steps=(TemplateStep("a", r".", 0),),
        )
        # max(1, 1-1) = max(1, 0) = 1
        assert t.effective_min_match == 1

    def test_frozen(self):
        t = CausalTemplate(name="t", description="", steps=())
        with pytest.raises(AttributeError):
            t.name = "changed"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════
# TEMPLATE_LIBRARY — completeness
# ═══════════════════════════════════════════════════════════════


class TestTemplateLibrary:
    def test_has_50_templates(self):
        assert len(TEMPLATE_LIBRARY) == 50

    def test_unique_names(self):
        names = [t.name for t in TEMPLATE_LIBRARY]
        assert len(names) == len(set(names))

    def test_all_templates_have_steps(self):
        for t in TEMPLATE_LIBRARY:
            assert len(t.steps) >= 2, f"Template {t.name} has fewer than 2 steps"

    def test_all_templates_have_descriptions(self):
        for t in TEMPLATE_LIBRARY:
            assert t.description, f"Template {t.name} has empty description"

    def test_all_step_zero_within_days_is_zero(self):
        """Trigger step (step 0) should have within_days=0."""
        for t in TEMPLATE_LIBRARY:
            assert (
                t.steps[0].within_days == 0
            ), f"Template {t.name} step 0 within_days != 0"

    def test_known_template_names(self):
        expected = {
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
            # Batch 2 (#23–#50)
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
        }
        actual = {t.name for t in TEMPLATE_LIBRARY}
        assert actual == expected

    def test_step_signal_patterns_are_valid_regex(self):
        """Every signal_pattern must compile without error."""
        import re

        for t in TEMPLATE_LIBRARY:
            for i, step in enumerate(t.steps):
                try:
                    re.compile(step.signal_pattern)
                except re.error as exc:
                    pytest.fail(
                        f"Template {t.name} step {i} has invalid regex "
                        f"{step.signal_pattern!r}: {exc}"
                    )


# ═══════════════════════════════════════════════════════════════
# match_template — single template matching
# ═══════════════════════════════════════════════════════════════


class TestMatchTemplate:
    """Test the core match_template function."""

    def test_perfect_supply_chain_match(self):
        """Sanctions → shipping → CFTC → PMI should match supply_chain_disruption."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "sanctions.russia.additions",
                "ais.baltic.vessel_count",
                "cftc.crude_oil.mm_net_long",
                "pmi.germany.manufacturing",
            ],
            ["regulatory_action", "physical_flow", "positioning", "macro_momentum"],
        )
        timeline = [
            _ev("sanctions.russia.additions", "regulatory_action", t0, +1),
            _ev("ais.baltic.vessel_count", "physical_flow", t0 + 5 * _DAY),
            _ev("cftc.crude_oil.mm_net_long", "positioning", t0 + 10 * _DAY, +1),
            _ev("pmi.germany.manufacturing", "macro_momentum", t0 + 25 * _DAY, -1),
        ]
        template = TEMPLATE_LIBRARY[0]  # supply_chain_disruption
        result = match_template(clique, timeline, template)
        assert result.match_score == 1.0
        assert result.matched_steps == 4
        assert result.lead_signal == "sanctions.russia.additions"
        assert result.temporal_order_valid is True
        assert len(result.lag_signals) == 3

    def test_partial_match_3_of_4(self):
        """Missing the PMI step → 3/4 = 0.75."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "sanctions.russia.additions",
                "ais.baltic.vessel_count",
                "cftc.crude_oil.mm_net_long",
            ],
            ["regulatory_action", "physical_flow", "positioning"],
        )
        timeline = [
            _ev("sanctions.russia.additions", "regulatory_action", t0, +1),
            _ev("ais.baltic.vessel_count", "physical_flow", t0 + 5 * _DAY),
            _ev("cftc.crude_oil.mm_net_long", "positioning", t0 + 10 * _DAY, +1),
        ]
        template = TEMPLATE_LIBRARY[0]
        result = match_template(clique, timeline, template)
        assert result.match_score == 0.75
        assert result.matched_steps == 3

    def test_2_of_4_match(self):
        """Only trigger + shipping → 2/4 = 0.5."""
        t0 = 1_000_000.0
        clique = _clique(
            ["sanctions.russia.additions", "ais.baltic.vessel_count"],
            ["regulatory_action", "physical_flow"],
        )
        timeline = [
            _ev("sanctions.russia.additions", "regulatory_action", t0, +1),
            _ev("ais.baltic.vessel_count", "physical_flow", t0 + 5 * _DAY),
        ]
        template = TEMPLATE_LIBRARY[0]
        result = match_template(clique, timeline, template)
        assert result.match_score == 0.5
        assert result.matched_steps == 2

    def test_no_match(self):
        """Unrelated signals → score 0."""
        clique = _clique(
            ["disease.covid.detection_rate", "wikipedia.neodymium.views"],
            ["biological", "behavioral_intent"],
        )
        timeline = [
            _ev("disease.covid.detection_rate", "biological", 1_000_000.0, +1),
            _ev("wikipedia.neodymium.views", "behavioral_intent", 1_000_000.0),
        ]
        template = TEMPLATE_LIBRARY[0]  # supply_chain_disruption
        result = match_template(clique, timeline, template)
        assert result.match_score < 0.5
        assert result.lead_signal is None or result.matched_steps <= 1

    def test_direction_constraint_violation(self):
        """Step expects +1 but evidence has -1 → step not matched."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "sanctions.russia.additions",
                "ais.baltic.vessel_count",
                "cftc.crude_oil.mm_net_long",
                "pmi.germany.manufacturing",
            ],
            ["regulatory_action", "physical_flow", "positioning", "macro_momentum"],
        )
        timeline = [
            # Trigger direction is wrong (expecting +1)
            _ev("sanctions.russia.additions", "regulatory_action", t0, -1),
            _ev("ais.baltic.vessel_count", "physical_flow", t0 + 5 * _DAY),
            _ev(
                "cftc.crude_oil.mm_net_long", "positioning", t0 + 10 * _DAY, -1
            ),  # expects +1
            _ev(
                "pmi.germany.manufacturing", "macro_momentum", t0 + 25 * _DAY, +1
            ),  # expects -1
        ]
        template = TEMPLATE_LIBRARY[0]
        result = match_template(clique, timeline, template)
        # Steps with direction mismatch are skipped
        assert result.match_score < 1.0

    def test_reverse_temporal_order(self):
        """PMI fires before sanctions → temporal_order_valid = False."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "sanctions.russia.additions",
                "ais.baltic.vessel_count",
                "cftc.crude_oil.mm_net_long",
                "pmi.germany.manufacturing",
            ],
            ["regulatory_action", "physical_flow", "positioning", "macro_momentum"],
        )
        timeline = [
            # Reversed: PMI fires first, sanctions fires last
            _ev("pmi.germany.manufacturing", "macro_momentum", t0, -1),
            _ev("cftc.crude_oil.mm_net_long", "positioning", t0 + 5 * _DAY, +1),
            _ev("ais.baltic.vessel_count", "physical_flow", t0 + 10 * _DAY),
            _ev("sanctions.russia.additions", "regulatory_action", t0 + 25 * _DAY, +1),
        ]
        template = TEMPLATE_LIBRARY[0]
        result = match_template(clique, timeline, template)
        # All 4 signals match semantically, but temporal order is invalid
        assert result.temporal_order_valid is False

    def test_empty_clique(self):
        clique = _clique([], [])
        result = match_template(clique, [], TEMPLATE_LIBRARY[0])
        assert result.match_score == 0.0
        assert result.matched_steps == 0
        assert result.lead_signal is None

    def test_empty_template(self):
        clique = _clique(["a"], ["positioning"])
        template = CausalTemplate(name="empty", description="", steps=())
        result = match_template(clique, [], template)
        assert result.match_score == 0.0
        assert result.total_steps == 0

    def test_lead_and_lag_signals_identified(self):
        """Lead signal is step-0 match, lag signals are the rest."""
        t0 = 1_000_000.0
        clique = _clique(
            ["sanctions.test.additions", "ais.test.vessel_count"],
            ["regulatory_action", "physical_flow"],
        )
        timeline = [
            _ev("sanctions.test.additions", "regulatory_action", t0, +1),
            _ev("ais.test.vessel_count", "physical_flow", t0 + 3 * _DAY),
        ]
        template = TEMPLATE_LIBRARY[0]
        result = match_template(clique, timeline, template)
        assert result.lead_signal == "sanctions.test.additions"
        assert "ais.test.vessel_count" in result.lag_signals

    def test_temporal_window_exceeded(self):
        """Lag signal arrives outside within_days → temporal_order_valid=False."""
        t0 = 1_000_000.0
        clique = _clique(
            ["weather.us.alert_count", "ais.us.vessel_count"],
            ["physical_disruption", "physical_flow"],
        )
        timeline = [
            _ev("weather.us.alert_count", "physical_disruption", t0, +1),
            # Step expects within_days=7, but signal arrives at day 50
            _ev("ais.us.vessel_count", "physical_flow", t0 + 50 * _DAY),
        ]
        template = TEMPLATE_LIBRARY[0]
        result = match_template(clique, timeline, template)
        # The match counts but temporal order is flagged
        assert result.temporal_order_valid is False


# ═══════════════════════════════════════════════════════════════
# match_all_templates — across all templates
# ═══════════════════════════════════════════════════════════════


class TestMatchAllTemplates:
    def test_returns_sorted_by_score(self):
        """Results should be sorted descending by match_score."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "sanctions.russia.additions",
                "ais.baltic.vessel_count",
                "cftc.crude_oil.mm_net_long",
                "pmi.germany.manufacturing",
            ],
            ["regulatory_action", "physical_flow", "positioning", "macro_momentum"],
        )
        timeline = [
            _ev("sanctions.russia.additions", "regulatory_action", t0, +1),
            _ev("ais.baltic.vessel_count", "physical_flow", t0 + 5 * _DAY),
            _ev("cftc.crude_oil.mm_net_long", "positioning", t0 + 10 * _DAY, +1),
            _ev("pmi.germany.manufacturing", "macro_momentum", t0 + 25 * _DAY, -1),
        ]
        results = match_all_templates(clique, timeline)
        assert len(results) == 50
        # Scores should be descending
        for i in range(len(results) - 1):
            assert results[i].match_score >= results[i + 1].match_score

    def test_supply_chain_is_best_for_supply_chain_scenario(self):
        """The supply-chain scenario should match supply_chain_disruption best."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "sanctions.russia.additions",
                "ais.baltic.vessel_count",
                "cftc.crude_oil.mm_net_long",
                "pmi.germany.manufacturing",
            ],
            ["regulatory_action", "physical_flow", "positioning", "macro_momentum"],
        )
        timeline = [
            _ev("sanctions.russia.additions", "regulatory_action", t0, +1),
            _ev("ais.baltic.vessel_count", "physical_flow", t0 + 5 * _DAY),
            _ev("cftc.crude_oil.mm_net_long", "positioning", t0 + 10 * _DAY, +1),
            _ev("pmi.germany.manufacturing", "macro_momentum", t0 + 25 * _DAY, -1),
        ]
        results = match_all_templates(clique, timeline)
        assert results[0].template_name == "supply_chain_disruption"
        assert results[0].match_score >= 0.75

    def test_credit_stress_scenario(self):
        """Bankruptcy → sovereign debt → CFTC → PMI → credit_stress_cascade."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "bankruptcy.us.filings",
                "sovereign_debt.it.spread",
                "cftc.sp500.mm_net_long",
                "pmi.us.manufacturing",
            ],
            ["financial_stress", "financial_stress", "positioning", "macro_momentum"],
        )
        timeline = [
            _ev("bankruptcy.us.filings", "financial_stress", t0, +1),
            _ev("sovereign_debt.it.spread", "financial_stress", t0 + 5 * _DAY, +1),
            _ev("cftc.sp500.mm_net_long", "positioning", t0 + 12 * _DAY, +1),
            _ev("pmi.us.manufacturing", "macro_momentum", t0 + 28 * _DAY, -1),
        ]
        results = match_all_templates(clique, timeline)
        # credit_stress_cascade should score highest
        credit_results = [
            r for r in results if r.template_name == "credit_stress_cascade"
        ]
        assert len(credit_results) == 1
        assert credit_results[0].match_score >= 0.75

    def test_health_crisis_scenario(self):
        """Disease → transport → wikipedia → PMI."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "disease.covid.detection_rate",
                "ais.global.vessel_count",
                "wikipedia.pandemic.views",
                "pmi.us.manufacturing",
            ],
            ["biological", "physical_flow", "behavioral_intent", "macro_momentum"],
        )
        timeline = [
            _ev("disease.covid.detection_rate", "biological", t0, +1),
            _ev("ais.global.vessel_count", "physical_flow", t0 + 10 * _DAY),
            _ev("wikipedia.pandemic.views", "behavioral_intent", t0 + 18 * _DAY),
            _ev("pmi.us.manufacturing", "macro_momentum", t0 + 28 * _DAY, -1),
        ]
        results = match_all_templates(clique, timeline)
        health_results = [r for r in results if r.template_name == "health_crisis"]
        assert len(health_results) == 1
        assert health_results[0].match_score >= 0.75

    def test_custom_template_list(self):
        """Passing a custom template list only tests those templates."""
        custom = [TEMPLATE_LIBRARY[0]]  # Only supply_chain_disruption
        clique = _clique(["a"], ["positioning"])
        results = match_all_templates(clique, [], templates=custom)
        assert len(results) == 1
        assert results[0].template_name == "supply_chain_disruption"


# ═══════════════════════════════════════════════════════════════
# best_match — convenience wrapper
# ═══════════════════════════════════════════════════════════════


class TestBestMatch:
    def test_returns_best_above_threshold(self):
        t0 = 1_000_000.0
        clique = _clique(
            [
                "sanctions.russia.additions",
                "ais.baltic.vessel_count",
                "cftc.crude_oil.mm_net_long",
                "pmi.germany.manufacturing",
            ],
            ["regulatory_action", "physical_flow", "positioning", "macro_momentum"],
        )
        timeline = [
            _ev("sanctions.russia.additions", "regulatory_action", t0, +1),
            _ev("ais.baltic.vessel_count", "physical_flow", t0 + 5 * _DAY),
            _ev("cftc.crude_oil.mm_net_long", "positioning", t0 + 10 * _DAY, +1),
            _ev("pmi.germany.manufacturing", "macro_momentum", t0 + 25 * _DAY, -1),
        ]
        result = best_match(clique, timeline)
        assert result is not None
        assert result.match_score >= 0.5

    def test_returns_none_below_threshold(self):
        """Unrelated signals → no template above 0.5."""
        clique = _clique(
            ["disease.covid.detection_rate"],
            ["biological"],
        )
        timeline = [
            _ev("disease.covid.detection_rate", "biological", 1_000_000.0, +1),
        ]
        result = best_match(clique, timeline, min_score=0.99)
        assert result is None

    def test_none_on_empty_clique(self):
        result = best_match(_clique([], []), [])
        assert result is None

    def test_custom_min_score(self):
        t0 = 1_000_000.0
        clique = _clique(
            ["sanctions.test.additions", "ais.test.vessel_count"],
            ["regulatory_action", "physical_flow"],
        )
        timeline = [
            _ev("sanctions.test.additions", "regulatory_action", t0, +1),
            _ev("ais.test.vessel_count", "physical_flow", t0 + 3 * _DAY),
        ]
        # 2/4 = 0.5 match → min_score=0.6 should reject
        result = best_match(clique, timeline, min_score=0.6)
        # Might match at 0.5, which is < 0.6
        if result is not None:
            assert result.match_score >= 0.6


# ═══════════════════════════════════════════════════════════════
# Geopolitical Escalation — specific scenario
# ═══════════════════════════════════════════════════════════════


class TestGeopoliticalEscalation:
    def test_gdelt_sanctions_shipping_cftc(self):
        """GDELT → sanctions → AIS → CFTC → geopolitical_escalation."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "gdelt.conflict.intensity",
                "sanctions.iran.additions",
                "ais.hormuz.vessel_count",
                "cftc.crude_oil.mm_net_long",
            ],
            ["geopolitical", "regulatory_action", "physical_flow", "positioning"],
        )
        timeline = [
            _ev("gdelt.conflict.intensity", "geopolitical", t0, +1),
            _ev("sanctions.iran.additions", "regulatory_action", t0 + 5 * _DAY, +1),
            _ev("ais.hormuz.vessel_count", "physical_flow", t0 + 12 * _DAY),
            _ev("cftc.crude_oil.mm_net_long", "positioning", t0 + 18 * _DAY, +1),
        ]
        results = match_all_templates(clique, timeline)
        geo_results = [
            r for r in results if r.template_name == "geopolitical_escalation"
        ]
        assert len(geo_results) == 1
        assert geo_results[0].match_score == 1.0
        assert geo_results[0].temporal_order_valid is True


# ═══════════════════════════════════════════════════════════════
# Energy Crisis — specific scenario
# ═══════════════════════════════════════════════════════════════


class TestEnergyCrisis:
    def test_energy_supply_ais_cftc_pmi(self):
        """Energy supply disruption → AIS → CFTC → PMI."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "energy_supply.eu.gas_flow",
                "ais.eu.vessel_count",
                "cftc.natural_gas.mm_net_long",
                "pmi.de.manufacturing",
            ],
            ["physical_flow", "physical_flow", "positioning", "macro_momentum"],
        )
        timeline = [
            _ev("energy_supply.eu.gas_flow", "physical_flow", t0, +1),
            _ev("ais.eu.vessel_count", "physical_flow", t0 + 5 * _DAY),
            _ev("cftc.natural_gas.mm_net_long", "positioning", t0 + 12 * _DAY, +1),
            _ev("pmi.de.manufacturing", "macro_momentum", t0 + 28 * _DAY, -1),
        ]
        results = match_all_templates(clique, timeline)
        energy_results = [r for r in results if r.template_name == "energy_crisis"]
        assert len(energy_results) == 1
        assert energy_results[0].match_score >= 0.75


# ═══════════════════════════════════════════════════════════════
# Digital Infrastructure Crisis — specific scenario
# ═══════════════════════════════════════════════════════════════


class TestDigitalInfrastructureCrisis:
    def test_dns_defi_wikipedia_pmi(self):
        """DNS → DeFi → Wikipedia → PMI."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "dns.global.changes",
                "defi.eth.outflows",
                "wikipedia.cybersecurity.views",
                "pmi.us.manufacturing",
            ],
            [
                "behavioral_intent",
                "financial_stress",
                "behavioral_intent",
                "macro_momentum",
            ],
        )
        timeline = [
            _ev("dns.global.changes", "behavioral_intent", t0, +1),
            _ev("defi.eth.outflows", "financial_stress", t0 + 5 * _DAY, +1),
            _ev("wikipedia.cybersecurity.views", "behavioral_intent", t0 + 12 * _DAY),
            _ev("pmi.us.manufacturing", "macro_momentum", t0 + 28 * _DAY, -1),
        ]
        results = match_all_templates(clique, timeline)
        digi = [
            r for r in results if r.template_name == "digital_infrastructure_crisis"
        ]
        assert len(digi) == 1
        assert digi[0].match_score >= 0.75


# ═══════════════════════════════════════════════════════════════
# Edge cases — direction, timing, degenerate inputs
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_single_signal_clique(self):
        """A single-signal clique can at most match step 0."""
        clique = _clique(
            ["sanctions.test.additions"],
            ["regulatory_action"],
        )
        timeline = [
            _ev("sanctions.test.additions", "regulatory_action", 1_000_000.0, +1),
        ]
        result = match_template(clique, timeline, TEMPLATE_LIBRARY[0])
        assert result.match_score == 0.25  # 1/4

    def test_no_evidence_in_timeline(self):
        """Clique has signals but timeline is empty → 0 matches."""
        clique = _clique(
            ["sanctions.test.additions", "ais.test.vessel_count"],
            ["regulatory_action", "physical_flow"],
        )
        result = match_template(clique, [], TEMPLATE_LIBRARY[0])
        assert result.match_score == 0.0

    def test_duplicate_signal_ids_in_timeline(self):
        """Multiple evidence entries for the same signal_id — should still work."""
        t0 = 1_000_000.0
        clique = _clique(
            ["sanctions.test.additions", "ais.test.vessel_count"],
            ["regulatory_action", "physical_flow"],
        )
        timeline = [
            _ev("sanctions.test.additions", "regulatory_action", t0, +1),
            _ev("sanctions.test.additions", "regulatory_action", t0 + 1 * _DAY, +1),
            _ev("ais.test.vessel_count", "physical_flow", t0 + 5 * _DAY),
            _ev("ais.test.vessel_count", "physical_flow", t0 + 6 * _DAY),
        ]
        result = match_template(clique, timeline, TEMPLATE_LIBRARY[0])
        assert result.matched_steps >= 2

    def test_all_signals_same_timestamp(self):
        """All signals fire simultaneously — temporal order is valid (delta ≥ 0)."""
        t0 = 1_000_000.0
        clique = _clique(
            [
                "sanctions.test.additions",
                "ais.test.vessel_count",
                "cftc.test.mm_net_long",
                "pmi.test.manufacturing",
            ],
            ["regulatory_action", "physical_flow", "positioning", "macro_momentum"],
        )
        timeline = [
            _ev("sanctions.test.additions", "regulatory_action", t0, +1),
            _ev("ais.test.vessel_count", "physical_flow", t0),
            _ev("cftc.test.mm_net_long", "positioning", t0, +1),
            _ev("pmi.test.manufacturing", "macro_momentum", t0, -1),
        ]
        result = match_template(clique, timeline, TEMPLATE_LIBRARY[0])
        # delta = 0 which is ≥ 0 and within the window — should be valid
        assert result.matched_steps == 4
        assert result.temporal_order_valid is True

    def test_match_all_with_empty_library(self):
        clique = _clique(["a"], ["positioning"])
        results = match_all_templates(clique, [], templates=[])
        assert results == []
