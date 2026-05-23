"""Causal chain template library and pattern matcher.

Encodes domain knowledge about known causal chains — e.g., "sanctions →
shipping diversion → commodity spike → PMI decline" — so that detected
convergence cliques can be classified against known event archetypes.

Each :class:`CausalTemplate` is a short ordered sequence of
:class:`TemplateStep` entries.  A step matches when the clique contains
a signal whose ``signal_id`` matches the step's regex **and** whose
``category`` is in the step's category set **and** (optionally) whose
``direction`` matches.  Temporal ordering is checked against the evidence
timeline so that earlier steps must fire *before* later steps
(within the ``within_days`` tolerance).

This module is deterministic and LLM-free (Pipeline Layer contract).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from agent.convergence.evidence import Evidence
from agent.convergence.graph import ConvergenceClique

log = logging.getLogger(__name__)


# ── Dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class TemplateStep:
    """One step in a causal chain template.

    Parameters
    ----------
    category_pattern : str
        Pipe-separated set of acceptable categories
        (e.g., ``"physical_disruption|regulatory_action"``).
    signal_pattern : str
        Python regex (``re.search``) applied to ``Evidence.signal_id``.
    within_days : int
        Maximum calendar days after the *trigger* step (step 0)
        that this step may occur.  Step 0 itself uses 0.
    direction : int | None
        Expected direction (+1 / −1) or ``None`` for any.
    """

    category_pattern: str
    signal_pattern: str
    within_days: int
    direction: int | None = None

    # Pre-compiled helpers (not part of frozen hash/eq) ─────────

    def matches_category(self, category: str) -> bool:
        """Return True if *category* matches this step's pattern."""
        return category in self.category_pattern.split("|")

    def matches_signal(self, signal_id: str) -> bool:
        """Return True if *signal_id* matches this step's regex."""
        try:
            return re.search(self.signal_pattern, signal_id) is not None
        except re.error:
            log.warning("Invalid regex in template step: %s", self.signal_pattern)
            return False


@dataclass(frozen=True)
class CausalTemplate:
    """A known causal chain archetype.

    Parameters
    ----------
    name : str
        Machine-readable template identifier
        (e.g., ``"supply_chain_disruption"``).
    description : str
        Human-readable explanation.
    steps : tuple[TemplateStep, ...]
        Ordered trigger → response sequence.
    min_match : int
        Minimum steps that must match for this template to be
        considered a hit.  Defaults to ``len(steps) - 1`` when set
        to 0 (sentinel).
    """

    name: str
    description: str
    steps: tuple[TemplateStep, ...]
    min_match: int = 0  # 0 = sentinel → auto-compute as len(steps) - 1

    @property
    def effective_min_match(self) -> int:
        if self.min_match > 0:
            return self.min_match
        return max(1, len(self.steps) - 1)


@dataclass
class TemplateMatchResult:
    """Result of matching one template against a convergence clique.

    Attributes
    ----------
    template_name : str
        Which template was tested.
    match_score : float
        Fraction of steps matched, in [0.0, 1.0].
    matched_steps : int
        Count of matched steps.
    total_steps : int
        Total steps in the template.
    lead_signal : str | None
        Signal that matched the trigger step (step 0), if any.
    lag_signals : list[str]
        Signals that matched subsequent steps.
    temporal_order_valid : bool
        True if every matched lag signal's earliest evidence
        timestamp is ≥ the trigger signal's earliest timestamp.
    """

    template_name: str
    match_score: float
    matched_steps: int
    total_steps: int
    lead_signal: str | None
    lag_signals: list[str] = field(default_factory=list)
    temporal_order_valid: bool = True


# ── Template Library (50 templates) ────────────────────────────


TEMPLATE_LIBRARY: list[CausalTemplate] = [
    # 1 — Supply Chain Disruption
    CausalTemplate(
        name="supply_chain_disruption",
        description=(
            "Physical or regulatory disruption propagates through shipping, "
            "commodity positioning, and macro indicators."
        ),
        steps=(
            TemplateStep(
                "physical_disruption|regulatory_action",
                r"sanctions\.|weather\.|earthquake\.",
                0,
                +1,
            ),
            TemplateStep("physical_flow", r"ais\.|transport\.|energy_supply\.", 7, None),
            TemplateStep("positioning", r"cftc\.|finra\.", 14, +1),
            TemplateStep("macro_momentum", r"pmi\.|treasury\.", 30, -1),
        ),
    ),
    # 2 — Monetary Policy Shift
    CausalTemplate(
        name="monetary_policy_shift",
        description=(
            "Central bank balance-sheet change triggers rate expectations, capital flows, and positioning adjustments."
        ),
        steps=(
            TemplateStep("monetary_policy", r"central_bank\.|rate_monitor\.", 0, None),
            TemplateStep(
                "monetary_policy|financial_stress",
                r"capital_flows\.|sovereign_debt\.",
                7,
                None,
            ),
            TemplateStep("positioning", r"cftc\.|polymarket\.", 14, None),
            TemplateStep("macro_momentum", r"pmi\.|consumer_sentiment\.", 30, None),
        ),
    ),
    # 3 — Geopolitical Escalation
    CausalTemplate(
        name="geopolitical_escalation",
        description=(
            "Geopolitical tension escalation triggers sanctions, shipping "
            "route changes, commodity positioning, and macro impact."
        ),
        steps=(
            TemplateStep("geopolitical", r"gdelt\.|political_risk\.|migration\.", 0, +1),
            TemplateStep("regulatory_action", r"sanctions\.", 7, +1),
            TemplateStep("physical_flow", r"ais\.|transport\.", 14, None),
            TemplateStep("positioning", r"cftc\.|finra\.", 21, +1),
        ),
    ),
    # 4 — Pandemic / Health Crisis
    CausalTemplate(
        name="health_crisis",
        description=("Biological signal escalates to travel disruption, supply chain impact, and macro decline."),
        steps=(
            TemplateStep("biological", r"disease\.|food_security\.", 0, +1),
            TemplateStep(
                "physical_flow|physical_disruption",
                r"ais\.|transport\.|weather\.",
                14,
                None,
            ),
            TemplateStep("behavioral_intent", r"wikipedia\.|jobs\.", 21, None),
            TemplateStep("macro_momentum", r"pmi\.|consumer_sentiment\.", 30, -1),
        ),
    ),
    # 5 — Agricultural Shock
    CausalTemplate(
        name="agricultural_shock",
        description=(
            "Weather disruption in agricultural regions triggers commodity positioning and food-security alerts."
        ),
        steps=(
            TemplateStep("physical_disruption", r"weather\.|earthquake\.", 0, +1),
            TemplateStep("biological", r"food_security\.", 14, +1),
            TemplateStep("positioning", r"cftc\.", 14, +1),
            TemplateStep("macro_momentum", r"pmi\.|consumer_sentiment\.", 30, -1),
        ),
    ),
    # 6 — Energy Crisis
    CausalTemplate(
        name="energy_crisis",
        description=(
            "Energy supply disruption propagates through grid stress, shipping, industrial output, and macro."
        ),
        steps=(
            TemplateStep(
                "physical_flow|physical_disruption",
                r"energy_supply\.|electricity\.",
                0,
                +1,
            ),
            TemplateStep("physical_flow", r"ais\.|transport\.", 7, None),
            TemplateStep("positioning", r"cftc\.|finra\.", 14, +1),
            TemplateStep("macro_momentum", r"pmi\.", 30, -1),
        ),
    ),
    # 7 — Credit Stress Cascade
    CausalTemplate(
        name="credit_stress_cascade",
        description=(
            "Financial stress signals compound: credit events, DeFi "
            "outflows, sovereign spreads, and corporate distress."
        ),
        steps=(
            TemplateStep("financial_stress", r"bankruptcy\.|creditor\.|defi\.", 0, +1),
            TemplateStep("financial_stress", r"sovereign_debt\.|liquidity\.", 7, +1),
            TemplateStep("positioning", r"cftc\.|finra\.|polymarket\.", 14, +1),
            TemplateStep("macro_momentum", r"pmi\.|consumer_sentiment\.", 30, -1),
        ),
    ),
    # 8 — Tech / Innovation Disruption
    CausalTemplate(
        name="tech_disruption",
        description=(
            "Patent or academic surge signals a technology shift that propagates to lobbying, hiring, and positioning."
        ),
        steps=(
            TemplateStep("behavioral_intent", r"patent\.|academic\.", 0, None),
            TemplateStep("behavioral_intent", r"lobbying\.|jobs\.", 14, +1),
            TemplateStep("supply_chain", r"interconnection\.|gov_contracts\.", 30, None),
            TemplateStep("positioning", r"polymarket\.|insider\.", 30, None),
        ),
    ),
    # 9 — Labor Market Shift
    CausalTemplate(
        name="labor_market_shift",
        description=("Job posting changes signal economic momentum shift, confirmed by macro and consumer indicators."),
        steps=(
            TemplateStep("behavioral_intent", r"jobs\.", 0, None),
            TemplateStep("macro_momentum", r"pmi\.|consumer_sentiment\.", 14, None),
            TemplateStep("macro_momentum", r"building_permits\.|treasury\.", 30, None),
            TemplateStep("positioning", r"cftc\.|finra\.", 30, None),
        ),
    ),
    # 10 — Trade War Escalation
    CausalTemplate(
        name="trade_war_escalation",
        description=("Regulatory/tariff actions → shipping re-routing → commodity shifts → macro impact."),
        steps=(
            TemplateStep("regulatory_action", r"regulatory_gazette\.|sanctions\.", 0, +1),
            TemplateStep("geopolitical", r"gdelt\.|political_risk\.", 7, +1),
            TemplateStep("physical_flow", r"ais\.|transport\.", 14, None),
            TemplateStep("positioning", r"cftc\.|finra\.", 21, +1),
        ),
    ),
    # 11 — Real Estate / Construction Cycle
    CausalTemplate(
        name="construction_cycle",
        description=(
            "Building permit changes signal construction cycle shifts, "
            "confirmed by financial stress and macro indicators."
        ),
        steps=(
            TemplateStep("macro_momentum|regulatory_action", r"building_permits\.", 0, None),
            TemplateStep("behavioral_intent", r"jobs\.", 14, None),
            TemplateStep("financial_stress", r"bankruptcy\.|creditor\.", 30, None),
            TemplateStep("macro_momentum", r"pmi\.|consumer_sentiment\.", 30, None),
        ),
    ),
    # 12 — Digital Infrastructure Crisis
    CausalTemplate(
        name="digital_infrastructure_crisis",
        description=(
            "DNS / cert transparency / internet infrastructure disruption propagates to DeFi, tech sector, and macro."
        ),
        steps=(
            TemplateStep(
                "physical_disruption|behavioral_intent",
                r"dns\.|cert_trans\.|internet\.",
                0,
                +1,
            ),
            TemplateStep("financial_stress", r"defi\.", 7, +1),
            TemplateStep("behavioral_intent", r"wikipedia\.", 14, None),
            TemplateStep("macro_momentum", r"pmi\.", 30, -1),
        ),
    ),
    # ── Advanced cross-domain templates (13–22) ────────────────
    #
    # These exploit TirraMind's unique data surface — satellite,
    # DeFi, DNS, Wikipedia, lobbying, Form 144, etc. — to detect
    # non-obvious causal chains that require connecting disparate
    # data sources.  Grounded in Soros reflexivity, Minsky financial
    # instability, carry-trade dynamics, and financial contagion
    # literature.
    # 13 — Silent Nationalization / Sovereign Resource Grab
    CausalTemplate(
        name="silent_nationalization",
        description=(
            "Lobbying surge, satellite physical activity, enabling regulation, "
            "insider selling, and shipping reroutes reveal state resource "
            "capture before it is announced."
        ),
        steps=(
            TemplateStep("behavioral_intent", r"lobbying\.", 0, +1),
            TemplateStep("physical_disruption", r"satellite\.", 14, +1),
            TemplateStep("regulatory_action", r"regulatory_gazette\.", 21, None),
            TemplateStep("positioning", r"insider\.|form144\.", 30, -1),
            TemplateStep("physical_flow", r"ais\.|transport\.", 45, None),
        ),
        min_match=3,
    ),
    # 14 — DeFi Canary in the Coal Mine
    CausalTemplate(
        name="defi_canary",
        description=(
            "DeFi protocol stress and whale exits precede TradFi credit "
            "events by 7-14 days; Wikipedia attention and prediction market "
            "shifts confirm the propagation (Minsky-inspired)."
        ),
        steps=(
            TemplateStep("financial_stress", r"defi\.", 0, +1),
            TemplateStep("financial_stress", r"whale_alert\.", 3, None),
            TemplateStep("behavioral_intent", r"wikipedia\.", 7, +1),
            TemplateStep("positioning", r"polymarket\.", 14, None),
            TemplateStep("financial_stress", r"bankruptcy\.|creditor\.", 30, +1),
        ),
        min_match=3,
    ),
    # 15 — Pandemic Physical Evidence
    CausalTemplate(
        name="pandemic_physical_evidence",
        description=(
            "Disease surveillance spike confirmed by satellite physical "
            "evidence of disruption, Wikipedia information cascade, "
            "transport decline, and commodity positioning shift."
        ),
        steps=(
            TemplateStep("biological", r"disease\.", 0, +1),
            TemplateStep("physical_disruption", r"satellite\.", 7, +1),
            TemplateStep("behavioral_intent", r"wikipedia\.", 14, +1),
            TemplateStep("physical_flow", r"transport\.", 21, -1),
            TemplateStep("positioning", r"cftc\.", 30, None),
        ),
        min_match=3,
    ),
    # 16 — Capital Flight via Crypto
    CausalTemplate(
        name="capital_flight_crypto",
        description=(
            "Political instability triggers traditional capital outflows "
            "and DeFi stablecoin surges (modern flight capital), widening "
            "sovereign spreads and forcing central bank response "
            "(Soros reflexivity loop)."
        ),
        steps=(
            TemplateStep("geopolitical", r"political_risk\.", 0, +1),
            TemplateStep("monetary_policy", r"capital_flows\.", 7, -1),
            TemplateStep("financial_stress", r"defi\.", 14, +1),
            TemplateStep("financial_stress", r"sovereign_debt\.", 21, +1),
            TemplateStep("monetary_policy", r"central_bank\.", 30, None),
        ),
        min_match=3,
    ),
    # 17 — Infrastructure Decay Cascade
    CausalTemplate(
        name="infrastructure_decay_cascade",
        description=(
            "Power grid instability and internet degradation precede "
            "construction decline, job losses, and consumer confidence "
            "drops — a slow-burn physical-to-economic cascade."
        ),
        steps=(
            TemplateStep("physical_flow", r"power_grid\.", 0, +1),
            TemplateStep("physical_disruption", r"internet\.|dns\.", 14, +1),
            TemplateStep("macro_momentum", r"building_permits\.", 30, -1),
            TemplateStep("behavioral_intent", r"jobs\.", 45, -1),
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 60, -1),
        ),
        min_match=3,
    ),
    # 18 — Commodity Hoarding
    CausalTemplate(
        name="commodity_hoarding",
        description=(
            "Weather disruption confirmed by satellite vegetation damage, "
            "shipping reroutes, speculative positioning surge, and food "
            "security alerts — verified supply shock, not forecast noise."
        ),
        steps=(
            TemplateStep("physical_disruption", r"weather\.", 0, +1),
            TemplateStep("supply_chain", r"satellite\.vegetation", 7, -1),
            TemplateStep("physical_flow", r"ais\.|transport\.", 14, None),
            TemplateStep("positioning", r"cftc\.", 21, +1),
            TemplateStep("biological", r"food_security\.", 30, +1),
        ),
        min_match=3,
    ),
    # 19 — Smart Money Divergence (Soros reflexivity turning point)
    CausalTemplate(
        name="smart_money_divergence",
        description=(
            "Retail euphoria (Wikipedia attention, bullish prediction "
            "markets) diverges from smart-money exits (insider selling, "
            "short volume, whale exits) — reflexive turning point."
        ),
        steps=(
            TemplateStep("behavioral_intent", r"wikipedia\.", 0, +1),
            TemplateStep("positioning", r"polymarket\.", 7, +1),
            TemplateStep("positioning", r"form144\.|insider\.", 14, -1),
            TemplateStep("positioning", r"finra\.", 21, +1),
            TemplateStep("financial_stress", r"defi\.|whale_alert\.", 30, +1),
        ),
        min_match=3,
    ),
    # 20 — Sanctions Evasion Network
    CausalTemplate(
        name="sanctions_evasion_network",
        description=(
            "After sanctions are imposed, evasion appears simultaneously: "
            "dark shipping, DNS/cert changes, DeFi circumvention flows, "
            "and GDELT escalation — predicts sanctions effectiveness."
        ),
        steps=(
            TemplateStep("regulatory_action", r"sanctions\.", 0, +1),
            TemplateStep("physical_flow", r"ais\.", 7, None),
            TemplateStep(
                "behavioral_intent|physical_disruption",
                r"cert_trans\.|dns\.",
                14,
                None,
            ),
            TemplateStep("financial_stress", r"defi\.", 21, +1),
            TemplateStep("geopolitical", r"gdelt\.", 30, +1),
        ),
        min_match=3,
    ),
    # 21 — Carry Trade Unwind
    CausalTemplate(
        name="carry_trade_unwind",
        description=(
            "Central bank rate surprise cascades through capital flow "
            "reversals, DeFi yield-farming unwind, sovereign stress, "
            "speculative positioning collapse, and macro deterioration "
            "(Lee et al. 2020, Plantin & Shin 2010)."
        ),
        steps=(
            TemplateStep("monetary_policy", r"central_bank\.|rate_monitor\.", 0, None),
            TemplateStep("monetary_policy", r"capital_flows\.", 7, -1),
            TemplateStep("financial_stress", r"defi\.", 14, +1),
            TemplateStep("financial_stress", r"sovereign_debt\.", 21, +1),
            TemplateStep("positioning", r"cftc\.", 30, -1),
            TemplateStep("macro_momentum", r"pmi\.", 45, -1),
        ),
        min_match=4,
    ),
    # 22 — Stealth Accumulation
    CausalTemplate(
        name="stealth_accumulation",
        description=(
            "Informed actors accumulate quietly ahead of a regulatory "
            "catalyst: Wikipedia research, lobbying spend, patent filings, "
            "position building — each noise alone, together a fingerprint."
        ),
        steps=(
            TemplateStep("behavioral_intent", r"wikipedia\.", 0, +1),
            TemplateStep("behavioral_intent", r"lobbying\.", 30, +1),
            TemplateStep("behavioral_intent", r"patent\.", 45, None),
            TemplateStep("positioning", r"cftc\.|insider\.", 60, +1),
            TemplateStep(
                "regulatory_action",
                r"regulatory_gazette\.|drug_regulatory\.",
                90,
                None,
            ),
        ),
        min_match=3,
    ),
    # ── Batch 2: templates 23–50 ──────────────────────────────
    # 23 — Currency Crisis (Emerging Market)
    CausalTemplate(
        name="currency_crisis_em",
        description=(
            "Emerging market currency crisis: central bank intervention "
            "triggers capital flight, sovereign stress, geopolitical "
            "turmoil, and speculative positioning."
        ),
        steps=(
            TemplateStep("monetary_policy", r"central_bank\.|capital_flows\.", 0, None),
            TemplateStep("monetary_policy", r"capital_flows\.", 7, -1),
            TemplateStep("financial_stress", r"sovereign_debt\.|sovereign\.", 14, +1),
            TemplateStep("geopolitical", r"gdelt\.|political_risk\.", 21, +1),
            TemplateStep("positioning", r"cftc\.|finra\.", 30, -1),
        ),
    ),
    # 24 — Dollar Squeeze
    CausalTemplate(
        name="dollar_squeeze",
        description=(
            "Dollar funding squeeze propagates from monetary tightening "
            "through sovereign stress, capital reversal, DeFi liquidations, "
            "and macro contraction."
        ),
        steps=(
            TemplateStep("monetary_policy", r"central_bank\.|rate_monitor\.", 0, None),
            TemplateStep("financial_stress", r"sovereign_debt\.|sovereign\.", 7, +1),
            TemplateStep("monetary_policy", r"capital_flows\.", 14, -1),
            TemplateStep("financial_stress", r"defi\.", 21, +1),
            TemplateStep("macro_momentum", r"pmi\.", 30, -1),
        ),
    ),
    # 25 — Twin Deficit Crisis
    CausalTemplate(
        name="twin_deficit_crisis",
        description=(
            "Budget and current account deficits converge into crisis: "
            "falling tax receipts, capital outflow, sovereign stress, "
            "political pressure, speculative positioning."
        ),
        steps=(
            TemplateStep("macro_momentum", r"treasury\.", 0, -1),
            TemplateStep("monetary_policy", r"capital_flows\.", 7, -1),
            TemplateStep("financial_stress", r"sovereign_debt\.|sovereign\.", 14, +1),
            TemplateStep("geopolitical", r"political_risk\.", 21, +1),
            TemplateStep("positioning", r"cftc\.|finra\.", 30, +1),
        ),
    ),
    # 26 — Sovereign Debt Spiral
    CausalTemplate(
        name="sovereign_debt_spiral",
        description=(
            "Sovereign debt spiral: spreads spike, capital flees, creditors "
            "file, political turmoil erupts, consumer sentiment collapses."
        ),
        steps=(
            TemplateStep("financial_stress", r"sovereign_debt\.|sovereign\.", 0, +1),
            TemplateStep("monetary_policy", r"capital_flows\.", 7, -1),
            TemplateStep("financial_stress", r"creditor\.|bankruptcy\.", 14, +1),
            TemplateStep("geopolitical", r"gdelt\.|political_risk\.", 21, +1),
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 30, -1),
        ),
    ),
    # 27 — Fiscal Dominance
    CausalTemplate(
        name="fiscal_dominance",
        description=(
            "Government fiscal pressures override monetary policy: central "
            "bank acts, treasury flows shift, sovereign stress rises, "
            "inflation expectations embed, bond positioning adjusts."
        ),
        steps=(
            TemplateStep("monetary_policy", r"central_bank\.", 0, None),
            TemplateStep("macro_momentum", r"treasury\.", 7, +1),
            TemplateStep("financial_stress", r"sovereign_debt\.|sovereign\.", 14, +1),
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 21, +1),
            TemplateStep("positioning", r"cftc\.", 30, -1),
        ),
    ),
    # 28 — Real Estate Bubble
    CausalTemplate(
        name="real_estate_bubble",
        description=(
            "Housing bubble formation: building permits surge, monetary "
            "policy accommodates, real estate lobbying intensifies, credit "
            "stress emerges, consumer confidence erodes."
        ),
        steps=(
            TemplateStep("macro_momentum", r"permits\.|building_permits\.", 0, +1),
            TemplateStep("monetary_policy", r"central_bank\.|rate_monitor\.", 14, None),
            TemplateStep("behavioral_intent", r"lobbying\.", 21, +1),
            TemplateStep("financial_stress", r"creditor\.|bankruptcy\.", 30, +1),
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 45, -1),
        ),
    ),
    # 29 — Construction Bust → Banking
    CausalTemplate(
        name="construction_bust_banking",
        description=(
            "Construction cycle collapse cascades into banking: permits "
            "fall, construction jobs vanish, bankruptcies spike, liquidity "
            "dries up, PMI contracts."
        ),
        steps=(
            TemplateStep("macro_momentum", r"permits\.|building_permits\.", 0, -1),
            TemplateStep("behavioral_intent", r"jobs\.", 14, -1),
            TemplateStep("financial_stress", r"bankruptcy\.|creditor\.", 21, +1),
            TemplateStep("financial_stress", r"liquidity\.", 30, +1),
            TemplateStep("macro_momentum", r"pmi\.", 45, -1),
        ),
    ),
    # 30 — Inflation Persistence (Wage-Price Spiral)
    CausalTemplate(
        name="inflation_persistence",
        description=(
            "Wage-price spiral builds: inflation expectations rise, labor "
            "demand surges, supply chain pressure increases, PMI input "
            "costs climb, central bank forced to respond."
        ),
        steps=(
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 0, +1),
            TemplateStep("behavioral_intent", r"jobs\.", 7, +1),
            TemplateStep("supply_chain", r"supply_chain\.", 14, +1),
            TemplateStep("macro_momentum", r"pmi\.", 21, +1),
            TemplateStep("monetary_policy", r"central_bank\.", 30, None),
        ),
    ),
    # 31 — Deflation Trap
    CausalTemplate(
        name="deflation_trap",
        description=(
            "Deflationary spiral: consumer confidence collapses, PMI "
            "contracts, job postings evaporate, bankruptcies mount, "
            "central bank forced to respond."
        ),
        steps=(
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 0, -1),
            TemplateStep("macro_momentum", r"pmi\.", 7, -1),
            TemplateStep("behavioral_intent", r"jobs\.", 14, -1),
            TemplateStep("financial_stress", r"bankruptcy\.", 21, +1),
            TemplateStep("monetary_policy", r"central_bank\.", 30, None),
        ),
    ),
    # 32 — Chokepoint Disruption
    CausalTemplate(
        name="chokepoint_disruption",
        description=(
            "Maritime chokepoint disruption: geopolitical event triggers "
            "AIS rerouting, supply chain stress, energy price spikes, "
            "and commodity positioning."
        ),
        steps=(
            TemplateStep("geopolitical", r"gdelt\.|political_risk\.", 0, +1),
            TemplateStep("physical_flow", r"ais\.", 7, None),
            TemplateStep("supply_chain", r"supply_chain\.", 14, +1),
            TemplateStep("physical_flow", r"energy_supply\.|energy\.", 21, +1),
            TemplateStep("positioning", r"cftc\.", 30, +1),
        ),
    ),
    # 33 — Dark Fleet Expansion
    CausalTemplate(
        name="dark_fleet_expansion",
        description=(
            "Sanctions-driven shadow fleet: new sanctions prompt dark "
            "shipping activity, geopolitical coverage, speculative "
            "positioning, and energy market disruption."
        ),
        steps=(
            TemplateStep("regulatory_action", r"sanctions\.", 0, +1),
            TemplateStep("physical_flow", r"ais\.", 7, None),
            TemplateStep("geopolitical", r"gdelt\.", 14, +1),
            TemplateStep("positioning", r"cftc\.", 21, +1),
            TemplateStep("physical_flow", r"energy_supply\.|energy\.", 30, None),
        ),
    ),
    # 34 — Shipping Regime Change
    CausalTemplate(
        name="shipping_regime_change",
        description=(
            "Shipping rate regime change: vessel and transport patterns "
            "shift, trade flows adjust, PMI reflects, positioning moves, "
            "public attention follows."
        ),
        steps=(
            TemplateStep("physical_flow", r"ais\.|transport\.", 0, None),
            TemplateStep("supply_chain", r"supply_chain\.|comtrade\.", 7, None),
            TemplateStep("macro_momentum", r"pmi\.", 14, None),
            TemplateStep("positioning", r"cftc\.", 21, None),
            TemplateStep("behavioral_intent", r"wikipedia\.|wiki\.", 30, +1),
        ),
    ),
    # 35 — Liquidity Freeze
    CausalTemplate(
        name="liquidity_freeze",
        description=(
            "Sudden liquidity freeze: regime change detected, sovereign "
            "spreads widen, central bank responds, short interest spikes, "
            "PMI contracts."
        ),
        steps=(
            TemplateStep("financial_stress", r"liquidity\.", 0, +1),
            TemplateStep("financial_stress", r"sovereign_debt\.|sovereign\.", 3, +1),
            TemplateStep("monetary_policy", r"central_bank\.", 7, None),
            TemplateStep("positioning", r"finra\.", 14, +1),
            TemplateStep("macro_momentum", r"pmi\.", 30, -1),
        ),
    ),
    # 36 — Digital Bank Run
    CausalTemplate(
        name="bank_run_digital",
        description=(
            "Digital bank run: DeFi stress triggers Wikipedia panic "
            "search, creditor filings emerge, liquidity freezes, "
            "prediction markets reprice."
        ),
        steps=(
            TemplateStep("financial_stress", r"defi\.", 0, +1),
            TemplateStep("behavioral_intent", r"wikipedia\.|wiki\.", 3, +1),
            TemplateStep("financial_stress", r"creditor\.|bankruptcy\.", 7, +1),
            TemplateStep("financial_stress", r"liquidity\.", 14, +1),
            TemplateStep("positioning", r"polymarket\.", 21, None),
        ),
    ),
    # 37 — Contagion Cascade
    CausalTemplate(
        name="contagion_cascade",
        description=(
            "Cross-sector financial contagion: corporate distress spreads "
            "to sovereign bonds, crypto, capital flows, and consumer "
            "sentiment."
        ),
        steps=(
            TemplateStep("financial_stress", r"bankruptcy\.|creditor\.", 0, +1),
            TemplateStep("financial_stress", r"sovereign_debt\.|sovereign\.", 7, +1),
            TemplateStep("financial_stress", r"defi\.", 14, +1),
            TemplateStep("monetary_policy", r"capital_flows\.", 21, -1),
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 30, -1),
        ),
    ),
    # 38 — Climate → Insurance Cascade
    CausalTemplate(
        name="climate_insurance_cascade",
        description=(
            "Climate event cascades through insurance and construction: "
            "physical event triggers satellite confirmation, creditor "
            "stress, construction pullback, sentiment drop."
        ),
        steps=(
            TemplateStep("physical_disruption", r"weather\.|earthquake\.", 0, +1),
            TemplateStep("physical_disruption", r"satellite\.fire", 7, +1),
            TemplateStep("financial_stress", r"creditor\.|bankruptcy\.", 14, +1),
            TemplateStep("macro_momentum", r"permits\.|building_permits\.", 30, -1),
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 45, -1),
        ),
    ),
    # 39 — Water Stress → Food Crisis
    CausalTemplate(
        name="water_stress_food_crisis",
        description=(
            "Water stress to food crisis: drought severity rises, satellite "
            "shows crop stress, food security alerts fire, trade flows "
            "shift, geopolitical tensions escalate."
        ),
        steps=(
            TemplateStep("physical_disruption", r"weather\.", 0, +1),
            TemplateStep("supply_chain", r"satellite\.vegetation", 14, -1),
            TemplateStep("biological", r"food_security\.", 21, +1),
            TemplateStep("physical_flow|supply_chain", r"ais\.|comtrade\.", 30, None),
            TemplateStep("geopolitical", r"gdelt\.", 45, +1),
        ),
    ),
    # 40 — Stablecoin Depeg
    CausalTemplate(
        name="stablecoin_depeg",
        description=(
            "Stablecoin depeg cascade: stablecoin metrics stress, DEX "
            "panic volume spikes, crypto whale movements, Wikipedia "
            "panic search, prediction markets reprice."
        ),
        steps=(
            TemplateStep("financial_stress", r"defi\.stablecoin", 0, +1),
            TemplateStep("financial_stress", r"defi\.dex", 3, +1),
            TemplateStep("financial_stress", r"crypto\.|whale_alert\.", 7, +1),
            TemplateStep("behavioral_intent", r"wikipedia\.|wiki\.", 14, +1),
            TemplateStep("positioning", r"polymarket\.", 21, None),
        ),
    ),
    # 41 — Crypto ↔ Energy Nexus
    CausalTemplate(
        name="crypto_energy_nexus",
        description=(
            "Crypto mining strains energy grid, triggering DeFi stress, "
            "energy supply disruption, regulatory response, and public "
            "attention."
        ),
        steps=(
            TemplateStep("physical_flow", r"power_grid\.", 0, +1),
            TemplateStep("financial_stress", r"defi\.", 7, +1),
            TemplateStep("physical_flow", r"energy_supply\.|energy\.", 14, +1),
            TemplateStep("regulatory_action", r"regulatory_gazette\.|regulatory\.", 21, None),
            TemplateStep("behavioral_intent", r"wikipedia\.|wiki\.", 30, +1),
        ),
    ),
    # 42 — Drug Safety Crisis
    CausalTemplate(
        name="drug_safety_crisis",
        description=(
            "Drug safety crisis: FDA adverse events spike, Wikipedia "
            "attention rises, insiders sell, creditor filings emerge, "
            "short sellers pile on."
        ),
        steps=(
            TemplateStep("regulatory_action", r"fda\.|drug_regulatory\.", 0, +1),
            TemplateStep("behavioral_intent", r"wikipedia\.|wiki\.", 7, +1),
            TemplateStep("positioning", r"form144\.|insider\.", 14, -1),
            TemplateStep("financial_stress", r"creditor\.", 21, +1),
            TemplateStep("positioning", r"finra\.", 30, +1),
        ),
    ),
    # 43 — Pharma Pipeline Collapse
    CausalTemplate(
        name="pharma_pipeline_collapse",
        description=(
            "Pharma pipeline failure: drug rejection triggers insider "
            "selling, patent activity drops, shorts accumulate, "
            "bankruptcy risk rises."
        ),
        steps=(
            TemplateStep("regulatory_action", r"fda\.|drug_regulatory\.", 0, -1),
            TemplateStep("positioning", r"form144\.|insider\.", 7, -1),
            TemplateStep("behavioral_intent", r"patent\.", 14, -1),
            TemplateStep("positioning", r"finra\.", 21, +1),
            TemplateStep("financial_stress", r"bankruptcy\.", 30, +1),
        ),
    ),
    # 44 — Election Positioning
    CausalTemplate(
        name="election_positioning",
        description=(
            "Election cycle market positioning: political risk rises, "
            "lobbying intensifies, prediction markets shift, regulatory "
            "activity changes, consumer sentiment adjusts."
        ),
        steps=(
            TemplateStep("geopolitical", r"political_risk\.", 0, +1),
            TemplateStep("behavioral_intent", r"lobbying\.", 14, +1),
            TemplateStep("positioning", r"polymarket\.", 21, None),
            TemplateStep("regulatory_action", r"regulatory_gazette\.|regulatory\.", 30, None),
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 45, None),
        ),
    ),
    # 45 — Regime Change Market
    CausalTemplate(
        name="regime_change_market",
        description=(
            "Political regime change triggers market reorientation: "
            "geopolitical event detected, Wikipedia attention surges, "
            "capital flows adjust, regulatory landscape shifts, "
            "positioning realigns."
        ),
        steps=(
            TemplateStep("geopolitical", r"gdelt\.|political_risk\.", 0, +1),
            TemplateStep("behavioral_intent", r"wikipedia\.|wiki\.", 7, +1),
            TemplateStep("monetary_policy", r"capital_flows\.", 14, None),
            TemplateStep("regulatory_action", r"sanctions\.|regulatory_gazette\.", 21, None),
            TemplateStep("positioning", r"cftc\.|insider\.", 30, None),
        ),
    ),
    # 46 — Critical Mineral Bottleneck
    CausalTemplate(
        name="critical_mineral_bottleneck",
        description=(
            "Critical mineral supply crunch: regulatory/sanctions action, "
            "trade flow disruption, patent race intensifies, shipping "
            "adjusts, speculative positioning builds."
        ),
        steps=(
            TemplateStep("regulatory_action", r"sanctions\.|regulatory_gazette\.", 0, +1),
            TemplateStep("supply_chain", r"comtrade\.|supply_chain\.", 7, None),
            TemplateStep("behavioral_intent", r"patent\.", 14, +1),
            TemplateStep("physical_flow", r"ais\.", 21, None),
            TemplateStep("positioning", r"cftc\.", 30, +1),
        ),
    ),
    # 47 — Supply Chain Decoupling (Friend-Shoring)
    CausalTemplate(
        name="supply_chain_decoupling",
        description=(
            "Friend-shoring and supply chain decoupling: regulatory "
            "triggers, trade flows shift, lobbying/patent activity surges, "
            "shipping reroutes, construction permits rise."
        ),
        steps=(
            TemplateStep("regulatory_action", r"sanctions\.|regulatory_gazette\.", 0, +1),
            TemplateStep("supply_chain", r"comtrade\.|supply_chain\.", 14, None),
            TemplateStep("behavioral_intent", r"lobbying\.|patent\.", 21, +1),
            TemplateStep("physical_flow", r"ais\.|transport\.", 30, None),
            TemplateStep("macro_momentum", r"permits\.|building_permits\.", 45, +1),
        ),
    ),
    # 48 — Bond–Equity Divergence
    CausalTemplate(
        name="bond_equity_divergence",
        description=(
            "Bond market signals recession while equities lag: sovereign "
            "stress, liquidity tightens, PMI contracts, positioning "
            "shifts, consumer sentiment confirms."
        ),
        steps=(
            TemplateStep("financial_stress", r"sovereign_debt\.|sovereign\.", 0, +1),
            TemplateStep("financial_stress", r"liquidity\.", 7, +1),
            TemplateStep("macro_momentum", r"pmi\.", 14, -1),
            TemplateStep("positioning", r"cftc\.", 21, -1),
            TemplateStep("macro_momentum", r"consumer_sentiment\.", 30, -1),
        ),
    ),
    # 49 — Commodity Demand Collapse
    CausalTemplate(
        name="commodity_demand_collapse",
        description=(
            "Physical demand collapse signals: energy/transport volumes "
            "drop, AIS confirms shipping decline, PMI contracts, supply "
            "chain pressure falls, positioning adjusts."
        ),
        steps=(
            TemplateStep("physical_flow", r"energy_supply\.|energy\.|transport\.", 0, -1),
            TemplateStep("physical_flow", r"ais\.", 7, None),
            TemplateStep("macro_momentum", r"pmi\.", 14, -1),
            TemplateStep("supply_chain", r"supply_chain\.", 21, -1),
            TemplateStep("positioning", r"cftc\.", 30, -1),
        ),
    ),
    # 50 — Internet Censorship Escalation
    CausalTemplate(
        name="internet_censorship_escalation",
        description=(
            "Internet censorship escalates: censorship anomaly detected, "
            "DNS/cert infrastructure changes, geopolitical coverage "
            "spikes, capital flees, crypto activity surges."
        ),
        steps=(
            TemplateStep("physical_disruption", r"internet\.censorship|internet\.outage", 0, +1),
            TemplateStep("behavioral_intent", r"dns\.|cert\.", 7, None),
            TemplateStep("geopolitical", r"gdelt\.", 14, +1),
            TemplateStep("monetary_policy", r"capital_flows\.", 21, -1),
            TemplateStep("financial_stress", r"defi\.", 30, +1),
        ),
    ),
]


# ── Matching logic ─────────────────────────────────────────────

# Seconds per day — used for temporal comparisons.
_SECONDS_PER_DAY = 86_400


def _earliest_timestamp(
    signal_id: str,
    evidence_timeline: list[Evidence],
) -> float | None:
    """Return the earliest timestamp for *signal_id* in the timeline."""
    ts: float | None = None
    for ev in evidence_timeline:
        if ev.signal_id == signal_id:
            if ts is None or ev.timestamp < ts:
                ts = ev.timestamp
    return ts


def _latest_direction(
    signal_id: str,
    evidence_timeline: list[Evidence],
) -> int | None:
    """Return the direction of the most recent evidence for *signal_id*."""
    best_ts = -1.0
    direction: int | None = None
    for ev in evidence_timeline:
        if ev.signal_id == signal_id and ev.timestamp > best_ts:
            best_ts = ev.timestamp
            direction = ev.direction
    return direction


def match_template(
    clique: ConvergenceClique,
    evidence_timeline: list[Evidence],
    template: CausalTemplate,
) -> TemplateMatchResult:
    """Score how well a convergence clique matches a causal template.

    The algorithm walks each template step and looks for *any* signal in
    the clique whose category and signal_id match the step's patterns.
    If a match is found and the template step has a ``direction``
    constraint, it must also agree with the evidence direction.

    Temporal ordering is checked:  every matched step (except step 0)
    must have an earliest evidence timestamp within ``step.within_days``
    of the trigger signal *and* not earlier than the trigger.

    Parameters
    ----------
    clique : ConvergenceClique
        The detected convergence clique to classify.
    evidence_timeline : list[Evidence]
        Full evidence history for the relevant detection period.
        Only entries whose ``signal_id`` is in the clique are examined.
    template : CausalTemplate
        The causal template to test.

    Returns
    -------
    TemplateMatchResult
    """
    if not clique.signals or not template.steps:
        return TemplateMatchResult(
            template_name=template.name,
            match_score=0.0,
            matched_steps=0,
            total_steps=len(template.steps),
            lead_signal=None,
        )

    # ── Build lookup structures ────────────────────────────────
    # signal_id → category (from clique.categories is a flat list; need a map)
    # We need per-signal category.  Clique stores sorted distinct categories
    # but NOT a per-signal mapping.  We reconstruct from evidence_timeline.
    sig_set = set(clique.signals)
    sig_category: dict[str, str] = {}
    for ev in evidence_timeline:
        if ev.signal_id in sig_set and ev.signal_id not in sig_category:
            sig_category[ev.signal_id] = ev.category
    # Fallback: if a signal has no evidence (shouldn't happen), skip it.

    total_steps = len(template.steps)
    matched_steps = 0
    lead_signal: str | None = None
    lag_signals: list[str] = []
    trigger_ts: float | None = None
    temporal_valid = True

    for step_idx, step in enumerate(template.steps):
        best_signal: str | None = None

        for sig_id in clique.signals:
            cat = sig_category.get(sig_id)
            if cat is None:
                continue

            # Category check
            if not step.matches_category(cat):
                continue

            # Signal regex check
            if not step.matches_signal(sig_id):
                continue

            # Direction check (if required by step)
            if step.direction is not None:
                actual_dir = _latest_direction(sig_id, evidence_timeline)
                if actual_dir is not None and actual_dir != step.direction:
                    continue

            best_signal = sig_id
            break  # first match wins for this step

        if best_signal is None:
            continue

        # Record match
        matched_steps += 1
        sig_ts = _earliest_timestamp(best_signal, evidence_timeline)

        if step_idx == 0:
            lead_signal = best_signal
            trigger_ts = sig_ts
        else:
            lag_signals.append(best_signal)

            # Temporal ordering check (only if we have a trigger timestamp)
            if trigger_ts is not None and sig_ts is not None:
                delta_days = (sig_ts - trigger_ts) / _SECONDS_PER_DAY
                if delta_days < 0:
                    # Lag signal fired before trigger → invalid order
                    temporal_valid = False
                if delta_days > step.within_days and step.within_days > 0:
                    # Outside the allowed window — still count the match
                    # but mark temporal order as invalid
                    temporal_valid = False

    match_score = matched_steps / total_steps if total_steps > 0 else 0.0

    return TemplateMatchResult(
        template_name=template.name,
        match_score=match_score,
        matched_steps=matched_steps,
        total_steps=total_steps,
        lead_signal=lead_signal,
        lag_signals=lag_signals,
        temporal_order_valid=temporal_valid,
    )


def match_all_templates(
    clique: ConvergenceClique,
    evidence_timeline: list[Evidence],
    templates: list[CausalTemplate] | None = None,
) -> list[TemplateMatchResult]:
    """Match a clique against all templates and return results sorted by score.

    Parameters
    ----------
    clique : ConvergenceClique
        The convergence clique to classify.
    evidence_timeline : list[Evidence]
        Evidence history for the detection period.
    templates : list[CausalTemplate] | None
        Templates to test.  Defaults to :data:`TEMPLATE_LIBRARY`.

    Returns
    -------
    list[TemplateMatchResult]
        Sorted descending by ``match_score``, then by
        ``temporal_order_valid`` (valid first).
    """
    if templates is None:
        templates = TEMPLATE_LIBRARY

    results: list[TemplateMatchResult] = []
    for tmpl in templates:
        result = match_template(clique, evidence_timeline, tmpl)
        results.append(result)

    # Sort: highest match_score first, then temporal validity as tiebreaker.
    results.sort(key=lambda r: (r.match_score, r.temporal_order_valid), reverse=True)
    return results


def best_match(
    clique: ConvergenceClique,
    evidence_timeline: list[Evidence],
    templates: list[CausalTemplate] | None = None,
    min_score: float = 0.5,
) -> TemplateMatchResult | None:
    """Return the best matching template, or ``None`` if nothing beats *min_score*.

    This is a convenience wrapper around :func:`match_all_templates`.
    """
    results = match_all_templates(clique, evidence_timeline, templates)
    if results and results[0].match_score >= min_score:
        return results[0]
    return None
