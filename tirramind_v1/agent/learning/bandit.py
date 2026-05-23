"""
TirraMind Agent — Strategy Bandit (RL Layer)

Thompson Sampling multi-armed bandit for goal-type selection.
Each arm is a category of work (backtest, explore asset, fetch data, etc.).
The bandit learns which categories produce the most reward over time.

This is the actual learning component — parameters (α, β) update from
observed rewards, producing measurably different behavior after experience.
No LLM in the loop. Pure RL.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class GoalArm:
    """A category of work the agent can choose to pursue.

    Each arm maps to a set of tools and a description that constrains
    the LLM's goal generation to that category.
    """

    name: str
    description: str
    tools: list[str]  # primary tools this category uses
    examples: list[str] = field(default_factory=list)  # example goals for the LLM


# Default arms covering the agent's full capability set.
# Add new arms here as capabilities expand. New arms start with
# uniform prior Beta(1,1) → high uncertainty → automatic exploration.
DEFAULT_ARMS: list[GoalArm] = [
    GoalArm(
        name="backtest_strategy",
        description="Run a backtest on a new strategy variant or asset",
        tools=["backtest"],
        examples=[
            "Backtest regime-avoid strategy on TLT using 2015-2024 data",
            "Backtest buy-and-hold vs regime-only on GLD",
        ],
    ),
    GoalArm(
        name="tune_parameters",
        description="Adjust parameters of an existing strategy and re-test",
        tools=["backtest"],
        examples=[
            "Re-run regime-avoid backtest with HMM K=4 instead of K=3",
            "Test regime-only strategy with min_train=200 weeks",
        ],
    ),
    GoalArm(
        name="explore_asset",
        description="Run regime detection or analysis on a new asset class",
        tools=["liquidity_regime", "market_data"],
        examples=[
            "Run regime detection on EUR/USD and compare to SPY regimes",
            "Fetch BTC-USD data and check regime-conditional returns",
        ],
    ),
    GoalArm(
        name="fetch_macro_data",
        description="Explore new macroeconomic data series from FRED",
        tools=["macro_data"],
        examples=[
            "Fetch and analyze ECBASSETSW trends over the last 5 years",
            "Pull yield curve data (T10Y2Y) and check for inversion signals",
        ],
    ),
    GoalArm(
        name="research_market",
        description="Web research on current market conditions or events",
        tools=["web_search", "web_browse"],
        examples=[
            "Research the latest Fed policy changes and their market impact",
            "Search for current global liquidity conditions commentary",
        ],
    ),
    GoalArm(
        name="prediction_markets",
        description="Scan prediction markets for probability shifts and consensus signals",
        tools=["polymarket", "polymarket_whales", "web_search", "web_browse"],
        examples=[
            "Fetch top Polymarket events in politics and finance, identify any with >70% implied probability",
            "Search for large Polymarket probability moves this week and cross-reference with news",
        ],
    ),
    GoalArm(
        name="insider_flow",
        description="Detect insider buying/selling signals from SEC Form 4 and Form 144 filings",
        tools=["insider_filings", "form144", "market_data", "web_search"],
        examples=[
            "Scan last 30 days of insider filings for buying clusters of 3+ insiders",
            "Check Form 144 sell-intent clusters and cross-reference with Form 4 buying",
            "Find companies where insiders are filing Form 144 to sell open-market-acquired shares",
        ],
    ),
    GoalArm(
        name="geopolitical_intelligence",
        description="Monitor geopolitical events and their market implications",
        tools=["gdelt", "market_data", "macro_data"],
        examples=[
            "Track conflict escalation in oil-producing regions",
            "Monitor sanctions events and cross-reference with commodity price moves",
        ],
    ),
    GoalArm(
        name="futures_positioning",
        description="Analyze CFTC futures positioning for crowding and flow signals",
        tools=["cftc", "market_data"],
        examples=[
            "Check managed money positioning in crude oil futures for crowding extremes",
            "Analyze gold futures COT data for producer/merchant accumulation signals",
        ],
    ),
    GoalArm(
        name="crypto_whale_flows",
        description="Monitor large crypto transfers for exchange inflow/outflow signals",
        tools=["whale_alert", "market_data"],
        examples=[
            "Scan BTC mempool for whale transactions above 50 BTC",
            "Check whale transfers for exchange inflow spikes indicating sell pressure",
        ],
    ),
    GoalArm(
        name="institutional_flow",
        description="Detect institutional short selling pressure and short squeeze setups from FINRA data",
        tools=["finra_short_volume", "market_data", "cftc"],
        examples=[
            "Check TSLA short volume ratio trend over the last 5 days for anomalies",
            "Scan all tickers for extreme short volume ratios today",
            "Get AAPL short interest and check days-to-cover for squeeze risk",
        ],
    ),
    GoalArm(
        name="energy_demand",
        description="Monitor power grid demand, fuel mix, and pricing for economic activity and energy stress signals",
        tools=["power_grid", "market_data", "macro_data"],
        examples=[
            "Check NYISO demand today for zone-level anomalies vs forecast",
            "Get current fuel mix and compare gas vs renewable generation share",
            "Analyze DA-RT LBMP spread for congestion and stress indicators",
        ],
    ),
    GoalArm(
        name="regulatory_pipeline",
        description="Monitor the US federal regulatory pipeline for upcoming rules, agency actions, and sector-specific regulation",
        tools=["regulatory_gazette", "web_search", "market_data"],
        examples=[
            "Check upcoming SEC proposed rules with open comment periods",
            "Search Federal Register for 'semiconductor' or 'crypto' regulation",
            "Scan all proposed rules from the last 7 days for market-moving regulation",
        ],
    ),
    GoalArm(
        name="weather_disruption",
        description="Monitor severe weather events and wildfires for supply chain and commodity disruption signals",
        tools=["weather_alerts", "market_data", "power_grid"],
        examples=[
            "Check for severe weather alerts near Gulf Coast refineries or Permian Basin oil",
            "Scan NASA FIRMS wildfire data for active fires near infrastructure zones",
            "Get weather disruption summary for commodity-relevant regions",
        ],
    ),
    GoalArm(
        name="seismic_risk",
        description="Monitor earthquake activity near critical infrastructure (semiconductor fabs, mines, ports, pipelines)",
        tools=["earthquake_proximity", "web_search", "market_data"],
        examples=[
            "Check recent earthquakes near TSMC fabs in Taiwan for supply chain risk",
            "Monitor seismic activity near Chilean copper mines for commodity impact",
            "Scan M5+ earthquakes in the last 7 days for infrastructure proximity alerts",
        ],
    ),
    GoalArm(
        name="transport_flow",
        description="Track US border crossing volumes (trucks, trains, rail containers) as trade throughput and economic activity proxies",
        tools=["transport_throughput", "ais_vessel_tracking", "market_data"],
        examples=[
            "Compare US-Canada vs US-Mexico truck crossing trends over the last 6 months",
            "Check latest month's rail container volumes for trade balance direction",
            "Get port-level truck data for Texas border crossings",
        ],
    ),
    GoalArm(
        name="defi_liquidity",
        description="Monitor DeFi protocol TVL, stablecoin supply, and DEX volumes for on-chain capital flow signals",
        tools=["defi_flows", "whale_alert", "market_data"],
        examples=[
            "Check top DeFi protocols by TVL for 24h change anomalies",
            "Monitor stablecoin supply changes for USDT/USDC mint or burn signals",
            "Compare DEX volumes across Ethereum vs Solana for capital rotation",
        ],
    ),
    GoalArm(
        name="government_spending",
        description="Track US federal contract awards for defense spending surges, contractor concentration, and policy signals",
        tools=["gov_contracts", "web_search", "market_data"],
        examples=[
            "Check top Department of Defense contract awards this quarter",
            "Search for federal contracts related to 'AI' or 'semiconductor'",
            "Scan recent large-dollar awards for contractor concentration patterns",
        ],
    ),
    GoalArm(
        name="research_pipeline",
        description="Track academic preprints and clinical trials for technology and pharma leading indicators",
        tools=["academic_preprints", "web_search", "market_data"],
        examples=[
            "Search arXiv for trending papers in quantum computing or AI safety",
            "Check ClinicalTrials.gov for Phase III completions by Pfizer or Moderna",
            "Monitor arXiv q-fin category for new quantitative finance research",
        ],
    ),
    GoalArm(
        name="sanctions_screening",
        description="Monitor OFAC SDN and UN Security Council sanctions lists for new designations, entity screening, and geopolitical escalation signals",
        tools=["sanctions_monitor", "gdelt", "market_data"],
        examples=[
            "Search OFAC and UN sanctions lists for entities named 'Huawei'",
            "Check for recently added sanctions designations in the last 30 days",
            "List active sanctions programs and entity counts by program",
        ],
    ),
    GoalArm(
        name="infrastructure_recon",
        description="Monitor Certificate Transparency logs and DNS record changes for infrastructure shifts, provider migrations, and corporate activity signals",
        tools=["cert_transparency", "dns_monitor", "web_search", "market_data"],
        examples=[
            "Search CT logs for certificates issued to openai.com",
            "Discover subdomains of palantir.com via certificate transparency",
            "Check DNS records for stripe.com — detect cloud provider and email infrastructure",
            "Run DNS diff on tesla.com to detect recent infrastructure changes",
        ],
    ),
    GoalArm(
        name="legal_filings",
        description="Monitor bankruptcy filings, SEC enforcement actions, and court proceedings for corporate distress and regulatory risk signals",
        tools=["bankruptcy_court", "insider_filings", "market_data"],
        examples=[
            "Scan PACER RSS for new Chapter 11 filings across major US courts",
            "Check SEC enforcement actions and litigation releases from the last 7 days",
            "Search 8-K Item 1.03 filings for recent bankruptcy notifications",
        ],
    ),
    GoalArm(
        name="sovereign_stress",
        description="Monitor government bond yields and cross-country spreads for fiscal stress, yield curve inversion, and sovereign risk signals",
        tools=["sovereign_debt", "macro_data", "market_data"],
        examples=[
            "Check US Treasury 2s10s yield curve spread for inversion signals",
            "Monitor Italy-Germany and Greece-Germany spread widening for eurozone stress",
            "Fetch Japan JGB 10Y yield for BOJ yield curve control breakout",
            "Compare sovereign spreads across peripheral eurozone countries",
        ],
    ),
    GoalArm(
        name="global_liquidity",
        description="Track global central bank balance sheets, net liquidity index, and cross-CB policy divergence for macro regime signals",
        tools=["central_bank_balance", "macro_data", "market_data"],
        examples=[
            "Compute global net liquidity index (CB assets minus RRP and TGA)",
            "Check which central banks are expanding vs contracting balance sheets",
            "Monitor Fed-ECB-BOJ policy rate differentials for carry trade signals",
            "Detect synchronized global tightening or easing across major central banks",
        ],
    ),
    GoalArm(
        name="investigation_signals",
        description="Monitor FOIA/FOI request patterns to detect investigation formation — surges in requests about an entity across agencies or jurisdictions signal upcoming disclosures",
        tools=["foia_requests", "web_search", "web_browse"],
        examples=[
            "Search FOIA requests mentioning Boeing for investigation formation signals",
            "Check if SEC is seeing a surge in FOIA requests (agency activity mode)",
            "Cluster FOIA/FOI requests about PFAS contamination across US and UK agencies",
            "Detect multi-agency investigation convergence on a pharmaceutical company",
        ],
    ),
    GoalArm(
        name="creditor_stress",
        description="Monitor creditor filings (SEC 8-K credit events, UK Companies House charges) for financial distress signals — filing surges and entity clusters precede credit downgrades",
        tools=["creditor_filings", "web_search", "insider_filings"],
        examples=[
            "Search SEC 8-K filings for credit facility changes mentioning Tesla",
            "Scan for creditor filing stress clusters in the last 30 days",
            "Check UK Companies House charges for Vodafone Group",
            "Detect entities with multiple recent credit-event 8-K filings",
        ],
    ),
    GoalArm(
        name="global_trade",
        description="Monitor bilateral trade flows via UN Comtrade — tariff exposure, supply-chain rerouting, and commodity chokepoints move currencies and equities",
        tools=["comtrade", "web_search", "macro_data"],
        examples=[
            "Check US-China bilateral trade flows for semiconductor commodities",
            "Track rare earth export patterns from China and Australia",
            "Compare ASEAN trade partner diversification trends over 2 years",
            "Detect rerouting of Russian energy exports via India and Turkey",
        ],
    ),
    GoalArm(
        name="labor_market",
        description="Track labor market dynamics via BLS JOLTS — quits/layoffs ratio, job openings, and market tightness signal recession probability and Fed rate decisions",
        tools=["job_postings", "macro_data", "web_search"],
        examples=[
            "Get latest JOLTS openings, quits, and hires data",
            "Check sector-level hiring signals across US industries",
            "Compute labor market tightness ratio (openings vs unemployed)",
            "Detect labor market softening from declining quits rate",
        ],
    ),
    GoalArm(
        name="construction_cycle",
        description="Monitor US building permits and housing starts — leading recession indicator with 12-18 month lead time, regional divergence reveals migration and credit stress",
        tools=["building_permits", "macro_data", "web_search"],
        examples=[
            "Get latest national building permit trends (single-family vs multi-family)",
            "Check regional permit divergence across US Census regions",
            "Compare housing starts to permits for builder confidence ratio",
            "Detect consecutive permit declines as recession early-warning",
        ],
    ),
    GoalArm(
        name="capital_flow_monitor",
        description="Track cross-border capital flows — foreign holdings of US Treasuries, net purchases, and FX reserves reveal coordinated de-dollarization, EM flow reversals, and reserve stress",
        tools=["capital_flows", "macro_data", "web_search"],
        examples=[
            "Check major foreign holders of US Treasuries for coordinated selling",
            "Monitor net foreign purchases of US long-term securities",
            "Detect EM reserve drawdown stress across China, Saudi, India",
            "Track flow reversals in official vs private capital movements",
        ],
    ),
    GoalArm(
        name="innovation_pipeline",
        description="Monitor patent filings to track global innovation pipeline — filing velocity acceleration, CPC class concentration shifts, and cross-company technology races",
        tools=["patent_filings", "web_search"],
        examples=[
            "Search recent AI/ML patent filings (CPC G06N) for technology leaders",
            "Analyze patent portfolio and filing velocity for a specific company",
            "Track semiconductor patent trends (CPC H01L) over the past 5 years",
            "Detect company pivoting into new technology areas via CPC distribution",
        ],
    ),
    GoalArm(
        name="lobbying_intelligence",
        description="Track US lobbying expenditure to detect upcoming regulation — abnormal spend spikes, new-issue lobbying, and coordinated industry campaigns precede policy changes by 12-18 months",
        tools=["lobbying", "web_search"],
        examples=[
            "Search lobbying filings by a specific company or lobbying firm",
            "Track annual lobbying spend trends for an industry",
            "Identify which companies are lobbying on healthcare regulation",
            "Detect abnormal lobbying spend increases signaling regulatory change",
        ],
    ),
    GoalArm(
        name="satellite_surveillance",
        description="Observe physical-world activity from space — NASA FIRMS thermal hotspots near refineries/factories reveal operational intensity, MODIS NDVI tracks crop health for agricultural commodity pricing, EONET natural events detect supply chain disruptions",
        tools=["satellite_activity", "weather_alerts", "web_search"],
        examples=[
            "Check fire/thermal hotspots near major oil refineries in the Gulf Coast",
            "Assess crop health via NDVI in the US Corn Belt during growing season",
            "Monitor active wildfires and volcanic events affecting shipping routes",
            "Detect industrial thermal activity changes in China manufacturing zones",
        ],
    ),
    GoalArm(
        name="electricity_demand",
        description="Monitor US-wide electricity demand, generation mix, and inter-regional flows via EIA — demand anomalies reveal economic activity shifts, fuel mix changes signal energy cost structure, interchange patterns expose grid stress",
        tools=["electricity_monitor", "power_grid", "macro_data"],
        examples=[
            "Compare electricity demand across PJM, ERCOT, and CAISO for regional divergence",
            "Track generation fuel mix shift toward renewables in California ISO",
            "Analyze inter-regional power flows to detect grid stress or surplus",
            "Identify demand-forecast deviation as unplanned industrial activity proxy",
        ],
    ),
    GoalArm(
        name="energy_infrastructure_pipeline",
        description="Track the US generator interconnection queue — planned and under-construction capacity reveals energy transition speed, data center buildout geography, and grid investment concentration",
        tools=["interconnection_queue", "electricity_monitor", "web_search"],
        examples=[
            "Survey planned solar and wind capacity pipeline across US states",
            "Detect suspected hyperscaler data center power projects in the queue",
            "Analyze battery storage buildout pipeline for grid reliability investment",
            "Compare planned vs under-construction capacity by fuel type nationally",
        ],
    ),
    GoalArm(
        name="pandemic_surveillance",
        description="Monitor global disease and pandemic signals — US wastewater pathogen concentrations (CDC NWSS, 6 pathogens), WHO outbreak declarations, EU surveillance data (ECDC), and genomic sequence velocity (NCBI) to detect emerging health crises before headlines",
        tools=["disease_surveillance", "weather_alerts", "web_search"],
        examples=[
            "Check CDC wastewater for multi-state surge in any pathogen",
            "Monitor WHO DON for novel pathogen outbreak declarations in last 30 days",
            "Track avian H5 wastewater detection rates across US states",
            "Compare NCBI genomic sequence submission velocity for H5N1 vs baseline",
        ],
    ),
    GoalArm(
        name="food_security_monitor",
        description="Track global food security via World Bank agricultural indicators — crop/food/livestock production indices, cereal yields, and food import dependency across 200+ countries to detect supply stress and food crisis vulnerability",
        tools=["food_security", "web_search"],
        examples=[
            "Compare food production indices for top wheat exporters over 5 years",
            "Check cereal yield trends for drought-prone Sub-Saharan African countries",
            "Identify import-dependent nations with food imports >30% of merchandise trade",
            "Track livestock production index divergence from crop production in India",
        ],
    ),
    GoalArm(
        name="political_risk_monitor",
        description="Monitor US political risk via FEC campaign finance — candidate filings, campaign cash on hand, and Super PAC independent expenditures to detect election spending surges and policy-relevant political uncertainty",
        tools=["political_risk", "web_search"],
        examples=[
            "Search for presidential candidates actively fundraising in 2024 cycle",
            "Track independent expenditure surge for/against Senate candidates",
            "Analyze Super PAC oppose-vs-support spending ratio by candidate",
            "Find recent campaign finance filings with largest cash-on-hand positions",
        ],
    ),
    GoalArm(
        name="internet_infrastructure_monitor",
        description="Monitor global internet outages, censorship, and connectivity — IODA BGP visibility and active probing for outage detection, OONI censorship measurements and ongoing incidents, RIPE Atlas probe connectivity, and normalized connectivity signals across 237 countries",
        tools=["internet_infrastructure", "internet_outages", "web_search"],
        examples=[
            "Check IODA for country-level internet outage alerts in the last 24 hours",
            "Monitor OONI censorship trends for Iran over last 30 days",
            "Get IODA normalized connectivity signal for Russia to detect BGP drops",
            "List all ongoing censorship incidents worldwide from OONI",
            "Compare WhatsApp and Telegram blocking measurements across authoritarian regimes",
        ],
    ),
    GoalArm(
        name="labor_disruption_monitor",
        description="Track US work stoppages and strike activity via BLS data — workers involved, idle days, and labor disruption trends that signal supply chain interruption, wage pressure, and sector-specific production risk",
        tools=["labor_disruptions", "web_search"],
        examples=[
            "Check recent US work stoppage activity for workers involved",
            "Analyze idle days trend from BLS strike data over last 4 years",
            "Get labor disruption overview combining workers and idle day metrics",
            "Assess labor disruption intensity ratio for inflationary signal",
        ],
    ),
    GoalArm(
        name="migration_flow_monitor",
        description="Monitor global migration and refugee flows via UNHCR displacement data, asylum decision trends, and World Bank remittance flows — detect humanitarian crises, policy shifts, and economic dependence on diaspora remittances",
        tools=["migration_flows", "web_search"],
        examples=[
            "Check UNHCR global displacement figures for refugees and IDPs",
            "Analyze asylum decision acceptance rates for Turkey",
            "Monitor World Bank remittance flows to Philippines for economic dependence signals",
            "Detect migration surges by comparing year-over-year displacement data",
        ],
    ),
    GoalArm(
        name="energy_supply_monitor",
        description="Monitor US energy supply via EIA petroleum stocks (crude, gasoline, distillate, SPR), weekly supply & disposition, and monthly rig counts — detect inventory surprises, supply tightening, and production leading indicators",
        tools=["energy_supply", "web_search"],
        examples=[
            "Check weekly crude oil stock levels for inventory surprises",
            "Monitor gasoline stock draws heading into summer driving season",
            "Track US rig count trend for future production signals",
            "Analyze petroleum supply & disposition for refinery input changes",
        ],
    ),
    GoalArm(
        name="treasury_receipt_monitor",
        description="Monitor US Treasury Daily Treasury Statement data — TGA operating balance, withheld income tax receipts, corporate tax deposits, customs duties, and public debt transactions for real-time fiscal and employment signals",
        tools=["treasury_receipts", "web_search"],
        examples=[
            "Check TGA operating cash balance for liquidity signals",
            "Monitor withheld income tax receipts as real-time employment proxy",
            "Track customs duty momentum for import volume signals",
            "Analyze public debt issuance pace for Treasury supply dynamics",
        ],
    ),
    GoalArm(
        name="drug_regulatory_monitor",
        description="Monitor FDA drug regulatory data — new approvals (NDA/BLA), FAERS adverse event spikes, label changes, and boxed warnings for pharma sector event-driven signals",
        tools=["drug_regulatory", "web_search"],
        examples=[
            "Check recent FDA drug approvals for pharma stock catalysts",
            "Monitor adverse event counts for safety signal spikes",
            "Detect new boxed warnings or label changes for prescribing impact",
            "Analyze approval rate by sponsor for competitive landscape shifts",
        ],
    ),
    GoalArm(
        name="global_pmi_monitor",
        description="Monitor OECD Composite Leading Indicators (CLI), Business Confidence (BCI), and Consumer Confidence (CCI) across 40+ countries — detect turning points, regime shifts, and cross-country growth momentum divergence",
        tools=["global_pmi", "web_search"],
        examples=[
            "Check CLI turning points for G7 countries",
            "Monitor US-China CLI divergence for macro regime signals",
            "Detect synchronized G7 CLI decline for global slowdown signals",
            "Analyze business confidence divergence from leading indicators",
        ],
    ),
    GoalArm(
        name="consumer_sentiment_monitor",
        description="Track consumer confidence and inflation expectations across EU and US — Eurostat country-level confidence, UMichigan sentiment, BLS CPI reality-check, and expectation gap analysis for cross-region divergence signals",
        tools=["consumer_sentiment", "macro_data", "web_search"],
        examples=[
            "Check Eurostat consumer confidence for EU27 and major economies",
            "Monitor UMichigan sentiment for recession-level pessimism signals",
            "Compare BLS CPI actual inflation with UMich 1yr expectations for gap analysis",
            "Detect transatlantic sentiment divergence between EU and US consumers",
        ],
    ),
    GoalArm(
        name="supply_chain_pressure",
        description="Track supply chain price pressure via BLS PPI and import price indices — semiconductor, steel, machinery, petroleum, and chemical producer prices plus import cost indices to detect cost-push inflation and sector margin squeeze",
        tools=["supply_chain_prices", "macro_data", "web_search"],
        examples=[
            "Check PPI trends for semiconductors and electronics for tech margin signals",
            "Monitor iron & steel and machinery PPI for capex inflation pressure",
            "Compute supply chain pressure index across all tracked sectors",
            "Compare import price changes vs domestic PPI for trade flow disruption",
        ],
    ),
    # --- Novel exploration arm (Change 8 — Tier 3) ---
    # The bandit treats this like any other arm.  When selected, the
    # orchestrator uses the LLM to generate an unconstrained goal using
    # any available tools.  Successful novel pulls are candidates for
    # promotion into permanent arms.
    GoalArm(
        name="novel_exploration",
        description="Open-ended exploration: pick any available tools and formulate a novel research goal that doesn't fit existing categories. The LLM is unconstrained — it should try creative, cross-domain combinations.",
        tools=[],  # empty = all tools allowed
        examples=[
            "Combine satellite imagery with shipping data to look for hidden commodity flows",
            "Cross-reference clinical trial failures with insider trading patterns in biotech",
            "Look for patterns between weather data and agricultural futures positioning",
        ],
    ),
]

# Minimum reward for a novel pull to be considered a success
_NOVEL_PROMOTE_REWARD_THRESHOLD = 0.6
# Number of successful novel pulls with similar tool signatures before promotion
_NOVEL_PROMOTE_MIN_SUCCESSES = 3


@dataclass
class ArmStats:
    """Statistics for a single arm — what the bandit has learned."""

    name: str
    alpha: float
    beta: float
    pulls: int
    total_reward: float

    @property
    def mean_reward(self) -> float:
        """Expected reward = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Variance of the Beta distribution. High = uncertain = explore more."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def __str__(self) -> str:
        return (
            f"{self.name}: mean={self.mean_reward:.3f} "
            f"pulls={self.pulls} total_r={self.total_reward:.2f} "
            f"α={self.alpha:.2f} β={self.beta:.2f} unc={self.uncertainty:.4f}"
        )


class StrategyBandit:
    """Thompson Sampling bandit over goal categories.

    Each arm has a Beta(α, β) distribution. To choose:
      1. Sample θ_i ~ Beta(α_i, β_i) for each arm
      2. Pick arm with highest θ_i

    To update after observing reward r ∈ [0, 1]:
      α += r
      β += (1 - r)

    Over time, arms with higher true reward rates accumulate
    higher α values → their Beta distribution shifts right →
    they get selected more often. This IS the learning.

    Persistence: saves α, β, pull counts to a JSON file so
    learning survives across sessions.
    """

    def __init__(
        self,
        arms: list[GoalArm] | None = None,
        persist_path: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._arms = {arm.name: arm for arm in (arms or DEFAULT_ARMS)}
        self._persist_path = persist_path
        self._rng = random.Random(seed)

        # Initialize Beta distribution parameters: uniform prior
        self._alpha: dict[str, float] = {name: 1.0 for name in self._arms}
        self._beta: dict[str, float] = {name: 1.0 for name in self._arms}
        self._pulls: dict[str, int] = {name: 0 for name in self._arms}
        self._total_reward: dict[str, float] = {name: 0.0 for name in self._arms}

        # Novel exploration history: list of {tools_used, reward, description}
        self._novel_history: list[dict] = []

        # Load persisted state if it exists
        if persist_path and persist_path.exists():
            self._load()

    def choose(self) -> GoalArm:
        """Select the arm with the highest Thompson sample.

        Each arm's Beta distribution is sampled once. The arm with the
        highest sample wins. Arms with few observations have high variance
        (exploration). Arms with many observations converge to their true
        mean (exploitation).

        Returns:
            The chosen GoalArm.
        """
        samples: dict[str, float] = {}
        for name in self._arms:
            # Sample from Beta(α, β) for this arm
            samples[name] = self._rng.betavariate(self._alpha[name], self._beta[name])

        best_arm = max(samples, key=samples.get)
        log.info(
            "Bandit chose '%s' (sample=%.3f). Samples: %s",
            best_arm,
            samples[best_arm],
            {k: f"{v:.3f}" for k, v in samples.items()},
        )
        return self._arms[best_arm]

    def update(self, arm_name: str, reward: float) -> None:
        """Update the arm's Beta distribution with observed reward.

        This is the actual learning step. After observing reward r ∈ [0,1]:
          α += r      (evidence of success)
          β += (1-r)  (evidence of failure)

        Args:
            arm_name: Which arm was pulled.
            reward: Observed reward, clamped to [0, 1].
        """
        reward = max(0.0, min(1.0, reward))

        if arm_name not in self._arms:
            log.warning("Unknown arm '%s', skipping update", arm_name)
            return

        self._alpha[arm_name] += reward
        self._beta[arm_name] += 1.0 - reward
        self._pulls[arm_name] += 1
        self._total_reward[arm_name] += reward

        log.info(
            "Bandit update: arm='%s' reward=%.3f → α=%.2f β=%.2f mean=%.3f",
            arm_name,
            reward,
            self._alpha[arm_name],
            self._beta[arm_name],
            self._alpha[arm_name] / (self._alpha[arm_name] + self._beta[arm_name]),
        )

        self._persist()

    def stats(self) -> list[ArmStats]:
        """Return statistics for all arms, sorted by mean reward descending."""
        result = []
        for name in self._arms:
            result.append(
                ArmStats(
                    name=name,
                    alpha=self._alpha[name],
                    beta=self._beta[name],
                    pulls=self._pulls[name],
                    total_reward=self._total_reward[name],
                )
            )
        result.sort(key=lambda s: s.mean_reward, reverse=True)
        return result

    def stats_report(self) -> str:
        """Human-readable report of bandit state."""
        lines = ["Bandit State (sorted by mean reward):"]
        total_pulls = sum(self._pulls.values())
        for s in self.stats():
            pct = (s.pulls / total_pulls * 100) if total_pulls > 0 else 0
            lines.append(
                f"  {s.name:25s}  mean={s.mean_reward:.3f}  "
                f"pulls={s.pulls} ({pct:.0f}%)  α={s.alpha:.1f} β={s.beta:.1f}"
            )
        lines.append(f"  Total pulls: {total_pulls}")
        return "\n".join(lines)

    def get_arm(self, name: str) -> GoalArm | None:
        """Look up an arm by name."""
        return self._arms.get(name)

    def is_first_pull(self, arm_name: str) -> bool:
        """Check if this arm has never been pulled before."""
        return self._pulls.get(arm_name, 0) == 0

    @property
    def total_pulls(self) -> int:
        return sum(self._pulls.values())

    # ------------------------------------------------------------------
    # Novel arm discovery (Change 8 — Tier 3)
    # ------------------------------------------------------------------

    def record_novel_pull(
        self,
        tools_used: list[str],
        reward: float,
        description: str = "",
    ) -> GoalArm | None:
        """Record the outcome of a novel_exploration pull.

        Stores the pull and checks if enough successful pulls with a
        similar tool signature exist to warrant promoting a new permanent arm.

        Args:
            tools_used: Which tools were actually used during execution.
            reward: Observed reward for this pull.
            description: Short description of what the novel goal was.

        Returns:
            A newly promoted GoalArm if promotion triggered, else None.
        """
        entry = {
            "tools": sorted(tools_used),
            "reward": float(reward),
            "description": description,
        }
        self._novel_history.append(entry)
        self._persist()

        if reward < _NOVEL_PROMOTE_REWARD_THRESHOLD:
            return None

        return self._check_promote(entry["tools"])

    def _check_promote(self, tools: list[str]) -> GoalArm | None:
        """Check if a tool signature has enough successes for promotion.

        Groups novel history by frozenset(tools_used), counts entries
        with reward >= threshold.  If count >= _NOVEL_PROMOTE_MIN_SUCCESSES,
        creates and adds a new permanent arm.
        """
        tool_key = frozenset(tools)

        successes = [
            h
            for h in self._novel_history
            if frozenset(h["tools"]) == tool_key and h["reward"] >= _NOVEL_PROMOTE_REWARD_THRESHOLD
        ]

        if len(successes) < _NOVEL_PROMOTE_MIN_SUCCESSES:
            return None

        # Build a name from the tool signature
        name = "_".join(tools) if tools else "general_exploration"
        # Avoid duplicates
        if name in self._arms:
            return None

        # Derive description from the successful pulls
        descriptions = [s["description"] for s in successes if s["description"]]
        desc = (
            f"Auto-discovered category from {len(successes)} successful novel "
            f"explorations using tools: {', '.join(tools)}. "
            f"Examples: {'; '.join(descriptions[:3])}"
        )

        # Use mean reward of successes as informative prior
        mean_r = sum(s["reward"] for s in successes) / len(successes)
        arm = GoalArm(
            name=name,
            description=desc,
            tools=list(tools),
            examples=descriptions[:3],
        )

        self.add_arm(arm, prior_reward=mean_r)
        log.info(
            "Promoted novel exploration → permanent arm '%s' (tools=%s, mean_reward=%.3f, n_successes=%d)",
            name,
            tools,
            mean_r,
            len(successes),
        )
        return arm

    def add_arm(self, arm: GoalArm, prior_reward: float | None = None) -> None:
        """Add a new arm to the bandit.

        Args:
            arm: The new GoalArm.
            prior_reward: If given, start with an informative prior
                Beta(1 + r, 1 + (1-r)) instead of uniform Beta(1,1).
        """
        if arm.name in self._arms:
            log.warning("Arm '%s' already exists, skipping add", arm.name)
            return

        self._arms[arm.name] = arm
        if prior_reward is not None:
            r = max(0.0, min(1.0, prior_reward))
            self._alpha[arm.name] = 1.0 + r
            self._beta[arm.name] = 1.0 + (1.0 - r)
        else:
            self._alpha[arm.name] = 1.0
            self._beta[arm.name] = 1.0
        self._pulls[arm.name] = 0
        self._total_reward[arm.name] = 0.0
        self._persist()

    @property
    def novel_history(self) -> list[dict]:
        """Read-only view of novel exploration pull history."""
        return list(self._novel_history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "arms": {
                name: {
                    "alpha": self._alpha[name],
                    "beta": self._beta[name],
                    "pulls": self._pulls[name],
                    "total_reward": self._total_reward[name],
                }
                for name in self._arms
            },
            "novel_history": self._novel_history,
        }
        self._persist_path.write_text(json.dumps(state, indent=2))

    def _load(self) -> None:
        try:
            state = json.loads(self._persist_path.read_text())
            for name, data in state.get("arms", {}).items():
                if name in self._arms:
                    self._alpha[name] = float(data["alpha"])
                    self._beta[name] = float(data["beta"])
                    self._pulls[name] = int(data["pulls"])
                    self._total_reward[name] = float(data["total_reward"])
            self._novel_history = state.get("novel_history", [])
            log.info(
                "Loaded bandit state: %d arms, %d total pulls, %d novel history",
                len(self._arms),
                self.total_pulls,
                len(self._novel_history),
            )
        except Exception as exc:
            log.warning("Failed to load bandit state: %s", exc)
