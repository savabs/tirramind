---
title: TirraMind — Project Memory
tags:
  - doc/memory
  - layer/surveillance
  - layer/world-model
---

# TirraMind — Project Memory

Persistent architectural knowledge. Keep this file current.

---

## Obsidian Knowledge Base

**The project root is an Obsidian vault** (`.obsidian/` at repo root). Every `.md` file is part of a single interconnected knowledge graph.

**Key conventions:**
- All `.md` files have YAML frontmatter with `title` and `tags`
- All cross-references use `[[wiki links]]`, never bare file paths
- Every research/spec/task file has a `## Related` section with `[[links]]` to the triad and topically related docs
- Tags follow a hierarchical taxonomy: `doc/*`, `phase/*`, `topic/*`, `layer/*`, `status/*`
- Graph View + backlinks + tag search are the primary navigation tools
- `python scripts/obsidian_linkify.py` auto-adds frontmatter, converts paths, and adds Related sections

**How to cold-start a session:** Read `[[agent_playground_doctrine]]` → `[[project_memory]]` → latest `doc/checkpoint` → follow `[[links]]` to the active task. Don't scan directories.

**Current graph stats (2026-04-07):** ~1,239 wiki links across 165 unique targets, 258 markdown files with frontmatter.

**Evaluated & archived research:**
- [[cc_cache_fix]] — Evaluated 2025-04-02, not useful for TirraMind (patches Anthropic's Claude Code CLI, not our codebase).
- [[learning_stack_spec]] — Stub spec for future learning stack work.

---

## Company identity (canonical)

**Canonical owner:** [[agent_playground_doctrine]] — read before planning product, niche, or hiring.

**TirraMind is an advanced self-improving agent company.** Markets are **playground #1** (first scoring environment for the agent). The core technical asset is a **learned cross-domain entity embedding space** (ℰ). **ML is the primary R&D path**; quant finance is implemented as differentiable constraints and readouts in `agent/quant/`, not as the firm’s identity.

**Archetype roles (one stack, not four companies):** Palantir = reality integration; DeepMind = ℰ + world model + alignment; Jane Street = deterministic systems truth; Renaissance = walk-forward economic proof.

**Commercial niche:** **N1 + N4 fused** — commodity/energy macro × sovereign/geopolitical state × **microstructure at every scale** (2026-06-02). Mathematical-field positioning, not sentiment. Spec: [[n1_n4_playground_spec]]. Global sensors unchanged.

**Parallel playgrounds:** Other agent projects (e.g. self-learning coding editor) share the same perceive → represent → act → learn loop but stay separate codebases until each playground closes its learning loop.

---

## This Is a Machine Intelligence Project

**TirraMind is a machine intelligence system. Not a chatbot. Not a wrapper. Not a dashboard.**

The bar is **agent + representation learning** at the depth of top ML labs, with **RenTech-grade evaluation** on the market playground — not “hire quants to hand-craft factors.”

**Expect SOTA-level work across every discipline that contributes to predictive edge:**

| Domain | What we mean by "SOTA" |
|--------|----------------------|
| **Mathematics** | Measure theory, stochastic calculus, information geometry, optimal transport, spectral methods, tensor decompositions — not textbook stats |
| **Machine Learning** | Foundation model architectures, neural ODEs/SDEs, meta-learning, causal inference, world models, reinforcement learning from scratch — not sklearn pipelines |
| **ML Engineering** | Distributed training, mixed-precision, efficient inference, data pipelines at scale, experiment tracking — production-grade, not notebook prototypes |
| **Software Engineering** | Clean architecture, type safety, correctness proofs where warranted, performance-critical paths in compiled code — engineering that doesn't break |
| **Physics** | Statistical mechanics (Boltzmann, Ising models for markets), field theory analogies, dynamical systems, chaos theory, renormalization — physics as a source of structural insight |
| **Biology & Complex Systems** | Evolutionary dynamics, predator-prey models (Lotka-Volterra for competing strategies), swarm intelligence, fitness landscapes, adaptation rate — markets are ecosystems |
| **Finance** | Microstructure, no-arbitrage theory, risk-neutral pricing, factor models, regime detection, cointegration, volatility surfaces — quant-desk level, every global market |
| **Geopolitics & Game Theory** | Bayesian games, mechanism design, signaling models, Nash equilibria applied to geopolitical actors, supply chain disruption modeling |
| **Information Theory** | Mutual information, rate-distortion, channel capacity as a lens on market efficiency, entropy-based feature selection |
| **Earth Observation & Physical Sensing** | Satellite imagery (SAR, multispectral, thermal), AIS vessel tracking, flight tracking, IoT sensor networks, seismology, weather modeling — observing reality directly, not reports about reality |

**This is not a "use an API and plot a chart" project.** Every component should reflect the question: *"Would a senior quant researcher or ML scientist at RenTech/DeepMind consider this serious?"* If the answer is no, the bar hasn't been met.

---

## The Observational Architecture: Model Reality, Not Markets

**Markets are outputs. Reality is the input. TirraMind observes reality.**

```
Layer 0: Physical reality (atoms, energy, weather, ships, factories, humans moving through space)
    ↓ generates
Layer 1: Human decisions (policy, trades, consumption, production, conflict, cooperation)
    ↓ generates
Layer 2: Information flows (news, filings, rumors, data releases, signals, prices)
    ↓ generates
Layer 3: Market prices (the scoreboard — what everyone stares at)
```

Most quant systems live at Layer 3 (fit model to prices). Smart ones reach Layer 2 (parse filings faster). TirraMind operates at **Layers 0 and 1** — observing the physical and behavioral reality that *generates* all downstream signals — and derives Layers 2-3 as consequences.

**Price is a symptom.** It's the last thing to move, not the first. A drought destroys Brazilian coffee crops (Layer 0) → farmers reduce supply commitments (Layer 1) → commodity traders adjust futures (Layer 2) → coffee prices spike (Layer 3). If you only see Layer 3, you're late. If you see Layer 0, you're early.

**The sensory surface must be global and complete.** Just as a brain integrates vision, hearing, touch, smell, and proprioception into a single world model — TirraMind must integrate every observable output of the global system into a coherent state representation. Not just US data. Not just market data. EVERYTHING:

| Sensory Domain | What it observes | Why it matters |
|---------------|------------------|----------------|
| **Global equities** | Every major stock exchange on Earth — NYSE, NASDAQ, LSE, TSE, HKEX, SSE, SZSE, BSE, ASX, Bovespa, JSE, etc. | Equity prices across countries reveal capital flow, risk appetite, and sector rotation at a global scale |
| **Fixed income & yield curves** | Government bonds across maturities for every major economy — US, Germany, Japan, UK, China, Australia, Brazil, India | Yield curve shapes encode growth expectations, inflation expectations, and central bank credibility — per country and cross-country |
| **FX & currency markets** | Every major and minor pair — DXY, EUR/USD, USD/JPY, GBP/USD, AUD/USD, emerging market FX | Currencies are the exchange rate between two economies' futures — FX reveals relative strength of entire economic systems |
| **Commodities** | Energy (WTI, Brent, nat gas, uranium), metals (gold, silver, copper, lithium, rare earths), agriculture (wheat, corn, soybeans, coffee, cocoa) | Commodity prices are the cost of physical reality — they reflect supply constraints, weather, geopolitics, and demand shifts before GDP reports |
| **Money markets & interbank rates** | SOFR, EURIBOR, SONIA, TONA, SHIBOR, OIS curves, FRA-OIS spreads | The plumbing of the global financial system — stress here propagates everywhere, and is visible before it hits headlines |
| **Central bank balance sheets** | Fed, ECB, BOJ, PBOC, BOE, RBA, SNB, Riksbank, BOK, RBI — assets, liabilities, repo operations | The single biggest driver of all risk assets. Global liquidity = the sum of what all central banks do. Not just the Fed. |
| **Volatility surfaces** | VIX, VSTOXX, VXEEM, individual equity/commodity/FX vol surfaces, skew, term structure | Vol surfaces encode the market's probability distribution over future states — the shape tells you what the market fears and expects |
| **Credit markets** | IG/HY spreads, sovereign CDS, corporate CDS, loan performance data | Credit is the canary in the coal mine — stress shows up in credit spreads before it shows up in equity prices |
| **Physical world sensing** | Satellite imagery (crop health, factory activity, military movement), AIS shipping data, flight tracking, weather, seismology | Layer 0 observations — seeing atoms move before anyone reports on it |
| **Trade & logistics flows** | UN Comtrade, container bookings, port throughput, customs data, trucking indices, rail traffic | The physical movement of goods through the world economy — reveals demand, supply chains, and bottlenecks in real-time |
| **Behavioral & social** | Hiring patterns, patent filings, lobbying spend, executive behavior, prediction market activity, dark pool flow | Layer 1 observations — human decisions that leak private information |
| **Geopolitical & institutional** | Government actions, military deployments, sanctions, trade policy, regulatory changes, election dynamics | Game theory in action — state actors making moves that reshape the economic landscape |
| **Crypto & on-chain** | BTC, ETH, major L1s/L2s, on-chain flows, wallet clustering, DeFi TVL, stablecoin supply | A 24/7 transparent financial system where every transaction is visible — the most information-rich market that exists |

**The principle: No blind spots.** Every observable output of the global system is a sensor. TirraMind should be wired to all of them — not because we'll use every one immediately, but because the *architecture* must assume that any data source might contain the next edge. The system discovers which signals matter; we don't pre-filter.

**Immediate vs. future scope:**
- **Now (Phase 1-2):** yfinance (global markets), FRED + ECB + BOJ + BIS (global macro/central banks), cache layer
- **Phase 3+:** Add physical world data (shipping, satellite, flight), behavioral data (filings, hiring, on-chain), geopolitical feeds
- **Agent-driven:** Eventually the agent itself discovers and integrates new data sources autonomously

---

## The Purpose Is Monetizable Predictive Intelligence

**The pure goal is monetization through superior prediction.** Everything we build exists to produce valuable predictive outputs that people or firms will pay for — not papers, not demos, not "interesting research." Intelligence without monetization is a hobby. This is not a hobby.

**The thesis:** Machine intelligence that is genuinely superior at prediction — across assets, across domains, across timescales — is one of the most valuable products you can build. The system's job is to find edge where humans can't, at a scale humans can't, and package that edge into decision advantage.

**Product hierarchy:** first build the core intelligence engine. That is the base asset of the business. Then build productized delivery layers on top of it for specific customer workflows, verticals, and operational decisions. The moat is the intelligence core; the delivery layer is how that core is captured economically.

**Commercial doctrine: sell the outcome through a tool-shaped product.** The customer should be buying decision advantage, monitored predictive coverage, probability updates, anomaly detection, regime-change alerts, and other concrete high-value outcomes. They may access that value through dashboards, APIs, alerts, workbenches, and embedded workflows, but the point is not generic model access. The model stack is internal production infrastructure and is allowed to change aggressively whenever a better model or workflow appears.

**Why this matters:** a tool company gets compressed every time foundation models improve. An outcome company gets stronger every time models improve, because delivery cost drops while the price remains anchored to the customer's value. Better models should widen gross margin, increase quality, and deepen the data moat.

**Long-term vision (locked):** [[long_term_vision]] — agent company, ℰ, ghost patterns, 5–10 year north star.

**Near-term income (active):** [[ghost_pattern_income_plan]] — micro-playground chain alerts (MP-1 Atlantic energy first); intelligence SKU, not API wrapper.

**Initial buyers (micro-niche GTM):** physical energy traders, ag specialists, soft-commodity boutiques per MP roster in [[ghost_pattern_income_plan]]. Quant/prop firm monitors remain year-2+ path.

**How this makes money:**

| Path | What | When |
|------|------|------|
| **Ghost pattern micro-playgrounds** | Chain alerts + public archive + paid Chain Brief per MP (G1–G4 gates). Phase A template engine before GNN gate. | **Now** ([[ghost_pattern_income_plan]]) |
| **Tool-shaped intelligence delivery** | Wrap the same engine in dashboards, telemetry, audit trails, recurring reports, alerting workflows, APIs, and customer-specific operating surfaces. These surfaces make delivery legible, habitual, and harder to replace because they embed the customer's workflow and accumulated context. | Layered on top of the core engine |
| **Custom intelligence programs** | Apply the same engine to customer-specific problems: geopolitical risk scoring, supply chain disruption probabilities, commodity flow prediction, strategic monitoring, internal workflows, and other forward-looking use cases. Price around business value and outcome sensitivity rather than raw seat count. | After initial wedge validates the delivery model |
| **Broader platform expansion** | Extend the same intelligence engine into wider enterprise and strategic-intelligence markets once the base product and delivery workflows are proven. Expose APIs or tools selectively when they reinforce the outcome contract rather than commoditize it. | After initial wedge validates the engine |
| **Trade our own edge later** | Deploy discovered strategies on our own capital if and when capital, execution infrastructure, and proven edge justify it. | Secondary path, after product and capital constraints change |

**Packaging rule:** avoid generic copilot products whose value depends on staying ahead of the next model release. Prefer tool-shaped intelligence delivery where the customer cares that the outcome arrives reliably and measurably, while the surrounding workflow, telemetry, and customer-specific memory make replacement non-trivial.

**Operational rule:** do the work manually long enough to learn the edge cases, collect the failures, and document the decision logic. Then automate the validated workflow. The data exhaust and process knowledge from delivery are part of the moat.

**What "edge" actually means here:**
- **Not backtested curves.** Edge means out-of-sample, risk-adjusted, transaction-cost-adjusted returns that survive regime changes.
- **Not one strategy.** Edge means a *machine that generates predictive products* — forecasts, early warnings, regime shifts, entity-level alerts, causal hypotheses, and, when useful, trading strategies.
- **Not what everyone else has.** The edge comes from fusion: combining signals across domains that siloed systems can't see (geopolitics → commodity flows → equity sector rotation → volatility surface shifts, or supply chain disruption → procurement risk → enterprise response). The machine sees the whole board.
- **Finding opportunities others miss.** The intelligence should scan across every asset class, every geography, every timescale, and every decision domain where forward-looking signal matters.

### Firm Identity: Technologically Advanced Predictive Intelligence

**TirraMind is not only a quant fund idea. It is a technology firm that finds data nobody else is looking at and applies the deepest math to extract predictive edge from it.**

The moat is not capital. It is not data subscriptions. It is the *combination* of:
1. **Finding unique, cheap/free information sources** that leak predictive signal (prediction markets, on-chain flows, regulatory filings, behavioral traces)
2. **Applying SOTA mathematics and computer science** to extract orders-of-magnitude more edge from that data than anyone else could
3. **Learning autonomously** which sources and methods produce the most alpha, and compounding that knowledge over time

This is a fundamentally different firm than one that buys Bloomberg + Refinitiv and runs regressions. Those firms compete on the same data with the same tools — alpha converges to zero. We compete on *what we choose to observe* and *how deeply we analyze it*.

**Cost discipline is a strategic advantage, not a limitation.** The most information-rich sources are often the cheapest:
- Polymarket whale tracking — **free** — leaks insider knowledge about real-world event outcomes
- SEC EDGAR insider filings — **free** — executives reveal private information through their trades
- On-chain wallet flows — **free** — every transaction on every blockchain is a public data point
- FRED/ECB/BOJ central bank data — **free** — the plumbing of global liquidity
- AIS shipping data (basic tier) — **free** — physical world observation before news reports
- NASA FIRMS fire data — **free** — satellite observation of physical disruptions
- Google Trends — **free** — attention shifts before price moves
- Government filings (patents, lobbying, FCC, FDA) — **free** — strategic intent revealed in regulatory paperwork

Expensive data (satellite subscriptions, real-time order flow feeds, alternative data vendors) is expensive *because* everyone already uses it — the alpha is competed away. **The asymmetric edge lives in the cheap, weird, overlooked sources that nobody else combines.** The system's job is to find those sources, apply rigorous math to them, and discover the causal chains that connect observable behavior to future market outcomes.

**The formula:** Unique observation × Advanced math = Asymmetric edge. Both halves are necessary. Unique data without math is just trivia. Math without unique data is commoditized. The combination is what makes us a different kind of firm.

### Workflow Rule: OSS and Documentation First

**Research external knowledge before coding.** For any new feature, unfamiliar technology, or externally inspired idea, the default workflow is:
1. Search GitHub for strong open-source repositories using multiple keyword variants and adjacent terminology.
2. Search authoritative documentation to understand the underlying API, concept, math, or integration surface.
3. Record the useful repositories, documents, search terms, and reuse constraints in `docs/research/<feature>.md` before implementation.
4. If a repository is incompatible with commercial use or its license is unclear, treat it as a concept source only. Capture the idea in research and reimplement it independently in TirraMind style.
5. Only then write specs, task steps, tests, and code.

This rule is part of TirraMind's DNA because it aligns with the firm's edge model: use OSS and public documentation as reconnaissance, extract the best concepts cheaply, and turn them into original, commercially safe implementations.

### Execution Rule: Move Fast Without Building Commodity Layers Too Early

- **Data first:** prioritize expanding the surveillance surface with high-value, cheap/free, under-watched sources.
- **Schemas now:** while building tools, enforce stable machine-readable outputs, lineage, timestamps, and normalized fields so later layers do not require rewrites.
- **Abstractions after coverage:** build higher-order context compression, execution modeling, factor objects, and world-model interfaces only after the raw evidence surface is broad enough to justify them.
- **Avoid the trap:** do not pause surveillance expansion to build Bloomberg-style dashboards, generic context terminals, or polished common-data workflows.
- **Decision test:** if a new task improves unique evidence collection or the mathematical use of that evidence, do it now; if it mainly improves presentation of common data, defer it.

### The Real Edge: Seeing What Nobody Else Sees

Traditional quant = analyze price/volume data with known models. That's table stakes. **The real alpha lives in the non-obvious** — the correlations, behaviors, and information leakages that people don't even think to look for.

**Concrete example: Polymarket insider tracking.** Prediction markets have whales who consistently bet correctly — because they have private information. If the machine can identify these wallets, track their betting patterns, and detect when they suddenly load up on a position, it has a window into the future that the price hasn't reflected yet. The bet itself is a data point about reality. This is the kind of edge we're building for.

**The principle:** It's not about analyzing data better — it's about **knowing which data to look at in the first place.** The machine should autonomously discover that:
- On-chain wallet behavior on prediction markets leaks information about real-world outcomes
- Unusual options flow in one sector predicts earnings surprises in a related sector
- Shipping vessel GPS patterns in the Strait of Hormuz predict oil volatility 3 days before news breaks
- A specific government official's travel schedule correlates with subsequent policy announcements
- Dark pool prints in a stock cluster reveal institutional accumulation before catalyst events
- Social graph changes between corporate executives precede M&A activity
- Patent filing patterns in a technology sector predict which company wins a government contract

**None of these are standard "quant signals."** They're adversarial intelligence — the machine treating the entire observable world as a data source and searching for information asymmetries. Most quant systems look where everyone else looks. TirraMind should look where nobody looks, and find the hidden causal chains that connect observable behavior to future outcomes.

**This requires a fundamentally different architecture than "fit model to OHLCV data":**
- **Autonomous data source discovery:** The agent doesn't just consume predefined datasets — it actively searches for new data sources that might contain edge (blockchain explorers, regulatory filings, satellite imagery providers, prediction market APIs, social networks, patent databases, shipping trackers)
- **Behavioral pattern detection:** Not just statistical correlations — detecting when an *agent* (person, institution, government) is acting on private information, and extracting that informational content
- **Causal graph inference:** Building causal models of how information flows through the world — who knows what, when do they act on it, and where does that action become visible before the market prices it in
- **Adversarial thinking:** Assume smart counterparties. The edge isn't in what's easy to find — it's in what's hard to find, or in combining easy-to-find things in ways nobody else combines them

### Information Asymmetry Taxonomy — Where Edge Actually Lives

The machine needs a structured understanding of *categories* of non-obvious information. This is the playbook for where to look. Each category represents a class of signals that most systems completely ignore.

#### 1. Behavioral Leakage (People reveal what they know through actions)

When someone has private information, they can't help but act on it. Those actions leave traces.

| Signal Class | What to detect | Why it matters |
|-------------|---------------|----------------|
| **Prediction market whale tracking** | Wallets that consistently win big on Polymarket/Kalshi suddenly loading positions | Someone with inside knowledge of an election/event outcome is betting. Their bet IS the signal about reality. |
| **Insider trading precursors** | Unusual options activity (especially OTM calls/puts) concentrated in time before events | SEC Form 4 filings are public but delayed. The options flow arrives first. Network analysis of who trades before whom reveals information chains. |
| **Executive behavior anomalies** | C-suite selling patterns, unusual stock lending, insider purchase clustering across related companies | An executive selling 80% of holdings 6 weeks before bad earnings isn't random. Cross-company clustering is even more telling. |
| **Lobbying spend shifts** | Sudden increases in lobbying expenditure targeting specific regulatory bodies | Companies lobby hardest right before regulation that affects them. Direction of lobbying + target agency = prediction of regulatory outcome. |
| **Legal filing patterns** | Patent applications, trademark filings, litigation pre-positioning | A pharma company filing very specific patents in a new therapeutic area tells you their pipeline 12-18 months before any announcement. |
| **Hiring signals** | Job postings by companies revealing strategic direction (new roles, new locations, new tech stacks) | A defense contractor hiring 200 satellite imagery analysts in a specific geographic specialty = contract win incoming. |

#### 2. Physical World Observables (Atoms move before bits)

The physical world generates information before the financial world prices it. Satellites, sensors, and logistics data are windows into reality.

| Signal Class | What to detect | Why it matters |
|-------------|---------------|----------------|
| **Vessel tracking (AIS data)** | Tanker congestion at ports, rerouting around conflict zones, unusual anchorage patterns | Oil tankers queueing at Ras Tanura = Saudi production changes. Tankers avoiding the Red Sea = insurance costs spike → shipping rates → inflation |
| **Satellite imagery** | Parking lot fullness (retail earnings), construction activity, crop health (NDVI), military buildup, factory output (heat signatures, smoke) | Walmart parking lots 15% fuller than last quarter on same dates = earnings beat. Chinese factory infrared output dropping = PMI miss incoming |
| **Flight tracking** | Private jet movements of CEOs, government officials, deal-makers | Two CEOs whose companies haven't been linked start flying to the same city repeatedly = M&A. Defense secretary flying to a region = geopolitical event |
| **Supply chain signals** | Container bookings, port throughput, customs data, trucking indices | If Chinese container bookings to US West Coast drop 30% and nobody has reported it yet, that's a leading indicator for retail |
| **Energy grid data** | Power consumption at facilities, data center energy usage in specific regions | A chip fab running at 110% power consumption = yield improvement = company ahead of street estimates |
| **Weather + agriculture** | Granular weather models applied to specific crop regions + storage facility monitoring | Combine county-level drought data with commodity warehouse receipts = predict grain prices before USDA report |

#### 3. Network & Graph Intelligence (Who connects to whom reveals intent)

Relationships between entities carry predictive information. Changes in the connection graph often precede changes in outcomes.

| Signal Class | What to detect | Why it matters |
|-------------|---------------|----------------|
| **Board interlock analysis** | Shared directors between companies, new board appointments, resignations | A director joining boards of two companies that could merge = early M&A signal |
| **Capital flow graphs** | VC fund investment patterns, LP commitments, fund-of-fund allocations | If three top VCs all invest in the same niche in Q1, that sector's public comps will re-rate in Q2-Q3 |
| **Regulatory revolving door** | Officials moving between government agencies and specific industries | Former SEC enforcement chief joining a law firm that represents crypto companies = regulatory stance shift |
| **Supply chain mapping** | Which company depends on which supplier, single-source dependencies | A fire at a sole-source supplier in Japan → identify every downstream company before the market does. This was literally the Fukushima trade. |
| **Political donation networks** | Who donates to whom, bundler networks, PAC structures | Donation pattern shifts before elections predict policy outcomes better than polls |
| **Academic publication graphs** | Which researchers collaborate with which companies, citation patterns in specific fields | A burst of publications co-authored by Company X researchers + university lab in a specific area = breakthrough incoming |

#### 4. Information Decay & Timing Arbitrage (Same info, different speeds)

Markets don't process information instantly. Different participants see the same information at different times and through different channels.

| Signal Class | What to detect | Why it matters |
|-------------|---------------|----------------|
| **Cross-market latency** | Event impacts on prediction markets vs. options vs. equities vs. credit | Polymarket prices in a political event in minutes; equity options take hours; credit markets take days. The machine can arbitrage the speed difference. |
| **Language/geography barriers** | Information published in non-English sources, local news in emerging markets, regional regulatory filings | A local Chinese newspaper reports a factory shutdown. English-speaking markets don't price it for 2 days. That's 2 days of edge. |
| **Expertise barriers** | Technical filings (FDA, patent offices, FCC spectrum auctions) that require domain knowledge to parse | An FDA filing uses specific regulatory language that implies approval probability changed from 40% to 80%. Biotech analysts figure this out in a week. The machine should figure it out in minutes. |
| **Structural market delays** | Index rebalancing announcements, ETF creation/redemption lags, settlement timing | When MSCI announces an index addition, the actual buying by index funds happens on a known future date. Everything between announcement and execution is a predictable flow. |
| **Attention arbitrage** | What the market is focused on vs. what actually matters right now | If every headline is about Fed rates but the real story is a drought destroying Brazilian coffee crops, the machine should be on the coffee trade, not watching the Fed with everyone else. |

#### 5. Meta-Signals (Signals about signals)

The most powerful edge comes from monitoring the behavior of OTHER prediction systems and information processors.

| Signal Class | What to detect | Why it matters |
|-------------|---------------|----------------|
| **Consensus crowding** | When too many quant funds hold the same position (factor crowding) | If every momentum fund is long the same basket, the unwind when it reverses will be catastrophic. Being short the crowd at the right time is edge. |
| **Model decay detection** | When a previously profitable signal starts losing power | If a signal we (or others) have been using starts degrading, that's information: either the opportunity was arbitraged away or the regime changed. Both are actionable. |
| **Sentiment divergence** | When retail sentiment and institutional positioning diverge sharply | Retail euphoria + institutional selling = top signal. Retail panic + institutional buying = bottom signal. The divergence IS the edge. |
| **Analyst behavior** | Timing of analyst upgrades/downgrades relative to price moves, herding patterns | Analysts who consistently upgrade AFTER a stock has already moved are lagging indicators. Analysts who move first (or whose silent periods predict) are leading. |
| **Order flow toxicity** | VPIN, Kyle's lambda, order imbalance metrics that detect informed trading in real-time | When order flow toxicity spikes, it means someone with private information is trading. The machine should detect this and figure out what they know. |

### Why This Thinking Is Core, Not Optional

This isn't a nice-to-have "alternative data" appendix. **This IS the project.**

The standard quant approach — take price data, fit statistical model, trade signal — is commoditized. Every fund with a Bloomberg terminal and a Python developer does this. The Sharpe ratio on commoditized signals converges to zero as more capital chases them.

**The entire value proposition of TirraMind is that it looks where others don't.** The machine intelligence isn't just faster or better at running regressions — it's operating on a fundamentally different information set because it autonomously discovers information sources and causal chains that humans haven't thought to look for.

**Priority ordering for the agent's intelligence development:**
1. **First:** Get basic data infrastructure working (price data, macro data) — this is Step 0-3 in our build sequence
2. **Simultaneously:** Design the agent's architecture so it can *discover new data sources on its own* — not just consume predefined ones
3. **Then:** Reward the agent for finding non-obvious correlations, not just fitting known models to known data
4. **Always:** The RL reward function should give EXTRA credit for edge that comes from novel data sources or novel combinations — pure price-based alpha is fine but information-asymmetry alpha is the real prize
5. **Cost-first:** Prefer free/cheap data sources that leak disproportionate signal. Expensive data is a last resort, not a first instinct. The smartest observation is the one nobody else thought to make, not the one you bought.

**The information edge hierarchy (most valuable → least):**
1. **Nobody else has this data** (e.g., you built a scraper that tracks a specific behavioral signal nobody monitors)
2. **Everyone has the data but nobody connects it this way** (e.g., combining shipping AIS data with options flow in commodity stocks)
3. **Everyone sees the connection but you see it faster** (e.g., parsing FDA filings in real-time with domain-specific NLP)
4. **Everyone does this but you do it better** (e.g., superior factor model) — this is where most quants live, and it's the LEAST valuable tier

**Cost-adjusted hierarchy:** Tiers 1-2 are also the CHEAPEST — nobody else monitors the data because they never thought to look, so it's unmonetized and often free. Tier 4 (better factor models) requires expensive feeds that everyone else also buys. The cheapest edge is the most valuable edge. This is not a coincidence — it's the core strategic insight.

**The compounding loop:** Edge → returns → capital → more data/compute → better intelligence → more edge. This is the flywheel. Every dollar earned funds the machine that earns the next dollar. The system should be designed from day one with this compounding in mind.

**Bottom line:** If TirraMind isn't on a trajectory to generate real money by selling predictive outcomes, decision support, productized intelligence delivery, or eventually direct trading performance, then something is wrong and we need to fix it.

---

## The Deep Surveillance Doctrine: Investigative Intelligence

**Motto: "We are secret journalists. Our job is to find data that has patterns — things that lead to things that not many people can see."**

Most data tools stop at the surface: fetch an aggregate, return a summary, move on. That is L1 work — the same layer every Bloomberg terminal covers. **TirraMind's mandate is to go into the dark ages of each tool** — drilling past the surface into entity-level resolution, cross-domain linkage, and temporal pattern extraction that nobody else performs.

### The Three Layers of Signal Depth

Every data source has three layers. Most systems touch only Layer 1. Our edge lives in Layers 2 and 3.

| Depth | Name | What It Looks At | Who Else Does It | Example |
|-------|------|------------------|-------------------|---------|
| **L1** | Aggregate | Top-line numbers, summaries, indices | Everyone with a terminal | "Fed balance sheet grew $50B this week" |
| **L2** | Entity-Level | Individual actors, transactions, filings, vessels, wallets | A few specialized desks | "This specific insider sold 80% of holdings across 3 related companies in the same week" |
| **L3** | Cross-Entity Combinations | Patterns that only emerge when you link entities across domains | **Nobody** — this is the moat | "A cluster of insiders at defense contractors sold while the same week a shadow fleet of tankers rerouted from sanctioned ports + CFTC large-trader positions in crude oil shifted — that combination predicts a sanctions regime change" |

### How Every Tool Must Evolve

Each of TirraMind's 31+ data tools must eventually support all three depth layers:

**L1 → L2: Go deeper inside the tool.** Don't just return aggregates — resolve to entities. Track those entities over time. Build time-series per entity.
- `insider_filings`: Don't just list filings. Build entity profiles — which insiders trade together? Who has the best track record? Cluster insiders by timing similarity.
- `whale_alert`: Don't just report large transactions. Build wallet identity graphs. Track wallet age, clustering behavior, exchange deposit patterns.
- `gdelt`: Don't just count events. Track specific actors over time. Build conflict escalation curves per actor-pair.
- `ais_vessel`: Don't just show ship positions. Build route deviation scores. Identify dark fleet vessels (AIS transponder gaps). Track port-to-port flow volumes by commodity type.
- `cftc`: Don't just report commitments. Track the *changes* in large-trader positioning at entity-category level. Detect crowding and positioning reversals.

**L2 → L3: Cross-domain entity linking.** The real patterns emerge when you combine entities across tools:
- **Shadow fleet detection:** AIS dark periods (vessel tool) + unusual crude oil CFTC positioning (commitments tool) + sanctions-adjacent country port calls (vessel tool) + insurance registry gaps (new data source) → identifies sanctions evasion before enforcement
- **Drug approval lifecycle:** ClinicalTrials.gov phase transitions (disease tool) + FDA filing language specificity (new parser) + insider trading at the pharma company (insider tool) + patent filing clustering in the therapeutic area (patent tool) → predicts approval probability months early
- **Industrial production proxy:** Energy grid consumption at specific facilities (grid tool) + hiring postings at those facilities (future tool) + trucking/rail volume near those facilities (future tool) + satellite thermal signatures (future tool) → reveals production changes before official reports
- **Geopolitical escalation timing:** GDELT event intensity curves (gdelt tool) + military flight tracking patterns (flight tool) + lobbying spend shifts at defense contractors (future tool) + UN voting pattern changes (future tool) → predicts conflict escalation timing

### The Entity Resolution Layer (Required Architecture)

To make L2 and L3 work, TirraMind needs an **entity resolution layer** that does not exist yet:

1. **Entity identity:** Map names, IDs, tickers, wallet addresses, vessel IMO numbers, CIK numbers, etc. to a unified entity graph
2. **Entity time-series:** Store per-entity observations over time (not just latest snapshots)
3. **Cross-domain linking:** Connect the same real-world entity across different data sources (e.g., a company's CIK in EDGAR = its ticker in market data = its vessels in AIS)
4. **Anomaly detection per entity:** Flag when an entity's behavior deviates from its own baseline — not just from a global average

### The Investigative Intelligence Principle

Every implementation decision about data tools should answer: **"Would an investigative journalist find this useful for uncovering something hidden?"**

- If the tool only returns what a Google search would find → it's not deep enough
- If the tool can't track a specific entity over time → it's not deep enough
- If the tool can't be combined with another tool to reveal a pattern → it's not deep enough
- If the output would surprise a domain expert who only reads headlines → you're getting somewhere

**The investigative journalist test applies to every new tool, every tool upgrade, and every feature engineering decision.** Surface-level data is infrastructure. Deep entity-level data with cross-domain linking is intelligence. TirraMind is an intelligence firm.

### Tool Depth Audit (as of 2026-04-07)

31 tools audited. Current depth ratings:

| Rating | Count | Examples |
|--------|-------|---------|
| **Deep (L2-capable)** | 11 | insider_filings, finra_short_volume, cftc, whale_alert, gdelt, bankruptcy_court, form144, ais_vessel, disease_surveillance, patent_filings, central_bank_balance |
| **Moderate (L1.5)** | 12 | polymarket, dark_pool, bond_market, options_flow, fred_macro, etc. |
| **Surface (L1 only)** | 8 | yfinance market data, web search, basic macro aggregates |

**Next phase:** Upgrade all 11 deep tools from L2-capable to L2-active (entity resolution + time-series tracking), then build L3 cross-domain combinators.

---

## Intelligence First, Data Second

**Data integration is not intelligence.** Fetching FRED data, aggregating it, and displaying a chart is ETL. Any Python script does this. Zero edge.

**Intelligence is what you do with the data:**

| Capability | What it means concretely | Not this |
|-----------|-------------------------|----------|
| **Regime detection** | Bayesian online changepoint detection, Hidden Markov Models inferring latent states from noisy observables | "Plot the Fed balance sheet and eyeball it" |
| **Causal inference** | Transfer entropy, Granger causality, do-calculus, instrumental variables — directional information flow, not correlation | `df.corr()` |
| **Spectral decomposition** | Separate multi-timescale structure — what's the 3-month cycle vs. the 3-year cycle in liquidity? | Moving averages |
| **Information-theoretic measures** | Mutual information for feature selection, entropy rate of price processes, channel capacity as a bound on market efficiency | "This R² is 0.3" |
| **Real-time streaming** | O(1) update per new data point via online algorithms; not O(n) recomputation from scratch | Rerun the whole notebook |
| **Compiled hot paths** | Performance-critical inner loops in Cython/Rust/C — microsecond latency where it matters | "Pure Python is fine" |
| **GPU-accelerated numerics** | Matrix operations, Monte Carlo, parameter sweeps — leverage hardware parallelism | Sequential loops |

**The bar:** Every mathematical operation TirraMind performs should withstand review by a senior quant researcher. If someone at RenTech would say "this is toy code," the bar hasn't been met.

**This means Phase 2 (Quant Primitives) is where TirraMind's real identity lives — not Phase 1 (data plumbing).** Phase 1 is necessary infrastructure, but the system's value comes from the mathematical and computational intelligence applied to that data.

---

## Strategic Focus: Liquidity Regimes (Market-Agnostic Edge)

**Problem:** We need a first target — something that exercises real math, is genuinely market-agnostic, and produces actionable edge. Three candidates were evaluated:

| Candidate | Verdict |
|-----------|---------|
| **Global Liquidity Regime Detection** | **CHOSEN** — uses FRED data we're building now, market-agnostic (equities/bonds/crypto/commodities all dance to the same liquidity conductor), requires real math (HMMs, changepoint detection, spectral methods), immediately tradeable (risk-on/risk-off positioning) |
| Cross-Market Information Propagation | Strong but depends on liquidity context — build second |
| Behavioral Leakage Detection | Strongest long-term edge but needs many data sources from day 1 — build third |

**Why liquidity is the right first target:**
1. **Every asset class responds to it.** Central bank balance sheets, reserve flows, dollar funding costs — when global liquidity expands, risk assets rise everywhere. When it contracts, everything falls. This is *the* market-agnostic signal.
2. **The data is free.** Fed balance sheet (WALCL), Treasury General Account (WTREGEN), Reverse Repo (RRPONTSYD), M2 (M2SL), foreign reserves — all FRED series. Our Phase 1 tools directly feed this.
3. **The edge is in construction, not data.** Everyone can see the raw series. Nobody aggregates them correctly in real-time with proper weighting, detrending, and regime inference. The intelligence IS the math, not the data.
4. **It's adversarial.** Most participants react to the *narrative* (what a Fed governor said). The machine reacts to the *plumbing* (what's happening in overnight funding markets). Plumbing leads narrative by days to weeks.
5. **It's the foundation for everything else.** You can't interpret cross-market propagation or behavioral anomalies without knowing what liquidity regime you're in. This layer contextualizes all future signals.

---

## Foundational Design Principle

**TirraMind is NOT an LLM wrapper.** The LLM is temporary scaffolding. The real product is a learning system that improves through trial and error and discovers strategies on its own — like AlphaZero, not ChatGPT.

---

## FIRST-CLASS PRIORITY: Always Trend Toward Learned

**Every implementation decision must be evaluated against: "Does this move toward more learned or more hand-coded?"** If a change adds hand-coded logic where a learned component could absorb the task, it needs explicit justification. The default direction is always toward more learning.

**The trajectory is non-negotiable:**

| Tier | % Learned | What Gets Learned |
|------|-----------|-------------------|
| Baseline | 25% | — |
| Tier 1-2 | 45% | World model parameters, surprise weights, loss weights, Kalman params |
| Tier 3 | 55% | Reward function, detector thresholds, goal space |
| Tier 4 | 65% | State representations, causal graph structure |
| Tier 5 | 75% | End-to-end gradient flow through the full pipeline |
| Tier 6 | 82% | Feature selection, tool routing |
| Tier 7 | 90% | Graph topology, scheduling, refit intervals |
| Tier 8 | 95% | Data source discovery, entity ontology |

**The residual 5% that stays hand-coded forever:** safety constraints, legal/regulatory rules, API plumbing, schema invariants, textbook equations. These are the only things that don't benefit from learning — everything else should be on a path to become learned.

**This is not aspirational — it is the project's primary architectural direction.** Building a self-improving machine intelligence system is TirraMind's richest and most valuable capability. Every phase of work should advance this trajectory. When in doubt between a hand-coded shortcut and a learned solution, invest in the learned solution.

See [[learned_vs_handcoded_architecture_spec]] for the full spec with Changes 1-16.

---

## FIRST PRINCIPLE: Mathematics and Physics, Not Sentiment

**The world has mathematical structure. Markets are emergent phenomena from deeper laws.**

Prices are not random walks perturbed by tweets. They are the output of a physical system governed by conservation laws (capital doesn't appear or disappear), information-theoretic constraints (Shannon entropy bounds how fast markets can price information), network dynamics (money flows through a graph), and phase transitions (regime changes are topological shifts in the system's free energy landscape).

**TirraMind models the GENERATING PROCESS, not surface observations.**

| The wrong way (shallow) | The right way (deep) | The physics/math underneath |
|------------------------|---------------------|---------------------------|
| "Sentiment is bullish → buy" | Why is sentiment clustering? Detect the regime shift in collective behavior | Ising model / mean-field theory — agents align like magnetic spins when coupling strength exceeds critical threshold. The phase transition IS the signal. |
| "X correlates with Y" | What is the CAUSAL direction? Does information flow X→Y or Y→X? At what lag? | Transfer entropy, Granger causality, do-calculus. Directed information flow, not symmetric correlation. |
| Moving average crossovers | Decompose the signal across timescales — what's noise, what's cyclical, what's structural? | Spectral methods: FFT, wavelets, empirical mode decomposition, Hilbert-Huang transform. Each timescale has its own generating process. |
| "BTC tracks M2 money supply" | How does the *distribution shape* of returns change as liquidity regimes shift? | Information geometry: KL divergence, Fisher information metric, optimal transport (Wasserstein distance). The manifold of probability distributions IS the state space. |
| Fit LSTM to price history | Model the stochastic process that GENERATES prices | SDEs, neural ODEs/SDEs, Fokker-Planck equations. The price path is a sample from a process; understand the process. |
| "Fear & Greed Index" | Markets have temperature (vol), energy (capital flow), entropy (information content). Phase transitions shift the free energy landscape. | Statistical mechanics: partition functions, Boltzmann distributions, renormalization group, critical exponents. |
| "Predict next candle" | What are the conserved quantities? What are the symmetries? What breaks them? | Noether's theorem adapted: every conservation law (capital, information) constrains the space of possible dynamics. Symmetry-breaking events (policy changes, supply shocks) are the signals. |

**Sentiment analysis, social media hype, "AI predicts price" — these are symptoms, not causes.** They're useful as INPUTS to the physics (they measure the coupling strength between agents, the temperature of the system, the information arrival rate). But they are never the MODEL itself.

**The hierarchy of understanding:**
1. **Laws** — conservation, entropy increase, information bounds (these never change)
2. **Structure** — network topology, feedback loops, causal graphs (these change slowly)
3. **Regimes** — which dynamical attractor the system is near (these change on medium timescales)
4. **Signals** — the data we observe (these change constantly)

Most quant systems operate at level 4. Good ones reach level 3. TirraMind must operate at levels 1-2 and let 3-4 fall out as consequences.

**Core identity:** An autonomous agent that learns to discover mathematical structure across heterogeneous data domains — and gets measurably better at it every time it runs. The agent itself embodies machine intelligence: its architecture, its learning algorithms, and its outputs should all reflect the SOTA bar defined above.

**Two sharply separated layers of work:**

| Layer | What | Who builds it | Success means |
|-------|------|--------------|---------------|
| **A — The Agent** | Learning infrastructure (memory, RL, world model) | Us (human + copilot) | Agent learns autonomously from experience |
| **B — The Training Ground** | A quant predictiveness system | The agent, on its own | Agent discovers mathematical structure in data without domain hand-holding |

**Quant predictiveness is TirraMind's first "Go".** The agent's job is to autonomously build a system that fuses heterogeneous data (market, economic, geopolitical, geographic) and discovers mathematical relationships with predictive power over asset prices. We do NOT inject domain knowledge or tell the agent which techniques to use. If the agent fails, we improve the learning infrastructure (Layer A), not give it answers.

**Why quant as the training ground:** Every signal domain (geopolitics, supply chains, game theory, technology) eventually manifests in asset prices. Predicting assets = fusing everything. This is the most general test of the agent's core capability — and the most direct path to making money.

**The LLM's role shrinks over time** as learned policies, accumulated experience, and the agent's own world model take over decision-making.

**Success criterion:** The agent can autonomously produce strategies with real, deployable edge — measured by out-of-sample risk-adjusted returns that survive transaction costs and regime changes. The system should not just find one strategy, but continuously discover, validate, and replace opportunities across asset classes. Ultimate success = the machine prints money.

---

## First Training Ground: Quant Predictiveness

**Goal:** Agent autonomously builds a system that finds mispricings and predictive structure in mixed data using advanced mathematics (cointegration, factor models, Bayesian inference, information theory, etc. — whatever the agent discovers works).

**Data sources (free):**

| Source | Data | Access | Scope |
|--------|------|--------|-------|
| Yahoo Finance (yfinance) | Equities, ETFs, commodities, FX, crypto — prices + volumes | Python library, no API key | **Global** — every major exchange (NYSE, NASDAQ, LSE, TSE, HKEX, ASX, BSE, Bovespa, etc.) |
| FRED (Federal Reserve) | US macro: rates, CPI, employment, GDP, Fed balance sheet, TGA, RRP, M2 | Free API key | US |
| ECB Statistical Data Warehouse | Eurozone macro: ECB balance sheet, EURIBOR, HICP, TARGET balances | Free API | Eurozone |
| Bank of Japan (BOJ) | BOJ balance sheet, JGB yields, TONA, Japan macro | Free CSV/API | Japan |
| BIS (Bank for International Settlements) | Global credit, cross-border flows, property prices, FX turnover, debt securities | Free bulk download | **Global** — the central bank of central banks |
| World Bank Open Data | GDP, trade flows, development indicators across 200+ countries | Free API | **Global** |
| IMF Data | World Economic Outlook, Balance of Payments, exchange rates, commodity prices | Free API | **Global** |
| OECD Data | Leading indicators, trade, productivity, composite leading indicators | Free API | OECD countries |
| UN Comtrade | International trade flows: who exports what to whom, quantities, values | Free API (rate limited) | **Global** |
| OpenSky Network | Global flight tracking (ADS-B) — aircraft positions, routes | Free API | **Global** |
| MarineTraffic / AIS | Vessel positions, port calls, shipping routes | Free tier (limited) | **Global** |
| USGS / Copernicus | Satellite imagery (Sentinel-1/2), land use, ocean temperature, NDVI | Free | **Global** |
| Web search (DuckDuckGo) | News, geopolitical events, sentiment | Already implemented | **Global** |
| Synthetic data | Known-structure pairs for validation | We generate | N/A |

**Scoring (reward signal for RL):**
- Walk-forward backtesting (train on window, test on next window, roll forward)
- Primary metric: Sharpe ratio (risk-adjusted return)
- Secondary: information ratio, max drawdown, hit rate
- Anti-overfitting: score ONLY on out-of-sample periods

**What the agent must discover autonomously:** which instruments are related, what mathematical model captures their relationship, how to estimate parameters, how to generate signals, how to validate without overfitting, how to incorporate macro data as regime filters.

---

## Build Sequence

**Two parallel tracks that converge:**
- **Track A: Quant Engine** — data pipelines, math primitives, signal generation, backtesting. This produces edge.
- **Track B: Agent Intelligence** — memory, learning loops, autonomous decision-making. This compounds edge.

We build the engine (Track A). The agent learns to drive it (Track B).
Only the NEXT phase is broken into atomic steps. Later phases stay high-level until they're next.

---

### Phase 0: Agent End-to-End ✅
Done. Agent runs, tools work, memory persists.

---

### Phase 1: Data Foundation ← CURRENT
Get real financial data flowing. Each sub-step is one change, one test.

- [ ] 1.1: Add `yfinance` to pyproject.toml
- [ ] 1.2: Create `agent/tools/market_data.py` — tool class skeleton (name, description, parameters schema)
- [ ] 1.3: Implement single-ticker OHLCV daily fetch (e.g., AAPL last 1 year)
- [ ] 1.4: Add period/interval parameters (1d, 1wk, 1mo; 1y, 5y, max)
- [ ] 1.5: Add multi-ticker support (fetch several tickers in one call)
- [ ] 1.6: Register market data tool in CLI tool setup
- [ ] 1.7: Smoke test — run agent with "get AAPL daily prices for the last year"
- [ ] 1.8: Add `fredapi` to pyproject.toml
- [ ] 1.9: Add FRED API key to config/settings.py
- [ ] 1.10: Create `agent/tools/macro_data.py` — tool class skeleton
- [ ] 1.11: Implement single-series FRED fetch (e.g., GDP, DFF, CPIAUCSL)
- [ ] 1.12: Register macro data tool in CLI
- [ ] 1.13: Smoke test — run agent with "get US CPI data for the last 5 years"
- [ ] 1.14: Create `agent/data/cache.py` — local file cache (save fetched data to disk, load if fresh)
- [ ] 1.15: Wire cache into both data tools (don't re-download if cached and recent)

---

### Phase 2: Liquidity Regime Detection (break down when Phase 1 done)
First real intelligence layer. Aggregate global liquidity composite (Fed balance sheet, TGA, reverse repo, foreign reserves, monetary aggregates). Changepoint detection (BOCPD, CUSUM). HMM-based regime classification (expansion / contraction / transition). Spectral decomposition across timescales. Cross-validate regime labels against actual asset class returns. The math here must be RenTech-grade — no toy implementations.

### Phase 3: Scoring & Validation (break down when Phase 2 done)
Walk-forward backtester, metrics suite, overfitting guards.

### Phase 4: Agent Autonomy (break down when Phase 3 done)
Register Phases 1-3 as agent tools, learning loop, strategy memory, first real autonomous run.

### Phase 5: Edge Compounding (break down when Phase 4 done)
Alternative data, cross-domain signal fusion, continuous strategy discovery, decay monitoring.

---

## Architecture Overview

**Identity:** Advanced agent company — learning infrastructure + embedding space ℰ — with **markets as playground #1**. See [[agent_playground_doctrine]].

**Stack:** Python 3.11+, OpenAI-compatible LLM (scaffolding), ChromaDB (planned), Pydantic, Rich CLI.

**Entry point:** `agent/cli.py` → `Orchestrator.run(goal)`

---

## Module Map

| Module | Responsibility | Key classes |
|--------|---------------|-------------|
| `agent/cli.py` | CLI + interactive REPL | `main()`, `build_tool_registry()` |
| `agent/config/settings.py` | Env-var config | `AgentConfig`, `LLMConfig` |
| `agent/core/orchestrator.py` | Agent loop: research → plan → execute → synthesize | `Orchestrator`, `AgentResult` |
| `agent/reasoning/llm_client.py` | OpenAI-compatible LLM wrapper | `LLMClient` |
| `agent/planner/task_planner.py` | Hierarchical LLM-powered task decomposition | `TaskPlanner`, `Task`, `TaskStatus` |
| `agent/memory/store.py` | Three-tier memory (episodic, semantic, working) | `EpisodicMemory`, `SemanticMemory`, `WorkingMemory` |
| `agent/tools/base.py` | Tool abstraction + registry | `Tool`, `ToolRegistry`, `ToolResult` |
| `agent/tools/web_search.py` | DuckDuckGo web search | `WebSearchTool` |
| `agent/tools/web_browse.py` | URL content extraction | `WebBrowseTool` |
| `agent/tools/code_executor.py` | Sandboxed Python execution | `CodeExecutorTool` |
| `agent/tools/shell_runner.py` | Shell command execution (with blocklist) | `ShellRunnerTool` |
| `agent/tools/file_manager.py` | File read/write/list | `FileReadTool`, `FileWriteTool`, `ListDirectoryTool` |
| `agent/tools/market_data.py` | yfinance market data | `MarketDataTool` |
| `agent/tools/macro_data.py` | FRED macro data | `MacroDataTool` |
| `agent/tools/liquidity_regime.py` | Liquidity regime query (HMM + BOCPD) | `LiquidityRegimeTool` |
| `agent/quant/liquidity.py` | US + global liquidity composite | `LiquidityComposite` |
| `agent/quant/changepoint.py` | BOCPD changepoint detection | `BOCPD`, `BOCPDResult` |
| `agent/quant/regime.py` | Gaussian HMM regime classification | `RegimeHMM`, `RegimeResult` |
| `agent/quant/spectral.py` | FFT + CWT spectral analysis | `power_spectrum()`, `scalogram()` |
| `agent/quant/scoring.py` | Performance metrics | `sharpe_ratio()`, `max_drawdown()`, `information_ratio()`, `hit_rate()` |

---

## Orchestrator Pipeline

```
Goal
 ↓
Research Phase — use tools to understand the problem space
 ↓
Plan Phase — LLM decomposes goal into hierarchical task tree
 ↓
Execute Phase — walk task tree depth-first, run tools, replan on failure (max 2 replans)
 ↓
Synthesize Phase — LLM produces final report from all completed task results
```

---

## Configuration

All config via environment variables (`TIRRA_` prefix):
- `TIRRA_LLM_PROVIDER` — "openai" | "ollama"
- `TIRRA_LLM_MODEL` — model name (default: gpt-4o)
- `TIRRA_LLM_BASE_URL` — custom endpoint
- `TIRRA_LLM_API_KEY` — API key
- `TIRRA_MAX_STEPS` — hard loop limit (default: 30)
- `TIRRA_MAX_PLAN_DEPTH` — hierarchical depth (default: 3)
- `TIRRA_MEMORY_DIR` — persistence path (default: .tirra_memory)

---

## Patterns & Conventions

- Tools inherit from `Tool` ABC, register in `ToolRegistry` via `cli.build_tool_registry()`.
- `ToolResult(success, output, data)` is the universal return type.
- LLM uses `structured_output()` for JSON, `ask()` for plain text, `decide_tool()` for function calling.
- Memory tiers: episodic (JSONL log), semantic (key-value facts), working (rolling LLM context).
- Task tree uses depth-first traversal for execution ordering.

---

## Development Workflow

This project follows a strict phased workflow (see `.github/copilot-instructions.md`):
1. **Research** → `docs/research/<feature>.md` (no code changes)
2. **Specification** → `docs/specs/<feature>_spec.md` (no code changes)
3. **Implementation** → follow spec, modify only listed files
4. **Task tracking** → `tasks/active/<task>.md`

---

## THE COMPUTATION STACK: From Data Pipes to Mathematical Intelligence

**As of Phase 5 completion, the system is a data-pipe-only architecture.** Tools fetch data, LLM reads it, produces text recommendations. This is a toy. The real system has 7 layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: SURVEILLANCE SURFACE (free APIs, physical sensors)    │
│  Polymarket │ EDGAR │ GDELT │ CFTC │ ADS-B │ AIS │ Grid │      │
│  Whale Alert │ DarkPool │ TRACE │ ClinicalTrials │ Patent       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: FEATURE ENGINEERING (quantitative signal extraction)  │
│  OFI │ VPIN │ Hurst exponent │ Transfer entropy │ MI │ Hawkes  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: WORLD MODEL (Bayesian network — causal graph)         │
│  Nodes: geopolitical tension, sector health, liquidity regime,  │
│  informed flow, macro cycle, credit stress, commodity supply    │
│  Edges: domain-knowledge causal links, learned CPDs             │
│  Evidence injection → belief propagation → posterior updates    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: SIGNAL FUSION (Kalman/particle filter)                │
│  Fuse noisy multi-source signals → optimal state estimate       │
│  Uncertainty quantification at every step                       │
│  Info-theoretic signal quality scoring (MI, transfer entropy)   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: RL POLICY + PORTFOLIO OPTIMIZER                       │
│  Model-based RL: simulate futures in world model, pick best     │
│  Kelly sizing │ Black-Litterman │ Robust optimization           │
│  Risk budget: VaR/CVaR constraints, max DD, correlation-aware   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: ADVERSARIAL INTELLIGENCE                              │
│  Manipulation detection (spoofing, stop hunting, pump & dump)   │
│  Edge decay monitoring │ Crowding risk │ Game theory             │
│  PIN model (Probability of Informed Trading)                    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 7: LLM (SUPPORT ROLE ONLY)                               │
│  Unstructured → structured │ Hypothesis generation              │
│  Narrative synthesis │ Report generation                        │
│  NEVER makes trading decisions — the math does.                 │
└─────────────────────────────────────────────────────────────────┘
```

**The LLM's demotion is deliberate.** The LLM is useful for parsing messy text (FDA filings, news articles, filing XML) and for generating hypotheses ("these anomalies might be connected because..."). But it NEVER decides what to trade or how much. That's the math + RL policy's job. The LLM is a tongue, not a brain.

---

## THE MATHEMATICAL CORE: What the Models Actually Are

### Stochastic Processes (Layer 0 — everything is a process, not a number)

| Math | What It Models | Why We Need It |
|------|---------------|----------------|
| **Hawkes processes** | Self-exciting point events — insider buys trigger more insider buys, conflict events escalate | Captures clustering/contagion. Intensity at time t = future event rate. Critical for timing. |
| **Lévy processes** | Fat-tailed jump distributions | Real markets have 10σ events weekly. Gaussian models miss this. Lévy captures the heavy tails truthfully. |
| **Fractional Brownian motion** | Long memory, persistence vs mean-reversion | Hurst exponent determines model class: H>0.5 = momentum, H<0.5 = reversion. Apply the wrong model class and you lose money. |
| **Jump-diffusion (Merton)** | Normal dynamics + sudden shocks | Models earnings surprises, geopolitical shocks, FDA decisions. Price doesn't just diffuse — it jumps. |
| **SDEs / Neural SDEs** | Stochastic differential equations, learnable drift and diffusion | Model the generating process of prices, not just the output. The price path is a sample from a process; understand the process. |

### State Estimation (Layer 1 — what's really going on behind the noise)

| Math | What It Does | When To Use |
|------|-------------|-------------|
| **Kalman filter** | Optimal linear state estimation from noisy observations | When multiple noisy signals point at the same hidden state (e.g., "oil supply disruption probability") |
| **Particle filter** | Non-linear, non-Gaussian state estimation via sequential Monte Carlo | When distributions are multi-modal or have fat tails — Kalman assumes Gaussian. Particle filter doesn't. |
| **HMM (Hidden Markov Model)** | Infer hidden regimes from observable data | We already use this for liquidity regime detection (agent/quant/regime.py). Extend to multi-factor regime states. |
| **BOCPD** | Bayesian online changepoint detection | We already have this (agent/quant/changepoint.py). Detects when the regime shifts in real-time. |

### Probabilistic Inference (Layer 2 — from evidence to beliefs)

| Math | What It Does | Why It's Critical |
|------|-------------|-------------------|
| **Bayesian Networks** | Directed causal graphs. Propagate evidence → compute posteriors. | The world model IS a Bayesian network. "If Iran tensions rise, what happens to oil?" is a query over the graph. |
| **Factor graphs + belief propagation** | More general than Bayes nets. Model complex interdependencies. | When the causal structure has loops or undirected connections. |
| **Granger causality + transfer entropy** | Does signal X actually predict Y, or just correlate? Which direction does information flow? | Kill false signals. If GDELT events predict oil but oil doesn't predict GDELT, the causal direction is one-way. |
| **Copulas** | Model tail dependencies between assets | Two stocks: 30% correlated normally, 90% correlated in a crash. Gaussian misses this. Clayton/Gumbel copulas capture asymmetric tail dependence. 2008 proved this matters. |
| **Mutual Information** | Non-linear dependency measure. Zero correlation ≠ zero dependence. | Feature selection: keep signals with high MI to target, discard those without. Catches relationships that correlation misses. |

### Action Optimization (Layer 3 — what to do given beliefs)

| Math | What It Does | When To Use |
|------|-------------|-------------|
| **Kelly criterion** | Optimal position sizing given edge and variance of edge | Every signal → Kelly-optimal bet size. Fractional Kelly (25-50%) for real deployment to account for model uncertainty. |
| **Black-Litterman** | Combine market equilibrium with subjective views (our signals) into portfolio weights | Our signals are "views." B-L blends them with market consensus, respecting uncertainty in each view. |
| **Robust optimization** | Optimize for worst case within uncertainty set | Don't optimize for the most likely future — survive the worst case. If P(crash)=5%, ensure portfolio survives that 5%. |
| **Convex optimization (cvxpy)** | Portfolio construction with constraints: max position, sector caps, turnover limits | Tractable, guaranteed-optimal solutions for constrained portfolio problems. |
| **Dynamic programming** | Multi-step optimal decisions: when to enter, scale in/out, exit | Finite-horizon stochastic control. Optimal execution given beliefs about future price paths. |
| **Model-based RL** | Simulate futures in world model, plan across simulated trajectories | The Dreamer/MuZero approach: don't just react to observations, IMAGINE future states and plan within them. |

---

## ADVERSARIAL INTELLIGENCE DOCTRINE

**Markets are adversarial environments. Every player optimizes against you.**

### Market Microstructure — Understanding the Machine

Markets are order books — physical systems with exploitable mechanics:

- **Order flow imbalance (OFI)**: If 70% of recent volume is buy-initiated, the ask side is being consumed. Price must rise mechanically.
- **VPIN**: Decomposes flow into informed vs uninformed. High VPIN = someone with private info is active. Predicted the 2010 Flash Crash hours early.
- **Liquidity cascades**: Price drops through stop-loss cluster → stops trigger → sell into next bid → triggers more stops. Deterministic if you know where the stops are.
- **Bid-ask spread behavior**: Spread widens before big moves because market makers detect informed flow and protect themselves.

### Manipulation Detection (Defense, Not Offense)

Understanding HOW manipulation works is defensive:

- **Spoofing**: Large orders placed and cancelled in milliseconds. Order-to-trade ratio anomaly detection.
- **Stop hunting**: Price pushed through known stop cluster, then immediately reversed. Detect via cascade model + reversal timing.
- **Pump and dump**: Social velocity spike + micro-cap volume surge + insider selling timing = pump in progress, dump incoming.
- **Wash trading**: Circular trades inflating volume. Graph analysis of counterparty patterns.

### Edge Decay & Competition

- Track signal Sharpe over rolling windows. Degradation = others found the same pattern.
- Crowding risk: if many participants hold the same factor exposure, unwind is catastrophic. Model via agent-based simulation.
- Adversarial robustness: can our signals be reverse-engineered from observing our trades? Design fusion-based signals that require access to the same multi-source data combination.

### Key Principles

1. **Output is ALWAYS a probability distribution.** Never "buy FMBM." Always "P(FMBM > $35 | 30d) = 0.72 [0.58, 0.83]."
2. **Position sizing flows from Kelly criterion** adjusted for model uncertainty (fractional Kelly) and portfolio correlation.
3. **The math decides. The LLM explains.** The Bayesian network, fusion layer, and RL policy generate the action. The LLM can narrate WHY, but it doesn't get a vote.
4. **Physics can't lie.** Prefer unforgeable physical traces (electricity consumption, jet flight paths, blockchain transactions, vessel positions) over human-authored disclosures that are delayed and can be manipulated.
5. **Every signal has a half-life.** No edge lasts forever. Build the machinery to discover new edge faster than old edge decays.

---

## DATA SOURCE TIERING (By Real-Time-ness, Not By Importance)

| Tier | Latency | Sources | Role |
|------|---------|---------|------|
| **T0: Real-time physical** | Seconds-minutes | ADS-B jets, AIS vessels, power grid, blockchain whale transfers | PRIMARY signals — unforgeable, pre-event |
| **T1: Near real-time behavioral** | Hours-days | GDELT events, dark pool volume, Polymarket odds, CBOE options flow, ClinicalTrials.gov status | ACTIVE intelligence — behavioral traces as they happen |
| **T2: Weekly positioning** | 1 week | CFTC COT, FINRA ATS aggregate, ICI fund flows | CONTEXT signals — what the big players are actually positioned for |
| **T3: Disclosure lag** | 2-45 days | Form 4, Form 144, congressional trades, 13F, FEC donations | CONFIRMATION signals — validates hypotheses formed from T0-T2 |
| **T4: Quarterly/structural** | 45+ days | 13F, patent filings, lobbying disclosure, job postings | STRATEGIC signals — long-term shifts in player behavior |

**The real edge is cross-tier fusion.** When T0 (jet tracker) + T1 (GDELT) + T2 (CFTC positioning) + T3 (insider cluster) all converge on the same thesis — that's a conviction level no single tier reaches alone.
