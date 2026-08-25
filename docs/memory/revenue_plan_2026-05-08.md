---
title: "Revenue Plan: Intelligence Brief to First Dollar (v2)"
tags:
  - doc/memory
  - topic/commercial
  - status/active
date: 2026-05-08
revised: 2026-05-08
---

# TirraMind: Market Intelligence Brief — Commercial Plan
## Version 2 — Revised after external critique, 2026-05-08

> **Superseded for GTM (2026-06-08):** Active income spec = [[ghost_pattern_income_plan]] (micro-playground ghost chains, not generic brief/API wrapper). Long-term north star = [[long_term_vision]]. Retain this file for tone rules, signal archive schema, and historical context.

> **Niche status (2026-06-02):** **Playground = N1 + N4 + microstructure.** Spec: [[n1_n4_playground_spec]]. **GTM deferred:** weekly brief / Substack / `generate_report.py` are UI layer — build **raw intelligence first** ([[n1_n4_microstructure_playground_task]]).

---

## The Core Reframe (read this first)

The v1 plan was technically correct but had a framing problem.

**Wrong framing:** "AI newsletter from a quant engineer."
**Correct framing:** "Market intelligence brief. Asymmetric observations. Verifiable signals."

This is not a semantic difference. It changes everything downstream:
- What the product is called
- Who it is written for
- How every sentence reads
- How it gets shared
- Who pays for it

The architecture is the engine. It is never mentioned externally.
What people buy is: **consistent asymmetric insight delivered sharply.**

A sophisticated system that produces generic-sounding output gets ignored.
A clear, confident observation about something unusual gets shared.

> "WTI positioning just hit a level seen only 4 times in 10 years."

That gets shared. "Our Bayesian convergence stack detected..." does not.

Think: intelligence analyst. Macro strategist. Anomaly hunter.
Not: ML researcher.

---

## Part 1: What We Have That Is Sellable Right Now

### The Sellable Layer (external description)

| Signal | External Description | Internal Source |
|---|---|---|
| Commodity positioning anomalies | When managed-money concentration in crude, gold, or grains hits multi-year extremes — before the move | CFTC Disaggregated COT, 5,080 observations |
| Regime change alerts | When market structure in a specific instrument breaks from recent pattern — historically precedes directional moves | Changepoint detection across 89 instruments |
| Cross-domain entity anomalies | When insider activity, regulatory filings, or capital flows in an entity cluster correlate with positioning extremes | Cross-domain entity graph, 2,502 entities, 11,915 links |
| Walk-forward signal audit | 3-year out-of-sample signal history, 40 folds, 89 instruments | `scripts/phase40_gnn_backtest.py` |

**Internal architecture is never mentioned externally.** The terms HetTGN, BOCPD, ICIR, SAC,
EWC, ListNet, Fisher diagonal — these exist inside the codebase. They do not exist in the brief.

### What We Are NOT Selling Yet

- A complete trading system (the GNN's standalone Sharpe is below EqualWeight)
- Real-time signals (the pipeline runs daily, not intraday)
- Guaranteed alpha (we publish historical patterns, not predictions)

### The Core Insight

We do not need a perfect model to make money.

Most successful financial intelligence products are not mathematically superior.
They are curated, consistently delivered, emotionally trustworthy, and framed well.

Our edge over existing products (COTBase, Barchart, SentimenTrader) is:
- Cross-domain context: positioning + entity graph + regime state simultaneously
- Verifiable sourcing: every claim traceable to a public data source
- No black box: readers can audit the raw CFTC report we pulled from

That combination does not exist as a packaged product anywhere. That is the moat at this stage.

---

## Part 2: The Product

### What It Is

**Name:** (TBD — do not call it a newsletter. Options: "The Tirra Brief", "Market Signal Brief",
"TirraMind Intelligence." Something that sounds institutional, not creator-economy.)

**Format:** Weekly market intelligence brief. 400–700 words. 1–2 charts. Source links.
Delivered via Substack but presented as a professional intelligence brief, not a newsletter.

**Cadence:** Weekly, published Friday morning (covers the full trading week).

**Tone:** Sharp. Minimal. Confident. No hedging language. No jargon. No architecture explanation.
Reads like a senior macro strategist wrote it, not a developer.

### What Each Issue Contains

**1. The observation** — one specific, unusual thing we found this week
> "Managed money net positioning on WTI Crude crossed +2.1 standard deviations above its
> 52-week mean. This level has occurred 4 times in the past decade. In 3 of those 4 cases,
> crude declined more than 8% over the following three weeks."

**2. The chart** — actual data, clean design, source cited at the bottom
> Source: CFTC Disaggregated Commitments of Traders, published 2026-05-06 (cftc.gov)

**3. The cross-domain context** — what the entity graph shows alongside the positioning
> "Simultaneously: 3 major commodity trading entities in our entity graph increased hedge
> positions in related instruments this week. Capital flow data for commodity-exporting
> countries shows net outflows."

**4. Prior call outcome** — last week's call, resolved
> "Issue #4 flagged GDX regime alert. GDX returned +4.2% over 7 days. Signal: ✓"
> "Running record: 7 calls, 5 confirmed directional, 2 inconclusive, 0 opposite direction."

---

## Part 3: The Real Target Customer

### Who Actually Pays First (revised)

The v1 plan said "quant traders and prop desk analysts." That was wrong for early stage.

Professional quant desks:
- Already have internal data infrastructure
- Distrust external signals by default
- Move slowly through procurement
- Will not pay until 12+ months of documented track record

**The real early adopter:**

| Segment | Description | Why They Pay |
|---|---|---|
| Finance-native internet people | Macro enthusiasts, advanced retail traders, people who follow geopolitical market dynamics on X | $9–19/month is trivial. They are already paying for 3 other signal services. They add this if it shows them something new. |
| Thematic / macro retail | Traders who think about commodity cycles, central bank policy, geopolitical risk | Our cross-domain signals (entity graph + positioning + regime) map exactly to how they think |
| Solo quant / indie algo trader | Running $50k–$500k, building their own system, looking for external signals to incorporate | They can evaluate signal quality technically. If CFTC anomaly has documented accuracy, they subscribe. |

**What they are NOT:**
- Passive investors
- People who want trading recommendations
- People who don't understand what a standard deviation is

**How to write for them:**

Assume they understand: standard deviation, positioning, regime, long/short, drawdown.
Do not explain these. Write at that level and they feel at home. Write below it and they leave.

### Secondary Customer (Month 6+): Junior Research Analysts

Prop desk juniors, boutique research associates, macro fund associates who need
to brief senior traders on positioning context. They would expense $49–99/month
without question if the brief saves them 2 hours of manual CFTC research per week.

### Tertiary Customer (Year 2+): Quant Firms

Will pay $500–5,000/month for custom signal delivery via API.
Requires 6–12 months of public documented track record.
The brief is the lead generation funnel for this customer — not the product they buy.

---

## Part 4: Trust Architecture

### The Problem With "Correct Calls"

Naive track record claims break down fast with sophisticated readers:
- "Gold rallied after signal ✓" — compared to what? Gold rallies most of the time.
- What was the false positive rate?
- Over what exact horizon?
- What would a random signal have done?

We need a more rigorous accountability system.

### The Signal Archive (most important asset in the business)

A public GitHub repository: `tirramind-signal-archive`

Contents:
```
signals/
  2026-05-08_WTI_CFTC_ANOMALY.json
  2026-05-15_GDX_REGIME_ALERT.json
  ...
SCORECARD.md   ← updated weekly, automated
```

Each signal file:
```json
{
  "issued_at": "2026-05-08T08:00:00Z",
  "instrument": "CL=F",
  "signal_type": "CFTC_POSITIONING_ANOMALY",
  "direction": "bearish",
  "sigma": 2.1,
  "source": "CFTC Disaggregated COT 2026-05-06",
  "source_url": "https://www.cftc.gov/dea/futures/deacmesf.htm",
  "evaluation_window_days": 21,
  "outcome_date": "2026-05-29",
  "actual_return": null,
  "baseline_return": null,
  "vs_baseline": null
}
```

`SCORECARD.md` tracks:
- Total signals issued
- Directionally correct (vs 0% baseline)
- Median return in direction vs median return of instrument in same period
- Hit rate by signal type
- Hit rate by instrument class

**Why this matters:** Immutable timestamps. Public. Anyone can audit. The timestamp on
a GitHub commit cannot be backdated. This is the only credibility mechanism that
sophisticated audiences fully trust.

### The Credibility Stack

| Element | What It Proves |
|---|---|
| Public GitHub archive with immutable timestamps | We cannot backfill or cherry-pick calls |
| Source link (cftc.gov URL) in every signal file | Every claim traceable to a public dataset |
| Baseline comparison in scorecard | Signal is better than random / buy-and-hold |
| Prior call outcome in every issue | Honest public accountability, week over week |
| No model description in external copy | Confidence. We describe what we see, not how we see it. |

---

## Part 5: Framing and Aesthetic

### This Matters More Than the Technology

The best intelligence products are not just accurate. They feel:
- Sharp
- Elite
- Minimal
- Confident
- Signal-dense

Your instinct toward premium engineering and tasteful complexity is useful here.
The brief should feel like reading a desk note from a senior macro strategist —
not like a Substack from a developer who built an AI thing.

### Language Rules

**Use:**
- "Positioning hit an extreme seen 4 times in a decade"
- "The cross-domain signal suggests..."
- "Historically, this pattern precedes..."
- "Resolved: +4.2% in 7 days. Signal confirmed."

**Never use:**
- "Our model detected..."
- "The AI flagged..."
- "Our algorithm shows..."
- "BOCPD", "GNN", "HetTGN", "ICIR", "EWC"
- "Bayesian", "neural network", "machine learning"
- Sentiment vocabulary: "bullish", "bearish", "fear", "greed", "market mood", "investors feel"
- Narrative hype without a number and a source

**Prefer (quant-desk tone, reader-facing):**
- "Managed-money net long hit +2.1σ vs 52-week distribution"
- "Regime shift: changepoint probability crossed 0.8 on 20d vol"
- "Country stress composite moved +1.4σ; sovereign spread proxy widened 12bp"
- "Positioning velocity (Δ net spec) at 90th percentile — micro shift before headline risk"

### Visual Rules

- Charts: clean, dark background or white background, no clutter, one message per chart
- Typography: consistent, minimal (Substack handles this adequately)
- No AI-generated images
- No stock photos
- The chart IS the content — make it self-explanatory

### Name / Brand

Do not use "newsletter" anywhere in the product description.
Candidates:
- "TirraMind Market Intelligence"
- "The Tirra Brief"
- "Asymmetric Signal" (generic but strong)
- "The Positioning Brief"

Decision: pick one before publishing issue #1. Do not change it after.

---

## Part 6: The Brief is the Bridge, Not the Business

This is the most important structural correction from v1.

The brief is:
- Trust engine
- Acquisition funnel
- Public track record generator

The brief is NOT:
- The long-term revenue model
- The moat
- The technical product

The actual business is structured signal infrastructure:
- APIs
- Datasets
- Signal feeds
- Alert systems
- Research workbench for quant teams

Every paid subscriber to the brief is a proof point that the signals have value.
That proof point is what eventually converts to API contracts and firm-level delivery.

**Treat the brief like a product trial, not a product.**

---

## Part 7: The Sales Mechanism

### You Document. You Don't Pitch.

Post the chart. State the observation. Show the source. Post the outcome next week.
Repeat. The track record sells itself.

### Channel 1: X — Primary

Format: one chart, 2–3 lines, historical rate, source.

> "Crude oil managed-money positioning: +2.1σ above 52-week mean.
> Last 4 times this occurred: crude fell 8%, 11%, 6%, 14% over 3 weeks.
> Full context in this week's brief 👇"

Tags: `#CFTC #commodities #crudeOil #macro #quant`

Do not over-explain. Do not mention the technology. The signal is the content.

### Channel 2: Reddit — Secondary

r/algotrading, r/quant, r/investing (for macro-level commodity calls)

Opening post: show the observation and the data. Let the community engage.
Answer every technical question honestly. If someone challenges your methodology,
engage with precision. That builds credibility faster than any marketing.

### Channel 3: Quant Communities — Tertiary

QuantConnect forums, Elite Trader, Nuclear Phynance Discord, Wilmott.

Post once with the methodology briefly described (no architecture jargon) + the first chart.
These communities will stress-test it. That is the point.

### What NOT to Do

- Do not explain the GNN to readers
- Do not describe the system as "AI-powered" — say "cross-domain signal fusion"
- Do not promise forward returns
- Do not post without a source link
- Do not skip the prior call outcome — this is non-negotiable

---

## Part 8: Technical Build

### The Minimum Viable Report Script

`scripts/generate_report.py` — reads `pipeline.db`, outputs formatted markdown.
~150 LOC. Three sections auto-generated; opening observation written manually.

**Section 1: CFTC Positioning Anomalies**
```
Query: latest week's managed_money_net per contract
Compute: z-score vs 52-week rolling mean and std
Flag: |z| > 1.5
Output: contract name, current net, σ level, direction, last 5 occurrence outcomes
```

**Section 2: Regime Change Alerts**
```
Query: changepoints detected in last 7 days from pipeline.db
Map: entity_id → instrument ticker
Output: instrument, trigger date, regime type, confidence
```

**Section 3: Entity Graph Anomalies**
```
Query: entity_types [company, cftc_contract, country]
       obs_type in [managed_money_net, insider_buy, capital_flow]
       observed_at > now - 7 days
Compute: deviation from 90-day rolling baseline per entity
Output: top 5 anomalous observations with human-readable context
```

**Section 4: Prior Call Resolver**
```
Read: .tirra_pipeline/signal_calls.json
For each call with outcome_date <= today and actual_return = null:
  Fetch forward return from price data
  Compute vs baseline (instrument mean return over same window)
  Write actual_return + vs_baseline back to the JSON
Output: resolved call table for the issue
```

### Signal Call Log Schema

Location: `.tirra_pipeline/signal_calls.json` (local) + `tirramind-signal-archive/` (public GitHub)

```json
{
  "issued_at": "2026-05-08T08:00:00Z",
  "instrument": "GLD",
  "signal_type": "CFTC_POSITIONING_ANOMALY",
  "direction": "bearish",
  "sigma": 2.3,
  "source": "CFTC Disaggregated COT 2026-05-06",
  "source_url": "https://www.cftc.gov/dea/futures/deacmesf.htm",
  "evaluation_window_days": 21,
  "outcome_date": "2026-05-29",
  "actual_return": null,
  "baseline_return": null,
  "vs_baseline": null
}
```

The local file is the working copy. The GitHub archive is the public record.
Both are updated together. The GitHub commit timestamp is the proof of prior art.

### Infrastructure

| Item | Platform | Cost | Time |
|---|---|---|---|
| Brief delivery | Substack | Free (10% of revenue) | 20 min |
| Domain | Namecheap | ~$12/year | 10 min |
| Signal archive | GitHub public repo | Free | 10 min |
| Charts | matplotlib (already installed) | $0 | In the script |
| Call log | JSON file + GitHub | $0 | 5 min |

**Total upfront cost: $12.**

---

## Part 9: Week-by-Week Execution

### Week 1

- [ ] Decide on product name (not "newsletter")
- [ ] Create public GitHub repo: `tirramind-signal-archive`
- [ ] Set up Substack with the chosen name
- [ ] Write issue #1 manually:
  - Pull latest CFTC data, find most anomalous contract
  - Look up last 5 historical occurrences of this σ level manually
  - Write the observation in plain language (no jargon)
  - Generate one clean chart
  - Link to the CFTC source report
- [ ] Commit the signal call to the archive with proper JSON
- [ ] Publish issue #1 as free
- [ ] Post chart to X — observation + historical rate + source link
- [ ] Post to r/algotrading: show the chart, explain the methodology briefly

### Week 2

- [ ] Build `scripts/generate_report.py` — automate CFTC section
- [ ] Write issue #2 (CFTC section automated, entity context written manually)
- [ ] Resolve week 1 call: fetch forward return, update JSON, post outcome to X
- [ ] Monitor: subscriber count, post engagement, any community questions

### Week 3

- [ ] Automate regime alert section
- [ ] Write issue #3 — majority automated, opening observation hand-written
- [ ] Update public GitHub scorecard with first 2 call outcomes
- [ ] Post scorecard link to X: "3 weeks of public signal log"

### Week 4

- [ ] Assess: ≥50 free subscribers → switch archive to paid at $9/month
- [ ] < 50 subscribers → post signal observations on X daily (not just weekly)
  The issue may be distribution, not signal quality. Test more posts.

### Month 2

- [ ] Full automation of report generation
- [ ] Evaluate: add insider filing anomalies section (already in the pipeline)
- [ ] Watch: which posts get the most engagement? Weight that signal type heavier.

### Month 3+

- [ ] ≥100 paid subscribers → start designing RapidAPI endpoints
- [ ] 0 paid after 60 days → the problem is packaging, not pipeline. Audit tone, framing, channels.

---

## Part 10: Stage Roadmap

### Stage 1: Brief (Now → Month 6)
**Goal:** Prove signal usefulness. Build attention. Build retention. Build willingness to pay.
**Revenue:** $0 → $3,800/month at scale

### Stage 2: API (Month 4–8, after demand proven)

3 FastAPI endpoints wrapping the same signals:
```
GET /v1/cftc/anomalies?week=latest&sigma_threshold=1.5
GET /v1/regime/alerts?days=7
GET /v1/entity/anomalies?entity_type=company
```

Host: Hetzner VPS ($5/month). Listed on RapidAPI marketplace.
Pricing: free tier (5 calls/day) / $49/month (500/day) / $199/month (unlimited).
**Revenue ceiling: $1,000–$10,000/month**

### Stage 3: Firm-Level Signal Delivery (Year 2)

Custom signal packages for quant firms. Daily delivery via S3 or SFTP.
Custom instruments. Entity graph exports. Workflow integration.
**Revenue: $500–5,000/month per firm, 3–10 firms → $15,000–$50,000/month**

Prerequisite: 6+ months of public track record + API infrastructure + GNN ICIR > 0.40.

### Stage 4: Intelligence Platform (Year 3+)

Full TirraMind vision. Research workbench. Real-time alerts. Entity risk terminal.
Gated on model quality proof and sufficient observation history per entity class.

---

## Part 11: Parallel Technical Tracks

The model improvement work is not blocked by the brief. They run simultaneously.

| Track | What | Timeline |
|---|---|---|
| Intelligence brief | Launch | Now |
| GDELT Goldstein filter | Remove noise, improve signal quality | 1 week |
| Kaggle GNN retrain | Epochs 21–40, ListNet + filtered graph | 2–3 weeks |
| `embedding_snapshots` table | Ghost pattern detector foundation | 2–3 weeks |
| Phase 48 (world model upgrade) | Gated — needs density audit + IC proof | 6+ months |

When the model improves, the signals improve. Subscribers don't see the internal work.
They see more accurate calls. That is the correct abstraction boundary.

---

## Summary

| Item | Answer |
|---|---|
| **Product name** | TBD — not "newsletter." Something institutional. |
| **Format** | Weekly intelligence brief, 400–700 words, 1–2 charts |
| **Platform** | Substack (delivery) + GitHub (public signal archive) |
| **Price** | Free for 4 weeks → $9/month |
| **Trust mechanism** | Public GitHub signal archive, immutable timestamps, baseline comparison, source links |
| **Target customer (early)** | Finance-native internet people, advanced retail traders, macro enthusiasts |
| **Target customer (later)** | Junior research analysts, eventually quant firms |
| **Sales mechanism** | Post the observation publicly. Document the outcome. Repeat. |
| **Time to first dollar** | 4–6 weeks |
| **Technical build** | 150-LOC report script + signal log JSON + Substack + GitHub archive |
| **Upfront cost** | $12 |
| **The brief is** | Trust engine + acquisition funnel + public track record. Not the final product. |
| **The actual business** | Structured signal infrastructure: APIs, feeds, firm-level delivery |

**The key discipline:**
Do not delay launch waiting for better model performance.
Do not explain the architecture to readers — ever.
Do not call it a newsletter.
Ship the observation. Document the outcome. Build the record publicly.
First revenue unlocks everything else.

## Related

- [[long_term_vision]] — locked north star
- [[ghost_pattern_income_plan]] — active GTM (supersedes this doc's product shape)
- [[quant_training_ground]] — technical roadmap
- [[tirramind_structure]] — canonical metrics
