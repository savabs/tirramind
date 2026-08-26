---
title: "Strategy: Deep-Intelligence Roadmap, Pricing & Infrastructure Cost"
tags:
  - doc/research
  - topic/strategy
  - topic/product
  - topic/north-star
  - status/active
date: 2026-08-25
---

# Strategy: Deep-Intelligence Roadmap, Pricing & Infrastructure Cost

> **Purpose:** confront the honest truth — a $19/wk brief is too thin to justify a
> subscription. Define the path to deeper analysis and real intelligence, with
> concrete pricing tiers and a realistic infrastructure cost model.

---

## 1. The honest problem

We launched a **weekly contract-information brief**: a ranked list of government
contracts with EV scores. It works end-to-end (live data → scoring → webhook →
delivery). But it is:

- **Information, not intelligence.** It tells a buyer *what's out there*, not
  *what to do and why*.
- **Thin per-week cadence.** One brief/week over ~weeks of public data is a
  shallow artifact.
- **Easily replicated.** Scraping contracts + z-scores is commodity. Anyone can
  build it in weeks.
- **Not priced for the value.** $19/mo can't support the engineering or the
  infrastructure the intelligence really costs.

**Conclusion:** the brief is a *foot in the door* (proves the pipeline + a
real customer loop), NOT the product. The product must be **predictive
intelligence** — answers, forecasts, and decisions, not lists.

---

## 2. What "deep intelligence" actually means (the extension)

The north star is: **predictive intelligence** (math + finance + systems +
architecture). Deep analysis = moving up the value stack:

| Level | What it is | Example for a contract buyer |
|---|---|---|
| **L1 — Lists** (current) | Raw opportunities, scored | "These 10 contracts exist, EV-ranked" |
| **L2 — Analysis** | Explain *why*, add context | "This agency is ramping spend; 3 past winners of this NAICS; competitive set" |
| **L3 — Prediction** | Forecast outcomes, learn from history | "You have a 62% chance this bid wins at ≤X; the winning bid last quarter was ~$40k" |
| **L4 — Decision** | Recommend *action* | "Bid on this one. Here's your draft price. Do these 3 things to win." |
| **L5 — Autonomous** | Act on your behalf | Track, run the pipeline, adapt weekly to what wins |

**The gap to close:** L1→L3 is where the *model + learning* lives (the GNN /
world-model we parked). L3→L5 is where subscription value compounds and where
$20/mo becomes $200/mo.

---

## 3. The concrete capability stack to build (deep-intelligence roadmap)

Clustered by the infrastructure cost item each needs.

### 3a. Deeper data (more + better sources)
| Source | Data it adds | Cost | Cost model |
|---|---|---|---|
| **Free (current):** USASpending, CFTC, GDELT, AIS, 60 tools | contracts, positioning, geo, physical | **$0** | — |
| **SAM.gov opportunities (all set-asides + full NAICS)** | *actionable* solicitations, not just awards | **$0** (free API w/ account) | — |
| **FRED premium / more macro** | deeper macro context | ~$0–50/mo | free tier often enough |
| **Compustat / CapIQ / Refinitiv** | firm-level fundamentals + filings | **$500–5,000/mo** | enterprise licensing |
| **Real-time market data** (Polygon, etc.) | intraday anomalies | $50–300/mo | tiered per-call |

### 3b. Databases (storage for prediction + learning)
| Option | Fits | Cost/month |
|---|---|---|
| **Current: SQLite** (`.tirra_pipeline`, 123 MB now) | single-node, low cost | **$0** |
| **Postgres (managed, e.g. RDS/Neon/Supabase)** | multi-user, relational + growth | ~$10–50/mo |
| **Vector DB (pgvector)** | embeddings for semantic retrieval (when model is on) | included in Postgres |
| **Time-series (Timescale/ClickHouse)** | high-frequency anomaly data if we go intraday | $50–200/mo |
| **Object storage (S3) + theme: raw archives + model weights** | cheap durable blobs | ~$1–10/mo |

**Honest path:** stay on SQLite → move to **Postgres (Supabase/Neon, ~$20/mo)**
when >1 user. Add **S3** for model weights/archives (~$5/mo).

### 3c. Compute (inference + pipeline)
| Work | Current cost | Rented cost |
|---|---|---|
| Deterministic pipeline (fetch→score) | ~seconds/day, **$0** on a Mac | $5 small VPS |
| **API backend** (webhook + brief serving) | **$0** on Mac/VPS | $5–10 VPS |
| **Stats / EV / bandit** | CPU, **$0** | included |

### 3d. Training (the GNN / weight-based layer — the real cost)
| Path | Cost/month | Notes |
|---|---|---|
| **Kaggle free GPU** (already used) | **$0** | 30h/wk, enough for small HetTGN retrain |
| **Colab free / Pro** | $0 / ~$10 | |
| **Rent GPU per-hour** (RunPod/Vast — 4090) | ~$1–2/hr, ~$40–80/mo if steady | only for frequent retraining |
| **Always-on GPU VPS** | $50–500/mo | not needed at this model size |

**Honest:** the HetTGN models are 7–24 MB — *small*. They do NOT need a big,
always-on GPU. Weekly retraining fits **Kaggle free credits ($0)**. Rented GPU
only if we retrain daily or grow the model.

---

## 4. Pricing tiers (up the value stack)

Anchor: L1 lists (current) = free / ~$19; the intelligence layers charge more.

| Tier | Price | What it delivers (L-level) | Target buyer |
|---|---|---|---|
| **Free / Freemium** | $0 | Sample brief; L1 teaser (n=3) plus market-anomaly headline | Top-of-funnel |
| **Opportunity** (current) | **$29–49/mo** | L1–L2: full ranked contracts + *analysis* (agency context, competitor set, why) | Small businesses / indie ops |
| **Alpha** | **$149–299/mo** | L3: *learned P(win)* personalized to your firm, bid-price suggestion, weekly forecasting | Serious bids / consultancies |
| **Institutional / API** | **$500–5k/mo** | L3–L4: batch/daily signals, custom universes, firm-level intelligence API | Quant teams / agencies |

**Why this fixes the $20 problem:** $19 only buys the thin list. Real value (a
personalized win-probability + decision) justifies **$150+**. The tiers give a
growth ladder and price on *outcome value*, not data volume.

---

## 5. Realistic total infrastructure cost (steady-state, ~50 subscribers)

| Item | Monthly |
|---|---|
| Domain (amortized) | ~$1 |
| Hosting (static) | $0 |
| Backend VPS (API + pipeline) | ~$10 |
| Database (Postgres) | ~$20 |
| Object storage | ~$5 |
| **Data** (free tiers) | ~$0–50 |
| **Compute** (pipeline, CPU) | ~$10 |
| **Training** (Kaggle free GPU) | ~$0 |
| **Subtotal** | **~$45–95/mo** |

**At $49/mo × 50 = $2,450/mo gross** → after Paddle (~5%+$0.5) ~$2,250/mo →
minus ~$95 infra → **~$2,150/mo net** at 50 subscribers. Even 15 subscribers at
$49 (~$700/mo) covers infra with room.

**The expensive path (only if you commit to enterprise/institutional):**
+ $500–5k/mo data, + $50–500/mo GPU, + $50–200/mo time-series DB → total
**$600–5,700/mo** — justified only *after* the lower tiers prove demand.

---

## 6. Phased build plan (deep-intelligence extension)

**Phase 1 — Deeper analysis (L1→L2), weeks 1–3** *(low cost, free data)*
- Wire **SAM.gov opportunities** (actionable solicitations, not just awards)
- Add agency context + competitor set + "why this matters" to each opportunity
- Improve P(win) realism (feed actual SAM response/bid counts where available)
- Move SQLite → managed Postgres

**Phase 2 — Predictive (L2→L3), weeks 4–8** *(Kaggle GPU, ~$0)*
- Re-engage the **GNN / world-model** so cross-domain relationships produce
  learnable signal (the genuine moat, currently parked)
- **Learned per-firm P(win)** from realized bids + market regime
- Weekly forecast artifact (not just rankings)

**Phase 3 — Decision (L3→L4) + tiers, weeks 8–12**
- Bid-price recommendation + go/no-go scoring
- Launch **Alpha tier ($149+/mo)** with the personalized intelligence

**Phase 4 — Scale / API (L4→L5), beyond**
- Firm API ($500–5k/mo), autonomous daily pipeline
- Reassess paid data + GPU only when demand proves it

---

## 7. Bottom line

- **The $19 brief is a proof-of-concept, not the product.** Real value and
  pricing come from **L2–L3: analysis → prediction**, which is exactly the
  GNN / world-model / learning layer we built scaffolding for but parked.
- **Infra cost to run L1–L2 is ~$45–95/mo** (free data, cheap GPU). The
  expensive items (paid data, big GPU, time-series DB) are **optional, later**,
  driven by demand — not required to start.
- **Pricing must ladder:** $49 → $149 → $500+. The intelligence (not the list) is
  what a buyer pays for.

The next implementation after this plan: **Phase 1 — deeper analysis on free
data** (SAM.gov opportunities + agency/competitor context + Postgres). That
raises the ceiling immediately without spending on data or GPU.

## Related
- [[revenue_plan_2026-05-08]]
- [[checkpoint_2026-08-24_front_door_deploy]]
- [[checkpoint_2026-08-24_product_subscription]]
- [[quant_training_ground]]
