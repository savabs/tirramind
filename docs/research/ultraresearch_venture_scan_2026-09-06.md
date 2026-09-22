---
title: "Ultraresearch venture scan — synthesis (2026-09-06)"
tags:
  - doc/research
  - topic/business-readiness
  - status/current
---

# Ultraresearch venture scan — synthesis (2026-09-06)

Decision-grade report. Inputs: 13 merged finalists, each attacked by two independent skeptics (reachability/comparable lens, moat/crowding lens), plus 27 merge-step drops and ~90 sub-fields swept empty across finance, quant-risk, AI infra, databases, low-level systems, vision, energy, OSINT, devtools, crypto and scientific compute. Method caveat carried from the agents: WebSearch budget was exhausted mid-run, so most verification is direct fetches of pricing/about/GitHub/PyPI pages; where a figure could not be fetched it is labelled inferred or unverified below, not asserted.

---

## 1. Verdict

The scan says one thing clearly: for this owner, a multi-million self-serve business with an engineering moat exists in exactly one shape, and it is not the shape the owner's instincts (developer infrastructure, GPU engines-as-API, energy/OSINT intelligence) point at. It is a **card-paid prosumer product for people risking their own money — options traders, systematic traders, DIY retirees, indie quants — who already pay $25–150/month to a UI-first incumbent whose numerics are demonstrably sloppy**, sold on correctness the buyer can check against a public reference (CBOE/Deribit marks, IRS worksheets, a published conformance suite). Every developer-facing "engine" idea died for one of two reasons, with almost no exceptions: the hardware or platform vendor gives the engine away free (NVIDIA cuOpt, NVIDIA-maintained sccache nvcc support, COLMAP/GLOMAP, gsplat with Meta/NVIDIA contributors, Hugging Face trackio, DOE-funded pvlib, OpenSanctions' MIT matcher), or the buyer who values correctness buys through procurement and pedigree (banks, energy operators, compliance teams, survey firms). Every energy, OSINT and intelligence idea failed on the second. The honest revenue ceiling the scan could *verify* for a ≤5-person, no-sales team is **$1M/yr** (ProjectionLab, Sidekiq, Plausible); every $3M+ small-team figure in the set (AmiBroker, Agisoft, OptionStrat) is inferred or behind a 403. The owner should plan for $1M as the credible three-year outcome of any single product, and treat $3M as requiring two products on one shared GPU kernel — which, as it happens, the top-ranked candidates permit.

---

## 2. TOP 5 — ranked

### #1 Surface-Correct Options Lab (equity + crypto, BYO feed, open-core GPU kernel)

**What it is.** A free pip GPU/SIMD kernel (the voltorch successor) that fits arbitrage-free SVI/SSVI surfaces and greeks for whole option chains, wrapped in a card-paid strategy builder / backtester / P&L-attribution terminal. Equity users paste a ThetaData/Polygon/broker key; crypto users get free Deribit/OKX/Bybit public books, which is why crypto is the launch wedge.

**Who pays and how they find it.** Retail/prop options traders (0DTE, condors, wheel, calendars) and crypto-options traders already paying OptionStrat ($39.99/$99.99), Option Omega (from $24.99) or Laevitas ($50). Found via YouTube/Discord/X, "options backtester" search, GitHub stars on the kernel, and the owner's write-ups ("why your backtester's greeks are wrong"). 7-day trial, card. Note the skeptic's warning: incumbents' traffic is 80–85% *direct* (habit/brand), so the owner must earn discovery through content, not inherit search demand.

**Verified comparable.** OptionStrat — LinkedIn lists 1 employee, privately held; ~400k visits/month, 10:38 avg session, self-serve tiers with 7-day trials (https://www.linkedin.com/company/optionstrat, https://www.similarweb.com/website/optionstrat.com/, https://r.jina.ai/https://optionstrat.com/membership). Revenue is **not** public; $1M+ is arithmetic from traffic × tiers. The founder name "Heath Milligan" in the finalist is unverified (Crunchbase 403). Option Omega: 2–10 staff, Knoxville, self-serve (https://www.linkedin.com/company/option-omega).

**Engineering moat.** Arbitrage-free surface fits (calendar + butterfly constraints) on sparse crypto wings and illiquid equity wings at every bar; American early exercise with discrete dividends; assignment/pin risk and bid-ask fill modelling; greek-based P&L attribution that reconciles to the cent; calibrated (not iid-normal) probability cones; multi-year 1-minute backtests in seconds via batched CUDA pricing. The only company selling this exact axis (Vola Dynamics) is institutional and sales-led (https://voladynamics.com/); no prosumer incumbent competes on numerics — OptionStrat's own footer says its calculations "are estimates". No maintained GPU-batched chain-scale arb-free fitter with a paid product on top exists on GitHub (QuantLib 7.6k stars, RustQuant 1.8k, py_vollib 434 are libraries only).

**Honest kill risk.** Retail buyers have never been shown to pay for correctness over UX; OptionStrat's 3,270 reviews praise UI, price and "chance of profit", not greeks. OptionStrat already ships net greeks, per-expiry IV and probability distributions, so "greeks" is not the hook — surface correctness and attribution are, and those are harder to see on a pricing page. The crypto wedge is smaller than the finalist claimed: Laevitas has 17 staff, a $50 tier with strategy builder + backtesting, and only ~7k visits/month (https://www.laevitas.ch/pricing, https://www.similarweb.com/website/laevitas.ch/). Equity data gravity was overstated — ThetaData sells 1-min OPRA history to individuals at $40–160/mo (https://www.thetadata.net/pricing), so BYO cost is one more subscription, not >$800/mo — but BYO still halves conversion.

**Revenue arithmetic.** $3M = 5,000 × $50/mo, or 2,500 × $100 blended with desk seats. $1M ≈ 1,700 × $50 is the credible three-year target; that is <10% of OptionStrat's claimed active base.

**Owner assets.** Uses voltorch directly (the kernel *is* voltorch v2: batched CUDA pricing + autodiff greeks). Uses the calibration/coverage engine for probability cones and for publishing coverage scorecards of P&L attribution. Ignores the append-only ledger store and the queue archive. voltorch currently has 0 stars / 0 forks, so the open-core funnel starts from zero.

**90-day first move.** Ship the crypto kernel first: pull Deribit BTC/ETH books, fit arb-free SSVI every minute, publish a live public page showing your surface vs Deribit marks and vs Laevitas' displayed IV with butterfly/calendar-arb violations counted per venue per day. That page is the marketing. Simultaneously publish one benchmark post reconstructing a known OptionStrat or Option Omega backtest and showing where the greeks/P&L attribution diverge from settlement. Card-paid terminal only after the kernel has a few hundred stars and a Discord.

---

### #2 Decumulate — joint tax/spending/Roth/annuitisation optimizer for DIY retirees

**What it is.** A retirement decumulation engine that solves spending, withdrawal order, Roth conversions, Social Security timing, IRMAA and annuitise-vs-drawdown as one stochastic-control problem with mortality weighting and calibrated (regime/bootstrap, not iid-lognormal) return simulation, wrapped in a planner UX good enough to compete with ProjectionLab/Boldin.

**Who pays and how they find it.** DIY near-retirees and FIRE households (Bogleheads, r/financialindependence, early-retirement.org) who already pay $109–149/yr for ProjectionLab, Boldin, Pralana or MaxiFi. Forum word-of-mouth, "best retirement calculator" comparisons, long-form posts on why iid Monte Carlo overstates success. Card, annual plans. Users bring their own balances.

**Verified comparable — the strongest in the entire scan.** ProjectionLab: $150 MRR (May 2021) → $83.3k MRR / $1M ARR (June 2025), zero funding, 100k+ households, no paid advertising (https://projectionlab.com/blog/we-reached-1m-arr-with-zero-funding). Correction: it was solo only for ~2 years (~$50k MRR / ~$600k/yr solo per https://www.starterstory.com/projection-lab-breakdown); at $1M it had a second full-timer on growth plus community contractors. Still ≤5 people, no sales team. Growth did include podcast appearances (ChooseFI, FI Show) and a Mr Money Mustache shout-out — public promotion, not sales calls, but more founder-visible than "write and wait".

**Engineering moat.** Correctness of the optimisation. Kotlikoff's public critique documents Boldin's Roth tool ignoring future taxes/IRMAA and applying a rigid 4% rule instead of solving jointly (https://larrykotlikoff.substack.com/p/boldins-new-retirement-roth-conversion); Boldin's own Roth page is a scenario comparison grid, not a solver, and its $3,200 advisor tier is where recommendations are sold (https://www.boldin.com/retirement/roth-conversion-calculator/, https://www.boldin.com/retirement/pricing/). A DP/stochastic-control solver over the real bracket/IRMAA/RMD/NIIT rule set with published test vectors against IRS worksheets, out-of-sample coverage tests on the return model, and differentiable simulation for instant plan sensitivities.

**Honest kill risk.** The finalist's "incumbents have not done joint optimisation" is false for MaxiFi, which has sold a joint spending + Roth + IRMAA + state-tax optimiser since 1993 at $149/yr and markets Kotlikoff's economist authority (https://www.maxifi.com/features/roth-conversion-optimizer, https://www.maxifi.com/pricing); Pralana already advertises bracket/IRMAA-bounded Roth optimisation and SS-age optimisation (https://pralanaretirementcalculator.com/pralana-gold/). The remaining differentiation is stochastic (mortality-weighted, regime/bootstrap) control vs MaxiFi's essentially deterministic solver — real, but invisible to a lay buyer. Price ceiling is $109–149/yr, so $3M needs ~20k subscribers vs ProjectionLab's ~7.7k at $1M; Boldin has 450k+ users and partner widgets (https://www.boldin.com/retirement/about-us/). US-tax-code-only, annual rule treadmill, and a large UX build for a solo founder. Owner fit is the weakest of the top four (3/5).

**Revenue arithmetic.** $3M at $150/yr = 20,000 subs; at a $30/mo optimiser tier = 8,300. $1M is a *proven* solo-ish outcome on this exact buyer; $3M is a 2.5–3× extension, not a new category.

**Owner assets.** Uses the calibration/coverage engine heavily (return-model coverage tests are the marketing). Uses voltorch's autodiff style for plan sensitivities but not the option-pricing code itself. Ignores the ledger store and queue archive.

**90-day first move.** Do not build the planner. Build the solver as a CLI/notebook and publish one post: "Three tools, one household, three different Roth answers — and what the DP optimum actually is", with IRS-worksheet test vectors and a downloadable script. Post it to Bogleheads and r/financialindependence. If it produces a waiting list, build the planner; if it produces a shrug, stop — the buyer has told you correctness is not the hook.

---

### #3 Sweep — GPU/SIMD portfolio backtester and parameter-sweep engine (BYO data)

**What it is.** pip-installable CUDA/SIMD backtesting + hyperparameter-sweep engine with bit-exact, auditable fill/margin/borrow/corporate-action/futures-roll semantics and a published conformance suite; paid tier on the VectorBT PRO / AmiBroker model plus a hosted GPU-sweep meter.

**Who pays and how they find it.** Retail/prosumer systematic traders, indie quants and 1–5 person shops on vectorbt/backtesting.py/AmiBroker/NautilusTrader. GitHub, PyPI, r/algotrading, YouTube algo channels, benchmark posts. Card licence exactly as VectorBT PRO ($25/mo, $240/yr, $500 lifetime) sells.

**Verified comparable.** VectorBT PRO — solo (Oleg Polakow), 379 GitHub sponsors at $25/mo (~$115k/yr from that channel; https://github.com/sponsors/polakowo, https://vectorbt.pro/become-a-member/). AmiBroker — self-described "small software company", two named staff, $299/$379/$499 perpetual sold purely through FastSpring/Paddle since 1995 (https://www.amibroker.com/about.html, https://www.amibroker.com/order.html); revenue **not** public (Owler estimate unfetchable). So the shape is proven; the $1M+ datapoint is inferred, and the only hard solo number is sub-$1M.

**Engineering moat.** The specific "CUDA sweep engine" lane is genuinely empty — a GitHub star-sorted search for GPU backtesting returns 0–1-star toys, and NVIDIA's gQuant/fsi-samples is dormant at 307 stars. Exact settlement/margin/borrow/corporate-action arithmetic that vectorised numpy engines silently fudge, deterministic reproducible runs, and a published conformance suite map directly onto the owner's CUDA + numerics background and the F-04 leakage lesson.

**Honest kill risk.** Both halves of the moat are already the headline claim of a live competitor: speed (VectorBT PRO, Numba + new Rust engine — and Numba can target CUDA in a quarter if a rival proves demand) and determinism (NautilusTrader, 28.4k stars, 6-person Nautech team with Pro/Cloud tiers, https://nautilustrader.io/team/). The hosted-sweep half is already occupied by QuantConnect (GPU nodes, credits) and MQL5 Cloud Network (61k agents, https://cloud.mql5.com/). Bar-data sweeps are embarrassingly parallel on multicore CPU, so GPU only bites for intraday/tick — a narrower paying niche. Price is anchored at $25/mo by the incumbent. The absence of a GPU backtester with traction is as consistent with "no demand" as with "open lane".

**Revenue arithmetic.** $3M = 5,000 × $50/mo, 2× the incumbent's price on ~13× its known payer count. $1M (1,700 × $50) is the credible ceiling absent AmiBroker-scale desktop adoption.

**Owner assets.** Shares the batched CUDA pricing kernel with #1 — this is the strongest structural argument for the candidate: one kernel, two products, two buyer pools. Uses the calibration engine for out-of-sample/overfitting statistics on sweeps (deflated Sharpe, coverage of walk-forward intervals). Ignores ledger store and queue archive. The dropped BookForge L2-replay idea could later be a tier here.

**90-day first move.** Publish the conformance suite before the engine: a repository of 50 hand-checked test vectors (splits, dividends, borrow, futures rolls, partial fills, signal-shift look-ahead) and a table showing which of vectorbt / backtesting.py / NautilusTrader / zipline-reloaded pass each. That post is the wedge regardless of whether the GPU engine ever ships. Then ship the CUDA sweep as a vectorbt-compatible drop-in so the migration cost is zero.

---

### #4 Vintage — point-in-time / bitemporal columnar store for research data

**What it is.** pip-installable Rust/Arrow engine storing tabular time series with two clocks (valid time + knowledge time) so a user can ask "what did I know on date T" with a physically enforced no-leakage guarantee; restatement diffs, delta-of-vintage encoding, forecast-vintage (issue × target) indexing. Free embedded core; small-team licence; metered S3-backed cloud.

**Who pays and how they find it.** Small quant shops, independent researchers, energy/weather forecasting teams and ML feature engineers who hand-roll snapshot folders. PyPI, GitHub, "point-in-time backtest" / "look-ahead bias" searches, the owner's leakage posts.

**Verified comparable.** **None in category** — the finalist said so and both skeptics confirmed. Shape comparables only: Sidekiq, solo, $80k/mo, self-serve $995/yr Pro (https://www.indiehackers.com/product/sidekiq, https://sidekiq.org/products/pro); Plausible, $1M ARR June 2022 with four people, no ads, no sales (https://plausible.io/blog/open-source-saas). Both category incumbents sell by contact: ArcticDB pricing is "email us" (https://docs.arcticdb.io/6.9.0/licensing/); XTDB is free MPL seeking design partners (https://docs.xtdb.com/).

**Engineering moat.** Row-level bitemporal as-of join kernels over Arrow with SIMD; a testable "query bound to knowledge time T cannot read later rows" guarantee; delta-of-vintage encoding for 10–100× over naive snapshots. ArcticDB calls itself "bitemporal" but means symbol-level version timestamps, not row-level valid/knowledge time (https://docs.arcticdb.io/latest/faq/); DuckDB has single-axis ASOF only; XTDB has the full semantics but is a JVM server. The one pip prior-art (pytemporal, Rust/PyO3) has 1 star and 52 downloads/month (https://github.com/gingermike/pytemporal).

**Honest kill risk.** Measured demand today is near zero: pytemporal's 52 downloads/month, zero ArcticDB issues asking for row-level PIT, xtdb PyPI at 89 downloads/month. The organic pool must be *created* by the owner's writing. ArcticDB (Man Group-incubated, 260k PyPI downloads/month) already owns the word "bitemporal" in quant-Python mindshare and could add a knowledge_time column semantic in one or two releases. $250/mo ARPU is unanchored because no incumbent publishes prices. The $3M target needs ~1,000 paying teams from 20–50k free users, and nothing verified supports that funnel.

**Revenue arithmetic.** $3M ≈ 600 small-team licences at $150–300/mo + 400 metered cloud accounts. Unanchored; treat $300–500k as the realistic outcome unless the writing creates a category.

**Owner assets.** This is the only candidate that uses the **append-only ledger store** (as the storage substrate) and could ingest the **queue archive** as a worked example of forecast vintages. Uses the calibration engine for forecast-vintage evaluation (issue-time coverage scorecards). Ignores voltorch. Fit 5, but it is the candidate where the owner's assets are most useful and the market signal is weakest — a classic build-first trap (see memory: two projects died this way).

**90-day first move.** Do not build the engine. Write the two posts that would create the category: "Your backtest is leaking through your fundamentals vendor's restatements — here is how to measure it" with a script that quantifies look-ahead in a user's own Compustat/FRED-style panel, and "ArcticDB is version-temporal, not bitemporal — the difference costs you X bps". Track PyPI downloads of the measurement script for 60 days. Build only if it clears ~2k/month organically.

---

### #5 Postgres-native durable job queue for Python ("Oban for Python") — CONDITIONAL, probably dead

**What it is.** Sidekiq/Oban-grade background-job and durable-workflow engine on the Postgres a team already has, Rust worker core, verified exactly-once/at-least-once semantics; free OSS core, card-paid Pro per application.

**Why it is ranked at all.** Both skeptics passed it: the shape is the best-proven in the scan (Sidekiq $80k/mo solo, https://www.indiehackers.com/product/sidekiq; Oban Pro $150/mo per app, https://oban.pro/pricing; River Pro $125/mo, https://riverqueue.com/pro), the pip channel is large (procrastinate 1.54M downloads/month, pgqueuer 379k, hatchet-sdk 1.33M), and Hatchet/DBOS prove Python devs card-pay for this category.

**Why the owner should not start it.** The scan's own database sweep records: **"Oban for Python shipped 25 Jul 2026 with a paid Pro tier" (https://oban.pro/articles/introducing-oban-python)** — the proven operator has taken the exact slot, and neither skeptic caught the contradiction. Independently, the moat-crowding skeptic found the Pro feature list (workflows, rate limits, unique jobs) is already free in DBOS Transact Python (2.28M downloads/month, https://docs.dbos.dev/python/tutorials/queue-tutorial) and Procrastinate, so the Pro tier must be re-derived from throughput, partitioned fairness, ops UI and support. Weakest link to the owner's finance/ML interests. If the Oban Python launch is confirmed on inspection, the list is four, not five.

**Revenue arithmetic (if it were alive).** $150/mo per application → $3M = ~1,650 apps; Python's lower payment culture implies 2–3× Sidekiq's customer count for the same revenue.

**Owner assets.** Ignores all four.

**90-day first move.** One afternoon: read the Oban Python announcement and pricing. If it is a real Pro tier, close the file.

---

## 3. Honourable mentions

Survivors of one skeptic, or residual slices the skeptics explicitly named as "a different, thinner candidate". None has a comparable; none should be started without one.

- **Differentiable GPU PV + storage simulator** (moat lens passed; killed only on the comparable — PVsyst is ~36 staff per LinkedIn, not 8, and ships PVsystCLI batch mode). The batch-throughput + autodiff lane is genuinely empty (no JAX/torch/CUDA pvlib exists; pvlib 1.45M downloads/month, https://pypistats.org/packages/pvlib) but correctness is free (pvlib, PySAM) and every vendor is quote-only.
- **Calibra** (moat lens passed; killed on reachability — every operator-tier vendor incl. indie pvnode is contact-sales; card-payable prosumer tier is ~$2–5/mo). The telemetry-conditioned calibrated-quantile niche is unoccupied in OSS but Amperon already sells probabilistic asset forecasts and Darts ships conformal wrappers free.
- **fp8/int4/bf16 kernel-correctness library** (the surviving half of KernelGuard): torch.testing.assert_close has no float8 entry and no ULP/distribution-aware mode; KernelBench-X reports 0/30 on quantisation tasks. A pip library with no demonstrated payer — only viable as open-source credibility for #1/#3.
- **Public financial-document reconciliation benchmark + on-prem engine** (the residual of the Reconciled Document Engine): DocuClipper and Suparse already *claim* balance reconciliation; a benchmark that measures it rather than claims it is the only unclaimed ground, and it is a content asset, not a business.
- **SPZ-class splat compressor / deterministic large-scene chunker as a library** (residual of the 3D reconstruction idea): a component, not a Metashape successor.
- **Order-book replay tier for Sweep** (from dropped BookForge): hftbacktest is free and high-fidelity; only makes sense as a feature of #3.

---

## 4. Killed by the skeptics

- **Reconciled Financial-Document Engine** — moat refuted: DocuClipper (3-person SEO incumbent) and Suparse already advertise the exact identities (opening + credits − debits = closing; Σ line items + tax = total) as features; Reducto ($108M raised) sells self-serve at $10–40/1k pages; Mistral OCR at $4/1k pages. A better solver behind an identical green tick is not buyer-perceivable.
- **KernelGuard** — both lenses refuted: the sole comparable (Depot "$1M with 3 people") is false — Depot has 18–22 staff, a VP of Revenue, $14.1M raised (https://depot.dev/about); CodSpeed owns performance-regression CI with 2.3M pytest-codspeed downloads/month and "GPU instruments: soon" on its pricing page; change-point detection is free in Apache Otava.
- **nvcache** — both lenses refuted: the moat premise is factually wrong — sccache has decomposed nvcc per SM-arch since v0.9.0 (NVIDIA's PR #2247, Nov 2024, https://github.com/mozilla/sccache/pull/2247) and RAPIDS runs it in CI; Depot gates GPU builds behind "schedule a call"; Buildless is still free-in-beta after three years.
- **Orphan-proof training-run tracker** — moat refuted: W&B Pro is $60/month for 10 seats, not $60/user, so "$50/seat undercuts W&B" is false; Comet $19/user with free academic Pro; ClearML $15; trackio (Hugging Face) is free with alerts; W&B Automations already do z-score/window divergence alerts; Neptune shutdown was March 2026 — the migration wave is over.
- **MatchKernel** — both lenses refuted: OpenSanctions ships free MIT cross-script name matching with public training data (https://www.opensanctions.org/matcher/) and has a "Talk to sales" function; its only self-serve revenue is metered *data* (the disqualified shape); Splink already outputs Fellegi-Sunter probabilities; Senzing is annual up-front via "Contact Us".
- **GPU-native 3D reconstruction** — moat refuted: COLMAP absorbed GLOMAP (1–2 orders of magnitude faster) and added the Caspar GPU bundle adjuster; gsplat has NVIDIA/Meta/Amazon contributors; Epic made RealityCapture free for anyone under $1M revenue — the exact target buyer; DJI Terra bundles splatting; Agisoft "5 people/$9M" unverifiable (403).
- **Ferrograph** — moat refuted: NVIDIA cuOpt is Apache-2.0, pip-installable, self-hostable, holds 23 SINTEF world records and already covers the "constraint richness" claimed as the differentiator; the "self-serve gap" does not exist (GraphHopper, Routific, OptimoRoute, Google all card-billed); any incumbent can wrap cuOpt.
- **Vectorised PV + storage engine** — comparable refuted: PVsyst is 11–50 staff (~36 on LinkedIn) with a quote channel and a batch CLI, not an 8-person $1M no-sales shop; HelioScope figures unverifiable. No ≤5-person comparable exists.
- **Calibra** — reachability refuted: the revenue-bearing operator tier is quote-gated at every observed vendor including the indie one (pvnode: "commercial use requires direct contact with founders"); the card-payable tier tops out at €200/yr (Forecast.Solar) and Charge HQ could not hold $84/yr.

---

## 5. Dropped at merge

- **Curveset (rates library)** — buyers are funds/banks (pedigree-gated); rateslib ~344 stars, QuantLib free; no comparable.
- **Generic document inference engine** — commoditised by Mistral OCR/VLMs above and Datalab/Marker (72k stars) below.
- **Local speech/dictation app** — 17+ superwhisper clones, $29 lifetime pricing, OS-native dictation; no speech background.
- **GPU kernel correctness CI / tensor snapshot CI** — merged into KernelGuard (then killed).
- **Eval-stats & judge-calibration engine** — a feature; evalstats is free; Braintrust/Langfuse add CIs as a checkbox.
- **Synthetic tabular/time-series engine** — Gretel self-serve killed post-NVIDIA, Mostly AI shut down 2026; budgets are enterprise.
- **Prodigy-style annotation tool** — LLM auto-labelling eats it; Label Studio/CVAT free; Explosion needed >5 people.
- **Arithmetic-verified invoice parser / reconciled statement API** — merged into the document engine (then killed).
- **EN 16931 e-invoice engine** — regulation-created need, free validators, transmission needs Peppol certification.
- **Exact Schedule Engine (leases/loans)** — only comparable (Cradle, 4 people) reached ~$182k ARR before selling.
- **Native sampling profiler with GPU timeline** — Nsight/Tracy/perf free; Superluminal still 2 people after 10 years.
- **Gaussian-splatting engine** — merged into 3D reconstruction (then killed).
- **GPU image/video transform server** — commodity (Cloudinary, imgproxy, libvips); fit 2.
- **BESS dispatch / Dispatchr** — buyers are financiers and Modo/Fluence customers on 36-month contracts; sales-led; confidence 1.
- **Hosted home-energy MPC for Home Assistant** — Charge HQ measured $125–140/yr savings and could not hold $84/yr; Predbat/EMHASS free.
- **FetchForge anti-bot collection API** — moat collapses to proxy purchasing; arms race; legal/abuse exposure.
- **RedactCore PII engine** — Presidio 6.2M downloads/month free; paying buyers are compliance teams (sales-led).
- **ReportSink security-report ingestion** — Report URI/URIports own the terms for a decade; ARPU $6–120; lifestyle ceiling.
- **VolDesk crypto options terminal** — merged into #1 as the launch wedge.
- **BookForge L2/L3 replay** — ceiling ~$200–480k; hftbacktest free; could be a Sweep tier.
- **LedgerDecode on-chain cost basis** — overlaps tabled tax engine; protocol-decoding treadmill; seasonal; no comparable.
- **Relief GPU terrain API** — Earth Engine and rasterio/WhiteboxTools cover it; no GIS track record.
- **Solid self-serve FEA** — fit 2; credibility is element libraries not solver speed; cuDSS free.

---

## 6. Fields swept that yielded nothing

- **Finance tooling:** developer ledger/balance APIs (all VC); small-fund NAV/admin (GP-trust-gated); crypto tax as new entrant (CoinLedger ~35 staff, Koinly ~93); budgeting/plaintext accounting (Actual open-sourced with no revenue); trade journals (Tradervue solo but overlaps tabled item); donation-funded ETF backtesters (testfol.io — dashboard niche commoditised to $0).
- **Quant/risk compute:** XVA/GPU Monte Carlo for banks; actuarial reserving (Slope raised $2M, 11–50 people, sold to Akur8); model-validation reference pricers; economic scenario generators; institutional stress testing — every one pedigree- and sales-gated.
- **AI inference infra:** quantisation-as-a-service (HF/Unsloth free); multimodal router (OpenRouter closed it at $5M ARR on 5 people, now ~50); $/token inference (capital war); local-LLM desktop runtimes (no verified revenue); edge NVR subscriptions (Frigate+/Scrypted revenue unverifiable); kernel auto-gen compilers (VC-saturated); constrained decoding (free).
- **AI training/data layer:** corpus dedup/curation (NeMo Curator free); fine-tuning infra (capital-intensive); classical ML monitoring (consolidated into sales-led vendors); conformal/UQ API (MAPIE free); RL environments; LLM trust scoring (Cleanlab $30M, acquired).
- **AI vertical apps:** legal drafting for individuals; insurance claims; customs docs; HS-code classification (data moat); rent-roll extraction (6 funded competitors); transaction categorisation (merchant-DB data moat); bank reconciliation matching (Xero/QB bundle it); procurement; logistics paperwork (contingency-fee shape).
- **Databases/storage:** job queues (Oban Python shipped Jul 2026); serverless vector search (turbopuffer, est. $100M ARR); DuckDB-on-S3 (MotherDuck, cloud vendors); edge IoT time-series (ReductStore me-too); DuckDB workbench (official free UI); trading event journals (Chronicle: $957k, 11 staff, bank sales); log stores (VictoriaMetrics 46 people); Kafka-on-S3/CDC/local-first sync (all VC); GPU OLAP/ANN (cuDF/cuVS free); bitemporal Postgres extensions (free; PG18 temporal constraints); compression formats (free by convention).
- **Low-level systems:** numerics JIT/DSL compilers (Modular/Taichi VC, rest free); WASM runtimes (VC hosting); RISC-V/FPGA toolchains (quote-driven); embedded/RTOS tooling (OEM procurement); binary analysis (Hex-Rays €20.6M at 13 people but VC-owned; Ghidra free).
- **Vision/sensing:** industrial inspection (sales-led, anomalib free); robotics/SLAM (hardware-coupled); medical imaging (grant-funded free tools); satellite analytics (data resale or sales-led); self-hosted NVR (Frigate owns it, revenue undisclosed); browser barcode SDK (STRICH solo/profitable but revenue unverified); LiDAR tools (CloudCompare free).
- **Energy/physical infra:** EV-charging CSMS (unverified — budget ran out); data-centre power/cooling sim (enterprise); building energy sim (Ladybug/Pollination own the Rhino/Revit channel; EnergyPlus free); battery cell-model fitting (Ionworks has the PyBaMM maintainers); power-flow/OPF API (pandapower free, utilities pedigree-gated); industrial IoT historians (crowded); commodity logistics tracking (data resale); day-ahead price forecast API (**unsearched**, not empty).
- **Security/intel/OSINT:** maritime AIS anomaly (free data, but insurers/navies buy via sales); threat-intel feeds (Shodan/HIBP/AbuseIPDB prove solo card-paid scale but are data-hoarding businesses); typosquat monitoring (dnstwist free, depends on paid passive-DNS); OSINT frameworks (SpiderFoot wraps third-party data; SOC buyers); disinformation detection (grant/platform-funded); secure comms (nonprofit/VC); SBOM/dependency risk (Snyk/Socket; Trivy free).
- **Devtools/platforms:** agent sandboxes (capital moat); MCP gateways (protocol churn, land-grab); deterministic simulation testing (Antithesis contact-sales — a gap, but no comparable and very high build cost); feature flags/gateways/auth/code search/docs hosting (crowded, low fit); package security (data business); GPU-kernel benchmark runner farms (thin usage-infra margins).
- **Crypto/DeFi microstructure:** MEV tooling (searchers build in-house; only revenue is data); wallet analytics (labelled-dataset moat); CL-LP simulators (demand collapses in bear markets — fails permanent-need); funding/basis dashboards (data resale); exchange infrastructure (BD-gated); hosted market-making bots (Hummingbot: $309k revenue with losses, fee-share BD model).
- **Scientific/engineering compute:** GPU LP/QP/conic solvers (cuOpt/HiGHS free); computational chemistry (Rowan VC, fit 1); differentiable simulation/FEM libraries (free or vendor-subsidised); simulation-as-a-service (contact-sales); sparse direct solvers (Pardiso unreachable; cuDSS free); mesh/CAD kernels (Zoo VC; Gmsh free).

---

## 7. Cross-cutting patterns — how the owner should think

**1. The buyer who pays is a person risking their own money, not an engineer spending a company's.** All four real survivors sell to individuals with capital at stake — options traders, systematic traders, retirees, independent quants — who already pay a card-billed incumbent. Every candidate whose buyer was "developers at companies" died: either the vendor gives the engine away (NVIDIA, Epic, Hugging Face, Man Group, DOE, MoJ) or the person who values correctness sits inside procurement. The scan found no exception. The owner's "developer API / open-core cloud" shape is eligible on paper but produced zero survivors with a comparable.

**2. Speed alone never survived; checkable correctness did.** cuOpt, COLMAP/Caspar, gsplat, sccache-nvcc, CodSpeed, MQL5 Cloud — every "we are faster on GPU" pitch met a free or funded incumbent already on that axis. What survived was a correctness claim the *buyer* can verify against a public reference: Deribit/CBOE marks, IRS worksheets, a conformance suite of hand-checked test vectors, a leakage guarantee. Speed is the enabler (it makes the correct thing interactive); correctness is the product. Build the reference and the test vectors first — they are simultaneously the moat, the marketing and the demand test.

**3. The verified ceiling is $1M, not $3M.** Across ~130 companies examined, the ≤5-person / no-sales / $1M+ bar was cleared with a public source by exactly three: ProjectionLab, Sidekiq, Plausible. Every larger small-team figure (AmiBroker, Agisoft, OptionStrat, Parseur's "7 figures" at 6 people) is inferred, estimated or 403'd. Plan on $1M per product. $3M requires two products, which is why #1 and #3 sharing one CUDA pricing/backtest kernel is the single most important structural fact in this report.

**4. BYO-data cuts both ways.** It keeps the owner out of data resale, but every skeptic noted it halves conversion. The survivors route around it: public exchange books (crypto), the user's own balances (retirement), the user's own bar data (backtesting). Pick wedges where the data the buyer needs is already free or already theirs.

**5. Incumbents win on habit, not search.** OptionStrat and Option Omega get 80–85% direct traffic. The owner refuses sales, so the content *is* the sales team — and the content that works is adversarial benchmarking of the incumbents ("your backtester's greeks are wrong", "three tools, three Roth answers"), not tutorials. Budget the writing as seriously as the kernel.

**6. The owner's existing assets are unevenly useful.** voltorch and the calibration/coverage engine map directly onto #1, #2 and #3 and should be the spine. The append-only ledger store and the queue archive were built for ideas this scan killed (grid-queue commercialisation); only Vintage (#4) would use them, and Vintage is the candidate with the weakest demand signal — a build-first trap the owner has fallen into twice before. Do not let sunk assets pick the market.

**7. Correlation is the hidden risk.** #1 and #3 both sell to retail traders; retail options and algo activity are cyclical even if the *need* is permanent. #2 (retirees) is the natural hedge and the strongest comparable, at the cost of the owner's interest. A two-product plan should pair one trader product with Decumulate, not stack two trader products.

**Bottom line.** Start with the crypto surface kernel (#1), publish the backtester conformance suite (#3) as a second content wedge on the same kernel, and run the Decumulate solver post (#2) as a cheap demand probe in parallel. Do not build Vintage until its measurement script has organic downloads. Close the Oban file after one afternoon of checking.
